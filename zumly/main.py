import argparse
import json
import logging
import os
import sys
import tempfile
import time
import uuid

from zumly_capture.audio import AudioCapture, cleanup_audio_tracks, mux_audio_tracks
from zumly_capture.identity import FILE_PREFIX, PRODUCT_NAME
from zumly_capture.session import CaptureSession, publish_recording
from zumly_capture.smart_zoom import render_smart_zoom

from zumly.app.screen_recorder import ScreenRecorder
from zumly.app.mouse_tracker import MouseTracker
from zumly.app.click_tracker import ClickTracker
from zumly.app.global_hotkeys import GlobalHotkeys
from zumly.app.session_timing import RecordingState, SessionTimelineClock

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _write_result_payload(result_file: str, payload: dict) -> bool:
    """Atomically publish a recorder result for the tray process."""
    if not result_file:
        return True
    temp_path = ""
    try:
        result_path = os.path.abspath(result_file)
        os.makedirs(os.path.dirname(result_path) or ".", exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=os.path.dirname(result_path) or ".",
            prefix=f"{FILE_PREFIX}_result_",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = handle.name
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, result_path)
        return True
    except OSError as exc:
        logging.error("Could not publish recorder result payload: %s", exc)
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass
        return False


def _failure(
    result_file: str,
    message: str,
    code: int = 1,
    **details: object,
) -> int:
    """Write a terminal failure payload and return the process exit code."""
    logging.error(message)
    payload = {"status": "failed", "error": str(message), "returnCode": int(code)}
    payload.update(details)
    _write_result_payload(
        result_file,
        payload,
    )
    return code


def _read_control_payload(path: str, last_sequence: int) -> tuple[int, str]:
    """Return the next complete sequenced recorder command, if available."""
    if not path:
        return last_sequence, ""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        sequence = int(payload.get("sequence", 0))
        action = str(payload.get("action", "")).strip().lower()
    except (FileNotFoundError, PermissionError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return last_sequence, ""
    if sequence <= last_sequence or action not in {"pause", "resume", "stop", "cancel"}:
        return last_sequence, ""
    return sequence, action


def _write_status_payload(
    path: str,
    sequence: int,
    state: RecordingState,
    clock: SessionTimelineClock,
    **details: object,
) -> None:
    payload = {
        "sequence": int(sequence),
        "state": state.value,
        "activeDurationMs": round(clock.active_time_ms(), 3),
        "pausedDurationMs": round(clock.paused_duration_ms, 3),
    }
    payload.update(details)
    _write_result_payload(
        path,
        payload,
    )


def _run(args: argparse.Namespace) -> int:
    target_kind = str(args.target_kind or "monitor").lower()
    capture_rect: dict = {}
    capture_target: dict = {"kind": target_kind}
    if target_kind == "monitor":
        for monitor in ScreenRecorder.get_monitors():
            if monitor["index"] == args.monitor:
                capture_rect = dict(monitor)
                break
        if not capture_rect:
            return _failure(
                args.result_file,
                f"Could not find monitor with index {args.monitor}",
            )
        capture_target["monitorIndex"] = int(args.monitor)
    elif target_kind == "window":
        from zumly.app.window_utils import get_window_rect

        capture_rect = get_window_rect(int(args.window_hwnd)) or {}
        if not capture_rect:
            return _failure(args.result_file, "The selected window is no longer available")
        capture_target["windowHandle"] = int(args.window_hwnd)
        capture_target["windowTitle"] = str(args.window_title or "")
    elif target_kind == "region":
        if len(args.region) != 4:
            return _failure(args.result_file, "Region capture requires left, top, width, height")
        left, top, width, height = (int(value) for value in args.region)
        width = max(2, width - width % 2)
        height = max(2, height - height % 2)
        capture_rect = {
            "left": left,
            "top": top,
            "width": width,
            "height": height,
        }
    else:
        return _failure(args.result_file, f"Unsupported capture target: {target_kind}")

    capture_target.update(
        {
            "left": int(capture_rect.get("left", 0)),
            "top": int(capture_rect.get("top", 0)),
            "width": int(capture_rect.get("width", 0)),
            "height": int(capture_rect.get("height", 0)),
        }
    )

    # Trackers
    click_tracker = ClickTracker()
    mouse_tracker = MouseTracker(click_state_provider=click_tracker.is_button_down)

    # Setup Hotkey Tracker (Callback based)
    recording_toggled = [False]
    pause_toggled = [False]

    def on_hotkey_triggered():
        recording_toggled[0] = True

    def on_pause_hotkey_triggered():
        pause_toggled[0] = True

    hotkey_tracker = GlobalHotkeys(
        callback=on_hotkey_triggered,
        pause_callback=on_pause_hotkey_triggered,
    )

    # Initialize recorder
    def on_recording_finished(path: str) -> None:
        pass

    def on_capture_backend_changed(backend: str) -> None:
        pass

    timeline_clock = SessionTimelineClock()
    recorder = ScreenRecorder(
        recording_finished_cb=on_recording_finished,
        capture_backend_changed_cb=on_capture_backend_changed,
        timeline_clock=timeline_clock,
    )

    logging.info("Starting %s capture at %s FPS...", target_kind, args.fps)
    if target_kind == "window":
        recorder.start_capture_window(int(args.window_hwnd), args.fps)
    elif target_kind == "region":
        recorder.start_capture_region(capture_rect, args.fps)
    else:
        recorder.start_capture(args.monitor, args.fps)

    # Let capture spin up
    time.sleep(2.0)

    # Complete all potentially blocking preparation before defining time zero.
    raw_video_path = recorder.prepare_recording()
    audio_capture = AudioCapture(
        [str(args.microphone or ""), str(args.system_audio or "")]
    )
    audio_capture.start()
    if args.microphone or args.system_audio:
        time.sleep(0.25)
    SESSION_EPOCH = time.perf_counter()
    audio_lead_ms = max(0.0, (SESSION_EPOCH - audio_capture.started_at) * 1000.0)
    recorder.start_recording(
        session_epoch=SESSION_EPOCH,
        timeline_clock=timeline_clock,
    )
    click_tracker.start(SESSION_EPOCH, capture_rect, timeline_clock=timeline_clock)
    mouse_tracker.start(SESSION_EPOCH, timeline_clock=timeline_clock)
    recording_wall_time_ms = time.time() * 1000.0
    if not args.stop_file:
        hotkey_tracker.register_record_hotkey()

    from zumly.app.recording_overlay import RecordingOverlay

    overlay = RecordingOverlay(capture_rect)
    overlay.start()
    last_control_sequence = 0
    _write_status_payload(
        args.status_file,
        last_control_sequence,
        RecordingState.RECORDING,
        timeline_clock,
    )

    logging.info(f"Recording started. Outputting raw video to: {raw_video_path}")
    if args.duration > 0:
        logging.info(f"Recording will stop automatically after {args.duration} seconds.")
    else:
        logging.info("Press CTRL+SHIFT+R to stop recording.")

    try:
        while True:
            time.sleep(0.1)
            sequence, action = _read_control_payload(
                args.control_file,
                last_control_sequence,
            )
            if action:
                last_control_sequence = sequence
                if action == "pause":
                    if recorder.pause_recording():
                        overlay.set_paused(True)
                elif action == "resume":
                    if recorder.resume_recording():
                        overlay.set_paused(False)
                elif action == "stop":
                    _write_status_payload(
                        args.status_file,
                        sequence,
                        RecordingState.STOPPING,
                        timeline_clock,
                    )
                    logging.info("Stop command received. Stopping recording...")
                    break
                _write_status_payload(
                    args.status_file,
                    sequence,
                    recorder.recording_state,
                    timeline_clock,
                )
            if args.stop_file and os.path.exists(args.stop_file):
                logging.info("Stop file detected. Stopping recording...")
                break
            if args.duration > 0 and timeline_clock.active_seconds() >= args.duration:
                logging.info(f"Reached duration of {args.duration}s. Stopping recording...")
                break
            if recording_toggled[0]:
                logging.info("Hotkey pressed. Stopping recording...")
                break
            if pause_toggled[0]:
                pause_toggled[0] = False
                if recorder.is_paused:
                    if recorder.resume_recording():
                        overlay.set_paused(False)
                else:
                    if recorder.pause_recording():
                        overlay.set_paused(True)
                _write_status_payload(
                    args.status_file,
                    last_control_sequence,
                    recorder.recording_state,
                    timeline_clock,
                )
    except KeyboardInterrupt:
        logging.info("Ctrl+C pressed. Stopping recording...")

    # Freeze the shared timeline before stopping input hooks so shutdown time
    # cannot leak into the serialized media duration.
    recorder.stop_recording()
    mouse_events = mouse_tracker.stop()
    click_events = click_tracker.stop()
    audio_tracks = audio_capture.stop()
    hotkey_tracker.unregister_record_hotkey()
    overlay.stop()
    recorder.stop_capture()
    if args.stop_file:
        try:
            os.remove(args.stop_file)
        except OSError:
            pass

    if recorder.recording_error:
        cleanup_audio_tracks(audio_tracks)
        return _failure(
            args.result_file,
            f"Recording failed before a usable video was written: {recorder.recording_error}",
        )

    session_id = str(uuid.uuid4())
    duration_ms = recorder.recording_duration_ms
    output_path = os.path.abspath(args.out)
    pause_boundaries = [boundary.to_dict() for boundary in timeline_clock.pause_boundaries]
    publication_source = raw_video_path
    warnings: list[str] = []
    audio_manifest = {
        "requestedDevices": [track.device for track in audio_tracks],
        "state": "disabled" if not (args.microphone or args.system_audio) else "unavailable",
    }
    if audio_tracks:
        mux_handle = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        mux_path = mux_handle.name
        mux_handle.close()
        try:
            os.remove(mux_path)
        except OSError:
            pass
        muxed, mux_error = mux_audio_tracks(
            raw_video_path,
            audio_tracks,
            mux_path,
            duration_ms,
            pause_boundaries,
            lead_ms=audio_lead_ms,
        )
        if muxed:
            publication_source = mux_path
            audio_manifest["state"] = "muxed"
        else:
            try:
                os.remove(mux_path)
            except OSError:
                pass
            warnings.append("Audio capture was unavailable; the video was saved without audio.")
            if mux_error:
                logging.warning("Audio mux failed: %s", mux_error)
    elif args.microphone or args.system_audio:
        warnings.append("The selected audio device did not produce a usable track.")
    cleanup_audio_tracks(audio_tracks)

    smart_zoom_manifest: dict[str, object] = {
        "state": "not_processed",
        "keyframes": [],
    }
    if args.smart_zoom:
        smart_zoom_level = max(1.1, min(3.0, float(args.smart_zoom_level)))
        render_sequence = [last_control_sequence]

        def smart_zoom_cancelled() -> bool:
            sequence, action = _read_control_payload(
                args.control_file,
                render_sequence[0],
            )
            if action:
                render_sequence[0] = sequence
            return action == "cancel"

        def smart_zoom_progress(progress: int) -> None:
            _write_status_payload(
                args.status_file,
                render_sequence[0],
                RecordingState.PROCESSING,
                timeline_clock,
                phase="smart_zoom",
                progress=max(0, min(100, int(progress))),
            )

        smart_zoom_progress(0)
        render_handle = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        render_path = render_handle.name
        render_handle.close()
        try:
            os.remove(render_path)
        except OSError:
            pass
        source_before_render = publication_source
        outcome = render_smart_zoom(
            source_before_render,
            render_path,
            mouse_events,
            click_events,
            capture_rect,
            duration_ms,
            recorder.actual_fps or float(args.fps),
            zoom_level=smart_zoom_level,
            render_cursor=bool(args.render_cursor),
            render_clicks=bool(args.render_clicks),
            progress_callback=smart_zoom_progress,
            cancel_callback=smart_zoom_cancelled,
        )
        smart_zoom_manifest = {
            "state": outcome.state,
            "keyframes": [keyframe.to_dict() for keyframe in outcome.keyframes],
            "renderCursor": bool(args.render_cursor),
            "renderClicks": bool(args.render_clicks),
            "zoomLevel": smart_zoom_level,
        }
        if outcome.error:
            smart_zoom_manifest["error"] = outcome.error
        if outcome.state == "processed" and outcome.output_path:
            publication_source = outcome.output_path
            if source_before_render != raw_video_path:
                try:
                    os.remove(source_before_render)
                except OSError:
                    pass
        elif outcome.state == "cancelled":
            warnings.append("Smart Zoom was cancelled; the unprocessed recording was saved.")
        elif outcome.state == "failed":
            warnings.append("Smart Zoom could not be applied; the unprocessed recording was saved.")
            if outcome.error:
                logging.warning("Smart Zoom render failed: %s", outcome.error)

    session = CaptureSession(
        session_id=session_id,
        media_path=output_path,
        capture_target=capture_target,
        started_at_unix_ms=recording_wall_time_ms,
        duration_ms=duration_ms,
        paused_duration_ms=timeline_clock.paused_duration_ms,
        pause_boundaries=pause_boundaries,
        requested_fps=float(args.fps),
        actual_fps=recorder.actual_fps,
        is_cfr=recorder.is_cfr,
        capture_backend=recorder.backend,
        frame_timestamps=recorder.frame_timestamps,
        mouse_track=[event.to_dict() for event in mouse_events],
        click_events=[event.to_dict() for event in click_events],
        capture_telemetry=recorder.capture_telemetry,
        audio=audio_manifest,
        smart_zoom=smart_zoom_manifest,
    )

    logging.info("Recording stopped. Publishing video to %s", output_path)
    try:
        published = publish_recording(publication_source, output_path, session)
    except Exception as exc:
        logging.exception("Could not publish the completed recording: %s", exc)
        recovery_candidate = publication_source if os.path.isfile(publication_source) else raw_video_path
        recovery_path = os.path.abspath(recovery_candidate) if os.path.isfile(recovery_candidate) else ""
        return _failure(
            args.result_file,
            f"Could not save the completed recording: {exc}",
            recoveryPath=recovery_path,
        )

    payload = {
        "status": "success",
        "mediaPath": published.media_path,
        "outputPath": published.media_path,
        "manifestPath": published.manifest_path,
        "sessionId": session_id,
        "durationMs": duration_ms,
        "returnCode": 0,
    }
    if publication_source != raw_video_path:
        try:
            os.remove(raw_video_path)
        except OSError:
            pass
    if published.warning:
        warnings.append(published.warning)
    if warnings:
        payload["warning"] = " ".join(warnings)
    _write_status_payload(
        args.status_file,
        last_control_sequence,
        RecordingState.FINISHED,
        timeline_clock,
        progress=100 if smart_zoom_manifest.get("state") == "processed" else 0,
    )
    if not _write_result_payload(args.result_file, payload):
        return 1

    if published.manifest_path:
        logging.info("Capture manifest saved to %s", published.manifest_path)
    for warning in warnings:
        logging.warning(warning)
    logging.info("Force exiting to release WGC hooks...")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=f"{PRODUCT_NAME} capture worker")
    parser.add_argument("--out", "-o", required=True, help="Output MP4 path")
    parser.add_argument("--monitor", "-m", type=int, default=1, help="Monitor index (default 1)")
    parser.add_argument("--fps", type=int, default=60, help="Recording FPS")
    parser.add_argument(
        "--target-kind",
        choices=("monitor", "window", "region"),
        default="monitor",
        help="Capture target type",
    )
    parser.add_argument("--window-hwnd", type=int, default=0, help="Window handle to capture")
    parser.add_argument("--window-title", default="", help="Selected window title for metadata")
    parser.add_argument(
        "--region",
        type=int,
        nargs=4,
        default=(),
        metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"),
        help="Physical-pixel region rectangle",
    )
    parser.add_argument("--microphone", default="", help="DirectShow microphone device")
    parser.add_argument("--system-audio", default="", help="DirectShow loopback audio device")
    parser.add_argument(
        "--smart-zoom",
        action="store_true",
        help="Apply automatic click-driven Smart Zoom after recording",
    )
    parser.add_argument(
        "--smart-zoom-level",
        type=float,
        default=1.5,
        help="Smart Zoom scale (1.1 to 3.0)",
    )
    parser.add_argument(
        "--render-cursor",
        action="store_true",
        help="Render the captured cursor as an optional video layer",
    )
    parser.add_argument(
        "--render-clicks",
        action="store_true",
        help="Render click indicators as an optional video layer",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=0.0,
        help="Optional duration to record in seconds (if 0, stops on hotkey CTRL+SHIFT+R)",
    )
    parser.add_argument(
        "--stop-file",
        type=str,
        default="",
        help="Optional file path used by the tray app to request a graceful stop",
    )
    parser.add_argument(
        "--result-file",
        type=str,
        default="",
        help="Atomic JSON result payload path for the tray process",
    )
    parser.add_argument(
        "--control-file",
        type=str,
        default="",
        help="Atomic sequenced pause/resume/stop/cancel command payload",
    )
    parser.add_argument(
        "--status-file",
        type=str,
        default="",
        help="Atomic recorder-state acknowledgement payload",
    )
    args = parser.parse_args()
    try:
        return _run(args)
    except KeyboardInterrupt:
        return _failure(args.result_file, "Recording cancelled by user", 130)
    except Exception as exc:
        logging.exception("Headless capture failed: %s", exc)
        return _failure(args.result_file, str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
