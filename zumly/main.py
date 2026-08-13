import argparse
import json
import logging
import os
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass

from zumly_capture.audio import AudioCapture, cleanup_audio_tracks, mux_audio_tracks
from zumly_capture.gif_export import export_gif
from zumly_capture.identity import FILE_PREFIX, PRODUCT_NAME
from zumly_capture.session import (
    CaptureSession,
    discard_unzoomed_recording,
    preserve_unzoomed_recording,
    publish_recording,
)
from zumly_capture.smart_zoom import render_smart_zoom

from zumly.app.screen_recorder import ScreenRecorder
from zumly.app.mouse_tracker import MouseTracker
from zumly.app.click_tracker import ClickTracker
from zumly.app.global_hotkeys import GlobalHotkeys
from zumly.app.recording_overlay import RecordingOverlay
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


@dataclass(slots=True)
class _RecordingArtifacts:
    recorder: ScreenRecorder
    timeline_clock: SessionTimelineClock
    raw_video_path: str
    mouse_events: list[object]
    click_events: list[object]
    audio_tracks: list[object]
    audio_lead_ms: float
    recording_wall_time_ms: float
    last_control_sequence: int


def _record_media(
    args: argparse.Namespace,
    target_kind: str,
    capture_rect: dict,
) -> _RecordingArtifacts:
    """Record media and always release every native capture resource."""
    click_tracker = ClickTracker()
    mouse_tracker = MouseTracker(click_state_provider=click_tracker.is_button_down)
    recording_toggled = [False]
    pause_toggled = [False]

    def on_hotkey_triggered() -> None:
        recording_toggled[0] = True

    def on_pause_hotkey_triggered() -> None:
        pause_toggled[0] = True

    hotkey_tracker = GlobalHotkeys(
        callback=on_hotkey_triggered,
        pause_callback=on_pause_hotkey_triggered,
    )
    timeline_clock = SessionTimelineClock()
    recorder = ScreenRecorder(
        recording_finished_cb=lambda _path: None,
        capture_backend_changed_cb=lambda _backend: None,
        timeline_clock=timeline_clock,
    )
    audio_capture = AudioCapture(
        [str(args.microphone or ""), str(args.system_audio or "")]
    )
    overlay: RecordingOverlay | None = None
    raw_video_path = ""
    mouse_events: list[object] = []
    click_events: list[object] = []
    audio_tracks: list[object] = []
    audio_lead_ms = 0.0
    recording_wall_time_ms = 0.0
    last_control_sequence = 0
    capture_started = False
    recording_started = False
    mouse_started = False
    click_started = False
    audio_started = False
    hotkey_registered = False

    def cleanup(label: str, callback) -> None:
        try:
            callback()
        except Exception as exc:
            logging.warning("Could not clean up %s: %s", label, exc)

    try:
        logging.info("Starting %s capture at %s FPS...", target_kind, args.fps)
        capture_started = True
        if target_kind == "window":
            recorder.start_capture_window(int(args.window_hwnd), args.fps)
        elif target_kind == "region":
            recorder.start_capture_region(capture_rect, args.fps)
        else:
            recorder.start_capture(args.monitor, args.fps)

        raw_video_path = recorder.prepare_recording()
        audio_started = True
        audio_capture.start()
        if args.microphone or args.system_audio:
            time.sleep(0.25)
        session_epoch = time.perf_counter()
        audio_lead_ms = max(0.0, (session_epoch - audio_capture.started_at) * 1000.0)
        recording_started = True
        recorder.start_recording(
            session_epoch=session_epoch,
            timeline_clock=timeline_clock,
        )
        click_started = True
        click_tracker.start(session_epoch, capture_rect, timeline_clock=timeline_clock)
        mouse_started = True
        mouse_tracker.start(session_epoch, timeline_clock=timeline_clock)
        recording_wall_time_ms = time.time() * 1000.0
        if not args.stop_file:
            hotkey_registered = True
            hotkey_tracker.register_record_hotkey()

        overlay = RecordingOverlay(capture_rect)
        overlay.start()
        _write_status_payload(
            args.status_file,
            last_control_sequence,
            RecordingState.RECORDING,
            timeline_clock,
        )

        logging.info("Recording started. Outputting raw video to: %s", raw_video_path)
        if args.duration > 0:
            logging.info("Recording will stop automatically after %s seconds.", args.duration)
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
                    logging.info("Reached duration of %ss. Stopping recording...", args.duration)
                    break
                if recording_toggled[0]:
                    logging.info("Hotkey pressed. Stopping recording...")
                    break
                if pause_toggled[0]:
                    pause_toggled[0] = False
                    if recorder.is_paused:
                        if recorder.resume_recording():
                            overlay.set_paused(False)
                    elif recorder.pause_recording():
                        overlay.set_paused(True)
                    _write_status_payload(
                        args.status_file,
                        last_control_sequence,
                        recorder.recording_state,
                        timeline_clock,
                    )
        except KeyboardInterrupt:
            logging.info("Ctrl+C pressed. Stopping recording...")
    finally:
        failed = sys.exc_info()[0] is not None
        if recording_started:
            cleanup("video recording", recorder.stop_recording)
        if mouse_started:
            def stop_mouse() -> None:
                nonlocal mouse_events
                mouse_events = mouse_tracker.stop()

            cleanup("mouse tracker", stop_mouse)
        if click_started:
            def stop_clicks() -> None:
                nonlocal click_events
                click_events = click_tracker.stop()

            cleanup("click tracker", stop_clicks)
        if audio_started:
            def stop_audio() -> None:
                nonlocal audio_tracks
                audio_tracks = audio_capture.stop()

            cleanup("audio capture", stop_audio)
        if hotkey_registered:
            cleanup("recording hotkey", hotkey_tracker.unregister_record_hotkey)
        if overlay is not None:
            cleanup("recording overlay", overlay.stop)
        if capture_started:
            cleanup("screen capture", recorder.stop_capture)
        if args.stop_file:
            try:
                os.remove(args.stop_file)
            except OSError:
                pass
        if failed and audio_tracks:
            cleanup("temporary audio tracks", lambda: cleanup_audio_tracks(audio_tracks))

    return _RecordingArtifacts(
        recorder=recorder,
        timeline_clock=timeline_clock,
        raw_video_path=raw_video_path,
        mouse_events=mouse_events,
        click_events=click_events,
        audio_tracks=audio_tracks,
        audio_lead_ms=audio_lead_ms,
        recording_wall_time_ms=recording_wall_time_ms,
        last_control_sequence=last_control_sequence,
    )


def _run(args: argparse.Namespace) -> int:
    output_format = str(getattr(args, "output_format", "mp4") or "mp4").lower()
    if output_format not in {"mp4", "gif"}:
        return _failure(args.result_file, f"Unsupported recording format: {output_format}")
    expected_suffix = f".{output_format}"
    if os.path.splitext(os.path.abspath(args.out))[1].lower() != expected_suffix:
        return _failure(
            args.result_file,
            f"{output_format.upper()} output must use the {expected_suffix} extension",
        )
    if output_format == "gif":
        # GIF has no audio stream. Avoid starting audio capture or muxing work.
        args.microphone = ""
        args.system_audio = ""
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

    artifacts = _record_media(args, target_kind, capture_rect)
    recorder = artifacts.recorder
    timeline_clock = artifacts.timeline_clock
    raw_video_path = artifacts.raw_video_path
    mouse_events = artifacts.mouse_events
    click_events = artifacts.click_events
    audio_tracks = artifacts.audio_tracks
    audio_lead_ms = artifacts.audio_lead_ms
    recording_wall_time_ms = artifacts.recording_wall_time_ms
    last_control_sequence = artifacts.last_control_sequence

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
    unzoomed_path = ""
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

        source_before_render = publication_source
        render_path = ""
        try:
            smart_zoom_progress(0)
            render_handle = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            render_path = render_handle.name
            render_handle.close()
            try:
                os.remove(render_path)
            except OSError:
                pass
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
                if bool(getattr(args, "preserve_unzoomed", False)):
                    try:
                        unzoomed_path = preserve_unzoomed_recording(
                            source_before_render,
                            session_id,
                        )
                    except Exception as exc:
                        warnings.append(
                            "The original recording could not be retained for Smart Zoom removal."
                        )
                        logging.warning(
                            "Could not preserve the unzoomed recording draft: %s",
                            exc,
                        )
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
        except Exception as exc:
            publication_source = source_before_render
            if render_path:
                try:
                    os.remove(render_path)
                except OSError:
                    pass
            smart_zoom_manifest = {
                "state": "failed",
                "keyframes": [],
                "renderCursor": bool(args.render_cursor),
                "renderClicks": bool(args.render_clicks),
                "zoomLevel": smart_zoom_level,
                "error": str(exc),
            }
            warnings.append("Smart Zoom could not be applied; the unprocessed recording was saved.")
            logging.exception("Smart Zoom orchestration failed; preserving source video: %s", exc)

    if output_format == "gif":
        gif_handle = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
        gif_path = gif_handle.name
        gif_handle.close()
        try:
            os.remove(gif_path)
        except OSError:
            pass
        conversion_sequence = [last_control_sequence]

        def gif_cancelled() -> bool:
            sequence, action = _read_control_payload(
                args.control_file,
                conversion_sequence[0],
            )
            if action:
                conversion_sequence[0] = sequence
            return action == "cancel"

        def gif_progress(progress: int) -> None:
            _write_status_payload(
                args.status_file,
                conversion_sequence[0],
                RecordingState.PROCESSING,
                timeline_clock,
                phase="gif_export",
                progress=max(0, min(100, int(progress))),
            )

        video_source = publication_source
        gif_result = export_gif(
            video_source,
            gif_path,
            duration_ms,
            progress_callback=gif_progress,
            cancel_callback=gif_cancelled,
        )
        if gif_result.state != "processed" or not gif_result.output_path:
            discard_unzoomed_recording(unzoomed_path)
            message = (
                "GIF creation was cancelled."
                if gif_result.state == "cancelled"
                else f"Could not create GIF: {gif_result.error or 'unknown error'}"
            )
            recovery_path = os.path.abspath(video_source) if os.path.isfile(video_source) else ""
            if video_source != raw_video_path:
                try:
                    os.remove(raw_video_path)
                except OSError:
                    pass
            return _failure(args.result_file, message, recoveryPath=recovery_path)
        publication_source = gif_result.output_path
        if video_source != raw_video_path:
            try:
                os.remove(video_source)
            except OSError:
                pass

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

    logging.info("Recording stopped. Publishing %s to %s", output_format.upper(), output_path)
    try:
        published = publish_recording(publication_source, output_path, session)
    except Exception as exc:
        discard_unzoomed_recording(unzoomed_path)
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
    if unzoomed_path and os.path.isfile(unzoomed_path):
        payload["unzoomedPath"] = unzoomed_path
    _write_status_payload(
        args.status_file,
        last_control_sequence,
        RecordingState.FINISHED,
        timeline_clock,
        progress=100 if smart_zoom_manifest.get("state") == "processed" else 0,
    )
    if not _write_result_payload(args.result_file, payload):
        discard_unzoomed_recording(unzoomed_path)
        return 1

    for warning in warnings:
        logging.warning(warning)
    logging.info("Force exiting to release WGC hooks...")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=f"{PRODUCT_NAME} capture worker")
    parser.add_argument(
        "--out",
        "-o",
        required=True,
        help="Output media path (.mp4 or .gif)",
    )
    parser.add_argument("--monitor", "-m", type=int, default=1, help="Monitor index (default 1)")
    parser.add_argument("--fps", type=int, default=60, help="Recording FPS")
    parser.add_argument(
        "--output-format",
        choices=("mp4", "gif"),
        default="mp4",
        help="Published recording format (default: mp4)",
    )
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
        "--preserve-unzoomed",
        action="store_true",
        help="Retain a private unzoomed draft for the post-capture remove action",
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
