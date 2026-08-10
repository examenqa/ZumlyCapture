import argparse
import json
import logging
import os
import sys
import tempfile
import time

from zumly_capture.identity import FILE_PREFIX, PRODUCT_NAME, RUNTIME_DIRECTORY_NAME

from zumly.app.screen_recorder import ScreenRecorder
from zumly.app.mouse_tracker import MouseTracker
from zumly.app.click_tracker import ClickTracker
from zumly.app.keyboard_tracker import KeyboardTracker
from zumly.app.global_hotkeys import GlobalHotkeys
from zumly.app.session_timing import RecordingState, SessionTimelineClock
from zumly.app.activity_analyzer import analyze_activity

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


def _capture_project_bridge_path(recording_wall_time_ms: float, session_id: str) -> str:
    """Return an ephemeral project bridge path for the tray/editor handoff.

    The bridge is an implementation detail of the three-process architecture,
    not a user-facing export. Keeping it under the runtime temp directory keeps
    the configured video output folder limited to media the user asked for.
    """
    bridge_dir = os.path.join(
        tempfile.gettempdir(),
        RUNTIME_DIRECTORY_NAME,
        "bridges",
    )
    os.makedirs(bridge_dir, exist_ok=True)
    timestamp = int(recording_wall_time_ms)
    safe_session_id = "".join(char for char in str(session_id) if char.isalnum())
    return os.path.join(bridge_dir, f"capture_{timestamp}_{safe_session_id}_project.json")


def _write_capture_project_bridge(path: str, payload: dict) -> None:
    """Atomically publish the recorder-to-editor project bridge."""
    directory = os.path.dirname(path)
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=directory,
            prefix=f"{FILE_PREFIX}_session_",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = handle.name
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise


def _failure(result_file: str, message: str, code: int = 1) -> int:
    """Write a terminal failure payload and return the process exit code."""
    logging.error(message)
    _write_result_payload(
        result_file,
        {"status": "failed", "error": str(message), "returnCode": int(code)},
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
    if sequence <= last_sequence or action not in {"pause", "resume", "stop"}:
        return last_sequence, ""
    return sequence, action


def _write_status_payload(
    path: str,
    sequence: int,
    state: RecordingState,
    clock: SessionTimelineClock,
) -> None:
    _write_result_payload(
        path,
        {
            "sequence": int(sequence),
            "state": state.value,
            "activeDurationMs": round(clock.active_time_ms(), 3),
            "pausedDurationMs": round(clock.paused_duration_ms, 3),
        },
    )


def _run(args: argparse.Namespace) -> int:
    # Determine monitor dimensions
    monitor_rect = {}
    for mon in ScreenRecorder.get_monitors():
        if mon["index"] == args.monitor:
            monitor_rect = mon
            break

    if not monitor_rect:
        return _failure(args.result_file, f"Could not find monitor with index {args.monitor}")

    # Trackers
    click_tracker = ClickTracker()
    mouse_tracker = MouseTracker(click_state_provider=click_tracker.is_button_down)
    kbd_tracker = KeyboardTracker()

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

    logging.info(f"Starting capture on monitor {args.monitor} at {args.fps} FPS...")
    recorder.start_capture(args.monitor, args.fps)

    # Let capture spin up
    time.sleep(2.0)

    # Complete all potentially blocking preparation before defining time zero.
    raw_video_path = recorder.prepare_recording()
    SESSION_EPOCH = time.perf_counter()
    recorder.start_recording(
        session_epoch=SESSION_EPOCH,
        timeline_clock=timeline_clock,
    )
    click_tracker.start(SESSION_EPOCH, monitor_rect, timeline_clock=timeline_clock)
    mouse_tracker.start(SESSION_EPOCH, timeline_clock=timeline_clock)
    kbd_tracker.start(SESSION_EPOCH)
    recording_wall_time_ms = time.time() * 1000.0
    if not args.stop_file:
        hotkey_tracker.register_record_hotkey()

    from zumly.app.recording_overlay import RecordingOverlay

    overlay = RecordingOverlay(monitor_rect)
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
    kbd_events = kbd_tracker.stop()
    hotkey_tracker.unregister_record_hotkey()
    overlay.stop()
    recorder.stop_capture()
    _write_status_payload(
        args.status_file,
        last_control_sequence,
        RecordingState.FINISHED,
        timeline_clock,
    )
    if args.stop_file:
        try:
            os.remove(args.stop_file)
        except OSError:
            pass

    if recorder.recording_error:
        return _failure(
            args.result_file,
            f"Recording failed before a usable video was written: {recorder.recording_error}",
        )

    logging.info("Recording stopped. Generating AI auto-zooms...")
    keyframes = analyze_activity(
        mouse_track=mouse_events,
        monitor_rect=monitor_rect,
        key_events=kbd_events,
        click_events=click_events,
    )

    logging.info(f"Generated {len(keyframes)} zoom keyframes. Saving session state...")

    import uuid
    from zumly.app.models import RecordingSession

    session_id = str(uuid.uuid4())
    duration_ms = recorder.recording_duration_ms

    session = RecordingSession(
        id=session_id,
        start_time=recording_wall_time_ms / 1000.0,
        duration=duration_ms,
        mouse_track=mouse_events,
        keyframes=keyframes,
        click_events=click_events,
        frame_timestamps=recorder.frame_timestamps,
        is_cfr=recorder.is_cfr,
        capture_telemetry=recorder.capture_telemetry,
    )

    data = json.loads(session.to_json())
    data["monitorRect"] = monitor_rect
    data["actualFps"] = recorder.actual_fps
    data["videoPath"] = raw_video_path
    data["outPath"] = args.out

    project_path = _capture_project_bridge_path(recording_wall_time_ms, session_id)
    _write_capture_project_bridge(project_path, data)

    payload = {
        "status": "success",
        "projectPath": os.path.abspath(project_path),
        "videoPath": os.path.abspath(raw_video_path),
        "outputPath": os.path.abspath(args.out),
        "sessionId": session_id,
        "durationMs": duration_ms,
        "returnCode": 0,
    }
    if not _write_result_payload(args.result_file, payload):
        return 1

    logging.info(f"Session serialized to {project_path}")
    logging.info("Force exiting to release WGC hooks...")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=f"{PRODUCT_NAME} capture worker")
    parser.add_argument("--out", "-o", required=True, help="Output MP4 path")
    parser.add_argument("--monitor", "-m", type=int, default=1, help="Monitor index (default 1)")
    parser.add_argument("--fps", type=int, default=60, help="Recording FPS")
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
        help="Atomic sequenced pause/resume/stop command payload",
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
