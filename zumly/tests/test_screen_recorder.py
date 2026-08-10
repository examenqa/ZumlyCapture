"""Unit tests for capture buffering and encoder selection."""

import threading
import time

import numpy as np
from queue import Queue

from app.screen_recorder import (
    CfrFrameScheduler,
    ScreenRecorder,
    _FrameBufferPool,
    _capture_encoder_args,
    _capture_record_fps,
    _bgra_frame_view,
    _mp4_output_path,
    _wait_for_capture_interval,
)
from app.click_tracker import ClickTracker
from app.mouse_tracker import MouseTracker
from app.session_timing import RecordingState, SessionTimelineClock


def test_capture_encoder_prefers_hardware_in_priority_order() -> None:
    """Capture should select the fastest available hardware encoder."""
    encoder, args = _capture_encoder_args(
        "ffmpeg",
        {"h264_amf", "h264_nvenc", "h264_qsv"},
    )

    assert encoder == "h264_nvenc"
    assert args[:2] == ["-c:v", "h264_nvenc"]


def test_capture_encoder_falls_back_to_libx264() -> None:
    """CPU encoding remains available when no GPU encoder is exposed."""
    encoder, args = _capture_encoder_args("ffmpeg", set())

    assert encoder == "libx264"
    assert args[:2] == ["-c:v", "libx264"]


def test_cpu_capture_uses_a_stable_30_fps_cadence() -> None:
    """A CPU encoder should not advertise a 60 FPS stream it cannot sustain."""
    assert _capture_record_fps(60, "libx264") == 30
    assert _capture_record_fps(30, "libx264") == 30
    assert _capture_record_fps(60, "h264_nvenc") == 60


def test_wgc_fifo_preserves_order_and_counts_true_overflow() -> None:
    """Only a full FIFO counts as a dropped WGC capture frame."""
    recorder = ScreenRecorder()
    recorder._recording = True
    frame = np.zeros((2, 2, 4), dtype=np.uint8)

    for timestamp in range(30):
        assert recorder._enqueue_wgc_frame(float(timestamp), frame)
    assert not recorder._enqueue_wgc_frame(30.0, frame)

    packets = [recorder._wgc_frame_queue.get_nowait() for _ in range(30)]
    telemetry = recorder.capture_telemetry

    assert [timestamp for timestamp, _ in packets] == [float(i) for i in range(30)]
    assert telemetry["sourceFrames"] == 31
    assert telemetry["overflowDrops"] == 1
    assert telemetry["maxQueueDepth"] == 30
    assert telemetry["queuedFrames"] == 0


def test_bgra_frame_view_avoids_a_second_byte_copy() -> None:
    frame = np.zeros((2, 3, 4), dtype=np.uint8)

    view = _bgra_frame_view(frame)

    assert view is not None
    assert view.nbytes == 2 * 3 * 4
    frame[0, 0, 0] = 255
    assert view[0] == 255
    assert _bgra_frame_view(frame[:, ::2]) is None


def test_frame_pool_uses_a_byte_budget_and_reuses_allocations() -> None:
    frame_bytes = 4 * 2 * 4
    pool = _FrameBufferPool(4, 2, byte_budget=frame_bytes * 3)

    assert pool.capacity == 3
    assert pool.allocated_bytes == frame_bytes * 3
    leases = [pool.checkout() for _ in range(3)]
    assert all(lease is not None for lease in leases)
    assert pool.checkout() is None

    source = np.arange(frame_bytes, dtype=np.uint8).reshape((2, 4, 4))
    assert leases[0].copy_from(source)
    assert bytes(_bgra_frame_view(leases[0])) == source.tobytes()

    for lease in leases:
        lease.release()
    assert pool.available == 3


def test_frame_pool_repacks_strided_wgc_rows_without_allocating_a_frame() -> None:
    pool = _FrameBufferPool(3, 2, byte_budget=3 * 2 * 4)
    lease = pool.checkout()
    assert lease is not None
    padded = np.arange(2 * 16, dtype=np.uint8).reshape((2, 16))
    source = padded[:, :12].reshape((2, 3, 4))
    assert source.flags.c_contiguous is False

    assert lease.copy_from(source)
    assert bytes(_bgra_frame_view(lease)) == source.tobytes()

    lease.release()
    assert pool.available == 1


def test_cfr_scheduler_returns_coalesced_and_superseded_pool_buffers() -> None:
    pool = _FrameBufferPool(1, 1, byte_budget=12)
    initial = pool.checkout()
    older = pool.checkout()
    freshest = pool.checkout()
    assert initial is not None and older is not None and freshest is not None

    frame_queue = Queue()
    frame_queue.put((-0.2, older))
    frame_queue.put((-0.1, freshest))
    scheduler = CfrFrameScheduler(epoch=0.0, target_fps=30)
    scheduler.last_frame = initial

    selected, stuffed = scheduler.select_frame_for_current_slot(
        frame_queue,
        boundary=0.0,
    )

    assert selected is freshest
    assert stuffed is False
    assert scheduler.coalesced_frames == 1
    assert pool.available == 2

    scheduler.close()
    assert pool.available == 3


def test_capture_wait_can_be_interrupted_without_busy_spinning() -> None:
    stop_event = threading.Event()
    assert _wait_for_capture_interval(stop_event, 0.001) is False

    stop_event.set()
    assert _wait_for_capture_interval(stop_event, 1.0) is True


def test_capture_telemetry_exposes_hot_path_metrics() -> None:
    recorder = ScreenRecorder()
    recorder._backend = "WGC"
    recorder._capture_encoder = "h264_nvenc"
    recorder._record_writer_write(9.0)
    recorder._record_wgc_callback(1.5, 2.0)

    telemetry = recorder.capture_telemetry

    assert telemetry["captureBackend"] == "WGC"
    assert telemetry["captureEncoder"] == "h264_nvenc"
    assert telemetry["writerStallCount"] == 1
    assert telemetry["wgcSnapshotMs"] == 1.5
    assert telemetry["wgcCallbackMs"] == 2.0
    assert telemetry["zeroCopyPipeWrites"] == 1


def test_cfr_scheduler_coalesces_backlog_without_skipping_slots() -> None:
    """A late writer keeps every output slot while choosing the freshest input."""
    frame_queue: Queue[tuple[float, str]] = Queue()
    scheduler = CfrFrameScheduler(epoch=100.0, target_fps=10)
    frame_queue.put((100.01, "initial"))
    initial = scheduler.wait_for_initial_packet(frame_queue, timeout=0.01)
    assert initial == (100.01, "initial")

    scheduler.last_frame = initial[1]
    scheduler.selected_frames = 1
    scheduler.advance()
    frame_queue.put((100.05, "older"))
    frame_queue.put((100.08, "fresh"))
    frame_queue.put((100.12, "future"))

    first = scheduler.select_frame_for_current_slot(frame_queue)
    assert first == ("fresh", False)
    assert scheduler.coalesced_frames == 1
    assert scheduler.output_timestamp_ms() == 100.0
    scheduler.advance()

    second = scheduler.select_frame_for_current_slot(frame_queue)
    assert second == ("future", False)
    assert scheduler.output_timestamp_ms() == 200.0
    scheduler.advance()

    assert scheduler.select_frame_for_current_slot(frame_queue) == ("future", True)
    assert scheduler.stuffed_frames == 1


def test_cfr_scheduler_initial_frame_timeout_is_bounded() -> None:
    scheduler = CfrFrameScheduler(epoch=0.0, target_fps=30)
    assert scheduler.wait_for_initial_packet(Queue(), timeout=0.01) is None


def test_mp4_output_path_preserves_dotted_parent_directory() -> None:
    assert _mp4_output_path(r"C:\Users\my.name\video") == r"C:\Users\my.name\video.mp4"


def test_failed_recording_becomes_non_cfr_and_finalizes() -> None:
    recorder = ScreenRecorder()
    recorder._recording = True
    recorder._is_cfr = True
    recorder._recording_finalized_event.clear()

    recorder._mark_recording_failed("writer stopped")

    assert not recorder.is_recording
    assert not recorder.is_cfr
    assert recorder.recording_error == "writer stopped"
    assert recorder._recording_finalized_event.is_set()


def test_recording_uses_supplied_monotonic_epoch() -> None:
    recorder = ScreenRecorder()
    recorder._output_path = "prepared.mp4"
    recorder._recording_prepared = True

    assert recorder.start_recording(session_epoch=123.456) == "prepared.mp4"
    assert recorder._start_time == 123.456
    assert recorder._perf_start == 123.456
    assert recorder.is_recording is True


def test_pause_resume_keeps_session_open_and_clears_stale_wgc_packets() -> None:
    clock = SessionTimelineClock()
    recorder = ScreenRecorder(timeline_clock=clock)
    recorder._output_path = "prepared.mp4"
    recorder._recording_prepared = True
    recorder.start_recording(session_epoch=100.0, timeline_clock=clock)
    recorder._wgc_frame_queue.put((100.01, "frame"))

    assert recorder.pause_recording()
    assert recorder.is_recording is True
    assert recorder.is_paused is True
    assert recorder.recording_state == RecordingState.PAUSED
    assert recorder._scheduler_wake_event.is_set()
    assert recorder._wgc_frame_queue.empty()
    assert not recorder._enqueue_wgc_frame(101.0, "paused-frame")

    assert recorder.resume_recording()
    assert recorder.recording_state == RecordingState.RECORDING
    assert recorder.is_recording is True
    assert not recorder._scheduler_wake_event.is_set()


def test_click_tracker_discards_points_outside_recorded_monitor() -> None:
    tracker = ClickTracker()
    tracker._monitor_rect = {"left": -1920, "top": 0, "width": 1920, "height": 1080}

    tracker._on_click(-100, 500, 100.0)
    tracker._on_click(50, 500, 200.0)

    assert [(event.x, event.y, event.timestamp) for event in tracker.events] == [
        (-100.0, 500.0, 100.0)
    ]


def test_click_tracker_routes_hook_events_to_state_and_click_telemetry() -> None:
    tracker = ClickTracker()
    tracker._monitor_rect = {"left": 0, "top": 0, "width": 1920, "height": 1080}

    tracker._on_mouse_button(400, 300, 100.0, "left", True)
    assert tracker.is_button_down() is True
    tracker._on_mouse_button(400, 300, 135.0, "left", False)

    assert tracker.is_button_down() is False
    assert [(event.x, event.y, event.timestamp) for event in tracker.events] == [
        (400.0, 300.0, 100.0)
    ]


def test_click_tracker_discards_events_and_button_state_during_pause() -> None:
    clock = SessionTimelineClock()
    clock.start(100.0)
    tracker = ClickTracker()
    tracker._timeline_clock = clock
    tracker._monitor_rect = {"left": 0, "top": 0, "width": 1920, "height": 1080}

    clock.pause(101.0)
    tracker._on_mouse_button(400, 300, 1000.0, "left", True)

    assert tracker.events == []
    assert tracker.is_button_down() is False


def test_mouse_tracker_flags_first_sample_after_resume(monkeypatch) -> None:
    monkeypatch.setattr("app.mouse_tracker._get_physical_cursor_pos", lambda: (320, 240))
    clock = SessionTimelineClock()
    clock.start()
    tracker = MouseTracker(interval_ms=2)
    tracker.start(timeline_clock=clock)
    time.sleep(0.02)
    clock.pause()
    time.sleep(0.01)
    clock.resume()
    time.sleep(0.02)

    samples = tracker.stop()

    boundaries = [sample for sample in samples if sample.resume_boundary]
    assert len(boundaries) == 1
    assert boundaries[0].x == 320
    assert boundaries[0].y == 240


def test_stop_capture_forces_stalled_writer_with_bounded_join() -> None:
    class NeverFinalized:
        def __init__(self) -> None:
            self.set_called = False

        def is_set(self) -> bool:
            return False

        def wait(self, timeout: float) -> bool:
            assert timeout <= 4.0
            return False

        def set(self) -> None:
            self.set_called = True

    class StalledWriter:
        pid = 42

        def __init__(self) -> None:
            self.killed = False

        def poll(self):
            return None if not self.killed else -9

        def kill(self) -> None:
            self.killed = True

        def wait(self, timeout: float = 0.0) -> int:
            assert timeout <= 1.0
            return -9

    class CaptureThread:
        def __init__(self) -> None:
            self.join_timeout = None

        def is_alive(self) -> bool:
            return False

        def join(self, timeout: float) -> None:
            self.join_timeout = timeout

    recorder = ScreenRecorder()
    finalized = NeverFinalized()
    writer = StalledWriter()
    thread = CaptureThread()
    recorder._recording_finalized_event = finalized
    recorder._writer_proc = writer
    recorder._thread = thread

    recorder.stop_capture()

    assert writer.killed is True
    assert finalized.set_called is True
    assert thread.join_timeout is not None
    assert thread.join_timeout <= 5.0
    assert recorder.is_capturing is False
    assert recorder.recording_error == "Capture shutdown timed out; FFmpeg was terminated."
