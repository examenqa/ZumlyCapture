"""Screen / window capture engine.

Captures a monitor or individual window at up to 60 fps using
Windows Graphics Capture (hardware-accelerated) with a GDI/mss
fallback.  During recording, raw BGRA frames are piped to ffmpeg
for lossless AVI encoding.  Non-recording mode emits QImage preview
frames via the ``frame_ready`` signal.
"""

from __future__ import annotations

import logging
import time
import threading
import tempfile
import os
import subprocess
from pathlib import Path
from functools import lru_cache
from queue import Empty, Full, Queue

logger = logging.getLogger(__name__)
import ctypes
from typing import Any, Optional, List, TYPE_CHECKING, Callable

# Windows Graphics Capture API (hardware-accelerated capture)
try:
    from windows_capture import WindowsCapture, Frame, InternalCaptureControl
    _HAS_WGC = True
except ImportError:
    _HAS_WGC = False

# Windows high-resolution timer
try:
    _winmm = ctypes.windll.winmm
except AttributeError:
    _winmm = None


from .utils import (
    detect_available_encoders,
    ffmpeg_exe as _ffmpeg_exe,
    subprocess_kwargs as _subprocess_kwargs,
)
from .session_timing import RecordingState, SessionTimelineClock


WGC_BUFFER_BUDGET_BYTES = 100 * 1024 * 1024


@lru_cache(maxsize=2)
def _available_capture_encoders(ffmpeg: str) -> set[str]:
    """Return encoders verified by the shared FFmpeg capability registry."""
    try:
        return set(detect_available_encoders())
    except Exception as exc:
        logger.debug("Unable to inspect FFmpeg encoders: %s", exc)
        return set()


def _capture_encoder_args(
    ffmpeg: str,
    available_encoders: Optional[set[str]] = None,
) -> tuple[str, list[str]]:
    """Select the fastest available real-time H.264 encoder.

    Raw BGRA frames arrive over stdin, so ``-hwaccel`` is not applicable to
    this input. Hardware acceleration is selected at the encoder output.
    """
    available = (
        _available_capture_encoders(ffmpeg)
        if available_encoders is None
        else available_encoders
    )
    if "h264_nvenc" in available:
        return "h264_nvenc", [
            "-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ull",
            "-rc", "vbr", "-cq", "18", "-b:v", "0",
        ]
    if "h264_qsv" in available:
        return "h264_qsv", [
            "-c:v", "h264_qsv", "-preset", "veryfast", "-global_quality", "18",
        ]
    if "h264_amf" in available:
        return "h264_amf", [
            "-c:v", "h264_amf", "-quality", "speed", "-rc", "cqp",
            "-qp_i", "18", "-qp_p", "18",
        ]
    return "libx264", [
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
        "-tune", "zerolatency",
    ]


def _capture_record_fps(requested_fps: int, encoder_name: str) -> int:
    """Keep CPU-only capture at a stable cadence instead of uneven stuffing."""
    requested = max(1, int(requested_fps))
    return min(requested, 30) if encoder_name == "libx264" else requested


def _mp4_output_path(output_path: str) -> str:
    """Return an MP4 path without treating dots in parent folders as a suffix."""
    return str(Path(output_path).with_suffix(".mp4"))


def _wait_for_capture_interval(stop_event: threading.Event, seconds: float) -> bool:
    """Yield the GIL while waiting for a capture slot or shutdown request."""
    if seconds <= 0:
        return stop_event.is_set()
    return stop_event.wait(timeout=seconds)


def _start_ffmpeg_writer(
    out_path: str, w: int, h: int, fps: int,
) -> Optional[subprocess.Popen]:
    """Launch an ffmpeg subprocess that accepts raw BGRA on stdin → MP4.

    Uses H.264 (CRF 18, ultrafast) in an MP4 container for
    near-lossless quality at a fraction of the file size compared to
    raw/huffyuv codecs.  MP4 properly handles H.264 NAL unit framing,
    avoiding the macroblock corruption that can occur with AVI.

    Returns the Popen object, or None if ffmpeg couldn't start.
    """
    try:
        ffmpeg = _ffmpeg_exe()
        encoder_name, encoder_args = _capture_encoder_args(ffmpeg)
        logger.info("Capture encoder: %s", encoder_name)
        cmd = [
            ffmpeg,
            "-y",                        # overwrite
            "-f", "rawvideo",
            "-pix_fmt", "bgra",
            "-s", f"{w}x{h}",
            "-r", str(fps),
            "-i", "pipe:0",
            *encoder_args,
            "-pix_fmt", "yuv420p",       # standard pixel format
            "-movflags", "+frag_keyframe+empty_moov",  # fragmented MP4: always valid
            out_path,
        ]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,   # avoid buffer deadlock on Windows
            **_subprocess_kwargs(),
        )
        # Give ffmpeg a moment to fail on bad args
        import time as _t
        _t.sleep(0.05)
        if proc.poll() is not None:
            logger.error("ffmpeg exited immediately (rc=%s)", proc.returncode)
            return None
        return proc
    except Exception as exc:
        logger.error("ffmpeg pipe launch failed: %s", exc)
        return None


def _warm_ffmpeg_capture_pipeline(w: int, h: int, fps: int) -> None:
    """Warm FFmpeg and the selected encoder before the user starts recording."""
    warm_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    warm_path = warm_file.name
    warm_file.close()
    writer: Optional[subprocess.Popen] = None
    try:
        writer = _start_ffmpeg_writer(warm_path, w, h, fps)
        if writer is None or not writer.stdin or writer.stdin.closed:
            return
        started = time.perf_counter()
        writer.stdin.write(bytes(w * h * 4))
        logger.info(
            "FFmpeg capture pipeline warmed in %.0f ms",
            (time.perf_counter() - started) * 1000.0,
        )
    except (BrokenPipeError, OSError) as exc:
        logger.debug("FFmpeg capture warm-up failed: %s", exc)
    finally:
        _stop_ffmpeg_writer(writer)
        try:
            os.remove(warm_path)
        except OSError:
            pass


def _stop_ffmpeg_writer(proc: Optional[subprocess.Popen]) -> None:
    """Cleanly close an ffmpeg writer subprocess.
    
    Proper cleanup sequence:
    1. Close stdin to signal EOF
    2. Wait with timeout for clean exit
    3. Kill if timeout expires
    4. Log all cleanup stages
    """
    if proc is None:
        return
    
    # Step 1: Close stdin to signal end of stream. Closing a full pipe can
    # itself block while the encoder is stalled, so perform it on a bounded
    # helper thread and fall through to process termination if necessary.
    if proc.stdin and not proc.stdin.closed:
        def close_stdin() -> None:
            try:
                proc.stdin.close()
            except OSError as exc:
                logger.debug("Failed to close ffmpeg stdin (pid=%s): %s", proc.pid, exc)

        closer = threading.Thread(target=close_stdin, daemon=True, name="FFmpegStdinCloser")
        closer.start()
        closer.join(timeout=0.5)
        if closer.is_alive():
            logger.warning("ffmpeg stdin close stalled (pid=%s); forcing termination", proc.pid)
        else:
            logger.debug("Closed ffmpeg stdin (pid=%s)", proc.pid)
    
    # Step 2: Wait for clean exit
    try:
        proc.wait(timeout=3)
        logger.debug("ffmpeg exited cleanly (pid=%s, rc=%s)", proc.pid, proc.returncode)
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg wait timeout (pid=%s), forcing termination", proc.pid)
        # Step 3: Kill if timeout expires
        try:
            proc.kill()
            proc.wait(timeout=1)
            logger.info("ffmpeg killed after timeout (pid=%s)", proc.pid)
        except subprocess.TimeoutExpired:
            logger.error("ffmpeg still alive after kill+1s (pid=%s)", proc.pid)
        except Exception as e:
            logger.error("Failed to kill ffmpeg (pid=%s): %s", proc.pid, e)
    except Exception as e:
        logger.error("Error waiting for ffmpeg (pid=%s): %s", proc.pid, e)
        # Ensure process is killed even on unexpected errors
        if proc.poll() is None:
            try:
                proc.kill()
                proc.wait(timeout=1)
            except Exception:
                logger.error("Failed to kill ffmpeg after wait error (pid=%s)", proc.pid)
    
    # Log non-zero exit codes
    if proc.returncode and proc.returncode != 0:
        logger.warning("ffmpeg exited with rc=%s (pid=%s)", proc.returncode, proc.pid)


def _bgra_frame_view(frame_bgra: Any) -> memoryview | None:
    """Expose a contiguous BGRA frame for direct pipe writing.

    WGC frames are copied once before they leave the native callback because
    the source pointer becomes invalid afterwards. This helper avoids the
    former second full-frame ``tobytes()`` allocation on the writer path.
    """
    if isinstance(frame_bgra, _PooledBgraFrame):
        return frame_bgra.view
    try:
        view = memoryview(frame_bgra)
        if not view.c_contiguous:
            return None
        return view.cast("B")
    except (TypeError, ValueError):
        return None


def _release_frame_buffer(frame_bgra: Any) -> None:
    """Return a pooled frame lease; ordinary frame objects are untouched."""
    release = getattr(frame_bgra, "release", None)
    if callable(release):
        release()


class _PooledBgraFrame:
    """One fixed-size BGRA allocation leased from a frame-buffer pool."""

    __slots__ = (
        "_pool",
        "_storage",
        "_view",
        "_array",
        "_leased",
        "width",
        "height",
    )

    def __init__(self, pool: "_FrameBufferPool", width: int, height: int) -> None:
        self._pool = pool
        self.width = int(width)
        self.height = int(height)
        self._storage = bytearray(self.width * self.height * 4)
        self._view = memoryview(self._storage)
        # windows-capture exposes padded D3D rows as a strided NumPy view at
        # some resolutions. Keep one ndarray header per pooled allocation so
        # np.copyto can repack those rows in C without allocating a frame.
        import numpy as np

        self._array = np.frombuffer(self._storage, dtype=np.uint8).reshape(
            self.height,
            self.width,
            4,
        )
        self._leased = False

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.height, self.width, 4

    @property
    def view(self) -> memoryview:
        return self._view

    def copy_from(self, source: Any) -> bool:
        try:
            source_view = memoryview(source)
        except (TypeError, ValueError, BufferError):
            return False
        if source_view.nbytes != self._view.nbytes:
            return False
        if source_view.c_contiguous:
            try:
                self._view[:] = source_view.cast("B")
                return True
            except (TypeError, ValueError, BufferError):
                return False

        try:
            import numpy as np

            if getattr(source, "shape", None) != self.shape:
                return False
            np.copyto(self._array, source, casting="no")
            return True
        except (TypeError, ValueError):
            return False

    def release(self) -> None:
        self._pool.release(self)


class _FrameBufferPool:
    """Preallocate a resolution-aware, byte-budgeted set of BGRA frames."""

    def __init__(
        self,
        width: int,
        height: int,
        byte_budget: int = WGC_BUFFER_BUDGET_BYTES,
    ) -> None:
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.frame_bytes = self.width * self.height * 4
        self.byte_budget = max(self.frame_bytes, int(byte_budget))
        self.capacity = max(1, self.byte_budget // self.frame_bytes)
        self.allocated_bytes = self.capacity * self.frame_bytes
        self._lease_lock = threading.Lock()
        self._free: Queue[_PooledBgraFrame] = Queue(maxsize=self.capacity)
        for _ in range(self.capacity):
            self._free.put_nowait(_PooledBgraFrame(self, self.width, self.height))

    @property
    def available(self) -> int:
        return self._free.qsize()

    def checkout(self) -> _PooledBgraFrame | None:
        try:
            frame = self._free.get_nowait()
        except Empty:
            return None
        with self._lease_lock:
            frame._leased = True
        return frame

    def release(self, frame: _PooledBgraFrame) -> None:
        with self._lease_lock:
            if not frame._leased:
                return
            frame._leased = False
        try:
            self._free.put_nowait(frame)
        except Full:
            logger.error("WGC frame pool received a duplicate buffer release")


class CfrFrameScheduler:
    """Select exactly one source frame for each absolute output frame slot.

    The WGC callback is intentionally unaware of output cadence.  It only
    appends timestamped packets to a FIFO.  This scheduler owns the one and
    only timing clock used by the FFmpeg writer, which prevents source-frame
    throttling and independently scheduled stuffing from drifting apart.
    """

    INITIAL_FRAME_TIMEOUT_S = 3.0

    def __init__(
        self,
        epoch: float,
        target_fps: int,
        timeline_clock: SessionTimelineClock | None = None,
    ) -> None:
        self.epoch = epoch
        self.timeline_clock = timeline_clock
        self.target_fps = max(1, int(target_fps))
        self.frame_interval = 1.0 / self.target_fps
        self.current_slot_index = 0
        self.last_frame: Any = None
        self._pending_packet: tuple[float, Any] | None = None
        self.selected_frames = 0
        self.coalesced_frames = 0
        self.stuffed_frames = 0

    def slot_time(self, slot_index: int | None = None) -> float:
        """Return the absolute performance-counter boundary for a slot."""
        index = self.current_slot_index if slot_index is None else slot_index
        if self.timeline_clock is not None:
            return self.timeline_clock.wall_deadline(index / self.target_fps)
        return self.epoch + (index / self.target_fps)

    def output_timestamp_ms(self) -> float:
        """Return the deterministic session timestamp for the current slot."""
        return self.current_slot_index * 1000.0 / self.target_fps

    def wait_for_initial_packet(
        self,
        frame_queue: Queue[tuple[float, Any]],
        timeout: float | None = None,
    ) -> tuple[float, Any] | None:
        """Wait once for a defined first frame without risking a deadlock."""
        try:
            return frame_queue.get(
                timeout=self.INITIAL_FRAME_TIMEOUT_S if timeout is None else timeout
            )
        except Empty:
            return None

    def select_frame_for_current_slot(
        self,
        frame_queue: Queue[tuple[float, Any]],
        *,
        boundary: float | None = None,
    ) -> tuple[Any, bool] | None:
        """Return the newest packet at or before this slot's boundary.

        A packet later than the boundary is retained for its future slot.
        When no eligible source packet exists, the previous output frame is
        duplicated to preserve the CFR timeline.  ``True`` means a duplicate.
        """
        slot_boundary = self.slot_time() if boundary is None else boundary
        newest: tuple[float, Any] | None = None

        def consider(packet: tuple[float, Any]) -> bool:
            nonlocal newest
            if packet[0] > slot_boundary:
                self._pending_packet = packet
                return False
            if newest is not None:
                self.coalesced_frames += 1
                _release_frame_buffer(newest[1])
            newest = packet
            return True

        if self._pending_packet is not None:
            pending = self._pending_packet
            self._pending_packet = None
            consider(pending)

        while self._pending_packet is None:
            try:
                packet = frame_queue.get_nowait()
            except Empty:
                break
            if not consider(packet):
                break

        if newest is not None:
            previous = self.last_frame
            self.last_frame = newest[1]
            if previous is not None and previous is not self.last_frame:
                _release_frame_buffer(previous)
            self.selected_frames += 1
            return self.last_frame, False
        if self.last_frame is None:
            return None
        self.stuffed_frames += 1
        return self.last_frame, True

    def advance(self) -> None:
        self.current_slot_index += 1

    def discard_pending_packet(self) -> None:
        """Drop a pre-pause packet retained for a future frame slot."""
        if self._pending_packet is not None:
            _release_frame_buffer(self._pending_packet[1])
        self._pending_packet = None

    def close(self) -> None:
        """Release the pending and retained CFR frame leases."""
        self.discard_pending_packet()
        if self.last_frame is not None:
            _release_frame_buffer(self.last_frame)
            self.last_frame = None


class ScreenRecorder:
    """Captures a monitor, emits preview frames, and optionally records to file."""

    def __init__(self,
                 recording_finished_cb: Optional[Callable[[str], None]] = None,
                 capture_backend_changed_cb: Optional[Callable[[str], None]] = None,
                 timeline_clock: SessionTimelineClock | None = None) -> None:
        self._recording_finished_cb = recording_finished_cb
        self._capture_backend_changed_cb = capture_backend_changed_cb
        self._monitor_index: int = 1
        self._capturing: bool = False
        self._recording: bool = False
        self._recording_prepared: bool = False
        self._output_path: str = ""
        self._thread: Optional[threading.Thread] = None
        self._writer_proc: Optional[subprocess.Popen] = None
        self._fps: int = 30
        self._start_time: float = 0.0
        self._perf_start: float = 0.0
        self._actual_fps: float = 30.0
        self._frame_count: int = 0
        self._frame_timestamps: List[float] = []  # ms offset per frame
        self._recording_duration_ms: float = 0.0
        self._lock = threading.Lock()
        # window capture mode
        self._capture_mode: str = "monitor"  # "monitor" | "window"
        self._window_hwnd: int = 0
        self._initial_size: tuple = (0, 0)
        self._backend: str = ""  # set once capture starts
        # WGC callback -> writer handoff. A FIFO keeps short encoder stalls
        # from discarding active frames; full is explicitly counted as a drop.
        self._wgc_frame_queue: Queue[tuple[float, Any]] = Queue(maxsize=30)
        self._wgc_buffer_pool: _FrameBufferPool | None = None
        self._wgc_pool_capacity: int = 0
        self._wgc_pool_bytes: int = 0
        self._wgc_first_frame_event = threading.Event()
        self._capture_pipeline_ready_event = threading.Event()
        self._capture_pipeline_ready_event.set()
        self._recording_finalized_event = threading.Event()
        self._recording_finalized_event.set()
        self._capture_stop_event = threading.Event()
        self._scheduler_wake_event = threading.Event()
        self._wgc_control = None
        self._frames_captured: int = 0
        self._frames_written: int = 0
        self._frames_stuffed: int = 0
        self._frames_dropped: int = 0
        self._frames_coalesced: int = 0
        self._max_queue_depth: int = 0
        self._writer_stall_ms: float = 0.0
        self._writer_stall_count: int = 0
        self._writer_write_max_ms: float = 0.0
        self._wgc_copy_ms: float = 0.0
        self._wgc_copy_max_ms: float = 0.0
        self._wgc_callback_ms: float = 0.0
        self._wgc_callback_max_ms: float = 0.0
        self._dimension_dropped_frames: int = 0
        self._zero_copy_pipe_writes: int = 0
        self._capture_encoder: str = ""
        self._target_fps: int = 30
        self._is_cfr: bool = True
        self._validation_error: str = ""
        self._recording_error: str = ""
        self._record_stop_perf: float = 0.0
        self._timeline_clock = timeline_clock or SessionTimelineClock()

    # ── static helpers ──────────────────────────────────────────────

    @staticmethod
    def get_monitors() -> List[dict]:
        """Return a list of available monitors with dimensions and positions."""
        import mss
        with mss.mss() as sct:
            monitors: List[dict] = []
            for i, m in enumerate(sct.monitors):
                if i == 0:  # "all monitors" virtual screen
                    continue
                monitors.append(
                    {
                        "index": i,
                        "name": f"Display {i}  ({m['width']}×{m['height']})",
                        "width": m["width"],
                        "height": m["height"],
                        "left": m["left"],
                        "top": m["top"],
                    }
                )
            return monitors

    # ── properties ──────────────────────────────────────────────────

    @property
    def is_capturing(self) -> bool:
        with self._lock:
            return self._capturing

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._recording

    @property
    def recording_state(self) -> RecordingState:
        return self._timeline_clock.state

    @property
    def is_paused(self) -> bool:
        return self._timeline_clock.is_paused

    @property
    def recording_duration_ms(self) -> float:
        with self._lock:
            if self._recording and self._start_time > 0:
                return self._timeline_clock.active_time_ms()
            return self._recording_duration_ms

    @property
    def actual_fps(self) -> float:
        """The encoded stream's intended cadence after finalization."""
        with self._lock:
            return self._actual_fps

    @property
    def is_cfr(self) -> bool:
        """Whether the final recording satisfied the CFR frame-count contract."""
        with self._lock:
            return self._is_cfr

    @property
    def recording_error(self) -> str:
        """A terminal recorder error, if the active recording could not be written."""
        with self._lock:
            return self._recording_error

    @property
    def frame_count(self) -> int:
        """Number of frames written to the pipe during the last recording."""
        with self._lock:
            return self._frame_count

    @property
    def frame_timestamps(self) -> List[float]:
        """Per-frame timestamps (ms from recording start) for each written frame."""
        with self._lock:
            return list(self._frame_timestamps)

    @property
    def capture_telemetry(self) -> dict[str, Any]:
        """Capture and scheduler counters retained with the session metadata."""
        with self._lock:
            pool = self._wgc_buffer_pool
            return {
                "sourceFrames": self._frames_captured,
                "selectedFrames": self._frames_written,
                "coalescedFrames": self._frames_coalesced,
                "stuffedFrames": self._frames_stuffed,
                "overflowDrops": self._frames_dropped,
                "encodedFrames": self._frame_count,
                "maxQueueDepth": self._max_queue_depth,
                "queuedFrames": self._wgc_frame_queue.qsize(),
                "framePoolCapacity": self._wgc_pool_capacity,
                "framePoolAvailable": (
                    pool.available if pool is not None else self._wgc_pool_capacity
                ),
                "framePoolBytes": self._wgc_pool_bytes,
                "targetFps": self._target_fps,
                "captureBackend": self._backend,
                "captureEncoder": self._capture_encoder,
                "writerStallMs": round(self._writer_stall_ms, 3),
                "writerStallCount": self._writer_stall_count,
                "writerWriteMaxMs": round(self._writer_write_max_ms, 3),
                "wgcSnapshotMs": round(self._wgc_copy_ms, 3),
                "wgcSnapshotMaxMs": round(self._wgc_copy_max_ms, 3),
                "wgcCallbackMs": round(self._wgc_callback_ms, 3),
                "wgcCallbackMaxMs": round(self._wgc_callback_max_ms, 3),
                "dimensionDroppedFrames": self._dimension_dropped_frames,
                "zeroCopyPipeWrites": self._zero_copy_pipe_writes,
                "validationPassed": self._is_cfr,
                "validationError": self._validation_error,
                "pauseCount": self._timeline_clock.pause_count,
                "pausedDurationMs": round(self._timeline_clock.paused_duration_ms, 3),
                "pauseBoundaries": [
                    boundary.to_dict()
                    for boundary in self._timeline_clock.pause_boundaries
                ],
            }

    @property
    def backend(self) -> str:
        """Current capture backend: 'DXGI' or 'GDI' (mss)."""
        with self._lock:
            return self._backend

    # ── public API ──────────────────────────────────────────────────

    def start_capture(self, monitor_index: int, fps: int = 60) -> None:
        """Begin capturing a monitor for live preview (no recording yet)."""
        self.stop_capture()
        self._capture_mode = "monitor"
        self._monitor_index = monitor_index
        self._fps = fps
        self._capture_stop_event.clear()
        self._scheduler_wake_event.clear()
        self._capture_pipeline_ready_event.clear()
        with self._lock:
            self._capturing = True
            self._recording = False
        # The headless capture worker has a bounded shutdown path; daemonizing
        # this last-resort thread prevents a native capture deadlock from
        # keeping the process alive after the guard expires.
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def start_capture_window(self, hwnd: int, fps: int = 60) -> None:
        """Start capturing a specific window by its handle."""
        self.stop_capture()
        self._capture_mode = "window"
        self._window_hwnd = hwnd
        self._fps = fps
        self._capture_stop_event.clear()
        self._scheduler_wake_event.clear()
        self._capture_pipeline_ready_event.clear()
        with self._lock:
            self._capturing = True
            self._recording = False
        from .window_utils import get_window_rect
        rect = get_window_rect(hwnd)
        if rect:
            self._initial_size = (rect["width"], rect["height"])
        else:
            self._initial_size = (0, 0)
        self._thread = threading.Thread(target=self._capture_loop_window, daemon=True)
        self._thread.start()

    def prepare_recording(self) -> str:
        """Prepare output and capture state without starting the session clock."""
        self._timeline_clock.prepare()
        self._scheduler_wake_event.clear()
        # Clean up previous temp recording file (if any) to avoid
        # accumulating large orphaned files in %TEMP%.
        with self._lock:
            old_path = self._output_path
        if old_path and os.path.isfile(old_path):
            try:
                os.remove(old_path)
                logger.info("Cleaned up previous temp recording: %s", old_path)
            except OSError:
                pass

        f = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_path = f.name
        f.close()
        if self.is_capturing and not self._capture_pipeline_ready_event.wait(timeout=5.0):
            logger.warning("Capture pipeline warm-up did not finish before recording started")
        self._clear_wgc_frame_queue()
        with self._lock:
            self._output_path = temp_path
            self._start_time = 0.0
            self._perf_start = 0.0
            self._frame_count = 0
            self._frame_timestamps = []
            self._recording_duration_ms = 0.0
            self._frames_captured = 0
            self._frames_written = 0
            self._frames_stuffed = 0
            self._frames_dropped = 0
            self._frames_coalesced = 0
            self._max_queue_depth = 0
            self._writer_stall_ms = 0.0
            self._writer_stall_count = 0
            self._writer_write_max_ms = 0.0
            self._wgc_copy_ms = 0.0
            self._wgc_copy_max_ms = 0.0
            self._wgc_callback_ms = 0.0
            self._wgc_callback_max_ms = 0.0
            self._dimension_dropped_frames = 0
            self._zero_copy_pipe_writes = 0
            self._capture_encoder = ""
            # Only the WGC scheduler may mark a recording as CFR after it
            # validates its frame-slot contract during finalization.
            self._is_cfr = False
            self._validation_error = ""
            self._recording_error = ""
            self._record_stop_perf = 0.0
            self._recording_finalized_event.clear()
            self._recording_prepared = True
            self._recording = False
        return temp_path

    def start_recording(
        self,
        session_epoch: float = 0.0,
        timeline_clock: SessionTimelineClock | None = None,
    ) -> str:
        """Start recording against one shared monotonic session epoch."""
        with self._lock:
            prepared = self._recording_prepared
        if not prepared:
            self.prepare_recording()

        if timeline_clock is not None and timeline_clock is not self._timeline_clock:
            self._timeline_clock = timeline_clock
        epoch = self._timeline_clock.start(session_epoch)
        self._scheduler_wake_event.clear()
        with self._lock:
            self._start_time = epoch
            self._perf_start = epoch
            self._recording_prepared = False
            self._recording = True
            return self._output_path

    def pause_recording(self) -> bool:
        """Freeze media time while leaving WGC and FFmpeg alive."""
        if not self._timeline_clock.pause():
            return False
        self._scheduler_wake_event.set()
        self._clear_wgc_frame_queue()
        logger.info("Recording paused at %.0f ms", self._timeline_clock.active_time_ms())
        return True

    def resume_recording(self) -> bool:
        """Resume media time without creating a new output stream."""
        self._clear_wgc_frame_queue()
        if not self._timeline_clock.resume():
            return False
        self._scheduler_wake_event.clear()
        logger.info("Recording resumed at %.0f ms", self._timeline_clock.active_time_ms())
        return True

    def stop_recording(self) -> str:
        """Stop recording and return the path to the raw MP4 file."""
        stopped_at = time.perf_counter()
        active_elapsed = self._timeline_clock.stop(stopped_at)
        self._scheduler_wake_event.set()
        with self._lock:
            self._recording_duration_ms = max(0.0, active_elapsed * 1000.0)
            self._record_stop_perf = stopped_at
            self._recording = False
            if self._backend != "WGC" and active_elapsed > 0 and self._frame_count > 0:
                self._actual_fps = self._frame_count / active_elapsed
            else:
                self._actual_fps = float(self._fps)
            return self._output_path

    def _enqueue_wgc_frame(self, captured_perf: float, frame_bgra: Any) -> bool:
        """Queue one WGC frame and report whether it fit in the FIFO."""
        with self._lock:
            if not self._recording or self._timeline_clock.state in {
                RecordingState.PAUSED,
                RecordingState.STOPPING,
                RecordingState.FINISHED,
            }:
                _release_frame_buffer(frame_bgra)
                return False
            self._frames_captured += 1

        try:
            self._wgc_frame_queue.put_nowait((captured_perf, frame_bgra))
            with self._lock:
                self._max_queue_depth = max(
                    self._max_queue_depth,
                    self._wgc_frame_queue.qsize(),
                )
            return True
        except Full:
            _release_frame_buffer(frame_bgra)
            self._record_wgc_overflow_drop(count_as_source=False)
            return False

    def _record_wgc_overflow_drop(self, *, count_as_source: bool = True) -> None:
        """Count an early pool/queue rejection without allocating a frame."""
        with self._lock:
            if count_as_source:
                self._frames_captured += 1
            self._frames_dropped += 1
            dropped = self._frames_dropped
        if dropped == 1 or dropped % 30 == 0:
            logger.warning(
                "WGC frame capacity exhausted; dropped %d captured frame(s)",
                dropped,
            )

    def _record_dimension_drop(
        self,
        width: int,
        height: int,
        expected_w: int,
        expected_h: int,
        *,
        count_as_source: bool = True,
    ) -> None:
        """Reject a resized live frame without introducing Python image work."""
        with self._lock:
            if count_as_source:
                self._frames_captured += 1
            self._dimension_dropped_frames += 1
            dropped = self._dimension_dropped_frames
        if dropped == 1 or dropped % 30 == 0:
            logger.warning(
                "Capture dimensions changed from %dx%d to %dx%d; dropping %d frame(s) "
                "until the source returns to the recording size.",
                expected_w,
                expected_h,
                width,
                height,
                dropped,
            )

    def _record_writer_write(self, elapsed_ms: float) -> None:
        """Track pipe backpressure separately from normal frame writes."""
        with self._lock:
            self._writer_stall_ms += elapsed_ms
            self._writer_write_max_ms = max(self._writer_write_max_ms, elapsed_ms)
            self._zero_copy_pipe_writes += 1
            if elapsed_ms >= 8.0:
                self._writer_stall_count += 1

    def _record_wgc_callback(self, copy_ms: float, callback_ms: float) -> None:
        with self._lock:
            self._wgc_copy_ms += copy_ms
            self._wgc_copy_max_ms = max(self._wgc_copy_max_ms, copy_ms)
            self._wgc_callback_ms += callback_ms
            self._wgc_callback_max_ms = max(self._wgc_callback_max_ms, callback_ms)

    def _log_capture_performance_summary(self) -> None:
        telemetry = self.capture_telemetry
        if not int(telemetry["encodedFrames"]) and not int(telemetry["sourceFrames"]):
            return
        logger.info(
            "Capture performance: backend=%s encoder=%s target_fps=%s source=%s "
            "encoded=%s wgc_copy_ms=%.1f max_copy_ms=%.2f callback_ms=%.1f "
            "max_callback_ms=%.2f writer_ms=%.1f max_writer_ms=%.2f stalls=%s "
            "zero_copy_writes=%s dimension_drops=%s",
            telemetry["captureBackend"],
            telemetry["captureEncoder"] or "unknown",
            telemetry["targetFps"],
            telemetry["sourceFrames"],
            telemetry["encodedFrames"],
            float(telemetry["wgcSnapshotMs"]),
            float(telemetry["wgcSnapshotMaxMs"]),
            float(telemetry["wgcCallbackMs"]),
            float(telemetry["wgcCallbackMaxMs"]),
            float(telemetry["writerStallMs"]),
            float(telemetry["writerWriteMaxMs"]),
            telemetry["writerStallCount"],
            telemetry["zeroCopyPipeWrites"],
            telemetry["dimensionDroppedFrames"],
        )

    def _clear_wgc_frame_queue(self) -> None:
        """Discard packets from a previous recording without blocking WGC."""
        while True:
            try:
                _timestamp, frame_bgra = self._wgc_frame_queue.get_nowait()
            except Empty:
                return
            _release_frame_buffer(frame_bgra)

    def _mark_recording_failed(self, reason: str) -> None:
        """Publish a terminal writer failure so the capture worker cannot hang."""
        logger.error("Recording failed: %s", reason)
        with self._lock:
            self._recording = False
            self._is_cfr = False
            self._recording_error = reason
            self._validation_error = reason
        self._timeline_clock.stop()
        self._timeline_clock.finish()
        self._recording_finalized_event.set()

    def _register_writer_process(self, proc: Optional[subprocess.Popen]) -> None:
        """Expose the active writer so shutdown can break a blocked pipe write."""
        with self._lock:
            self._writer_proc = proc

    def _force_terminate_writer(self) -> None:
        """Kill the active FFmpeg writer without waiting on the capture thread."""
        with self._lock:
            proc = self._writer_proc
        if proc is None or proc.poll() is not None:
            return
        try:
            logger.warning("Forcing stalled FFmpeg writer to terminate (pid=%s)", proc.pid)
            proc.kill()
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg writer did not exit after forced termination (pid=%s)", proc.pid)
        except (OSError, ValueError) as exc:
            logger.warning("Could not force FFmpeg writer termination: %s", exc)

    def stop_capture(self) -> None:
        """Stop capture with a bounded finalization window.

        The normal path gives the writer time to finish its remaining CFR
        slots. If it stalls, the active FFmpeg process is killed to unblock a
        pipe write before the capture thread is joined.
        """
        if self.is_recording:
            self.stop_recording()
        # Wake cadence waits immediately while the recording transition above
        # lets the writer finalize its remaining scheduled slots.
        self._capture_stop_event.set()
        self._scheduler_wake_event.set()

        deadline = time.monotonic() + 5.0
        finalized = self._recording_finalized_event.is_set()
        if not finalized:
            finalized = self._recording_finalized_event.wait(
                timeout=max(0.0, min(4.0, deadline - time.monotonic()))
            )

        if not finalized:
            self._force_terminate_writer()
            with self._lock:
                self._recording = False
                self._capturing = False
                if not self._recording_error:
                    self._recording_error = "Capture shutdown timed out; FFmpeg was terminated."
                self._is_cfr = False
            self._recording_finalized_event.set()
            self._capture_pipeline_ready_event.set()
        else:
            with self._lock:
                self._capturing = False
                self._recording = False

        thread = self._thread
        remaining = max(0.0, deadline - time.monotonic())
        if thread and thread is not threading.current_thread():
            thread.join(timeout=remaining)
            if thread.is_alive():
                self._force_terminate_writer()
                self._recording_finalized_event.set()
                logger.error("Capture thread did not stop within 5 seconds")
        with self._lock:
            self._writer_proc = None
        self._thread = None
        self._timeline_clock.finish()
        self._log_capture_performance_summary()

    # ── internal ────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        """Monitor capture — prefers WGC hardware capture, falls back to mss."""
        if _winmm:
            _winmm.timeBeginPeriod(1)
        try:
            if _HAS_WGC:
                try:
                    self._capture_loop_wgc(monitor_index=self._monitor_index)
                    return
                except Exception as exc:
                    logger.warning("WGC capture failed: %r", exc)
                    logger.info("Falling back to GDI (mss)")
                    self._capture_pipeline_ready_event.set()
            self._backend = "GDI"
            if self._capture_backend_changed_cb:
                self._capture_backend_changed_cb("GDI")
            self._capture_loop_mss()
        finally:
            if _winmm:
                _winmm.timeEndPeriod(1)

    def _capture_loop_wgc(
        self,
        monitor_index: Optional[int] = None,
        window_hwnd: Optional[int] = None,
    ) -> None:
        """Hardware-accelerated capture via Windows Graphics Capture API.

        Works for both monitor capture (pass monitor_index) and window
        capture (pass window_hwnd).  The WGC callback writes BGRA frames
        into a shared buffer; this loop polls it at the target FPS.
        """
        import mss
        self._backend = "WGC"
        if self._capture_backend_changed_cb:
            self._capture_backend_changed_cb("WGC")

        # Resolve window_hwnd → window_name (WGC v1.5 uses title string)
        window_name: Optional[str] = None
        if window_hwnd is not None:
            _user32 = ctypes.windll.user32
            length = _user32.GetWindowTextLengthW(window_hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                _user32.GetWindowTextW(window_hwnd, buf, length + 1)
                window_name = buf.value
            if not window_name:
                raise RuntimeError("WGC: cannot get window title from HWND")

        # Get expected dimensions from mss (used for FFmpeg writer init)
        if monitor_index is not None:
            with mss.mss() as sct:
                mon_info = sct.monitors[monitor_index]
                w, h = mon_info["width"], mon_info["height"]
        elif window_hwnd is not None:
            from .window_utils import get_window_rect
            rect = get_window_rect(window_hwnd)
            if not rect:
                raise RuntimeError("WGC: target window not found")
            w, h = rect["width"], rect["height"]
        else:
            raise ValueError("monitor_index or window_hwnd required")

        # ── FIFO handoff written by the WGC callback ──────────────
        self._wgc_first_frame_event.clear()
        self._clear_wgc_frame_queue()
        buffer_pool = _FrameBufferPool(w, h)
        with self._lock:
            self._wgc_buffer_pool = buffer_pool
            self._wgc_pool_capacity = buffer_pool.capacity
            self._wgc_pool_bytes = buffer_pool.allocated_bytes
            self._wgc_frame_queue = Queue(maxsize=buffer_pool.capacity)
        logger.info(
            "WGC frame pool: %d buffers, %.1f MiB at %dx%d",
            buffer_pool.capacity,
            buffer_pool.allocated_bytes / (1024.0 * 1024.0),
            w,
            h,
        )
        encoder_name, _ = _capture_encoder_args(_ffmpeg_exe())
        record_fps = _capture_record_fps(self._fps, encoder_name)
        if record_fps != self._fps:
            logger.info("CPU capture encoder detected; recording at stable 30 FPS")
        with self._lock:
            self._target_fps = record_fps
            self._capture_encoder = encoder_name

        def _on_frame(frame: Frame, ctl: InternalCaptureControl) -> None:
            """Enqueue captured BGRA frames in their original WGC order."""
            callback_started = time.perf_counter()
            copy_ms = 0.0
            buf = frame.frame_buffer
            if buf is None:
                return

            self._wgc_first_frame_event.set()

            with self._lock:
                is_recording = (
                    self._recording
                    and self._timeline_clock.state == RecordingState.RECORDING
                )
                is_capturing = self._capturing
            if is_recording:
                # The producer never applies an FPS gate.  Backpressure is
                # represented only by a full FIFO and is counted explicitly.
                if frame.width != w or frame.height != h:
                    self._record_dimension_drop(frame.width, frame.height, w, h)
                else:
                    pooled_frame = buffer_pool.checkout()
                    if pooled_frame is None:
                        self._record_wgc_overflow_drop()
                    else:
                        # The native frame expires with this callback. Copy it
                        # into an already-owned buffer without allocating.
                        copy_started = time.perf_counter()
                        copied = pooled_frame.copy_from(buf)
                        copy_ms = (time.perf_counter() - copy_started) * 1000.0
                        if copied:
                            self._enqueue_wgc_frame(
                                time.perf_counter(),
                                pooled_frame,
                            )
                        else:
                            pooled_frame.release()
                            self._record_wgc_overflow_drop()
                self._record_wgc_callback(
                    copy_ms,
                    (time.perf_counter() - callback_started) * 1000.0,
                )

            if not is_capturing:
                ctl.stop()

        def _on_closed() -> None:
            pass

        # ── build and start the WGC session ───────────────────────
        capture = WindowsCapture(
            cursor_capture=False,   # we render our own cursor in export
            draw_border=False,
            monitor_index=monitor_index,
            window_name=window_name,
        )
        capture.frame_handler = _on_frame
        capture.closed_handler = _on_closed
        capture_control = capture.start_free_threaded()
        self._wgc_control = capture_control

        try:
            # Wait for WGC to prove it is producing frames before recording.
            deadline = time.monotonic() + 3.0
            while self.is_capturing and time.monotonic() < deadline:
                if self._wgc_first_frame_event.wait(timeout=0.01):
                    break
                time.sleep(0.01)
            if not self._wgc_first_frame_event.is_set():
                raise RuntimeError("WGC: no frames received within 3 seconds")

            # This runs during the capture spin-up window, before recording
            # starts. It prevents FFmpeg's first-frame initialization from
            # consuming the live FIFO budget after the user presses Record.
            _warm_ffmpeg_capture_pipeline(w, h, record_fps)
            self._capture_pipeline_ready_event.set()

            writer_proc: Optional[subprocess.Popen] = None
            was_recording = False
            was_paused = False
            awaiting_fresh_resume = False
            record_perf_start = 0.0
            scheduler: CfrFrameScheduler | None = None
            writer_failed = False

            def _write_frame(
                frame_bgra: Any,
                timestamp_ms: float,
            ) -> bool:
                """Write one already-scheduled CFR frame to FFmpeg."""
                nonlocal writer_proc
                if frame_bgra is None or not writer_proc or not writer_proc.stdin:
                    return False
                if writer_proc.stdin.closed:
                    return False

                fh, fw = frame_bgra.shape[:2]
                if fw != w or fh != h:
                    self._record_dimension_drop(fw, fh, w, h)
                    return False
                frame_view = _bgra_frame_view(frame_bgra)
                expected = w * h * 4
                if frame_view is None or frame_view.nbytes != expected:
                    logger.warning(
                        "Frame size mismatch: %d vs expected %d — skipping",
                        0 if frame_view is None else frame_view.nbytes,
                        expected,
                    )
                    return False
                try:
                    write_started = time.perf_counter()
                    writer_proc.stdin.write(frame_view)
                    write_elapsed_ms = (time.perf_counter() - write_started) * 1000.0
                except (BrokenPipeError, OSError) as exc:
                    logger.error("ffmpeg pipe write error: %s", exc)
                    _stop_ffmpeg_writer(writer_proc)
                    writer_proc = None
                    return False

                with self._lock:
                    self._frame_timestamps.append(max(timestamp_ms, 0.0))
                    self._frame_count += 1
                self._record_writer_write(write_elapsed_ms)
                return True

            def _finalize_recording(record_stop_perf: float) -> None:
                """Complete remaining slots, validate the stream, and close FFmpeg."""
                nonlocal writer_proc, scheduler, writer_failed
                if scheduler is None:
                    return

                with self._lock:
                    elapsed = max(0.0, self._recording_duration_ms / 1000.0)
                target_slots = max(0, round(elapsed * record_fps))
                while scheduler.current_slot_index < target_slots:
                    # The final partial slot has no future source frame by
                    # definition; it is deliberately resolved with the last
                    # known frame instead of being skipped.
                    selection = scheduler.select_frame_for_current_slot(
                        self._wgc_frame_queue,
                        boundary=min(scheduler.slot_time(), record_stop_perf),
                    )
                    if selection is None:
                        writer_failed = True
                        logger.error("CFR scheduler lost its initial frame during finalization")
                        break
                    frame_bgra, _stuffed = selection
                    if not _write_frame(frame_bgra, scheduler.output_timestamp_ms()):
                        writer_failed = True
                        break
                    scheduler.advance()

                scheduler.close()
                self._clear_wgc_frame_queue()
                _stop_ffmpeg_writer(writer_proc)
                writer_proc = None
                with self._lock:
                    self._frames_written = scheduler.selected_frames
                    self._frames_stuffed = scheduler.stuffed_frames
                    self._frames_coalesced = scheduler.coalesced_frames
                    self._actual_fps = float(record_fps)
                    frame_delta = abs(self._frame_count - target_slots)
                    self._is_cfr = not writer_failed and frame_delta <= 1
                    self._validation_error = "" if self._is_cfr else (
                        f"encoded {self._frame_count} frame(s), expected "
                        f"{target_slots} at {record_fps} FPS"
                    )
                    if not self._is_cfr:
                        logger.error("CFR validation failed: %s", self._validation_error)

                telemetry = self.capture_telemetry
                logger.info(
                    "WGC scheduler: target_fps=%d source=%d selected=%d "
                    "coalesced=%d stuffed=%d overflow=%d encoded=%d cfr=%s",
                    record_fps,
                    telemetry["sourceFrames"],
                    telemetry["selectedFrames"],
                    telemetry["coalescedFrames"],
                    telemetry["stuffedFrames"],
                    telemetry["overflowDrops"],
                    telemetry["encodedFrames"],
                    telemetry["validationPassed"],
                )

            def _abort_recording(reason: str) -> None:
                """Leave a failed writer in a known stopped state without killing WGC."""
                nonlocal writer_proc, scheduler
                if scheduler is not None:
                    scheduler.close()
                self._clear_wgc_frame_queue()
                _stop_ffmpeg_writer(writer_proc)
                writer_proc = None
                stopped_at = time.perf_counter()
                active_elapsed = self._timeline_clock.stop(stopped_at)
                with self._lock:
                    self._recording = False
                    self._record_stop_perf = stopped_at
                    self._recording_duration_ms = max(
                        0.0,
                        active_elapsed * 1000.0,
                    )
                    self._is_cfr = False
                    self._validation_error = reason
                    self._recording_error = reason
                self._timeline_clock.finish()
                self._recording_finalized_event.set()
                if self._recording_finished_cb:
                    self._recording_finished_cb(output_path)

            while self.is_capturing:
                with self._lock:
                    state = self._timeline_clock.state
                    session_open = self._recording and state in {
                        RecordingState.RECORDING,
                        RecordingState.PAUSED,
                    }
                    output_path = self._output_path

                # ── state transitions (start / stop recording) ────
                if state == RecordingState.PAUSED and not was_recording:
                    self._timeline_clock.wait_until_active(timeout=0.05)
                    continue
                if state == RecordingState.STOPPING and not was_recording:
                    self._timeline_clock.finish()
                    self._recording_finalized_event.set()
                    if self._recording_finished_cb:
                        self._recording_finished_cb(output_path)
                    time.sleep(0.01)
                    continue
                if session_open and not was_recording:
                    out_path = output_path
                    if not out_path.lower().endswith(".mp4"):
                        out_path = _mp4_output_path(output_path)
                        with self._lock:
                            self._output_path = out_path
                        output_path = out_path
                    # Try ffmpeg pipe (fast, out-of-process encoding)
                    writer_proc = _start_ffmpeg_writer(out_path, w, h, record_fps)
                    self._register_writer_process(writer_proc)
                    if writer_proc is None:
                        raise RuntimeError("FFmpeg writer failed; OpenCV recording fallback has been removed")
                    with self._lock:
                        record_perf_start = self._perf_start
                    scheduler = CfrFrameScheduler(
                        record_perf_start or time.perf_counter(),
                        record_fps,
                        timeline_clock=self._timeline_clock,
                    )
                    initial_packet = scheduler.wait_for_initial_packet(self._wgc_frame_queue)
                    if initial_packet is None:
                        logger.error("WGC recording aborted: no first frame within 3 seconds")
                        _abort_recording("No WGC frame arrived within 3 seconds")
                        was_recording = False
                        continue

                    scheduler.last_frame = initial_packet[1]
                    scheduler.selected_frames = 1
                    if not _write_frame(scheduler.last_frame, scheduler.output_timestamp_ms()):
                        _abort_recording("FFmpeg rejected the first scheduled frame")
                        was_recording = False
                        continue
                    scheduler.advance()
                    was_recording = True
                    was_paused = False
                    awaiting_fresh_resume = False
                    continue
                elif not session_open and was_recording:
                    with self._lock:
                        record_stop_perf = self._record_stop_perf or time.perf_counter()
                    _finalize_recording(record_stop_perf)
                    self._timeline_clock.finish()
                    self._recording_finalized_event.set()
                    if self._recording_finished_cb:
                        self._recording_finished_cb(output_path)
                    record_perf_start = 0.0
                    scheduler = None
                    was_recording = False
                    was_paused = False
                    awaiting_fresh_resume = False
                    continue

                if not was_recording:
                    time.sleep(0.01)
                    continue

                assert scheduler is not None
                if state == RecordingState.PAUSED:
                    if not was_paused:
                        self._clear_wgc_frame_queue()
                        scheduler.discard_pending_packet()
                        was_paused = True
                    self._timeline_clock.wait_until_active(timeout=0.05)
                    continue
                if was_paused:
                    scheduler.discard_pending_packet()
                    was_paused = False
                    awaiting_fresh_resume = True
                slot_deadline = scheduler.slot_time()
                now = time.perf_counter()
                if now < slot_deadline:
                    interrupted = _wait_for_capture_interval(
                        self._scheduler_wake_event,
                        slot_deadline - now,
                    )
                    if (
                        interrupted
                        and self._timeline_clock.state == RecordingState.RECORDING
                    ):
                        self._scheduler_wake_event.clear()
                    continue

                # A blocked pipe may make several slots overdue.  We still
                # walk them in order, never collapsing the output timeline.
                selection = scheduler.select_frame_for_current_slot(self._wgc_frame_queue)
                if selection is None:
                    logger.error("CFR scheduler has no frame for a live slot")
                    _abort_recording("CFR scheduler lost its initial frame")
                    was_recording = False
                    continue
                frame_bgra, _stuffed = selection
                if awaiting_fresh_resume and not _stuffed:
                    self._timeline_clock.mark_first_fresh_resume_frame(
                        scheduler.output_timestamp_ms(),
                        1000.0 / max(record_fps, 1),
                    )
                    awaiting_fresh_resume = False
                if not _write_frame(frame_bgra, scheduler.output_timestamp_ms()):
                    _abort_recording("FFmpeg rejected a scheduled frame")
                    was_recording = False
                    continue
                scheduler.advance()

            # cleanup
            if scheduler is not None:
                scheduler.close()
            self._clear_wgc_frame_queue()
            _stop_ffmpeg_writer(writer_proc)
            if was_recording:
                self._recording_finalized_event.set()
                if self._recording_finished_cb:
                    self._recording_finished_cb(output_path)
        finally:
            self._capture_pipeline_ready_event.set()
            self._recording_finalized_event.set()
            try:
                capture_control.stop()
            except Exception:
                pass
            self._wgc_control = None
            self._clear_wgc_frame_queue()
            with self._lock:
                self._wgc_buffer_pool = None
            self._wgc_first_frame_event.clear()
            self._register_writer_process(None)

    def _capture_loop_mss(self) -> None:
        """GDI-based monitor capture via mss (fallback)."""
        import numpy as np
        import mss
        # mss feeds a best-effort capture loop. It is intentionally not allowed
        # to opt into WGC's validated CFR timing path.
        with self._lock:
            self._is_cfr = False
            self._validation_error = "GDI fallback capture is not CFR-validated"
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[self._monitor_index]
                w, h = monitor["width"], monitor["height"]
                writer_proc: Optional[subprocess.Popen] = None
                was_recording = False
                encoder_name, _ = _capture_encoder_args(_ffmpeg_exe())
                record_fps = _capture_record_fps(self._fps, encoder_name)
                with self._lock:
                    self._target_fps = record_fps
                    self._capture_encoder = encoder_name

                while self.is_capturing:
                    t0 = time.perf_counter()

                    with self._lock:
                        state = self._timeline_clock.state
                        session_open = self._recording and state in {
                            RecordingState.RECORDING,
                            RecordingState.PAUSED,
                        }
                        is_recording = state == RecordingState.RECORDING
                        output_path = self._output_path

                    # state transitions
                    if state == RecordingState.PAUSED and not was_recording:
                        self._timeline_clock.wait_until_active(timeout=0.05)
                        continue
                    if session_open and not was_recording:
                        out_path = output_path
                        if not out_path.lower().endswith(".mp4"):
                            out_path = _mp4_output_path(output_path)
                            with self._lock:
                                self._output_path = out_path
                            output_path = out_path
                        writer_proc = _start_ffmpeg_writer(out_path, w, h, record_fps)
                        self._register_writer_process(writer_proc)
                        if writer_proc is None:
                            raise RuntimeError("FFmpeg writer failed; OpenCV recording fallback has been removed")
                        if record_fps != self._fps:
                            logger.info("GDI recording capped at %d fps for stability", record_fps)
                        was_recording = True
                    elif not session_open and was_recording:
                        _stop_ffmpeg_writer(writer_proc)
                        writer_proc = None
                        self._timeline_clock.finish()
                        self._recording_finalized_event.set()
                        if self._recording_finished_cb:
                            self._recording_finished_cb(output_path)
                        was_recording = False

                    if state == RecordingState.PAUSED:
                        self._timeline_clock.wait_until_active(timeout=0.05)
                        continue

                    # grab frame (BGRA from mss)
                    img = sct.grab(monitor)
                    frame = np.asarray(img)

                    if was_recording and is_recording:
                        with self._lock:
                            self._frames_captured += 1
                        if writer_proc and writer_proc.stdin and not writer_proc.stdin.closed:
                            try:
                                frame_view = _bgra_frame_view(frame)
                                expected = w * h * 4
                                if frame_view is None or frame_view.nbytes != expected:
                                    self._record_dimension_drop(
                                        frame.shape[1], frame.shape[0], w, h,
                                        count_as_source=False,
                                    )
                                else:
                                    write_started = time.perf_counter()
                                    writer_proc.stdin.write(frame_view)
                                    self._record_writer_write(
                                        (time.perf_counter() - write_started) * 1000.0
                                    )
                                    with self._lock:
                                        ts = self._timeline_clock.active_time_ms()
                                        self._frame_timestamps.append(ts)
                                        self._frame_count += 1
                                        self._frames_written += 1
                            except (BrokenPipeError, OSError) as exc:
                                _stop_ffmpeg_writer(writer_proc)
                                writer_proc = None
                                self._mark_recording_failed(f"GDI FFmpeg writer stopped: {exc}")
                                was_recording = False
                    else:
                        pass  # Preview disabled in headless mode

                    # frame-rate cap (precise timing)
                    elapsed = time.perf_counter() - t0
                    sleep_time = max(0, (1.0 / record_fps) - elapsed)
                    if sleep_time > 0:
                        _wait_for_capture_interval(self._capture_stop_event, sleep_time)

                # cleanup
                _stop_ffmpeg_writer(writer_proc)
                if was_recording:
                    self._recording_finalized_event.set()
                    if self._recording_finished_cb:
                        self._recording_finished_cb(output_path)
        except Exception as exc:
            self._mark_recording_failed(f"GDI capture error: {exc}")

        finally:
            self._capture_pipeline_ready_event.set()
            self._recording_finalized_event.set()
            self._register_writer_process(None)

    def _capture_loop_window(self) -> None:
        """Capture loop for a specific window — prefers WGC, falls back to mss."""
        import numpy as np
        import mss
        if _winmm:
            _winmm.timeBeginPeriod(1)
        try:
            if _HAS_WGC:
                try:
                    self._capture_loop_wgc(window_hwnd=self._window_hwnd)
                    return
                except Exception as exc:
                    logger.warning("WGC window capture failed: %r", exc)
                    logger.info("Falling back to GDI (mss) for window")
                    self._capture_pipeline_ready_event.set()

            # ── GDI fallback for window capture ───────────────────
            from .window_utils import get_window_rect

            self._backend = "GDI"
            with self._lock:
                self._is_cfr = False
                self._validation_error = "GDI fallback capture is not CFR-validated"
            if self._capture_backend_changed_cb:
                self._capture_backend_changed_cb("GDI")

            with mss.mss() as sct:
                rect = get_window_rect(self._window_hwnd)
                if not rect:
                    return
                w, h = self._initial_size
                if w == 0 or h == 0:
                    w, h = rect["width"], rect["height"]
                    self._initial_size = (w, h)

                writer_proc: Optional[subprocess.Popen] = None
                was_recording = False
                encoder_name, _ = _capture_encoder_args(_ffmpeg_exe())
                record_fps = _capture_record_fps(self._fps, encoder_name)
                with self._lock:
                    self._target_fps = record_fps
                    self._capture_encoder = encoder_name

                while self.is_capturing:
                    t0 = time.perf_counter()

                    with self._lock:
                        state = self._timeline_clock.state
                        session_open = self._recording and state in {
                            RecordingState.RECORDING,
                            RecordingState.PAUSED,
                        }
                        is_recording = state == RecordingState.RECORDING
                        output_path = self._output_path

                    rect = get_window_rect(self._window_hwnd)
                    if not rect:
                        _stop_ffmpeg_writer(writer_proc)
                        if was_recording:
                            self._recording_finalized_event.set()
                            if self._recording_finished_cb:
                                self._recording_finished_cb(output_path)
                        with self._lock:
                            self._capturing = False
                        break

                    if state == RecordingState.PAUSED and not was_recording:
                        self._timeline_clock.wait_until_active(timeout=0.05)
                        continue
                    if session_open and not was_recording:
                        out_path = output_path
                        if not out_path.lower().endswith(".mp4"):
                            out_path = _mp4_output_path(output_path)
                            with self._lock:
                                self._output_path = out_path
                            output_path = out_path
                        writer_proc = _start_ffmpeg_writer(out_path, w, h, record_fps)
                        self._register_writer_process(writer_proc)
                        if writer_proc is None:
                            raise RuntimeError("FFmpeg writer failed; OpenCV recording fallback has been removed")
                        if record_fps != self._fps:
                            logger.info("GDI window recording capped at %d fps for stability", record_fps)
                        was_recording = True
                    elif not session_open and was_recording:
                        _stop_ffmpeg_writer(writer_proc)
                        writer_proc = None
                        self._timeline_clock.finish()
                        self._recording_finalized_event.set()
                        if self._recording_finished_cb:
                            self._recording_finished_cb(output_path)
                        was_recording = False

                    if state == RecordingState.PAUSED:
                        self._timeline_clock.wait_until_active(timeout=0.05)
                        continue

                    monitor = {
                        "left": rect["left"],
                        "top": rect["top"],
                        "width": rect["width"],
                        "height": rect["height"],
                    }
                    img = sct.grab(monitor)
                    frame = np.asarray(img)
                    cw, ch = rect["width"], rect["height"]

                    if was_recording and is_recording:
                        if cw != w or ch != h:
                            self._record_dimension_drop(cw, ch, w, h)
                        elif writer_proc and writer_proc.stdin and not writer_proc.stdin.closed:
                            with self._lock:
                                self._frames_captured += 1
                            try:
                                frame_view = _bgra_frame_view(frame)
                                expected = w * h * 4
                                if frame_view is None or frame_view.nbytes != expected:
                                    self._record_dimension_drop(
                                        cw, ch, w, h, count_as_source=False,
                                    )
                                else:
                                    write_started = time.perf_counter()
                                    writer_proc.stdin.write(frame_view)
                                    self._record_writer_write(
                                        (time.perf_counter() - write_started) * 1000.0
                                    )
                                    with self._lock:
                                        ts = self._timeline_clock.active_time_ms()
                                        self._frame_timestamps.append(ts)
                                        self._frame_count += 1
                                        self._frames_written += 1
                            except (BrokenPipeError, OSError) as exc:
                                _stop_ffmpeg_writer(writer_proc)
                                writer_proc = None
                                self._mark_recording_failed(f"GDI window FFmpeg writer stopped: {exc}")
                                was_recording = False
                    else:
                        pass  # Preview disabled in headless mode

                    elapsed = time.perf_counter() - t0
                    sleep_time = max(0, (1.0 / record_fps) - elapsed)
                    if sleep_time > 0:
                        _wait_for_capture_interval(self._capture_stop_event, sleep_time)

                _stop_ffmpeg_writer(writer_proc)
                if was_recording:
                    self._recording_finalized_event.set()
                    if self._recording_finished_cb:
                        self._recording_finished_cb(output_path)
        except Exception as exc:
            self._mark_recording_failed(f"GDI window capture error: {exc}")
        finally:
            self._capture_pipeline_ready_event.set()
            self._recording_finalized_event.set()
            self._register_writer_process(None)
            if _winmm:
                _winmm.timeEndPeriod(1)
