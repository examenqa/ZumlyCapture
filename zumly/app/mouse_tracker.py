"""Mouse position tracker — polls cursor position at 60 Hz via Win32.

Uses ``GetPhysicalCursorPos`` for physical pixel coordinates.
so they match the capture APIs (mss, WGC, PrintWindow).  Runs on a
native background thread.
"""

import sys
import time
import threading
import logging
from typing import Callable, List

from .models import MousePosition
from .session_timing import RecordingState, SessionTimelineClock

logger = logging.getLogger(__name__)

# Use the explicit Win32 physical-coordinate API for DPI-stable samples.
if sys.platform == "win32":
    import ctypes
    import ctypes.wintypes as wintypes
    try:
        _winmm = ctypes.windll.winmm
    except AttributeError:
        _winmm = None
else:
    _winmm = None


def _get_physical_cursor_pos() -> tuple[int, int]:
    """Return the cursor position in physical screen pixels via Win32."""
    if sys.platform != "win32":
        return 0, 0
    pt = wintypes.POINT()
    if not ctypes.windll.user32.GetPhysicalCursorPos(ctypes.byref(pt)):
        raise ctypes.WinError()
    return pt.x, pt.y


class MouseTracker:
    """Thread-based cursor poller that records :class:`MousePosition` samples.

    The polling interval defaults to 16 ms (~60 Hz).  All timestamps
    are relative to a shared epoch so they align with keyboard and
    click trackers.
    """

    def __init__(
        self,
        interval_ms: int = 16,
        click_state_provider: Callable[[], bool] | None = None,
    ) -> None:
        self._interval_sec = interval_ms / 1000.0
        self._start_time: float = 0.0
        self._positions: List[MousePosition] = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._click_state_provider = click_state_provider
        self._timeline_clock: SessionTimelineClock | None = None
        self._resume_generation = 0

    def start(
        self,
        session_epoch: float = 0.0,
        timeline_clock: SessionTimelineClock | None = None,
    ) -> None:
        """Begin polling cursor position.

        ``session_epoch`` is the shared ``time.perf_counter()`` value used by
        the recorder and every input tracker.
        """
        if timeline_clock is None:
            timeline_clock = SessionTimelineClock()
            timeline_clock.start(session_epoch)
        self._timeline_clock = timeline_clock
        self._start_time = timeline_clock.epoch
        self._resume_generation = timeline_clock.resume_generation
        with self._lock:
            self._positions.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> List[MousePosition]:
        """Stop polling and return the collected position samples."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning("Mouse tracker thread did not stop within timeout")
            self._thread = None
        
        with self._lock:
            return list(self._positions)

    def _run_loop(self) -> None:
        """Polling loop running in background thread."""
        if _winmm:
            _winmm.timeBeginPeriod(1)
        try:
            while not self._stop_event.is_set():
                t0 = time.perf_counter()

                clock = self._timeline_clock
                if clock is not None and clock.state == RecordingState.PAUSED:
                    clock.wait_until_active(timeout=self._interval_sec)
                    continue
                if clock is not None and clock.state not in {
                    RecordingState.RECORDING,
                    RecordingState.PAUSED,
                }:
                    if self._stop_event.wait(timeout=self._interval_sec):
                        break
                    continue
                
                px, py = _get_physical_cursor_pos()
                click_state = None
                if self._click_state_provider is not None:
                    try:
                        click_state = bool(self._click_state_provider())
                    except Exception:
                        logger.debug("Mouse button state provider failed", exc_info=True)
                if clock is not None:
                    state, timestamp_ms, resume_generation = clock.snapshot()
                    if state != RecordingState.RECORDING:
                        continue
                else:
                    timestamp_ms = (time.perf_counter() - self._start_time) * 1000.0
                    resume_generation = 0
                resume_boundary = resume_generation != self._resume_generation
                mp = MousePosition(
                    x=px,
                    y=py,
                    timestamp=timestamp_ms,
                    click_state=click_state,
                    resume_boundary=resume_boundary,
                )
                with self._lock:
                    self._positions.append(mp)
                self._resume_generation = resume_generation
                
                elapsed = time.perf_counter() - t0
                sleep_time = max(0.0, self._interval_sec - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            if _winmm:
                _winmm.timeEndPeriod(1)
