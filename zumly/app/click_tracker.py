"""Mouse click tracker — records click positions via a Win32 low-level hook.

Records timestamp and screen coordinates of left/right mouse-button-down events.
Runs the hook in a dedicated daemon thread.
"""

import logging
import sys
import time
import ctypes
import ctypes.wintypes as wintypes
import threading
from typing import List, Optional, Callable

from .models import ClickEvent
from .session_timing import RecordingState, SessionTimelineClock

logger = logging.getLogger(__name__)

# Win32 constants
WH_MOUSE_LL = 14
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
WM_RBUTTONDOWN = 0x0204
WM_RBUTTONUP = 0x0205
WM_QUIT = 0x0012

if sys.platform == "win32":
    class MSLLHOOKSTRUCT(ctypes.Structure):
        _fields_ = [
            ("pt", wintypes.POINT),
            ("mouseData", wintypes.DWORD),
            ("flags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.c_void_p),
        ]

    # Use WINFUNCTYPE with proper pointer-sized types for 64-bit compat
    HOOKPROC = ctypes.WINFUNCTYPE(
        wintypes.LPARAM,   # LRESULT (pointer-sized)
        ctypes.c_int,      # nCode
        wintypes.WPARAM,   # wParam (pointer-sized)
        wintypes.LPARAM,   # lParam (pointer-sized)
    )


class _MouseHookThread(threading.Thread):
    """Runs a Win32 message loop with a low-level mouse hook."""

    def __init__(
        self,
        start_ms: float = 0.0,
        callback: Optional[Callable[[int, int, float, str, bool], None]] = None,
        timeline_clock: SessionTimelineClock | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self._thread_id: int = 0
        self._hook = None
        self._start_ms = start_ms
        self._proc = None  # prevent GC of the callback
        self._callback = callback
        self._timeline_clock = timeline_clock
        self._started_event = threading.Event()

    def run(self) -> None:
        if sys.platform != "win32":
            return

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        self._thread_id = kernel32.GetCurrentThreadId()

        # Set argtypes/restype for 64-bit pointer compatibility
        user32.SetWindowsHookExW.argtypes = [
            ctypes.c_int, HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD
        ]
        user32.SetWindowsHookExW.restype = wintypes.HHOOK

        user32.CallNextHookEx.argtypes = [
            wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM
        ]
        user32.CallNextHookEx.restype = wintypes.LPARAM

        user32.UnhookWindowsHookEx.argtypes = [wintypes.HHOOK]
        user32.UnhookWindowsHookEx.restype = wintypes.BOOL

        start_ms = self._start_ms

        def low_level_handler(n_code, w_param, l_param):
            try:
                button_messages = {
                    WM_LBUTTONDOWN: ("left", True),
                    WM_LBUTTONUP: ("left", False),
                    WM_RBUTTONDOWN: ("right", True),
                    WM_RBUTTONUP: ("right", False),
                }
                if n_code >= 0 and w_param in button_messages:
                    if self._timeline_clock is not None:
                        state, ts, _generation = self._timeline_clock.snapshot()
                        if state != RecordingState.RECORDING:
                            return user32.CallNextHookEx(
                                self._hook, n_code, w_param, l_param
                            )
                    else:
                        ts = (time.perf_counter() - start_ms) * 1000.0
                    # Cast the raw LPARAM integer to a pointer to MSLLHOOKSTRUCT
                    info = ctypes.cast(l_param, ctypes.POINTER(MSLLHOOKSTRUCT)).contents
                    if self._callback:
                        button, is_down = button_messages[w_param]
                        self._callback(info.pt.x, info.pt.y, ts, button, is_down)
            except Exception:
                logger.exception("Error in mouse click hook callback")
            return user32.CallNextHookEx(self._hook, n_code, w_param, l_param)

        self._proc = HOOKPROC(low_level_handler)
        self._hook = user32.SetWindowsHookExW(
            WH_MOUSE_LL, self._proc, None, 0
        )

        self._started_event.set()

        # Pump messages so the hook receives events
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

        if self._hook:
            user32.UnhookWindowsHookEx(self._hook)
            self._hook = None

    def request_stop(self) -> None:
        if self._thread_id and sys.platform == "win32":
            ctypes.windll.user32.PostThreadMessageW(
                self._thread_id, WM_QUIT, 0, 0
            )


class ClickTracker:
    """Records positions and timestamps of mouse clicks during a recording."""

    def __init__(self) -> None:
        self._thread: Optional[_MouseHookThread] = None
        self._events: List[ClickEvent] = []
        self._start_time: float = 0.0
        self._monitor_rect: dict | None = None
        self._lock = threading.Lock()
        self._pressed_buttons: set[str] = set()
        self._timeline_clock: SessionTimelineClock | None = None
        self._resume_generation = 0

    def start(
        self,
        session_epoch: float = 0.0,
        monitor_rect: Optional[dict] = None,
        timeline_clock: SessionTimelineClock | None = None,
    ) -> None:
        """Begin tracking mouse clicks.

        ``session_epoch`` is the shared ``time.perf_counter()`` value used by
        the recorder and every input tracker. Clicks outside ``monitor_rect``
        are discarded before they enter the session data contract.
        """
        if sys.platform != "win32":
            return
        with self._lock:
            self._events.clear()
            self._pressed_buttons.clear()
            self._monitor_rect = dict(monitor_rect) if monitor_rect else None
        if timeline_clock is None:
            timeline_clock = SessionTimelineClock()
            timeline_clock.start(session_epoch)
        self._timeline_clock = timeline_clock
        self._start_time = timeline_clock.epoch
        self._resume_generation = timeline_clock.resume_generation
        self._thread = _MouseHookThread(
            start_ms=self._start_time,
            callback=self._on_mouse_button,
            timeline_clock=timeline_clock,
        )
        self._thread.start()
        self._thread._started_event.wait(timeout=1.0)

    def stop(self) -> List[ClickEvent]:
        """Stop tracking and return collected events."""
        if self._thread is not None:
            self._thread.request_stop()
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning("Click hook thread did not stop within timeout")
            self._thread = None
        with self._lock:
            result = list(self._events)
            self._events.clear()
        return result

    def _on_mouse_button(self, x: int, y: int, ts: float, button: str, is_down: bool) -> None:
        with self._lock:
            clock = self._timeline_clock
            if clock is not None:
                generation = clock.resume_generation
                if generation != self._resume_generation:
                    self._pressed_buttons.clear()
                    self._resume_generation = generation
                if clock.state != RecordingState.RECORDING:
                    self._pressed_buttons.clear()
                    return
            if is_down:
                self._pressed_buttons.add(button)
            else:
                self._pressed_buttons.discard(button)
            if not is_down:
                return
            if self._monitor_rect:
                left = float(self._monitor_rect.get("left", self._monitor_rect.get("x", 0.0)))
                top = float(self._monitor_rect.get("top", self._monitor_rect.get("y", 0.0)))
                width = max(float(self._monitor_rect.get("width", self._monitor_rect.get("w", 0.0))), 0.0)
                height = max(float(self._monitor_rect.get("height", self._monitor_rect.get("h", 0.0))), 0.0)
                if not (left <= float(x) < left + width and top <= float(y) < top + height):
                    return
            self._events.append(ClickEvent(x=float(x), y=float(y), timestamp=ts))

    def _on_click(self, x: int, y: int, ts: float) -> None:
        """Compatibility shim for existing callers and tests."""
        self._on_mouse_button(x, y, ts, "left", True)

    def is_button_down(self) -> bool:
        """Return the current state without exposing hook internals."""
        with self._lock:
            clock = self._timeline_clock
            if clock is not None:
                if clock.state != RecordingState.RECORDING:
                    return False
                if clock.resume_generation != self._resume_generation:
                    self._pressed_buttons.clear()
                    self._resume_generation = clock.resume_generation
            return bool(self._pressed_buttons)

    @property
    def events(self) -> List[ClickEvent]:
        with self._lock:
            return list(self._events)
