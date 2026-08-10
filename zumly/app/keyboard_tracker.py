"""Keyboard compatibility stubs for removed keystroke capture.

Zumly no longer installs a keyboard hook or records keystrokes.
This module keeps a tiny import-compatible surface for legacy controller
paths and for ABI tests that validate ``KBDLLHOOKSTRUCT`` on Windows.
"""

import ctypes
import ctypes.wintypes as wintypes
import logging
import time
from typing import List



from .models import KeyEvent

logger = logging.getLogger(__name__)

# Win32 constants retained for import compatibility.
WH_KEYBOARD_LL = 13
WM_KEYDOWN = 0x0100
WM_SYSKEYDOWN = 0x0104


class KBDLLHOOKSTRUCT(ctypes.Structure):
    """Win32 keyboard-hook payload retained for ABI regression tests."""

    _fields_ = [
        ("vkCode", wintypes.DWORD),
        ("scanCode", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_void_p),
    ]


class KeyboardTracker:
    """Compatibility wrapper that no longer records keyboard activity."""

    def __init__(self) -> None:
        self._events: List[KeyEvent] = []
        self._start_time: float = 0.0

    def start(self, session_epoch: float = 0.0) -> None:
        """Compatibility no-op for removed keystroke capture."""
        self._events.clear()
        self._start_time = session_epoch if session_epoch > 0 else time.perf_counter()
        logger.info("Keystroke capture has been removed; ignoring tracker start")

    def stop(self) -> List[KeyEvent]:
        """Stop tracking and return an empty event list."""
        if self._events:
            logger.info("Discarding %d legacy keystroke event(s)", len(self._events))
        self._events.clear()
        return []

    @property
    def events(self) -> List[KeyEvent]:
        return []
