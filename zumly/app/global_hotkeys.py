"""Windows global hotkeys via RegisterHotKey / GetMessage loop."""

import logging
import sys
import ctypes
import ctypes.wintypes as wintypes
import threading
from typing import Optional, Callable

logger = logging.getLogger(__name__)

WM_HOTKEY = 0x0312
WM_QUIT = 0x0012
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
VK_R = 0x52           # R
VK_P = 0x50           # P

HOTKEY_RECORD_TOGGLE = 3
HOTKEY_PAUSE_TOGGLE = 4


class _HotkeyThread(threading.Thread):
    """Runs a Win32 message loop that listens for registered hotkeys."""

    def __init__(self, hotkeys: list, callback: Callable[[int], None]) -> None:
        super().__init__(daemon=True)
        self._thread_id: int = 0
        self._hotkeys = hotkeys  # list of (id, modifiers, vk)
        self._callback = callback
        self._started_event = threading.Event()

    def run(self) -> None:
        if sys.platform != "win32":
            return

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        self._thread_id = kernel32.GetCurrentThreadId()

        for hk_id, mods, vk in self._hotkeys:
            user32.RegisterHotKey(None, hk_id, mods, vk)

        self._started_event.set()

        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == WM_HOTKEY:
                self._callback(msg.wParam)

        for hk_id, _mods, _vk in self._hotkeys:
            user32.UnregisterHotKey(None, hk_id)

    def request_stop(self) -> None:
        if self._thread_id and sys.platform == "win32":
            ctypes.windll.user32.PostThreadMessageW(
                self._thread_id, WM_QUIT, 0, 0
            )


class GlobalHotkeys:
    """Global recording shortcuts used by the standalone capture worker."""

    def __init__(
        self,
        callback: Callable[[], None],
        pause_callback: Callable[[], None] | None = None,
    ) -> None:
        self._record_thread: Optional[_HotkeyThread] = None
        self._callback = callback
        self._pause_callback = pause_callback

    # ── persistent record hotkey ────────────────────────────────────

    def register_record_hotkey(self) -> None:
        """Register Ctrl+Shift+R globally (call once at startup)."""
        if sys.platform != "win32" or self._record_thread is not None:
            return
        self._record_thread = _HotkeyThread(
            [
                (HOTKEY_RECORD_TOGGLE, MOD_CONTROL | MOD_SHIFT, VK_R),
                (HOTKEY_PAUSE_TOGGLE, MOD_CONTROL | MOD_SHIFT, VK_P),
            ],
            self._on_triggered,
        )
        self._record_thread.start()
        self._record_thread._started_event.wait(timeout=1.0)

    def unregister_record_hotkey(self) -> None:
        """Stop listening for the record-toggle hotkey and clean up the thread."""
        if self._record_thread is not None:
            self._record_thread.request_stop()
            self._record_thread.join(timeout=2.0)
            if self._record_thread.is_alive():
                logger.warning("Hotkey hook thread did not stop within timeout")
            self._record_thread = None

    # ── dispatch ────────────────────────────────────────────────────

    def _on_triggered(self, hotkey_id: int) -> None:
        if hotkey_id == HOTKEY_RECORD_TOGGLE:
            if self._callback:
                self._callback()
        elif hotkey_id == HOTKEY_PAUSE_TOGGLE and self._pause_callback:
            self._pause_callback()
