"""NumPy-free adapter for the native windows-capture extension."""

from __future__ import annotations

import ctypes
import importlib
from pathlib import Path
import sys
import types
from typing import Callable


def _load_native_capture_class():
    """Load the extension without executing its NumPy/OpenCV package wrapper."""
    package_name = "windows_capture"
    if package_name not in sys.modules:
        roots = [Path(getattr(sys, "_MEIPASS", ""))] if getattr(sys, "_MEIPASS", "") else []
        roots.extend(Path(entry) for entry in sys.path if entry)
        package_path = next(
            (
                root / package_name
                for root in roots
                if (root / package_name).is_dir()
                and any((root / package_name).glob("windows_capture*.pyd"))
            ),
            None,
        )
        if package_path is None:
            raise ImportError("Native windows-capture extension was not found")
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_path)]
        package.__package__ = package_name
        sys.modules[package_name] = package
    native_module = importlib.import_module("windows_capture.windows_capture")
    return native_module.NativeWindowsCapture


NativeWindowsCapture = _load_native_capture_class()


class NativeFrameBuffer:
    __slots__ = ("_address", "_length", "_row_pitch", "_width", "_height")

    def __init__(self, pointer, length: int, width: int, height: int) -> None:
        self._address = ctypes.addressof(
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_uint8)).contents
        )
        self._length = int(length)
        self._width = int(width)
        self._height = int(height)
        self._row_pitch = int(length) // max(1, int(height))

    def copy_into(self, destination: memoryview, width: int, height: int) -> bool:
        row_bytes = int(width) * 4
        if width != self._width or height != self._height:
            return False
        if destination.nbytes != row_bytes * int(height):
            return False
        try:
            destination_address = ctypes.addressof(
                ctypes.c_ubyte.from_buffer(destination)
            )
            if self._row_pitch == row_bytes:
                ctypes.memmove(destination_address, self._address, destination.nbytes)
            else:
                for row in range(int(height)):
                    ctypes.memmove(
                        destination_address + row * row_bytes,
                        self._address + row * self._row_pitch,
                        row_bytes,
                    )
            return True
        except (TypeError, ValueError, BufferError, OSError):
            return False


class Frame:
    __slots__ = ("frame_buffer", "width", "height", "timespan")

    def __init__(self, pointer, length: int, width: int, height: int, timespan: int) -> None:
        self.frame_buffer = NativeFrameBuffer(pointer, length, width, height)
        self.width = int(width)
        self.height = int(height)
        self.timespan = int(timespan)


class InternalCaptureControl:
    __slots__ = ("_stop_list",)

    def __init__(self, stop_list: list) -> None:
        self._stop_list = stop_list

    def stop(self) -> None:
        self._stop_list[0] = True


class CaptureControl:
    __slots__ = ("_native",)

    def __init__(self, native_control) -> None:
        self._native = native_control

    def stop(self) -> None:
        self._native.stop()


class WindowsCapture:
    def __init__(
        self,
        cursor_capture: bool = False,
        draw_border: bool = False,
        monitor_index: int | None = None,
        window_name: str | None = None,
        window_hwnd: int | None = None,
    ) -> None:
        if window_name is not None or window_hwnd is not None:
            monitor_index = None
        self.frame_handler: Callable[[Frame, InternalCaptureControl], None] | None = None
        self.closed_handler: Callable[[], None] | None = None
        self._native = NativeWindowsCapture(
            self._on_frame,
            self._on_closed,
            cursor_capture,
            draw_border,
            None,
            None,
            None,
            monitor_index,
            window_name,
            window_hwnd,
        )

    def _on_frame(
        self,
        pointer,
        length: int,
        width: int,
        height: int,
        stop_list: list,
        timespan: int,
    ) -> None:
        if self.frame_handler is None:
            return
        self.frame_handler(
            Frame(pointer, length, width, height, timespan),
            InternalCaptureControl(stop_list),
        )

    def _on_closed(self) -> None:
        if self.closed_handler is not None:
            self.closed_handler()

    def start_free_threaded(self) -> CaptureControl:
        if self.frame_handler is None or self.closed_handler is None:
            raise RuntimeError("WGC handlers must be set before capture starts")
        return CaptureControl(self._native.start_free_threaded())
