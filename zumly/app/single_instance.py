"""Windows single-instance ownership and activation signalling."""

from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import time
from typing import Any, Callable

from zumly_capture.identity import (
    ACTIVATION_EVENT_PREFIX,
    MUTEX_PREFIX,
    PRODUCT_NAME,
)


logger = logging.getLogger("zumly_capture.single_instance")

_ERROR_ALREADY_EXISTS = 183
_EVENT_MODIFY_STATE = 0x0002
_SYNCHRONIZE = 0x00100000
_WAIT_OBJECT_0 = 0


def _instance_suffix() -> str:
    """Keep ownership scoped to the current Windows user and logon session."""
    identity = "|".join(
        (
            os.environ.get("USERNAME", "unknown"),
            os.environ.get("USERDOMAIN", ""),
            os.environ.get("SESSIONNAME", ""),
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _load_kernel32() -> Any:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = (ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p)
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    kernel32.CreateEventW.argtypes = (
        ctypes.c_void_p,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_wchar_p,
    )
    kernel32.CreateEventW.restype = ctypes.c_void_p
    kernel32.OpenEventW.argtypes = (ctypes.c_uint32, ctypes.c_bool, ctypes.c_wchar_p)
    kernel32.OpenEventW.restype = ctypes.c_void_p
    kernel32.OpenMutexW.argtypes = (ctypes.c_uint32, ctypes.c_bool, ctypes.c_wchar_p)
    kernel32.OpenMutexW.restype = ctypes.c_void_p
    kernel32.SetEvent.argtypes = (ctypes.c_void_p,)
    kernel32.SetEvent.restype = ctypes.c_bool
    kernel32.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    kernel32.WaitForSingleObject.restype = ctypes.c_uint32
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    kernel32.CloseHandle.restype = ctypes.c_bool
    return kernel32


class ZumlyCaptureSingleInstance:
    """Own one tray process and signal it when a second launcher is invoked.

    A named mutex is released by Windows if the primary process crashes, so it
    cannot leave a stale lock behind. The auto-reset event is intentionally
    payload-free: a secondary launch only asks the primary instance to activate.
    """

    def __init__(
        self,
        *,
        mutex_name: str | None = None,
        activation_event_name: str | None = None,
        kernel32: Any | None = None,
        get_last_error: Callable[[], int] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        suffix = _instance_suffix()
        self.mutex_name = mutex_name or f"{MUTEX_PREFIX}.{suffix}"
        self.activation_event_name = (
            activation_event_name or f"{ACTIVATION_EVENT_PREFIX}.{suffix}"
        )
        self._kernel32 = kernel32 or (_load_kernel32() if os.name == "nt" else None)
        self._get_last_error = get_last_error or ctypes.get_last_error
        self._monotonic = monotonic
        self._sleep = sleep
        self._mutex_handle: int | None = None
        self._activation_event_handle: int | None = None
        self.is_primary = False

    def acquire(self) -> bool:
        """Return True only for the process that owns the tray lifecycle."""
        if self._kernel32 is None:
            # The app is Windows-only, but keep local tooling usable elsewhere.
            self.is_primary = True
            return True

        ctypes.set_last_error(0)
        mutex_handle = self._kernel32.CreateMutexW(None, False, self.mutex_name)
        if not mutex_handle:
            raise OSError(
                self._get_last_error(),
                f"Unable to create {PRODUCT_NAME} instance mutex",
            )
        if self._get_last_error() == _ERROR_ALREADY_EXISTS:
            self._kernel32.CloseHandle(mutex_handle)
            return False

        event_handle = self._kernel32.CreateEventW(
            None,
            False,
            False,
            self.activation_event_name,
        )
        if not event_handle:
            self._kernel32.CloseHandle(mutex_handle)
            raise OSError(
                self._get_last_error(),
                f"Unable to create {PRODUCT_NAME} activation event",
            )

        self._mutex_handle = mutex_handle
        self._activation_event_handle = event_handle
        self.is_primary = True
        return True

    def signal_primary(self, timeout_s: float = 1.0) -> bool:
        """Request activation without starting a second tray or hotkey owner."""
        if self._kernel32 is None:
            return False
        deadline = self._monotonic() + max(0.0, timeout_s)
        while True:
            event_handle = self._kernel32.OpenEventW(
                _EVENT_MODIFY_STATE,
                False,
                self.activation_event_name,
            )
            if event_handle:
                try:
                    return bool(self._kernel32.SetEvent(event_handle))
                finally:
                    self._kernel32.CloseHandle(event_handle)
            if self._monotonic() >= deadline:
                logger.warning(
                    "Existing %s instance did not expose its activation channel",
                    PRODUCT_NAME,
                )
                return False
            self._sleep(0.05)

    def primary_exists(self) -> bool:
        """Return whether the long-lived tray owner is already running."""
        if self._kernel32 is None:
            return False
        mutex_handle = self._kernel32.OpenMutexW(_SYNCHRONIZE, False, self.mutex_name)
        if not mutex_handle:
            return False
        self._kernel32.CloseHandle(mutex_handle)
        return True

    def consume_activation(self) -> bool:
        """Consume at most one pending secondary-launch request."""
        if self._kernel32 is None or not self._activation_event_handle:
            return False
        return self._kernel32.WaitForSingleObject(self._activation_event_handle, 0) == _WAIT_OBJECT_0

    def close(self) -> None:
        """Release handles during normal shutdown; crashes are released by Windows."""
        for attribute in ("_activation_event_handle", "_mutex_handle"):
            handle = getattr(self, attribute)
            if handle and self._kernel32 is not None:
                self._kernel32.CloseHandle(handle)
            setattr(self, attribute, None)
        self.is_primary = False


# Compatibility alias for copied seed tests and modules. New code must use the
# product-specific class name above.
ZumlySingleInstance = ZumlyCaptureSingleInstance
