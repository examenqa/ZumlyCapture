"""Qt system-tray controller for standalone screen recording."""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from PySide6.QtCore import QMimeData, QObject, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QIcon, QImage
from PySide6.QtWidgets import QApplication, QDialog, QMenu, QSystemTrayIcon

from zumly_capture.capture_ui import RegionSelector, WindowPickerDialog
from zumly_capture.identity import FILE_PREFIX, PRODUCT_NAME
from zumly_capture.preview_dialog import CapturePreviewDialog
from zumly_capture.screenshot import foreground_window_handle, publish_screenshot
from zumly_capture.session import discard_recording_draft, discard_unzoomed_recording
from zumly_capture.settings import load_settings, save_settings
from zumly_capture.settings_dialog import CaptureSettingsDialog
from zumly_capture.windows_shell import reveal_in_folder

from .icon_loader import get_brand_icon
from .screen_recorder import CAPTURE_ENCODERS_ENV, ScreenRecorder
from .session_timing import RecordingState
from .utils import detect_available_encoders
from .window_utils import enumerate_windows, get_window_rect

logger = logging.getLogger("zumly_capture.tray")

_ROOT_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[2]

WM_HOTKEY = 0x0312
WM_QUIT = 0x0012
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
VK_R = 0x52
VK_P = 0x50
HOTKEY_SCREENSHOT_MONITOR_ID = 9901
HOTKEY_SCREENSHOT_WINDOW_ID = 9902
HOTKEY_SCREENSHOT_REGION_ID = 9903
HOTKEY_RECORD_MONITOR_ID = 9904
HOTKEY_RECORD_WINDOW_ID = 9905
HOTKEY_RECORD_REGION_ID = 9906
HOTKEY_PAUSE_ID = 9909
HOTKEY_STOP_ID = 9910


def _parse_hotkey(value: str) -> tuple[int, int]:
    """Translate a portable Qt shortcut into RegisterHotKey values."""
    parts = [part.strip() for part in str(value).split("+") if part.strip()]
    if not parts:
        raise ValueError("Shortcut cannot be empty")
    modifiers = 0
    key_name = parts[-1].upper()
    for part in parts[:-1]:
        name = part.lower()
        if name in {"ctrl", "control"}:
            modifiers |= MOD_CONTROL
        elif name == "shift":
            modifiers |= MOD_SHIFT
        elif name == "alt":
            modifiers |= 0x0001
        elif name in {"win", "meta"}:
            modifiers |= 0x0008
        else:
            raise ValueError(f"Unsupported shortcut modifier: {part}")
    if len(key_name) == 1 and key_name.isalnum():
        virtual_key = ord(key_name)
    elif key_name.startswith("F") and key_name[1:].isdigit():
        number = int(key_name[1:])
        if not 1 <= number <= 24:
            raise ValueError(f"Unsupported function key: {key_name}")
        virtual_key = 0x70 + number - 1
    else:
        raise ValueError(f"Unsupported shortcut key: {key_name}")
    return modifiers, virtual_key


class _HotkeyThread(threading.Thread):
    """Register recording shortcuts without blocking the Qt event loop."""

    def __init__(self, callback, shortcuts: dict[int, str]):
        super().__init__(daemon=True, name="TrayHotkey")
        self._callback = callback
        self._shortcuts = dict(shortcuts)
        self._thread_id = 0
        self._ready = threading.Event()

    def run(self) -> None:
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        self._thread_id = kernel32.GetCurrentThreadId()
        registered: list[int] = []
        for hotkey_id, shortcut in self._shortcuts.items():
            try:
                modifiers, virtual_key = _parse_hotkey(shortcut)
            except ValueError as exc:
                logger.warning("Invalid shortcut %s: %s", shortcut, exc)
                continue
            if user32.RegisterHotKey(None, hotkey_id, modifiers, virtual_key):
                registered.append(hotkey_id)
            else:
                logger.warning("Could not register shortcut %s", shortcut)
        self._ready.set()

        message = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
            if message.message == WM_HOTKEY and int(message.wParam) in self._shortcuts:
                self._callback(int(message.wParam))
        for hotkey_id in registered:
            user32.UnregisterHotKey(None, hotkey_id)

    def stop(self) -> None:
        if self._thread_id:
            ctypes.windll.user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
            self.join(timeout=2.0)


class QtZumlyCaptureTray(QObject):
    """Own the Qt tray UI while capture runs in a subprocess."""

    hotkey_requested = Signal(int)
    recording_finished = Signal(object, int)
    engine_state_changed = Signal(object)

    def __init__(self, app: QApplication, instance_guard: object | None = None) -> None:
        super().__init__()
        self._app = app
        self._instance_guard = instance_guard
        self._activation_timer: QTimer | None = None
        self._recording = False
        self._stopping = False
        self._state = RecordingState.IDLE
        self._process: subprocess.Popen | None = None
        self._hotkey_thread: _HotkeyThread | None = None
        self._stop_file = ""
        self._control_file = ""
        self._status_file = ""
        self._result_file = ""
        self._command_sequence = 0
        self._ack_sequence = -1
        self._cfg = load_settings()
        self._last_capture_path = ""
        self._settings_dialog = None
        self._preview_dialog: CapturePreviewDialog | None = None
        self._region_selector: RegionSelector | None = None
        self._countdown_timer: QTimer | None = None
        self._pending_record_target: dict | None = None
        self._tray_icon: QSystemTrayIcon | None = None
        self._idle_tray_icon: QIcon | None = None
        self._recording_tray_icon: QIcon | None = None
        self._paused_tray_icon: QIcon | None = None
        self._toggle_action: QAction | None = None
        self._pause_action: QAction | None = None
        self._open_capture_action: QAction | None = None
        self._copy_capture_action: QAction | None = None
        self._reveal_capture_action: QAction | None = None
        self._cursor_zoom_action: QAction | None = None
        self._capture_encoder_hint = ""
        self.recording_finished.connect(self._handle_recording_finished)
        self.hotkey_requested.connect(self._handle_hotkey)
        self.engine_state_changed.connect(self._handle_engine_state)

    def run(self) -> None:
        """Create the tray menu and return control to QApplication.exec()."""
        os.makedirs(self._cfg["output_folder"], exist_ok=True)
        self._initialize_tray_ui()
        self._start_hotkey_thread()
        threading.Thread(
            target=self._probe_capture_encoders,
            daemon=True,
            name="CaptureEncoderProbe",
        ).start()
        if self._instance_guard is not None:
            self._activation_timer = QTimer(self)
            self._activation_timer.setInterval(100)
            self._activation_timer.timeout.connect(self._consume_instance_activation)
            self._activation_timer.start()

    def _consume_instance_activation(self) -> None:
        """Handle a second launcher request inside the Qt UI thread."""
        guard = self._instance_guard
        if guard is None or not bool(guard.consume_activation()):
            return

        if self._state in {
            RecordingState.STARTING,
            RecordingState.RECORDING,
            RecordingState.PAUSED,
            RecordingState.STOPPING,
            RecordingState.PROCESSING,
        }:
            self._notify(f"{PRODUCT_NAME} is already recording")
            return

        if self._last_capture_path and os.path.isfile(self._last_capture_path):
            self._show_last_capture()
        else:
            self._notify(f"{PRODUCT_NAME} is ready")

    def _initialize_tray_ui(self) -> None:
        """Create the persistent tray icon and menu exactly once."""
        if self._tray_icon is not None:
            self._tray_icon.show()
            return

        self._idle_tray_icon = get_brand_icon()
        if self._idle_tray_icon.isNull():
            logger.warning("Packaged %s tray icon could not be loaded", PRODUCT_NAME)
        self._recording_tray_icon = self._idle_tray_icon
        self._paused_tray_icon = self._idle_tray_icon
        self._tray_icon = QSystemTrayIcon(self._idle_tray_icon, self)

        menu = QMenu()
        menu.setObjectName("captureCommandMenu")
        menu.setMinimumWidth(310)
        menu.setStyleSheet(
            """
            QMenu#captureCommandMenu {
                background: #202936;
                color: #edf4ff;
                border: 1px solid #3f5067;
                border-radius: 12px;
                padding: 9px;
                font-size: 13px;
            }
            QMenu#captureCommandMenu::item {
                min-height: 25px;
                padding: 7px 18px 7px 14px;
                margin: 2px 0;
                border-radius: 7px;
            }
            QMenu#captureCommandMenu::item:selected {
                background: #356da8;
                color: white;
            }
            QMenu#captureCommandMenu::item:disabled { color: #8292a6; }
            QMenu#captureCommandMenu::separator {
                height: 1px;
                background: #3a4759;
                margin: 8px 10px;
            }
            QMenu#captureCommandMenu::indicator { width: 16px; height: 16px; }
            QMenu#captureCommandMenu QMenu {
                background: #202936;
                color: #edf4ff;
                border: 1px solid #3f5067;
                padding: 8px;
            }
            """
        )
        brand = QAction(PRODUCT_NAME.upper(), self)
        brand.setEnabled(False)
        menu.addAction(brand)
        menu.addSeparator()
        self._toggle_action = QAction("Start Recording", self)
        self._toggle_action.triggered.connect(self._on_toggle)
        menu.addAction(self._toggle_action)

        self._pause_action = QAction("Pause Recording", self)
        self._pause_action.setShortcut("Ctrl+Alt+9")
        self._pause_action.setEnabled(False)
        self._pause_action.triggered.connect(self._on_pause_toggle)
        menu.addAction(self._pause_action)

        screenshot_menu = menu.addMenu("Take Screenshot")
        screenshot_monitor = QAction("Full Monitor", self)
        screenshot_monitor.setShortcut("Ctrl+Alt+1")
        screenshot_monitor.triggered.connect(self._screenshot_monitor)
        screenshot_menu.addAction(screenshot_monitor)
        screenshot_window = QAction("Active Window", self)
        screenshot_window.setShortcut("Ctrl+Alt+2")
        screenshot_window.triggered.connect(self._screenshot_active_window)
        screenshot_menu.addAction(screenshot_window)
        screenshot_region = QAction("Select Region", self)
        screenshot_region.setShortcut("Ctrl+Alt+3")
        screenshot_region.triggered.connect(self._screenshot_region)
        screenshot_menu.addAction(screenshot_region)

        recording_menu = menu.addMenu("Record")
        record_monitor = QAction("Full Monitor", self)
        record_monitor.setShortcut("Ctrl+Alt+4")
        record_monitor.triggered.connect(self._record_monitor)
        recording_menu.addAction(record_monitor)
        record_window = QAction("Choose Window…", self)
        record_window.setShortcut("Ctrl+Alt+5")
        record_window.triggered.connect(self._record_window)
        recording_menu.addAction(record_window)
        record_region = QAction("Select Region…", self)
        record_region.setShortcut("Ctrl+Alt+6")
        record_region.triggered.connect(self._record_region)
        recording_menu.addAction(record_region)

        self._cursor_zoom_action = QAction("Automatic Smart Zoom", self)
        self._cursor_zoom_action.setCheckable(True)
        self._cursor_zoom_action.setChecked(bool(self._cfg.get("smart_zoom_enabled", True)))
        self._cursor_zoom_action.triggered.connect(self._toggle_cursor_zoom)
        recording_menu.addSeparator()
        recording_menu.addAction(self._cursor_zoom_action)

        menu.addSeparator()
        self._open_capture_action = QAction("Open Last Capture", self)
        self._open_capture_action.setEnabled(False)
        self._open_capture_action.triggered.connect(self._open_last_capture)
        menu.addAction(self._open_capture_action)

        self._copy_capture_action = QAction("Copy Last Capture", self)
        self._copy_capture_action.setEnabled(False)
        self._copy_capture_action.triggered.connect(self._copy_last_capture)
        menu.addAction(self._copy_capture_action)

        self._reveal_capture_action = QAction("Show in Folder", self)
        self._reveal_capture_action.setEnabled(False)
        self._reveal_capture_action.triggered.connect(self._show_last_capture)
        menu.addAction(self._reveal_capture_action)

        menu.addSeparator()
        settings = QAction("Settings", self)
        settings.triggered.connect(self._on_settings)
        menu.addAction(settings)

        menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self._on_quit)
        menu.addAction(quit_action)

        self._tray_icon.setContextMenu(menu)
        self._tray_icon.setToolTip(f"{PRODUCT_NAME} - Ready · Ctrl+Alt+1–6")
        self._tray_icon.show()

    def _dispatch_hotkey(self, hotkey_id: int) -> None:
        self.hotkey_requested.emit(int(hotkey_id))

    def _handle_hotkey(self, hotkey_id: int) -> None:
        actions = {
            HOTKEY_SCREENSHOT_MONITOR_ID: self._screenshot_monitor,
            HOTKEY_SCREENSHOT_WINDOW_ID: self._screenshot_active_window,
            HOTKEY_SCREENSHOT_REGION_ID: self._screenshot_region,
            HOTKEY_RECORD_MONITOR_ID: self._record_monitor,
            HOTKEY_RECORD_WINDOW_ID: self._record_window,
            HOTKEY_RECORD_REGION_ID: self._record_region,
            HOTKEY_PAUSE_ID: self._on_pause_toggle,
            HOTKEY_STOP_ID: self._on_stop_hotkey,
        }
        action = actions.get(int(hotkey_id))
        if action is not None:
            action()

    def _on_stop_hotkey(self) -> None:
        if self._state in {
            RecordingState.STARTING,
            RecordingState.RECORDING,
            RecordingState.PAUSED,
            RecordingState.PROCESSING,
        } or self._recording:
            self._on_toggle()

    def _toggle_cursor_zoom(self, enabled: bool) -> None:
        self._cfg["smart_zoom_enabled"] = bool(enabled)
        if enabled:
            self._cfg["render_cursor"] = True
        self._cfg = save_settings(self._cfg)
        self._notify(
            "Automatic Smart Zoom enabled"
            if enabled
            else "Automatic Smart Zoom disabled"
        )

    def _on_toggle(self, _checked: bool = False) -> None:
        if self._state == RecordingState.PROCESSING:
            if not self._stopping and self._process is not None:
                self._stopping = True
                self._send_control("cancel")
                self._update_tray("Cancelling Smart Zoom...")
                if self._toggle_action is not None:
                    self._toggle_action.setEnabled(False)
            return
        if self._stopping:
            return
        if self._state == RecordingState.STARTING and self._process is None:
            self._cancel_countdown()
            return
        if self._state in {RecordingState.RECORDING, RecordingState.PAUSED} or self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _on_pause_toggle(self, _checked: bool = False) -> None:
        if self._stopping or self._process is None:
            return
        if self._state == RecordingState.RECORDING:
            self._send_control("pause")
        elif self._state == RecordingState.PAUSED:
            self._send_control("resume")

    def _on_settings(self, _checked: bool = False) -> None:
        if self._settings_dialog is not None and self._settings_dialog.isVisible():
            self._settings_dialog.raise_()
            self._settings_dialog.activateWindow()
            return
        self._settings_dialog = CaptureSettingsDialog(self._cfg)
        self._settings_dialog.settings_saved.connect(self._save_capture_settings)
        self._settings_dialog.show()
        self._settings_dialog.raise_()
        self._settings_dialog.activateWindow()

    def _save_capture_settings(self, settings: object) -> None:
        try:
            if not isinstance(settings, dict):
                raise ValueError("Capture settings must be an object")
            self._cfg = save_settings(settings)
        except (OSError, ValueError) as exc:
            logger.error("Settings save rejected: %s", exc)
            self._notify(f"Settings were not saved: {exc}")
            return
        self._restart_hotkey_thread()
        if self._cursor_zoom_action is not None:
            self._cursor_zoom_action.setChecked(
                bool(self._cfg.get("smart_zoom_enabled", True))
            )
        logger.info("Capture settings saved")

    def _start_hotkey_thread(self) -> None:
        if self._hotkey_thread is not None:
            return
        shortcuts = {
            HOTKEY_SCREENSHOT_MONITOR_ID: str(self._cfg["screenshot_monitor_hotkey"]),
            HOTKEY_SCREENSHOT_WINDOW_ID: str(self._cfg["screenshot_window_hotkey"]),
            HOTKEY_SCREENSHOT_REGION_ID: str(self._cfg["screenshot_region_hotkey"]),
            HOTKEY_RECORD_MONITOR_ID: str(self._cfg["record_monitor_hotkey"]),
            HOTKEY_RECORD_WINDOW_ID: str(self._cfg["record_window_hotkey"]),
            HOTKEY_RECORD_REGION_ID: str(self._cfg["record_region_hotkey"]),
            HOTKEY_PAUSE_ID: str(self._cfg["pause_hotkey"]),
            HOTKEY_STOP_ID: str(self._cfg["stop_hotkey"]),
        }
        self._hotkey_thread = _HotkeyThread(self._dispatch_hotkey, shortcuts)
        self._hotkey_thread.start()
        self._hotkey_thread._ready.wait(timeout=2.0)

    def _restart_hotkey_thread(self) -> None:
        if self._hotkey_thread is not None:
            self._hotkey_thread.stop()
            self._hotkey_thread = None
        self._start_hotkey_thread()

    def _capture_is_available(self) -> bool:
        available = bool(self._last_capture_path and os.path.isfile(self._last_capture_path))
        if not available:
            self._last_capture_path = ""
            self._set_capture_actions_enabled(False)
        return available

    def _set_capture_actions_enabled(self, enabled: bool) -> None:
        for action in (
            self._open_capture_action,
            self._copy_capture_action,
            self._reveal_capture_action,
        ):
            if action is not None:
                action.setEnabled(enabled)

    def _open_last_capture(self, _checked: bool = False) -> None:
        if not self._capture_is_available():
            self._notify("The last capture is no longer available")
            return
        try:
            os.startfile(self._last_capture_path)
        except OSError as exc:
            logger.error("Could not open capture: %s", exc)
            self._notify("Could not open the last capture")

    def _copy_last_capture(self, _checked: bool = False) -> None:
        if not self._capture_is_available():
            self._notify("The last capture is no longer available")
            return
        mime_data = QMimeData()
        mime_data.setUrls([QUrl.fromLocalFile(self._last_capture_path)])
        self._app.clipboard().setMimeData(mime_data)
        self._notify("Capture copied to the clipboard")

    def _show_last_capture(self, _checked: bool = False) -> None:
        if not self._capture_is_available():
            self._notify("The last capture is no longer available")
            return
        try:
            reveal_in_folder(self._last_capture_path)
        except OSError as exc:
            logger.error("Could not reveal capture: %s", exc)
            self._notify("Could not show the last capture in its folder")

    def _probe_capture_encoders(self) -> None:
        """Move FFmpeg hardware probing off the recording startup path."""
        try:
            self._capture_encoder_hint = ",".join(detect_available_encoders())
        except Exception as exc:
            logger.debug("Background capture encoder probe failed: %s", exc)

    def _show_capture_preview(
        self,
        capture_path: str,
        unzoomed_path: str = "",
        format_source_path: str = "",
        preferred_output_format: str = "",
    ) -> None:
        if not bool(self._cfg.get("preview_after_capture", True)):
            discard_unzoomed_recording(unzoomed_path)
            discard_recording_draft(format_source_path)
            return
        if not capture_path or not os.path.isfile(capture_path):
            discard_unzoomed_recording(unzoomed_path)
            discard_recording_draft(format_source_path)
            return
        if self._preview_dialog is not None:
            self._preview_dialog.close()
        try:
            preview = CapturePreviewDialog(
                capture_path,
                unzoomed_path=unzoomed_path,
                format_source_path=format_source_path,
                preferred_output_format=preferred_output_format,
            )
        except Exception as exc:
            discard_unzoomed_recording(unzoomed_path)
            discard_recording_draft(format_source_path)
            logger.warning("Could not open capture preview: %s", exc)
            return
        self._preview_dialog = preview
        preview.saved.connect(lambda path: setattr(self, "_last_capture_path", path))
        preview.destroyed.connect(lambda _obj=None: setattr(self, "_preview_dialog", None))
        preview.show()
        preview.raise_()
        preview.activateWindow()

    def _monitor_rect(self, monitor_index: int | None = None) -> dict | None:
        selected = int(monitor_index or self._cfg.get("monitor", 1))
        for monitor in ScreenRecorder.get_monitors():
            if int(monitor.get("index", 0)) == selected:
                return dict(monitor)
        return None

    def _next_output_path(self, folder: str, extension: str) -> str:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        millis = int(time.time_ns() // 1_000_000) % 1000
        stem = f"{FILE_PREFIX}_{timestamp}_{millis:03d}"
        candidate = os.path.join(folder, f"{stem}.{extension}")
        suffix = 1
        while os.path.exists(candidate):
            candidate = os.path.join(folder, f"{stem}_{suffix}.{extension}")
            suffix += 1
        return candidate

    def _capture_screenshot(self, rect: dict) -> None:
        self._cfg = load_settings()
        folder = str(self._cfg["screenshot_folder"])
        os.makedirs(folder, exist_ok=True)
        image_format = str(self._cfg.get("screenshot_format", "png"))
        extension = "jpg" if image_format == "jpg" else "png"
        output_path = self._next_output_path(folder, extension)
        try:
            saved_path = publish_screenshot(rect, output_path, image_format)
        except Exception as exc:
            logger.exception("Screenshot failed: %s", exc)
            self._notify(f"Screenshot failed: {exc}")
            return
        self._last_capture_path = saved_path
        self._set_capture_actions_enabled(True)
        if self._cfg.get("copy_screenshot", True):
            image = QImage(saved_path)
            if not image.isNull():
                self._app.clipboard().setImage(image)
        self._notify(f"Screenshot saved: {os.path.basename(saved_path)}")
        QTimer.singleShot(160, lambda: self._show_capture_preview(saved_path))

    def _schedule_screenshot(self, rect: dict) -> None:
        delay = int(self._cfg.get("screenshot_delay_seconds", 0) or 0)
        if delay > 0:
            self._notify(f"Screenshot in {delay} seconds…")
            QTimer.singleShot(delay * 1000, lambda: self._capture_screenshot(rect))
        else:
            QTimer.singleShot(120, lambda: self._capture_screenshot(rect))

    def _screenshot_monitor(self, _checked: bool = False) -> None:
        rect = self._monitor_rect()
        if rect is None:
            self._notify("The configured monitor is not available")
            return
        self._schedule_screenshot(rect)

    def _screenshot_active_window(self, _checked: bool = False) -> None:
        delay = int(self._cfg.get("screenshot_delay_seconds", 0) or 0)
        if delay > 0:
            self._notify(f"Screenshot in {delay} seconds…")
        # Resolve foreground state after the tray menu has closed. Resolving it
        # synchronously here can select Qt's transient popup instead of the app.
        QTimer.singleShot(max(180, delay * 1000), self._capture_active_window_screenshot)

    def _capture_active_window_screenshot(self) -> None:
        handle = foreground_window_handle()
        rect = get_window_rect(handle) if handle else None
        if rect is None:
            self._notify("Could not identify the active window")
            return
        self._capture_screenshot(rect)

    def _select_region(self, callback) -> None:
        if self._region_selector is not None:
            self._region_selector.close()
        selector = RegionSelector()
        self._region_selector = selector

        def selected(rect: object) -> None:
            self._region_selector = None
            selector.deleteLater()
            if isinstance(rect, dict):
                callback(dict(rect))

        def cancelled() -> None:
            self._region_selector = None
            selector.deleteLater()

        selector.region_selected.connect(selected)
        selector.cancelled.connect(cancelled)
        selector.begin()

    def _screenshot_region(self, _checked: bool = False) -> None:
        self._select_region(self._schedule_screenshot)

    def _record_monitor(self, _checked: bool = False) -> None:
        self._begin_recording(
            {"kind": "monitor", "monitorIndex": int(self._cfg.get("monitor", 1))}
        )

    def _record_window(self, _checked: bool = False) -> None:
        windows = enumerate_windows()
        if not windows:
            self._notify("No recordable windows were found")
            return
        dialog = WindowPickerDialog(windows, "Record a Window")
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        selected = dialog.selected_window()
        if selected:
            self._begin_recording(
                {
                    "kind": "window",
                    "windowHandle": int(selected["hwnd"]),
                    "windowTitle": str(selected.get("title", "")),
                }
            )

    def _record_region(self, _checked: bool = False) -> None:
        self._select_region(
            lambda rect: self._begin_recording({"kind": "region", **rect})
        )

    def _start_recording(self) -> None:
        self._record_monitor()

    def _begin_recording(self, target: dict) -> None:
        if self._state in {
            RecordingState.STARTING,
            RecordingState.RECORDING,
            RecordingState.PAUSED,
            RecordingState.STOPPING,
            RecordingState.PROCESSING,
        }:
            return
        self._state = RecordingState.STARTING
        self._pending_record_target = dict(target)
        seconds = int(self._cfg.get("countdown_seconds", 0) or 0)
        if seconds <= 0:
            self._launch_recording()
            return
        self._countdown_remaining = seconds
        self._update_tray(f"Recording starts in {seconds}…")
        self._countdown_timer = QTimer(self)
        self._countdown_timer.setInterval(1000)
        self._countdown_timer.timeout.connect(self._countdown_tick)
        self._countdown_timer.start()

    def _countdown_tick(self) -> None:
        self._countdown_remaining -= 1
        if self._countdown_remaining <= 0:
            if self._countdown_timer is not None:
                self._countdown_timer.stop()
                self._countdown_timer.deleteLater()
                self._countdown_timer = None
            self._launch_recording()
            return
        self._update_tray(f"Recording starts in {self._countdown_remaining}…")

    def _cancel_countdown(self) -> None:
        if self._countdown_timer is not None:
            self._countdown_timer.stop()
            self._countdown_timer.deleteLater()
            self._countdown_timer = None
        self._pending_record_target = None
        self._state = RecordingState.IDLE
        self._update_tray("Ready · Ctrl+Alt+1–6")
        self._notify("Recording countdown cancelled")

    def _launch_recording(self) -> None:
        if self._process is not None or self._recording:
            return
        target = dict(
            self._pending_record_target
            or {"kind": "monitor", "monitorIndex": int(self._cfg.get("monitor", 1))}
        )
        self._pending_record_target = None
        self._state = RecordingState.STARTING
        self._cfg = load_settings()
        os.makedirs(self._cfg["output_folder"], exist_ok=True)
        recording_format = str(self._cfg.get("recording_format", "mp4")).lower()
        if recording_format not in {"mp4", "gif"}:
            recording_format = "mp4"
        out_path = self._next_output_path(self._cfg["output_folder"], recording_format)

        if getattr(sys, "frozen", False):
            command = [sys.executable, "--headless-engine"]
        else:
            command = [sys.executable, "-m", "zumly.main"]
        command += [
            "--out", out_path,
            "--monitor", str(target.get("monitorIndex", self._cfg.get("monitor", 1))),
            "--fps", str(self._cfg.get("fps", 60)),
            "--output-format", recording_format,
            "--target-kind", str(target.get("kind", "monitor")),
        ]
        if target.get("kind") == "window":
            command += ["--window-hwnd", str(target.get("windowHandle", 0))]
            command += ["--window-title", str(target.get("windowTitle", ""))]
        elif target.get("kind") == "region":
            command += [
                "--region",
                str(target.get("left", 0)),
                str(target.get("top", 0)),
                str(target.get("width", 0)),
                str(target.get("height", 0)),
            ]
        microphone = str(self._cfg.get("microphone_device", "") or "")
        system_audio = str(self._cfg.get("system_audio_device", "") or "")
        retain_mp4_source = recording_format == "gif" and bool(
            self._cfg.get("preview_after_capture", True)
        )
        if microphone and (recording_format == "mp4" or retain_mp4_source):
            command += ["--microphone", microphone]
        if system_audio and (recording_format == "mp4" or retain_mp4_source):
            command += ["--system-audio", system_audio]
        if bool(self._cfg.get("smart_zoom_enabled", False)):
            command += [
                "--smart-zoom",
                "--smart-zoom-level",
                str(self._cfg.get("smart_zoom_level", 1.5)),
            ]
            if bool(self._cfg.get("preview_after_capture", True)):
                command.append("--preserve-unzoomed")
            if bool(self._cfg.get("render_cursor", False)):
                command.append("--render-cursor")
            if bool(self._cfg.get("render_clicks", True)):
                command.append("--render-clicks")
        if recording_format == "gif" and bool(
            self._cfg.get("preview_after_capture", True)
        ):
            command.append("--preserve-format-source")

        self._stop_file = os.path.join(
            tempfile.gettempdir(),
            f"{FILE_PREFIX}_stop_{os.getpid()}_{int(time.time() * 1000)}.signal",
        )
        self._result_file = os.path.join(
            tempfile.gettempdir(),
            f"{FILE_PREFIX}_result_{os.getpid()}_{int(time.time() * 1000)}.json",
        )
        self._control_file = os.path.join(
            tempfile.gettempdir(),
            f"{FILE_PREFIX}_control_{os.getpid()}_{int(time.time() * 1000)}.json",
        )
        self._status_file = os.path.join(
            tempfile.gettempdir(),
            f"{FILE_PREFIX}_status_{os.getpid()}_{int(time.time() * 1000)}.json",
        )
        self._command_sequence = 0
        self._ack_sequence = -1
        for path in (
            self._stop_file,
            self._result_file,
            self._control_file,
            self._status_file,
        ):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass
        command += ["--stop-file", self._stop_file]
        command += ["--result-file", self._result_file]
        command += ["--control-file", self._control_file]
        command += ["--status-file", self._status_file]

        process_environment = None
        if self._capture_encoder_hint:
            process_environment = os.environ.copy()
            process_environment[CAPTURE_ENCODERS_ENV] = self._capture_encoder_hint
        try:
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(_ROOT_DIR),
                env=process_environment,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            logger.error("Failed to start recording engine: %s", exc)
            self._cleanup_ipc_files()
            self._state = RecordingState.IDLE
            self._update_tray("Ready · Ctrl+Alt+1–6")
            self._notify("Could not start the recording engine")
            return

        self._recording = True
        self._stopping = False
        self._update_tray("Starting recording...")
        threading.Thread(
            target=self._monitor_process,
            daemon=True,
            name="ProcMon",
        ).start()
        threading.Thread(
            target=self._monitor_engine_state,
            daemon=True,
            name="StateMon",
        ).start()

    def _stop_recording(self) -> None:
        if not self._recording or self._stopping or self._process is None:
            return
        self._stopping = True
        self._state = RecordingState.STOPPING
        if self._process.poll() is None:
            if self._control_file:
                self._send_control("stop")
            elif self._stop_file:
                # Backward compatibility for older frozen capture workers.
                try:
                    Path(self._stop_file).write_text("stop", encoding="utf-8")
                except OSError as exc:
                    logger.error("Failed to write stop signal: %s", exc)
            self._update_tray("Stopping recording...")
            if self._toggle_action is not None:
                self._toggle_action.setEnabled(False)

    def _send_control(self, action: str) -> bool:
        """Atomically publish one ordered command to the capture worker."""
        if not self._control_file:
            return False
        self._command_sequence += 1
        payload = {
            "sequence": self._command_sequence,
            "action": str(action),
            "requestedAt": time.time(),
        }
        target = Path(self._control_file)
        temp_path = ""
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(target.parent),
                prefix=f"{FILE_PREFIX}_control_",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = handle.name
                json.dump(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, target)
            if self._pause_action is not None and action in {"pause", "resume"}:
                self._pause_action.setEnabled(False)
            return True
        except OSError as exc:
            logger.error("Failed to send recorder command %s: %s", action, exc)
            if temp_path:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            return False

    def _monitor_engine_state(self) -> None:
        """Poll worker acknowledgements without touching Qt from this thread."""
        process = self._process
        last_snapshot = ""
        while process is not None and process.poll() is None:
            payload = self._read_json_payload(self._status_file)
            try:
                snapshot = json.dumps(payload, sort_keys=True) if payload.get("state") else ""
            except (TypeError, ValueError):
                snapshot = ""
            if snapshot and snapshot != last_snapshot:
                last_snapshot = snapshot
                self.engine_state_changed.emit(dict(payload))
            time.sleep(0.05)

    @staticmethod
    def _read_json_payload(path: str) -> dict:
        if not path:
            return {}
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return payload if isinstance(payload, dict) else {}
        except (FileNotFoundError, PermissionError, json.JSONDecodeError, OSError):
            return {}

    def _handle_engine_state(self, payload: object, legacy_sequence: int | None = None) -> None:
        if isinstance(payload, str):
            payload = {"state": payload, "sequence": legacy_sequence or 0}
        if not isinstance(payload, dict):
            return
        state_value = str(payload.get("state", ""))
        try:
            sequence = int(payload.get("sequence", -1))
        except (TypeError, ValueError):
            sequence = -1
        try:
            state = RecordingState(state_value)
        except ValueError:
            return
        if sequence < self._ack_sequence and state != RecordingState.FINISHED:
            return
        self._ack_sequence = max(self._ack_sequence, sequence)
        self._state = state
        self._recording = state in {
            RecordingState.STARTING,
            RecordingState.RECORDING,
            RecordingState.PAUSED,
            RecordingState.STOPPING,
            RecordingState.PROCESSING,
        }
        if state == RecordingState.RECORDING:
            self._stopping = False
            self._update_tray("Recording · Ctrl+Alt+9 pauses · Ctrl+Alt+0 stops")
        elif state == RecordingState.PAUSED:
            self._update_tray("Paused · Ctrl+Alt+9 resumes · Ctrl+Alt+0 stops")
        elif state == RecordingState.STOPPING:
            self._stopping = True
            self._update_tray("Stopping recording...")
        elif state == RecordingState.PROCESSING:
            self._stopping = False
            try:
                progress = max(0, min(100, int(payload.get("progress", 0))))
            except (TypeError, ValueError):
                progress = 0
            phase = str(payload.get("phase", "smart_zoom"))
            if phase == "gif_export":
                self._update_tray(f"Creating GIF... {progress}%")
            else:
                self._update_tray(f"Applying Smart Zoom... {progress}%")
            if self._toggle_action is not None:
                self._toggle_action.setEnabled(True)

    def _monitor_process(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            if process.stdout is not None:
                for line in process.stdout:
                    line = line.rstrip()
                    if line:
                        logger.info("[engine] %s", line)
        except (ValueError, OSError):
            pass
        process.wait()
        payload = self._read_result_payload(self._result_file, timeout=2.0)
        return_code = int(process.returncode or 0)
        if payload.get("status") != "success" and return_code == 0:
            return_code = 1
        self.recording_finished.emit(payload, return_code)

    @staticmethod
    def _read_result_payload(path: str, timeout: float) -> dict:
        """Read the recorder's atomic result payload without touching Qt UI."""
        if not path:
            return {}
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() <= deadline:
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                return payload if isinstance(payload, dict) else {}
            except (FileNotFoundError, PermissionError, json.JSONDecodeError, OSError):
                time.sleep(0.05)
        logger.warning("Recorder result payload was not available: %s", path)
        return {}

    def _cleanup_ipc_files(self) -> None:
        for attribute in (
            "_stop_file",
            "_control_file",
            "_status_file",
            "_result_file",
        ):
            path = str(getattr(self, attribute, "") or "")
            if path:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except OSError:
                    logger.debug("Could not remove recorder IPC file: %s", path)
            setattr(self, attribute, "")

    def _handle_recording_finished(self, payload: object, return_code: int) -> None:
        result = payload if isinstance(payload, dict) else {}
        self._state = RecordingState.FINISHED
        self._recording = False
        self._stopping = False
        self._process = None
        self._cleanup_ipc_files()

        if self._toggle_action is not None:
            self._toggle_action.setEnabled(True)
        if self._pause_action is not None:
            self._pause_action.setEnabled(False)

        self._update_tray("Ready · Ctrl+Alt+1–6")

        media_path = str(result.get("mediaPath") or result.get("outputPath") or "")
        if (
            return_code == 0
            and result.get("status") == "success"
            and media_path
            and os.path.isfile(media_path)
        ):
            self._last_capture_path = os.path.abspath(media_path)
            self._set_capture_actions_enabled(True)
            warning = str(result.get("warning", "") or "")
            if warning:
                logger.warning("Capture completed with warning: %s", warning)
                self._notify(f"Recording saved. {warning}")
            else:
                self._notify(f"Recording saved: {os.path.basename(media_path)}")
            QTimer.singleShot(
                160,
                lambda path=self._last_capture_path,
                original=str(result.get("unzoomedPath", "") or ""),
                format_source=str(result.get("formatSourcePath", "") or ""),
                preferred=str(
                    result.get("outputFormat")
                    or self._cfg.get("recording_format", "mp4")
                ): self._show_capture_preview(
                    path,
                    original,
                    format_source,
                    preferred,
                ),
            )
        else:
            error = str(result.get("error", "") or "Recording engine did not publish media")
            recovery_path = str(result.get("recoveryPath", "") or "")
            if recovery_path and os.path.isfile(recovery_path):
                logger.warning("Completed recording remains recoverable at %s", recovery_path)
            logger.error("Recording failed: %s", error)
            self._notify(f"Recording failed: {error}")
        self._reregister_hotkey()

    def _reregister_hotkey(self) -> None:
        self._start_hotkey_thread()

    def _update_tray(self, title: str) -> None:
        if self._tray_icon is not None:
            self._tray_icon.setToolTip(f"{PRODUCT_NAME} - {title}")
            if self._state == RecordingState.PAUSED:
                self._tray_icon.setIcon(self._paused_tray_icon or get_brand_icon())
            elif self._state in {
                RecordingState.STARTING,
                RecordingState.RECORDING,
                RecordingState.STOPPING,
                RecordingState.PROCESSING,
            }:
                self._tray_icon.setIcon(self._recording_tray_icon or get_brand_icon())
            else:
                self._tray_icon.setIcon(self._idle_tray_icon or get_brand_icon())
        if self._toggle_action is not None:
            active = self._state in {
                RecordingState.STARTING,
                RecordingState.RECORDING,
                RecordingState.PAUSED,
                RecordingState.STOPPING,
                RecordingState.PROCESSING,
            }
            if self._state == RecordingState.PROCESSING:
                self._toggle_action.setText("Cancel Processing")
            else:
                self._toggle_action.setText("Stop Recording" if active else "Start Recording")
        if self._pause_action is not None:
            self._pause_action.setText(
                "Resume Recording" if self._state == RecordingState.PAUSED else "Pause Recording"
            )
            self._pause_action.setEnabled(
                self._state in {RecordingState.RECORDING, RecordingState.PAUSED}
                and not self._stopping
            )

    def _notify(self, message: str) -> None:
        if self._tray_icon is not None:
            self._tray_icon.showMessage(
                PRODUCT_NAME,
                message,
                QSystemTrayIcon.MessageIcon.Information,
                3500,
            )

    def _on_quit(self) -> None:
        if self._state == RecordingState.PROCESSING and self._process is not None:
            self._send_control("cancel")
        elif self._state in {RecordingState.STARTING, RecordingState.RECORDING, RecordingState.PAUSED}:
            self._stop_recording()
        if self._hotkey_thread is not None:
            self._hotkey_thread.stop()
            self._hotkey_thread = None
        if self._activation_timer is not None:
            self._activation_timer.stop()
            self._activation_timer = None
        if self._tray_icon is not None:
            self._tray_icon.hide()
        self._app.quit()


# Compatibility alias while copied tests and seed modules are migrated.
QtZumlyTray = QtZumlyCaptureTray
