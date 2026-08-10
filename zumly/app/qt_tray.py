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
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon, QMessageBox

from zumly_capture.identity import FILE_PREFIX, PRODUCT_NAME

from .widgets.unified_settings_dialog import (
    UnifiedSettingsDialog,
    load_ai_settings,
    load_export_settings,
    load_general_config,
    save_ai_settings,
    save_export_settings,
    save_general_config,
)
from .credentials import DPAPIEncryptionError
from .icon_loader import get_brand_icon
from .session_timing import RecordingState

logger = logging.getLogger("zumly_capture.tray")

_ROOT_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[2]

WM_HOTKEY = 0x0312
WM_QUIT = 0x0012
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
VK_R = 0x52
VK_P = 0x50
HOTKEY_RECORD_ID = 9901
HOTKEY_PAUSE_ID = 9902


class _HotkeyThread(threading.Thread):
    """Register recording shortcuts without blocking the Qt event loop."""

    def __init__(self, callback):
        super().__init__(daemon=True, name="TrayHotkey")
        self._callback = callback
        self._thread_id = 0
        self._ready = threading.Event()

    def run(self) -> None:
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        self._thread_id = kernel32.GetCurrentThreadId()
        if not user32.RegisterHotKey(None, HOTKEY_RECORD_ID, MOD_CONTROL | MOD_SHIFT, VK_R):
            logger.warning("Could not register Ctrl+Shift+R")
        if not user32.RegisterHotKey(None, HOTKEY_PAUSE_ID, MOD_CONTROL | MOD_SHIFT, VK_P):
            logger.warning("Could not register Ctrl+Shift+P")
        self._ready.set()

        message = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
            if message.message == WM_HOTKEY and message.wParam in {
                HOTKEY_RECORD_ID,
                HOTKEY_PAUSE_ID,
            }:
                self._callback(int(message.wParam))
        user32.UnregisterHotKey(None, HOTKEY_RECORD_ID)
        user32.UnregisterHotKey(None, HOTKEY_PAUSE_ID)

    def stop(self) -> None:
        if self._thread_id:
            ctypes.windll.user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
            self.join(timeout=2.0)


class QtZumlyCaptureTray(QObject):
    """Own the Qt tray UI while capture runs in a subprocess."""

    toggle_requested = Signal()
    pause_toggle_requested = Signal()
    recording_finished = Signal(object, int)
    engine_state_changed = Signal(str, int)

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
        self._cfg = load_general_config()
        self._last_capture_path = ""
        self._settings_dialog = None
        self._tray_icon: QSystemTrayIcon | None = None
        self._idle_tray_icon: QIcon | None = None
        self._recording_tray_icon: QIcon | None = None
        self._paused_tray_icon: QIcon | None = None
        self._toggle_action: QAction | None = None
        self._pause_action: QAction | None = None
        self._open_capture_action: QAction | None = None
        self._copy_capture_action: QAction | None = None
        self._reveal_capture_action: QAction | None = None
        self.recording_finished.connect(self._handle_recording_finished)
        self.toggle_requested.connect(self._on_toggle)
        self.pause_toggle_requested.connect(self._on_pause_toggle)
        self.engine_state_changed.connect(self._handle_engine_state)

    def run(self) -> None:
        """Create the tray menu and return control to QApplication.exec()."""
        os.makedirs(self._cfg["output_folder"], exist_ok=True)
        self._initialize_tray_ui()
        self._hotkey_thread = _HotkeyThread(self._dispatch_hotkey)
        self._hotkey_thread.start()
        self._hotkey_thread._ready.wait(timeout=2.0)
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
        self._toggle_action = QAction("Start Recording", self)
        self._toggle_action.triggered.connect(self._on_toggle)
        menu.addAction(self._toggle_action)

        self._pause_action = QAction("Pause Recording", self)
        self._pause_action.setEnabled(False)
        self._pause_action.triggered.connect(self._on_pause_toggle)
        menu.addAction(self._pause_action)

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
        self._tray_icon.setToolTip(f"{PRODUCT_NAME} - Ready (Ctrl+Shift+R)")
        self._tray_icon.show()

    def _dispatch_hotkey(self, hotkey_id: int) -> None:
        if hotkey_id == HOTKEY_PAUSE_ID:
            self.pause_toggle_requested.emit()
        elif hotkey_id == HOTKEY_RECORD_ID:
            self.toggle_requested.emit()

    def _on_toggle(self, _checked: bool = False) -> None:
        if self._stopping:
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
        self._settings_dialog = UnifiedSettingsDialog(
            self._cfg,
            load_ai_settings(),
            load_export_settings(),
        )
        self._settings_dialog.settings_saved.connect(self._save_unified_settings)
        self._settings_dialog.show()
        self._settings_dialog.raise_()
        self._settings_dialog.activateWindow()

    def _save_unified_settings(
        self,
        general_config: dict,
        ai_settings: object,
        export_settings: dict,
    ) -> None:
        try:
            save_ai_settings(ai_settings)
            save_general_config(general_config)
            save_export_settings(export_settings)
        except (DPAPIEncryptionError, ValueError) as exc:
            logger.error("Settings save rejected: %s", exc)
            QMessageBox.critical(
                self._settings_dialog,
                "Settings not saved",
                str(exc),
            )
            return
        self._cfg = general_config
        logger.info("General, AI, and export settings saved")

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
            subprocess.Popen(
                ["explorer.exe", f"/select,{self._last_capture_path}"],
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            logger.error("Could not reveal capture: %s", exc)
            self._notify("Could not show the last capture in its folder")

    def _start_recording(self) -> None:
        if self._state in {
            RecordingState.STARTING,
            RecordingState.RECORDING,
            RecordingState.PAUSED,
            RecordingState.STOPPING,
        }:
            return
        self._state = RecordingState.STARTING
        self._cfg = load_general_config()
        os.makedirs(self._cfg["output_folder"], exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        millis = int(time.time_ns() // 1_000_000) % 1000
        stem = f"{FILE_PREFIX}_{timestamp}_{millis:03d}"
        out_path = os.path.join(self._cfg["output_folder"], f"{stem}.mp4")
        suffix = 1
        while os.path.exists(out_path):
            out_path = os.path.join(
                self._cfg["output_folder"],
                f"{stem}_{suffix}.mp4",
            )
            suffix += 1

        if getattr(sys, "frozen", False):
            command = [sys.executable, "--headless-engine"]
        else:
            command = [sys.executable, "-m", "zumly.main"]
        command += [
            "--out", out_path,
            "--monitor", str(self._cfg.get("monitor", 1)),
            "--fps", str(self._cfg.get("fps", 60)),
        ]

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

        try:
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(_ROOT_DIR),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            logger.error("Failed to start recording engine: %s", exc)
            self._cleanup_ipc_files()
            self._state = RecordingState.IDLE
            self._update_tray("Ready (Ctrl+Shift+R)")
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
        last_seen = -1
        last_state = ""
        while process is not None and process.poll() is None:
            payload = self._read_json_payload(self._status_file)
            try:
                sequence = int(payload.get("sequence", -1))
                state = str(payload.get("state", ""))
            except (TypeError, ValueError):
                sequence, state = -1, ""
            if state and (sequence > last_seen or state != last_state):
                last_seen = sequence
                last_state = state
                self.engine_state_changed.emit(state, sequence)
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

    def _handle_engine_state(self, state_value: str, sequence: int) -> None:
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
        }
        if state == RecordingState.RECORDING:
            self._stopping = False
            self._update_tray("Recording (Ctrl+Shift+P to pause)")
        elif state == RecordingState.PAUSED:
            self._update_tray("Paused (Ctrl+Shift+P to resume)")
        elif state == RecordingState.STOPPING:
            self._stopping = True
            self._update_tray("Stopping recording...")

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

        self._update_tray("Ready (Ctrl+Shift+R)")

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
        else:
            error = str(result.get("error", "") or "Recording engine did not publish a video")
            recovery_path = str(result.get("recoveryPath", "") or "")
            if recovery_path and os.path.isfile(recovery_path):
                logger.warning("Completed recording remains recoverable at %s", recovery_path)
            logger.error("Recording failed: %s", error)
            self._notify(f"Recording failed: {error}")
        self._reregister_hotkey()

    def _reregister_hotkey(self) -> None:
        if self._hotkey_thread is not None:
            return
        self._hotkey_thread = _HotkeyThread(self._dispatch_hotkey)
        self._hotkey_thread.start()
        self._hotkey_thread._ready.wait(timeout=2.0)

    def _update_tray(self, title: str) -> None:
        if self._tray_icon is not None:
            self._tray_icon.setToolTip(f"{PRODUCT_NAME} - {title}")
            if self._state == RecordingState.PAUSED:
                self._tray_icon.setIcon(self._paused_tray_icon or get_brand_icon())
            elif self._state in {RecordingState.STARTING, RecordingState.RECORDING, RecordingState.STOPPING}:
                self._tray_icon.setIcon(self._recording_tray_icon or get_brand_icon())
            else:
                self._tray_icon.setIcon(self._idle_tray_icon or get_brand_icon())
        if self._toggle_action is not None:
            active = self._state in {
                RecordingState.STARTING,
                RecordingState.RECORDING,
                RecordingState.PAUSED,
                RecordingState.STOPPING,
            }
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
        if self._state in {RecordingState.STARTING, RecordingState.RECORDING, RecordingState.PAUSED}:
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
