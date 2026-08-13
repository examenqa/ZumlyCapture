"""Compact settings dialog for capture-only workflows."""

from __future__ import annotations

import threading

from PySide6.QtCore import QTimer, Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QKeySequenceEdit,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .audio import list_dshow_audio_devices
from .settings import normalize_settings


class CaptureSettingsDialog(QDialog):
    settings_saved = Signal(object)
    _audio_devices_loaded = Signal(object)

    def __init__(self, settings: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Zumly Capture Settings")
        self.resize(620, 560)
        self._settings = normalize_settings(settings)
        self._audio_devices_ready = False
        self._audio_devices_loaded.connect(self._populate_audio_devices)
        root = QVBoxLayout(self)
        tabs = QTabWidget()
        tabs.addTab(self._capture_tab(), "Capture")
        tabs.addTab(self._shortcuts_tab(), "Shortcuts")
        tabs.addTab(self._audio_tab(), "Audio")
        tabs.addTab(self._smart_zoom_tab(), "Smart Zoom")
        root.addWidget(tabs, 1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        QTimer.singleShot(0, self._start_audio_discovery)

    def _folder_row(self, value: str, caption: str) -> tuple[QWidget, QLineEdit]:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit(value)
        button = QPushButton("Browse")
        button.clicked.connect(
            lambda: self._choose_folder(edit, caption)
        )
        layout.addWidget(edit, 1)
        layout.addWidget(button)
        return widget, edit

    def _capture_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        video_row, self._output_folder = self._folder_row(
            self._settings["output_folder"], "Choose recording folder"
        )
        form.addRow("Recording folder:", video_row)
        shot_row, self._screenshot_folder = self._folder_row(
            self._settings["screenshot_folder"], "Choose screenshot folder"
        )
        form.addRow("Screenshot folder:", shot_row)
        self._format = QComboBox()
        self._format.addItem("PNG", "png")
        self._format.addItem("JPEG", "jpg")
        self._format.setCurrentIndex(max(0, self._format.findData(self._settings["screenshot_format"])))
        form.addRow("Screenshot format:", self._format)
        self._copy_screenshot = QCheckBox("Copy screenshots to clipboard")
        self._copy_screenshot.setChecked(self._settings["copy_screenshot"])
        form.addRow("", self._copy_screenshot)
        self._preview_after_capture = QCheckBox(
            "Show preview after screenshots and recordings"
        )
        self._preview_after_capture.setChecked(self._settings["preview_after_capture"])
        form.addRow("", self._preview_after_capture)
        self._screenshot_delay = QSpinBox()
        self._screenshot_delay.setRange(0, 10)
        self._screenshot_delay.setSuffix(" seconds")
        self._screenshot_delay.setValue(self._settings["screenshot_delay_seconds"])
        form.addRow("Screenshot delay:", self._screenshot_delay)
        self._fps = QSpinBox()
        self._fps.setRange(15, 120)
        self._fps.setValue(self._settings["fps"])
        form.addRow("Recording FPS:", self._fps)
        self._monitor = QSpinBox()
        self._monitor.setRange(1, 32)
        self._monitor.setValue(self._settings["monitor"])
        form.addRow("Default monitor:", self._monitor)
        self._countdown = QSpinBox()
        self._countdown.setRange(0, 10)
        self._countdown.setSuffix(" seconds")
        self._countdown.setValue(self._settings["countdown_seconds"])
        form.addRow("Recording countdown:", self._countdown)
        return tab

    def _shortcuts_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._shortcut_edits: dict[str, QKeySequenceEdit] = {}
        shortcut_rows = (
            ("screenshot_monitor_hotkey", "Screenshot monitor:"),
            ("screenshot_window_hotkey", "Screenshot active window:"),
            ("screenshot_region_hotkey", "Screenshot selected region:"),
            ("record_monitor_hotkey", "Record monitor:"),
            ("record_window_hotkey", "Record selected window:"),
            ("record_region_hotkey", "Record selected region:"),
            ("pause_hotkey", "Pause/resume recording:"),
            ("stop_hotkey", "Stop recording:"),
        )
        for key, label in shortcut_rows:
            edit = QKeySequenceEdit(QKeySequence(self._settings[key]))
            edit.setMaximumSequenceLength(1)
            self._shortcut_edits[key] = edit
            form.addRow(label, edit)
        return tab

    def _audio_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._microphone = QComboBox()
        self._system_audio = QComboBox()
        for combo in (self._microphone, self._system_audio):
            combo.addItem("Detecting devices…", "")
            combo.setEnabled(False)
        form.addRow("Microphone:", self._microphone)
        form.addRow("System/loopback device:", self._system_audio)
        return tab

    def _start_audio_discovery(self) -> None:
        def discover() -> None:
            try:
                devices = list_dshow_audio_devices()
            except Exception:
                devices = []
            self._audio_devices_loaded.emit(devices)

        threading.Thread(
            target=discover,
            daemon=True,
            name="AudioDeviceDiscovery",
        ).start()

    def _populate_audio_devices(self, devices: object) -> None:
        names = [str(device) for device in devices] if isinstance(devices, list) else []
        for combo in (self._microphone, self._system_audio):
            combo.clear()
            combo.addItem("None", "")
            for device in names:
                combo.addItem(device, device)
            combo.setEnabled(True)
        self._select_device(self._microphone, self._settings["microphone_device"])
        self._select_device(self._system_audio, self._settings["system_audio_device"])
        self._audio_devices_ready = True

    def _smart_zoom_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._smart_zoom = QCheckBox(
            "Enable automatic click-driven cursor-follow zoom after recording"
        )
        self._smart_zoom.setChecked(self._settings["smart_zoom_enabled"])
        form.addRow("", self._smart_zoom)
        self._zoom_level = QDoubleSpinBox()
        self._zoom_level.setRange(1.1, 3.0)
        self._zoom_level.setSingleStep(0.1)
        self._zoom_level.setValue(self._settings["smart_zoom_level"])
        form.addRow("Zoom level:", self._zoom_level)
        self._render_cursor = QCheckBox("Keep the cursor visible in the zoomed video")
        self._render_cursor.setChecked(self._settings["render_cursor"])
        form.addRow("", self._render_cursor)
        self._render_clicks = QCheckBox("Render click indicators")
        self._render_clicks.setChecked(self._settings["render_clicks"])
        form.addRow("", self._render_clicks)
        return tab

    @staticmethod
    def _select_device(combo: QComboBox, value: str) -> None:
        index = combo.findData(value)
        if index < 0 and value:
            combo.addItem(value, value)
            index = combo.count() - 1
        combo.setCurrentIndex(max(0, index))

    def _choose_folder(self, edit: QLineEdit, caption: str) -> None:
        selected = QFileDialog.getExistingDirectory(self, caption, edit.text())
        if selected:
            edit.setText(selected)

    @staticmethod
    def _portable_sequence(edit: QKeySequenceEdit) -> str:
        return edit.keySequence().toString(QKeySequence.SequenceFormat.PortableText)

    def _save(self) -> None:
        value = dict(self._settings)
        microphone = (
            self._microphone.currentData()
            if self._audio_devices_ready
            else self._settings["microphone_device"]
        )
        system_audio = (
            self._system_audio.currentData()
            if self._audio_devices_ready
            else self._settings["system_audio_device"]
        )
        value.update(
            {
                "output_folder": self._output_folder.text().strip(),
                "screenshot_folder": self._screenshot_folder.text().strip(),
                "screenshot_format": self._format.currentData(),
                "copy_screenshot": self._copy_screenshot.isChecked(),
                "preview_after_capture": self._preview_after_capture.isChecked(),
                "screenshot_delay_seconds": self._screenshot_delay.value(),
                "fps": self._fps.value(),
                "monitor": self._monitor.value(),
                "countdown_seconds": self._countdown.value(),
                "microphone_device": microphone,
                "system_audio_device": system_audio,
                "smart_zoom_enabled": self._smart_zoom.isChecked(),
                "smart_zoom_level": self._zoom_level.value(),
                "render_cursor": self._render_cursor.isChecked(),
                "render_clicks": self._render_clicks.isChecked(),
            }
        )
        value.update(
            {
                key: self._portable_sequence(edit)
                for key, edit in self._shortcut_edits.items()
            }
        )
        value = normalize_settings(value)
        self.settings_saved.emit(value)
        self.accept()
