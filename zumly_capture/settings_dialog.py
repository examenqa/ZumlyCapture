"""Compact settings dialog for capture-only workflows."""

from __future__ import annotations

from PySide6.QtCore import Signal
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

    def __init__(self, settings: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Zumly Capture Settings")
        self.resize(560, 470)
        self._settings = normalize_settings(settings)
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
        self._record_hotkey = QKeySequenceEdit(QKeySequence(self._settings["record_hotkey"]))
        self._pause_hotkey = QKeySequenceEdit(QKeySequence(self._settings["pause_hotkey"]))
        self._screenshot_hotkey = QKeySequenceEdit(
            QKeySequence(self._settings["screenshot_hotkey"])
        )
        form.addRow("Start/stop recording:", self._record_hotkey)
        form.addRow("Pause/resume:", self._pause_hotkey)
        form.addRow("Capture monitor screenshot:", self._screenshot_hotkey)
        return tab

    def _audio_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        devices = list_dshow_audio_devices()
        self._microphone = QComboBox()
        self._system_audio = QComboBox()
        for combo in (self._microphone, self._system_audio):
            combo.addItem("None", "")
            for device in devices:
                combo.addItem(device, device)
        self._select_device(self._microphone, self._settings["microphone_device"])
        self._select_device(self._system_audio, self._settings["system_audio_device"])
        form.addRow("Microphone:", self._microphone)
        form.addRow("System/loopback device:", self._system_audio)
        return tab

    def _smart_zoom_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._smart_zoom = QCheckBox("Apply Smart Zoom after recording")
        self._smart_zoom.setChecked(self._settings["smart_zoom_enabled"])
        form.addRow("", self._smart_zoom)
        self._zoom_level = QDoubleSpinBox()
        self._zoom_level.setRange(1.1, 3.0)
        self._zoom_level.setSingleStep(0.1)
        self._zoom_level.setValue(self._settings["smart_zoom_level"])
        form.addRow("Zoom level:", self._zoom_level)
        self._render_cursor = QCheckBox("Render captured cursor telemetry")
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
        value.update(
            {
                "output_folder": self._output_folder.text().strip(),
                "screenshot_folder": self._screenshot_folder.text().strip(),
                "screenshot_format": self._format.currentData(),
                "copy_screenshot": self._copy_screenshot.isChecked(),
                "screenshot_delay_seconds": self._screenshot_delay.value(),
                "fps": self._fps.value(),
                "monitor": self._monitor.value(),
                "countdown_seconds": self._countdown.value(),
                "record_hotkey": self._portable_sequence(self._record_hotkey),
                "pause_hotkey": self._portable_sequence(self._pause_hotkey),
                "screenshot_hotkey": self._portable_sequence(self._screenshot_hotkey),
                "microphone_device": self._microphone.currentData(),
                "system_audio_device": self._system_audio.currentData(),
                "smart_zoom_enabled": self._smart_zoom.isChecked(),
                "smart_zoom_level": self._zoom_level.value(),
                "render_cursor": self._render_cursor.isChecked(),
                "render_clicks": self._render_clicks.isChecked(),
            }
        )
        value = normalize_settings(value)
        self.settings_saved.emit(value)
        self.accept()
