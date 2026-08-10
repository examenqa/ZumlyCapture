"""Transitional settings dialog using the Zumly Capture namespace."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from PySide6.QtCore import QSettings, QStandardPaths, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QButtonGroup, QCheckBox, QRadioButton, QMessageBox
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from zumly_capture.identity import (
    PRODUCT_NAME,
    SETTINGS_APPLICATION_NAME,
    SETTINGS_DIRECTORY_NAME,
    ORGANIZATION_NAME,
)

from .. import tokens as T
from ..ai_service import (
    AIConnectionResult,
    AIConnectionWorker,
    AI_PROVIDER_PRESETS,
    AISettings,
    DEFAULT_AI_CHAT_MODEL,
    DEFAULT_AI_ENDPOINT,
    DEFAULT_AI_PROVIDER,
    DEFAULT_NARRATION_MODEL,
    DEFAULT_TTS_VOICE,
    is_standard_google_ai_endpoint,
    validate_ai_endpoint,
)
from ..credentials import (
    DPAPIEncryptionError,
    DPAPIDecryptionError,
    CredentialReadResult,
    protect,
    read_credential,
)
from ..fluent_effects import apply_shadow, install_focus_ring
from ..hardware_utils import (
    ENCODER_AMD,
    ENCODER_INTEL,
    ENCODER_NVIDIA,
    ENCODER_SOFTWARE,
    detect_supported_hardware_encoders,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "output_folder": str(Path.home() / "Videos" / SETTINGS_DIRECTORY_NAME),
    "fps": 60,
    "monitor": 1,
}


def _app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[3]


def _app_config_path() -> Path:
    """Return the standalone Zumly Capture config path."""
    location = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppConfigLocation
    )
    if not location:
        return Path.home() / ".config" / SETTINGS_DIRECTORY_NAME / "config.json"
    return Path(location).parent / SETTINGS_DIRECTORY_NAME / "config.json"


CONFIG_PATH = _app_config_path()


def _qt_settings() -> QSettings:
    return QSettings(ORGANIZATION_NAME, SETTINGS_APPLICATION_NAME)


def load_general_config() -> dict:
    """Load recorder settings shared by the tray and unified dialog."""
    config = DEFAULT_CONFIG.copy()
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            saved = json.load(handle)
        if isinstance(saved, dict):
            config.update(saved)
    except FileNotFoundError:
        pass
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("General settings load failed: %s", exc)
    return config


def save_general_config(config: dict) -> None:
    """Atomically persist recorder settings under the OS app-config root."""
    temp_path = CONFIG_PATH.with_name(f"{CONFIG_PATH.name}.tmp")
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        with temp_path.open("r", encoding="utf-8") as handle:
            validated = json.load(handle)
        if not isinstance(validated, dict):
            raise ValueError("General settings must be a JSON object")
        os.replace(temp_path, CONFIG_PATH)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.error("General settings save failed: %s", exc)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("Could not remove staged settings file %s: %s", temp_path, exc)


def load_ai_settings(*, strict: bool = False) -> AISettings:
    """Read AI settings from QSettings and decrypt the API key."""
    settings = _qt_settings()
    encrypted_key = settings.value("ai/apiKey", "") or ""
    try:
        credential: CredentialReadResult = read_credential(encrypted_key)
        api_key = credential.value
        if credential.is_legacy_plaintext:
            logger.info("Loaded a legacy plaintext AI key; it will be upgraded on save.")
    except DPAPIDecryptionError as exc:
        if strict:
            raise
        logger.warning("Could not decrypt the saved AI key: %s", exc)
        api_key = ""
    return AISettings(
        provider=settings.value("ai/provider", DEFAULT_AI_PROVIDER) or DEFAULT_AI_PROVIDER,
        endpoint=settings.value("ai/endpoint", DEFAULT_AI_ENDPOINT) or DEFAULT_AI_ENDPOINT,
        api_key=api_key,
        chat_model=settings.value("ai/chatModel", DEFAULT_AI_CHAT_MODEL) or DEFAULT_AI_CHAT_MODEL,
        narration_model=settings.value("ai/narrationModel", DEFAULT_NARRATION_MODEL) or DEFAULT_NARRATION_MODEL,
        tts_voice=settings.value("ai/ttsVoice", DEFAULT_TTS_VOICE) or DEFAULT_TTS_VOICE,
    )


def save_ai_settings(value: AISettings) -> None:
    """Persist AI settings while keeping the API key DPAPI-protected."""
    endpoint = validate_ai_endpoint(value.endpoint)
    protected_key = protect(value.api_key)
    settings = _qt_settings()
    settings.setValue("ai/provider", value.provider)
    settings.setValue("ai/endpoint", endpoint)
    settings.setValue("ai/apiKey", protected_key)
    settings.setValue("ai/chatModel", value.chat_model)
    settings.setValue("ai/narrationModel", value.narration_model)
    settings.setValue("ai/ttsVoice", value.tts_voice)
    settings.sync()


def load_export_settings(fallback: dict | None = None) -> dict:
    """Load global export preferences, using the supplied project fallback."""
    fallback = fallback or {}
    settings = _qt_settings()
    encoder_id = str(
        settings.value("export/encoderId", fallback.get("encoder_id", "libx264"))
        or "libx264"
    )
    raw_debug = settings.value(
        "editor/showDebugOverlay",
        fallback.get("debug_overlay", False),
    )
    if isinstance(raw_debug, str):
        debug_overlay = raw_debug.strip().lower() in {"1", "true", "yes", "on"}
    else:
        debug_overlay = bool(raw_debug)
    return {
        "encoder_id": encoder_id,
        "debug_overlay": debug_overlay,
        "encoder_configured": settings.contains("export/encoderId"),
    }


def save_export_settings(value: dict) -> None:
    """Persist encoder and preview-debug preferences in QSettings."""
    settings = _qt_settings()
    settings.setValue("export/encoderId", str(value.get("encoder_id", "libx264")))
    settings.setValue("editor/showDebugOverlay", bool(value.get("debug_overlay", False)))
    settings.sync()


class UnifiedSettingsDialog(QDialog):
    """Tabbed General, AI, and Export settings with one Save action."""

    settings_saved = Signal(dict, object, dict)  # general config, AISettings, export settings

    def __init__(
        self,
        general_config: dict | None = None,
        ai_settings: AISettings | None = None,
        export_settings: dict | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{PRODUCT_NAME} Settings")
        self.setMinimumSize(580, 430)
        self.setStyleSheet(
            f"QDialog {{ background: {T.BG_SURFACE}; color: {T.FG_PRIMARY}; }}"
            f"QLabel {{ color: {T.FG_PRIMARY}; }}"
            f"QLineEdit, QComboBox, QSpinBox {{ background: {T.BG_INTERACTIVE};"
            f" color: {T.FG_PRIMARY}; border: 1px solid {T.CARD_BORDER};"
            f" border-radius: {T.RADIUS_SMALL}px; padding: 6px; }}"
            f"QPushButton {{ background: {T.BG_INTERACTIVE}; color: {T.FG_PRIMARY};"
            f" border: 1px solid {T.CARD_BORDER}; border-radius: {T.RADIUS_SMALL}px;"
            f" padding: 6px 12px; }}"
            f"QPushButton:hover {{ background: {T.BRAND}; }}"
            f"QTabWidget::pane {{ border: 1px solid {T.STROKE_2}; }}"
            f"QTabBar::tab {{ background: {T.BG_LAYER_2}; color: {T.FG_SECONDARY};"
            f" padding: 8px 18px; border: 1px solid {T.STROKE_2}; }}"
            f"QTabBar::tab:selected {{ color: {T.FG_PRIMARY}; background: {T.BG_INTERACTIVE}; }}"
        )
        apply_shadow(self, level="medium")

        config = DEFAULT_CONFIG.copy()
        config.update(general_config or {})
        self._general_config_seed = dict(config)
        current_ai = ai_settings or load_ai_settings()
        current_export = load_export_settings(export_settings)
        self._connection_worker: AIConnectionWorker | None = None
        self._dialog_buttons: QDialogButtonBox | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(T.SPACE_LG, T.SPACE_LG, T.SPACE_LG, T.SPACE_LG)
        layout.setSpacing(T.SPACE_MD)

        intro = QLabel("Configure recording defaults and AI assistance in one place.")
        intro.setStyleSheet(f"color: {T.FG_SECONDARY};")
        layout.addWidget(intro)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_general_tab(config), "General")
        self._tabs.addTab(self._build_ai_tab(current_ai), "AI Provider")
        self._tabs.addTab(self._build_export_tab(current_export), "Export")
        layout.addWidget(self._tabs, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        self._dialog_buttons = buttons
        layout.addWidget(buttons)

    def _build_general_tab(self, config: dict) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        form = QFormLayout()
        form.setSpacing(T.SPACE_MD)

        folder_row = QHBoxLayout()
        self._folder = QLineEdit(str(config.get("output_folder", "")))
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse_folder)
        folder_row.addWidget(self._folder, 1)
        folder_row.addWidget(browse)
        form.addRow("Output folder:", folder_row)

        self._fps = QSpinBox()
        self._fps.setRange(15, 120)
        self._fps.setSingleStep(5)
        self._fps.setValue(int(config.get("fps", 60) or 60))
        form.addRow("Recording FPS:", self._fps)

        self._monitor = QSpinBox()
        self._monitor.setRange(1, 8)
        self._monitor.setValue(int(config.get("monitor", 1) or 1))
        form.addRow("Monitor:", self._monitor)

        layout.addLayout(form)
        layout.addStretch(1)
        return tab

    def _build_ai_tab(self, current: AISettings) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        info = QLabel(
            "Choose a provider to fill safe defaults, then add your API key. "
            "Endpoint, models, and voice remain editable."
        )
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {T.FG_SECONDARY};")
        layout.addWidget(info)

        form = QFormLayout()
        form.setSpacing(T.SPACE_SM)
        self._provider = QComboBox()
        for preset in AI_PROVIDER_PRESETS.values():
            self._provider.addItem(preset.label, preset.id)
        form.addRow("Provider:", self._provider)

        self._endpoint = QLineEdit(current.endpoint)
        form.addRow("Endpoint:", self._endpoint)
        self._endpoint_security_note = QLabel()
        self._endpoint_security_note.setWordWrap(True)
        self._endpoint_security_note.setVisible(False)
        layout.addLayout(form)
        layout.addWidget(self._endpoint_security_note)

        self._endpoint_trust_ack = QCheckBox(
            "I understand this sends my API key to an untrusted third-party host."
        )
        self._endpoint_trust_ack.setToolTip(
            "Required before saving a custom HTTPS endpoint."
        )
        self._endpoint_trust_ack.setVisible(False)
        layout.addWidget(self._endpoint_trust_ack)

        self._api_key = QLineEdit(current.api_key)
        self._api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key.setPlaceholderText("Your Gemini API key")
        form.addRow("API key:", self._api_key)
        self._chat_model = QLineEdit(current.chat_model)
        form.addRow("Chat model:", self._chat_model)
        self._narration_model = QLineEdit(current.narration_model)
        form.addRow("Narration model:", self._narration_model)
        self._tts_voice = QLineEdit(current.tts_voice)
        form.addRow("TTS voice:", self._tts_voice)
        self._provider_note = QLabel()
        self._provider_note.setWordWrap(True)
        self._provider_note.setStyleSheet(
            f"color: {T.FG_SECONDARY}; background: {T.BG_LAYER_2};"
            f" border: 1px solid {T.STROKE_2}; padding: {T.SPACE_SM}px;"
        )
        layout.addWidget(self._provider_note)

        connection_row = QHBoxLayout()
        self._test_connection_button = QPushButton("Test Connection")
        self._test_connection_button.setToolTip(
            "Send a small ping to the configured Gemini chat model."
        )
        self._test_connection_button.clicked.connect(self._test_ai_connection)
        connection_row.addWidget(self._test_connection_button)
        self._connection_status = QLabel("")
        self._connection_status.setWordWrap(True)
        self._connection_status.setObjectName("ConnectionStatus")
        connection_row.addWidget(self._connection_status, 1)
        layout.addLayout(connection_row)
        layout.addStretch(1)

        provider_id = current.provider if current.provider in AI_PROVIDER_PRESETS else DEFAULT_AI_PROVIDER
        self._provider.setCurrentIndex(max(0, self._provider.findData(provider_id)))
        self._fill_provider_defaults(provider_id, only_missing=True)
        self._provider.currentIndexChanged.connect(self._on_provider_changed)
        self._endpoint.textChanged.connect(self._update_endpoint_trust_ui)
        self._update_provider_note()
        self._update_endpoint_trust_ui()
        for child in self.findChildren(QLineEdit):
            install_focus_ring(child)
        return tab

    def _test_ai_connection(self) -> None:
        """Run the provider diagnostic without blocking the settings dialog."""
        if self._connection_worker and self._connection_worker.isRunning():
            return

        self._connection_status.setText("Testing Gemini connection...")
        self._connection_status.setStyleSheet(f"color: {T.INFO_FG};")
        self._test_connection_button.setEnabled(False)
        if self._dialog_buttons:
            self._dialog_buttons.setEnabled(False)

        worker = AIConnectionWorker(self._ai_settings(), self)
        self._connection_worker = worker
        worker.result.connect(self._show_connection_result)
        worker.finished.connect(self._connection_test_finished)
        worker.start()

    def _show_connection_result(self, result: AIConnectionResult) -> None:
        status = result.status
        if status == "success":
            text = f"Connected to {result.model_used} in {result.latency_ms} ms."
            color = T.SUCCESS_FG
        elif status == "auth_error":
            text = result.message or "Authentication failed. Check the API key."
            color = T.DANGER_FG
        elif status == "rate_limited":
            text = result.message or "Gemini is rate-limiting requests. Try again later."
            color = T.WARNING_FG
        elif status == "timeout":
            text = result.message or "The connection test timed out."
            color = T.WARNING_FG
        else:
            text = result.message or "Gemini is unavailable. Check the endpoint and model."
            color = T.DANGER_FG

        if result.model_used and status != "success":
            text = f"{text} Model: {result.model_used}."
        self._connection_status.setText(text)
        self._connection_status.setStyleSheet(f"color: {color};")

    def _connection_test_finished(self) -> None:
        worker = self._connection_worker
        self._connection_worker = None
        self._test_connection_button.setEnabled(True)
        if self._dialog_buttons:
            self._dialog_buttons.setEnabled(True)
        if worker:
            worker.deleteLater()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._connection_worker and self._connection_worker.isRunning():
            self._connection_status.setText("Wait for the connection test to finish.")
            self._connection_status.setStyleSheet(f"color: {T.WARNING_FG};")
            event.ignore()
            return
        super().closeEvent(event)

    def _build_export_tab(self, current: dict) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        label = QLabel("Choose the encoder used for the next export.")
        label.setStyleSheet(f"color: {T.FG_SECONDARY};")
        layout.addWidget(label)

        self._encoder_group = QButtonGroup(self)
        encoders = (
            ("NVIDIA NVENC", ENCODER_NVIDIA),
            ("Intel QuickSync", ENCODER_INTEL),
            ("AMD AMF", ENCODER_AMD),
            ("Software (x264)", ENCODER_SOFTWARE),
        )
        supported_hardware = detect_supported_hardware_encoders()
        available_hardware = [
            encoder_id for _, encoder_id in encoders[:-1]
            if encoder_id in supported_hardware
        ]
        requested_encoder = str(
            current.get("encoder_id", ENCODER_SOFTWARE) or ENCODER_SOFTWARE
        )
        has_saved_encoder = _qt_settings().contains("export/encoderId")
        if requested_encoder == ENCODER_SOFTWARE and not has_saved_encoder:
            selected_encoder = available_hardware[0] if available_hardware else ENCODER_SOFTWARE
        elif requested_encoder in available_hardware or requested_encoder == ENCODER_SOFTWARE:
            selected_encoder = requested_encoder
        else:
            selected_encoder = available_hardware[0] if available_hardware else ENCODER_SOFTWARE

        self._encoder_buttons = []
        for title, encoder_id in encoders:
            radio = QRadioButton(title)
            radio.setProperty("encoder_id", encoder_id)
            self._encoder_group.addButton(radio)
            self._encoder_buttons.append(radio)
            is_supported = encoder_id == ENCODER_SOFTWARE or encoder_id in supported_hardware
            radio.setEnabled(is_supported)
            if not is_supported:
                radio.setToolTip("No compatible GPU was detected for this encoder.")
            radio.setChecked(encoder_id == selected_encoder)
            layout.addWidget(radio)

        self._debug_overlay = QCheckBox("Show zoom debug overlay")
        self._debug_overlay.setChecked(bool(current.get("debug_overlay", False)))
        self._debug_overlay.setToolTip(
            "Show keyframe markers and activity diagnostics in the editor preview."
        )
        layout.addSpacing(T.SPACE_MD)
        layout.addWidget(self._debug_overlay)
        layout.addStretch(1)
        return tab

    def _current_provider_id(self) -> str:
        return self._provider.currentData() or DEFAULT_AI_PROVIDER

    def _fill_provider_defaults(self, provider_id: str, *, only_missing: bool) -> None:
        preset = AI_PROVIDER_PRESETS.get(provider_id) or AI_PROVIDER_PRESETS[DEFAULT_AI_PROVIDER]
        fields = (
            (self._endpoint, preset.endpoint),
            (self._chat_model, preset.chat_model),
            (self._narration_model, preset.narration_model),
            (self._tts_voice, preset.tts_voice),
        )
        for field, value in fields:
            if not only_missing or not field.text().strip():
                field.setText(value)

    def _on_provider_changed(self, _index: int) -> None:
        self._fill_provider_defaults(self._current_provider_id(), only_missing=False)
        self._api_key.clear()
        self._update_provider_note()
        self._update_endpoint_trust_ui()

    def _update_endpoint_trust_ui(self) -> None:
        """Show an acknowledgement gate for non-Google HTTPS endpoints."""
        endpoint = self._endpoint.text().strip() or DEFAULT_AI_ENDPOINT
        try:
            validate_ai_endpoint(endpoint)
        except ValueError as exc:
            self._endpoint_security_note.setText(str(exc))
            self._endpoint_security_note.setStyleSheet(f"color: {T.DANGER_FG};")
            self._endpoint_security_note.setVisible(True)
            self._endpoint_trust_ack.setChecked(False)
            self._endpoint_trust_ack.setEnabled(False)
            self._endpoint_trust_ack.setVisible(False)
            return

        custom = not is_standard_google_ai_endpoint(endpoint)
        self._endpoint_trust_ack.setEnabled(custom)
        self._endpoint_trust_ack.setVisible(custom)
        if custom:
            self._endpoint_security_note.setText(
                "This custom HTTPS host is not a standard Google endpoint. "
                "Your API key will be transmitted to that third-party host."
            )
            self._endpoint_security_note.setStyleSheet(f"color: {T.WARNING_FG};")
            self._endpoint_security_note.setVisible(True)
        else:
            self._endpoint_security_note.clear()
            self._endpoint_security_note.setVisible(False)

    def _update_provider_note(self) -> None:
        preset = AI_PROVIDER_PRESETS.get(self._current_provider_id())
        self._provider_note.setText(
            (preset.description if preset else "")
            + "\n\nThe API key is stored with local Windows protection when saved."
        )

    def _browse_folder(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self,
            f"Choose {PRODUCT_NAME} output folder",
            self._folder.text(),
        )
        if chosen:
            self._folder.setText(chosen)

    def _general_config(self) -> dict:
        config = dict(self._general_config_seed)
        config.update({
            "output_folder": self._folder.text().strip(),
            "fps": self._fps.value(),
            "monitor": self._monitor.value(),
        })
        return config

    def _ai_settings(self) -> AISettings:
        return AISettings(
            provider=self._current_provider_id(),
            endpoint=self._endpoint.text().strip() or DEFAULT_AI_ENDPOINT,
            api_key=self._api_key.text().strip(),
            chat_model=self._chat_model.text().strip() or DEFAULT_AI_CHAT_MODEL,
            narration_model=self._narration_model.text().strip() or DEFAULT_NARRATION_MODEL,
            tts_voice=self._tts_voice.text().strip() or DEFAULT_TTS_VOICE,
        )

    def show_security_warning(self, message: str) -> None:
        """Open the AI tab with a credential recovery warning."""
        self._tabs.setCurrentIndex(1)
        self._endpoint_security_note.setText(message)
        self._endpoint_security_note.setStyleSheet(f"color: {T.DANGER_FG};")
        self._endpoint_security_note.setVisible(True)

    def _export_settings(self) -> dict:
        button = self._encoder_group.checkedButton()
        return {
            "encoder_id": str(button.property("encoder_id") if button else "libx264"),
            "debug_overlay": self._debug_overlay.isChecked(),
        }

    def _save(self) -> None:
        endpoint = self._endpoint.text().strip() or DEFAULT_AI_ENDPOINT
        try:
            validate_ai_endpoint(endpoint)
        except ValueError as exc:
            self._tabs.setCurrentIndex(1)
            QMessageBox.warning(self, "Invalid AI endpoint", str(exc))
            return
        if not is_standard_google_ai_endpoint(endpoint) and not self._endpoint_trust_ack.isChecked():
            self._tabs.setCurrentIndex(1)
            QMessageBox.warning(
                self,
                "Confirm custom AI endpoint",
                "Please acknowledge that your API key will be sent to an untrusted third-party host.",
            )
            return
        try:
            # Preflight before emitting the save signal so a DPAPI failure
            # cannot close the dialog while leaving the user with no key.
            protect(self._api_key.text().strip())
        except DPAPIEncryptionError as exc:
            self._tabs.setCurrentIndex(1)
            QMessageBox.critical(self, "Credential storage unavailable", str(exc))
            return
        self.settings_saved.emit(
            self._general_config(),
            self._ai_settings(),
            self._export_settings(),
        )
        self.accept()
