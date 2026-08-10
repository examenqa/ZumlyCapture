"""Tests for non-blocking tray recording shutdown."""

import os
import json
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from app.qt_tray import QtZumlyTray
from app.icon_loader import get_brand_icon, get_resource_path
from app.session_timing import RecordingState
from app.widgets.editor_window import EditorWindow


def test_editor_window_deletes_project_state_on_close() -> None:
    app = QApplication.instance() or QApplication([])
    window = EditorWindow(project_path=None)

    assert window.testAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

    window.close()
    app.processEvents()


def test_packaged_resource_resolver_prefers_pyinstaller_internal_path(tmp_path, monkeypatch) -> None:
    internal = tmp_path / "_internal"
    packaged_icon = internal / "zumly" / "app" / "branding" / "generated" / "zumly.ico"
    packaged_icon.parent.mkdir(parents=True)
    packaged_icon.write_bytes(b"icon")
    monkeypatch.setattr("app.icon_loader.sys.frozen", True, raising=False)
    monkeypatch.setattr("app.icon_loader.sys._MEIPASS", str(internal), raising=False)
    monkeypatch.setattr("app.icon_loader.sys.executable", str(tmp_path / "tray_app.exe"), raising=False)

    assert (
        get_resource_path("zumly/app/branding/generated/zumly.ico")
        == packaged_icon
    )


def test_brand_icon_loads_from_generated_asset() -> None:
    app = QApplication.instance() or QApplication([])
    icon = get_brand_icon()

    assert icon.isNull() is False
    assert icon.pixmap(64, 64).isNull() is False


def test_tray_initializes_persistent_non_null_icons() -> None:
    app = QApplication.instance() or QApplication([])
    tray = QtZumlyTray(app)
    tray._initialize_tray_ui()

    assert tray._tray_icon is not None
    assert tray._idle_tray_icon is not None
    assert tray._recording_tray_icon is not None
    assert tray._paused_tray_icon is not None
    assert tray._tray_icon.icon().isNull() is False

    tray._state = RecordingState.RECORDING
    tray._update_tray("Recording")
    assert tray._tray_icon.icon().isNull() is False
    tray._state = RecordingState.IDLE
    tray._update_tray("Ready")
    assert tray._tray_icon.icon().isNull() is False
    tray.deleteLater()


def test_cold_tray_open_discards_hidden_editor(monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    tray = QtZumlyTray(app)

    class RetainedEditor:
        def __init__(self) -> None:
            self.deleted = False

        def isVisible(self) -> bool:
            return False

        def deleteLater(self) -> None:
            self.deleted = True

    retained = RetainedEditor()
    tray._editor_window = retained

    created = []

    class SignalStub:
        def connect(self, _slot) -> None:
            pass

    class FreshEditor:
        def __init__(self, project_path=None) -> None:
            self.project_path = project_path
            self.destroyed = SignalStub()
            created.append(self)

        def show(self) -> None:
            pass

        def raise_(self) -> None:
            pass

        def activateWindow(self) -> None:
            pass

    monkeypatch.setattr("app.widgets.editor_window.EditorWindow", FreshEditor)

    tray._open_editor()

    assert retained.deleted is True
    assert len(created) == 1
    assert created[0].project_path is None
    assert tray._editor_window is created[0]

    tray.deleteLater()


def test_second_launch_opens_editor_only_when_tray_is_idle(monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    tray = QtZumlyTray(app)
    opened = []

    class ActivationGuard:
        def __init__(self) -> None:
            self.pending = True

        def consume_activation(self) -> bool:
            value, self.pending = self.pending, False
            return value

    tray._instance_guard = ActivationGuard()
    monkeypatch.setattr(tray, "_open_editor", lambda: opened.append(True))

    tray._consume_instance_activation()

    assert opened == [True]
    tray.deleteLater()


def test_second_launch_preserves_active_recording(monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    tray = QtZumlyTray(app)
    notices = []

    class ActivationGuard:
        def consume_activation(self) -> bool:
            return True

    tray._instance_guard = ActivationGuard()
    tray._state = RecordingState.RECORDING
    monkeypatch.setattr(tray, "_notify", notices.append)
    monkeypatch.setattr(tray, "_open_editor", lambda: (_ for _ in ()).throw(AssertionError("must not open editor")))

    tray._consume_instance_activation()

    assert notices == ["Zumly is already recording"]
    tray.deleteLater()


def test_stop_recording_only_signals_child_and_returns_immediately(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    tray = QtZumlyTray(app)

    class RunningProcess:
        def __init__(self) -> None:
            self.stop_wait_called = False

        def poll(self):
            return None

        def wait(self, *args, **kwargs):
            self.stop_wait_called = True
            raise AssertionError("Qt tray must not wait for the recorder")

    process = RunningProcess()
    stop_file = tmp_path / "stop.signal"
    tray._recording = True
    tray._process = process
    tray._stop_file = str(stop_file)
    tray._toggle_action = None

    started = time.monotonic()
    tray._stop_recording()
    elapsed = time.monotonic() - started

    assert elapsed < 0.5
    assert tray._stopping is True
    assert stop_file.read_text(encoding="utf-8") == "stop"
    assert process.stop_wait_called is False

    tray.deleteLater()


def test_pause_resume_commands_are_atomic_and_sequenced(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    tray = QtZumlyTray(app)

    class RunningProcess:
        def poll(self):
            return None

    control_file = tmp_path / "control.json"
    tray._process = RunningProcess()
    tray._control_file = str(control_file)
    tray._state = RecordingState.RECORDING
    tray._recording = True

    tray._on_pause_toggle()
    first = json.loads(control_file.read_text(encoding="utf-8"))
    assert first["sequence"] == 1
    assert first["action"] == "pause"

    tray._handle_engine_state("paused", 1)
    assert tray._state == RecordingState.PAUSED

    tray._on_pause_toggle()
    second = json.loads(control_file.read_text(encoding="utf-8"))
    assert second["sequence"] == 2
    assert second["action"] == "resume"

    tray.deleteLater()
