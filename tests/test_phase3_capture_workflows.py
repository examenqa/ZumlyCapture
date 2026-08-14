import argparse
import json
import ctypes
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from PIL import Image
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QImage, QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLineEdit, QPushButton, QToolButton

from zumly import main as capture_main
from zumly.app import qt_tray, recording_overlay, screen_recorder
from zumly.app.qt_tray import (
    HOTKEY_PAUSE_ID,
    HOTKEY_RECORD_MONITOR_ID,
    HOTKEY_RECORD_REGION_ID,
    HOTKEY_RECORD_WINDOW_ID,
    HOTKEY_SCREENSHOT_MONITOR_ID,
    HOTKEY_SCREENSHOT_REGION_ID,
    HOTKEY_SCREENSHOT_WINDOW_ID,
    HOTKEY_STOP_ID,
    MOD_CONTROL,
    MOD_SHIFT,
    QtZumlyCaptureTray,
    _HotkeyThread,
    _parse_hotkey,
)
from zumly_capture import preview_dialog, screenshot, windows_shell
from zumly_capture import settings_dialog
from zumly_capture.audio import _active_wall_segments, parse_dshow_audio_devices
from zumly_capture.capture_ui import physical_selection_rect
from zumly_capture.preview_dialog import (
    Annotation,
    AnnotationCanvas,
    CapturePreviewDialog,
    render_annotations,
)
from zumly_capture.settings import (
    default_output_folder,
    load_settings,
    normalize_settings,
    save_settings,
)
from zumly_capture.settings_dialog import CaptureSettingsDialog
from zumly_capture.wgc import NativeFrameBuffer


def test_recording_indicator_uses_smooth_premultiplied_alpha() -> None:
    pixels = recording_overlay._render_indicator_pixels(24, 24, paused=False)
    alpha = pixels[3::4]

    assert len(pixels) == 24 * 24 * 4
    assert min(alpha) == 0
    assert max(alpha) == 255
    assert any(0 < value < 255 for value in alpha)
    assert pixels[(12 * 24 + 12) * 4 : (12 * 24 + 12) * 4 + 4] == bytes(
        (68, 68, 239, 255)
    )


def test_recording_indicator_keeps_paused_state_color() -> None:
    pixels = recording_overlay._render_indicator_pixels(24, 24, paused=True)
    center = (12 * 24 + 12) * 4

    assert pixels[center : center + 4] == bytes((11, 158, 245, 255))


def test_settings_normalize_and_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "capture-settings.json"
    saved = save_settings(
        {
            "fps": 500,
            "monitor": 2,
            "countdown_seconds": -2,
            "screenshot_format": "JPEG",
            "recording_format": "GIF",
            "record_monitor_hotkey": "Alt+F9",
            "settings_schema_version": 2,
            "smart_zoom_level": 9,
        },
        path,
    )

    assert saved["fps"] == 120
    assert saved["countdown_seconds"] == 0
    assert saved["screenshot_format"] == "png"
    assert saved["recording_format"] == "gif"
    assert saved["smart_zoom_level"] == 3.0
    assert load_settings(path) == saved
    assert json.loads(path.read_text(encoding="utf-8"))["monitor"] == 2


def test_normalize_settings_migrates_phase2_general_config() -> None:
    settings = normalize_settings({"output_folder": "D:/Captures", "fps": 30, "monitor": 3})

    assert settings["output_folder"] == "D:/Captures"
    assert settings["fps"] == 30
    assert settings["monitor"] == 3
    assert settings["screenshot_monitor_hotkey"] == "Ctrl+Alt+1"
    assert settings["screenshot_window_hotkey"] == "Ctrl+Alt+2"
    assert settings["screenshot_region_hotkey"] == "Ctrl+Alt+3"
    assert settings["record_monitor_hotkey"] == "Ctrl+Alt+4"
    assert settings["record_window_hotkey"] == "Ctrl+Alt+5"
    assert settings["record_region_hotkey"] == "Ctrl+Alt+6"
    assert settings["pause_hotkey"] == "Ctrl+Alt+9"
    assert settings["stop_hotkey"] == "Ctrl+Alt+0"
    assert settings["preview_after_capture"] is True
    assert settings["smart_zoom_enabled"] is True
    assert settings["render_cursor"] is True


def test_default_recording_location_and_countdown_are_lightweight() -> None:
    settings = normalize_settings({"settings_schema_version": 2})

    assert Path(default_output_folder()).parts[-2:] == ("Videos", "Zumly Capture")
    assert settings["countdown_seconds"] == 1
    assert settings["recording_format"] == "mp4"


def test_invalid_recording_format_falls_back_to_mp4() -> None:
    settings = normalize_settings({"recording_format": "webm"})

    assert settings["recording_format"] == "mp4"


def test_show_in_folder_opens_the_capture_parent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_folder = tmp_path / "Videos" / "Zumly Capture"
    capture_folder.mkdir(parents=True)
    capture = capture_folder / "recording.mp4"
    capture.write_bytes(b"video")
    opened: list[str] = []
    monkeypatch.setattr(windows_shell.os, "startfile", opened.append)

    revealed = windows_shell.reveal_in_folder(str(capture))

    assert revealed == str(capture_folder.resolve())
    assert opened == [str(capture_folder.resolve())]


def test_capture_encoder_hint_skips_fresh_process_probe(monkeypatch) -> None:
    screen_recorder._available_capture_encoders.cache_clear()
    monkeypatch.setenv(
        screen_recorder.CAPTURE_ENCODERS_ENV,
        "h264_qsv,unknown_encoder,libx264",
    )
    monkeypatch.setattr(
        screen_recorder,
        "detect_available_encoders",
        lambda: (_ for _ in ()).throw(AssertionError("encoder probe ran")),
    )

    available = screen_recorder._available_capture_encoders("ffmpeg.exe")

    assert available == {"h264_qsv", "libx264"}
    screen_recorder._available_capture_encoders.cache_clear()


def test_gdi_monitor_signals_ready_before_its_capture_loop(monkeypatch) -> None:
    recorder = screen_recorder.ScreenRecorder()
    recorder._capturing = True
    recorder._monitor_index = 1
    recorder._fps = 30
    grab_count = [0]
    ready_after_grabs: list[int] = []

    class ReadyEvent:
        def set(self) -> None:
            ready_after_grabs.append(grab_count[0])

    class FakeCapture:
        monitors = [{}, {"width": 8, "height": 6}]

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def grab(self, _monitor):
            grab_count[0] += 1
            recorder._capturing = False
            return SimpleNamespace(width=8, height=6, bgra=bytes(8 * 6 * 4))

    recorder._capture_pipeline_ready_event = ReadyEvent()
    monkeypatch.setitem(sys.modules, "mss", SimpleNamespace(mss=FakeCapture))
    monkeypatch.setattr(
        screen_recorder,
        "_capture_encoder_args",
        lambda _ffmpeg: ("libx264", []),
    )
    monkeypatch.setattr(screen_recorder, "_ffmpeg_exe", lambda: "ffmpeg.exe")

    recorder._capture_loop_mss()

    assert ready_after_grabs[0] == 0


def test_phase5_settings_preserve_custom_number_shortcuts() -> None:
    settings = normalize_settings(
        {
            "settings_schema_version": 2,
            "screenshot_monitor_hotkey": "Alt+F1",
            "smart_zoom_enabled": False,
        }
    )

    assert settings["screenshot_monitor_hotkey"] == "Alt+F1"
    assert settings["smart_zoom_enabled"] is False


def test_dshow_device_parser_keeps_audio_devices_only() -> None:
    output = """
    [dshow @ 1] \"Integrated Camera\" (video)
    [dshow @ 1] \"Microphone (USB Audio)\" (audio)
    [dshow @ 1] \"Stereo Mix\" (audio)
    [dshow @ 1] \"Microphone (USB Audio)\" (audio)
    """

    assert parse_dshow_audio_devices(output) == [
        "Microphone (USB Audio)",
        "Stereo Mix",
    ]


def test_audio_segments_remove_pause_wall_time() -> None:
    segments = _active_wall_segments(
        10_000,
        [
            {"timelineMs": 2_000, "wallDurationMs": 3_000},
            {"timelineMs": 7_000, "wallDurationMs": 1_000},
        ],
        100,
    )

    assert segments == [
        (100.0, 2100.0),
        (5100.0, 10100.0),
        (11100.0, 14100.0),
    ]


def test_hotkey_parser_supports_letters_and_function_keys() -> None:
    assert _parse_hotkey("Ctrl+Shift+R") == (MOD_CONTROL | MOD_SHIFT, ord("R"))
    assert _parse_hotkey("Alt+F9") == (0x0001, 0x78)
    assert _parse_hotkey("Ctrl+Alt+1") == (MOD_CONTROL | 0x0001, ord("1"))


def test_hotkey_thread_reports_registration_conflicts(monkeypatch) -> None:
    unregistered: list[int] = []

    class User32:
        @staticmethod
        def RegisterHotKey(_window, hotkey_id, _modifiers, _key) -> bool:
            return hotkey_id != 1

        @staticmethod
        def GetMessageW(_message, _window, _minimum, _maximum) -> int:
            return 0

        @staticmethod
        def UnregisterHotKey(_window, hotkey_id) -> None:
            unregistered.append(hotkey_id)

    class Kernel32:
        @staticmethod
        def GetCurrentThreadId() -> int:
            return 42

        @staticmethod
        def GetLastError() -> int:
            return qt_tray.ERROR_HOTKEY_ALREADY_REGISTERED

    monkeypatch.setattr(
        qt_tray.ctypes,
        "windll",
        SimpleNamespace(user32=User32(), kernel32=Kernel32()),
    )
    thread = _HotkeyThread(
        lambda _hotkey_id: None,
        {1: "Ctrl+Alt+1", 2: "Ctrl+Alt+2"},
    )

    thread.run()

    assert thread.failures == (
        ("Ctrl+Alt+1", "Already in use by another app"),
    )
    assert unregistered == [2]


def test_tray_menu_lists_shortcut_registration_issues() -> None:
    app = QApplication.instance() or QApplication([])
    tray = QtZumlyCaptureTray(app)
    tray._initialize_tray_ui()

    tray._update_hotkey_status(
        (
            ("Ctrl+Alt+1", "Already in use by another app"),
            ("Ctrl+Alt+5", "Could not be registered"),
        )
    )

    assert tray._hotkey_status_menu is not None
    assert tray._hotkey_status_menu.menuAction().isVisible() is True
    assert tray._hotkey_status_menu.title() == "Shortcut issues (2)"
    assert [action.text() for action in tray._hotkey_status_menu.actions()] == [
        "Ctrl+Alt+1 — Already in use by another app",
        "Ctrl+Alt+5 — Could not be registered",
    ]

    tray._update_hotkey_status(())
    assert tray._hotkey_status_menu.menuAction().isVisible() is False
    tray.deleteLater()


def test_tray_menu_has_compact_aligned_capture_mode_submenus() -> None:
    app = QApplication.instance() or QApplication([])
    tray = QtZumlyCaptureTray(app)
    tray._initialize_tray_ui()

    assert tray._tray_icon is not None
    menu = tray._tray_icon.contextMenu()
    assert menu is not None
    assert menu.minimumWidth() == 282

    assert tray._screenshot_menu is not None
    assert tray._recording_menu is not None
    screenshot_menu = tray._screenshot_menu
    recording_menu = tray._recording_menu

    assert screenshot_menu.minimumWidth() == 236
    assert recording_menu.minimumWidth() == 236
    assert [action.text() for action in screenshot_menu.actions()] == [
        "Full Monitor",
        "Active Window",
        "Select Region",
    ]
    assert [action.shortcut().toString() for action in screenshot_menu.actions()] == [
        "Ctrl+Alt+1",
        "Ctrl+Alt+2",
        "Ctrl+Alt+3",
    ]
    assert [action.text() for action in recording_menu.actions() if not action.isSeparator()] == [
        "Full Monitor",
        "Choose Window…",
        "Select Region…",
        "Automatic Smart Zoom",
    ]
    assert not any(action.text() == "Record" for action in menu.actions())

    assert tray._toggle_action is not None
    assert tray._toggle_action.isVisible() is False
    tray._state = qt_tray.RecordingState.RECORDING
    tray._update_tray("Recording")
    assert tray._toggle_action.isVisible() is True
    assert tray._toggle_action.text() == "Stop Recording"
    assert recording_menu.menuAction().isVisible() is False

    tray._state = qt_tray.RecordingState.IDLE
    tray._update_tray("Ready")
    assert tray._toggle_action.isVisible() is False
    assert recording_menu.menuAction().isVisible() is True
    tray.deleteLater()


def test_number_hotkeys_route_to_all_capture_actions() -> None:
    calls: list[str] = []

    class TrayActions:
        _screenshot_monitor = lambda self: calls.append("screenshot-monitor")
        _screenshot_active_window = lambda self: calls.append("screenshot-window")
        _screenshot_region = lambda self: calls.append("screenshot-region")
        _record_monitor = lambda self: calls.append("record-monitor")
        _record_window = lambda self: calls.append("record-window")
        _record_region = lambda self: calls.append("record-region")
        _on_pause_toggle = lambda self: calls.append("pause")
        _on_stop_hotkey = lambda self: calls.append("stop")

    tray = TrayActions()
    for hotkey_id in (
        HOTKEY_SCREENSHOT_MONITOR_ID,
        HOTKEY_SCREENSHOT_WINDOW_ID,
        HOTKEY_SCREENSHOT_REGION_ID,
        HOTKEY_RECORD_MONITOR_ID,
        HOTKEY_RECORD_WINDOW_ID,
        HOTKEY_RECORD_REGION_ID,
        HOTKEY_PAUSE_ID,
        HOTKEY_STOP_ID,
    ):
        QtZumlyCaptureTray._handle_hotkey(tray, hotkey_id)

    assert calls == [
        "screenshot-monitor",
        "screenshot-window",
        "screenshot-region",
        "record-monitor",
        "record-window",
        "record-region",
        "pause",
        "stop",
    ]


def test_region_selection_uses_physical_desktop_coordinates() -> None:
    assert physical_selection_rect(QPoint(2400, -120), QPoint(1200, 780)) == {
        "left": 1200,
        "top": -120,
        "width": 1200,
        "height": 900,
    }


def test_annotation_renderer_composites_without_mutating_source() -> None:
    source = QImage(120, 80, QImage.Format.Format_ARGB32)
    source.fill(QColor("white"))
    annotation = Annotation(
        kind="rectangle",
        start=QPoint(10, 10),
        end=QPoint(100, 60),
        color=QColor("#ff0000"),
        width=5,
    )

    rendered = render_annotations(source, [annotation])

    assert source.pixelColor(10, 10) == QColor("white")
    assert rendered.pixelColor(10, 10) != QColor("white")


def test_arrow_annotation_uses_a_substantial_filled_head() -> None:
    source = QImage(150, 100, QImage.Format.Format_ARGB32)
    source.fill(QColor("white"))
    arrow = Annotation(
        kind="arrow",
        start=QPointF(20, 50),
        end=QPointF(125, 50),
        color=QColor("#ff0000"),
        width=5,
    )

    rendered = render_annotations(source, [arrow])

    assert rendered.pixelColor(100, 44).red() > 220
    assert rendered.pixelColor(100, 44).green() < 40


def test_arrow_has_a_pointed_tapered_tail_and_angled_head() -> None:
    source = QImage(190, 120, QImage.Format.Format_ARGB32)
    source.fill(QColor("white"))
    arrow = Annotation(
        kind="arrow",
        start=QPointF(20, 60),
        end=QPointF(165, 60),
        color=QColor("#ff0000"),
        width=5,
    )

    rendered = render_annotations(source, [arrow])

    assert rendered.pixelColor(20, 60).red() > 220
    assert rendered.pixelColor(22, 55) == QColor("white")
    assert rendered.pixelColor(45, 57) == QColor("white")
    assert rendered.pixelColor(115, 57).red() > 220
    assert rendered.pixelColor(130, 45).red() > 220
    assert rendered.pixelColor(165, 60).red() > 220


def test_filled_arrow_does_not_fill_later_outline_shapes() -> None:
    source = QImage(180, 120, QImage.Format.Format_ARGB32)
    source.fill(QColor("white"))
    annotations = [
        Annotation(
            "arrow",
            QPointF(10, 100),
            QPointF(70, 70),
            QColor("#ff0000"),
            5,
        ),
        Annotation(
            "rectangle",
            QPointF(90, 20),
            QPointF(170, 90),
            QColor("#0088ff"),
            4,
        ),
    ]

    rendered = render_annotations(source, annotations)

    assert rendered.pixelColor(130, 55) == QColor("white")


def test_mask_annotation_uses_an_opaque_thick_stroke() -> None:
    source = QImage(140, 80, QImage.Format.Format_ARGB32)
    source.fill(QColor("white"))
    mask = Annotation(
        "mask",
        QPointF(20, 40),
        QPointF(120, 40),
        QColor("#ff0000"),
        26,
    )

    rendered = render_annotations(source, [mask])

    assert rendered.pixelColor(70, 40) == QColor("#ff0000")
    assert rendered.pixelColor(70, 28).red() > 220


def test_text_annotation_is_entered_directly_on_the_canvas(tmp_path: Path) -> None:
    _app = QApplication.instance() or QApplication([])
    image_path = tmp_path / "canvas.png"
    image = QImage(320, 180, QImage.Format.Format_ARGB32)
    image.fill(QColor("white"))
    assert image.save(str(image_path), "PNG")
    canvas = AnnotationCanvas(str(image_path))
    canvas.resize(640, 380)

    canvas._begin_inline_text(QPointF(80, 60), QPointF(160, 130))
    assert canvas._inline_editor is not None
    assert canvas._inline_editor.autoFillBackground() is False
    assert canvas._inline_editor.testAttribute(
        Qt.WidgetAttribute.WA_TranslucentBackground
    )
    assert canvas._inline_editor.styleSheet() == ""
    assert canvas.findChildren(QLineEdit) == []
    empty_width = canvas._inline_editor.width()
    QTest.keyClicks(canvas._inline_editor, "Direct canvas text")
    assert canvas._inline_editor.width() > empty_width
    QTest.keyClick(canvas._inline_editor, Qt.Key.Key_Return)

    assert canvas.annotations[-1].kind == "text"
    assert canvas.annotations[-1].text == "Direct canvas text"
    canvas.undo()
    assert canvas.annotations == []
    canvas.deleteLater()


def test_annotation_toolbar_defaults_to_wider_arrow_and_uses_mask(
    tmp_path: Path,
) -> None:
    _app = QApplication.instance() or QApplication([])
    image_path = tmp_path / "annotation-tools.png"
    image = QImage(320, 180, QImage.Format.Format_ARGB32)
    image.fill(QColor("white"))
    assert image.save(str(image_path), "PNG")

    dialog = CapturePreviewDialog(str(image_path))
    tools = {button.text(): button for button in dialog.findChildren(QToolButton)}

    assert "Mask" in tools
    assert "Highlight" not in tools
    assert tools["Arrow"].isChecked() is True
    assert tools["Pen"].isChecked() is False
    assert dialog._canvas._tool == "arrow"

    dialog._canvas.resize(640, 380)
    QTest.mousePress(
        dialog._canvas,
        Qt.MouseButton.LeftButton,
        pos=QPoint(100, 100),
    )
    assert dialog._canvas._preview is not None
    assert dialog._canvas._preview.width == 8.0
    QTest.mouseRelease(
        dialog._canvas,
        Qt.MouseButton.LeftButton,
        pos=QPoint(280, 100),
    )

    tools["Mask"].click()
    QTest.mousePress(
        dialog._canvas,
        Qt.MouseButton.LeftButton,
        pos=QPoint(100, 140),
    )
    assert dialog._canvas._preview is not None
    assert dialog._canvas._preview.kind == "mask"
    assert dialog._canvas._preview.width == 26.0
    QTest.mouseRelease(
        dialog._canvas,
        Qt.MouseButton.LeftButton,
        pos=QPoint(280, 140),
    )
    dialog.close()
    dialog.deleteLater()


def test_undo_removes_whole_strokes_including_the_initial_action(
    tmp_path: Path,
) -> None:
    _app = QApplication.instance() or QApplication([])
    image_path = tmp_path / "undo-actions.png"
    image = QImage(320, 180, QImage.Format.Format_ARGB32)
    image.fill(QColor("white"))
    assert image.save(str(image_path), "PNG")
    canvas = AnnotationCanvas(str(image_path))
    canvas.resize(640, 380)
    canvas.set_tool("mask")

    for y in (70, 110):
        QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=QPoint(80, y))
        QTest.mouseMove(canvas, QPoint(180, y))
        QTest.mouseMove(canvas, QPoint(280, y))
        QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=QPoint(300, y))

    assert len(canvas.annotations) == 4
    canvas.undo()
    assert len(canvas.annotations) == 2
    canvas.undo()
    assert canvas.annotations == []
    canvas.deleteLater()


def test_undo_button_uses_ctrl_z_and_removes_the_first_arrow(tmp_path: Path) -> None:
    _app = QApplication.instance() or QApplication([])
    image_path = tmp_path / "undo-shortcut.png"
    image = QImage(320, 180, QImage.Format.Format_ARGB32)
    image.fill(QColor("white"))
    assert image.save(str(image_path), "PNG")
    dialog = CapturePreviewDialog(str(image_path))
    dialog.resize(760, 520)
    dialog.show()
    QApplication.processEvents()

    assert dialog._undo_shortcut is not None
    assert dialog._undo_shortcut.key() == QKeySequence(QKeySequence.StandardKey.Undo)
    QTest.mousePress(dialog._canvas, Qt.MouseButton.LeftButton, pos=QPoint(100, 100))
    QTest.mouseRelease(dialog._canvas, Qt.MouseButton.LeftButton, pos=QPoint(280, 100))
    assert len(dialog._canvas.annotations) == 1

    QTest.keyClick(dialog, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    QApplication.processEvents()
    assert dialog._canvas.annotations == []
    dialog.close()
    dialog.deleteLater()


def test_preview_actions_have_one_save_and_no_redundant_open(tmp_path: Path) -> None:
    _app = QApplication.instance() or QApplication([])
    image_path = tmp_path / "preview.png"
    image = QImage(320, 180, QImage.Format.Format_ARGB32)
    image.fill(QColor("white"))
    assert image.save(str(image_path), "PNG")

    dialog = CapturePreviewDialog(str(image_path))
    buttons = {button.text(): button for button in dialog.findChildren(QPushButton)}

    assert "Save" in buttons
    assert "Open" not in buttons
    assert buttons["Show in folder"].isEnabled() is False
    dialog.close()
    dialog.deleteLater()


def test_recording_preview_controls_reserve_clear_indicator_space(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(preview_dialog, "QMediaPlayer", None)
    monkeypatch.setattr(preview_dialog, "QVideoWidget", None)
    monkeypatch.setattr(preview_dialog, "QAudioOutput", None)
    recording_path = tmp_path / "preview.mp4"
    unzoomed_path = tmp_path / "preview-unzoomed.mp4"
    recording_path.write_bytes(b"placeholder")
    unzoomed_path.write_bytes(b"placeholder")

    dialog = CapturePreviewDialog(
        str(recording_path),
        unzoomed_path=str(unzoomed_path),
    )

    assert dialog._remove_zoom is not None
    assert dialog._remove_zoom.objectName() == "removeSmartZoom"
    assert dialog._format_combo is not None
    assert dialog._format_combo.objectName() == "recordingFormat"
    style = dialog.styleSheet()
    assert "QCheckBox#removeSmartZoom::indicator" in style
    assert "border: 2px solid #62aaf0" in style
    assert "QComboBox#recordingFormat::drop-down" in style
    assert "width: 32px" in style
    assert "padding: 7px 40px 7px 10px" in style
    dialog.close()
    dialog.deleteLater()


def test_active_window_is_resolved_after_tray_menu_closes(monkeypatch) -> None:
    tray = QtZumlyCaptureTray(object())
    tray._cfg = {"screenshot_delay_seconds": 0}
    callbacks: list[object] = []
    captures: list[dict] = []
    foreground_calls: list[bool] = []
    monkeypatch.setattr(
        qt_tray.QTimer,
        "singleShot",
        lambda delay, callback: callbacks.append((delay, callback)),
    )
    monkeypatch.setattr(
        qt_tray,
        "foreground_window_handle",
        lambda: foreground_calls.append(True) or 123,
    )
    monkeypatch.setattr(
        qt_tray,
        "get_window_rect",
        lambda handle: {"left": handle, "top": 0, "width": 10, "height": 10},
    )
    tray._capture_screenshot = captures.append

    tray._screenshot_active_window()

    assert foreground_calls == []
    assert callbacks[0][0] == 180
    callbacks[0][1]()
    assert foreground_calls == [True]
    assert captures[0]["left"] == 123
    tray.deleteLater()


def test_settings_dialog_defers_audio_device_enumeration(monkeypatch) -> None:
    _app = QApplication.instance() or QApplication([])
    discovery_scheduled: list[object] = []
    monkeypatch.setattr(
        settings_dialog.QTimer,
        "singleShot",
        lambda delay, callback: discovery_scheduled.append((delay, callback)),
    )
    monkeypatch.setattr(
        settings_dialog,
        "list_dshow_audio_devices",
        lambda: (_ for _ in ()).throw(AssertionError("enumerated during construction")),
    )

    dialog = CaptureSettingsDialog({"microphone_device": "Saved microphone"})

    assert discovery_scheduled[0][0] == 0
    assert dialog._microphone.isEnabled() is False
    assert dialog._microphone.currentText() == "Detecting devices…"
    dialog._populate_audio_devices(["USB microphone"])
    assert dialog._microphone.isEnabled() is True
    assert dialog._microphone.currentData() == "Saved microphone"
    dialog.deleteLater()


def test_native_wgc_buffer_copies_without_numpy() -> None:
    source = (ctypes.c_ubyte * 16)(*range(16))
    destination = bytearray(16)
    buffer = NativeFrameBuffer(source, 16, 2, 2)

    assert buffer.copy_into(memoryview(destination), 2, 2) is True
    assert destination == bytes(range(16))


def test_screenshot_publication_is_atomic_and_non_overwriting(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        screenshot,
        "capture_rect_image",
        lambda _rect: Image.new("RGB", (8, 6), (10, 20, 30)),
    )
    target = tmp_path / "shot.png"
    rect = {"left": 0, "top": 0, "width": 8, "height": 6}

    published = screenshot.publish_screenshot(rect, target, "png")

    assert published == str(target.resolve())
    with Image.open(target) as image:
        assert image.size == (8, 6)
    try:
        screenshot.publish_screenshot(rect, target, "png")
    except FileExistsError:
        pass
    else:
        raise AssertionError("Existing screenshot was overwritten")


def test_capture_setup_failure_releases_all_started_resources(monkeypatch) -> None:
    events: list[str] = []
    startup_sleeps: list[float] = []

    class Recorder:
        def __init__(self, **_kwargs) -> None:
            pass

        def start_capture(self, *_args) -> None:
            events.append("capture-start")

        def prepare_recording(self) -> str:
            return "raw.mp4"

        def start_recording(self, **_kwargs) -> None:
            events.append("recording-start")

        def stop_recording(self) -> None:
            events.append("recording-stop")

        def stop_capture(self) -> None:
            events.append("capture-stop")

    class Audio:
        started_at = 0.0

        def __init__(self, _devices) -> None:
            pass

        def start(self) -> None:
            self.started_at = capture_main.time.perf_counter()
            events.append("audio-start")

        def stop(self) -> list:
            events.append("audio-stop")
            return []

    class Clicks:
        def is_button_down(self) -> bool:
            return False

        def start(self, *_args, **_kwargs) -> None:
            events.append("click-start")

        def stop(self) -> list:
            events.append("click-stop")
            return []

    class Mouse:
        def __init__(self, **_kwargs) -> None:
            pass

        def start(self, *_args, **_kwargs) -> None:
            events.append("mouse-start")

        def stop(self) -> list:
            events.append("mouse-stop")
            return []

    class Hotkeys:
        def __init__(self, **_kwargs) -> None:
            pass

        def register_record_hotkey(self) -> None:
            events.append("hotkey-start")

        def unregister_record_hotkey(self) -> None:
            events.append("hotkey-stop")

    class Overlay:
        def __init__(self, _rect) -> None:
            pass

        def start(self) -> None:
            events.append("overlay-start")

        def stop(self) -> None:
            events.append("overlay-stop")

    monkeypatch.setattr(capture_main, "ScreenRecorder", Recorder)
    monkeypatch.setattr(capture_main, "AudioCapture", Audio)
    monkeypatch.setattr(capture_main, "ClickTracker", Clicks)
    monkeypatch.setattr(capture_main, "MouseTracker", Mouse)
    monkeypatch.setattr(capture_main, "GlobalHotkeys", Hotkeys)
    monkeypatch.setattr(capture_main, "RecordingOverlay", Overlay)
    monkeypatch.setattr(capture_main.time, "sleep", startup_sleeps.append)
    monkeypatch.setattr(
        capture_main,
        "_write_status_payload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("status failed")),
    )
    args = argparse.Namespace(
        fps=60,
        monitor=1,
        window_hwnd=0,
        microphone="",
        system_audio="",
        stop_file="",
        status_file="status.json",
        control_file="",
        duration=0.0,
    )

    with pytest.raises(RuntimeError, match="status failed"):
        capture_main._record_media(
            args,
            "monitor",
            {"left": 0, "top": 0, "width": 100, "height": 100},
        )

    assert events[-7:] == [
        "recording-stop",
        "mouse-stop",
        "click-stop",
        "audio-stop",
        "hotkey-stop",
        "overlay-stop",
        "capture-stop",
    ]
    assert 2.0 not in startup_sleeps
