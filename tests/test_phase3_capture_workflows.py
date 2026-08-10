import json
import ctypes
from pathlib import Path

from PIL import Image

from zumly.app.qt_tray import MOD_CONTROL, MOD_SHIFT, _parse_hotkey
from zumly_capture import screenshot
from zumly_capture.audio import _active_wall_segments, parse_dshow_audio_devices
from zumly_capture.settings import load_settings, normalize_settings, save_settings
from zumly_capture.wgc import NativeFrameBuffer


def test_settings_normalize_and_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "capture-settings.json"
    saved = save_settings(
        {
            "fps": 500,
            "monitor": 2,
            "countdown_seconds": -2,
            "screenshot_format": "JPEG",
            "record_hotkey": "Alt+F9",
            "smart_zoom_level": 9,
        },
        path,
    )

    assert saved["fps"] == 120
    assert saved["countdown_seconds"] == 0
    assert saved["screenshot_format"] == "png"
    assert saved["smart_zoom_level"] == 3.0
    assert load_settings(path) == saved
    assert json.loads(path.read_text(encoding="utf-8"))["monitor"] == 2


def test_normalize_settings_migrates_phase2_general_config() -> None:
    settings = normalize_settings({"output_folder": "D:/Captures", "fps": 30, "monitor": 3})

    assert settings["output_folder"] == "D:/Captures"
    assert settings["fps"] == 30
    assert settings["monitor"] == 3
    assert settings["screenshot_hotkey"] == "Ctrl+Shift+S"
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
