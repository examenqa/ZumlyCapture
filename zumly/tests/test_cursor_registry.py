from pathlib import Path
import json

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from app.cursor_registry import (
    CURSOR_PRESETS,
    click_effect_name_for_cursor,
    cursor_asset_scale,
    cursor_hotspot,
    cursor_svg_path,
    ensure_cursor_asset,
)
from app.widgets.cursor_panel import CursorPanelWidget
from app.models import MousePosition, RecordingSession
from app.cursor_renderer import draw_cursor_qpainter


def test_builtin_cursor_assets_have_declared_hotspots() -> None:
    ensure_cursor_asset.cache_clear()

    for preset in CURSOR_PRESETS:
        if preset.style_id == "custom":
            continue
        if preset.asset_name:
            assert Path(cursor_svg_path(preset.style_id)).is_file()
        path = ensure_cursor_asset(preset.style_id)
        with Image.open(path) as image:
            scale = int(cursor_asset_scale(preset.style_id))
            assert image.size == (preset.width * scale, preset.height * scale)
        assert cursor_hotspot(preset.style_id) == (preset.hotspot_x, preset.hotspot_y)
        assert 0 <= preset.hotspot_x < preset.width
        assert 0 <= preset.hotspot_y < preset.height
    ensure_cursor_asset.cache_clear()


def test_custom_cursor_defaults_to_origin_hotspot() -> None:
    assert cursor_hotspot("custom") == (0, 0)


def test_cursor_styles_choose_complementary_click_effects() -> None:
    assert click_effect_name_for_cursor("violet_pointer") == "Subtle Purple"
    assert click_effect_name_for_cursor("sky_pointer") == "Neon Cyan"
    assert click_effect_name_for_cursor("mint_head") == "Soft Green"
    assert click_effect_name_for_cursor("custom") == "Clean White"


def test_svg_cursor_renders_directly_into_qpainter_target() -> None:
    app = QApplication.instance() or QApplication([])
    image = QImage(320, 240, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(0)
    painter = QPainter(image)

    draw_cursor_qpainter(
        painter,
        track=[MousePosition(x=50, y=50, timestamp=0.0)],
        time_ms=0.0,
        monitor_rect={"left": 0, "top": 0, "width": 100, "height": 100},
        screen_x=0,
        screen_y=0,
        screen_w=320,
        screen_h=240,
        cursor_style_id="filled_arrow",
    )
    painter.end()

    assert any(
        image.pixelColor(x, y).alpha() > 0
        for y in range(image.height())
        for x in range(image.width())
    )


def test_cursor_gallery_exposes_stylized_heads_and_emits_selection() -> None:
    app = QApplication.instance() or QApplication([])
    panel = CursorPanelWidget()
    assert {"ink_pointer", "violet_pointer", "sky_pointer", "midnight_pointer"} <= set(panel._buttons)
    assert {
        "tangerine_wedge",
        "cobalt_arrow",
        "orchid_pointer",
        "coral_pointer",
        "signal_pointer",
        "aqua_pointer",
        "periwinkle_pointer",
        "prism_pointer",
        "aurora_pointer",
        "ember_head",
        "sapphire_head",
        "lilac_head",
        "ruby_head",
        "mint_head",
        "violet_head",
        "cyan_wedge",
        "outline_soft",
        "graphite_stripe",
        "lavender_soft",
    } <= set(panel._buttons)
    assert panel._buttons["violet_pointer"].icon().isNull() is False
    assert panel._buttons["prism_pointer"].icon().isNull() is False
    assert panel._buttons["violet_head"].icon().isNull() is False

    selected = []
    panel.cursor_selection_changed.connect(lambda path, style: selected.append((path, style)))
    panel._buttons["sky_pointer"].click()

    assert selected[-1] == ("", "sky_pointer")
    assert panel._buttons["sky_pointer"].isChecked() is True


def test_cursor_gallery_size_control_emits_bounded_scale() -> None:
    app = QApplication.instance() or QApplication([])
    panel = CursorPanelWidget()
    scales = []
    panel.cursor_scale_changed.connect(scales.append)

    assert panel._scale_slider.minimum() == 50
    assert panel._scale_slider.maximum() == 150
    assert panel._scale_slider.value() == 100
    assert panel._cursor_scale == 1.75

    panel._scale_slider.setValue(125)

    assert scales[-1] == 2.1875
    assert panel._scale_value.text() == "125%"

    panel._scale_slider.setValue(50)
    assert scales[-1] == 0.875
    panel._scale_slider.setValue(150)
    assert scales[-1] == 2.625


def test_existing_cursor_scale_keeps_visual_size_with_recalibrated_label() -> None:
    app = QApplication.instance() or QApplication([])
    panel = CursorPanelWidget()

    panel.set_cursor_asset("", "arrow", 2.5)

    assert panel._cursor_scale == 2.5
    assert panel._scale_slider.value() == 143
    assert panel._scale_value.text() == "143%"


def test_cursor_gallery_exposes_one_click_ripple_toggle() -> None:
    app = QApplication.instance() or QApplication([])
    panel = CursorPanelWidget()
    changes = []
    panel.click_effect_enabled_changed.connect(changes.append)

    panel._click_effect_toggle.setChecked(False)

    assert changes == [False]
    panel.set_click_effect_enabled(True)
    assert panel._click_effect_toggle.isChecked() is True
    assert changes == [False]


def test_cursor_scale_roundtrips_in_recording_session() -> None:
    session = RecordingSession(
        id="cursor-scale",
        start_time=0.0,
        duration=1000.0,
        mouse_track=[],
        keyframes=[],
        cursor_style_id="violet_pointer",
        cursor_scale=2.25,
    )

    loaded = RecordingSession.from_json(session.to_json())

    assert loaded.cursor_style_id == "violet_pointer"
    assert loaded.cursor_scale == 2.25


def test_new_and_legacy_cursor_scale_defaults_are_backward_compatible() -> None:
    session = RecordingSession(
        id="cursor-default",
        start_time=0.0,
        duration=1000.0,
        mouse_track=[],
        keyframes=[],
    )
    assert session.cursor_scale == 1.75

    legacy_payload = json.loads(session.to_json())
    legacy_payload.pop("cursorScale")
    legacy = RecordingSession.from_json(json.dumps(legacy_payload))
    assert legacy.cursor_scale == 2.5
