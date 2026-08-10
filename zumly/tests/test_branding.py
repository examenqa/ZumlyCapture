"""Regression coverage for canonical Zumly branding assets and tokens."""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication

from app import tokens as T
from app.icon_loader import get_brand_asset_path, get_brand_icon, get_brand_pixmap


BRANDING_DIR = Path(__file__).resolve().parents[1] / "app" / "branding"
GENERATED_DIR = BRANDING_DIR / "generated"
ICON_SIZES = (16, 20, 24, 32, 48, 64, 128, 256, 512)


def test_approved_brand_sources_are_vector_assets() -> None:
    symbol = BRANDING_DIR / "zumly-logo-symbol.svg"
    wordmark = BRANDING_DIR / "zumly-wordmark.svg"

    assert QSvgRenderer(str(symbol)).isValid()
    assert QSvgRenderer(str(wordmark)).isValid()
    assert "<text" not in wordmark.read_text(encoding="utf-8").lower()


def test_generated_icon_pngs_have_exact_target_dimensions() -> None:
    for size in ICON_SIZES:
        with Image.open(GENERATED_DIR / f"zumly-icon-{size}.png") as image:
            assert image.size == (size, size)
            assert image.mode == "RGBA"


def test_generated_ico_contains_all_windows_icon_sizes() -> None:
    with Image.open(GENERATED_DIR / "zumly.ico") as icon:
        sizes = set(icon.info.get("sizes", ()))

    assert set((size, size) for size in ICON_SIZES if size <= 256) <= sizes


def test_horizontal_variants_keep_identical_geometry() -> None:
    light = (GENERATED_DIR / "zumly-horizontal-light.svg").read_text(encoding="utf-8")
    dark = (GENERATED_DIR / "zumly-horizontal-dark.svg").read_text(encoding="utf-8")
    purple = (GENERATED_DIR / "zumly-horizontal-purple.svg").read_text(encoding="utf-8")

    for lockup in (light, dark, purple):
        assert 'viewBox="0 0 1120 360"' in lockup
        assert 'transform="translate(40 40) scale(0.546875)"' in lockup
        assert 'transform="translate(360 30) scale(1.166666667)"' in lockup
    assert 'stroke="#0F1738"' in light
    assert 'stroke="#FFFFFF"' in dark
    assert 'stroke="#FFFFFF"' in purple


def test_brand_loader_returns_sharp_non_null_assets() -> None:
    app = QApplication.instance() or QApplication([])

    assert get_brand_asset_path("zumly-logo-symbol.svg").is_file()
    assert get_brand_icon().isNull() is False
    for scale in (1.0, 1.5, 2.0):
        size = int(round(64 * scale))
        pixmap = get_brand_pixmap("zumly-logo-symbol.svg", size, size)
        assert pixmap.isNull() is False
        assert pixmap.deviceIndependentSize().width() == size
    _ = app


def test_brand_tokens_match_approved_palette() -> None:
    assert T.BRAND_NAVY == "#0F1738"
    assert T.BRAND_CYAN == "#08AFC0"
    assert T.BRAND_CYAN_DARK == "#079EB0"
    assert T.BRAND_PURPLE == "#6D2BD6"
    assert T.BRAND_VIOLET == "#6940DA"
    assert T.WHITE == "#FFFFFF"
    assert T.BG_LAYER_1 != "#FFFFFF"


def _relative_luminance(hex_color: str) -> float:
    channels = [int(hex_color[index:index + 2], 16) / 255.0 for index in (1, 3, 5)]
    linear = [
        channel / 12.92
        if channel <= 0.04045
        else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast_ratio(foreground: str, background: str) -> float:
    lighter, darker = sorted(
        (_relative_luminance(foreground), _relative_luminance(background)),
        reverse=True,
    )
    return (lighter + 0.05) / (darker + 0.05)


def test_core_dark_theme_pairs_keep_readable_contrast() -> None:
    assert _contrast_ratio(T.FG_PRIMARY, T.BG_LAYER_1) >= 7.0
    assert _contrast_ratio(T.FG_2, T.BG_LAYER_2) >= 4.5
    assert _contrast_ratio(T.ACCENT, T.BG_LAYER_1) >= 4.5
    assert _contrast_ratio(T.WHITE, T.BRAND) >= 4.5
