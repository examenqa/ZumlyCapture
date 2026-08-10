"""Built-in cursor styles and their click hotspots."""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)
_CURSOR_RENDER_VERSION = 4
_CURSOR_SUPERSAMPLE = 4


@dataclass(frozen=True)
class CursorPreset:
    style_id: str
    name: str
    width: int
    height: int
    hotspot_x: int
    hotspot_y: int
    asset_name: str = ""
    asset_scale: int = _CURSOR_SUPERSAMPLE


CURSOR_PRESETS = (
    # High-resolution presentation cursors. These are intentionally listed
    # first so the visual gallery leads with the styles used in polished demos.
    CursorPreset("tangerine_wedge", "Tangerine Wedge", 128, 160, 9, 10, "modern_tangerine.svg", 1),
    CursorPreset("cobalt_arrow", "Cobalt Arrow", 128, 160, 10, 9, "modern_cobalt.svg", 1),
    CursorPreset("orchid_pointer", "Orchid Pointer", 128, 160, 10, 10, "modern_orchid.svg", 1),
    CursorPreset("coral_pointer", "Coral Pointer", 128, 160, 10, 9, "modern_coral.svg", 1),
    CursorPreset("signal_pointer", "Signal Red", 128, 160, 9, 8, "modern_signal.svg", 1),
    CursorPreset("aqua_pointer", "Aqua Pointer", 128, 160, 10, 10, "modern_aqua.svg", 1),
    CursorPreset("periwinkle_pointer", "Periwinkle", 128, 160, 10, 10, "modern_periwinkle.svg", 1),
    CursorPreset("prism_pointer", "Prism", 128, 160, 9, 10, "modern_prism.svg", 1),
    CursorPreset("aurora_pointer", "Aurora", 128, 160, 9, 9, "modern_aurora.svg", 1),
    CursorPreset("ember_head", "Ember Head", 128, 128, 9, 9, "head_ember.svg", 1),
    CursorPreset("sapphire_head", "Sapphire Head", 128, 128, 9, 9, "head_sapphire.svg", 1),
    CursorPreset("lilac_head", "Lilac Head", 128, 128, 9, 9, "head_lilac.svg", 1),
    CursorPreset("ruby_head", "Ruby Head", 128, 128, 9, 9, "head_ruby.svg", 1),
    CursorPreset("mint_head", "Mint Head", 128, 128, 9, 9, "head_mint.svg", 1),
    CursorPreset("violet_head", "Violet Head", 128, 128, 9, 9, "head_violet.svg", 1),
    CursorPreset("cyan_wedge", "Cyan Wedge", 128, 128, 9, 9, "head_cyan_wedge.svg", 1),
    CursorPreset("outline_soft", "Outline Soft", 128, 128, 9, 9, "head_outline_soft.svg", 1),
    CursorPreset("graphite_stripe", "Graphite Stripe", 128, 128, 9, 9, "head_graphite.svg", 1),
    CursorPreset("lavender_soft", "Lavender Soft", 128, 128, 9, 9, "head_lavender.svg", 1),
    CursorPreset("ink_pointer", "Ink Black", 128, 160, 14, 14),
    CursorPreset("violet_pointer", "Violet", 128, 160, 14, 14),
    CursorPreset("sky_pointer", "Sky Blue", 128, 160, 16, 14),
    CursorPreset("midnight_pointer", "Midnight", 128, 160, 14, 14),
    CursorPreset("arrow", "Standard Arrow", 28, 38, 4, 4),
    CursorPreset("hand", "Hand Pointer", 32, 38, 10, 2),
    CursorPreset("ibeam", "Text I-Beam", 32, 38, 8, 19),
    CursorPreset("crosshair", "Crosshair", 32, 32, 16, 16),
    CursorPreset("filled_arrow", "Filled Arrow", 128, 160, 12, 12, "filled_arrow.svg", 1),
    CursorPreset("rounded_arrow", "Rounded Arrow", 128, 160, 12, 12, "rounded_arrow.svg", 1),
    CursorPreset(
        "highlighted_pointer",
        "Highlighted Pointer",
        160,
        160,
        20,
        20,
        "highlighted_pointer.svg",
        1,
    ),
    # A custom asset is supplied by the project and never generated from this
    # preset. The zero hotspot is the stable default for uploaded PNGs.
    CursorPreset("custom", "Custom PNG", 96, 128, 0, 0),
)
CURSOR_PRESETS_BY_ID = {item.style_id: item for item in CURSOR_PRESETS}
DEFAULT_CURSOR_STYLE_ID = "arrow"


# Cursor artwork and click feedback form one visual system. Keeping this
# mapping beside the cursor registry gives the editor one deterministic source
# while preserving the existing click-effect preset IDs in project files.
CURSOR_CLICK_EFFECT_NAMES = {
    "tangerine_wedge": "High Contrast Yellow",
    "cobalt_arrow": "Neon Cyan",
    "orchid_pointer": "Subtle Purple",
    "coral_pointer": "Bold Red",
    "signal_pointer": "Bold Red",
    "aqua_pointer": "Neon Cyan",
    "periwinkle_pointer": "Subtle Purple",
    "prism_pointer": "Subtle Purple",
    "aurora_pointer": "Neon Cyan",
    "ember_head": "Bold Red",
    "sapphire_head": "Neon Cyan",
    "lilac_head": "Subtle Purple",
    "ruby_head": "Bold Red",
    "mint_head": "Soft Green",
    "violet_head": "Subtle Purple",
    "cyan_wedge": "Neon Cyan",
    "outline_soft": "Clean White",
    "graphite_stripe": "Minimal Gray",
    "lavender_soft": "Subtle Purple",
    "ink_pointer": "Clean White",
    "violet_pointer": "Subtle Purple",
    "sky_pointer": "Neon Cyan",
    "midnight_pointer": "Minimal Gray",
    "highlighted_pointer": "High Contrast Yellow",
    "rounded_arrow": "Subtle Purple",
}


def click_effect_name_for_cursor(style_id: str | None) -> str:
    """Return the complementary built-in ripple for one cursor style."""
    return CURSOR_CLICK_EFFECT_NAMES.get(
        str(style_id or DEFAULT_CURSOR_STYLE_ID),
        "Clean White",
    )


def get_cursor_preset(style_id: str | None) -> CursorPreset:
    return CURSOR_PRESETS_BY_ID.get(
        str(style_id or DEFAULT_CURSOR_STYLE_ID),
        CURSOR_PRESETS_BY_ID[DEFAULT_CURSOR_STYLE_ID],
    )


def cursor_svg_path(style_id: str | None) -> str:
    """Return the packaged SVG source for an artistic cursor, if present."""
    preset = get_cursor_preset(style_id)
    if not preset.asset_name:
        return ""
    return str(Path(__file__).with_name("cursors") / preset.asset_name)


def _cursor_cache_dir() -> Path:
    directory = Path(tempfile.gettempdir()) / "Zumly" / "cursors"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


class _ScaledDraw:
    """Small ImageDraw adapter for antialiased procedural cursor assets."""

    def __init__(self, image: Image.Image, scale: float) -> None:
        self._draw = ImageDraw.Draw(image)
        self._scale = max(float(scale), 1.0)

    def _point(self, point: tuple[float, float]) -> tuple[int, int]:
        return (
            int(round(float(point[0]) * self._scale)),
            int(round(float(point[1]) * self._scale)),
        )

    def _box(self, box) -> tuple[int, int, int, int]:
        values = list(box)
        return tuple(int(round(float(value) * self._scale)) for value in values)

    def polygon(self, points, **kwargs) -> None:
        self._draw.polygon([self._point(point) for point in points], **kwargs)

    def line(self, points, fill=None, width=1, joint=None) -> None:
        options = {"fill": fill, "width": max(1, int(round(width * self._scale)))}
        if joint is not None:
            options["joint"] = joint
        if len(points) == 4 and all(isinstance(value, (int, float)) for value in points):
            points = ((points[0], points[1]), (points[2], points[3]))
        self._draw.line([self._point(point) for point in points], **options)

    def ellipse(self, box, **kwargs) -> None:
        self._draw.ellipse(self._box(box), **kwargs)

    def rounded_rectangle(self, box, radius=0, **kwargs) -> None:
        self._draw.rounded_rectangle(
            self._box(box),
            radius=max(1, int(round(float(radius) * self._scale))),
            **kwargs,
        )


def _scaled_draw(image: Image.Image, logical_width: int) -> _ScaledDraw:
    return _ScaledDraw(image, image.width / max(float(logical_width), 1.0))


def _draw_arrow(image: Image.Image) -> None:
    draw = _scaled_draw(image, 28)
    points = [(0, 0), (0, 17), (4, 13), (7, 20), (9, 19), (6, 12), (12, 12)]
    points = [(x + 4, y + 4) for x, y in points]
    shadow = [(x + 2, y + 2) for x, y in points]
    draw.polygon(shadow, fill=(0, 0, 0, 120))
    draw.line(shadow + [shadow[0]], fill=(0, 0, 0, 160), width=3)
    draw.polygon(points, fill=(255, 255, 255, 255))
    draw.line(points + [points[0]], fill=(20, 20, 20, 255), width=2)


def _draw_hand(image: Image.Image) -> None:
    draw = _scaled_draw(image, 32)
    points = [(10, 2), (14, 2), (15, 16), (18, 11), (21, 11), (21, 19),
              (24, 17), (27, 19), (24, 29), (17, 35), (9, 31), (5, 22), (5, 18), (9, 18)]
    draw.polygon([(x + 2, y + 2) for x, y in points], fill=(0, 0, 0, 120))
    draw.polygon(points, fill=(255, 255, 255, 255), outline=(20, 20, 20, 255))


def _draw_ibeam(image: Image.Image) -> None:
    draw = _scaled_draw(image, 32)
    x = 16
    draw.line([(x, 3), (x, 35)], fill=(20, 20, 20, 255), width=5)
    draw.line([(x, 3), (x, 35)], fill=(255, 255, 255, 255), width=2)
    draw.line([(7, 4), (25, 4)], fill=(20, 20, 20, 255), width=5)
    draw.line([(7, 34), (25, 34)], fill=(20, 20, 20, 255), width=5)
    draw.line([(7, 4), (25, 4)], fill=(255, 255, 255, 255), width=2)
    draw.line([(7, 34), (25, 34)], fill=(255, 255, 255, 255), width=2)


def _draw_crosshair(image: Image.Image) -> None:
    draw = _scaled_draw(image, 32)
    cx = cy = 16
    draw.ellipse((5, 5, 27, 27), outline=(20, 20, 20, 255), width=4)
    draw.ellipse((5, 5, 27, 27), outline=(255, 255, 255, 255), width=2)
    draw.line((cx, 0, cx, 32), fill=(20, 20, 20, 255), width=3)
    draw.line((0, cy, 32, cy), fill=(20, 20, 20, 255), width=3)


def _draw_filled_arrow(image: Image.Image) -> None:
    draw = _scaled_draw(image, 128)
    points = [(12, 12), (12, 104), (38, 78), (62, 136), (82, 126), (56, 70), (104, 70)]
    shadow = [(x + 5, y + 6) for x, y in points]
    draw.polygon(shadow, fill=(0, 0, 0, 120))
    draw.line(shadow + [shadow[0]], fill=(0, 0, 0, 180), width=6, joint="curve")
    draw.polygon(points, fill=(255, 255, 255, 255))
    draw.line(points + [points[0]], fill=(24, 24, 28, 255), width=5, joint="curve")


def _draw_rounded_arrow(image: Image.Image) -> None:
    draw = _scaled_draw(image, 128)
    # A softened pointer silhouette built at 4x the legacy cursor resolution.
    points = [(12, 12), (12, 104), (39, 79), (61, 135), (82, 125), (56, 70), (104, 70)]
    shadow = [(x + 5, y + 6) for x, y in points]
    draw.polygon(shadow, fill=(0, 0, 0, 110))
    draw.polygon(points, fill=(139, 92, 246, 255))
    draw.line(points + [points[0]], fill=(247, 244, 255, 255), width=5, joint="curve")
    draw.rounded_rectangle((34, 66, 106, 77), radius=5, fill=(139, 92, 246, 255))


def _draw_highlighted_pointer(image: Image.Image) -> None:
    draw = _scaled_draw(image, 160)
    draw.ellipse((0, 0, 80, 80), fill=(250, 204, 21, 70), outline=(250, 204, 21, 180), width=3)
    points = [(20, 20), (20, 106), (44, 82), (65, 138), (84, 128), (60, 73), (112, 73)]
    shadow = [(x + 5, y + 6) for x, y in points]
    draw.polygon(shadow, fill=(0, 0, 0, 125))
    draw.polygon(points, fill=(255, 255, 255, 255))
    draw.line(points + [points[0]], fill=(25, 25, 25, 255), width=5, joint="curve")


def _presentation_pointer_points() -> list[tuple[int, int]]:
    """Return the common oversized pointer silhouette used by gallery styles."""
    return [(14, 14), (14, 112), (42, 84), (68, 142), (90, 130), (62, 76), (114, 76)]


def _draw_ink_pointer(image: Image.Image) -> None:
    draw = _scaled_draw(image, 128)
    points = _presentation_pointer_points()
    shadow = [(x + 6, y + 7) for x, y in points]
    draw.polygon(shadow, fill=(0, 0, 0, 90))
    draw.polygon(points, fill=(12, 12, 16, 255))
    draw.line(points + [points[0]], fill=(255, 255, 255, 220), width=4, joint="curve")
    draw.line(points + [points[0]], fill=(12, 12, 16, 255), width=2, joint="curve")


def _draw_violet_pointer(image: Image.Image) -> None:
    draw = _scaled_draw(image, 128)
    points = _presentation_pointer_points()
    shadow = [(x + 6, y + 7) for x, y in points]
    draw.polygon(shadow, fill=(0, 0, 0, 95))
    draw.polygon(points, fill=(126, 34, 206, 255))
    draw.line(points + [points[0]], fill=(244, 226, 255, 255), width=5, joint="curve")
    draw.line(points + [points[0]], fill=(88, 28, 150, 255), width=2, joint="curve")


def _draw_sky_pointer(image: Image.Image) -> None:
    draw = _scaled_draw(image, 128)
    # A folded, paper-like pointer inspired by the blue reference cursor.
    points = [(16, 14), (116, 62), (70, 78), (52, 138), (38, 80)]
    draw.polygon([(x + 5, y + 6) for x, y in points], fill=(0, 0, 0, 65))
    draw.polygon(points, fill=(71, 157, 226, 255))
    draw.polygon([(16, 14), (70, 78), (38, 80)], fill=(130, 199, 245, 255))
    draw.line(points + [points[0]], fill=(25, 92, 151, 230), width=3, joint="curve")


def _draw_midnight_pointer(image: Image.Image) -> None:
    draw = _scaled_draw(image, 128)
    points = [(14, 14), (28, 118), (54, 86), (84, 130), (104, 114), (70, 72), (118, 68)]
    draw.polygon([(x + 6, y + 7) for x, y in points], fill=(0, 0, 0, 100))
    draw.polygon(points, fill=(3, 7, 18, 255))
    draw.line(points + [points[0]], fill=(108, 137, 190, 230), width=4, joint="curve")


@lru_cache(maxsize=32)
def ensure_cursor_asset(style_id: str = DEFAULT_CURSOR_STYLE_ID) -> str:
    """Generate and cache a built-in cursor bitmap for both renderers."""
    preset = get_cursor_preset(style_id)
    if preset.style_id == "custom":
        # There is no standalone custom asset until the project supplies one.
        # Keep the renderer deterministic by using the standard arrow.
        return ensure_cursor_asset(DEFAULT_CURSOR_STYLE_ID)
    path = _cursor_cache_dir() / (
        f"v{_CURSOR_RENDER_VERSION}_{preset.style_id}.png"
    )
    if path.is_file():
        return str(path)

    asset_scale = max(int(preset.asset_scale), 1)
    image = Image.new(
        "RGBA",
        (preset.width * asset_scale, preset.height * asset_scale),
        (0, 0, 0, 0),
    )
    if preset.style_id == "hand":
        _draw_hand(image)
    elif preset.style_id == "ibeam":
        _draw_ibeam(image)
    elif preset.style_id == "crosshair":
        _draw_crosshair(image)
    elif preset.style_id == "filled_arrow":
        _draw_filled_arrow(image)
    elif preset.style_id == "rounded_arrow":
        _draw_rounded_arrow(image)
    elif preset.style_id == "highlighted_pointer":
        _draw_highlighted_pointer(image)
    elif preset.style_id == "ink_pointer":
        _draw_ink_pointer(image)
    elif preset.style_id == "violet_pointer":
        _draw_violet_pointer(image)
    elif preset.style_id == "sky_pointer":
        _draw_sky_pointer(image)
    elif preset.style_id == "midnight_pointer":
        _draw_midnight_pointer(image)
    else:
        _draw_arrow(image)
    image.save(path)
    return str(path)


def cursor_hotspot(
    style_id: str | None,
    override: tuple[float, float] | None = None,
) -> tuple[float, float]:
    if override is not None:
        return float(override[0]), float(override[1])
    preset = get_cursor_preset(style_id)
    return preset.hotspot_x, preset.hotspot_y


def cursor_asset_scale(style_id: str | None) -> float:
    """Return the raster scale used by a built-in cursor asset.

    Built-in cursors are retained at their supersampled source resolution so
    preview and FFmpeg can reduce them with a high-quality filter at the final
    display size.  Hotspot coordinates remain in logical preset pixels and
    must be converted with this scale before drawing.
    """
    return float(get_cursor_preset(style_id).asset_scale)
