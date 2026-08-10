"""Shared screen-change transition semantics and pause-boundary suggestions."""

from __future__ import annotations

import io
import hashlib
import colorsys
import logging
import math
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps, ImageStat

from .models import ScreenTransition
from . import tokens as T
from .utils import ffmpeg_exe, subprocess_kwargs

logger = logging.getLogger(__name__)


def transition_resume_source_ms(
    transition: ScreenTransition,
    source_duration_ms: float,
) -> float:
    """Return the first source frame owned by normal playback after a transition.

    Pause-boundary analysis may move the incoming endpoint past one or more
    stuffed/outgoing capture frames.  Those bridge frames belong to the
    transition and must not be replayed after it completes.
    """
    duration = max(0.0, float(source_duration_ms))
    boundary = max(0.0, min(float(transition.timestamp_ms), duration))
    incoming = transition.incoming_frame_ms
    if incoming is None:
        return boundary
    return max(boundary, min(float(incoming), duration))


@dataclass(frozen=True)
class TransitionLayerState:
    scale_x: float = 1.0
    scale_y: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    anchor_x: float = 0.5
    anchor_y: float = 0.5
    opacity: float = 1.0
    visible: bool = True

    @property
    def scale(self) -> float:
        """Compatibility alias for callers that only need uniform scale."""
        return self.scale_x


@dataclass(frozen=True)
class GraphicBarState:
    """One fully opaque styled bar in normalized Video Space."""

    index: int
    rect: tuple[float, float, float, float]
    color_start: str
    color_end: str
    style: str
    grain_seed: int
    texture_bands: tuple[float, ...]
    edge_shading: float

    @property
    def color(self) -> str:
        """Compatibility alias for flat-color callers."""
        return self.color_start


@dataclass(frozen=True)
class GraphicBarDescription:
    index: int
    start_rect: tuple[float, float, float, float]
    covered_rect: tuple[float, float, float, float]
    end_rect: tuple[float, float, float, float]
    delay: float
    color_start: str
    color_end: str
    style: str
    grain_seed: int
    texture_bands: tuple[float, ...]
    edge_shading: float

    @property
    def color(self) -> str:
        """Compatibility alias for flat-color callers."""
        return self.color_start


@dataclass(frozen=True)
class GraphicTransitionDescription:
    bars: tuple[GraphicBarDescription, ...]
    switch_progress: float = 0.5


@dataclass(frozen=True)
class ScreenTransitionState:
    progress: float
    outgoing: TransitionLayerState
    incoming: TransitionLayerState
    layer_order: tuple[str, str] = ("incoming", "outgoing")
    clip_rect: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    easing: str = "quintic_in_out"
    graphic_bars: tuple[GraphicBarState, ...] = ()
    incoming_owns_aperture: bool = False

    @property
    def is_graphic(self) -> bool:
        return bool(self.graphic_bars)


@dataclass(frozen=True)
class TransitionSceneDescription:
    """Endpoint states shared by the preview and FFmpeg renderers."""

    effect_type: str
    outgoing_start: TransitionLayerState
    outgoing_end: TransitionLayerState
    incoming_start: TransitionLayerState
    incoming_end: TransitionLayerState
    layer_order: tuple[str, str] = ("incoming", "outgoing")
    clip_rect: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    easing: str = "quintic_in_out"


@dataclass(frozen=True)
class FFmpegTransitionLayerExpressions:
    scale_x: str
    scale_y: str
    offset_x: str
    offset_y: str
    anchor_x: str
    anchor_y: str
    opacity: str
    visible: str

    @property
    def scale(self) -> str:
        return self.scale_x


@dataclass(frozen=True)
class FFmpegTransitionSceneExpressions:
    progress: str
    outgoing: FFmpegTransitionLayerExpressions
    incoming: FFmpegTransitionLayerExpressions
    layer_order: tuple[str, str]
    clip_rect: tuple[float, float, float, float]


@dataclass(frozen=True)
class FFmpegGraphicBarExpressions:
    x: str
    y: str
    width: str
    height: str
    color_start: str
    color_end: str
    style: str
    grain_seed: int
    texture_bands: tuple[float, ...]
    edge_shading: float
    enable: str

    @property
    def color(self) -> str:
        return self.color_start


@dataclass(frozen=True)
class FFmpegGraphicTransitionExpressions:
    progress: str
    incoming_owns_aperture: str
    bars: tuple[FFmpegGraphicBarExpressions, ...]


GRAPHIC_TRANSITION_COLORS = {
    "zumly_editorial": (
        "#10B7C4", "#078FA4", "#6F35E5", "#8B5CF6",
        "#111936", "#F06467", "#E9A23B", "#C7D936",
    ),
    "modern_editorial": (
        "#C33BB6", "#087D9C", "#E58C35", "#E86872",
        "#03495A", "#171717", "#C8DA32",
    ),
    "cool_spectrum": (
        "#16B8C4", "#0D9488", "#3154D8", "#5046C8",
        "#7C3AED", "#C43C91", "#121725",
    ),
    "dark_premium": (
        "#0B1020", "#123C49", "#066A73", "#4338CA",
        "#702DBD", "#9B3158", "#C89432",
    ),
    "warm_creative": (
        "#D9654F", "#E8677A", "#E9933A", "#C8A936",
        "#AABF3B", "#075B64", "#202124",
    ),
}

def is_graphic_transition(effect_type: str) -> bool:
    return str(effect_type or "").startswith("graphic_")


def _ease_in_out_quint(value: float) -> float:
    t = max(0.0, min(float(value), 1.0))
    return (6.0 * t**5) - (15.0 * t**4) + (10.0 * t**3)


def _ease_value(value: float, easing: str) -> float:
    value = max(0.0, min(float(value), 1.0))
    if easing == "linear":
        return value
    if easing == "ease_out":
        return 1.0 - ((1.0 - value) ** 3)
    return _ease_in_out_quint(value)


def _rect_shifted(
    rect: tuple[float, float, float, float],
    direction: str,
) -> tuple[float, float, float, float]:
    x, y, width, height = rect
    if direction == "left":
        return (-width, y, width, height)
    if direction == "right":
        return (1.0, y, width, height)
    if direction == "up":
        return (x, -height, width, height)
    return (x, 1.0, width, height)


def _opposite_direction(direction: str) -> str:
    return {
        "left": "right",
        "right": "left",
        "up": "down",
        "down": "up",
    }.get(direction, "right")


def _graphic_orientation(transition: ScreenTransition) -> str:
    if transition.bar_orientation != "auto":
        return transition.bar_orientation
    effect = transition.effect_type
    if effect in {"graphic_horizontal_bars"}:
        return "horizontal"
    if effect == "graphic_diagonal_bars":
        return "diagonal"
    if effect in {"graphic_sweep"} and transition.direction in {"up", "down"}:
        return "horizontal"
    return "vertical"


def _color_distance(left: str, right: str) -> float:
    left_rgb = _hex_rgb(left)
    right_rgb = _hex_rgb(right)
    red = (left_rgb[0] - right_rgb[0]) * 0.30
    green = (left_rgb[1] - right_rgb[1]) * 0.59
    blue = (left_rgb[2] - right_rgb[2]) * 0.41
    luminance = (
        (left_rgb[0] * 0.2126 + left_rgb[1] * 0.7152 + left_rgb[2] * 0.0722)
        - (right_rgb[0] * 0.2126 + right_rgb[1] * 0.7152 + right_rgb[2] * 0.0722)
    )
    return (
        (red * red)
        + (green * green)
        + (blue * blue)
        + (luminance * luminance * 0.18)
    )


def _color_luminance(color: str) -> float:
    red, green, blue = _hex_rgb(color)
    return (red * 0.2126) + (green * 0.7152) + (blue * 0.0722)


def _curated_color_sequence(
    palette: tuple[str, ...],
    count: int,
    seed: int,
) -> tuple[str, ...]:
    """Order a palette with rich dark edge anchors and contrasting interiors."""
    unique = tuple(dict.fromkeys(color.upper() for color in palette))
    target_count = max(1, int(count))
    if not unique:
        unique = GRAPHIC_TRANSITION_COLORS["zumly_editorial"]
    if len(unique) == 1:
        return (unique[0],) * target_count

    # The first and last bars frame the composition. Keeping the two darkest
    # available palette members there makes light accents feel intentional
    # instead of leaving a pale, low-weight edge.
    dark_order = sorted(
        range(len(unique)),
        key=lambda index: (
            _color_luminance(unique[index]),
            _seed_step(seed + index) / 0xFFFFFFFF,
        ),
    )
    left_index = dark_order[0]
    right_index = dark_order[1] if len(dark_order) > 1 else left_index
    if target_count == 1:
        return (unique[left_index],)

    sequence = [unique[left_index]]
    available = [
        index
        for index in range(len(unique))
        if index not in {left_index, right_index}
    ]
    previous_index = left_index

    for position in range(1, target_count - 1):
        if not available:
            available = [
                index
                for index in range(len(unique))
                if index not in {previous_index, right_index}
            ]
        selected = max(
            available,
            key=lambda index: (
                _color_distance(unique[previous_index], unique[index]),
                _seed_step(seed + (position * 131) + index) / 0xFFFFFFFF,
            ),
        )
        sequence.append(unique[selected])
        available.remove(selected)
        previous_index = selected

    sequence.append(unique[right_index])
    return tuple(sequence)


def _premium_tone(color: str, darken: float, saturation_gain: float = 0.06) -> str:
    """Deepen a palette color while preserving its hue and material identity."""
    red, green, blue = _hex_rgb(color)
    hue, saturation, value = colorsys.rgb_to_hsv(
        red / 255.0,
        green / 255.0,
        blue / 255.0,
    )
    saturation = min(1.0, saturation + max(0.0, float(saturation_gain)))
    value = max(0.0, min(1.0, value * (1.0 - max(0.0, float(darken)))))
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return "#" + "".join(
        f"{round(channel * 255.0):02X}" for channel in (red, green, blue)
    )


def _graphic_colors(
    transition: ScreenTransition,
    count: int,
    seed: int,
) -> tuple[str, ...]:
    if transition.color_preset == "custom":
        return (transition.custom_color,) * max(1, int(count))
    palette = GRAPHIC_TRANSITION_COLORS.get(
        transition.color_preset,
        GRAPHIC_TRANSITION_COLORS["zumly_editorial"],
    )
    return _curated_color_sequence(palette, count, seed)

def _stable_graphic_seed(transition: ScreenTransition) -> int:
    explicit = max(0, int(getattr(transition, "bar_seed", 0) or 0))
    if explicit:
        return explicit & 0xFFFFFFFF
    # Palette selection is intentionally excluded: changing only the Color Set
    # must never reshuffle bar geometry or material assignment.
    identity = f"{transition.id}:{transition.effect_type}:{transition.bar_count}"
    return int.from_bytes(
        hashlib.sha256(identity.encode("utf-8")).digest()[:4], "big"
    )


def _seed_step(seed: int) -> int:
    return ((1664525 * int(seed)) + 1013904223) & 0xFFFFFFFF


def _bar_boundaries(
    count: int,
    width_mode: str,
    seed: int,
) -> tuple[float, ...]:
    """Return cumulative shared boundaries whose final value is exactly 1."""
    count = max(2, int(count))
    mode = str(width_mode or "varied")
    if mode == "uniform":
        weights = [1.0] * count
    elif mode == "seeded":
        state = seed
        weights = []
        for _ in range(count):
            state = _seed_step(state)
            weights.append(0.68 + (state / 0xFFFFFFFF) * 0.72)
    else:
        pattern = (0.72, 1.18, 0.88, 1.34, 0.78, 1.10, 0.94)
        offset = seed % len(pattern)
        weights = [pattern[(offset + index) % len(pattern)] for index in range(count)]

    total = sum(weights)
    boundaries = [0.0]
    running = 0.0
    for weight in weights[:-1]:
        running += weight / total
        boundaries.append(running)
    boundaries.append(1.0)
    return tuple(boundaries)


def _hex_rgb(color: str) -> tuple[int, int, int]:
    normalized = str(color or T.BRAND_PURPLE).lstrip("#")
    if len(normalized) != 6:
        normalized = T.BRAND_PURPLE.lstrip("#")
    return tuple(int(normalized[index : index + 2], 16) for index in (0, 2, 4))


def _mix_hex(color: str, target: str, amount: float) -> str:
    amount = max(0.0, min(float(amount), 1.0))
    left = _hex_rgb(color)
    right = _hex_rgb(target)
    return "#" + "".join(
        f"{round(a + ((b - a) * amount)):02X}" for a, b in zip(left, right)
    )


def _styled_bar_colors(
    base_color: str,
    style: str,
    seed: int,
) -> tuple[str, str, tuple[float, ...], float]:
    """Return deterministic opaque gradient and subtle texture parameters."""
    state = _seed_step(seed)
    depth = 0.05 + ((state / 0xFFFFFFFF) * 0.04)
    toned = _premium_tone(base_color, depth, saturation_gain=0.06)
    state = _seed_step(state)
    gradient_amount = 0.05 + ((state / 0xFFFFFFFF) * 0.05)
    end = _mix_hex(toned, T.BRAND_NAVY, gradient_amount)

    # Fine grain is generated from this seed by each renderer. Keeping it
    # two-dimensional avoids the enlarged scanline look of directional bands.
    return toned, end, (), 0.16


_MATERIAL_FILES = {
    "material_vinyl": "matte-vinyl.png",
    "material_paint": "brushed-paint.png",
    "material_leather": "pseudo-leather.png",
    "material_cloth": "heavy-cloth.png",
}
_MATERIAL_MIX = tuple(_MATERIAL_FILES.values())


@dataclass(frozen=True)
class MaterialDepthProfile:
    source_zoom: float
    height_contrast: float
    normal_radius: float
    normal_strength: float
    ao_radius: float
    ao_strength: float
    highlight_strength: float
    shadow_strength: float
    crest_strength: float


_MATERIAL_DEPTH_PROFILES = {
    "matte-vinyl.png": MaterialDepthProfile(
        0.76, 1.07, 1.20, 0.92, 5.0, 0.82, 0.42, 0.36, 0.20
    ),
    "brushed-paint.png": MaterialDepthProfile(
        0.92, 1.08, 1.35, 0.98, 6.0, 0.86, 0.46, 0.40, 0.22
    ),
    "pseudo-leather.png": MaterialDepthProfile(
        0.86, 1.10, 1.05, 1.00, 5.5, 0.92, 0.48, 0.42, 0.24
    ),
    "heavy-cloth.png": MaterialDepthProfile(
        0.88, 1.08, 0.95, 0.98, 5.0, 0.96, 0.50, 0.44, 0.26
    ),
}


def _material_relief_layers(
    relief: Image.Image,
    profile: MaterialDepthProfile,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    """Derive height, ridge highlights, and recessed occlusion from artwork."""
    height = ImageOps.autocontrast(relief, cutoff=(1.0, 1.0))
    height = ImageEnhance.Contrast(height).enhance(profile.height_contrast)

    normal_source = height.filter(
        ImageFilter.GaussianBlur(radius=profile.normal_radius)
    )
    horizontal = ImageChops.subtract(
        ImageChops.offset(normal_source, -1, 0),
        ImageChops.offset(normal_source, 1, 0),
        scale=1.0,
        offset=128,
    )
    vertical = ImageChops.subtract(
        ImageChops.offset(normal_source, 0, -1),
        ImageChops.offset(normal_source, 0, 1),
        scale=1.0,
        offset=128,
    )
    directional = ImageChops.add(horizontal, vertical, scale=2.0, offset=0)
    directional = ImageEnhance.Contrast(directional).enhance(
        profile.normal_strength
    )

    highlight = directional.point(
        lambda value: max(
            0,
            min(
                255,
                round((value - 128) * 2.0 * profile.highlight_strength),
            ),
        )
    )
    directional_shadow = directional.point(
        lambda value: max(
            0,
            min(
                255,
                round((128 - value) * 2.0 * profile.shadow_strength),
            ),
        )
    )

    neighborhood = height.filter(ImageFilter.GaussianBlur(radius=profile.ao_radius))
    recessed = ImageChops.subtract(neighborhood, height, scale=1.0, offset=0)
    occlusion = recessed.point(
        lambda value: min(255, round(value * profile.ao_strength))
    )
    shadow = ImageChops.lighter(directional_shadow, occlusion)

    raised = ImageChops.subtract(height, neighborhood, scale=1.0, offset=0)
    crest = raised.point(
        lambda value: min(255, round(value * profile.crest_strength * 3.0))
    )
    highlight = ImageChops.lighter(highlight, crest)
    return height, highlight, shadow

def _material_asset_path(filename: str) -> Path:
    """Resolve one generated material scan in source and PyInstaller layouts."""
    if getattr(sys, "frozen", False):
        bundle_root = Path(
            getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent)
        )
        bundled = bundle_root / "zumly" / "app" / "transition_materials" / filename
        if bundled.is_file():
            return bundled
    return Path(__file__).resolve().parent / "transition_materials" / filename


def _material_filename(style: str, seed: int) -> str | None:
    if style == "material_mix":
        return _MATERIAL_MIX[int(seed) % len(_MATERIAL_MIX)]
    return _MATERIAL_FILES.get(style)


@lru_cache(maxsize=96)
def graphic_bar_material_png(
    color_start: str,
    color_end: str,
    style: str,
    seed: int,
    edge_shading: float,
    size: int = 384,
    *,
    width: int | None = None,
    height: int | None = None,
) -> bytes:
    """Render an opaque aspect-correct bar surface for preview and export.

    Real material artwork is mirrored into a seamless tile and repeated at a
    fixed texel scale. Horizontal bars therefore retain the same crisp grain as
    vertical bars instead of stretching one square bitmap across the surface.
    """
    tile_size = max(96, min(int(size), 768))
    output_width = max(2, min(int(width or tile_size), 4096))
    output_height = max(2, min(int(height or tile_size), 4096))
    start = _hex_rgb(color_start)
    end = _hex_rgb(color_end)
    filename = _material_filename(style, seed)

    if filename:
        asset_path = _material_asset_path(filename)
        if not asset_path.is_file():
            raise FileNotFoundError(f"Transition material asset is missing: {asset_path}")
        half = max(48, tile_size // 2)
        profile = _MATERIAL_DEPTH_PROFILES[filename]
        with Image.open(asset_path) as source:
            source_height = source.convert("L")
            zoom = max(0.25, min(float(profile.source_zoom), 1.0))
            if zoom < 1.0:
                crop_width = max(2, round(source_height.width * zoom))
                crop_height = max(2, round(source_height.height * zoom))
                crop_left = (source_height.width - crop_width) // 2
                crop_top = (source_height.height - crop_height) // 2
                source_height = source_height.crop(
                    (
                        crop_left,
                        crop_top,
                        crop_left + crop_width,
                        crop_top + crop_height,
                    )
                )
            relief = ImageOps.fit(
                source_height,
                (half, half),
                method=Image.Resampling.LANCZOS,
            )
        # Mirroring makes all outer tile boundaries meet without a hard seam.
        relief_tile = Image.new("L", (half * 2, half * 2))
        relief_tile.paste(relief, (0, 0))
        relief_tile.paste(ImageOps.mirror(relief), (half, 0))
        relief_tile.paste(ImageOps.flip(relief), (0, half))
        relief_tile.paste(ImageOps.flip(ImageOps.mirror(relief)), (half, half))

        height, highlight_mask, shadow_mask = _material_relief_layers(
            relief_tile,
            profile,
        )
        mid = tuple(
            round((left * 0.72) + (right * 0.28))
            for left, right in zip(start, end)
        )
        # Keep the selected hue, but give the material a real tonal range:
        # deep recesses, a confident body tone, and controlled soft highlights.
        base_shadow = tuple(max(0, round(channel * 0.80)) for channel in mid)
        base_highlight = tuple(
            min(255, round(channel + ((255 - channel) * 0.12)))
            for channel in mid
        )
        colored_tile = ImageOps.colorize(
            height,
            black=base_shadow,
            mid=mid,
            white=base_highlight,
        ).convert("RGB")

        ridge_color = tuple(
            min(255, round(channel + ((255 - channel) * 0.18)))
            for channel in mid
        )
        recess_color = tuple(max(0, round(channel * 0.64)) for channel in mid)
        # Use the source relief as a restrained internal grade. This keeps the
        # surface matte and tactile without turning the bar into glossy 3D art.
        highlight_mask = highlight_mask.point(
            lambda value: round(value * 0.52)
        )
        shadow_mask = shadow_mask.point(
            lambda value: round(value * 0.44)
        )
        colored_tile = Image.composite(
            Image.new("RGB", colored_tile.size, ridge_color),
            colored_tile,
            highlight_mask,
        )
        colored_tile = Image.composite(
            Image.new("RGB", colored_tile.size, recess_color),
            colored_tile,
            shadow_mask,
        )
        image = Image.new("RGB", (output_width, output_height))
        for top in range(0, output_height, colored_tile.height):
            for left in range(0, output_width, colored_tile.width):
                image.paste(colored_tile, (left, top))

        # A restrained surface-wide color drift adds depth without blurring the
        # underlying vinyl, paint, leather, or cloth relief.
        if start != end:
            gradient = Image.new("RGB", (output_width, 1))
            denominator = max(output_width - 1, 1)
            gradient.putdata(
                [
                    tuple(
                        round(left + ((right - left) * (x / denominator)))
                        for left, right in zip(start, end)
                    )
                    for x in range(output_width)
                ]
            )
            gradient = gradient.resize((output_width, output_height))
            image = Image.blend(image, gradient, 0.06)
    else:
        pixels: list[tuple[int, int, int]] = []
        width_denominator = max(output_width - 1, 1)
        height_denominator = max(output_height - 1, 1)
        for y in range(output_height):
            ny = y / height_denominator
            for x in range(output_width):
                nx = x / width_denominator
                ramp = max(0.0, min((0.68 * nx) + (0.32 * ny), 1.0))
                rgb = [
                    left + ((right - left) * ramp)
                    for left, right in zip(start, end)
                ]
                pixels.append(
                    tuple(max(0, min(round(channel), 255)) for channel in rgb)
                )
        image = Image.new("RGB", (output_width, output_height))
        image.putdata(pixels)

    if edge_shading > 0.0 and output_width > 8:
        # Apply a color-preserving falloff at both outer edges. It deepens the
        # panel without introducing a black stripe or changing its geometry.
        edge_width = max(2, round(output_width * 0.10))
        edge_limit = max(edge_width - 1, 1)
        strength = min(max(float(edge_shading), 0.0), 0.35)
        mask = Image.new("L", (output_width, 1))
        mask.putdata(
            [
                round(
                    255
                    * strength
                    * (
                        max(
                            0.0,
                            max(
                                1.0 - (x / edge_limit),
                                1.0 - ((output_width - 1 - x) / edge_limit),
                            ),
                        )
                        ** 1.6
                    )
                )
                for x in range(output_width)
            ]
        )
        mask = mask.resize((output_width, output_height))
        darkened = ImageEnhance.Brightness(image).enhance(0.78)
        image = Image.composite(darkened, image, mask)

    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()

def graphic_transition_description(
    transition: ScreenTransition,
) -> GraphicTransitionDescription:
    """Describe opaque bars with shared, seam-free normalized boundaries."""
    count = max(2, min(int(transition.bar_count), 20))
    orientation = _graphic_orientation(transition)
    horizontal_tracks = orientation == "horizontal"
    seed = _stable_graphic_seed(transition)
    boundaries = _bar_boundaries(
        count,
        getattr(transition, "bar_width_mode", "varied"),
        seed,
    )
    colors = _graphic_colors(transition, count, seed)
    style = str(getattr(transition, "bar_style", "material_mix") or "material_mix")
    effect = transition.effect_type
    direction = transition.direction
    if effect in {"graphic_vertical_bars", "graphic_diagonal_bars"}:
        direction = direction if direction in {"up", "down"} else "down"
    elif effect == "graphic_horizontal_bars":
        direction = direction if direction in {"left", "right"} else "right"
    elif effect == "graphic_fold":
        direction = direction if direction in {"left", "right"} else "left"
    exit_direction = (
        direction
        if transition.enter_exit_mode == "same"
        else _opposite_direction(direction)
    )

    bars: list[GraphicBarDescription] = []
    for index in range(count):
        edge_start = boundaries[index]
        edge_end = boundaries[index + 1]
        extent = edge_end - edge_start
        covered = (
            (0.0, edge_start, 1.0, extent)
            if horizontal_tracks
            else (edge_start, 0.0, extent, 1.0)
        )
        moving = covered

        order = index / max(count - 1, 1)
        if direction in {"right", "down"}:
            order = 1.0 - order
        if effect == "graphic_diagonal_bars":
            order = index / max(count - 1, 1)
            if transition.direction in {"left", "up"}:
                order = 1.0 - order

        if effect == "graphic_split_in":
            start_direction = "left" if index < (count / 2.0) else "right"
            start = _rect_shifted(moving, start_direction)
            end = _rect_shifted(moving, _opposite_direction(start_direction))
            order = abs(((index + 0.5) / count) - 0.5) * 2.0
        elif effect == "graphic_split_out":
            x, y, width, height = moving
            start = (
                (x, 0.5, width, 0.0)
                if horizontal_tracks
                else (0.5, y, 0.0, height)
            )
            end = _rect_shifted(
                moving, "left" if index < (count / 2.0) else "right"
            )
            order = 1.0 - (abs(((index + 0.5) / count) - 0.5) * 2.0)
        elif effect == "graphic_fold":
            x, y, width, height = covered
            if direction == "right":
                start = (x + width, y, 0.0, height)
            else:
                start = (x, y, 0.0, height)
            if exit_direction == "right":
                end = (x + width, y, 0.0, height)
            else:
                end = (x, y, 0.0, height)
        else:
            start = _rect_shifted(moving, direction)
            end = _rect_shifted(moving, exit_direction)

        bar_seed = _seed_step(seed + index) & 0x7FFFFFFF
        color_start, color_end, texture_bands, edge_shading = _styled_bar_colors(
            colors[index], style, bar_seed
        )
        bars.append(
            GraphicBarDescription(
                index=index,
                start_rect=start,
                covered_rect=covered,
                end_rect=end,
                delay=max(0.0, min(float(transition.bar_stagger) * order, 0.8)),
                color_start=color_start,
                color_end=color_end,
                style=style,
                grain_seed=bar_seed,
                texture_bands=texture_bands,
                edge_shading=edge_shading,
            )
        )
    return GraphicTransitionDescription(tuple(bars))

def _interpolate_rect(
    start: tuple[float, float, float, float],
    end: tuple[float, float, float, float],
    progress: float,
) -> tuple[float, float, float, float]:
    return tuple(
        left + ((right - left) * progress)
        for left, right in zip(start, end)
    )


def _bar_phase_progress(phase: float, delay: float, easing: str) -> float:
    available = max(1.0 - delay, 0.001)
    return _ease_value((phase - delay) / available, easing)


def evaluate_graphic_transition(
    transition: ScreenTransition,
    elapsed_ms: float,
) -> ScreenTransitionState:
    duration = max(float(transition.duration_ms), 1.0)
    progress = max(0.0, min(float(elapsed_ms) / duration, 1.0))
    description = graphic_transition_description(transition)
    entering = progress <= description.switch_progress
    phase = (
        progress / description.switch_progress
        if entering
        else (progress - description.switch_progress)
        / max(1.0 - description.switch_progress, 0.001)
    )
    bars: list[GraphicBarState] = []
    for bar in description.bars:
        local_progress = _bar_phase_progress(phase, bar.delay, transition.easing)
        rect = _interpolate_rect(
            bar.start_rect if entering else bar.covered_rect,
            bar.covered_rect if entering else bar.end_rect,
            local_progress,
        )
        bars.append(
            GraphicBarState(
                bar.index,
                rect,
                bar.color_start,
                bar.color_end,
                bar.style,
                bar.grain_seed,
                bar.texture_bands,
                bar.edge_shading,
            )
        )
    incoming_owns = progress >= description.switch_progress
    return ScreenTransitionState(
        progress=progress,
        outgoing=TransitionLayerState(visible=not incoming_owns),
        incoming=TransitionLayerState(visible=incoming_owns),
        graphic_bars=tuple(bars),
        incoming_owns_aperture=incoming_owns,
        easing=transition.easing,
    )


def transition_scene_description(
    effect_type: str, direction: str = "left"
) -> TransitionSceneDescription:
    """Return the canonical layer endpoints for a transition preset."""
    legacy = {
        "smooth_settle": "scale_swap",
        "blur_dissolve": "zoom_through",
        "dip_to_canvas": "scale_swap",
    }
    effect = legacy.get(str(effect_type or "scale_swap"), str(effect_type or "scale_swap"))
    direction = str(direction or "left").lower()
    if direction not in {"left", "right", "up", "down"}:
        direction = "left"
    if effect == "directional_push":
        old_x = -1.0 if direction == "left" else 1.0 if direction == "right" else 0.0
        old_y = -1.0 if direction == "up" else 1.0 if direction == "down" else 0.0
        return TransitionSceneDescription(
            effect,
            TransitionLayerState(),
            TransitionLayerState(
                offset_x=old_x, offset_y=old_y, visible=False
            ),
            TransitionLayerState(offset_x=-old_x, offset_y=-old_y),
            TransitionLayerState(),
        )
    if effect == "axis_flip":
        horizontal = direction in {"left", "right"}
        anchor_x = 0.0 if direction == "left" else 1.0 if direction == "right" else 0.5
        anchor_y = 0.0 if direction == "up" else 1.0 if direction == "down" else 0.5
        return TransitionSceneDescription(
            effect,
            TransitionLayerState(anchor_x=anchor_x, anchor_y=anchor_y),
            TransitionLayerState(
                scale_x=0.02 if horizontal else 1.0,
                scale_y=1.0 if horizontal else 0.02,
                anchor_x=anchor_x,
                anchor_y=anchor_y,
                visible=False,
            ),
            TransitionLayerState(scale_x=1.04, scale_y=1.04),
            TransitionLayerState(),
        )
    if effect == "zoom_through":
        return TransitionSceneDescription(
            effect,
            TransitionLayerState(),
            TransitionLayerState(
                scale_x=1.35,
                scale_y=1.35,
                visible=False,
            ),
            TransitionLayerState(scale_x=0.02, scale_y=0.02),
            TransitionLayerState(),
            layer_order=("outgoing", "incoming"),
        )
    return TransitionSceneDescription(
        "scale_swap",
        TransitionLayerState(),
        TransitionLayerState(scale_x=0.02, scale_y=0.02, visible=False),
        TransitionLayerState(scale_x=1.08, scale_y=1.08),
        TransitionLayerState(),
    )


def _interpolate_layer(
    start: TransitionLayerState,
    end: TransitionLayerState,
    progress: float,
) -> TransitionLayerState:
    def lerp(left: float, right: float) -> float:
        return left + ((right - left) * progress)

    return TransitionLayerState(
        scale_x=lerp(start.scale_x, end.scale_x),
        scale_y=lerp(start.scale_y, end.scale_y),
        offset_x=lerp(start.offset_x, end.offset_x),
        offset_y=lerp(start.offset_y, end.offset_y),
        anchor_x=lerp(start.anchor_x, end.anchor_x),
        anchor_y=lerp(start.anchor_y, end.anchor_y),
        opacity=lerp(start.opacity, end.opacity),
        visible=start.visible if progress < 1.0 else end.visible,
    )


def evaluate_screen_transition(
    transition: ScreenTransition,
    elapsed_ms: float,
) -> ScreenTransitionState:
    """Return preview/export-neutral layer state for an inserted transition."""
    if is_graphic_transition(transition.effect_type):
        return evaluate_graphic_transition(transition, elapsed_ms)
    duration = max(float(transition.duration_ms), 1.0)
    progress = _ease_in_out_quint(float(elapsed_ms) / duration)
    description = transition_scene_description(
        transition.effect_type, transition.direction
    )
    return ScreenTransitionState(
        progress,
        _interpolate_layer(
            description.outgoing_start, description.outgoing_end, progress
        ),
        _interpolate_layer(
            description.incoming_start, description.incoming_end, progress
        ),
        layer_order=description.layer_order,
        clip_rect=description.clip_rect,
        easing=description.easing,
    )


def _ffmpeg_lerp(start: float, end: float, progress: str) -> str:
    if abs(float(start) - float(end)) <= 1e-12:
        return f"{float(start):.6f}"
    return f"({float(start):.6f}+({float(end - start):.6f})*{progress})"


def ffmpeg_transition_scene_expressions(
    effect_type: str,
    duration_sec: float,
    *,
    direction: str = "left",
    variable: str = "T",
) -> FFmpegTransitionSceneExpressions:
    """Translate the canonical scene description into FFmpeg expressions."""
    description = transition_scene_description(effect_type, direction)
    progress = ffmpeg_quintic_time_expression(duration_sec, variable=variable)

    def layer(
        start: TransitionLayerState,
        end: TransitionLayerState,
    ) -> FFmpegTransitionLayerExpressions:
        return FFmpegTransitionLayerExpressions(
            scale_x=_ffmpeg_lerp(start.scale_x, end.scale_x, progress),
            scale_y=_ffmpeg_lerp(start.scale_y, end.scale_y, progress),
            offset_x=_ffmpeg_lerp(start.offset_x, end.offset_x, progress),
            offset_y=_ffmpeg_lerp(start.offset_y, end.offset_y, progress),
            anchor_x=_ffmpeg_lerp(start.anchor_x, end.anchor_x, progress),
            anchor_y=_ffmpeg_lerp(start.anchor_y, end.anchor_y, progress),
            opacity=_ffmpeg_lerp(start.opacity, end.opacity, progress),
            visible=(
                "1"
                if start.visible and end.visible
                else "0"
                if not start.visible and not end.visible
                else f"lt({progress},1)"
                if start.visible
                else f"gte({progress},1)"
            ),
        )

    outgoing = layer(description.outgoing_start, description.outgoing_end)
    incoming = layer(description.incoming_start, description.incoming_end)
    return FFmpegTransitionSceneExpressions(
        progress=progress,
        outgoing=outgoing,
        incoming=incoming,
        layer_order=description.layer_order,
        clip_rect=description.clip_rect,
    )


def _ffmpeg_ease_expression(value: str, easing: str) -> str:
    if easing == "linear":
        return value
    if easing == "ease_out":
        return f"(1-pow(1-({value}),3))"
    return f"(6*pow({value},5)-15*pow({value},4)+10*pow({value},3))"


def ffmpeg_graphic_transition_expressions(
    transition: ScreenTransition,
    duration_sec: float,
    *,
    variable: str = "t",
) -> FFmpegGraphicTransitionExpressions:
    """Translate the same normalized bar descriptions used by Qt to FFmpeg."""
    duration = max(float(duration_sec), 0.001)
    progress = f"clip({variable}/{duration:.6f},0,1)"
    description = graphic_transition_description(transition)
    bars: list[FFmpegGraphicBarExpressions] = []
    for bar in description.bars:
        available = max(1.0 - bar.delay, 0.001)
        enter_raw = f"clip(((2*({progress}))-{bar.delay:.6f})/{available:.6f},0,1)"
        exit_raw = (
            f"clip(((2*({progress})-1)-{bar.delay:.6f})/{available:.6f},0,1)"
        )
        enter = _ffmpeg_ease_expression(enter_raw, transition.easing)
        exit_value = _ffmpeg_ease_expression(exit_raw, transition.easing)

        def component(index: int) -> str:
            start = bar.start_rect[index]
            covered = bar.covered_rect[index]
            end = bar.end_rect[index]
            entering = _ffmpeg_lerp(start, covered, enter)
            exiting = _ffmpeg_lerp(covered, end, exit_value)
            return f"if(lt({progress},0.5),{entering},{exiting})"

        start_time = (duration * 0.5) * bar.delay
        end_time = duration - ((duration * 0.5) * bar.delay)
        bars.append(
            FFmpegGraphicBarExpressions(
                x=component(0),
                y=component(1),
                width=component(2),
                height=component(3),
                color_start=bar.color_start,
                color_end=bar.color_end,
                style=bar.style,
                grain_seed=bar.grain_seed,
                texture_bands=bar.texture_bands,
                edge_shading=bar.edge_shading,
                enable=f"between({variable},{start_time:.6f},{end_time:.6f})",
            )
        )
    return FFmpegGraphicTransitionExpressions(
        progress=progress,
        incoming_owns_aperture=f"gte({progress},0.5)",
        bars=tuple(bars),
    )


def ffmpeg_quintic_time_expression(
    duration_sec: float,
    variable: str = "t",
) -> str:
    """Return forward 0..1 quintic progress for a filter time variable."""
    duration = max(float(duration_sec), 0.001)
    t = f"clip({variable}/{duration:.6f},0,1)"
    return f"(6*pow({t},5)-15*pow({t},4)+10*pow({t},3))"


def _extract_probe_frame(video_path: str, timestamp_ms: float) -> Image.Image:
    command = [
        ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
        "-ss", f"{max(float(timestamp_ms), 0.0) / 1000.0:.6f}",
        "-i", video_path, "-frames:v", "1", "-vf", "scale=96:54",
        "-c:v", "png", "-f", "image2pipe", "pipe:1",
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=8,
        check=False,
        **subprocess_kwargs(),
    )
    if result.returncode != 0 or not result.stdout:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    with Image.open(io.BytesIO(result.stdout)) as frame:
        return frame.convert("RGB")


def _extract_probe_frames(
    video_path: str,
    timestamps_ms: Iterable[float],
    *,
    cancel_event: threading.Event | None = None,
    process_observer: Callable[[subprocess.Popen | None], None] | None = None,
) -> dict[float, Image.Image]:
    """Extract all pause-analysis frames with one cancellable FFmpeg process."""
    timestamps = sorted({max(0.0, float(value)) for value in timestamps_ms})
    if not timestamps or (cancel_event is not None and cancel_event.is_set()):
        return {}

    with tempfile.TemporaryDirectory(prefix="zumly-pause-analysis-") as temp_dir:
        root = Path(temp_dir)
        graph_path = root / "extract.ffgraph"
        stderr_path = root / "ffmpeg.stderr.log"
        outputs = [root / f"frame-{index:04d}.png" for index in range(len(timestamps))]

        graph_parts: list[str] = []
        if len(timestamps) == 1:
            sources = ["[0:v]"]
        else:
            split_outputs = "".join(f"[probe{index}]" for index in range(len(timestamps)))
            graph_parts.append(f"[0:v]split={len(timestamps)}{split_outputs}")
            sources = [f"[probe{index}]" for index in range(len(timestamps))]
        for index, (source, timestamp_ms) in enumerate(zip(sources, timestamps)):
            graph_parts.append(
                f"{source}trim=start={timestamp_ms / 1000.0:.6f},"
                f"setpts=PTS-STARTPTS,scale=96:54:flags=bilinear[frame{index}]"
            )
        graph_path.write_text(";\n".join(graph_parts), encoding="utf-8")

        command = [
            ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            video_path,
            "-filter_complex_script",
            str(graph_path),
        ]
        for index, output_path in enumerate(outputs):
            command.extend(
                [
                    "-map",
                    f"[frame{index}]",
                    "-frames:v",
                    "1",
                    "-an",
                    "-c:v",
                    "png",
                    "-y",
                    str(output_path),
                ]
            )

        process: subprocess.Popen | None = None
        return_code = -1
        with stderr_path.open("wb") as stderr_file:
            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=stderr_file,
                    **subprocess_kwargs(),
                )
                if process_observer is not None:
                    process_observer(process)
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        if process.poll() is None:
                            try:
                                process.terminate()
                            except OSError:
                                pass
                        try:
                            process.wait(timeout=0.5)
                        except subprocess.TimeoutExpired:
                            try:
                                process.kill()
                            except OSError:
                                pass
                            process.wait(timeout=1.0)
                        return {}
                    try:
                        return_code = process.wait(timeout=0.05)
                        break
                    except subprocess.TimeoutExpired:
                        continue
            finally:
                if process_observer is not None:
                    process_observer(None)

        if return_code != 0:
            error = stderr_path.read_text(encoding="utf-8", errors="replace").strip()
            raise RuntimeError(error or f"FFmpeg pause analysis failed ({return_code})")

        frames: dict[float, Image.Image] = {}
        for timestamp_ms, output_path in zip(timestamps, outputs):
            if not output_path.is_file():
                continue
            with Image.open(output_path) as frame:
                frames[timestamp_ms] = frame.convert("RGB")
        return frames


def frame_change_score(before: Image.Image, after: Image.Image) -> float:
    """Return normalized mean RGB difference in the inclusive 0..1 range."""
    left = before.convert("RGB").resize((96, 54))
    right = after.convert("RGB").resize((96, 54))
    means = ImageStat.Stat(ImageChops.difference(left, right)).mean
    return max(0.0, min(sum(means) / (len(means) * 255.0), 1.0))


def pause_transition_id(timestamp_ms: float) -> str:
    """Return a stable suggestion ID for one active-time pause boundary."""
    canonical = f"{max(0.0, float(timestamp_ms)):.3f}"
    digest = hashlib.sha256(canonical.encode("ascii")).hexdigest()[:20]
    return f"pause-{digest}"


def suggest_screen_transitions(
    video_path: str,
    capture_telemetry: dict | None,
    duration_ms: float,
    *,
    dismissed_ids: Iterable[str] | None = None,
    threshold: float = 0.08,
    frame_extractor: Callable[[str, float], Image.Image] = _extract_probe_frame,
) -> list[ScreenTransition]:
    """Create an immediate active suggestion for every valid pause boundary.

    ``threshold`` and ``frame_extractor`` remain accepted for compatibility;
    visual scoring is intentionally handled by
    :func:`analyze_screen_transition_changes` after project load.
    """
    del video_path, threshold, frame_extractor
    if not isinstance(capture_telemetry, dict):
        return []
    boundaries: Iterable[dict] = capture_telemetry.get("pauseBoundaries") or []
    dismissed = {str(item) for item in (dismissed_ids or [])}
    suggestions: list[ScreenTransition] = []
    seen: set[str] = set()
    for boundary in boundaries:
        try:
            timestamp_ms = float(boundary.get("timelineMs", -1.0))
        except (AttributeError, TypeError, ValueError):
            continue
        if timestamp_ms <= 0.0 or timestamp_ms >= float(duration_ms):
            continue
        suggestion_id = pause_transition_id(timestamp_ms)
        if suggestion_id in dismissed or suggestion_id in seen:
            continue
        seen.add(suggestion_id)
        suggestions.append(
            ScreenTransition.create(
                timestamp_ms,
                enabled=True,
                suggested=True,
                transition_id=suggestion_id,
                outgoing_frame_ms=boundary.get("outgoingFrameMs"),
                incoming_frame_ms=boundary.get("incomingFrameMs"),
            )
        )
    return suggestions


def analyze_screen_transition_changes(
    video_path: str,
    transitions: Iterable[ScreenTransition],
    duration_ms: float,
    *,
    frame_extractor: Callable[[str, float], Image.Image] = _extract_probe_frame,
    cancel_event: threading.Event | None = None,
    process_observer: Callable[[subprocess.Popen | None], None] | None = None,
) -> dict[str, dict[str, float]]:
    """Resolve legacy endpoint frames without blocking project loading."""
    if not Path(video_path).is_file():
        return {}
    plans: list[tuple[ScreenTransition, float, list[float]]] = []
    requested_timestamps: list[float] = []
    for transition in transitions:
        if cancel_event is not None and cancel_event.is_set():
            return {}
        if not transition.suggested or (
            transition.change_score is not None
            and transition.outgoing_frame_ms is not None
            and transition.incoming_frame_ms is not None
        ):
            continue
        timestamp_ms = float(transition.timestamp_ms)
        if timestamp_ms <= 0.0 or timestamp_ms >= float(duration_ms):
            continue
        outgoing_ms = (
            float(transition.outgoing_frame_ms)
            if transition.outgoing_frame_ms is not None
            else max(0.0, timestamp_ms - 50.0)
        )
        candidate_times = (
            [float(transition.incoming_frame_ms)]
            if transition.incoming_frame_ms is not None
            else [
                min(float(duration_ms), timestamp_ms + offset)
                for offset in (16.0, 33.0, 50.0, 83.0, 116.0, 166.0, 250.0, 400.0)
            ]
        )
        plans.append((transition, outgoing_ms, candidate_times))
        requested_timestamps.append(outgoing_ms)
        requested_timestamps.extend(candidate_times)

    if not plans:
        return {}

    uses_batched_extractor = frame_extractor is _extract_probe_frame
    try:
        if uses_batched_extractor:
            extracted_frames = _extract_probe_frames(
                video_path,
                requested_timestamps,
                cancel_event=cancel_event,
                process_observer=process_observer,
            )
        else:
            extracted_frames = {
                timestamp_ms: frame_extractor(video_path, timestamp_ms)
                for timestamp_ms in sorted(set(requested_timestamps))
                if cancel_event is None or not cancel_event.is_set()
            }
    except Exception as exc:
        logger.warning("Could not batch pause-boundary frames: %s", exc)
        return {}

    scores: dict[str, dict[str, float]] = {}
    for transition, outgoing_ms, candidate_times in plans:
        if cancel_event is not None and cancel_event.is_set():
            return {}
        timestamp_ms = float(transition.timestamp_ms)
        try:
            before = extracted_frames[outgoing_ms]
            best_score = -1.0
            incoming_ms = candidate_times[0]
            for candidate_ms in candidate_times:
                after = extracted_frames.get(candidate_ms)
                if after is None:
                    continue
                score = frame_change_score(before, after)
                if score > best_score:
                    best_score = score
                    incoming_ms = candidate_ms
                if score >= 0.01:
                    break
            scores[transition.id] = {
                "changeScore": max(0.0, best_score),
                "outgoingFrameMs": outgoing_ms,
                "incomingFrameMs": incoming_ms,
            }
        except Exception as exc:
            logger.warning("Could not inspect pause boundary %.1fms: %s", timestamp_ms, exc)
    return scores
