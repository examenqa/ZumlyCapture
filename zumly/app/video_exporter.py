import logging
import math
import os
import subprocess
import threading
import time
import tempfile
import re
import bisect
import uuid
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, List, Optional, Callable

from PIL import Image, ImageDraw, ImageFont

from .models import (
    ZoomKeyframe,
    MousePosition,
    ClickEvent,
    HighlightBox,
    TextAnnotation,
    TimelineOverlay,
    TimelineFrame,
    ScreenTransition,
    VideoSegment,
    VoiceoverSegment,
    BackgroundMusic,
    ClickEffectPreset,
    DEFAULT_CLICK_EFFECT,
    DEFAULT_CURSOR_SCALE,
    normalize_cursor_scale,
    VideoSpaceTransform,
    CanvasLayoutScene,
    ExplainerScene,
    canvas_layout_scene_at,
    canvas_layout_transition_for_range,
    interpolated_canvas_layout_scene,
)
from .audio_mix_builder import AudioInputSpec, AudioMixBuilder
from .backgrounds import BackgroundPreset, DEFAULT_PRESET
from .frames import FramePreset, DEFAULT_FRAME
from .utils import (
    build_encoder_args,
    ffmpeg_exe as _ffmpeg_exe,
    is_hardware_encoder,
    subprocess_kwargs as _subprocess_kwargs,
)
from .font_resolver import FontResolver
from .cursor_registry import (
    cursor_asset_scale as registry_cursor_asset_scale,
    cursor_hotspot as registry_cursor_hotspot,
    cursor_svg_path,
    ensure_cursor_asset,
    get_cursor_preset,
)
from .geometry_math import (
    LayoutSpaceTransform,
    PresentationGroupGeometry,
    Rect2D,
    ease_in_out_quint,
)
from .text_renderer import aligned_offset, canvas_text_metrics, design_px, layout_canvas_text, wrap_canvas_text
from .text_reveal import (
    TEXT_REVEAL_SLIDE_Y,
    bounded_text_reveal_duration_ms,
    normalize_text_reveal_effect,
)
from .explainer_scene import (
    explainer_phase_timing,
    explainer_text_annotation,
    ffmpeg_clamped_progress,
    ffmpeg_lerp,
    ffmpeg_quintic_ease,
)
from .transitions import (
    ffmpeg_graphic_transition_expressions,
    ffmpeg_transition_scene_expressions,
    graphic_bar_material_png,
    graphic_transition_description,
    is_graphic_transition,
    transition_resume_source_ms,
)
from .timeline import EditedTimelineMapper, ordered_video_segments
from .overlay_timeline import visible_overlays_at_output_time
from .video_masks import ffmpeg_static_mask_filters, video_space_masks
from .video_annotations import render_video_annotation_overlay, video_space_annotations

logger = logging.getLogger(__name__)

try:
    _PIL_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9.0
    _PIL_LANCZOS = Image.LANCZOS


def _new_temp_asset_path(suffix: str) -> str:
    """Return a writable, closed-before-use runtime asset path on Windows."""
    root = Path(tempfile.gettempdir()) / "zumly-runtime"
    root.mkdir(parents=True, exist_ok=True)
    return str(root / f"{uuid.uuid4().hex}{suffix}")


def generate_background_png(
    preset: BackgroundPreset,
    width: int,
    height: int,
) -> str:
    """Render the shared background painter into a static FFmpeg input."""
    from PySide6.QtGui import QImage, QPainter

    from .background_renderer import paint_background

    image = QImage(
        max(1, int(width)),
        max(1, int(height)),
        QImage.Format.Format_RGB32,
    )
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    paint_background(painter, float(width), float(height), preset)
    painter.end()

    path = _new_temp_asset_path(".png")
    if not image.save(path, "PNG"):
        raise RuntimeError("Could not generate the export background asset")
    return path


@dataclass
class VideoProbeResult:
    src_fps: float
    total_frames: int
    src_w: int
    src_h: int
    out_w: int
    out_h: int
    fps: float
    is_gif: bool


@dataclass
class GeometryResult:
    scr_x: int
    scr_y: int
    scr_w: int
    scr_h: int
    base_canvas: Any
    screen_mask: Any
    device_mask_u8: Any
    bg: Any


@dataclass
class ExportFilterGraphPlan:
    """Prepared FFmpeg graph and static assets, before subprocess execution."""

    filtergraph: str
    background_img_path: str
    frame_img_path: str
    click_img_path: Optional[str]
    cursor_img_path: Optional[str]
    cursor_motion_tracks: List["CursorMotionTrack"]
    highlight_img_paths: List[str]
    video_annotation_img_paths: List[str]
    text_annotation_img_paths: List[str]
    transition_text_img_paths: List[str]
    timeline_frame_img_paths: List[str]
    voiceover_audio_paths: List[str]
    audio_input_specs: List[AudioInputSpec]
    temp_files: List[str]
    temp_dirs: List[str]
    output_total_sec: float
    has_speed_changes: bool
    has_timeline_edits: bool
    has_source_audio: bool
    has_voiceover_audio: bool
    has_audio_output: bool = False
    audio_output_node: str = ""


@dataclass
class ExportAssetBundle:
    """Static image/audio inputs prepared independently from graph math."""

    background_img_path: str
    background_input_index: int
    frame_img_path: str
    click_img_path: Optional[str]
    cursor_img_path: Optional[str]
    highlight_img_paths: List[str]
    video_annotation_img_paths: List[str]
    local_video_annotation_assets: list[list[tuple[TimelineOverlay, int]]]
    local_highlight_assets: list[list[tuple[HighlightBox, int]]]
    text_annotation_img_paths: List[str]
    local_text_annotation_assets: list[list[tuple[TextAnnotation, int]]]
    timeline_frame_img_paths: List[str]
    temp_files: List[str]
    highlight_base_input: int
    frame_base_input: int


@dataclass(frozen=True)
class CursorMotionTrack:
    """External FFmpeg commands that move one compact cursor overlay."""

    command_path: str
    segment_index: int
    node_index: int
    initial_x: int
    initial_y: int


@dataclass
class ExportSourceProbe:
    """Source video metadata required before graph construction."""

    src_w: int
    src_h: int
    src_fps: float
    total_sec: float
    has_audio: bool = False


@dataclass(frozen=True)
class ExportResult:
    """Authoritative result of a single export attempt."""

    success: bool
    output_path: str
    error_message: str = ""
    ffmpeg_exit_code: int = -1
    requested_encoder_id: str = ""
    encoder_id: str = ""
    fallback_used: bool = False


def export_staging_path(output_path: str) -> str:
    """Return the deterministic sibling used for an atomic MP4 export."""
    return f"{output_path}.tmp"


def _should_retry_hardware_encoder(encoder_id: str, error_message: str) -> bool:
    """Retry only failures attributable to hardware encoder initialization."""
    if not is_hardware_encoder(encoder_id):
        return False
    message = str(error_message or "").lower()
    if any(
        marker in message
        for marker in (
            "cannot allocate memory",
            "out of memory",
            "no space left on device",
            "error initializing complex filters",
            "error binding filtergraph",
            "unconnected output",
            "error reinitializing filters",
            "failed to configure output pad",
            "failed to inject frame",
        )
    ):
        return False
    encoder_markers = (
        "nvenc",
        "cuda",
        "qsv",
        "quick sync",
        "mfx",
        "amf",
        "hardware encoder",
        "no capable devices found",
        "device setup failed",
        "unsupported device",
        "error while opening encoder",
        "initialization failed",
        "initialise encoder",
        "initialize encoder",
    )
    return any(marker in message for marker in encoder_markers)


class GeometryComputer:
    """Pure geometry helper shared by tests and the FFmpeg export graph."""

    def __init__(
        self,
        canvas_w: int,
        canvas_h: int,
        src_w: int,
        src_h: int,
        frame_preset: Optional[FramePreset] = None,
        layout_transform: Optional[LayoutSpaceTransform] = None,
    ) -> None:
        self.canvas_w = int(canvas_w)
        self.canvas_h = int(canvas_h)
        self.src_w = max(int(src_w), 1)
        self.src_h = max(int(src_h), 1)
        self.frame_preset = frame_preset or DEFAULT_FRAME
        self.layout_transform = layout_transform or LayoutSpaceTransform.identity()

    def compute(self) -> dict:
        W = max(self.canvas_w, 1)
        H = max(self.canvas_h, 1)
        video_aspect = self.src_w / self.src_h
        fp = self.frame_preset

        def map_rect(x: float, y: float, width: float, height: float) -> Rect2D:
            mapped = self.layout_transform.map_rect(
                Rect2D(x / W, y / H, width / W, height / H)
            )
            return Rect2D(mapped.x * W, mapped.y * H, mapped.width * W, mapped.height * H)

        def apply_layout(values: dict, *, scale_keys: tuple[str, ...] = ()) -> dict:
            screen = map_rect(values["scr_x"], values["scr_y"], values["scr_w"], values["scr_h"])
            values["scr_x"] = int(round(screen.x))
            values["scr_y"] = int(round(screen.y))
            values["scr_w"] = max(1, int(round(screen.width)))
            values["scr_h"] = max(1, int(round(screen.height)))
            if "dev_x" in values:
                device = map_rect(values["dev_x"], values["dev_y"], values["dev_w"], values["dev_h"])
                values["dev_x"] = int(round(device.x))
                values["dev_y"] = int(round(device.y))
                values["dev_w"] = max(1, int(round(device.width)))
                values["dev_h"] = max(1, int(round(device.height)))
            scale = max(0.01, min(self.layout_transform.width, self.layout_transform.height))
            for key in scale_keys:
                values[key] = int(round(values[key] * scale))
            return values

        if fp.is_none:
            if W / H > video_aspect:
                scr_h = H
                scr_w = int(H * video_aspect)
            else:
                scr_w = W
                scr_h = int(W / video_aspect)
            return apply_layout({
                "scr_x": (W - scr_w) // 2,
                "scr_y": (H - scr_h) // 2,
                "scr_w": max(scr_w, 1),
                "scr_h": max(scr_h, 1),
            })

        preliminary_scale = max((W - 2 * W * fp.padding) / 900.0, 0.01)
        bw_est = fp.bezel_width * preliminary_scale
        pad_x = W * fp.padding
        pad_y = H * fp.padding
        avail_w = max(W - 2 * pad_x, 1.0)
        avail_h = max(H - 2 * pad_y, 1.0)

        dev_h = avail_h
        scr_h_try = max(dev_h - 2 * bw_est, 1.0)
        scr_w_try = scr_h_try * video_aspect
        dev_w = scr_w_try + 2 * bw_est
        if dev_w > avail_w:
            dev_w = avail_w
            scr_w_try = max(dev_w - 2 * bw_est, 1.0)
            scr_h_try = scr_w_try / video_aspect
            dev_h = scr_h_try + 2 * bw_est

        dev_x = (W - dev_w) / 2
        dev_y = (H - dev_h) / 2
        scale = max(dev_w / 900.0, 0.01)
        bw = fp.bezel_width * scale

        scr_x = dev_x + bw
        scr_y = dev_y + bw
        scr_w = max(dev_w - 2 * bw, 1.0)
        scr_h = max(dev_h - 2 * bw, 1.0)

        return apply_layout({
            "scr_x": int(scr_x),
            "scr_y": int(scr_y),
            "scr_w": max(int(scr_w), 1),
            "scr_h": max(int(scr_h), 1),
            "dev_x": int(dev_x),
            "dev_y": int(dev_y),
            "dev_w": max(int(dev_w), 1),
            "dev_h": max(int(dev_h), 1),
            "bw": int(round(bw)),
            "outer_r": int(round(fp.outer_radius * scale)),
            "inner_r": int(round(fp.inner_radius * scale)),
            "edge_thickness": max(int(round(fp.edge_width * scale)), 0),
        }, scale_keys=("bw", "outer_r", "inner_r", "edge_thickness"))


def generate_device_frame_png(preset: FramePreset, w: int, h: int, geom: dict) -> str:
    """Generate a device frame PNG and return the path."""
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    if preset and not preset.is_none and "dev_x" in geom:
        dev_box = [
            geom["dev_x"],
            geom["dev_y"],
            geom["dev_x"] + geom["dev_w"],
            geom["dev_y"] + geom["dev_h"],
        ]
        scr_box = [
            geom["scr_x"],
            geom["scr_y"],
            geom["scr_x"] + geom["scr_w"],
            geom["scr_y"] + geom["scr_h"],
        ]
        if preset.shadow_layers > 0:
            for layer in range(preset.shadow_layers, 0, -1):
                spread = layer * 4
                alpha = max(8, 34 - layer * 5)
                draw.rounded_rectangle(
                    [
                        dev_box[0] - spread,
                        dev_box[1] - spread,
                        dev_box[2] + spread,
                        dev_box[3] + spread,
                    ],
                    radius=geom["outer_r"] + spread,
                    fill=(0, 0, 0, alpha),
                )
        if geom["bw"] > 0:
            draw.rounded_rectangle(
                dev_box,
                radius=geom["outer_r"],
                fill=preset.bezel_color + (255,),
                outline=preset.edge_color + (255,),
                width=max(geom["edge_thickness"], 1),
            )
            draw.rounded_rectangle(
                scr_box,
                radius=geom["inner_r"],
                fill=(0, 0, 0, 0),
            )
        elif preset.shadow_layers > 0:
            draw.rounded_rectangle(
                scr_box,
                radius=geom["inner_r"],
                outline=preset.edge_color + (80,),
                width=max(geom["edge_thickness"], 1),
            )

    path = _new_temp_asset_path(".png")
    img.save(path)
    return path

def generate_click_png(preset: ClickEffectPreset) -> str:
    """Generate a visible click marker PNG for FFmpeg overlay."""
    r = max(int(preset.radius), 12)
    d = max(1, r * 2)
    img = Image.new("RGBA", (d, d), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    color = preset.color
    style = preset.style if preset.style in ("ripple", "burst", "highlight") else "ripple"

    if style == "highlight":
        fill = (color[0], color[1], color[2], min(color[3], 120))
        draw.ellipse([1, 1, d - 2, d - 2], fill=fill, outline=color, width=max(2, r // 8))
    elif style == "burst":
        import math
        cx = cy = r
        for i in range(8):
            angle = 2.0 * math.pi * i / 8
            x1 = cx + math.cos(angle) * r * 0.35
            y1 = cy + math.sin(angle) * r * 0.35
            x2 = cx + math.cos(angle) * r * 0.95
            y2 = cy + math.sin(angle) * r * 0.95
            draw.line([x1, y1, x2, y2], fill=color, width=max(2, r // 8))
        draw.ellipse([r - 4, r - 4, r + 4, r + 4], fill=color)
    else:
        draw.ellipse([2, 2, d - 3, d - 3], outline=color, width=max(3, r // 6))
        draw.ellipse([r - 5, r - 5, r + 5, r + 5], fill=color)

    path = _new_temp_asset_path(".png")
    img.save(path)
    return path


def generate_cursor_png(cursor_asset_path: str = "", cursor_style_id: str = "arrow") -> str:
    """Prepare a custom cursor bitmap or a registered cursor preset."""
    # Packaged artistic cursors stay vector until this export-only raster step.
    # The SVG source is already larger than the maximum presentation cursor
    # size, so the later Lanczos reduction retains crisp edges.
    svg_path = ""
    if str(cursor_asset_path).lower().endswith(".svg"):
        svg_path = cursor_asset_path
    elif not cursor_asset_path:
        svg_path = cursor_svg_path(cursor_style_id)
    if svg_path and os.path.isfile(svg_path):
        try:
            from PySide6.QtCore import QRectF, Qt
            from PySide6.QtGui import QImage, QPainter
            from PySide6.QtSvg import QSvgRenderer

            renderer = QSvgRenderer(svg_path)
            if renderer.isValid():
                size = renderer.defaultSize()
                width = max(int(size.width()), 1)
                height = max(int(size.height()), 1)
                image = QImage(width, height, QImage.Format.Format_RGBA8888)
                image.fill(Qt.GlobalColor.transparent)
                painter = QPainter(image)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
                renderer.render(painter, QRectF(0, 0, width, height))
                painter.end()
                path = _new_temp_asset_path(".png")
                if image.save(path, "PNG"):
                    return path
                raise OSError("QSvgRenderer could not save cursor PNG")
        except Exception:
            logger.warning("Could not rasterize SVG cursor %s", svg_path, exc_info=True)

    if cursor_asset_path and os.path.isfile(cursor_asset_path):
        try:
            with Image.open(cursor_asset_path) as source:
                img = source.convert("RGBA")
                if img.width <= 0 or img.height <= 0:
                    raise ValueError("cursor image has invalid dimensions")
                img.thumbnail((96, 96), _PIL_LANCZOS)
                padded = Image.new(
                    "RGBA", (img.width + 8, img.height + 8), (0, 0, 0, 0)
                )
                padded.alpha_composite(img, (4, 4))
                path = _new_temp_asset_path(".png")
                padded.save(path)
                return path
        except (OSError, ValueError):
            logger.warning(
                "Custom cursor asset is unreadable; using the standard arrow: %s",
                cursor_asset_path,
            )

    try:
        registered_path = ensure_cursor_asset(cursor_style_id)
        with Image.open(registered_path) as source:
            img = source.convert("RGBA")
    except Exception:
        logger.warning("Cursor preset %s could not be rendered; using arrow", cursor_style_id, exc_info=True)
        with Image.open(ensure_cursor_asset("arrow")) as source:
            img = source.convert("RGBA")

    path = _new_temp_asset_path(".png")
    img.save(path)
    return path


def generate_highlight_png(highlight: HighlightBox, w: int, h: int, geom: dict) -> str:
    """Generate an optional highlight border asset.

    Dimming and the transparent shape hole are constructed in FFmpeg so the
    exported geometry is evaluated in the same timeline as the video.
    """
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    scr_x = int(geom["scr_x"])
    scr_y = int(geom["scr_y"])
    scr_w = int(geom["scr_w"])
    scr_h = int(geom["scr_h"])
    hx = int(scr_x + max(0.0, min(1.0, float(highlight.x))) * scr_w)
    hy = int(scr_y + max(0.0, min(1.0, float(highlight.y))) * scr_h)
    hx = max(scr_x, min(scr_x + scr_w - 1, hx))
    hy = max(scr_y, min(scr_y + scr_h - 1, hy))
    hw = max(1, int(max(0.0, min(1.0, float(highlight.width))) * scr_w))
    hh = max(1, int(max(0.0, min(1.0, float(highlight.height))) * scr_h))
    box = [hx, hy, max(hx + 1, min(scr_x + scr_w, hx + hw)), max(hy + 1, min(scr_y + scr_h, hy + hh))]

    border_width = max(0, int(getattr(highlight, "border_width", 0)))
    border_alpha = round(max(0.0, min(1.0, float(getattr(highlight, "opacity", 1.0)))) * 255)
    border_color = tuple(highlight.color[:3]) + (border_alpha,)

    if getattr(highlight, "shape", "rect") == "circle":
        draw.ellipse(box, fill=(0, 0, 0, 0))
        if border_width > 0:
            draw.ellipse(
                box,
                outline=border_color,
                width=border_width,
            )
    else:
        radius = round(
            min(hw, hh)
            * max(0.0, min(0.5, float(getattr(highlight, "corner_radius", 0.14))))
        )
        draw.rounded_rectangle(box, radius=radius, fill=(0, 0, 0, 0))
        if border_width > 0:
            draw.rounded_rectangle(
                box,
                radius=radius,
                outline=border_color,
                width=border_width,
            )

    path = _new_temp_asset_path(".png")
    img.save(path)
    return path


def _ffmpeg_filter_path(path: str) -> str:
    """Escape an external sidecar path for FFmpeg filtergraph syntax."""
    normalized = str(Path(path).resolve()).replace("\\", "/")
    return normalized.replace(":", r"\:").replace("'", r"\'")


def _generate_cursor_motion_track(
    *,
    points: list[tuple],
    anchor_x: float,
    anchor_y: float,
    segment_index: int,
    node_index: int,
) -> CursorMotionTrack:
    """Write compact cursor x/y commands instead of full-canvas PNG frames."""
    if not points:
        raise ValueError("Cursor motion track requires at least one point")

    path = _new_temp_asset_path(".cmd")
    overlay_name = f"overlay@cursor{segment_index}"
    commands: list[str] = []
    last_position: tuple[int, int] | None = None
    for point in points:
        timestamp = max(0.0, float(point[0]))
        x = int(round(float(point[1]) - anchor_x))
        y = int(round(float(point[2]) - anchor_y))
        if last_position == (x, y):
            continue
        commands.append(
            f"{timestamp:.6f} {overlay_name} x {x}, {overlay_name} y {y};"
        )
        last_position = (x, y)
    if not commands:
        commands.append(f"0.000000 {overlay_name} x 0, {overlay_name} y 0;")
    Path(path).write_text("\n".join(commands), encoding="utf-8")
    first = points[0]
    return CursorMotionTrack(
        command_path=path,
        segment_index=segment_index,
        node_index=node_index,
        initial_x=int(round(float(first[1]) - anchor_x)),
        initial_y=int(round(float(first[2]) - anchor_y)),
    )


def _is_valid_cursor_asset(path: str) -> bool:
    """Return whether a custom cursor bitmap or SVG can be rendered."""
    if not path or not os.path.isfile(path):
        return False
    if str(path).lower().endswith(".svg"):
        try:
            from PySide6.QtSvg import QSvgRenderer

            return QSvgRenderer(path).isValid()
        except Exception:
            return False
    try:
        with Image.open(path) as image:
            image.load()
            return image.width > 0 and image.height > 0
    except (OSError, ValueError):
        return False


_FONT_RESOLVER = FontResolver()


def _load_annotation_font(font_family: str, pixel_size: int):
    """Resolve the same deterministic font path used by export assets."""
    try:
        return ImageFont.truetype(
            _FONT_RESOLVER.resolve(font_family),
            pixel_size,
        )
    except OSError:
        logger.warning("Font asset could not be loaded: %s", font_family)
        return ImageFont.load_default()


def generate_text_annotation_png(annotation: TextAnnotation, w: int, h: int) -> str:
    """Render one full-canvas transparent TextAnnotation asset."""
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    resolved = canvas_text_metrics(annotation.font_size, h, dpi_scale=1.0)
    font = _load_annotation_font(annotation.font_family, resolved.font_px)
    text = str(annotation.text or "")
    opacity = max(0.0, min(1.0, float(annotation.opacity)))
    measure = lambda value: draw.textlength(value, font=font)
    anchor_x = max(0.0, min(float(annotation.x) * w, float(w)))
    anchor_y = max(0.0, min(float(annotation.y) * h, float(h)))
    region_w = max(
        min(
            float(w) - anchor_x,
            float(w) * float(annotation.text_width or annotation.max_width),
        ),
        1.0,
    )
    layout = layout_canvas_text(
        text,
        region_w,
        resolved,
        measure,
    )
    padding = resolved.padding_px
    box_w = min(w, max(1, int(round(layout.box_width_px))))
    box_h = min(h, max(1, int(round(layout.box_height_px))))
    horizontal_alignment = (
        "left" if annotation.horizontal_alignment == "auto" else annotation.horizontal_alignment
    )
    left = anchor_x + aligned_offset(region_w, box_w, horizontal_alignment)
    left = max(0, min(int(round(left)), w - box_w))
    region_h = (
        max(box_h, min(float(h) - anchor_y, float(h) * float(annotation.text_height)))
        if float(annotation.text_height) > 0.0
        else float(box_h)
    )
    vertical_room = max(0.0, region_h - box_h)
    if annotation.vertical_alignment == "center":
        vertical_offset = vertical_room / 2.0
    elif annotation.vertical_alignment == "bottom":
        vertical_offset = vertical_room
    else:
        vertical_offset = 0.0
    top = max(0, min(int(round(anchor_y + vertical_offset)), h - box_h))

    if annotation.background_color is not None:
        bg = tuple(annotation.background_color[:3]) + (
            int(annotation.background_color[3] * opacity),
        )
        draw.rounded_rectangle(
            [left, top, left + box_w, top + box_h],
            radius=min(6, padding),
            fill=bg,
        )
    fg = tuple(annotation.color[:3]) + (int(annotation.color[3] * opacity),)
    for line_index, line in enumerate(layout.lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        draw.text(
            (
                left + padding - bbox[0],
                top + padding + line_index * resolved.line_height_px - bbox[1],
            ),
            line,
            font=font,
            fill=fg,
        )
    path = _new_temp_asset_path(".png")
    img.save(path)
    return path


def _highlight_mask_filter(
    highlight: HighlightBox,
    *,
    out_w: int,
    out_h: int,
    src_fps: float,
    duration_sec: float,
    geom: dict,
    output_node: str,
) -> str:
    """Build a bounded black dim layer with a shape-specific alpha hole."""
    screen_left = float(geom["scr_x"])
    screen_top = float(geom["scr_y"])
    screen_right = screen_left + float(geom["scr_w"])
    screen_bottom = screen_top + float(geom["scr_h"])
    x = max(0.0, min(1.0, float(highlight.x)))
    y = max(0.0, min(1.0, float(highlight.y)))
    width = max(0.0, min(1.0, float(highlight.width)))
    height = max(0.0, min(1.0, float(highlight.height)))
    left = screen_left + x * float(geom["scr_w"])
    top = screen_top + y * float(geom["scr_h"])
    right = left + width * float(geom["scr_w"])
    bottom = top + height * float(geom["scr_h"])
    dim_alpha = int(max(0.0, min(0.9, float(highlight.dim_opacity))) * 255)

    # Escaped commas keep the comparison functions inside the geq expression
    # when this string is parsed from filter_complex_script.
    inside_screen = (
        f"gt(X\\,{screen_left:.6f})*lt(X\\,{screen_right:.6f})*"
        f"gt(Y\\,{screen_top:.6f})*lt(Y\\,{screen_bottom:.6f})"
    )
    if getattr(highlight, "shape", "rect") == "circle":
        cx = (left + right) / 2.0
        cy = (top + bottom) / 2.0
        radius = max(0.5, min(right - left, bottom - top) / 2.0)
        hole = (
            f"lt((X-{cx:.6f})*(X-{cx:.6f})+(Y-{cy:.6f})*(Y-{cy:.6f})"
            f"\\,{radius * radius:.6f})"
        )
    else:
        radius = min(right - left, bottom - top) * max(
            0.0, min(0.5, float(getattr(highlight, "corner_radius", 0.14)))
        )
        if radius <= 0.5:
            hole = (
                f"gt(X\\,{left:.6f})*lt(X\\,{right:.6f})*"
                f"gt(Y\\,{top:.6f})*lt(Y\\,{bottom:.6f})"
            )
        else:
            cx = (left + right) / 2.0
            cy = (top + bottom) / 2.0
            inner_w = max(0.0, (right - left) / 2.0 - radius)
            inner_h = max(0.0, (bottom - top) / 2.0 - radius)
            dx = f"max(abs(X-{cx:.6f})-{inner_w:.6f}\\,0)"
            dy = f"max(abs(Y-{cy:.6f})-{inner_h:.6f}\\,0)"
            hole = f"lt(({dx})*({dx})+({dy})*({dy})\\,{radius * radius:.6f})"

    alpha = f"{dim_alpha}*({inside_screen})*(1-({hole}))"
    return (
        f"color=c=black@1.0:s={out_w}x{out_h}:r={src_fps}:d={duration_sec:.6f},"
        f"format=rgba,geq=r=0:g=0:b=0:a='{alpha}'[{output_node}]"
    )


def _parse_hex_color(value: str, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    text = (value or "").strip().lstrip("#")
    if len(text) != 6:
        return fallback
    try:
        return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError:
        return fallback


def _load_frame_font(size: int, family: str = "Segoe UI") -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            _FONT_RESOLVER.resolve(family),
            size=max(12, int(size)),
        )
    except OSError:
        logger.warning("Timeline frame font asset could not be loaded: %s", family)
        return ImageFont.load_default()


def _wrap_text_for_width(text: str, font: ImageFont.ImageFont, max_width: int, draw: ImageDraw.ImageDraw) -> list[str]:
    lines: list[str] = []
    for paragraph in (text or "").splitlines() or [""]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def _clean_timeline_frame_text(text: str) -> str:
    bidi_controls = {
        "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",
        "\u2066", "\u2067", "\u2068", "\u2069",
        "\u200e", "\u200f",
    }
    return "".join(ch for ch in text if ch not in bidi_controls)


def _timeline_frame_text_sections(frame: TimelineFrame) -> list[tuple[str, int]]:
    """Return structured copy while preserving legacy single-text cards."""
    title = _clean_timeline_frame_text(str(getattr(frame, "title", "") or "")).strip()
    description = _clean_timeline_frame_text(
        str(getattr(frame, "description", "") or "")
    ).strip()
    if title or description:
        rows: list[tuple[str, int]] = []
        if title:
            rows.append((title, int(getattr(frame, "title_font_size", 64) or 64)))
        if description:
            rows.append((description, int(getattr(frame, "body_font_size", 38) or 38)))
        return rows
    return [
        (
            _clean_timeline_frame_text(frame.text or "Add your text"),
            int(frame.font_size or 54),
        )
    ]


def generate_timeline_frame_png(
    frame: TimelineFrame,
    w: int,
    h: int,
    *,
    text_only: bool = False,
) -> str:
    """Generate a full card or a transparent typography layer for export."""
    bg = _parse_hex_color(frame.background_color, (17, 24, 39))
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0) if text_only else bg + (255,))

    if frame.kind == "image" and frame.image_path and os.path.isfile(frame.image_path):
        try:
            with Image.open(frame.image_path) as source:
                source = source.convert("RGBA")
                if getattr(frame, "image_fit", "fit") == "fill":
                    scale = max(w / max(source.width, 1), h / max(source.height, 1))
                    source = source.resize(
                        (max(1, round(source.width * scale)), max(1, round(source.height * scale))),
                        _PIL_LANCZOS,
                    )
                    left = max(0, (source.width - w) // 2)
                    top = max(0, (source.height - h) // 2)
                    source = source.crop((left, top, left + w, top + h))
                else:
                    source.thumbnail((int(w * 0.92), int(h * 0.88)), _PIL_LANCZOS)
                x = (w - source.width) // 2
                y = (h - source.height) // 2
                canvas.alpha_composite(source, (x, y))
        except Exception as exc:
            logger.warning("Failed to render image frame %s: %s", frame.image_path, exc)

    if frame.kind == "text" or not (frame.kind == "image" and frame.image_path and os.path.isfile(frame.image_path)):
        draw = ImageDraw.Draw(canvas)
        text_color = _parse_hex_color(frame.text_color, (249, 250, 251))
        max_width = int(w * 0.78)
        rendered_sections: list[tuple[ImageFont.ImageFont, tuple[str, ...], int]] = []
        for text, design_size in _timeline_frame_text_sections(frame):
            font = _load_frame_font(
                design_px(design_size, h),
                getattr(frame, "font_family", "Segoe UI"),
            )
            lines = wrap_canvas_text(
                text,
                max_width,
                lambda value, current=font: draw.textlength(value, font=current),
            )
            line_height = max(1, int(round(getattr(font, "size", design_px(design_size, h)) * 1.2)))
            rendered_sections.append((font, lines, line_height))
        section_spacing = design_px(int(getattr(frame, "content_spacing", 22) or 0), h)
        total_height = sum(len(lines) * line_height for _font, lines, line_height in rendered_sections)
        total_height += max(0, len(rendered_sections) - 1) * section_spacing
        y = max(40, (h - total_height) // 2)
        alignment = str(getattr(frame, "text_alignment", "center") or "center")
        for section_index, (font, lines, line_height) in enumerate(rendered_sections):
            for line in lines:
                bbox = draw.textbbox((0, 0), line or " ", font=font)
                line_width = max(0, bbox[2] - bbox[0])
                if alignment == "left":
                    x = int(w * 0.11)
                elif alignment == "right":
                    x = int(w * 0.89) - line_width
                else:
                    x = (w - line_width) // 2
                draw.text((x - bbox[0], y - bbox[1]), line, font=font, fill=text_color + (255,))
                y += line_height
            if section_index < len(rendered_sections) - 1:
                y += section_spacing

    path = _new_temp_asset_path("_timeline_frame.png")
    canvas.save(path)
    return path


def _segment_speed(segment: VideoSegment) -> float:
    """Return a bounded playback speed for export retiming."""
    try:
        speed = float(segment.speed)
    except (TypeError, ValueError):
        return 1.0
    if speed <= 0:
        return 1.0
    return min(speed, 10.0)


def _atempo_filters(tempo: float) -> list[str]:
    """Return FFmpeg-safe atempo stages for an arbitrary bounded tempo."""
    remaining = max(0.1, min(float(tempo), 10.0))
    stages: list[str] = []
    while remaining > 2.0 + 1e-9:
        stages.append("atempo=2.000000")
        remaining /= 2.0
    while remaining < 0.5 - 1e-9:
        stages.append("atempo=0.500000")
        remaining /= 0.5
    stages.append(f"atempo={remaining:.6f}")
    return stages


def _normalize_video_segments(
    video_segments: Optional[List[VideoSegment]],
    duration_ms: float,
    fill_gaps: bool = True,
) -> List[VideoSegment]:
    """Return source-time segments for export.

    With editor-authored segments, list order is the output order and gaps are
    real cuts. Legacy/no-segment payloads still get one full-duration segment.
    """
    duration_ms = max(float(duration_ms or 0.0), 0.0)
    if duration_ms <= 0:
        return []

    valid: List[VideoSegment] = []
    source_segments = ordered_video_segments(video_segments)
    if fill_gaps:
        source_segments.sort(key=lambda s: float(s.start_ms))
    for segment in source_segments:
        start = max(0.0, min(float(segment.start_ms), duration_ms))
        end = max(0.0, min(float(segment.end_ms), duration_ms))
        if end <= start:
            continue
        valid.append(
            VideoSegment(
                id=segment.id,
                start_ms=start,
                end_ms=end,
                speed=_segment_speed(segment),
                sequence_index=int(getattr(segment, "sequence_index", len(valid))),
            )
        )

    if not valid or not fill_gaps:
        if valid:
            return valid
        return [VideoSegment.create(0.0, duration_ms, 1.0)]

    normalized: List[VideoSegment] = []
    cursor = 0.0
    for segment in valid:
        if segment.start_ms > cursor:
            normalized.append(VideoSegment.create(cursor, segment.start_ms, 1.0))
        start = max(segment.start_ms, cursor)
        if segment.end_ms > start:
            normalized.append(
                VideoSegment(
                    id=segment.id,
                    start_ms=start,
                    end_ms=segment.end_ms,
                    speed=segment.speed,
                )
            )
            cursor = segment.end_ms
    if cursor < duration_ms:
        normalized.append(VideoSegment.create(cursor, duration_ms, 1.0))
    return normalized


def _split_segments_at_timeline_frames(
    segments: List[VideoSegment],
    timeline_frames: List[TimelineFrame],
) -> List[VideoSegment]:
    """Split export-only source segments at every inserted-card anchor."""
    split_points = sorted({
        float(frame.timestamp_ms)
        for frame in timeline_frames
    })
    if not split_points:
        return segments

    result: List[VideoSegment] = []
    epsilon = 0.5
    for segment in segments:
        internal_points = [
            point
            for point in split_points
            if float(segment.start_ms) + epsilon < point < float(segment.end_ms) - epsilon
        ]
        boundaries = [float(segment.start_ms), *internal_points, float(segment.end_ms)]
        for part_index, (start_ms, end_ms) in enumerate(zip(boundaries, boundaries[1:])):
            result.append(
                VideoSegment(
                    id=(segment.id if part_index == 0 else f"{segment.id}:frame:{part_index}"),
                    start_ms=start_ms,
                    end_ms=end_ms,
                    speed=_segment_speed(segment),
                )
            )
    return result


def _split_segments_at_screen_transitions(
    segments: List[VideoSegment],
    transitions: List[ScreenTransition],
) -> List[VideoSegment]:
    """Split at transitions and retire stale boundary frames after each one."""
    if not transitions:
        return segments

    source_duration_ms = max(
        (float(segment.end_ms) for segment in segments), default=0.0
    )
    result: List[VideoSegment] = []
    epsilon = 0.5
    for segment in segments:
        segment_id = str(segment.id or "")
        boundaries = []
        for transition in transitions:
            if not transition.enabled:
                continue
            clip_id = str(getattr(transition, "clip_id", "") or "")
            if clip_id and not (
                segment_id == clip_id or segment_id.startswith(f"{clip_id}:")
            ):
                continue
            anchor = float(transition.timestamp_ms)
            resume = transition_resume_source_ms(transition, source_duration_ms)
            if (
                float(segment.start_ms) - epsilon <= anchor
                < float(segment.end_ms) - epsilon
            ):
                boundaries.append((anchor, max(anchor, resume)))
        cursor = float(segment.start_ms)
        part_index = 0
        for anchor, resume in sorted(boundaries):
            if anchor > cursor + epsilon:
                result.append(
                    VideoSegment(
                        id=(segment.id if part_index == 0 else f"{segment.id}:transition:{part_index}"),
                        start_ms=cursor,
                        end_ms=min(anchor, float(segment.end_ms)),
                        speed=_segment_speed(segment),
                    )
                )
                part_index += 1
            cursor = max(cursor, min(resume, float(segment.end_ms)))
        if cursor < float(segment.end_ms) - epsilon:
            result.append(
                VideoSegment(
                    id=(segment.id if part_index == 0 else f"{segment.id}:transition:{part_index}"),
                    start_ms=cursor,
                    end_ms=float(segment.end_ms),
                    speed=_segment_speed(segment),
                )
            )
    return result


def _split_segments_at_layout_scenes(
    segments: List[VideoSegment],
    scenes: Optional[List[CanvasLayoutScene]],
) -> List[VideoSegment]:
    """Split source ranges at layout boundaries and ease transition starts."""
    if not scenes:
        return segments
    split_points: set[float] = set()
    for scene in scenes:
        split_points.add(float(scene.start_ms))
        split_points.add(float(scene.end_ms))
        if scene.transition == "ease" and scene.transition_duration_ms > 0:
            split_points.add(
                max(0.0, float(scene.start_ms) - float(scene.transition_duration_ms))
            )

    result: List[VideoSegment] = []
    epsilon = 0.5
    for segment in segments:
        internal_points = sorted(
            point
            for point in split_points
            if float(segment.start_ms) + epsilon < point < float(segment.end_ms) - epsilon
        )
        boundaries = [float(segment.start_ms), *internal_points, float(segment.end_ms)]
        for part_index, (start_ms, end_ms) in enumerate(zip(boundaries, boundaries[1:])):
            result.append(
                VideoSegment(
                    id=(segment.id if part_index == 0 else f"{segment.id}:layout:{part_index}"),
                    start_ms=start_ms,
                    end_ms=end_ms,
                    speed=_segment_speed(segment),
                )
            )
    return result


def _split_segments_at_text_animations(
    segments: List[VideoSegment],
    annotations: Optional[List[TextAnnotation]],
) -> List[VideoSegment]:
    """Split at text phase boundaries so animation state never restarts."""
    split_points: set[float] = set()
    for annotation in annotations or []:
        if str(annotation.animation or "none") == "none":
            continue
        start = float(annotation.start_ms)
        end = float(annotation.end_ms)
        enter_start = min(end, start + float(annotation.animation_delay_ms))
        enter_end = min(end, enter_start + float(annotation.animation_in_ms))
        exit_start = max(enter_start, end - float(annotation.animation_out_ms))
        split_points.update((start, enter_start, enter_end, exit_start, end))
    if not split_points:
        return segments

    result: List[VideoSegment] = []
    epsilon = 0.5
    for segment in segments:
        internal_points = sorted(
            point
            for point in split_points
            if float(segment.start_ms) + epsilon < point < float(segment.end_ms) - epsilon
        )
        boundaries = [float(segment.start_ms), *internal_points, float(segment.end_ms)]
        for part_index, (start_ms, end_ms) in enumerate(zip(boundaries, boundaries[1:])):
            result.append(
                VideoSegment(
                    id=(
                        segment.id
                        if part_index == 0
                        else f"{segment.id}:text:{part_index}"
                    ),
                    start_ms=start_ms,
                    end_ms=end_ms,
                    speed=_segment_speed(segment),
                )
            )
    return result


class _SessionMediaMapper:
    """Map recorder session timestamps onto the encoded MP4 media timeline."""

    def __init__(
        self,
        frame_timestamps: Optional[List[float]],
        media_duration_sec: float,
        fps: float,
    ) -> None:
        self._timestamps = sorted(float(ts) for ts in (frame_timestamps or []) if ts is not None)
        self._media_duration_sec = max(float(media_duration_sec or 0.0), 0.0)
        self._fps = max(float(fps or 0.0), 0.0)
        if self._timestamps and self._media_duration_sec > 0:
            self._frame_duration_sec = self._media_duration_sec / len(self._timestamps)
        elif self._fps > 0:
            self._frame_duration_sec = 1.0 / self._fps
        else:
            self._frame_duration_sec = 1.0 / 30.0

    @property
    def has_timestamps(self) -> bool:
        return bool(self._timestamps)

    def to_media_sec(self, session_time_ms: float, *, for_end: bool = False) -> float:
        if not self._timestamps:
            return max(float(session_time_ms), 0.0) / 1000.0

        target = float(session_time_ms)
        last_media_start = max(0.0, self._media_duration_sec - self._frame_duration_sec)
        if target <= self._timestamps[0]:
            return min(self._frame_duration_sec, self._media_duration_sec) if for_end else 0.0
        if target >= self._timestamps[-1]:
            return self._media_duration_sec if for_end else last_media_start

        idx = bisect.bisect_left(self._timestamps, target)
        if idx <= 0:
            frame_pos = 0.0
        elif idx >= len(self._timestamps):
            frame_pos = float(len(self._timestamps) if for_end else len(self._timestamps) - 1)
        else:
            prev_ts = self._timestamps[idx - 1]
            next_ts = self._timestamps[idx]
            if next_ts <= prev_ts:
                frame_pos = float(idx)
            else:
                ratio = (target - prev_ts) / (next_ts - prev_ts)
                frame_pos = float(idx - 1) + max(0.0, min(1.0, ratio))

        media_sec = frame_pos * self._frame_duration_sec
        return max(0.0, min(self._media_duration_sec, media_sec))

    def get_media_time(self, session_time_ms: float, *, for_end: bool = False) -> float:
        """Map session milliseconds to encoded media milliseconds."""
        return self.to_media_sec(session_time_ms, for_end=for_end) * 1000.0

    def segment_bounds(self, segment: VideoSegment) -> tuple[float, float, float]:
        start_sec = self.to_media_sec(segment.start_ms, for_end=False)
        end_sec = self.to_media_sec(segment.end_ms, for_end=True)
        min_duration = max(self._frame_duration_sec, 0.001)
        if end_sec <= start_sec:
            end_sec = min(self._media_duration_sec or start_sec + min_duration, start_sec + min_duration)
        media_duration = max(end_sec - start_sec, min_duration)
        return start_sec, end_sec, media_duration


class _RawSessionMediaMapper:
    """Use recorder session timestamps directly for normal-duration sources."""

    def __init__(self, fps: float) -> None:
        self._fps = max(float(fps or 0.0), 0.0)
        self._frame_duration_sec = 1.0 / self._fps if self._fps > 0 else 1.0 / 30.0

    @property
    def has_timestamps(self) -> bool:
        return False

    def get_media_time(self, session_time_ms: float, *, for_end: bool = False) -> float:
        return max(float(session_time_ms), 0.0)

    def segment_bounds(self, segment: VideoSegment) -> tuple[float, float, float]:
        start_sec = max(float(segment.start_ms), 0.0) / 1000.0
        end_sec = max(float(segment.end_ms), float(segment.start_ms)) / 1000.0
        if end_sec <= start_sec:
            end_sec = start_sec + max(self._frame_duration_sec, 0.001)
        duration_sec = max(end_sec - start_sec, max(self._frame_duration_sec, 0.001))
        return start_sec, end_sec, duration_sec


def _should_use_compressed_time_mapper(
    frame_timestamps: Optional[List[float]],
    source_duration_ms: float,
    media_duration_sec: float,
) -> bool:
    """Detect old time-lapse captures whose encoded media is much shorter."""
    if not frame_timestamps or source_duration_ms <= 0 or media_duration_sec <= 0:
        return False
    source_duration_sec = source_duration_ms / 1000.0
    if source_duration_sec <= 0:
        return False
    shortfall_sec = source_duration_sec - media_duration_sec
    tolerance_sec = max(0.75, source_duration_sec * 0.05)
    return shortfall_sec > tolerance_sec


def _select_session_media_mapper(
    frame_timestamps: Optional[List[float]],
    source_duration_ms: float,
    media_duration_sec: float,
    fps: float,
    is_cfr: bool = False,
) -> _SessionMediaMapper | _RawSessionMediaMapper:
    """Choose direct wall-clock timing for normal captures, compressed mapping for old ones."""
    if is_cfr:
        logger.info("Using raw session timeline for CFR-stabilized capture")
        return _RawSessionMediaMapper(fps)

    if _should_use_compressed_time_mapper(frame_timestamps, source_duration_ms, media_duration_sec):
        logger.info(
            "Using %d frame timestamps to map %.3fs session timeline onto %.3fs compressed media timeline",
            len(frame_timestamps or []),
            source_duration_ms / 1000.0,
            media_duration_sec,
        )
        return _SessionMediaMapper(frame_timestamps, media_duration_sec, fps)

    if source_duration_ms > 0 and media_duration_sec > 0:
        logger.info(
            "Using raw session timeline: %.3fs session, %.3fs media",
            source_duration_ms / 1000.0,
            media_duration_sec,
        )
    return _RawSessionMediaMapper(fps)


def _media_keyframes_for_segment(
    keyframes: List[ZoomKeyframe],
    segment: VideoSegment,
    mapper: _SessionMediaMapper | _RawSessionMediaMapper,
    media_start_sec: float,
) -> List[ZoomKeyframe]:
    """Filter keyframes into a segment-local mapped media timeline for FFmpeg trim."""
    media_keyframes: List[ZoomKeyframe] = []
    for keyframe in keyframes:
        keyframe_time = float(keyframe.timestamp)
        if not (segment.start_ms <= keyframe_time < segment.end_ms):
            continue
        media_start_ms = mapper.get_media_time(keyframe_time)
        media_time_ms = max(0.0, media_start_ms - (media_start_sec * 1000.0))
        duration = max(float(keyframe.duration), 0.0)
        if duration > 0:
            keyframe_end = min(float(segment.end_ms), keyframe_time + duration)
            media_end_ms = max(
                0.0,
                mapper.get_media_time(keyframe_end, for_end=True) - (media_start_sec * 1000.0),
            )
            duration = max(0.0, media_end_ms - media_time_ms)
        media_keyframes.append(replace(keyframe, timestamp=media_time_ms, duration=duration))
    return media_keyframes


def _media_time_for_segment(
    timestamp_ms: float,
    segment: VideoSegment,
    mapper: _SessionMediaMapper | _RawSessionMediaMapper,
    media_start_sec: float,
    *,
    for_end: bool = False,
) -> float:
    """Map an absolute session timestamp to segment-local mapped media seconds."""
    bounded = max(float(segment.start_ms), min(float(timestamp_ms), float(segment.end_ms)))
    return max(0.0, (mapper.get_media_time(bounded, for_end=for_end) / 1000.0) - media_start_sec)


def _media_window_for_segment(
    timestamp_ms: float,
    duration_ms: float,
    segment: VideoSegment,
    mapper: _SessionMediaMapper | _RawSessionMediaMapper,
    media_start_sec: float,
) -> tuple[float, float]:
    """Map a session-time overlay window to segment-local mapped media seconds."""
    start_ms = max(float(segment.start_ms), min(float(timestamp_ms), float(segment.end_ms)))
    end_ms = max(start_ms, min(start_ms + max(float(duration_ms), 0.0), float(segment.end_ms)))
    start_sec = _media_time_for_segment(start_ms, segment, mapper, media_start_sec)
    end_sec = _media_time_for_segment(end_ms, segment, mapper, media_start_sec, for_end=True)
    return start_sec, end_sec


def _timed_overlay_filter(
    main_node: str,
    overlay_node: str,
    output_node: str,
    *,
    x: int = 0,
    y: int = 0,
    start_sec: float,
    end_sec: float,
    end_with_main: bool = False,
) -> str:
    """Overlay a static asset in the main stream's timeline domain.

    The asset input is a single PNG frame. ``repeatlast`` holds that frame in
    the normalized main-video clock without feeding an unbounded image stream
    through shared ``split`` nodes.
    """
    start = max(float(start_sec), 0.0)
    end = max(float(end_sec), start + 0.001)
    lifetime = (
        "shortest=1:eof_action=pass:repeatlast=1"
        if end_with_main
        else "eof_action=repeat:repeatlast=1"
    )
    return (
        f"[{main_node}][{overlay_node}]overlay=x={x}:y={y}:"
        f"{lifetime}:"
        f"enable='between(t,{start:.6f},{end:.6f})'[{output_node}]"
    )


def _animated_text_overlay_filters(
    main_node: str,
    overlay_node: str,
    output_node: str,
    annotation: TextAnnotation,
    *,
    start_sec: float,
    end_sec: float,
    out_w: int,
    out_h: int,
) -> list[str]:
    """Build the bounded text animation shared with ExplainerScene semantics."""
    animation = str(annotation.animation or "none")
    if animation == "none":
        return [
            _timed_overlay_filter(
                main_node,
                overlay_node,
                output_node,
                start_sec=start_sec,
                end_sec=end_sec,
                end_with_main=True,
            )
        ]

    enter_start = start_sec + max(annotation.animation_delay_ms, 0.0) / 1000.0
    enter_duration = max(annotation.animation_in_ms / 1000.0, 0.001)
    exit_duration = max(annotation.animation_out_ms / 1000.0, 0.001)
    exit_start = max(enter_start, end_sec - exit_duration)
    apply_fade_in = bool(getattr(annotation, "_apply_fade_in", True))
    apply_fade_out = bool(getattr(annotation, "_apply_fade_out", True))
    prepared = f"{output_node}asset"
    asset_filters = ["format=rgba"]
    if apply_fade_in:
        asset_filters.append(
            f"fade=t=in:st={enter_start:.6f}:d={enter_duration:.6f}:alpha=1"
        )
    if apply_fade_out:
        asset_filters.append(
            f"fade=t=out:st={exit_start:.6f}:d={exit_duration:.6f}:alpha=1"
        )
    filters = [f"[{overlay_node}]{','.join(asset_filters)}[{prepared}]"]
    overlay_x = "0"
    overlay_y = "0"
    if animation == "fade-slide":
        in_progress = (
            ffmpeg_clamped_progress("t", enter_start, enter_start + enter_duration)
            if apply_fade_in
            else "1"
        )
        out_progress = (
            ffmpeg_clamped_progress("t", exit_start, end_sec)
            if apply_fade_out
            else "0"
        )
        motion = f"clip(1-({in_progress})+({out_progress}),0,1)"
        overlay_x = f"({annotation.slide_offset_x * out_w:.6f})*({motion})"
        overlay_y = f"({annotation.slide_offset_y * out_h:.6f})*({motion})"
    elif animation == "soft-reveal":
        in_progress = (
            ffmpeg_clamped_progress("T", enter_start, enter_start + enter_duration)
            if apply_fade_in
            else "1"
        )
        out_progress = (
            ffmpeg_clamped_progress("T", exit_start, end_sec)
            if apply_fade_out
            else "0"
        )
        progress = f"clip(({in_progress})*(1-({out_progress})),0,1)"
        region_x = max(0, min(out_w - 1, int(round(annotation.x * out_w))))
        region_w = max(2, min(out_w - region_x, int(round(annotation.max_width * out_w))))
        revealed = f"{prepared}reveal"
        filters.append(
            f"[{prepared}]geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':"
            f"a='if(between(X,{region_x},{region_x}+{region_w}*({progress})),alpha(X,Y),0)'"
            f"[{revealed}]"
        )
        prepared = revealed

    filters.append(
        f"[{main_node}][{prepared}]overlay=x='{overlay_x}':y='{overlay_y}':"
        f"shortest=1:eof_action=pass:repeatlast=1:"
        f"enable='between(t,{start_sec:.6f},{end_sec:.6f})'[{output_node}]"
    )
    return filters


def _layout_background_color(default_color: str, scene: CanvasLayoutScene) -> str:
    """Resolve a scene background to an FFmpeg-safe six-digit RGB value."""
    value = str(getattr(scene, "background_color", "") or "").strip().lstrip("#")
    if len(value) == 3:
        value = "".join(char * 2 for char in value)
    if len(value) == 6 and all(char in "0123456789abcdefABCDEF" for char in value):
        return value.lower()
    return default_color


def _layout_scene_expressions(
    scenes: List[CanvasLayoutScene],
    segment: VideoSegment,
    mapper: _SessionMediaMapper | _RawSessionMediaMapper,
    media_start_sec: float,
) -> tuple[CanvasLayoutScene, str, str, str]:
    """Return scene metadata plus FFmpeg expressions for scale/x/y.

    ``t`` is local to the already-trimmed segment.  The mapper converts the
    session transition bounds into that same media-time domain, so VFR legacy
    projects and CFR captures share one expression contract.
    """
    transition_window = canvas_layout_transition_for_range(
        scenes,
        segment.start_ms,
        segment.end_ms,
    )
    if transition_window is not None:
        previous, upcoming, transition_start_ms, transition_end_ms = transition_window
        media_start = mapper.get_media_time(transition_start_ms) / 1000.0
        media_end = mapper.get_media_time(transition_end_ms, for_end=True) / 1000.0
        duration = max(media_end - media_start, 0.001)
        progress = f"clip((t+{media_start_sec:.6f}-{media_start:.6f})/{duration:.6f},0,1)"
        eased = ffmpeg_quintic_ease(progress)

        def lerp(start: float, end: float) -> str:
            return f"({start:.6f}+({end - start:.6f})*({eased}))"

        scene = previous
        return (
            scene,
            lerp(previous.video_scale, upcoming.video_scale),
            lerp(previous.video_x, upcoming.video_x),
            lerp(previous.video_y, upcoming.video_y),
        )

    scene = interpolated_canvas_layout_scene(scenes, segment.start_ms, segment.end_ms)
    return (
        scene,
        f"{scene.video_scale:.6f}",
        f"{scene.video_x:.6f}",
        f"{scene.video_y:.6f}",
    )


def _local_clicks_for_segment(
    click_events: Optional[List[ClickEvent]],
    segment: VideoSegment,
    monitor_rect: Optional[dict] = None,
) -> List[ClickEvent]:
    """Filter absolute click events into a segment-local zero timeline."""
    if monitor_rect:
        left = float(monitor_rect.get("left", monitor_rect.get("x", 0.0)))
        top = float(monitor_rect.get("top", monitor_rect.get("y", 0.0)))
        width = max(float(monitor_rect.get("width", monitor_rect.get("w", 0.0))), 0.0)
        height = max(float(monitor_rect.get("height", monitor_rect.get("h", 0.0))), 0.0)
    else:
        left = top = 0.0
        width = height = 0.0

    def in_recorded_monitor(click: ClickEvent) -> bool:
        if not monitor_rect:
            return True
        return (
            left <= float(click.x) < left + width
            and top <= float(click.y) < top + height
        )

    return [
        replace(click, timestamp=float(click.timestamp) - segment.start_ms)
        for click in (click_events or [])
        if segment.start_ms <= float(click.timestamp) < segment.end_ms
        and in_recorded_monitor(click)
    ]


def _local_mouse_track_for_segment(
    mouse_track: Optional[List[MousePosition]],
    segment: VideoSegment,
    monitor_rect: Optional[dict] = None,
) -> List[MousePosition]:
    """Filter absolute mouse telemetry into one segment-local timeline."""
    if monitor_rect:
        left = float(monitor_rect.get("left", monitor_rect.get("x", 0.0)))
        top = float(monitor_rect.get("top", monitor_rect.get("y", 0.0)))
        width = max(float(monitor_rect.get("width", monitor_rect.get("w", 0.0))), 0.0)
        height = max(float(monitor_rect.get("height", monitor_rect.get("h", 0.0))), 0.0)
    else:
        left = top = 0.0
        width = height = 0.0

    def in_recorded_monitor(point: MousePosition) -> bool:
        if not monitor_rect:
            return True
        return (
            left <= float(point.x) < left + width
            and top <= float(point.y) < top + height
        )

    return [
        replace(point, timestamp=float(point.timestamp) - float(segment.start_ms))
        for point in (mouse_track or [])
        if float(segment.start_ms) <= float(point.timestamp) < float(segment.end_ms)
        and in_recorded_monitor(point)
    ]


def _media_cursor_points_for_segment(
    mouse_track: Optional[List[MousePosition]],
    segment: VideoSegment,
    mapper: _SessionMediaMapper | _RawSessionMediaMapper,
    media_start_sec: float,
    media_keyframes: List[ZoomKeyframe],
    monitor_rect: Optional[dict],
    src_w: int,
    src_h: int,
    scr_w: int,
    scr_h: int,
    *,
    max_points: int | None = None,
) -> list[tuple]:
    """Map continuous mouse samples into segment-local Video Space pixels."""
    local_points = _local_mouse_track_for_segment(mouse_track, segment, monitor_rect)
    if not local_points:
        return []

    if monitor_rect:
        monitor_left = float(monitor_rect.get("left", monitor_rect.get("x", 0.0)))
        monitor_top = float(monitor_rect.get("top", monitor_rect.get("y", 0.0)))
        monitor_w = max(float(monitor_rect.get("width", monitor_rect.get("w", src_w))), 1.0)
        monitor_h = max(float(monitor_rect.get("height", monitor_rect.get("h", src_h))), 1.0)
    else:
        monitor_left = monitor_top = 0.0
        monitor_w = max(float(src_w), 1.0)
        monitor_h = max(float(src_h), 1.0)

    mapped: list[tuple] = []
    for point in local_points:
        absolute_ms = float(segment.start_ms) + float(point.timestamp)
        media_sec = _media_time_for_segment(
            absolute_ms,
            segment,
            mapper,
            media_start_sec,
        )
        rel_x = (float(point.x) - monitor_left) / monitor_w
        rel_y = (float(point.y) - monitor_top) / monitor_h
        if not VideoSpaceTransform.contains_source_point(rel_x, rel_y):
            continue
        zoomed_x, zoomed_y = _map_zoomed_relative_point(
            rel_x,
            rel_y,
            media_sec * 1000.0,
            media_keyframes,
        )
        mapped.append(
            (
                media_sec,
                zoomed_x * float(scr_w),
                zoomed_y * float(scr_h),
                point.click_state,
                point.resume_boundary,
            )
        )

    deduped: dict[float, tuple] = {}
    for point in mapped:
        deduped[round(point[0], 6)] = point
    ordered = [deduped[key] for key in sorted(deduped)]
    if max_points is None or len(ordered) <= max_points:
        return ordered
    indices = {
        int(round(index * (len(ordered) - 1) / max(max_points - 1, 1)))
        for index in range(max_points)
    }
    indices.update(
        index
        for index, point in enumerate(ordered)
        if len(point) > 4 and bool(point[4])
    )
    return [point for index, point in enumerate(ordered) if index in indices]


def _cursor_anchor_for_export(
    *,
    cursor_asset_path: str,
    cursor_style_id: str,
    cursor_hotspot: tuple[float, float] | None,
    cursor_scale: float,
) -> tuple[float, float]:
    """Resolve the same cursor hotspot contract used by preview and export."""
    if _is_valid_cursor_asset(cursor_asset_path):
        base_hotspot = cursor_hotspot or (0.0, 0.0)
        if str(cursor_asset_path).lower().endswith(".svg"):
            return (
                float(base_hotspot[0]) * cursor_scale,
                float(base_hotspot[1]) * cursor_scale,
            )
        # generate_cursor_png pads custom content by four pixels.
        return (
            (float(base_hotspot[0]) + 4.0) * cursor_scale,
            (float(base_hotspot[1]) + 4.0) * cursor_scale,
        )

    effective_style_id = "arrow" if cursor_style_id == "custom" else cursor_style_id
    base_hotspot = cursor_hotspot or registry_cursor_hotspot(effective_style_id)
    asset_scale = registry_cursor_asset_scale(effective_style_id)
    return (
        float(base_hotspot[0]) * asset_scale * cursor_scale,
        float(base_hotspot[1]) * asset_scale * cursor_scale,
    )


def _media_highlights_for_segment(
    highlights: Optional[List[HighlightBox]],
    segment: VideoSegment,
    mapper: _SessionMediaMapper | _RawSessionMediaMapper,
    media_start_sec: float,
) -> List[HighlightBox]:
    """Filter highlights into a segment-local mapped media timeline."""
    local: List[HighlightBox] = []
    for highlight in highlights or []:
        start = max(float(highlight.start_ms), float(segment.start_ms))
        end = min(float(highlight.end_ms), float(segment.end_ms))
        if end <= start:
            continue
        media_start_ms = _media_time_for_segment(start, segment, mapper, media_start_sec) * 1000.0
        media_end_ms = _media_time_for_segment(end, segment, mapper, media_start_sec, for_end=True) * 1000.0
        if media_end_ms <= media_start_ms:
            continue
        local.append(
            replace(
                highlight,
                start_ms=media_start_ms,
                end_ms=media_end_ms,
            )
        )
    return local


def _media_text_annotations_for_segment(
    annotations: Optional[List[TextAnnotation]],
    segment: VideoSegment,
    mapper: _SessionMediaMapper | _RawSessionMediaMapper,
    media_start_sec: float,
) -> List[TextAnnotation]:
    """Map Canvas Space text timing into one segment-local media timeline."""
    local: List[TextAnnotation] = []
    for annotation in annotations or []:
        start = max(float(annotation.start_ms), float(segment.start_ms))
        end = min(float(annotation.end_ms), float(segment.end_ms))
        if end <= start:
            continue
        animation_start = float(annotation.start_ms) + float(
            annotation.animation_delay_ms
        )
        if (
            str(annotation.animation or "none") != "none"
            and float(segment.end_ms) <= animation_start + 0.5
        ):
            # Text-animation boundaries split the source timeline at the
            # reveal start.  A segment before that boundary must contain no
            # text at all.  Treating "fade not active in this segment" as a
            # completed fade made Explainer copy appear while the video was
            # still moving into place.
            continue
        local_start_ms = _media_time_for_segment(
            start, segment, mapper, media_start_sec
        ) * 1000.0
        local_end_ms = _media_time_for_segment(
            end, segment, mapper, media_start_sec, for_end=True
        ) * 1000.0
        if local_end_ms <= local_start_ms:
            continue
        animation_end = animation_start + float(annotation.animation_in_ms)
        exit_start = float(annotation.end_ms) - float(annotation.animation_out_ms)
        local_animation_start_ms = _media_time_for_segment(
            animation_start, segment, mapper, media_start_sec
        ) * 1000.0
        local_animation_end_ms = _media_time_for_segment(
            animation_end, segment, mapper, media_start_sec, for_end=True
        ) * 1000.0
        local_exit_start_ms = _media_time_for_segment(
            exit_start, segment, mapper, media_start_sec
        ) * 1000.0
        mapped = replace(
            annotation,
            start_ms=local_start_ms,
            end_ms=local_end_ms,
            animation_delay_ms=max(
                0.0, local_animation_start_ms - local_start_ms
            ),
            animation_in_ms=max(
                1.0, local_animation_end_ms - local_animation_start_ms
            ),
            animation_out_ms=max(
                1.0, local_end_ms - local_exit_start_ms
            ),
        )
        setattr(
            mapped,
            "_apply_fade_in",
            float(segment.start_ms) < animation_end - 0.5
            and float(segment.end_ms) > animation_start + 0.5,
        )
        setattr(
            mapped,
            "_apply_fade_out",
            float(segment.start_ms) < float(annotation.end_ms) - 0.5
            and float(segment.end_ms) > exit_start + 0.5,
        )
        local.append(mapped)
    return local


def _ease_in_out(progress: float) -> float:
    """Canonical quintic smoothstep shared with the editor preview."""
    return ease_in_out_quint(progress)


def _zoom_state_at_time(keyframes: List[ZoomKeyframe], time_ms: float) -> tuple[float, float, float]:
    """Compute local zoom/pan state for overlay coordinate mapping."""
    sorted_kfs = sorted(keyframes, key=lambda k: float(k.timestamp))
    active_idx = -1
    for idx in range(len(sorted_kfs) - 1, -1, -1):
        if time_ms >= float(sorted_kfs[idx].timestamp):
            active_idx = idx
            break
    if active_idx < 0:
        return 1.0, 0.5, 0.5

    active = sorted_kfs[active_idx]
    duration = max(float(active.duration), 0.0)
    elapsed = max(0.0, float(time_ms) - float(active.timestamp))
    progress = elapsed / duration if duration > 0 else 1.0
    eased = _ease_in_out(progress)

    prev_zoom = float(sorted_kfs[active_idx - 1].zoom) if active_idx > 0 else 1.0
    prev_x = float(sorted_kfs[active_idx - 1].x) if active_idx > 0 else 0.5
    prev_y = float(sorted_kfs[active_idx - 1].y) if active_idx > 0 else 0.5

    zoom = prev_zoom + (float(active.zoom) - prev_zoom) * eased
    pan_x = prev_x + (float(active.x) - prev_x) * eased
    pan_y = prev_y + (float(active.y) - prev_y) * eased
    return max(1.0, zoom), max(0.0, min(1.0, pan_x)), max(0.0, min(1.0, pan_y))


def _map_zoomed_relative_point(
    rel_x: float,
    rel_y: float,
    local_time_ms: float,
    keyframes: List[ZoomKeyframe],
) -> tuple[float, float]:
    zoom, pan_x, pan_y = _zoom_state_at_time(keyframes, local_time_ms)
    return VideoSpaceTransform(zoom=zoom, pan_x=pan_x, pan_y=pan_y).map_point(rel_x, rel_y)


def _click_point_for_export(
    click: ClickEvent,
) -> tuple[float, float]:
    """Return the low-level hook coordinate captured at the click instant."""
    return float(click.x), float(click.y)


def _log_click_coordinate_audit(
    *,
    overlay_kind: str,
    segment_index: int,
    click_index: int,
    click: ClickEvent,
    abs_click_ms: float,
    media_time_sec: float,
    resolved_x: float,
    resolved_y: float,
    monitor_rect: Optional[dict],
    monitor_left: float,
    monitor_top: float,
    monitor_width: float,
    monitor_height: float,
    rel_x_before_zoom: float,
    rel_y_before_zoom: float,
    rel_x_after_zoom: float,
    rel_y_after_zoom: float,
    scr_x: int,
    scr_y: int,
    scr_w: int,
    scr_h: int,
    anchor_x: int,
    anchor_y: int,
    final_x: int,
    final_y: int,
    current_comp_node: str,
    timed_node: str,
    next_node: str,
) -> None:
    """Log current click coordinate math without changing export behavior."""
    if not logger.isEnabledFor(logging.DEBUG):
        return

    logger.debug(
        "\n"
        "=== Zumly Click Coordinate Audit ===\n"
        "overlay_kind      : %s\n"
        "segment/click     : segment=%d click=%d\n"
        "timeline          : absolute_ms=%.3f media_local_sec=%.6f local_click_ms=%.3f\n"
        "\n"
        "Original Data\n"
        "  ClickEvent raw physical px      : x=%.3f y=%.3f\n"
        "  Resolved export point physical  : x=%.3f y=%.3f\n"
        "\n"
        "Monitor Normalization\n"
        "  monitorRect                     : left=%.3f top=%.3f width=%.3f height=%.3f raw=%s\n"
        "  relative before zoompan crop    : x=(%.3f - %.3f) / %.3f = %.6f\n"
        "                                    y=(%.3f - %.3f) / %.3f = %.6f\n"
        "  relative after zoompan crop     : x=%.6f y=%.6f\n"
        "\n"
        "Final Node Math\n"
        "  Video Space overlay rect         : x=%d y=%d width=%d height=%d\n"
        "  overlay x equation              : int(%d + %.6f * %d - %d) = %d\n"
        "  overlay y equation              : int(%d + %.6f * %d - %d) = %d\n"
        "  ffmpeg overlay node             : [%s][%s]overlay=x=%d:y=%d:eof_action=pass:repeatlast=0[%s]\n"
        "=== End Zumly Click Coordinate Audit ===",
        overlay_kind,
        segment_index,
        click_index,
        abs_click_ms,
        media_time_sec,
        float(click.timestamp),
        float(click.x),
        float(click.y),
        resolved_x,
        resolved_y,
        monitor_left,
        monitor_top,
        monitor_width,
        monitor_height,
        monitor_rect,
        resolved_x,
        monitor_left,
        monitor_width,
        rel_x_before_zoom,
        resolved_y,
        monitor_top,
        monitor_height,
        rel_y_before_zoom,
        rel_x_after_zoom,
        rel_y_after_zoom,
        scr_x,
        scr_y,
        scr_w,
        scr_h,
        scr_x,
        rel_x_after_zoom,
        scr_w,
        anchor_x,
        final_x,
        scr_y,
        rel_y_after_zoom,
        scr_h,
        anchor_y,
        final_y,
        current_comp_node,
        timed_node,
        final_x,
        final_y,
        next_node,
    )


def _build_zoompan_filter(keyframes: List[ZoomKeyframe], fps: float) -> str:
    """Build a linear-size FFmpeg zoompan expression for a local timeline.

    Each keyframe owns one mutually exclusive time interval.  Its transition
    starts at the previous keyframe's target state, matching ZoomEngine's
    preview semantics without embedding prior FFmpeg expressions recursively.
    """
    native_fps = max(float(fps or 0.0), 1.0)
    ordered = sorted(keyframes, key=lambda k: float(k.timestamp))
    if not ordered:
        return f"zoompan=z='1':x='0':y='0':d=1:fps={native_fps}"

    starts = [max(float(kf.timestamp) / 1000.0, 0.0) for kf in ordered]

    def _smoothstep_expression(start_sec: float, duration_sec: float) -> str:
        progress = f"clip((time - {start_sec:.6f})/{duration_sec:.6f},0,1)"
        return (
            f"(6*pow({progress},5)-15*pow({progress},4)+10*pow({progress},3))"
        )

    def _property_expression(default: float, attribute: str, minimum: Optional[float] = None) -> str:
        targets = [float(getattr(kf, attribute)) for kf in ordered]
        if minimum is not None:
            targets = [max(minimum, value) for value in targets]

        # Before the first keyframe the camera uses the default state. Every
        # subsequent term has a disjoint interval, so this remains O(n).
        terms = [f"if(lt(time,{starts[0]:.6f}),{default:.6f},0)"]
        for index, keyframe in enumerate(ordered):
            start_sec = starts[index]
            duration_sec = max(float(keyframe.duration) / 1000.0, 0.0)
            end_sec = start_sec + duration_sec
            previous = default if index == 0 else targets[index - 1]
            target = targets[index]

            if duration_sec > 0:
                eased = _smoothstep_expression(start_sec, duration_sec)
                local_value = (
                    f"if(lt(time,{end_sec:.6f}),"
                    f"{previous:.6f}+({target:.6f}-{previous:.6f})*{eased},"
                    f"{target:.6f})"
                )
            else:
                local_value = f"{target:.6f}"

            if index + 1 < len(ordered):
                next_start = starts[index + 1]
                interval = f"gte(time,{start_sec:.6f})*lt(time,{next_start:.6f})"
            else:
                interval = f"gte(time,{start_sec:.6f})"
            terms.append(f"if({interval},{local_value},0)")

        return "(" + "+".join(terms) + ")"

    z_var = _property_expression(1.0, "zoom", minimum=1.0)
    pan_x = _property_expression(0.5, "x")
    pan_y = _property_expression(0.5, "y")
    zoompan_x = f"clip(({pan_x}) * iw - (iw/{z_var})/2, 0, iw - iw/{z_var})"
    zoompan_y = f"clip(({pan_y}) * ih - (ih/{z_var})/2, 0, ih - ih/{z_var})"
    return f"zoompan=z='{z_var}':x='{zoompan_x}':y='{zoompan_y}':d=1:fps={native_fps}"


def _build_export_assets(
    *,
    bg_preset: BackgroundPreset,
    frame_preset: FramePreset,
    out_w: int,
    out_h: int,
    src_w: int,
    src_h: int,
    geom: dict,
    click_preset: Optional[ClickEffectPreset],
    local_click_count: int,
    local_cursor_count: int,
    cursor_asset_path: str = "",
    cursor_style_id: str = "arrow",
    local_highlight_sets: list[list[HighlightBox]],
    local_video_annotation_sets: list[list[TimelineOverlay]],
    local_text_annotation_sets: list[list[TextAnnotation]],
    ordered_frames: list[TimelineFrame],
) -> ExportAssetBundle:
    """Generate static FFmpeg inputs without mixing them into graph assembly."""
    temp_files: List[str] = []
    frame_img_path = generate_device_frame_png(frame_preset, out_w, out_h, geom)
    temp_files.append(frame_img_path)

    click_img_path = None
    if (
        local_click_count > 0
        and click_preset
        and click_preset.duration_ms > 0
        and click_preset.color[3] > 0
    ):
        click_img_path = generate_click_png(click_preset)
        temp_files.append(click_img_path)

    cursor_img_path = None
    if local_cursor_count > 0:
        cursor_img_path = generate_cursor_png(cursor_asset_path, cursor_style_id)
        temp_files.append(cursor_img_path)

    annotation_base_input = 2 + (1 if click_img_path else 0) + (1 if cursor_img_path else 0)
    video_annotation_img_paths: List[str] = []
    local_video_annotation_assets: list[list[tuple[TimelineOverlay, int]]] = []
    for local_annotations in local_video_annotation_sets:
        rows: list[tuple[TimelineOverlay, int]] = []
        for annotation in local_annotations:
            image = render_video_annotation_overlay([annotation], src_w, src_h)
            path = _new_temp_asset_path(".png")
            if not image.save(path, "PNG"):
                raise RuntimeError("Could not create annotation export asset")
            temp_files.append(path)
            rows.append((annotation, annotation_base_input + len(video_annotation_img_paths)))
            video_annotation_img_paths.append(path)
        local_video_annotation_assets.append(rows)

    highlight_base_input = annotation_base_input + len(video_annotation_img_paths)
    highlight_img_paths: List[str] = []
    local_highlight_assets: list[list[tuple[HighlightBox, int]]] = []
    for local_highlights in local_highlight_sets:
        asset_rows: list[tuple[HighlightBox, int]] = []
        for highlight in local_highlights:
            # Dimming is generated directly in FFmpeg. Only allocate a PNG
            # input when a project explicitly requests a visible border.
            if max(0, int(getattr(highlight, "border_width", 0))) <= 0:
                continue
            # Highlight geometry belongs to Video Space.  Rasterize optional
            # borders at the internal screen resolution so the complete
            # highlight layer follows layout and transition transforms with
            # the captured video, rather than covering the outer canvas.
            screen_w = max(2, int(geom["scr_w"]))
            screen_h = max(2, int(geom["scr_h"]))
            screen_geom = {
                "scr_x": 0,
                "scr_y": 0,
                "scr_w": screen_w,
                "scr_h": screen_h,
            }
            highlight_path = generate_highlight_png(
                highlight,
                screen_w,
                screen_h,
                screen_geom,
            )
            temp_files.append(highlight_path)
            asset_rows.append((highlight, highlight_base_input + len(highlight_img_paths)))
            highlight_img_paths.append(highlight_path)
        local_highlight_assets.append(asset_rows)

    text_base_input = highlight_base_input + len(highlight_img_paths)
    text_annotation_img_paths: List[str] = []
    local_text_annotation_assets: list[list[tuple[TextAnnotation, int]]] = []
    for local_annotations in local_text_annotation_sets:
        rows: list[tuple[TextAnnotation, int]] = []
        for annotation in local_annotations:
            if not str(annotation.text or "") or float(annotation.opacity) <= 0.0:
                continue
            text_path = generate_text_annotation_png(annotation, out_w, out_h)
            temp_files.append(text_path)
            input_index = text_base_input + len(text_annotation_img_paths)
            text_annotation_img_paths.append(text_path)
            rows.append((annotation, input_index))
        local_text_annotation_assets.append(rows)

    frame_base_input = text_base_input + len(text_annotation_img_paths)
    timeline_frame_img_paths: List[str] = []
    for timeline_frame in ordered_frames:
        animated_text = (
            timeline_frame.kind == "text"
            and normalize_text_reveal_effect(
                getattr(timeline_frame, "text_animation", "none")
            ) != "none"
        )
        frame_path = generate_timeline_frame_png(
            timeline_frame,
            out_w,
            out_h,
            text_only=animated_text,
        )
        temp_files.append(frame_path)
        timeline_frame_img_paths.append(frame_path)

    background_img_path = generate_background_png(bg_preset, out_w, out_h)
    temp_files.append(background_img_path)
    background_input_index = frame_base_input + len(timeline_frame_img_paths)

    return ExportAssetBundle(
        background_img_path=background_img_path,
        background_input_index=background_input_index,
        frame_img_path=frame_img_path,
        click_img_path=click_img_path,
        cursor_img_path=cursor_img_path,
        highlight_img_paths=highlight_img_paths,
        video_annotation_img_paths=video_annotation_img_paths,
        local_video_annotation_assets=local_video_annotation_assets,
        local_highlight_assets=local_highlight_assets,
        text_annotation_img_paths=text_annotation_img_paths,
        local_text_annotation_assets=local_text_annotation_assets,
        timeline_frame_img_paths=timeline_frame_img_paths,
        temp_files=temp_files,
        highlight_base_input=highlight_base_input,
        frame_base_input=frame_base_input,
    )


def _build_export_filtergraph(
    *,
    bg_preset: Optional[BackgroundPreset],
    frame_preset: Optional[FramePreset],
    target_resolution: Optional[tuple[int, int]],
    duration_ms: float,
    frame_timestamps: Optional[List[float]],
    keyframes: List[ZoomKeyframe],
    mouse_track: Optional[List[MousePosition]],
    click_events: Optional[List[ClickEvent]],
    click_preset: Optional[ClickEffectPreset],
    cursor_asset_path: str = "",
    cursor_style_id: str = "arrow",
    monitor_rect: Optional[dict],
    video_segments: Optional[List[VideoSegment]],
    timeline_frames: Optional[List[TimelineFrame]],
    highlights: Optional[List[HighlightBox]],
    screen_transitions: Optional[List[ScreenTransition]] = None,
    text_annotations: Optional[List[TextAnnotation]] = None,
    timeline_overlays: Optional[List[TimelineOverlay]] = None,
    cursor_hotspot: tuple[float, float] | None = None,
    cursor_scale: float = DEFAULT_CURSOR_SCALE,
    src_w: int,
    src_h: int,
    src_fps: float,
    total_sec: float,
    is_cfr: bool = False,
    source_has_audio: bool = False,
    layout_transform: Optional[LayoutSpaceTransform] = None,
    canvas_layout_scenes: Optional[List[CanvasLayoutScene]] = None,
    explainer_scenes: Optional[List[ExplainerScene]] = None,
) -> ExportFilterGraphPlan:
    """Build the FFmpeg filtergraph and static assets without running FFmpeg."""
    out_w, out_h = int(src_w), int(src_h)
    if target_resolution:
        out_w, out_h = target_resolution
    out_w = out_w + (out_w % 2)
    out_h = out_h + (out_h % 2)

    frame_preset = frame_preset or DEFAULT_FRAME
    source_duration_ms = duration_ms or ((frame_timestamps[-1] if frame_timestamps else 0.0) or (total_sec * 1000.0))
    if source_duration_ms > 0:
        keyframes = [kf for kf in keyframes if float(kf.timestamp) <= source_duration_ms]
        if click_events:
            click_events = [ce for ce in click_events if float(ce.timestamp) <= source_duration_ms]

    has_layout_scenes = bool(canvas_layout_scenes)
    base_layout_transform = (
        LayoutSpaceTransform.identity() if has_layout_scenes else layout_transform
    )
    geom = GeometryComputer(
        canvas_w=out_w,
        canvas_h=out_h,
        src_w=src_w,
        src_h=src_h,
        frame_preset=frame_preset,
        layout_transform=base_layout_transform,
    ).compute()
    scr_x = geom["scr_x"]
    scr_y = geom["scr_y"]
    scr_w = geom["scr_w"]
    scr_h = geom["scr_h"]

    bg_color = "000000"
    if bg_preset and hasattr(bg_preset, "color_top") and bg_preset.color_top:
        r, g, b = bg_preset.color_top
        bg_color = f"{r:02x}{g:02x}{b:02x}"

    explicit_segments = video_segments is not None
    segments = _normalize_video_segments(
        video_segments,
        source_duration_ms,
        fill_gaps=not explicit_segments,
    )
    if not segments:
        raise ValueError("Export failed: unknown source duration")
    # Keep equal-timestamp cards in creation order. Random UUID ordering made
    # same-anchor text/image cards swap unpredictably between saves.
    ordered_frames = sorted(
        timeline_frames or [],
        key=lambda frame: float(frame.timestamp_ms),
    )
    enabled_transitions = sorted(
        [item for item in (screen_transitions or []) if item.enabled],
        key=lambda item: float(item.timestamp_ms),
    )
    ordered_transitions: list[ScreenTransition] = []
    for transition in enabled_transitions:
        timestamp_ms = float(transition.timestamp_ms)
        has_outgoing_video = any(
            float(segment.start_ms) + 0.5 < timestamp_ms
            <= float(segment.end_ms) + 0.5
            for segment in segments
        )
        has_incoming_video = any(
            float(segment.start_ms) - 0.5 <= timestamp_ms
            < float(segment.end_ms) - 0.5
            for segment in segments
        )
        if has_outgoing_video and has_incoming_video:
            ordered_transitions.append(transition)
        else:
            logger.warning(
                "Skipping screen transition %s because its timestamp %.3f ms "
                "does not have video on both sides of the edited timeline",
                transition.id,
                timestamp_ms,
            )
    ordered_explainers = sorted(
        list(explainer_scenes or []),
        key=lambda scene: (float(scene.start_ms), scene.id),
    )
    # Preview and export must use the same output-time contract. Keep this
    # mapper based on the unsplit user clips; later visual splits preserve
    # duration but should not create a second timeline calculation.
    output_timeline = EditedTimelineMapper(
        source_duration_ms,
        list(segments),
        ordered_frames,
        ordered_transitions,
        ordered_explainers,
    )
    explainer_visual_sources = {
        scene.id: output_timeline.synthetic_visual_source_ms(
            scene.id,
            float(scene.start_ms),
        )
        for scene in ordered_explainers
    }
    # Only Video Space masks participate in source-video rendering. Canvas
    # overlays remain a future annotation concern and must not be accidentally
    # transformed through this path.
    static_video_masks = video_space_masks(timeline_overlays)
    static_video_annotations = video_space_annotations(timeline_overlays)
    segments = _split_segments_at_timeline_frames(segments, ordered_frames)
    segments = _split_segments_at_screen_transitions(segments, ordered_transitions)
    segments = _split_segments_at_timeline_frames(
        segments,
        [
            TimelineFrame(
                id=scene.id,
                timestamp_ms=scene.start_ms,
                duration_ms=max(scene.end_ms - scene.start_ms, 1.0),
            )
            for scene in ordered_explainers
        ],
    )
    segments = _split_segments_at_layout_scenes(segments, canvas_layout_scenes)
    segments = _split_segments_at_text_animations(segments, text_annotations)

    media_mapper = _select_session_media_mapper(
        frame_timestamps,
        source_duration_ms,
        total_sec,
        src_fps,
        is_cfr=is_cfr,
    )
    segment_media_bounds = [
        media_mapper.segment_bounds(segment)
        for segment in segments
    ]


    output_total_sec = output_timeline.output_duration_ms / 1000.0
    has_speed_changes = any(abs(_segment_speed(segment) - 1.0) > 0.01 for segment in segments)
    has_timeline_edits = bool(ordered_frames or ordered_transitions or ordered_explainers) or (
        explicit_segments
        and (
            len(segments) != 1
            or abs(segments[0].start_ms) > 0.5
            or abs(segments[0].end_ms - source_duration_ms) > 0.5
        )
    )
    local_click_sets = [
        _local_clicks_for_segment(click_events, segment, monitor_rect)
        for segment in segments
    ]
    local_mouse_sets = [
        _local_mouse_track_for_segment(mouse_track, segment, monitor_rect)
        for segment in segments
    ]
    local_highlight_sets = [
        _media_highlights_for_segment(highlights, segment, media_mapper, segment_media_bounds[idx][0])
        for idx, segment in enumerate(segments)
    ]
    local_video_annotation_sets = [
        [
            item for item in static_video_annotations
            if (not item.clip_id or item.clip_id == segment.id)
            and float(item.end_ms) > float(segment.start_ms)
            and float(item.start_ms) < float(segment.end_ms)
        ]
        for segment in segments
    ]
    local_text_annotation_sets = [
        _media_text_annotations_for_segment(
            text_annotations, segment, media_mapper, segment_media_bounds[idx][0]
        )
        for idx, segment in enumerate(segments)
    ]
    media_keyframe_sets = [
        _media_keyframes_for_segment(
            keyframes,
            segment,
            media_mapper,
            segment_media_bounds[idx][0],
        )
        for idx, segment in enumerate(segments)
    ]
    continuous_cursor_points_by_segment = [
        _media_cursor_points_for_segment(
            mouse_track,
            segment,
            media_mapper,
            segment_media_bounds[idx][0],
            media_keyframe_sets[idx],
            monitor_rect,
            src_w,
            src_h,
            scr_w,
            scr_h,
        )
        for idx, segment in enumerate(segments)
    ]
    local_click_count = sum(len(items) for items in local_click_sets)
    # Continuous telemetry uses one compact cursor asset per graph branch and
    # an external sendcmd sidecar. No full-size alpha frame sequence is made.
    fallback_cursor_count = sum(
        len(clicks)
        for mouse, clicks in zip(local_mouse_sets, local_click_sets)
        if not mouse
    )
    continuous_cursor_count = sum(
        1 for points in continuous_cursor_points_by_segment if points
    )
    local_cursor_count = continuous_cursor_count + fallback_cursor_count

    assets = _build_export_assets(
        bg_preset=bg_preset or DEFAULT_PRESET,
        frame_preset=frame_preset,
        out_w=out_w,
        out_h=out_h,
        src_w=src_w,
        src_h=src_h,
        geom=geom,
        click_preset=click_preset,
        local_click_count=local_click_count,
        local_cursor_count=local_cursor_count,
        cursor_asset_path=cursor_asset_path,
        cursor_style_id=cursor_style_id,
        local_highlight_sets=local_highlight_sets,
        local_video_annotation_sets=local_video_annotation_sets,
        local_text_annotation_sets=local_text_annotation_sets,
        ordered_frames=ordered_frames,
    )
    temp_files = assets.temp_files
    background_img_path = assets.background_img_path
    background_input_index = assets.background_input_index
    frame_img_path = assets.frame_img_path
    click_img_path = assets.click_img_path
    cursor_img_path = assets.cursor_img_path
    highlight_img_paths = assets.highlight_img_paths
    video_annotation_img_paths = assets.video_annotation_img_paths
    local_video_annotation_assets = assets.local_video_annotation_assets
    local_highlight_assets = assets.local_highlight_assets
    text_annotation_img_paths = assets.text_annotation_img_paths
    local_text_annotation_assets = assets.local_text_annotation_assets
    frame_base_input = assets.frame_base_input
    timeline_frame_img_paths = assets.timeline_frame_img_paths
    cursor_motion_tracks: list[CursorMotionTrack] = []
    text_asset_inputs = {
        annotation.id: input_index
        for rows in local_text_annotation_assets
        for annotation, input_index in rows
    }
    transition_text_img_paths: list[str] = []
    transition_text_assets: dict[int, list[tuple[TextAnnotation, int]]] = {}
    explainer_text_assets: dict[str, tuple[TextAnnotation, int]] = {}
    transition_text_base_input = background_input_index + 1
    for transition_index, transition in enumerate(ordered_transitions):
        rows: list[tuple[TextAnnotation, int]] = []
        for annotation in text_annotations or []:
            if not (
                float(annotation.start_ms) <= float(transition.timestamp_ms)
                <= float(annotation.end_ms)
            ):
                continue
            path = generate_text_annotation_png(annotation, out_w, out_h)
            temp_files.append(path)
            input_index = transition_text_base_input + len(transition_text_img_paths)
            transition_text_img_paths.append(path)
            rows.append((annotation, input_index))
        transition_text_assets[transition_index] = rows
    for scene in ordered_explainers:
        annotation = explainer_text_annotation(scene)
        duration_ms = max(float(scene.end_ms) - float(scene.start_ms), 1.0)
        annotation = replace(
            annotation,
            start_ms=0.0,
            end_ms=duration_ms,
        )
        path = generate_text_annotation_png(annotation, out_w, out_h)
        temp_files.append(path)
        input_index = transition_text_base_input + len(transition_text_img_paths)
        transition_text_img_paths.append(path)
        explainer_text_assets[scene.id] = (annotation, input_index)

    # The cursor bitmap is static. Movement is supplied later through external
    # sendcmd sidecars, so preparation remains O(telemetry points), not
    # O(output frames * canvas pixels).
    if any(continuous_cursor_points_by_segment):
        if not cursor_img_path:
            raise ValueError("Could not prepare cursor asset for telemetry export")
        try:
            with Image.open(cursor_img_path) as source:
                cursor_source = source.convert("RGBA")
        except (OSError, ValueError):
            logger.warning(
                "Cursor asset could not be decoded for telemetry export; using arrow"
            )
            try:
                with Image.open(ensure_cursor_asset("arrow")) as source:
                    cursor_source = source.convert("RGBA")
            except (OSError, ValueError) as fallback_exc:
                raise ValueError("Could not prepare cursor asset for export") from fallback_exc
        desired_cursor_height = max(
            28,
            min(
                144,
                int(
                    round(
                        out_h * 0.035 * normalize_cursor_scale(cursor_scale)
                    )
                ),
            ),
        )
        cursor_render_scale = desired_cursor_height / max(float(cursor_source.height), 1.0)
        cursor_anchor_x, cursor_anchor_y = _cursor_anchor_for_export(
            cursor_asset_path=cursor_asset_path,
            cursor_style_id=cursor_style_id,
            cursor_hotspot=cursor_hotspot,
            cursor_scale=cursor_render_scale,
        )
        for seg_index, points in enumerate(continuous_cursor_points_by_segment):
            if not points:
                continue
            track = _generate_cursor_motion_track(
                points=points,
                anchor_x=cursor_anchor_x,
                anchor_y=cursor_anchor_y,
                segment_index=seg_index,
                node_index=len(cursor_motion_tracks),
            )
            cursor_motion_tracks.append(track)
            temp_files.append(track.command_path)

    filter_lines: List[str] = []

    frame_branch_count = len(segments) + len(ordered_transitions) + len(ordered_explainers)
    if frame_branch_count > 1:
        frame_nodes = "".join(f"[fr{i}]" for i in range(len(segments)))
        transition_frame_nodes = "".join(
            f"[stfr{i}]" for i in range(len(ordered_transitions))
        )
        explainer_frame_nodes = "".join(
            f"[exfr{i}]" for i in range(len(ordered_explainers))
        )
        filter_lines.append(
            f"[1:v]split={frame_branch_count}{frame_nodes}"
            f"{transition_frame_nodes}{explainer_frame_nodes}"
        )
    else:
        filter_lines.append("[1:v]null[fr0]")

    if not has_layout_scenes:
        filter_lines.append(
            f"[{background_input_index}:v]scale={out_w}:{out_h}:flags=lanczos,"
            f"fps={src_fps},format=rgba[backgroundsource]"
        )
        background_branch_count = (
            len(segments) + len(ordered_transitions) + len(ordered_explainers)
        )
        if background_branch_count > 1:
            background_nodes = "".join(
                f"[background{i}]" for i in range(len(segments))
            )
            transition_background_nodes = "".join(
                f"[stbackground{i}]" for i in range(len(ordered_transitions))
            )
            explainer_background_nodes = "".join(
                f"[exbackground{i}]" for i in range(len(ordered_explainers))
            )
            filter_lines.append(
                f"[backgroundsource]split={background_branch_count}"
                f"{background_nodes}{transition_background_nodes}"
                f"{explainer_background_nodes}"
            )
        else:
            filter_lines.append("[backgroundsource]null[background0]")

    click_node_index = 0
    if click_img_path and local_click_count > 1:
        click_nodes = "".join(f"[cl{i}]" for i in range(local_click_count))
        filter_lines.append(f"[2:v]split={local_click_count}{click_nodes}")
    elif click_img_path and local_click_count == 1:
        filter_lines.append("[2:v]null[cl0]")

    cursor_input_index = 2 + (1 if click_img_path else 0)
    render_cursor_scale = 1.0
    cursor_source_node = f"[{cursor_input_index}:v]"
    if cursor_img_path:
        try:
            with Image.open(cursor_img_path) as cursor_source:
                source_height = max(int(cursor_source.height), 1)
        except (OSError, ValueError):
            fallback_style_id = "arrow" if cursor_style_id == "custom" else cursor_style_id
            fallback_preset = get_cursor_preset(fallback_style_id)
            source_height = max(
                int(round(fallback_preset.height * registry_cursor_asset_scale(fallback_style_id))),
                1,
            )
        user_cursor_scale = normalize_cursor_scale(cursor_scale)
        desired_height = max(
            28,
            min(144, int(round(out_h * 0.035 * user_cursor_scale))),
        )
        render_cursor_scale = desired_height / float(source_height)
        cursor_source_node = "[cursorasset]"
        filter_lines.append(
            f"[{cursor_input_index}:v]scale=-1:{desired_height}:flags=lanczos{cursor_source_node}"
        )
    if cursor_img_path and local_cursor_count > 1:
        cursor_nodes = "".join(f"[cu{i}]" for i in range(local_cursor_count))
        filter_lines.append(f"{cursor_source_node}split={local_cursor_count}{cursor_nodes}")
    elif cursor_img_path and local_cursor_count == 1:
        filter_lines.append(f"{cursor_source_node}null[cu0]")

    cursor_motion_by_segment = {
        track.segment_index: track for track in cursor_motion_tracks
    }

    output_items: list[tuple[float, int, str, str]] = []
    transition_endpoint_nodes: dict[
        tuple[str, str], tuple[str, float, float]
    ] = {}
    cursor_node_index = len(cursor_motion_tracks)
    frame_node_index = 0
    appended_frame_ids: set[str] = set()
    appended_transition_ids: set[str] = set()

    def append_timeline_frame_output(frame_idx: int, timeline_frame: TimelineFrame) -> None:
        nonlocal frame_node_index
        if timeline_frame.id in appended_frame_ids:
            return
        input_index = frame_base_input + frame_idx
        frame_out = f"tf{frame_node_index}out"
        frame_duration = max(float(timeline_frame.duration_ms), 250.0) / 1000.0
        text_animation = normalize_text_reveal_effect(
            getattr(timeline_frame, "text_animation", "none")
        )
        if timeline_frame.kind == "text" and text_animation != "none":
            frame_asset = f"tf{frame_node_index}asset"
            frame_base = f"tf{frame_node_index}base"
            bg = _parse_hex_color(timeline_frame.background_color, (17, 24, 39))
            bg_hex = f"0x{bg[0]:02x}{bg[1]:02x}{bg[2]:02x}"
            filter_lines.append(
                f"[{input_index}:v]scale={out_w}:{out_h}:flags=lanczos,format=rgba,"
                f"fps={src_fps},trim=duration={frame_duration:.6f},"
                f"setpts=PTS-STARTPTS[{frame_asset}]"
            )
            filter_lines.append(
                f"color=c={bg_hex}:s={out_w}x{out_h}:r={src_fps}:"
                f"d={frame_duration:.6f},format=rgba,setpts=PTS-STARTPTS[{frame_base}]"
            )
            reveal_duration_ms = bounded_text_reveal_duration_ms(
                timeline_frame.duration_ms,
                getattr(timeline_frame, "text_animation_duration_ms", 700.0),
            )
            animation = TextAnnotation.create(
                0.0,
                timeline_frame.duration_ms,
                x=0.0,
                y=0.0,
                text="Text Frame",
                max_width=1.0,
                animation=text_animation,
                animation_in_ms=reveal_duration_ms,
                animation_out_ms=1.0,
                slide_offset_y=TEXT_REVEAL_SLIDE_Y,
            )
            setattr(animation, "_apply_fade_out", False)
            filter_lines.extend(
                _animated_text_overlay_filters(
                    frame_base,
                    frame_asset,
                    frame_out,
                    animation,
                    start_sec=0.0,
                    end_sec=frame_duration,
                    out_w=out_w,
                    out_h=out_h,
                )
            )
        else:
            filter_lines.append(
                f"[{input_index}:v]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,"
                f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2,"
                f"fps={src_fps},trim=duration={frame_duration:.6f},"
                f"setpts=PTS-STARTPTS[{frame_out}]"
            )
        current_frame_node = frame_out
        for text_index, annotation in enumerate(text_annotations or []):
            if annotation.id not in text_asset_inputs:
                continue
            if not (
                float(annotation.start_ms) <= float(timeline_frame.timestamp_ms)
                <= float(annotation.end_ms)
            ):
                continue
            next_frame_node = f"tf{frame_node_index}text{text_index}"
            filter_lines.append(
                _timed_overlay_filter(
                    current_frame_node,
                    f"{text_asset_inputs[annotation.id]}:v",
                    next_frame_node,
                    start_sec=0.0,
                    end_sec=frame_duration,
                    end_with_main=True,
                )
            )
            current_frame_node = next_frame_node
        frame_audio_node = ""
        if source_has_audio:
            frame_audio_node = f"tf{frame_node_index}aout"
            filter_lines.append(
                "anullsrc=r=48000:cl=stereo,"
                f"atrim=duration={frame_duration:.6f},"
                f"asetpts=PTS-STARTPTS[{frame_audio_node}]"
            )
        output_items.append(
            (
                float(timeline_frame.timestamp_ms),
                10,
                current_frame_node,
                frame_audio_node,
            )
        )
        appended_frame_ids.add(timeline_frame.id)
        frame_node_index += 1

    def append_screen_transition_output(
        transition_index: int, transition: ScreenTransition
    ) -> None:
        if transition.id in appended_transition_ids:
            return
        duration_sec = max(float(transition.duration_ms), 150.0) / 1000.0
        frame_sec = 1.0 / max(src_fps, 1.0)
        outgoing = transition_endpoint_nodes.get((transition.id, "outgoing"))
        incoming = transition_endpoint_nodes.get((transition.id, "incoming"))
        if outgoing is None or incoming is None:
            logger.warning(
                "Skipping screen transition %s because a processed endpoint is missing",
                transition.id,
            )
            return
        outgoing_node, outgoing_duration, outgoing_sample = outgoing
        incoming_node, incoming_duration, incoming_sample = incoming
        transition_w = max(2, int(scr_w) - (int(scr_w) % 2))
        transition_h = max(2, int(scr_h) - (int(scr_h) % 2))
        transition_x = int(scr_x) + max(0, (int(scr_w) - transition_w) // 2)
        transition_y = int(scr_y) + max(0, (int(scr_h) - transition_h) // 2)
        old_still_node = f"st{transition_index}oldstill"
        new_still_node = f"st{transition_index}newstill"
        old_node = f"st{transition_index}old"
        new_node = f"st{transition_index}new"
        video_out_node = f"st{transition_index}video"
        out_node = f"st{transition_index}out"
        outgoing_window_start = max(0.0, outgoing_sample - frame_sec)
        outgoing_window_end = min(
            outgoing_duration, max(outgoing_sample + frame_sec, frame_sec)
        )
        incoming_window_start = max(0.0, incoming_sample - (frame_sec / 2.0))
        incoming_window_end = min(
            incoming_duration, max(incoming_sample + frame_sec, frame_sec)
        )
        filter_lines.append(
            f"[{outgoing_node}]trim=start={outgoing_window_start:.6f}:"
            f"end={outgoing_window_end:.6f},select='eq(n,0)',"
            f"setpts=PTS-STARTPTS,scale={transition_w}:{transition_h}:flags=lanczos,"
            f"format=rgba[{old_still_node}]"
        )
        filter_lines.append(
            f"[{incoming_node}]trim=start={incoming_window_start:.6f}:"
            f"end={incoming_window_end:.6f},select='eq(n,0)',"
            f"setpts=PTS-STARTPTS,scale={transition_w}:{transition_h}:flags=lanczos,"
            f"format=rgba[{new_still_node}]"
        )
        filter_lines.append(
            f"[{old_still_node}]loop=loop=-1:size=1:start=0,settb=AVTB,"
            f"setpts=N/({src_fps}*TB),trim=duration={duration_sec:.6f},"
            f"format=rgba[{old_node}]"
        )
        filter_lines.append(
            f"[{new_still_node}]loop=loop=-1:size=1:start=0,settb=AVTB,"
            f"setpts=N/({src_fps}*TB),trim=duration={duration_sec:.6f},"
            f"format=rgba[{new_node}]"
        )
        if is_graphic_transition(transition.effect_type):
            transition_layout_scene = canvas_layout_scene_at(
                list(canvas_layout_scenes or []),
                float(transition.timestamp_ms),
                source_duration_ms,
            )
            transition_bg_color = _layout_background_color(
                bg_color, transition_layout_scene
            )
            bar_description = graphic_transition_description(transition)
            bar_expressions = ffmpeg_graphic_transition_expressions(
                transition,
                duration_sec,
                variable="t",
            )
            graphic_base_node = f"st{transition_index}graphicbase"
            filter_lines.append(
                f"[{old_node}][{new_node}]overlay=x=0:y=0:eval=frame:"
                f"enable='gte(t,{duration_sec * 0.5:.6f})':"
                f"shortest=1:eof_action=pass[{graphic_base_node}]"
            )
            current_stage = graphic_base_node
            for bar_index, bar in enumerate(bar_expressions.bars):
                covered = bar_description.bars[bar_index].covered_rect
                material_width = max(2, math.ceil(transition_w * covered[2]) + 2)
                material_height = max(2, math.ceil(transition_h * covered[3]) + 2)
                bar_canvas_node = f"st{transition_index}bar{bar_index}canvas"
                bar_scaled_node = f"st{transition_index}bar{bar_index}scaled"
                next_stage = f"st{transition_index}bar{bar_index}stage"

                # Preview and export use the same generated material artwork at
                # the bar's covered aspect ratio, preserving sharp surface grain.
                material_path = _new_temp_asset_path(".png")
                Path(material_path).write_bytes(
                    graphic_bar_material_png(
                        bar.color_start,
                        bar.color_end,
                        bar.style,
                        bar.grain_seed,
                        bar.edge_shading,
                        width=material_width,
                        height=material_height,
                    )
                )
                temp_files.append(material_path)
                filter_lines.append(
                    "movie="
                    f"filename='{_ffmpeg_filter_path(material_path)}':loop=1,"
                    "loop=loop=-1:size=1:start=0,settb=AVTB,"
                    f"setpts=N/({src_fps}*TB),"
                    f"trim=duration={duration_sec:.6f},format=rgba"
                    f"[{bar_canvas_node}]"
                )
                filter_lines.append(
                    f"[{bar_canvas_node}]scale="
                    f"w='max(2,ceil({transition_w}*({bar.width}))+2)':"
                    f"h='max(2,ceil({transition_h}*({bar.height}))+2)':"
                    f"eval=frame:flags=lanczos,setsar=1,format=rgba"
                    f"[{bar_scaled_node}]"
                )
                filter_lines.append(
                    f"[{current_stage}][{bar_scaled_node}]overlay="
                    f"x='floor(W*({bar.x}))-1':"
                    f"y='floor(H*({bar.y}))-1':eval=frame:"
                    f"enable='{bar.enable}':shortest=1:eof_action=pass"
                    f"[{next_stage}]"
                )
                current_stage = next_stage
            filter_lines.append(
                f"[{current_stage}]trim=duration={duration_sec:.6f},"
                f"setpts=PTS-STARTPTS,format=rgba"
                f"[st{transition_index}videofinal]"
            )
            video_out_node = f"st{transition_index}videofinal"
        else:
            motion_expressions = ffmpeg_transition_scene_expressions(
                transition.effect_type,
                duration_sec,
                direction=transition.direction,
                variable="t",
            )
            transition_layout_scene = canvas_layout_scene_at(
                list(canvas_layout_scenes or []),
                float(transition.timestamp_ms),
                source_duration_ms,
            )
            transition_bg_color = _layout_background_color(
                bg_color, transition_layout_scene
            )
            background_node = f"st{transition_index}background"
            filter_lines.append(
                f"color=c=0x{transition_bg_color}:s={transition_w}x{transition_h}:"
                f"r={src_fps}:d={duration_sec:.6f},format=rgba[{background_node}]"
            )

            def append_transformed_layer(
                source_node: str,
                output_node: str,
                expressions,
            ) -> tuple[str, str]:
                """Scale a timestamped endpoint and return its overlay placement."""
                scaled_node = f"{output_node}scaled"
                filter_lines.append(
                    f"[{source_node}]scale="
                    f"w='max(2,trunc(iw*({expressions.scale_x})/2)*2)':"
                    f"h='max(2,trunc(ih*({expressions.scale_y})/2)*2)':"
                    f"eval=frame:flags=bicubic,setsar=1[{scaled_node}]"
                )
                placement = (
                    f"x='({expressions.anchor_x})*W-({expressions.anchor_x})*w+"
                    f"W*({expressions.offset_x})':"
                    f"y='({expressions.anchor_y})*H-({expressions.anchor_y})*h+"
                    f"H*({expressions.offset_y})'"
                )
                return scaled_node, placement

            transformed = {}
            transformed["incoming"] = append_transformed_layer(
                new_node,
                f"st{transition_index}incoming",
                motion_expressions.incoming,
            )
            transformed["outgoing"] = append_transformed_layer(
                old_node,
                f"st{transition_index}outgoing",
                motion_expressions.outgoing,
            )
            current_stage = background_node
            for layer_position, layer_name in enumerate(motion_expressions.layer_order):
                layer_node, placement = transformed[layer_name]
                next_stage = (
                    video_out_node
                    if layer_position == len(motion_expressions.layer_order) - 1
                    else f"st{transition_index}stage{layer_position}"
                )
                filter_lines.append(
                    f"[{current_stage}][{layer_node}]overlay={placement}:"
                    f"eval=frame:shortest=1:eof_action=pass"
                    f"[{next_stage}]"
                )
                current_stage = next_stage
            filter_lines.append(
                f"[{video_out_node}]trim=duration={duration_sec:.6f},"
                f"setpts=PTS-STARTPTS,format=rgba[st{transition_index}videofinal]"
            )
            video_out_node = f"st{transition_index}videofinal"

        # The transition owns only Video Space. Rebuild the stable canvas,
        # device frame, and text overlays after the animated screen stream.
        transition_canvas_node = f"st{transition_index}canvas"
        if has_layout_scenes:
            filter_lines.append(
                f"color=c=0x{transition_bg_color}:s={out_w}x{out_h}:"
                f"r={src_fps}:d={duration_sec:.6f},format=rgba"
                f"[{transition_canvas_node}]"
            )
        else:
            filter_lines.append(
                f"[stbackground{transition_index}]tpad=stop_mode=clone:"
                f"stop_duration={duration_sec:.6f},trim=duration={duration_sec:.6f},"
                f"setpts=PTS-STARTPTS[{transition_canvas_node}]"
            )

        placed_video_node = f"st{transition_index}placed"
        if has_layout_scenes:
            transition_scale = max(float(transition_layout_scene.video_scale), 0.01)
            transition_video_w = max(2, int(round(transition_w * transition_scale / 2.0) * 2))
            transition_video_h = max(2, int(round(transition_h * transition_scale / 2.0) * 2))
            transition_video_x = int(round(
                transition_x * transition_scale + transition_layout_scene.video_x * out_w
            ))
            transition_video_y = int(round(
                transition_y * transition_scale + transition_layout_scene.video_y * out_h
            ))
            scaled_video_node = f"st{transition_index}scaledvideo"
            filter_lines.append(
                f"[{video_out_node}]scale={transition_video_w}:{transition_video_h}:"
                f"flags=lanczos[{scaled_video_node}]"
            )
            filter_lines.append(
                f"[{transition_canvas_node}][{scaled_video_node}]overlay="
                f"x={transition_video_x}:y={transition_video_y}:shortest=1:"
                f"eof_action=pass[{placed_video_node}]"
            )
        else:
            filter_lines.append(
                f"[{transition_canvas_node}][{video_out_node}]overlay="
                f"x={transition_x}:y={transition_y}:shortest=1:eof_action=pass"
                f"[{placed_video_node}]"
            )

        framed_transition_node = f"st{transition_index}framed"
        if has_layout_scenes and not transition_layout_scene.device_frame_visible:
            filter_lines.append(f"[stfr{transition_index}]nullsink")
            filter_lines.append(
                f"[{placed_video_node}]null[{framed_transition_node}]"
            )
        else:
            held_transition_frame = f"st{transition_index}frameheld"
            filter_lines.append(
                f"[stfr{transition_index}]tpad=stop_mode=clone:"
                f"stop_duration={duration_sec:.6f},trim=duration={duration_sec:.6f},"
                f"setpts=PTS-STARTPTS[{held_transition_frame}]"
            )
        if has_layout_scenes and transition_layout_scene.device_frame_visible:
            transition_scale = max(float(transition_layout_scene.video_scale), 0.01)
            transition_frame_w = max(2, int(round(out_w * transition_scale / 2.0) * 2))
            transition_frame_h = max(2, int(round(out_h * transition_scale / 2.0) * 2))
            transition_frame_x = int(round(transition_layout_scene.video_x * out_w))
            transition_frame_y = int(round(transition_layout_scene.video_y * out_h))
            scaled_frame_node = f"st{transition_index}scaledframe"
            filter_lines.append(
                f"[{held_transition_frame}]scale={transition_frame_w}:"
                f"{transition_frame_h}:flags=lanczos[{scaled_frame_node}]"
            )
            filter_lines.append(
                f"[{placed_video_node}][{scaled_frame_node}]overlay="
                f"x={transition_frame_x}:y={transition_frame_y}:"
                f"eof_action=repeat:repeatlast=1[{framed_transition_node}]"
            )
        elif not has_layout_scenes:
            filter_lines.append(
                f"[{placed_video_node}][{held_transition_frame}]overlay=x=0:y=0:"
                f"eof_action=repeat:repeatlast=1[{framed_transition_node}]"
            )

        current_transition_node = framed_transition_node
        for text_index, (_annotation, input_index) in enumerate(
            transition_text_assets.get(transition_index, [])
        ):
            text_node = f"st{transition_index}text{text_index}"
            filter_lines.append(
                f"[{current_transition_node}][{input_index}:v]overlay=x=0:y=0:"
                f"shortest=1:eof_action=pass:repeatlast=1[{text_node}]"
            )
            current_transition_node = text_node
        filter_lines.append(
            f"[{current_transition_node}]trim=duration={duration_sec:.6f},"
            f"setpts=PTS-STARTPTS,format=yuv420p[{out_node}]"
        )
        audio_node = ""
        if source_has_audio:
            audio_node = f"st{transition_index}aout"
            filter_lines.append(
                "anullsrc=r=48000:cl=stereo,"
                f"atrim=duration={duration_sec:.6f},asetpts=PTS-STARTPTS[{audio_node}]"
            )
        output_items.append(
            (float(transition.timestamp_ms), 15, out_node, audio_node)
        )
        appended_transition_ids.add(transition.id)

    def append_explainer_output(
        explainer_index: int, scene: ExplainerScene
    ) -> None:
        """Insert an animated presentation moment without consuming source time."""
        endpoint = transition_endpoint_nodes.get((scene.id, "explainer"))
        text_asset = explainer_text_assets.get(scene.id)
        if endpoint is None or text_asset is None:
            logger.warning(
                "Skipping explainer %s because its frozen endpoint or text asset is missing",
                scene.id,
            )
            return

        duration_ms = max(float(scene.end_ms) - float(scene.start_ms), 1.0)
        duration_sec = duration_ms / 1000.0
        endpoint_node, endpoint_duration, endpoint_sample = endpoint
        frame_sec = 1.0 / max(src_fps, 1.0)
        sample_start = max(0.0, min(endpoint_sample, max(endpoint_duration - frame_sec, 0.0)))
        sample_end = min(endpoint_duration, sample_start + max(frame_sec * 2.0, 0.002))
        target_layout = scene.layout_scene
        aperture_overlap = (
            1
            if target_layout.device_frame_visible and not frame_preset.is_none
            else 0
        )
        covered_aperture = PresentationGroupGeometry.bezel_covered_aperture(
            Rect2D(scr_x, scr_y, scr_w, scr_h),
            out_w,
            out_h,
            overlap_px=aperture_overlap,
        )
        frozen_node = f"ex{explainer_index}frozen"
        filter_lines.append(
            f"[{endpoint_node}]trim=start={sample_start:.6f}:end={sample_end:.6f},"
            f"select='eq(n,0)',setpts=PTS-STARTPTS,"
            f"scale={covered_aperture.width}:{covered_aperture.height}:flags=lanczos,"
            f"format=rgba,loop=loop=-1:size=1:start=0,settb=AVTB,"
            f"setpts=N/({src_fps}*TB),trim=duration={duration_sec:.6f},"
            f"format=rgba[{frozen_node}]"
        )

        base_layout = canvas_layout_scene_at(
            list(canvas_layout_scenes or []),
            max(0.0, float(scene.start_ms) - 0.001),
            source_duration_ms,
        )
        phases = explainer_phase_timing(scene)
        transition_sec = max(
            0.0,
            (phases.layout_enter_end - phases.layout_enter_start) / 1000.0,
        )
        restore_start_sec = max(
            0.0,
            (phases.layout_restore_start - float(scene.start_ms)) / 1000.0,
        )
        restore_duration_sec = max(
            0.0,
            (phases.layout_restore_end - phases.layout_restore_start) / 1000.0,
        )
        if transition_sec <= 0.000001:
            layout_progress = "1"
        else:
            enter = ffmpeg_quintic_ease(
                ffmpeg_clamped_progress("t", 0.0, transition_sec)
            )
            if scene.restore_previous and restore_duration_sec > 0.000001:
                leave = ffmpeg_quintic_ease(
                    ffmpeg_clamped_progress(
                        "t", restore_start_sec, duration_sec
                    )
                )
                layout_progress = (
                    f"if(lt(t,{transition_sec:.6f}),{enter},"
                    f"if(gt(t,{restore_start_sec:.6f}),1-({leave}),1))"
                )
            else:
                layout_progress = f"if(lt(t,{transition_sec:.6f}),{enter},1)"

        scale_expr = ffmpeg_lerp(
            base_layout.video_scale,
            target_layout.video_scale,
            layout_progress,
        )
        x_expr = ffmpeg_lerp(
            base_layout.video_x,
            target_layout.video_x,
            layout_progress,
        )
        y_expr = ffmpeg_lerp(
            base_layout.video_y,
            target_layout.video_y,
            layout_progress,
        )
        background_color = _layout_background_color(bg_color, target_layout)
        canvas_node = f"ex{explainer_index}canvas"
        if has_layout_scenes:
            filter_lines.append(
                f"color=c=0x{background_color}:s={out_w}x{out_h}:"
                f"r={src_fps}:d={duration_sec:.6f},format=rgba[{canvas_node}]"
            )
        else:
            filter_lines.append(
                f"[exbackground{explainer_index}]tpad=stop_mode=clone:"
                f"stop_duration={duration_sec:.6f},trim=duration={duration_sec:.6f},"
                f"setpts=PTS-STARTPTS[{canvas_node}]"
            )

        # Assemble video and bezel at identity geometry first. The complete
        # transparent presentation group then receives one scale/translation,
        # keeping the aperture locked to the frame at every rasterized step.
        group_canvas_node = f"ex{explainer_index}groupcanvas"
        group_video_node = f"ex{explainer_index}groupvideo"
        filter_lines.append(
            f"color=c=black@0.0:s={out_w}x{out_h}:r={src_fps}:"
            f"d={duration_sec:.6f},format=rgba[{group_canvas_node}]"
        )
        filter_lines.append(
            f"[{group_canvas_node}][{frozen_node}]overlay="
            f"x={covered_aperture.x}:y={covered_aperture.y}:"
            f"shortest=1:eof_action=pass:format=auto[{group_video_node}]"
        )

        group_node = group_video_node
        if target_layout.device_frame_visible:
            held_frame_node = f"ex{explainer_index}frameheld"
            framed_group_node = f"ex{explainer_index}groupframed"
            filter_lines.append(
                f"[exfr{explainer_index}]tpad=stop_mode=clone:"
                f"stop_duration={duration_sec:.6f},trim=duration={duration_sec:.6f},"
                f"setpts=PTS-STARTPTS[{held_frame_node}]"
            )
            filter_lines.append(
                f"[{group_video_node}][{held_frame_node}]overlay=x=0:y=0:"
                f"eof_action=repeat:repeatlast=1:format=auto[{framed_group_node}]"
            )
            group_node = framed_group_node
        else:
            filter_lines.append(f"[exfr{explainer_index}]nullsink")

        scaled_group_node = f"ex{explainer_index}group"
        filter_lines.append(
            f"[{group_node}]scale=w='max(2,trunc({out_w}*({scale_expr})/2)*2)':"
            f"h='max(2,trunc({out_h}*({scale_expr})/2)*2)':"
            f"eval=frame:flags=lanczos[{scaled_group_node}]"
        )
        framed_node = f"ex{explainer_index}framed"
        filter_lines.append(
            f"[{canvas_node}][{scaled_group_node}]overlay=x='({x_expr})*{out_w}':"
            f"y='({y_expr})*{out_h}':eval=frame:shortest=1:eof_action=pass"
            f"[{framed_node}]"
        )

        annotation, input_index = text_asset
        text_end_sec = max(
            0.001,
            (phases.text_exit_end - float(scene.start_ms)) / 1000.0,
        )
        text_node = f"ex{explainer_index}text"
        filter_lines.extend(
            _animated_text_overlay_filters(
                framed_node,
                f"{input_index}:v",
                text_node,
                annotation,
                start_sec=0.0,
                end_sec=text_end_sec,
                out_w=out_w,
                out_h=out_h,
            )
        )
        out_node = f"ex{explainer_index}out"
        filter_lines.append(
            f"[{text_node}]trim=duration={duration_sec:.6f},"
            f"setpts=PTS-STARTPTS,format=yuv420p[{out_node}]"
        )
        audio_node = ""
        if source_has_audio:
            audio_node = f"ex{explainer_index}aout"
            filter_lines.append(
                "anullsrc=r=48000:cl=stereo,"
                f"atrim=duration={duration_sec:.6f},"
                f"asetpts=PTS-STARTPTS[{audio_node}]"
            )
        output_items.append((float(scene.start_ms), 18, out_node, audio_node))

    for seg_index, segment in enumerate(segments):
        # A card inside a deleted source gap belongs between the surrounding
        # kept segments. A card at source time zero belongs before all video.
        for frame_idx, timeline_frame in enumerate(ordered_frames):
            if float(timeline_frame.timestamp_ms) <= float(segment.start_ms) + 0.5:
                append_timeline_frame_output(frame_idx, timeline_frame)

        media_start_sec, media_end_sec, media_duration_sec = segment_media_bounds[seg_index]
        session_duration_sec = max((segment.end_ms - segment.start_ms) / 1000.0, 0.001)
        output_duration_sec = session_duration_sec / _segment_speed(segment)
        retime_scale = output_duration_sec / max(media_duration_sec, 0.001)

        media_kfs = media_keyframe_sets[seg_index]
        local_clicks = local_click_sets[seg_index]
        zoompan_filter = _build_zoompan_filter(media_kfs, src_fps)

        filter_lines.append(
            f"[0:v]trim=start={media_start_sec:.6f}:end={media_end_sec:.6f},setpts=PTS-STARTPTS[s{seg_index}trim]"
        )
        # Render text/shapes once with Qt, then overlay the same transparent
        # asset before zoom. This makes export typography and geometry match
        # the preview rather than relying on a separate FFmpeg drawtext path.
        annotated_source_node = f"s{seg_index}trim"
        for annotation_index, (annotation, input_index) in enumerate(
            local_video_annotation_assets[seg_index]
        ):
            source_start = max(float(annotation.start_ms), float(segment.start_ms))
            source_end = min(float(annotation.end_ms), float(segment.end_ms))
            local_start = _media_time_for_segment(source_start, segment, media_mapper, media_start_sec)
            local_end = _media_time_for_segment(source_end, segment, media_mapper, media_start_sec, for_end=True)
            if local_end <= local_start:
                continue
            next_node = f"s{seg_index}annotation{annotation_index}"
            filter_lines.append(
                _timed_overlay_filter(
                    annotated_source_node,
                    f"{input_index}:v",
                    next_node,
                    start_sec=local_start,
                    end_sec=local_end,
                )
            )
            annotated_source_node = next_node
        local_masks: list[tuple[TimelineOverlay, float, float]] = []
        for mask in static_video_masks:
            if mask.clip_id and mask.clip_id != segment.id:
                continue
            source_start = max(float(mask.start_ms), float(segment.start_ms))
            source_end = min(float(mask.end_ms), float(segment.end_ms))
            if source_end <= source_start:
                continue
            local_start = _media_time_for_segment(
                source_start, segment, media_mapper, media_start_sec
            )
            local_end = _media_time_for_segment(
                source_end, segment, media_mapper, media_start_sec, for_end=True
            )
            if local_end > local_start:
                local_masks.append((mask, local_start, local_end))
        mask_filters, masked_source_node = ffmpeg_static_mask_filters(
            annotated_source_node,
            f"s{seg_index}",
            local_masks,
            src_w,
            src_h,
        )
        filter_lines.extend(mask_filters)
        filter_lines.append(f"[{masked_source_node}]{zoompan_filter}[s{seg_index}zoom]")
        filter_lines.append(f"[s{seg_index}zoom]scale={scr_w}:{scr_h}[s{seg_index}vid]")

        # Cursor and click evidence live in Video Space. Composite them onto
        # the zoomed video before placing that video on the static canvas.
        current_video_node = f"s{seg_index}vid"
        if (
            local_clicks
            and click_img_path
            and click_preset
            and click_preset.duration_ms > 0
            and click_preset.color[3] > 0
        ):
            r = max(int(click_preset.radius), 1)
            m_left = monitor_rect.get("left", 0) if monitor_rect else 0
            m_top = monitor_rect.get("top", 0) if monitor_rect else 0
            m_w = monitor_rect.get("width", src_w) if monitor_rect else src_w
            m_h = monitor_rect.get("height", src_h) if monitor_rect else src_h

            for local_click in local_clicks:
                abs_click_ms = float(segment.start_ms) + float(local_click.timestamp)
                t_s, t_e = _media_window_for_segment(
                    abs_click_ms,
                    click_preset.duration_ms,
                    segment,
                    media_mapper,
                    media_start_sec,
                )
                t_e = min(t_e, media_duration_sec)
                click_x, click_y = _click_point_for_export(local_click)
                rel_x = (click_x - m_left) / max(m_w, 1)
                rel_y = (click_y - m_top) / max(m_h, 1)
                rel_x_before_zoom = rel_x
                rel_y_before_zoom = rel_y
                rel_x, rel_y = _map_zoomed_relative_point(
                    rel_x,
                    rel_y,
                    t_s * 1000.0,
                    media_kfs,
                )
                cx = int(rel_x * scr_w - r)
                cy = int(rel_y * scr_h - r)

                timed_node = f"cl{click_node_index}"
                next_node = f"s{seg_index}click{click_node_index}"
                _log_click_coordinate_audit(
                    overlay_kind="click-effect",
                    segment_index=seg_index,
                    click_index=click_node_index,
                    click=local_click,
                    abs_click_ms=abs_click_ms,
                    media_time_sec=t_s,
                    resolved_x=click_x,
                    resolved_y=click_y,
                    monitor_rect=monitor_rect,
                    monitor_left=m_left,
                    monitor_top=m_top,
                    monitor_width=m_w,
                    monitor_height=m_h,
                    rel_x_before_zoom=rel_x_before_zoom,
                    rel_y_before_zoom=rel_y_before_zoom,
                    rel_x_after_zoom=rel_x,
                    rel_y_after_zoom=rel_y,
                    scr_x=0,
                    scr_y=0,
                    scr_w=scr_w,
                    scr_h=scr_h,
                    anchor_x=r,
                    anchor_y=r,
                    final_x=cx,
                    final_y=cy,
                    current_comp_node=current_video_node,
                    timed_node=timed_node,
                    next_node=next_node,
                )
                filter_lines.append(
                    _timed_overlay_filter(
                        current_video_node,
                        f"cl{click_node_index}",
                        next_node,
                        x=cx,
                        y=cy,
                        start_sec=t_s,
                        end_sec=t_e,
                    )
                )
                current_video_node = next_node
                click_node_index += 1

        cursor_track = cursor_motion_by_segment.get(seg_index)
        if cursor_track is not None:
            command_node = f"s{seg_index}cursorcmd"
            next_node = f"s{seg_index}cursor"
            filter_lines.append(
                f"[{current_video_node}]sendcmd=f='{_ffmpeg_filter_path(cursor_track.command_path)}'"
                f"[{command_node}]"
            )
            filter_lines.append(
                f"[{command_node}][cu{cursor_track.node_index}]overlay@cursor{seg_index}="
                f"x={cursor_track.initial_x}:y={cursor_track.initial_y}:eval=init:"
                f"eof_action=repeat:repeatlast=1[{next_node}]"
            )
            current_video_node = next_node
        elif local_clicks and cursor_img_path:
            cursor_hold_sec = max(
                (click_preset.duration_ms / 1000.0) if click_preset else 0.0,
                0.75,
            )
            m_left = monitor_rect.get("left", 0) if monitor_rect else 0
            m_top = monitor_rect.get("top", 0) if monitor_rect else 0
            m_w = monitor_rect.get("width", src_w) if monitor_rect else src_w
            m_h = monitor_rect.get("height", src_h) if monitor_rect else src_h

            for local_click in local_clicks:
                abs_click_ms = float(segment.start_ms) + float(local_click.timestamp)
                t_s, t_e = _media_window_for_segment(
                    abs_click_ms,
                    cursor_hold_sec * 1000.0,
                    segment,
                    media_mapper,
                    media_start_sec,
                )
                t_e = min(t_e, media_duration_sec)
                click_x, click_y = _click_point_for_export(local_click)
                rel_x = (click_x - m_left) / max(m_w, 1)
                rel_y = (click_y - m_top) / max(m_h, 1)
                rel_x_before_zoom = rel_x
                rel_y_before_zoom = rel_y
                rel_x, rel_y = _map_zoomed_relative_point(
                    rel_x,
                    rel_y,
                    t_s * 1000.0,
                    media_kfs,
                )
                anchor_x, anchor_y = _cursor_anchor_for_export(
                    cursor_asset_path=cursor_asset_path,
                    cursor_style_id=cursor_style_id,
                    cursor_hotspot=cursor_hotspot,
                    cursor_scale=render_cursor_scale,
                )
                cx = int(rel_x * scr_w - anchor_x)
                cy = int(rel_y * scr_h - anchor_y)

                timed_node = f"cu{cursor_node_index}"
                next_node = f"s{seg_index}cursor{cursor_node_index}"
                _log_click_coordinate_audit(
                    overlay_kind="cursor",
                    segment_index=seg_index,
                    click_index=cursor_node_index,
                    click=local_click,
                    abs_click_ms=abs_click_ms,
                    media_time_sec=t_s,
                    resolved_x=click_x,
                    resolved_y=click_y,
                    monitor_rect=monitor_rect,
                    monitor_left=m_left,
                    monitor_top=m_top,
                    monitor_width=m_w,
                    monitor_height=m_h,
                    rel_x_before_zoom=rel_x_before_zoom,
                    rel_y_before_zoom=rel_y_before_zoom,
                    rel_x_after_zoom=rel_x,
                    rel_y_after_zoom=rel_y,
                    scr_x=0,
                    scr_y=0,
                    scr_w=scr_w,
                    scr_h=scr_h,
                    anchor_x=anchor_x,
                    anchor_y=anchor_y,
                    final_x=cx,
                    final_y=cy,
                    current_comp_node=current_video_node,
                    timed_node=timed_node,
                    next_node=next_node,
                )
                filter_lines.append(
                    _timed_overlay_filter(
                        current_video_node,
                        f"cu{cursor_node_index}",
                        next_node,
                        x=cx,
                        y=cy,
                        start_sec=t_s,
                        end_sec=t_e,
                    )
                )
                current_video_node = next_node
                cursor_node_index += 1

        # Highlights are video-anchored evidence.  Composite their dim mask
        # and optional border before transition endpoint extraction and before
        # the video is placed into the Presentation Shell.
        video_geom = {
            "scr_x": 0,
            "scr_y": 0,
            "scr_w": scr_w,
            "scr_h": scr_h,
        }
        for highlight_index, highlight in enumerate(local_highlight_sets[seg_index]):
            t_s = max(float(highlight.start_ms) / 1000.0, 0.0)
            t_e = min(float(highlight.end_ms) / 1000.0, media_duration_sec)
            if t_e <= t_s:
                continue
            mask_node = f"s{seg_index}highlight{highlight_index}mask"
            filter_lines.append(
                _highlight_mask_filter(
                    highlight,
                    out_w=scr_w,
                    out_h=scr_h,
                    src_fps=src_fps,
                    duration_sec=media_duration_sec,
                    geom=video_geom,
                    output_node=mask_node,
                )
            )
            dim_node = f"s{seg_index}highlight{highlight_index}dim"
            filter_lines.append(
                _timed_overlay_filter(
                    current_video_node,
                    mask_node,
                    dim_node,
                    start_sec=t_s,
                    end_sec=t_e,
                )
            )
            current_video_node = dim_node

        for border_index, (highlight, input_index) in enumerate(
            local_highlight_assets[seg_index]
        ):
            t_s = max(float(highlight.start_ms) / 1000.0, 0.0)
            t_e = min(float(highlight.end_ms) / 1000.0, media_duration_sec)
            if t_e <= t_s:
                continue
            border_node = f"s{seg_index}highlightborder{border_index}"
            filter_lines.append(
                _timed_overlay_filter(
                    current_video_node,
                    f"{input_index}:v",
                    border_node,
                    start_sec=t_s,
                    end_sec=t_e,
                )
            )
            current_video_node = border_node

        # Capture transition endpoints here, at the end of Video Space and
        # before the canvas background, device frame, or text are introduced.
        endpoint_roles: list[tuple[object, str]] = []
        for transition in ordered_transitions:
            if abs(float(transition.timestamp_ms) - float(segment.end_ms)) <= 0.5:
                endpoint_roles.append((transition, "outgoing"))
            transition_resume_ms = transition_resume_source_ms(
                transition, source_duration_ms
            )
            if abs(transition_resume_ms - float(segment.start_ms)) <= 0.5:
                endpoint_roles.append((transition, "incoming"))
        for scene in ordered_explainers:
            visual_source_ms = explainer_visual_sources.get(
                scene.id,
                float(scene.start_ms),
            )
            if abs(visual_source_ms - float(segment.start_ms)) <= 0.5:
                endpoint_roles.append((scene, "explainer"))
            elif (
                abs(visual_source_ms - float(segment.end_ms)) <= 0.5
                and not any(
                    abs(visual_source_ms - float(candidate.start_ms)) <= 0.5
                    for candidate in segments
                )
            ):
                endpoint_roles.append((scene, "explainer_outgoing"))
        if endpoint_roles:
            split_nodes = [f"s{seg_index}videomain"] + [
                f"s{seg_index}transitionvideo{index}"
                for index in range(len(endpoint_roles))
            ]
            filter_lines.append(
                f"[{current_video_node}]split={len(split_nodes)}"
                + "".join(f"[{node}]" for node in split_nodes)
            )
            current_video_node = split_nodes[0]
            for role_index, (transition, role) in enumerate(endpoint_roles):
                sample_ms = (
                    transition.outgoing_frame_ms
                    if role == "outgoing"
                    else transition.incoming_frame_ms
                    if role == "incoming"
                    else explainer_visual_sources.get(
                        transition.id,
                        float(getattr(transition, "start_ms", segment.start_ms)),
                    )
                )
                if sample_ms is None:
                    sample_ms = (
                        float(segment.end_ms) - (1000.0 / max(src_fps, 1.0))
                        if role == "outgoing"
                        else float(segment.start_ms) + (1000.0 / max(src_fps, 1.0))
                        if role == "incoming"
                        else explainer_visual_sources.get(
                            transition.id,
                            float(getattr(transition, "start_ms", segment.start_ms)),
                        )
                    )
                sample_media_sec, _ = _media_window_for_segment(
                    float(sample_ms),
                    1.0,
                    segment,
                    media_mapper,
                    media_start_sec,
                )
                sample_media_sec = max(
                    0.0,
                    min(
                        max(0.0, media_duration_sec - (1.0 / max(src_fps, 1.0))),
                        sample_media_sec,
                    ),
                )
                endpoint_key = (
                    "explainer"
                    if role in {"explainer", "explainer_outgoing"}
                    else role
                )
                transition_endpoint_nodes[(transition.id, endpoint_key)] = (
                    split_nodes[role_index + 1],
                    media_duration_sec,
                    sample_media_sec,
                )

        # Canvas Space begins here. In the scene-aware branch the video and
        # frame are transformed as one Presentation Group. The legacy branch
        # remains byte-compatible for projects without layout scenes.
        layout_scene = None
        layout_is_static = False
        if has_layout_scenes:
            layout_is_static = canvas_layout_transition_for_range(
                list(canvas_layout_scenes or []),
                segment.start_ms,
                segment.end_ms,
            ) is None
            layout_scene, layout_scale_expr, layout_x_expr, layout_y_expr = _layout_scene_expressions(
                list(canvas_layout_scenes or []),
                segment,
                media_mapper,
                media_start_sec,
            )
            segment_bg_color = _layout_background_color(bg_color, layout_scene)
        else:
            layout_scale_expr = layout_x_expr = layout_y_expr = "1.000000"

        if has_layout_scenes:
            color_args = (
                f"color=c=0x{segment_bg_color}:s={out_w}x{out_h}:"
                f"r={src_fps}:d={media_duration_sec}"
            )
            filter_lines.append(f"{color_args}[s{seg_index}bg]")
            if layout_is_static and layout_scene is not None:
                static_scale = max(float(layout_scene.video_scale), 0.01)
                scaled_w = max(2, int(round(scr_w * static_scale / 2.0) * 2))
                scaled_h = max(2, int(round(scr_h * static_scale / 2.0) * 2))
                video_x = int(round(scr_x * static_scale + layout_scene.video_x * out_w))
                video_y = int(round(scr_y * static_scale + layout_scene.video_y * out_h))
                filter_lines.append(
                    f"[{current_video_node}]scale={scaled_w}:{scaled_h}[s{seg_index}layoutvid]"
                )
                filter_lines.append(
                    f"[s{seg_index}bg][s{seg_index}layoutvid]overlay="
                    f"x={video_x}:y={video_y}:shortest=1:eof_action=pass:"
                    f"enable='between(t,0.000000,{media_duration_sec:.6f})'[s{seg_index}base]"
                )
            else:
                scaled_w = f"trunc(({scr_w})*({layout_scale_expr})/2)*2"
                scaled_h = f"trunc(({scr_h})*({layout_scale_expr})/2)*2"
                filter_lines.append(
                    f"[{current_video_node}]scale=w='{scaled_w}':h='{scaled_h}':eval=frame"
                    f"[s{seg_index}layoutvid]"
                )
                video_x_expr = f"({scr_x})*({layout_scale_expr})+({layout_x_expr})*{out_w}"
                video_y_expr = f"({scr_y})*({layout_scale_expr})+({layout_y_expr})*{out_h}"
                filter_lines.append(
                    f"[s{seg_index}bg][s{seg_index}layoutvid]overlay="
                    f"x='{video_x_expr}':y='{video_y_expr}':"
                    f"shortest=1:eof_action=pass:"
                    f"enable='between(t,0.000000,{media_duration_sec:.6f})'[s{seg_index}base]"
                )
        else:
            filter_lines.append(
                f"[background{seg_index}]tpad=stop_mode=clone:"
                f"stop_duration={media_duration_sec:.6f},"
                f"trim=duration={media_duration_sec:.6f},"
                f"setpts=PTS-STARTPTS[s{seg_index}bg]"
            )
            filter_lines.append(
                f"[s{seg_index}bg][{current_video_node}]overlay=x={scr_x}:y={scr_y}:"
                f"shortest=1:eof_action=pass[s{seg_index}base]"
            )
        filter_lines.append(
            f"[s{seg_index}base]setpts=PTS-STARTPTS[s{seg_index}comp0]"
        )
        current_comp_node = f"s{seg_index}comp0"

        framed_node = f"s{seg_index}framed"
        frame_is_used = not (
            has_layout_scenes
            and layout_scene is not None
            and not layout_scene.device_frame_visible
        )
        held_frame_node = f"s{seg_index}frameheld"
        if frame_is_used:
            filter_lines.append(
                f"[fr{seg_index}]tpad=stop_mode=clone:"
                f"stop_duration={media_duration_sec:.6f},"
                f"trim=duration={media_duration_sec:.6f},"
                f"setpts=PTS-STARTPTS[{held_frame_node}]"
            )
        else:
            filter_lines.append(f"[fr{seg_index}]nullsink")
        if has_layout_scenes and layout_scene is not None:
            if layout_scene.device_frame_visible:
                if layout_is_static:
                    frame_w = max(2, int(round(out_w * float(layout_scene.video_scale) / 2.0) * 2))
                    frame_h = max(2, int(round(out_h * float(layout_scene.video_scale) / 2.0) * 2))
                    frame_x = int(round(layout_scene.video_x * out_w))
                    frame_y = int(round(layout_scene.video_y * out_h))
                    filter_lines.append(
                        f"[{held_frame_node}]scale={frame_w}:{frame_h}[s{seg_index}layoutframe]"
                    )
                    filter_lines.append(
                        f"[{current_comp_node}][s{seg_index}layoutframe]overlay="
                        f"x={frame_x}:y={frame_y}:"
                        f"eof_action=repeat:repeatlast=1:"
                        f"enable='between(t,0.000000,{media_duration_sec:.6f})'[{framed_node}]"
                    )
                else:
                    frame_w = f"trunc({out_w}*({layout_scale_expr})/2)*2"
                    frame_h = f"trunc({out_h}*({layout_scale_expr})/2)*2"
                    filter_lines.append(
                        f"[{held_frame_node}]scale=w='{frame_w}':h='{frame_h}':eval=frame"
                        f"[s{seg_index}layoutframe]"
                    )
                    filter_lines.append(
                        f"[{current_comp_node}][s{seg_index}layoutframe]overlay="
                        f"x='({layout_x_expr})*{out_w}':y='({layout_y_expr})*{out_h}':"
                        f"eof_action=repeat:repeatlast=1:"
                        f"enable='between(t,0.000000,{media_duration_sec:.6f})'[{framed_node}]"
                    )
            else:
                filter_lines.append(f"[{current_comp_node}]null[{framed_node}]")
        else:
            filter_lines.append(
                f"[{current_comp_node}][{held_frame_node}]overlay=x=0:y=0:"
                f"eof_action=repeat:repeatlast=1[{framed_node}]"
            )

        # TextAnnotation is absolute Canvas Space and is intentionally the
        # final visual layer after video zoom, background, and device frame.
        current_canvas_node = framed_node
        for text_index, (annotation, input_index) in enumerate(
            local_text_annotation_assets[seg_index]
        ):
            t_s = max(float(annotation.start_ms) / 1000.0, 0.0)
            t_e = min(float(annotation.end_ms) / 1000.0, media_duration_sec)
            if t_e <= t_s:
                continue
            text_node = f"s{seg_index}text{text_index}"
            filter_lines.extend(
                _animated_text_overlay_filters(
                    current_canvas_node,
                    f"{input_index}:v",
                    text_node,
                    annotation,
                    start_sec=t_s,
                    end_sec=t_e,
                    out_w=out_w,
                    out_h=out_h,
                )
            )
            current_canvas_node = text_node

        out_node = f"s{seg_index}out"
        filter_lines.append(f"[{current_canvas_node}]setpts={retime_scale:.8f}*PTS[{out_node}]")
        main_video_node = out_node
        audio_node = ""
        if source_has_audio:
            audio_node = f"s{seg_index}aout"
            audio_filters = [
                f"[0:a]atrim=start={media_start_sec:.6f}:end={media_end_sec:.6f}",
                "asetpts=PTS-STARTPTS",
                "aresample=48000",
                "aformat=sample_rates=48000:channel_layouts=stereo",
                *_atempo_filters(1.0 / max(retime_scale, 0.001)),
                "asetpts=PTS-STARTPTS",
            ]
            filter_lines.append(
                ",".join(audio_filters) + f"[{audio_node}]"
            )
        output_items.append(
            (float(segment.start_ms), 20, main_video_node, audio_node)
        )

        for frame_idx, timeline_frame in enumerate(ordered_frames):
            if float(timeline_frame.timestamp_ms) <= float(segment.end_ms) + 0.5:
                append_timeline_frame_output(frame_idx, timeline_frame)

    for frame_idx, timeline_frame in enumerate(ordered_frames):
        append_timeline_frame_output(frame_idx, timeline_frame)
    for transition_index, transition in enumerate(ordered_transitions):
        append_screen_transition_output(transition_index, transition)
    for explainer_index, scene in enumerate(ordered_explainers):
        append_explainer_output(explainer_index, scene)

    ordered_output_items = sorted(output_items, key=lambda item: (item[0], item[1]))
    segment_outputs = [item[2] for item in ordered_output_items]
    audio_segment_outputs = [item[3] for item in ordered_output_items if item[3]]

    if source_has_audio and len(audio_segment_outputs) != len(segment_outputs):
        raise ValueError("Export failed: audio and video timeline items are out of sync")

    audio_output_node = ""
    if source_has_audio:
        if len(segment_outputs) > 1:
            concat_inputs = "".join(
                f"[{video_node}][{audio_node}]"
                for video_node, audio_node in zip(segment_outputs, audio_segment_outputs)
            )
            filter_lines.append(
                f"{concat_inputs}concat=n={len(segment_outputs)}:v=1:a=1[out][sourceaout]"
            )
        else:
            filter_lines.append(f"[{segment_outputs[0]}]null[out]")
            filter_lines.append(f"[{audio_segment_outputs[0]}]anull[sourceaout]")
        audio_output_node = "sourceaout"
    elif len(segment_outputs) > 1:
        concat_inputs = "".join(f"[{node}]" for node in segment_outputs)
        filter_lines.append(f"{concat_inputs}concat=n={len(segment_outputs)}:v=1:a=0[out]")
    else:
        filter_lines.append(f"[{segment_outputs[0]}]null[out]")

    return ExportFilterGraphPlan(
        filtergraph=";\n".join(filter_lines),
        background_img_path=background_img_path,
        frame_img_path=frame_img_path,
        click_img_path=click_img_path,
        cursor_img_path=cursor_img_path,
        cursor_motion_tracks=cursor_motion_tracks,
        highlight_img_paths=highlight_img_paths,
        video_annotation_img_paths=video_annotation_img_paths,
        text_annotation_img_paths=text_annotation_img_paths,
        transition_text_img_paths=transition_text_img_paths,
        timeline_frame_img_paths=timeline_frame_img_paths,
        voiceover_audio_paths=[],
        audio_input_specs=[],
        temp_files=temp_files,
        temp_dirs=[],
        output_total_sec=output_total_sec,
        has_speed_changes=has_speed_changes,
        has_timeline_edits=has_timeline_edits,
        has_source_audio=source_has_audio,
        has_voiceover_audio=False,
        has_audio_output=bool(audio_output_node),
        audio_output_node=audio_output_node,
    )


def _apply_trim_to_video_segments(
    video_segments: Optional[List[VideoSegment]],
    duration_ms: float,
    trim_start_ms: float,
    trim_end_ms: float,
) -> Optional[List[VideoSegment]]:
    """Apply legacy trim handles as a segment intersection layer."""
    duration = max(float(duration_ms or 0.0), 0.0)
    if duration <= 0:
        return video_segments

    start = max(0.0, min(float(trim_start_ms or 0.0), duration))
    raw_end = float(trim_end_ms or 0.0)
    end = max(start, min(raw_end, duration)) if raw_end > 0 else duration
    if start <= 0.5 and end >= duration - 0.5:
        return video_segments

    source_segments = ordered_video_segments(
        video_segments or [VideoSegment.create(0.0, duration, 1.0)]
    )
    trimmed: List[VideoSegment] = []
    for segment in source_segments:
        seg_start = max(float(segment.start_ms), start)
        seg_end = min(float(segment.end_ms), end)
        if seg_end <= seg_start:
            continue
        trimmed.append(
            VideoSegment(
                id=segment.id,
                start_ms=seg_start,
                end_ms=seg_end,
                speed=_segment_speed(segment),
                sequence_index=int(getattr(segment, "sequence_index", len(trimmed))),
            )
        )
    return trimmed


def _output_time_for_source_timestamp(
    timestamp_ms: float,
    segments: List[VideoSegment],
    timeline_frames: Optional[List[TimelineFrame]],
    screen_transitions: Optional[List[ScreenTransition]] = None,
) -> Optional[float]:
    """Map a source timestamp to the edited output timeline in seconds."""
    ordered_frames = sorted(timeline_frames or [], key=lambda frame: float(frame.timestamp_ms))
    ordered_transitions = sorted(
        [item for item in (screen_transitions or []) if item.enabled],
        key=lambda item: float(item.timestamp_ms),
    )
    segments = _split_segments_at_timeline_frames(segments, ordered_frames)
    segments = _split_segments_at_screen_transitions(segments, ordered_transitions)
    appended_frame_ids: set[str] = set()
    appended_transition_ids: set[str] = set()
    elapsed_sec = 0.0
    for index, segment in enumerate(segments):
        seg_start = float(segment.start_ms)
        seg_end = float(segment.end_ms)
        is_last = index == len(segments) - 1

        for frame in ordered_frames:
            if frame.id in appended_frame_ids:
                continue
            if float(frame.timestamp_ms) <= seg_start + 0.5:
                elapsed_sec += max(float(frame.duration_ms), 250.0) / 1000.0
                appended_frame_ids.add(frame.id)
        for transition in ordered_transitions:
            if transition.id in appended_transition_ids:
                continue
            if float(transition.timestamp_ms) <= seg_start + 0.5:
                elapsed_sec += max(float(transition.duration_ms), 150.0) / 1000.0
                appended_transition_ids.add(transition.id)

        if seg_start <= timestamp_ms < seg_end or (is_last and abs(timestamp_ms - seg_end) <= 0.5):
            local_ms = max(0.0, min(float(timestamp_ms), seg_end) - seg_start)
            return elapsed_sec + (local_ms / 1000.0) / _segment_speed(segment)

        elapsed_sec += ((seg_end - seg_start) / 1000.0) / _segment_speed(segment)
        for frame in ordered_frames:
            if frame.id in appended_frame_ids:
                continue
            if float(frame.timestamp_ms) <= seg_end + 0.5:
                elapsed_sec += max(float(frame.duration_ms), 250.0) / 1000.0
                appended_frame_ids.add(frame.id)
        for transition in ordered_transitions:
            if transition.id in appended_transition_ids:
                continue
            if float(transition.timestamp_ms) <= seg_end + 0.5:
                elapsed_sec += max(float(transition.duration_ms), 150.0) / 1000.0
                appended_transition_ids.add(transition.id)

    return None


def _voiceover_audio_rows(
    voiceover_segments: Optional[List[VoiceoverSegment]],
    video_segments: Optional[List[VideoSegment]],
    timeline_frames: Optional[List[TimelineFrame]],
    duration_ms: float,
    screen_transitions: Optional[List[ScreenTransition]] = None,
) -> list[tuple[str, int, float]]:
    """Return (audio_path, delay_ms, volume) rows for FFmpeg audio inputs."""
    source_duration_ms = max(float(duration_ms or 0.0), 0.0)
    segments = _normalize_video_segments(
        video_segments,
        source_duration_ms,
        fill_gaps=not bool(video_segments),
    )
    rows: list[tuple[str, int, float]] = []
    for segment in voiceover_segments or []:
        audio_path = str(getattr(segment, "audio_path", "") or "")
        if not audio_path or not os.path.isfile(audio_path):
            continue
        output_sec = _output_time_for_source_timestamp(
            float(segment.timestamp),
            segments,
            timeline_frames,
            screen_transitions,
        )
        if output_sec is None:
            logger.info("Skipping voiceover outside kept video segments: %s", segment.id)
            continue
        volume = max(0.0, min(3.0, float(getattr(segment, "volume", 1.0) or 1.0)))
        if volume <= 0.0:
            continue
        rows.append((audio_path, max(0, int(round(output_sec * 1000.0))), volume))
    return rows


def _attach_voiceover_audio(
    plan: ExportFilterGraphPlan,
    *,
    voiceover_segments: Optional[List[VoiceoverSegment]],
    video_segments: Optional[List[VideoSegment]],
    timeline_frames: Optional[List[TimelineFrame]],
    duration_ms: float,
    screen_transitions: Optional[List[ScreenTransition]] = None,
) -> None:
    """Compatibility wrapper for attaching voiceover without background music."""
    rows = _voiceover_audio_rows(
        voiceover_segments,
        video_segments,
        timeline_frames,
        duration_ms,
        screen_transitions,
    )
    AudioMixBuilder().attach(plan, voiceover_rows=rows)


class VideoExporter:
    """Export pipeline for rendering Zumly sessions to MP4 (Phase 5 Motion Engine)."""

    def __init__(
        self,
        progress_cb: Optional[Callable[[float], None]] = None,
        finished_cb: Optional[Callable[[str], None]] = None,
        error_cb: Optional[Callable[[str], None]] = None,
        status_cb: Optional[Callable[[str], None]] = None,
        result_cb: Optional[Callable[[ExportResult], None]] = None,
    ) -> None:
        self._progress_cb = progress_cb
        self._finished_cb = finished_cb
        self._error_cb = error_cb
        self._status_cb = status_cb
        self._result_cb = result_cb
        self._thread: Optional[threading.Thread] = None
        self._last_result: Optional[ExportResult] = None

    @staticmethod
    def visible_timeline_overlays(
        mapper: EditedTimelineMapper,
        overlays: List[TimelineOverlay] | None,
        output_time_ms: float,
    ) -> List[TimelineOverlay]:
        """Use the preview's exact overlay visibility contract at export time.

        The first visual overlay renderer will consume this from the graph
        builder. Keeping it here now makes the parity contract executable and
        testable before any annotation tool is exposed.
        """
        return visible_overlays_at_output_time(mapper, overlays, output_time_ms)

    def export(
        self,
        input_path: str,
        output_path: str,
        keyframes: List[ZoomKeyframe],
        actual_fps: float = 0.0,
        mouse_track: Optional[List[MousePosition]] = None,
        monitor_rect: Optional[dict] = None,
        bg_preset: Optional[BackgroundPreset] = None,
        frame_preset: Optional[FramePreset] = None,
        target_resolution: Optional[tuple[int, int]] = None,
        click_events: Optional[List[ClickEvent]] = None,
        click_preset: Optional[ClickEffectPreset] = None,
        duration_ms: float = 0.0,
        frame_timestamps: Optional[List[float]] = None,
        trim_start_ms: float = 0.0,
        trim_end_ms: float = 0.0,
        encoder_id: str = "libx264",
        voiceover_segments: Optional[List[VoiceoverSegment]] = None,
        video_segments: Optional[List[VideoSegment]] = None,
        timeline_frames: Optional[List[TimelineFrame]] = None,
        screen_transitions: Optional[List[ScreenTransition]] = None,
        highlights: Optional[List[HighlightBox]] = None,
        text_annotations: Optional[List[TextAnnotation]] = None,
        timeline_overlays: Optional[List[TimelineOverlay]] = None,
        cursor_asset_path: str = "",
        cursor_style_id: str = "arrow",
        cursor_hotspot: tuple[float, float] | None = None,
        cursor_scale: float = DEFAULT_CURSOR_SCALE,
        is_cfr: bool = False,
        canvas_layout_scenes: Optional[List[CanvasLayoutScene]] = None,
        explainer_scenes: Optional[List[ExplainerScene]] = None,
        background_music: BackgroundMusic | None = None,
        wait: bool = False,
    ) -> Optional[ExportResult]:
        if not input_path or not output_path:
            result = ExportResult(
                success=False,
                output_path=str(output_path or ""),
                error_message="Input and output paths are required.",
                ffmpeg_exit_code=-1,
            )
            self._publish_result(result)
            return result if wait else None
        try:
            same_file = os.path.normcase(os.path.realpath(input_path)) == os.path.normcase(
                os.path.realpath(output_path)
            )
        except OSError:
            same_file = False
        if same_file:
            result = ExportResult(
                success=False,
                output_path=str(output_path),
                error_message="Input and output paths must be different.",
                ffmpeg_exit_code=-1,
            )
            self._publish_result(result)
            return result if wait else None
        effective_segments = _apply_trim_to_video_segments(
            video_segments,
            duration_ms,
            trim_start_ms,
            trim_end_ms,
        )
        self._thread = threading.Thread(
            target=self._run,
            args=(
                input_path,
                output_path,
                bg_preset,
                frame_preset,
                target_resolution,
                duration_ms,
                frame_timestamps,
                is_cfr,
                keyframes,
                mouse_track,
                click_events,
                click_preset,
                actual_fps,
                monitor_rect,
                effective_segments,
                timeline_frames,
                highlights,
                encoder_id,
                voiceover_segments,
                text_annotations,
                cursor_asset_path,
                cursor_style_id,
                cursor_hotspot,
                cursor_scale,
                canvas_layout_scenes,
                explainer_scenes,
                screen_transitions,
                background_music,
                timeline_overlays,
            ),
            daemon=True,
        )
        self._thread.start()
        if wait:
            self._thread.join()
            return self._last_result
        return None

    def _publish_result(self, result: ExportResult) -> None:
        """Store the result and preserve the existing callback API."""
        self._last_result = result
        try:
            if result.success:
                if self._finished_cb:
                    self._finished_cb(result.output_path)
            elif self._error_cb:
                self._error_cb(result.error_message or "Export failed")
            if self._result_cb:
                self._result_cb(result)
        except Exception:
            logger.exception("Export result callback failed")

    def _build_filtergraph(
        self,
        *,
        bg_preset: Optional[BackgroundPreset],
        frame_preset: Optional[FramePreset],
        target_resolution: Optional[tuple[int, int]],
        duration_ms: float,
        frame_timestamps: Optional[List[float]],
        keyframes: List[ZoomKeyframe],
        mouse_track: Optional[List[MousePosition]],
        click_events: Optional[List[ClickEvent]],
        click_preset: Optional[ClickEffectPreset],
        monitor_rect: Optional[dict],
        video_segments: Optional[List[VideoSegment]],
        timeline_frames: Optional[List[TimelineFrame]],
        highlights: Optional[List[HighlightBox]],
        screen_transitions: Optional[List[ScreenTransition]] = None,
        text_annotations: Optional[List[TextAnnotation]] = None,
        timeline_overlays: Optional[List[TimelineOverlay]] = None,
        cursor_asset_path: str = "",
        cursor_style_id: str = "arrow",
        cursor_hotspot: tuple[float, float] | None = None,
        cursor_scale: float = DEFAULT_CURSOR_SCALE,
        src_w: int,
        src_h: int,
        src_fps: float,
        total_sec: float,
        is_cfr: bool = False,
        source_has_audio: bool = False,
        layout_transform: Optional[LayoutSpaceTransform] = None,
        canvas_layout_scenes: Optional[List[CanvasLayoutScene]] = None,
        explainer_scenes: Optional[List[ExplainerScene]] = None,
    ) -> ExportFilterGraphPlan:
        """Build a synthetic export graph for tests or future _run refactors."""
        return _build_export_filtergraph(
            bg_preset=bg_preset,
            frame_preset=frame_preset,
            target_resolution=target_resolution,
            duration_ms=duration_ms,
            frame_timestamps=frame_timestamps,
            keyframes=keyframes,
            mouse_track=mouse_track,
            click_events=click_events,
            click_preset=click_preset,
            monitor_rect=monitor_rect,
            video_segments=video_segments,
            timeline_frames=timeline_frames,
            screen_transitions=screen_transitions,
            highlights=highlights,
            text_annotations=text_annotations,
            timeline_overlays=timeline_overlays,
            cursor_asset_path=cursor_asset_path,
            cursor_style_id=cursor_style_id,
            cursor_hotspot=cursor_hotspot,
            cursor_scale=cursor_scale,
            src_w=src_w,
            src_h=src_h,
            src_fps=src_fps,
            total_sec=total_sec,
            is_cfr=is_cfr,
            source_has_audio=source_has_audio,
            layout_transform=layout_transform,
            canvas_layout_scenes=canvas_layout_scenes,
            explainer_scenes=explainer_scenes,
        )

    def _probe_source(
        self,
        ffmpeg: str,
        input_path: str,
        actual_fps: float,
        duration_ms: float,
    ) -> ExportSourceProbe:
        """Probe source geometry, framerate, and duration for export setup."""
        ffprobe_cmd = [ffmpeg, "-i", input_path]
        p = subprocess.run(ffprobe_cmd, capture_output=True, text=True, **_subprocess_kwargs())

        src_w, src_h = 1920, 1080
        src_fps = 30.0

        m = re.search(r"Video:.* (\d{3,5})x(\d{3,5})", p.stderr)
        if m:
            src_w, src_h = int(m.group(1)), int(m.group(2))

        fps_m = re.search(r"(\d+(?:\.\d+)?) fps", p.stderr)
        if fps_m:
            src_fps = float(fps_m.group(1))
        elif actual_fps > 0:
            src_fps = actual_fps

        dur_m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", p.stderr)
        total_sec = 0.0
        if dur_m:
            total_sec = int(dur_m.group(1)) * 3600 + int(dur_m.group(2)) * 60 + float(dur_m.group(3))
        elif duration_ms:
            total_sec = duration_ms / 1000.0

        has_audio = bool(
            re.search(r"^\s*Stream #\d+:\d+.*:\s*Audio:", p.stderr, re.MULTILINE)
        )
        return ExportSourceProbe(
            src_w=src_w,
            src_h=src_h,
            src_fps=src_fps,
            total_sec=total_sec,
            has_audio=has_audio,
        )

    def _write_filtergraph_script(self, filtergraph: str, temp_files: List[str]) -> str:
        """Persist filtergraph to a script file to bypass Windows CLI limits."""
        graph_path = _new_temp_asset_path(".txt")
        temp_files.append(graph_path)

        with open(graph_path, "w", encoding="utf-8") as f:
            f.write(filtergraph)
        logger.info("FFmpeg filter graph retained for debugging: %s", graph_path)
        return graph_path

    def _build_ffmpeg_command(
        self,
        *,
        ffmpeg: str,
        input_path: str,
        output_path: str,
        graph_path: str,
        plan: ExportFilterGraphPlan,
        encoder_id: str,
    ) -> List[str]:
        """Build the FFmpeg argv from a prepared filtergraph plan."""
        cmd = [
            ffmpeg, "-y",
            "-i", input_path,
            "-loop", "1", "-t", "0.100",
            "-i", plan.frame_img_path,
        ]
        if plan.click_img_path:
            cmd.extend(["-loop", "1", "-t", "0.100", "-i", plan.click_img_path])
        if plan.cursor_img_path:
            cmd.extend(["-loop", "1", "-t", "0.100", "-i", plan.cursor_img_path])
        for annotation_path in getattr(plan, "video_annotation_img_paths", []):
            cmd.extend(["-loop", "1", "-i", annotation_path])
        for highlight_path in plan.highlight_img_paths:
            cmd.extend(["-loop", "1", "-i", highlight_path])
        for text_path in getattr(plan, "text_annotation_img_paths", []):
            cmd.extend(["-loop", "1", "-i", text_path])
        for frame_path in plan.timeline_frame_img_paths:
            cmd.extend(["-loop", "1", "-i", frame_path])
        background_img_path = getattr(plan, "background_img_path", "")
        if background_img_path:
            cmd.extend(["-loop", "1", "-t", "0.100", "-i", background_img_path])
        for text_path in getattr(plan, "transition_text_img_paths", []):
            cmd.extend(["-loop", "1", "-i", text_path])
        audio_specs = list(getattr(plan, "audio_input_specs", []) or [])
        if audio_specs:
            for audio_spec in audio_specs:
                if audio_spec.stream_loop:
                    cmd.extend(["-stream_loop", "-1"])
                cmd.extend(["-i", audio_spec.path])
        else:
            for audio_path in plan.voiceover_audio_paths:
                cmd.extend(["-i", audio_path])

        cmd.extend([
            "-filter_complex_script", graph_path,
            "-map", "[out]",
        ])
        cmd.extend(build_encoder_args(encoder_id or "libx264"))
        if getattr(plan, "has_audio_output", False) and getattr(plan, "audio_output_node", ""):
            cmd.extend([
                "-map", f"[{plan.audio_output_node}]",
                "-c:a", "aac",
                "-b:a", "192k",
            ])
        elif getattr(plan, "has_source_audio", False):
            logger.warning("Source audio was detected but no synchronized audio output was built.")
        else:
            logger.debug("Source contains no audio stream; exporting video only.")
        if plan.output_total_sec > 0:
            cmd.extend(["-t", f"{plan.output_total_sec:.3f}"])
        # Staged files end in .tmp, so do not rely on suffix-based muxer
        # detection. The committed destination remains a normal .mp4 path.
        cmd.extend(["-f", "mp4", output_path])
        return cmd

    def _execute_ffmpeg_command(
        self,
        cmd: List[str],
        output_path: str,
        output_total_sec: float,
        encoder_id: str = "",
    ) -> ExportResult:
        """Run FFmpeg and stream progress callbacks."""
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                **_subprocess_kwargs()
            )
        except Exception as exc:
            return ExportResult(
                success=False,
                output_path=output_path,
                error_message=f"Could not start FFmpeg: {exc}",
                ffmpeg_exit_code=-1,
                encoder_id=encoder_id,
            )

        stderr_tail = []
        try:
            while True:
                line = proc.stderr.readline()
                if not line:
                    break
                stderr_tail.append(line)
                if len(stderr_tail) > 80:
                    stderr_tail = stderr_tail[-80:]

                time_m = re.search(r"time=(\d+):(\d+):(\d+\.\d+)", line)
                if time_m and output_total_sec > 0:
                    curr_sec = int(time_m.group(1)) * 3600 + int(time_m.group(2)) * 60 + float(time_m.group(3))
                    prog = min(1.0, curr_sec / output_total_sec)
                    if self._progress_cb:
                        self._progress_cb(prog)
        finally:
            proc.wait()
            for stream_name in ("stdout", "stderr"):
                stream = getattr(proc, stream_name, None)
                close = getattr(stream, "close", None)
                if close is not None:
                    try:
                        close()
                    except OSError:
                        logger.debug("Could not close FFmpeg %s pipe", stream_name, exc_info=True)

        if proc.returncode != 0:
            stderr_excerpt = "".join(stderr_tail)[-4000:]
            logger.error("FFmpeg export failed with return code %d. Stderr: %s", proc.returncode, stderr_excerpt)
            message = f"FFmpeg export failed with exit code {proc.returncode}."
            if stderr_excerpt.strip():
                message += f" {stderr_excerpt.strip()[-2000:]}"
            return ExportResult(
                success=False,
                output_path=output_path,
                error_message=message,
                ffmpeg_exit_code=int(proc.returncode),
                encoder_id=encoder_id,
            )

        if self._progress_cb:
            self._progress_cb(1.0)
        if not os.path.isfile(output_path) or os.path.getsize(output_path) <= 0:
            return ExportResult(
                success=False,
                output_path=output_path,
                error_message="FFmpeg exited successfully but produced no usable output file.",
                ffmpeg_exit_code=0,
                encoder_id=encoder_id,
            )
        logger.info("Export completed successfully: %s", output_path)
        return ExportResult(
            success=True,
            output_path=output_path,
            ffmpeg_exit_code=0,
            encoder_id=encoder_id,
        )

    def _cleanup_temp_files(
        self,
        temp_files: List[str],
        temp_dirs: Optional[List[str]] = None,
    ) -> None:
        for path in temp_files:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning("Could not remove export temp file %s: %s", path, exc)
        for path in temp_dirs or []:
            try:
                shutil.rmtree(path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning("Could not remove cursor sequence directory %s: %s", path, exc)

    def _remove_failed_output(self, input_path: str, output_path: str) -> None:
        if not output_path:
            return
        try:
            if os.path.abspath(input_path) == os.path.abspath(output_path):
                return
            if os.path.isfile(output_path):
                os.remove(output_path)
        except OSError as exc:
            logger.warning("Could not remove failed export output %s: %s", output_path, exc)

    def _run(self, input_path: str, output_path: str, bg_preset: BackgroundPreset, frame_preset: FramePreset, target_resolution: Optional[tuple[int, int]], duration_ms: float, frame_timestamps: Optional[List[float]], is_cfr: bool, keyframes: List[ZoomKeyframe], mouse_track: Optional[List[MousePosition]], click_events: Optional[List[ClickEvent]], click_preset: Optional[ClickEffectPreset], actual_fps: float, monitor_rect: Optional[dict], video_segments: Optional[List[VideoSegment]], timeline_frames: Optional[List[TimelineFrame]], highlights: Optional[List[HighlightBox]], encoder_id: str, voiceover_segments: Optional[List[VoiceoverSegment]], text_annotations: Optional[List[TextAnnotation]] = None, cursor_asset_path: str = "", cursor_style_id: str = "arrow", cursor_hotspot: tuple[float, float] | None = None, cursor_scale: float = DEFAULT_CURSOR_SCALE, canvas_layout_scenes: Optional[List[CanvasLayoutScene]] = None, explainer_scenes: Optional[List[ExplainerScene]] = None, screen_transitions: Optional[List[ScreenTransition]] = None, background_music: BackgroundMusic | None = None, timeline_overlays: Optional[List[TimelineOverlay]] = None):
        temp_files: List[str] = []
        temp_dirs: List[str] = []
        staged_output_path = export_staging_path(output_path)
        requested_encoder_id = str(encoder_id or "libx264")
        result: Optional[ExportResult] = None
        try:
            self._remove_failed_output(input_path, staged_output_path)
            if self._status_cb:
                self._status_cb("Starting export...")

            ffmpeg = _ffmpeg_exe()
            probe = self._probe_source(ffmpeg, input_path, actual_fps, duration_ms)
            try:
                plan = self._build_filtergraph(
                    bg_preset=bg_preset,
                    frame_preset=frame_preset,
                    target_resolution=target_resolution,
                    duration_ms=duration_ms,
                    frame_timestamps=frame_timestamps,
                    keyframes=keyframes,
                    mouse_track=mouse_track,
                    click_events=click_events,
                    click_preset=click_preset,
                    monitor_rect=monitor_rect,
                    video_segments=video_segments,
                    timeline_frames=timeline_frames,
                    screen_transitions=screen_transitions,
                    highlights=highlights,
                    text_annotations=text_annotations,
                    timeline_overlays=timeline_overlays,
                    cursor_asset_path=cursor_asset_path,
                    cursor_style_id=cursor_style_id,
                    cursor_hotspot=cursor_hotspot,
                    cursor_scale=cursor_scale,
                    src_w=probe.src_w,
                    src_h=probe.src_h,
                    src_fps=probe.src_fps,
                    total_sec=probe.total_sec,
                    is_cfr=is_cfr,
                    source_has_audio=probe.has_audio,
                    canvas_layout_scenes=canvas_layout_scenes,
                    explainer_scenes=explainer_scenes,
                )
            except ValueError as exc:
                result = ExportResult(
                    success=False,
                    output_path=output_path,
                    error_message=str(exc),
                    ffmpeg_exit_code=-1,
                    requested_encoder_id=requested_encoder_id,
                    encoder_id=requested_encoder_id,
                )
                return result

            voiceover_rows = _voiceover_audio_rows(
                voiceover_segments,
                video_segments,
                timeline_frames,
                duration_ms,
                screen_transitions,
            )
            AudioMixBuilder().attach(
                plan,
                voiceover_rows=voiceover_rows,
                background_music=background_music,
            )

            temp_files.extend(plan.temp_files)
            temp_dirs.extend(getattr(plan, "temp_dirs", []))
            graph_path = self._write_filtergraph_script(plan.filtergraph, temp_files)
            cmd = self._build_ffmpeg_command(
                ffmpeg=ffmpeg,
                input_path=input_path,
                output_path=staged_output_path,
                graph_path=graph_path,
                plan=plan,
                encoder_id=encoder_id,
            )

            logger.info("Running FFmpeg with graph: %s", graph_path)
            result = self._execute_ffmpeg_command(
                cmd,
                staged_output_path,
                plan.output_total_sec,
                encoder_id=requested_encoder_id,
            )

            # Hardware initialization failures can happen after FFmpeg has
            # started, so capability detection alone cannot guarantee a
            # successful export. Retry the same prepared graph with the
            # deterministic software encoder before surfacing an error.
            if not result.success and _should_retry_hardware_encoder(
                requested_encoder_id,
                result.error_message,
            ):
                first_error = result.error_message
                self._remove_failed_output(input_path, staged_output_path)
                warning = (
                    f"Hardware encoder {requested_encoder_id} failed; "
                    "retrying export with libx264."
                )
                logger.warning(warning)
                if self._status_cb:
                    self._status_cb(warning)
                fallback_cmd = self._build_ffmpeg_command(
                    ffmpeg=ffmpeg,
                    input_path=input_path,
                    output_path=staged_output_path,
                    graph_path=graph_path,
                    plan=plan,
                    encoder_id="libx264",
                )
                fallback_result = self._execute_ffmpeg_command(
                    fallback_cmd,
                    staged_output_path,
                    plan.output_total_sec,
                    encoder_id="libx264",
                )
                if fallback_result.success:
                    result = replace(
                        fallback_result,
                        requested_encoder_id=requested_encoder_id,
                        error_message=(
                            f"{warning} Software fallback succeeded."
                        ),
                        fallback_used=True,
                    )
                else:
                    result = replace(
                        fallback_result,
                        requested_encoder_id=requested_encoder_id,
                        error_message=(
                            f"{warning} Software fallback also failed. "
                            f"Hardware error: {first_error} Fallback error: "
                            f"{fallback_result.error_message}"
                        ),
                        fallback_used=True,
                    )

            if result.success:
                os.replace(staged_output_path, output_path)
                result = replace(result, output_path=output_path)

        except Exception as exc:
            logger.exception("Export crashed")
            result = ExportResult(
                success=False,
                output_path=output_path,
                error_message=str(exc),
                ffmpeg_exit_code=-1,
                requested_encoder_id=requested_encoder_id,
                encoder_id=requested_encoder_id,
            )
        finally:
            self._cleanup_temp_files(temp_files, temp_dirs)
            if result is None:
                result = ExportResult(
                    success=False,
                    output_path=output_path,
                    error_message="Export ended without a result.",
                    ffmpeg_exit_code=-1,
                    requested_encoder_id=requested_encoder_id,
                    encoder_id=requested_encoder_id,
                )
            if not result.success:
                self._remove_failed_output(input_path, staged_output_path)
            self._publish_result(result)
        return result
