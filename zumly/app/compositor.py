"""Compositor — renders a screen recording inside a drawn device bezel.

Used by both the live preview widget and the video exporter so the
exported MP4 looks identical to the in-app preview.
"""

from dataclasses import dataclass, replace
from typing import List, Optional

from PySide6.QtCore import QRectF, QPointF, Qt
from PySide6.QtGui import (
    QImage,
    QPainter,
    QColor,
    QPen,
    QBrush,
    QPainterPath,
)

from .models import (
    CanvasLayoutScene,
    MousePosition,
    TextAnnotation,
    TimelineOverlay,
    interpolated_canvas_layout_scene,
    DEFAULT_CURSOR_SCALE,
)
from .geometry_math import (
    CanvasSpaceTransform,
    LayoutSpaceTransform,
    Point2D,
    PresentationGroupGeometry,
    Rect2D,
    VideoSpaceTransform,
)
from .text_renderer import aligned_offset, canvas_text_metrics, layout_canvas_text
from .backgrounds import BackgroundPreset, DEFAULT_PRESET
from .background_renderer import paint_background as _paint_bg
from .frames import FramePreset, FRAME_PRESETS, DEFAULT_FRAME
from .models import ClickEffectPreset, DEFAULT_CLICK_EFFECT
from .video_masks import apply_video_space_masks
from .video_annotations import apply_video_space_annotations

# Re-export ClickEvent so compositor callers don't need to import models
from .models import ClickEvent

# ── Visual constants ────────────────────────────────────────────────

# Device frame bezel
BEZEL_COLOR     = QColor("#1a1a1a")       # dark bezel body
BEZEL_EDGE      = QColor("#6b6b6b")       # silver outer rim
BEZEL_HIGHLIGHT = QColor(255, 255, 255, 18)  # subtle top/left highlight
OUTER_RADIUS    = 18.0                     # corner radius of device
CAMERA_DOT_R    = 3.0                      # front camera
CAMERA_COLOR    = QColor("#333333")

# Padding around device frame (fraction of canvas)
DEVICE_PAD = 0.04
# Device aspect ratio — derived from video frame + bezel at runtime
# Fallback used only for empty state (no video)
DEVICE_ASPECT_FALLBACK = 16.0 / 9.0
COMPOSITOR_LAYER_ORDER = ("background", "presentation_group", "canvas_text")
NO_FRAME = next((preset for preset in FRAME_PRESETS if preset.is_none), DEFAULT_FRAME)


@dataclass(frozen=True)
class CursorRenderRequest:
    """Cursor placement expressed relative to the final output canvas.

    The interactive preview may render the scene into a bounded offscreen
    image.  Keeping cursor placement normalized lets it be rasterized directly
    into the actual widget viewport instead of being enlarged with that image.
    """

    mouse_track: List[MousePosition]
    time_ms: float
    monitor_rect: dict
    screen_x: float
    screen_y: float
    screen_w: float
    screen_h: float
    video_transform: VideoSpaceTransform
    cursor_asset_path: str
    cursor_style_id: str
    cursor_hotspot: tuple[float, float] | None
    cursor_scale: float
    click_events: List[ClickEvent]
    click_preset: ClickEffectPreset | None


@dataclass(frozen=True)
class PresentationGroupRender:
    image: QImage
    geometry: PresentationGroupGeometry
    cursor_request: CursorRenderRequest | None


def draw_scene_cursor(
    painter: QPainter,
    request: CursorRenderRequest,
    canvas_x: float,
    canvas_y: float,
    canvas_w: float,
    canvas_h: float,
) -> None:
    """Render a deferred cursor request at its final viewport resolution."""
    from .cursor_renderer import draw_cursor_qpainter

    screen_x = float(canvas_x) + request.screen_x * float(canvas_w)
    screen_y = float(canvas_y) + request.screen_y * float(canvas_h)
    screen_w = request.screen_w * float(canvas_w)
    screen_h = request.screen_h * float(canvas_h)

    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
    painter.setClipRect(QRectF(screen_x, screen_y, screen_w, screen_h))
    try:
        draw_cursor_qpainter(
            painter,
            request.mouse_track,
            request.time_ms,
            request.monitor_rect,
            screen_x,
            screen_y,
            screen_w,
            screen_h,
            video_transform=request.video_transform,
            cursor_asset_path=request.cursor_asset_path,
            cursor_style_id=request.cursor_style_id,
            cursor_hotspot=request.cursor_hotspot,
            cursor_scale=request.cursor_scale,
            click_events=request.click_events,
        )
    finally:
        painter.restore()


def draw_scene_clicks(
    painter: QPainter,
    request: CursorRenderRequest,
    canvas_x: float,
    canvas_y: float,
    canvas_w: float,
    canvas_h: float,
) -> None:
    """Paint transient click effects without recompositing the video scene."""
    if not request.click_events or not request.monitor_rect:
        return

    from .cursor_renderer import draw_clicks_qpainter

    screen_x = float(canvas_x) + request.screen_x * float(canvas_w)
    screen_y = float(canvas_y) + request.screen_y * float(canvas_h)
    screen_w = request.screen_w * float(canvas_w)
    screen_h = request.screen_h * float(canvas_h)
    if screen_w <= 0.0 or screen_h <= 0.0:
        return

    painter.save()
    painter.setClipRect(QRectF(screen_x, screen_y, screen_w, screen_h))
    try:
        draw_clicks_qpainter(
            painter,
            request.click_events,
            request.time_ms,
            request.monitor_rect,
            screen_x,
            screen_y,
            screen_w,
            screen_h,
            request.click_preset or DEFAULT_CLICK_EFFECT,
            video_transform=request.video_transform,
        )
    finally:
        painter.restore()


def render_presentation_group(
    frame: QImage,
    canvas_w: int,
    canvas_h: int,
    *,
    layout_scene: CanvasLayoutScene | None = None,
    **scene_kwargs,
) -> PresentationGroupRender:
    """Rasterize video and bezel once, independently of layout animation."""
    width = max(1, int(canvas_w))
    height = max(1, int(canvas_h))
    image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QColor(0, 0, 0, 0))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    geometry_out: dict[str, float] = {}
    identity_scene = None
    if layout_scene is not None:
        identity_scene = replace(
            layout_scene,
            video_scale=1.0,
            video_x=0.0,
            video_y=0.0,
            transition="cut",
            transition_duration_ms=0.0,
        )
    try:
        request = compose_scene(
            painter,
            frame,
            width,
            height,
            canvas_layout_scene=identity_scene,
            canvas_layout_scenes=None,
            geometry_out=geometry_out,
            draw_background=False,
            draw_canvas_text=False,
            draw_cursor=False,
            draw_clicks=False,
            **scene_kwargs,
        )
    finally:
        painter.end()

    layout = (
        LayoutSpaceTransform(
            x=float(layout_scene.video_x),
            y=float(layout_scene.video_y),
            width=float(layout_scene.video_scale),
            height=float(layout_scene.video_scale),
        )
        if layout_scene is not None
        else LayoutSpaceTransform.identity()
    )
    geometry = PresentationGroupGeometry.create(
        width,
        height,
        Rect2D(
            geometry_out.get("device_x", 0.0) * width,
            geometry_out.get("device_y", 0.0) * height,
            geometry_out.get("device_w", 1.0) * width,
            geometry_out.get("device_h", 1.0) * height,
        ),
        Rect2D(
            geometry_out.get("screen_x", 0.0) * width,
            geometry_out.get("screen_y", 0.0) * height,
            geometry_out.get("screen_w", 1.0) * width,
            geometry_out.get("screen_h", 1.0) * height,
        ),
        layout,
    )
    if request is not None:
        mapped = geometry.mapped_video_aperture
        request = replace(
            request,
            screen_x=mapped.x / width,
            screen_y=mapped.y / height,
            screen_w=mapped.width / width,
            screen_h=mapped.height / height,
        )
    return PresentationGroupRender(image, geometry, request)


# ── Public API ──────────────────────────────────────────────────────


def compose_scene(
    painter: QPainter,
    frame: QImage,
    canvas_w: float,
    canvas_h: float,
    zoom: float = 1.0,
    pan_x: float = 0.5,
    pan_y: float = 0.5,
    mouse_track: Optional[List[MousePosition]] = None,
    time_ms: float = 0.0,
    monitor_rect: Optional[dict] = None,
    bg_preset: Optional[BackgroundPreset] = None,
    frame_preset: Optional[FramePreset] = None,
    click_events: Optional[List[ClickEvent]] = None,
    click_preset: Optional[ClickEffectPreset] = None,
    cursor_asset_path: str = "",
    cursor_style_id: str = "arrow",
    cursor_hotspot: tuple[float, float] | None = None,
    cursor_scale: float = DEFAULT_CURSOR_SCALE,
    layout_transform: Optional[LayoutSpaceTransform] = None,
    text_annotations: Optional[List[TextAnnotation]] = None,
    ephemeral_text_annotation: Optional[TextAnnotation] = None,
    canvas_layout_scene: Optional[CanvasLayoutScene] = None,
    canvas_layout_scenes: Optional[List[CanvasLayoutScene]] = None,
    video_masks: Optional[List[TimelineOverlay]] = None,
    video_annotations: Optional[List[TimelineOverlay]] = None,
    text_dpi_scale: float = 1.0,
    layer_trace: Optional[list[str]] = None,
    geometry_out: Optional[dict[str, float]] = None,
    *,
    draw_background: bool = True,
    draw_video: bool = True,
    draw_device_frame: bool = True,
    draw_cursor: bool = True,
    draw_clicks: bool = True,
    draw_canvas_text: bool = True,
) -> CursorRenderRequest | None:
    """Paint the device-frame composition onto *painter*.

    Draws:  gradient background  →  device bezel  →  video inside screen
            →  mouse cursor (if mouse_track provided).

    When *zoom* > 1 the device scales up and pans so the point
    (*pan_x*, *pan_y*) in the video stays centred in the output.
    """
    # Redact Video Space before any zoom/cursor/frame composition.  The
    # returned image is owned, so QVideoSink-backed frames are never mutated.
    if draw_video:
        frame = apply_video_space_annotations(frame, video_annotations)
        frame = apply_video_space_masks(frame, video_masks)
    W, H = float(canvas_w), float(canvas_h)
    iw, ih = frame.width(), frame.height()
    if iw <= 0 or ih <= 0:
        return None

    fp = frame_preset or DEFAULT_FRAME
    has_layout_scenes = canvas_layout_scene is not None or bool(canvas_layout_scenes)
    active_layout_scene = None
    if has_layout_scenes:
        active_layout_scene = canvas_layout_scene or interpolated_canvas_layout_scene(
            canvas_layout_scenes,
            time_ms,
            0.0,
        )
        if not active_layout_scene.device_frame_visible:
            fp = NO_FRAME

    # ── background ─────────────────────────────────────────────────
    preset = bg_preset or DEFAULT_PRESET
    if draw_background:
        if active_layout_scene and active_layout_scene.background_color:
            scene_color = QColor(active_layout_scene.background_color)
            if scene_color.isValid():
                painter.fillRect(QRectF(0, 0, W, H), scene_color)
            else:
                _paint_bg(painter, W, H, preset)
        else:
            _paint_bg(painter, W, H, preset)
    if layer_trace is not None and draw_background:
        layer_trace.append("background")

    # ── fit device into canvas ──────────────────────────────────────
    video_aspect = iw / ih
    dev_pad = fp.padding

    if fp.is_none:
        # No frame — video fills the entire canvas
        scr_x, scr_y = 0.0, 0.0
        scr_w, scr_h = W, H
        # Letterbox / pillarbox to maintain aspect
        if W / H > video_aspect:
            scr_h = H
            scr_w = H * video_aspect
        else:
            scr_w = W
            scr_h = W / video_aspect
        scr_x = (W - scr_w) / 2
        scr_y = (H - scr_h) / 2
        dev_x, dev_y, dev_w, dev_h = scr_x, scr_y, scr_w, scr_h
        bw = 0.0
        outer_r = 0.0
        inner_r = 0.0
        scale = 1.0
    else:
        preliminary_scale = (W - 2 * W * dev_pad) / 900.0
        bw_est = fp.bezel_width * preliminary_scale
        pad_x = W * dev_pad
        pad_y = H * dev_pad
        avail_w = W - 2 * pad_x
        avail_h = H - 2 * pad_y

        dev_h = avail_h
        scr_h_try = dev_h - 2 * bw_est
        scr_w_try = scr_h_try * video_aspect
        dev_w = scr_w_try + 2 * bw_est
        if dev_w > avail_w:
            dev_w = avail_w
            scr_w_try = dev_w - 2 * bw_est
            scr_h_try = scr_w_try / video_aspect
            dev_h = scr_h_try + 2 * bw_est

        dev_x = (W - dev_w) / 2
        dev_y = (H - dev_h) / 2
        scale = dev_w / 900.0
        bw = fp.bezel_width * scale
        outer_r = fp.outer_radius * scale
        inner_r = fp.inner_radius * scale

        scr_x = dev_x + bw
        scr_y = dev_y + bw
        scr_w = dev_w - 2 * bw
        scr_h = dev_h - 2 * bw

    # Layout Space moves the complete presentation group together.  Canvas
    # annotations are painted later and therefore never inherit this transform.
    if active_layout_scene:
        layout = LayoutSpaceTransform(
            x=active_layout_scene.video_x,
            y=active_layout_scene.video_y,
            width=active_layout_scene.video_scale,
            height=active_layout_scene.video_scale,
        )
    else:
        layout = layout_transform or LayoutSpaceTransform.identity()
    if layer_trace is not None:
        layer_trace.append("presentation_group")

    def map_layout_rect(x: float, y: float, width: float, height: float) -> Rect2D:
        normalized = Rect2D(x / max(W, 1.0), y / max(H, 1.0), width / max(W, 1.0), height / max(H, 1.0))
        mapped = layout.map_rect(normalized)
        return Rect2D(mapped.x * W, mapped.y * H, mapped.width * W, mapped.height * H)

    device_group_rect = map_layout_rect(dev_x, dev_y, dev_w, dev_h)
    screen_group_rect = map_layout_rect(scr_x, scr_y, scr_w, scr_h)
    layout_scale = max(0.01, min(float(layout.width), float(layout.height)))
    dev_x, dev_y, dev_w, dev_h = (
        device_group_rect.x,
        device_group_rect.y,
        device_group_rect.width,
        device_group_rect.height,
    )
    scr_x, scr_y, scr_w, scr_h = (
        screen_group_rect.x,
        screen_group_rect.y,
        screen_group_rect.width,
        screen_group_rect.height,
    )
    bw *= layout_scale
    outer_r *= layout_scale
    inner_r *= layout_scale
    scale *= layout_scale
    if geometry_out is not None:
        geometry_out.update({
            "device_x": dev_x / max(W, 1.0),
            "device_y": dev_y / max(H, 1.0),
            "device_w": dev_w / max(W, 1.0),
            "device_h": dev_h / max(H, 1.0),
            "screen_x": scr_x / max(W, 1.0),
            "screen_y": scr_y / max(H, 1.0),
            "screen_w": scr_w / max(W, 1.0),
            "screen_h": scr_h / max(H, 1.0),
        })

    # ── zoom ─────────────────────────────────────────────────────────
    # Zoom belongs exclusively to Video Space. Backgrounds, device frames,
    # and future Canvas Space annotations remain fixed on the output canvas.
    video_transform = VideoSpaceTransform(zoom=zoom, pan_x=pan_x, pan_y=pan_y)
    crop_x, crop_y, crop_w, crop_h = video_transform.viewport()
    source_rect = QRectF(crop_x * iw, crop_y * ih, crop_w * iw, crop_h * ih)

    # ── device body (outer shell) ───────────────────────────────────
    if draw_device_frame and not fp.is_none:
        device_rect = QRectF(dev_x, dev_y, dev_w, dev_h)

        # Drop shadow
        for i in range(fp.shadow_layers):
            shadow_off = 2 + i * 2
            shadow_rect = QRectF(dev_x + shadow_off * 0.3, dev_y + shadow_off, dev_w, dev_h)
            sp = QPainterPath()
            sp.addRoundedRect(shadow_rect, outer_r + 2, outer_r + 2)
            painter.fillPath(sp, QColor(0, 0, 0, max(40 - i * 10, 5)))

        if bw > 0:
            # Outer edge + bezel body
            bezel_c = QColor(*fp.bezel_color)
            edge_c = QColor(*fp.edge_color)
            painter.setPen(QPen(edge_c, max(fp.edge_width * scale, 0.5)))
            painter.setBrush(QBrush(bezel_c))
            painter.drawRoundedRect(device_rect, outer_r, outer_r)

            # Subtle highlight on top edge
            highlight_rect = QRectF(dev_x + outer_r, dev_y + 0.5, dev_w - 2 * outer_r, 1.0)
            painter.fillRect(highlight_rect, BEZEL_HIGHLIGHT)

    # ── screen area ─────────────────────────────────────────────────
    screen_rect = QRectF(scr_x, scr_y, scr_w, scr_h)
    aperture_overlap = 1 if draw_device_frame and not fp.is_none and bw > 0 else 0
    aperture_pixels = PresentationGroupGeometry.bezel_covered_aperture(
        Rect2D(scr_x, scr_y, scr_w, scr_h),
        W,
        H,
        overlap_px=aperture_overlap,
    )
    video_draw_rect = QRectF(
        aperture_pixels.x,
        aperture_pixels.y,
        aperture_pixels.width,
        aperture_pixels.height,
    )

    if draw_video and inner_r > 0:
        screen_path = QPainterPath()
        screen_path.addRoundedRect(
            video_draw_rect,
            inner_r + aperture_overlap,
            inner_r + aperture_overlap,
        )
        painter.save()
        painter.setClipPath(screen_path)
        try:
            painter.fillPath(screen_path, QColor(0, 0, 0, 255))
            painter.drawImage(video_draw_rect, frame, source_rect)
        finally:
            painter.restore()
    elif draw_video:
        painter.fillRect(video_draw_rect, QColor(0, 0, 0, 255))
        painter.drawImage(video_draw_rect, frame, source_rect)

    # Paint the aperture edge after the expanded video. The video may extend
    # one pixel beneath the bezel, but can never cover or expose its edge.
    if draw_device_frame and draw_video and not fp.is_none and bw > 0:
        painter.setPen(
            QPen(QColor(*fp.bezel_color), max(2.0 * scale, 1.0))
        )
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(screen_rect, inner_r, inner_r)

    # ── front camera dot ────────────────────────────────────────────
    if draw_device_frame and fp.show_camera and bw > 0:
        cam_r = CAMERA_DOT_R * scale
        cam_cx = dev_x + dev_w / 2
        cam_cy = dev_y + bw / 2
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(CAMERA_COLOR))
        painter.drawEllipse(QPointF(cam_cx, cam_cy), cam_r, cam_r)
        painter.setBrush(QBrush(QColor(80, 80, 80)))
        painter.drawEllipse(QPointF(cam_cx - cam_r * 0.2, cam_cy - cam_r * 0.2),
                            cam_r * 0.35, cam_r * 0.35)

    # ── mouse cursor overlay ───────────────────────────────────────
    cursor_request = None
    if draw_video and mouse_track and monitor_rect:
        cursor_request = CursorRenderRequest(
            mouse_track=mouse_track,
            time_ms=time_ms,
            monitor_rect=monitor_rect,
            screen_x=scr_x / max(W, 1.0),
            screen_y=scr_y / max(H, 1.0),
            screen_w=scr_w / max(W, 1.0),
            screen_h=scr_h / max(H, 1.0),
            video_transform=video_transform,
            cursor_asset_path=cursor_asset_path,
            cursor_style_id=cursor_style_id,
            cursor_hotspot=cursor_hotspot,
            cursor_scale=cursor_scale,
            click_events=list(click_events or []),
            click_preset=click_preset,
        )
        if draw_cursor:
            draw_scene_cursor(painter, cursor_request, 0.0, 0.0, W, H)

    # ── click effects overlay ──────────────────────────────────────
    if cursor_request is not None and draw_clicks:
        draw_scene_clicks(painter, cursor_request, 0.0, 0.0, W, H)

    # Canvas Space is the final layer. Text positions are normalized against
    # the full output canvas and never inherit the Video Space transform.
    if draw_canvas_text:
        _draw_canvas_text_annotations(
            painter,
            W,
            H,
            time_ms,
            text_annotations or [],
            ephemeral_text_annotation,
            dpi_scale=text_dpi_scale,
        )
        if layer_trace is not None:
            layer_trace.append("canvas_text")

    return cursor_request


def _draw_canvas_text_annotations(
    painter: QPainter,
    canvas_w: float,
    canvas_h: float,
    time_ms: float,
    annotations: List[TextAnnotation],
    ephemeral: Optional[TextAnnotation] = None,
    *,
    dpi_scale: float = 1.0,
) -> None:
    """Render committed and draft text in absolute normalized Canvas Space."""
    from PySide6.QtGui import QFont, QFontMetricsF

    canvas_transform = CanvasSpaceTransform(canvas_w, canvas_h)
    rows = [
        item for item in annotations
        if float(item.start_ms) <= float(time_ms) <= float(item.end_ms)
    ]
    if ephemeral is not None:
        rows.append(ephemeral)

    for item in rows:
        text = str(item.text or "")
        if not text:
            continue
        opacity = max(0.0, min(1.0, float(item.opacity)))
        if opacity <= 0.0:
            continue

        font = QFont(item.font_family or "Segoe UI")
        resolved = canvas_text_metrics(
            item.font_size,
            canvas_h,
            dpi_scale=dpi_scale,
        )
        font.setPixelSize(resolved.font_px)
        painter.save()
        painter.setFont(font)
        metrics = QFontMetricsF(font)
        anchor = canvas_transform.forward(Point2D(float(item.x), float(item.y)))
        measure = lambda value: metrics.horizontalAdvance(value)
        region_w = max(
            1.0,
            min(
                canvas_w - anchor.x,
                canvas_w * float(item.text_width or item.max_width),
            ),
        )
        layout = layout_canvas_text(
            text,
            region_w,
            resolved,
            measure,
        )
        padding = float(resolved.padding_px)
        box_w = min(canvas_w, layout.box_width_px)
        box_h = min(canvas_h, layout.box_height_px)
        horizontal_alignment = (
            "left" if item.horizontal_alignment == "auto" else item.horizontal_alignment
        )
        left = anchor.x + aligned_offset(region_w, box_w, horizontal_alignment)
        left = max(0.0, min(left, canvas_w - box_w))
        region_h = (
            max(box_h, min(canvas_h - anchor.y, canvas_h * float(item.text_height)))
            if float(item.text_height) > 0.0
            else box_h
        )
        vertical_room = max(0.0, region_h - box_h)
        if item.vertical_alignment == "center":
            vertical_offset = vertical_room / 2.0
        elif item.vertical_alignment == "bottom":
            vertical_offset = vertical_room
        else:
            vertical_offset = 0.0
        top = max(0.0, min(anchor.y + vertical_offset, canvas_h - box_h))
        box = QRectF(left, top, box_w, box_h)
        reveal_progress = max(
            0.0,
            min(1.0, float(getattr(item, "_reveal_progress", 1.0))),
        )
        if reveal_progress < 0.999:
            painter.setClipRect(
                QRectF(left, top, box_w * reveal_progress, box_h),
                Qt.ClipOperation.IntersectClip,
            )

        if item.background_color is not None:
            bg = QColor(*item.background_color)
            bg.setAlpha(int(bg.alpha() * opacity))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(bg)
            painter.drawRoundedRect(box, min(6.0, padding), min(6.0, padding))

        fg = QColor(*item.color)
        fg.setAlpha(int(fg.alpha() * opacity))
        painter.setPen(fg)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for line_index, line in enumerate(layout.lines):
            baseline = top + padding + metrics.ascent() + line_index * resolved.line_height_px
            painter.drawText(QPointF(left + padding, baseline), line)
        painter.restore()

def draw_empty_bg(painter: QPainter, w: float, h: float,
                   bg_preset: Optional[BackgroundPreset] = None) -> None:
    """Draw the background with an empty device frame."""
    preset = bg_preset or DEFAULT_PRESET
    _paint_bg(painter, float(w), float(h), preset)

    # Draw device with black screen
    W, H = float(w), float(h)
    pad_x = W * DEVICE_PAD
    pad_y = H * DEVICE_PAD
    avail_w = W - 2 * pad_x
    avail_h = H - 2 * pad_y

    if avail_w / max(avail_h, 1) > DEVICE_ASPECT_FALLBACK:
        dev_h = avail_h
        dev_w = dev_h * DEVICE_ASPECT_FALLBACK
    else:
        dev_w = avail_w
        dev_h = dev_w / DEVICE_ASPECT_FALLBACK

    dev_x = (W - dev_w) / 2
    dev_y = (H - dev_h) / 2
    scale = dev_w / 900.0
    outer_r = OUTER_RADIUS * scale

    device_rect = QRectF(dev_x, dev_y, dev_w, dev_h)
    painter.setPen(QPen(BEZEL_EDGE, max(1.5 * scale, 1.0)))
    painter.setBrush(QBrush(BEZEL_COLOR))
    painter.drawRoundedRect(device_rect, outer_r, outer_r)
