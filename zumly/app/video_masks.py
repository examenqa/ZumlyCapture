"""Static Video Space masking shared by preview and FFmpeg export.

Masks are applied to the captured video before zoom, cursor, device-frame, or
canvas composition.  That order is deliberate: every downstream consumer,
including cached transition endpoints and frozen Explainer frames, receives
already-redacted pixels.
"""

from __future__ import annotations

from typing import Iterable

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QImage, QPainter

from .models import MaskMode, MaskOverlayContent, OverlayKind, SceneSpace, TimelineOverlay


def video_space_masks(overlays: Iterable[TimelineOverlay] | None) -> list[TimelineOverlay]:
    """Return only renderable Video Space masks in stable project order."""
    return [
        overlay
        for overlay in overlays or ()
        if overlay.kind is OverlayKind.MASK
        and overlay.geometry.space is SceneSpace.VIDEO
        and isinstance(overlay.content, MaskOverlayContent)
    ]


def mask_rect_pixels(mask: TimelineOverlay, width: int, height: int) -> QRect:
    """Map normalized Video Space geometry to an in-frame, non-empty rect."""
    x = max(0, min(max(0, width - 1), int(round(mask.geometry.x * width))))
    y = max(0, min(max(0, height - 1), int(round(mask.geometry.y * height))))
    w = max(1, int(round(mask.geometry.width * width)))
    h = max(1, int(round(mask.geometry.height * height)))
    return QRect(x, y, min(w, width - x), min(h, height - y))


def _mask_color(mask: TimelineOverlay) -> QColor:
    r, g, b, a = mask.style.color
    color = QColor(r, g, b, a)
    color.setAlpha(max(0, min(255, int(round(a * mask.style.opacity)))))
    return color


def _pixelated_region(source: QImage, rect: QRect, strength: float) -> QImage:
    region = source.copy(rect)
    factor = max(2, int(round(4 + max(0.0, min(1.0, strength)) * 28)))
    small = region.scaled(
        max(1, region.width() // factor),
        max(1, region.height() // factor),
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.FastTransformation,
    )
    return small.scaled(
        region.size(),
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.FastTransformation,
    )


def _blurred_region(source: QImage, rect: QRect, strength: float) -> QImage:
    """Use a bounded multi-pass scale blur without adding an image dependency."""
    region = source.copy(rect)
    factor = max(2, int(round(2 + max(0.0, min(1.0, strength)) * 12)))
    reduced = region.scaled(
        max(1, region.width() // factor),
        max(1, region.height() // factor),
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    return reduced.scaled(
        region.size(),
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def apply_video_space_masks(
    frame: QImage,
    overlays: Iterable[TimelineOverlay] | None,
) -> QImage:
    """Return an owned frame with every active static mask applied.

    The input frame is never mutated because QVideoSink and transition caches
    may still own it.  Solid masks are exact; blur and pixelate have Qt
    fallbacks matching their FFmpeg counterparts closely enough for editing.
    """
    masks = video_space_masks(overlays)
    if frame.isNull() or not masks:
        return frame
    result = frame.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied).copy()
    painter = QPainter(result)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
    try:
        for mask in masks:
            rect = mask_rect_pixels(mask, result.width(), result.height())
            if rect.isEmpty():
                continue
            content = mask.content
            mode = content.mode
            if mode is MaskMode.PIXELATE:
                painter.drawImage(rect, _pixelated_region(result, rect, content.strength))
            elif mode is MaskMode.BLUR:
                painter.drawImage(rect, _blurred_region(result, rect, content.strength))
            else:
                painter.fillRect(rect, _mask_color(mask))
    finally:
        painter.end()
    return result


def ffmpeg_static_mask_filters(
    input_node: str,
    output_prefix: str,
    masks: Iterable[tuple[TimelineOverlay, float, float]],
    width: int,
    height: int,
) -> tuple[list[str], str]:
    """Build Video Space FFmpeg filters for a segment-local mask interval.

    ``masks`` contains ``(mask, local_start_sec, local_end_sec)`` rows.  The
    function is intentionally graph-only, letting the exporter remain the
    sole owner of input stream and timeline mapping decisions.
    """
    lines: list[str] = []
    current = input_node
    for index, (mask, start_sec, end_sec) in enumerate(masks):
        if end_sec <= start_sec:
            continue
        rect = mask_rect_pixels(mask, width, height)
        content = mask.content
        end = max(float(end_sec), float(start_sec) + 0.001)
        enable = f"between(t,{max(float(start_sec), 0.0):.6f},{end:.6f})"
        next_node = f"{output_prefix}mask{index}"
        if content.mode is MaskMode.SOLID:
            r, g, b, a = mask.style.color
            alpha = max(0.0, min(1.0, (a / 255.0) * mask.style.opacity))
            lines.append(
                f"[{current}]drawbox=x={rect.x()}:y={rect.y()}:w={rect.width()}:"
                f"h={rect.height()}:color=0x{r:02x}{g:02x}{b:02x}@{alpha:.4f}:"
                f"t=fill:enable='{enable}'[{next_node}]"
            )
        else:
            base_node = f"{output_prefix}mask{index}base"
            crop_node = f"{output_prefix}mask{index}crop"
            filtered_node = f"{output_prefix}mask{index}filtered"
            lines.append(f"[{current}]split=2[{base_node}][{crop_node}]")
            if content.mode is MaskMode.PIXELATE:
                factor = max(2, int(round(4 + content.strength * 28)))
                small_w = max(1, rect.width() // factor)
                small_h = max(1, rect.height() // factor)
                effect = (
                    f"crop={rect.width()}:{rect.height()}:{rect.x()}:{rect.y()},"
                    f"scale={small_w}:{small_h}:flags=neighbor,"
                    f"scale={rect.width()}:{rect.height()}:flags=neighbor"
                )
            else:
                radius = max(1, int(round(1 + content.strength * 18)))
                # FFmpeg's boxblur accepts a wider luma radius than chroma
                # on common Windows builds. Keep the visual strength on luma
                # while clamping chroma to the documented six-pixel ceiling.
                chroma_radius = min(radius, 6)
                effect = (
                    f"crop={rect.width()}:{rect.height()}:{rect.x()}:{rect.y()},"
                    f"boxblur=luma_radius={radius}:luma_power=1:"
                    f"chroma_radius={chroma_radius}:chroma_power=1"
                )
            lines.append(f"[{crop_node}]{effect}[{filtered_node}]")
            lines.append(
                f"[{base_node}][{filtered_node}]overlay=x={rect.x()}:y={rect.y()}:"
                f"eof_action=pass:repeatlast=1:enable='{enable}'[{next_node}]"
            )
        current = next_node
    return lines, current
