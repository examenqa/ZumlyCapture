"""Shared Video Space text and shape overlay rendering.

Preview composites this renderer directly.  Export writes its transparent
output as a still asset, so Qt owns text metrics and vector geometry in both
paths instead of trying to duplicate them in FFmpeg expressions.
"""

from __future__ import annotations

from typing import Iterable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QLinearGradient, QPainter, QPainterPath, QPen

from .models import (
    OverlayKind,
    OverlayShape,
    SceneSpace,
    ShapeOverlayContent,
    PathOverlayContent,
    TextOverlayContent,
    KeystrokeOverlayContent,
    TimelineOverlay,
)


_annotation_gui_app: QGuiApplication | None = None


def _ensure_annotation_gui_app() -> None:
    """Keep headless export font metrics valid without replacing the editor app."""
    global _annotation_gui_app
    if QGuiApplication.instance() is None:
        _annotation_gui_app = QGuiApplication([])


def video_space_annotations(overlays: Iterable[TimelineOverlay] | None) -> list[TimelineOverlay]:
    """Return renderable VIDEO overlays in stable project order."""
    return [
        item for item in overlays or ()
        if item.kind in {OverlayKind.SHAPE, OverlayKind.TEXT, OverlayKind.PATH, OverlayKind.KEYSTROKE}
        and item.geometry.space is SceneSpace.VIDEO
    ]


def _color(overlay: TimelineOverlay) -> QColor:
    r, g, b, a = overlay.style.color
    color = QColor(r, g, b, a)
    color.setAlpha(max(0, min(255, round(a * overlay.style.opacity))))
    return color


def _rect(overlay: TimelineOverlay, width: int, height: int) -> QRectF:
    g = overlay.geometry
    return QRectF(g.x * width, g.y * height, max(1.0, g.width * width), max(1.0, g.height * height))


def _keystroke_theme(theme: str) -> tuple[QColor, QColor, QColor, QColor, QColor]:
    themes = {
        "light": (
            QColor("#F7F8FC"), QColor("#0F1738"), QColor("#CBD2E1"),
            QColor(15, 23, 56, 55), QColor("#68718A"),
        ),
        "brand": (
            QColor("#6D2BD6"), QColor("#FFFFFF"), QColor("#08AFC0"),
            QColor(15, 23, 56, 105), QColor("#D8CBFF"),
        ),
        "dark": (
            QColor("#171D32"), QColor("#FFFFFF"), QColor("#48536F"),
            QColor(0, 0, 0, 115), QColor("#AEB8D0"),
        ),
    }
    return themes.get(str(theme or "dark").lower(), themes["dark"])


def _keystroke_labels(content: KeystrokeOverlayContent, platform: str | None = None) -> list[str]:
    target = platform or content.platform
    if target != "mac":
        return list(content.tokens)
    aliases = {"Ctrl": "Cmd", "Control": "Cmd", "Alt": "Option", "Win": "Cmd"}
    return [aliases.get(token, token) for token in content.tokens]


def _draw_keycap_surface(
    painter: QPainter,
    rect: QRectF,
    fill: QColor,
    border: QColor,
    shadow: QColor,
) -> None:
    radius = rect.height() * 0.18
    shadow_offset = max(1.0, rect.height() * 0.055)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(shadow)
    painter.drawRoundedRect(rect.translated(0.0, shadow_offset), radius, radius)
    gradient = QLinearGradient(rect.topLeft(), rect.bottomLeft())
    gradient.setColorAt(0.0, QColor(fill).lighter(108))
    gradient.setColorAt(1.0, QColor(fill).darker(108))
    painter.setBrush(gradient)
    painter.setPen(QPen(border, max(1.0, rect.height() * 0.025)))
    painter.drawRoundedRect(rect, radius, radius)


def _draw_platform_icon(painter: QPainter, rect: QRectF, platform: str, color: QColor) -> None:
    inset = rect.height() * 0.25
    icon_rect = rect.adjusted(inset, inset, -inset, -inset)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(color)
    if platform == "mac":
        width = icon_rect.width()
        height = icon_rect.height()
        left = icon_rect.left()
        top = icon_rect.top()
        apple = QPainterPath()
        apple.moveTo(left + width * 0.50, top + height * 0.30)
        apple.cubicTo(
            left + width * 0.40, top + height * 0.18,
            left + width * 0.23, top + height * 0.23,
            left + width * 0.20, top + height * 0.42,
        )
        apple.cubicTo(
            left + width * 0.16, top + height * 0.63,
            left + width * 0.30, top + height * 0.88,
            left + width * 0.46, top + height * 0.92,
        )
        apple.cubicTo(
            left + width * 0.54, top + height * 0.94,
            left + width * 0.58, top + height * 0.88,
            left + width * 0.66, top + height * 0.89,
        )
        apple.cubicTo(
            left + width * 0.80, top + height * 0.91,
            left + width * 0.91, top + height * 0.68,
            left + width * 0.94, top + height * 0.54,
        )
        apple.cubicTo(
            left + width * 0.77, top + height * 0.49,
            left + width * 0.75, top + height * 0.31,
            left + width * 0.83, top + height * 0.22,
        )
        apple.cubicTo(
            left + width * 0.70, top + height * 0.14,
            left + width * 0.56, top + height * 0.20,
            left + width * 0.50, top + height * 0.30,
        )
        painter.drawPath(apple)
        leaf = QPainterPath()
        leaf.moveTo(left + width * 0.54, top + height * 0.18)
        leaf.cubicTo(
            left + width * 0.55, top + height * 0.04,
            left + width * 0.67, top - height * 0.01,
            left + width * 0.76, top + height * 0.02,
        )
        leaf.cubicTo(
            left + width * 0.75, top + height * 0.14,
            left + width * 0.66, top + height * 0.20,
            left + width * 0.54, top + height * 0.18,
        )
        painter.drawPath(leaf)
        return
    gap = max(1.0, rect.height() * 0.045)
    half_width = (icon_rect.width() - gap) / 2.0
    half_height = (icon_rect.height() - gap) / 2.0
    for row in range(2):
        for column in range(2):
            painter.drawRect(
                QRectF(
                    icon_rect.x() + column * (half_width + gap),
                    icon_rect.y() + row * (half_height + gap),
                    half_width,
                    half_height,
                )
            )


def _draw_platform_badge(
    painter: QPainter,
    rect: QRectF,
    platform: str,
    theme: str,
    icon_color: QColor,
) -> None:
    badge = rect.adjusted(1.0, 1.0, -1.0, -1.0)
    if platform == "windows":
        background = QColor("#1479D1")
        border = QColor("#72C9FF")
        painter.setPen(QPen(border, max(1.0, rect.height() * 0.025)))
        painter.setBrush(background)
        painter.drawRoundedRect(badge, rect.height() * 0.28, rect.height() * 0.28)
        _draw_platform_icon(painter, badge, platform, QColor("#FFFFFF"))
        return
    if str(theme or "dark").lower() == "light":
        background = QColor("#E8ECF4")
        icon = QColor("#0F1738")
        border = QColor("#B8C2D4")
    else:
        background = QColor("#0F1738")
        icon = QColor("#FFFFFF")
        border = QColor("#A9B8D4")
    painter.setPen(QPen(border, max(1.0, rect.height() * 0.025)))
    painter.setBrush(background)
    painter.drawEllipse(badge)
    _draw_platform_icon(painter, badge, platform, icon)


def _draw_keystroke_row(
    painter: QPainter,
    rect: QRectF,
    labels: list[str],
    platform: str,
    content: KeystrokeOverlayContent,
) -> None:
    fill, text_color, border, shadow, separator = _keystroke_theme(content.theme)
    font = QFont("Segoe UI")
    font.setBold(True)
    font.setPixelSize(max(8, int(round(rect.height() * 0.34))))
    painter.setFont(font)

    def layout() -> tuple[list[float], float, float, float]:
        metrics = painter.fontMetrics()
        cap_height = max(16.0, rect.height() * 0.68)
        horizontal_pad = cap_height * 0.28
        widths = [
            max(cap_height * 0.72, metrics.horizontalAdvance(label) + horizontal_pad * 2.0)
            for label in labels
        ]
        separator_width = max(
            cap_height * 0.34,
            metrics.horizontalAdvance("+") + cap_height * 0.12,
        )
        platform_width = cap_height + cap_height * 0.18 if content.show_platform_icon else 0.0
        total_width = (
            sum(widths)
            + separator_width * max(0, len(widths) - 1)
            + platform_width
        )
        return widths, cap_height, separator_width, total_width

    widths, cap_height, separator_width, total_width = layout()
    while total_width > rect.width() * 0.98 and font.pixelSize() > 8:
        font.setPixelSize(font.pixelSize() - 1)
        painter.setFont(font)
        widths, cap_height, separator_width, total_width = layout()
    scale = min(1.0, rect.width() / max(total_width, 1.0))
    if scale < 1.0:
        widths = [value * scale for value in widths]
        cap_height *= scale
        separator_width *= scale
        total_width *= scale

    x = rect.x() + (rect.width() - total_width) / 2.0
    y = rect.y() + (rect.height() - cap_height) / 2.0
    if content.show_platform_icon:
        platform_rect = QRectF(x, y, cap_height, cap_height)
        _draw_platform_badge(
            painter,
            platform_rect,
            platform,
            content.theme,
            text_color,
        )
        x += cap_height * 1.18
    for index, (label, cap_width) in enumerate(zip(labels, widths)):
        key_rect = QRectF(x, y, cap_width, cap_height)
        _draw_keycap_surface(painter, key_rect, fill, border, shadow)
        painter.setPen(text_color)
        painter.drawText(key_rect, Qt.AlignmentFlag.AlignCenter, label)
        x += cap_width
        if index < len(labels) - 1:
            separator_rect = QRectF(x, y, separator_width, cap_height)
            painter.setPen(separator)
            painter.drawText(separator_rect, Qt.AlignmentFlag.AlignCenter, "+")
            x += separator_width


def _draw_text_card(
    painter: QPainter,
    rect: QRectF,
    image_size: tuple[int, int],
    content: TextOverlayContent,
) -> None:
    """Draw a self-contained text card whose typography scales with its box."""
    base_w = max(1.0, content.base_width * image_size[0])
    base_h = max(1.0, content.base_height * image_size[1])
    scale = max(0.2, min(rect.width() / base_w, rect.height() / base_h))
    padding = max(2.0, content.padding * scale)
    radius = max(0.0, min(rect.width(), rect.height()) * content.corner_radius)
    shadow_offset = max(1.0, min(rect.width(), rect.height()) * 0.05)
    bg = QColor(*content.background_color)
    bg.setAlpha(round(bg.alpha() * content.background_opacity))
    shadow = QColor(0, 0, 0, round(255 * content.shadow_opacity))
    border = QColor(*content.border_color)
    text_color = QColor(*content.text_color)

    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(shadow)
    painter.drawRoundedRect(rect.translated(0.0, shadow_offset), radius, radius)
    surface = QLinearGradient(rect.topLeft(), rect.bottomLeft())
    surface.setColorAt(0.0, bg.lighter(108))
    surface.setColorAt(0.54, bg)
    surface.setColorAt(1.0, bg.darker(112))
    painter.setBrush(surface)
    painter.setPen(QPen(border, max(0.0, content.border_width * scale)))
    painter.drawRoundedRect(rect, radius, radius)

    content_rect = rect.adjusted(padding, padding, -padding, -padding)
    font = QFont(content.font_family)
    font.setPixelSize(max(8, int(round(content.font_size * scale))))
    painter.setFont(font)
    painter.setPen(text_color)
    painter.drawText(
        content_rect,
        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap,
        content.text,
    )


def _draw_keystroke_overlay(
    painter: QPainter,
    rect: QRectF,
    overlay: TimelineOverlay,
    content: KeystrokeOverlayContent,
) -> None:
    platforms = ["windows", "mac"] if content.platform == "both" else [content.platform]
    if len(platforms) == 1:
        _draw_keystroke_row(
            painter,
            rect,
            _keystroke_labels(content, platforms[0]),
            platforms[0],
            content,
        )
        return
    row_gap = max(2.0, rect.height() * 0.08)
    row_height = max(1.0, (rect.height() - row_gap) / 2.0)
    for index, platform in enumerate(platforms):
        row_rect = QRectF(
            rect.x(),
            rect.y() + index * (row_height + row_gap),
            rect.width(),
            row_height,
        )
        _draw_keystroke_row(
            painter,
            row_rect,
            _keystroke_labels(content, platform),
            platform,
            content,
        )


def render_video_annotation_overlay(
    overlays: Iterable[TimelineOverlay] | None,
    width: int,
    height: int,
) -> QImage:
    """Create an owned transparent annotation layer at source-video pixels."""
    _ensure_annotation_gui_app()
    image = QImage(max(1, int(width)), max(1, int(height)), QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QColor(0, 0, 0, 0))
    annotations = video_space_annotations(overlays)
    if not annotations:
        return image
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
    try:
        for overlay in annotations:
            rect = _rect(overlay, image.width(), image.height())
            color = _color(overlay)
            if overlay.kind is OverlayKind.TEXT and isinstance(overlay.content, TextOverlayContent):
                painter.save()
                painter.setClipRect(rect)
                _draw_text_card(
                    painter,
                    rect,
                    (image.width(), image.height()),
                    overlay.content,
                )
                painter.restore()
                continue
            if overlay.kind is OverlayKind.KEYSTROKE and isinstance(overlay.content, KeystrokeOverlayContent):
                painter.save()
                painter.setClipRect(rect)
                _draw_keystroke_overlay(painter, rect, overlay, overlay.content)
                painter.restore()
                continue
            if overlay.kind is OverlayKind.PATH and isinstance(overlay.content, PathOverlayContent):
                if len(overlay.content.points) < 2:
                    continue
                painter.save()
                painter.setPen(QPen(color, max(1.0, overlay.style.stroke_width), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
                path = QPainterPath()
                first_x, first_y = overlay.content.points[0]
                path.moveTo(first_x * image.width(), first_y * image.height())
                for x, y in overlay.content.points[1:]:
                    path.lineTo(x * image.width(), y * image.height())
                painter.drawPath(path)
                painter.restore()
                continue
            if not isinstance(overlay.content, ShapeOverlayContent):
                continue
            painter.save()
            pen = QPen(color, max(1.0, overlay.style.stroke_width), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            shape = overlay.content.shape
            if shape is OverlayShape.RECTANGLE:
                radius = overlay.style.corner_radius * min(rect.width(), rect.height())
                painter.drawRoundedRect(rect, radius, radius)
            elif shape is OverlayShape.CIRCLE:
                painter.drawEllipse(rect)
            else:
                start = QPointF(
                    rect.x() + overlay.content.start_x * rect.width(),
                    rect.y() + overlay.content.start_y * rect.height(),
                )
                end = QPointF(
                    rect.x() + overlay.content.end_x * rect.width(),
                    rect.y() + overlay.content.end_y * rect.height(),
                )
                painter.drawLine(start, end)
                if shape is OverlayShape.ARROW:
                    dx, dy = end.x() - start.x(), end.y() - start.y()
                    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
                    ux, uy = dx / length, dy / length
                    wing = max(10.0, min(32.0, length * 0.18))
                    painter.drawLine(
                        end,
                        end + QPointF(
                            -ux * wing - uy * wing * 0.55,
                            -uy * wing + ux * wing * 0.55,
                        ),
                    )
                    painter.drawLine(
                        end,
                        end + QPointF(
                            -ux * wing + uy * wing * 0.55,
                            -uy * wing - ux * wing * 0.55,
                        ),
                    )
            painter.restore()
    finally:
        painter.end()
    return image


def apply_video_space_annotations(frame: QImage, overlays: Iterable[TimelineOverlay] | None) -> QImage:
    """Paint active annotations into an owned video frame before masking."""
    annotations = video_space_annotations(overlays)
    if frame.isNull() or not annotations:
        return frame
    result = frame.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied).copy()
    painter = QPainter(result)
    try:
        painter.drawImage(0, 0, render_video_annotation_overlay(annotations, result.width(), result.height()))
    finally:
        painter.end()
    return result
