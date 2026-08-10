"""Shared Qt background painter for swatches, preview, and export assets."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QRadialGradient,
)

from .backgrounds import BackgroundPreset, WAVE_LAYERS

QtNoPen = Qt.PenStyle.NoPen
QtNoBrush = Qt.BrushStyle.NoBrush
QtPenCapFlat = Qt.PenCapStyle.FlatCap
QtPenJoinRound = Qt.PenJoinStyle.RoundJoin


def paint_background(
    painter: QPainter,
    width: float,
    height: float,
    preset: BackgroundPreset,
) -> None:
    """Paint one background identically at thumbnail or canvas resolution."""
    width = max(float(width), 1.0)
    height = max(float(height), 1.0)
    rect = QRectF(0, 0, width, height)
    top = QColor(*preset.color_top)
    bottom = QColor(*preset.color_bottom)
    kind = preset.kind

    if kind == "wavy":
        gradient = QLinearGradient(0, 0, width * 0.3, height)
        gradient.setColorAt(0.0, top)
        gradient.setColorAt(1.0, bottom)
        painter.fillRect(rect, gradient)
        for y_frac, amp_frac, frequency, phase, alpha, use_top in WAVE_LAYERS:
            color = QColor(top if use_top else bottom)
            color.setAlphaF(alpha)
            path = QPainterPath()
            path.moveTo(0, height)
            step = max(int(width / 300), 1)
            for x_pixel in range(0, int(width) + 1, step):
                progress = x_pixel / width
                y_pixel = height * y_frac + height * amp_frac * math.sin(
                    2 * math.pi * frequency * progress + phase
                )
                path.lineTo(x_pixel, y_pixel)
            path.lineTo(width, height)
            path.closeSubpath()
            painter.fillPath(path, color)
        return

    if kind == "stripes":
        painter.fillRect(rect, bottom)
        stripe_width = max(2.0, min(width, height) * 0.13)
        pen = QPen(top, stripe_width)
        pen.setCapStyle(QtPenCapFlat)
        painter.setPen(pen)
        start = -height
        while start < width + height:
            painter.drawLine(QPointF(start, height), QPointF(start + height, 0))
            start += stripe_width * 2.5
        return

    if kind == "dots":
        painter.fillRect(rect, bottom)
        spacing = max(6.0, min(width, height) * 0.22)
        radius = max(1.4, spacing * 0.25)
        painter.setPen(QPen(QtNoPen))
        painter.setBrush(QBrush(top))
        row = 0
        y = spacing * 0.5
        while y < height:
            x = spacing * (1.0 if row % 2 else 0.5)
            while x < width:
                painter.drawEllipse(QPointF(x, y), radius, radius)
                x += spacing
            row += 1
            y += spacing
        return

    if kind == "grid":
        painter.fillRect(rect, bottom)
        spacing = max(7.0, min(width, height) * 0.2)
        line = QColor(top)
        line.setAlphaF(0.62)
        painter.setPen(QPen(line, max(1.0, min(width, height) * 0.025)))
        position = spacing
        while position < width:
            painter.drawLine(QPointF(position, 0), QPointF(position, height))
            position += spacing
        position = spacing
        while position < height:
            painter.drawLine(QPointF(0, position), QPointF(width, position))
            position += spacing
        return

    if kind == "rings":
        painter.fillRect(rect, bottom)
        painter.setBrush(QBrush(QtNoBrush))
        painter.setPen(QPen(top, max(1.5, min(width, height) * 0.055)))
        center = QPointF(width * 0.5, height * 0.5)
        radius = max(4.0, min(width, height) * 0.14)
        while radius < max(width, height) * 0.8:
            painter.drawEllipse(center, radius, radius)
            radius += max(5.0, min(width, height) * 0.18)
        return

    if kind == "checker":
        painter.fillRect(rect, bottom)
        tile = max(5.0, min(width, height) * 0.18)
        row = 0
        y = 0.0
        while y < height:
            column = 0
            x = 0.0
            while x < width:
                if (row + column) % 2 == 0:
                    painter.fillRect(QRectF(x, y, tile, tile), top)
                column += 1
                x += tile
            row += 1
            y += tile
        return

    if kind == "chevron":
        painter.fillRect(rect, bottom)
        step = max(8.0, min(width, height) * 0.25)
        pen = QPen(top, max(2.0, step * 0.28))
        pen.setJoinStyle(QtPenJoinRound)
        painter.setPen(pen)
        y = -step
        while y < height + step:
            x = -step
            while x < width + step:
                painter.drawPolyline(
                    QPolygonF([
                        QPointF(x, y),
                        QPointF(x + step * 0.5, y + step * 0.5),
                        QPointF(x + step, y),
                    ])
                )
                x += step
            y += step
        return

    if kind == "radial":
        painter.fillRect(rect, bottom)
        gradient = QRadialGradient(width / 2, height / 2, max(width, height) * 0.6)
        gradient.setColorAt(0.0, top)
        transparent_bottom = QColor(bottom)
        transparent_bottom.setAlpha(0)
        gradient.setColorAt(1.0, transparent_bottom)
        painter.fillRect(rect, QBrush(gradient))
        return

    if kind == "spotlight":
        painter.fillRect(rect, bottom)
        gradient = QRadialGradient(
            width * 0.8,
            height * 0.2,
            max(width, height) * 0.75,
        )
        gradient.setColorAt(0.0, top)
        transparent_bottom = QColor(bottom)
        transparent_bottom.setAlpha(0)
        gradient.setColorAt(1.0, transparent_bottom)
        painter.fillRect(rect, QBrush(gradient))
        return

    if kind == "gradient":
        gradient = QLinearGradient(0, 0, width * 0.5, height)
        gradient.setColorAt(0.0, top)
        gradient.setColorAt(1.0, bottom)
        painter.fillRect(rect, gradient)
        return

    painter.fillRect(rect, top)
