"""Regression tests for the shared preview compositor scene contract."""

import os

from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication

from app.compositor import (
    COMPOSITOR_LAYER_ORDER,
    NO_FRAME,
    _draw_canvas_text_annotations,
    compose_scene,
    render_presentation_group,
)
from app.models import CanvasLayoutScene, MousePosition, TextAnnotation
from app.video_exporter import generate_text_annotation_png


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_compose_scene_reports_background_group_and_canvas_text_order(qapp) -> None:
    frame = QImage(64, 36, QImage.Format.Format_ARGB32_Premultiplied)
    frame.fill(0xFF336699)
    canvas = QImage(128, 72, QImage.Format.Format_ARGB32_Premultiplied)
    canvas.fill(0)
    painter = QPainter(canvas)
    trace: list[str] = []

    try:
        compose_scene(
            painter,
            frame,
            128,
            72,
            canvas_layout_scene=CanvasLayoutScene.create(
                0,
                1000,
                video_scale=0.5,
                video_x=0.25,
                video_y=0.25,
                background_color="#112233",
            ),
            text_annotations=[
                TextAnnotation.create(
                    0,
                    1000,
                    x=0.05,
                    y=0.05,
                    text="Canvas text",
                    color=(255, 255, 255, 255),
                    background_color=(0, 0, 0, 255),
                )
            ],
            layer_trace=trace,
        )
    finally:
        painter.end()

    assert tuple(trace) == COMPOSITOR_LAYER_ORDER
    assert canvas.pixelColor(0, 0).name() == "#112233"


def test_compose_scene_can_defer_cursor_to_final_viewport(qapp) -> None:
    frame = QImage(64, 36, QImage.Format.Format_ARGB32_Premultiplied)
    frame.fill(0xFF336699)
    canvas = QImage(128, 72, QImage.Format.Format_ARGB32_Premultiplied)
    canvas.fill(0)
    painter = QPainter(canvas)

    try:
        request = compose_scene(
            painter,
            frame,
            128,
            72,
            mouse_track=[MousePosition(x=32, y=18, timestamp=0)],
            time_ms=0,
            monitor_rect={"left": 0, "top": 0, "width": 64, "height": 36},
            frame_preset=NO_FRAME,
            draw_cursor=False,
            draw_canvas_text=False,
        )
    finally:
        painter.end()

    assert request is not None
    assert request.screen_x == pytest.approx(0.0)
    assert request.screen_y == pytest.approx(0.0)
    assert request.screen_w == pytest.approx(1.0)
    assert request.screen_h == pytest.approx(1.0)


def test_scaled_presentation_group_does_not_expose_canvas_at_aperture_edges(qapp) -> None:
    frame = QImage(320, 180, QImage.Format.Format_ARGB32_Premultiplied)
    frame.fill(QColor("#336699"))
    layout = CanvasLayoutScene.create(
        0,
        1000,
        video_scale=0.58,
        video_x=0.38,
        video_y=0.21,
    )
    group = render_presentation_group(frame, 640, 360, layout_scene=layout)
    canvas = QImage(640, 360, QImage.Format.Format_ARGB32_Premultiplied)
    canvas.fill(QColor("#ff3344"))
    painter = QPainter(canvas)
    destination = group.geometry.destination_rect
    try:
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawImage(
            QRectF(destination.x, destination.y, destination.width, destination.height),
            group.image,
        )
    finally:
        painter.end()

    aperture = group.geometry.mapped_video_aperture
    center_x = int(round(aperture.x + aperture.width / 2.0))
    center_y = int(round(aperture.y + aperture.height / 2.0))
    bottom_inside = int(aperture.y + aperture.height) - 1
    right_inside = int(aperture.x + aperture.width) - 1

    assert canvas.pixelColor(center_x, bottom_inside).name() != "#ff3344"
    assert canvas.pixelColor(right_inside, center_y).name() != "#ff3344"


def test_centered_text_region_matches_qt_and_export_asset(qapp) -> None:
    annotation = TextAnnotation.create(
        0,
        1000,
        x=0.1,
        y=0.1,
        text="Short explanation",
        font_size=28,
        text_width=0.8,
        text_height=0.8,
        vertical_alignment="center",
        color=(255, 255, 255, 255),
        background_color=(20, 20, 20, 255),
    )
    width, height = 320, 180
    qt_image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
    qt_image.fill(0)
    painter = QPainter(qt_image)
    try:
        _draw_canvas_text_annotations(
            painter,
            width,
            height,
            500,
            [annotation],
        )
    finally:
        painter.end()

    qt_rows = [
        y
        for y in range(height)
        if any(qt_image.pixelColor(x, y).alpha() for x in range(width))
    ]
    export_path = generate_text_annotation_png(annotation, width, height)
    try:
        with Image.open(export_path) as exported:
            bbox = exported.getchannel("A").getbbox()
        assert qt_rows and bbox is not None
        qt_center = (min(qt_rows) + max(qt_rows) + 1) / 2.0
        export_center = (bbox[1] + bbox[3]) / 2.0
        expected_center = height * 0.5
        assert qt_center == pytest.approx(expected_center, abs=2.0)
        assert export_center == pytest.approx(expected_center, abs=2.0)
        assert qt_center == pytest.approx(export_center, abs=2.0)
    finally:
        os.remove(export_path)
