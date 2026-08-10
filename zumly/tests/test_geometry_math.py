import pytest

from app.geometry_math import (
    CanvasSpaceTransform,
    LayoutSpaceTransform,
    PresentationGroupGeometry,
    Point2D,
    Rect2D,
    VideoSpaceTransform,
)
from app.models import CanvasLayoutScene


def assert_point_close(actual: Point2D, expected: Point2D) -> None:
    assert actual.x == pytest.approx(expected.x)
    assert actual.y == pytest.approx(expected.y)


def test_video_transform_round_trips_a_source_point() -> None:
    transform = VideoSpaceTransform(zoom=2.0, pan_x=0.62, pan_y=0.42)
    source = Point2D(0.61, 0.44)

    assert_point_close(transform.inverse(transform.forward(source)), source)


def test_layout_transform_round_trips_and_maps_rect() -> None:
    transform = LayoutSpaceTransform(x=0.12, y=0.18, width=0.7, height=0.6)
    point = Point2D(0.35, 0.4)
    mapped = transform.forward(point)

    assert_point_close(transform.inverse(mapped), point)
    assert transform.map_rect(Rect2D(0.0, 0.0, 0.5, 0.5)) == Rect2D(
        0.12, 0.18, 0.35, 0.3
    )


def test_canvas_transform_round_trips_pixel_coordinates() -> None:
    transform = CanvasSpaceTransform(1920, 1080)
    normalized = Point2D(0.25, 0.75)

    assert_point_close(transform.inverse(transform.forward(normalized)), normalized)


def test_canvas_layout_scene_maps_presentation_group_without_moving_canvas_text() -> None:
    scene = CanvasLayoutScene.create(
        0,
        5000,
        video_scale=0.6,
        video_x=0.2,
        video_y=0.1,
    )
    presentation = LayoutSpaceTransform(
        x=scene.video_x,
        y=scene.video_y,
        width=scene.video_scale,
        height=scene.video_scale,
    )
    video_point = Point2D(0.5, 0.5)
    canvas_text_point = Point2D(0.8, 0.7)

    assert_point_close(
        presentation.forward(video_point),
        Point2D(0.5, 0.4),
    )
    assert_point_close(presentation.inverse(presentation.forward(video_point)), video_point)
    assert_point_close(canvas_text_point, Point2D(0.8, 0.7))


def test_presentation_group_keeps_aperture_locked_to_device_during_motion() -> None:
    device = Rect2D(80.0, 40.0, 840.0, 520.0)
    aperture = Rect2D(110.0, 70.0, 780.0, 460.0)
    local_offset = (
        (aperture.x - device.x) / device.width,
        (aperture.y - device.y) / device.height,
        aperture.width / device.width,
        aperture.height / device.height,
    )

    for scale, x, y in ((1.0, 0.0, 0.0), (0.79, 0.08, 0.12), (0.58, 0.38, 0.21)):
        geometry = PresentationGroupGeometry.create(
            1000,
            600,
            device,
            aperture,
            LayoutSpaceTransform(x=x, y=y, width=scale, height=scale),
        )
        mapped_device = geometry.mapped_device_rect
        mapped_aperture = geometry.mapped_video_aperture
        assert geometry.destination_rect == Rect2D(
            x * 1000,
            y * 600,
            scale * 1000,
            scale * 600,
        )
        assert (
            (mapped_aperture.x - mapped_device.x) / mapped_device.width,
            (mapped_aperture.y - mapped_device.y) / mapped_device.height,
            mapped_aperture.width / mapped_device.width,
            mapped_aperture.height / mapped_device.height,
        ) == pytest.approx(local_offset)


def test_bezel_covered_aperture_covers_fractional_bottom_and_right_edges() -> None:
    aperture = Rect2D(10.25, 20.75, 100.10, 50.10)

    covered = PresentationGroupGeometry.bezel_covered_aperture(
        aperture,
        200,
        120,
        overlap_px=1,
    )

    assert covered.x == 9
    assert covered.y == 19
    assert covered.x + covered.width == 112
    assert covered.y + covered.height == 72
    assert covered.x <= aperture.x
    assert covered.y <= aperture.y
    assert covered.x + covered.width >= aperture.x + aperture.width
    assert covered.y + covered.height >= aperture.y + aperture.height
