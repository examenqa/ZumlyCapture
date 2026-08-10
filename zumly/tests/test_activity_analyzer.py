"""Tests for app.activity_analyzer click-driven auto-zoom logic."""

import pytest

from app.activity_analyzer import (
    ANTICIPATION_MS,
    TRANSITION_MS,
    ZOOM_PRE_ROLL_MS,
    analyze_activity,
    detect_chapters,
    _dampen_pan,
)
from app.models import ClickEvent, KeyEvent, MousePosition, VideoSegment


MONITOR = {"left": 0, "top": 0, "width": 1920, "height": 1080}


def _make_track(
    duration_ms: int,
    interval: int = 16,
    *,
    x: float = 500.0,
    y: float = 500.0,
) -> list[MousePosition]:
    """Generate a mostly stationary mouse track of given duration."""
    return [
        MousePosition(x=x, y=y, timestamp=float(t))
        for t in range(0, duration_ms + 1, interval)
    ]


def _make_shift_track() -> list[MousePosition]:
    """Generate a track with a large cursor move around the midpoint."""
    track: list[MousePosition] = []
    for t in range(0, 20001, 16):
        if t < 10000:
            x, y = 200.0, 200.0
        else:
            x, y = 1700.0, 900.0
        track.append(MousePosition(x=x, y=y, timestamp=float(t)))
    return track


def _extract_zoom_segments(keyframes: list) -> list[tuple[float, float]]:
    """Return (start, end) tuples for each zoom block."""
    segments: list[tuple[float, float]] = []
    start: float | None = None
    for keyframe in sorted(keyframes, key=lambda item: item.timestamp):
        if keyframe.zoom > 1.01 and start is None:
            start = float(keyframe.timestamp)
        elif keyframe.zoom <= 1.01 and start is not None:
            segments.append((start, float(keyframe.timestamp + keyframe.duration)))
            start = None
    return segments


class TestDampenPan:
    def test_no_zoom_returns_center(self) -> None:
        px, py = _dampen_pan(0.3, 0.7, zoom=1.0)
        assert px == 0.5
        assert py == 0.5

    def test_target_far_shifts_viewport(self) -> None:
        px, py = _dampen_pan(0.9, 0.9, zoom=2.0, from_x=0.5, from_y=0.5)
        assert px > 0.5
        assert py > 0.5

    def test_clamps_to_visible_bounds(self) -> None:
        px, py = _dampen_pan(1.0, 1.0, zoom=2.0)
        half = 0.5 / 2.0
        assert half <= px <= 1.0 - half
        assert half <= py <= 1.0 - half


class TestAnalyzeActivity:
    def test_empty_track_returns_no_keyframes(self) -> None:
        assert analyze_activity([], MONITOR) == []

    def test_too_few_samples_returns_no_keyframes(self) -> None:
        short = [MousePosition(x=0, y=0, timestamp=float(i * 16)) for i in range(5)]
        assert analyze_activity(short, MONITOR) == []

    def test_stationary_mouse_without_clicks_returns_no_keyframes(self) -> None:
        assert analyze_activity(_make_track(5000), MONITOR) == []

    def test_removed_keystrokes_are_ignored(self) -> None:
        track = _make_track(10000, x=960, y=540)
        keys = [KeyEvent(timestamp=3000.0 + i * 50.0) for i in range(20)]
        assert analyze_activity(track, MONITOR, key_events=keys) == []

    def test_zoom_level_zero_does_not_raise(self) -> None:
        track = _make_track(10000, x=960, y=540)
        clicks = [ClickEvent(x=960, y=540, timestamp=3000.0)]
        result = analyze_activity(track, MONITOR, click_events=clicks, zoom_level=0.0)
        assert isinstance(result, list)

    def test_single_click_generates_zoom(self) -> None:
        track = _make_track(10000, x=960, y=540)
        clicks = [ClickEvent(x=800, y=400, timestamp=5000.0)]
        keyframes = analyze_activity(track, MONITOR, click_events=clicks)
        assert len(keyframes) >= 2
        assert any(keyframe.zoom > 1.01 for keyframe in keyframes)

    def test_click_zoom_keyframe_starts_with_pre_roll(self) -> None:
        track = _make_track(10000, x=960, y=540)
        click_time = 5000.0
        clicks = [ClickEvent(x=800, y=400, timestamp=click_time)]

        keyframes = analyze_activity(track, MONITOR, click_events=clicks)
        zoom_in = next(keyframe for keyframe in keyframes if keyframe.zoom > 1.01)

        expected_start = click_time - TRANSITION_MS - ANTICIPATION_MS - ZOOM_PRE_ROLL_MS
        assert zoom_in.timestamp == pytest.approx(expected_start)
        assert all(keyframe.is_auto_generated for keyframe in keyframes)

    def test_click_zoom_pre_roll_clamps_to_recording_start(self) -> None:
        track = _make_track(5000, x=960, y=540)
        clicks = [ClickEvent(x=800, y=400, timestamp=300.0)]

        keyframes = analyze_activity(track, MONITOR, click_events=clicks)
        zoom_in = next(keyframe for keyframe in keyframes if keyframe.zoom > 1.01)

        assert zoom_in.timestamp == 0.0

    def test_click_cluster_targets_click_position(self) -> None:
        track = _make_track(10000, x=100, y=100)
        clicks = [
            ClickEvent(x=1600, y=820, timestamp=5000.0),
            ClickEvent(x=1620, y=830, timestamp=5200.0),
        ]
        keyframes = analyze_activity(track, MONITOR, click_events=clicks)
        zoom_ins = [keyframe for keyframe in keyframes if keyframe.zoom > 1.01]
        assert zoom_ins
        assert zoom_ins[0].x > 0.6
        assert zoom_ins[0].y > 0.6

    def test_close_in_time_far_apart_clicks_generate_pan_targets(self) -> None:
        track = _make_track(12000, x=960, y=540)
        clicks = [
            ClickEvent(x=120, y=180, timestamp=5000.0),
            ClickEvent(x=1780, y=900, timestamp=7000.0),
        ]

        keyframes = analyze_activity(track, MONITOR, click_events=clicks)
        zoom_targets = [keyframe for keyframe in keyframes if keyframe.zoom > 1.01]

        assert len(zoom_targets) >= 2
        assert zoom_targets[0].x < 0.4
        assert zoom_targets[0].y < 0.4
        assert any(
            keyframe.reason.startswith("Pan to:")
            and keyframe.x > 0.6
            and keyframe.y > 0.6
            for keyframe in zoom_targets[1:]
        )

    def test_moderately_spaced_far_apart_clicks_use_separate_zoom_blocks(self) -> None:
        track = _make_track(14000, x=960, y=540)
        clicks = [
            ClickEvent(x=120, y=180, timestamp=5000.0),
            ClickEvent(x=1780, y=900, timestamp=8000.0),
        ]

        keyframes = analyze_activity(track, MONITOR, click_events=clicks)
        segments = _extract_zoom_segments(keyframes)

        assert len(segments) >= 2
        assert not any(keyframe.reason.startswith("Pan to:") for keyframe in keyframes)

    def test_follow_cursor_false_centers_zoom(self) -> None:
        track = _make_track(10000, x=100, y=100)
        clicks = [ClickEvent(x=1600, y=820, timestamp=5000.0)]
        keyframes = analyze_activity(
            track,
            MONITOR,
            click_events=clicks,
            follow_cursor=False,
        )
        zoom_ins = [keyframe for keyframe in keyframes if keyframe.zoom > 1.01]
        assert zoom_ins
        assert zoom_ins[0].x == pytest.approx(0.5, abs=0.1)
        assert zoom_ins[0].y == pytest.approx(0.5, abs=0.1)

    def test_custom_zoom_level_is_respected(self) -> None:
        track = _make_track(10000, x=960, y=540)
        clicks = [ClickEvent(x=960, y=540, timestamp=5000.0)]
        keyframes = analyze_activity(track, MONITOR, click_events=clicks, zoom_level=2.5)
        zoom_ins = [keyframe for keyframe in keyframes if keyframe.zoom > 1.01]
        assert zoom_ins
        assert all(keyframe.zoom == pytest.approx(2.5) for keyframe in zoom_ins)

    def test_keyframes_are_sorted(self) -> None:
        track = _make_track(10000, x=960, y=540)
        clicks = [
            ClickEvent(x=300, y=200, timestamp=2000.0),
            ClickEvent(x=1600, y=800, timestamp=7000.0),
        ]
        keyframes = analyze_activity(track, MONITOR, click_events=clicks)
        assert keyframes == sorted(keyframes, key=lambda item: item.timestamp)

    def test_zoom_segments_do_not_overlap(self) -> None:
        track = _make_track(20000, x=960, y=540)
        clicks = [
            ClickEvent(x=300, y=200, timestamp=3000.0),
            ClickEvent(x=1600, y=800, timestamp=11000.0),
        ]
        segments = _extract_zoom_segments(analyze_activity(track, MONITOR, click_events=clicks))
        assert len(segments) >= 2
        for previous, current in zip(segments, segments[1:]):
            assert previous[1] <= current[0] + 1

    def test_far_apart_click_clusters_create_multiple_zoom_blocks(self) -> None:
        track = _make_shift_track()
        clicks = [
            ClickEvent(x=200, y=200, timestamp=2000.0),
            ClickEvent(x=1700, y=900, timestamp=14000.0),
        ]
        segments = _extract_zoom_segments(analyze_activity(track, MONITOR, click_events=clicks))
        assert len(segments) >= 2

    def test_max_clusters_limits_zoom_blocks(self) -> None:
        track = _make_track(30000, x=960, y=540)
        clicks = [
            ClickEvent(x=250, y=200, timestamp=2000.0),
            ClickEvent(x=1650, y=250, timestamp=8000.0),
            ClickEvent(x=250, y=850, timestamp=14000.0),
            ClickEvent(x=1650, y=850, timestamp=20000.0),
            ClickEvent(x=960, y=540, timestamp=26000.0),
        ]
        segments = _extract_zoom_segments(
            analyze_activity(track, MONITOR, click_events=clicks, max_clusters=3)
        )
        assert len(segments) <= 3

    def test_auto_zoom_processes_more_than_ten_clicks_across_visible_clips(self) -> None:
        track = _make_track(150000, interval=100, x=960, y=540)
        click_times = [
            5000, 15000, 25000, 35000, 45000,
            55000, 65000,  # deleted source range
            75000, 85000, 95000, 105000,
            125000, 135000, 145000,
        ]
        clicks = [
            ClickEvent(
                x=250 if index % 2 == 0 else 1650,
                y=220 if index % 3 == 0 else 850,
                timestamp=float(timestamp),
            )
            for index, timestamp in enumerate(click_times)
        ]
        visible_clips = [
            VideoSegment.create(0.0, 50000.0, sequence_index=0),
            VideoSegment.create(70000.0, 110000.0, sequence_index=1),
            VideoSegment.create(120000.0, 150000.0, sequence_index=2),
        ]

        keyframes = analyze_activity(
            track,
            MONITOR,
            click_events=clicks,
            visible_segments=visible_clips,
            min_gap_ms=4000,
        )
        segments = _extract_zoom_segments(keyframes)
        zoom_ins = [keyframe for keyframe in keyframes if keyframe.zoom > 1.01]

        assert len(segments) == 12
        assert max(keyframe.timestamp for keyframe in zoom_ins) > 140000.0
        assert not any(52000.0 <= keyframe.timestamp <= 67000.0 for keyframe in zoom_ins)


class TestDetectChapters:
    def test_click_gaps_create_boundaries(self) -> None:
        mouse_events = [
            MousePosition(100, 100, 0.0),
            MousePosition(100, 100, 500.0),
            MousePosition(100, 100, 6000.0),
            MousePosition(100, 100, 6500.0),
        ]

        chapters = detect_chapters(mouse_events, None, None, 10000.0)

        assert [chapter.timestamp_ms for chapter in chapters] == [0, 6000]

    def test_removed_keystrokes_do_not_change_chapters(self) -> None:
        mouse_events = [
            MousePosition(100, 100, 0.0),
            MousePosition(100, 100, 500.0),
            MousePosition(100, 100, 6000.0),
            MousePosition(100, 100, 6500.0),
        ]
        key_events = [KeyEvent(timestamp=2500.0), KeyEvent(timestamp=7000.0)]

        without_keys = detect_chapters(mouse_events, None, None, 10000.0)
        with_keys = detect_chapters(mouse_events, key_events, None, 10000.0)

        assert with_keys == without_keys

    def test_jump_detection_requires_monitor_dimensions(self) -> None:
        mouse_events = [MousePosition(100, 100, float(timestamp)) for timestamp in range(0, 6000, 500)]
        mouse_events.extend([
            MousePosition(1000, 100, 6000.0),
            MousePosition(1000, 100, 6500.0),
            MousePosition(1000, 100, 7000.0),
        ])

        without_dimensions = detect_chapters(mouse_events, None, None, 8000.0)
        with_dimensions = detect_chapters(mouse_events, None, None, 8000.0, monitor_rect=MONITOR)

        assert [chapter.timestamp_ms for chapter in without_dimensions] == [0]
        assert [chapter.timestamp_ms for chapter in with_dimensions] == [0, 6000]

    def test_resume_boundary_does_not_create_a_jump_chapter(self) -> None:
        mouse_events = [
            MousePosition(100, 100, 0.0),
            MousePosition(100, 100, 1000.0),
            MousePosition(1800, 900, 2000.0, resume_boundary=True),
            MousePosition(1800, 900, 2500.0),
        ]

        chapters = detect_chapters(
            mouse_events,
            None,
            None,
            4000.0,
            monitor_rect=MONITOR,
        )

        assert [chapter.timestamp_ms for chapter in chapters] == [0]
