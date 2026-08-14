from zumly.app import recording_overlay


def _center(bounds: tuple[int, int, int, int]) -> tuple[float, float]:
    left, top, right, bottom = bounds
    return ((left + right) / 2, (top + bottom) / 2)


def test_recording_badge_is_a_compact_square() -> None:
    assert recording_overlay.BASE_WIDTH == 24
    assert recording_overlay.BASE_HEIGHT == 24
    assert recording_overlay.TOP_OFFSET == 16
    assert recording_overlay.NULL_PEN == 8


def test_recording_badge_circles_are_centered_and_nested() -> None:
    circles = (
        recording_overlay.OUTER_CIRCLE,
        recording_overlay.RING_CIRCLE,
        recording_overlay.DOT_CIRCLE,
    )

    assert {_center(circle) for circle in circles} == {(12.0, 12.0)}
    for outer, inner in zip(circles, circles[1:]):
        assert outer[0] < inner[0]
        assert outer[1] < inner[1]
        assert outer[2] > inner[2]
        assert outer[3] > inner[3]


def test_recording_and_paused_states_use_distinct_signal_colors() -> None:
    assert recording_overlay.RECORDING_COLOR != recording_overlay.PAUSED_COLOR
    assert recording_overlay.RING_COLOR not in {
        recording_overlay.RECORDING_COLOR,
        recording_overlay.PAUSED_COLOR,
    }
