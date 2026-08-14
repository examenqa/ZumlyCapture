from zumly.app import recording_overlay


def test_recording_badge_is_a_compact_square() -> None:
    assert recording_overlay.BASE_WIDTH == 24
    assert recording_overlay.BASE_HEIGHT == 24
    assert recording_overlay.TOP_OFFSET == 16
    assert recording_overlay.ANTIALIAS_GRID == 8


def test_recording_badge_circles_are_nested() -> None:
    assert recording_overlay.OUTER_RADIUS > recording_overlay.RING_RADIUS
    assert recording_overlay.RING_RADIUS > recording_overlay.DOT_RADIUS
    assert recording_overlay.DOT_RADIUS > 0


def test_recording_and_paused_states_use_distinct_signal_colors() -> None:
    assert recording_overlay.RECORDING_COLOR != recording_overlay.PAUSED_COLOR
    assert recording_overlay.RING_COLOR not in {
        recording_overlay.RECORDING_COLOR,
        recording_overlay.PAUSED_COLOR,
    }
