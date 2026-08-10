"""Pure timing regression tests for the shared cursor press animation."""

from app.cursor_animation import PRESS_SCALE, cursor_click_scale_at
from app.cursor_renderer import _interp_mouse
from app.models import ClickEvent, MousePosition


def test_button_state_press_and_release_use_shared_easing() -> None:
    samples = [
        MousePosition(100, 100, 0, click_state=False),
        MousePosition(100, 100, 20, click_state=True),
        MousePosition(100, 100, 60, click_state=False),
    ]

    assert cursor_click_scale_at(30, samples) == PRESS_SCALE
    assert PRESS_SCALE < cursor_click_scale_at(90, samples) < 1.0
    assert cursor_click_scale_at(200, samples) == 1.0


def test_click_event_fallback_animates_legacy_project_without_state_samples() -> None:
    samples = [MousePosition(100, 100, 0), MousePosition(100, 100, 200)]
    clicks = [ClickEvent(100, 100, 40)]

    assert cursor_click_scale_at(40, samples, clicks) == PRESS_SCALE
    assert PRESS_SCALE < cursor_click_scale_at(100, samples, clicks) < 1.0
    assert cursor_click_scale_at(220, samples, clicks) == 1.0


def test_cursor_snaps_instead_of_interpolating_across_resume_boundary() -> None:
    samples = [
        MousePosition(100, 100, 1000),
        MousePosition(900, 700, 1020, resume_boundary=True),
    ]

    assert _interp_mouse(samples, 1010) == (100, 100)
    assert _interp_mouse(samples, 1020) == (900, 700)
