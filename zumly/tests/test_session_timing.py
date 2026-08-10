import pytest

from app.session_timing import RecordingState, SessionTimelineClock
from app.screen_recorder import CfrFrameScheduler


def test_pause_freezes_active_time_and_resume_removes_wall_gap() -> None:
    clock = SessionTimelineClock()
    assert clock.start(100.0) == 100.0
    assert clock.active_seconds(105.0) == 5.0

    assert clock.pause(105.0)
    assert clock.state == RecordingState.PAUSED
    assert clock.active_seconds(112.0) == 5.0

    assert clock.resume(115.0)
    assert clock.state == RecordingState.RECORDING
    assert clock.active_seconds(116.0) == 6.0
    assert clock.paused_duration_ms == 10_000.0
    assert clock.resume_generation == 1
    assert clock.pause_boundaries[0].timeline_ms == 5_000.0

    assert clock.stop(120.0) == 10.0
    clock.finish()
    assert clock.state == RecordingState.FINISHED
    assert clock.active_time_ms() == 10_000.0


def test_cfr_slot_deadlines_shift_by_completed_pause_duration() -> None:
    clock = SessionTimelineClock()
    clock.start(100.0)
    scheduler = CfrFrameScheduler(100.0, 10, timeline_clock=clock)
    scheduler.advance()

    assert scheduler.slot_time() == 100.1
    clock.pause(100.05)
    clock.resume(105.05)
    assert scheduler.slot_time() == 105.1


def test_duplicate_pause_resume_commands_are_idempotent() -> None:
    clock = SessionTimelineClock()
    clock.start(10.0)

    assert clock.pause(11.0)
    assert not clock.pause(12.0)
    assert clock.resume(13.0)
    assert not clock.resume(14.0)
    assert clock.pause_count == 1


def test_first_fresh_resume_frame_is_attached_to_pause_boundary() -> None:
    clock = SessionTimelineClock()
    clock.start(100.0)
    assert clock.pause(101.0)
    assert clock.resume(102.0)

    assert clock.mark_first_fresh_resume_frame(1033.333, 33.333)
    boundary = clock.pause_boundaries[0]
    assert boundary.outgoing_frame_ms == pytest.approx(966.667)
    assert boundary.incoming_frame_ms == pytest.approx(1033.333)
    assert not clock.mark_first_fresh_resume_frame(1066.666, 33.333)


def test_snapshot_returns_one_consistent_clock_state() -> None:
    clock = SessionTimelineClock()
    clock.start(50.0)

    assert clock.snapshot(52.0) == (RecordingState.RECORDING, 2_000.0, 0)
    clock.pause(53.0)
    assert clock.snapshot(60.0) == (RecordingState.PAUSED, 3_000.0, 0)
    clock.resume(63.0)
    assert clock.snapshot(64.0) == (RecordingState.RECORDING, 4_000.0, 1)
