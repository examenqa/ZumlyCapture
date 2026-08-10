"""Shared pause-aware clock and recording state contract.

The capture worker, frame scheduler, and input trackers all consume this
clock so wall-clock pauses never enter the media timeline.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum


class RecordingState(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    RECORDING = "recording"
    PAUSED = "paused"
    STOPPING = "stopping"
    FINISHED = "finished"


@dataclass(frozen=True)
class PauseBoundary:
    """One removed wall-time interval at an active-media timestamp."""

    timeline_ms: float
    wall_duration_ms: float
    outgoing_frame_ms: float | None = None
    incoming_frame_ms: float | None = None

    def to_dict(self) -> dict[str, float]:
        data = {
            "timelineMs": max(0.0, float(self.timeline_ms)),
            "wallDurationMs": max(0.0, float(self.wall_duration_ms)),
        }
        if self.outgoing_frame_ms is not None:
            data["outgoingFrameMs"] = max(0.0, float(self.outgoing_frame_ms))
        if self.incoming_frame_ms is not None:
            data["incomingFrameMs"] = max(0.0, float(self.incoming_frame_ms))
        return data


class SessionTimelineClock:
    """Map monotonic wall time onto a pause-free active media timeline."""

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._state = RecordingState.IDLE
        self._epoch = 0.0
        self._pause_started = 0.0
        self._paused_total = 0.0
        self._stopped_active = 0.0
        self._pause_timeline = 0.0
        self._pause_boundaries: list[PauseBoundary] = []
        self._resume_generation = 0

    @property
    def state(self) -> RecordingState:
        with self._condition:
            return self._state

    @property
    def epoch(self) -> float:
        with self._condition:
            return self._epoch

    @property
    def is_paused(self) -> bool:
        with self._condition:
            return self._state == RecordingState.PAUSED

    @property
    def resume_generation(self) -> int:
        with self._condition:
            return self._resume_generation

    @property
    def pause_count(self) -> int:
        with self._condition:
            return len(self._pause_boundaries) + int(self._state == RecordingState.PAUSED)

    @property
    def pause_boundaries(self) -> list[PauseBoundary]:
        with self._condition:
            return list(self._pause_boundaries)

    @property
    def paused_duration_ms(self) -> float:
        with self._condition:
            total = self._paused_total
            if self._state == RecordingState.PAUSED and self._pause_started > 0:
                total += max(0.0, time.perf_counter() - self._pause_started)
            return total * 1000.0

    def prepare(self) -> None:
        with self._condition:
            self._state = RecordingState.STARTING
            self._epoch = 0.0
            self._pause_started = 0.0
            self._paused_total = 0.0
            self._stopped_active = 0.0
            self._pause_timeline = 0.0
            self._pause_boundaries.clear()
            self._resume_generation = 0
            self._condition.notify_all()

    def start(self, epoch: float = 0.0) -> float:
        with self._condition:
            self._epoch = epoch if epoch > 0 else time.perf_counter()
            self._pause_started = 0.0
            self._paused_total = 0.0
            self._stopped_active = 0.0
            self._pause_timeline = 0.0
            self._pause_boundaries.clear()
            self._resume_generation = 0
            self._state = RecordingState.RECORDING
            self._condition.notify_all()
            return self._epoch

    def pause(self, now: float | None = None) -> bool:
        with self._condition:
            if self._state != RecordingState.RECORDING:
                return False
            pause_at = time.perf_counter() if now is None else float(now)
            self._pause_timeline = self._active_seconds_locked(pause_at)
            self._pause_started = pause_at
            self._state = RecordingState.PAUSED
            self._condition.notify_all()
            return True

    def resume(self, now: float | None = None) -> bool:
        with self._condition:
            if self._state != RecordingState.PAUSED:
                return False
            resume_at = time.perf_counter() if now is None else float(now)
            paused_for = max(0.0, resume_at - self._pause_started)
            self._paused_total += paused_for
            self._pause_boundaries.append(
                PauseBoundary(self._pause_timeline * 1000.0, paused_for * 1000.0)
            )
            self._pause_started = 0.0
            self._resume_generation += 1
            self._state = RecordingState.RECORDING
            self._condition.notify_all()
            return True

    def mark_first_fresh_resume_frame(
        self,
        output_timestamp_ms: float,
        frame_interval_ms: float,
    ) -> bool:
        """Attach exact CFR endpoint slots to the latest pause boundary.

        The recorder calls this only after the scheduler selects a genuine
        post-resume WGC packet. Stuffed duplicates therefore never become the
        incoming screen-change endpoint.
        """
        with self._condition:
            if not self._pause_boundaries:
                return False
            boundary = self._pause_boundaries[-1]
            if boundary.incoming_frame_ms is not None:
                return False
            incoming_ms = max(float(output_timestamp_ms), boundary.timeline_ms)
            outgoing_ms = max(
                0.0,
                min(
                    boundary.timeline_ms,
                    boundary.timeline_ms - max(float(frame_interval_ms), 0.0),
                ),
            )
            self._pause_boundaries[-1] = PauseBoundary(
                boundary.timeline_ms,
                boundary.wall_duration_ms,
                outgoing_ms,
                incoming_ms,
            )
            return True

    def stop(self, now: float | None = None) -> float:
        with self._condition:
            if self._state in {RecordingState.STOPPING, RecordingState.FINISHED}:
                return max(0.0, self._stopped_active)
            if self._epoch <= 0:
                self._stopped_active = 0.0
                self._state = RecordingState.STOPPING
                self._condition.notify_all()
                return 0.0
            stop_at = time.perf_counter() if now is None else float(now)
            if self._state == RecordingState.PAUSED:
                paused_for = max(0.0, stop_at - self._pause_started)
                self._paused_total += paused_for
                self._pause_boundaries.append(
                    PauseBoundary(self._pause_timeline * 1000.0, paused_for * 1000.0)
                )
                self._pause_started = 0.0
            self._stopped_active = max(0.0, stop_at - self._epoch - self._paused_total)
            self._state = RecordingState.STOPPING
            self._condition.notify_all()
            return self._stopped_active

    def finish(self) -> None:
        with self._condition:
            self._state = RecordingState.FINISHED
            self._condition.notify_all()

    def reset(self) -> None:
        with self._condition:
            self._state = RecordingState.IDLE
            self._epoch = 0.0
            self._pause_started = 0.0
            self._paused_total = 0.0
            self._stopped_active = 0.0
            self._pause_timeline = 0.0
            self._pause_boundaries.clear()
            self._resume_generation = 0
            self._condition.notify_all()

    def active_seconds(self, now: float | None = None) -> float:
        with self._condition:
            return self._active_seconds_locked(
                time.perf_counter() if now is None else float(now)
            )

    def active_time_ms(self, now: float | None = None) -> float:
        return self.active_seconds(now) * 1000.0

    def snapshot(self, now: float | None = None) -> tuple[RecordingState, float, int]:
        """Atomically sample state, active-media time, and resume generation."""
        with self._condition:
            sampled_at = time.perf_counter() if now is None else float(now)
            return (
                self._state,
                self._active_seconds_locked(sampled_at) * 1000.0,
                self._resume_generation,
            )

    def wall_deadline(self, active_seconds: float) -> float:
        """Return the wall-clock deadline for an active-media offset."""
        with self._condition:
            return self._epoch + self._paused_total + max(0.0, float(active_seconds))

    def wait_until_active(self, timeout: float = 0.05) -> RecordingState:
        with self._condition:
            if self._state == RecordingState.PAUSED:
                self._condition.wait(timeout=max(0.0, float(timeout)))
            return self._state

    def _active_seconds_locked(self, now: float) -> float:
        if self._epoch <= 0:
            return 0.0
        if self._state in {RecordingState.STOPPING, RecordingState.FINISHED}:
            return max(0.0, self._stopped_active)
        effective_now = self._pause_started if self._state == RecordingState.PAUSED else now
        return max(0.0, effective_now - self._epoch - self._paused_total)
