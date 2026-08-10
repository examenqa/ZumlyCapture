"""Shared timing math for the cursor press animation.

The preview and export sequence use this module so a recorded mouse press has
the same hotspot-anchored scale at every timestamp in both render paths.
"""

from __future__ import annotations

import bisect
from typing import Iterable, Protocol


PRESS_SCALE = 0.90
RELEASE_DURATION_MS = 100.0
# Older recordings do not contain button-state samples. Their click event is
# represented as a brief press, then the same release easing used by new data.
LEGACY_PRESS_HOLD_MS = 34.0


class _CursorSample(Protocol):
    timestamp: float
    click_state: bool | None


class _ClickEvent(Protocol):
    timestamp: float


def _ease_out_cubic(progress: float) -> float:
    clamped = max(0.0, min(1.0, float(progress)))
    return 1.0 - (1.0 - clamped) ** 3


def _release_scale(elapsed_ms: float) -> float:
    if elapsed_ms <= 0.0:
        return PRESS_SCALE
    if elapsed_ms >= RELEASE_DURATION_MS:
        return 1.0
    return PRESS_SCALE + (1.0 - PRESS_SCALE) * _ease_out_cubic(
        elapsed_ms / RELEASE_DURATION_MS
    )


def _telemetry_scale_at(time_ms: float, samples: list[_CursorSample]) -> float:
    """Return the state-driven scale using sampled button transitions."""
    state_samples = [sample for sample in samples if sample.click_state is not None]
    if not state_samples:
        return 1.0

    timestamps = [float(sample.timestamp) for sample in state_samples]
    index = bisect.bisect_right(timestamps, float(time_ms)) - 1
    if index < 0:
        return 1.0
    if bool(state_samples[index].click_state):
        return PRESS_SCALE

    # Find the most recent pressed run, then use its first released sample as
    # the origin of the brief return-to-normal easing curve.
    pressed_index = index
    while pressed_index >= 0 and not bool(state_samples[pressed_index].click_state):
        pressed_index -= 1
    if pressed_index < 0:
        return 1.0
    release_index = pressed_index + 1
    if release_index >= len(state_samples):
        return PRESS_SCALE
    return _release_scale(float(time_ms) - float(state_samples[release_index].timestamp))


def _legacy_click_scale_at(time_ms: float, click_events: Iterable[_ClickEvent]) -> float:
    most_recent = max(
        (float(event.timestamp) for event in click_events if float(event.timestamp) <= float(time_ms)),
        default=None,
    )
    if most_recent is None:
        return 1.0
    elapsed = float(time_ms) - most_recent
    if elapsed <= LEGACY_PRESS_HOLD_MS:
        return PRESS_SCALE
    return _release_scale(elapsed - LEGACY_PRESS_HOLD_MS)


def cursor_click_scale_at(
    time_ms: float,
    samples: Iterable[_CursorSample],
    click_events: Iterable[_ClickEvent] = (),
) -> float:
    """Return the cursor scale for a session/media-local timestamp.

    Real button-state telemetry takes priority, while click events provide a
    deterministic fallback for legacy projects and very short button presses
    that occur between two mouse samples.
    """
    sample_list = list(samples)
    telemetry_scale = _telemetry_scale_at(float(time_ms), sample_list)
    legacy_scale = _legacy_click_scale_at(float(time_ms), click_events)
    return min(telemetry_scale, legacy_scale)
