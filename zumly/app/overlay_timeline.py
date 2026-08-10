"""Shared output-time visibility rules for clip-aware timeline overlays.

This module deliberately has no Qt, Pillow, or FFmpeg dependency. Preview and
export use the same result so future renderers cannot quietly diverge on copied
clips, retiming, or inserted synthetic timeline spans.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .models import TimelineOverlay
from .timeline import EditedTimelineMapper


@dataclass(frozen=True)
class OverlayVisibility:
    overlay_id: str
    output_start_ms: float
    output_end_ms: float
    visible: bool


def overlay_output_bounds_ms(
    mapper: EditedTimelineMapper,
    overlay: TimelineOverlay,
) -> tuple[float, float]:
    """Project one source-time overlay interval into the edited timeline."""
    start = mapper.source_time_to_output_ms(
        overlay.timing.start_ms,
        clip_id=overlay.timing.clip_id,
    )
    end = mapper.source_time_to_output_ms(
        overlay.timing.end_ms,
        clip_id=overlay.timing.clip_id,
    )
    return max(0.0, start), max(start, end)


def evaluate_overlay_visibility(
    mapper: EditedTimelineMapper,
    overlay: TimelineOverlay,
    output_time_ms: float,
) -> OverlayVisibility:
    """Return the sole preview/export visibility decision for one overlay.

    A zero-duration draft is visible at its anchor only. Persisted overlays are
    half-open intervals, which makes adjacent blocks deterministic.
    """
    start, end = overlay_output_bounds_ms(mapper, overlay)
    time_ms = max(0.0, float(output_time_ms))
    if end <= start:
        visible = abs(time_ms - start) <= 0.5
    else:
        visible = start <= time_ms < end
    return OverlayVisibility(overlay.id, start, end, visible)


def visible_overlays_at_output_time(
    mapper: EditedTimelineMapper,
    overlays: Iterable[TimelineOverlay] | None,
    output_time_ms: float,
) -> list[TimelineOverlay]:
    """Return stable timeline-order overlays active at one output instant."""
    rows = list(overlays or ())
    return [
        overlay
        for overlay in sorted(
            rows,
            key=lambda item: (*overlay_output_bounds_ms(mapper, item), item.id),
        )
        if evaluate_overlay_visibility(mapper, overlay, output_time_ms).visible
    ]
