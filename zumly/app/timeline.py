"""Shared source-media to edited-output timeline mapping.

Synthetic timeline items, such as inserted cards and screen transitions, add
output time without consuming source media.  This mapper is intentionally
UI-neutral so the preview, seek bar, and exporter can share the same contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .models import ExplainerScene, ScreenTransition, TimelineFrame, VideoSegment
from .transitions import transition_resume_source_ms


def ordered_video_segments(
    video_segments: Sequence[VideoSegment] | None,
) -> list[VideoSegment]:
    """Return clips in their stable edited-output order.

    Source timestamps are deliberately not part of this ordering: copied
    clips may reference an earlier source interval while appearing later in
    the rendered output.
    """
    indexed = list(enumerate(video_segments or ()))

    def sequence_key(item: tuple[int, VideoSegment]) -> tuple[int, int]:
        original_index, segment = item
        try:
            sequence_index = max(0, int(getattr(segment, "sequence_index", original_index)))
        except (TypeError, ValueError):
            sequence_index = original_index
        return sequence_index, original_index

    return [
        segment
        for _original_index, segment in sorted(
            indexed,
            key=sequence_key,
        )
    ]


@dataclass(frozen=True)
class TimelineSpan:
    kind: str
    output_start_ms: float
    output_end_ms: float
    source_start_ms: float
    source_end_ms: float
    item_id: str = ""
    clip_id: str = ""
    visual_source_ms: float | None = None

    @property
    def output_duration_ms(self) -> float:
        return max(0.0, self.output_end_ms - self.output_start_ms)


class EditedTimelineMapper:
    """Map a cut/retimed source timeline onto the rendered output timeline."""

    def __init__(
        self,
        source_duration_ms: float,
        video_segments: Sequence[VideoSegment] | None = None,
        timeline_frames: Sequence[TimelineFrame] | None = None,
        screen_transitions: Sequence[ScreenTransition] | None = None,
        explainer_scenes: Sequence[ExplainerScene] | None = None,
    ) -> None:
        self.source_duration_ms = max(0.0, float(source_duration_ms))
        self._spans = self._build_spans(
            video_segments,
            timeline_frames or (),
            screen_transitions or (),
            explainer_scenes or (),
        )
        self.output_duration_ms = (
            self._spans[-1].output_end_ms if self._spans else self.source_duration_ms
        )
        self._synthetic_by_id = {
            span.item_id: span
            for span in self._spans
            if span.item_id and span.kind != "source"
        }

    @property
    def spans(self) -> tuple[TimelineSpan, ...]:
        return tuple(self._spans)

    def synthetic_span(self, item_id: str) -> TimelineSpan | None:
        return self._synthetic_by_id.get(str(item_id or ""))

    def synthetic_visual_source_ms(
        self,
        item_id: str,
        fallback_source_ms: float,
    ) -> float:
        """Return the source frame owned by one inserted output block.

        Multiple synthetic blocks may share one source anchor. When a Screen
        Change precedes an Explainer at that boundary, the Explainer owns the
        transition's fresh incoming frame rather than seeking back to the raw
        boundary and exposing an outgoing/stuffed frame.
        """
        span = self.synthetic_span(item_id)
        source_ms = (
            span.visual_source_ms
            if span is not None and span.visual_source_ms is not None
            else fallback_source_ms
        )
        return max(0.0, min(float(source_ms), self.source_duration_ms))

    def source_time_to_output_ms(
        self,
        source_ms: float,
        *,
        clip_id: str = "",
    ) -> float:
        """Map source-media time to output time.

        ``clip_id`` disambiguates copied source ranges. Without it, the first
        occurrence in output sequence is returned for backward compatibility.
        """
        source = max(0.0, min(float(source_ms), self.source_duration_ms))
        for span in self._spans:
            if span.kind != "source":
                if clip_id and span.clip_id != clip_id:
                    continue
                if abs(source - span.source_start_ms) <= 0.5:
                    return span.output_start_ms
                continue
            if clip_id and span.item_id != clip_id:
                continue
            if span.source_start_ms <= source <= span.source_end_ms:
                source_length = max(span.source_end_ms - span.source_start_ms, 0.001)
                ratio = (source - span.source_start_ms) / source_length
                return span.output_start_ms + ratio * span.output_duration_ms
        previous = [span for span in self._spans if span.source_end_ms <= source]
        if previous:
            return previous[-1].output_end_ms
        return 0.0

    def output_time_to_source_ms(self, output_ms: float) -> float:
        """Map an edited-output playhead time to source-media time."""
        output = max(0.0, min(float(output_ms), self.output_duration_ms))
        for index, span in enumerate(self._spans):
            owns_output = (
                span.output_start_ms <= output < span.output_end_ms
                or (
                    index == len(self._spans) - 1
                    and abs(output - span.output_end_ms) <= 1e-9
                )
            )
            if owns_output:
                if span.kind != "source":
                    return (
                        span.visual_source_ms
                        if span.visual_source_ms is not None
                        else span.source_start_ms
                    )
                output_length = max(span.output_duration_ms, 0.001)
                ratio = (output - span.output_start_ms) / output_length
                return span.source_start_ms + ratio * (
                    span.source_end_ms - span.source_start_ms
                )
        return self.source_duration_ms

    # Compatibility aliases for callers outside the editor. New code should
    # name the time domain explicitly.
    def source_to_output(self, source_ms: float) -> float:
        return self.source_time_to_output_ms(source_ms)

    def output_to_source(self, output_ms: float) -> float:
        return self.output_time_to_source_ms(output_ms)

    def clip_output_bounds_ms(self, clip_id: str) -> tuple[float, float] | None:
        """Return the output interval occupied by one ordered source clip."""
        spans = [
            span
            for span in self._spans
            if span.kind == "source" and span.item_id == clip_id
        ]
        if not spans:
            return None
        return spans[0].output_start_ms, spans[-1].output_end_ms

    def clip_id_at_output_time(self, output_ms: float) -> str:
        """Return the stable source-clip occurrence at one output position."""
        output = max(0.0, min(float(output_ms), self.output_duration_ms))
        for index, span in enumerate(self._spans):
            owns_output = (
                span.output_start_ms <= output < span.output_end_ms
                or (
                    index == len(self._spans) - 1
                    and abs(output - span.output_end_ms) <= 1e-9
                )
            )
            if owns_output:
                return span.clip_id or (span.item_id if span.kind == "source" else "")
        return ""

    def clip_id_for_source_time(
        self,
        source_ms: float,
        *,
        preferred_clip_id: str = "",
    ) -> str:
        """Resolve a source timestamp to a clip occurrence without changing domains."""
        source = max(0.0, min(float(source_ms), self.source_duration_ms))
        candidates = [
            span
            for span in self._spans
            if span.kind == "source"
            and span.source_start_ms <= source <= span.source_end_ms
        ]
        preferred = str(preferred_clip_id or "")
        if preferred:
            match = next((span for span in candidates if span.item_id == preferred), None)
            if match is not None:
                return match.item_id
        return candidates[0].item_id if candidates else ""

    def _build_spans(
        self,
        video_segments: Sequence[VideoSegment] | None,
        timeline_frames: Sequence[TimelineFrame],
        screen_transitions: Sequence[ScreenTransition],
        explainer_scenes: Sequence[ExplainerScene],
    ) -> list[TimelineSpan]:
        segments = ordered_video_segments(video_segments)
        if not segments and self.source_duration_ms > 0:
            segments = [VideoSegment.create(0.0, self.source_duration_ms)]

        events: list[tuple[float, str, int, str, float, str, float]] = []
        transition_bridges: list[tuple[float, float, str]] = []
        transition_visual_sources: list[tuple[float, str, float]] = []
        for frame in timeline_frames:
            anchor = max(0.0, min(float(frame.timestamp_ms), self.source_duration_ms))
            events.append(
                (
                    anchor,
                    str(getattr(frame, "clip_id", "") or ""),
                    0,
                    "frame",
                    max(float(frame.duration_ms), 250.0),
                    frame.id,
                    anchor,
                )
            )
        for transition in screen_transitions:
            if not transition.enabled:
                continue
            anchor = max(
                0.0, min(float(transition.timestamp_ms), self.source_duration_ms)
            )
            events.append(
                (
                    anchor,
                    str(getattr(transition, "clip_id", "") or ""),
                    1,
                    "transition",
                    max(float(transition.duration_ms), 150.0),
                    transition.id,
                    anchor,
                )
            )
            resume = transition_resume_source_ms(
                transition, self.source_duration_ms
            )
            transition_visual_sources.append(
                (
                    anchor,
                    str(getattr(transition, "clip_id", "") or ""),
                    resume,
                )
            )
            if resume > anchor + 0.001:
                transition_bridges.append(
                    (
                        anchor,
                        resume,
                        str(getattr(transition, "clip_id", "") or ""),
                    )
                )
        for scene in explainer_scenes:
            anchor = max(0.0, min(float(scene.start_ms), self.source_duration_ms))
            scene_clip_id = str(getattr(scene, "clip_id", "") or "")
            visual_source_ms = anchor
            for transition_anchor, transition_clip_id, incoming_source_ms in (
                transition_visual_sources
            ):
                same_clip = (
                    not scene_clip_id
                    or not transition_clip_id
                    or scene_clip_id == transition_clip_id
                )
                if same_clip and abs(transition_anchor - anchor) <= 0.001:
                    visual_source_ms = max(visual_source_ms, incoming_source_ms)
            events.append(
                (
                    anchor,
                    scene_clip_id,
                    2,
                    "explainer",
                    max(float(scene.end_ms) - float(scene.start_ms), 1.0),
                    scene.id,
                    visual_source_ms,
                )
            )

        spans: list[TimelineSpan] = []
        output_cursor = 0.0
        emitted_events: set[str] = set()

        def append_events(anchor: float, clip_id: str) -> None:
            nonlocal output_cursor
            matching = [
                event
                for event in events
                if abs(event[0] - anchor) <= 0.001
                and (not event[1] or event[1] == clip_id)
            ]
            for (
                _source_anchor,
                event_clip_id,
                _priority,
                kind,
                duration,
                item_id,
                visual_source_ms,
            ) in sorted(
                matching, key=lambda event: (event[2], event[5])
            ):
                if item_id in emitted_events:
                    continue
                spans.append(
                    TimelineSpan(
                        kind,
                        output_cursor,
                        output_cursor + duration,
                        anchor,
                        anchor,
                        item_id,
                        event_clip_id or clip_id,
                        visual_source_ms,
                    )
                )
                output_cursor += duration
                emitted_events.add(item_id)

        for segment in segments:
            start = max(0.0, min(float(segment.start_ms), self.source_duration_ms))
            end = max(start, min(float(segment.end_ms), self.source_duration_ms))
            segment_events = [
                event[0]
                for event in events
                if start <= event[0] <= end
                and (not event[1] or event[1] == segment.id)
            ]
            bridge_boundaries = [
                point
                for bridge_start, bridge_end, bridge_clip_id in transition_bridges
                for point in (bridge_start, bridge_end)
                if start <= point <= end
                and (not bridge_clip_id or bridge_clip_id == segment.id)
            ]
            boundaries = sorted({start, end, *segment_events, *bridge_boundaries})
            append_events(start, segment.id)
            for left, right in zip(boundaries, boundaries[1:]):
                if right <= left:
                    continue
                transition_owns_interval = any(
                    left >= bridge_start - 0.001
                    and right <= bridge_end + 0.001
                    and (not bridge_clip_id or bridge_clip_id == segment.id)
                    for bridge_start, bridge_end, bridge_clip_id in transition_bridges
                )
                if transition_owns_interval:
                    append_events(right, segment.id)
                    continue
                speed = max(float(getattr(segment, "speed", 1.0) or 1.0), 0.01)
                output_length = (right - left) / speed
                spans.append(
                    TimelineSpan(
                        "source",
                        output_cursor,
                        output_cursor + output_length,
                        left,
                        right,
                        segment.id,
                        segment.id,
                    )
                )
                output_cursor += output_length
                append_events(right, segment.id)

        for anchor, clip_id, *_rest in sorted(events, key=lambda event: event[0]):
            if not clip_id:
                append_events(anchor, "")
        return spans
