"""Zoom engine — manages keyframes and interpolates zoom/pan state.

The engine holds an ordered list of :class:`ZoomKeyframe` objects and
computes the current ``(zoom, pan_x, pan_y)`` at any point in time
using quintic ease-out interpolation.  It also maintains an undo/redo
stack (deep-copy snapshots, max 50 entries).

Snapshots capture zoom keyframes, click events, video segments, and
voiceover segments so that undo/redo covers click deletions, segment
deletions, voiceover removals, and keyframe edits.
"""

import copy
from dataclasses import dataclass, field
from typing import List, Tuple
from .models import (
    CanvasLayoutScene,
    DEFAULT_CURSOR_SCALE,
    ClickEvent,
    ExplainerScene,
    HighlightBox,
    RedactionSuggestion,
    TextAnnotation,
    TimelineOverlay,
    TimelineFrame,
    ScreenTransition,
    VideoSegment,
    VoiceoverSegment,
    ZoomKeyframe,
)
from .geometry_math import ease_in_out_quint


@dataclass
class _Snapshot:
    """Internal snapshot for undo/redo — keyframes + click/voiceover/video segments."""
    keyframes: List[ZoomKeyframe] = field(default_factory=list)
    click_events: List[ClickEvent] = field(default_factory=list)
    video_segments: List[VideoSegment] = field(default_factory=list)
    timeline_frames: List[TimelineFrame] = field(default_factory=list)
    screen_transitions: List[ScreenTransition] = field(default_factory=list)
    dismissed_screen_transition_ids: List[str] = field(default_factory=list)
    voiceover_segments: List[VoiceoverSegment] = field(default_factory=list)
    highlights: List[HighlightBox] = field(default_factory=list)
    text_annotations: List[TextAnnotation] = field(default_factory=list)
    timeline_overlays: List[TimelineOverlay] = field(default_factory=list)
    redaction_suggestions: List[RedactionSuggestion] = field(default_factory=list)
    cursor_asset_path: str = ""
    cursor_style_id: str = "arrow"
    cursor_hotspot: tuple[float, float] | None = None
    cursor_scale: float = DEFAULT_CURSOR_SCALE
    canvas_layout_scenes: List[CanvasLayoutScene] = field(default_factory=list)
    explainer_scenes: List[ExplainerScene] = field(default_factory=list)


@dataclass
class _StateCommand:
    """One reversible editor mutation backed by an exact state snapshot."""

    label: str
    state: _Snapshot


def ease_out(t: float) -> float:
    """Quintic ease-out — fast start, decelerates asymptotically to zero.

    f(t) = 1 - (1-t)⁵

    The fifth-power curve gives a very pronounced deceleration at the
    end of the transition: roughly 80% of the movement happens in the
    first 40% of the duration, and the remaining 20% stretches out
    into a smooth, almost-motionless arrival.
    """
    inv = 1.0 - t
    return 1.0 - inv * inv * inv * inv * inv


def ease_in_out(t: float) -> float:
    """Quintic smoothstep — zero velocity at both endpoints.

    f(t) = 6t⁵ − 15t⁴ + 10t³

    Ideal for camera pans: starts gently, accelerates through the
    middle, and decelerates smoothly to a stop.  Both first and
    second derivatives are zero at t=0 and t=1.
    """
    return ease_in_out_quint(t)


# Keep the old name as an alias for callers that use the camera transition.
smooth_step = ease_in_out


def speed_at_time(keyframes: List[ZoomKeyframe], time_ms: float, duration_ms: float = 0.0) -> float:
    """Return the playback speed at *time_ms* from a list of keyframes.

    The speed is stored on the zoom-in keyframe that starts each
    segment.  Returns 1.0 outside zoom segments.

    The caller is expected to provide *keyframes* sorted by timestamp.
    """
    i = 0
    while i < len(keyframes):
        kf = keyframes[i]
        if kf.zoom > 1.01:
            start_ms = kf.timestamp
            speed = kf.speed
            j = i + 1
            while j < len(keyframes) and keyframes[j].zoom > 1.01:
                j += 1
            if j < len(keyframes) and keyframes[j].zoom <= 1.01:
                end_ms = (
                    min(keyframes[j].timestamp + keyframes[j].duration, duration_ms)
                    if duration_ms > 0
                    else keyframes[j].timestamp + keyframes[j].duration
                )
                i = j + 1
            else:
                end_ms = duration_ms if duration_ms > 0 else float('inf')
                i = len(keyframes)
            if start_ms <= time_ms <= end_ms:
                return speed
        else:
            i += 1
    return 1.0


MAX_UNDO = 50  # maximum undo history depth


class ZoomEngine:
    """Stateful zoom/pan interpolator with undo/redo support.

    Keyframes are kept sorted by timestamp.  ``compute_at(time_ms)``
    finds the most recent keyframe and eases from the previous state
    to the target over ``keyframe.duration`` milliseconds.
    """
    def __init__(self) -> None:
        self.keyframes: List[ZoomKeyframe] = []
        self.click_events: List[ClickEvent] = []
        self.video_segments: List[VideoSegment] = []
        self.timeline_frames: List[TimelineFrame] = []
        self.screen_transitions: List[ScreenTransition] = []
        self.dismissed_screen_transition_ids: List[str] = []
        self.voiceover_segments: List[VoiceoverSegment] = []
        self.highlights: List[HighlightBox] = []
        self.text_annotations: List[TextAnnotation] = []
        self.timeline_overlays: List[TimelineOverlay] = []
        self.redaction_suggestions: List[RedactionSuggestion] = []
        self.cursor_asset_path: str = ""
        self.cursor_style_id: str = "arrow"
        self.cursor_hotspot: tuple[float, float] | None = None
        self.cursor_scale: float = DEFAULT_CURSOR_SCALE
        self.canvas_layout_scenes: List[CanvasLayoutScene] = []
        self.explainer_scenes: List[ExplainerScene] = []
        self.current_zoom: float = 1.0
        self.current_pan_x: float = 0.5
        self.current_pan_y: float = 0.5

        # Trim state — included in undo/redo snapshots
        self.trim_start_ms: float = 0.0
        self.trim_end_ms: float = 0.0  # 0 = no trim (use full duration)

        # Undo / redo stacks — each entry is a snapshot of keyframes + click events + trim
        self._undo_stack: List[_StateCommand] = []
        self._redo_stack: List[_StateCommand] = []

    # ── snapshot helpers ────────────────────────────────────────────

    def _snapshot(self) -> _Snapshot:
        """Return a deep copy of the current keyframes, click events, video and voiceover segments."""
        return _Snapshot(
            keyframes=copy.deepcopy(self.keyframes),
            click_events=copy.deepcopy(self.click_events),
            video_segments=copy.deepcopy(self.video_segments),
            timeline_frames=copy.deepcopy(self.timeline_frames),
            screen_transitions=copy.deepcopy(self.screen_transitions),
            dismissed_screen_transition_ids=copy.deepcopy(
                self.dismissed_screen_transition_ids
            ),
            voiceover_segments=copy.deepcopy(self.voiceover_segments),
            highlights=copy.deepcopy(self.highlights),
            text_annotations=copy.deepcopy(self.text_annotations),
            timeline_overlays=copy.deepcopy(self.timeline_overlays),
            redaction_suggestions=copy.deepcopy(self.redaction_suggestions),
            cursor_asset_path=self.cursor_asset_path,
            cursor_style_id=self.cursor_style_id,
            cursor_hotspot=self.cursor_hotspot,
            cursor_scale=self.cursor_scale,
            canvas_layout_scenes=copy.deepcopy(self.canvas_layout_scenes),
            explainer_scenes=copy.deepcopy(self.explainer_scenes),
        )

    def push_undo(self, label: str = "Edit") -> None:
        """Save the current state onto the undo stack.

        Call this *before* any mutation so the previous state can be
        restored.  Clears the redo stack (new edit branch).
        """
        self._undo_stack.append(_StateCommand(label=label, state=self._snapshot()))
        if len(self._undo_stack) > MAX_UNDO:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self) -> bool:
        """Restore the previous state.  Returns True if successful."""
        if not self._undo_stack:
            return False
        command = self._undo_stack.pop()
        self._redo_stack.append(_StateCommand(label=command.label, state=self._snapshot()))
        snap = command.state
        self.keyframes = snap.keyframes
        self.click_events = snap.click_events
        self.video_segments = snap.video_segments
        self.timeline_frames = snap.timeline_frames
        self.screen_transitions = snap.screen_transitions
        self.dismissed_screen_transition_ids = snap.dismissed_screen_transition_ids
        self.voiceover_segments = snap.voiceover_segments
        self.highlights = snap.highlights
        self.text_annotations = snap.text_annotations
        self.timeline_overlays = snap.timeline_overlays
        self.redaction_suggestions = snap.redaction_suggestions
        self.cursor_asset_path = snap.cursor_asset_path
        self.cursor_style_id = snap.cursor_style_id
        self.cursor_hotspot = snap.cursor_hotspot
        self.cursor_scale = snap.cursor_scale
        self.canvas_layout_scenes = snap.canvas_layout_scenes
        self.explainer_scenes = snap.explainer_scenes
        return True

    def redo(self) -> bool:
        """Re-apply the last undone change.  Returns True if successful."""
        if not self._redo_stack:
            return False
        command = self._redo_stack.pop()
        self._undo_stack.append(_StateCommand(label=command.label, state=self._snapshot()))
        snap = command.state
        self.keyframes = snap.keyframes
        self.click_events = snap.click_events
        self.video_segments = snap.video_segments
        self.timeline_frames = snap.timeline_frames
        self.screen_transitions = snap.screen_transitions
        self.dismissed_screen_transition_ids = snap.dismissed_screen_transition_ids
        self.voiceover_segments = snap.voiceover_segments
        self.highlights = snap.highlights
        self.text_annotations = snap.text_annotations
        self.timeline_overlays = snap.timeline_overlays
        self.redaction_suggestions = snap.redaction_suggestions
        self.cursor_asset_path = snap.cursor_asset_path
        self.cursor_style_id = snap.cursor_style_id
        self.cursor_hotspot = snap.cursor_hotspot
        self.cursor_scale = snap.cursor_scale
        self.canvas_layout_scenes = snap.canvas_layout_scenes
        self.explainer_scenes = snap.explainer_scenes
        return True

    @property
    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    def clear_history(self) -> None:
        """Discard all undo/redo history."""
        self._undo_stack.clear()
        self._redo_stack.clear()

    def add_keyframe(self, kf: ZoomKeyframe) -> None:
        """Insert a keyframe, keeping the list sorted by timestamp."""
        self.keyframes.append(kf)
        self.keyframes.sort(key=lambda k: k.timestamp)

    def remove_keyframe(self, kf_id: str) -> None:
        """Remove a keyframe by its unique ID."""
        self.keyframes = [kf for kf in self.keyframes if kf.id != kf_id]

    def clear(self) -> None:
        """Remove all keyframes and reset zoom/pan to defaults."""
        self.keyframes.clear()
        self.current_zoom = 1.0
        self.current_pan_x = 0.5
        self.current_pan_y = 0.5

    def compute_at(self, time_ms: float) -> Tuple[float, float, float]:
        """Returns (zoom, pan_x, pan_y) at given time."""
        if not self.keyframes:
            return 1.0, 0.5, 0.5

        active_kf = None
        active_idx = -1
        for i in range(len(self.keyframes) - 1, -1, -1):
            if time_ms >= self.keyframes[i].timestamp:
                active_kf = self.keyframes[i]
                active_idx = i
                break

        if active_kf is None:
            return 1.0, 0.5, 0.5

        elapsed = time_ms - active_kf.timestamp
        progress = (
            min(elapsed / active_kf.duration, 1.0) if active_kf.duration > 0 else 1.0
        )
        # All transitions use ease-in-out for smooth camera movement:
        # gentle start → accelerate → gentle stop.
        eased = ease_in_out(progress)

        prev_zoom = self.keyframes[active_idx - 1].zoom if active_idx > 0 else 1.0
        prev_x = self.keyframes[active_idx - 1].x if active_idx > 0 else 0.5
        prev_y = self.keyframes[active_idx - 1].y if active_idx > 0 else 0.5

        zoom = prev_zoom + (active_kf.zoom - prev_zoom) * eased
        pan_x = prev_x + (active_kf.x - prev_x) * eased
        pan_y = prev_y + (active_kf.y - prev_y) * eased

        return zoom, pan_x, pan_y

    # ── per-segment speed helpers ───────────────────────────────────

    def _build_speed_segments(self, duration_ms: float) -> List[Tuple[float, float, float]]:
        """Return a list of (start_ms, end_ms, speed) covering the full timeline.

        Zoom segments with a non-default speed on their start keyframe
        define speed regions; everything else is 1.0×.
        """
        segments: List[Tuple[float, float, float]] = []
        sorted_kfs = sorted(self.keyframes, key=lambda k: k.timestamp)
        i = 0
        while i < len(sorted_kfs):
            kf = sorted_kfs[i]
            if kf.zoom > 1.01:
                start_ms = kf.timestamp
                speed = kf.speed
                j = i + 1
                while j < len(sorted_kfs) and sorted_kfs[j].zoom > 1.01:
                    j += 1
                if j < len(sorted_kfs) and sorted_kfs[j].zoom <= 1.01:
                    end_ms = min(sorted_kfs[j].timestamp + sorted_kfs[j].duration, duration_ms)
                    i = j + 1
                else:
                    end_ms = duration_ms
                    i = len(sorted_kfs)
                segments.append((start_ms, end_ms, speed))
            else:
                i += 1
        return segments

    def get_speed_at(self, time_ms: float, duration_ms: float = 0.0) -> float:
        """Return the playback speed at the given recording time.

        Delegates to the module-level ``speed_at_time()`` function.
        """
        return speed_at_time(self.keyframes, time_ms, duration_ms)

    def compute_output_duration(
        self, duration_ms: float, trim_start_ms: float = 0.0, trim_end_ms: float = 0.0,
    ) -> float:
        """Compute the total output duration accounting for per-segment speeds.

        Regions inside zoom segments play at their assigned speed;
        regions outside play at 1.0×.  The returned value is in ms.
        """
        eff_start = trim_start_ms if trim_start_ms > 0 else 0.0
        eff_end = trim_end_ms if trim_end_ms > 0 else duration_ms
        if eff_end <= eff_start:
            return 0.0

        speed_segs = self._build_speed_segments(duration_ms)
        output_ms = 0.0
        cursor = eff_start

        for seg_start, seg_end, speed in speed_segs:
            if seg_end <= cursor or seg_start >= eff_end:
                continue
            # Gap before this segment (speed 1.0)
            gap_end = min(seg_start, eff_end)
            if gap_end > cursor:
                output_ms += gap_end - cursor
                cursor = gap_end
            # Overlap of this segment with [cursor, eff_end]
            overlap_start = max(cursor, seg_start)
            overlap_end = min(seg_end, eff_end)
            if overlap_end > overlap_start:
                safe_speed = speed if speed > 0.0 else 1.0
                output_ms += (overlap_end - overlap_start) / safe_speed
                cursor = overlap_end

        # Remaining gap after last segment
        if cursor < eff_end:
            output_ms += eff_end - cursor

        return output_ms

    def update(self, time_ms: float) -> None:
        """Evaluate zoom state at *time_ms* and cache the result.

        Convenience wrapper around ``compute_at()`` that stores the
        result in ``current_zoom``, ``current_pan_x``, ``current_pan_y``.
        """
        self.current_zoom, self.current_pan_x, self.current_pan_y = self.compute_at(
            time_ms
        )
