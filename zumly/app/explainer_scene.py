"""Shared layout solving and animation semantics for cinematic explainers."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .geometry_math import ease_in_out_quint
from .models import (
    CanvasLayoutScene,
    ExplainerScene,
    TextAnnotation,
    canvas_layout_scene_at,
)


DESTINATIONS = (
    "left",
    "right",
    "top",
    "bottom",
    "top-left",
    "top-right",
    "bottom-left",
    "bottom-right",
)


@dataclass(frozen=True)
class ExplainerLayoutSolution:
    video_scale: float
    video_x: float
    video_y: float
    text_x: float
    text_y: float
    text_max_width: float
    text_height: float


@dataclass(frozen=True)
class ExplainerAnimationState:
    active: bool
    layout_scene: CanvasLayoutScene
    text_annotation: TextAnnotation | None
    text_opacity: float = 0.0
    text_offset_x: float = 0.0
    text_offset_y: float = 0.0
    reveal_progress: float = 0.0


@dataclass(frozen=True)
class ExplainerPhaseTiming:
    """Absolute timeline boundaries for one atomic explainer scene."""

    layout_enter_start: float
    layout_enter_end: float
    settled_hold_start: float
    settled_hold_end: float
    text_enter_start: float
    text_enter_end: float
    reading_start: float
    reading_end: float
    text_exit_start: float
    text_exit_end: float
    layout_restore_start: float
    layout_restore_end: float


EXPLAINER_SETTLED_HOLD_MS = 200.0


def solve_explainer_layout(
    destination: str,
    *,
    video_scale: float = 0.58,
    safe_margin: float = 0.04,
    gutter: float = 0.03,
) -> ExplainerLayoutSolution:
    """Return a balanced normalized video/text composition for a destination."""
    destination = str(destination or "right").strip().lower()
    if destination not in DESTINATIONS:
        destination = "right"
    margin = max(0.02, min(0.12, float(safe_margin)))
    gap = max(0.02, min(0.10, float(gutter)))
    scale = max(0.38, min(0.78, float(video_scale)))
    if "-" in destination:
        scale = min(scale, 0.54)

    centered = (1.0 - scale) / 2.0
    left = margin
    right = 1.0 - margin - scale
    top = margin
    bottom = 1.0 - margin - scale

    full_text_height = max(0.12, 1.0 - (2.0 * margin))
    if destination == "left":
        text_x = left + scale + gap
        return ExplainerLayoutSolution(
            scale, left, centered, text_x, margin,
            max(0.12, 1.0 - text_x - margin), full_text_height,
        )
    if destination == "right":
        return ExplainerLayoutSolution(
            scale, right, centered, margin, margin,
            max(0.12, right - gap - margin), full_text_height,
        )
    if destination == "top":
        text_y = top + scale + gap
        return ExplainerLayoutSolution(
            scale, centered, top, 0.10, text_y, 0.80,
            max(0.12, 1.0 - text_y - margin),
        )
    if destination == "bottom":
        return ExplainerLayoutSolution(
            scale, centered, bottom, 0.10, margin, 0.80,
            max(0.12, bottom - gap - margin),
        )

    video_x = left if destination.endswith("left") else right
    video_y = top if destination.startswith("top") else bottom
    text_x = max(0.08, min(0.56, 1.0 - video_x - scale + margin))
    if destination.endswith("right"):
        text_x = margin
    else:
        text_x = min(0.56, video_x + scale + gap)
    text_y = margin
    return ExplainerLayoutSolution(
        scale,
        video_x,
        video_y,
        text_x,
        text_y,
        max(0.22, min(0.40, 1.0 - text_x - margin)),
        full_text_height,
    )


def apply_explainer_destination(
    scene: ExplainerScene,
    destination: str,
) -> ExplainerScene:
    """Return a draft scene with its target layout and text region solved."""
    solved = solve_explainer_layout(
        destination,
        gutter=float(getattr(scene, "text_gutter", 0.03)),
    )
    layout = replace(
        scene.layout_scene,
        video_scale=solved.video_scale,
        video_x=solved.video_x,
        video_y=solved.video_y,
    )
    text = replace(
        scene.text_annotation,
        x=solved.text_x,
        y=solved.text_y,
        max_width=solved.text_max_width,
        text_width=solved.text_max_width,
        text_height=solved.text_height,
        vertical_alignment=(scene.text_annotation.vertical_alignment or "center"),
        horizontal_alignment=_default_text_alignment(destination),
    )
    return replace(
        scene,
        destination=destination,
        layout_scene=layout,
        text_annotation=text,
    )


def _default_text_alignment(destination: str) -> str:
    """Align explainer copy toward its related video, not the outer canvas edge."""
    destination = str(destination or "right")
    if destination in {"right", "top-right", "bottom-right"}:
        return "right"
    if destination in {"top", "bottom"}:
        return "center"
    return "left"


def resolved_explainer_text(scene: ExplainerScene) -> TextAnnotation:
    """Restore destination bounds missing from early ExplainerScene projects."""
    text = scene.text_annotation
    solved = solve_explainer_layout(
        scene.destination,
        video_scale=float(scene.layout_scene.video_scale),
        gutter=float(getattr(scene, "text_gutter", 0.03)),
    )
    return replace(
        text,
        text_width=(float(text.text_width) if float(text.text_width) > 0.0 else solved.text_max_width),
        text_height=(float(text.text_height) if float(text.text_height) > 0.0 else solved.text_height),
        vertical_alignment=(text.vertical_alignment or "center"),
        horizontal_alignment=(
            _default_text_alignment(scene.destination)
            if text.horizontal_alignment == "auto"
            else text.horizontal_alignment
        ),
    )
def explainer_scene_at(
    scenes: list[ExplainerScene] | None,
    time_ms: float,
) -> ExplainerScene | None:
    ordered = sorted(
        list(scenes or []),
        key=lambda scene: (float(scene.start_ms), float(scene.end_ms), scene.id),
    )
    for scene in reversed(ordered):
        if scene.contains(time_ms):
            return scene
    return None


def explainer_phase_boundaries(scene: ExplainerScene) -> tuple[float, ...]:
    """Return source-time boundaries where evaluator behavior changes."""
    phases = explainer_phase_timing(scene)
    return tuple(sorted({
        phases.layout_enter_start,
        phases.layout_enter_end,
        phases.settled_hold_end,
        phases.text_enter_start,
        phases.text_enter_end,
        phases.reading_end,
        phases.text_exit_end,
        phases.layout_restore_start,
        phases.layout_restore_end,
    }))


def _lerp(start: float, end: float, progress: float) -> float:
    return float(start) + (float(end) - float(start)) * float(progress)


def _slide_origin(destination: str) -> tuple[float, float]:
    destination = str(destination or "right")
    if destination in {"right", "top-right", "bottom-right"}:
        return -0.035, 0.0
    if destination in {"left", "top-left", "bottom-left"}:
        return 0.035, 0.0
    if destination == "top":
        return 0.0, 0.035
    return 0.0, -0.035


def explainer_text_animation_duration(scene: ExplainerScene) -> float:
    """Return a scene's bounded text entrance and exit duration in milliseconds."""
    phases = explainer_phase_timing(scene)
    return max(0.0, phases.text_enter_end - phases.text_enter_start)


def explainer_phase_timing(scene: ExplainerScene) -> ExplainerPhaseTiming:
    """Allocate non-overlapping movement, text, reading, and restore phases.

    Short scenes sacrifice reading and text-animation time before reducing the
    200 ms settled hold. This keeps text from appearing over a still-moving
    presentation group while preserving old serialized scene fields.
    """
    start = float(scene.start_ms)
    end = max(start + 1.0, float(scene.end_ms))
    duration = end - start
    requested_transition = max(0.0, float(scene.video_transition_ms))
    transition = min(requested_transition, duration * 0.35, duration / 2.0)
    restore_duration = transition if scene.restore_previous else 0.0
    layout_enter_end = start + transition
    layout_restore_start = end - restore_duration

    available = max(0.0, layout_restore_start - layout_enter_end)
    hold = min(EXPLAINER_SETTLED_HOLD_MS, available)
    after_hold = max(0.0, available - hold)

    requested_animation = max(
        0.0,
        min(float(getattr(scene, "text_animation_duration_ms", 700.0)), duration * 0.25),
    )
    animation = min(requested_animation, after_hold / 2.0)

    settled_hold_end = layout_enter_end + hold
    requested_enter = start + max(0.0, float(scene.text_enter_offset_ms))
    text_enter_start = min(
        layout_restore_start,
        max(settled_hold_end, requested_enter),
    )

    requested_exit_end = end - max(0.0, float(scene.text_exit_offset_ms))
    text_exit_end = min(layout_restore_start, max(text_enter_start, requested_exit_end))
    text_window = max(0.0, text_exit_end - text_enter_start)
    animation = min(animation, text_window / 2.0)
    text_enter_end = text_enter_start + animation
    text_exit_start = text_exit_end - animation

    return ExplainerPhaseTiming(
        layout_enter_start=start,
        layout_enter_end=layout_enter_end,
        settled_hold_start=layout_enter_end,
        settled_hold_end=settled_hold_end,
        text_enter_start=text_enter_start,
        text_enter_end=text_enter_end,
        reading_start=text_enter_end,
        reading_end=max(text_enter_end, text_exit_start),
        text_exit_start=max(text_enter_end, text_exit_start),
        text_exit_end=max(text_enter_end, text_exit_end),
        layout_restore_start=layout_restore_start,
        layout_restore_end=end,
    )


def explainer_text_enter_start(scene: ExplainerScene) -> float:
    """Start copy only after the presentation group has had time to settle."""
    return explainer_phase_timing(scene).text_enter_start


def explainer_text_exit_window(scene: ExplainerScene) -> tuple[float, float]:
    """Fade copy out before the presentation group begins its return move."""
    phases = explainer_phase_timing(scene)
    return phases.text_exit_start, phases.text_exit_end


def explainer_text_annotation(scene: ExplainerScene) -> TextAnnotation:
    """Flatten scene-owned text animation metadata for render backends."""
    slide_x, slide_y = _slide_origin(scene.destination)
    phases = explainer_phase_timing(scene)
    return replace(
        resolved_explainer_text(scene),
        start_ms=scene.start_ms,
        end_ms=phases.text_exit_end,
        animation=scene.text_animation,
        animation_delay_ms=max(0.0, phases.text_enter_start - scene.start_ms),
        animation_in_ms=max(1.0, phases.text_enter_end - phases.text_enter_start),
        animation_out_ms=max(1.0, phases.text_exit_end - phases.text_exit_start),
        slide_offset_x=slide_x,
        slide_offset_y=slide_y,
    )


def explainer_layout_scenes(
    explainers: list[ExplainerScene] | None,
    duration_ms: float,
    base_scenes: list[CanvasLayoutScene] | None = None,
) -> list[CanvasLayoutScene]:
    """Flatten atomic explainers into the existing bounded layout contract."""
    ordered = sorted(
        list(explainers or []),
        key=lambda scene: (scene.start_ms, scene.end_ms, scene.id),
    )
    if not ordered:
        return list(base_scenes or [])
    source_scenes = list(base_scenes or [CanvasLayoutScene.default(duration_ms)])
    base = canvas_layout_scene_at(source_scenes, 0.0, duration_ms)
    rows: list[CanvasLayoutScene] = [
        replace(base, id=f"explainer-base:{base.id}", start_ms=0.0, end_ms=duration_ms)
    ]
    for scene in ordered:
        previous_layout = canvas_layout_scene_at(
            source_scenes,
            max(0.0, float(scene.start_ms) - 0.001),
            duration_ms,
        )
        phases = explainer_phase_timing(scene)
        transition = phases.layout_enter_end - phases.layout_enter_start
        target = replace(
            scene.layout_scene,
            id=f"explainer-target:{scene.id}",
            start_ms=phases.layout_enter_end,
            end_ms=scene.end_ms,
            transition="ease",
            transition_duration_ms=transition,
        )
        restore = replace(
            previous_layout,
            id=f"explainer-restore:{scene.id}",
            start_ms=scene.end_ms,
            end_ms=duration_ms,
            transition="ease" if scene.restore_previous else "cut",
            transition_duration_ms=(
                phases.layout_restore_end - phases.layout_restore_start
                if scene.restore_previous else 0.0
            ),
        )
        rows.extend((target, restore))
    return sorted(rows, key=lambda item: (item.start_ms, item.id))


def evaluate_explainer_scene(
    scene: ExplainerScene,
    time_ms: float,
    base_layout: CanvasLayoutScene | None = None,
) -> ExplainerAnimationState:
    """Evaluate the authoritative preview/export animation state at one time."""
    base = base_layout or CanvasLayoutScene.default(scene.end_ms)
    target = scene.layout_scene
    now = float(time_ms)
    if now < scene.start_ms or now > scene.end_ms:
        return ExplainerAnimationState(False, replace(base), None)

    phases = explainer_phase_timing(scene)
    transition = phases.layout_enter_end - phases.layout_enter_start
    if transition <= 0.0:
        layout_progress = 1.0
    elif now < phases.layout_enter_end:
        layout_progress = ease_in_out_quint(
            (now - phases.layout_enter_start) / transition
        )
    elif scene.restore_previous and now > phases.layout_restore_start:
        restore_duration = max(
            phases.layout_restore_end - phases.layout_restore_start,
            1.0,
        )
        layout_progress = 1.0 - ease_in_out_quint(
            (now - phases.layout_restore_start) / restore_duration
        )
    else:
        layout_progress = 1.0

    layout = replace(
        target,
        video_scale=_lerp(base.video_scale, target.video_scale, layout_progress),
        video_x=_lerp(base.video_x, target.video_x, layout_progress),
        video_y=_lerp(base.video_y, target.video_y, layout_progress),
    )

    enter_duration = max(phases.text_enter_end - phases.text_enter_start, 1.0)
    enter_progress = max(
        0.0, min(1.0, (now - phases.text_enter_start) / enter_duration)
    )
    exit_duration = max(phases.text_exit_end - phases.text_exit_start, 1.0)
    exit_progress = max(
        0.0,
        min(1.0, (now - phases.text_exit_start) / exit_duration),
    )
    visibility = max(0.0, min(1.0, enter_progress * (1.0 - exit_progress)))
    offset_x = offset_y = 0.0
    reveal = 1.0
    if scene.text_animation == "fade-slide":
        origin_x, origin_y = _slide_origin(scene.destination)
        offset_x = origin_x * (1.0 - visibility)
        offset_y = origin_y * (1.0 - visibility)
    elif scene.text_animation == "soft-reveal":
        reveal = visibility

    base_text = resolved_explainer_text(scene)
    text = replace(
        base_text,
        x=max(0.0, min(1.0, base_text.x + offset_x)),
        y=max(0.0, min(1.0, base_text.y + offset_y)),
        opacity=base_text.opacity * visibility,
    )
    return ExplainerAnimationState(
        True,
        layout,
        text,
        text_opacity=visibility,
        text_offset_x=offset_x,
        text_offset_y=offset_y,
        reveal_progress=reveal,
    )


def ffmpeg_clamped_progress(time_expr: str, start_sec: float, end_sec: float) -> str:
    duration = max(float(end_sec) - float(start_sec), 0.000001)
    return f"clip((({time_expr})-{start_sec:.6f})/{duration:.6f},0,1)"


def ffmpeg_quintic_ease(progress_expr: str) -> str:
    """FFmpeg form of the same quintic smoothstep used by Python preview."""
    p = f"({progress_expr})"
    return f"(6*pow({p},5)-15*pow({p},4)+10*pow({p},3))"


def ffmpeg_lerp(start: float, end: float, progress_expr: str) -> str:
    return f"({float(start):.6f}+({float(end) - float(start):.6f})*({progress_expr}))"
