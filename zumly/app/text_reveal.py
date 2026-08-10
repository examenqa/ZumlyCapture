"""Shared entrance animation semantics for inserted Text Frames."""

from __future__ import annotations

from dataclasses import dataclass

TEXT_REVEAL_EFFECTS = {"none", "fade", "fade-slide", "soft-reveal"}
TEXT_REVEAL_SLIDE_Y = 0.035


def normalize_text_reveal_effect(value: object) -> str:
    effect = str(value or "none").strip().lower()
    return effect if effect in TEXT_REVEAL_EFFECTS else "none"


def bounded_text_reveal_duration_ms(
    frame_duration_ms: float,
    requested_duration_ms: float,
) -> float:
    """Keep an entrance finite while preserving a readable settled hold."""
    frame_duration = max(1.0, float(frame_duration_ms))
    requested = max(100.0, float(requested_duration_ms))
    return min(requested, max(100.0, frame_duration * 0.45))


@dataclass(frozen=True)
class TextRevealState:
    opacity: float = 1.0
    reveal_progress: float = 1.0
    offset_y: float = 0.0


def evaluate_text_reveal(
    effect: object,
    elapsed_ms: float,
    duration_ms: float,
) -> TextRevealState:
    """Evaluate a Text Frame entrance in normalized Canvas Space."""
    normalized = normalize_text_reveal_effect(effect)
    if normalized == "none":
        return TextRevealState()
    raw_progress = max(0.0, min(1.0, float(elapsed_ms) / max(float(duration_ms), 1.0)))
    progress = raw_progress
    if normalized == "fade-slide":
        return TextRevealState(
            opacity=progress,
            reveal_progress=1.0,
            offset_y=TEXT_REVEAL_SLIDE_Y * (1.0 - progress),
        )
    if normalized == "soft-reveal":
        return TextRevealState(opacity=progress, reveal_progress=progress)
    return TextRevealState(opacity=progress)
