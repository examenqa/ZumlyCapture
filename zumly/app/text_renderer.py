"""Backend-neutral text layout math shared by preview and export.

The preview uses Qt metrics and export uses Pillow metrics, but both backends
must agree on the canvas-scale math, wrapping decisions, line height, and
padding.  This module intentionally contains no Qt or Pillow imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class CanvasTextMetrics:
    """Resolved text dimensions in final Canvas Space pixels."""

    font_px: int
    line_height_px: int
    padding_px: int
    dpi_scale: float
    canvas_scale: float


@dataclass(frozen=True)
class CanvasTextLayout:
    """Measured multiline text box in Canvas Space pixels."""

    lines: tuple[str, ...]
    line_widths: tuple[float, ...]
    box_width_px: float
    box_height_px: float


def design_px(
    value: float,
    canvas_height: float,
    *,
    baseline_height: float = 720.0,
) -> int:
    """Scale a design-space pixel value onto the current output canvas."""
    scale = max(float(canvas_height) / max(float(baseline_height), 1.0), 0.01)
    return max(1, int(round(float(value) * scale)))


def aligned_offset(container_size: float, content_size: float, alignment: str) -> float:
    """Return a bounded left/top offset for a simple alignment choice."""
    room = max(0.0, float(container_size) - float(content_size))
    choice = str(alignment or "left").strip().lower()
    if choice == "center":
        return room / 2.0
    if choice in {"right", "bottom"}:
        return room
    return 0.0


def canvas_text_metrics(
    font_size: float,
    canvas_height: float,
    *,
    dpi_scale: float = 1.0,
    baseline_height: float = 1080.0,
) -> CanvasTextMetrics:
    """Resolve normalized text settings into deterministic canvas pixels.

    ``dpi_scale`` is explicit rather than inferred by either renderer.  The
    editor can pass its normalized output scale and Pillow export uses 1.0,
    which keeps both paths on the same coordinate contract.
    """
    height_scale = max(float(canvas_height) / max(float(baseline_height), 1.0), 0.01)
    resolved_dpi = max(float(dpi_scale), 0.25)
    font_px = max(8, int(round(float(font_size) * height_scale * resolved_dpi)))
    line_height_px = max(font_px + 1, int(round(font_px * 1.20)))
    padding_px = max(4, int(round(font_px * 0.28)))
    return CanvasTextMetrics(
        font_px=font_px,
        line_height_px=line_height_px,
        padding_px=padding_px,
        dpi_scale=resolved_dpi,
        canvas_scale=height_scale,
    )


def _split_long_word(
    word: str,
    max_width: float,
    measure: Callable[[str], float],
) -> list[str]:
    """Split an unbroken token so it cannot force a canvas wider."""
    if not word:
        return [""]
    chunks: list[str] = []
    current = ""
    for char in word:
        candidate = current + char
        if current and measure(candidate) > max_width:
            chunks.append(current)
            current = char
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks or [word]


def wrap_canvas_text(
    text: str,
    max_width: float,
    measure: Callable[[str], float],
) -> tuple[str, ...]:
    """Wrap paragraphs using the renderer's supplied font measurement."""
    width = max(float(max_width), 1.0)
    paragraphs = str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines: list[str] = []
    for paragraph in paragraphs or [""]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = ""
        for word in words:
            if measure(word) > width:
                if current:
                    lines.append(current)
                    current = ""
                chunks = _split_long_word(word, width, measure)
                lines.extend(chunks[:-1])
                current = chunks[-1]
                continue
            candidate = word if not current else f"{current} {word}"
            if current and measure(candidate) > width:
                lines.append(current)
                current = word
            else:
                current = candidate
        lines.append(current)
    return tuple(lines or [""])


def layout_canvas_text(
    text: str,
    max_box_width: float,
    metrics: CanvasTextMetrics,
    measure: Callable[[str], float],
) -> CanvasTextLayout:
    """Return one shared multiline layout for Qt and Pillow render passes."""
    max_width = max(float(max_box_width), float(metrics.padding_px * 2 + 1))
    text_width = max_width - metrics.padding_px * 2
    lines = wrap_canvas_text(text, text_width, measure)
    widths = tuple(max(0.0, float(measure(line))) for line in lines)
    content_width = max(widths or (1.0,))
    box_width = min(max_width, content_width + metrics.padding_px * 2)
    box_height = max(
        float(metrics.padding_px * 2 + 1),
        float(len(lines) * metrics.line_height_px + metrics.padding_px * 2),
    )
    return CanvasTextLayout(
        lines=lines,
        line_widths=widths,
        box_width_px=box_width,
        box_height_px=box_height,
    )
