"""Shared normalized coordinate transforms for preview and export.

The transforms in this module are deliberately independent of Qt, Pillow, and
FFmpeg.  They operate on normalized coordinates so the same math can be used
for canvas hit-testing and for final pixel placement.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


def ease_in_out_quint(progress: float) -> float:
    """Return the quintic smoothstep used by preview and export."""
    t = max(0.0, min(1.0, float(progress)))
    return 6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3


def clamp_layout_position(
    x: float,
    y: float,
    width: float,
    height: float | None = None,
    *,
    min_visible: float = 0.10,
) -> tuple[float, float]:
    """Keep a layout group partially visible while it is being dragged.

    The group may extend beyond the canvas when it is larger than the output,
    but at least ``min_visible`` of the normalized group remains reachable on
    each axis.  Keeping this rule in the shared geometry module makes canvas
    hit-testing and future export controls use the same bounds.
    """
    group_w = max(float(width), 1e-9)
    group_h = max(float(height if height is not None else width), 1e-9)
    visible = max(0.0, min(float(min_visible), 1.0))

    min_x = -(group_w - visible)
    max_x = 1.0 - visible
    min_y = -(group_h - visible)
    max_y = 1.0 - visible
    return (
        max(min_x, min(max_x, float(x))),
        max(min_y, min(max_y, float(y))),
    )


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float


@dataclass(frozen=True)
class Rect2D:
    x: float
    y: float
    width: float
    height: float

    def contains(self, point: Point2D) -> bool:
        return (
            self.x <= point.x <= self.x + self.width
            and self.y <= point.y <= self.y + self.height
        )


def normalized_rect_to_pixels(
    x: float,
    y: float,
    width: float,
    height: float,
    bounds: "Rect2D",
) -> "Rect2D":
    """Map one normalized overlay rectangle into a concrete coordinate space.

    Preview and export both use this helper before applying their own draw
    primitive, so a Video Space or Canvas Space overlay has one geometry
    contract rather than two independently rounded implementations.
    """
    left = max(0.0, min(1.0, float(x)))
    top = max(0.0, min(1.0, float(y)))
    right = max(left, min(1.0, left + max(0.0, float(width))))
    bottom = max(top, min(1.0, top + max(0.0, float(height))))
    return Rect2D(
        bounds.x + left * bounds.width,
        bounds.y + top * bounds.height,
        (right - left) * bounds.width,
        (bottom - top) * bounds.height,
    )


def pixels_to_normalized_rect(rect: "Rect2D", bounds: "Rect2D") -> "Rect2D":
    """Inverse of :func:`normalized_rect_to_pixels` with bounded output."""
    width = max(bounds.width, 1e-9)
    height = max(bounds.height, 1e-9)
    x = max(0.0, min(1.0, (rect.x - bounds.x) / width))
    y = max(0.0, min(1.0, (rect.y - bounds.y) / height))
    right = max(x, min(1.0, (rect.x + rect.width - bounds.x) / width))
    bottom = max(y, min(1.0, (rect.y + rect.height - bounds.y) / height))
    return Rect2D(x, y, right - x, bottom - y)


@dataclass(frozen=True)
class PixelRect:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class PresentationGroupGeometry:
    """One transform shared by a device bezel and its video aperture."""

    canvas_width: float
    canvas_height: float
    device_rect: Rect2D
    video_aperture: Rect2D
    layout: "LayoutSpaceTransform"

    @classmethod
    def create(
        cls,
        canvas_width: float,
        canvas_height: float,
        device_rect: Rect2D,
        video_aperture: Rect2D,
        layout: "LayoutSpaceTransform | None" = None,
    ) -> "PresentationGroupGeometry":
        return cls(
            max(float(canvas_width), 1.0),
            max(float(canvas_height), 1.0),
            device_rect,
            video_aperture,
            layout or LayoutSpaceTransform.identity(),
        )

    @property
    def destination_rect(self) -> Rect2D:
        return self._map_canvas_rect(
            Rect2D(0.0, 0.0, self.canvas_width, self.canvas_height)
        )

    @property
    def mapped_device_rect(self) -> Rect2D:
        return self._map_canvas_rect(self.device_rect)

    @property
    def mapped_video_aperture(self) -> Rect2D:
        return self._map_canvas_rect(self.video_aperture)

    def _map_canvas_rect(self, rect: Rect2D) -> Rect2D:
        return Rect2D(
            float(self.layout.x) * self.canvas_width
            + float(rect.x) * float(self.layout.width),
            float(self.layout.y) * self.canvas_height
            + float(rect.y) * float(self.layout.height),
            float(rect.width) * float(self.layout.width),
            float(rect.height) * float(self.layout.height),
        )

    @staticmethod
    def quantize_rect(rect: Rect2D, *, even_size: bool = False) -> PixelRect:
        """Quantize shared edges once so nested layers cannot drift apart."""
        left = int(round(float(rect.x)))
        top = int(round(float(rect.y)))
        right = int(round(float(rect.x) + float(rect.width)))
        bottom = int(round(float(rect.y) + float(rect.height)))
        width = max(1, right - left)
        height = max(1, bottom - top)
        if even_size:
            width = max(2, width - (width % 2))
            height = max(2, height - (height % 2))
        return PixelRect(left, top, width, height)

    def quantized_destination(self, *, even_size: bool = False) -> PixelRect:
        return self.quantize_rect(self.destination_rect, even_size=even_size)

    @staticmethod
    def bezel_covered_aperture(
        rect: Rect2D,
        canvas_width: float,
        canvas_height: float,
        *,
        overlap_px: int = 1,
    ) -> PixelRect:
        """Cover a fractional screen aperture underneath the bezel.

        The video is deliberately extended by a small integer overlap and the
        bezel is painted afterwards. This prevents smooth scaling from
        sampling a transparent canvas pixel along the bottom or right edge.
        """
        overlap = max(0, int(overlap_px))
        left = max(0, int(math.floor(float(rect.x))) - overlap)
        top = max(0, int(math.floor(float(rect.y))) - overlap)
        right = min(
            max(1, int(math.ceil(float(canvas_width)))),
            int(math.ceil(float(rect.x) + float(rect.width))) + overlap,
        )
        bottom = min(
            max(1, int(math.ceil(float(canvas_height)))),
            int(math.ceil(float(rect.y) + float(rect.height))) + overlap,
        )
        return PixelRect(left, top, max(1, right - left), max(1, bottom - top))


@dataclass(frozen=True)
class VideoSpaceTransform:
    """Zoom/pan transform inside the normalized source video."""

    zoom: float = 1.0
    pan_x: float = 0.5
    pan_y: float = 0.5

    def viewport(self) -> tuple[float, float, float, float]:
        zoom = max(1.0, float(self.zoom))
        visible_w = 1.0 / zoom
        visible_h = 1.0 / zoom
        pan_x = max(0.0, min(1.0, float(self.pan_x)))
        pan_y = max(0.0, min(1.0, float(self.pan_y)))
        crop_x = max(0.0, min(1.0 - visible_w, pan_x - visible_w / 2.0))
        crop_y = max(0.0, min(1.0 - visible_h, pan_y - visible_h / 2.0))
        return crop_x, crop_y, visible_w, visible_h

    def forward(self, point: Point2D) -> Point2D:
        crop_x, crop_y, visible_w, visible_h = self.viewport()
        return Point2D(
            (float(point.x) - crop_x) / max(visible_w, 1e-9),
            (float(point.y) - crop_y) / max(visible_h, 1e-9),
        )

    def inverse(self, point: Point2D) -> Point2D:
        crop_x, crop_y, visible_w, visible_h = self.viewport()
        return Point2D(
            crop_x + float(point.x) * visible_w,
            crop_y + float(point.y) * visible_h,
        )

    def map_point(self, x: float, y: float) -> tuple[float, float]:
        mapped = self.forward(Point2D(x, y))
        return mapped.x, mapped.y

    @staticmethod
    def contains_source_point(x: float, y: float) -> bool:
        return 0.0 <= float(x) < 1.0 and 0.0 <= float(y) < 1.0


@dataclass(frozen=True)
class LayoutSpaceTransform:
    """Placement of the complete video/device presentation group on canvas."""

    x: float = 0.0
    y: float = 0.0
    width: float = 1.0
    height: float = 1.0

    def __post_init__(self) -> None:
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("Layout transform dimensions must be positive")

    @classmethod
    def identity(cls) -> "LayoutSpaceTransform":
        return cls()

    def forward(self, point: Point2D) -> Point2D:
        return Point2D(
            self.x + float(point.x) * self.width,
            self.y + float(point.y) * self.height,
        )

    def inverse(self, point: Point2D) -> Point2D:
        return Point2D(
            (float(point.x) - self.x) / self.width,
            (float(point.y) - self.y) / self.height,
        )

    def map_rect(self, rect: Rect2D) -> Rect2D:
        top_left = self.forward(Point2D(rect.x, rect.y))
        return Rect2D(
            top_left.x,
            top_left.y,
            rect.width * self.width,
            rect.height * self.height,
        )


@dataclass(frozen=True)
class CanvasSpaceTransform:
    """Map normalized final-canvas coordinates to output pixels."""

    width_px: float
    height_px: float

    def forward(self, point: Point2D) -> Point2D:
        return Point2D(
            float(point.x) * self.width_px,
            float(point.y) * self.height_px,
        )

    def inverse(self, point: Point2D) -> Point2D:
        return Point2D(
            float(point.x) / max(self.width_px, 1e-9),
            float(point.y) / max(self.height_px, 1e-9),
        )

    def map_point(self, x: float, y: float) -> tuple[float, float]:
        mapped = self.forward(Point2D(x, y))
        return mapped.x, mapped.y
