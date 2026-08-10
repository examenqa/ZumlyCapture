"""Core data models for Zumly.

Defines the dataclasses used throughout the application for recording
sessions, input events, and zoom keyframes.  All models support
JSON serialization via ``to_dict()`` / ``from_dict()`` (or ``to_json()``
/ ``from_json()`` for top-level sessions).
"""

import logging
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, ClassVar, List
import uuid
import json
import re

from .geometry_math import (
    CanvasSpaceTransform,
    LayoutSpaceTransform,
    Point2D,
    Rect2D,
    VideoSpaceTransform,
    ease_in_out_quint,
)

logger = logging.getLogger(__name__)

# Allowed values for KeystrokeOverlayConfig.filter_mode
VALID_FILTER_MODES = frozenset({"all", "modifiers-only", "shortcuts-only"})
DEFAULT_VOICEOVER_VOICE = "Kore"


class SceneSpace(str, Enum):
    """Coordinate domain for scene elements.

    VIDEO coordinates are normalized against the captured video and follow
    zoom/pan. CANVAS coordinates are normalized against the final output and
    remain fixed while the video viewport moves beneath them.
    """

    VIDEO = "video"
    CANVAS = "canvas"


class OverlayKind(str, Enum):
    """The durable overlay families supported by the shared timeline layer."""

    MASK = "mask"
    SHAPE = "shape"
    TEXT = "text"
    PATH = "path"
    CALLOUT = "callout"
    KEYSTROKE = "keystroke"


class MaskMode(str, Enum):
    BLUR = "blur"
    PIXELATE = "pixelate"
    SOLID = "solid"


class RedactionDetectionType(str, Enum):
    EMAIL = "email"
    PHONE = "phone"


class RedactionSuggestionStatus(str, Enum):
    SUGGESTED = "suggested"
    ACCEPTED = "accepted"
    DISMISSED = "dismissed"


class OverlayShape(str, Enum):
    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    LINE = "line"
    ARROW = "arrow"


@dataclass
class OverlayTiming:
    """A clip-aware source-media interval for a non-destructive overlay."""

    start_ms: float
    end_ms: float
    clip_id: str = ""

    def __post_init__(self) -> None:
        self.start_ms = max(0.0, float(self.start_ms))
        self.end_ms = max(self.start_ms, float(self.end_ms))
        self.clip_id = str(self.clip_id or "")

    def to_dict(self) -> dict:
        data = {"startMs": self.start_ms, "endMs": self.end_ms}
        if self.clip_id:
            data["clipId"] = self.clip_id
        return data

    @staticmethod
    def from_dict(data: dict) -> "OverlayTiming":
        return OverlayTiming(
            start_ms=data.get("startMs", 0.0),
            end_ms=data.get("endMs", data.get("startMs", 0.0)),
            clip_id=data.get("clipId", ""),
        )


@dataclass
class OverlayGeometry:
    """Normalized geometry in either Video or Canvas Space."""

    x: float = 0.25
    y: float = 0.25
    width: float = 0.25
    height: float = 0.25
    space: SceneSpace = SceneSpace.VIDEO

    def __post_init__(self) -> None:
        self.x = max(0.0, min(1.0, float(self.x)))
        self.y = max(0.0, min(1.0, float(self.y)))
        self.width = max(0.001, min(1.0 - self.x, float(self.width)))
        self.height = max(0.001, min(1.0 - self.y, float(self.height)))
        self.space = _scene_space(self.space)

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "space": self.space.value,
        }

    @staticmethod
    def from_dict(data: dict) -> "OverlayGeometry":
        return OverlayGeometry(
            x=data.get("x", 0.25),
            y=data.get("y", 0.25),
            width=data.get("width", 0.25),
            height=data.get("height", 0.25),
            space=_scene_space(data.get("space")),
        )


@dataclass
class OverlayStyle:
    """Shared visual style; future renderers should not invent private fields."""

    color: tuple[int, int, int, int] = (255, 204, 0, 255)
    opacity: float = 1.0
    stroke_width: float = 4.0
    corner_radius: float = 0.0

    def __post_init__(self) -> None:
        rgba = tuple(self.color or (255, 204, 0, 255))
        rgba = (rgba + (255, 255, 255, 255))[:4]
        self.color = tuple(max(0, min(255, int(value))) for value in rgba)
        self.opacity = max(0.0, min(1.0, float(self.opacity)))
        self.stroke_width = max(0.0, min(256.0, float(self.stroke_width)))
        self.corner_radius = max(0.0, min(0.5, float(self.corner_radius)))

    def to_dict(self) -> dict:
        return {
            "color": list(self.color),
            "opacity": self.opacity,
            "strokeWidth": self.stroke_width,
            "cornerRadius": self.corner_radius,
        }

    @staticmethod
    def from_dict(data: dict) -> "OverlayStyle":
        return OverlayStyle(
            color=tuple(data.get("color", (255, 204, 0, 255))),
            opacity=data.get("opacity", 1.0),
            stroke_width=data.get("strokeWidth", 4.0),
            corner_radius=data.get("cornerRadius", 0.0),
        )


@dataclass
class MaskOverlayContent:
    # Solid is the safe first-use default: it cannot briefly reveal a source
    # pixel while a project is loading or a renderer lacks an effect filter.
    mode: MaskMode = MaskMode.SOLID
    strength: float = 0.5

    def __post_init__(self) -> None:
        try:
            self.mode = self.mode if isinstance(self.mode, MaskMode) else MaskMode(str(self.mode))
        except ValueError:
            self.mode = MaskMode.BLUR
        self.strength = max(0.0, min(1.0, float(self.strength)))

    def to_dict(self) -> dict:
        return {"mode": self.mode.value, "strength": self.strength}

    @staticmethod
    def from_dict(data: dict) -> "MaskOverlayContent":
        return MaskOverlayContent(data.get("mode", MaskMode.SOLID.value), data.get("strength", 0.5))


@dataclass
class ShapeOverlayContent:
    """Shape type plus an explicit start/end vector for lines and arrows."""

    shape: OverlayShape = OverlayShape.RECTANGLE
    start_x: float = 0.0
    start_y: float = 0.0
    end_x: float = 1.0
    end_y: float = 1.0

    def __post_init__(self) -> None:
        try:
            self.shape = self.shape if isinstance(self.shape, OverlayShape) else OverlayShape(str(self.shape))
        except ValueError:
            self.shape = OverlayShape.RECTANGLE
        self.start_x = max(0.0, min(1.0, float(self.start_x)))
        self.start_y = max(0.0, min(1.0, float(self.start_y)))
        self.end_x = max(0.0, min(1.0, float(self.end_x)))
        self.end_y = max(0.0, min(1.0, float(self.end_y)))

    def to_dict(self) -> dict:
        return {
            "shape": self.shape.value,
            "startX": self.start_x,
            "startY": self.start_y,
            "endX": self.end_x,
            "endY": self.end_y,
        }

    @staticmethod
    def from_dict(data: dict) -> "ShapeOverlayContent":
        return ShapeOverlayContent(
            data.get("shape", OverlayShape.RECTANGLE.value),
            data.get("startX", 0.0),
            data.get("startY", 0.0),
            data.get("endX", 1.0),
            data.get("endY", 1.0),
        )


@dataclass
class TextOverlayContent:
    """Video Space text card rendered through the shared preview/export path."""

    text: str = "Add annotation text"
    font_family: str = "Segoe UI"
    font_size: float = 36.0
    preset: str = "dark"
    background_color: tuple[int, int, int, int] = (23, 29, 50, 245)
    text_color: tuple[int, int, int, int] = (255, 255, 255, 255)
    border_color: tuple[int, int, int, int] = (82, 95, 130, 255)
    padding: float = 14.0
    corner_radius: float = 0.16
    border_width: float = 1.0
    background_opacity: float = 0.96
    shadow_opacity: float = 0.22
    base_width: float = 0.34
    base_height: float = 0.14

    _PRESETS: ClassVar[dict[str, tuple[tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int, int, int]]]] = {
        "dark": ((23, 29, 50, 245), (255, 255, 255, 255), (82, 95, 130, 255)),
        "light": ((247, 248, 252, 248), (15, 23, 56, 255), (203, 210, 225, 255)),
        "brand": ((109, 43, 214, 245), (255, 255, 255, 255), (8, 175, 192, 255)),
    }

    @staticmethod
    def _rgba(value, fallback: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        try:
            rgba = tuple(int(item) for item in value)
        except (TypeError, ValueError):
            rgba = fallback
        rgba = (rgba + fallback)[:4]
        return tuple(max(0, min(255, item)) for item in rgba)

    def __post_init__(self) -> None:
        self.text = str(self.text or "Add annotation text")[:4096]
        self.font_family = str(self.font_family or "Segoe UI")[:160]
        self.font_size = max(8.0, min(256.0, float(self.font_size)))
        self.preset = str(self.preset or "dark").lower()
        if self.preset not in self._PRESETS:
            self.preset = "dark"
        defaults = self._PRESETS[self.preset]
        self.background_color = self._rgba(self.background_color, defaults[0])
        self.text_color = self._rgba(self.text_color, defaults[1])
        self.border_color = self._rgba(self.border_color, defaults[2])
        self.padding = max(2.0, min(96.0, float(self.padding)))
        self.corner_radius = max(0.0, min(0.5, float(self.corner_radius)))
        self.border_width = max(0.0, min(16.0, float(self.border_width)))
        self.background_opacity = max(0.0, min(1.0, float(self.background_opacity)))
        self.shadow_opacity = max(0.0, min(1.0, float(self.shadow_opacity)))
        self.base_width = max(0.01, min(1.0, float(self.base_width)))
        self.base_height = max(0.01, min(1.0, float(self.base_height)))

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "fontFamily": self.font_family,
            "fontSize": self.font_size,
            "preset": self.preset,
            "backgroundColor": list(self.background_color),
            "textColor": list(self.text_color),
            "borderColor": list(self.border_color),
            "padding": self.padding,
            "cornerRadius": self.corner_radius,
            "borderWidth": self.border_width,
            "backgroundOpacity": self.background_opacity,
            "shadowOpacity": self.shadow_opacity,
            "baseWidth": self.base_width,
            "baseHeight": self.base_height,
        }

    @staticmethod
    def from_dict(data: dict) -> "TextOverlayContent":
        return TextOverlayContent(
            text=data.get("text", "Add annotation text"),
            font_family=data.get("fontFamily", "Segoe UI"),
            font_size=data.get("fontSize", 36.0),
            preset=data.get("preset", "dark"),
            background_color=tuple(data.get("backgroundColor", (23, 29, 50, 245))),
            text_color=tuple(data.get("textColor", (255, 255, 255, 255))),
            border_color=tuple(data.get("borderColor", (82, 95, 130, 255))),
            padding=data.get("padding", 14.0),
            corner_radius=data.get("cornerRadius", 0.16),
            border_width=data.get("borderWidth", 1.0),
            background_opacity=data.get("backgroundOpacity", 0.96),
            shadow_opacity=data.get("shadowOpacity", 0.22),
            base_width=data.get("baseWidth", 0.34),
            base_height=data.get("baseHeight", 0.14),
        )


_KEYSTROKE_ALIASES = {
    "control": "Ctrl",
    "ctrl": "Ctrl",
    "command": "Cmd",
    "cmd": "Cmd",
    "option": "Option",
    "opt": "Option",
    "alt": "Alt",
    "shift": "Shift",
    "windows": "Win",
    "win": "Win",
    "return": "Enter",
    "enter": "Enter",
    "escape": "Esc",
    "esc": "Esc",
    "spacebar": "Space",
    "space": "Space",
    "backspace": "Backspace",
    "delete": "Delete",
    "del": "Delete",
    "tab": "Tab",
}


def parse_keystroke_tokens(shortcut: str) -> list[str]:
    """Normalize manual shortcut input without coupling it to key capture."""
    parts = re.split(r"\s*\+\s*", str(shortcut or ""))
    tokens: list[str] = []
    for raw in parts:
        value = " ".join(raw.strip().split())[:32]
        if not value:
            continue
        canonical = _KEYSTROKE_ALIASES.get(value.casefold())
        if canonical is None:
            canonical = value.upper() if len(value) == 1 else value.title()
        tokens.append(canonical)
        if len(tokens) >= 8:
            break
    return tokens or ["Ctrl", "N"]


@dataclass
class KeystrokeOverlayContent:
    """Structured shortcut content rendered as individual keycaps."""

    shortcut: str = "Ctrl + N"
    tokens: list[str] = field(default_factory=list)
    theme: str = "dark"
    platform: str = "windows"
    show_platform_icon: bool = False

    def __post_init__(self) -> None:
        source = " + ".join(str(token) for token in self.tokens) if self.tokens else self.shortcut
        self.tokens = parse_keystroke_tokens(source)
        self.shortcut = " + ".join(self.tokens)
        self.theme = str(self.theme or "dark").lower()
        if self.theme not in {"light", "dark", "brand"}:
            self.theme = "dark"
        self.platform = str(self.platform or "windows").strip().lower()
        if self.platform in {"windows + mac", "windows & mac", "windows_mac", "win_mac"}:
            self.platform = "both"
        if self.platform not in {"windows", "mac", "both"}:
            self.platform = "windows"
        self.show_platform_icon = bool(self.show_platform_icon)

    def to_dict(self) -> dict:
        return {
            "shortcut": self.shortcut,
            "tokens": list(self.tokens),
            "theme": self.theme,
            "platform": self.platform,
            "showPlatformIcon": self.show_platform_icon,
        }

    @staticmethod
    def from_dict(data: dict) -> "KeystrokeOverlayContent":
        return KeystrokeOverlayContent(
            shortcut=data.get("shortcut", "Ctrl + N"),
            tokens=list(data.get("tokens", [])),
            theme=data.get("theme", "dark"),
            platform=data.get("platform", "windows"),
            show_platform_icon=bool(data.get("showPlatformIcon", False)),
        )


@dataclass
class PathOverlayContent:
    """A compact normalized polyline for the future freehand tool."""

    points: list[tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        normalized: list[tuple[float, float]] = []
        for point in self.points[:4096]:
            try:
                x, y = point
                normalized.append((max(0.0, min(1.0, float(x))), max(0.0, min(1.0, float(y)))))
            except (TypeError, ValueError):
                continue
        self.points = normalized

    def to_dict(self) -> dict:
        return {"points": [[x, y] for x, y in self.points]}

    @staticmethod
    def from_dict(data: dict) -> "PathOverlayContent":
        return PathOverlayContent(list(data.get("points", [])))


OverlayContent = MaskOverlayContent | ShapeOverlayContent | TextOverlayContent | PathOverlayContent | KeystrokeOverlayContent


@dataclass
class TimelineOverlay:
    """A typed, clip-aware overlay item shared by masking and annotations.

    ``timing`` remains in source-media time. The editor and exporter project it
    through the same output mapper; this avoids duplicate-clip ambiguity.
    """

    id: str
    kind: OverlayKind
    timing: OverlayTiming
    geometry: OverlayGeometry = field(default_factory=OverlayGeometry)
    style: OverlayStyle = field(default_factory=OverlayStyle)
    content: OverlayContent = field(default_factory=MaskOverlayContent)

    @property
    def start_ms(self) -> float:
        return self.timing.start_ms

    @property
    def end_ms(self) -> float:
        return self.timing.end_ms

    @property
    def clip_id(self) -> str:
        return self.timing.clip_id

    @staticmethod
    def create(
        kind: OverlayKind | str,
        start_ms: float,
        end_ms: float,
        *,
        clip_id: str = "",
        geometry: OverlayGeometry | None = None,
        style: OverlayStyle | None = None,
        content: OverlayContent | None = None,
    ) -> "TimelineOverlay":
        try:
            normalized_kind = kind if isinstance(kind, OverlayKind) else OverlayKind(str(kind))
        except ValueError:
            normalized_kind = OverlayKind.MASK
        if content is None:
            content = (
                MaskOverlayContent() if normalized_kind is OverlayKind.MASK
                else TextOverlayContent() if normalized_kind is OverlayKind.TEXT
                else KeystrokeOverlayContent() if normalized_kind is OverlayKind.KEYSTROKE
                else PathOverlayContent() if normalized_kind is OverlayKind.PATH
                else ShapeOverlayContent()
            )
        if style is None and normalized_kind is OverlayKind.MASK:
            style = OverlayStyle(color=(0, 0, 0, 255), opacity=1.0)
        return TimelineOverlay(
            id=str(uuid.uuid4()),
            kind=normalized_kind,
            timing=OverlayTiming(start_ms, end_ms, clip_id),
            geometry=geometry or OverlayGeometry(),
            style=style or OverlayStyle(),
            content=content,
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "timing": self.timing.to_dict(),
            "geometry": self.geometry.to_dict(),
            "style": self.style.to_dict(),
            "content": self.content.to_dict(),
        }

    @staticmethod
    def from_dict(data: dict) -> "TimelineOverlay":
        """Load canonical nested payloads and the short-lived flat prototype."""
        raw_kind = data.get("kind", OverlayKind.MASK.value)
        try:
            kind = OverlayKind(str(raw_kind))
        except ValueError:
            kind = OverlayKind.MASK
        timing_data = data.get("timing") if isinstance(data.get("timing"), dict) else data
        geometry_data = data.get("geometry") if isinstance(data.get("geometry"), dict) else data
        style_data = data.get("style") if isinstance(data.get("style"), dict) else data
        content_data = data.get("content") if isinstance(data.get("content"), dict) else data
        if kind is OverlayKind.MASK:
            content: OverlayContent = MaskOverlayContent.from_dict(content_data)
        elif kind is OverlayKind.TEXT:
            content = TextOverlayContent.from_dict(content_data)
        elif kind is OverlayKind.PATH:
            content = PathOverlayContent.from_dict(content_data)
        elif kind is OverlayKind.KEYSTROKE:
            content = KeystrokeOverlayContent.from_dict(content_data)
        else:
            content = ShapeOverlayContent.from_dict(content_data)
        return TimelineOverlay(
            id=str(data.get("id") or uuid.uuid4()),
            kind=kind,
            timing=OverlayTiming.from_dict(timing_data),
            geometry=OverlayGeometry.from_dict(geometry_data),
            style=OverlayStyle.from_dict(style_data),
            content=content,
        )


@dataclass
class RedactionSuggestion:
    """Privacy-safe OCR finding awaiting an explicit user decision.

    Recognized text is deliberately absent from this durable model. Suggestions
    retain only the detection category, source-relative timing, normalized
    Video Space geometry, and a coarse confidence score.
    """

    id: str
    detection_type: RedactionDetectionType
    timing: OverlayTiming
    geometry: OverlayGeometry
    confidence: float = 0.0
    status: RedactionSuggestionStatus = RedactionSuggestionStatus.SUGGESTED
    accepted_overlay_id: str = ""

    def __post_init__(self) -> None:
        try:
            self.detection_type = (
                self.detection_type
                if isinstance(self.detection_type, RedactionDetectionType)
                else RedactionDetectionType(str(self.detection_type))
            )
        except ValueError:
            self.detection_type = RedactionDetectionType.EMAIL
        try:
            self.status = (
                self.status
                if isinstance(self.status, RedactionSuggestionStatus)
                else RedactionSuggestionStatus(str(self.status))
            )
        except ValueError:
            self.status = RedactionSuggestionStatus.SUGGESTED
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.accepted_overlay_id = str(self.accepted_overlay_id or "")
        if self.geometry.space is not SceneSpace.VIDEO:
            self.geometry.space = SceneSpace.VIDEO

    @property
    def start_ms(self) -> float:
        return self.timing.start_ms

    @property
    def end_ms(self) -> float:
        return self.timing.end_ms

    @property
    def clip_id(self) -> str:
        return self.timing.clip_id

    def to_mask_overlay(self) -> TimelineOverlay:
        """Compile an accepted suggestion into the existing mask renderer."""
        overlay = TimelineOverlay.create(
            OverlayKind.MASK,
            self.start_ms,
            self.end_ms,
            clip_id=self.clip_id,
            geometry=OverlayGeometry(
                self.geometry.x,
                self.geometry.y,
                self.geometry.width,
                self.geometry.height,
                SceneSpace.VIDEO,
            ),
            style=OverlayStyle(color=(0, 0, 0, 255), opacity=1.0),
            content=MaskOverlayContent(MaskMode.SOLID, 1.0),
        )
        self.status = RedactionSuggestionStatus.ACCEPTED
        self.accepted_overlay_id = overlay.id
        return overlay

    def to_dict(self) -> dict:
        data = {
            "id": self.id,
            "detectionType": self.detection_type.value,
            "timing": self.timing.to_dict(),
            "geometry": self.geometry.to_dict(),
            "confidence": self.confidence,
            "status": self.status.value,
        }
        if self.accepted_overlay_id:
            data["acceptedOverlayId"] = self.accepted_overlay_id
        return data

    @staticmethod
    def from_dict(data: dict) -> "RedactionSuggestion":
        return RedactionSuggestion(
            id=str(data.get("id") or uuid.uuid4()),
            detection_type=data.get("detectionType", RedactionDetectionType.EMAIL.value),
            timing=OverlayTiming.from_dict(data.get("timing", {})),
            geometry=OverlayGeometry.from_dict(data.get("geometry", {})),
            confidence=data.get("confidence", 0.0),
            status=data.get("status", RedactionSuggestionStatus.SUGGESTED.value),
            accepted_overlay_id=str(data.get("acceptedOverlayId", "") or ""),
        )


def _as_bool(value: Any, default: bool = False) -> bool:
    """Parse JSON booleans without treating the string ``"false"`` as true."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
    return bool(value)


def _scene_space(value: SceneSpace | str | None, default: SceneSpace = SceneSpace.VIDEO) -> SceneSpace:
    try:
        return value if isinstance(value, SceneSpace) else SceneSpace(str(value))
    except (TypeError, ValueError):
        return default


@dataclass
class MousePosition:
    """A single cursor position sample captured during recording.

    Coordinates are in **physical screen pixels** (not DPI-scaled).
    """
    x: float
    y: float
    timestamp: float  # ms since recording start
    # ``None`` denotes a legacy project recorded before button-state telemetry
    # existed. New recordings store a boolean on every cursor sample.
    click_state: bool | None = None
    # The first sample after a recording resume starts a new interpolation
    # span. Preview/export snap to it instead of animating across paused time.
    resume_boundary: bool = False

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON storage."""
        data = {"x": self.x, "y": self.y, "timestamp": self.timestamp}
        if self.click_state is not None:
            data["clickState"] = bool(self.click_state)
        if self.resume_boundary:
            data["resumeBoundary"] = True
        return data

    @staticmethod
    def from_dict(d: dict) -> "MousePosition":
        """Reconstruct from a dict produced by ``to_dict()``."""
        try:
            return MousePosition(
                x=d["x"],
                y=d["y"],
                timestamp=d["timestamp"],
                click_state=(bool(d["clickState"]) if "clickState" in d else None),
                resume_boundary=_as_bool(d.get("resumeBoundary"), False),
            )
        except KeyError as exc:
            raise ValueError(f"MousePosition missing required field: {exc}") from exc


@dataclass
class KeyEvent:
    """Legacy keystroke payload retained for old project compatibility.

    New recordings no longer capture or persist keystrokes, but older
    ``.fcproj`` files may still contain this data and should load safely.
    """
    timestamp: float  # ms since recording start
    x: float | None = None  # cursor x at keystroke time (physical px)
    y: float | None = None  # cursor y at keystroke time (physical px)
    vk_code: int | None = None  # Windows virtual key code

    def to_dict(self) -> dict:
        d: dict = {"timestamp": self.timestamp}
        if self.x is not None:
            d["x"] = self.x
        if self.y is not None:
            d["y"] = self.y
        if self.vk_code is not None:
            d["vkCode"] = self.vk_code
        return d

    @staticmethod
    def from_dict(d: dict) -> "KeyEvent":
        return KeyEvent(
            timestamp=d["timestamp"],
            x=d.get("x"),
            y=d.get("y"),
            vk_code=d.get("vkCode", d.get("vk_code")),
        )


@dataclass
class ClickEvent:
    """A mouse click with position and timestamp."""
    x: float
    y: float
    timestamp: float  # ms since recording start

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "timestamp": self.timestamp}

    @staticmethod
    def from_dict(d: dict) -> "ClickEvent":
        try:
            return ClickEvent(x=d["x"], y=d["y"], timestamp=d["timestamp"])
        except KeyError as exc:
            raise ValueError(f"ClickEvent missing required field: {exc}") from exc


@dataclass
class ZoomKeyframe:
    """A single zoom/pan keyframe used by the zoom engine.

    Keyframes come in pairs: a zoom-in (``zoom > 1``) and a matching
    zoom-out (``zoom = 1``).  The engine interpolates between
    consecutive keyframes using quintic ease-out easing.
    """

    id: str
    timestamp: float  # ms
    zoom: float
    x: float  # 0-1 normalized pan
    y: float
    duration: float  # ms for transition
    reason: str = ""  # human-readable reason (e.g. "Mouse activity burst")
    speed: float = 1.0  # playback speed multiplier (0.5–10.0, stored on zoom-in kf)
    is_auto_generated: bool = False  # replaceable output from local/remote Smart Zoom

    @staticmethod
    def create(
        timestamp: float,
        zoom: float,
        x: float = 0.5,
        y: float = 0.5,
        duration: float = 600.0,
        reason: str = "",
        speed: float = 1.0,
        is_auto_generated: bool = False,
    ) -> "ZoomKeyframe":
        """Factory that auto-generates a UUID for the keyframe."""
        return ZoomKeyframe(
            id=str(uuid.uuid4()),
            timestamp=timestamp,
            zoom=zoom,
            x=x,
            y=y,
            duration=duration,
            reason=reason,
            speed=speed,
            is_auto_generated=is_auto_generated,
        )

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON storage."""
        d = {
            "id": self.id,
            "timestamp": self.timestamp,
            "zoom": self.zoom,
            "x": self.x,
            "y": self.y,
            "duration": self.duration,
        }
        if self.reason:
            d["reason"] = self.reason
        if self.speed != 1.0:
            d["speed"] = self.speed
        if self.is_auto_generated:
            d["isAutoGenerated"] = True
        return d

    @staticmethod
    def from_dict(d: dict) -> "ZoomKeyframe":
        """Reconstruct from a dict, ignoring unknown keys for forward compat."""
        # Filter to only known fields to avoid TypeError from extra keys
        known = {
            "id", "timestamp", "zoom", "x", "y", "duration", "reason", "speed",
            "isAutoGenerated", "is_auto_generated",
        }
        filtered = {k: v for k, v in d.items() if k in known}
        raw_auto = filtered.pop("isAutoGenerated", filtered.pop("is_auto_generated", False))
        if isinstance(raw_auto, bool):
            filtered["is_auto_generated"] = raw_auto
        else:
            filtered["is_auto_generated"] = str(raw_auto).strip().lower() in {
                "1", "true", "yes", "on",
            }
        # Validate speed to prevent division-by-zero and hangs on
        # malformed/corrupt project files.
        raw_speed = filtered.get("speed", 1.0)
        try:
            speed = float(raw_speed)
        except (TypeError, ValueError):
            speed = 1.0
        if speed <= 0.0:
            speed = 1.0
        elif speed > 10.0:
            speed = 10.0
        filtered["speed"] = speed
        return ZoomKeyframe(**filtered)


@dataclass
class Chapter:
    """A chapter marker for navigation within a recording.

    Chapters help users navigate long recordings by marking scene boundaries.
    They can be AI-generated from shared recording context or manually created.
    """

    timestamp_ms: int  # start time of this chapter
    name: str  # display name (e.g., "Chapter 1", "Scene 2", or custom name)
    auto_detected: bool = True  # True if generated, False if manual
    is_auto_generated: bool = False  # replaceable output from AI chapter generation

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON storage."""
        data = {
            "timestampMs": self.timestamp_ms,
            "name": self.name,
            "autoDetected": self.auto_detected,
        }
        if self.is_auto_generated:
            data["isAutoGenerated"] = True
        return data

    @staticmethod
    def from_dict(d: dict) -> "Chapter":
        """Reconstruct from a dict produced by ``to_dict()``."""
        try:
            raw_auto = d.get("isAutoGenerated", d.get("is_auto_generated", False))
            if isinstance(raw_auto, bool):
                is_auto_generated = raw_auto
            else:
                is_auto_generated = str(raw_auto).strip().lower() in {
                    "1", "true", "yes", "on",
                }
            return Chapter(
                timestamp_ms=int(d["timestampMs"]),
                name=d["name"],
                auto_detected=d.get("autoDetected", True),
                is_auto_generated=is_auto_generated,
            )
        except KeyError as exc:
            raise ValueError(f"Chapter missing required field: {exc}") from exc


@dataclass
class VideoSegment:
    """A contiguous section of the recording timeline.

    A recording starts as one segment spanning the full duration.
    Splitting at the playhead subdivides it into two adjacent segments.
    Each segment can later be independently deleted or speed-adjusted.
    """

    id: str
    start_ms: float  # inclusive start time (ms since recording start)
    end_ms: float    # exclusive end time (ms)
    speed: float = 1.0  # playback speed multiplier (1.0 = normal)
    sequence_index: int = 0  # stable position on the edited output timeline

    @property
    def source_in_ms(self) -> float:
        """Inclusive source-media boundary for this output clip."""
        return float(self.start_ms)

    @source_in_ms.setter
    def source_in_ms(self, value: float) -> None:
        self.start_ms = float(value)

    @property
    def source_out_ms(self) -> float:
        """Exclusive source-media boundary for this output clip."""
        return float(self.end_ms)

    @source_out_ms.setter
    def source_out_ms(self, value: float) -> None:
        self.end_ms = float(value)

    @staticmethod
    def create(
        start_ms: float,
        end_ms: float,
        speed: float = 1.0,
        sequence_index: int = 0,
    ) -> "VideoSegment":
        """Factory that auto-generates a UUID."""
        return VideoSegment(
            id=str(uuid.uuid4()),
            start_ms=start_ms,
            end_ms=end_ms,
            speed=speed,
            sequence_index=max(0, int(sequence_index)),
        )

    def to_dict(self) -> dict:
        d: dict = {
            "id": self.id,
            "startMs": self.start_ms,
            "endMs": self.end_ms,
            "sequenceIndex": max(0, int(self.sequence_index)),
        }
        if self.speed != 1.0:
            d["speed"] = self.speed
        return d

    @staticmethod
    def from_dict(d: dict) -> "VideoSegment":
        try:
            raw_speed = d.get("speed", 1.0)
            try:
                speed = float(raw_speed)
            except (TypeError, ValueError):
                speed = 1.0
            if speed <= 0.0:
                speed = 0.1  # minimum non-zero speed to prevent division-by-zero in duration calculations
            elif speed > 10.0:
                speed = 10.0
            try:
                sequence_index = max(0, int(d.get("sequenceIndex", 0) or 0))
            except (TypeError, ValueError):
                sequence_index = 0
            return VideoSegment(
                id=d["id"],
                start_ms=d["startMs"],
                end_ms=d["endMs"],
                speed=speed,
                sequence_index=sequence_index,
            )
        except KeyError as exc:
            raise ValueError(f"VideoSegment missing required field: {exc}") from exc


@dataclass
class CanvasLayoutScene:
    """A time-bounded Canvas Space layout for the presentation group."""

    id: str
    start_ms: float
    end_ms: float
    video_scale: float = 1.0
    video_x: float = 0.0
    video_y: float = 0.0
    background_color: str = ""
    device_frame_visible: bool = True
    transition_duration_ms: float = 0.0
    transition: str = "cut"

    def __post_init__(self) -> None:
        self.start_ms = max(0.0, float(self.start_ms))
        self.end_ms = max(self.start_ms, float(self.end_ms))
        self.video_scale = max(0.05, min(4.0, float(self.video_scale)))
        self.video_x = max(-2.0, min(2.0, float(self.video_x)))
        self.video_y = max(-2.0, min(2.0, float(self.video_y)))
        self.background_color = str(self.background_color or "")
        self.device_frame_visible = bool(self.device_frame_visible)
        self.transition_duration_ms = max(0.0, min(10000.0, float(self.transition_duration_ms)))
        transition = str(self.transition or "cut").strip().lower().replace("_", "-")
        if transition in {"ease-in-out", "ease in/out", "ease", "easeinout"}:
            transition = "ease"
        else:
            transition = "cut"
        self.transition = transition

    @staticmethod
    def create(
        start_ms: float,
        end_ms: float,
        *,
        video_scale: float = 1.0,
        video_x: float = 0.0,
        video_y: float = 0.0,
        background_color: str = "",
        device_frame_visible: bool = True,
        transition_duration_ms: float = 0.0,
        transition: str = "cut",
    ) -> "CanvasLayoutScene":
        return CanvasLayoutScene(
            id=str(uuid.uuid4()),
            start_ms=start_ms,
            end_ms=end_ms,
            video_scale=video_scale,
            video_x=video_x,
            video_y=video_y,
            background_color=background_color,
            device_frame_visible=device_frame_visible,
            transition_duration_ms=transition_duration_ms,
            transition=transition,
        )

    @staticmethod
    def default(duration_ms: float) -> "CanvasLayoutScene":
        return CanvasLayoutScene.create(0.0, max(float(duration_ms), 0.0))

    def contains(self, time_ms: float) -> bool:
        return self.start_ms <= float(time_ms) <= self.end_ms

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "startMs": self.start_ms,
            "endMs": self.end_ms,
            "videoScale": self.video_scale,
            "videoX": self.video_x,
            "videoY": self.video_y,
            "backgroundColor": self.background_color,
            "deviceFrameVisible": self.device_frame_visible,
            "transitionDurationMs": self.transition_duration_ms,
            "transition": self.transition,
        }

    @staticmethod
    def from_dict(data: dict) -> "CanvasLayoutScene":
        try:
            return CanvasLayoutScene(
                id=str(data.get("id") or uuid.uuid4()),
                start_ms=float(data["startMs"]),
                end_ms=float(data["endMs"]),
                video_scale=float(data.get("videoScale", 1.0)),
                video_x=float(data.get("videoX", 0.0)),
                video_y=float(data.get("videoY", 0.0)),
                background_color=str(data.get("backgroundColor", "") or ""),
                device_frame_visible=_as_bool(data.get("deviceFrameVisible", True), True),
                transition_duration_ms=float(data.get("transitionDurationMs", 0.0)),
                transition=str(data.get("transition", data.get("transitionType", "cut")) or "cut"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"CanvasLayoutScene is invalid: {exc}") from exc


def canvas_layout_scene_at(
    scenes: List[CanvasLayoutScene] | None,
    time_ms: float,
    duration_ms: float = 0.0,
) -> CanvasLayoutScene:
    """Resolve the active scene, falling back to a full-canvas scene."""
    ordered = sorted(
        list(scenes or []),
        key=lambda scene: (float(scene.start_ms), float(scene.end_ms)),
    )
    # Later scenes win at shared boundaries, which makes a scene inserted at
    # the previous scene's end timestamp take effect immediately.
    for scene in reversed(ordered):
        if scene.contains(time_ms):
            return scene
    return CanvasLayoutScene.default(duration_ms)


def canvas_layout_transition_at(
    scenes: List[CanvasLayoutScene] | None,
    time_ms: float,
) -> tuple[CanvasLayoutScene, CanvasLayoutScene, float] | None:
    """Return ``(previous, next, progress)`` while an ease transition runs."""
    window = canvas_layout_transition_for_range(scenes, time_ms, time_ms)
    if window is None:
        return None
    previous, upcoming, transition_start, transition_end = window
    duration = max(transition_end - transition_start, 0.001)
    progress = (float(time_ms) - transition_start) / duration
    return previous, upcoming, max(0.0, min(1.0, progress))


def canvas_layout_transition_for_range(
    scenes: List[CanvasLayoutScene] | None,
    start_ms: float,
    end_ms: float | None = None,
) -> tuple[CanvasLayoutScene, CanvasLayoutScene, float, float] | None:
    """Resolve an ease window that intersects a source-time range.

    ``end_ms`` is inclusive only for a point lookup.  For a real range, the
    returned window must overlap the half-open interval ``[start_ms, end_ms)``.
    This is important when a cut divides an ease transition into two source
    segments: both halves must retain the same transition math.
    """
    ordered = sorted(
        list(scenes or []),
        key=lambda scene: (float(scene.start_ms), float(scene.end_ms), scene.id),
    )
    range_start = float(start_ms)
    point_lookup = end_ms is None or abs(float(end_ms) - range_start) <= 1e-9
    range_end = range_start if point_lookup else float(end_ms)
    for index in range(1, len(ordered)):
        previous = ordered[index - 1]
        upcoming = ordered[index]
        duration = min(
            max(float(upcoming.transition_duration_ms), 0.0),
            max(float(upcoming.start_ms) - float(previous.start_ms), 0.0),
        )
        if upcoming.transition != "ease" or duration <= 0.0:
            continue
        transition_start = float(upcoming.start_ms) - duration
        transition_end = float(upcoming.start_ms)
        if point_lookup:
            overlaps = transition_start <= range_start < transition_end
        else:
            overlaps = transition_start < range_end and transition_end > range_start
        if overlaps:
            return previous, upcoming, transition_start, transition_end
    return None


def interpolated_canvas_layout_scene(
    scenes: List[CanvasLayoutScene] | None,
    time_ms: float,
    duration_ms: float = 0.0,
) -> CanvasLayoutScene:
    """Resolve a scene and interpolate its presentation geometry for preview."""
    transition = canvas_layout_transition_at(scenes, time_ms)
    if transition is None:
        return canvas_layout_scene_at(scenes, time_ms, duration_ms)
    previous, upcoming, progress = transition
    return CanvasLayoutScene(
        id=upcoming.id,
        start_ms=previous.start_ms,
        end_ms=upcoming.end_ms,
        video_scale=previous.video_scale + (upcoming.video_scale - previous.video_scale) * progress,
        video_x=previous.video_x + (upcoming.video_x - previous.video_x) * progress,
        video_y=previous.video_y + (upcoming.video_y - previous.video_y) * progress,
        background_color=previous.background_color,
        device_frame_visible=previous.device_frame_visible,
        transition_duration_ms=upcoming.transition_duration_ms,
        transition=upcoming.transition,
    )


@dataclass
class TimelineFrame:
    """A still text or image card inserted between video ranges."""

    id: str
    timestamp_ms: float
    duration_ms: float = 2500.0
    kind: str = "text"  # "text" | "image"
    text: str = "Add your text"
    title: str = ""
    description: str = ""
    image_path: str = ""
    background_color: str = "#111827"
    text_color: str = "#f9fafb"
    font_size: int = 54
    title_font_size: int = 64
    body_font_size: int = 38
    font_family: str = "Segoe UI"
    text_alignment: str = "center"
    content_spacing: int = 22
    text_animation: str = "none"
    text_animation_duration_ms: float = 700.0
    image_fit: str = "fit"  # "fit" | "fill" (center crop)
    clip_id: str = ""

    @staticmethod
    def create(
        timestamp_ms: float,
        *,
        kind: str = "text",
        duration_ms: float = 2500.0,
        text: str = "Add your text",
        image_path: str = "",
        clip_id: str = "",
    ) -> "TimelineFrame":
        frame_kind = kind if kind in ("text", "image") else "text"
        return TimelineFrame(
            id=str(uuid.uuid4()),
            timestamp_ms=max(0.0, float(timestamp_ms)),
            duration_ms=max(250.0, float(duration_ms)),
            kind=frame_kind,
            text=text,
            image_path=image_path,
            image_fit="fill" if frame_kind == "image" else "fit",
            clip_id=str(clip_id or ""),
        )

    def to_dict(self) -> dict:
        data = {
            "id": self.id,
            "timestampMs": self.timestamp_ms,
            "durationMs": self.duration_ms,
            "kind": self.kind,
        }
        if self.text:
            data["text"] = self.text
        if self.title:
            data["title"] = self.title
        if self.description:
            data["description"] = self.description
        if self.image_path:
            data["imagePath"] = self.image_path
        if self.background_color != "#111827":
            data["backgroundColor"] = self.background_color
        if self.text_color != "#f9fafb":
            data["textColor"] = self.text_color
        if self.font_size != 54:
            data["fontSize"] = self.font_size
        if self.title_font_size != 64:
            data["titleFontSize"] = self.title_font_size
        if self.body_font_size != 38:
            data["bodyFontSize"] = self.body_font_size
        if self.font_family != "Segoe UI":
            data["fontFamily"] = self.font_family
        if self.text_alignment != "center":
            data["textAlignment"] = self.text_alignment
        if self.content_spacing != 22:
            data["contentSpacing"] = self.content_spacing
        if self.text_animation != "none":
            data["textAnimation"] = self.text_animation
        if self.text_animation_duration_ms != 700.0:
            data["textAnimationDurationMs"] = self.text_animation_duration_ms
        if self.image_fit != "fit":
            data["imageFit"] = self.image_fit
        if self.clip_id:
            data["clipId"] = self.clip_id
        return data

    @staticmethod
    def from_dict(d: dict) -> "TimelineFrame":
        try:
            kind = d.get("kind", "text")
            if kind not in ("text", "image"):
                kind = "text"
            duration_ms = float(d.get("durationMs", 2500.0))
            duration_ms = max(250.0, min(duration_ms, 600000.0))
            return TimelineFrame(
                id=d["id"],
                timestamp_ms=float(d["timestampMs"]),
                duration_ms=duration_ms,
                kind=kind,
                text=d.get("text", "Add your text"),
                title=str(d.get("title", "") or ""),
                description=str(d.get("description", "") or ""),
                image_path=d.get("imagePath", ""),
                background_color=d.get("backgroundColor", "#111827"),
                text_color=d.get("textColor", "#f9fafb"),
                font_size=max(12, min(int(d.get("fontSize", 54)), 180)),
                title_font_size=max(18, min(int(d.get("titleFontSize", 64)), 220)),
                body_font_size=max(12, min(int(d.get("bodyFontSize", 38)), 180)),
                font_family=d.get("fontFamily", "Segoe UI") or "Segoe UI",
                text_alignment=(
                    str(d.get("textAlignment", "center")).lower()
                    if str(d.get("textAlignment", "center")).lower() in {"left", "center", "right"}
                    else "center"
                ),
                content_spacing=max(0, min(int(d.get("contentSpacing", 22)), 160)),
                text_animation=(
                    str(d.get("textAnimation", "none")).strip().lower()
                    if str(d.get("textAnimation", "none")).strip().lower()
                    in {"none", "fade", "fade-slide", "soft-reveal"}
                    else "none"
                ),
                text_animation_duration_ms=max(
                    100.0,
                    min(float(d.get("textAnimationDurationMs", 700.0)), 5000.0),
                ),
                image_fit=("fill" if d.get("imageFit") == "fill" else "fit"),
                clip_id=str(d.get("clipId", "") or ""),
            )
        except KeyError as exc:
            raise ValueError(f"TimelineFrame missing required field: {exc}") from exc


SCREEN_TRANSITION_EFFECTS = {
    "directional_push",
    "axis_flip",
    "scale_swap",
    "zoom_through",
    "graphic_vertical_bars",
    "graphic_horizontal_bars",
    "graphic_diagonal_bars",
    "graphic_split_in",
    "graphic_split_out",
    "graphic_sweep",
    "graphic_fold",
}
GRAPHIC_SCREEN_TRANSITION_EFFECTS = {
    effect for effect in SCREEN_TRANSITION_EFFECTS if effect.startswith("graphic_")
}
LEGACY_SCREEN_TRANSITION_EFFECTS = {
    "smooth_settle": "scale_swap",
    "blur_dissolve": "zoom_through",
    "dip_to_canvas": "scale_swap",
}
SCREEN_TRANSITION_DIRECTIONS = {"left", "right", "up", "down"}
SCREEN_TRANSITION_BAR_ORIENTATIONS = {"auto", "vertical", "horizontal", "diagonal"}
SCREEN_TRANSITION_EASINGS = {"linear", "ease_out", "quintic_in_out"}
SCREEN_TRANSITION_COLOR_PRESETS = {
    "zumly_editorial",
    "modern_editorial",
    "cool_spectrum",
    "dark_premium",
    "warm_creative",
}
LEGACY_SCREEN_TRANSITION_COLOR_PRESETS = {
    "editorial_bars": "zumly_editorial",
    "brand_film": "zumly_editorial",
    "brand_cyan": "zumly_editorial",
    "alternating_brand": "zumly_editorial",
    "brand_purple": "cool_spectrum",
    "deep_focus": "dark_premium",
    "dark": "dark_premium",
    "light": "zumly_editorial",
    # Legacy custom solid palettes now resolve to the curated default.
    "custom": "zumly_editorial",
}
SCREEN_TRANSITION_BAR_WIDTH_MODES = {"uniform", "varied", "seeded"}
SCREEN_TRANSITION_BAR_STYLES = {
    "material_mix",
    "material_vinyl",
    "material_paint",
    "material_leather",
    "material_cloth",
}
LEGACY_SCREEN_TRANSITION_BAR_STYLES = {
    "flat": "material_mix",
    "gradient_depth": "material_mix",
    "editorial_matte": "material_mix",
    "soft_grain": "material_mix",
}
SCREEN_TRANSITION_ENTER_EXIT_MODES = {"opposed", "same"}
DEFAULT_SCREEN_TRANSITION_DURATION_MS = 2000.0
MAX_SCREEN_TRANSITION_DURATION_MS = 10000.0


@dataclass
class ScreenTransition:
    """A non-destructive visual bridge inserted at a source-time boundary."""

    id: str
    timestamp_ms: float
    effect_type: str = "scale_swap"
    direction: str = "left"
    bar_orientation: str = "auto"
    bar_count: int = 8
    # Legacy-only compatibility field. Graphic bars always tile the aperture.
    bar_gap: float = 0.0
    bar_stagger: float = 0.18
    bar_width_mode: str = "varied"
    bar_style: str = "material_mix"
    bar_seed: int = 0
    easing: str = "quintic_in_out"
    color_preset: str = "zumly_editorial"
    custom_color: str = "#6D2BD6"
    enter_exit_mode: str = "opposed"
    duration_ms: float = DEFAULT_SCREEN_TRANSITION_DURATION_MS
    enabled: bool = True
    suggested: bool = False
    change_score: float | None = None
    outgoing_frame_ms: float | None = None
    incoming_frame_ms: float | None = None
    clip_id: str = ""
    legacy_effect_type: str = ""
    migration_pending: bool = False

    @staticmethod
    def create(
        timestamp_ms: float,
        *,
        effect_type: str = "scale_swap",
        direction: str = "left",
        bar_orientation: str = "auto",
        bar_count: int = 8,
        bar_gap: float = 0.0,
        bar_stagger: float = 0.18,
        bar_width_mode: str = "varied",
        bar_style: str = "material_mix",
        bar_seed: int = 0,
        easing: str = "quintic_in_out",
        color_preset: str = "zumly_editorial",
        custom_color: str = "#6D2BD6",
        enter_exit_mode: str = "opposed",
        duration_ms: float = DEFAULT_SCREEN_TRANSITION_DURATION_MS,
        enabled: bool = True,
        suggested: bool = False,
        transition_id: str = "",
        change_score: float | None = None,
        outgoing_frame_ms: float | None = None,
        incoming_frame_ms: float | None = None,
        clip_id: str = "",
    ) -> "ScreenTransition":
        requested_effect = str(effect_type or "scale_swap")
        legacy_effect = (
            requested_effect
            if requested_effect in LEGACY_SCREEN_TRANSITION_EFFECTS
            else ""
        )
        effect = LEGACY_SCREEN_TRANSITION_EFFECTS.get(
            requested_effect, requested_effect
        )
        if effect not in SCREEN_TRANSITION_EFFECTS:
            effect = "scale_swap"
        normalized_direction = str(direction or "left").lower()
        if normalized_direction not in SCREEN_TRANSITION_DIRECTIONS:
            normalized_direction = "left"
        normalized_orientation = str(bar_orientation or "auto").lower()
        if normalized_orientation not in SCREEN_TRANSITION_BAR_ORIENTATIONS:
            normalized_orientation = "auto"
        normalized_easing = str(easing or "quintic_in_out").lower()
        if normalized_easing not in SCREEN_TRANSITION_EASINGS:
            normalized_easing = "quintic_in_out"
        normalized_width_mode = str(bar_width_mode or "varied").lower()
        if normalized_width_mode not in SCREEN_TRANSITION_BAR_WIDTH_MODES:
            normalized_width_mode = "varied"
        normalized_bar_style = str(bar_style or "material_mix").lower()
        normalized_bar_style = LEGACY_SCREEN_TRANSITION_BAR_STYLES.get(
            normalized_bar_style, normalized_bar_style
        )
        if normalized_bar_style not in SCREEN_TRANSITION_BAR_STYLES:
            normalized_bar_style = "material_mix"
        normalized_color_preset = str(color_preset or "zumly_editorial").lower()
        normalized_color_preset = LEGACY_SCREEN_TRANSITION_COLOR_PRESETS.get(
            normalized_color_preset, normalized_color_preset
        )
        if normalized_color_preset not in SCREEN_TRANSITION_COLOR_PRESETS:
            normalized_color_preset = "zumly_editorial"
        normalized_enter_exit = str(enter_exit_mode or "opposed").lower()
        if normalized_enter_exit not in SCREEN_TRANSITION_ENTER_EXIT_MODES:
            normalized_enter_exit = "opposed"
        normalized_custom_color = str(custom_color or "#6D2BD6").strip().upper()
        if (
            len(normalized_custom_color) != 7
            or not normalized_custom_color.startswith("#")
            or any(
                character not in "0123456789ABCDEF"
                for character in normalized_custom_color[1:]
            )
        ):
            normalized_custom_color = "#6D2BD6"
        try:
            normalized_score = (
                None
                if change_score is None
                else max(0.0, min(float(change_score), 1.0))
            )
        except (TypeError, ValueError):
            normalized_score = None
        return ScreenTransition(
            id=str(transition_id or uuid.uuid4()),
            timestamp_ms=max(0.0, float(timestamp_ms)),
            effect_type=effect,
            direction=normalized_direction,
            bar_orientation=normalized_orientation,
            bar_count=max(2, min(int(bar_count), 20)),
            bar_gap=0.0,
            bar_stagger=max(0.0, min(float(bar_stagger), 0.8)),
            bar_width_mode=normalized_width_mode,
            bar_style=normalized_bar_style,
            bar_seed=max(0, min(int(bar_seed), 0xFFFFFFFF)),
            easing=normalized_easing,
            color_preset=normalized_color_preset,
            custom_color=normalized_custom_color,
            enter_exit_mode=normalized_enter_exit,
            duration_ms=max(
                150.0,
                min(float(duration_ms), MAX_SCREEN_TRANSITION_DURATION_MS),
            ),
            enabled=bool(enabled),
            suggested=bool(suggested),
            change_score=normalized_score,
            outgoing_frame_ms=(
                None if outgoing_frame_ms is None else max(0.0, float(outgoing_frame_ms))
            ),
            incoming_frame_ms=(
                None if incoming_frame_ms is None else max(0.0, float(incoming_frame_ms))
            ),
            clip_id=str(clip_id or ""),
            legacy_effect_type=legacy_effect,
            migration_pending=bool(legacy_effect),
        )

    def to_dict(self) -> dict:
        data = {
            "id": self.id,
            "timestampMs": self.timestamp_ms,
            "effectType": (
                self.legacy_effect_type
                if self.migration_pending and self.legacy_effect_type
                else self.effect_type
            ),
            "direction": self.direction,
            "durationMs": self.duration_ms,
            "enabled": self.enabled,
            "suggested": self.suggested,
        }
        if self.effect_type in GRAPHIC_SCREEN_TRANSITION_EFFECTS:
            data.update(
                {
                    "barOrientation": self.bar_orientation,
                    "barCount": self.bar_count,
                    "barStagger": self.bar_stagger,
                    "barWidthMode": self.bar_width_mode,
                    "barStyle": self.bar_style,
                    "barSeed": self.bar_seed,
                    "easing": self.easing,
                    "colorPreset": self.color_preset,
                    "customColor": self.custom_color,
                    "enterExitMode": self.enter_exit_mode,
                }
            )
        if self.change_score is not None:
            data["changeScore"] = max(0.0, min(float(self.change_score), 1.0))
        if self.outgoing_frame_ms is not None:
            data["outgoingFrameMs"] = max(0.0, float(self.outgoing_frame_ms))
        if self.incoming_frame_ms is not None:
            data["incomingFrameMs"] = max(0.0, float(self.incoming_frame_ms))
        if self.clip_id:
            data["clipId"] = self.clip_id
        return data

    @staticmethod
    def from_dict(data: dict) -> "ScreenTransition":
        transition = ScreenTransition.create(
            float(data.get("timestampMs", 0.0)),
            effect_type=str(data.get("effectType", "scale_swap")),
            direction=str(data.get("direction", "left")),
            bar_orientation=str(data.get("barOrientation", "auto")),
            bar_count=int(data.get("barCount", 8)),
            bar_gap=float(data.get("barGap", 0.0)),
            bar_stagger=float(data.get("barStagger", 0.18)),
            bar_width_mode=str(data.get("barWidthMode", "varied")),
            bar_style=str(data.get("barStyle", "material_mix")),
            bar_seed=int(data.get("barSeed", 0)),
            easing=str(data.get("easing", "quintic_in_out")),
            color_preset=str(data.get("colorPreset", "zumly_editorial")),
            custom_color=str(data.get("customColor", "#6D2BD6")),
            enter_exit_mode=str(data.get("enterExitMode", "opposed")),
            duration_ms=float(
                data.get("durationMs", DEFAULT_SCREEN_TRANSITION_DURATION_MS)
            ),
            enabled=_as_bool(data.get("enabled", True), True),
            suggested=_as_bool(data.get("suggested", False), False),
            change_score=data.get("changeScore"),
            outgoing_frame_ms=data.get("outgoingFrameMs"),
            incoming_frame_ms=data.get("incomingFrameMs"),
            clip_id=str(data.get("clipId", "") or ""),
        )
        transition.id = str(data.get("id") or transition.id)
        return transition


@dataclass
class BackgroundMusic:
    """One project-wide music bed, independent from source timeline clips."""

    asset_id: str
    asset_path: str = ""
    title: str = ""
    volume: float = 0.15
    enable_ducking: bool = True

    def __post_init__(self) -> None:
        self.asset_id = str(self.asset_id or "").strip()
        if not self.asset_id:
            raise ValueError("BackgroundMusic requires an asset_id")
        self.asset_path = str(self.asset_path or "").strip()
        self.title = str(self.title or "").strip()
        self.volume = max(0.0, min(1.0, float(self.volume)))
        self.enable_ducking = bool(self.enable_ducking)

    @property
    def is_custom(self) -> bool:
        return self.asset_id.startswith("custom:")

    def to_dict(self) -> dict:
        data = {
            "assetId": self.asset_id,
            "volume": self.volume,
            "enableDucking": self.enable_ducking,
        }
        if self.asset_path:
            data["assetPath"] = self.asset_path
        if self.title:
            data["title"] = self.title
        return data

    @staticmethod
    def from_dict(data: dict) -> "BackgroundMusic":
        if not isinstance(data, dict):
            raise ValueError("BackgroundMusic must be a JSON object")
        return BackgroundMusic(
            asset_id=str(data.get("assetId", data.get("asset_id", "")) or ""),
            asset_path=str(data.get("assetPath", data.get("asset_path", "")) or ""),
            title=str(data.get("title", "") or ""),
            volume=float(data.get("volume", 0.15)),
            enable_ducking=_as_bool(
                data.get("enableDucking", data.get("enable_ducking", True)),
                True,
            ),
        )


def _background_music_from_payload(data: dict) -> BackgroundMusic | None:
    """Read the global contract, or migrate the first valid legacy clip."""
    current = data.get("backgroundMusic")
    if isinstance(current, dict):
        return BackgroundMusic.from_dict(current)

    legacy = data.get("musicClips")
    if not isinstance(legacy, list):
        return None
    for item in legacy:
        if not isinstance(item, dict):
            continue
        try:
            return BackgroundMusic.from_dict(item)
        except (TypeError, ValueError):
            continue
    return None


MIN_CURSOR_SCALE = 0.875
DEFAULT_CURSOR_SCALE = 1.75
MAX_CURSOR_SCALE = 2.625


def normalize_cursor_scale(
    value: float | int | None,
    default: float = DEFAULT_CURSOR_SCALE,
) -> float:
    """Clamp the renderer scale used by both preview and export."""
    try:
        parsed = float(default if value is None else value)
    except (TypeError, ValueError):
        parsed = float(default)
    return max(MIN_CURSOR_SCALE, min(MAX_CURSOR_SCALE, parsed))


@dataclass
class RecordingSession:
    """Top-level container for everything captured in one recording.

    Includes mouse track, click events, zoom keyframes, trim points,
    and per-frame timestamps. Legacy keystroke data may still be held in
    ``key_events`` while older project files are being loaded, but it is
    no longer serialized back out.
    """

    id: str
    start_time: float
    duration: float
    mouse_track: List[MousePosition]
    keyframes: List[ZoomKeyframe]
    key_events: List[KeyEvent] | None = None  # legacy load-only data
    click_events: List[ClickEvent] | None = None
    frame_timestamps: List[float] | None = None
    trim_start_ms: float = 0.0
    trim_end_ms: float = 0.0  # 0 = no trim (use full duration)
    voiceover_segments: List["VoiceoverSegment"] | None = None
    video_segments: List["VideoSegment"] | None = None
    timeline_frames: List["TimelineFrame"] | None = None
    screen_transitions: List["ScreenTransition"] | None = None
    dismissed_screen_transition_ids: List[str] | None = None
    chapters: List["Chapter"] | None = None
    highlights: List["HighlightBox"] | None = None
    text_annotations: List["TextAnnotation"] | None = None
    timeline_overlays: List[TimelineOverlay] | None = None
    redaction_suggestions: List[RedactionSuggestion] | None = None
    canvas_layout_scenes: List[CanvasLayoutScene] | None = None
    explainer_scenes: List["ExplainerScene"] | None = None
    background_music: BackgroundMusic | None = None
    is_cfr: bool = False
    capture_telemetry: dict[str, Any] | None = None
    
    # Aesthetic settings
    background_id: str | None = None
    frame_id: str | None = None
    click_effect_id: str | None = None
    cursor_style_id: str = "arrow"
    cursor_asset_path: str = ""
    cursor_hotspot: tuple[float, float] | None = None
    # Physical render factor.  The UI presents this as 70%-100%, where 100%
    # is the demo-sized cursor and 70% is the smallest supported size.
    cursor_scale: float = DEFAULT_CURSOR_SCALE
    output_dimensions: list | str | None = None  # e.g. [1920, 1080] or "auto"

    def __post_init__(self) -> None:
        if self.canvas_layout_scenes is None:
            self.canvas_layout_scenes = [CanvasLayoutScene.default(self.duration)]

    def to_json(self) -> str:
        """Serialize the entire session to a JSON string."""
        data = {
            "id": self.id,
            "startTime": self.start_time,
            "duration": self.duration,
            "mouseTrack": [m.to_dict() for m in self.mouse_track],
            "keyframes": [k.to_dict() for k in self.keyframes],
        }
        if self.click_events:
            data["clickEvents"] = [c.to_dict() for c in self.click_events]
        if self.frame_timestamps:
            data["frameTimestamps"] = self.frame_timestamps
        if self.trim_start_ms > 0:
            data["trimStartMs"] = self.trim_start_ms
        if self.trim_end_ms > 0:
            data["trimEndMs"] = self.trim_end_ms
        if self.voiceover_segments:
            data["voiceoverSegments"] = [s.to_dict() for s in self.voiceover_segments]
        if self.video_segments:
            data["videoSegments"] = [s.to_dict() for s in self.video_segments]
        if self.timeline_frames:
            data["timelineFrames"] = [f.to_dict() for f in self.timeline_frames]
        if self.screen_transitions is not None:
            data["screenTransitions"] = [item.to_dict() for item in self.screen_transitions]
        if self.dismissed_screen_transition_ids:
            data["dismissedScreenTransitionIds"] = sorted(
                {str(item) for item in self.dismissed_screen_transition_ids if str(item)}
            )
        if self.chapters:
            data["chapters"] = [c.to_dict() for c in self.chapters]
        if self.highlights:
            data["highlights"] = [h.to_dict() for h in self.highlights]
        if self.text_annotations:
            data["textAnnotations"] = [item.to_dict() for item in self.text_annotations]
        if self.timeline_overlays:
            data["timelineOverlays"] = [item.to_dict() for item in self.timeline_overlays]
        if self.redaction_suggestions:
            data["redactionSuggestions"] = [
                item.to_dict() for item in self.redaction_suggestions
            ]
        if self.canvas_layout_scenes:
            data["canvasLayoutScenes"] = [scene.to_dict() for scene in self.canvas_layout_scenes]
        if self.explainer_scenes:
            data["explainerScenes"] = [scene.to_dict() for scene in self.explainer_scenes]
        if self.background_music is not None:
            data["backgroundMusic"] = self.background_music.to_dict()
        data["isCfr"] = self.is_cfr
        if self.capture_telemetry is not None:
            data["captureTelemetry"] = self.capture_telemetry
            
        # Add aesthetic settings if they exist
        if self.background_id:
            data["backgroundId"] = self.background_id
        if self.frame_id:
            data["frameId"] = self.frame_id
        if self.click_effect_id:
            data["clickEffectId"] = self.click_effect_id
        if self.cursor_asset_path:
            data["cursorAssetPath"] = self.cursor_asset_path
        if self.cursor_style_id or self.cursor_asset_path:
            # Older callers only know about cursorAssetPath. Preserve that
            # payload as an explicit custom style on the next round trip.
            style_id = self.cursor_style_id or "arrow"
            if self.cursor_asset_path and style_id == "arrow":
                style_id = "custom"
            data["cursorStyleId"] = style_id
        if self.cursor_hotspot is not None:
            data["cursorHotspot"] = [
                max(0.0, float(self.cursor_hotspot[0])),
                max(0.0, float(self.cursor_hotspot[1])),
            ]
        data["cursorScale"] = normalize_cursor_scale(self.cursor_scale)
        if self.output_dimensions:
            data["outputDimensions"] = self.output_dimensions
        return json.dumps(data, indent=2)

    @staticmethod
    def from_json(s: str) -> "RecordingSession":
        """Reconstruct a full session from its JSON representation.

        Tolerates missing optional fields for backward compatibility with
        older .fcproj versions.  Required fields (``id``, ``startTime``,
        ``duration``, ``mouseTrack``) raise ``ValueError`` with a clear
        message instead of raw ``KeyError``.  ``keyframes`` is optional
        (defaults to an empty list when absent) for backward compatibility.
        """
        """Deserialize from a JSON string."""
        d = json.loads(s)

        try:
            session_id = d["id"]
            start_time = d["startTime"]
            duration = d["duration"]
        except KeyError as exc:
            raise ValueError(f"RecordingSession missing required field: {exc}") from exc
        
        # Parse simple tracks
        mouse_track = [MousePosition.from_dict(m) for m in d.get("mouseTrack", [])]
        keyframes = [ZoomKeyframe.from_dict(k) for k in d.get("keyframes", [])]
        click_events = None
        if "clickEvents" in d:
            click_events = [ClickEvent.from_dict(c) for c in d["clickEvents"]]
        
        # Legacy fallback
        key_events = None
        if "keyEvents" in d:
            logger.debug("Ignoring legacy keyEvents during RecordingSession load")
            
        voiceover_segments = None
        if "voiceoverSegments" in d:
            voiceover_segments = [VoiceoverSegment.from_dict(v) for v in d["voiceoverSegments"]]
            
        video_segments = None
        if "videoSegments" in d:
            video_segments = [VideoSegment.from_dict(v) for v in d["videoSegments"]]
            # Legacy payloads used list order as their only ordering contract.
            # Normalize every load so stale or duplicate sequence values cannot
            # reorder copied clips unexpectedly.
            for sequence_index, segment in enumerate(video_segments):
                segment.sequence_index = sequence_index

        timeline_frames = None
        if "timelineFrames" in d:
            timeline_frames = [TimelineFrame.from_dict(v) for v in d["timelineFrames"]]

        screen_transitions = None
        if isinstance(d.get("screenTransitions"), list):
            screen_transitions = [
                ScreenTransition.from_dict(item) for item in d["screenTransitions"]
            ]
            
        chapters = None
        if "chapters" in d:
            chapters = [Chapter.from_dict(c) for c in d["chapters"]]

        highlights = None
        if "highlights" in d:
            highlights = [HighlightBox.from_dict(h) for h in d["highlights"]]

        text_annotations = None
        if "textAnnotations" in d:
            text_annotations = [TextAnnotation.from_dict(item) for item in d["textAnnotations"]]

        # ``overlays`` was used by a short-lived internal prototype. Keep its
        # payload readable, but serialize only the durable timelineOverlays key.
        raw_overlays = d.get("timelineOverlays", d.get("overlays"))
        timeline_overlays = None
        if isinstance(raw_overlays, list):
            timeline_overlays = [TimelineOverlay.from_dict(item) for item in raw_overlays]

        redaction_suggestions = None
        if isinstance(d.get("redactionSuggestions"), list):
            redaction_suggestions = [
                RedactionSuggestion.from_dict(item)
                for item in d["redactionSuggestions"]
                if isinstance(item, dict)
            ]

        raw_layout_scenes = d.get("canvasLayoutScenes")
        if isinstance(raw_layout_scenes, list):
            canvas_layout_scenes = [CanvasLayoutScene.from_dict(item) for item in raw_layout_scenes]
        else:
            canvas_layout_scenes = [CanvasLayoutScene.default(duration)]

        explainer_scenes = None
        if isinstance(d.get("explainerScenes"), list):
            explainer_scenes = [
                ExplainerScene.from_dict(item) for item in d["explainerScenes"]
            ]

        is_cfr = d.get("isCfr", False)
        background_music = _background_music_from_payload(d)

        raw_hotspot = d.get("cursorHotspot")
        cursor_hotspot = None
        if isinstance(raw_hotspot, (list, tuple)) and len(raw_hotspot) == 2:
            try:
                cursor_hotspot = (
                    max(0.0, float(raw_hotspot[0])),
                    max(0.0, float(raw_hotspot[1])),
                )
            except (TypeError, ValueError):
                cursor_hotspot = None

        try:
            # A missing value identifies a project from before cursor sizing
            # was persisted. Preserve that release's 2.5 visual default.
            cursor_scale = normalize_cursor_scale(d.get("cursorScale", 2.5), 2.5)
        except (TypeError, ValueError):
            cursor_scale = 2.5

        return RecordingSession(
            id=session_id,
            start_time=start_time,
            duration=duration,
            mouse_track=mouse_track,
            keyframes=keyframes,
            key_events=key_events,
            click_events=click_events,
            frame_timestamps=d.get("frameTimestamps"),
            trim_start_ms=d.get("trimStartMs", 0.0),
            trim_end_ms=d.get("trimEndMs", 0.0),
            voiceover_segments=voiceover_segments,
            video_segments=video_segments,
            timeline_frames=timeline_frames,
            screen_transitions=screen_transitions,
            dismissed_screen_transition_ids=[
                str(item)
                for item in d.get("dismissedScreenTransitionIds", [])
                if str(item)
            ],
            chapters=chapters,
            highlights=highlights,
            text_annotations=text_annotations,
            timeline_overlays=timeline_overlays,
            redaction_suggestions=redaction_suggestions,
            canvas_layout_scenes=canvas_layout_scenes,
            explainer_scenes=explainer_scenes,
            background_music=background_music,
            is_cfr=is_cfr,
            capture_telemetry=d.get("captureTelemetry"),
            background_id=d.get("backgroundId"),
            frame_id=d.get("frameId"),
            click_effect_id=d.get("clickEffectId"),
            cursor_style_id=str(
                d.get("cursorStyleId", "custom" if d.get("cursorAssetPath") else "arrow")
                or "arrow"
            ),
            cursor_asset_path=str(d.get("cursorAssetPath", "") or ""),
            cursor_hotspot=cursor_hotspot,
            cursor_scale=cursor_scale,
            output_dimensions=d.get("outputDimensions")
        )


@dataclass
class VoiceoverSegment:
    """A single voiceover segment with text, position, and audio.

    Segments can be user-authored (manual) or AI-generated narration.
    TTS synthesis converts the spoken text to speech and stores the
    audio file path.  Generated narration may also keep a markdown
    script for save/load roundtrips and file export.
    """

    id: str
    timestamp: float  # ms — start position on the timeline
    text: str  # user-authored voiceover text
    voice: str = DEFAULT_VOICEOVER_VOICE  # TTS voice name
    audio_path: str = ""  # path to synthesized audio file (empty = not yet synthesized)
    duration_ms: float = 0.0  # audio duration in ms (0 = unknown/not synthesized)
    rate: float = 1.0  # speech rate multiplier (0.0–3.0, 1.0 = normal)
    volume: float = 1.0  # volume multiplier (0.0–3.0, 1.0 = normal)
    source: str = "manual"  # "manual" | "generated"
    script_markdown: str = ""  # markdown script for generated narration
    script_path: str = ""  # last exported markdown path on disk
    # Runtime-only: True while TTS synthesis is actively in progress.
    # Never persisted — loaded segments always start with False.
    tts_generating: bool = field(default=False, compare=False, repr=False)

    @staticmethod
    def create(
        timestamp: float,
        text: str,
        voice: str = DEFAULT_VOICEOVER_VOICE,
        rate: float = 1.0,
        volume: float = 1.0,
        source: str = "manual",
        script_markdown: str = "",
        script_path: str = "",
    ) -> "VoiceoverSegment":
        """Factory that auto-generates a UUID."""
        return VoiceoverSegment(
            id=str(uuid.uuid4()),
            timestamp=timestamp,
            text=text,
            voice=voice,
            rate=rate,
            volume=volume,
            source=source,
            script_markdown=script_markdown,
            script_path=script_path,
        )

    def to_dict(self) -> dict:
        d: dict = {
            "id": self.id,
            "timestamp": self.timestamp,
            "text": self.text,
            "voice": self.voice,
        }
        if self.duration_ms > 0:
            d["durationMs"] = self.duration_ms
        if self.rate != 1.0:
            d["rate"] = self.rate
        if self.volume != 1.0:
            d["volume"] = self.volume
        if self.source != "manual":
            d["source"] = self.source
        if self.script_markdown:
            d["scriptMarkdown"] = self.script_markdown
        if self.script_path:
            d["scriptPath"] = self.script_path
        return d

    @property
    def is_generated_narration(self) -> bool:
        """True when this segment came from the AI narration pipeline."""
        return self.source == "generated"

    @property
    def generated_narration_label(self) -> str:
        """Return the section label for a generated narration segment."""
        if not self.script_markdown:
            return "AI narration"

        mapping = {
            "context": "Context",
            "background": "Background",
            "prompt / action": "Prompt / Action",
            "prompt/action": "Prompt / Action",
            "action": "Prompt / Action",
            "walkthrough": "Walkthrough",
            "result": "Result",
        }
        for raw_line in self.script_markdown.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                heading = " ".join(stripped.lstrip("#").strip().replace("/", " / ").split())
                return mapping.get(heading.lower(), heading or "AI narration")
            break
        return "AI narration"

    @staticmethod
    def from_dict(d: dict) -> "VoiceoverSegment":
        try:
            # Validate rate and volume bounds
            raw_rate = d.get("rate", 1.0)
            try:
                rate = float(raw_rate)
            except (TypeError, ValueError):
                rate = 1.0
            rate = max(0.0, min(3.0, rate))

            raw_volume = d.get("volume", 1.0)
            try:
                volume = float(raw_volume)
            except (TypeError, ValueError):
                volume = 1.0
            volume = max(0.0, min(3.0, volume))

            return VoiceoverSegment(
                id=d["id"],
                timestamp=d["timestamp"],
                text=d["text"],
                voice=d.get("voice", DEFAULT_VOICEOVER_VOICE),
                duration_ms=d.get("durationMs", 0.0),
                rate=rate,
                volume=volume,
                source=str(d.get("source", "manual") or "manual"),
                script_markdown=d.get("scriptMarkdown", ""),
                script_path=d.get("scriptPath", ""),
            )
        except KeyError as exc:
            raise ValueError(f"VoiceoverSegment missing required field: {exc}") from exc


@dataclass
class ClickEffectPreset:
    """A click effect style preset with color, style, duration, and radius.

    Defines the visual appearance of click ripple effects in preview and export.
    """

    name: str
    color: tuple[int, int, int, int]  # RGBA (0-255)
    style: str  # "ripple" | "burst" | "highlight"
    duration_ms: int
    radius: int

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "color": list(self.color),
            "style": self.style,
            "durationMs": self.duration_ms,
            "radius": self.radius,
        }

    @staticmethod
    def from_dict(d: dict) -> "ClickEffectPreset":
        try:
            raw_color = list(d["color"])
            # Coerce to 4 RGBA ints clamped to 0-255
            while len(raw_color) < 4:
                raw_color.append(255)
            color = tuple(max(0, min(255, int(c))) for c in raw_color[:4])
            return ClickEffectPreset(
                name=d["name"],
                color=color,
                style=d.get("style", "ripple"),
                duration_ms=d.get("durationMs", 400),
                radius=d.get("radius", 24),
            )
        except KeyError as exc:
            raise ValueError(f"ClickEffectPreset missing required field: {exc}") from exc


# ── Built-in click effect presets ───────────────────────────────────

CLICK_EFFECT_PRESETS = [
    ClickEffectPreset("Subtle Purple", (138, 92, 246, 220), "ripple", 400, 24),
    ClickEffectPreset("Bold Red", (239, 68, 68, 240), "ripple", 350, 28),
    ClickEffectPreset("Neon Cyan", (34, 211, 238, 230), "ripple", 450, 26),
    ClickEffectPreset("Minimal Gray", (156, 163, 175, 180), "ripple", 300, 20),
    ClickEffectPreset("High Contrast Yellow", (250, 204, 21, 250), "ripple", 380, 30),
    ClickEffectPreset("Clean White", (255, 255, 255, 200), "ripple", 350, 22),
    ClickEffectPreset("Soft Green", (74, 222, 128, 210), "ripple", 400, 24),
    ClickEffectPreset("Invisible", (0, 0, 0, 0), "ripple", 0, 0),
]

DEFAULT_CLICK_EFFECT = CLICK_EFFECT_PRESETS[0]  # Subtle Purple


@dataclass
class KeystrokeOverlayConfig:
    """Configuration for keystroke visualization overlay.

    Controls how keystrokes are rendered during video export and preview.
    """
    enabled: bool = False
    position: str = "bottom-center"  # "bottom-center", "bottom-left", "near-cursor"
    style: str = "floating-badge"    # "floating-badge", "minimal-text", "key-cap"
    display_duration_ms: int = 1500  # how long keystrokes remain visible
    filter_mode: str = "shortcuts-only"  # "all", "modifiers-only", "shortcuts-only"
    font_size: int = 18
    opacity: float = 0.85            # 0.0 - 1.0

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON storage."""
        return {
            "enabled": self.enabled,
            "position": self.position,
            "style": self.style,
            "displayDurationMs": self.display_duration_ms,
            "filterMode": self.filter_mode,
            "fontSize": self.font_size,
            "opacity": self.opacity,
        }

    @staticmethod
    def from_dict(d: dict) -> "KeystrokeOverlayConfig":
        """Reconstruct from a dict produced by ``to_dict()``."""
        raw_mode = d.get("filterMode", "shortcuts-only")
        if raw_mode not in VALID_FILTER_MODES:
            logger.warning(
                "Unknown keystroke filter_mode %r, defaulting to 'shortcuts-only'",
                raw_mode,
            )
            raw_mode = "shortcuts-only"
        return KeystrokeOverlayConfig(
            enabled=d.get("enabled", False),
            position=d.get("position", "bottom-center"),
            style=d.get("style", "floating-badge"),
            display_duration_ms=d.get("displayDurationMs", 1500),
            filter_mode=raw_mode,
            font_size=d.get("fontSize", 18),
            opacity=d.get("opacity", 0.85),
        )


@dataclass
class TextAnnotation:
    """Time-bounded text anchored to normalized absolute Canvas Space.

    ``x`` and ``y`` are normalized against the final export resolution. They
    never pass through the Video Space zoom/pan transform.
    """
    id: str
    start_ms: float
    end_ms: float
    x: float  # 0-1 normalized Canvas Space left anchor
    y: float  # 0-1 normalized Canvas Space top anchor
    text: str
    font_family: str = "Segoe UI"
    font_size: int = 18
    color: tuple[int, int, int, int] = (255, 255, 255, 255)  # RGBA
    opacity: float = 1.0
    background_color: tuple[int, int, int, int] | None = (30, 30, 30, 200)  # RGBA or None for no background
    max_width: float = 0.84
    text_width: float = 0.0
    text_height: float = 0.0
    vertical_alignment: str = "top"
    horizontal_alignment: str = "auto"
    animation: str = "none"
    animation_delay_ms: float = 0.0
    animation_in_ms: float = 350.0
    animation_out_ms: float = 300.0
    slide_offset_x: float = 0.0
    slide_offset_y: float = 0.0
    space: SceneSpace = SceneSpace.CANVAS

    def __post_init__(self) -> None:
        # TextAnnotation is intentionally not polymorphic across scene spaces.
        # A separate Video Space annotation type should be introduced if that
        # behavior is needed later.
        self.space = SceneSpace.CANVAS
        self.x = max(0.0, min(1.0, float(self.x)))
        self.y = max(0.0, min(1.0, float(self.y)))
        self.opacity = max(0.0, min(1.0, float(self.opacity)))
        self.max_width = max(0.05, min(1.0, float(self.max_width)))
        self.text_width = max(0.0, min(1.0, float(self.text_width)))
        self.text_height = max(0.0, min(1.0, float(self.text_height)))
        alignment = str(self.vertical_alignment or "top").strip().lower()
        self.vertical_alignment = alignment if alignment in {"top", "center", "bottom"} else "top"
        horizontal = str(self.horizontal_alignment or "auto").strip().lower()
        self.horizontal_alignment = (
            horizontal if horizontal in {"auto", "left", "center", "right"} else "auto"
        )
        animation = str(self.animation or "none").strip().lower()
        self.animation = animation if animation in {
            "none", "fade", "fade-slide", "soft-reveal"
        } else "none"
        self.animation_delay_ms = max(0.0, float(self.animation_delay_ms))
        self.animation_in_ms = max(1.0, float(self.animation_in_ms))
        self.animation_out_ms = max(1.0, float(self.animation_out_ms))
        self.font_size = max(8, min(int(self.font_size), 512))
        self.font_family = str(self.font_family or "Segoe UI")
    
    @staticmethod
    def create(
        start_ms: float,
        end_ms: float,
        x: float = 0.5,
        y: float = 0.5,
        text: str = "Text",
        font_family: str = "Segoe UI",
        font_size: int = 18,
        color: tuple[int, int, int, int] = (255, 255, 255, 255),
        opacity: float = 1.0,
        background_color: tuple[int, int, int, int] | None = (30, 30, 30, 200),
        max_width: float = 0.84,
        text_width: float = 0.0,
        text_height: float = 0.0,
        vertical_alignment: str = "top",
        horizontal_alignment: str = "auto",
        animation: str = "none",
        animation_delay_ms: float = 0.0,
        animation_in_ms: float = 350.0,
        animation_out_ms: float = 300.0,
        slide_offset_x: float = 0.0,
        slide_offset_y: float = 0.0,
        space: SceneSpace | str = SceneSpace.CANVAS,
    ) -> "TextAnnotation":
        """Factory that auto-generates a UUID."""
        return TextAnnotation(
            id=str(uuid.uuid4()),
            start_ms=start_ms,
            end_ms=end_ms,
            x=max(0.0, min(1.0, float(x))),
            y=max(0.0, min(1.0, float(y))),
            text=text,
            font_family=font_family or "Segoe UI",
            font_size=font_size,
            color=color,
            opacity=max(0.0, min(1.0, float(opacity))),
            background_color=background_color,
            max_width=max(0.05, min(1.0, float(max_width))),
            text_width=max(0.0, min(1.0, float(text_width))),
            text_height=max(0.0, min(1.0, float(text_height))),
            vertical_alignment=vertical_alignment,
            horizontal_alignment=horizontal_alignment,
            animation=animation,
            animation_delay_ms=animation_delay_ms,
            animation_in_ms=animation_in_ms,
            animation_out_ms=animation_out_ms,
            slide_offset_x=slide_offset_x,
            slide_offset_y=slide_offset_y,
            space=_scene_space(space, SceneSpace.CANVAS),
        )
    
    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON storage."""
        d = {
            "id": self.id,
            "startMs": self.start_ms,
            "endMs": self.end_ms,
            "x": self.x,
            "y": self.y,
            "text": self.text,
            "fontFamily": self.font_family,
            "fontSize": self.font_size,
            "color": list(self.color),
            "opacity": max(0.0, min(1.0, float(self.opacity))),
            "maxWidth": max(0.05, min(1.0, float(self.max_width))),
            "textWidth": max(0.0, min(1.0, float(self.text_width))),
            "textHeight": max(0.0, min(1.0, float(self.text_height))),
            "verticalAlignment": self.vertical_alignment,
            "horizontalAlignment": self.horizontal_alignment,
            "animation": self.animation,
            "animationDelayMs": self.animation_delay_ms,
            "animationInMs": self.animation_in_ms,
            "animationOutMs": self.animation_out_ms,
            "slideOffsetX": self.slide_offset_x,
            "slideOffsetY": self.slide_offset_y,
            "space": SceneSpace.CANVAS.value,
        }
        if self.background_color is not None:
            d["backgroundColor"] = list(self.background_color)
        return d
    
    @staticmethod
    def from_dict(d: dict) -> "TextAnnotation":
        """Reconstruct from a dict produced by ``to_dict()``."""
        try:
            bg_color = None
            if "backgroundColor" in d:
                bg_color = tuple(d["backgroundColor"])
            return TextAnnotation(
                id=d["id"],
                start_ms=d["startMs"],
                end_ms=d["endMs"],
                x=max(0.0, min(1.0, float(d["x"]))),
                y=max(0.0, min(1.0, float(d["y"]))),
                text=d["text"],
                font_family=d.get("fontFamily", "Segoe UI") or "Segoe UI",
                font_size=d.get("fontSize", 18),
                color=tuple(d.get("color", [255, 255, 255, 255])),
                opacity=max(0.0, min(1.0, float(d.get("opacity", 1.0)))),
                background_color=bg_color,
                max_width=max(0.05, min(1.0, float(d.get("maxWidth", 0.84)))),
                text_width=max(0.0, min(1.0, float(d.get("textWidth", 0.0)))),
                text_height=max(0.0, min(1.0, float(d.get("textHeight", 0.0)))),
                vertical_alignment=str(d.get("verticalAlignment", "top") or "top"),
                horizontal_alignment=str(d.get("horizontalAlignment", "auto") or "auto"),
                animation=str(d.get("animation", "none")),
                animation_delay_ms=float(d.get("animationDelayMs", 0.0)),
                animation_in_ms=float(d.get("animationInMs", 350.0)),
                animation_out_ms=float(d.get("animationOutMs", 300.0)),
                slide_offset_x=float(d.get("slideOffsetX", 0.0)),
                slide_offset_y=float(d.get("slideOffsetY", 0.0)),
                space=SceneSpace.CANVAS,
            )
        except KeyError as exc:
            raise ValueError(f"TextAnnotation missing required field: {exc}") from exc


@dataclass
class ExplainerScene:
    """Atomic cinematic scene coordinating presentation layout and canvas text."""

    id: str
    start_ms: float
    end_ms: float
    destination: str
    layout_scene: CanvasLayoutScene
    text_annotation: TextAnnotation
    video_transition_ms: float = 1100.0
    text_animation: str = "fade-slide"
    text_animation_duration_ms: float = 700.0
    text_enter_offset_ms: float = 350.0
    text_exit_offset_ms: float = 450.0
    restore_previous: bool = True
    text_gutter: float = 0.03
    clip_id: str = ""

    def __post_init__(self) -> None:
        self.id = str(self.id or uuid.uuid4())
        self.start_ms = max(0.0, float(self.start_ms))
        self.end_ms = max(self.start_ms + 1.0, float(self.end_ms))
        self.destination = str(self.destination or "right").strip().lower()
        self.video_transition_ms = max(0.0, min(5000.0, float(self.video_transition_ms)))
        self.text_animation_duration_ms = max(
            120.0, min(1500.0, float(self.text_animation_duration_ms))
        )
        self.text_enter_offset_ms = max(0.0, float(self.text_enter_offset_ms))
        self.text_exit_offset_ms = max(0.0, float(self.text_exit_offset_ms))
        animation = str(self.text_animation or "fade-slide").strip().lower()
        self.text_animation = animation if animation in {
            "fade", "fade-slide", "soft-reveal"
        } else "fade-slide"
        self.restore_previous = bool(self.restore_previous)
        self.text_gutter = max(0.01, min(0.10, float(self.text_gutter)))
        self.clip_id = str(self.clip_id or "")
        self.layout_scene.start_ms = self.start_ms
        self.layout_scene.end_ms = self.end_ms
        self.layout_scene.transition = "ease"
        self.layout_scene.transition_duration_ms = self.video_transition_ms
        self.text_annotation.start_ms = self.start_ms
        self.text_annotation.end_ms = self.end_ms

    @staticmethod
    def create(
        start_ms: float,
        end_ms: float,
        destination: str,
        layout_scene: CanvasLayoutScene,
        text_annotation: TextAnnotation,
        *,
        video_transition_ms: float = 1100.0,
        text_animation: str = "fade-slide",
        text_animation_duration_ms: float = 700.0,
        text_enter_offset_ms: float = 350.0,
        text_exit_offset_ms: float = 450.0,
        restore_previous: bool = True,
        text_gutter: float = 0.03,
        clip_id: str = "",
    ) -> "ExplainerScene":
        return ExplainerScene(
            id=str(uuid.uuid4()),
            start_ms=start_ms,
            end_ms=end_ms,
            destination=destination,
            layout_scene=layout_scene,
            text_annotation=replace(text_annotation, vertical_alignment="center"),
            video_transition_ms=video_transition_ms,
            text_animation=text_animation,
            text_animation_duration_ms=text_animation_duration_ms,
            text_enter_offset_ms=text_enter_offset_ms,
            text_exit_offset_ms=text_exit_offset_ms,
            restore_previous=restore_previous,
            text_gutter=text_gutter,
            clip_id=str(clip_id or ""),
        )

    def contains(self, time_ms: float) -> bool:
        return self.start_ms <= float(time_ms) <= self.end_ms

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "startMs": self.start_ms,
            "endMs": self.end_ms,
            "destination": self.destination,
            "layoutScene": self.layout_scene.to_dict(),
            "textAnnotation": self.text_annotation.to_dict(),
            "videoTransitionMs": self.video_transition_ms,
            "textAnimation": self.text_animation,
            "textAnimationDurationMs": self.text_animation_duration_ms,
            "textEnterOffsetMs": self.text_enter_offset_ms,
            "textExitOffsetMs": self.text_exit_offset_ms,
            "restorePrevious": self.restore_previous,
            "textGutter": self.text_gutter,
        }
        if self.clip_id:
            data["clipId"] = self.clip_id
        return data

    @staticmethod
    def from_dict(data: dict) -> "ExplainerScene":
        try:
            return ExplainerScene(
                id=str(data.get("id") or uuid.uuid4()),
                start_ms=float(data["startMs"]),
                end_ms=float(data["endMs"]),
                destination=str(data.get("destination", "right") or "right"),
                layout_scene=CanvasLayoutScene.from_dict(data["layoutScene"]),
                text_annotation=TextAnnotation.from_dict(data["textAnnotation"]),
                video_transition_ms=float(data.get("videoTransitionMs", 1100.0)),
                text_animation=str(data.get("textAnimation", "fade-slide")),
                text_animation_duration_ms=float(data.get("textAnimationDurationMs", 700.0)),
                text_enter_offset_ms=float(data.get("textEnterOffsetMs", 350.0)),
                text_exit_offset_ms=float(data.get("textExitOffsetMs", 450.0)),
                restore_previous=_as_bool(data.get("restorePrevious", True), True),
                text_gutter=float(data.get("textGutter", 0.03)),
                clip_id=str(data.get("clipId", "") or ""),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"ExplainerScene is invalid: {exc}") from exc


@dataclass
class ArrowAnnotation:
    """An arrow annotation pointing from one location to another.
    
    Arrow annotations draw directional arrows to highlight movement or
    connections between UI elements.
    """
    id: str
    start_ms: float
    end_ms: float
    x1: float  # 0-1 normalized start position
    y1: float
    x2: float  # 0-1 normalized end position
    y2: float
    color: tuple[int, int, int, int] = (255, 204, 0, 255)  # RGBA (yellow)
    thickness: int = 3
    head_size: int = 12
    space: SceneSpace = SceneSpace.VIDEO
    
    @staticmethod
    def create(
        start_ms: float,
        end_ms: float,
        x1: float = 0.3,
        y1: float = 0.3,
        x2: float = 0.5,
        y2: float = 0.5,
        color: tuple[int, int, int, int] = (255, 204, 0, 255),
        thickness: int = 3,
        head_size: int = 12,
        space: SceneSpace | str = SceneSpace.VIDEO,
    ) -> "ArrowAnnotation":
        """Factory that auto-generates a UUID."""
        return ArrowAnnotation(
            id=str(uuid.uuid4()),
            start_ms=start_ms,
            end_ms=end_ms,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            color=color,
            thickness=thickness,
            head_size=head_size,
            space=_scene_space(space),
        )
    
    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON storage."""
        return {
            "id": self.id,
            "startMs": self.start_ms,
            "endMs": self.end_ms,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "color": list(self.color),
            "thickness": self.thickness,
            "headSize": self.head_size,
            "space": _scene_space(self.space).value,
        }
    
    @staticmethod
    def from_dict(d: dict) -> "ArrowAnnotation":
        """Reconstruct from a dict produced by ``to_dict()``."""
        try:
            return ArrowAnnotation(
                id=d["id"],
                start_ms=d["startMs"],
                end_ms=d["endMs"],
                x1=d["x1"],
                y1=d["y1"],
                x2=d["x2"],
                y2=d["y2"],
                color=tuple(d.get("color", [255, 204, 0, 255])),
                thickness=d.get("thickness", 3),
                head_size=d.get("headSize", 12),
                space=_scene_space(d.get("space")),
            )
        except KeyError as exc:
            raise ValueError(f"ArrowAnnotation missing required field: {exc}") from exc


@dataclass
class HighlightBox:
    """A spotlight highlight annotation to emphasize a region.
    
    Highlights dim the surrounding screen and leave a rectangular or
    circular region visually emphasized.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    # Defaults keep the legacy positional constructor order intact while
    # allowing new callers and older JSON payloads to omit an identifier.
    start_ms: float = 0.0
    end_ms: float = 0.0
    x: float = 0.3  # 0-1 normalized position
    y: float = 0.3
    width: float = 0.2  # 0-1 normalized size
    height: float = 0.15
    color: tuple[int, int, int, int] = (255, 204, 0, 100)  # RGBA (yellow, semi-transparent)
    opacity: float = 0.4  # 0.0 - 1.0
    border_width: int = 0
    shape: str = "rect"  # "rect" or "circle"
    dim_opacity: float = 0.58
    corner_radius: float = 0.14  # Fraction of the shortest highlight edge.
    space: SceneSpace = SceneSpace.VIDEO
    
    @staticmethod
    def create(
        start_ms: float,
        end_ms: float,
        x: float = 0.3,
        y: float = 0.3,
        width: float = 0.2,
        height: float = 0.15,
        color: tuple[int, int, int, int] = (255, 204, 0, 100),
        opacity: float = 0.4,
        border_width: int = 0,
        shape: str = "rect",
        dim_opacity: float = 0.58,
        corner_radius: float = 0.14,
        space: SceneSpace | str = SceneSpace.VIDEO,
    ) -> "HighlightBox":
        """Factory that auto-generates a UUID."""
        return HighlightBox(
            id=str(uuid.uuid4()),
            start_ms=start_ms,
            end_ms=end_ms,
            x=x,
            y=y,
            width=width,
            height=height,
            color=color,
            opacity=opacity,
            border_width=border_width,
            shape=shape if shape in ("rect", "circle") else "rect",
            dim_opacity=max(0.0, min(0.9, float(dim_opacity))),
            corner_radius=max(0.0, min(0.5, float(corner_radius))),
            space=_scene_space(space),
        )
    
    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON storage."""
        return {
            "id": self.id,
            "startMs": self.start_ms,
            "endMs": self.end_ms,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "color": list(self.color),
            "opacity": self.opacity,
            "borderWidth": self.border_width,
            "shape": self.shape,
            "dimOpacity": self.dim_opacity,
            "cornerRadius": self.corner_radius,
            "space": _scene_space(self.space).value,
        }
    
    @staticmethod
    def from_dict(d: dict) -> "HighlightBox":
        """Reconstruct from a dict produced by ``to_dict()``."""
        try:
            return HighlightBox(
                start_ms=d["startMs"],
                end_ms=d["endMs"],
                x=d["x"],
                y=d["y"],
                width=d["width"],
                height=d["height"],
                id=str(d.get("id") or uuid.uuid4()),
                color=tuple(d.get("color", [255, 204, 0, 100])),
                opacity=max(0.0, min(1.0, float(d.get("opacity", 0.4)))),
                border_width=max(0, int(d.get("borderWidth", 0))),
                shape=d.get("shape", "rect") if d.get("shape", "rect") in ("rect", "circle") else "rect",
                dim_opacity=max(0.0, min(0.9, float(d.get("dimOpacity", 0.58)))),
                corner_radius=max(0.0, min(0.5, float(d.get("cornerRadius", 0.14)))),
                space=_scene_space(d.get("space")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"HighlightBox missing required field: {exc}") from exc


@dataclass
class AnnotationCollection:
    """Container for all annotation types in a recording session.
    
    Groups text, arrow, and highlight annotations together for easy
    serialization and rendering.
    """
    texts: List[TextAnnotation] | None = None
    arrows: List[ArrowAnnotation] | None = None
    highlights: List[HighlightBox] | None = None
    
    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON storage."""
        d = {}
        if self.texts:
            d["texts"] = [t.to_dict() for t in self.texts]
        if self.arrows:
            d["arrows"] = [a.to_dict() for a in self.arrows]
        if self.highlights:
            d["highlights"] = [h.to_dict() for h in self.highlights]
        return d
    
    @staticmethod
    def from_dict(d: dict) -> "AnnotationCollection":
        """Reconstruct from a dict produced by ``to_dict()``."""
        texts = None
        if "texts" in d:
            texts = [TextAnnotation.from_dict(t) for t in d["texts"]]
        arrows = None
        if "arrows" in d:
            arrows = [ArrowAnnotation.from_dict(a) for a in d["arrows"]]
        highlights = None
        if "highlights" in d:
            highlights = [HighlightBox.from_dict(h) for h in d["highlights"]]
        return AnnotationCollection(texts=texts, arrows=arrows, highlights=highlights)


DEFAULT_FPS = 60
DEFAULT_MOUSE_INTERVAL = 16
