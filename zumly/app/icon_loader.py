"""Shared loader for Zumly branding and Fluent UI system icons."""

import logging
import sys
from typing import Dict, Optional
from pathlib import Path

from PySide6.QtCore import QByteArray, QSize, QRectF
from PySide6.QtGui import QGuiApplication, QIcon, QPixmap, QPainter, QColor
from PySide6.QtSvg import QSvgRenderer

logger = logging.getLogger(__name__)

# Icon cache to avoid reloading the same icon multiple times
_ICON_CACHE: Dict[str, QIcon] = {}
_PIXMAP_CACHE: Dict[str, QPixmap] = {}

# Path to the icons directory
_ICONS_DIR = Path(__file__).parent / "icons"
_BRANDING_RESOURCE_ROOT = "zumly/app/branding"
_BRAND_ICON_RESOURCE = f"{_BRANDING_RESOURCE_ROOT}/generated/zumly.ico"
_BRAND_SYMBOL_RESOURCE = f"{_BRANDING_RESOURCE_ROOT}/zumly-logo-symbol.svg"


def get_resource_path(relative_path: str) -> Path:
    """Resolve a bundled resource in source and PyInstaller layouts.

    In a one-folder PyInstaller build, data files live below ``sys._MEIPASS``
    (the bundle's ``_internal`` directory), not beside the executable.
    Keeping this lookup here gives tray and UI resources one consistent path
    contract while retaining source-tree and adjacent-file fallbacks.
    """
    relative = Path(relative_path)
    candidates: list[Path] = []
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", "")
        if meipass:
            candidates.append(Path(meipass) / relative)
        candidates.append(Path(sys.executable).resolve().parent / relative)
    candidates.append(Path(__file__).resolve().parents[2] / relative)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0] if candidates else relative


def get_brand_asset_path(asset_name: str) -> Path:
    """Resolve a canonical or generated Zumly brand asset."""
    return get_resource_path(f"{_BRANDING_RESOURCE_ROOT}/{asset_name}")


def get_brand_icon() -> QIcon:
    """Return the packaged multi-resolution Zumly application icon."""
    cache_key = "zumly_application_icon"
    if cache_key in _ICON_CACHE:
        return _ICON_CACHE[cache_key]

    icon_path = get_resource_path(_BRAND_ICON_RESOURCE)
    icon = QIcon(str(icon_path))
    if icon.isNull():
        # Development checkouts can regenerate assets lazily; the approved SVG
        # remains a lossless fallback for Qt surfaces.
        symbol_path = get_resource_path(_BRAND_SYMBOL_RESOURCE)
        icon = QIcon(str(symbol_path))
    _ICON_CACHE[cache_key] = icon
    return icon


def get_brand_pixmap(asset_name: str, width: int, height: int) -> QPixmap:
    """Render a Zumly SVG asset at an exact UI target size."""
    screen = QGuiApplication.primaryScreen()
    device_ratio = max(1.0, float(screen.devicePixelRatio())) if screen else 1.0
    pixel_width = max(1, int(round(width * device_ratio)))
    pixel_height = max(1, int(round(height * device_ratio)))
    cache_key = (
        f"brand_pixmap_{asset_name}_{width}x{height}_dpr{device_ratio:.3f}"
    )
    cached = _PIXMAP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    asset_path = get_brand_asset_path(asset_name)
    renderer = QSvgRenderer(str(asset_path))
    pixmap = QPixmap(pixel_width, pixel_height)
    pixmap.setDevicePixelRatio(device_ratio)
    pixmap.fill(QColor(0, 0, 0, 0))
    if renderer.isValid():
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        renderer.render(painter, QRectF(0, 0, pixel_width, pixel_height))
        painter.end()
    _PIXMAP_CACHE[cache_key] = pixmap
    return pixmap


def load_icon(
    name: str,
    size: int = 20,
    variant: str = "regular",
    color: Optional[str] = None,
) -> QIcon:
    """Load a Fluent UI System Icon with optional color replacement.

    Args:
        name: Icon name (e.g., "record", "play", "save")
        size: Icon size in pixels (default: 20)
        variant: Icon variant - "regular" or "filled" (default: "regular")
        color: Optional color to apply (hex string or rgba). If None, uses default SVG color.

    Returns:
        QIcon object ready to use in Qt widgets

    Example:
        >>> from . import tokens as T
        >>> icon = load_icon("record", size=20, variant="filled", color=T.BRAND)
        >>> button.setIcon(icon)
    """
    svg_file = _ICONS_DIR / f"{name}_{size}_{variant}.svg"
    return _load_svg_file(svg_file, size=size, color=color)


def load_svg_icon(
    asset_name: str,
    size: int = 20,
    color: Optional[str] = None,
) -> QIcon:
    """Load an exact SVG asset such as ``background_selected.svg``."""
    filename = asset_name if asset_name.lower().endswith(".svg") else f"{asset_name}.svg"
    return _load_svg_file(_ICONS_DIR / filename, size=size, color=color)


def _load_svg_file(svg_file: Path, *, size: int, color: Optional[str]) -> QIcon:
    cache_key = f"{svg_file.name}_{size}_{color or 'default'}"
    if cache_key in _ICON_CACHE:
        return _ICON_CACHE[cache_key]

    if not svg_file.exists():
        logger.warning(f"Icon file not found: {svg_file}")
        return QIcon()  # Return empty icon as fallback

    try:
        # Read SVG content
        svg_content = svg_file.read_text(encoding="utf-8")

        # Apply color if specified
        if color:
            # Replace the fill color in the SVG
            # Fluent icons typically use fill="#212121" or similar
            svg_content = _apply_color(svg_content, color)

        # Render SVG to QPixmap
        svg_bytes = QByteArray(svg_content.encode("utf-8"))
        renderer = QSvgRenderer(svg_bytes)
        
        if not renderer.isValid():
            logger.warning(f"Invalid SVG content in: {svg_file}")
            return QIcon()

        # Create pixmap with the requested size
        pixmap = QPixmap(QSize(size, size))
        pixmap.fill(QColor(0, 0, 0, 0))  # Transparent background
        
        painter = QPainter(pixmap)
        renderer.render(painter, QRectF(0, 0, size, size))
        painter.end()

        # Create icon from pixmap
        icon = QIcon(pixmap)
        
        # Cache it
        _ICON_CACHE[cache_key] = icon
        
        return icon

    except Exception:
        logger.exception("Failed to load icon %s", svg_file.name)
        return QIcon()


def _apply_color(svg_content: str, color: str) -> str:
    """Replace fill color in SVG content with the specified color.

    Args:
        svg_content: Original SVG content as string
        color: Color to apply (hex string like "#8b5cf6" or rgba string like "rgba(139, 92, 246, 1)")

    Returns:
        Modified SVG content with color applied
    """
    # Convert rgba() to hex if needed
    if color.startswith("rgba("):
        color = _rgba_to_hex(color)

    # Purpose-built icons use currentColor so the same vector can inherit the
    # editor's inactive, selected, and page-header colors.
    svg_content = svg_content.replace("currentColor", color)

    # Replace fill attributes
    # Fluent icons use fill="#212121" or fill='#212121'
    import re
    
    # Replace fill="#..." with our color
    svg_content = re.sub(r'fill="#[0-9a-fA-F]{6}"', f'fill="{color}"', svg_content)
    svg_content = re.sub(r"fill='#[0-9a-fA-F]{6}'", f"fill='{color}'", svg_content)
    
    # Also handle style attributes like style="fill:#..."
    svg_content = re.sub(r'fill:#[0-9a-fA-F]{6}', f'fill:{color}', svg_content)

    return svg_content


def _rgba_to_hex(rgba_str: str) -> str:
    """Convert rgba(r, g, b, a) string to #RRGGBB hex.

    Args:
        rgba_str: String like "rgba(139, 92, 246, 1)"

    Returns:
        Hex color string like "#8b5cf6"
    """
    import re
    match = re.match(r"rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*[\d.]+)?\)", rgba_str)
    if match:
        r, g, b = map(int, match.groups())
        return f"#{r:02x}{g:02x}{b:02x}"
    return "#212121"  # Fallback to default dark gray


def clear_cache() -> None:
    """Clear the icon cache. Useful when theme changes."""
    global _ICON_CACHE
    _ICON_CACHE.clear()
    _PIXMAP_CACHE.clear()
    logger.debug("Icon cache cleared")

def get_zumly_icon(is_recording: bool = False, is_paused: bool = False) -> QIcon:
    """Return the approved Zumly application symbol.

    Recording state is communicated through menu text and tooltip copy. The
    application icon itself remains stable throughout the process lifecycle.
    """
    _ = is_recording, is_paused
    return get_brand_icon()
