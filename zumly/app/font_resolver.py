"""Deterministic font-family to font-file resolution for export rendering."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

_SUPPORTED_FONT_EXTENSIONS = {".ttf", ".otf", ".ttc"}
_FAMILY_VARIANT_SUFFIXES = ("display", "text", "small", "caption")


def _font_key(value: str) -> str:
    return "".join(ch for ch in str(value or "").casefold() if ch.isalnum())


def _font_name_matches(requested_key: str, candidate_key: str) -> bool:
    """Match a family name without accepting unrelated substring matches."""
    if not requested_key or not candidate_key:
        return False
    if candidate_key == requested_key or candidate_key.startswith(requested_key):
        return True
    # Qt exposes some variable-font optical families with a suffix that the
    # Windows registry omits (for example, "Segoe UI Variable Display" maps
    # to the registered "Segoe UI Variable" file).
    return any(
        requested_key == f"{candidate_key}{suffix}"
        for suffix in _FAMILY_VARIANT_SUFFIXES
    )


class FontResolver:
    """Resolve Qt/Pillow font family names to deterministic font files."""

    def __init__(
        self,
        *,
        bundle_dir: str | os.PathLike[str] | None = None,
        font_dirs: Iterable[str | os.PathLike[str]] | None = None,
    ) -> None:
        self.bundle_dir = Path(bundle_dir) if bundle_dir else Path(__file__).with_name("fonts")
        default_dirs = [self.bundle_dir, *self._system_font_dirs()]
        if font_dirs:
            default_dirs.extend(Path(item) for item in font_dirs)
        self.font_dirs = tuple(dict.fromkeys(default_dirs))

    @staticmethod
    def _system_font_dirs() -> tuple[Path, ...]:
        """Return conventional font directories for the current platform."""
        home = Path.home()
        if sys.platform == "win32":
            return (Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts",)
        if sys.platform == "darwin":
            return (
                Path("/System/Library/Fonts"),
                Path("/Library/Fonts"),
                home / "Library" / "Fonts",
            )
        return (
            Path("/usr/share/fonts"),
            Path("/usr/local/share/fonts"),
            home / ".fonts",
            home / ".local" / "share" / "fonts",
        )

    @property
    def fallback_path(self) -> str:
        """Return the bundled proportional fallback font."""
        return str(self.bundle_dir / "CascadiaCode.ttf")

    def resolve(self, family: str | None) -> str:
        requested = str(family or "").strip()
        key = _font_key(requested)
        if key:
            bundled = self._find_in_dirs(key)
            if bundled:
                return bundled
            registry_path = self._find_windows_registry(key)
            if registry_path:
                return registry_path
            system_path = self._find_windows_fonts(key)
            if system_path:
                return system_path

        fallback = self.fallback_path
        logger.warning(
            "Could not resolve font family %r; using bundled fallback %s",
            requested,
            fallback,
        )
        return fallback

    def resolve_qfont(self, font) -> str:
        """Resolve a Qt QFont-like object without importing Qt in this module."""
        family = font.family() if hasattr(font, "family") else str(font or "")
        return self.resolve(family)

    def _find_in_dirs(self, key: str) -> str | None:
        for directory in self.font_dirs:
            if not directory.is_dir():
                continue
            try:
                candidates = directory.rglob("*")
                for path in candidates:
                    if path.suffix.casefold() not in _SUPPORTED_FONT_EXTENSIONS:
                        continue
                    file_key = _font_key(path.stem)
                    if _font_name_matches(key, file_key):
                        return str(path)
            except OSError:
                logger.debug("Could not scan font directory %s", directory, exc_info=True)
        return None

    def _find_windows_fonts(self, key: str) -> str | None:
        if sys.platform != "win32":
            return None
        windows_fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        if not windows_fonts.is_dir():
            return None
        for path in windows_fonts.iterdir():
            if path.suffix.casefold() not in _SUPPORTED_FONT_EXTENSIONS:
                continue
            if _font_key(path.stem) == key:
                return str(path)
        return None

    def _find_windows_registry(self, key: str) -> str | None:
        if sys.platform != "win32":
            return None
        try:
            import winreg
        except ImportError:
            return None

        locations = (
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts"),
            (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts"),
        )
        fonts_root = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        for hive, subkey in locations:
            try:
                with winreg.OpenKey(hive, subkey) as handle:
                    index = 0
                    while True:
                        try:
                            name, value, _ = winreg.EnumValue(handle, index)
                        except OSError:
                            break
                        index += 1
                        name_key = _font_key(name.split("(", 1)[0])
                        if not _font_name_matches(key, name_key):
                            continue
                        candidate = Path(str(value))
                        if not candidate.is_absolute():
                            candidate = fonts_root / candidate
                        if (
                            candidate.suffix.casefold() in _SUPPORTED_FONT_EXTENSIONS
                            and candidate.is_file()
                        ):
                            return str(candidate)
            except OSError:
                continue
        return None
