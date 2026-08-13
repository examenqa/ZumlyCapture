"""Standalone settings for capture workflows."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any

from .identity import SETTINGS_DIRECTORY_NAME


logger = logging.getLogger(__name__)


def default_output_folder() -> str:
    return str(Path.home() / "Videos" / SETTINGS_DIRECTORY_NAME)


def default_screenshot_folder() -> str:
    return str(Path.home() / "Pictures" / SETTINGS_DIRECTORY_NAME)


DEFAULT_SETTINGS: dict[str, Any] = {
    "settings_schema_version": 3,
    "output_folder": default_output_folder(),
    "screenshot_folder": default_screenshot_folder(),
    "screenshot_format": "png",
    "recording_format": "mp4",
    "copy_screenshot": True,
    "screenshot_delay_seconds": 0,
    "fps": 60,
    "monitor": 1,
    "countdown_seconds": 1,
    "screenshot_monitor_hotkey": "Ctrl+Alt+1",
    "screenshot_window_hotkey": "Ctrl+Alt+2",
    "screenshot_region_hotkey": "Ctrl+Alt+3",
    "record_monitor_hotkey": "Ctrl+Alt+4",
    "record_window_hotkey": "Ctrl+Alt+5",
    "record_region_hotkey": "Ctrl+Alt+6",
    "pause_hotkey": "Ctrl+Alt+9",
    "stop_hotkey": "Ctrl+Alt+0",
    "preview_after_capture": True,
    "microphone_device": "",
    "system_audio_device": "",
    "smart_zoom_enabled": True,
    "smart_zoom_level": 1.5,
    "render_cursor": True,
    "render_clicks": True,
}


def settings_path() -> Path:
    appdata = os.environ.get("APPDATA", "").strip()
    root = Path(appdata) if appdata else Path.home() / ".config"
    return root / SETTINGS_DIRECTORY_NAME / "config.json"


def _bounded_int(value: object, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def normalize_settings(value: dict[str, Any] | None) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    settings = dict(DEFAULT_SETTINGS)
    settings.update(source)
    try:
        source_version = int(source.get("settings_schema_version", 1))
    except (TypeError, ValueError):
        source_version = 1
    if source_version < 2:
        # Phase 5 replaces the original three Ctrl+Shift shortcuts with a
        # complete, memorable Ctrl+Alt+number command map. It also turns on
        # the post-capture experience users expect from a capture utility.
        for key in (
            "screenshot_monitor_hotkey",
            "screenshot_window_hotkey",
            "screenshot_region_hotkey",
            "record_monitor_hotkey",
            "record_window_hotkey",
            "record_region_hotkey",
            "pause_hotkey",
            "stop_hotkey",
        ):
            settings[key] = DEFAULT_SETTINGS[key]
        settings["preview_after_capture"] = True
        settings["smart_zoom_enabled"] = True
        settings["render_cursor"] = True
    settings["settings_schema_version"] = 3
    settings["output_folder"] = str(settings.get("output_folder") or default_output_folder())
    settings["screenshot_folder"] = str(
        settings.get("screenshot_folder") or default_screenshot_folder()
    )
    fmt = str(settings.get("screenshot_format", "png")).lower()
    settings["screenshot_format"] = fmt if fmt in {"png", "jpg"} else "png"
    recording_format = str(settings.get("recording_format", "mp4")).lower()
    settings["recording_format"] = (
        recording_format if recording_format in {"mp4", "gif"} else "mp4"
    )
    settings["copy_screenshot"] = bool(settings.get("copy_screenshot", True))
    settings["screenshot_delay_seconds"] = _bounded_int(
        settings.get("screenshot_delay_seconds"), 0, 0, 10
    )
    settings["fps"] = _bounded_int(settings.get("fps"), 60, 15, 120)
    settings["monitor"] = _bounded_int(settings.get("monitor"), 1, 1, 32)
    settings["countdown_seconds"] = _bounded_int(
        settings.get("countdown_seconds"), 1, 0, 10
    )
    for key in (
        "screenshot_monitor_hotkey",
        "screenshot_window_hotkey",
        "screenshot_region_hotkey",
        "record_monitor_hotkey",
        "record_window_hotkey",
        "record_region_hotkey",
        "pause_hotkey",
        "stop_hotkey",
    ):
        fallback = str(DEFAULT_SETTINGS[key])
        settings[key] = str(settings.get(key) or fallback)
    settings["preview_after_capture"] = bool(
        settings.get("preview_after_capture", True)
    )
    settings["microphone_device"] = str(settings.get("microphone_device") or "")
    settings["system_audio_device"] = str(settings.get("system_audio_device") or "")
    settings["smart_zoom_enabled"] = bool(settings.get("smart_zoom_enabled", False))
    try:
        zoom = float(settings.get("smart_zoom_level", 1.5))
    except (TypeError, ValueError):
        zoom = 1.5
    settings["smart_zoom_level"] = max(1.1, min(3.0, zoom))
    settings["render_cursor"] = bool(settings.get("render_cursor", False))
    settings["render_clicks"] = bool(settings.get("render_clicks", True))
    return settings


def load_settings(path: Path | None = None) -> dict[str, Any]:
    target = path or settings_path()
    try:
        with target.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return normalize_settings(payload if isinstance(payload, dict) else {})
    except FileNotFoundError:
        return normalize_settings({})
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Capture settings load failed: %s", exc)
        return normalize_settings({})


def save_settings(value: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    target = path or settings_path()
    normalized = normalize_settings(value)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(target.parent),
            prefix="settings_",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = handle.name
            json.dump(normalized, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
        return normalized
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                pass
            except OSError:
                logger.debug("Could not remove staged settings file: %s", temp_path)
