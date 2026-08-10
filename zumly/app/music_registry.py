"""Bundled background-music catalog and custom-audio validation."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .utils import ffmpeg_exe, subprocess_kwargs

logger = logging.getLogger(__name__)

MAX_CUSTOM_MUSIC_BYTES = 50 * 1024 * 1024
SUPPORTED_MUSIC_EXTENSIONS = frozenset({".wav", ".mp3", ".m4a"})


class MusicValidationCancelled(RuntimeError):
    """Raised when a custom-music validation job is explicitly canceled."""


@dataclass(frozen=True, slots=True)
class MusicPreset:
    asset_id: str
    title: str
    mood: str
    asset_name: str = ""
    license_id: str = "CC0-1.0"
    creator: str = "Zumly"

    @property
    def path(self) -> str:
        if not self.asset_name:
            return ""
        return str(Path(__file__).with_name("music") / self.asset_name)

    @property
    def available(self) -> bool:
        return bool(self.path and os.path.isfile(self.path))


BUNDLED_MUSIC_TRACKS = (
    MusicPreset("builtin:ambient_focus", "Ambient Focus", "Quiet and spacious"),
    MusicPreset("builtin:tech_corporate", "Tech Corporate", "Professional and restrained"),
    MusicPreset("builtin:acoustic_flow", "Acoustic Flow", "Warm and conversational"),
)
BUNDLED_MUSIC_BY_ID = {preset.asset_id: preset for preset in BUNDLED_MUSIC_TRACKS}


def bundled_music_preset(asset_id: str) -> MusicPreset | None:
    return BUNDLED_MUSIC_BY_ID.get(str(asset_id or ""))


def resolve_music_asset(asset_id: str, asset_path: str = "") -> str:
    """Resolve a built-in ID or validated custom runtime path."""
    preset = bundled_music_preset(asset_id)
    if preset is not None:
        return preset.path if preset.available else ""
    path = str(asset_path or "")
    return path if path and os.path.isfile(path) else ""


def custom_music_asset_id(
    path: str,
    *,
    cancel_event: threading.Event | None = None,
) -> str:
    """Return a portable content-derived identifier for one custom asset."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            if cancel_event is not None and cancel_event.is_set():
                raise MusicValidationCancelled("Music validation canceled.")
            digest.update(chunk)
    return f"custom:{digest.hexdigest()[:20]}"


def probe_audio_duration_ms(
    path: str,
    *,
    cancel_event: threading.Event | None = None,
    process_observer: Callable[[subprocess.Popen | None], None] | None = None,
) -> float:
    """Validate audio with Zumly's bundled FFmpeg and return its duration."""
    command = [
        ffmpeg_exe(),
        "-hide_banner",
        "-nostdin",
        "-i",
        path,
        "-map",
        "0:a:0",
        "-vn",
        "-t",
        "0.05",
        "-f",
        "null",
        "-",
    ]
    if cancel_event is None and process_observer is None:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=12,
                check=False,
                **subprocess_kwargs(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError(f"Could not inspect audio file: {exc}") from exc
        if result.returncode != 0:
            raise ValueError("The selected file does not contain readable audio.")
        match = re.search(
            r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)",
            result.stderr or "",
        )
        if match:
            duration = (
                int(match.group(1)) * 3600
                + int(match.group(2)) * 60
                + float(match.group(3))
            )
            if duration > 0.0:
                return duration * 1000.0
        return 1.0

    process: subprocess.Popen | None = None
    started = time.monotonic()
    stdout = ""
    stderr = ""
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            **subprocess_kwargs(),
        )
        if process_observer is not None:
            process_observer(process)
        while True:
            if cancel_event is not None and cancel_event.is_set():
                if process.poll() is None:
                    process.terminate()
                raise MusicValidationCancelled("Music validation canceled.")
            if time.monotonic() - started > 12.0:
                if process.poll() is None:
                    process.kill()
                raise ValueError("Could not inspect audio file: validation timed out.")
            try:
                stdout, stderr = process.communicate(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                continue
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"Could not inspect audio file: {exc}") from exc
    finally:
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=0.5)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    process.kill()
                except OSError:
                    pass
        if process_observer is not None:
            process_observer(None)
    if process is None or process.returncode != 0:
        raise ValueError("The selected file does not contain readable audio.")
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr or "")
    if match:
        duration = (
            int(match.group(1)) * 3600
            + int(match.group(2)) * 60
            + float(match.group(3))
        )
        if duration > 0.0:
            return duration * 1000.0
    # Some valid containers do not publish a duration. Successful decoding is
    # sufficient because the preview player will report its runtime duration.
    return 1.0


def validate_music_asset(
    path: str,
    *,
    probe: bool = True,
    cancel_event: threading.Event | None = None,
    process_observer: Callable[[subprocess.Popen | None], None] | None = None,
) -> float:
    """Validate a custom upload and return its decoded duration in ms."""
    resolved = os.path.realpath(str(path or ""))
    if not resolved or not os.path.isfile(resolved):
        raise ValueError("Choose an existing audio file.")
    extension = Path(resolved).suffix.lower()
    if extension not in SUPPORTED_MUSIC_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_MUSIC_EXTENSIONS))
        raise ValueError(f"Unsupported audio type. Choose one of: {allowed}")
    size = os.path.getsize(resolved)
    if size <= 0:
        raise ValueError("The selected audio file is empty.")
    if size > MAX_CUSTOM_MUSIC_BYTES:
        raise ValueError("Custom music must be 50 MB or smaller.")
    if cancel_event is not None and cancel_event.is_set():
        raise MusicValidationCancelled("Music validation canceled.")
    return (
        probe_audio_duration_ms(
            resolved,
            cancel_event=cancel_event,
            process_observer=process_observer,
        )
        if probe
        else 0.0
    )
