"""Standalone capture-session metadata and durable media publication."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from .identity import FILE_PREFIX, SETTINGS_DIRECTORY_NAME


CAPTURE_SESSION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class CaptureSession:
    """In-memory metadata describing a directly playable recording."""

    session_id: str
    media_path: str
    capture_target: dict[str, Any]
    started_at_unix_ms: float
    duration_ms: float
    paused_duration_ms: float
    pause_boundaries: list[dict[str, Any]] = field(default_factory=list)
    requested_fps: float = 0.0
    actual_fps: float = 0.0
    is_cfr: bool = False
    capture_backend: str = ""
    frame_timestamps: list[float] = field(default_factory=list)
    mouse_track: list[dict[str, Any]] = field(default_factory=list)
    click_events: list[dict[str, Any]] = field(default_factory=list)
    capture_telemetry: dict[str, Any] = field(default_factory=dict)
    audio: dict[str, Any] = field(default_factory=dict)
    smart_zoom: dict[str, Any] = field(
        default_factory=lambda: {"state": "not_processed", "keyframes": []}
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": CAPTURE_SESSION_SCHEMA_VERSION,
            "sessionId": self.session_id,
            "mediaPath": os.path.abspath(self.media_path),
            "captureTarget": dict(self.capture_target),
            "startedAtUnixMs": round(float(self.started_at_unix_ms), 3),
            "durationMs": round(float(self.duration_ms), 3),
            "pausedDurationMs": round(float(self.paused_duration_ms), 3),
            "pauseBoundaries": list(self.pause_boundaries),
            "requestedFps": float(self.requested_fps),
            "actualFps": float(self.actual_fps),
            "isCfr": bool(self.is_cfr),
            "captureBackend": str(self.capture_backend),
            "frameTimestamps": [float(value) for value in self.frame_timestamps],
            "mouseTrack": list(self.mouse_track),
            "clickEvents": list(self.click_events),
            "captureTelemetry": dict(self.capture_telemetry),
            "audio": dict(self.audio),
            "smartZoom": dict(self.smart_zoom),
        }


@dataclass(frozen=True, slots=True)
class CapturePublishResult:
    media_path: str
    warning: str = ""


def capture_drafts_directory() -> Path:
    """Return the private per-user folder for reversible capture drafts."""
    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    root = Path(local_app_data) if local_app_data else Path(tempfile.gettempdir())
    return root / SETTINGS_DIRECTORY_NAME / "Drafts"


def _preserve_recording_draft(
    source_path: str | os.PathLike[str],
    session_id: str,
    purpose: str,
) -> str:
    """Copy a recording to a private, purpose-scoped draft."""
    source = Path(source_path).resolve()
    if not source.is_file() or source.stat().st_size <= 0:
        raise ValueError(f"Recording draft source is not usable: {source}")
    safe_session_id = "".join(
        character for character in str(session_id) if character.isalnum() or character in "-_"
    )
    if not safe_session_id:
        raise ValueError("A valid session id is required to preserve a recording draft")
    safe_purpose = "".join(
        character for character in str(purpose) if character.isalnum() or character in "-_"
    )
    if not safe_purpose:
        raise ValueError("A valid purpose is required to preserve a recording draft")

    drafts = capture_drafts_directory()
    drafts.mkdir(parents=True, exist_ok=True)
    destination = drafts / f"{safe_session_id}.{safe_purpose}.mp4"
    if destination.exists():
        raise FileExistsError(f"Recording draft already exists: {destination}")

    try:
        os.link(source, destination)
        return str(destination)
    except OSError:
        # Cross-volume and non-NTFS sources cannot be linked. Fall back to a
        # durable copy while keeping the published output independent.
        pass

    staged = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(drafts),
            prefix=f"{FILE_PREFIX}_{safe_purpose}_",
            suffix=".tmp",
            delete=False,
        ) as handle:
            staged = handle.name
            with source.open("rb") as source_handle:
                shutil.copyfileobj(source_handle, handle, length=1024 * 1024)
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(staged, destination)
        staged = ""
        return str(destination)
    finally:
        if staged:
            try:
                os.remove(staged)
            except OSError:
                pass


def preserve_unzoomed_recording(
    source_path: str | os.PathLike[str],
    session_id: str,
) -> str:
    """Copy the unzoomed recording to a private draft for post-capture removal."""
    return _preserve_recording_draft(source_path, session_id, "unzoomed")


def preserve_format_source_recording(
    source_path: str | os.PathLike[str],
    session_id: str,
) -> str:
    """Retain the processed MP4 used to create a GIF for preview format changes."""
    return _preserve_recording_draft(source_path, session_id, "format-source")


def discard_recording_draft(path: str | os.PathLike[str]) -> None:
    """Discard one private recording draft without touching other media."""
    if not path:
        return
    draft = Path(path)
    try:
        draft.unlink()
    except FileNotFoundError:
        return
    try:
        draft.parent.rmdir()
    except OSError:
        pass


def discard_unzoomed_recording(path: str | os.PathLike[str]) -> None:
    """Backward-compatible name for discarding a private recording draft."""
    discard_recording_draft(path)


def restore_unzoomed_recording(
    media_path: str | os.PathLike[str],
    draft_path: str | os.PathLike[str],
) -> str:
    """Atomically replace a zoomed recording with its preserved original."""
    media = Path(media_path).resolve()
    draft = Path(draft_path).resolve()
    if not media.is_file() or media.stat().st_size <= 0:
        raise ValueError(f"Published recording is not usable: {media}")
    if not draft.is_file() or draft.stat().st_size <= 0:
        raise ValueError(f"Unzoomed draft is not usable: {draft}")

    staged = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(media.parent),
            prefix=f"{FILE_PREFIX}_restore_",
            suffix=media.suffix,
            delete=False,
        ) as handle:
            staged = handle.name
            with draft.open("rb") as source_handle:
                shutil.copyfileobj(source_handle, handle, length=1024 * 1024)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staged, media)
        staged = ""

        discard_unzoomed_recording(draft)
        return ""
    finally:
        if staged:
            try:
                os.remove(staged)
            except OSError:
                pass

def publish_recording(
    source_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    session: CaptureSession,
) -> CapturePublishResult:
    """Publish a non-empty recording without overwriting existing media.

    The recording is copied to a staging file in the destination directory and
    renamed into its final name. On Windows this same-volume rename is atomic
    and fails if the destination already exists. The source temp file is removed
    only after the final media exists.
    """
    source = Path(source_path).resolve()
    output = Path(output_path).resolve()

    if source == output:
        raise ValueError("Capture source and output paths must be different")
    if Path(session.media_path).resolve() != output:
        raise ValueError("Capture session media path must match the publish destination")
    if not source.is_file() or source.stat().st_size <= 0:
        raise ValueError(f"Capture engine did not produce a usable recording: {source}")
    if output.exists():
        raise FileExistsError(f"Capture output already exists: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    media_stage = ""
    warning = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(output.parent),
            prefix=f"{FILE_PREFIX}_media_",
            suffix=".tmp",
            delete=False,
        ) as handle:
            media_stage = handle.name
            with source.open("rb") as source_handle:
                shutil.copyfileobj(source_handle, handle, length=1024 * 1024)
            handle.flush()
            os.fsync(handle.fileno())

        os.rename(media_stage, output)
        media_stage = ""

        try:
            source.unlink()
        except OSError as exc:
            cleanup_warning = f"Could not remove temporary capture file: {exc}"
            warning = f"{warning} {cleanup_warning}".strip()

        return CapturePublishResult(
            media_path=str(output),
            warning=warning,
        )
    finally:
        if media_stage:
            try:
                os.remove(media_stage)
            except OSError:
                pass
