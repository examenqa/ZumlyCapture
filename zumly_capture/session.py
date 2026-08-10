"""Standalone capture-session metadata and durable media publication."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from .identity import FILE_PREFIX


CAPTURE_SESSION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class CaptureSession:
    """Metadata retained beside a directly playable recording."""

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
    manifest_path: str
    warning: str = ""


def manifest_path_for(media_path: str | os.PathLike[str]) -> str:
    """Return the stable sidecar name for a published recording."""
    media = Path(media_path).resolve()
    return str(media.with_suffix(".zumly-capture.json"))


def _stage_json(directory: Path, payload: dict[str, Any]) -> str:
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(directory),
            prefix=f"{FILE_PREFIX}_manifest_",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = handle.name
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return temp_path
    except Exception:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise


def publish_recording(
    source_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    session: CaptureSession,
) -> CapturePublishResult:
    """Publish a non-empty MP4 and its session sidecar without overwriting media.

    The recording is copied to a staging file in the destination directory and
    hard-linked into its final name. That final link is atomic and fails if the
    destination already exists. The source temp file is removed only after the
    final media exists.
    """
    source = Path(source_path).resolve()
    output = Path(output_path).resolve()
    manifest = Path(manifest_path_for(output))

    if source == output:
        raise ValueError("Capture source and output paths must be different")
    if Path(session.media_path).resolve() != output:
        raise ValueError("Capture session media path must match the publish destination")
    if not source.is_file() or source.stat().st_size <= 0:
        raise ValueError(f"Capture engine did not produce a usable video: {source}")
    if output.exists():
        raise FileExistsError(f"Capture output already exists: {output}")
    if manifest.exists():
        raise FileExistsError(f"Capture manifest already exists: {manifest}")

    output.parent.mkdir(parents=True, exist_ok=True)
    media_stage = ""
    manifest_stage = ""
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

        manifest_stage = _stage_json(output.parent, session.to_dict())
        os.link(media_stage, output)
        os.remove(media_stage)
        media_stage = ""

        try:
            os.link(manifest_stage, manifest)
            os.remove(manifest_stage)
            manifest_stage = ""
        except OSError as exc:
            warning = f"Recording saved, but its capture manifest could not be published: {exc}"

        try:
            source.unlink()
        except OSError as exc:
            cleanup_warning = f"Could not remove temporary capture file: {exc}"
            warning = f"{warning} {cleanup_warning}".strip()

        return CapturePublishResult(
            media_path=str(output),
            manifest_path=str(manifest) if manifest.is_file() else "",
            warning=warning,
        )
    finally:
        for staged_path in (media_stage, manifest_stage):
            if staged_path:
                try:
                    os.remove(staged_path)
                except OSError:
                    pass
