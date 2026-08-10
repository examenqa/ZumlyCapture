"""Session-scoped visual frame cache used by AI analysis features.

This module owns filesystem lifecycle and traversal protection. It deliberately
does not know about Qt, providers, prompts, or recording models.
"""

import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

_AI_FRAME_CACHE_DIR_NAME = "zumly-ai-frame-cache"
_AI_FRAME_CACHE_VERSION = 1
_AI_FRAME_CACHE_TTL_SECONDS = 7 * 24 * 60 * 60
_AI_FRAME_CACHE_MAX_BYTES = 512 * 1024 * 1024


class AIFrameCacheSecurityError(ValueError):
    """Raised when a cache operation would leave the managed cache root."""


def safe_frame_cache_scope(session_id: str, cache_key: str) -> str:
    """Return an opaque, filesystem-safe scope for one recording session."""
    import hashlib

    raw_scope = str(session_id or "").strip()
    if raw_scope and (
        "/" in raw_scope
        or "\\" in raw_scope
        or "\x00" in raw_scope
        or raw_scope in {".", ".."}
    ):
        raise AIFrameCacheSecurityError("Unsafe recording session identifier")
    scope_source = raw_scope or f"legacy-{cache_key[:12]}"
    return hashlib.sha256(scope_source.encode("utf-8")).hexdigest()


def cache_root_path() -> Path:
    """Return the resolved cache root, rejecting a redirected temp root."""
    temp_root = Path(tempfile.gettempdir()).resolve()
    cache_root = (temp_root / _AI_FRAME_CACHE_DIR_NAME).resolve()
    temp_text = os.path.normcase(str(temp_root))
    cache_text = os.path.normcase(str(cache_root))
    try:
        common = os.path.commonpath([temp_text, cache_text])
    except ValueError as exc:
        raise AIFrameCacheSecurityError("AI cache root is on a different filesystem") from exc
    if cache_text == temp_text or common != temp_text:
        raise AIFrameCacheSecurityError("AI cache root resolves outside the temp directory")
    return cache_root


def safe_cache_child(root: Path, child: str | Path) -> Path:
    """Resolve a cache child and enforce containment under *root*."""
    root_resolved = root.resolve()
    target = (root_resolved / child).resolve()
    root_text = os.path.normcase(str(root_resolved))
    target_text = os.path.normcase(str(target))
    try:
        common = os.path.commonpath([root_text, target_text])
    except ValueError as exc:
        raise AIFrameCacheSecurityError("AI cache path is on a different filesystem") from exc
    if target_text == root_text or common != root_text:
        raise AIFrameCacheSecurityError("AI cache path escapes its managed root")
    return target


def frame_cache_manifest_is_valid(directory: Path) -> bool:
    """Validate the small manifest used to describe cached frame timestamps."""
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return True
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("frames"), list):
            raise ValueError("manifest does not contain a frame list")
        for record in payload["frames"]:
            if not isinstance(record, dict):
                raise ValueError("manifest contains a non-object frame record")
            filename = str(record.get("file", ""))
            if filename and Path(filename).name != filename:
                raise ValueError("manifest contains an unsafe frame path")
        return True
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("Recovering corrupted AI frame manifest %s: %s", manifest_path, exc)
        try:
            manifest_path.unlink(missing_ok=True)
        except OSError:
            logger.debug("Could not remove corrupted AI frame manifest", exc_info=True)
        return False


def frame_cache_directory(cache_key: str, session_id: str = "") -> Optional[Path]:
    """Create a session-scoped temporary JPEG cache directory."""
    try:
        root = cache_root_path()
        scope_dir = safe_cache_child(root, safe_frame_cache_scope(session_id, cache_key))
        directory = safe_cache_child(scope_dir, cache_key)
        directory.mkdir(parents=True, exist_ok=True)
        frame_cache_manifest_is_valid(directory)
        return directory
    except OSError:
        logger.warning("Could not create AI frame cache directory", exc_info=True)
        return None


def frame_cache_path(directory: Path, timestamp_ms: float) -> Path:
    """Return a stable filename for a tenth-of-a-millisecond-rounded sample."""
    sample_key = int(round(max(0.0, float(timestamp_ms)) * 10.0))
    return directory / f"frame-{sample_key:016d}.jpg"


def directory_size(directory: Path) -> int:
    """Return a best-effort byte count for a cache directory."""
    total = 0
    try:
        for path in directory.rglob("*"):
            if path.is_file():
                try:
                    total += path.stat().st_size
                except OSError:
                    continue
    except OSError:
        return total
    return total


def cleanup_ai_frame_cache(session_id: str | None = None) -> None:
    """Remove a closed session cache and expire old cache scopes safely."""
    root = cache_root_path()
    active_scope = None
    if session_id:
        active_scope = safe_frame_cache_scope(session_id, "legacy-cache")
    if not root.is_dir():
        return

    if session_id:
        scope_dir = safe_cache_child(root, active_scope)
        try:
            if scope_dir.is_dir():
                shutil.rmtree(scope_dir)
                logger.debug("Removed AI frame cache for session %s", session_id)
        except OSError:
            logger.warning("Could not remove AI frame cache for session %s", session_id, exc_info=True)

    now = time.time()
    cache_scopes: list[tuple[float, Path, int]] = []
    try:
        scope_dirs = [path for path in root.iterdir() if path.is_dir()]
    except OSError:
        return

    for candidate in scope_dirs:
        try:
            scope_dir = safe_cache_child(root, candidate.name)
        except AIFrameCacheSecurityError:
            logger.warning("Skipping AI cache path outside the managed root: %s", candidate)
            continue
        if active_scope and scope_dir.name == active_scope:
            continue
        try:
            modified = scope_dir.stat().st_mtime
        except OSError:
            continue
        if now - modified > _AI_FRAME_CACHE_TTL_SECONDS:
            try:
                shutil.rmtree(scope_dir)
            except OSError:
                logger.debug("Could not expire AI frame cache %s", scope_dir, exc_info=True)
            continue
        cache_scopes.append((modified, scope_dir, directory_size(scope_dir)))

    total_bytes = sum(size for _, _, size in cache_scopes)
    if total_bytes <= _AI_FRAME_CACHE_MAX_BYTES:
        return
    for _, scope_dir, size in sorted(cache_scopes):
        if total_bytes <= _AI_FRAME_CACHE_MAX_BYTES:
            break
        try:
            shutil.rmtree(scope_dir)
            total_bytes -= size
        except OSError:
            logger.debug("Could not trim oversized AI frame cache %s", scope_dir, exc_info=True)


def read_cached_frame(path: Path) -> bytes:
    """Read a cached JPEG, returning empty bytes for stale or locked files."""
    try:
        data = path.read_bytes()
        return data if data else b""
    except OSError:
        return b""


def write_frame_cache_manifest(directory: Optional[Path], records: List[dict[str, Any]]) -> None:
    """Persist timestamp-to-image mapping beside the cached JPEGs."""
    if directory is None:
        return
    manifest_path = directory / "manifest.json"
    temporary_path = directory / "manifest.json.tmp"
    try:
        merged_records: dict[str, dict[str, Any]] = {}
        if manifest_path.is_file():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                for record in existing.get("frames", []):
                    if isinstance(record, dict) and record.get("file"):
                        merged_records[str(record["file"])] = record
            except (OSError, TypeError, ValueError):
                logger.debug("Ignoring unreadable AI frame cache manifest", exc_info=True)
        for record in records:
            filename = str(record.get("file", ""))
            if filename:
                merged_records[filename] = record

        def manifest_sort_key(record: dict[str, Any]) -> float:
            try:
                return float(record.get("timestamp_ms", 0.0))
            except (TypeError, ValueError):
                return 0.0

        ordered_records = sorted(merged_records.values(), key=manifest_sort_key)
        temporary_path.write_text(
            json.dumps(
                {"version": _AI_FRAME_CACHE_VERSION, "frames": ordered_records},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(temporary_path, manifest_path)
    except OSError:
        logger.debug("Could not persist AI frame cache manifest", exc_info=True)
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass


# The service keeps the historical private names as import aliases.
_safe_frame_cache_scope = safe_frame_cache_scope
_cache_root_path = cache_root_path
_safe_cache_child = safe_cache_child
_frame_cache_manifest_is_valid = frame_cache_manifest_is_valid
_frame_cache_directory = frame_cache_directory
_frame_cache_path = frame_cache_path
_directory_size = directory_size
_read_cached_frame = read_cached_frame
_write_frame_cache_manifest = write_frame_cache_manifest
