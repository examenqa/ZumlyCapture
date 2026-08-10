"""Project file management — save / load .fcproj bundles.

A .fcproj file is a ZIP archive containing:
  - project.json   — session metadata (mouse track, keyframes, click events,
                      voiceover segments, generated narration scripts, etc.)
  - recording.mp4  — the raw H.264 intermediate video
  - voiceover_*.wav — synthesized voiceover audio files (one per segment)
  - assets/audio/* — one optional custom global background-music asset

This lets users save their work and resume editing later.
"""

import atexit
import hashlib
import json
import logging
import os
import shutil
import struct
import time
import uuid
import zipfile
import zlib
from pathlib import Path
from typing import List, Optional

# Track extraction directories so they can be cleaned up on exit.
_extract_dirs: List[str] = []


def _cleanup_extract_dirs() -> None:
    """Remove temporary extraction directories on interpreter exit."""
    for d in _extract_dirs:
        try:
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass


atexit.register(_cleanup_extract_dirs)


def _discard_extract_dir(extract_dir: str) -> None:
    """Delete one tracked extraction directory after a failed load."""
    try:
        if os.path.isdir(extract_dir):
            shutil.rmtree(extract_dir, ignore_errors=True)
    finally:
        if extract_dir in _extract_dirs:
            _extract_dirs.remove(extract_dir)


def _new_extract_dir() -> str:
    """Create a writable extraction directory without an open temp handle."""
    root = Path(os.environ.get("TEMP", os.getcwd())) / "Zumly" / "projects"
    root.mkdir(parents=True, exist_ok=True)
    directory = root / f"project-{uuid.uuid4().hex}"
    directory.mkdir(parents=False, exist_ok=False)
    return str(directory)

from .models import RecordingSession, ClickEffectPreset, KeystrokeOverlayConfig, AnnotationCollection
from .backgrounds import BackgroundPreset
from .frames import FramePreset
from .utils import validate_imported_image

logger = logging.getLogger(__name__)

PROJ_EXT = ".fcproj"
_JSON_NAME = "project.json"
_VIDEO_NAME = "recording.mp4"
_FRAME_IMAGE_DIR = "frame_images"
_CURSOR_ASSET_DIR = "cursor_assets"
_AUDIO_ASSET_DIR = "assets/audio"
_PROJECT_AUDIO_EXTENSIONS = frozenset({".mp3", ".wav", ".m4a"})
MAX_PROJECT_AUDIO_ASSET_SIZE = 50 * 1024 * 1024
MAX_PROJECT_ARCHIVE_MEMBERS = 500
MAX_PROJECT_ARCHIVE_TOTAL_SIZE = 500 * 1024 * 1024
MAX_PROJECT_ARCHIVE_MEMBER_SIZE = 500 * 1024 * 1024
MAX_PROJECT_ARCHIVE_COMPRESSION_RATIO = 200.0
_COMPRESSION_RATIO_MIN_SIZE = 1024 * 1024
_EXTRACTION_CHUNK_SIZE = 1024 * 1024


class SecurityError(ValueError):
    """Raised when a project archive attempts to write outside its extract dir."""


def _sibling_temp_path(path: str) -> str:
    """Return a temporary path on the destination filesystem."""
    return f"{path}.tmp"


def _validate_project_archive(path: str, *, require_video: bool = False) -> None:
    """Reopen and validate a fully written project bundle before commit."""
    with zipfile.ZipFile(path, "r") as archive:
        names = set(archive.namelist())
        if _JSON_NAME not in names:
            raise ValueError("Project bundle is missing project.json")
        if require_video and _VIDEO_NAME not in names:
            raise ValueError("Project bundle is missing recording.mp4")
        corrupt_member = archive.testzip()
        if corrupt_member:
            raise ValueError(f"Project bundle contains a corrupt entry: {corrupt_member}")
        payload = json.loads(archive.read(_JSON_NAME).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Project metadata must be a JSON object")
        music = payload.get("backgroundMusic")
        if isinstance(music, dict):
            music_path = str(music.get("assetPath", "") or "").replace("\\", "/")
            if music_path and music_path not in names:
                raise ValueError("Project bundle is missing its background music asset")


def atomic_write_json(path: str, payload: dict) -> None:
    """Atomically replace a JSON bridge after validating the staged payload."""
    temp_path = _sibling_temp_path(path)
    try:
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        with open(temp_path, "r", encoding="utf-8") as handle:
            validated = json.load(handle)
        if not isinstance(validated, dict):
            raise ValueError("Project bridge must contain a JSON object")
        os.replace(temp_path, path)
    finally:
        try:
            os.remove(temp_path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("Could not remove staged JSON %s: %s", temp_path, exc)


def _contained_path(root: str, path: str, *, allow_root: bool = False) -> str:
    """Resolve *path* and require it to remain under *root*."""
    root_abs = os.path.realpath(root)
    path_abs = os.path.realpath(path)
    root_cmp = os.path.normcase(root_abs)
    path_cmp = os.path.normcase(path_abs)
    try:
        common = os.path.commonpath([root_cmp, path_cmp])
    except ValueError as exc:
        raise SecurityError("Project path is on a different filesystem") from exc
    if common != root_cmp or (not allow_root and path_cmp == root_cmp):
        raise SecurityError(f"Project path escapes extract directory: {path}")
    return path_abs


def _validate_archive_limits(members) -> None:
    """Reject oversized archives before any member is opened or written."""
    if len(members) > MAX_PROJECT_ARCHIVE_MEMBERS:
        raise SecurityError(
            f"Project archive contains too many members (maximum {MAX_PROJECT_ARCHIVE_MEMBERS})"
        )

    total_size = 0
    for member in members:
        try:
            member_size = int(member.file_size)
        except (AttributeError, TypeError, ValueError) as exc:
            raise SecurityError("Project archive contains an invalid member size") from exc
        if member_size < 0 or member_size > MAX_PROJECT_ARCHIVE_MEMBER_SIZE:
            raise SecurityError(
                f"Project archive member is too large: {member.filename}"
            )
        normalized_name = str(member.filename or "").replace("\\", "/")
        if (
            normalized_name.startswith(f"{_AUDIO_ASSET_DIR}/")
            and member_size > MAX_PROJECT_AUDIO_ASSET_SIZE
        ):
            raise SecurityError(
                f"Project audio asset is too large: {member.filename}"
            )
        try:
            compressed_size = int(member.compress_size)
        except (AttributeError, TypeError, ValueError) as exc:
            raise SecurityError("Project archive contains an invalid compressed size") from exc
        if compressed_size < 0:
            raise SecurityError("Project archive contains an invalid compressed size")
        if member_size >= _COMPRESSION_RATIO_MIN_SIZE:
            if compressed_size == 0:
                raise SecurityError(
                    f"Project archive member has an unsafe compression ratio: {member.filename}"
                )
            ratio = member_size / compressed_size
            if ratio > MAX_PROJECT_ARCHIVE_COMPRESSION_RATIO:
                raise SecurityError(
                    f"Project archive member has an unsafe compression ratio: {member.filename}"
                )
        total_size += member_size
        if total_size > MAX_PROJECT_ARCHIVE_TOTAL_SIZE:
            raise SecurityError(
                f"Project archive is too large when extracted (maximum {MAX_PROJECT_ARCHIVE_TOTAL_SIZE} bytes)"
            )


def _resolve_internal_asset(
    extract_dir: str,
    asset_path: str,
    *,
    required_prefix: str,
) -> str:
    """Resolve an asset reference that must point inside the project bundle."""
    normalized = str(asset_path or "").replace("\\", "/")
    prefix = f"{required_prefix}/"
    if not normalized.startswith(prefix):
        raise SecurityError(f"Project asset path is not bundle-owned: {asset_path}")
    if ".." in normalized.split("/"):
        raise SecurityError(f"Project asset path contains traversal: {asset_path}")
    asset_root = os.path.join(extract_dir, *required_prefix.split("/"))
    candidate = os.path.join(extract_dir, *normalized.split("/"))
    resolved = _contained_path(asset_root, candidate)
    return resolved if os.path.isfile(resolved) else ""


def _resolve_voiceover_asset(extract_dir: str, asset_name: str) -> str:
    """Resolve a generated voiceover archive member without path components."""
    normalized = str(asset_name or "").replace("\\", "/")
    if (
        os.path.basename(normalized) != normalized
        or not normalized.startswith("voiceover_")
        or not normalized.lower().endswith((".wav", ".mp3"))
    ):
        raise SecurityError(f"Project voiceover asset path is unsafe: {asset_name}")
    resolved = _contained_path(extract_dir, os.path.join(extract_dir, normalized))
    return resolved if os.path.isfile(resolved) else ""


def _validate_project_audio_asset(path: str) -> str:
    """Validate a custom music file without decoding it into memory."""
    resolved = os.path.realpath(str(path or ""))
    if not resolved or not os.path.isfile(resolved):
        raise ValueError("Background music file is missing")
    extension = Path(resolved).suffix.lower()
    if extension not in _PROJECT_AUDIO_EXTENSIONS:
        allowed = ", ".join(sorted(_PROJECT_AUDIO_EXTENSIONS))
        raise ValueError(f"Unsupported background music type; choose {allowed}")
    size = os.path.getsize(resolved)
    if size <= 0:
        raise ValueError("Background music file is empty")
    if size > MAX_PROJECT_AUDIO_ASSET_SIZE:
        raise ValueError("Background music must be 50 MB or smaller")
    return resolved


def _content_addressed_audio_name(path: str) -> str:
    """Return a stable bundle member name without exposing the source path."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"{_AUDIO_ASSET_DIR}/{digest.hexdigest()[:24]}{Path(path).suffix.lower()}"


def _validate_serialized_asset_references(data: dict) -> None:
    """Reject asset references that cannot belong to a project archive."""
    for frame in data.get("timelineFrames") or []:
        if not isinstance(frame, dict):
            continue
        raw_path = str(frame.get("imagePath", "") or "")
        if raw_path and not raw_path.replace("\\", "/").startswith(f"{_FRAME_IMAGE_DIR}/"):
            raise SecurityError(f"Timeline image asset is not bundle-owned: {raw_path}")

    for segment in data.get("voiceoverSegments") or []:
        if not isinstance(segment, dict):
            continue
        raw_path = str(segment.get("audioPath", "") or "")
        if not raw_path:
            continue
        normalized = raw_path.replace("\\", "/")
        if (
            os.path.basename(normalized) != normalized
            or not normalized.startswith("voiceover_")
            or not normalized.lower().endswith((".wav", ".mp3"))
        ):
            raise SecurityError(f"Voiceover asset is not bundle-owned: {raw_path}")

    cursor_path = str(data.get("cursorAssetPath", "") or "")
    if cursor_path and not cursor_path.replace("\\", "/").startswith(
        f"{_CURSOR_ASSET_DIR}/"
    ):
        raise SecurityError(f"Cursor asset is not bundle-owned: {cursor_path}")

    music_rows: list[dict] = []
    music = data.get("backgroundMusic")
    if isinstance(music, dict):
        music_rows.append(music)
    legacy_music = data.get("musicClips")
    if isinstance(legacy_music, list):
        music_rows.extend(item for item in legacy_music if isinstance(item, dict))
    for music in music_rows:
        music_path = str(music.get("assetPath", "") or "")
        if music_path:
            normalized = music_path.replace("\\", "/")
            if not normalized.startswith(f"{_AUDIO_ASSET_DIR}/"):
                raise SecurityError(
                    f"Background music asset is not bundle-owned: {music_path}"
                )
            if Path(normalized).suffix.lower() not in _PROJECT_AUDIO_EXTENSIONS:
                raise SecurityError(
                    f"Background music asset type is unsafe: {music_path}"
                )


def _annotation_count(annotations: Optional[AnnotationCollection]) -> int:
    """Return the number of legacy annotations carried by *annotations*."""
    if not annotations:
        return 0
    return sum(
        len(items or [])
        for items in (annotations.texts, annotations.arrows, annotations.highlights)
    )


def save_project(
    output_path: str,
    video_path: str,
    session: RecordingSession,
    monitor_rect: Optional[dict] = None,
    actual_fps: float = 30.0,
    bg_preset: Optional[BackgroundPreset] = None,
    frame_preset: Optional[FramePreset] = None,
    click_preset: Optional[ClickEffectPreset] = None,
    keystroke_config: Optional[KeystrokeOverlayConfig] = None,
    annotations = None,
    metadata_only: bool = False,
) -> str:
    """Bundle session + raw video into a .fcproj ZIP file.

    When *metadata_only* is True and the output file already exists,
    the save is optimised to avoid re-reading or re-copying the large
    video entry:

    1. **In-place rewrite** (preferred) — if the video is the first
       entry at offset 0, everything after the video's raw data is
       replaced with a fresh JSON entry, central directory, and EOCD
       record.  Total write is O(JSON), the multi-MB video is never
       read or copied.
    2. **Streaming copy** (fallback) — if the layout doesn't allow
       in-place rewrite, the video is streamed in 8 MB chunks to a new
       ZIP (no huge single allocation).

    Returns the final output path.
    """
    if not output_path.lower().endswith(PROJ_EXT):
        output_path += PROJ_EXT

    # Build project JSON (session data + extras)
    data = json.loads(session.to_json())
    frame_image_entries: list[tuple[str, str]] = []
    cursor_asset_entry: tuple[str, str] | None = None
    music_asset_entry: tuple[str, str] | None = None
    if session.timeline_frames:
        frames_json = data.get("timelineFrames", [])
        for frame, frame_json in zip(session.timeline_frames, frames_json):
            if frame.kind != "image" or not frame.image_path or not os.path.isfile(frame.image_path):
                continue
            validate_imported_image(frame.image_path)
            _, ext = os.path.splitext(frame.image_path)
            ext = ext if ext else ".png"
            arc_name = f"{_FRAME_IMAGE_DIR}/{frame.id}{ext.lower()}"
            frame_json["imagePath"] = arc_name
            frame_image_entries.append((frame.image_path, arc_name))
    if session.cursor_asset_path and os.path.isfile(session.cursor_asset_path):
        _, ext = os.path.splitext(session.cursor_asset_path)
        ext = ext.lower() if ext else ".png"
        arc_name = f"{_CURSOR_ASSET_DIR}/cursor{ext}"
        data["cursorAssetPath"] = arc_name
        cursor_asset_entry = (session.cursor_asset_path, arc_name)
    else:
        data.pop("cursorAssetPath", None)
    if session.background_music is not None:
        music_json = data.get("backgroundMusic")
        if not isinstance(music_json, dict):
            raise ValueError("Background music metadata is invalid")
        if session.background_music.is_custom:
            source_path = _validate_project_audio_asset(
                session.background_music.asset_path
            )
            arc_name = _content_addressed_audio_name(source_path)
            music_json["assetPath"] = arc_name
            music_asset_entry = (source_path, arc_name)
        else:
            # Built-in tracks resolve from the installed application bundle.
            music_json.pop("assetPath", None)
    if session.key_events:
        logger.info(
            "Ignoring %d removed keystroke event(s) during project save",
            len(session.key_events),
        )
    data.pop("keyEvents", None)
    if monitor_rect:
        data["monitorRect"] = monitor_rect
    data["actualFps"] = actual_fps
    if bg_preset:
        data["bgPreset"] = bg_preset.to_dict()
    if frame_preset:
        data["framePreset"] = frame_preset.to_dict()
    if click_preset:
        data["clickPreset"] = click_preset.to_dict()
    if keystroke_config and getattr(keystroke_config, "enabled", False):
        logger.info("Ignoring removed keystroke overlay settings during project save")
    annotation_count = _annotation_count(annotations)
    if annotation_count:
        logger.info(
            "Ignoring %d removed annotation(s) during project save",
            annotation_count,
        )

    json_str = json.dumps(data, indent=2)

    # When voiceover audio files exist, always do a full save since
    # the fast metadata rewrite and streaming copy don't handle the
    # extra ZIP entries for voiceover audio.
    has_vo_audio = (
        session.voiceover_segments
        and any(s.audio_path and os.path.isfile(s.audio_path)
                for s in session.voiceover_segments)
    )
    has_frame_images = bool(frame_image_entries)
    has_cursor_asset = cursor_asset_entry is not None
    has_music_asset = music_asset_entry is not None

    if (
        metadata_only
        and os.path.isfile(output_path)
        and not has_vo_audio
        and not has_frame_images
        and not has_cursor_asset
        and not has_music_asset
    ):
        t0 = time.perf_counter()
        _streaming_metadata_save(output_path, json_str)
        logger.info(
            "Metadata save (atomic streaming): %.1f ms",
            (time.perf_counter() - t0) * 1000,
        )
        return output_path

    temp_path = _sibling_temp_path(output_path)
    has_video = bool(video_path and os.path.isfile(video_path))
    try:
        with zipfile.ZipFile(temp_path, "w", zipfile.ZIP_STORED) as zf:
            if has_video:
                zf.write(video_path, _VIDEO_NAME)
            if session.voiceover_segments:
                for seg in session.voiceover_segments:
                    if seg.audio_path and os.path.isfile(seg.audio_path):
                        arc_name = f"voiceover_{seg.id[:8]}.wav"
                        zf.write(seg.audio_path, arc_name)
            for source_path, arc_name in frame_image_entries:
                zf.write(source_path, arc_name)
            if cursor_asset_entry:
                zf.write(*cursor_asset_entry)
            if music_asset_entry:
                zf.write(*music_asset_entry)
            zf.writestr(_JSON_NAME, json_str)
        _validate_project_archive(temp_path, require_video=has_video)
        os.replace(temp_path, output_path)
    finally:
        try:
            os.remove(temp_path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("Could not remove staged project %s: %s", temp_path, exc)

    return output_path


# ── fast metadata helpers ───────────────────────────────────────────


def _fast_metadata_rewrite(zip_path: str, json_str: str) -> bool:
    """Rewrite only the JSON in a .fcproj ZIP without touching video data.

    Requires the video entry to be the first entry at offset 0 — the
    layout produced by ``save_project`` full saves.  Everything after
    the video's raw data is replaced with a fresh JSON local-file-header,
    central directory, and end-of-central-directory record.

    Returns True on success, False when a fallback is needed.
    """
    try:
        # ── Validate layout ─────────────────────────────────────────
        with zipfile.ZipFile(zip_path, "r") as zf:
            if _VIDEO_NAME not in zf.namelist():
                return False
            vi = zf.getinfo(_VIDEO_NAME)
            if vi.header_offset != 0:
                return False          # video not first — can't truncate
            if vi.file_size >= 0x7FFFFFFF:
                return False          # ZIP64 territory — fall back

        # Parse the actual local-file-header to get field lengths
        with open(zip_path, "rb") as f:
            lfh = f.read(30)
        if lfh[:4] != b"PK\x03\x04":
            return False
        fn_len, extra_len = struct.unpack_from("<HH", lfh, 26)
        video_end = 30 + fn_len + extra_len + vi.compress_size

        # ── Build new tail (JSON + CD + EOCD) ───────────────────────
        json_raw = json_str.encode("utf-8")
        json_crc = zlib.crc32(json_raw) & 0xFFFFFFFF
        jfn = _JSON_NAME.encode("utf-8")
        vfn = _VIDEO_NAME.encode("utf-8")

        buf = bytearray()

        # JSON local-file-header + filename + data
        json_lfh_offset = video_end
        buf += struct.pack(
            "<4sHHHHHIIIHH",
            b"PK\x03\x04", 20, 0, 0, 0, 0,
            json_crc, len(json_raw), len(json_raw), len(jfn), 0,
        )
        buf += jfn
        buf += json_raw

        # Central directory
        cd_offset = video_end + len(buf)
        cd_start = len(buf)

        # Video CD entry
        buf += struct.pack(
            "<4sHHHHHHIIIHHHHHII",
            b"PK\x01\x02", 20, 20, 0, 0, 0, 0,
            vi.CRC, vi.compress_size, vi.file_size,
            len(vfn), 0, 0, 0, 0, 0, 0,
        )
        buf += vfn

        # JSON CD entry
        buf += struct.pack(
            "<4sHHHHHHIIIHHHHHII",
            b"PK\x01\x02", 20, 20, 0, 0, 0, 0,
            json_crc, len(json_raw), len(json_raw),
            len(jfn), 0, 0, 0, 0, 0, json_lfh_offset,
        )
        buf += jfn

        cd_size = len(buf) - cd_start

        # End of central directory
        buf += struct.pack(
            "<4sHHHHIIH",
            b"PK\x05\x06", 0, 0, 2, 2,
            cd_size, cd_offset, 0,
        )

        # ── Write in-place ──────────────────────────────────────────
        with open(zip_path, "r+b") as f:
            f.seek(video_end)
            f.write(bytes(buf))
            f.truncate()

        return True
    except Exception:
        logger.exception("Fast metadata rewrite failed, will use fallback")
        return False


def _streaming_metadata_save(zip_path: str, json_str: str) -> None:
    """Atomically rewrite project metadata while preserving every asset."""
    tmp_path = _sibling_temp_path(zip_path)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf_old, \
             zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_STORED) as zf_new:
            require_video = _VIDEO_NAME in zf_old.namelist()
            for info in zf_old.infolist():
                if info.filename == _JSON_NAME:
                    continue
                if info.is_dir():
                    zf_new.writestr(info, b"")
                    continue
                with zf_old.open(info) as src, zf_new.open(info, "w") as dst:
                    shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
            zf_new.writestr(_JSON_NAME, json_str)
        _validate_project_archive(tmp_path, require_video=require_video)
        os.replace(tmp_path, zip_path)
    finally:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("Could not remove staged project %s: %s", tmp_path, exc)


def load_project(input_path: str) -> dict:
    """Extract a .fcproj ZIP and return all project data.

    Returns dict with keys:
      - session: RecordingSession
      - video_path: str — path to extracted video (in temp dir)
      - monitor_rect: dict | None
      - actual_fps: float
    """
    if not zipfile.is_zipfile(input_path):
        raise ValueError(f"Not a valid project file: {input_path}")

    # Extract to a temp directory
    extract_dir = _new_extract_dir()
    _extract_dirs.append(extract_dir)

    try:
        with zipfile.ZipFile(input_path, "r") as zf:
            members = zf.infolist()
            _validate_archive_limits(members)
            extracted_bytes = 0
            for member in members:
                member_name = member.filename
                target_path = _contained_path(
                    extract_dir,
                    os.path.join(extract_dir, member_name),
                    allow_root=member.is_dir(),
                )

                if member.is_dir():
                    os.makedirs(target_path, exist_ok=True)
                    continue

                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with zf.open(member, "r") as src, open(target_path, "wb") as dst:
                    member_bytes = 0
                    while True:
                        chunk = src.read(_EXTRACTION_CHUNK_SIZE)
                        if not chunk:
                            break
                        member_bytes += len(chunk)
                        extracted_bytes += len(chunk)
                        if member_bytes > MAX_PROJECT_ARCHIVE_MEMBER_SIZE:
                            raise SecurityError(
                                f"Project archive member exceeded its extraction limit: {member_name}"
                            )
                        if extracted_bytes > MAX_PROJECT_ARCHIVE_TOTAL_SIZE:
                            raise SecurityError(
                                "Project archive exceeded the maximum extracted size "
                                f"({MAX_PROJECT_ARCHIVE_TOTAL_SIZE} bytes)"
                            )
                        dst.write(chunk)
                    if member_bytes != int(member.file_size):
                        raise SecurityError(
                            f"Project archive member size does not match its metadata: {member_name}"
                        )
    except Exception:
        _discard_extract_dir(extract_dir)
        raise

    try:
        json_path = os.path.join(extract_dir, _JSON_NAME)
        video_path = os.path.join(extract_dir, _VIDEO_NAME)

        if not os.path.isfile(json_path):
            raise ValueError(f"Project file missing {_JSON_NAME}")

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.loads(f.read())
            if not isinstance(data, dict):
                raise ValueError("Project metadata must be a JSON object")
            _validate_serialized_asset_references(data)
            session = RecordingSession.from_json(json.dumps(data))
        except SecurityError:
            raise
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Corrupted project file: {exc}") from exc

        # Restore voiceover audio paths from extracted files
        if session.voiceover_segments:
            for seg in session.voiceover_segments:
                # Check for both .wav (new) and .mp3 (legacy) files
                extracted_audio = ""
                for ext in (".wav", ".mp3"):
                    arc_name = f"voiceover_{seg.id[:8]}{ext}"
                    extracted = _resolve_voiceover_asset(extract_dir, arc_name)
                    if extracted:
                        extracted_audio = extracted
                        break
                if extracted_audio:
                    seg.audio_path = extracted_audio
                elif seg.audio_path:
                    raise SecurityError(
                        f"Voiceover asset is not bundled in the project: {seg.audio_path}"
                    )
                else:
                    seg.audio_path = ""

        # Restore inserted picture frame image paths from extracted files.
        if session.timeline_frames:
            for frame in session.timeline_frames:
                if frame.kind != "image" or not frame.image_path:
                    continue
                normalized = frame.image_path.replace("\\", "/")
                if not normalized.startswith(f"{_FRAME_IMAGE_DIR}/"):
                    raise SecurityError(
                        f"Timeline image asset is not bundle-owned: {frame.image_path}"
                    )
                frame.image_path = _resolve_internal_asset(
                    extract_dir,
                    normalized,
                    required_prefix=_FRAME_IMAGE_DIR,
                )
                validate_imported_image(frame.image_path)

        if session.cursor_asset_path:
            session.cursor_asset_path = _resolve_internal_asset(
                extract_dir,
                session.cursor_asset_path,
                required_prefix=_CURSOR_ASSET_DIR,
            )

        if session.background_music is not None and session.background_music.asset_path:
            extracted_music = _resolve_internal_asset(
                extract_dir,
                session.background_music.asset_path,
                required_prefix=_AUDIO_ASSET_DIR,
            )
            if not extracted_music:
                raise SecurityError("Background music asset is missing from the project")
            session.background_music.asset_path = _validate_project_audio_asset(
                extracted_music
            )
        elif session.background_music is not None and session.background_music.is_custom:
            raise ValueError("Custom background music is missing its bundled audio asset")

        monitor_rect = data.get("monitorRect")
        actual_fps = data.get("actualFps", 30.0)

        bg_preset = None
        if "bgPreset" in data:
            try:
                bg_preset = BackgroundPreset.from_dict(data["bgPreset"])
            except Exception:
                pass

        frame_preset = None
        if "framePreset" in data:
            try:
                frame_preset = FramePreset.from_dict(data["framePreset"])
            except Exception:
                pass

        click_preset = None
        if "clickPreset" in data:
            try:
                click_preset = ClickEffectPreset.from_dict(data["clickPreset"])
            except Exception:
                pass

        if data.get("keyEvents"):
            logger.info("Ignoring %d legacy keystroke event(s) in project file", len(data["keyEvents"]))
        if "keystrokeConfig" in data:
            logger.info("Ignoring removed keystroke overlay settings in project file")
        if "annotations" in data:
            raw_annotations = data.get("annotations") or {}
            legacy_annotation_count = sum(
                len(raw_annotations.get(key) or [])
                for key in ("texts", "arrows", "highlights")
            )
            logger.info(
                "Ignoring %d legacy annotation(s) in project file",
                legacy_annotation_count,
            )

        return {
            "session": session,
            "project_data": data,
            "video_path": video_path if os.path.isfile(video_path) else "",
            "monitor_rect": monitor_rect,
            "actual_fps": actual_fps,
            "bg_preset": bg_preset,
            "frame_preset": frame_preset,
            "click_preset": click_preset,
            "keystroke_config": None,
            "annotations": None,
        }
    except Exception:
        _discard_extract_dir(extract_dir)
        raise
