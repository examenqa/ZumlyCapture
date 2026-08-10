"""Shared utilities used by multiple modules."""

import logging
import os
import re
import subprocess
import sys
from glob import glob
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

MAX_IMPORTED_IMAGE_EDGE = 8192
MAX_IMPORTED_IMAGE_PIXELS = 7680 * 4320


class ImageValidationError(ValueError):
    """Raised when a user or project image is unsafe to decode."""


def validate_imported_image(
    path: str,
    *,
    max_edge: int = MAX_IMPORTED_IMAGE_EDGE,
    max_pixels: int = MAX_IMPORTED_IMAGE_PIXELS,
) -> tuple[int, int]:
    """Probe image headers without allocating the decoded pixel buffer."""
    candidate = os.path.abspath(os.fspath(path))
    if not os.path.isfile(candidate):
        raise ImageValidationError("The selected image file does not exist.")
    try:
        from PIL import Image, UnidentifiedImageError

        with Image.open(candidate) as image:
            width, height = (int(image.size[0]), int(image.size[1]))
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError(
            "The selected file is not a readable supported image."
        ) from exc
    except Exception as exc:
        # Pillow raises a dedicated decompression-bomb exception before a
        # pixel buffer is allocated for pathologically large headers.
        raise ImageValidationError(
            "The selected image could not be validated safely."
        ) from exc
    if width <= 0 or height <= 0:
        raise ImageValidationError("The selected image has invalid dimensions.")
    if width > max_edge or height > max_edge or width * height > max_pixels:
        raise ImageValidationError(
            "The selected image is too large. Use an image no larger than "
            f"{max_edge}px on either edge and {max_pixels:,} total pixels."
        )
    return width, height


def ffmpeg_exe() -> str:
    """Returns the path to the ffmpeg executable."""
    if getattr(sys, 'frozen', False):
        base_dir = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
        exe_dir = os.path.dirname(sys.executable)
        candidates = [
            os.path.join(base_dir, "ffmpeg.exe"),
            os.path.join(exe_dir, "ffmpeg.exe"),
            os.path.join(base_dir, "imageio_ffmpeg", "binaries", "ffmpeg.exe"),
            os.path.join(exe_dir, "_internal", "imageio_ffmpeg", "binaries", "ffmpeg.exe"),
        ]
        candidates.extend(glob(os.path.join(base_dir, "imageio_ffmpeg", "binaries", "ffmpeg*.exe")))
        candidates.extend(glob(os.path.join(exe_dir, "_internal", "imageio_ffmpeg", "binaries", "ffmpeg*.exe")))
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(f"Bundled FFmpeg executable not found under {base_dir}")

    from imageio_ffmpeg import get_ffmpeg_exe
    return get_ffmpeg_exe()


def subprocess_kwargs() -> dict:
    """Extra kwargs to hide the console window on Windows."""
    kw: dict = {}
    if sys.platform == "win32":
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        kw["startupinfo"] = si
        kw["creationflags"] = subprocess.CREATE_NO_WINDOW
    return kw


def fmt_time(ms: float) -> str:
    """Format milliseconds as m:ss."""
    s = int(ms / 1000)
    m = s // 60
    return f"{m}:{s % 60:02d}"


# ── Hardware-accelerated encoder support ────────────────────────────

# Canonical encoder IDs shared by settings, capture, and export.
ENCODER_NVIDIA = "h264_nvenc"
ENCODER_INTEL = "h264_qsv"
ENCODER_AMD = "h264_amf"
ENCODER_SOFTWARE = "libx264"

# Encoder ID → (display name, ffmpeg codec name, quality args)
# Quality args approximate CRF 18 equivalent for each encoder.
ENCODER_PROFILES: Dict[str, Tuple[str, str, List[str]]] = {
    ENCODER_NVIDIA:  ("NVIDIA NVENC",    ENCODER_NVIDIA, ["-preset", "p4", "-cq", "18", "-b:v", "0"]),
    ENCODER_INTEL:    ("Intel QuickSync",  ENCODER_INTEL,  ["-preset", "medium", "-global_quality", "18"]),
    ENCODER_AMD:      ("AMD AMF",          ENCODER_AMD,    ["-quality", "quality", "-qp_i", "18", "-qp_p", "18"]),
    "h264_mf":     ("Media Foundation", "h264_mf",     ["-rate_control", "quality", "-quality", "80"]),
    ENCODER_SOFTWARE:     ("Software (x264)",  ENCODER_SOFTWARE,     ["-preset", "medium", "-crf", "18"]),
}

# Order of preference for auto-detection.
# h264_mf (Media Foundation) is excluded: its quality is far below libx264
# and it cannot match CRF-18-equivalent output.  libx264 is the preferred
# software fallback and is always appended last by detect_available_encoders().
_HW_ENCODER_ORDER = [ENCODER_NVIDIA, ENCODER_INTEL, ENCODER_AMD]

# Cached result so we only probe once per process
_available_encoders: List[str] | None = None


def _process_output(value: object) -> str:
    """Normalize subprocess output from byte- and text-mode test doubles."""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value or "")


def is_hardware_encoder(enc_id: str) -> bool:
    """Return whether ``enc_id`` is one of Zumly's hardware profiles."""
    return str(enc_id or "") in _HW_ENCODER_ORDER


def probe_encoder_initialization(enc_id: str, ffmpeg: str | None = None) -> bool:
    """Verify that FFmpeg can initialize an encoder, not just list it.

    ``ffmpeg -encoders`` only proves that a codec was compiled into the
    binary.  Hardware codecs can still fail at runtime because the driver is
    missing, the device is busy, or the packaged FFmpeg build cannot access
    the GPU.  A one-frame lavfi encode makes that distinction explicit while
    keeping settings detection independent of WMI vendor names.
    """
    profile = ENCODER_PROFILES.get(str(enc_id or ""))
    if profile is None:
        return False
    ffmpeg = ffmpeg or ffmpeg_exe()
    _, codec, quality_args = profile
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=256x256:r=1",
        "-frames:v",
        "1",
        "-an",
        "-c:v",
        codec,
        *quality_args,
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            timeout=8,
            **subprocess_kwargs(),
        )
    except Exception as exc:
        logger.debug("Encoder initialization probe failed for %s: %s", enc_id, exc)
        return False
    if int(getattr(result, "returncode", 1)) == 0:
        return True
    detail = _process_output(getattr(result, "stderr", "")).strip()
    logger.debug("Encoder %s is unavailable: %s", enc_id, detail[-500:])
    return False


def detect_available_encoders() -> List[str]:
    """Probe ffmpeg for available H.264 encoders.

    Returns a list of encoder IDs (e.g. ``["h264_nvenc", "libx264"]``)
    in preference order.  The software fallback ``libx264`` is always
    included last.  Results are cached after the first call.

    Detection algorithm:
    1. Execute ``ffmpeg -encoders`` once to find compiled codecs.
    2. Run a one-frame initialization probe for every hardware codec found.
    3. Return only hardware encoders that pass both checks, then append the
       software fallback ``libx264``.

    Notes:
    - This is a capability probe, not a guaranteed successful encode probe;
      runtime export still has a fallback chain for launch/mid-stream errors.
    - Caching avoids repeated subprocess overhead on every export.
    """
    global _available_encoders
    if _available_encoders is not None:
        return _available_encoders

    available: List[str] = []
    try:
        ffmpeg = ffmpeg_exe()
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True, timeout=10,
            **subprocess_kwargs(),
        )
        output = _process_output(getattr(result, "stdout", ""))
        output += "\n" + _process_output(getattr(result, "stderr", ""))
        for enc_id in (*_HW_ENCODER_ORDER, "libx264"):
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(enc_id)}(?![A-Za-z0-9_])", output):
                if enc_id == "libx264" or probe_encoder_initialization(enc_id, ffmpeg):
                    available.append(enc_id)
                else:
                    logger.info("FFmpeg listed %s but initialization failed", enc_id)
    except Exception as exc:
        logger.warning("Encoder probe failed: %s", exc)

    # Keep the software fallback selectable even with a partial/broken probe.
    # A real export still verifies its output and reports a failure if the
    # packaged FFmpeg binary truly lacks libx264.
    if "libx264" not in available:
        available.append("libx264")
    _available_encoders = available
    return available


def best_hw_encoder() -> str:
    """Return the best available encoder ID, preferring HW acceleration.

    Falls back to ``"libx264"`` if no HW encoder is found.

    Selection policy is deterministic and follows ``_HW_ENCODER_ORDER`` so
    UI defaults and exporter behavior are consistent across launches.
    """
    encoders = detect_available_encoders()
    return encoders[0] if encoders else "libx264"


def encoder_display_name(enc_id: str) -> str:
    """Human-readable name for an encoder ID."""
    profile = ENCODER_PROFILES.get(enc_id)
    return profile[0] if profile else enc_id


def build_encoder_args(enc_id: str) -> List[str]:
    """Return ffmpeg arguments for the given encoder ID.

    Returns ``["-c:v", "<codec>", ...quality_args..., "-pix_fmt", "yuv420p"]``.

    Argument construction rules:
    - Unknown IDs fall back to ``libx264`` profile.
    - Profile quality args target roughly CRF-18-equivalent quality per codec.
    - ``yuv420p`` is always appended for broad player compatibility.
    """
    profile = ENCODER_PROFILES.get(enc_id)
    if profile is None:
        profile = ENCODER_PROFILES["libx264"]
    _, codec, quality_args = profile
    args = ["-c:v", codec] + quality_args + ["-pix_fmt", "yuv420p"]
    # Place moov atom at the start so players can open the file
    # immediately without seeking to the end first.
    args += ["-movflags", "+faststart"]
    return args


# ── GIF export support ───────────────────────────────────────────────

# Default frames per second for GIF output (balances quality and file size)
GIF_FPS: int = 15


def build_gif_args(gif_fps: int = GIF_FPS) -> List[str]:
    """Return ffmpeg ``-vf`` filter arguments for high-quality GIF output.

    Uses palette generation (``palettegen`` + ``paletteuse``) for accurate
    colours and bayer dithering to reduce banding artefacts. The graph is
    applied in one ffmpeg pass via ``split``:

    ``fps -> split -> palettegen + paletteuse``

    This avoids temporary files and keeps GIF encoding deterministic.

    Returns ``["-vf", "<filtergraph>", "-loop", "0"]``.
    """
    vf = (
        f"fps={gif_fps},"
        "split[s0][s1];"
        "[s0]palettegen=max_colors=256:stats_mode=diff[p];"
        "[s1][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
    )
    return ["-vf", vf, "-loop", "0"]
