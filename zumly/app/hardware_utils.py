"""Small Windows hardware probes used by the export settings UI."""

from __future__ import annotations

import logging
import subprocess

from .utils import (
    ENCODER_AMD,
    ENCODER_INTEL,
    ENCODER_NVIDIA,
    ENCODER_SOFTWARE,
    detect_available_encoders,
)

logger = logging.getLogger(__name__)

def detect_gpu_manufacturers() -> set[str]:
    """Return all detected GPU manufacturers from Windows WMI.

    WMIC was removed from some current Windows installations. In that case we
    use the built-in Windows PowerShell WMI cmdlet before falling back to an
    empty result. Software encoding remains available in every case.
    """
    output = _query_wmic()
    if not output.strip():
        output = _query_powershell_wmi()

    has_nvidia = False
    has_amd = False
    has_intel = False
    for line in output.splitlines():
        name = line.strip().upper()
        if not name or name == "NAME":
            continue
        if "NVIDIA" in name:
            has_nvidia = True
        if "AMD" in name or "RADEON" in name:
            has_amd = True
        if "INTEL" in name:
            has_intel = True

    manufacturers: set[str] = set()
    if has_nvidia:
        manufacturers.add("nvidia")
    if has_amd:
        manufacturers.add("amd")
    if has_intel:
        manufacturers.add("intel")
    return manufacturers


def _query_wmic() -> str:
    """Query the legacy WMIC executable when it is available."""
    try:
        return subprocess.check_output(
            ["wmic", "path", "win32_VideoController", "get", "name"],
            text=True,
            timeout=3,
        )
    except (subprocess.CalledProcessError, OSError, subprocess.SubprocessError, UnicodeError) as exc:
        logger.warning("WMIC GPU discovery failed; trying PowerShell WMI: %s", exc)
        return ""


def _query_powershell_wmi() -> str:
    """Query Win32_VideoController through built-in Windows PowerShell."""
    command = "Get-WmiObject -Class Win32_VideoController | Select-Object -ExpandProperty Name"
    try:
        return subprocess.check_output(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                command,
            ],
            text=True,
            timeout=5,
        )
    except (subprocess.CalledProcessError, OSError, subprocess.SubprocessError, UnicodeError) as exc:
        logger.warning("PowerShell WMI GPU discovery failed; using software encoding: %s", exc)
        return ""


def detect_supported_hardware_encoders() -> set[str]:
    """Return hardware encoders that FFmpeg can actually initialize.

    GPU vendor discovery remains available for diagnostics, but it is not
    used as the capability decision.  The shared utility probe verifies the
    codec in the exact FFmpeg binary Zumly will use, which handles multi-GPU
    systems and stale/missing drivers correctly.
    """
    try:
        available = set(detect_available_encoders())
    except Exception as exc:
        logger.warning("FFmpeg encoder capability discovery failed: %s", exc)
        return set()
    return {
        encoder_id
        for encoder_id in (ENCODER_NVIDIA, ENCODER_INTEL, ENCODER_AMD)
        if encoder_id in available
    }
