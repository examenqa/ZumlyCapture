"""Bounded, palette-based GIF export for standalone screen recordings."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
from typing import Callable

from zumly.app.utils import ffmpeg_exe


GIF_FPS = 15
GIF_MAX_EDGE = 1280

ProgressCallback = Callable[[int], None]
CancelCallback = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class GifExportResult:
    state: str
    output_path: str = ""
    error: str = ""


def build_gif_filter(
    fps: int = GIF_FPS,
    max_edge: int = GIF_MAX_EDGE,
) -> str:
    """Return a high-quality GIF filter graph with practical size limits."""
    bounded_fps = max(1, min(30, int(fps)))
    bounded_edge = max(320, min(2560, int(max_edge)))
    scale = (
        "scale="
        f"w='if(gte(iw,ih),min({bounded_edge},iw),-2)':"
        f"h='if(gte(iw,ih),-2,min({bounded_edge},ih))':"
        "flags=lanczos"
    )
    return (
        f"[0:v]fps={bounded_fps},{scale},split[palette_source][gif_source];"
        "[palette_source]palettegen=max_colors=256:stats_mode=diff[palette];"
        "[gif_source][palette]paletteuse="
        "dither=bayer:bayer_scale=5:diff_mode=rectangle[outv]"
    )


def _remove_file(path: str | os.PathLike[str]) -> None:
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def export_gif(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    duration_ms: float,
    *,
    fps: int = GIF_FPS,
    max_edge: int = GIF_MAX_EDGE,
    progress_callback: ProgressCallback | None = None,
    cancel_callback: CancelCallback | None = None,
) -> GifExportResult:
    """Convert one video to a looping, silent GIF without overwriting files."""
    source = Path(input_path).resolve()
    output = Path(output_path).resolve()
    if source == output:
        return GifExportResult(state="failed", error="GIF source and output must differ")
    if not source.is_file() or source.stat().st_size <= 0:
        return GifExportResult(state="failed", error=f"GIF source is not usable: {source}")
    if output.exists():
        return GifExportResult(state="failed", error=f"GIF output already exists: {output}")
    if output.suffix.lower() != ".gif":
        return GifExportResult(state="failed", error="GIF output must use the .gif extension")
    if cancel_callback is not None and cancel_callback():
        return GifExportResult(state="cancelled")

    output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics: deque[str] = deque(maxlen=20)
    process: subprocess.Popen[str] | None = None
    last_progress = -1
    command = [
        ffmpeg_exe(),
        "-n",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-filter_complex",
        build_gif_filter(fps=fps, max_edge=max_edge),
        "-map",
        "[outv]",
        "-an",
        "-loop",
        "0",
        "-progress",
        "pipe:1",
        "-stats_period",
        "0.25",
        "-nostats",
        str(output),
    ]
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if progress_callback is not None:
            progress_callback(0)
        assert process.stdout is not None
        for raw_line in process.stdout:
            if cancel_callback is not None and cancel_callback():
                process.terminate()
                try:
                    process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3.0)
                _remove_file(output)
                return GifExportResult(state="cancelled")
            name, separator, value = raw_line.strip().partition("=")
            if not separator or name not in {"out_time_us", "out_time_ms"}:
                line = raw_line.strip()
                if line and name not in {
                    "bitrate",
                    "drop_frames",
                    "dup_frames",
                    "fps",
                    "frame",
                    "out_time",
                    "progress",
                    "speed",
                    "stream_0_0_q",
                    "total_size",
                }:
                    diagnostics.append(line[-900:])
                continue
            try:
                elapsed_ms = float(value) / 1000.0
            except ValueError:
                continue
            progress = max(0, min(99, int(elapsed_ms * 100.0 / max(duration_ms, 1.0))))
            if progress > last_progress:
                last_progress = progress
                if progress_callback is not None:
                    progress_callback(progress)
        return_code = process.wait()
        if return_code != 0 or not output.is_file() or output.stat().st_size <= 0:
            _remove_file(output)
            details = " | ".join(diagnostics)[-4000:]
            message = f"FFmpeg GIF export exited with code {return_code}"
            if details:
                message = f"{message}: {details}"
            return GifExportResult(state="failed", error=message)
        if progress_callback is not None:
            progress_callback(100)
        return GifExportResult(state="processed", output_path=str(output))
    except Exception as exc:
        if process is not None and process.poll() is None:
            process.terminate()
        _remove_file(output)
        return GifExportResult(state="failed", error=str(exc))

