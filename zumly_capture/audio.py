"""Optional DirectShow audio capture and pause-safe video muxing."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Callable

import imageio_ffmpeg


logger = logging.getLogger(__name__)


def _subprocess_kwargs() -> dict:
    return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}


def parse_dshow_audio_devices(output: str) -> list[str]:
    devices: list[str] = []
    for line in str(output).splitlines():
        match = re.search(r'"([^"]+)"\s+\(audio\)', line)
        if match and match.group(1) not in devices:
            devices.append(match.group(1))
    return devices


def list_dshow_audio_devices() -> list[str]:
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-list_devices",
        "true",
        "-f",
        "dshow",
        "-i",
        "dummy",
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=8,
            check=False,
            **_subprocess_kwargs(),
        )
        return parse_dshow_audio_devices(result.stderr)
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("Could not enumerate DirectShow audio devices: %s", exc)
        return []


@dataclass(slots=True)
class AudioTrack:
    device: str
    path: str


class AudioCapture:
    def __init__(self, devices: list[str]) -> None:
        self._devices = list(dict.fromkeys(device for device in devices if device))
        self._processes: list[tuple[AudioTrack, subprocess.Popen]] = []
        self.started_at = 0.0

    def start(self) -> None:
        self.started_at = time.perf_counter()
        for device in self._devices:
            handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            path = handle.name
            handle.close()
            command = [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "dshow",
                "-i",
                f"audio={device}",
                "-ac",
                "2",
                "-ar",
                "48000",
                "-c:a",
                "pcm_s16le",
                path,
            ]
            try:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    **_subprocess_kwargs(),
                )
                self._processes.append((AudioTrack(device=device, path=path), process))
            except OSError as exc:
                logger.warning("Could not start audio device %s: %s", device, exc)
                try:
                    os.remove(path)
                except OSError:
                    pass

    def stop(self) -> list[AudioTrack]:
        tracks: list[AudioTrack] = []
        for track, process in self._processes:
            try:
                if process.stdin is not None:
                    process.stdin.write(b"q\n")
                    process.stdin.flush()
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
            if Path(track.path).is_file() and Path(track.path).stat().st_size > 44:
                tracks.append(track)
            else:
                try:
                    os.remove(track.path)
                except OSError:
                    pass
        self._processes.clear()
        return tracks


def _active_wall_segments(
    duration_ms: float,
    pause_boundaries: list[dict],
    lead_ms: float,
) -> list[tuple[float, float]]:
    segments: list[tuple[float, float]] = []
    active_cursor = 0.0
    paused_before = 0.0
    for boundary in sorted(pause_boundaries, key=lambda item: float(item.get("timelineMs", 0.0))):
        pause_at = max(active_cursor, min(duration_ms, float(boundary.get("timelineMs", 0.0))))
        if pause_at > active_cursor:
            segments.append(
                (
                    lead_ms + active_cursor + paused_before,
                    lead_ms + pause_at + paused_before,
                )
            )
        paused_before += max(0.0, float(boundary.get("wallDurationMs", 0.0)))
        active_cursor = pause_at
    if duration_ms > active_cursor:
        segments.append(
            (
                lead_ms + active_cursor + paused_before,
                lead_ms + duration_ms + paused_before,
            )
        )
    return segments or [(max(0.0, lead_ms), max(0.001, lead_ms + duration_ms))]


def mux_audio_tracks(
    video_path: str,
    tracks: list[AudioTrack],
    output_path: str,
    duration_ms: float,
    pause_boundaries: list[dict],
    lead_ms: float = 0.0,
    progress: Callable[[float], None] | None = None,
) -> tuple[bool, str]:
    if not tracks:
        return False, ""
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", video_path]
    for track in tracks:
        command += ["-i", track.path]

    segments = _active_wall_segments(duration_ms, pause_boundaries, max(0.0, lead_ms))
    filters: list[str] = []
    active_labels: list[str] = []
    for track_index in range(len(tracks)):
        part_labels: list[str] = []
        input_index = track_index + 1
        for segment_index, (start_ms, end_ms) in enumerate(segments):
            label = f"a{track_index}_{segment_index}"
            filters.append(
                f"[{input_index}:a]atrim=start={start_ms / 1000.0:.6f}:"
                f"end={end_ms / 1000.0:.6f},asetpts=PTS-STARTPTS[{label}]"
            )
            part_labels.append(f"[{label}]")
        active_label = f"active{track_index}"
        if len(part_labels) == 1:
            filters.append(f"{part_labels[0]}anull[{active_label}]")
        else:
            filters.append(
                f"{''.join(part_labels)}concat=n={len(part_labels)}:v=0:a=1[{active_label}]"
            )
        active_labels.append(f"[{active_label}]")

    if len(active_labels) == 1:
        filters.append(f"{active_labels[0]}anull[aout]")
    else:
        filters.append(
            f"{''.join(active_labels)}amix=inputs={len(active_labels)}:"
            "duration=longest:dropout_transition=0[aout]"
        )
    command += [
        "-filter_complex",
        ";".join(filters),
        "-map",
        "0:v:0",
        "-map",
        "[aout]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-t",
        f"{max(0.001, duration_ms / 1000.0):.6f}",
        "-movflags",
        "+faststart",
        output_path,
    ]
    try:
        if progress:
            progress(0.1)
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            **_subprocess_kwargs(),
        )
        if result.returncode != 0:
            return False, result.stderr[-2000:]
        if progress:
            progress(1.0)
        return Path(output_path).is_file() and Path(output_path).stat().st_size > 0, ""
    except OSError as exc:
        return False, str(exc)


def cleanup_audio_tracks(tracks: list[AudioTrack]) -> None:
    for track in tracks:
        try:
            os.remove(track.path)
        except OSError:
            pass
