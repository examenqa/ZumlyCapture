"""Optional, cancellation-safe Smart Zoom post-processing."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import subprocess
import tempfile
from typing import Callable, Iterable, Optional

from PIL import Image, ImageDraw

from zumly.app.activity_analyzer import analyze_activity
from zumly.app.models import ClickEvent, MousePosition, ZoomKeyframe
from zumly.app.utils import ffmpeg_exe


ProgressCallback = Callable[[int], None]
CancelCallback = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class SmartZoomResult:
    """Outcome of one best-effort Smart Zoom render."""

    state: str
    output_path: str = ""
    keyframes: list[ZoomKeyframe] = field(default_factory=list)
    error: str = ""


def build_zoompan_filter(
    keyframes: Iterable[ZoomKeyframe],
    fps: float,
    width: int,
    height: int,
) -> str:
    """Build a linear-size FFmpeg zoompan expression at source dimensions."""
    native_fps = max(float(fps or 0.0), 1.0)
    ordered = sorted(keyframes, key=lambda keyframe: float(keyframe.timestamp))
    out_width = max(2, int(width) - int(width) % 2)
    out_height = max(2, int(height) - int(height) % 2)
    if not ordered:
        return (
            "zoompan=z='1':x='0':y='0':d=1:"
            f"s={out_width}x{out_height}:fps={native_fps}"
        )

    starts = [max(float(keyframe.timestamp) / 1000.0, 0.0) for keyframe in ordered]

    def smoothstep(start_seconds: float, duration_seconds: float) -> str:
        progress = f"clip((time-{start_seconds:.6f})/{duration_seconds:.6f},0,1)"
        return f"(6*pow({progress},5)-15*pow({progress},4)+10*pow({progress},3))"

    def property_expression(
        default: float,
        attribute: str,
        minimum: Optional[float] = None,
    ) -> str:
        targets = [float(getattr(keyframe, attribute)) for keyframe in ordered]
        if minimum is not None:
            targets = [max(minimum, value) for value in targets]
        terms = [f"if(lt(time,{starts[0]:.6f}),{default:.6f},0)"]
        for index, keyframe in enumerate(ordered):
            start_seconds = starts[index]
            duration_seconds = max(float(keyframe.duration) / 1000.0, 0.0)
            end_seconds = start_seconds + duration_seconds
            previous = default if index == 0 else targets[index - 1]
            target = targets[index]
            if duration_seconds > 0.0:
                eased = smoothstep(start_seconds, duration_seconds)
                local_value = (
                    f"if(lt(time,{end_seconds:.6f}),"
                    f"{previous:.6f}+({target:.6f}-{previous:.6f})*{eased},"
                    f"{target:.6f})"
                )
            else:
                local_value = f"{target:.6f}"
            if index + 1 < len(ordered):
                interval = (
                    f"gte(time,{start_seconds:.6f})*"
                    f"lt(time,{starts[index + 1]:.6f})"
                )
            else:
                interval = f"gte(time,{start_seconds:.6f})"
            terms.append(f"if({interval},{local_value},0)")
        return "(" + "+".join(terms) + ")"

    zoom = property_expression(1.0, "zoom", minimum=1.0)
    pan_x = property_expression(0.5, "x")
    pan_y = property_expression(0.5, "y")
    x = f"clip(({pan_x})*iw-(iw/{zoom})/2,0,iw-iw/{zoom})"
    y = f"clip(({pan_y})*ih-(ih/{zoom})/2,0,ih-ih/{zoom})"
    return (
        f"zoompan=z='{zoom}':x='{x}':y='{y}':d=1:"
        f"s={out_width}x{out_height}:fps={native_fps}"
    )


def downsample_cursor_track(
    mouse_track: Iterable[MousePosition],
    interval_ms: float = 50.0,
) -> list[MousePosition]:
    """Bound expression size while preserving pause/resume discontinuities."""
    ordered = sorted(mouse_track, key=lambda sample: float(sample.timestamp))
    if len(ordered) <= 2:
        return ordered
    selected = [ordered[0]]
    last_timestamp = float(ordered[0].timestamp)
    for sample in ordered[1:-1]:
        timestamp = float(sample.timestamp)
        if sample.resume_boundary or timestamp - last_timestamp >= interval_ms:
            selected.append(sample)
            last_timestamp = timestamp
    if selected[-1] is not ordered[-1]:
        selected.append(ordered[-1])
    return selected


def build_cursor_axis_expression(
    mouse_track: Iterable[MousePosition],
    axis: str,
    capture_rect: dict,
    cursor_extent: int,
) -> str:
    """Build cursor motion without interpolating across a resume boundary."""
    samples = downsample_cursor_track(mouse_track)
    if not samples:
        return "-100"
    if axis not in {"x", "y"}:
        raise ValueError("Cursor axis must be x or y")
    origin = float(capture_rect.get("left" if axis == "x" else "top", 0))
    limit = max(
        0.0,
        float(capture_rect.get("width" if axis == "x" else "height", 1))
        - float(cursor_extent),
    )

    def coordinate(sample: MousePosition) -> float:
        value = float(getattr(sample, axis)) - origin - 2.0
        return max(0.0, min(limit, value))

    values = [coordinate(sample) for sample in samples]
    expression = f"{values[-1]:.3f}"
    for index in range(len(samples) - 2, -1, -1):
        current = samples[index]
        following = samples[index + 1]
        current_time = max(0.0, float(current.timestamp) / 1000.0)
        following_time = max(current_time, float(following.timestamp) / 1000.0)
        if following.resume_boundary or following_time <= current_time:
            segment_value = f"{values[index]:.3f}"
        else:
            progress = (
                f"clip((t-{current_time:.6f})/"
                f"{following_time - current_time:.6f},0,1)"
            )
            segment_value = (
                f"{values[index]:.3f}+"
                f"({values[index + 1]:.3f}-{values[index]:.3f})*{progress}"
            )
        expression = f"if(lt(t,{following_time:.6f}),{segment_value},{expression})"
    return expression


def build_click_filter_chain(
    input_label: str,
    click_events: Iterable[ClickEvent],
    capture_rect: dict,
) -> tuple[str, str]:
    """Create one visible indicator filter for every click event."""
    filters: list[str] = []
    current = input_label
    left = float(capture_rect.get("left", 0))
    top = float(capture_rect.get("top", 0))
    width = max(1.0, float(capture_rect.get("width", 1)))
    height = max(1.0, float(capture_rect.get("height", 1)))
    for index, click in enumerate(sorted(click_events, key=lambda event: event.timestamp)):
        x = max(0, min(int(width) - 28, int(round(float(click.x) - left - 14))))
        y = max(0, min(int(height) - 28, int(round(float(click.y) - top - 14))))
        start = max(0.0, float(click.timestamp) / 1000.0)
        output = f"click{index}"
        filters.append(
            f"[{current}]drawbox=x={x}:y={y}:w=28:h=28:"
            f"color=0xFF5A5ACC:t=4:enable='between(t,{start:.6f},{start + 0.45:.6f})'"
            f"[{output}]"
        )
        current = output
    return ";".join(filters), current


def _create_cursor_image(path: str) -> None:
    image = Image.new("RGBA", (24, 32), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    points = [(2, 1), (2, 25), (8, 19), (13, 30), (18, 27), (12, 17), (22, 17)]
    draw.polygon(points, fill=(255, 255, 255, 255), outline=(20, 20, 20, 255))
    image.save(path, format="PNG")


def _build_filter_graph(
    keyframes: list[ZoomKeyframe],
    mouse_track: list[MousePosition],
    click_events: list[ClickEvent],
    capture_rect: dict,
    fps: float,
    render_cursor: bool,
    render_clicks: bool,
) -> str:
    filters = ["[0:v]setpts=PTS-STARTPTS,format=rgba[base]"]
    current = "base"
    if render_clicks:
        click_chain, current = build_click_filter_chain(current, click_events, capture_rect)
        if click_chain:
            filters.append(click_chain)
    if render_cursor and mouse_track:
        x = build_cursor_axis_expression(mouse_track, "x", capture_rect, 24)
        y = build_cursor_axis_expression(mouse_track, "y", capture_rect, 32)
        filters.append(
            f"[{current}][1:v]overlay=x='{x}':y='{y}':"
            "eval=frame:eof_action=repeat:shortest=1[decorated]"
        )
        current = "decorated"
    zoompan = build_zoompan_filter(
        keyframes,
        fps,
        int(capture_rect.get("width", 2)),
        int(capture_rect.get("height", 2)),
    )
    filters.append(f"[{current}]{zoompan},format=yuv420p[outv]")
    return ";".join(filters)


def _remove_file(path: str) -> None:
    if not path:
        return
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError:
        pass


def render_smart_zoom(
    input_path: str,
    output_path: str,
    mouse_track: list[MousePosition],
    click_events: list[ClickEvent],
    capture_rect: dict,
    duration_ms: float,
    fps: float,
    zoom_level: float = 1.5,
    render_cursor: bool = False,
    render_clicks: bool = True,
    progress_callback: ProgressCallback | None = None,
    cancel_callback: CancelCallback | None = None,
) -> SmartZoomResult:
    """Analyze every click and render Smart Zoom without risking source media."""
    keyframes: list[ZoomKeyframe] = []
    cursor_path = ""
    filter_path = ""
    process: subprocess.Popen[str] | None = None
    last_progress = -1
    try:
        keyframes = analyze_activity(
            mouse_track,
            capture_rect,
            click_events=click_events,
            max_clusters=None,
            zoom_level=float(zoom_level),
            follow_cursor=True,
        )
        if not keyframes:
            return SmartZoomResult(state="no_activity")
        if cancel_callback is not None and cancel_callback():
            return SmartZoomResult(state="cancelled", keyframes=keyframes)

        with tempfile.NamedTemporaryFile(suffix=".filter", delete=False) as handle:
            filter_path = handle.name
        command = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-i", input_path]
        if render_cursor and mouse_track:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
                cursor_path = handle.name
            _create_cursor_image(cursor_path)
            command += ["-loop", "1", "-framerate", str(max(float(fps), 1.0)), "-i", cursor_path]
        graph = _build_filter_graph(
            keyframes,
            mouse_track,
            click_events,
            capture_rect,
            fps,
            render_cursor,
            render_clicks,
        )
        with open(filter_path, "w", encoding="utf-8") as handle:
            handle.write(graph)
            handle.write("\n")
        command += [
            "-filter_complex_script", filter_path,
            "-map", "[outv]",
            "-map", "0:a?",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-movflags", "+faststart",
            "-progress", "pipe:1",
            "-stats_period", "0.25",
            "-nostats",
            output_path,
        ]
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
                _remove_file(output_path)
                return SmartZoomResult(state="cancelled", keyframes=keyframes)
            name, separator, value = raw_line.strip().partition("=")
            if not separator or name not in {"out_time_us", "out_time_ms"}:
                continue
            try:
                # Modern FFmpeg emits both values in microseconds despite the
                # legacy out_time_ms label.
                elapsed_ms = float(value) / 1000.0
            except ValueError:
                continue
            progress = max(0, min(99, int(elapsed_ms * 100.0 / max(duration_ms, 1.0))))
            if progress > last_progress:
                last_progress = progress
                if progress_callback is not None:
                    progress_callback(progress)
        return_code = process.wait()
        if return_code != 0 or not os.path.isfile(output_path) or os.path.getsize(output_path) <= 0:
            _remove_file(output_path)
            return SmartZoomResult(
                state="failed",
                keyframes=keyframes,
                error=f"FFmpeg exited with code {return_code}",
            )
        if progress_callback is not None:
            progress_callback(100)
        return SmartZoomResult(
            state="processed",
            output_path=os.path.abspath(output_path),
            keyframes=keyframes,
        )
    except Exception as exc:
        if process is not None and process.poll() is None:
            process.terminate()
        _remove_file(output_path)
        return SmartZoomResult(state="failed", keyframes=keyframes, error=str(exc))
    finally:
        _remove_file(cursor_path)
        _remove_file(filter_path)
