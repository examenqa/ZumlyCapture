"""Optional, cancellation-safe Smart Zoom post-processing."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import os
import subprocess
import tempfile
from typing import Callable, Iterable, Optional

from PIL import Image, ImageDraw

from zumly.app.activity_analyzer import analyze_activity
from zumly.app.icon_loader import get_resource_path
from zumly.app.models import ClickEvent, MousePosition, ZoomKeyframe
from zumly.app.utils import ffmpeg_exe


ProgressCallback = Callable[[int], None]
CancelCallback = Callable[[], bool]

CURSOR_WIDTH = 44
CURSOR_HEIGHT = 48
CURSOR_HOTSPOT_X = 2
CURSOR_HOTSPOT_Y = 2
CURSOR_ASSET_PATH = "zumly/app/cursors/recording_cursor.png"
CLICK_RIPPLE_SIZE = 84
CLICK_RIPPLE_DURATION_SECONDS = 0.62


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
        hotspot = CURSOR_HOTSPOT_X if axis == "x" else CURSOR_HOTSPOT_Y
        value = float(getattr(sample, axis)) - origin - float(hotspot)
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


def build_cursor_command_script(
    mouse_track: Iterable[MousePosition],
    capture_rect: dict,
    target: str = "overlay@cursor",
) -> str:
    """Build bounded FFmpeg commands for smooth cursor motion.

    A single deeply nested overlay expression exceeds FFmpeg's expression
    parser on ordinary 20–30 second recordings. ``sendcmd`` keeps each motion
    segment independent while retaining frame-by-frame interpolation.
    """
    samples = downsample_cursor_track(mouse_track)
    if not samples:
        return ""

    left = float(capture_rect.get("left", 0))
    top = float(capture_rect.get("top", 0))
    width = max(float(capture_rect.get("width", 1)), 1.0)
    height = max(float(capture_rect.get("height", 1)), 1.0)

    def coordinates(sample: MousePosition) -> tuple[float, float]:
        x = max(
            0.0,
            min(
                max(width - float(CURSOR_WIDTH), 0.0),
                float(sample.x) - left - float(CURSOR_HOTSPOT_X),
            ),
        )
        y = max(
            0.0,
            min(
                max(height - float(CURSOR_HEIGHT), 0.0),
                float(sample.y) - top - float(CURSOR_HOTSPOT_Y),
            ),
        )
        return x, y

    commands: list[str] = []
    first_x, first_y = coordinates(samples[0])
    first_time = max(0.0, float(samples[0].timestamp) / 1000.0)
    commands.append(
        f"{first_time:.6f} {target} x {first_x:.3f}, {target} y {first_y:.3f};"
    )

    for current, following in zip(samples, samples[1:]):
        start = max(0.0, float(current.timestamp) / 1000.0)
        end = max(start, float(following.timestamp) / 1000.0)
        current_x, current_y = coordinates(current)
        following_x, following_y = coordinates(following)
        if following.resume_boundary or end - start <= 0.000001:
            commands.append(
                f"{end:.6f} {target} x {following_x:.3f}, "
                f"{target} y {following_y:.3f};"
            )
            continue
        commands.append(
            f"{start:.6f}-{end:.6f} "
            f"[enter+expr] {target} x "
            f"'{current_x:.3f}+({following_x:.3f}-{current_x:.3f})*TI', "
            f"[enter+expr] {target} y "
            f"'{current_y:.3f}+({following_y:.3f}-{current_y:.3f})*TI';"
        )

    last_x, last_y = coordinates(samples[-1])
    last_time = max(0.0, float(samples[-1].timestamp) / 1000.0)
    commands.append(
        f"{last_time:.6f} {target} x {last_x:.3f}, {target} y {last_y:.3f};"
    )
    return "\n".join(commands)


def _escape_filter_path(path: str) -> str:
    """Escape an absolute Windows path for an FFmpeg filter option."""
    return (
        os.path.abspath(path)
        .replace("\\", "/")
        .replace(":", r"\:")
        .replace("'", r"\'")
    )


def build_click_filter_chain(
    input_label: str,
    click_events: Iterable[ClickEvent],
    capture_rect: dict,
    ripple_input_label: str = "1:v",
) -> tuple[str, str]:
    """Overlay an expanding circular ripple for every recorded click."""
    filters: list[str] = []
    current = input_label
    left = float(capture_rect.get("left", 0))
    top = float(capture_rect.get("top", 0))
    clicks = sorted(click_events, key=lambda event: event.timestamp)
    if not clicks:
        return "", current

    ripple_sources: list[str]
    if len(clicks) == 1:
        ripple_sources = [ripple_input_label]
    else:
        ripple_sources = [f"ripple_source{index}" for index in range(len(clicks))]
        outputs = "".join(f"[{source}]" for source in ripple_sources)
        filters.append(f"[{ripple_input_label}]split={len(clicks)}{outputs}")

    for index, click in enumerate(clicks):
        x = int(round(float(click.x) - left - CLICK_RIPPLE_SIZE / 2.0))
        y = int(round(float(click.y) - top - CLICK_RIPPLE_SIZE / 2.0))
        start = max(0.0, float(click.timestamp) / 1000.0)
        ripple = f"ripple{index}"
        output = f"click{index}"
        filters.append(
            f"[{ripple_sources[index]}]format=rgba,"
            f"setpts=PTS-STARTPTS+{start:.6f}/TB[{ripple}]"
        )
        filters.append(
            f"[{current}][{ripple}]overlay=x={x}:y={y}:"
            f"eof_action=pass:repeatlast=0:shortest=0:format=auto[{output}]"
        )
        current = output
    return ";".join(filters), current


def _create_cursor_image(path: str) -> None:
    """Create the compact recording cursor from the bundled cyan asset."""
    source_path = get_resource_path(CURSOR_ASSET_PATH)
    with Image.open(source_path) as opened:
        source = opened.convert("RGBA")
    bounds = source.getchannel("A").getbbox()
    if bounds is None:
        raise ValueError("Recording cursor asset is empty")
    subject = source.crop(bounds)
    subject.thumbnail(
        (CURSOR_WIDTH - 4, CURSOR_HEIGHT - 4),
        Image.Resampling.LANCZOS,
    )
    image = Image.new("RGBA", (CURSOR_WIDTH, CURSOR_HEIGHT), (0, 0, 0, 0))
    image.alpha_composite(subject, (2, 2))
    image.save(path, format="PNG")


def _create_click_ripple(path: str, fps: float) -> None:
    """Create a short transparent APNG used for every automatic click pulse."""
    ripple_fps = max(24.0, min(float(fps or 30.0), 60.0))
    frame_count = max(12, int(round(CLICK_RIPPLE_DURATION_SECONDS * ripple_fps)))
    frame_duration_ms = max(16, int(round(1000.0 / ripple_fps)))
    render_scale = 4
    rendered_size = CLICK_RIPPLE_SIZE * render_scale
    center = rendered_size / 2.0
    frames: list[Image.Image] = []

    for index in range(frame_count):
        progress = index / max(frame_count - 1, 1)
        eased = 1.0 - pow(1.0 - progress, 3)
        radius = (7.0 + (CLICK_RIPPLE_SIZE / 2.0 - 4.0 - 7.0) * eased) * render_scale
        opacity = int(round(230.0 * pow(1.0 - progress, 1.45)))
        line_width = max(2, int(round((4.5 - 1.5 * progress) * render_scale)))
        rendered = Image.new("RGBA", (rendered_size, rendered_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(rendered)
        box = (
            center - radius,
            center - radius,
            center + radius,
            center + radius,
        )
        draw.ellipse(
            box,
            outline=(20, 92, 142, max(0, opacity // 3)),
            width=line_width + render_scale * 2,
        )
        draw.ellipse(
            box,
            outline=(38, 196, 244, opacity),
            width=line_width,
        )
        if progress < 0.3:
            dot_progress = progress / 0.3
            dot_radius = (5.5 - 2.5 * dot_progress) * render_scale
            dot_opacity = int(round(190.0 * (1.0 - dot_progress)))
            draw.ellipse(
                (
                    center - dot_radius,
                    center - dot_radius,
                    center + dot_radius,
                    center + dot_radius,
                ),
                fill=(38, 196, 244, dot_opacity),
            )
        frames.append(
            rendered.resize(
                (CLICK_RIPPLE_SIZE, CLICK_RIPPLE_SIZE),
                Image.Resampling.LANCZOS,
            )
        )

    frames[0].save(
        path,
        format="PNG",
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        disposal=1,
        blend=1,
    )


def _build_filter_graph(
    keyframes: list[ZoomKeyframe],
    mouse_track: list[MousePosition],
    click_events: list[ClickEvent],
    capture_rect: dict,
    fps: float,
    render_cursor: bool,
    render_clicks: bool,
    cursor_command_path: str = "",
) -> str:
    filters = ["[0:v]setpts=PTS-STARTPTS,format=rgba[base]"]
    current = "base"
    has_cursor = bool(render_cursor and mouse_track)
    has_clicks = bool(render_clicks and click_events)
    if has_clicks:
        ripple_input_index = 2 if has_cursor else 1
        click_chain, current = build_click_filter_chain(
            current,
            click_events,
            capture_rect,
            f"{ripple_input_index}:v",
        )
        if click_chain:
            filters.append(click_chain)
    if has_cursor:
        if not cursor_command_path:
            raise ValueError("Cursor rendering requires a command script")
        command_path = _escape_filter_path(cursor_command_path)
        filters.append(f"[{current}]sendcmd=f='{command_path}'[commanded]")
        filters.append(
            "[commanded][1:v]overlay@cursor=x=0:y=0:"
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
    click_ripple_path = ""
    cursor_command_path = ""
    filter_path = ""
    process: subprocess.Popen[str] | None = None
    last_progress = -1
    diagnostic_lines: deque[str] = deque(maxlen=16)
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
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".commands",
                delete=False,
            ) as handle:
                cursor_command_path = handle.name
                handle.write(build_cursor_command_script(mouse_track, capture_rect))
                handle.write("\n")
            command += ["-loop", "1", "-framerate", str(max(float(fps), 1.0)), "-i", cursor_path]
        if render_clicks and click_events:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
                click_ripple_path = handle.name
            _create_click_ripple(click_ripple_path, fps)
            command += ["-ignore_loop", "1", "-i", click_ripple_path]
        graph = _build_filter_graph(
            keyframes,
            mouse_track,
            click_events,
            capture_rect,
            fps,
            render_cursor,
            render_clicks,
            cursor_command_path,
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
                stripped = raw_line.strip()
                if stripped and name not in {
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
                    if len(stripped) > 900:
                        stripped = (
                            f"{stripped[:420]}...<truncated>...{stripped[-420:]}"
                        )
                    diagnostic_lines.append(stripped)
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
            signed_code = return_code if return_code < 2**31 else return_code - 2**32
            details = " | ".join(diagnostic_lines)[-4000:]
            message = f"FFmpeg exited with code {signed_code}"
            if details:
                message = f"{message}: {details}"
            return SmartZoomResult(
                state="failed",
                keyframes=keyframes,
                error=message,
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
        _remove_file(click_ripple_path)
        _remove_file(cursor_command_path)
        _remove_file(filter_path)
