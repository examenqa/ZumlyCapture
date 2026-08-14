from pathlib import Path
import json
import subprocess

from PIL import Image

from zumly.app.activity_analyzer import analyze_activity
from zumly.app.models import ClickEvent, MousePosition, ZoomKeyframe
from zumly.app.qt_tray import QtZumlyCaptureTray
from zumly.app.session_timing import RecordingState
from zumly.app.utils import ffmpeg_exe
from zumly.main import _read_control_payload
from zumly_capture import session as capture_session
from zumly_capture.session import (
    CaptureSession,
    preserve_unzoomed_recording,
    restore_unzoomed_recording,
)
from zumly_capture import smart_zoom
from zumly_capture.smart_zoom import (
    build_click_filter_chain,
    build_cursor_axis_expression,
    build_cursor_click_scale_expression,
    build_cursor_command_script,
    build_zoompan_filter,
    render_smart_zoom,
)


CAPTURE_RECT = {"left": 0, "top": 0, "width": 1280, "height": 720}


def _mouse_track(duration_ms: int = 12_000) -> list[MousePosition]:
    return [
        MousePosition(x=100 + index, y=100, timestamp=index * duration_ms / 11)
        for index in range(12)
    ]


def test_activity_analysis_keeps_more_than_four_clusters_in_one_pan_chain() -> None:
    clicks = [
        ClickEvent(
            x=80 if index % 2 == 0 else 1180,
            y=100 if index % 3 == 0 else 650,
            timestamp=1000 + index * 1500,
        )
        for index in range(6)
    ]

    keyframes = analyze_activity(
        _mouse_track(),
        CAPTURE_RECT,
        click_events=clicks,
        max_clusters=None,
    )

    assert sum(keyframe.reason.startswith("Pan to:") for keyframe in keyframes) == 5
    assert sum(keyframe.zoom == 1.0 for keyframe in keyframes) == 1


def test_cursor_expression_snaps_across_resume_boundary() -> None:
    samples = [
        MousePosition(x=10, y=20, timestamp=0),
        MousePosition(x=100, y=20, timestamp=1000, resume_boundary=True),
    ]

    expression = build_cursor_axis_expression(samples, "x", CAPTURE_RECT, 24)

    assert "if(lt(t,1.000000),8.000,98.000)" == expression
    assert "clip((t-" not in expression


def test_cursor_command_script_avoids_deeply_nested_expressions() -> None:
    samples = [
        MousePosition(
            x=(index * 13) % 1280,
            y=(index * 7) % 720,
            timestamp=index * 16,
            resume_boundary=index == 750,
        )
        for index in range(1500)
    ]

    script = build_cursor_command_script(samples, CAPTURE_RECT)

    assert "overlay@cursor" in script
    assert "[enter+expr]" in script
    assert "*TI" in script
    assert "if(" not in script
    assert "12.000000 overlay@cursor" in script


def test_cursor_click_scale_uses_original_zumly_press_and_release_timing() -> None:
    expression = build_cursor_click_scale_expression(
        [ClickEvent(x=100, y=200, timestamp=500)]
    )

    assert "between(t,0.500000,0.634000)" in expression
    assert "if(lt(t,0.534000),0.900000" in expression
    assert "clip((t-0.534000)/0.100000,0,1)" in expression
    assert "1-pow(" in expression


def test_cursor_filter_keeps_hotspot_fixed_while_pressing_forward() -> None:
    graph = smart_zoom._build_filter_graph(
        [ZoomKeyframe.create(0, 1.0)],
        [MousePosition(x=100, y=100, timestamp=0)],
        [ClickEvent(x=100, y=100, timestamp=500)],
        CAPTURE_RECT,
        30,
        True,
        True,
        "C:/Temp/cursor.commands",
    )

    assert "scale=w='max(1,iw*(" in graph
    assert "eval=frame" in graph
    assert f"pad={smart_zoom.CURSOR_WIDTH}:{smart_zoom.CURSOR_HEIGHT}:0:0" in graph
    assert "[commanded][animatedcursor]overlay@cursor" in graph


def test_effects_preserving_output_splits_before_smart_zoom() -> None:
    graph = smart_zoom._build_filter_graph(
        [ZoomKeyframe.create(0, 1.5)],
        [MousePosition(x=100, y=100, timestamp=0)],
        [ClickEvent(x=100, y=100, timestamp=500)],
        CAPTURE_RECT,
        30,
        True,
        True,
        "C:/Temp/cursor.commands",
        include_unzoomed_output=True,
    )

    assert "split=2[unzoomed_source][zoom_source]" in graph
    assert "[unzoomed_source]format=yuv420p[unzoomedv]" in graph
    assert "[zoom_source]zoompan=" in graph


def test_click_filter_chain_renders_every_click() -> None:
    clicks = [ClickEvent(x=100 + index, y=200, timestamp=index * 100) for index in range(9)]

    chain, output = build_click_filter_chain("base", clicks, CAPTURE_RECT)

    assert chain.count("overlay=x=") == len(clicks)
    assert chain.count("setpts=PTS-STARTPTS+") == len(clicks)
    assert "split=9" in chain
    assert "drawbox=" not in chain
    assert output == "click8"


def test_recording_cursor_uses_transparent_cyan_asset(tmp_path: Path) -> None:
    cursor = tmp_path / "cursor.png"

    smart_zoom._create_cursor_image(str(cursor))

    with Image.open(cursor) as image:
        rgba = image.convert("RGBA")
        assert rgba.size == (smart_zoom.CURSOR_WIDTH, smart_zoom.CURSOR_HEIGHT)
        assert rgba.getpixel((0, 0))[3] == 0
        visible = [
            rgba.getpixel((x, y))
            for y in range(rgba.height)
            for x in range(rgba.width)
            if rgba.getpixel((x, y))[3] > 200
        ]
        assert visible
        assert max(pixel[2] for pixel in visible) > 180
        assert max(pixel[1] for pixel in visible) > 140


def test_click_ripple_is_an_animated_transparent_circle(tmp_path: Path) -> None:
    ripple = tmp_path / "ripple.png"

    smart_zoom._create_click_ripple(str(ripple), 30)

    with Image.open(ripple) as image:
        assert image.size == (
            smart_zoom.CLICK_RIPPLE_SIZE,
            smart_zoom.CLICK_RIPPLE_SIZE,
        )
        assert image.n_frames >= 12
        image.seek(image.n_frames // 2)
        rgba = image.convert("RGBA")
        assert rgba.getpixel((0, 0))[3] == 0
        assert rgba.getchannel("A").getextrema()[1] > 0


def test_zoompan_filter_is_linear_and_preserves_dimensions() -> None:
    keyframes = [
        ZoomKeyframe.create(index * 1000, 1.5 if index % 2 == 0 else 1.0)
        for index in range(8)
    ]

    expression = build_zoompan_filter(keyframes, 60, 1919, 1079)
    doubled = build_zoompan_filter(keyframes + [
        ZoomKeyframe.create((index + 8) * 1000, 1.5 if index % 2 == 0 else 1.0)
        for index in range(8)
    ], 60, 1919, 1079)

    assert len(doubled) < len(expression) * 2.2
    assert "s=1918x1078" in expression


def test_cancel_before_render_preserves_source_and_cleans_output(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source remains recoverable")
    output = tmp_path / "processed.mp4"
    clicks = [ClickEvent(x=200, y=150, timestamp=1000)]

    result = render_smart_zoom(
        str(source),
        str(output),
        _mouse_track(),
        clicks,
        CAPTURE_RECT,
        12_000,
        60,
        cancel_callback=lambda: True,
    )

    assert result.state == "cancelled"
    assert source.read_bytes() == b"source remains recoverable"
    assert not output.exists()


def test_analysis_failure_preserves_source_and_returns_failed_result(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source remains recoverable")
    output = tmp_path / "processed.mp4"
    monkeypatch.setattr(
        smart_zoom,
        "analyze_activity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("analysis failed")),
    )

    result = render_smart_zoom(
        str(source),
        str(output),
        _mouse_track(),
        [],
        CAPTURE_RECT,
        12_000,
        60,
    )

    assert result.state == "failed"
    assert result.error == "analysis failed"
    assert source.read_bytes() == b"source remains recoverable"
    assert not output.exists()


def test_capture_control_channel_accepts_processing_cancel(tmp_path: Path) -> None:
    control = tmp_path / "control.json"
    control.write_text(json.dumps({"sequence": 4, "action": "cancel"}), encoding="utf-8")

    assert _read_control_payload(str(control), 3) == (4, "cancel")


def test_tray_toggle_becomes_processing_cancel_action() -> None:
    class RunningProcess:
        @staticmethod
        def poll() -> None:
            return None

    class ToggleAction:
        enabled = True

        def setEnabled(self, enabled: bool) -> None:
            self.enabled = enabled

    tray = QtZumlyCaptureTray(object())
    actions: list[str] = []
    titles: list[str] = []
    tray._state = RecordingState.PROCESSING
    tray._process = RunningProcess()
    tray._toggle_action = ToggleAction()
    tray._send_control = lambda action: actions.append(action) or True
    tray._update_tray = titles.append

    tray._on_toggle()

    assert actions == ["cancel"]
    assert titles == ["Cancelling Smart Zoom..."]
    assert tray._toggle_action.enabled is False
    tray.deleteLater()


def test_capture_session_records_smart_zoom_result(tmp_path: Path) -> None:
    media_path = tmp_path / "capture.mp4"
    session = CaptureSession(
        session_id="phase4",
        media_path=str(media_path),
        capture_target=CAPTURE_RECT,
        started_at_unix_ms=1,
        duration_ms=1000,
        paused_duration_ms=0,
        smart_zoom={"state": "processed", "keyframes": [{"zoom": 1.5}]},
    )

    capture_data = session.to_dict()

    assert capture_data["smartZoom"]["state"] == "processed"
    assert capture_data["smartZoom"]["keyframes"] == [{"zoom": 1.5}]


def test_automatic_smart_zoom_can_be_removed_as_one_effect(
    tmp_path: Path,
    monkeypatch,
) -> None:
    drafts = tmp_path / "private-drafts"
    monkeypatch.setattr(capture_session, "capture_drafts_directory", lambda: drafts)
    original = tmp_path / "original.mp4"
    original.write_bytes(b"unzoomed recording")
    published = tmp_path / "capture.mp4"
    published.write_bytes(b"automatic smart zoom recording")
    draft = preserve_unzoomed_recording(original, "session-123")

    warning = restore_unzoomed_recording(published, draft)

    assert warning == ""
    assert published.read_bytes() == b"unzoomed recording"
    assert not Path(draft).exists()
    assert not published.with_suffix(".zumly-capture.json").exists()


def test_renderer_produces_playable_video_and_reports_progress(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "processed.mp4"
    effects_only = tmp_path / "effects-only.mp4"
    subprocess.run(
        [
            ffmpeg_exe(),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=0x345678:s=320x180:r=30:d=1.5",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=1.5",
            "-shortest",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
    )
    rect = {"left": 0, "top": 0, "width": 320, "height": 180}
    mouse = [
        MousePosition(
            x=20 + (index * 7) % 280,
            y=20 + (index * 3) % 140,
            timestamp=index * 16,
        )
        for index in range(1500)
    ]
    progress: list[int] = []

    result = render_smart_zoom(
        str(source),
        str(output),
        mouse,
        [
            ClickEvent(x=120, y=60, timestamp=300),
            ClickEvent(x=180, y=90, timestamp=700),
            ClickEvent(x=260, y=120, timestamp=1100),
        ],
        rect,
        1500,
        30,
        render_cursor=True,
        render_clicks=True,
        unzoomed_output_path=str(effects_only),
        progress_callback=progress.append,
    )

    assert result.state == "processed", result.error
    assert output.stat().st_size > 0
    assert effects_only.stat().st_size > 0
    assert progress[0] == 0
    assert progress[-1] == 100
    subprocess.run(
        [
            ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(output),
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        ],
        check=True,
    )
    subprocess.run(
        [
            ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(effects_only),
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        ],
        check=True,
    )
