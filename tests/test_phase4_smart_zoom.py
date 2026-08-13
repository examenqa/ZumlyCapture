from pathlib import Path
import json
import subprocess

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


def test_click_filter_chain_renders_every_click() -> None:
    clicks = [ClickEvent(x=100 + index, y=200, timestamp=index * 100) for index in range(9)]

    chain, output = build_click_filter_chain("base", clicks, CAPTURE_RECT)

    assert chain.count("drawbox=") == len(clicks)
    assert output == "click8"


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


def test_capture_manifest_records_smart_zoom_result(tmp_path: Path) -> None:
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

    manifest = session.to_dict()

    assert manifest["smartZoom"]["state"] == "processed"
    assert manifest["smartZoom"]["keyframes"] == [{"zoom": 1.5}]


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
    manifest = published.with_suffix(".zumly-capture.json")
    manifest.write_text(
        json.dumps(
            {
                "mediaPath": str(published),
                "smartZoom": {
                    "state": "processed",
                    "keyframes": [{"zoom": 1.5}, {"zoom": 1.0}],
                },
            }
        ),
        encoding="utf-8",
    )
    draft = preserve_unzoomed_recording(original, "session-123")

    warning = restore_unzoomed_recording(published, draft)

    assert warning == ""
    assert published.read_bytes() == b"unzoomed recording"
    assert not Path(draft).exists()
    updated = json.loads(manifest.read_text(encoding="utf-8"))
    assert updated["smartZoom"]["state"] == "removed"
    assert updated["smartZoom"]["keyframes"] == []


def test_renderer_produces_playable_video_and_reports_progress(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "processed.mp4"
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
        MousePosition(x=40 + index * 10, y=90, timestamp=index * 120)
        for index in range(12)
    ]
    progress: list[int] = []

    result = render_smart_zoom(
        str(source),
        str(output),
        mouse,
        [ClickEvent(x=180, y=90, timestamp=500)],
        rect,
        1500,
        30,
        render_cursor=True,
        render_clicks=True,
        progress_callback=progress.append,
    )

    assert result.state == "processed", result.error
    assert output.stat().st_size > 0
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
