from pathlib import Path

import pytest

from zumly_capture.session import (
    CaptureSession,
    publish_recording,
)


def _session(media_path: Path) -> CaptureSession:
    return CaptureSession(
        session_id="session-123",
        media_path=str(media_path),
        capture_target={
            "kind": "monitor",
            "monitorIndex": 1,
            "left": 0,
            "top": 0,
            "width": 1920,
            "height": 1080,
        },
        started_at_unix_ms=1234.5,
        duration_ms=8000.25,
        paused_duration_ms=500.0,
        pause_boundaries=[{"timelineMs": 3500.0, "wallDurationMs": 500.0}],
        requested_fps=60,
        actual_fps=59.94,
        is_cfr=True,
        capture_backend="WGC",
        frame_timestamps=[0.0, 16.68],
        mouse_track=[{"x": 10, "y": 20, "timestamp": 0.0}],
        click_events=[{"x": 10, "y": 20, "timestamp": 4.0}],
        capture_telemetry={"framesWritten": 480},
    )


def test_publish_recording_creates_only_playable_media(tmp_path: Path) -> None:
    source = tmp_path / "engine-temp.mp4"
    source.write_bytes(b"mock-mp4-payload")
    output = tmp_path / "capture.mp4"

    result = publish_recording(source, output, _session(output))

    assert output.read_bytes() == b"mock-mp4-payload"
    assert not source.exists()
    assert result.media_path == str(output.resolve())
    assert result.warning == ""
    assert sorted(path.name for path in tmp_path.iterdir()) == ["capture.mp4"]


def test_publish_recording_never_overwrites_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "engine-temp.mp4"
    source.write_bytes(b"new")
    output = tmp_path / "capture.mp4"
    output.write_bytes(b"existing")

    with pytest.raises(FileExistsError):
        publish_recording(source, output, _session(output))

    assert output.read_bytes() == b"existing"
    assert source.read_bytes() == b"new"


def test_publish_recording_rejects_empty_engine_output(tmp_path: Path) -> None:
    source = tmp_path / "empty.mp4"
    source.touch()
    output = tmp_path / "capture.mp4"

    with pytest.raises(ValueError, match="usable video"):
        publish_recording(source, output, _session(output))

    assert not output.exists()
