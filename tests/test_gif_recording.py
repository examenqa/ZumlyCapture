from __future__ import annotations

from pathlib import Path
import subprocess

from PIL import Image
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from zumly.app import qt_tray
from zumly.app.qt_tray import QtZumlyCaptureTray
from zumly.app.utils import ffmpeg_exe
from zumly_capture.gif_export import GIF_FPS, GIF_MAX_EDGE, build_gif_filter, export_gif
from zumly_capture.preview_dialog import CapturePreviewDialog, _RecordingFormatWorker


def _animated_gif(path: Path, colors: tuple[str, str] = ("red", "blue")) -> None:
    frames = [Image.new("RGB", (160, 90), color) for color in colors]
    frames[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=80,
        loop=0,
    )


def test_gif_filter_is_palette_based_and_bounded() -> None:
    graph = build_gif_filter()

    assert f"fps={GIF_FPS}" in graph
    assert str(GIF_MAX_EDGE) in graph
    assert "palettegen" in graph
    assert "paletteuse" in graph
    assert graph.endswith("[outv]")


def test_gif_export_produces_a_looping_bounded_animation(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "capture.gif"
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
            "testsrc2=size=1440x810:rate=30:duration=0.7",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )

    progress: list[int] = []
    result = export_gif(source, output, 700, progress_callback=progress.append)

    assert result.state == "processed"
    assert result.output_path == str(output.resolve())
    assert progress[0] == 0
    assert progress[-1] == 100
    with Image.open(output) as animation:
        assert animation.is_animated
        assert animation.n_frames > 1
        assert max(animation.size) <= GIF_MAX_EDGE
        assert animation.info.get("loop") == 0
    assert sorted(path.name for path in tmp_path.iterdir()) == ["capture.gif", "source.mp4"]


def test_gif_export_honors_cancellation_before_start(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    output = tmp_path / "capture.gif"

    result = export_gif(source, output, 1000, cancel_callback=lambda: True)

    assert result.state == "cancelled"
    assert not output.exists()
    assert source.exists()


def test_gif_preview_uses_an_animation_and_can_remove_smart_zoom(tmp_path: Path) -> None:
    _app = QApplication.instance() or QApplication([])
    zoomed = tmp_path / "zoomed.gif"
    original_gif = tmp_path / "original.gif"
    original_video = tmp_path / "original.mp4"
    _animated_gif(zoomed, ("red", "blue"))
    _animated_gif(original_gif, ("green", "yellow"))
    original_bytes = original_gif.read_bytes()
    original_video.write_bytes(b"private unzoomed video")

    dialog = CapturePreviewDialog(str(zoomed), unzoomed_path=str(original_video))
    dialog._gif_original_path = str(original_gif)
    assert dialog._gif_movie is not None
    assert dialog._remove_zoom is not None
    dialog._remove_zoom.setChecked(True)
    dialog._save_video()

    assert zoomed.read_bytes() == original_bytes
    assert not original_gif.exists()
    assert not original_video.exists()
    assert dialog._saved is True
    dialog.close()
    dialog.deleteLater()


def test_recording_preview_defaults_save_as_to_the_settings_choice(tmp_path: Path) -> None:
    _app = QApplication.instance() or QApplication([])
    video = tmp_path / "capture.mp4"
    video.write_bytes(b"preview video")

    dialog = CapturePreviewDialog(
        str(video),
        preferred_output_format="gif",
    )

    assert dialog._format_combo is not None
    assert dialog._format_combo.currentData() == "gif"
    assert [
        dialog._format_combo.itemData(index)
        for index in range(dialog._format_combo.count())
    ] == ["mp4", "gif"]
    dialog.close()
    dialog.deleteLater()


def test_mp4_preview_routes_gif_choice_to_the_format_worker(tmp_path: Path) -> None:
    _app = QApplication.instance() or QApplication([])
    video = tmp_path / "capture.mp4"
    video.write_bytes(b"preview video")
    dialog = CapturePreviewDialog(str(video), preferred_output_format="gif")
    requested: list[tuple[str, str, str]] = []
    dialog._start_format_save = (  # type: ignore[method-assign]
        lambda source, output, output_format: requested.append(
            (source, output, output_format)
        )
    )

    dialog._save_video()

    assert requested == [(str(video.resolve()), str(tmp_path / "capture.gif"), "gif")]
    dialog.close()
    dialog.deleteLater()


def test_recording_format_worker_saves_preserved_mp4(tmp_path: Path) -> None:
    source = tmp_path / "private-source.mp4"
    output = tmp_path / "capture.mp4"
    source.write_bytes(b"processed MP4 with audio")
    outcomes: list[tuple[str, str]] = []
    worker = _RecordingFormatWorker(str(source), str(output), "mp4")
    worker.finished.connect(lambda path, error: outcomes.append((path, error)))

    worker.run()

    assert outcomes == [(str(output.resolve()), "")]
    assert output.read_bytes() == source.read_bytes()
    assert source.exists()


def test_gif_preview_can_save_as_mp4_from_its_private_source(tmp_path: Path) -> None:
    _app = QApplication.instance() or QApplication([])
    gif = tmp_path / "capture.gif"
    source = tmp_path / "private-format-source.mp4"
    _animated_gif(gif)
    source.write_bytes(b"processed MP4 with audio")
    dialog = CapturePreviewDialog(
        str(gif),
        format_source_path=str(source),
        preferred_output_format="mp4",
    )
    saved_paths: list[str] = []
    loop = QEventLoop()
    dialog.saved.connect(lambda path: (saved_paths.append(path), loop.quit()))

    dialog._save_video()
    QTimer.singleShot(5000, loop.quit)
    loop.exec()

    expected = tmp_path / "capture.mp4"
    assert saved_paths == [str(expected.resolve())]
    assert expected.read_bytes() == b"processed MP4 with audio"
    assert not gif.exists()
    assert not source.exists()
    assert dialog._saved is True
    dialog.close()
    dialog.deleteLater()


def test_tray_launches_gif_capture_with_preview_format_source(tmp_path: Path, monkeypatch) -> None:
    _app = QApplication.instance() or QApplication([])
    commands: list[list[str]] = []

    class FakeProcess:
        stdout = None

        def poll(self):
            return None

    class DeferredThread:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def start(self) -> None:
            pass

    settings = {
        "output_folder": str(tmp_path),
        "recording_format": "gif",
        "fps": 60,
        "monitor": 1,
        "microphone_device": "Microphone",
        "system_audio_device": "Loopback",
        "smart_zoom_enabled": False,
        "preview_after_capture": True,
    }
    monkeypatch.setattr(qt_tray, "load_settings", lambda: dict(settings))
    monkeypatch.setattr(
        qt_tray.subprocess,
        "Popen",
        lambda command, **_kwargs: commands.append(list(command)) or FakeProcess(),
    )
    monkeypatch.setattr(qt_tray.threading, "Thread", DeferredThread)
    tray = QtZumlyCaptureTray(_app)
    tray._pending_record_target = {"kind": "monitor", "monitorIndex": 1}

    tray._launch_recording()

    command = commands[0]
    output = command[command.index("--out") + 1]
    assert output.endswith(".gif")
    assert command[command.index("--output-format") + 1] == "gif"
    assert command[command.index("--microphone") + 1] == "Microphone"
    assert command[command.index("--system-audio") + 1] == "Loopback"
    assert "--preserve-format-source" in command
    tray.deleteLater()
