"""Tests for video_exporter geometry calculations.

Tests the GeometryComputer class and phase methods without requiring
Qt or actual video files.
"""

import os
import io
import re
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
from pathlib import Path

import pytest
import numpy as np
from PIL import Image
import app.video_exporter as video_exporter

from app.models import (
    ClickEvent,
    HighlightBox,
    MaskOverlayContent,
    MaskMode,
    MousePosition,
    OverlayGeometry,
    OverlayKind,
    OverlayStyle,
    OverlayShape,
    ShapeOverlayContent,
    TextOverlayContent,
    RecordingSession,
    TextAnnotation,
    TimelineFrame,
    ScreenTransition,
    TimelineOverlay,
    VideoSegment,
    ZoomKeyframe,
    CanvasLayoutScene,
    ExplainerScene,
)
from app.video_exporter import (
    GeometryComputer,
    VideoProbeResult,
    GeometryResult,
    ExportResult,
    VideoExporter,
    _RawSessionMediaMapper,
    _SessionMediaMapper,
    _build_zoompan_filter,
    _atempo_filters,
    _click_point_for_export,
    _ease_in_out,
    generate_timeline_frame_png,
    _map_zoomed_relative_point,
    _media_time_for_segment,
    _media_window_for_segment,
    _media_keyframes_for_segment,
    _local_clicks_for_segment,
    _local_mouse_track_for_segment,
    _media_cursor_points_for_segment,
    _generate_cursor_motion_track,
    _normalize_video_segments,
    _output_time_for_source_timestamp,
    _select_session_media_mapper,
    _split_segments_at_timeline_frames,
    _timed_overlay_filter,
    generate_cursor_png,
)
from app.frames import FramePreset, DEFAULT_FRAME, FRAME_PRESETS
from app.backgrounds import PRESETS as BACKGROUND_PRESETS
from app.geometry_math import LayoutSpaceTransform
from app.cursor_registry import get_cursor_preset
from app.explainer_scene import solve_explainer_layout


def test_static_video_masks_are_applied_before_zoom_and_transition_caches() -> None:
    mask = TimelineOverlay.create(
        OverlayKind.MASK,
        250.0,
        750.0,
        geometry=OverlayGeometry(0.1, 0.2, 0.3, 0.4),
        style=OverlayStyle(color=(0, 0, 0, 255)),
        content=MaskOverlayContent(MaskMode.SOLID),
    )
    transition = ScreenTransition.create(1000.0, duration_ms=400.0, enabled=True)
    plan = VideoExporter()._build_filtergraph(
        bg_preset=None,
        frame_preset=None,
        target_resolution=(320, 180),
        duration_ms=2000.0,
        frame_timestamps=None,
        keyframes=[],
        mouse_track=[],
        click_events=[],
        click_preset=None,
        monitor_rect={"left": 0, "top": 0, "width": 320, "height": 180},
        video_segments=[VideoSegment.create(0.0, 2000.0)],
        timeline_frames=None,
        highlights=None,
        timeline_overlays=[mask],
        screen_transitions=[transition],
        src_w=320,
        src_h=180,
        src_fps=30.0,
        total_sec=2.0,
        is_cfr=True,
        source_has_audio=False,
    )
    try:
        graph = plan.filtergraph
        token = "drawbox=x=32:y=36:w=96:h=72"
        assert token in graph
        assert graph.index(token) < graph.index("zoompan=")
        # Transition stills split from the masked/zoomed Video Space branch.
        assert "[s0mask0]" in graph
    finally:
        VideoExporter()._cleanup_temp_files(plan.temp_files, plan.temp_dirs)


def test_accepted_mask_precedes_explainer_frozen_frame_and_export_intermediates() -> None:
    mask = TimelineOverlay.create(
        OverlayKind.MASK,
        0.0,
        2_000.0,
        geometry=OverlayGeometry(0.0, 0.0, 0.5, 1.0),
        style=OverlayStyle(color=(0, 0, 0, 255)),
        content=MaskOverlayContent(MaskMode.SOLID),
    )
    scene = ExplainerScene.create(
        500.0,
        1_500.0,
        "right",
        CanvasLayoutScene.create(
            500.0, 1_500.0, video_scale=0.58, video_x=0.02, video_y=0.2
        ),
        TextAnnotation.create(500.0, 1_500.0, text="Private detail"),
    )
    exporter = VideoExporter()
    plan = exporter._build_filtergraph(
        bg_preset=None,
        frame_preset=DEFAULT_FRAME,
        target_resolution=(320, 180),
        duration_ms=2_000.0,
        frame_timestamps=None,
        keyframes=[],
        mouse_track=[],
        click_events=[],
        click_preset=None,
        monitor_rect={"left": 0, "top": 0, "width": 320, "height": 180},
        video_segments=[VideoSegment.create(0.0, 2_000.0)],
        timeline_frames=[],
        highlights=[],
        timeline_overlays=[mask],
        explainer_scenes=[scene],
        src_w=320,
        src_h=180,
        src_fps=30.0,
        total_sec=2.0,
        is_cfr=True,
        source_has_audio=False,
    )
    try:
        graph = plan.filtergraph
        mask_index = graph.index("drawbox=x=0:y=0:w=160:h=180")
        assert mask_index < graph.index("zoompan=")
        assert mask_index < graph.index("tpad=stop_mode=clone")
    finally:
        exporter._cleanup_temp_files(plan.temp_files, plan.temp_dirs)


def test_video_space_annotations_use_shared_png_assets_before_zoom() -> None:
    shape = TimelineOverlay.create(
        OverlayKind.SHAPE, 250.0, 750.0,
        geometry=OverlayGeometry(0.1, 0.1, 0.4, 0.3),
        content=ShapeOverlayContent(OverlayShape.ARROW),
    )
    text = TimelineOverlay.create(
        OverlayKind.TEXT, 500.0, 1_250.0,
        geometry=OverlayGeometry(0.2, 0.2, 0.5, 0.2),
        content=TextOverlayContent("Shared raster", "Segoe UI", 20),
    )
    plan = VideoExporter()._build_filtergraph(
        bg_preset=None, frame_preset=None, target_resolution=(320, 180),
        duration_ms=2_000.0, frame_timestamps=None, keyframes=[], mouse_track=[],
        click_events=[], click_preset=None,
        monitor_rect={"left": 0, "top": 0, "width": 320, "height": 180},
        video_segments=[VideoSegment.create(0.0, 2_000.0)], timeline_frames=None,
        highlights=None, timeline_overlays=[shape, text], src_w=320, src_h=180,
        src_fps=30.0, total_sec=2.0, is_cfr=True, source_has_audio=False,
    )
    try:
        assert len(plan.video_annotation_img_paths) == 2
        assert "[s0trim][2:v]overlay=" in plan.filtergraph
        assert plan.filtergraph.index("[s0trim][2:v]overlay=") < plan.filtergraph.index("zoompan=")
    finally:
        VideoExporter()._cleanup_temp_files(plan.temp_files, plan.temp_dirs)


def test_geometry_computer_applies_layout_space_to_presentation_group() -> None:
    geom = GeometryComputer(
        canvas_w=100,
        canvas_h=100,
        src_w=100,
        src_h=100,
        frame_preset=_no_frame_preset(),
        layout_transform=LayoutSpaceTransform(x=0.1, y=0.2, width=0.5, height=0.4),
    ).compute()

    assert geom["scr_x"] == 10
    assert geom["scr_y"] == 20
    assert geom["scr_w"] == 50
    assert geom["scr_h"] == 40


def test_continuous_cursor_track_maps_to_segment_local_video_space() -> None:
    segment = VideoSegment.create(1000.0, 2000.0, speed=2.0)
    track = [
        MousePosition(x=100.0, y=200.0, timestamp=1000.0),
        MousePosition(x=500.0, y=600.0, timestamp=1500.0),
    ]
    mapper = _RawSessionMediaMapper(30.0)
    points = _media_cursor_points_for_segment(
        track,
        segment,
        mapper,
        1.0,
        [],
        {"left": 0, "top": 0, "width": 1000, "height": 1000},
        1000,
        1000,
        500,
        500,
    )

    assert [round(row[0], 3) for row in points] == [0.0, 0.5]
    assert points[0][1:3] == (50.0, 100.0)
    assert points[1][1:3] == (250.0, 300.0)
    assert points[0][3] is None


def test_cursor_motion_track_uses_external_ffmpeg_commands(tmp_path, monkeypatch) -> None:
    command_path = tmp_path / "cursor-motion.cmd"
    monkeypatch.setattr(video_exporter, "_new_temp_asset_path", lambda _suffix: str(command_path))

    track = _generate_cursor_motion_track(
        points=[(0.0, 12.0, 8.0, False), (0.5, 24.0, 16.0, True)],
        anchor_x=5.0,
        anchor_y=3.0,
        segment_index=2,
        node_index=0,
    )

    assert track.command_path == str(command_path)
    assert (track.initial_x, track.initial_y) == (7, 5)
    commands = command_path.read_text(encoding="utf-8")
    assert "overlay@cursor2 x 7" in commands
    assert "overlay@cursor2 y 13" in commands
    assert "cursor_%05d.png" not in commands


def test_ffmpeg_accepts_cursor_sendcmd_sidecar(tmp_path) -> None:
    """The compact cursor path must be accepted by the bundled FFmpeg build."""
    ffmpeg = video_exporter._ffmpeg_exe()
    if not ffmpeg or not os.path.isfile(ffmpeg):
        pytest.skip("FFmpeg is unavailable")

    cursor = tmp_path / "cursor.png"
    Image.new("RGBA", (12, 12), (255, 255, 255, 255)).save(cursor)
    command_file = tmp_path / "cursor.cmd"
    command_file.write_text(
        "0.000000 overlay@cursor0 x 4, overlay@cursor0 y 4;\n"
        "0.500000 overlay@cursor0 x 28, overlay@cursor0 y 20;",
        encoding="utf-8",
    )
    graph = tmp_path / "cursor-graph.txt"
    graph.write_text(
        "[0:v]sendcmd=f='"
        + video_exporter._ffmpeg_filter_path(str(command_file))
        + "'[main];[1:v]format=rgba[cursor];"
        "[main][cursor]overlay@cursor0=x=4:y=4:eval=init:shortest=1[out]",
        encoding="utf-8",
    )
    output = tmp_path / "cursor.mp4"
    result = subprocess.run(
        [
            ffmpeg, "-y", "-f", "lavfi", "-i", "color=c=black:s=64x48:r=30:d=1",
            "-loop", "1", "-i", str(cursor), "-filter_complex_script", str(graph),
            "-map", "[out]", "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert output.exists() and output.stat().st_size > 0


def test_ffmpeg_nonzero_exit_returns_structured_failure(tmp_path, monkeypatch) -> None:
    class EmptyStream:
        def readline(self):
            return ""

    class FailedProcess:
        returncode = 17
        stderr = EmptyStream()

        def wait(self):
            return self.returncode

    monkeypatch.setattr(video_exporter.subprocess, "Popen", lambda *a, **k: FailedProcess())
    result = VideoExporter()._execute_ffmpeg_command(
        ["ffmpeg", "-version"],
        str(tmp_path / "failed.mp4"),
        1.0,
    )

    assert isinstance(result, ExportResult)
    assert result.success is False
    assert result.ffmpeg_exit_code == 17
    assert "exit code 17" in result.error_message


def test_hardware_export_retries_with_libx264(tmp_path, monkeypatch) -> None:
    """A hardware init failure retries the prepared graph in software."""
    exporter = VideoExporter()
    output = tmp_path / "fallback.mp4"
    plan = SimpleNamespace(
        filtergraph="[0:v]null[out]",
        temp_files=[],
        voiceover_audio_paths=[],
        has_voiceover_audio=False,
        has_speed_changes=False,
        has_timeline_edits=False,
        output_total_sec=1.0,
    )
    calls = []

    monkeypatch.setattr(
        exporter,
        "_probe_source",
        lambda *args, **kwargs: video_exporter.ExportSourceProbe(320, 180, 30.0, 1.0),
    )
    monkeypatch.setattr(exporter, "_build_filtergraph", lambda **kwargs: plan)
    monkeypatch.setattr(exporter, "_write_filtergraph_script", lambda graph, files: "graph.txt")
    monkeypatch.setattr(
        exporter,
        "_build_ffmpeg_command",
        lambda **kwargs: ["ffmpeg", kwargs["encoder_id"]],
    )

    def execute(_cmd, output_path, _duration, encoder_id=""):
        calls.append(encoder_id)
        if encoder_id == "h264_nvenc":
            return ExportResult(
                success=False,
                output_path=output_path,
                error_message="NVENC initialization failed",
                ffmpeg_exit_code=1,
                encoder_id=encoder_id,
            )
        with open(output_path, "wb") as handle:
            handle.write(b"software output")
        return ExportResult(success=True, output_path=output_path, ffmpeg_exit_code=0, encoder_id=encoder_id)

    monkeypatch.setattr(exporter, "_execute_ffmpeg_command", execute)
    result = exporter._run(
        input_path=str(tmp_path / "source.mp4"),
        output_path=str(output),
        bg_preset=None,
        frame_preset=None,
        target_resolution=(320, 180),
        duration_ms=1000.0,
        frame_timestamps=None,
        is_cfr=True,
        keyframes=[],
        mouse_track=[],
        click_events=[],
        click_preset=None,
        actual_fps=30.0,
        monitor_rect={},
        video_segments=None,
        timeline_frames=None,
        highlights=None,
        encoder_id="h264_nvenc",
        voiceover_segments=None,
    )

    assert calls == ["h264_nvenc", "libx264"]
    assert result.success is True
    assert result.fallback_used is True
    assert result.requested_encoder_id == "h264_nvenc"
    assert result.encoder_id == "libx264"
    assert "fallback succeeded" in result.error_message.lower()
    assert output.read_bytes() == b"software output"


def test_hardware_export_does_not_retry_graph_memory_failure(
    tmp_path, monkeypatch
) -> None:
    """A filtergraph OOM is encoder-independent and must not run twice."""
    exporter = VideoExporter()
    output = tmp_path / "oom.mp4"
    plan = SimpleNamespace(
        filtergraph="[0:v]null[out]",
        temp_files=[],
        temp_dirs=[],
        voiceover_audio_paths=[],
        has_voiceover_audio=False,
        has_speed_changes=False,
        has_timeline_edits=False,
        output_total_sec=1.0,
    )
    calls = []

    monkeypatch.setattr(
        exporter,
        "_probe_source",
        lambda *args, **kwargs: video_exporter.ExportSourceProbe(320, 180, 30.0, 1.0),
    )
    monkeypatch.setattr(exporter, "_build_filtergraph", lambda **kwargs: plan)
    monkeypatch.setattr(exporter, "_write_filtergraph_script", lambda graph, files: "graph.txt")
    monkeypatch.setattr(
        exporter,
        "_build_ffmpeg_command",
        lambda **kwargs: ["ffmpeg", kwargs["encoder_id"]],
    )

    def execute(_cmd, output_path, _duration, encoder_id=""):
        calls.append(encoder_id)
        return ExportResult(
            success=False,
            output_path=output_path,
            error_message="Error while filtering: Cannot allocate memory",
            ffmpeg_exit_code=-12,
            encoder_id=encoder_id,
        )

    monkeypatch.setattr(exporter, "_execute_ffmpeg_command", execute)
    result = exporter._run(
        input_path=str(tmp_path / "source.mp4"),
        output_path=str(output),
        bg_preset=None,
        frame_preset=None,
        target_resolution=(320, 180),
        duration_ms=1000.0,
        frame_timestamps=None,
        is_cfr=True,
        keyframes=[],
        mouse_track=[],
        click_events=[],
        click_preset=None,
        actual_fps=30.0,
        monitor_rect={},
        video_segments=None,
        timeline_frames=None,
        highlights=None,
        encoder_id="h264_nvenc",
        voiceover_segments=None,
    )

    assert calls == ["h264_nvenc"]
    assert result.success is False
    assert result.fallback_used is False
    assert "cannot allocate memory" in result.error_message.lower()


def test_hardware_export_does_not_retry_filtergraph_binding_failure(
    tmp_path, monkeypatch
) -> None:
    """Source metadata mentioning NVENC must not misclassify a graph error."""
    exporter = VideoExporter()
    output = tmp_path / "unconnected.mp4"
    plan = SimpleNamespace(
        filtergraph="[0:v]null[out]",
        temp_files=[],
        temp_dirs=[],
        voiceover_audio_paths=[],
        has_voiceover_audio=False,
        has_speed_changes=False,
        has_timeline_edits=False,
        output_total_sec=1.0,
    )
    calls = []
    monkeypatch.setattr(
        exporter,
        "_probe_source",
        lambda *args, **kwargs: video_exporter.ExportSourceProbe(320, 180, 30.0, 1.0),
    )
    monkeypatch.setattr(exporter, "_build_filtergraph", lambda **kwargs: plan)
    monkeypatch.setattr(exporter, "_write_filtergraph_script", lambda graph, files: "graph.txt")
    monkeypatch.setattr(
        exporter,
        "_build_ffmpeg_command",
        lambda **kwargs: ["ffmpeg", kwargs["encoder_id"]],
    )

    def execute(_cmd, output_path, _duration, encoder_id=""):
        calls.append(encoder_id)
        return ExportResult(
            success=False,
            output_path=output_path,
            error_message=(
                "encoder: Lavc h264_nvenc\n"
                "Filter sendcmd has an unconnected output\n"
                "Error binding filtergraph inputs/outputs: Invalid argument"
            ),
            ffmpeg_exit_code=-22,
            encoder_id=encoder_id,
        )

    monkeypatch.setattr(exporter, "_execute_ffmpeg_command", execute)
    result = exporter._run(
        input_path=str(tmp_path / "source.mp4"),
        output_path=str(output),
        bg_preset=None,
        frame_preset=None,
        target_resolution=(320, 180),
        duration_ms=1000.0,
        frame_timestamps=None,
        is_cfr=True,
        keyframes=[],
        mouse_track=[],
        click_events=[],
        click_preset=None,
        actual_fps=30.0,
        monitor_rect={},
        video_segments=None,
        timeline_frames=None,
        highlights=None,
        encoder_id="h264_nvenc",
        voiceover_segments=None,
    )

    assert calls == ["h264_nvenc"]
    assert result.success is False
    assert result.fallback_used is False


def test_failed_export_preserves_existing_destination(tmp_path, monkeypatch) -> None:
    exporter = VideoExporter()
    output = tmp_path / "existing.mp4"
    output.write_bytes(b"previous successful export")
    plan = SimpleNamespace(
        filtergraph="[0:v]null[out]",
        temp_files=[],
        temp_dirs=[],
        voiceover_audio_paths=[],
        output_total_sec=1.0,
    )
    monkeypatch.setattr(
        exporter,
        "_probe_source",
        lambda *args, **kwargs: video_exporter.ExportSourceProbe(320, 180, 30.0, 1.0),
    )
    monkeypatch.setattr(exporter, "_build_filtergraph", lambda **kwargs: plan)
    monkeypatch.setattr(exporter, "_write_filtergraph_script", lambda graph, files: "graph.txt")
    monkeypatch.setattr(exporter, "_build_ffmpeg_command", lambda **kwargs: ["ffmpeg"])
    monkeypatch.setattr(
        exporter,
        "_execute_ffmpeg_command",
        lambda _cmd, path, _duration, encoder_id="": ExportResult(
            success=False,
            output_path=path,
            error_message="render failed",
            ffmpeg_exit_code=9,
            encoder_id=encoder_id,
        ),
    )

    result = exporter._run(
        input_path=str(tmp_path / "source.mp4"),
        output_path=str(output),
        bg_preset=None,
        frame_preset=None,
        target_resolution=(320, 180),
        duration_ms=1000.0,
        frame_timestamps=None,
        is_cfr=True,
        keyframes=[],
        mouse_track=[],
        click_events=[],
        click_preset=None,
        actual_fps=30.0,
        monitor_rect={},
        video_segments=None,
        timeline_frames=None,
        highlights=None,
        encoder_id="libx264",
        voiceover_segments=None,
    )

    assert result.success is False
    assert output.read_bytes() == b"previous successful export"
    assert not (tmp_path / "existing.mp4.tmp").exists()


def test_ffmpeg_command_uses_encoder_profile_args() -> None:
    plan = SimpleNamespace(
        frame_img_path="frame.png",
        click_img_path=None,
        cursor_img_path=None,
        highlight_img_paths=[],
        timeline_frame_img_paths=[],
        voiceover_audio_paths=[],
        has_voiceover_audio=False,
        has_speed_changes=False,
        has_timeline_edits=False,
        output_total_sec=0.0,
    )
    exporter = VideoExporter()
    for encoder_id in ("h264_nvenc", "h264_qsv", "h264_amf", "libx264"):
        command = exporter._build_ffmpeg_command(
            ffmpeg="ffmpeg",
            input_path="input.mp4",
            output_path="output.mp4",
            graph_path="graph.txt",
            plan=plan,
            encoder_id=encoder_id,
        )
        assert command[command.index("-c:v") + 1] == encoder_id
        assert "-preset" in command or encoder_id == "h264_amf"


def test_ffmpeg_command_bounds_shared_static_image_inputs() -> None:
    plan = SimpleNamespace(
        frame_img_path="frame.png",
        click_img_path="click.png",
        cursor_img_path="cursor.png",
        highlight_img_paths=[],
        text_annotation_img_paths=[],
        timeline_frame_img_paths=[],
        background_img_path="background.png",
        transition_text_img_paths=[],
        voiceover_audio_paths=[],
        has_source_audio=False,
        has_audio_output=False,
        output_total_sec=10.0,
    )
    command = VideoExporter()._build_ffmpeg_command(
        ffmpeg="ffmpeg",
        input_path="input.mp4",
        output_path="output.mp4",
        graph_path="graph.txt",
        plan=plan,
        encoder_id="libx264",
    )

    for path in ("frame.png", "click.png", "cursor.png", "background.png"):
        path_index = command.index(path)
        assert command[path_index - 5 : path_index] == [
            "-loop",
            "1",
            "-t",
            "0.100",
            "-i",
        ]


def test_ffmpeg_command_maps_generated_audio_without_shortest() -> None:
    """Audio-enabled plans must map the concat output rather than source audio."""
    plan = SimpleNamespace(
        frame_img_path="frame.png",
        click_img_path=None,
        cursor_img_path=None,
        highlight_img_paths=[],
        text_annotation_img_paths=[],
        timeline_frame_img_paths=[],
        background_img_path="background.png",
        cursor_motion_tracks=[],
        voiceover_audio_paths=[],
        has_voiceover_audio=False,
        has_speed_changes=True,
        has_timeline_edits=True,
        has_source_audio=True,
        has_audio_output=True,
        audio_output_node="sourceaout",
        output_total_sec=4.5,
    )

    command = VideoExporter()._build_ffmpeg_command(
        ffmpeg="ffmpeg",
        input_path="input.mp4",
        output_path="output.mp4",
        graph_path="graph.txt",
        plan=plan,
        encoder_id="libx264",
    )

    assert ["-map", "[out]"] == command[command.index("-map") : command.index("-map") + 2]
    audio_map_index = command.index("[sourceaout]")
    assert command[audio_map_index - 1] == "-map"
    assert command[command.index("-c:a") + 1] == "aac"
    assert command[command.index("-b:a") + 1] == "192k"
    assert "-shortest" not in command


@pytest.fixture
def synthetic_cut_retime_highlight_frame_session() -> RecordingSession:
    """Session with a cut gap, a 4x segment, one highlight, and one inserted frame."""
    return RecordingSession(
        id="synthetic-graph-session",
        start_time=0.0,
        duration=9000.0,
        mouse_track=[
            MousePosition(x=100.0, y=100.0, timestamp=0.0),
            MousePosition(x=900.0, y=500.0, timestamp=6000.0),
        ],
        keyframes=[
            ZoomKeyframe.create(timestamp=5500.0, zoom=1.5, x=0.45, y=0.45, duration=250.0),
            ZoomKeyframe.create(timestamp=7500.0, zoom=1.0, x=0.5, y=0.5, duration=250.0),
        ],
        click_events=[],
        video_segments=[
            VideoSegment(id="seg-a", start_ms=0.0, end_ms=2000.0, speed=1.0),
            VideoSegment(id="seg-b", start_ms=5000.0, end_ms=9000.0, speed=4.0),
        ],
        timeline_frames=[
            TimelineFrame(
                id="frame-a",
                timestamp_ms=6500.0,
                duration_ms=1500.0,
                kind="text",
                text="Checkpoint",
            )
        ],
        highlights=[
            HighlightBox(
                id="hl-a",
                start_ms=6000.0,
                end_ms=7000.0,
                x=0.2,
                y=0.25,
                width=0.3,
                height=0.2,
                shape="rect",
            )
        ],
    )


# Helper to create a "No Frame" preset
def _no_frame_preset() -> FramePreset:
    """Create a No Frame preset for testing."""
    return FramePreset(
        name="None",
        bezel_width=0,
        outer_radius=0,
        inner_radius=0,
        bezel_color=(0, 0, 0),
        edge_color=(0, 0, 0),
        edge_width=0,
        show_camera=False,
        shadow_layers=0,
        padding=0.0,
    )


class TestGeometryComputer:
    """Test the GeometryComputer pure-logic class."""

    def test_no_frame_mode_landscape(self):
        """No-frame mode with landscape video should center video on canvas."""
        # 1920x1080 video on 1920x1080 canvas
        gc = GeometryComputer(
            canvas_w=1920,
            canvas_h=1080,
            src_w=1920,
            src_h=1080,
            frame_preset=_no_frame_preset(),
        )
        geom = gc.compute()

        assert geom["scr_w"] == 1920
        assert geom["scr_h"] == 1080
        assert geom["scr_x"] == 0
        assert geom["scr_y"] == 0
        # No device keys in no-frame mode
        assert "dev_x" not in geom

    def test_no_frame_mode_portrait_video_landscape_canvas(self):
        """Portrait video on landscape canvas should be centered with pillarbox."""
        gc = GeometryComputer(
            canvas_w=1920,
            canvas_h=1080,
            src_w=1080,
            src_h=1920,
            frame_preset=_no_frame_preset(),
        )
        geom = gc.compute()

        # Video should be scaled to fit height, width constrained by aspect ratio
        assert geom["scr_h"] == 1080
        # 1080 * (1080/1920) = 607.5 → 607
        assert geom["scr_w"] == 607
        # Centered horizontally
        assert geom["scr_x"] == (1920 - 607) // 2
        assert geom["scr_y"] == 0

    def test_with_frame_standard_bezel(self):
        """Standard device frame with bezel should compute correct geometry."""
        # Use the default frame preset (Laptop)
        gc = GeometryComputer(
            canvas_w=1920,
            canvas_h=1080,
            src_w=1600,
            src_h=900,
            frame_preset=DEFAULT_FRAME,
        )
        geom = gc.compute()

        # Should have device keys
        assert "dev_x" in geom
        assert "dev_y" in geom
        assert "dev_w" in geom
        assert "dev_h" in geom
        assert "bw" in geom
        assert "outer_r" in geom
        assert "inner_r" in geom
        assert "edge_thickness" in geom

        # Screen should be smaller than device (due to bezel)
        assert geom["scr_w"] < geom["dev_w"]
        assert geom["scr_h"] < geom["dev_h"]

        # Bezel width should be positive
        assert geom["bw"] > 0

        # Screen should be inside device bounds
        assert geom["scr_x"] >= geom["dev_x"]
        assert geom["scr_y"] >= geom["dev_y"]
        assert geom["scr_x"] + geom["scr_w"] <= geom["dev_x"] + geom["dev_w"]
        assert geom["scr_y"] + geom["scr_h"] <= geom["dev_y"] + geom["dev_h"]

    def test_with_frame_aspect_ratio_preserved(self):
        """Device frame should preserve video aspect ratio."""
        video_aspect = 16 / 9
        gc = GeometryComputer(
            canvas_w=1920,
            canvas_h=1080,
            src_w=1600,
            src_h=900,
            frame_preset=DEFAULT_FRAME,
        )
        geom = gc.compute()

        # Screen area should maintain video aspect ratio
        screen_aspect = geom["scr_w"] / max(geom["scr_h"], 1)
        assert abs(screen_aspect - video_aspect) < 0.01  # Within 1% tolerance

    def test_small_canvas_no_frame(self):
        """Small canvas should still compute valid geometry."""
        gc = GeometryComputer(
            canvas_w=640,
            canvas_h=480,
            src_w=1920,
            src_h=1080,
            frame_preset=_no_frame_preset(),
        )
        geom = gc.compute()

        # Video should fit within canvas
        assert geom["scr_w"] <= 640
        assert geom["scr_h"] <= 480
        assert geom["scr_x"] >= 0
        assert geom["scr_y"] >= 0

    def test_zero_bezel_width_frame(self):
        """Frame with zero bezel width should still compute correctly."""
        # Minimal frame (shadow-only, no bezel)
        gc = GeometryComputer(
            canvas_w=1920,
            canvas_h=1080,
            src_w=1600,
            src_h=900,
            frame_preset=FramePreset(
                name="Shadow Only",
                bezel_width=0,
                outer_radius=20,
                inner_radius=8,
                bezel_color=(0, 0, 0),
                edge_color=(0, 0, 0),
                edge_width=0,
                show_camera=False,
                shadow_layers=4,
                padding=0.08,
            ),
        )
        geom = gc.compute()

        # Should have device keys even with zero bezel
        assert "dev_x" in geom
        assert geom["bw"] == 0
        # Screen and device dimensions should be very close (only padding matters)
        assert abs(geom["scr_w"] - geom["dev_w"]) < 5

    def test_high_padding_frame(self):
        """High padding should leave space around device."""
        gc = GeometryComputer(
            canvas_w=1920,
            canvas_h=1080,
            src_w=1600,
            src_h=900,
            frame_preset=FramePreset(
                name="Padded",
                bezel_width=40,
                outer_radius=20,
                inner_radius=8,
                bezel_color=(26, 26, 26),
                edge_color=(107, 107, 107),
                edge_width=2,
                show_camera=False,
                shadow_layers=4,
                padding=0.15,  # High padding
            ),
        )
        geom = gc.compute()

        # Device should be smaller than canvas due to padding
        padding_px = 1920 * 0.15
        assert geom["dev_x"] >= padding_px * 0.9  # Allow 10% tolerance
        assert geom["dev_y"] >= 1080 * 0.15 * 0.9


class TestVideoProbeResult:
    """Test the VideoProbeResult dataclass."""

    def test_dataclass_construction(self):
        """VideoProbeResult should construct with all fields."""
        result = VideoProbeResult(
            src_fps=30.0,
            total_frames=900,
            src_w=1920,
            src_h=1080,
            out_w=1920,
            out_h=1080,
            fps=30.0,
            is_gif=False,
        )
        assert result.src_fps == 30.0
        assert result.total_frames == 900
        assert result.src_w == 1920
        assert result.src_h == 1080
        assert result.out_w == 1920
        assert result.out_h == 1080
        assert result.fps == 30.0
        assert result.is_gif is False


def test_probe_source_prefers_encoded_fps_over_measured_capture_fps(monkeypatch) -> None:
    """Dropped-frame metadata must not lower zoompan/export cadence."""
    import app.video_exporter as video_exporter

    class _ProbeResult:
        stderr = (
            "Duration: 00:00:28.02, start: 0.000000, bitrate: 1000 kb/s\n"
            "Stream #0:0: Video: h264, yuv420p, 1920x1080, 60 fps, 60 tbr"
        )

    monkeypatch.setattr(video_exporter.subprocess, "run", lambda *args, **kwargs: _ProbeResult())

    probe = VideoExporter()._probe_source(
        ffmpeg="ffmpeg",
        input_path="capture.mp4",
        actual_fps=33.9,
        duration_ms=49597.0,
    )

    assert probe.src_fps == pytest.approx(60.0)
    assert probe.total_sec == pytest.approx(28.02)
    assert probe.has_audio is False


def test_probe_source_detects_audio_stream(monkeypatch) -> None:
    """A source audio stream enables parallel audio graph construction."""

    class _ProbeResult:
        stderr = (
            "Duration: 00:00:01.00, start: 0.000000, bitrate: 1000 kb/s\n"
            "Stream #0:0: Video: h264, yuv420p, 1920x1080, 30 fps, 30 tbr\n"
            "Stream #0:1: Audio: aac, 48000 Hz, stereo, fltp"
        )

    monkeypatch.setattr(video_exporter.subprocess, "run", lambda *args, **kwargs: _ProbeResult())

    probe = VideoExporter()._probe_source(
        ffmpeg="ffmpeg",
        input_path="capture-with-audio.mp4",
        actual_fps=30.0,
        duration_ms=1000.0,
    )

    assert probe.has_audio is True


def test_atempo_filters_chain_out_of_range_speed_changes() -> None:
    """FFmpeg accepts 4x and quarter-speed retimes only as chained stages."""
    assert _atempo_filters(4.0) == ["atempo=2.000000", "atempo=2.000000"]
    assert _atempo_filters(0.25) == ["atempo=0.500000", "atempo=0.500000"]
    assert _atempo_filters(0.5) == ["atempo=0.500000"]
    assert _atempo_filters(0.75) == ["atempo=0.750000"]


@pytest.mark.parametrize(
    ("speed", "setpts", "atempo", "duration"),
    [
        (0.5, "setpts=2.00000000*PTS", "atempo=0.500000", 4.0),
        (0.75, "setpts=1.33333333*PTS", "atempo=0.750000", 8.0 / 3.0),
    ],
)
def test_slow_motion_retimes_video_audio_and_output_timeline(
    speed: float,
    setpts: str,
    atempo: str,
    duration: float,
) -> None:
    segment = VideoSegment.create(0.0, 2000.0, speed=speed)
    plan = VideoExporter()._build_filtergraph(
        bg_preset=None,
        frame_preset=None,
        target_resolution=(320, 180),
        duration_ms=2000.0,
        frame_timestamps=None,
        keyframes=[],
        mouse_track=[MousePosition(x=100.0, y=100.0, timestamp=1000.0)],
        click_events=[],
        click_preset=None,
        monitor_rect={"left": 0, "top": 0, "width": 320, "height": 180},
        video_segments=[segment],
        timeline_frames=None,
        highlights=None,
        src_w=320,
        src_h=180,
        src_fps=30.0,
        total_sec=2.0,
        is_cfr=True,
        source_has_audio=True,
    )
    try:
        assert setpts in plan.filtergraph
        assert atempo in plan.filtergraph
        assert plan.output_total_sec == pytest.approx(duration)
        assert _output_time_for_source_timestamp(1000.0, [segment], None) == pytest.approx(
            1.0 / speed
        )
    finally:
        VideoExporter()._cleanup_temp_files(plan.temp_files, plan.temp_dirs)


def test_filtergraph_builder_receives_explicit_cfr_state(monkeypatch) -> None:
    """CFR timing must flow from the export request, never hidden instance state."""
    captured: dict[str, object] = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(video_exporter, "_build_export_filtergraph", fake_build)
    exporter = VideoExporter()
    plan = exporter._build_filtergraph(
        bg_preset=None,
        frame_preset=None,
        target_resolution=None,
        duration_ms=1000.0,
        frame_timestamps=[],
        keyframes=[],
        mouse_track=[],
        click_events=[],
        click_preset=None,
        monitor_rect=None,
        video_segments=None,
        timeline_frames=None,
        highlights=None,
        src_w=1280,
        src_h=720,
        src_fps=30.0,
        total_sec=1.0,
        is_cfr=True,
    )

    assert plan is not None
    assert captured["is_cfr"] is True


def test_screen_transition_inserts_video_and_matching_silence() -> None:
    transition = ScreenTransition.create(
        1000.0,
        effect_type="directional_push",
        duration_ms=400.0,
        enabled=True,
    )
    plan = VideoExporter()._build_filtergraph(
        bg_preset=None,
        frame_preset=None,
        target_resolution=(320, 180),
        duration_ms=2000.0,
        frame_timestamps=None,
        keyframes=[],
        mouse_track=[],
        click_events=[],
        click_preset=None,
        monitor_rect={"left": 0, "top": 0, "width": 320, "height": 180},
        video_segments=[VideoSegment.create(0.0, 2000.0)],
        timeline_frames=None,
        highlights=None,
        screen_transitions=[transition],
        src_w=320,
        src_h=180,
        src_fps=30.0,
        total_sec=2.0,
        is_cfr=True,
        source_has_audio=True,
    )
    try:
        graph = plan.filtergraph
        # Transition endpoints are the internal Video Space dimensions, not
        # the 320x180 Presentation Shell.
        assert "scale=262:148:flags=lanczos,format=rgba[st0oldstill]" in graph
        assert "zoompan=z=1:x=0:y=0:d=13:s=262x148:fps=30" not in graph
        assert "[st0new]scale=w='max(2,trunc(iw*" in graph
        assert "[st0old]scale=w='max(2,trunc(iw*" in graph
        assert "[st0background][st0incomingscaled]overlay=" in graph
        assert "[st0stage0][st0outgoingscaled]overlay=" in graph
        assert ",geq=" not in graph
        assert "anchor" not in graph
        assert "clip(t/0.400000,0,1)" in graph
        assert "[0:v]trim=start=0.966667" not in graph
        assert "[s0transitionvideo0]trim=" in graph
        assert "[s1transitionvideo0]trim=" in graph
        assert graph.index("[st0video]") < graph.index("[st0framed]")
        assert "anullsrc=r=48000:cl=stereo,atrim=duration=0.400000" in graph
        assert "concat=n=3:v=1:a=1[out][sourceaout]" in graph
        assert plan.output_total_sec == pytest.approx(2.4)
    finally:
        for path in plan.temp_files:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass


def test_graphic_transitions_after_cut_retime_and_same_boundary_share_output_clock() -> None:
    transitions = [
        ScreenTransition.create(
            3500.0,
            effect_type="graphic_sweep",
            direction="right",
            duration_ms=700.0,
        ),
        ScreenTransition.create(
            3500.0,
            effect_type="graphic_fold",
            direction="left",
            duration_ms=900.0,
        ),
    ]
    exporter = VideoExporter()
    plan = exporter._build_filtergraph(
        bg_preset=None,
        frame_preset=None,
        target_resolution=(320, 180),
        duration_ms=6000.0,
        frame_timestamps=None,
        keyframes=[],
        mouse_track=[],
        click_events=[],
        click_preset=None,
        monitor_rect={"left": 0, "top": 0, "width": 320, "height": 180},
        video_segments=[
            VideoSegment.create(0.0, 2000.0, speed=2.0, sequence_index=0),
            VideoSegment.create(3000.0, 6000.0, speed=0.75, sequence_index=1),
        ],
        timeline_frames=None,
        highlights=None,
        screen_transitions=transitions,
        src_w=320,
        src_h=180,
        src_fps=30.0,
        total_sec=6.0,
        is_cfr=True,
        source_has_audio=False,
    )
    try:
        graph = plan.filtergraph
        assert "[st0graphicbase]" in graph
        assert "[st1graphicbase]" in graph
        assert "[st0bar0stage]" in graph
        assert "[st1bar0stage]" in graph
        assert "movie=filename=" in graph
        assert "noise=" not in graph
        assert "gradients=" not in graph
        assert "drawbox=" not in graph
        assert "w='max(2,ceil(" in graph
        assert "floor(W*(" in graph
        assert any(str(path).endswith(".png") for path in plan.temp_files)
        assert plan.output_total_sec == pytest.approx(6.6)
    finally:
        exporter._cleanup_temp_files(plan.temp_files, plan.temp_dirs)

def test_screen_transition_inside_deleted_gap_does_not_leave_split_outputs() -> None:
    transition = ScreenTransition.create(
        1250.0,
        effect_type="blur_dissolve",
        duration_ms=400.0,
        enabled=True,
    )
    plan = VideoExporter()._build_filtergraph(
        bg_preset=None,
        frame_preset=None,
        target_resolution=(320, 180),
        duration_ms=2000.0,
        frame_timestamps=None,
        keyframes=[],
        mouse_track=[],
        click_events=[],
        click_preset=None,
        monitor_rect={"left": 0, "top": 0, "width": 320, "height": 180},
        video_segments=[
            VideoSegment.create(0.0, 1000.0),
            VideoSegment.create(1500.0, 2000.0),
        ],
        timeline_frames=None,
        highlights=None,
        screen_transitions=[transition],
        src_w=320,
        src_h=180,
        src_fps=30.0,
        total_sec=2.0,
        is_cfr=True,
        source_has_audio=False,
    )
    try:
        graph = plan.filtergraph
        assert "[1:v]split=2[fr0][fr1]" in graph
        assert "[stfr" not in graph
        assert "transitionvideo" not in graph
        assert "[st0out]" not in graph
        assert plan.output_total_sec == pytest.approx(1.5)
    finally:
        VideoExporter()._cleanup_temp_files(plan.temp_files, plan.temp_dirs)


def test_transition_and_inserted_explainer_render_as_independent_output_blocks() -> None:
    transition = ScreenTransition.create(
        1000.0,
        effect_type="smooth_settle",
        duration_ms=400.0,
        enabled=True,
    )
    explainer = ExplainerScene.create(
        0.0,
        2000.0,
        "right",
        CanvasLayoutScene.create(
            0.0,
            2000.0,
            video_scale=0.58,
            video_x=0.02,
            video_y=0.2,
            background_color="#193047",
        ),
        TextAnnotation.create(
            0.0,
            2000.0,
            text="Stable explainer",
            color=(255, 255, 255, 255),
        ),
    )
    plan = VideoExporter()._build_filtergraph(
        bg_preset=None,
        frame_preset=DEFAULT_FRAME,
        target_resolution=(320, 180),
        duration_ms=2000.0,
        frame_timestamps=None,
        keyframes=[],
        mouse_track=[],
        click_events=[],
        click_preset=None,
        monitor_rect={"left": 0, "top": 0, "width": 320, "height": 180},
        video_segments=[VideoSegment.create(0.0, 2000.0)],
        timeline_frames=None,
        highlights=None,
        screen_transitions=[transition],
        explainer_scenes=[explainer],
        src_w=320,
        src_h=180,
        src_fps=30.0,
        total_sec=2.0,
        is_cfr=True,
        source_has_audio=False,
    )
    try:
        graph = plan.filtergraph
        video_pos = graph.index("[st0video]")
        placed_pos = graph.index("[st0placed]")
        frame_pos = graph.index("[st0framed]")
        output_pos = graph.index("[st0out]")
        assert video_pos < placed_pos < frame_pos < output_pos
        assert "[st0text0]" not in graph
        assert "[ex0frozen]" in graph
        assert "[ex0text]" in graph
        assert "[ex0out]" in graph
        assert "sendcmd=f=" not in graph
        assert "[st0background][st0incomingscaled]overlay=" in graph
        assert "[st0stage0][st0outgoingscaled]overlay=" in graph
        assert "blend=all_expr=" not in graph
        assert not any(path.endswith(".cmd") for path in plan.temp_files)
        assert plan.output_total_sec == pytest.approx(4.4)
        endpoint_line = next(
            line
            for line in graph.split(";\n")
            if "transitionvideo0]" in line and "split=" in line
        )
        endpoint_segment = re.match(r"\[s(\d+)", endpoint_line)
        assert endpoint_segment is not None
        segment_index = endpoint_segment.group(1)
        assert graph.index(endpoint_line) < graph.index(f"[s{segment_index}base]")
        assert plan.transition_text_img_paths
    finally:
        VideoExporter()._cleanup_temp_files(plan.temp_files, plan.temp_dirs)


def test_scale_swap_normalizes_both_frames_to_even_rgba_canvas() -> None:
    transition = ScreenTransition.create(
        1000.0,
        effect_type="scale_swap",
        duration_ms=600.0,
        enabled=True,
    )
    plan = VideoExporter()._build_filtergraph(
        bg_preset=None,
        frame_preset=DEFAULT_FRAME,
        target_resolution=(1920, 1080),
        duration_ms=2000.0,
        frame_timestamps=None,
        keyframes=[],
        mouse_track=[],
        click_events=[],
        click_preset=None,
        monitor_rect={"left": 0, "top": 0, "width": 1920, "height": 1080},
        video_segments=[VideoSegment.create(0.0, 2000.0)],
        timeline_frames=[],
        highlights=[],
        screen_transitions=[transition],
        src_w=1920,
        src_h=1080,
        src_fps=60.0,
        total_sec=2.0,
        is_cfr=True,
    )
    try:
        assert "scale=1576:888:flags=lanczos" in plan.filtergraph
        assert "s=1576x888" in plan.filtergraph
        assert "[st0new]scale=w='max(2,trunc(iw*" in plan.filtergraph
        assert "[st0old]scale=w='max(2,trunc(iw*" in plan.filtergraph
        assert "[st0background][st0incomingscaled]overlay=" in plan.filtergraph
        assert "[st0stage0][st0outgoingscaled]overlay=" in plan.filtergraph
        assert ",geq=" not in plan.filtergraph
        assert "format=rgba[st0background]" in plan.filtergraph
    finally:
        VideoExporter()._cleanup_temp_files(plan.temp_files, plan.temp_dirs)


@pytest.mark.parametrize(
    ("effect_type", "expected_midpoint"),
    [
        ("directional_push", ("red", "blue", "blue")),
        ("axis_flip", ("red", "red", "blue")),
        ("scale_swap", ("blue", "red", "blue")),
        ("zoom_through", ("red", "blue", "red")),
        # Graphic presets contain cyan, teal, cobalt, violet, lime, amber, and
        # charcoal panels. Red/blue dominance cannot identify scene ownership
        # while those opaque material bars are present.
        ("graphic_vertical_bars", None),
        ("graphic_horizontal_bars", None),
        ("graphic_diagonal_bars", None),
        ("graphic_split_in", None),
        ("graphic_split_out", None),
        ("graphic_sweep", None),
        ("graphic_fold", None),
    ],
)
def test_ffmpeg_compiles_processed_screen_transition(
    tmp_path,
    monkeypatch,
    effect_type,
    expected_midpoint,
) -> None:
    ffmpeg = video_exporter._ffmpeg_exe()
    if not (os.path.exists(ffmpeg) or shutil.which(ffmpeg)):
        ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is not available")

    source = tmp_path / f"{effect_type}-source.mp4"
    source_result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            # Preserve one stale outgoing frame after the logical boundary.
            # The transition owns that bridge frame and must not replay it.
            "color=c=red:s=320x180:r=30:d=1.033333",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=320x180:r=30:d=0.966667",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0[outv]",
            "-map",
            "[outv]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert source_result.returncode == 0, source_result.stderr

    frame_path = tmp_path / "transparent-frame.png"
    Image.new("RGBA", (320, 180), (0, 0, 0, 0)).save(frame_path)
    monkeypatch.setattr(
        video_exporter,
        "generate_device_frame_png",
        lambda *args, **kwargs: str(frame_path),
    )
    transition = ScreenTransition.create(
        1000.0,
        effect_type=effect_type,
        duration_ms=400.0,
        outgoing_frame_ms=966.667,
        incoming_frame_ms=1033.333,
    )
    plan = VideoExporter()._build_filtergraph(
        bg_preset=BACKGROUND_PRESETS[0],
        frame_preset=_no_frame_preset(),
        target_resolution=(320, 180),
        duration_ms=2000.0,
        frame_timestamps=None,
        keyframes=[],
        mouse_track=[],
        click_events=[],
        click_preset=None,
        monitor_rect={"left": 0, "top": 0, "width": 320, "height": 180},
        video_segments=[VideoSegment.create(0.0, 2000.0)],
        timeline_frames=[],
        highlights=[],
        screen_transitions=[transition],
        canvas_layout_scenes=[
            CanvasLayoutScene.create(
                0.0,
                2000.0,
                background_color="#00ff00",
                device_frame_visible=False,
            )
        ],
        src_w=320,
        src_h=180,
        src_fps=30.0,
        total_sec=2.0,
        is_cfr=True,
    )
    graph_path = tmp_path / f"{effect_type}.txt"
    graph_path.write_text(plan.filtergraph, encoding="utf-8")
    output = tmp_path / f"{effect_type}.mp4"
    command = VideoExporter()._build_ffmpeg_command(
        ffmpeg=ffmpeg,
        input_path=str(source),
        output_path=str(output),
        graph_path=str(graph_path),
        plan=plan,
        encoder_id="libx264",
    )
    result = subprocess.run(command, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    assert output.exists() and output.stat().st_size > 0
    samples = []
    for timestamp in (
        0.50,
        0.90,
        0.97,
        1.00,
        1.02,
        1.05,
        1.10,
        1.20,
        1.30,
        1.40,
        1.43,
        1.47,
    ):
        midpoint = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                str(output),
                "-frames:v",
                "1",
                "-c:v",
                "png",
                "-f",
                "image2pipe",
                "pipe:1",
            ],
            capture_output=True,
            timeout=20,
        )
        assert midpoint.returncode == 0 and midpoint.stdout
        with Image.open(io.BytesIO(midpoint.stdout)) as frame:
            rgb = frame.convert("RGB")
            colors = [
                rgb.getpixel((20, 20)),
                rgb.getpixel((160, 90)),
                rgb.getpixel((299, 159)),
            ]
        samples.append((timestamp, colors))
    transition_samples = [
        colors for timestamp, colors in samples if 1.0 <= timestamp <= 1.4
    ]
    assert any(
        any(blue > red + 35 for red, _green, blue in colors)
        for colors in transition_samples
    ), samples
    assert all(
        all(not (green > red + 35 and green > blue + 35) for red, green, blue in colors)
        for colors in transition_samples
    ), samples
    post_transition_samples = [
        colors for timestamp, colors in samples if timestamp >= 1.40
    ]
    assert all(
        all(blue > red + 35 for red, _green, blue in colors)
        for colors in post_transition_samples
    ), samples
    midpoint_colors = min(samples, key=lambda item: abs(item[0] - 1.20))[1]

    def dominant(pixel) -> str:
        red, _green, blue = pixel
        return "red" if red > blue else "blue"

    if expected_midpoint is not None:
        assert (
            tuple(dominant(pixel) for pixel in midpoint_colors)
            == expected_midpoint
        ), samples


def test_screen_transition_keeps_video_inside_real_device_frame(tmp_path) -> None:
    ffmpeg = video_exporter._ffmpeg_exe()
    if not (os.path.exists(ffmpeg) or shutil.which(ffmpeg)):
        ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is not available")

    source = tmp_path / "transition-device-source.mp4"
    source_result = subprocess.run(
        [
            ffmpeg, "-y", "-f", "lavfi", "-i", "color=c=red:s=320x180:r=30:d=1",
            "-f", "lavfi", "-i", "color=c=blue:s=320x180:r=30:d=1",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[outv]",
            "-map", "[outv]", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert source_result.returncode == 0, source_result.stderr

    transition = ScreenTransition.create(
        1000.0,
        effect_type="zoom_through",
        duration_ms=1000.0,
        outgoing_frame_ms=966.667,
        incoming_frame_ms=1033.333,
    )
    layout = CanvasLayoutScene.create(
        0.0, 2000.0, background_color="#00ff00", device_frame_visible=True
    )
    exporter = VideoExporter()
    plan = exporter._build_filtergraph(
        bg_preset=BACKGROUND_PRESETS[0],
        frame_preset=DEFAULT_FRAME,
        target_resolution=(320, 180),
        duration_ms=2000.0,
        frame_timestamps=None,
        keyframes=[],
        mouse_track=[],
        click_events=[],
        click_preset=None,
        monitor_rect={"left": 0, "top": 0, "width": 320, "height": 180},
        video_segments=[VideoSegment.create(0.0, 2000.0)],
        timeline_frames=[],
        highlights=[],
        screen_transitions=[transition],
        canvas_layout_scenes=[layout],
        src_w=320,
        src_h=180,
        src_fps=30.0,
        total_sec=2.0,
        is_cfr=True,
    )
    graph_path = tmp_path / "transition-device-graph.txt"
    graph_path.write_text(plan.filtergraph, encoding="utf-8")
    output = tmp_path / "transition-device.mp4"
    try:
        result = subprocess.run(
            exporter._build_ffmpeg_command(
                ffmpeg=ffmpeg,
                input_path=str(source),
                output_path=str(output),
                graph_path=str(graph_path),
                plan=plan,
                encoder_id="libx264",
            ),
            capture_output=True,
            text=True,
            timeout=45,
        )
        assert result.returncode == 0, result.stderr
        sample = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", "1.5", "-i", str(output),
             "-frames:v", "1", "-c:v", "png", "-f", "image2pipe", "pipe:1"],
            capture_output=True,
            timeout=20,
        )
        assert sample.returncode == 0 and sample.stdout
        with Image.open(io.BytesIO(sample.stdout)) as frame:
            red, green, blue = frame.convert("RGB").getpixel((160, 90))
        assert max(red, blue) > green + 30, (red, green, blue)
        assert green < 32, (red, green, blue)
    finally:
        exporter._cleanup_temp_files(plan.temp_files, plan.temp_dirs)


def test_explainer_keeps_frozen_video_inside_real_device_frame(tmp_path) -> None:
    ffmpeg = video_exporter._ffmpeg_exe()
    if not (os.path.exists(ffmpeg) or shutil.which(ffmpeg)):
        ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is not available")

    source = tmp_path / "explainer-device-source.mp4"
    source_result = subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i", "color=c=red:s=320x180:r=30:d=3",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert source_result.returncode == 0, source_result.stderr

    solved = solve_explainer_layout("right")
    scene = ExplainerScene.create(
        1000.0,
        4000.0,
        "right",
        CanvasLayoutScene.create(
            1000.0,
            4000.0,
            video_scale=solved.video_scale,
            video_x=solved.video_x,
            video_y=solved.video_y,
            background_color="#00ff00",
            device_frame_visible=True,
        ),
        TextAnnotation.create(
            1000.0,
            4000.0,
            x=solved.text_x,
            y=solved.text_y,
            text="Explain",
            max_width=solved.text_max_width,
            background_color=None,
        ),
    )
    layout = CanvasLayoutScene.create(
        0.0, 3000.0, background_color="#00ff00", device_frame_visible=True
    )
    exporter = VideoExporter()
    plan = exporter._build_filtergraph(
        bg_preset=BACKGROUND_PRESETS[0],
        frame_preset=DEFAULT_FRAME,
        target_resolution=(320, 180),
        duration_ms=3000.0,
        frame_timestamps=None,
        keyframes=[],
        mouse_track=[],
        click_events=[],
        click_preset=None,
        monitor_rect={"left": 0, "top": 0, "width": 320, "height": 180},
        video_segments=[VideoSegment.create(0.0, 3000.0)],
        timeline_frames=[],
        highlights=[],
        explainer_scenes=[scene],
        canvas_layout_scenes=[layout],
        src_w=320,
        src_h=180,
        src_fps=30.0,
        total_sec=3.0,
        is_cfr=True,
    )
    graph_path = tmp_path / "explainer-device-graph.txt"
    graph_path.write_text(plan.filtergraph, encoding="utf-8")
    output = tmp_path / "explainer-device.mp4"
    try:
        result = subprocess.run(
            exporter._build_ffmpeg_command(
                ffmpeg=ffmpeg,
                input_path=str(source),
                output_path=str(output),
                graph_path=str(graph_path),
                plan=plan,
                encoder_id="libx264",
            ),
            capture_output=True,
            text=True,
            timeout=45,
        )
        assert result.returncode == 0, result.stderr
        samples = {}
        for timestamp in (0.5, 2.5):
            sample = subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", str(timestamp), "-i", str(output),
                 "-frames:v", "1", "-c:v", "png", "-f", "image2pipe", "pipe:1"],
                capture_output=True,
                timeout=20,
            )
            assert sample.returncode == 0 and sample.stdout
            with Image.open(io.BytesIO(sample.stdout)) as frame:
                samples[timestamp] = frame.convert("RGB").copy()
        pixels = samples[2.5]
        if pixels:
            red_pixels = sum(
                1
                for y in range(pixels.height)
                for x in range(pixels.width)
                if pixels.getpixel((x, y))[0] > pixels.getpixel((x, y))[1] + 40
            )
            common_colors = sorted(
                pixels.getcolors(maxcolors=pixels.width * pixels.height) or [],
                reverse=True,
            )[:5]
        before_red = samples[0.5].getpixel((160, 90))[0]
        assert red_pixels > 500, (before_red, common_colors)
    finally:
        exporter._cleanup_temp_files(plan.temp_files, plan.temp_dirs)


def test_explainer_at_transition_boundary_after_cut_freezes_incoming_video(
    tmp_path,
) -> None:
    ffmpeg = video_exporter._ffmpeg_exe()
    if not (os.path.exists(ffmpeg) or shutil.which(ffmpeg)):
        ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is not available")

    source = tmp_path / "cut-transition-explainer-source.mp4"
    source_result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=320x180:r=30:d=2.55",
            "-f",
            "lavfi",
            "-i",
            "color=c=lime:s=320x180:r=30:d=1.45",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0[outv]",
            "-map",
            "[outv]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert source_result.returncode == 0, source_result.stderr

    transition = ScreenTransition.create(
        2500.0,
        effect_type="scale_swap",
        duration_ms=400.0,
        outgoing_frame_ms=2466.667,
        incoming_frame_ms=2600.0,
    )
    solved = solve_explainer_layout("right")
    scene = ExplainerScene.create(
        2500.0,
        3500.0,
        "right",
        CanvasLayoutScene.create(
            2500.0,
            3500.0,
            video_scale=solved.video_scale,
            video_x=solved.video_x,
            video_y=solved.video_y,
            background_color="#ff00ff",
            device_frame_visible=False,
        ),
        TextAnnotation.create(
            2500.0,
            3500.0,
            x=solved.text_x,
            y=solved.text_y,
            text="Incoming frame",
            max_width=solved.text_max_width,
        ),
        video_transition_ms=200.0,
    )
    exporter = VideoExporter()
    plan = exporter._build_filtergraph(
        bg_preset=BACKGROUND_PRESETS[0],
        frame_preset=_no_frame_preset(),
        target_resolution=(320, 180),
        duration_ms=4000.0,
        frame_timestamps=None,
        keyframes=[],
        mouse_track=[],
        click_events=[],
        click_preset=None,
        monitor_rect={"left": 0, "top": 0, "width": 320, "height": 180},
        video_segments=[
            VideoSegment.create(0.0, 1000.0, sequence_index=0),
            VideoSegment.create(2000.0, 4000.0, sequence_index=1),
        ],
        timeline_frames=[],
        highlights=[],
        screen_transitions=[transition],
        explainer_scenes=[scene],
        canvas_layout_scenes=[
            CanvasLayoutScene.create(
                0.0,
                4000.0,
                background_color="#ff00ff",
                device_frame_visible=False,
            )
        ],
        src_w=320,
        src_h=180,
        src_fps=30.0,
        total_sec=4.0,
        is_cfr=True,
        source_has_audio=False,
    )
    graph_path = tmp_path / "cut-transition-explainer.ffgraph"
    graph_path.write_text(plan.filtergraph, encoding="utf-8")
    output = tmp_path / "cut-transition-explainer.mp4"
    try:
        result = subprocess.run(
            exporter._build_ffmpeg_command(
                ffmpeg=ffmpeg,
                input_path=str(source),
                output_path=str(output),
                graph_path=str(graph_path),
                plan=plan,
                encoder_id="libx264",
            ),
            capture_output=True,
            text=True,
            timeout=45,
        )
        assert result.returncode == 0, result.stderr
        sample = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                "2.5",
                "-i",
                str(output),
                "-frames:v",
                "1",
                "-c:v",
                "png",
                "-f",
                "image2pipe",
                "pipe:1",
            ],
            capture_output=True,
            timeout=20,
        )
        assert sample.returncode == 0 and sample.stdout
        with Image.open(io.BytesIO(sample.stdout)) as frame:
            red, green, blue = frame.convert("RGB").getpixel((220, 90))
        assert green > red + 60, (red, green, blue)
        assert green > blue + 60, (red, green, blue)
    finally:
        exporter._cleanup_temp_files(plan.temp_files, plan.temp_dirs)


def test_click_point_for_export_uses_exact_hook_coordinate() -> None:
    click = ClickEvent(x=300.0, y=300.0, timestamp=1500.0)
    assert _click_point_for_export(click) == (300.0, 300.0)


def test_local_clicks_cull_global_events_outside_recorded_monitor() -> None:
    segment = VideoSegment.create(0.0, 2000.0, 1.0)
    monitor = {"left": -1920, "top": 0, "width": 1920, "height": 1080}
    clicks = [
        ClickEvent(x=-100.0, y=500.0, timestamp=1000.0),
        ClickEvent(x=50.0, y=500.0, timestamp=1100.0),
    ]

    local = _local_clicks_for_segment(clicks, segment, monitor)

    assert [(click.x, click.y, click.timestamp) for click in local] == [
        (-100.0, 500.0, 1000.0)
    ]


def test_session_media_mapper_maps_cut_segments_to_encoded_timeline() -> None:
    frame_timestamps = [0.0, 1000.0, 2000.0, 4000.0, 6000.0, 8000.0]
    mapper = _SessionMediaMapper(frame_timestamps, media_duration_sec=3.0, fps=2.0)
    first = VideoSegment.create(0.0, 2000.0, 1.0)
    second = VideoSegment.create(4000.0, 8000.0, 1.0)

    assert mapper.segment_bounds(first) == (0.0, 1.0, 1.0)
    assert mapper.segment_bounds(second) == (1.5, 3.0, 1.5)


def test_timing_mapper_uses_compressed_time_for_short_media() -> None:
    frame_timestamps = [0.0, 1000.0, 2000.0, 4000.0, 6000.0, 8000.0]

    mapper = _select_session_media_mapper(
        frame_timestamps,
        source_duration_ms=8000.0,
        media_duration_sec=3.0,
        fps=2.0,
        is_cfr=False,
    )

    assert isinstance(mapper, _SessionMediaMapper)
    assert mapper.segment_bounds(VideoSegment.create(4000.0, 8000.0, 1.0)) == (1.5, 3.0, 1.5)


def test_timing_mapper_uses_raw_time_for_normal_duration_media() -> None:
    frame_timestamps = [0.0, 1000.0, 2000.0, 4000.0, 6000.0, 8000.0]

    mapper = _select_session_media_mapper(
        frame_timestamps,
        source_duration_ms=8000.0,
        media_duration_sec=8.02,
        fps=2.0,
    )
    segment = VideoSegment.create(4000.0, 8000.0, 1.0)
    media_start, _, _ = mapper.segment_bounds(segment)

    assert isinstance(mapper, _RawSessionMediaMapper)
    assert mapper.segment_bounds(segment) == (4.0, 8.0, 4.0)
    assert _media_time_for_segment(6000.0, segment, mapper, media_start) == pytest.approx(2.0)


def test_media_keyframes_use_media_local_time() -> None:
    frame_timestamps = [0.0, 1000.0, 2000.0, 4000.0, 6000.0, 8000.0]
    mapper = _SessionMediaMapper(frame_timestamps, media_duration_sec=3.0, fps=2.0)
    segment = VideoSegment.create(4000.0, 8000.0, 1.0)
    media_start, _, _ = mapper.segment_bounds(segment)
    keyframe = ZoomKeyframe.create(timestamp=6000.0, zoom=1.5, duration=500.0)

    mapped = _media_keyframes_for_segment([keyframe], segment, mapper, media_start)

    assert len(mapped) == 1
    assert mapped[0].timestamp == 500.0
    assert mapped[0].duration == pytest.approx(125.0)


def test_zoompan_uses_media_relative_keyframe_time_for_time_lapse_recordings() -> None:
    frame_timestamps = [0.0, 1000.0, 2000.0, 4000.0, 6000.0, 8000.0]
    mapper = _SessionMediaMapper(frame_timestamps, media_duration_sec=3.0, fps=2.0)
    segment = VideoSegment.create(4000.0, 8000.0, 1.0)
    media_start, _, _ = mapper.segment_bounds(segment)
    keyframe = ZoomKeyframe.create(timestamp=6000.0, zoom=1.5, duration=500.0)

    media_keyframes = _media_keyframes_for_segment([keyframe], segment, mapper, media_start)
    zoompan = _build_zoompan_filter(media_keyframes, fps=2.0)

    assert "time - 0.500000" in zoompan
    assert "lt(time,0.625000)" in zoompan
    assert "time - 6.0" not in zoompan
    assert "lt(time, 6.5)" not in zoompan


def test_zoompan_uses_native_capture_framerate() -> None:
    keyframe = ZoomKeyframe.create(timestamp=0.0, zoom=1.5, duration=250.0)

    zoompan = _build_zoompan_filter([keyframe], fps=60.0)

    assert ":fps=60.0" in zoompan
    assert ":fps=25" not in zoompan


def test_zoompan_expression_grows_linearly_for_many_keyframes() -> None:
    keyframes = [
        ZoomKeyframe.create(
            timestamp=float(index * 500),
            zoom=1.5 if index % 2 == 0 else 1.0,
            x=0.25 if index % 2 == 0 else 0.5,
            y=0.75 if index % 2 == 0 else 0.5,
            duration=250.0,
        )
        for index in range(32)
    ]

    zoompan = _build_zoompan_filter(keyframes, fps=60.0)

    assert len(zoompan) < 60000
    # The zoom expression is reused in z and both crop axes; pan expressions
    # appear once each. The count remains a fixed multiple of keyframes.
    assert zoompan.count("gte(time,") == len(keyframes) * 7
    assert "pow(clip((time - " in zoompan


@pytest.mark.parametrize("progress", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_export_easing_matches_preview_smoothstep(progress: float) -> None:
    from app.zoom_engine import ease_in_out

    assert _ease_in_out(progress) == pytest.approx(ease_in_out(progress))


def test_media_keyframes_do_not_carry_zoom_state_from_cut_gap() -> None:
    frame_timestamps = [0.0, 1000.0, 2000.0, 4000.0, 6000.0, 8000.0]
    mapper = _SessionMediaMapper(frame_timestamps, media_duration_sec=3.0, fps=2.0)
    segment = VideoSegment.create(4000.0, 8000.0, 1.0)
    media_start, _, _ = mapper.segment_bounds(segment)
    keyframes = [
        ZoomKeyframe.create(timestamp=2000.0, zoom=2.0, x=0.25, y=0.75, duration=0.0),
        ZoomKeyframe.create(timestamp=7000.0, zoom=1.0, x=0.5, y=0.5, duration=1000.0),
    ]

    mapped = _media_keyframes_for_segment(keyframes, segment, mapper, media_start)

    assert len(mapped) == 1
    assert mapped[0].timestamp == pytest.approx(750.0)
    assert mapped[0].zoom == pytest.approx(1.0)
    assert mapped[0].x == pytest.approx(0.5)
    assert mapped[0].y == pytest.approx(0.5)


def test_media_time_for_segment_maps_click_to_source_frame_time() -> None:
    frame_timestamps = [0.0, 1000.0, 2000.0, 4000.0, 6000.0, 8000.0]
    mapper = _SessionMediaMapper(frame_timestamps, media_duration_sec=3.0, fps=2.0)
    segment = VideoSegment.create(4000.0, 8000.0, 1.0)
    media_start, _, _ = mapper.segment_bounds(segment)

    assert _media_time_for_segment(6000.0, segment, mapper, media_start) == pytest.approx(0.5)


def test_click_overlay_window_uses_media_relative_time_for_time_lapse_recordings() -> None:
    frame_timestamps = [0.0, 1000.0, 2000.0, 4000.0, 6000.0, 8000.0]
    mapper = _SessionMediaMapper(frame_timestamps, media_duration_sec=3.0, fps=2.0)
    segment = VideoSegment.create(4000.0, 8000.0, 1.0)
    media_start, _, _ = mapper.segment_bounds(segment)

    start_sec, end_sec = _media_window_for_segment(
        6000.0,
        1000.0,
        segment,
        mapper,
        media_start,
    )

    assert start_sec == pytest.approx(0.5)
    assert end_sec == pytest.approx(0.75)


def test_timed_overlay_filter_uses_main_video_timeline() -> None:
    assert _timed_overlay_filter("base", "cl0", "click0", x=11, y=12, start_sec=1.25, end_sec=1.65) == (
        "[base][cl0]overlay=x=11:y=12:eof_action=repeat:repeatlast=1:"
        "enable='between(t,1.250000,1.650000)'[click0]"
    )


def test_filtergraph_builder_handles_cut_retime_highlight_and_frame(
    monkeypatch,
    request,
    synthetic_cut_retime_highlight_frame_session: RecordingSession,
) -> None:
    """Synthetic graph contract test without invoking the FFmpeg subprocess."""
    import app.video_exporter as video_exporter

    monkeypatch.setattr(video_exporter, "generate_device_frame_png", lambda *args, **kwargs: "frame.png")
    monkeypatch.setattr(video_exporter, "generate_highlight_png", lambda *args, **kwargs: "highlight.png")
    monkeypatch.setattr(video_exporter, "generate_timeline_frame_png", lambda *args, **kwargs: "timeline-frame.png")

    session = synthetic_cut_retime_highlight_frame_session
    exporter = VideoExporter()
    plan = exporter._build_filtergraph(
        bg_preset=BACKGROUND_PRESETS[0],
        frame_preset=_no_frame_preset(),
        target_resolution=(1280, 720),
        duration_ms=session.duration,
        frame_timestamps=session.frame_timestamps,
        keyframes=session.keyframes,
        mouse_track=session.mouse_track,
        click_events=session.click_events,
        click_preset=None,
        monitor_rect={"left": 0, "top": 0, "width": 1280, "height": 720},
        video_segments=session.video_segments,
        timeline_frames=session.timeline_frames,
        highlights=session.highlights,
        src_w=1280,
        src_h=720,
        src_fps=30.0,
        total_sec=9.0,
    )
    request.addfinalizer(
        lambda: exporter._cleanup_temp_files(plan.temp_files, plan.temp_dirs)
    )

    graph = plan.filtergraph

    assert plan.has_timeline_edits is True
    assert plan.has_speed_changes is True
    assert plan.output_total_sec == pytest.approx(4.5)
    assert plan.highlight_img_paths == []
    assert plan.timeline_frame_img_paths == ["timeline-frame.png"]

    expected_nodes = [
        "[1:v]split=3[fr0][fr1][fr2]",
        "[0:v]trim=start=0.000000:end=2.000000,setpts=PTS-STARTPTS[s0trim]",
        "[s0trim]zoompan=",
            "[s0bg][s0cursor]overlay=",
        "[s0framed]setpts=1.00000000*PTS[s0out]",
        "[0:v]trim=start=5.000000:end=6.500000,setpts=PTS-STARTPTS[s1trim]",
        "[s1trim]zoompan=",
        "color=c=black@1.0:s=1280x720:r=30.0:d=1.500000,format=rgba,geq=r=0:g=0:b=0:a='",
        "s1highlight0mask]overlay=x=0:y=0:eof_action=repeat:repeatlast=1:enable='between(t,1.000000,1.500000)'[s1highlight0dim]",
        "[s1bg][s1highlight0dim]overlay=x=0:y=0:shortest=1:eof_action=pass[s1base]",
        "[s1framed]setpts=0.25000000*PTS[s1out]",
        "[3:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=30.0,trim=duration=1.500000,setpts=PTS-STARTPTS[tf0out]",
        "[0:v]trim=start=6.500000:end=9.000000,setpts=PTS-STARTPTS[s2trim]",
        "[s2trim]zoompan=",
        "format=rgba,geq=r=0:g=0:b=0:a='",
        "s2highlight0mask]overlay=x=0:y=0:eof_action=repeat:repeatlast=1:enable='between(t,0.000000,0.500000)'[s2highlight0dim]",
        "[s2bg][s2highlight0dim]overlay=x=0:y=0:shortest=1:eof_action=pass[s2base]",
        "[s2framed]setpts=0.25000000*PTS[s2out]",
        "[s0out][s1out][tf0out][s2out]concat=n=4:v=1:a=0[out]",
    ]
    cursor = -1
    for node in expected_nodes:
        next_pos = graph.find(node, cursor + 1)
        assert next_pos != -1, f"Missing graph node after offset {cursor}: {node}"
        cursor = next_pos


def test_audio_filtergraph_tracks_cuts_retimes_and_inserted_frames(
    monkeypatch,
    request,
    synthetic_cut_retime_highlight_frame_session: RecordingSession,
) -> None:
    """Source audio and silent inserted frames share the video concat timeline."""
    monkeypatch.setattr(video_exporter, "generate_device_frame_png", lambda *args, **kwargs: "frame.png")
    monkeypatch.setattr(video_exporter, "generate_highlight_png", lambda *args, **kwargs: "highlight.png")
    monkeypatch.setattr(video_exporter, "generate_timeline_frame_png", lambda *args, **kwargs: "timeline-frame.png")

    session = synthetic_cut_retime_highlight_frame_session
    exporter = VideoExporter()
    plan = exporter._build_filtergraph(
        bg_preset=BACKGROUND_PRESETS[0],
        frame_preset=_no_frame_preset(),
        target_resolution=(1280, 720),
        duration_ms=session.duration,
        frame_timestamps=session.frame_timestamps,
        keyframes=session.keyframes,
        mouse_track=session.mouse_track,
        click_events=session.click_events,
        click_preset=None,
        monitor_rect={"left": 0, "top": 0, "width": 1280, "height": 720},
        video_segments=session.video_segments,
        timeline_frames=session.timeline_frames,
        highlights=session.highlights,
        src_w=1280,
        src_h=720,
        src_fps=30.0,
        total_sec=9.0,
        source_has_audio=True,
    )
    request.addfinalizer(lambda: exporter._cleanup_temp_files(plan.temp_files, plan.temp_dirs))

    graph = plan.filtergraph
    assert plan.has_source_audio is True
    assert plan.has_audio_output is True
    assert plan.audio_output_node == "sourceaout"

    expected_nodes = [
        "[0:a]atrim=start=0.000000:end=2.000000,asetpts=PTS-STARTPTS,aresample=48000,aformat=sample_rates=48000:channel_layouts=stereo,atempo=1.000000,asetpts=PTS-STARTPTS[s0aout]",
        "[0:a]atrim=start=5.000000:end=6.500000,asetpts=PTS-STARTPTS,aresample=48000,aformat=sample_rates=48000:channel_layouts=stereo,atempo=2.000000,atempo=2.000000,asetpts=PTS-STARTPTS[s1aout]",
        "anullsrc=r=48000:cl=stereo,atrim=duration=1.500000,asetpts=PTS-STARTPTS[tf0aout]",
        "[0:a]atrim=start=6.500000:end=9.000000,asetpts=PTS-STARTPTS,aresample=48000,aformat=sample_rates=48000:channel_layouts=stereo,atempo=2.000000,atempo=2.000000,asetpts=PTS-STARTPTS[s2aout]",
        "[s0out][s0aout][s1out][s1aout][tf0out][tf0aout][s2out][s2aout]concat=n=4:v=1:a=1[out][sourceaout]",
    ]
    cursor = -1
    for node in expected_nodes:
        next_pos = graph.find(node, cursor + 1)
        assert next_pos != -1, f"Missing audio node after offset {cursor}: {node}"
        cursor = next_pos


def test_text_frame_reveal_animates_only_the_transparent_text_layer(
    monkeypatch,
    request,
    synthetic_cut_retime_highlight_frame_session: RecordingSession,
) -> None:
    monkeypatch.setattr(video_exporter, "generate_device_frame_png", lambda *args, **kwargs: "frame.png")
    monkeypatch.setattr(video_exporter, "generate_highlight_png", lambda *args, **kwargs: "highlight.png")
    generated: list[dict] = []

    def fake_frame_asset(*args, **kwargs):
        generated.append(kwargs)
        return "timeline-frame.png"

    monkeypatch.setattr(video_exporter, "generate_timeline_frame_png", fake_frame_asset)
    session = synthetic_cut_retime_highlight_frame_session
    frame = session.timeline_frames[0]
    frame.text_animation = "fade-slide"
    frame.text_animation_duration_ms = 900.0

    exporter = VideoExporter()
    plan = exporter._build_filtergraph(
        bg_preset=BACKGROUND_PRESETS[0],
        frame_preset=_no_frame_preset(),
        target_resolution=(1280, 720),
        duration_ms=session.duration,
        frame_timestamps=session.frame_timestamps,
        keyframes=session.keyframes,
        mouse_track=session.mouse_track,
        click_events=session.click_events,
        click_preset=None,
        monitor_rect={"left": 0, "top": 0, "width": 1280, "height": 720},
        video_segments=session.video_segments,
        timeline_frames=session.timeline_frames,
        highlights=session.highlights,
        src_w=1280,
        src_h=720,
        src_fps=30.0,
        total_sec=9.0,
    )
    request.addfinalizer(lambda: exporter._cleanup_temp_files(plan.temp_files, plan.temp_dirs))

    assert generated == [{"text_only": True}]
    assert "color=c=0x111827:s=1280x720:r=30.0:d=1.500000" in plan.filtergraph
    assert "[3:v]scale=1280:720:flags=lanczos,format=rgba" in plan.filtergraph
    assert "fade=t=in:st=0.000000:d=0.675000:alpha=1" in plan.filtergraph
    assert "fade=t=out" not in plan.filtergraph
    assert "[tf0base][tf0outasset]overlay=" in plan.filtergraph


def test_filtergraph_builder_emits_ease_layout_group_expression(monkeypatch) -> None:
    monkeypatch.setattr(video_exporter, "generate_device_frame_png", lambda *args, **kwargs: "frame.png")

    first = CanvasLayoutScene.create(0, 5000, video_scale=1.0, video_x=0.0)
    second = CanvasLayoutScene.create(
        5000,
        10000,
        video_scale=0.5,
        video_x=0.5,
        video_y=0.25,
        transition="ease",
        transition_duration_ms=2000,
    )
    plan = VideoExporter()._build_filtergraph(
        bg_preset=BACKGROUND_PRESETS[0],
        frame_preset=_no_frame_preset(),
        target_resolution=(1280, 720),
        duration_ms=10000,
        frame_timestamps=None,
        keyframes=[],
        mouse_track=[],
        click_events=[],
        click_preset=None,
        monitor_rect={"left": 0, "top": 0, "width": 1280, "height": 720},
        video_segments=[VideoSegment.create(0, 10000, 1.0)],
        timeline_frames=[],
        highlights=[],
        src_w=1280,
        src_h=720,
        src_fps=30.0,
        total_sec=10.0,
        is_cfr=True,
        canvas_layout_scenes=[first, second],
    )

    graph = plan.filtergraph
    assert "scale=w='trunc((1280)*(" in graph
    assert "6*pow((clip((t+" in graph
    assert "15*pow((clip((t+" in graph
    assert "overlay=x='(" in graph
    assert "[s2out]" in graph


@pytest.mark.parametrize("animation", ["fade", "fade-slide", "soft-reveal"])
def test_filtergraph_builder_composes_atomic_explainer_scene(
    monkeypatch, animation: str
) -> None:
    monkeypatch.setattr(video_exporter, "generate_device_frame_png", lambda *a, **k: "frame.png")
    monkeypatch.setattr(video_exporter, "generate_text_annotation_png", lambda *a, **k: "text.png")
    solved = solve_explainer_layout("right")
    scene = ExplainerScene.create(
        1000,
        6000,
        "right",
        CanvasLayoutScene.create(
            1000,
            6000,
            video_scale=solved.video_scale,
            video_x=solved.video_x,
            video_y=solved.video_y,
            transition="ease",
            transition_duration_ms=600,
        ),
        TextAnnotation.create(
            1000,
            6000,
            x=solved.text_x,
            y=solved.text_y,
            text="Explain this step",
            max_width=solved.text_max_width,
            background_color=None,
        ),
        text_animation=animation,
    )

    plan = VideoExporter()._build_filtergraph(
        bg_preset=BACKGROUND_PRESETS[0],
        frame_preset=_no_frame_preset(),
        target_resolution=(1280, 720),
        duration_ms=10000,
        frame_timestamps=None,
        keyframes=[],
        mouse_track=[],
        click_events=[],
        click_preset=None,
        monitor_rect={"left": 0, "top": 0, "width": 1280, "height": 720},
        video_segments=[VideoSegment.create(0, 10000, 1.0)],
        timeline_frames=[],
        highlights=[],
        src_w=1280,
        src_h=720,
        src_fps=30.0,
        total_sec=10.0,
        is_cfr=True,
        explainer_scenes=[scene],
    )

    graph = plan.filtergraph
    phases = video_exporter.explainer_phase_timing(scene)
    text_end_sec = (phases.text_exit_end - scene.start_ms) / 1000.0
    text_exit_start_sec = (phases.text_exit_start - scene.start_ms) / 1000.0
    assert "[ex0frozen]" in graph
    assert "6*pow(" in graph
    assert "fade=t=in" in graph
    assert "fade=t=out" in graph
    assert "shortest=1:eof_action=pass:repeatlast=1" in graph
    assert "[ex0groupcanvas][ex0frozen]overlay=" in graph
    assert "[ex0groupvideo][ex0frameheld]overlay=" in graph
    assert "[ex0groupframed]scale=w=" in graph
    assert "[ex0frozen]scale=w=" not in graph
    assert "[ex0frameheld]scale=w=" not in graph
    assert f"fade=t=out:st={text_exit_start_sec:.6f}" in graph
    assert f"enable='between(t,0.000000,{text_end_sec:.6f})'" in graph
    assert text_end_sec <= (
        phases.layout_restore_start - scene.start_ms
    ) / 1000.0
    assert plan.transition_text_img_paths
    assert set(plan.transition_text_img_paths) == {"text.png"}
    if animation == "fade-slide":
        assert "overlay=x='(-44.800000)*(" in graph
    elif animation == "soft-reveal":
        assert "geq=r='r(X,Y)'" in graph


def test_explainer_text_is_absent_before_its_reveal_boundary() -> None:
    solved = solve_explainer_layout("right")
    scene = ExplainerScene.create(
        1000.0,
        6000.0,
        "right",
        CanvasLayoutScene.create(
            1000.0,
            6000.0,
            video_scale=solved.video_scale,
            video_x=solved.video_x,
            video_y=solved.video_y,
        ),
        TextAnnotation.create(
            1000.0,
            6000.0,
            x=solved.text_x,
            y=solved.text_y,
            text="Wait until the video settles",
            max_width=solved.text_max_width,
        ),
    )
    annotation = video_exporter.explainer_text_annotation(scene)
    reveal_start = annotation.start_ms + annotation.animation_delay_ms
    mapper = video_exporter._RawSessionMediaMapper(30.0)

    before_reveal = VideoSegment.create(scene.start_ms, reveal_start, 1.0)
    during_reveal = VideoSegment.create(
        reveal_start,
        reveal_start + annotation.animation_in_ms,
        1.0,
    )

    assert video_exporter._media_text_annotations_for_segment(
        [annotation], before_reveal, mapper, scene.start_ms / 1000.0
    ) == []
    assert len(
        video_exporter._media_text_annotations_for_segment(
            [annotation], during_reveal, mapper, reveal_start / 1000.0
        )
    ) == 1


def test_legacy_vfr_graph_preserves_final_wall_clock_duration(
    monkeypatch,
    synthetic_cut_retime_highlight_frame_session: RecordingSession,
) -> None:
    """Compressed source timing must expand back to the edited output timeline."""
    monkeypatch.setattr(video_exporter, "generate_device_frame_png", lambda *args, **kwargs: "frame.png")
    monkeypatch.setattr(video_exporter, "generate_highlight_png", lambda *args, **kwargs: "highlight.png")
    monkeypatch.setattr(video_exporter, "generate_timeline_frame_png", lambda *args, **kwargs: "timeline-frame.png")

    session = synthetic_cut_retime_highlight_frame_session
    session.frame_timestamps = [float(index * 1000) for index in range(10)]
    plan = VideoExporter()._build_filtergraph(
        bg_preset=BACKGROUND_PRESETS[0],
        frame_preset=_no_frame_preset(),
        target_resolution=(1280, 720),
        duration_ms=session.duration,
        frame_timestamps=session.frame_timestamps,
        keyframes=session.keyframes,
        mouse_track=session.mouse_track,
        click_events=session.click_events,
        click_preset=None,
        monitor_rect={"left": 0, "top": 0, "width": 1280, "height": 720},
        video_segments=session.video_segments,
        timeline_frames=session.timeline_frames,
        highlights=session.highlights,
        src_w=1280,
        src_h=720,
        src_fps=30.0,
        total_sec=4.5,
        is_cfr=False,
    )

    # The encoded source is 4.5 seconds, but the edited output is 2s +
    # 0.375s + 1.5s + 0.625s = 4.5s after retiming and card insertion.
    assert plan.output_total_sec == pytest.approx(4.5)
    assert "setpts=2.22222222*PTS" in plan.filtergraph
    assert "setpts=0.55555556*PTS" in plan.filtergraph
    assert "setpts=0.39682540*PTS" in plan.filtergraph


def test_cfr_graph_uses_raw_session_bounds_even_with_short_media(
    monkeypatch,
    synthetic_cut_retime_highlight_frame_session: RecordingSession,
) -> None:
    """Validated CFR captures must bypass legacy compressed-time mapping."""
    monkeypatch.setattr(video_exporter, "generate_device_frame_png", lambda *args, **kwargs: "frame.png")
    monkeypatch.setattr(video_exporter, "generate_highlight_png", lambda *args, **kwargs: "highlight.png")
    monkeypatch.setattr(video_exporter, "generate_timeline_frame_png", lambda *args, **kwargs: "timeline-frame.png")

    session = synthetic_cut_retime_highlight_frame_session
    session.frame_timestamps = [float(index * 1000) for index in range(10)]
    kwargs = dict(
        bg_preset=BACKGROUND_PRESETS[0],
        frame_preset=_no_frame_preset(),
        target_resolution=(1280, 720),
        duration_ms=session.duration,
        frame_timestamps=session.frame_timestamps,
        keyframes=session.keyframes,
        mouse_track=session.mouse_track,
        click_events=session.click_events,
        click_preset=None,
        monitor_rect={"left": 0, "top": 0, "width": 1280, "height": 720},
        video_segments=session.video_segments,
        timeline_frames=session.timeline_frames,
        highlights=session.highlights,
        src_w=1280,
        src_h=720,
        src_fps=30.0,
        total_sec=4.5,
    )

    cfr_graph = VideoExporter()._build_filtergraph(**kwargs, is_cfr=True).filtergraph
    legacy_graph = VideoExporter()._build_filtergraph(**kwargs, is_cfr=False).filtergraph

    assert "[0:v]trim=start=5.000000:end=6.500000" in cfr_graph
    assert "[0:v]trim=start=2.250000:end=2.925000" in legacy_graph


def test_ffmpeg_export_compiles_rectangle_and_circle_highlights(
    tmp_path,
    monkeypatch,
) -> None:
    """Compile a short real export containing both supported highlight shapes."""
    ffmpeg = video_exporter._ffmpeg_exe()
    if not (os.path.exists(ffmpeg) or shutil.which(ffmpeg)):
        ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is not available")

    source = tmp_path / "source.mp4"
    source_result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x180:rate=30",
            "-t",
            "1",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert source_result.returncode == 0, source_result.stderr

    frame_path = tmp_path / "frame.png"
    Image.new("RGBA", (320, 180), (20, 20, 20, 255)).save(frame_path)
    monkeypatch.setattr(
        video_exporter,
        "generate_device_frame_png",
        lambda *args, **kwargs: str(frame_path),
    )

    rect = HighlightBox.create(
        start_ms=0.0,
        end_ms=500.0,
        x=0.1,
        y=0.1,
        width=0.35,
        height=0.35,
        shape="rect",
        dim_opacity=0.45,
    )
    circle = HighlightBox.create(
        start_ms=500.0,
        end_ms=1000.0,
        x=0.55,
        y=0.25,
        width=0.3,
        height=0.3,
        shape="circle",
        dim_opacity=0.55,
    )
    session = RecordingSession(
        id="ffmpeg-highlight-shapes",
        start_time=0.0,
        duration=1000.0,
        mouse_track=[],
        keyframes=[],
        video_segments=[VideoSegment(id="segment", start_ms=0.0, end_ms=1000.0)],
        highlights=[rect, circle],
    )
    plan = VideoExporter()._build_filtergraph(
        bg_preset=BACKGROUND_PRESETS[0],
        frame_preset=_no_frame_preset(),
        target_resolution=(320, 180),
        duration_ms=session.duration,
        frame_timestamps=None,
        keyframes=[],
        mouse_track=[],
        click_events=[],
        click_preset=None,
        monitor_rect={"left": 0, "top": 0, "width": 320, "height": 180},
        video_segments=session.video_segments,
        timeline_frames=[],
        highlights=session.highlights,
        src_w=320,
        src_h=180,
        src_fps=30.0,
        total_sec=1.0,
        is_cfr=True,
    )
    assert "geq=r=0:g=0:b=0:a='" in plan.filtergraph
    assert plan.filtergraph.count("geq=r=0:g=0:b=0:a='") == 2
    assert "max(abs(X-" in plan.filtergraph

    graph_path = tmp_path / "graph.txt"
    graph_path.write_text(plan.filtergraph, encoding="utf-8")
    output = tmp_path / "highlights.mp4"
    command = VideoExporter()._build_ffmpeg_command(
        ffmpeg=ffmpeg,
        input_path=str(source),
        output_path=str(output),
        graph_path=str(graph_path),
        plan=plan,
        encoder_id="libx264",
    )
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert output.exists()
    assert output.stat().st_size > 0
def test_ffmpeg_export_concatenates_source_audio_with_static_frame_silence(tmp_path) -> None:
    """A timeline edit exports a playable AAC stream, including silent cards."""
    ffmpeg = video_exporter._ffmpeg_exe()
    if not (os.path.exists(ffmpeg) or shutil.which(ffmpeg)):
        ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is not available")

    source = tmp_path / "source-with-audio.mp4"
    source_result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x180:rate=30",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
            "-t",
            "3",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(source),
        ],
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert source_result.returncode == 0, source_result.stderr

    animated_card = TimelineFrame.create(
        1000.0,
        kind="text",
        duration_ms=500.0,
        text="Pause",
    )
    animated_card.text_animation = "fade"
    animated_card.text_animation_duration_ms = 300.0
    session = RecordingSession(
        id="audio-concat",
        start_time=0.0,
        duration=3000.0,
        mouse_track=[],
        keyframes=[],
        video_segments=[
            VideoSegment.create(0.0, 1000.0, speed=1.0),
            VideoSegment.create(1000.0, 3000.0, speed=2.0),
        ],
        timeline_frames=[animated_card],
        highlights=[],
    )
    exporter = VideoExporter()
    plan = exporter._build_filtergraph(
        bg_preset=BACKGROUND_PRESETS[0],
        frame_preset=_no_frame_preset(),
        target_resolution=(320, 180),
        duration_ms=session.duration,
        frame_timestamps=None,
        keyframes=[],
        mouse_track=[],
        click_events=[],
        click_preset=None,
        monitor_rect={"left": 0, "top": 0, "width": 320, "height": 180},
        video_segments=session.video_segments,
        timeline_frames=session.timeline_frames,
        highlights=[],
        src_w=320,
        src_h=180,
        src_fps=30.0,
        total_sec=3.0,
        is_cfr=True,
        source_has_audio=True,
    )
    graph_path = tmp_path / "audio-graph.txt"
    graph_path.write_text(plan.filtergraph, encoding="utf-8")
    output = tmp_path / "audio-output.mp4"
    try:
        command = exporter._build_ffmpeg_command(
            ffmpeg=ffmpeg,
            input_path=str(source),
            output_path=str(output),
            graph_path=str(graph_path),
            plan=plan,
            encoder_id="libx264",
        )
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        assert output.exists() and output.stat().st_size > 0

        probe = subprocess.run(
            [ffmpeg, "-i", str(output)], capture_output=True, text=True, timeout=20
        )
        assert "Audio:" in probe.stderr
    finally:
        exporter._cleanup_temp_files(plan.temp_files, plan.temp_dirs)


@pytest.mark.parametrize("animation", ["fade", "fade-slide", "soft-reveal"])
def test_ffmpeg_export_compiles_explainer_scene(
    tmp_path,
    monkeypatch,
    animation: str,
) -> None:
    """Compile the real transition/text graph, including its phase splits."""
    ffmpeg = video_exporter._ffmpeg_exe()
    if not (os.path.exists(ffmpeg) or shutil.which(ffmpeg)):
        ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is not available")

    source = tmp_path / "source.mp4"
    source_result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x180:rate=30",
            "-t",
            "3",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert source_result.returncode == 0, source_result.stderr

    frame_path = tmp_path / "frame.png"
    Image.new("RGBA", (320, 180), (20, 20, 20, 255)).save(frame_path)
    text_path = tmp_path / "text.png"
    text_image = Image.new("RGBA", (320, 180), (0, 0, 0, 0))
    for x in range(12, 120):
        for y in range(60, 92):
            text_image.putpixel((x, y), (255, 255, 255, 255))
    text_image.save(text_path)
    monkeypatch.setattr(
        video_exporter,
        "generate_device_frame_png",
        lambda *args, **kwargs: str(frame_path),
    )
    monkeypatch.setattr(
        video_exporter,
        "generate_text_annotation_png",
        lambda *args, **kwargs: str(text_path),
    )

    solved = solve_explainer_layout("right")
    scene = ExplainerScene.create(
        300.0,
        2700.0,
        "right",
        CanvasLayoutScene.create(
            300.0,
            2700.0,
            video_scale=solved.video_scale,
            video_x=solved.video_x,
            video_y=solved.video_y,
            transition="ease",
            transition_duration_ms=400.0,
        ),
        TextAnnotation.create(
            300.0,
            2700.0,
            x=solved.text_x,
            y=solved.text_y,
            text="Explain this step",
            max_width=solved.text_max_width,
            background_color=None,
        ),
        video_transition_ms=400.0,
        text_animation=animation,
    )
    plan = VideoExporter()._build_filtergraph(
        bg_preset=BACKGROUND_PRESETS[0],
        frame_preset=_no_frame_preset(),
        target_resolution=(320, 180),
        duration_ms=3000.0,
        frame_timestamps=None,
        keyframes=[],
        mouse_track=[],
        click_events=[],
        click_preset=None,
        monitor_rect={"left": 0, "top": 0, "width": 320, "height": 180},
        video_segments=[VideoSegment.create(0.0, 3000.0, 1.0)],
        timeline_frames=[
            TimelineFrame.create(
                0.0,
                kind="text",
                duration_ms=300.0,
                text="Opening card",
            )
        ],
        highlights=[],
        src_w=320,
        src_h=180,
        src_fps=30.0,
        total_sec=3.0,
        is_cfr=True,
        explainer_scenes=[scene],
    )

    graph_path = tmp_path / f"{animation}-graph.txt"
    graph_path.write_text(plan.filtergraph, encoding="utf-8")
    output = tmp_path / f"{animation}.mp4"
    command = VideoExporter()._build_ffmpeg_command(
        ffmpeg=ffmpeg,
        input_path=str(source),
        output_path=str(output),
        graph_path=str(graph_path),
        plan=plan,
        encoder_id="libx264",
    )
    result = subprocess.run(command, capture_output=True, text=True, timeout=45)

    assert result.returncode == 0, result.stderr
    assert output.exists()
    assert output.stat().st_size > 0
    if animation == "fade":
        def frame_md5(timestamp: float) -> str:
            probe = subprocess.run(
                [
                    ffmpeg,
                    "-ss",
                    f"{timestamp:.3f}",
                    "-i",
                    str(output),
                    "-frames:v",
                    "1",
                    "-f",
                    "md5",
                    "-",
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )
            assert probe.returncode == 0, probe.stderr
            return probe.stdout.strip()

        # A finite opening card plus an infinite text PNG must not hold the
        # first composition for the rest of the export.
        assert frame_md5(0.1) != frame_md5(2.8)


def test_timeline_frame_asset_releases_windows_file_handle(tmp_path) -> None:
    """Generated PNGs can be replaced immediately after rendering on Windows."""
    frame = TimelineFrame.create(0.0, kind="text", duration_ms=500.0, text="Test")
    path = generate_timeline_frame_png(frame, 320, 180)
    replacement = tmp_path / "replacement.png"

    os.replace(path, replacement)

    assert replacement.is_file()


def test_animated_text_frame_asset_contains_only_transparent_typography() -> None:
    frame = TimelineFrame.create(0.0, kind="text", duration_ms=1000.0, text="Reveal")
    path = generate_timeline_frame_png(frame, 320, 180, text_only=True)
    try:
        with Image.open(path) as rendered:
            alpha = np.asarray(rendered.convert("RGBA"))[:, :, 3]
            assert alpha[0, 0] == 0
            assert alpha.max() == 255
            assert np.count_nonzero(alpha) > 0
    finally:
        os.remove(path)


def test_structured_text_frame_scales_typography_with_output_canvas() -> None:
    frame = TimelineFrame.create(0.0, kind="text", duration_ms=500.0)
    frame.title = "Create a request"
    frame.description = "Add the details and submit it for review."
    frame.title_font_size = 64
    frame.body_font_size = 38
    frame.text_alignment = "center"
    small_path = generate_timeline_frame_png(frame, 1280, 720)
    large_path = generate_timeline_frame_png(frame, 1920, 1080)
    try:
        with Image.open(small_path) as small, Image.open(large_path) as large:
            small_bg = Image.new("RGBA", small.size, (17, 24, 39, 255))
            large_bg = Image.new("RGBA", large.size, (17, 24, 39, 255))
            small_box = Image.fromarray(
                np.any(np.asarray(small) != np.asarray(small_bg), axis=2).astype(np.uint8) * 255
            ).getbbox()
            large_box = Image.fromarray(
                np.any(np.asarray(large) != np.asarray(large_bg), axis=2).astype(np.uint8) * 255
            ).getbbox()
            assert small_box is not None and large_box is not None
            small_height = small_box[3] - small_box[1]
            large_height = large_box[3] - large_box[1]
            assert large_height / small_height == pytest.approx(1.5, rel=0.12)
    finally:
        os.remove(small_path)
        os.remove(large_path)


def test_new_picture_frame_fill_is_full_bleed_while_fit_preserves_gutters(tmp_path) -> None:
    source = tmp_path / "square.png"
    Image.new("RGBA", (100, 100), (230, 40, 70, 255)).save(source)
    frame = TimelineFrame.create(
        0.0,
        kind="image",
        duration_ms=500.0,
        image_path=str(source),
    )

    assert frame.image_fit == "fill"
    fill_path = generate_timeline_frame_png(frame, 320, 180)
    try:
        with Image.open(fill_path) as rendered:
            assert rendered.getpixel((0, 0))[:3] == (230, 40, 70)
            assert rendered.getpixel((319, 179))[:3] == (230, 40, 70)
    finally:
        os.remove(fill_path)

    frame.image_fit = "fit"
    fit_path = generate_timeline_frame_png(frame, 320, 180)
    try:
        with Image.open(fit_path) as rendered:
            assert rendered.getpixel((0, 0))[:3] == (17, 24, 39)
    finally:
        os.remove(fit_path)


class TestVideoSegments:
    def test_timeline_frames_split_source_without_consuming_video(self):
        segment = VideoSegment.create(0.0, 10000.0, 1.0)
        frames = [
            TimelineFrame(id="text", timestamp_ms=3000.0, duration_ms=2500.0),
            TimelineFrame(id="image", timestamp_ms=3000.0, duration_ms=2500.0, kind="image"),
            TimelineFrame(id="later", timestamp_ms=7000.0, duration_ms=1000.0),
        ]

        split = _split_segments_at_timeline_frames([segment], frames)

        assert [(item.start_ms, item.end_ms) for item in split] == [
            (0.0, 3000.0),
            (3000.0, 7000.0),
            (7000.0, 10000.0),
        ]
        assert _output_time_for_source_timestamp(7000.0, [segment], frames[:2]) == pytest.approx(12.0)

    def test_explicit_segments_preserve_order_and_gaps(self):
        segments = [
            VideoSegment.create(5000.0, 8000.0, 2.0),
            VideoSegment.create(1000.0, 2000.0, 1.0),
        ]

        normalized = _normalize_video_segments(segments, 10000.0, fill_gaps=False)

        assert [(s.start_ms, s.end_ms, s.speed) for s in normalized] == [
            (5000.0, 8000.0, 2.0),
            (1000.0, 2000.0, 1.0),
        ]

    def test_legacy_segments_fill_gaps(self):
        segments = [VideoSegment.create(3000.0, 5000.0, 1.0)]

        normalized = _normalize_video_segments(segments, 7000.0, fill_gaps=True)

        assert [(s.start_ms, s.end_ms) for s in normalized] == [
            (0.0, 3000.0),
            (3000.0, 5000.0),
            (5000.0, 7000.0),
        ]


class TestOverlayCoordinateMapping:
    def test_click_coordinate_follows_zoom_crop(self):
        keyframes = [
            ZoomKeyframe.create(timestamp=0.0, zoom=2.0, x=0.25, y=0.25, duration=0.0)
        ]

        rel_x, rel_y = _map_zoomed_relative_point(0.25, 0.25, 100.0, keyframes)

        assert rel_x == pytest.approx(0.5)
        assert rel_y == pytest.approx(0.5)

    def test_click_coordinate_uses_export_zoom_easing(self):
        keyframes = [
            ZoomKeyframe.create(timestamp=0.0, zoom=2.0, x=0.25, y=0.25, duration=1000.0)
        ]

        rel_x, rel_y = _map_zoomed_relative_point(0.25, 0.25, 500.0, keyframes)

        eased = 6.0 * pow(0.5, 5.0) - 15.0 * pow(0.5, 4.0) + 10.0 * pow(0.5, 3.0)
        zoom = 1.0 + (2.0 - 1.0) * eased
        pan = 0.5 + (0.25 - 0.5) * eased
        visible = 1.0 / zoom
        crop = max(0.0, min(1.0 - visible, pan - visible / 2.0))
        expected = (0.25 - crop) * zoom

        assert rel_x == pytest.approx(expected)
        assert rel_y == pytest.approx(expected)

    def test_click_coordinate_remains_outside_when_zoom_crop_hides_it(self):
        keyframes = [
            ZoomKeyframe.create(timestamp=0.0, zoom=1.5, x=0.6, y=0.6, duration=0.0)
        ]

        rel_x, rel_y = _map_zoomed_relative_point(0.01, 0.01, 100.0, keyframes)

        assert rel_x < 0.0
        assert rel_y < 0.0


def test_click_and_cursor_are_composited_in_video_space_before_canvas(monkeypatch, tmp_path) -> None:
    cursor_path = tmp_path / "cursor.png"
    Image.new("RGBA", (16, 20), (255, 255, 255, 255)).save(cursor_path)
    monkeypatch.setattr(video_exporter, "generate_device_frame_png", lambda *a, **k: "frame.png")
    monkeypatch.setattr(video_exporter, "generate_click_png", lambda *a, **k: "click.png")
    monkeypatch.setattr(video_exporter, "generate_cursor_png", lambda *a, **k: str(cursor_path))

    plan = VideoExporter()._build_filtergraph(
        bg_preset=BACKGROUND_PRESETS[0],
        frame_preset=_no_frame_preset(),
        target_resolution=(100, 100),
        duration_ms=1000.0,
        frame_timestamps=None,
        keyframes=[ZoomKeyframe.create(timestamp=0.0, zoom=2.0, x=0.5, y=0.5, duration=0.0)],
        mouse_track=[MousePosition(x=10.0, y=10.0, timestamp=500.0)],
        click_events=[ClickEvent(x=50.0, y=50.0, timestamp=500.0)],
        click_preset=video_exporter.DEFAULT_CLICK_EFFECT,
        monitor_rect={"left": 0, "top": 0, "width": 100, "height": 100},
        video_segments=None,
        timeline_frames=None,
        highlights=None,
        src_w=100,
        src_h=100,
        src_fps=30.0,
        total_sec=1.0,
        is_cfr=True,
    )

    try:
        graph = plan.filtergraph
        click_pos = graph.index("[s0vid][cl0]overlay=")
        cursor_pos = graph.index("sendcmd=f=")
        canvas_pos = graph.index("[s0bg][s0cursor]overlay=")
        frame_pos = graph.index("[s0comp0][s0frameheld]overlay=")
        assert click_pos < cursor_pos < canvas_pos < frame_pos
        assert "if(lt(t," not in graph
        assert len(plan.cursor_motion_tracks) == 1
        assert plan.cursor_motion_tracks[0].command_path in plan.temp_files
        assert "overlay@cursor0" in graph

        cmd = VideoExporter()._build_ffmpeg_command(
            ffmpeg="ffmpeg",
            input_path="source.mp4",
            output_path="output.mp4",
            graph_path="graph.txt",
            plan=plan,
            encoder_id="libx264",
        )
        assert "-framerate" not in cmd
    finally:
        VideoExporter()._cleanup_temp_files(plan.temp_files, plan.temp_dirs)


def test_canvas_text_is_composited_after_device_frame(monkeypatch) -> None:
    monkeypatch.setattr(video_exporter, "generate_device_frame_png", lambda *a, **k: "frame.png")
    monkeypatch.setattr(video_exporter, "generate_text_annotation_png", lambda *a, **k: "text.png")
    annotation = TextAnnotation.create(
        100.0,
        900.0,
        x=0.8,
        y=0.1,
        text="Canvas label",
    )

    plan = VideoExporter()._build_filtergraph(
        bg_preset=BACKGROUND_PRESETS[0],
        frame_preset=_no_frame_preset(),
        target_resolution=(100, 100),
        duration_ms=1000.0,
        frame_timestamps=None,
        keyframes=[ZoomKeyframe.create(timestamp=0.0, zoom=2.0, x=0.2, y=0.2, duration=0.0)],
        mouse_track=[],
        click_events=[],
        click_preset=video_exporter.DEFAULT_CLICK_EFFECT,
        monitor_rect={"left": 0, "top": 0, "width": 100, "height": 100},
        video_segments=None,
        timeline_frames=None,
        highlights=None,
        text_annotations=[annotation],
        src_w=100,
        src_h=100,
        src_fps=30.0,
        total_sec=1.0,
        is_cfr=True,
    )

    graph = plan.filtergraph
    frame_pos = graph.index("[s0comp0][s0frameheld]overlay=")
    text_pos = graph.index("[s0framed][2:v]overlay=")
    retime_pos = graph.index("[s0text0]setpts=")
    assert frame_pos < text_pos < retime_pos
    assert plan.text_annotation_img_paths == ["text.png"]


def test_cursor_asset_uses_custom_bitmap_and_corrupt_asset_falls_back() -> None:
    source_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    source_path = source_file.name
    source_file.close()
    corrupt_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    corrupt_path = corrupt_file.name
    corrupt_file.write(b"not an image")
    corrupt_file.close()
    outputs = []
    try:
        Image.new("RGBA", (12, 18), (255, 0, 0, 255)).save(source_path)
        custom_output = generate_cursor_png(source_path)
        fallback_output = generate_cursor_png(corrupt_path)
        outputs.extend([custom_output, fallback_output])

        with Image.open(custom_output) as custom:
            assert custom.size == (20, 26)
        with Image.open(fallback_output) as fallback:
            # Built-in cursors stay supersampled until the preview/export
            # renderer reduces them with a high-quality filter.
            assert fallback.size == (112, 152)
    finally:
        for path in [source_path, corrupt_path, *outputs]:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass


@pytest.mark.parametrize(
    "style_id",
    [
        "filled_arrow",
        "tangerine_wedge",
        "cobalt_arrow",
        "orchid_pointer",
        "coral_pointer",
        "signal_pointer",
        "aqua_pointer",
        "periwinkle_pointer",
        "prism_pointer",
        "aurora_pointer",
        "ember_head",
        "sapphire_head",
        "lilac_head",
        "ruby_head",
        "mint_head",
        "violet_head",
        "cyan_wedge",
        "outline_soft",
        "graphite_stripe",
        "lavender_soft",
    ],
)
def test_svg_cursor_is_rasterized_from_vector_source_for_export(style_id) -> None:
    output = generate_cursor_png("", style_id)
    try:
        with Image.open(output) as image:
            preset = get_cursor_preset(style_id)
            assert image.size == (preset.width, preset.height)
            assert image.getbbox() is not None
    finally:
        try:
            os.remove(output)
        except FileNotFoundError:
            pass


class TestGeometryResult:
    """Test the GeometryResult dataclass."""

    def test_dataclass_construction(self):
        """GeometryResult should construct with required fields."""
        # Create dummy numpy arrays
        canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mask = np.zeros((1080, 1920), dtype=np.uint8)
        bg = np.zeros((1080, 1920, 3), dtype=np.uint8)

        result = GeometryResult(
            scr_x=100,
            scr_y=100,
            scr_w=1720,
            scr_h=880,
            base_canvas=canvas,
            screen_mask=mask,
            device_mask_u8=None,
            bg=bg,
        )
        assert result.scr_x == 100
        assert result.scr_y == 100
        assert result.scr_w == 1720
        assert result.scr_h == 880
        assert result.base_canvas.shape == (1080, 1920, 3)
        assert result.screen_mask.shape == (1080, 1920)
        assert result.device_mask_u8 is None
        assert result.bg.shape == (1080, 1920, 3)


class TestIntegrationScenarios:
    """Integration tests combining geometry computation with realistic scenarios."""

    def test_4k_export_with_laptop_frame(self):
        """4K export with laptop frame should compute valid geometry."""
        gc = GeometryComputer(
            canvas_w=3840,
            canvas_h=2160,
            src_w=2560,
            src_h=1440,
            frame_preset=DEFAULT_FRAME,
        )
        geom = gc.compute()

        # Should fit within 4K canvas
        assert geom["scr_x"] + geom["scr_w"] <= 3840
        assert geom["scr_y"] + geom["scr_h"] <= 2160
        # Device should be visible (not zero-sized)
        assert geom["dev_w"] > 0
        assert geom["dev_h"] > 0

    def test_gif_export_small_canvas(self):
        """GIF export with smaller canvas (800x600) should work."""
        gc = GeometryComputer(
            canvas_w=800,
            canvas_h=600,
            src_w=1920,
            src_h=1080,
            frame_preset=_no_frame_preset(),
        )
        geom = gc.compute()

        # Video should be downscaled to fit
        assert geom["scr_w"] <= 800
        assert geom["scr_h"] <= 600

    def test_all_frame_presets_valid(self):
        """All frame presets should produce valid geometry."""
        for preset in FRAME_PRESETS:
            gc = GeometryComputer(
                canvas_w=1920,
                canvas_h=1080,
                src_w=1600,
                src_h=900,
                frame_preset=preset,
            )
            geom = gc.compute()

            # Basic validity checks
            assert geom["scr_w"] > 0
            assert geom["scr_h"] > 0
            assert geom["scr_x"] >= 0
            assert geom["scr_y"] >= 0
            assert geom["scr_x"] + geom["scr_w"] <= 1920
            assert geom["scr_y"] + geom["scr_h"] <= 1080
