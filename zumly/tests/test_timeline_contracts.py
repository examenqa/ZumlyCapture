"""Explicit source/output time and ordered clip regressions."""

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from app.models import (
    CanvasLayoutScene,
    ExplainerScene,
    RecordingSession,
    ScreenTransition,
    TextAnnotation,
    TimelineFrame,
    VideoSegment,
)
from app.timeline import EditedTimelineMapper, ordered_video_segments
from app.signals import EditorEventBus
from app.widgets.editor_window import EditorWindow
from app.widgets.preview_widget import PreviewWidget
from app.widgets.timeline_widget import TimelineWidget, _fmt_precise


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_mapper_preserves_output_sequence_for_copied_source_ranges() -> None:
    first = VideoSegment.create(4000.0, 5000.0, sequence_index=0)
    middle = VideoSegment.create(0.0, 1000.0, speed=2.0, sequence_index=1)
    copied = VideoSegment.create(4000.0, 5000.0, sequence_index=2)

    mapper = EditedTimelineMapper(5000.0, [first, middle, copied])

    assert mapper.clip_output_bounds_ms(first.id) == pytest.approx((0.0, 1000.0))
    assert mapper.clip_output_bounds_ms(middle.id) == pytest.approx((1000.0, 1500.0))
    assert mapper.clip_output_bounds_ms(copied.id) == pytest.approx((1500.0, 2500.0))
    assert mapper.source_time_to_output_ms(4500.0) == pytest.approx(500.0)
    assert mapper.source_time_to_output_ms(4500.0, clip_id=copied.id) == pytest.approx(2000.0)
    assert mapper.output_time_to_source_ms(1750.0) == pytest.approx(4250.0)


def test_inserted_effect_anchor_targets_specific_copied_clip_occurrence() -> None:
    first = VideoSegment.create(4000.0, 5000.0, sequence_index=0)
    copied = VideoSegment.create(4000.0, 5000.0, sequence_index=1)
    anchored = TimelineFrame.create(4500.0, duration_ms=250.0, clip_id=copied.id)

    mapper = EditedTimelineMapper(
        5000.0,
        [first, copied],
        timeline_frames=[anchored],
    )
    span = mapper.synthetic_span(anchored.id)

    assert span is not None
    assert span.clip_id == copied.id
    assert span.output_start_ms == pytest.approx(1500.0)
    assert mapper.clip_id_at_output_time(span.output_start_ms) == copied.id
    assert TimelineFrame.from_dict(anchored.to_dict()).clip_id == copied.id


def test_recording_session_roundtrip_keeps_list_as_canonical_clip_order() -> None:
    clips = [
        VideoSegment.create(5000.0, 6000.0, sequence_index=0),
        VideoSegment.create(0.0, 1000.0, sequence_index=1),
    ]
    session = RecordingSession(
        id="ordered-clips",
        start_time=0.0,
        duration=6000.0,
        mouse_track=[],
        keyframes=[],
        video_segments=clips,
    )

    loaded = RecordingSession.from_json(session.to_json())

    assert [clip.id for clip in ordered_video_segments(loaded.video_segments)] == [
        clips[0].id,
        clips[1].id,
    ]
    assert [clip.sequence_index for clip in loaded.video_segments] == [0, 1]


def test_explainer_is_inserted_output_time_without_consuming_source() -> None:
    scene = ExplainerScene.create(
        2000.0,
        5000.0,
        "right",
        CanvasLayoutScene.create(2000.0, 5000.0, video_scale=0.58),
        TextAnnotation.create(2000.0, 5000.0, text="Pause and explain"),
    )

    mapper = EditedTimelineMapper(
        8000.0,
        [VideoSegment.create(0.0, 8000.0)],
        explainer_scenes=[scene],
    )
    span = mapper.synthetic_span(scene.id)

    assert span is not None
    assert span.kind == "explainer"
    assert span.output_start_ms == pytest.approx(2000.0)
    assert span.output_end_ms == pytest.approx(5000.0)
    assert mapper.output_duration_ms == pytest.approx(11000.0)
    assert mapper.output_time_to_source_ms(3500.0) == pytest.approx(2000.0)
    assert mapper.output_time_to_source_ms(6000.0) == pytest.approx(3000.0)


def test_same_boundary_explainer_owns_transition_incoming_frame_after_cut() -> None:
    clips = [
        VideoSegment.create(0.0, 132_618.0, sequence_index=0),
        VideoSegment.create(141_817.0, 230_314.0, sequence_index=1),
    ]
    transition = ScreenTransition.create(
        144_625.681,
        duration_ms=2000.0,
        outgoing_frame_ms=144_609.014,
        incoming_frame_ms=144_650.0,
    )
    scene = ExplainerScene.create(
        144_625.681,
        149_625.681,
        "right",
        CanvasLayoutScene.create(144_625.681, 149_625.681, video_scale=0.58),
        TextAnnotation.create(
            144_625.681,
            149_625.681,
            text="Freeze the post-transition recording frame",
        ),
    )

    mapper = EditedTimelineMapper(
        230_314.0,
        clips,
        screen_transitions=[transition],
        explainer_scenes=[scene],
    )
    transition_span = mapper.synthetic_span(transition.id)
    explainer_span = mapper.synthetic_span(scene.id)

    assert transition_span is not None
    assert explainer_span is not None
    assert explainer_span.output_start_ms == pytest.approx(
        transition_span.output_end_ms
    )
    assert mapper.synthetic_visual_source_ms(
        scene.id,
        scene.start_ms,
    ) == pytest.approx(144_650.0)
    assert mapper.output_time_to_source_ms(
        transition_span.output_end_ms
    ) == pytest.approx(144_650.0)
    assert mapper.output_time_to_source_ms(
        explainer_span.output_start_ms + 100.0
    ) == pytest.approx(144_650.0)


def test_global_output_clock_stays_synchronized_across_same_boundary_effects(
    qapp,
) -> None:
    class FakePlayer:
        def __init__(self) -> None:
            self.position_ms = 0
            self.play_calls = 0
            self.pause_calls = 0

        def position(self) -> int:
            return self.position_ms

        def setPosition(self, value: int) -> None:
            self.position_ms = int(value)

        def pause(self) -> None:
            self.pause_calls += 1

        def play(self) -> None:
            self.play_calls += 1

        def setPlaybackRate(self, _rate: float) -> None:
            pass

    class FakeEditor:
        def __init__(self) -> None:
            self.active_settings_panel_id = "screen_changes"
            self.screen_playhead_ms = -1.0
            self.explainer_playhead_ms = -1.0
            self.frame_playhead_ms = -1.0

        def set_screen_change_playhead(self, value: float) -> None:
            self.screen_playhead_ms = float(value)

        def set_explainer_insert_timestamp(
            self,
            value: float,
            _duration: float,
        ) -> None:
            self.explainer_playhead_ms = float(value)

        def set_frame_insert_timestamp(
            self,
            value: float,
            _duration: float,
        ) -> None:
            self.frame_playhead_ms = float(value)

    class FakeZoomEngine:
        @staticmethod
        def compute_at(_source_ms: float) -> tuple[float, float, float]:
            return 1.0, 0.5, 0.5

    clips = [
        VideoSegment.create(0.0, 4000.0, speed=2.0, sequence_index=0),
        VideoSegment.create(6000.0, 12_000.0, speed=2.0, sequence_index=1),
    ]
    transition = ScreenTransition.create(
        7000.0,
        duration_ms=400.0,
        outgoing_frame_ms=6966.667,
        incoming_frame_ms=7100.0,
    )
    explainer = ExplainerScene.create(
        7000.0,
        8000.0,
        "right",
        CanvasLayoutScene.create(7000.0, 8000.0, video_scale=0.58),
        TextAnnotation.create(7000.0, 8000.0, text="Shared boundary"),
    )
    later_frame = TimelineFrame.create(9000.0, duration_ms=500.0)
    mapper = EditedTimelineMapper(
        12_000.0,
        clips,
        timeline_frames=[later_frame],
        screen_transitions=[transition],
        explainer_scenes=[explainer],
    )

    preview = PreviewWidget()
    timeline = TimelineWidget()
    player = FakePlayer()
    editor = FakeEditor()
    window = EditorWindow.__new__(EditorWindow)
    window._preview = preview
    window._timeline = timeline
    window._editor = editor
    window._zoom_engine = FakeZoomEngine()
    window._session = RecordingSession(
        id="clock-contract",
        start_time=0.0,
        duration=12_000.0,
        mouse_track=[],
        keyframes=[],
        video_segments=clips,
        timeline_frames=[later_frame],
        screen_transitions=[transition],
        explainer_scenes=[explainer],
    )
    window._timeline_mapper = mapper
    window._insert_source_playhead_ms = 0.0
    window._insert_output_playhead_ms = 0.0

    preview._media_player = player
    preview._media_loaded = True
    preview._playing = True
    preview._video_duration_ms = 12_000.0
    preview._video_segments = clips
    preview._timeline_frames = [later_frame]
    preview._screen_transitions = [transition]
    preview._explainer_scenes = [explainer]
    preview._output_timeline = mapper
    preview._external_timeline_mapper = True
    preview._last_ready_source_ms = 7100.0
    timeline.set_timeline_mapper(mapper, 1)

    outgoing = QImage(64, 36, QImage.Format.Format_RGBA8888)
    outgoing.fill(QColor("#ef4444"))
    incoming = QImage(64, 36, QImage.Format.Format_RGBA8888)
    incoming.fill(QColor("#22c55e"))
    preview._transition_prefetches[transition.id] = {
        "outgoing": outgoing,
        "incoming": incoming,
        "incoming_target_ms": 7100.0,
    }

    output_positions: list[float] = []
    preview.playback_time_changed.connect(window._on_playback_time_changed)
    preview.output_playback_time_changed.connect(timeline.set_current_time)
    preview.output_playback_time_changed.connect(output_positions.append)

    def assert_synchronized(panel_value: float) -> None:
        output_ms = preview._output_playback_pos_ms
        assert timeline._track.current_time == pytest.approx(output_ms)
        assert timeline._time_display.current_text == _fmt_precise(output_ms)
        assert panel_value == pytest.approx(output_ms)
        assert window._insert_output_playhead_ms == pytest.approx(output_ms)

    # Normal retimed source immediately before the shared boundary.
    preview._playback_pos_ms = 6999.0
    preview._current_time_ms = 6999.0
    preview._output_playback_pos_ms = mapper.source_time_to_output_ms(6999.0)
    preview._emit_playback_positions()
    assert_synchronized(editor.screen_playhead_ms)

    # The one master progression method advances Screen Change elapsed state.
    assert preview._begin_screen_transition(transition)
    assert_synchronized(editor.screen_playhead_ms)
    preview._last_preview_tick = time.perf_counter() - 0.2
    preview._advance_playback()
    assert preview._playback_pos_ms == pytest.approx(7000.0)
    assert_synchronized(editor.screen_playhead_ms)

    preview._transition_decoder_ready = True
    preview._last_preview_tick = time.perf_counter() - 0.25
    preview._advance_playback()
    transition_span = mapper.synthetic_span(transition.id)
    assert transition_span is not None
    assert preview._output_playback_pos_ms == pytest.approx(
        transition_span.output_end_ms
    )

    # The immediately following Explainer uses the same output clock while
    # source decoding remains frozen at the transition's incoming frame.
    editor.active_settings_panel_id = "canvas_text"
    player.position_ms = 7100
    preview._advance_playback()
    explainer_span = mapper.synthetic_span(explainer.id)
    assert explainer_span is not None
    assert preview._active_explainer_scene_id == explainer.id
    assert preview._playback_pos_ms == pytest.approx(7000.0)
    assert player.position_ms == 7100
    assert_synchronized(editor.explainer_playhead_ms)

    preview._last_preview_tick = time.perf_counter() - 0.4
    preview._advance_playback()
    assert preview._playback_pos_ms == pytest.approx(7000.0)
    assert_synchronized(editor.explainer_playhead_ms)

    preview._last_preview_tick = time.perf_counter() - 0.7
    preview._advance_playback()
    assert preview._output_playback_pos_ms == pytest.approx(
        explainer_span.output_end_ms
    )
    assert preview._playback_pos_ms == pytest.approx(7100.0)
    assert_synchronized(editor.explainer_playhead_ms)

    # A later inserted effect still resolves against the same authoritative
    # mapper after the cut, speed change, transition, and Explainer.
    editor.active_settings_panel_id = "frames"
    player.position_ms = 9000
    preview._advance_playback()
    frame_span = mapper.synthetic_span(later_frame.id)
    assert frame_span is not None
    assert preview._active_timeline_frame_id == later_frame.id
    assert preview._output_playback_pos_ms == pytest.approx(
        frame_span.output_start_ms
    )
    assert_synchronized(editor.frame_playhead_ms)

    assert output_positions == sorted(output_positions)
    assert not preview._playback_timer.isActive()


def test_editor_publishes_one_mapper_instance_to_preview_and_timeline() -> None:
    class Consumer:
        _video_duration_ms = 8000.0

        def set_timeline_mapper(self, mapper, revision):
            self.mapper = mapper
            self.revision = revision

    window = EditorWindow.__new__(EditorWindow)
    window._session = RecordingSession(
        id="shared-mapper",
        start_time=0.0,
        duration=8000.0,
        mouse_track=[],
        keyframes=[],
        video_segments=[VideoSegment.create(0.0, 8000.0, speed=2.0)],
    )
    window._preview = Consumer()
    window._timeline = Consumer()
    window.event_bus = EditorEventBus()
    window.event_bus.timeline_mapping_changed.connect(
        window._preview.set_timeline_mapper
    )
    window.event_bus.timeline_mapping_changed.connect(
        window._timeline.set_timeline_mapper
    )
    window._timeline_mapping_revision = 0

    mapper = EditorWindow._publish_timeline_mapping(window)

    assert window._preview.mapper is mapper
    assert window._timeline.mapper is mapper
    assert window.mapping_revision == 1
    assert mapper.output_duration_ms == pytest.approx(4000.0)


def test_mapping_revision_changes_only_when_timeline_structure_changes() -> None:
    class Consumer:
        _video_duration_ms = 8000.0

        def set_timeline_mapper(self, mapper, revision):
            self.mapper = mapper
            self.revision = revision

    segment = VideoSegment.create(0.0, 8000.0)
    window = EditorWindow.__new__(EditorWindow)
    window._session = RecordingSession(
        id="mapper-revision",
        start_time=0.0,
        duration=8000.0,
        mouse_track=[],
        keyframes=[],
        video_segments=[segment],
    )
    window._preview = Consumer()
    window._timeline = Consumer()
    window._timeline_mapping_revision = 0

    first = EditorWindow._publish_timeline_mapping(window)
    unchanged = EditorWindow._publish_timeline_mapping(window)
    segment.speed = 2.0
    changed = EditorWindow._publish_timeline_mapping(window)

    assert unchanged is first
    assert changed is not first
    assert window.mapping_revision == 2
    assert window._preview.mapper is changed
    assert window._timeline.mapper is changed
