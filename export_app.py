import argparse
import sys
import logging
import json
from zumly.app.video_exporter import VideoExporter
from zumly.app.models import RecordingSession
from zumly.app.backgrounds import DEFAULT_PRESET as DEFAULT_BG, PRESETS as BG_PRESETS
from zumly.app.frames import DEFAULT_FRAME, FRAME_PRESETS
from zumly.app.models import DEFAULT_CLICK_EFFECT, CLICK_EFFECT_PRESETS

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def _preset_by_name(presets, name: str, default):
    """Resolve a persisted preset name against a preset list."""
    return next((preset for preset in presets if preset.name == name), default)

def main() -> int:
    parser = argparse.ArgumentParser(description="Zumly Headless Exporter")
    parser.add_argument("--project", type=str, required=True, help="Path to the project JSON")
    args = parser.parse_args()

    try:
        with open(args.project, "r", encoding="utf-8") as f:
            data = json.load(f)

        session = RecordingSession.from_json(json.dumps(data))
        out_path = data.get("outPath", "")
        video_path = data.get("videoPath", "")
        monitor_rect = data.get("monitorRect", {})
        actual_fps = data.get("actualFps", 30.0)
        encoder_id = data.get("encoderId", "libx264") or "libx264"

        if not out_path or not video_path:
            raise ValueError("Project JSON missing outPath or videoPath")

        bg_preset = _preset_by_name(BG_PRESETS, session.background_id or "", DEFAULT_BG)
        frame_preset = _preset_by_name(FRAME_PRESETS, session.frame_id or "", DEFAULT_FRAME)
        click_preset = _preset_by_name(
            CLICK_EFFECT_PRESETS, session.click_effect_id or "", DEFAULT_CLICK_EFFECT
        )

        target_res = None
        if session.output_dimensions and isinstance(session.output_dimensions, list) and len(session.output_dimensions) == 2:
            target_res = tuple(session.output_dimensions)

        def on_progress(p: float) -> None:
            sys.stdout.write(f"\rExport progress: {p*100:.1f}%")
            sys.stdout.flush()

        def on_finished(path: str) -> None:
            print(f"\nExport finished: {path}")

        def on_error(err: str) -> None:
            print(f"\nExport error: {err}")

        exporter = VideoExporter(
            progress_cb=on_progress,
            finished_cb=on_finished,
            error_cb=on_error,
        )
        result = exporter.export(
            input_path=video_path,
            output_path=out_path,
            keyframes=session.keyframes,
            actual_fps=actual_fps,
            mouse_track=session.mouse_track,
            monitor_rect=monitor_rect,
            bg_preset=bg_preset,
            frame_preset=frame_preset,
            target_resolution=target_res,
            click_events=session.click_events,
            click_preset=click_preset,
            duration_ms=session.duration,
            frame_timestamps=session.frame_timestamps,
            is_cfr=session.is_cfr,
            trim_start_ms=session.trim_start_ms,
            trim_end_ms=session.trim_end_ms,
            encoder_id=encoder_id,
            voiceover_segments=session.voiceover_segments,
            video_segments=session.video_segments,
            timeline_frames=session.timeline_frames,
            screen_transitions=session.screen_transitions,
            highlights=session.highlights,
            text_annotations=session.text_annotations,
            timeline_overlays=session.timeline_overlays,
            cursor_asset_path=session.cursor_asset_path,
            cursor_style_id=session.cursor_style_id,
            cursor_hotspot=session.cursor_hotspot,
            cursor_scale=session.cursor_scale,
            canvas_layout_scenes=session.canvas_layout_scenes,
            explainer_scenes=session.explainer_scenes,
            background_music=session.background_music,
            wait=True,
        )
        if result is None or not result.success:
            logging.error(
                "Headless export failed (FFmpeg exit code %s): %s",
                getattr(result, "ffmpeg_exit_code", -1),
                getattr(result, "error_message", "No export result"),
            )
            return 1
        if result.fallback_used:
            logging.warning(result.error_message)
        logging.info("Headless export done.")
        return 0
    except KeyboardInterrupt:
        logging.error("Export cancelled by user")
        return 130
    except Exception as exc:
        logging.exception("Headless export failed: %s", exc)
        return 1

if __name__ == "__main__":
    sys.exit(main())
