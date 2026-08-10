"""Final audio-bus composition for video exports.

Timeline/source audio is prepared by ``video_exporter`` alongside the edited
video segments. This module owns the later, independent composition stage:
voiceover normalization, global background music, ducking, and the final mix.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Protocol

from .models import BackgroundMusic
from .music_registry import resolve_music_asset

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AudioInputSpec:
    """One external audio input and the CLI options that must precede it."""

    path: str
    stream_loop: bool = False


class _AudioPlan(Protocol):
    filtergraph: str
    output_total_sec: float
    has_audio_output: bool
    audio_output_node: str
    has_source_audio: bool
    has_voiceover_audio: bool
    voiceover_audio_paths: list[str]
    audio_input_specs: list[AudioInputSpec]


def static_input_count(plan: object) -> int:
    """Return the first free FFmpeg input index after all visual assets."""

    return (
        2
        + (1 if getattr(plan, "click_img_path", None) else 0)
        + (1 if getattr(plan, "cursor_img_path", None) else 0)
        + len(getattr(plan, "highlight_img_paths", []))
        + len(getattr(plan, "text_annotation_img_paths", []))
        + len(getattr(plan, "timeline_frame_img_paths", []))
        + (1 if getattr(plan, "background_img_path", "") else 0)
        + len(getattr(plan, "transition_text_img_paths", []))
    )


class AudioMixBuilder:
    """Attach voiceover and global music to an existing source-audio graph."""

    def attach(
        self,
        plan: _AudioPlan,
        *,
        voiceover_rows: Iterable[tuple[str, int, float]] = (),
        background_music: BackgroundMusic | None = None,
    ) -> None:
        duration = max(float(plan.output_total_sec or 0.0), 0.0)
        rows = list(voiceover_rows)
        input_specs = [AudioInputSpec(path) for path, _, _ in rows]
        music_path = ""
        if background_music is not None:
            music_path = resolve_music_asset(
                background_music.asset_id,
                background_music.asset_path,
            )
            if not music_path:
                logger.warning(
                    "Background music asset is unavailable; exporting without it: %s",
                    background_music.asset_id or background_music.asset_path,
                )
            else:
                input_specs.append(AudioInputSpec(music_path, stream_loop=True))

        if not rows and not music_path:
            return
        if duration <= 0.0:
            logger.warning("Audio mix skipped because the edited output duration is zero.")
            return

        first_input = static_input_count(plan)
        lines: list[str] = []
        source_node = plan.audio_output_node if plan.has_audio_output else ""
        voice_bus = ""

        voice_nodes: list[str] = []
        for index, (_, delay_ms, volume) in enumerate(rows):
            node = f"voice{index}"
            lines.append(
                f"[{first_input + index}:a]aresample=48000,"
                "aformat=sample_rates=48000:channel_layouts=stereo,"
                "asetpts=PTS-STARTPTS,"
                f"volume={max(0.0, min(float(volume), 3.0)):.6f},"
                f"adelay={max(0, int(delay_ms))}:all=1,apad,"
                f"atrim=duration={duration:.6f},asetpts=PTS-STARTPTS[{node}]"
            )
            voice_nodes.append(node)

        if voice_nodes:
            voice_bus = "voicebus"
            if len(voice_nodes) == 1:
                lines.append(f"[{voice_nodes[0]}]anull[{voice_bus}]")
            else:
                inputs = "".join(f"[{node}]" for node in voice_nodes)
                lines.append(
                    f"{inputs}amix=inputs={len(voice_nodes)}:duration=longest:"
                    f"dropout_transition=0:normalize=0,apad,atrim=duration={duration:.6f},"
                    f"asetpts=PTS-STARTPTS[{voice_bus}]"
                )

        voice_play_node = voice_bus
        voice_sidechain_node = ""
        if music_path and voice_bus and background_music and background_music.enable_ducking:
            voice_play_node = "voiceplay"
            voice_sidechain_node = "voicesidechain"
            lines.append(f"[{voice_bus}]asplit=2[{voice_play_node}][{voice_sidechain_node}]")

        music_node = ""
        if music_path and background_music is not None:
            music_input = first_input + len(rows)
            fade = min(2.0, duration / 4.0)
            fade_out_start = max(0.0, duration - fade)
            lines.append(
                f"[{music_input}:a]aresample=48000,"
                "aformat=sample_rates=48000:channel_layouts=stereo,"
                "asetpts=PTS-STARTPTS,"
                f"volume={max(0.0, min(float(background_music.volume), 1.0)):.6f},"
                f"atrim=duration={duration:.6f},"
                f"afade=t=in:st=0:d={fade:.6f},"
                f"afade=t=out:st={fade_out_start:.6f}:d={fade:.6f}[musicbase]"
            )
            music_node = "musicbase"
            if voice_sidechain_node:
                music_node = "musicducked"
                lines.append(
                    f"[musicbase][{voice_sidechain_node}]"
                    "sidechaincompress=threshold=0.025:ratio=8:attack=20:release=350"
                    f"[{music_node}]"
                )

        mix_nodes = [node for node in (source_node, voice_play_node, music_node) if node]
        if len(mix_nodes) == 1:
            lines.append(
                f"[{mix_nodes[0]}]apad,atrim=duration={duration:.6f},"
                "asetpts=PTS-STARTPTS[aout]"
            )
        else:
            inputs = "".join(f"[{node}]" for node in mix_nodes)
            lines.append(
                f"{inputs}amix=inputs={len(mix_nodes)}:duration=longest:"
                f"dropout_transition=0:normalize=0,alimiter=limit=0.95,"
                f"apad,atrim=duration={duration:.6f},"
                "asetpts=PTS-STARTPTS[aout]"
            )

        plan.filtergraph = f"{plan.filtergraph};\n" + ";\n".join(lines)
        plan.voiceover_audio_paths = [path for path, _, _ in rows]
        plan.audio_input_specs = input_specs
        plan.has_voiceover_audio = bool(rows)
        plan.has_audio_output = True
        plan.audio_output_node = "aout"
