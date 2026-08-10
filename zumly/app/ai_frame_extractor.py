"""Small FFmpeg command builder for multimodal AI screenshot sampling."""


def build_mjpeg_frame_command(ffmpeg: str, video_path: str, timestamp_ms: float) -> list[str]:
    """Build a deterministic one-frame JPEG extraction command.

    Keeping this command construction separate makes the frame extraction
    contract testable without coupling it to Gemini transport or prompts.
    """
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, timestamp_ms) / 1000.0:.3f}",
        "-i",
        video_path,
        "-frames:v",
        "1",
        "-vf",
        "scale=960:960:force_original_aspect_ratio=decrease,scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-q:v",
        "4",
        "-strict",
        "unofficial",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "pipe:1",
    ]
