"""First-class screenshot capture and publication."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import tempfile
from typing import Any

import mss
from PIL import Image

from .identity import FILE_PREFIX


def foreground_window_handle() -> int:
    try:
        return int(ctypes.windll.user32.GetForegroundWindow())
    except (AttributeError, OSError):
        return 0


def capture_rect_image(rect: dict[str, Any]) -> Image.Image:
    area = {
        "left": int(rect["left"]),
        "top": int(rect["top"]),
        "width": int(rect["width"]),
        "height": int(rect["height"]),
    }
    if area["width"] <= 0 or area["height"] <= 0:
        raise ValueError("Screenshot area must have positive dimensions")
    with mss.mss() as capture:
        shot = capture.grab(area)
    return Image.frombytes("RGB", shot.size, shot.rgb)


def publish_screenshot(
    rect: dict[str, Any],
    output_path: str | os.PathLike[str],
    image_format: str = "png",
) -> str:
    target = Path(output_path).resolve()
    if target.exists():
        raise FileExistsError(f"Screenshot already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    fmt = "JPEG" if image_format.lower() in {"jpg", "jpeg"} else "PNG"
    suffix = ".jpg" if fmt == "JPEG" else ".png"
    temp_path = ""
    try:
        image = capture_rect_image(rect)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(target.parent),
            prefix=f"{FILE_PREFIX}_screenshot_",
            suffix=suffix,
            delete=False,
        ) as handle:
            temp_path = handle.name
        image.save(temp_path, format=fmt, quality=95)
        with open(temp_path, "r+b") as handle:
            os.fsync(handle.fileno())
        os.link(temp_path, target)
        os.remove(temp_path)
        temp_path = ""
        return str(target)
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass
