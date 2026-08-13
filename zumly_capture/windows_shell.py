"""Small Windows shell helpers shared by capture surfaces."""

from __future__ import annotations

import os
from pathlib import Path


def reveal_in_folder(capture_path: str) -> str:
    """Open the actual parent folder for a saved capture.

    Passing ``/select,<path with spaces>`` as one quoted Explorer argument can
    be misparsed as a location request and fall back to Documents. Opening the
    already-resolved parent directory is deterministic and matches the action
    label even for folders such as ``Videos/Zumly Capture``.
    """
    target = Path(capture_path).expanduser().resolve(strict=False)
    folder = target.parent
    if not folder.is_dir():
        raise FileNotFoundError(str(folder))
    os.startfile(str(folder))
    return str(folder)
