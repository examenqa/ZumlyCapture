"""Shared pytest fixtures for Zumly tests.

Pytest's default temporary-directory cleanup is deliberately aggressive. On
Windows, Qt and FFmpeg can release a handle a few milliseconds after a test
returns, so the suite owns a small bounded retry layer here.
"""

import logging
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)


def pytest_configure(config: pytest.Config) -> None:
    """Keep pytest media fixtures out of the repository's locked folders."""
    root = Path(tempfile.gettempdir()) / "zumly_pytest_sandbox"
    base = root / f"pytest-{os.getpid()}"
    base.mkdir(parents=True, exist_ok=True)
    config.option.basetemp = str(base)
    config._zumly_basetemp = base


def _remove_tree_with_retries(path: Path, attempts: int = 5, delay: float = 0.2) -> bool:
    """Remove a temporary tree despite short-lived Windows sharing locks."""
    target = Path(path)
    for attempt in range(attempts):
        try:
            shutil.rmtree(target)
            return True
        except FileNotFoundError:
            return True
        except (PermissionError, OSError) as exc:
            if attempt == attempts - 1:
                logger.warning("Could not remove pytest temp tree %s: %s", target, exc)
                return False
            time.sleep(delay)
    return not target.exists()


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest):
    """Provide an isolated per-test path without pytest's locked tmp plugin."""
    base = Path(tempfile.gettempdir()) / "zumly_pytest_sandbox" / f"pytest-{os.getpid()}"
    base.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(
        char if char.isalnum() or char in "-_" else "_"
        for char in request.node.name
    )[:80]
    path = base / f"{safe_name}-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        _remove_tree_with_retries(path)


def pytest_unconfigure(config: pytest.Config) -> None:
    """Clean the isolated base after pytest's own tmpdir plugin has finished."""
    base_temp = getattr(config, "_zumly_basetemp", None)
    if base_temp is not None:
        _remove_tree_with_retries(Path(base_temp))

from app.models import (
    MousePosition,
    KeyEvent,
    ClickEvent,
    ZoomKeyframe,
    RecordingSession,
)
from app.backgrounds import BackgroundPreset, PRESETS as BACKGROUND_PRESETS
from app.frames import FramePreset, FRAME_PRESETS, DEFAULT_FRAME


# ── Monitor rect ────────────────────────────────────────────────────

@pytest.fixture
def monitor_rect() -> dict:
    """A 1920×1080 monitor at origin."""
    return {"left": 0, "top": 0, "width": 1920, "height": 1080}


# ── Mouse track helpers ────────────────────────────────────────────

@pytest.fixture
def simple_mouse_track() -> list[MousePosition]:
    """Short straight-line mouse track (20 samples, 320ms)."""
    return [
        MousePosition(x=100.0 + i * 10, y=200.0, timestamp=i * 16.0)
        for i in range(20)
    ]


@pytest.fixture
def long_mouse_track() -> list[MousePosition]:
    """10-second mouse track with a fast→slow settlement at 5s."""
    track: list[MousePosition] = []
    for i in range(625):  # ~10s at 16ms intervals
        t = i * 16.0
        if t < 4000:
            # Slow drift
            x = 500.0 + i * 0.5
            y = 500.0
        elif t < 5000:
            # Fast move
            x = 500.0 + (t - 4000) * 1.0
            y = 500.0 + (t - 4000) * 0.5
        else:
            # Settle
            x = 1500.0
            y = 1000.0
        track.append(MousePosition(x=x, y=y, timestamp=t))
    return track


# ── Key / click event helpers ──────────────────────────────────────

@pytest.fixture
def typing_burst() -> list[KeyEvent]:
    """Rapid typing burst at ~3s (20 keys over 1s)."""
    return [KeyEvent(timestamp=3000.0 + i * 50) for i in range(20)]


@pytest.fixture
def click_cluster() -> list[ClickEvent]:
    """3 clicks near (960, 540) around 6s."""
    return [
        ClickEvent(x=950, y=530, timestamp=6000),
        ClickEvent(x=960, y=540, timestamp=6200),
        ClickEvent(x=970, y=550, timestamp=6400),
    ]


# ── Zoom keyframes ─────────────────────────────────────────────────

@pytest.fixture
def zoom_in_out_pair() -> list[ZoomKeyframe]:
    """A simple zoom-in / zoom-out keyframe pair."""
    return [
        ZoomKeyframe.create(timestamp=1000, zoom=1.5, x=0.3, y=0.4, duration=600, reason="Test zoom in"),
        ZoomKeyframe.create(timestamp=4000, zoom=1.0, x=0.5, y=0.5, duration=1200, reason="Test zoom out"),
    ]


# ── Recording session ──────────────────────────────────────────────

@pytest.fixture
def sample_session(simple_mouse_track: list[MousePosition]) -> RecordingSession:
    """Minimal recording session for serialization tests."""
    return RecordingSession(
        id="test-session-001",
        start_time=0.0,
        duration=320.0,
        mouse_track=simple_mouse_track,
        keyframes=[
            ZoomKeyframe.create(timestamp=100, zoom=1.5, x=0.3, y=0.4, duration=600),
        ],
        key_events=[KeyEvent(timestamp=50), KeyEvent(timestamp=150)],
        click_events=[ClickEvent(x=110, y=200, timestamp=80)],
        frame_timestamps=[i * 16.0 for i in range(20)],
        trim_start_ms=32.0,
        trim_end_ms=288.0,
    )


# ── Presets ─────────────────────────────────────────────────────────

@pytest.fixture
def sample_bg_preset() -> BackgroundPreset:
    return BACKGROUND_PRESETS[0]


@pytest.fixture
def sample_frame_preset() -> FramePreset:
    return DEFAULT_FRAME
