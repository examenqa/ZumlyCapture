# Zumly Capture

Zumly Capture is a standalone Windows screenshot and screen-recording app
derived from selected Zumly capture components. Its distinguishing recording
feature is optional, automatic Smart Zoom after capture.

The product intentionally excludes Zumly's video editor and manual timeline.
The copied `zumly/` tree remains a transitional extraction seed while its
capture components are moved behind the standalone `zumly_capture` package.

## Current status

Phases 1 through 4 establish an independent repository, the complete capture
surface, and optional Smart Zoom post-processing. The tray can take monitor,
active-window, and selected-region
screenshots; record monitors, windows, and regions; run delayed screenshots and
recording countdowns; and use configurable global shortcuts. Optional
DirectShow microphone or loopback tracks are trimmed across recording pauses
before they are muxed into the result.

When Smart Zoom is enabled, every eligible click cluster contributes to a
continuous zoom/pan plan—there is no chain-length or click-count cap. The
pause-free cursor/click telemetry drives the render, with optional cursor and
click-indicator layers. The tray reports render progress and exposes a Cancel
Smart Zoom action. Cancellation or an FFmpeg failure falls back to publishing
the complete unprocessed recording, so post-processing cannot discard a
successful capture.

The recording worker publishes a playable MP4 without launching the Zumly
editor or exporter. The tray exposes actions to open, copy, or reveal the last
capture while retaining pause/resume support. Hardware WGC remains available
through a NumPy-free native adapter, with a NumPy-free GDI fallback.

Each successful MP4 receives a sibling `*.zumly-capture.json` manifest. The
manifest preserves monitor geometry, timing, pause boundaries, mouse/click
telemetry, frame cadence, capture diagnostics, and the final Smart Zoom state
and generated keyframes.

Copied editor modules remain in the transitional seed tree, but they are no
longer imported or launched by the capture worker or tray completion flow.

The package specification intentionally excludes NumPy and OpenCV. Active WGC,
GDI, screenshot, and Smart Zoom paths do not require either dependency; copied
inactive editor helpers remain transitional extraction work.

## Development

Zumly Capture targets CPython 3.13 on Windows x64. Python 3.14 is currently
excluded because the copied Windows/Qt stack crashes under that runtime.

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest
.venv\Scripts\python -m zumly_capture
```

The default pytest target is the standalone `tests/` suite. Copied seed tests
remain under `zumly/tests/` and must be invoked explicitly while they are
classified and migrated.

## Packaging baseline

```powershell
.venv\Scripts\python -m PyInstaller --noconfirm zumly_capture.spec
```

The resulting application directory is `dist/zumly-capture`. The Inno Setup
definition in `zumly_capture_setup.iss` uses an installer identity distinct
from Zumly, allowing both applications to coexist.

See [NOTICE.md](NOTICE.md) for derivation and attribution notes.
