# Zumly Capture

Zumly Capture is a standalone Windows screenshot and screen-recording app
derived from selected Zumly capture components. Its distinguishing recording
feature is optional, automatic Smart Zoom after capture.

The product intentionally excludes Zumly's video editor and manual timeline.
The copied `zumly/` tree remains a transitional extraction seed while its
capture components are moved behind the standalone `zumly_capture` package.

## Current status

Phases 1 through 5 establish an independent repository, the complete capture
surface, post-capture preview and annotation, and Smart Zoom post-processing.
The tray can take monitor,
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

Every successful screenshot or recording opens in a compact preview. Image
previews include pen, highlighter, arrow, rectangle, text, color, undo, copy,
and one explicit Save action; text is entered directly on the canvas. Recording
previews remain editing-free but include a Save as selector for MP4 or animated
GIF. The preview retains its canonical MP4 source until it closes, so both
formats can be saved and previewed during the same session without deleting the
other output. When automatic Smart Zoom was rendered, the user may remove the
entire effect before saving, but cannot place or edit individual zooms. Show in
Folder becomes available after Save, and the redundant preview Open action is
omitted. Automatic Smart Zoom is enabled by default and is also available as a
quick toggle in the tray menu.

Default global shortcuts use the `Ctrl+Alt+number` family:

- `Ctrl+Alt+1`, `2`, `3`: monitor, active-window, and region screenshots.
- `Ctrl+Alt+4`, `5`, `6`: monitor, window, and region recordings.
- `Ctrl+Alt+9`: pause or resume recording.
- `Ctrl+Alt+0`: stop recording or cancel post-processing.

The recording worker publishes a playable MP4 without launching the Zumly
editor or exporter. The tray exposes actions to open, copy, or reveal the last
capture while retaining pause/resume support. Hardware WGC remains available
through a NumPy-free native adapter, with a NumPy-free GDI fallback.

Each successful recording publishes only its directly playable media file. Capture
telemetry and Smart Zoom processing data remain internal and do not leave JSON
sidecars in the user's output folder.

Settings defines the default recording output, while each recording preview can
switch its Save as choice between MP4 and animated GIF. MP4 remains the default
and supports configured audio devices. GIF output loops automatically, omits
audio, uses a palette-based 15 FPS encoder, and limits the longest edge to 1280
pixels so the result remains practical to share. Smart Zoom is applied before
GIF conversion and can still be removed from the preview.

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

Every push to `main` runs the Windows test and packaging workflow, then updates
the rolling `continuous` prerelease with a fresh `ZumlyCaptureSetup.exe` and
SHA-256 checksum. Numbered releases such as `v0.5.5` remain immutable.

See [NOTICE.md](NOTICE.md) for derivation and attribution notes.
