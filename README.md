# Zumly Capture

Zumly Capture is a standalone Windows screenshot and screen-recording app
derived from selected Zumly capture components. Its distinguishing recording
feature is optional, automatic Smart Zoom after capture.

The product intentionally excludes Zumly's video editor and manual timeline.
The copied `zumly/` tree remains a transitional extraction seed while its
capture components are moved behind the standalone `zumly_capture` package.

## Current status

Phase 1 establishes an independent repository, package identity, settings and
single-instance namespaces, dependency contract, tests, and packaging baseline.
The old editor handoff and copied editor modules remain until Phase 2 replaces
the recording session bridge with a direct capture result flow.

The package specification intentionally excludes NumPy and OpenCV. Some copied
capture fallbacks still import NumPy and therefore remain transitional; they
must be replaced before the packaged application is considered release-ready.

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
