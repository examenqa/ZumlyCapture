# Screen, Window, and Region Recording

Zumly Capture records a monitor, an application window, or a selected region.
The resulting MP4 is saved directly; the full Zumly editor and manual timeline
are intentionally not part of this application.

## Starting and stopping

Use the tray menu or the default global shortcuts:

- `Ctrl+Alt+4`: record the configured monitor.
- `Ctrl+Alt+5`: choose a window to record.
- `Ctrl+Alt+6`: select a region to record.
- `Ctrl+Alt+9`: pause or resume.
- `Ctrl+Alt+0`: stop recording.

The shortcuts are configurable in Settings. A countdown runs before capture,
and the tray plus recording border show whether recording is active or paused.

## Automatic Smart Zoom

Smart Zoom is an optional Settings toggle and is enabled by default. It is not
a manual recording editor.

When enabled, Zumly Capture records click and cursor telemetry on the same
pause-aware active-time clock as the video. After recording stops, every
eligible click is analyzed and an automatic cursor-follow zoom plan is rendered
through FFmpeg. The tray shows post-processing progress, and cancelling that
step safely publishes the original unzoomed recording.

When Smart Zoom is disabled, the app skips analysis and publishes the recording
without the extra render.

## Post-capture preview

The recording preview supports playback only. It has no annotation tools,
timeline, or controls for adding, moving, or changing zooms.

If Smart Zoom was successfully applied, the preview offers one reversible
choice: **Remove automatic Smart Zoom**. Selecting it previews the original
recording; pressing **Save** replaces the zoomed result with that complete
unzoomed recording. Individual automatic zooms cannot be edited or removed.

After Save, **Show in folder** reveals the final MP4. **Copy** places the media
file on the clipboard.
