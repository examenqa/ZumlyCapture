# Design QA — Recording Cursor and Click Ripple

- Source visual truth: `C:\Users\vali2\AppData\Local\Temp\codex-clipboard-e17c025e-575a-499c-8ba8-cb5b85c2793f.png`
- Rendered implementation: `C:\MyApps\Zumly Capture\docs\qa\recording-cursor-ripple-implementation.png`
- Side-by-side comparison: `C:\MyApps\Zumly Capture\docs\qa\recording-cursor-ripple-comparison.png`
- Viewport: 640 × 360 synthetic screen recording at 30 fps
- Pixels and density: source 362 × 310; implementation frames 640 × 360 at 1×; focused implementation crops normalized to 264 × 264 for comparison
- State: cursor held at the recorded click hotspot; ripple inspected at +0.02 s, +0.18 s, and +0.42 s

## Findings

No actionable P0, P1, or P2 differences remain.

- The rendered cursor preserves the source's rounded upper-left pointer silhouette and cyan-to-blue color direction.
- A thin navy edge is an intentional recording constraint so the cyan pointer remains legible on both bright and dark footage.
- The click marker is circular, centered on the cursor hotspot, expands smoothly, and fades without leaving the previous square-box artifact.
- Transparency edges are clean at recording scale; no magenta chroma fringe is visible.
- Fonts and typography: not applicable to this asset/effect comparison.
- Spacing and layout rhythm: pointer scale and ripple diameter remain proportional at the 640 × 360 validation viewport.
- Colors and visual tokens: cyan pointer and ripple align with the supplied blue/cyan reference; the dark edge provides necessary contrast.
- Image quality and asset fidelity: the implementation uses the generated transparent raster cursor asset, not a code-drawn placeholder; the cursor stays sharp at playback size.
- Copy and content: not applicable.

## Full-view Comparison Evidence

Three full recording frames verify pointer contrast across SMPTE light, saturated, and dark regions. The effect does not crop or shift the recording canvas.

## Focused Region Comparison Evidence

The side-by-side comparison places the 362 × 310 source reference beside equal-size focused crops of the rendered cursor and ripple. A separate focused region was necessary because the playback-size cursor is too small for silhouette and alpha-edge inspection in the full frame.

## Comparison History

- Pass 1: no P0/P1/P2 mismatch was found. The cursor silhouette, hotspot alignment, and animated circular ripple passed without a corrective visual iteration.

## Implementation Checklist

- [x] Replace the white default pointer with the rounded cyan recording cursor.
- [x] Keep the cursor hotspot aligned with recorded mouse coordinates.
- [x] Replace every square click marker with an automatic circular ripple.
- [x] Verify early, middle, and late ripple frames in an encoded MP4.
- [x] Verify transparent edges and contrast on varied backgrounds.

final result: passed
