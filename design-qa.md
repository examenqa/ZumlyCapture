# Design QA

Reference: the user-provided dark screenshot annotation editor.

Validated at a 1000 × 710 native preview window using a 1280 × 720 sample
screenshot. The rendered editor was compared side by side with the reference.

- The annotation tool rail sits below the canvas.
- The arrow has a substantial filled head and a rounded, thick stem.
- Rectangle annotations remain transparent after arrow rendering.
- Text entry is initiated directly on the canvas.
- Save is the primary bottom-right action.
- Show in Folder is disabled until Save.
- The redundant Open action is absent.
- The video surface remains playback-only; automatic Smart Zoom is removable
  only as a complete effect.

Automated coverage also verifies the filled arrow, inline text flow, action
labels, reversible Smart Zoom media replacement, and manifest update.
