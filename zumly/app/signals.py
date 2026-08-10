"""Editor event bus signals shared by panels and preview surfaces.

The bus carries transient editor state and derived, revisioned views of the
active ``RecordingSession``.  It must not become a second mutable source of
truth for project data.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class EditorEventBus(QObject):
    """Signals for editor-only state that has not been committed yet."""

    highlight_preview_updated = Signal(object)
    clear_highlight_preview = Signal()
    highlight_geometry_dragged = Signal(float, float, float, float)
    text_annotation_preview_updated = Signal(object)
    clear_text_annotation_preview = Signal()
    text_annotation_geometry_dragged = Signal(float, float)
    layout_scene_preview_updated = Signal(object)
    clear_layout_scene_preview = Signal()
    explainer_scene_preview_updated = Signal(object)
    clear_explainer_scene_preview = Signal()
    timeline_mapping_changed = Signal(object, int)
