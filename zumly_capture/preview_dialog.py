"""Intentional post-capture preview with lightweight screenshot annotation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import tempfile
import threading

from PySide6.QtCore import (
    QMimeData,
    QObject,
    QPointF,
    QRectF,
    QSize,
    Qt,
    QThread,
    QTimer,
    QUrl,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QImage,
    QImageReader,
    QKeySequence,
    QMouseEvent,
    QMovie,
    QPainter,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .identity import FILE_PREFIX
from .session import (
    discard_recording_draft,
    discard_unzoomed_recording,
    restore_unzoomed_recording,
)
from .gif_export import export_gif
from .windows_shell import reveal_in_folder

try:
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
    from PySide6.QtMultimediaWidgets import QVideoWidget
except ImportError:  # pragma: no cover - only relevant to incomplete Qt installs
    QAudioOutput = QMediaPlayer = QVideoWidget = None


@dataclass(slots=True)
class Annotation:
    kind: str
    start: QPointF
    end: QPointF
    color: QColor
    width: float
    text: str = ""


class _GifPreviewWorker(QObject):
    """Create the unzoomed GIF only when the user asks to preview/remove zoom."""

    finished = Signal(str, str)

    def __init__(self, source_path: str, output_path: str) -> None:
        super().__init__()
        self._source_path = source_path
        self._output_path = output_path
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        self._cancelled.set()

    def run(self) -> None:
        result = export_gif(
            self._source_path,
            self._output_path,
            1.0,
            cancel_callback=self._cancelled.is_set,
        )
        if result.state == "processed":
            self.finished.emit(result.output_path, "")
        elif result.state == "cancelled":
            self.finished.emit("", "GIF preparation was cancelled.")
        else:
            self.finished.emit("", result.error or "Could not prepare the original GIF.")


class _RecordingFormatWorker(QObject):
    """Create the selected recording format without blocking the preview UI."""

    finished = Signal(str, str)

    def __init__(self, source_path: str, output_path: str, output_format: str) -> None:
        super().__init__()
        self._source_path = source_path
        self._output_path = output_path
        self._output_format = output_format
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        self._cancelled.set()

    def run(self) -> None:
        try:
            if self._output_format == "gif":
                result = export_gif(
                    self._source_path,
                    self._output_path,
                    1.0,
                    cancel_callback=self._cancelled.is_set,
                )
                if result.state == "processed":
                    self.finished.emit(result.output_path, "")
                elif result.state == "cancelled":
                    self.finished.emit("", "GIF creation was cancelled.")
                else:
                    self.finished.emit("", result.error or "Could not create the GIF.")
                return

            source = Path(self._source_path).resolve()
            destination = Path(self._output_path).resolve()
            if not source.is_file() or source.stat().st_size <= 0:
                raise ValueError(f"The MP4 source is not usable: {source}")
            if destination.exists():
                raise FileExistsError(f"The save destination already exists: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            staged = ""
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=str(destination.parent),
                    prefix=f"{FILE_PREFIX}_format_",
                    suffix=".tmp",
                    delete=False,
                ) as target:
                    staged = target.name
                    with source.open("rb") as source_handle:
                        while True:
                            if self._cancelled.is_set():
                                self.finished.emit("", "MP4 save was cancelled.")
                                return
                            chunk = source_handle.read(1024 * 1024)
                            if not chunk:
                                break
                            target.write(chunk)
                    target.flush()
                    os.fsync(target.fileno())
                if self._cancelled.is_set():
                    self.finished.emit("", "MP4 save was cancelled.")
                    return
                os.rename(staged, destination)
                staged = ""
                self.finished.emit(str(destination), "")
            finally:
                if staged:
                    try:
                        os.remove(staged)
                    except OSError:
                        pass
        except Exception as exc:
            self.finished.emit("", str(exc))


def _draw_annotation(painter: QPainter, annotation: Annotation) -> None:
    color = QColor(annotation.color)
    if annotation.kind == "highlight":
        color.setAlpha(90)
    painter.setPen(
        QPen(
            color,
            annotation.width,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
    )
    painter.setBrush(Qt.BrushStyle.NoBrush)
    if annotation.kind in {"pen", "highlight"}:
        painter.drawLine(annotation.start, annotation.end)
    elif annotation.kind == "rectangle":
        painter.drawRect(QRectF(annotation.start, annotation.end).normalized())
    elif annotation.kind == "arrow":
        dx = annotation.end.x() - annotation.start.x()
        dy = annotation.end.y() - annotation.start.y()
        length = math.hypot(dx, dy)
        if length <= 0.5:
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(annotation.start, annotation.width, annotation.width)
            return
        direction = QPointF(dx / length, dy / length)
        perpendicular = QPointF(-direction.y(), direction.x())
        head_length = min(length * 0.48, max(24.0, annotation.width * 6.0))
        head_half = max(
            4.0,
            min(length * 0.34, max(13.0, annotation.width * 3.0)),
        )
        stem_half = min(head_half * 0.46, max(2.0, annotation.width * 0.75))
        neck = annotation.end - direction * head_length
        shoulder = neck - direction * min(
            head_length * 0.18,
            max(2.0, annotation.width * 0.9),
        )
        polygon = QPolygonF(
            [
                annotation.start,
                neck + perpendicular * stem_half,
                shoulder + perpendicular * head_half,
                annotation.end,
                shoulder - perpendicular * head_half,
                neck - perpendicular * stem_half,
            ]
        )
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPolygon(polygon)
    elif annotation.kind == "text":
        font = painter.font()
        font.setBold(True)
        font.setPixelSize(max(18, int(annotation.width * 5)))
        painter.setFont(font)
        painter.drawText(annotation.start, annotation.text)


def render_annotations(image: QImage, annotations: list[Annotation]) -> QImage:
    """Return a composited image without mutating the source image."""
    rendered = image.convertToFormat(QImage.Format.Format_ARGB32)
    painter = QPainter(rendered)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    for annotation in annotations:
        _draw_annotation(painter, annotation)
    painter.end()
    return rendered


class InlineTextEdit(QWidget):
    """Paint editable text and its caret directly over the screenshot."""

    committed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        parent: QWidget,
        color: QColor,
        font_pixel_size: int,
        maximum_width: int,
    ) -> None:
        super().__init__(parent)
        self._finished = False
        self._text = ""
        self._cursor_position = 0
        self._caret_visible = True
        self._horizontal_offset = 0
        self._maximum_editor_width = max(24, int(maximum_width))
        font = self.font()
        font.setBold(True)
        font.setPixelSize(max(10, int(font_pixel_size)))
        self.setFont(font)
        self._color = QColor(color)
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_InputMethodEnabled, True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.IBeamCursor)
        self._caret_timer = QTimer(self)
        self._caret_timer.setInterval(520)
        self._caret_timer.timeout.connect(self._toggle_caret)
        self._caret_timer.start()
        self._resize_to_text()

    def text(self) -> str:
        return self._text

    def setText(self, text: str) -> None:
        self._text = str(text)[:240]
        self._cursor_position = len(self._text)
        self._resize_to_text()
        self.update()

    def _resize_to_text(self) -> None:
        metrics = self.fontMetrics()
        content_width = metrics.horizontalAdvance(self._text or "M") + 6
        self.resize(
            min(self._maximum_editor_width, max(12, content_width)),
            metrics.height() + 6,
        )
        cursor_x = metrics.horizontalAdvance(self._text[: self._cursor_position]) + 2
        self._horizontal_offset = max(0, cursor_x - max(4, self.width() - 4))

    def _toggle_caret(self) -> None:
        self._caret_visible = not self._caret_visible
        self.update()

    def _reset_caret(self) -> None:
        self._caret_visible = True
        self._caret_timer.start()
        self.update()

    def _insert_text(self, text: str) -> None:
        cleaned = " ".join(str(text).replace("\r", "\n").splitlines())
        if not cleaned:
            return
        available = 240 - len(self._text)
        inserted = cleaned[:available]
        self._text = (
            self._text[: self._cursor_position]
            + inserted
            + self._text[self._cursor_position :]
        )
        self._cursor_position += len(inserted)
        self._resize_to_text()
        self._reset_caret()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setFont(self.font())
        painter.setPen(self._color)
        metrics = painter.fontMetrics()
        baseline = metrics.ascent()
        painter.save()
        painter.translate(-self._horizontal_offset, 0)
        painter.drawText(QPointF(2, baseline), self._text)
        if self.hasFocus() and self._caret_visible:
            cursor_x = metrics.horizontalAdvance(self._text[: self._cursor_position]) + 2
            painter.drawLine(
                QPointF(cursor_x, 2),
                QPointF(cursor_x, self.height() - 3),
            )
        painter.restore()
        painter.end()

    def _finish(self, commit: bool) -> None:
        if self._finished:
            return
        self._finished = True
        self._caret_timer.stop()
        text = self._text.strip()
        if commit and text:
            self.committed.emit(text)
        else:
            self.cancelled.emit()

    def keyPressEvent(self, event) -> None:
        if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter}:
            self._finish(True)
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape:
            self._finish(False)
            event.accept()
            return
        if event.matches(QKeySequence.StandardKey.Paste):
            self._insert_text(QApplication.clipboard().text())
            event.accept()
            return
        if event.key() == Qt.Key.Key_Backspace:
            if self._cursor_position > 0:
                self._text = (
                    self._text[: self._cursor_position - 1]
                    + self._text[self._cursor_position :]
                )
                self._cursor_position -= 1
                self._resize_to_text()
                self._reset_caret()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Delete:
            if self._cursor_position < len(self._text):
                self._text = (
                    self._text[: self._cursor_position]
                    + self._text[self._cursor_position + 1 :]
                )
                self._resize_to_text()
                self._reset_caret()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Left:
            self._cursor_position = max(0, self._cursor_position - 1)
            self._resize_to_text()
            self._reset_caret()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Right:
            self._cursor_position = min(len(self._text), self._cursor_position + 1)
            self._resize_to_text()
            self._reset_caret()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Home:
            self._cursor_position = 0
            self._resize_to_text()
            self._reset_caret()
            event.accept()
            return
        if event.key() == Qt.Key.Key_End:
            self._cursor_position = len(self._text)
            self._resize_to_text()
            self._reset_caret()
            event.accept()
            return
        typed = event.text()
        if (typed and not typed.isspace()) or typed == " ":
            self._insert_text(typed)
            event.accept()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        x = event.position().x() + self._horizontal_offset - 2
        metrics = self.fontMetrics()
        closest = 0
        closest_distance = abs(x)
        for index in range(1, len(self._text) + 1):
            distance = abs(metrics.horizontalAdvance(self._text[:index]) - x)
            if distance <= closest_distance:
                closest = index
                closest_distance = distance
        self._cursor_position = closest
        self._resize_to_text()
        self._reset_caret()
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        event.accept()

    def focusInEvent(self, event) -> None:
        self._reset_caret()
        super().focusInEvent(event)

    def focusOutEvent(self, event) -> None:
        self._finish(True)
        super().focusOutEvent(event)


class AnnotationCanvas(QWidget):
    changed = Signal()

    def __init__(self, path: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image = QImage(path)
        if self._image.isNull():
            raise ValueError(f"Could not load screenshot: {path}")
        self._annotations: list[Annotation] = []
        self._tool = "pen"
        self._color = QColor("#ff4d67")
        self._start: QPointF | None = None
        self._preview: Annotation | None = None
        self._inline_editor: InlineTextEdit | None = None
        self._inline_origin: QPointF | None = None
        self.setMinimumSize(640, 380)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)

    @property
    def annotations(self) -> list[Annotation]:
        return list(self._annotations)

    def set_tool(self, tool: str) -> None:
        self._tool = str(tool)

    def set_color(self, color: QColor) -> None:
        if color.isValid():
            self._color = QColor(color)

    def undo(self) -> None:
        if self._annotations:
            self._annotations.pop()
            self.changed.emit()
            self.update()

    def rendered_image(self) -> QImage:
        return render_annotations(self._image, self._annotations)

    def _begin_inline_text(self, image_point: QPointF, widget_point: QPointF) -> None:
        if self._inline_editor is not None:
            self._inline_editor._finish(True)
        image_rect = self._image_rect()
        scale = image_rect.width() / max(self._image.width(), 1)
        saved_font_size = max(18, int(4.0 * 5))
        editor = InlineTextEdit(
            self,
            self._color,
            max(10, round(saved_font_size * scale)),
            max(24, int(image_rect.right() - widget_point.x())),
        )
        x = max(int(image_rect.left()), min(int(widget_point.x()), self.width() - editor.width()))
        y = max(
            int(image_rect.top()),
            min(
                int(widget_point.y()) - editor.fontMetrics().ascent(),
                int(image_rect.bottom()) - editor.height(),
            ),
        )
        editor.move(x, y)
        self._inline_editor = editor
        self._inline_origin = QPointF(image_point)
        editor.committed.connect(self._commit_inline_text)
        editor.cancelled.connect(self._clear_inline_text)
        editor.show()
        editor.raise_()
        editor.setFocus(Qt.FocusReason.MouseFocusReason)

    def _commit_inline_text(self, text: str) -> None:
        origin = self._inline_origin
        if origin is not None and text.strip():
            self._annotations.append(
                Annotation("text", origin, origin, self._color, 4.0, text.strip())
            )
            self.changed.emit()
            self.update()
        self._clear_inline_text()

    def _clear_inline_text(self) -> None:
        editor = self._inline_editor
        self._inline_editor = None
        self._inline_origin = None
        if editor is not None:
            editor.hide()
            editor.deleteLater()

    def _image_rect(self) -> QRectF:
        scale = min(
            self.width() / max(self._image.width(), 1),
            self.height() / max(self._image.height(), 1),
        )
        width = self._image.width() * scale
        height = self._image.height() * scale
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    def _to_image(self, point: QPointF) -> QPointF | None:
        target = self._image_rect()
        if not target.contains(point):
            return None
        scale = target.width() / max(self._image.width(), 1)
        return QPointF(
            (point.x() - target.left()) / scale,
            (point.y() - target.top()) / scale,
        )

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.fillRect(self.rect(), QColor("#111827"))
        target = self._image_rect()
        painter.drawImage(target, self._image)
        scale = target.width() / max(self._image.width(), 1)
        painter.save()
        painter.translate(target.left(), target.top())
        painter.scale(scale, scale)
        for annotation in self._annotations:
            _draw_annotation(painter, annotation)
        if self._preview is not None:
            _draw_annotation(painter, self._preview)
        painter.restore()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        point = self._to_image(event.position())
        if point is None:
            return
        if self._tool == "text":
            self._begin_inline_text(point, event.position())
            return
        self._start = point
        self._preview = Annotation(
            self._tool,
            point,
            point,
            self._color,
            18.0 if self._tool == "highlight" else 5.0,
        )

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._start is None or self._preview is None:
            return
        point = self._to_image(event.position())
        if point is None:
            return
        if self._tool in {"pen", "highlight"}:
            self._annotations.append(
                Annotation(
                    self._tool,
                    self._preview.end,
                    point,
                    self._color,
                    self._preview.width,
                )
            )
            self._preview.start = point
        self._preview.end = point
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self._preview is None:
            return
        point = self._to_image(event.position())
        if point is not None:
            self._preview.end = point
        if self._tool not in {"pen", "highlight"}:
            self._annotations.append(self._preview)
        self._start = None
        self._preview = None
        self.changed.emit()
        self.update()


class CapturePreviewDialog(QDialog):
    saved = Signal(str)

    def __init__(
        self,
        capture_path: str,
        parent: QWidget | None = None,
        *,
        unzoomed_path: str = "",
        format_source_path: str = "",
        preferred_output_format: str = "",
    ) -> None:
        super().__init__(parent)
        self.capture_path = os.path.abspath(capture_path)
        candidate = os.path.abspath(unzoomed_path) if unzoomed_path else ""
        self._unzoomed_path = candidate if candidate and os.path.isfile(candidate) else ""
        format_candidate = (
            os.path.abspath(format_source_path) if format_source_path else ""
        )
        self._format_source_path = (
            format_candidate
            if format_candidate and os.path.isfile(format_candidate)
            else ""
        )
        preferred = str(preferred_output_format or "").lower()
        self._preferred_output_format = preferred if preferred in {"mp4", "gif"} else ""
        initial_format = Path(self.capture_path).suffix.lower().lstrip(".")
        self._recording_outputs: dict[str, str] = (
            {initial_format: self.capture_path}
            if initial_format in {"mp4", "gif"}
            else {}
        )
        self._obsolete_recording_paths: set[str] = set()
        self._canonical_mp4_path = (
            self.capture_path if initial_format == "mp4" else self._format_source_path
        )
        self._zoom_finalized = False
        self._final_remove_zoom = False
        self._saved = False
        self._player = None
        self._audio = None
        self._video_widget = None
        self._position: QSlider | None = None
        self._gif_label: QLabel | None = None
        self._gif_movie: QMovie | None = None
        self._gif_original_path = ""
        self._gif_pending_path = ""
        self._gif_thread: QThread | None = None
        self._gif_worker: _GifPreviewWorker | None = None
        self._format_pending_path = ""
        self._format_pending_source = ""
        self._format_pending_remove_zoom = False
        self._format_save_completed = False
        self._format_thread: QThread | None = None
        self._format_worker: _RecordingFormatWorker | None = None
        self._play_button: QPushButton | None = None
        self._remove_zoom: QCheckBox | None = None
        self._zoom_status: QLabel | None = None
        self._reveal_button: QPushButton | None = None
        self._save_button: QPushButton | None = None
        self._format_combo: QComboBox | None = None
        self._recording_shell: QWidget | None = None
        self._recording_layout: QVBoxLayout | None = None
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(f"Preview · {Path(self.capture_path).name}")
        self.resize(980, 690)
        self.setStyleSheet(
            """
            QDialog { background: #111827; color: #edf4ff; }
            QLabel { color: #c8d5e5; }
            QPushButton, QToolButton {
                background: #263449; color: #f6f9ff; border: 1px solid #40536c;
                border-radius: 7px; padding: 7px 12px; min-height: 22px;
            }
            QPushButton:hover, QToolButton:hover { background: #356da8; }
            QToolButton:checked { background: #2f7fca; border-color: #70b7ff; }
            QPushButton:disabled { color: #718096; background: #1b2637; border-color: #2c3a4d; }
            QCheckBox { color: #d6e4f5; spacing: 8px; padding: 5px; }
            QComboBox {
                background: #1d2a3c; color: #f6f9ff; border: 1px solid #40536c;
                border-radius: 7px; padding: 7px 10px; min-height: 22px;
                min-width: 116px;
            }
            QComboBox:hover { border-color: #62aaf0; }
            QComboBox QAbstractItemView {
                background: #1d2a3c; color: #f6f9ff; selection-background-color: #356da8;
            }
            QSlider::groove:horizontal { height: 5px; background: #33445b; border-radius: 2px; }
            QSlider::handle:horizontal { width: 14px; margin: -5px 0; background: #66aef2; border-radius: 7px; }
            """
        )
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)
        self._header = QLabel(Path(self.capture_path).name)
        self._header.setStyleSheet("font-size: 16px; font-weight: 650; color: white;")
        root.addWidget(self._header)
        suffix = Path(self.capture_path).suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg"}:
            self._build_image_preview(root)
        else:
            self._recording_shell = QWidget()
            self._recording_layout = QVBoxLayout(self._recording_shell)
            self._recording_layout.setContentsMargins(0, 0, 0, 0)
            self._recording_layout.setSpacing(10)
            root.addWidget(self._recording_shell, 1)
            self._build_recording_preview()
        root.addLayout(self._common_actions())

    def _build_recording_preview(self) -> None:
        if self._recording_layout is None:
            return
        if Path(self.capture_path).suffix.lower() == ".gif":
            self._build_gif_preview(self._recording_layout)
        else:
            self._build_video_preview(self._recording_layout)

    def _clear_layout(self, layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            child_layout = item.layout()
            widget = item.widget()
            if child_layout is not None:
                self._clear_layout(child_layout)
                child_layout.deleteLater()
            if widget is not None:
                widget.deleteLater()

    def _dispose_recording_preview(self) -> None:
        self._release_recording_preview()
        if self._player is not None:
            self._player.setVideoOutput(None)
            self._player.deleteLater()
        if self._audio is not None:
            self._audio.deleteLater()
        if self._gif_movie is not None:
            self._gif_movie.deleteLater()
        if self._recording_layout is not None:
            self._clear_layout(self._recording_layout)
        self._player = None
        self._audio = None
        self._video_widget = None
        self._position = None
        self._gif_label = None
        self._gif_movie = None
        self._play_button = None
        self._remove_zoom = None
        self._zoom_status = None

    def _rebuild_recording_preview(self) -> None:
        self._dispose_recording_preview()
        self._header.setText(Path(self.capture_path).name)
        self._build_recording_preview()
        if self._zoom_finalized and self._remove_zoom is not None:
            self._remove_zoom.blockSignals(True)
            self._remove_zoom.setChecked(self._final_remove_zoom)
            self._remove_zoom.blockSignals(False)
            self._remove_zoom.setEnabled(False)
        if self._zoom_finalized and self._zoom_status is not None:
            self._zoom_status.setText(
                "Automatic Smart Zoom removed"
                if self._final_remove_zoom
                else "Automatic Smart Zoom retained"
            )

    def _build_image_preview(self, root: QVBoxLayout) -> None:
        self._canvas = AnnotationCanvas(self.capture_path)
        root.addWidget(self._canvas, 1)
        toolbar_shell = QWidget()
        toolbar_shell.setStyleSheet(
            "background: #202c3d; border: 1px solid #34465d; border-radius: 9px;"
        )
        toolbar = QHBoxLayout(toolbar_shell)
        toolbar.setContentsMargins(7, 5, 7, 5)
        toolbar.setSpacing(5)
        group = QButtonGroup(self)
        group.setExclusive(True)
        for label, tool in (
            ("Pen", "pen"),
            ("Highlight", "highlight"),
            ("Arrow", "arrow"),
            ("Rectangle", "rectangle"),
            ("Text", "text"),
        ):
            button = QToolButton()
            button.setText(label)
            button.setCheckable(True)
            button.setChecked(tool == "pen")
            group.addButton(button)
            button.clicked.connect(lambda _checked=False, value=tool: self._canvas.set_tool(value))
            toolbar.addWidget(button)
        color = QPushButton("Color")
        color.clicked.connect(self._choose_color)
        toolbar.addWidget(color)
        undo = QPushButton("Undo")
        undo.clicked.connect(self._canvas.undo)
        toolbar.addWidget(undo)
        toolbar.addStretch(1)
        hint = QLabel("Text: click the canvas and type")
        hint.setStyleSheet("color: #91a6bd; padding-right: 6px;")
        toolbar.addWidget(hint)
        root.addWidget(toolbar_shell)

    def _build_video_preview(self, root: QVBoxLayout) -> None:
        if QMediaPlayer is None or QVideoWidget is None or QAudioOutput is None:
            message = QLabel("Video preview components are unavailable.")
            message.setAlignment(Qt.AlignmentFlag.AlignCenter)
            root.addWidget(message, 1)
            self._build_zoom_choice(root)
            return
        video = QVideoWidget()
        video.setMinimumHeight(420)
        self._video_widget = video
        root.addWidget(video, 1)
        self._player = QMediaPlayer(self)
        self._audio = QAudioOutput(self)
        self._player.setAudioOutput(self._audio)
        self._player.setVideoOutput(video)
        self._player.setSource(QUrl.fromLocalFile(self.capture_path))
        controls = QHBoxLayout()
        self._play_button = QPushButton("Play")
        self._play_button.clicked.connect(self._toggle_playback)
        controls.addWidget(self._play_button)
        self._position = QSlider(Qt.Orientation.Horizontal)
        self._position.sliderMoved.connect(self._player.setPosition)
        self._player.durationChanged.connect(self._position.setMaximum)
        self._player.positionChanged.connect(self._position.setValue)
        self._player.playbackStateChanged.connect(self._playback_changed)
        controls.addWidget(self._position, 1)
        root.addLayout(controls)
        self._build_zoom_choice(root)

    def _build_gif_preview(self, root: QVBoxLayout) -> None:
        label = QLabel()
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setMinimumHeight(420)
        label.setStyleSheet("background: #0b1220; border-radius: 8px;")
        self._gif_label = label
        root.addWidget(label, 1)
        movie = QMovie(self.capture_path)
        self._gif_movie = movie
        label.setMovie(movie)
        self._set_gif_source(self.capture_path)
        controls = QHBoxLayout()
        self._play_button = QPushButton("Pause")
        self._play_button.clicked.connect(self._toggle_gif_playback)
        controls.addWidget(self._play_button)
        controls.addStretch(1)
        note = QLabel("Animated GIF · loops automatically · no audio")
        note.setStyleSheet("color: #91a6bd;")
        controls.addWidget(note)
        root.addLayout(controls)
        self._build_zoom_choice(root)

    def _set_gif_source(self, path: str) -> None:
        if self._gif_movie is None:
            return
        self._gif_movie.stop()
        self._gif_movie.setFileName(path)
        source_size = QImageReader(path).size()
        if source_size.isValid():
            self._gif_movie.setScaledSize(
                source_size.scaled(
                    QSize(920, 520),
                    Qt.AspectRatioMode.KeepAspectRatio,
                )
            )
        self._gif_movie.start()
        if self._play_button is not None:
            self._play_button.setText("Pause")

    def _build_zoom_choice(self, root: QVBoxLayout) -> None:
        if not self._unzoomed_path:
            return
        zoom_row = QHBoxLayout()
        self._zoom_status = QLabel("Automatic Smart Zoom applied")
        self._zoom_status.setStyleSheet("color: #9ecbff; font-weight: 650;")
        zoom_row.addWidget(self._zoom_status)
        zoom_row.addStretch(1)
        self._remove_zoom = QCheckBox("Remove automatic Smart Zoom")
        self._remove_zoom.toggled.connect(self._preview_zoom_choice)
        zoom_row.addWidget(self._remove_zoom)
        root.addLayout(zoom_row)

    def _common_actions(self) -> QHBoxLayout:
        row = QHBoxLayout()
        copy = QPushButton("Copy")
        copy.clicked.connect(self._copy_capture)
        row.addWidget(copy)
        self._reveal_button = QPushButton("Show in folder")
        self._reveal_button.setEnabled(False)
        self._reveal_button.clicked.connect(self._reveal_capture)
        row.addWidget(self._reveal_button)
        row.addStretch(1)
        if Path(self.capture_path).suffix.lower() in {".mp4", ".gif"}:
            format_label = QLabel("Save as:")
            format_label.setStyleSheet("color: #d6e4f5; font-weight: 650;")
            row.addWidget(format_label)
            self._format_combo = QComboBox()
            self._format_combo.setObjectName("recordingFormat")
            self._format_combo.addItem("MP4 video", "mp4")
            self._format_combo.addItem("Animated GIF", "gif")
            current_format = Path(self.capture_path).suffix.lower().lstrip(".")
            selected_format = self._preferred_output_format or current_format
            selected_index = self._format_combo.findData(selected_format)
            self._format_combo.setCurrentIndex(max(0, selected_index))
            row.addWidget(self._format_combo)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        row.addWidget(close)
        self._save_button = QPushButton("Save")
        self._save_button.setStyleSheet(
            "background: #2f7fca; border-color: #62aaf0; font-weight: 700;"
        )
        self._save_button.clicked.connect(self._save_capture)
        row.addWidget(self._save_button)
        return row

    def _choose_color(self) -> None:
        chosen = QColorDialog.getColor(QColor("#ff4d67"), self, "Annotation color")
        self._canvas.set_color(chosen)

    def _save_capture(self) -> None:
        try:
            if hasattr(self, "_canvas"):
                self._save_image()
            else:
                self._save_video()
        except Exception as exc:
            QMessageBox.critical(self, "Could not save", str(exc))

    def _mark_saved(self) -> None:
        self._saved = True
        if self._reveal_button is not None:
            self._reveal_button.setEnabled(True)
        if self._save_button is not None:
            self._save_button.setText("Saved")
            self._save_button.setEnabled(False)
        self.saved.emit(self.capture_path)
        self.setWindowTitle(f"Saved · {Path(self.capture_path).name}")

    def _mark_recording_saved(self) -> None:
        """Mark one recording format saved while keeping the preview reusable."""
        self._saved = True
        if self._reveal_button is not None:
            self._reveal_button.setEnabled(True)
        if self._save_button is not None:
            self._save_button.setText("Save")
            self._save_button.setEnabled(True)
        if self._format_combo is not None:
            self._format_combo.setEnabled(True)
        self.saved.emit(self.capture_path)
        self.setWindowTitle(f"Saved · {Path(self.capture_path).name}")

    def _save_image(self) -> None:
        image = self._canvas.rendered_image()
        suffix = Path(self.capture_path).suffix.lower()
        image_format = "JPEG" if suffix in {".jpg", ".jpeg"} else "PNG"
        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(
                dir=str(Path(self.capture_path).parent),
                suffix=suffix,
                delete=False,
            ) as handle:
                temp_path = handle.name
            if not image.save(temp_path, image_format, 95):
                raise OSError("Qt could not encode the annotated screenshot")
            os.replace(temp_path, self.capture_path)
            temp_path = ""
            self._mark_saved()
        finally:
            if temp_path:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def _preview_zoom_choice(self, remove_zoom: bool) -> None:
        if self._gif_movie is not None:
            if not remove_zoom:
                self._set_gif_source(self.capture_path)
                if self._zoom_status is not None:
                    self._zoom_status.setText("Automatic Smart Zoom applied")
                return
            if self._gif_original_path and os.path.isfile(self._gif_original_path):
                self._set_gif_source(self._gif_original_path)
                if self._zoom_status is not None:
                    self._zoom_status.setText("Previewing original GIF")
                return
            self._prepare_original_gif()
            return
        if self._player is None or not self._unzoomed_path:
            return
        self._player.stop()
        source = self._unzoomed_path if remove_zoom else self.capture_path
        self._player.setSource(QUrl.fromLocalFile(source))
        if self._zoom_status is not None:
            self._zoom_status.setText(
                "Previewing original recording"
                if remove_zoom
                else "Automatic Smart Zoom applied"
            )

    def _prepare_original_gif(self) -> None:
        if not self._unzoomed_path or self._gif_thread is not None:
            return
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as handle:
            output_path = handle.name
        try:
            os.remove(output_path)
        except OSError:
            pass
        thread = QThread(self)
        worker = _GifPreviewWorker(self._unzoomed_path, output_path)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._original_gif_ready)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._gif_thread_finished)
        self._gif_thread = thread
        self._gif_worker = worker
        self._gif_pending_path = output_path
        if self._remove_zoom is not None:
            self._remove_zoom.setEnabled(False)
        if self._save_button is not None:
            self._save_button.setEnabled(False)
        if self._zoom_status is not None:
            self._zoom_status.setText("Preparing original GIF…")
        thread.start()

    def _original_gif_ready(self, path: str, error: str) -> None:
        if path and os.path.isfile(path):
            self._gif_original_path = os.path.abspath(path)
            self._gif_pending_path = ""
            if self._remove_zoom is not None and self._remove_zoom.isChecked():
                self._set_gif_source(self._gif_original_path)
                if self._zoom_status is not None:
                    self._zoom_status.setText("Previewing original GIF")
        else:
            discard_unzoomed_recording(self._gif_pending_path)
            self._gif_pending_path = ""
            if self._remove_zoom is not None:
                self._remove_zoom.blockSignals(True)
                self._remove_zoom.setChecked(False)
                self._remove_zoom.blockSignals(False)
            if self._zoom_status is not None:
                self._zoom_status.setText(error or "Could not prepare original GIF")
        if self._remove_zoom is not None:
            self._remove_zoom.setEnabled(bool(self._unzoomed_path))
        if self._save_button is not None:
            self._save_button.setEnabled(True)

    def _gif_thread_finished(self) -> None:
        self._gif_thread = None
        self._gif_worker = None

    def _selected_recording_format(self) -> str:
        if self._format_combo is None:
            return Path(self.capture_path).suffix.lower().lstrip(".")
        selected = str(self._format_combo.currentData() or "").lower()
        return selected if selected in {"mp4", "gif"} else "mp4"

    def _alternate_format_path(self, output_format: str) -> str:
        source = Path(self.capture_path)
        candidate = source.with_suffix(f".{output_format}")
        if not candidate.exists():
            return str(candidate)
        for counter in range(1, 10_000):
            candidate = source.with_name(
                f"{source.stem}_{counter}.{output_format}"
            )
            if not candidate.exists():
                return str(candidate)
        raise FileExistsError("Could not find an unused recording filename")

    def _release_recording_preview(self) -> None:
        if self._player is not None:
            self._player.stop()
            self._player.setSource(QUrl())
        if self._gif_movie is not None:
            self._gif_movie.stop()
            # QMovie keeps its current GIF handle open on Windows.
            self._gif_movie.setFileName("")

    def _cleanup_recording_drafts(self, *excluded_paths: str) -> None:
        excluded = {
            os.path.abspath(path)
            for path in excluded_paths
            if path
        }
        for path in (
            self._unzoomed_path,
            self._format_source_path,
            self._gif_original_path,
            self._gif_pending_path,
        ):
            if path and os.path.abspath(path) not in excluded:
                discard_recording_draft(path)

    def _clear_recording_draft_paths(self) -> None:
        self._unzoomed_path = ""
        self._format_source_path = ""
        self._gif_original_path = ""
        self._gif_pending_path = ""

    def _set_format_save_enabled(self, enabled: bool) -> None:
        if self._format_combo is not None:
            self._format_combo.setEnabled(enabled)
        if self._remove_zoom is not None:
            self._remove_zoom.setEnabled(enabled and bool(self._unzoomed_path))
        if self._save_button is not None:
            self._save_button.setEnabled(enabled)

    def _recording_mp4_source(self, remove_zoom: bool) -> str:
        source = self._unzoomed_path if remove_zoom else self._canonical_mp4_path
        if not source or not os.path.isfile(source):
            raise RuntimeError(
                "The MP4 source for this format is no longer available. "
                "Record again and keep the preview open until saving."
            )
        return os.path.abspath(source)

    def _finalize_zoom_choice(
        self,
        remove_zoom: bool,
        mp4_source: str = "",
        saved_format: str = "",
    ) -> None:
        if self._zoom_finalized:
            return
        self._zoom_finalized = True
        self._final_remove_zoom = remove_zoom
        if mp4_source and os.path.isfile(mp4_source):
            self._canonical_mp4_path = os.path.abspath(mp4_source)
        if remove_zoom and saved_format:
            for output_format, output_path in list(self._recording_outputs.items()):
                if output_format != saved_format:
                    self._obsolete_recording_paths.add(os.path.abspath(output_path))
                    self._recording_outputs.pop(output_format, None)
        if self._remove_zoom is not None:
            self._remove_zoom.setEnabled(False)
        if self._zoom_status is not None:
            self._zoom_status.setText(
                "Automatic Smart Zoom removed"
                if remove_zoom
                else "Automatic Smart Zoom retained"
            )

    def _show_recording_output(self, path: str) -> None:
        self.capture_path = os.path.abspath(path)
        if self._format_combo is not None:
            output_format = Path(self.capture_path).suffix.lower().lstrip(".")
            index = self._format_combo.findData(output_format)
            if index >= 0:
                self._format_combo.setCurrentIndex(index)
        self._rebuild_recording_preview()

    def _save_current_recording(self, remove_zoom: bool) -> None:
        current_format = Path(self.capture_path).suffix.lower().lstrip(".")
        if not self._zoom_finalized:
            self._release_recording_preview()
            canonical_source = self._canonical_mp4_path
            if remove_zoom:
                if current_format == "gif":
                    if not self._gif_original_path or not os.path.isfile(
                        self._gif_original_path
                    ):
                        raise RuntimeError("The original GIF is still being prepared")
                    restore_unzoomed_recording(
                        self.capture_path,
                        self._gif_original_path,
                    )
                    self._gif_original_path = ""
                    canonical_source = self._unzoomed_path
                else:
                    if not self._unzoomed_path or not os.path.isfile(
                        self._unzoomed_path
                    ):
                        raise RuntimeError("The original MP4 is no longer available")
                    restore_unzoomed_recording(
                        self.capture_path,
                        self._unzoomed_path,
                    )
                    self._unzoomed_path = ""
                    canonical_source = self.capture_path
            self._recording_outputs[current_format] = self.capture_path
            self._finalize_zoom_choice(
                remove_zoom,
                canonical_source,
                saved_format=current_format,
            )
            self._rebuild_recording_preview()
        self._mark_recording_saved()

    def _save_video(self) -> None:
        remove_zoom = (
            self._final_remove_zoom
            if self._zoom_finalized
            else self._remove_zoom is not None and self._remove_zoom.isChecked()
        )
        current_format = Path(self.capture_path).suffix.lower().lstrip(".")
        selected_format = self._selected_recording_format()
        existing_output = self._recording_outputs.get(selected_format, "")
        if existing_output and os.path.isfile(existing_output):
            if selected_format == current_format:
                self._save_current_recording(remove_zoom)
            else:
                self._show_recording_output(existing_output)
                self._mark_recording_saved()
            return

        source = self._recording_mp4_source(remove_zoom)
        target = self._alternate_format_path(selected_format)
        self._release_recording_preview()
        self._start_format_save(
            source,
            target,
            selected_format,
            remove_zoom=remove_zoom,
        )

    def _start_format_save(
        self,
        source_path: str,
        output_path: str,
        output_format: str,
        *,
        remove_zoom: bool = False,
    ) -> None:
        if self._format_thread is not None:
            return
        thread = QThread(self)
        worker = _RecordingFormatWorker(source_path, output_path, output_format)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._recording_format_ready)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._format_thread_finished)
        self._format_thread = thread
        self._format_worker = worker
        self._format_pending_path = os.path.abspath(output_path)
        self._format_pending_source = os.path.abspath(source_path)
        self._format_pending_remove_zoom = remove_zoom
        self._format_save_completed = False
        self._set_format_save_enabled(False)
        if self._save_button is not None:
            self._save_button.setText(
                "Creating GIF…" if output_format == "gif" else "Saving MP4…"
            )
        if self._zoom_status is not None:
            self._zoom_status.setText(
                "Creating animated GIF…"
                if output_format == "gif"
                else "Saving MP4 video…"
            )
        thread.start()

    def _recording_format_ready(self, path: str, error: str) -> None:
        pending_path = self._format_pending_path
        pending_source = self._format_pending_source
        pending_remove_zoom = self._format_pending_remove_zoom
        if not path or not os.path.isfile(path):
            discard_recording_draft(pending_path)
            self._format_pending_path = ""
            self._format_pending_source = ""
            self._format_pending_remove_zoom = False
            self._set_format_save_enabled(True)
            if self._save_button is not None:
                self._save_button.setText("Save")
            self._restore_recording_preview()
            QMessageBox.critical(
                self,
                "Could not save recording",
                error or "The selected recording format could not be created.",
            )
            return

        saved_path = os.path.abspath(path)
        saved_format = Path(saved_path).suffix.lower().lstrip(".")
        self._recording_outputs[saved_format] = saved_path
        self._finalize_zoom_choice(
            pending_remove_zoom,
            pending_source,
            saved_format=saved_format,
        )
        self._format_pending_path = ""
        self._format_pending_source = ""
        self._format_pending_remove_zoom = False
        self._show_recording_output(saved_path)
        self._format_save_completed = True

    def _restore_recording_preview(self) -> None:
        remove_zoom = self._remove_zoom is not None and self._remove_zoom.isChecked()
        if self._player is not None:
            source = self._unzoomed_path if remove_zoom and self._unzoomed_path else self.capture_path
            self._player.setSource(QUrl.fromLocalFile(source))
        if self._gif_movie is not None:
            source = (
                self._gif_original_path
                if remove_zoom and self._gif_original_path
                else self.capture_path
            )
            self._set_gif_source(source)

    def _format_thread_finished(self) -> None:
        self._format_thread = None
        self._format_worker = None
        if self._format_save_completed:
            self._format_save_completed = False
            self._mark_recording_saved()

    def _copy_capture(self) -> None:
        if hasattr(self, "_canvas"):
            QApplication.clipboard().setImage(self._canvas.rendered_image())
        else:
            mime = QMimeData()
            mime.setUrls([QUrl.fromLocalFile(self.capture_path)])
            QApplication.clipboard().setMimeData(mime)

    def _reveal_capture(self) -> None:
        reveal_in_folder(self.capture_path)

    def _toggle_playback(self) -> None:
        if self._player is None:
            return
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _toggle_gif_playback(self) -> None:
        if self._gif_movie is None:
            return
        if self._gif_movie.state() == QMovie.MovieState.Running:
            self._gif_movie.setPaused(True)
            if self._play_button is not None:
                self._play_button.setText("Play")
        else:
            self._gif_movie.setPaused(False)
            if self._play_button is not None:
                self._play_button.setText("Pause")

    def _playback_changed(self, state) -> None:
        if self._play_button is not None:
            self._play_button.setText(
                "Pause"
                if state == QMediaPlayer.PlaybackState.PlayingState
                else "Play"
            )

    def closeEvent(self, event) -> None:
        if self._player is not None:
            self._player.stop()
        if self._gif_movie is not None:
            self._gif_movie.stop()
            self._gif_movie.setFileName("")
        if self._gif_thread is not None and self._gif_thread.isRunning():
            if self._gif_worker is not None:
                self._gif_worker.cancel()
            self._gif_thread.quit()
            if not self._gif_thread.wait(5000):
                event.ignore()
                return
        if self._format_thread is not None and self._format_thread.isRunning():
            if self._format_worker is not None:
                self._format_worker.cancel()
            self._format_thread.quit()
            if not self._format_thread.wait(5000):
                event.ignore()
                return
        self._cleanup_recording_drafts()
        discard_recording_draft(self._format_pending_path)
        self._clear_recording_draft_paths()
        self._format_pending_path = ""
        self._format_pending_source = ""
        self._format_pending_remove_zoom = False
        retained_outputs = {
            os.path.abspath(path) for path in self._recording_outputs.values()
        }
        for obsolete_path in self._obsolete_recording_paths:
            if obsolete_path in retained_outputs:
                continue
            try:
                os.remove(obsolete_path)
            except FileNotFoundError:
                pass
            except OSError:
                # The saved outputs are already durable; a stale provisional
                # file is safer than risking the retained copies.
                pass
        self._obsolete_recording_paths.clear()
        super().closeEvent(event)
