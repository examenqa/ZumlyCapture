"""Intentional post-capture preview with lightweight screenshot annotation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import subprocess
import tempfile

from PySide6.QtCore import QMimeData, QPointF, QRectF, Qt, QUrl, Signal
from PySide6.QtGui import (
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .session import discard_unzoomed_recording, restore_unzoomed_recording

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
        stem_half = max(2.5, annotation.width * 0.7)
        head_length = min(length * 0.55, max(24.0, annotation.width * 5.5))
        head_half = max(13.0, annotation.width * 2.7)
        neck = annotation.end - direction * head_length
        polygon = QPolygonF(
            [
                annotation.start - perpendicular * stem_half,
                neck - perpendicular * stem_half,
                neck - perpendicular * head_half,
                annotation.end,
                neck + perpendicular * head_half,
                neck + perpendicular * stem_half,
                annotation.start + perpendicular * stem_half,
            ]
        )
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPolygon(polygon)
        painter.drawEllipse(annotation.start, stem_half, stem_half)
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


class InlineTextEdit(QLineEdit):
    """A single-line text editor positioned directly over the screenshot."""

    committed = Signal(str)
    cancelled = Signal()

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self._finished = False
        self.setPlaceholderText("Type annotation…")
        self.setMinimumSize(220, 38)
        self.setMaxLength(240)

    def _finish(self, commit: bool) -> None:
        if self._finished:
            return
        self._finished = True
        text = self.text().strip()
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
        super().keyPressEvent(event)

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
        editor = InlineTextEdit(self)
        editor.setStyleSheet(
            f"""
            QLineEdit {{
                color: {self._color.name()};
                background: rgba(15, 23, 42, 220);
                border: 2px solid {self._color.name()};
                border-radius: 6px;
                padding: 5px 8px;
                font-size: 18px;
                font-weight: 700;
            }}
            """
        )
        x = max(0, min(int(widget_point.x()), max(0, self.width() - editor.width())))
        y = max(0, min(int(widget_point.y()) - 19, max(0, self.height() - 38)))
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
    ) -> None:
        super().__init__(parent)
        self.capture_path = os.path.abspath(capture_path)
        candidate = os.path.abspath(unzoomed_path) if unzoomed_path else ""
        self._unzoomed_path = candidate if candidate and os.path.isfile(candidate) else ""
        self._saved = False
        self._player = None
        self._audio = None
        self._video_widget = None
        self._play_button: QPushButton | None = None
        self._remove_zoom: QCheckBox | None = None
        self._zoom_status: QLabel | None = None
        self._reveal_button: QPushButton | None = None
        self._save_button: QPushButton | None = None
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
            QSlider::groove:horizontal { height: 5px; background: #33445b; border-radius: 2px; }
            QSlider::handle:horizontal { width: 14px; margin: -5px 0; background: #66aef2; border-radius: 7px; }
            """
        )
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)
        header = QLabel(Path(self.capture_path).name)
        header.setStyleSheet("font-size: 16px; font-weight: 650; color: white;")
        root.addWidget(header)
        if Path(self.capture_path).suffix.lower() in {".png", ".jpg", ".jpeg"}:
            self._build_image_preview(root)
        else:
            self._build_video_preview(root)
        root.addLayout(self._common_actions())

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

    def _save_video(self) -> None:
        remove_zoom = self._remove_zoom is not None and self._remove_zoom.isChecked()
        if self._player is not None:
            self._player.stop()
            self._player.setSource(QUrl())
        if remove_zoom and self._unzoomed_path:
            warning = restore_unzoomed_recording(
                self.capture_path,
                self._unzoomed_path,
            )
            if warning:
                QMessageBox.warning(self, "Recording saved", warning)
        else:
            discard_unzoomed_recording(self._unzoomed_path)
        self._unzoomed_path = ""
        if self._remove_zoom is not None:
            self._remove_zoom.setEnabled(False)
        if self._zoom_status is not None:
            self._zoom_status.setText(
                "Automatic Smart Zoom removed"
                if remove_zoom
                else "Automatic Smart Zoom saved"
            )
        if self._player is not None:
            self._player.setSource(QUrl.fromLocalFile(self.capture_path))
        self._mark_saved()

    def _copy_capture(self) -> None:
        if hasattr(self, "_canvas"):
            QApplication.clipboard().setImage(self._canvas.rendered_image())
        else:
            mime = QMimeData()
            mime.setUrls([QUrl.fromLocalFile(self.capture_path)])
            QApplication.clipboard().setMimeData(mime)

    def _reveal_capture(self) -> None:
        subprocess.Popen(
            ["explorer.exe", f"/select,{self.capture_path}"],
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    def _toggle_playback(self) -> None:
        if self._player is None:
            return
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

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
        if not self._saved:
            discard_unzoomed_recording(self._unzoomed_path)
            self._unzoomed_path = ""
        super().closeEvent(event)
