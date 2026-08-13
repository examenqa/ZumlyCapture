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
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QColorDialog,
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QPushButton,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

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
    if annotation.kind in {"pen", "highlight"}:
        painter.drawLine(annotation.start, annotation.end)
    elif annotation.kind == "rectangle":
        painter.drawRect(QRectF(annotation.start, annotation.end).normalized())
    elif annotation.kind == "arrow":
        painter.drawLine(annotation.start, annotation.end)
        angle = math.atan2(
            annotation.end.y() - annotation.start.y(),
            annotation.end.x() - annotation.start.x(),
        )
        size = max(12.0, annotation.width * 4.0)
        for offset in (math.pi * 0.84, -math.pi * 0.84):
            tip = QPointF(
                annotation.end.x() + math.cos(angle + offset) * size,
                annotation.end.y() + math.sin(angle + offset) * size,
            )
            painter.drawLine(annotation.end, tip)
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
            text, accepted = QInputDialog.getText(self, "Add text", "Annotation text:")
            if accepted and text.strip():
                self._annotations.append(
                    Annotation("text", point, point, self._color, 4.0, text.strip())
                )
                self.changed.emit()
                self.update()
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

    def __init__(self, capture_path: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.capture_path = os.path.abspath(capture_path)
        self._player = None
        self._audio = None
        self._play_button: QPushButton | None = None
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
        toolbar = QHBoxLayout()
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
        save = QPushButton("Save annotations")
        save.setStyleSheet("background: #2f7fca; font-weight: 650;")
        save.clicked.connect(self._save_image)
        toolbar.addWidget(save)
        root.addLayout(toolbar)
        root.addWidget(self._canvas, 1)

    def _build_video_preview(self, root: QVBoxLayout) -> None:
        if QMediaPlayer is None or QVideoWidget is None or QAudioOutput is None:
            message = QLabel("Video preview components are unavailable. Use Open to play the recording.")
            message.setAlignment(Qt.AlignmentFlag.AlignCenter)
            root.addWidget(message, 1)
            return
        video = QVideoWidget()
        video.setMinimumHeight(420)
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

    def _common_actions(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addStretch(1)
        copy = QPushButton("Copy")
        copy.clicked.connect(self._copy_capture)
        row.addWidget(copy)
        reveal = QPushButton("Show in folder")
        reveal.clicked.connect(self._reveal_capture)
        row.addWidget(reveal)
        open_button = QPushButton("Open")
        open_button.clicked.connect(self._open_capture)
        row.addWidget(open_button)
        close = QPushButton("Done")
        close.clicked.connect(self.close)
        row.addWidget(close)
        return row

    def _choose_color(self) -> None:
        chosen = QColorDialog.getColor(QColor("#ff4d67"), self, "Annotation color")
        self._canvas.set_color(chosen)

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
            self.saved.emit(self.capture_path)
            self.setWindowTitle(f"Saved · {Path(self.capture_path).name}")
        finally:
            if temp_path:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

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

    def _open_capture(self) -> None:
        os.startfile(self.capture_path)

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
