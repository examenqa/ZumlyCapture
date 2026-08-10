"""Small capture-target pickers independent of the video editor."""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)


class WindowPickerDialog(QDialog):
    def __init__(self, windows: list[dict], title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(520, 420)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Choose a window:"))
        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        for window in windows:
            item = QListWidgetItem(str(window.get("name") or window.get("title") or "Window"))
            item.setData(Qt.ItemDataRole.UserRole, dict(window))
            self._list.addItem(item)
        if self._list.count():
            self._list.setCurrentRow(0)
        self._list.itemDoubleClicked.connect(lambda _item: self.accept())
        layout.addWidget(self._list, 1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_window(self) -> dict | None:
        item = self._list.currentItem()
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        return dict(value) if isinstance(value, dict) else None


class RegionSelector(QWidget):
    region_selected = Signal(object)
    cancelled = Signal()

    def __init__(self) -> None:
        super().__init__(None)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._origin: QPoint | None = None
        self._current: QPoint | None = None

    def begin(self) -> None:
        screens = self.screen().virtualSiblings() if self.screen() is not None else []
        geometry = QRect()
        for screen in screens:
            geometry = geometry.united(screen.geometry())
        if geometry.isNull() and self.screen() is not None:
            geometry = self.screen().geometry()
        self.setGeometry(geometry)
        self.show()
        self.raise_()
        self.activateWindow()
        self.setFocus()

    def selection_rect(self) -> QRect:
        if self._origin is None or self._current is None:
            return QRect()
        return QRect(self._origin, self._current).normalized()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 105))
        selection = self.selection_rect()
        if not selection.isNull():
            painter.fillRect(selection, QColor(255, 255, 255, 35))
            painter.setPen(QPen(QColor(79, 156, 255), 2))
            painter.drawRect(selection)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(
                selection.adjusted(8, 8, -8, -8),
                Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
                f"{selection.width()} × {selection.height()}",
            )
        else:
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Drag to select a region · Esc to cancel",
            )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._origin = event.position().toPoint()
            self._current = self._origin
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._origin is not None:
            self._current = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._current = event.position().toPoint()
        selection = self.selection_rect()
        if selection.width() < 8 or selection.height() < 8:
            self._origin = None
            self._current = None
            self.update()
            return
        top_left = self.mapToGlobal(selection.topLeft())
        payload = {
            "left": top_left.x(),
            "top": top_left.y(),
            "width": selection.width(),
            "height": selection.height(),
        }
        self.hide()
        QTimer.singleShot(160, lambda: self.region_selected.emit(payload))

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.hide()
            self.cancelled.emit()
            return
        super().keyPressEvent(event)
