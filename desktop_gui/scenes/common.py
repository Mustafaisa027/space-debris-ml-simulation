from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget


def panel(title: str, subtitle: str | None = None) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setProperty("panel", True)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(18, 16, 18, 16)
    layout.setSpacing(10)
    heading = QLabel(title)
    heading.setProperty("heading", True)
    layout.addWidget(heading)
    if subtitle:
        detail = QLabel(subtitle)
        detail.setProperty("muted", True)
        detail.setWordWrap(True)
        layout.addWidget(detail)
    return frame, layout


def metric_value(value: object, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "EVET" if value else "HAYIR"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def clear_layout(layout: QVBoxLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()


def centered_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setWordWrap(True)
    return label

