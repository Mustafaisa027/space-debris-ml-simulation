from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from desktop_gui.data_loader import SimulationData
from desktop_gui.strings_tr import APP_SUBTITLE, APP_TITLE, QUALITY_WARNING


class BriefingScene(QWidget):
    continue_requested = Signal()

    def __init__(self, data: SimulationData, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(72, 52, 72, 52)
        root.setSpacing(18)
        root.addStretch(2)

        kicker = QLabel(data.paper["paper_code"])
        kicker.setProperty("accent", True)
        root.addWidget(kicker)

        title = QLabel(APP_TITLE)
        title.setObjectName("heroTitle")
        root.addWidget(title)

        subtitle = QLabel(APP_SUBTITLE)
        subtitle.setObjectName("heroSubtitle")
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)

        mission = QFrame()
        mission.setProperty("panel", True)
        mission_layout = QHBoxLayout(mission)
        mission_layout.setContentsMargins(22, 18, 22, 18)
        for label, value in (
            ("DOGRULANMIS SNAPSHOT", data.counts["verified_snapshots"]),
            ("ADAY GOZLEM", data.counts["candidate_observations"]),
            ("REPLAY SENARYOSU", len(data.scenarios)),
            ("MODEL", len(data.models)),
        ):
            block = QVBoxLayout()
            value_label = QLabel(f"{value:,}")
            value_label.setProperty("metric", True)
            name_label = QLabel(label)
            name_label.setProperty("muted", True)
            block.addWidget(value_label)
            block.addWidget(name_label)
            mission_layout.addLayout(block)
        root.addWidget(mission)

        warning = QLabel(QUALITY_WARNING)
        warning.setProperty("warning", True)
        warning.setWordWrap(True)
        root.addWidget(warning)

        start = QPushButton("Gorevi Baslat")
        start.setProperty("primary", True)
        start.setCursor(Qt.CursorShape.PointingHandCursor)
        start.clicked.connect(self.continue_requested)
        root.addWidget(start, 0, Qt.AlignmentFlag.AlignLeft)
        root.addStretch(3)
        self._effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._effect)
        self._animation = QPropertyAnimation(self._effect, b"opacity", self)
        self._animation.setDuration(900)
        self._animation.setStartValue(0.0)
        self._animation.setEndValue(1.0)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)

    def start_animation(self) -> None:
        self._animation.start()
