from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from desktop_gui.data_loader import SimulationData
from desktop_gui.scenes.briefing_scene import BriefingScene
from desktop_gui.scenes.conjunction_panel import ConjunctionPanel
from desktop_gui.scenes.model_dashboard import ModelDashboard
from desktop_gui.scenes.orbit_scene import OrbitScene
from desktop_gui.scenes.quality_gates_panel import QualityGatesPanel
from desktop_gui.scenes.summary_scene import SummaryScene
from desktop_gui.strings_tr import (
    APP_TITLE,
    SCENE_BRIEFING,
    SCENE_CONJUNCTION,
    SCENE_MODELS,
    SCENE_ORBIT,
    SCENE_QUALITY,
    SCENE_SUMMARY,
)
from desktop_gui.theme import tokens as theme_tokens


class OrbitalSentinelWindow(QMainWindow):
    def __init__(
        self,
        data: SimulationData,
        enable_3d: bool | None = None,
        safe_mode_message: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if enable_3d is None:
            enable_3d = os.environ.get("QT_QPA_PLATFORM", "").lower() != "offscreen"
        self.data = data
        self.setWindowTitle(f"{APP_TITLE} | {data.paper['paper_code']}")
        self.resize(1440, 900)
        self.setMinimumSize(1080, 700)

        central = QWidget()
        shell = QHBoxLayout(central)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)
        self.setCentralWidget(central)

        rail = QFrame()
        rail.setObjectName("navigationRail")
        rail.setFixedWidth(192)
        rail_layout = QVBoxLayout(rail)
        rail_layout.setContentsMargins(12, 16, 12, 14)
        rail_layout.setSpacing(7)
        brand = QLabel("ORBITAL\nSENTINEL")
        brand.setObjectName("brand")
        rail_layout.addWidget(brand)
        code = QLabel(str(data.paper["paper_code"]))
        code.setProperty("mono", True)
        code.setProperty("muted", True)
        rail_layout.addWidget(code)
        rail_layout.addSpacing(18)

        self.stack = QStackedWidget()
        self.briefing_scene = BriefingScene(data)
        self.orbit_scene = OrbitScene(
            data,
            enable_3d=enable_3d,
            safe_mode_message=safe_mode_message,
        )
        self.conjunction_scene = ConjunctionPanel(data)
        self.model_scene = ModelDashboard(data)
        self.quality_scene = QualityGatesPanel(data)
        self.summary_scene = SummaryScene(data)
        self.scenes = (
            self.briefing_scene,
            self.orbit_scene,
            self.conjunction_scene,
            self.model_scene,
            self.quality_scene,
            self.summary_scene,
        )
        for scene in self.scenes:
            self.stack.addWidget(scene)

        labels = (
            SCENE_BRIEFING,
            SCENE_ORBIT,
            SCENE_CONJUNCTION,
            SCENE_MODELS,
            SCENE_QUALITY,
            SCENE_SUMMARY,
        )
        self.nav_buttons: list[QPushButton] = []
        for index, label in enumerate(labels):
            button = QPushButton(f"{index + 1:02d}  {label}")
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, target=index: self.show_scene(target))
            rail_layout.addWidget(button)
            self.nav_buttons.append(button)
        rail_layout.addStretch()
        status = QLabel("EXPLORATORY\nNON-OPERATIONAL")
        status.setProperty("warning", True)
        status.setWordWrap(True)
        rail_layout.addWidget(status)
        shell.addWidget(rail)
        shell.addWidget(self.stack, 1)

        self.briefing_scene.continue_requested.connect(lambda: self.show_scene(1))
        self.orbit_scene.details_requested.connect(lambda: self.show_scene(2))
        self.orbit_scene.scenario_changed.connect(self.conjunction_scene.set_scenario)
        self.conjunction_scene.scenario_changed.connect(self.orbit_scene.set_scenario)
        self.orbit_scene.playback_updated.connect(self.conjunction_scene.set_playback_state)
        self._animation: QPropertyAnimation | None = None
        self.show_scene(0, animate=False)
        QTimer.singleShot(0, self.briefing_scene.start_animation)
        QTimer.singleShot(3500, self._auto_advance_briefing)

    def _auto_advance_briefing(self) -> None:
        if self.stack.currentIndex() == 0 and self.isVisible():
            self.show_scene(1)

    def show_scene(self, index: int, animate: bool = True) -> None:
        index = min(max(int(index), 0), self.stack.count() - 1)
        if index != 1:
            self.orbit_scene.controller.pause()
        self.stack.setCurrentIndex(index)
        for button_index, button in enumerate(self.nav_buttons):
            button.setChecked(button_index == index)
        if index == 5:
            self.summary_scene.start_animation()
        if animate and index != 1:
            widget = self.stack.currentWidget()
            effect = QGraphicsOpacityEffect(widget)
            widget.setGraphicsEffect(effect)
            animation = QPropertyAnimation(effect, b"opacity", self)
            animation.setDuration(220)
            animation.setStartValue(0.15)
            animation.setEndValue(1.0)
            animation.setEasingCurve(QEasingCurve.Type.OutCubic)
            animation.finished.connect(lambda target=widget: target.setGraphicsEffect(None))
            self._animation = animation
            animation.start()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.orbit_scene.shutdown()
        super().closeEvent(event)


def load_stylesheet() -> str:
    path = Path(__file__).resolve().parent / "theme" / "mission_control.qss"
    stylesheet = path.read_text(encoding="utf-8")
    for name in (
        "BACKGROUND",
        "PANEL",
        "PANEL_ALT",
        "RAIL",
        "CYAN",
        "AMBER",
        "RED",
        "TEXT",
        "BRIGHT",
        "MUTED",
        "GRID",
        "BORDER",
        "GREEN",
        "PRIMARY_BUTTON",
        "CHECKED_BUTTON",
        "MONO_FONT",
        "SANS_FONT",
    ):
        stylesheet = stylesheet.replace(f"@{name}@", str(getattr(theme_tokens, name)))
    return stylesheet
