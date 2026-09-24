from __future__ import annotations

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QVBoxLayout, QWidget

from desktop_gui.data_loader import SimulationData
from desktop_gui.strings_tr import QUALITY_WARNING, REPOSITORY_URL


class SummaryScene(QWidget):
    def __init__(self, data: SimulationData, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.data = data
        self._progress = 0
        self._targets = (
            data.counts["verified_snapshots"],
            data.counts["candidate_observations"],
            data.counts["proxy_positive_observations"],
            data.counts["held_out_observations"],
        )
        self._counter_labels: list[QLabel] = []
        self._timer = QTimer(self)
        self._timer.setInterval(24)
        self._timer.timeout.connect(self._advance_counts)

        root = QVBoxLayout(self)
        root.setContentsMargins(64, 38, 64, 38)
        root.setSpacing(16)
        root.addStretch()
        kicker = QLabel(data.paper["paper_code"])
        kicker.setProperty("accent", True)
        root.addWidget(kicker)
        title = QLabel("Mission Summary")
        title.setObjectName("heroTitle")
        root.addWidget(title)
        paper_title = QLabel(data.paper["title"])
        paper_title.setObjectName("heroSubtitle")
        paper_title.setWordWrap(True)
        root.addWidget(paper_title)

        frame = QFrame()
        frame.setProperty("panel", True)
        grid = QGridLayout(frame)
        labels = ("VERIFIED SNAPSHOTS", "CANDIDATE OBSERVATIONS", "PROXY POSITIVE", "HELD-OUT OBSERVATIONS")
        for column, label in enumerate(labels):
            value = QLabel("0")
            value.setProperty("metric", True)
            name = QLabel(label)
            name.setProperty("muted", True)
            self._counter_labels.append(value)
            grid.addWidget(value, 0, column)
            grid.addWidget(name, 1, column)
        root.addWidget(frame)

        gate_state = QLabel(
            f"Quality gates: {sum(gate['passed'] for gate in data.quality_gates)}/{len(data.quality_gates)} passed"
        )
        gate_state.setProperty("statusFail", True)
        root.addWidget(gate_state)
        warning = QLabel(QUALITY_WARNING)
        warning.setProperty("warning", True)
        warning.setWordWrap(True)
        root.addWidget(warning)
        author = QLabel(f"{data.paper['author']}  |  {data.paper['evidence_status']}")
        author.setProperty("muted", True)
        root.addWidget(author)
        repository = QLabel(REPOSITORY_URL)
        repository.setProperty("mono", True)
        repository.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        repository.setOpenExternalLinks(True)
        repository.setText(f'<a style="color:#37E6E0" href="{REPOSITORY_URL}">{REPOSITORY_URL}</a>')
        root.addWidget(repository)
        root.addStretch()

    def start_animation(self) -> None:
        self._progress = 0
        self._timer.start()

    def _advance_counts(self) -> None:
        self._progress += 1
        fraction = min(self._progress / 35.0, 1.0)
        eased = 1.0 - (1.0 - fraction) ** 3
        for label, target in zip(self._counter_labels, self._targets):
            label.setText(f"{round(target * eased):,}")
        if fraction >= 1.0:
            self._timer.stop()
