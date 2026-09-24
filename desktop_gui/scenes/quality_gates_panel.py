from __future__ import annotations

from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QVBoxLayout, QWidget

from desktop_gui.data_loader import SimulationData
from desktop_gui.strings_tr import QUALITY_WARNING


class QualityGatesPanel(QWidget):
    def __init__(self, data: SimulationData, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(14)
        title = QLabel("Data Quality and Transparency")
        title.setObjectName("sceneTitle")
        root.addWidget(title)

        warning = QLabel(QUALITY_WARNING)
        warning.setProperty("warning", True)
        warning.setWordWrap(True)
        root.addWidget(warning)

        grid = QGridLayout()
        grid.setSpacing(12)
        for column, gate in enumerate(data.quality_gates):
            card = QFrame()
            card.setProperty("gateFailed", not gate["passed"])
            card_layout = QVBoxLayout(card)
            state = QLabel("PASSED" if gate["passed"] else "X FAILED")
            state.setProperty("statusPass" if gate["passed"] else "statusFail", True)
            label = QLabel(str(gate["label"]))
            label.setProperty("heading", True)
            label.setWordWrap(True)
            value = QLabel(f"{gate['value']} {gate['unit']}")
            value.setProperty("metric", True)
            requirement = QLabel(f"Requirement: {gate['requirement']}")
            requirement.setProperty("mono", True)
            card_layout.addWidget(state)
            card_layout.addWidget(label)
            card_layout.addStretch()
            card_layout.addWidget(value)
            card_layout.addWidget(requirement)
            grid.addWidget(card, 0, column)
        root.addLayout(grid)

        source = QFrame()
        source.setProperty("panel", True)
        source_layout = QGridLayout(source)
        metadata = data.replay["source_metadata"]
        rows = (
            ("Generation time", metadata["generated_utc"]),
            ("Git commit", metadata["git_commit"]),
            ("Conjunction source", data.replay["source"]),
            ("TLE source", data.replay["tle_source"]),
            ("Series generation", data.replay["distance_series_metadata"]["generation_mode"]),
        )
        for row, (label, value) in enumerate(rows):
            name = QLabel(label.upper())
            name.setProperty("muted", True)
            item = QLabel(str(value))
            item.setProperty("mono", True)
            item.setWordWrap(True)
            source_layout.addWidget(name, row, 0)
            source_layout.addWidget(item, row, 1)
        root.addWidget(source)

        disclaimer = QLabel(data.disclaimer)
        disclaimer.setWordWrap(True)
        disclaimer.setProperty("muted", True)
        root.addWidget(disclaimer)
        root.addStretch()
