from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from desktop_gui.data_loader import SimulationData
from desktop_gui.strings_tr import DISTANCE_COMPARATOR_NOTE
from desktop_gui.theme.tokens import AMBER, CYAN, GRID, MUTED, PANEL, TEXT


METRICS = (
    ("PR-AUC", "pr_auc"),
    ("F1", "f1"),
    ("Recall", "recall"),
    ("ROC-AUC (local)", "roc_auc"),
    ("Precision (local)", "precision"),
    ("False alarm rate (local)", "false_alarm_rate"),
)


class ModelDashboard(QWidget):
    def __init__(self, data: SimulationData, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.data = data
        self.supplemental = data.supplemental_by_model()
        root = QVBoxLayout(self)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(12)

        header = QHBoxLayout()
        title_block = QVBoxLayout()
        title = QLabel("Model Comparison Dashboard")
        title.setObjectName("sceneTitle")
        title_block.addWidget(title)
        subtitle = QLabel("Fixed paper aggregation; local metrics come from a separate development report")
        subtitle.setProperty("muted", True)
        subtitle.setWordWrap(True)
        title_block.addWidget(subtitle)
        header.addLayout(title_block, 1)
        self.metric_selector = QComboBox()
        self.metric_selector.addItems([label for label, _key in METRICS])
        self.metric_selector.currentIndexChanged.connect(self._refresh_chart)
        header.addWidget(self.metric_selector)
        root.addLayout(header)

        chart_frame = QFrame()
        chart_frame.setProperty("panel", True)
        chart_layout = QVBoxLayout(chart_frame)
        self.chart = pg.PlotWidget(background=PANEL)
        self.chart.setToolTip(
            "Distance comparator: target-aligned control, not independent collision truth"
        )
        self.chart.setMinimumHeight(280)
        self.chart.showGrid(x=False, y=True, alpha=0.22)
        self.chart.setYRange(0.0, 1.05)
        self.chart.getAxis("left").setTextPen(QColor(MUTED))
        self.chart.getAxis("bottom").setTextPen(QColor(MUTED))
        chart_layout.addWidget(self.chart)
        root.addWidget(chart_frame, 1)

        self.table = QTableWidget(len(data.models), 7)
        self.table.setHorizontalHeaderLabels(
            ["Model", "PR-AUC", "F1", "Recall", "ROC-AUC*", "Precision*", "FAR*"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._populate_table()
        root.addWidget(self.table)

        note = QLabel(
            f"* Local 75-object random split report. Distance comparator: {DISTANCE_COMPARATOR_NOTE}. "
            "Sources must not be interpreted as one combined performance claim."
        )
        note.setProperty("warning", True)
        note.setWordWrap(True)
        root.addWidget(note)
        self._refresh_chart(0)

    def _metric_value(self, model: dict, key: str) -> float | None:
        if key in model:
            value = model.get(key)
        else:
            value = self.supplemental.get(str(model["key"]), {}).get(key)
        return float(value) if value is not None else None

    def _refresh_chart(self, index: int) -> None:
        self.chart.clear()
        _label, key = METRICS[index]
        models = self.data.models
        x_values = list(range(len(models)))
        values = [self._metric_value(model, key) for model in models]
        heights = [value if value is not None else 0.0 for value in values]
        brushes = [
            QColor(AMBER if index == 0 and value is not None else CYAN if value is not None else GRID)
            for index, value in enumerate(values)
        ]
        bars = pg.BarGraphItem(x=x_values, height=heights, width=0.62, brushes=brushes)
        self.chart.addItem(bars)
        ticks = [(position, str(model["label"])) for position, model in zip(x_values, models)]
        self.chart.getAxis("bottom").setTicks([ticks])
        self.chart.getAxis("bottom").setStyle(tickTextOffset=8, autoExpandTextSpace=True)
        for position, value in zip(x_values, values):
            text = "N/A" if value is None else f"{value:.3f}"
            item = pg.TextItem(text, color=TEXT if value is not None else MUTED, anchor=(0.5, 1.0))
            item.setPos(position, (value if value is not None else 0.02) + 0.02)
            self.chart.addItem(item)

    def _populate_table(self) -> None:
        keys = ("pr_auc", "f1", "recall", "roc_auc", "precision", "false_alarm_rate")
        for row, model in enumerate(self.data.models):
            name = QTableWidgetItem(str(model["label"]))
            if model["key"] == "distance":
                name.setForeground(QColor(AMBER))
            self.table.setItem(row, 0, name)
            for column, key in enumerate(keys, start=1):
                value = self._metric_value(model, key)
                cell = QTableWidgetItem("N/A" if value is None else f"{value:.3f}")
                cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, column, cell)
