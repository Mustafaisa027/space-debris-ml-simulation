from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QGridLayout, QLabel, QVBoxLayout, QWidget

from desktop_gui.data_loader import Scenario, SimulationData
from desktop_gui.scenes.common import metric_value, panel


def engineering_value(value: object, unit: str = "", digits: int = 3) -> str:
    if value is None:
        return "N/A"
    try:
        rendered = f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"
    return f"{rendered} {unit}".strip()


def format_tca_offset(delta_minutes: float) -> str:
    total_seconds = round(abs(float(delta_minutes)) * 60.0)
    if total_seconds == 0:
        return "TCA"
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    clock = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"TCA {'+' if delta_minutes > 0 else '-'} {clock}"


class ConjunctionPanel(QWidget):
    scenario_changed = Signal(int)

    def __init__(self, data: SimulationData, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.data = data
        self._scenario_index = 0
        self._current_minute = 0.0
        root = QVBoxLayout(self)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(14)

        header = QLabel("Konjonksiyon Detayi")
        header.setObjectName("sceneTitle")
        root.addWidget(header)
        self.selector = QComboBox()
        self.selector.addItems([scenario.label for scenario in data.scenarios])
        self.selector.currentIndexChanged.connect(self._select)
        root.addWidget(self.selector)

        encounter, encounter_layout = panel("Yakin Yaklasim")
        self.encounter_grid = QGridLayout()
        self.encounter_grid.setHorizontalSpacing(28)
        self.encounter_grid.setVerticalSpacing(10)
        encounter_layout.addLayout(self.encounter_grid)
        root.addWidget(encounter)

        live, live_layout = panel("Canli Playback", "Kanonik zaman cizelgesi ve secili cift")
        self.live_grid = QGridLayout()
        self.live_grid.setHorizontalSpacing(28)
        self.live_labels: dict[str, QLabel] = {}
        for column, (key, label) in enumerate(
            (
                ("simulation_time", "Simulasyon zamani"),
                ("tca_offset", "TCA'ya gore"),
                ("current_distance", "Guncel mesafe (screened seri)"),
                ("primary_velocity", "Primary hiz"),
                ("secondary_velocity", "Secondary hiz"),
            )
        ):
            name = QLabel(label.upper())
            name.setProperty("muted", True)
            value = QLabel("N/A")
            value.setProperty("metricSmall", True)
            self.live_labels[key] = value
            self.live_grid.addWidget(name, (column // 3) * 2, column % 3)
            self.live_grid.addWidget(value, (column // 3) * 2 + 1, column % 3)
        live_layout.addLayout(self.live_grid)
        root.addWidget(live)

        ric, ric_layout = panel("RIC Ayrisimi", "Radial / in-track / cross-track bilesenleri")
        self.ric_grid = QGridLayout()
        ric_layout.addLayout(self.ric_grid)
        root.addWidget(ric)

        provenance, provenance_layout = panel("Replay Kaynagi")
        source = QLabel(str(data.replay["source"]))
        source.setProperty("mono", True)
        source.setWordWrap(True)
        provenance_layout.addWidget(source)
        mode = QLabel(str(data.replay["distance_series_source"]))
        mode.setProperty("muted", True)
        provenance_layout.addWidget(mode)
        root.addWidget(provenance)
        root.addStretch()
        self.set_scenario(0)

    def set_scenario(self, index: int) -> None:
        index = min(max(index, 0), len(self.data.scenarios) - 1)
        self._scenario_index = index
        if self.selector.currentIndex() != index:
            self.selector.blockSignals(True)
            self.selector.setCurrentIndex(index)
            self.selector.blockSignals(False)
        self._populate(self.data.scenarios[index])
        self.set_playback_state(index, 0.0, None, None, None)

    def _select(self, index: int) -> None:
        self._scenario_index = index
        self._populate(self.data.scenarios[index])
        self.set_playback_state(index, 0.0, None, None, None)
        self.scenario_changed.emit(index)

    def set_playback_state(
        self,
        scenario_index: int,
        minute: float,
        current_distance_km: float | None,
        primary_velocity_km_s: float | None,
        secondary_velocity_km_s: float | None,
    ) -> None:
        if int(scenario_index) != self._scenario_index:
            return
        scenario = self.data.scenarios[self._scenario_index]
        self._current_minute = float(minute)
        self.live_labels["simulation_time"].setText(engineering_value(minute, "dk"))
        self.live_labels["tca_offset"].setText(format_tca_offset(minute - scenario.tca_minute))
        self.live_labels["current_distance"].setText(
            engineering_value(current_distance_km, "km")
        )
        self.live_labels["primary_velocity"].setText(
            engineering_value(primary_velocity_km_s, "km/s")
        )
        self.live_labels["secondary_velocity"].setText(
            engineering_value(secondary_velocity_km_s, "km/s")
        )

    def _populate(self, scenario: Scenario) -> None:
        self._clear_grid(self.encounter_grid)
        fields = (
            ("Encounter ID", scenario.id),
            ("Primary", f"{scenario.value('object_1')} ({scenario.value('catalog_id_1')})"),
            ("Secondary", f"{scenario.value('object_2')} ({scenario.value('catalog_id_2')})"),
            ("TCA (UTC)", scenario.value("tca_utc")),
            ("Minimum mesafe", engineering_value(scenario.value("min_distance_km"), "km")),
            ("Bagil hiz", engineering_value(scenario.value("relative_velocity_km_s"), "km/s")),
            ("Heuristic ranking score (NOT Pc)", metric_value(scenario.value("risk_score"), 6)),
            ("Sabit esik alarmi", metric_value(scenario.value("fixed_threshold_alarm"))),
            ("Proxy positive (distance + velocity rule)", metric_value(scenario.value("proxy_positive"))),
        )
        self._add_fields(self.encounter_grid, fields, columns=3)

        self._clear_grid(self.ric_grid)
        ric_fields = (
            ("Radial", f"{metric_value(scenario.value('relative_radial_km'))} km"),
            ("In-track", f"{metric_value(scenario.value('relative_intrack_km'))} km"),
            ("Cross-track", f"{metric_value(scenario.value('relative_crosstrack_km'))} km"),
            ("Yaklasma acisi", f"{metric_value(scenario.value('approach_angle_deg'))} deg"),
            ("Primary irtifa", engineering_value(scenario.value("altitude_1_km"), "km")),
            ("Secondary irtifa", engineering_value(scenario.value("altitude_2_km"), "km")),
        )
        self._add_fields(self.ric_grid, ric_fields, columns=3)

    @staticmethod
    def _clear_grid(grid: QGridLayout) -> None:
        while grid.count():
            item = grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    @staticmethod
    def _add_fields(grid: QGridLayout, fields: tuple[tuple[str, str], ...], columns: int) -> None:
        for index, (label, value) in enumerate(fields):
            row = (index // columns) * 2
            column = index % columns
            name = QLabel(label.upper())
            name.setProperty("muted", True)
            result = QLabel(value)
            result.setProperty("metricSmall", True)
            result.setWordWrap(True)
            grid.addWidget(name, row, column)
            grid.addWidget(result, row + 1, column)
