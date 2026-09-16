from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_left
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "gui" / "data" / "simulation.json"


@dataclass(frozen=True)
class DistancePoint:
    minute: float
    utc_iso: str
    distance_km: float


@dataclass(frozen=True)
class Scenario:
    raw: dict[str, Any]
    distance_series: tuple[DistancePoint, ...]

    @property
    def id(self) -> str:
        return str(self.raw["id"])

    @property
    def label(self) -> str:
        return str(self.raw["label"])

    def value(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def tca_minute(self) -> float:
        return float(self.raw["time_to_tca_min"])

    @property
    def minimum_distance_km(self) -> float:
        return float(self.raw["min_distance_km"])

    def distance_at(self, minute: float) -> float:
        """Linearly interpolate the canonical distance series at a playback minute."""
        target = float(minute)
        points = self.distance_series
        if target <= points[0].minute:
            return points[0].distance_km
        if target >= points[-1].minute:
            return points[-1].distance_km
        right = bisect_left([point.minute for point in points], target)
        first, second = points[right - 1], points[right]
        fraction = (target - first.minute) / max(second.minute - first.minute, 1e-9)
        return first.distance_km + fraction * (second.distance_km - first.distance_km)


@dataclass(frozen=True)
class SimulationData:
    raw: dict[str, Any]
    scenarios: tuple[Scenario, ...]

    @property
    def paper(self) -> dict[str, Any]:
        return self.raw["paper"]

    @property
    def replay(self) -> dict[str, Any]:
        return self.raw["replay"]

    @property
    def disclaimer(self) -> str:
        return str(self.raw.get("disclaimer", ""))

    @property
    def counts(self) -> dict[str, Any]:
        return self.paper["counts"]

    @property
    def quality_gates(self) -> list[dict[str, Any]]:
        return list(self.paper["quality_gates"])

    @property
    def models(self) -> list[dict[str, Any]]:
        return list(self.paper["models"])

    @property
    def supplemental_rows(self) -> list[dict[str, Any]]:
        reports = self.raw.get("local_reruns", {}).get("supplemental_model_reports", [])
        rows: list[dict[str, Any]] = []
        for report in reports:
            for row in report.get("rows", []):
                item = dict(row)
                item["_report_key"] = report.get("key")
                item["_report_label"] = report.get("label")
                rows.append(item)
        return rows

    def supplemental_by_model(self) -> dict[str, dict[str, Any]]:
        aliases = {
            "distance": "fixed_threshold",
            "logistic": "logistic_regression",
            "decision_tree": "decision_tree",
            "random_forest": "random_forest",
            "svm": "svm",
            "xgboost": "xgboost",
            "lightgbm": "lightgbm",
        }
        grouped: dict[str, dict[str, Any]] = {}
        for row in self.supplemental_rows:
            key = str(row.get("model"))
            grouped[key] = row
        return {paper_key: grouped.get(row_key, {}) for paper_key, row_key in aliases.items()}


def load_simulation_data(path: Path | str = DEFAULT_DATA_PATH) -> SimulationData:
    payload_path = Path(path)
    with payload_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    _validate_payload(raw)
    scenarios = tuple(_load_scenario(item) for item in raw["replay"]["scenarios"])
    return SimulationData(raw=raw, scenarios=scenarios)


def _load_scenario(item: dict[str, Any]) -> Scenario:
    points = tuple(
        DistancePoint(
            minute=float(point["minute"]),
            utc_iso=str(point["utc_iso"]),
            distance_km=float(point["distance_km"]),
        )
        for point in item["distance_series"]
    )
    if not points:
        raise ValueError(f"Scenario {item.get('id')} has an empty distance_series")
    return Scenario(raw=dict(item), distance_series=points)


def _validate_payload(raw: dict[str, Any]) -> None:
    required_top = {"schema_version", "paper", "replay", "disclaimer"}
    missing = required_top.difference(raw)
    if missing:
        raise ValueError(f"simulation.json missing top-level keys: {sorted(missing)}")
    paper = raw["paper"]
    replay = raw["replay"]
    if len(paper.get("models", [])) != 7:
        raise ValueError("simulation.json must contain exactly 7 paper models")
    if len(paper.get("quality_gates", [])) != 3:
        raise ValueError("simulation.json must contain exactly 3 quality gates")
    if len(replay.get("scenarios", [])) != 8:
        raise ValueError("simulation.json must contain exactly 8 replay scenarios")
    scenario_required = {
        "id",
        "label",
        "object_1",
        "object_2",
        "tca_utc",
        "min_distance_km",
        "relative_velocity_km_s",
        "proxy_positive",
        "distance_series",
    }
    for scenario in replay["scenarios"]:
        missing_scenario = scenario_required.difference(scenario)
        if missing_scenario:
            raise ValueError(
                f"Scenario {scenario.get('id')} missing keys: {sorted(missing_scenario)}"
            )
