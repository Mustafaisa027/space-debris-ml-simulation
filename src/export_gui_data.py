"""Export a small, self-contained payload for the offline simulation GUI.

The encounter replay is sourced from a local validation artifact.  The paper
aggregate is intentionally embedded separately so a single replay row is never
mistaken for the manuscript's held-out evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


PAPER_AGGREGATE: dict[str, Any] = {
    "paper_code": "IAC-26-A6.IP.6",
    "title": (
        "Machine Learning-Based Simulation Approach for Assessing Space Debris "
        "Collision Risk in Low Earth Orbit"
    ),
    "author": "Mustafa İsa Oruçtutan",
    "evidence_status": "exploratory",
    "counts": {
        "verified_snapshots": 97,
        "candidate_observations": 19018,
        "proxy_positive_observations": 1469,
        "held_out_observations": 2795,
        "held_out_positive_observations": 223,
        "active_catalog_pairs": 1656,
        "held_out_catalog_pairs": 361,
    },
    "quality_gates": [
        {
            "key": "coverage",
            "label": "Snapshot coverage",
            "value": 80.833,
            "unit": "%",
            "requirement": ">= 90%",
            "passed": False,
        },
        {
            "key": "gap",
            "label": "Maximum snapshot gap",
            "value": 16.844,
            "unit": "h",
            "requirement": "<= 6 h",
            "passed": False,
        },
        {
            "key": "identical_run",
            "label": "Longest identical-input run",
            "value": 8,
            "unit": "bins",
            "requirement": "<= 6 bins",
            "passed": False,
        },
    ],
    "models": [
        {"key": "distance", "label": "Distance comparator", "pr_auc": 0.725, "f1": 0.510, "recall": 0.386},
        {"key": "logistic", "label": "Logistic regression", "pr_auc": 0.385, "f1": 0.283, "recall": 0.839},
        {"key": "decision_tree", "label": "Decision tree", "pr_auc": 0.137, "f1": 0.281, "recall": 0.274},
        {"key": "random_forest", "label": "Random forest", "pr_auc": 0.412, "f1": 0.299, "recall": 0.197},
        {"key": "svm", "label": "SVM", "pr_auc": 0.418, "f1": 0.406, "recall": 0.874},
        {"key": "xgboost", "label": "XGBoost", "pr_auc": 0.367, "f1": 0.396, "recall": 0.439},
        {"key": "lightgbm", "label": "LightGBM", "pr_auc": 0.408, "f1": 0.438, "recall": 0.623},
    ],
}


def _read_commented_csv(path: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    metadata: dict[str, str] = {}
    data_lines: list[str] = []
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip("\ufeff")
        if line.startswith("#"):
            key, separator, value = line[1:].strip().partition(":")
            if separator:
                metadata[key.strip()] = value.strip()
        elif line.strip():
            data_lines.append(line)
    if not data_lines:
        return metadata, []
    return metadata, list(csv.DictReader(data_lines))


def _number(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _scenario(row: dict[str, str], index: int) -> dict[str, Any]:
    object_1 = row.get("object_1", "Object A")
    object_2 = row.get("object_2", "Object B")
    synthetic = object_1.startswith("DEMO-") or object_2.startswith("DEMO-")
    return {
        "id": f"encounter-{index + 1}",
        "label": (
            f"{object_1} ({row.get('object_1_catalog_id', '')}) × "
            f"{object_2} ({row.get('object_2_catalog_id', '')})"
        ),
        "object_1": object_1,
        "object_2": object_2,
        "catalog_id_1": row.get("object_1_catalog_id", ""),
        "catalog_id_2": row.get("object_2_catalog_id", ""),
        "snapshot_utc": row.get("snapshot_utc", ""),
        "tca_utc": row.get("tca_utc", ""),
        "time_to_tca_min": _number(row, "time_to_tca_min"),
        "current_distance_km": _number(row, "current_distance_km"),
        "min_distance_km": _number(row, "min_distance_km"),
        "relative_velocity_km_s": _number(row, "relative_velocity_km_s"),
        "altitude_1_km": _number(row, "altitude_1_km"),
        "altitude_2_km": _number(row, "altitude_2_km"),
        "altitude_difference_km": _number(row, "altitude_difference_km"),
        "relative_radial_km": _number(row, "relative_radial_km"),
        "relative_intrack_km": _number(row, "relative_intrack_km"),
        "relative_crosstrack_km": _number(row, "relative_crosstrack_km"),
        "approach_angle_deg": _number(row, "approach_angle_deg"),
        "risk_score": _number(row, "risk_score"),
        "fixed_threshold_alarm": row.get("fixed_threshold_alarm", "0") == "1",
        "proxy_positive": row.get("risk_label", "0") == "1",
        "synthetic": synthetic,
        "replay_label": "Synthetic demonstration" if synthetic else "Validation snapshot replay",
        "distance_series": [],
    }


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _tle_distance_series(
    tle_path: Path,
    scenario: dict[str, Any],
    *,
    horizon_minutes: int = 720,
    step_minutes: int = 5,
) -> list[dict[str, Any]]:
    import numpy as np
    from skyfield.api import load

    from space_debris.core import build_satellites, read_tles

    satellites = build_satellites(read_tles(tle_path))
    by_catalog_id = {str(satellite.model.satnum): satellite for satellite in satellites}
    sat1 = by_catalog_id.get(str(scenario["catalog_id_1"]))
    sat2 = by_catalog_id.get(str(scenario["catalog_id_2"]))
    if sat1 is None or sat2 is None:
        raise ValueError(
            "Selected encounter catalogue IDs are not both present in the TLE file"
        )
    start = _parse_utc(scenario["snapshot_utc"])
    tca_offset = float(scenario["time_to_tca_min"])
    offsets = set(range(0, horizon_minutes + 1, step_minutes))
    offsets.add(round(tca_offset, 6))
    timescale = load.timescale()
    series: list[dict[str, Any]] = []
    for offset in sorted(offsets):
        current = start + timedelta(minutes=float(offset))
        time = timescale.from_datetime(current)
        p1 = sat1.at(time).position.km
        p2 = sat2.at(time).position.km
        distance = float(np.linalg.norm(p1 - p2))
        series.append(
            {
                "minute": float(offset),
                "utc_iso": current.isoformat().replace("+00:00", "Z"),
                "distance_km": round(distance, 3),
            }
        )
    return series


def export_gui_data(
    conjunctions_path: Path,
    distance_series_path: Path | None,
    output_path: Path,
    limit: int = 8,
    tle_path: Path | None = None,
) -> dict[str, Any]:
    metadata, rows = _read_commented_csv(conjunctions_path)
    if not rows:
        raise ValueError(f"No encounter rows found in {conjunctions_path}")
    rows.sort(key=lambda row: _number(row, "risk_score"), reverse=True)
    scenarios = [_scenario(row, index) for index, row in enumerate(rows[:limit])]

    series_metadata: dict[str, str] = {}
    if distance_series_path and distance_series_path.exists():
        series_metadata, series_rows = _read_commented_csv(distance_series_path)
        scenarios[0]["distance_series"] = [
            {
                "minute": _number(row, "minute"),
                "utc_iso": row.get("utc_iso", ""),
                "distance_km": _number(row, "distance_km"),
            }
            for row in series_rows
        ]
        # The archived curve is sampled every five minutes, whereas TCA is
        # refined between samples. Preserve both facts by inserting the
        # independently calculated event point into the visual series.
        refined_tca = float(scenarios[0]["time_to_tca_min"])
        if not any(
            abs(point["minute"] - refined_tca) < 1e-6
            for point in scenarios[0]["distance_series"]
        ):
            scenarios[0]["distance_series"].append(
                {
                    "minute": refined_tca,
                    "utc_iso": scenarios[0]["tca_utc"],
                    "distance_km": float(scenarios[0]["min_distance_km"]),
                }
            )
            scenarios[0]["distance_series"].sort(key=lambda point: point["minute"])
    elif tle_path and tle_path.exists():
        scenarios[0]["distance_series"] = _tle_distance_series(
            tle_path, scenarios[0]
        )

    payload = {
        "schema_version": 1,
        "paper": PAPER_AGGREGATE,
        "replay": {
            "label": "Local validation artifact",
            "not_to_scale": True,
            "source": str(conjunctions_path.as_posix()),
            "source_metadata": metadata,
            "distance_series_source": (
                str(distance_series_path.as_posix()) if distance_series_path else None
            ),
            "distance_series_metadata": series_metadata,
            "tle_source": str(tle_path.as_posix()) if tle_path else None,
            "scenarios": scenarios,
        },
        "disclaimer": (
            "Public TLE/SGP4 geometry without covariance or hard-body radius. "
            "The proxy label and risk score are not probability of collision (Pc)."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conjunctions",
        type=Path,
        default=Path("outputs/catalog_75_validation_pipeline/identified_conjunctions.csv"),
    )
    parser.add_argument(
        "--distance-series",
        type=Path,
        default=None,
    )
    parser.add_argument("--tle", type=Path, default=None)
    parser.add_argument(
        "--output", type=Path, default=Path("gui/data/simulation.json")
    )
    parser.add_argument("--limit", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    export_gui_data(
        args.conjunctions,
        args.distance_series,
        args.output,
        max(1, args.limit),
        args.tle,
    )
    print(f"GUI payload written to {args.output}")


if __name__ == "__main__":
    main()
