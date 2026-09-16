"""Export a small, self-contained payload for the offline simulation GUI.

The encounter replay is sourced from local validation artifacts. The paper
aggregate is intentionally embedded separately so a single replay row is never
mistaken for the manuscript's held-out evaluation.
"""

from __future__ import annotations

import argparse
import copy
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
    "author": "Mustafa \u0130sa Oru\u00e7tutan",
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

DEFAULT_SUPPLEMENTAL_MODEL_REPORTS = (
    {
        "key": "catalog_75_validation_random_split",
        "label": "75-object validation random split",
        "path": Path("outputs/catalog_75_validation_pipeline/model_comparison.csv"),
    },
    {
        "key": "history_pair_grouped_time_split",
        "label": "History pair-grouped time split",
        "path": Path("outputs/history/model_comparison_pair_grouped_time_split.csv"),
    },
)

LOCAL_RERUN_SOURCE_NOTE = (
    "Local validation/development reruns. This block is not an official "
    "manuscript result and must not be merged with the fixed paper aggregate."
)


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


def _optional_number(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(row: dict[str, str], key: str) -> int | None:
    value = row.get(key, "")
    if value in ("", None):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _scenario(row: dict[str, str], index: int) -> dict[str, Any]:
    object_1 = row.get("object_1", "Object A")
    object_2 = row.get("object_2", "Object B")
    synthetic = object_1.startswith("DEMO-") or object_2.startswith("DEMO-")
    return {
        "id": f"encounter-{index + 1}",
        "label": (
            f"{object_1} ({row.get('object_1_catalog_id', '')}) \u00d7 "
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


def _load_model_report(report_key: str, label: str, path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    metadata, rows = _read_commented_csv(path)
    normalized_rows = []
    for row in rows:
        model_name = row.get("model", "")
        normalized_rows.append(
            {
                "model": model_name,
                "status": "not_enough_data" if model_name == "not_enough_data" else "ok",
                "pr_auc": _optional_number(row, "pr_auc"),
                "roc_auc": _optional_number(row, "roc_auc"),
                "precision": _optional_number(row, "precision"),
                "recall": _optional_number(row, "recall"),
                "f1": _optional_number(row, "f1"),
                "accuracy": _optional_number(row, "accuracy"),
                "false_alarm_rate": _optional_number(row, "false_alarm_rate"),
                "false_positive": _optional_int(row, "false_positive"),
                "false_negative": _optional_int(row, "false_negative"),
                "true_positive": _optional_int(row, "true_positive"),
                "true_negative": _optional_int(row, "true_negative"),
                "train_rows": _optional_int(row, "train_rows"),
                "test_rows": _optional_int(row, "test_rows"),
                "train_pairs": _optional_int(row, "train_pairs"),
                "test_pairs": _optional_int(row, "test_pairs"),
                "train_positive_rows": _optional_int(row, "train_positive_rows"),
                "test_positive_rows": _optional_int(row, "test_positive_rows"),
                "train_positive_pairs": _optional_int(row, "train_positive_pairs"),
                "test_positive_pairs": _optional_int(row, "test_positive_pairs"),
                "train_positive_snapshots": _optional_int(row, "train_positive_snapshots"),
                "test_positive_snapshots": _optional_int(row, "test_positive_snapshots"),
                "excluded_rows": _optional_int(row, "excluded_rows"),
                "cutoff_utc": row.get("cutoff_utc") or None,
                "split": row.get("split") or None,
                "note": row.get("note") or None,
            }
        )

    return {
        "key": report_key,
        "label": label,
        "source": str(path.as_posix()),
        "source_metadata": metadata,
        "rows": normalized_rows,
    }


def _load_supplemental_model_reports() -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for spec in DEFAULT_SUPPLEMENTAL_MODEL_REPORTS:
        report = _load_model_report(spec["key"], spec["label"], spec["path"])
        if report is not None:
            reports.append(report)
    return reports


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


def _load_archived_distance_series(
    distance_series_path: Path,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    series_metadata, series_rows = _read_commented_csv(distance_series_path)
    return (
        series_metadata,
        [
            {
                "minute": _number(row, "minute"),
                "utc_iso": row.get("utc_iso", ""),
                "distance_km": _number(row, "distance_km"),
            }
            for row in series_rows
        ],
    )


def _annotate_series_with_tca(
    series: list[dict[str, Any]], scenario: dict[str, Any]
) -> list[dict[str, Any]]:
    annotated = list(series)
    refined_tca = float(scenario["time_to_tca_min"])
    if not any(abs(point["minute"] - refined_tca) < 1e-6 for point in annotated):
        annotated.append(
            {
                "minute": refined_tca,
                "utc_iso": scenario["tca_utc"],
                "distance_km": float(scenario["min_distance_km"]),
            }
        )
    annotated.sort(key=lambda point: point["minute"])
    return annotated


def _distance_series_matches_scenario(
    series: list[dict[str, Any]], scenario: dict[str, Any]
) -> bool:
    if not series:
        return False
    first_point = series[0]
    return abs(first_point["minute"]) < 1e-9 and abs(
        first_point["distance_km"] - float(scenario["current_distance_km"])
    ) < 1e-3


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
    archived_series: list[dict[str, Any]] = []
    archived_matches_first_scenario: bool | None = None
    distance_series_source: str | None = None
    if distance_series_path and distance_series_path.exists():
        series_metadata, archived_series = _load_archived_distance_series(distance_series_path)

    if tle_path and tle_path.exists():
        for scenario in scenarios:
            scenario["distance_series"] = _tle_distance_series(tle_path, scenario)
        series_metadata["generation_mode"] = "tle_all_scenarios"
        distance_series_source = "generated_from_tle"
        if archived_series:
            archived_matches_first_scenario = _distance_series_matches_scenario(
                archived_series, scenarios[0]
            )
    elif archived_series:
        archived_matches_first_scenario = _distance_series_matches_scenario(
            archived_series, scenarios[0]
        )
        if archived_matches_first_scenario:
            scenarios[0]["distance_series"] = _annotate_series_with_tca(
                archived_series, scenarios[0]
            )
            series_metadata["generation_mode"] = "archived_first_scenario_only"
            distance_series_source = str(distance_series_path.as_posix())
        else:
            series_metadata["generation_mode"] = "archived_series_mismatch_skipped"

    if archived_matches_first_scenario is not None:
        series_metadata["archived_matches_first_scenario"] = (
            "true" if archived_matches_first_scenario else "false"
        )

    local_rerun_reports = _load_supplemental_model_reports()

    payload = {
        "schema_version": 1,
        "paper": copy.deepcopy(PAPER_AGGREGATE),
        "replay": {
            "label": "Local validation artifact",
            "not_to_scale": True,
            "source": str(conjunctions_path.as_posix()),
            "source_metadata": metadata,
            "distance_series_source": distance_series_source,
            "distance_series_metadata": series_metadata,
            "tle_source": str(tle_path.as_posix()) if tle_path else None,
            "scenarios": scenarios,
        },
        "disclaimer": (
            "Public TLE/SGP4 geometry without covariance or hard-body radius. "
            "The proxy label and risk score are not probability of collision (Pc)."
        ),
    }
    if local_rerun_reports:
        payload["local_reruns"] = {
            "source_note": LOCAL_RERUN_SOURCE_NOTE,
            "supplemental_model_reports": local_rerun_reports,
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
    parser.add_argument("--distance-series", type=Path, default=None)
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
