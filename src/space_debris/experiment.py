from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


DEFAULT_EXPERIMENT_CONFIG = Path("config/experiment_60_days.json")
DEFAULT_QUALITY_GATE = {
    "min_train_positive_rows": 30,
    "min_test_positive_rows": 20,
    "min_train_positive_pairs": 10,
    "min_test_positive_pairs": 5,
    "min_train_positive_snapshots": 5,
    "min_test_positive_snapshots": 3,
}


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    duration_days: float
    poll_interval_hours: float
    provider: str
    preset: str
    catalog_version: str
    catalog_sha256: str
    max_objects: int
    horizon_minutes: int
    step_minutes: int
    screening_step_seconds: float
    leo_min_altitude_km: float
    leo_max_altitude_km: float
    candidate_threshold_km: float
    fixed_threshold_km: float
    label_threshold_km: float
    label_relative_velocity_km_s: float
    max_tle_age_hours: float
    time_column: str
    train_time_fraction: float
    test_pair_fraction: float
    pair_seed: str
    min_train_positive_rows: int
    min_test_positive_rows: int
    min_train_positive_pairs: int
    min_test_positive_pairs: int
    min_train_positive_snapshots: int
    min_test_positive_snapshots: int
    snapshot_dir: str
    run_root: str
    history: str
    history_rebuild_report: str
    resimulated_runs: str
    resimulated_history: str
    resimulation_report: str
    time_split_report: str


def load_experiment_config(path: str | Path = DEFAULT_EXPERIMENT_CONFIG) -> ExperimentConfig:
    """Load and validate the single source of truth for the IAC experiment."""
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        query = raw["default_query"]
        simulation = raw["simulation"]
        evaluation = raw["evaluation"]
        quality_gate = {
            **DEFAULT_QUALITY_GATE,
            **evaluation.get("quality_gate", {}),
        }
        outputs = raw["outputs"]
        config = ExperimentConfig(
            path=config_path,
            duration_days=float(raw["duration_days"]),
            poll_interval_hours=float(raw["poll_interval_hours"]),
            provider=str(raw.get("provider", "auto")),
            preset=str(query["preset"]),
            catalog_version=str(
                query.get("catalog_version", f"{query['preset']}-legacy-unspecified")
            ),
            catalog_sha256=str(query.get("catalog_sha256", "")),
            max_objects=int(query["max_objects"]),
            horizon_minutes=int(simulation["horizon_minutes"]),
            step_minutes=int(simulation["step_minutes"]),
            screening_step_seconds=float(simulation["screening_step_seconds"]),
            leo_min_altitude_km=float(simulation["leo_min_altitude_km"]),
            leo_max_altitude_km=float(simulation["leo_max_altitude_km"]),
            candidate_threshold_km=float(simulation["candidate_threshold_km"]),
            fixed_threshold_km=float(simulation["fixed_threshold_km"]),
            label_threshold_km=float(simulation["label_threshold_km"]),
            label_relative_velocity_km_s=float(simulation["label_relative_velocity_km_s"]),
            max_tle_age_hours=float(simulation["max_tle_age_hours"]),
            time_column=str(evaluation["time_column"]),
            train_time_fraction=float(evaluation["train_time_fraction"]),
            test_pair_fraction=float(evaluation["test_pair_fraction"]),
            pair_seed=str(evaluation["pair_seed"]),
            min_train_positive_rows=_positive_integer(
                quality_gate["min_train_positive_rows"], "min_train_positive_rows"
            ),
            min_test_positive_rows=_positive_integer(
                quality_gate["min_test_positive_rows"], "min_test_positive_rows"
            ),
            min_train_positive_pairs=_positive_integer(
                quality_gate["min_train_positive_pairs"], "min_train_positive_pairs"
            ),
            min_test_positive_pairs=_positive_integer(
                quality_gate["min_test_positive_pairs"], "min_test_positive_pairs"
            ),
            min_train_positive_snapshots=_positive_integer(
                quality_gate["min_train_positive_snapshots"], "min_train_positive_snapshots"
            ),
            min_test_positive_snapshots=_positive_integer(
                quality_gate["min_test_positive_snapshots"], "min_test_positive_snapshots"
            ),
            snapshot_dir=str(outputs["snapshots"]),
            run_root=str(outputs["runs"]),
            history=str(outputs["history"]),
            history_rebuild_report=str(outputs["history_rebuild_report"]),
            resimulated_runs=str(outputs["resimulated_runs"]),
            resimulated_history=str(outputs["resimulated_history"]),
            resimulation_report=str(outputs["resimulation_report"]),
            time_split_report=str(outputs["time_split_report"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid experiment config {config_path}: {exc}") from exc

    if config.duration_days <= 0:
        raise ValueError("duration_days must be positive")
    if config.poll_interval_hours < 2:
        raise ValueError("poll_interval_hours must be at least 2 to respect provider cadence")
    if config.provider not in {"auto", "celestrak", "space-track"}:
        raise ValueError(f"Unsupported provider: {config.provider}")
    if config.max_objects < 2:
        raise ValueError("max_objects must be at least 2")
    if config.horizon_minutes <= 0 or config.step_minutes <= 0 or config.screening_step_seconds <= 0:
        raise ValueError("horizon, TCA step and screening step must be positive")
    if not 0 < config.leo_min_altitude_km < config.leo_max_altitude_km:
        raise ValueError("LEO altitude bounds are invalid")
    if min(
        config.candidate_threshold_km,
        config.fixed_threshold_km,
        config.label_threshold_km,
        config.label_relative_velocity_km_s,
    ) <= 0:
        raise ValueError("distance and velocity thresholds must be positive")
    numeric_values = [
        config.duration_days,
        config.poll_interval_hours,
        config.max_objects,
        config.horizon_minutes,
        config.step_minutes,
        config.screening_step_seconds,
        config.leo_min_altitude_km,
        config.leo_max_altitude_km,
        config.candidate_threshold_km,
        config.fixed_threshold_km,
        config.label_threshold_km,
        config.label_relative_velocity_km_s,
        config.max_tle_age_hours,
        config.train_time_fraction,
        config.test_pair_fraction,
    ]
    if not all(math.isfinite(float(value)) for value in numeric_values):
        raise ValueError("experiment numeric values must be finite")
    if config.candidate_threshold_km < max(config.fixed_threshold_km, config.label_threshold_km):
        raise ValueError("candidate_threshold_km must include both fixed and proxy-label thresholds")
    if not 0 < config.train_time_fraction < 1 or not 0 < config.test_pair_fraction < 1:
        raise ValueError("evaluation split fractions must be between 0 and 1")
    if not config.time_column or not config.pair_seed or not config.catalog_version:
        raise ValueError("evaluation identity fields and catalog_version must be non-empty")
    if not config.catalog_version.endswith("-legacy-unspecified"):
        if len(config.catalog_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in config.catalog_sha256.lower()
        ):
            raise ValueError("frozen catalog_sha256 must be a 64-character hexadecimal digest")
    return config
