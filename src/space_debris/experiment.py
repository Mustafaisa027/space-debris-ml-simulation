from __future__ import annotations

import json
import hashlib
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_EXPERIMENT_CONFIG = Path("config/experiment_60_days.json")
# Active prospective confirmation experiment. v5 keeps prior cohorts isolated
# and separates scientific slot width from provider-request cadence.
# v2/v3 are retained as pilot/audit only and are intentionally not registered
# as claim-eligible inputs for the independent v5 confirmation run.
ACTIVE_EXPERIMENT_CONFIG = Path("config/experiment_15_days_v5.json")
PILOT_EXPERIMENT_CONFIG = Path("config/experiment_10_days_v2.json")
CLAIM_ELIGIBLE_EXPERIMENT_CONFIGS = (
    DEFAULT_EXPERIMENT_CONFIG,
    ACTIVE_EXPERIMENT_CONFIG,
)
DEFAULT_QUALITY_GATE = {
    "min_train_positive_rows": 30,
    "min_test_positive_rows": 20,
    "min_train_positive_pairs": 10,
    "min_test_positive_pairs": 5,
    "min_train_positive_snapshots": 5,
    "min_test_positive_snapshots": 3,
}
DEFAULT_EVIDENCE = {
    "primary_model": "xgboost",
    "primary_feature_set": "snapshot_only",
    "required_models": ["logistic_regression", "random_forest", "svm", "xgboost"],
    "inner_train_time_fraction": 0.75,
    "inner_validation_pair_fraction": 0.25,
    "inner_pair_seed": "iac26-operating-point-v1",
    "min_inner_train_positive_rows": 20,
    "min_inner_validation_positive_rows": 10,
    "min_inner_train_positive_pairs": 5,
    "min_inner_validation_positive_pairs": 3,
    "min_inner_train_positive_snapshots": 3,
    "min_inner_validation_positive_snapshots": 2,
    "bootstrap_replicates": 2000,
    "bootstrap_seed": 114764,
    "confidence_level": 0.95,
    "recall_noninferiority_margin": 0.05,
    "min_valid_bootstrap_fraction": 0.90,
    "adaptability_min_folds": 5,
    "adaptability_bootstrap_replicates": 10000,
    "adaptability_block_hours": 48,
    "adaptability_embargo_hours": 2,
    "adaptability_min_blocks": 10,
    "adaptability_min_unique_objects": 30,
    "adaptability_min_pairs": 30,
    "adaptability_min_positive_pairs_per_block": 5,
    "adaptability_min_positive_days_per_block": 2,
}


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _optional_utc(value: object, field: str) -> str | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO-8601 UTC string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 UTC string") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{field} must include the UTC timezone")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    experiment_id: str
    config_sha256: str
    duration_days: float
    poll_interval_hours: float
    minimum_provider_poll_interval_hours: float
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
    collection_start_utc: str | None
    collection_end_utc: str | None
    collection_slot_anchor_basis: str | None
    collection_slot_timestamp_field: str | None
    collection_slot_assignment: str | None
    min_snapshot_coverage_fraction: float | None
    max_snapshot_gap_hours: float | None
    min_tle_hash_diversity_fraction: float | None
    max_identical_tle_hash_run_bins: int | None
    inner_train_time_fraction: float
    inner_validation_pair_fraction: float
    inner_pair_seed: str
    primary_model: str
    primary_feature_set: str
    required_models: tuple[str, ...]
    min_inner_train_positive_rows: int
    min_inner_validation_positive_rows: int
    min_inner_train_positive_pairs: int
    min_inner_validation_positive_pairs: int
    min_inner_train_positive_snapshots: int
    min_inner_validation_positive_snapshots: int
    bootstrap_replicates: int
    bootstrap_seed: int
    confidence_level: float
    recall_noninferiority_margin: float
    min_valid_bootstrap_fraction: float
    adaptability_min_folds: int
    adaptability_bootstrap_replicates: int
    adaptability_block_hours: float
    adaptability_embargo_hours: float
    adaptability_min_blocks: int
    adaptability_min_unique_objects: int
    adaptability_min_pairs: int
    adaptability_min_positive_pairs_per_block: int
    adaptability_min_positive_days_per_block: int
    min_train_positive_rows: int
    min_test_positive_rows: int
    min_train_positive_pairs: int
    min_test_positive_pairs: int
    min_train_positive_snapshots: int
    min_test_positive_snapshots: int
    archive_collections: str
    snapshot_dir: str
    run_root: str
    history: str
    archive_import_report: str
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
        collection_window = evaluation.get("collection_window", {})
        evidence = {**DEFAULT_EVIDENCE, **evaluation.get("evidence", {})}
        quality_gate = {
            **DEFAULT_QUALITY_GATE,
            **evaluation.get("quality_gate", {}),
        }
        outputs = raw["outputs"]
        canonical_config = json.dumps(
            raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        config = ExperimentConfig(
            path=config_path,
            experiment_id=str(
                raw.get("experiment_id")
                or query.get(
                    "catalog_version",
                    f"{query['preset']}-legacy-unspecified",
                )
            ),
            config_sha256=hashlib.sha256(canonical_config).hexdigest(),
            duration_days=float(raw["duration_days"]),
            poll_interval_hours=float(raw["poll_interval_hours"]),
            minimum_provider_poll_interval_hours=float(
                raw.get(
                    "minimum_provider_poll_interval_hours",
                    raw["poll_interval_hours"],
                )
            ),
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
            collection_start_utc=_optional_utc(
                collection_window.get("start_utc"), "collection_window.start_utc"
            ),
            collection_end_utc=_optional_utc(
                collection_window.get("end_utc"), "collection_window.end_utc"
            ),
            collection_slot_anchor_basis=(
                str(collection_window["slot_anchor_basis"])
                if "slot_anchor_basis" in collection_window
                else None
            ),
            collection_slot_timestamp_field=(
                str(collection_window["slot_timestamp_field"])
                if "slot_timestamp_field" in collection_window
                else None
            ),
            collection_slot_assignment=(
                str(collection_window["slot_assignment"])
                if "slot_assignment" in collection_window
                else None
            ),
            min_snapshot_coverage_fraction=(
                float(collection_window["min_snapshot_coverage_fraction"])
                if "min_snapshot_coverage_fraction" in collection_window
                else None
            ),
            max_snapshot_gap_hours=(
                float(collection_window["max_snapshot_gap_hours"])
                if "max_snapshot_gap_hours" in collection_window
                else None
            ),
            min_tle_hash_diversity_fraction=(
                float(collection_window["min_tle_hash_diversity_fraction"])
                if "min_tle_hash_diversity_fraction" in collection_window
                else None
            ),
            max_identical_tle_hash_run_bins=(
                _positive_integer(
                    collection_window["max_identical_tle_hash_run_bins"],
                    "collection_window.max_identical_tle_hash_run_bins",
                )
                if "max_identical_tle_hash_run_bins" in collection_window
                else None
            ),
            inner_train_time_fraction=float(evidence["inner_train_time_fraction"]),
            inner_validation_pair_fraction=float(
                evidence["inner_validation_pair_fraction"]
            ),
            inner_pair_seed=str(evidence["inner_pair_seed"]),
            primary_model=str(evidence["primary_model"]),
            primary_feature_set=str(evidence["primary_feature_set"]),
            required_models=tuple(str(value) for value in evidence["required_models"]),
            min_inner_train_positive_rows=_positive_integer(
                evidence["min_inner_train_positive_rows"], "min_inner_train_positive_rows"
            ),
            min_inner_validation_positive_rows=_positive_integer(
                evidence["min_inner_validation_positive_rows"],
                "min_inner_validation_positive_rows",
            ),
            min_inner_train_positive_pairs=_positive_integer(
                evidence["min_inner_train_positive_pairs"], "min_inner_train_positive_pairs"
            ),
            min_inner_validation_positive_pairs=_positive_integer(
                evidence["min_inner_validation_positive_pairs"],
                "min_inner_validation_positive_pairs",
            ),
            min_inner_train_positive_snapshots=_positive_integer(
                evidence["min_inner_train_positive_snapshots"],
                "min_inner_train_positive_snapshots",
            ),
            min_inner_validation_positive_snapshots=_positive_integer(
                evidence["min_inner_validation_positive_snapshots"],
                "min_inner_validation_positive_snapshots",
            ),
            bootstrap_replicates=_positive_integer(
                evidence["bootstrap_replicates"], "bootstrap_replicates"
            ),
            bootstrap_seed=_nonnegative_integer(evidence["bootstrap_seed"], "bootstrap_seed"),
            confidence_level=float(evidence["confidence_level"]),
            recall_noninferiority_margin=float(
                evidence["recall_noninferiority_margin"]
            ),
            min_valid_bootstrap_fraction=float(
                evidence["min_valid_bootstrap_fraction"]
            ),
            adaptability_min_folds=_positive_integer(
                evidence["adaptability_min_folds"], "adaptability_min_folds"
            ),
            adaptability_bootstrap_replicates=_positive_integer(
                evidence["adaptability_bootstrap_replicates"],
                "adaptability_bootstrap_replicates",
            ),
            adaptability_block_hours=float(evidence["adaptability_block_hours"]),
            adaptability_embargo_hours=float(evidence["adaptability_embargo_hours"]),
            adaptability_min_blocks=_positive_integer(
                evidence["adaptability_min_blocks"], "adaptability_min_blocks"
            ),
            adaptability_min_unique_objects=_positive_integer(
                evidence["adaptability_min_unique_objects"],
                "adaptability_min_unique_objects",
            ),
            adaptability_min_pairs=_positive_integer(
                evidence["adaptability_min_pairs"], "adaptability_min_pairs"
            ),
            adaptability_min_positive_pairs_per_block=_positive_integer(
                evidence["adaptability_min_positive_pairs_per_block"],
                "adaptability_min_positive_pairs_per_block",
            ),
            adaptability_min_positive_days_per_block=_positive_integer(
                evidence["adaptability_min_positive_days_per_block"],
                "adaptability_min_positive_days_per_block",
            ),
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
            archive_collections=str(outputs.get("archive_collections", "collections")),
            snapshot_dir=str(outputs["snapshots"]),
            run_root=str(outputs["runs"]),
            history=str(outputs["history"]),
            archive_import_report=str(
                outputs.get(
                    "archive_import_report",
                    "outputs/history/collection_archive_import.json",
                )
            ),
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
        raise ValueError("poll_interval_hours must be at least 2")
    if config.minimum_provider_poll_interval_hours < 2:
        raise ValueError(
            "minimum_provider_poll_interval_hours must be at least 2"
        )
    if config.minimum_provider_poll_interval_hours > config.poll_interval_hours:
        raise ValueError(
            "minimum_provider_poll_interval_hours cannot exceed the scientific "
            "poll_interval_hours"
        )
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
        config.minimum_provider_poll_interval_hours,
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
        config.adaptability_block_hours,
        config.adaptability_embargo_hours,
    ]
    if not all(math.isfinite(float(value)) for value in numeric_values):
        raise ValueError("experiment numeric values must be finite")
    if config.candidate_threshold_km < max(config.fixed_threshold_km, config.label_threshold_km):
        raise ValueError("candidate_threshold_km must include both fixed and proxy-label thresholds")
    if not 0 < config.train_time_fraction < 1 or not 0 < config.test_pair_fraction < 1:
        raise ValueError("evaluation split fractions must be between 0 and 1")
    if config.adaptability_block_hours <= 0 or config.adaptability_embargo_hours < 0:
        raise ValueError("adaptability block hours must be positive and embargo non-negative")
    if config.adaptability_embargo_hours >= config.adaptability_block_hours:
        raise ValueError("adaptability embargo must be shorter than each test block")
    if (
        not config.time_column
        or not config.pair_seed
        or not config.catalog_version
        or not config.experiment_id
    ):
        raise ValueError("evaluation identity fields and catalog_version must be non-empty")
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", config.experiment_id):
        raise ValueError("experiment_id must be a safe lowercase versioned identifier")
    archive_collections = Path(config.archive_collections)
    if (
        archive_collections.is_absolute()
        or ".." in archive_collections.parts
        or not archive_collections.parts
        or archive_collections.name != "collections"
    ):
        raise ValueError(
            "outputs.archive_collections must be a relative path ending in 'collections'"
        )
    window_values = (
        config.collection_start_utc,
        config.collection_end_utc,
        config.collection_slot_anchor_basis,
        config.collection_slot_timestamp_field,
        config.collection_slot_assignment,
        config.min_snapshot_coverage_fraction,
        config.max_snapshot_gap_hours,
        config.min_tle_hash_diversity_fraction,
        config.max_identical_tle_hash_run_bins,
    )
    if any(value is not None for value in window_values) and not all(
        value is not None for value in window_values
    ):
        raise ValueError("collection_window fields must be configured together")
    if config.collection_start_utc is not None:
        if config.collection_slot_anchor_basis not in {
            "github_actions_cron_17_even_utc",
            "github_actions_multi_cron_utc",
        }:
            raise ValueError(
                "collection_window.slot_anchor_basis must identify a frozen GitHub cron anchor"
            )
        if config.collection_slot_timestamp_field != "snapshot_utc":
            raise ValueError(
                "collection_window.slot_timestamp_field must be 'snapshot_utc'"
            )
        if config.collection_slot_assignment != "half_open_snapshot_bins":
            raise ValueError(
                "collection_window.slot_assignment must be 'half_open_snapshot_bins'"
            )
        start = datetime.fromisoformat(config.collection_start_utc.replace("Z", "+00:00"))
        end = datetime.fromisoformat(config.collection_end_utc.replace("Z", "+00:00"))
        if end <= start:
            raise ValueError("collection_window.end_utc must be after start_utc")
        configured_days = (end - start).total_seconds() / 86400.0
        if not math.isclose(configured_days, config.duration_days, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("collection_window duration must equal duration_days")
        if not 0 < config.min_snapshot_coverage_fraction <= 1:
            raise ValueError("min_snapshot_coverage_fraction must be in (0, 1]")
        if not math.isfinite(config.max_snapshot_gap_hours) or config.max_snapshot_gap_hours <= 0:
            raise ValueError("max_snapshot_gap_hours must be positive and finite")
        if config.max_snapshot_gap_hours < config.poll_interval_hours:
            raise ValueError("max_snapshot_gap_hours cannot be below the normal poll interval")
        if not 0 < config.min_tle_hash_diversity_fraction <= 1:
            raise ValueError("min_tle_hash_diversity_fraction must be in (0, 1]")
    if not 0 < config.inner_train_time_fraction < 1:
        raise ValueError("inner_train_time_fraction must be between 0 and 1")
    if not 0 < config.inner_validation_pair_fraction < 1:
        raise ValueError("inner_validation_pair_fraction must be between 0 and 1")
    if not config.inner_pair_seed:
        raise ValueError("inner_pair_seed must be non-empty")
    if not config.primary_model or not config.required_models:
        raise ValueError("primary_model and required_models must be non-empty")
    if len(set(config.required_models)) != len(config.required_models):
        raise ValueError("required_models cannot contain duplicates")
    if config.primary_model not in config.required_models:
        raise ValueError("primary_model must be included in required_models")
    if config.primary_feature_set != "snapshot_only":
        raise ValueError(
            "primary_feature_set must be 'snapshot_only' for the frozen IAC protocol"
        )
    if not 0 < config.confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    if not 0 <= config.recall_noninferiority_margin < 1:
        raise ValueError("recall_noninferiority_margin must be in [0, 1)")
    if not 0 < config.min_valid_bootstrap_fraction <= 1:
        raise ValueError("min_valid_bootstrap_fraction must be in (0, 1]")
    if not config.catalog_version.endswith("-legacy-unspecified"):
        if len(config.catalog_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in config.catalog_sha256.lower()
        ):
            raise ValueError("frozen catalog_sha256 must be a 64-character hexadecimal digest")
    return config
