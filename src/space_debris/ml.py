from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from space_debris.provenance import write_csv_text_with_provenance
from space_debris.tle_validation import partition_tle_hash_quality


# NOTE: ``risk_score``, ``risk_label`` and ``fixed_threshold_alarm`` are
# intentionally absent. The proxy label is a function of min_distance_km and
# relative_velocity_km_s; the canonical model receives those two physical
# predictors and measures recovery of that transparent rule. ``risk_score`` is
# excluded because it is a monotone transform of the same quantities and would
# add a duplicate target proxy. A separately reported feature ablation removes
# the two rule-defining predictors to quantify how much performance survives.
#
# The relative_*/radial_velocity/tangential_velocity/approach_angle features
# (ROADMAP_YOL1.md GOREV 5) are geometric decompositions computed from the
# position/velocity vectors. Individually, the RIC components describe
# direction, but their joint Euclidean norm reconstructs ``min_distance_km``;
# the strict feature ablation therefore removes the entire RIC position trio.
# Radial/tangential velocity and approach_angle are evaluated at snapshot time
# rather than at TCA, and relative_inclination_deg is orbital-plane geometry.
FEATURES = [
    "time_to_tca_min",
    "current_distance_km",
    "min_distance_km",
    "relative_velocity_km_s",
    "altitude_difference_km",
    "max_tle_age_hours",
    "relative_radial_km",
    "relative_intrack_km",
    "relative_crosstrack_km",
    "relative_inclination_deg",
    "radial_velocity_km_s",
    "tangential_velocity_km_s",
    "approach_angle_deg",
]

# Pre-specified primary predictors available at the observation snapshot,
# before the forward SGP4 TCA search is evaluated.  The publication model must
# not receive the future-propagation quantities that define (or reconstruct)
# the deterministic proxy label.
SNAPSHOT_ONLY_FEATURES = [
    "current_distance_km",
    "altitude_difference_km",
    "max_tle_age_hours",
    "radial_velocity_km_s",
    "tangential_velocity_km_s",
    "approach_angle_deg",
]

FEATURE_SETS = {
    "snapshot_only": tuple(SNAPSHOT_ONLY_FEATURES),
    "full_rule_recovery": tuple(FEATURES),
}


def feature_set_by_name(name: str) -> list[str]:
    """Return a copy of a pre-registered feature set."""
    try:
        return list(FEATURE_SETS[name])
    except KeyError as exc:
        raise ValueError(
            f"Unknown feature set {name!r}; expected one of {sorted(FEATURE_SETS)}"
        ) from exc

FEATURES_WITHOUT_LABEL_RULE = [
    feature
    for feature in FEATURES
    if feature
    not in {
        "min_distance_km",
        "relative_velocity_km_s",
        "relative_radial_km",
        "relative_intrack_km",
        "relative_crosstrack_km",
    }
]

FORBIDDEN_MODEL_FEATURES = frozenset(
    {"risk_label", "fixed_threshold_alarm", "risk_score"}
)


def _validated_feature_columns(feature_columns: Sequence[str] | None) -> list[str]:
    selected = list(FEATURES if feature_columns is None else feature_columns)
    if not selected:
        raise ValueError("feature_columns cannot be empty")
    duplicates = sorted({feature for feature in selected if selected.count(feature) > 1})
    if duplicates:
        raise ValueError(f"feature_columns contains duplicates: {duplicates}")
    forbidden = sorted(set(selected) & FORBIDDEN_MODEL_FEATURES)
    if forbidden:
        raise ValueError(f"feature_columns contains forbidden target/leakage columns: {forbidden}")
    return selected

# Reported for every model. pr_auc/roc_auc lead the report and are the
# metrics the paper leans on; accuracy is kept for context but is
# deliberately NOT the headline -- with rare positives, a model that always
# predicts "not risky" still scores >99% accuracy while missing every
# conjunction (see ROADMAP_YOL1.md observations G5/G6).
REPORT_COLUMNS = [
    "model",
    "pr_auc",
    "roc_auc",
    "precision",
    "recall",
    "f1",
    "accuracy",
    "false_alarm_rate",
    "false_positive",
    "false_negative",
    "true_positive",
    "true_negative",
    "train_rows",
    "test_rows",
    "train_pairs",
    "test_pairs",
    "train_positive_rows",
    "test_positive_rows",
    "train_positive_pairs",
    "test_positive_pairs",
    "train_positive_snapshots",
    "test_positive_snapshots",
    "observation_span_days",
    "evaluation_window_start_utc",
    "evaluation_window_end_utc",
    "expected_snapshot_slots",
    "observed_snapshot_times",
    "occupied_snapshot_slots",
    "duplicate_slot_snapshot_times",
    "duplicate_slot_rows_excluded",
    "unique_tle_input_hashes",
    "tle_input_hash_diversity_fraction",
    "max_identical_tle_hash_run_bins",
    "maximum_observed_tle_age_hours",
    "orphan_history_rows_excluded",
    "coverage_source",
    "snapshot_coverage_fraction",
    "max_nominal_bin_gap_hours",
    "max_snapshot_gap_hours",
    "window_excluded_rows",
    "excluded_rows",
    "cutoff_utc",
    "split",
    "note",
]


PAIR_COLUMNS = ["object_1", "object_2"]
PAIR_CATALOG_ID_COLUMNS = ["object_1_catalog_id", "object_2_catalog_id"]
PAIR_SPLIT_SEED = "iac26-pair-split-v1"

WINDOW_QUALITY_KEYS = (
    "evaluation_window_start_utc",
    "evaluation_window_end_utc",
    "expected_snapshot_slots",
    "observed_snapshot_times",
    "occupied_snapshot_slots",
    "duplicate_slot_snapshot_times",
    "duplicate_slot_rows_excluded",
    "unique_tle_input_hashes",
    "tle_input_hash_diversity_fraction",
    "max_identical_tle_hash_run_bins",
    "maximum_observed_tle_age_hours",
    "orphan_history_rows_excluded",
    "coverage_source",
    "snapshot_coverage_fraction",
    "max_nominal_bin_gap_hours",
    "max_snapshot_gap_hours",
    "window_excluded_rows",
)


class InsufficientGroupedSplitError(ValueError):
    """Raised when independent pair cohorts cannot form train and test sets."""


@dataclass(frozen=True)
class SplitResult:
    train: pd.DataFrame
    test: pd.DataFrame
    excluded: pd.DataFrame
    split_name: str
    metadata: dict[str, object]


SUPPORT_KEYS = (
    "train_positive_rows",
    "test_positive_rows",
    "train_positive_pairs",
    "test_positive_pairs",
    "train_positive_snapshots",
    "test_positive_snapshots",
)


def _positive_partition_support(
    frame: pd.DataFrame,
    time_column: str | None,
) -> dict[str, int | None]:
    positive = frame.loc[frame["risk_label"].astype(int).eq(1)]
    if positive.empty:
        return {"rows": 0, "pairs": 0, "snapshots": 0}
    has_pair_identity = all(column in positive.columns for column in PAIR_COLUMNS)
    pairs = int(_canonical_pair_ids(positive).nunique()) if has_pair_identity else None
    snapshots = (
        int(_validated_times(positive, time_column).nunique())
        if time_column
        else None
    )
    return {"rows": int(len(positive)), "pairs": pairs, "snapshots": snapshots}


def _split_positive_support(
    split: SplitResult,
    time_column: str | None,
) -> dict[str, int | None]:
    train = _positive_partition_support(split.train, time_column)
    test = _positive_partition_support(split.test, time_column)
    return {
        "train_positive_rows": train["rows"],
        "test_positive_rows": test["rows"],
        "train_positive_pairs": train["pairs"],
        "test_positive_pairs": test["pairs"],
        "train_positive_snapshots": train["snapshots"],
        "test_positive_snapshots": test["snapshots"],
    }


def _observation_span_days(df: pd.DataFrame, time_column: str | None) -> float | None:
    if not time_column or df.empty:
        return None
    timestamps = _validated_times(df, time_column)
    return float((timestamps.max() - timestamps.min()).total_seconds() / 86400.0)


def _apply_collection_window(
    df: pd.DataFrame,
    time_column: str | None,
    *,
    start_utc: str | None,
    end_utc: str | None,
    poll_interval_hours: float | None,
    snapshot_records: Sequence[dict[str, object]] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Select the frozen half-open experiment window and measure cadence.

    Coverage partitions the frozen half-open window into equal half-open cadence
    bins using authoritative snapshot timestamps. Retries in one bin cannot hide
    a missing bin elsewhere, and delayed GitHub jobs are never rounded into a
    future nominal slot.
    """
    values = (start_utc, end_utc, poll_interval_hours)
    if not any(value is not None for value in values):
        return df, {key: None for key in WINDOW_QUALITY_KEYS}
    if not all(value is not None for value in values) or not time_column:
        raise ValueError(
            "collection window requires start_utc, end_utc, poll_interval_hours and time_column"
        )
    poll_interval_hours = float(poll_interval_hours)
    if not np.isfinite(poll_interval_hours) or poll_interval_hours <= 0:
        raise ValueError("poll_interval_hours must be positive and finite")
    start = pd.Timestamp(start_utc)
    end = pd.Timestamp(end_utc)
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("collection window timestamps must include a timezone")
    start = start.tz_convert("UTC")
    end = end.tz_convert("UTC")
    if end <= start:
        raise ValueError("collection window end_utc must be after start_utc")
    duration_hours = float((end - start).total_seconds() / 3600.0)
    expected_slots_float = duration_hours / poll_interval_hours
    expected_slots = int(round(expected_slots_float))
    if expected_slots <= 0 or not np.isclose(expected_slots_float, expected_slots, atol=1e-9):
        raise ValueError("collection window duration must be divisible by poll_interval_hours")

    timestamps = _validated_times(df, time_column)
    history_window_mask = timestamps.ge(start) & timestamps.lt(end)
    record_collection_ids: set[str] | None = None
    if snapshot_records is not None:
        normalized_records: list[tuple[str, pd.Timestamp, str]] = []
        seen_record_ids: dict[str, tuple[pd.Timestamp, str]] = {}
        for record in snapshot_records:
            if not isinstance(record, dict):
                raise ValueError("snapshot_records entries must be mappings")
            collection_id = str(record.get("collection_id", "")).strip()
            if not collection_id:
                raise ValueError("snapshot_records require non-empty collection_id")
            timestamp = pd.Timestamp(record.get("snapshot_utc"))
            if timestamp.tzinfo is None:
                raise ValueError("snapshot_records timestamps must include a timezone")
            timestamp = timestamp.tz_convert("UTC")
            input_sha256 = str(record.get("input_sha256", "")).strip().lower()
            if len(input_sha256) != 64 or any(
                character not in "0123456789abcdef" for character in input_sha256
            ):
                raise ValueError("snapshot_records require a valid input_sha256")
            identity = (timestamp, input_sha256)
            if collection_id in seen_record_ids and seen_record_ids[collection_id] != identity:
                raise ValueError(f"Conflicting snapshot_records for collection_id={collection_id}")
            seen_record_ids[collection_id] = identity
        normalized_records = sorted(
            (
                (collection_id, identity[0], identity[1])
                for collection_id, identity in seen_record_ids.items()
            ),
            key=lambda item: (item[1], item[0]),
        )
        candidates = [
            (collection_id, timestamp, input_sha256)
            for collection_id, timestamp, input_sha256 in normalized_records
            if start <= timestamp < end
        ]
        record_collection_ids = {
            collection_id for collection_id, _, _ in normalized_records
        }
        if "collection_id" not in df.columns:
            raise ValueError("snapshot_records require collection_id in model history")
        history_ids = df["collection_id"].fillna("").astype(str).str.strip()
        if history_ids.eq("").any():
            raise ValueError("model history collection_id cannot be empty")
        coverage_source = "resimulation_snapshot_records"
    else:
        candidates = [
            (timestamp.isoformat(), timestamp, None)
            for timestamp in pd.Index(timestamps.loc[history_window_mask].unique()).sort_values()
        ]
        coverage_source = "history_candidate_rows"

    unique_times = pd.Index(sorted({timestamp for _, timestamp, _ in candidates}))
    chosen_by_slot: dict[int, tuple[pd.Timestamp, str, str | None]] = {}
    for collection_id, timestamp, input_sha256 in candidates:
        elapsed_hours = float((timestamp - start).total_seconds() / 3600.0)
        slot = int(np.floor(elapsed_hours / poll_interval_hours))
        candidate = (timestamp, collection_id, input_sha256)
        if slot not in chosen_by_slot or candidate < chosen_by_slot[slot]:
            chosen_by_slot[slot] = candidate
    occupied = sorted(chosen_by_slot)
    chosen_times = {candidate[0] for candidate in chosen_by_slot.values()}
    chosen_collection_ids = {candidate[1] for candidate in chosen_by_slot.values()}
    chosen_input_hashes = {
        candidate[2] for candidate in chosen_by_slot.values() if candidate[2] is not None
    }
    ordered_input_hashes = [
        chosen_by_slot[slot][2] for slot in occupied if chosen_by_slot[slot][2] is not None
    ]
    maximum_identical_hash_run = 0
    current_identical_hash_run = 0
    previous_input_hash: str | None = None
    for input_hash in ordered_input_hashes:
        if input_hash == previous_input_hash:
            current_identical_hash_run += 1
        else:
            current_identical_hash_run = 1
            previous_input_hash = input_hash
        maximum_identical_hash_run = max(
            maximum_identical_hash_run, current_identical_hash_run
        )
    if snapshot_records is not None:
        history_ids = df["collection_id"].fillna("").astype(str).str.strip()
        selected_mask = history_window_mask & history_ids.isin(chosen_collection_ids)
        orphan_history_rows = int(
            (history_window_mask & ~history_ids.isin(record_collection_ids)).sum()
        )
    else:
        selected_mask = history_window_mask & timestamps.isin(chosen_times)
        orphan_history_rows = 0
    selected = df.loc[selected_mask].copy()
    duplicate_slot_times = max(0, len(unique_times) - len(chosen_times))
    duplicate_slot_rows = int(
        (history_window_mask & ~selected_mask).sum() - orphan_history_rows
    )
    if occupied:
        slot_gaps = [occupied[0] * poll_interval_hours]
        slot_gaps.extend(
            (right - left) * poll_interval_hours
            for left, right in zip(occupied, occupied[1:])
        )
        # In a half-open window, a complete cadence still leaves one normal
        # poll interval between the final nominal snapshot and ``end``.
        slot_gaps.append((expected_slots - occupied[-1]) * poll_interval_hours)
        max_nominal_gap_hours = float(max(slot_gaps))
        ordered_snapshot_times = sorted(chosen_times)
        actual_gaps = [
            float((ordered_snapshot_times[0] - start).total_seconds() / 3600.0)
        ]
        actual_gaps.extend(
            float((right - left).total_seconds() / 3600.0)
            for left, right in zip(
                ordered_snapshot_times, ordered_snapshot_times[1:]
            )
        )
        actual_gaps.append(
            float((end - ordered_snapshot_times[-1]).total_seconds() / 3600.0)
        )
        max_gap_hours = float(max(actual_gaps))
    else:
        max_nominal_gap_hours = duration_hours
        max_gap_hours = duration_hours
    quality = {
        "evaluation_window_start_utc": start.isoformat().replace("+00:00", "Z"),
        "evaluation_window_end_utc": end.isoformat().replace("+00:00", "Z"),
        "expected_snapshot_slots": expected_slots,
        "observed_snapshot_times": int(len(unique_times)),
        "occupied_snapshot_slots": int(len(occupied)),
        "duplicate_slot_snapshot_times": int(duplicate_slot_times),
        "duplicate_slot_rows_excluded": duplicate_slot_rows,
        "unique_tle_input_hashes": (
            int(len(chosen_input_hashes)) if snapshot_records is not None else None
        ),
        "tle_input_hash_diversity_fraction": (
            float(len(chosen_input_hashes) / len(occupied))
            if snapshot_records is not None and occupied
            else (0.0 if snapshot_records is not None else None)
        ),
        "max_identical_tle_hash_run_bins": (
            int(maximum_identical_hash_run) if snapshot_records is not None else None
        ),
        "maximum_observed_tle_age_hours": None,
        "orphan_history_rows_excluded": orphan_history_rows,
        "coverage_source": coverage_source,
        "snapshot_coverage_fraction": float(len(occupied) / expected_slots),
        "max_nominal_bin_gap_hours": max_nominal_gap_hours,
        "max_snapshot_gap_hours": max_gap_hours,
        "window_excluded_rows": int((~selected_mask).sum()),
    }
    return selected, quality


def _partition_tle_hash_quality(
    frame: pd.DataFrame,
    snapshot_records: Sequence[dict[str, object]],
) -> dict[str, object]:
    return partition_tle_hash_quality(frame, snapshot_records)


def _validated_minimum_support(
    minimum_support: dict[str, int] | None,
) -> dict[str, int]:
    if minimum_support is None:
        return {}
    unknown = sorted(set(minimum_support) - set(SUPPORT_KEYS))
    if unknown:
        raise ValueError(f"Unknown minimum-support keys: {unknown}")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) or value <= 0
        for value in minimum_support.values()
    ):
        raise ValueError("minimum-support values must be positive integers")
    return {key: int(value) for key, value in minimum_support.items()}


def _canonical_pair_ids(df: pd.DataFrame) -> pd.Series:
    missing = [column for column in PAIR_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Pair-grouped split requires columns: {missing}")
    if df[PAIR_COLUMNS].isna().any().any():
        raise ValueError("Pair identity columns cannot contain missing values")
    use_catalog_ids = all(column in df.columns for column in PAIR_CATALOG_ID_COLUMNS)
    if use_catalog_ids:
        catalog = df[PAIR_CATALOG_ID_COLUMNS]
        use_catalog_ids = not catalog.isna().any().any() and not (
            catalog.astype(str).apply(lambda column: column.str.strip().eq("")).any().any()
        )
    identity_columns = PAIR_CATALOG_ID_COLUMNS if use_catalog_ids else PAIR_COLUMNS
    left = df[identity_columns[0]].astype(str).str.strip()
    right = df[identity_columns[1]].astype(str).str.strip()
    if (left == "").any() or (right == "").any():
        raise ValueError("Pair identity columns cannot contain empty values")
    return pd.Series(
        np.where(left <= right, left + "|" + right, right + "|" + left),
        index=df.index,
        name="canonical_pair_id",
    )


def _validated_times(df: pd.DataFrame, time_column: str) -> pd.Series:
    if time_column not in df.columns:
        raise ValueError(f"time_column={time_column!r} not found in dataset")
    return pd.to_datetime(df[time_column], utc=True, errors="raise", format="ISO8601")


def _pair_is_test(pair_id: str, test_pair_fraction: float, pair_seed: str) -> bool:
    digest = hashlib.sha256(f"{pair_seed}|{pair_id}".encode("utf-8")).digest()
    score = int.from_bytes(digest[:8], "big") / 2**64
    return score < test_pair_fraction


def _pair_grouped_time_split(
    df: pd.DataFrame,
    time_column: str,
    train_time_fraction: float = 0.75,
    test_pair_fraction: float = 0.25,
    pair_seed: str = PAIR_SPLIT_SEED,
) -> SplitResult:
    """Hold out deterministic pairs in a strictly future snapshot window.

    SHA-256 assigns every canonical pair permanently to the train or test
    group. The cutoff lies between complete timestamp blocks. Training uses
    only train-pair observations before the cutoff; testing uses only test-pair
    observations at/after it. Cross-quadrant rows are retained as ``excluded``
    metadata rather than silently leaking into the opposite partition.
    """
    if not 0.0 < train_time_fraction < 1.0:
        raise ValueError("train_time_fraction must be between 0 and 1")
    if not 0.0 < test_pair_fraction < 1.0:
        raise ValueError("test_pair_fraction must be between 0 and 1")
    timestamps = _validated_times(df, time_column)
    pair_ids = _canonical_pair_ids(df)
    unique_times = pd.Index(timestamps.unique()).sort_values()
    if len(unique_times) < 2:
        raise InsufficientGroupedSplitError("Need at least two distinct snapshot timestamps.")
    cutoff_index = min(max(1, int(np.ceil(len(unique_times) * train_time_fraction))), len(unique_times) - 1)
    cutoff = unique_times[cutoff_index]

    pair_assignment = {
        pair_id: _pair_is_test(pair_id, test_pair_fraction, pair_seed)
        for pair_id in pair_ids.unique()
    }
    is_test_pair = pair_ids.map(pair_assignment).astype(bool)
    train_mask = (~is_test_pair) & (timestamps < cutoff)
    test_mask = is_test_pair & (timestamps >= cutoff)
    excluded_mask = ~(train_mask | test_mask)
    train = df.loc[train_mask].copy()
    test = df.loc[test_mask].copy()
    excluded = df.loc[excluded_mask].copy()
    if train.empty or test.empty:
        raise InsufficientGroupedSplitError(
            "Pair/time intersection produced an empty train or test partition."
        )

    train_pairs = set(pair_ids[train_mask])
    test_pairs = set(pair_ids[test_mask])
    if train_pairs & test_pairs:
        raise AssertionError("Canonical pair leakage detected in grouped split")
    train_times = timestamps[train_mask]
    test_times = timestamps[test_mask]
    if not train_times.max() < test_times.min():
        raise AssertionError("Chronological leakage detected in grouped split")

    metadata = {
        "cutoff_utc": cutoff.isoformat(),
        "pair_seed": pair_seed,
        "test_pair_fraction": test_pair_fraction,
        "train_time_fraction": train_time_fraction,
        "train_pairs": len(train_pairs),
        "test_pairs": len(test_pairs),
        "train_rows": len(train),
        "test_rows": len(test),
        "excluded_rows": len(excluded),
    }
    return SplitResult(train, test, excluded, "pair_grouped_time", metadata)


def _split_data(
    df: pd.DataFrame,
    time_column: str | None,
    *,
    train_time_fraction: float = 0.75,
    test_pair_fraction: float = 0.25,
    pair_seed: str = PAIR_SPLIT_SEED,
) -> SplitResult:
    y = df["risk_label"].astype(int)
    if time_column:
        return _pair_grouped_time_split(
            df, time_column, train_time_fraction, test_pair_fraction, pair_seed
        )

    stratify = y if y.value_counts().min() >= 2 else None
    train, test = train_test_split(
        df, test_size=0.35, random_state=42, stratify=stratify
    )
    return SplitResult(
        train.copy(), test.copy(), df.iloc[0:0].copy(), "random",
        {"train_rows": len(train), "test_rows": len(test), "excluded_rows": 0},
    )


def _ranking_score(model, x_test):
    """Continuous score for ranking-based metrics (PR-AUC/ROC-AUC).

    Prefers predict_proba (probability of the positive class), falls back to
    decision_function (e.g. an SVC without probability calibration enabled),
    and finally to hard 0/1 predictions if neither is available.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(x_test)
    return model.predict(x_test)


def _distance_baseline_score(test: pd.DataFrame) -> np.ndarray:
    """Continuous ranking score for the classical distance-only baseline.

    Smaller miss distance means higher conjunction risk, hence the negative
    sign.  The configured fixed-distance alarm remains the baseline's binary
    operating point for precision/recall/F1 and confusion counts; using the
    continuous distance here keeps PR-AUC/ROC-AUC meaningful instead of
    reducing the ranking to two score levels.
    """
    return -test["min_distance_km"].astype(float).to_numpy()


def _ranking_metrics(y_true, scores) -> tuple[float | None, float | None, str]:
    """average_precision_score / roc_auc_score are undefined with a single
    class in y_true; report None with a note instead of letting sklearn raise.
    """
    if pd.Series(y_true).nunique() < 2:
        return None, None, "single class in test split; PR-AUC/ROC-AUC undefined"
    return (
        float(average_precision_score(y_true, scores)),
        float(roc_auc_score(y_true, scores)),
        "",
    )


def _build_models(scale_pos_weight: float) -> dict:
    """Every learner is imbalance-aware: class_weight='balanced' re-weights
    the rare positive class for LogReg/RF/SVM, and XGBoost's scale_pos_weight
    (n_negative / n_positive of the TRAINING split) plays the same role.
    fixed_threshold is not a learner (None) -- it needs no weighting.
    """
    models: dict = {
        "fixed_threshold": None,
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
        ),
        "decision_tree": DecisionTreeClassifier(
            random_state=42, class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"
        ),
        "svm": make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced")),
    }
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            eval_metric="logloss", random_state=42, scale_pos_weight=scale_pos_weight
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = LGBMClassifier(
            random_state=42,
            class_weight="balanced",
            verbosity=-1,
        )
    except Exception:
        pass
    return models


def _not_enough_data_row(model: str, note: str, train_rows=None, test_rows=None, split=None) -> dict:
    return {
        "model": model,
        "pr_auc": None,
        "roc_auc": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "accuracy": None,
        "false_alarm_rate": None,
        "false_positive": None,
        "false_negative": None,
        "true_positive": None,
        "true_negative": None,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "train_pairs": None,
        "test_pairs": None,
        "train_positive_rows": None,
        "test_positive_rows": None,
        "train_positive_pairs": None,
        "test_positive_pairs": None,
        "train_positive_snapshots": None,
        "test_positive_snapshots": None,
        "observation_span_days": None,
        "evaluation_window_start_utc": None,
        "evaluation_window_end_utc": None,
        "expected_snapshot_slots": None,
        "observed_snapshot_times": None,
        "occupied_snapshot_slots": None,
        "duplicate_slot_snapshot_times": None,
        "duplicate_slot_rows_excluded": None,
        "unique_tle_input_hashes": None,
        "tle_input_hash_diversity_fraction": None,
        "max_identical_tle_hash_run_bins": None,
        "maximum_observed_tle_age_hours": None,
        "orphan_history_rows_excluded": None,
        "coverage_source": None,
        "snapshot_coverage_fraction": None,
        "max_nominal_bin_gap_hours": None,
        "max_snapshot_gap_hours": None,
        "window_excluded_rows": None,
        "excluded_rows": None,
        "cutoff_utc": None,
        "split": split,
        "note": note,
    }


def _evaluate_split(
    split: SplitResult,
    split_name: str | None = None,
    feature_columns: Sequence[str] = FEATURES,
    time_column: str | None = None,
    observation_span_days: float | None = None,
    dataset_quality: dict[str, object] | None = None,
) -> list[dict]:
    """Fit every model on one train/test split and score it. Shared by the
    single chronological split (compare_models) and each fold of
    TimeSeriesSplit cross-validation (time_series_cv_report).
    """
    train = split.train
    test = split.test
    feature_columns = _validated_feature_columns(feature_columns)
    x_train, x_test = train[feature_columns], test[feature_columns]
    y_train = train["risk_label"].astype(int)
    y_test = test["risk_label"].astype(int)
    split_name = split_name or split.split_name
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    models = _build_models(scale_pos_weight)
    trainable = y_train.nunique() >= 2
    support = _split_positive_support(split, time_column)
    support["observation_span_days"] = observation_span_days
    support.update(dataset_quality or {})

    rows: list[dict] = []
    for name, model in models.items():
        if name == "fixed_threshold":
            # Purely distance-based baseline: no training required. The
            # configured alarm is its operating point; continuous negative
            # miss distance is its ranking score for PR-AUC/ROC-AUC.
            pred = test["fixed_threshold_alarm"].astype(int).to_numpy()
            scores = _distance_baseline_score(test)
        elif not trainable:
            row = _not_enough_data_row(
                name,
                "Training split had a single class; model skipped.",
                train_rows=int(len(y_train)),
                test_rows=int(len(y_test)),
                split=split_name,
            )
            row.update(**support)
            rows.append(row)
            continue
        else:
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            scores = _ranking_score(model, x_test)

        pr_auc, roc_auc, ranking_note = _ranking_metrics(y_test, scores)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, pred, average="binary", zero_division=0
        )
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        false_alarm_rate = float(fp / (fp + tn)) if (fp + tn) else None
        rows.append(
            {
                "model": name,
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy_score(y_test, pred),
                "false_alarm_rate": false_alarm_rate,
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
                "true_negative": int(tn),
                "train_rows": int(len(y_train)),
                "test_rows": int(len(y_test)),
                "train_pairs": split.metadata.get("train_pairs"),
                "test_pairs": split.metadata.get("test_pairs"),
                **support,
                "excluded_rows": split.metadata.get("excluded_rows", len(split.excluded)),
                "cutoff_utc": split.metadata.get("cutoff_utc"),
                "split": split_name,
                "note": ranking_note,
            }
        )

    if "xgboost" not in models:
        row = _not_enough_data_row(
            "xgboost",
            "xgboost is not installed.",
            train_rows=int(len(y_train)),
            test_rows=int(len(y_test)),
            split=split_name,
        )
        row.update(**support)
        rows.append(row)
    if "lightgbm" not in models:
        row = _not_enough_data_row(
            "lightgbm",
            "lightgbm is not installed.",
            train_rows=int(len(y_train)),
            test_rows=int(len(y_test)),
            split=split_name,
        )
        row.update(**support)
        rows.append(row)

    return rows


def model_predictions_for_plotting(
    dataset_path: Path,
    time_column: str | None = None,
    *,
    train_time_fraction: float = 0.75,
    test_pair_fraction: float = 0.25,
    pair_seed: str = PAIR_SPLIT_SEED,
) -> dict:
    """Per-model test-split predictions/scores/fitted estimator, for
    PUBLICATION FIGURES ONLY (PR curves, confusion matrices, feature
    importance in space_debris.plots) -- compare_models()/
    time_series_cv_report() remain the source of truth for the numeric report.

    Returns {model_name: {"y_true", "scores", "pred", "model"}}; empty dict if
    there is not enough data to form a meaningful split (mirrors
    compare_models()'s "not enough data" guard).
    """
    df = pd.read_csv(dataset_path, comment="#")
    missing = [feature for feature in FEATURES + ["risk_label", "fixed_threshold_alarm"] if feature not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if df.empty or df["risk_label"].nunique() < 2 or len(df) < 6:
        return {}

    try:
        split = _split_data(
            df,
            time_column,
            train_time_fraction=train_time_fraction,
            test_pair_fraction=test_pair_fraction,
            pair_seed=pair_seed,
        )
    except InsufficientGroupedSplitError:
        return {}
    train, test = split.train, split.test
    if train["risk_label"].nunique() < 2 or test["risk_label"].nunique() < 2:
        return {}
    x_train, x_test = train[FEATURES], test[FEATURES]
    y_train, y_test = train["risk_label"].astype(int), test["risk_label"].astype(int)
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    models = _build_models(scale_pos_weight)
    trainable = y_train.nunique() >= 2

    predictions: dict = {}
    for name, model in models.items():
        if name == "fixed_threshold":
            pred = test["fixed_threshold_alarm"].astype(int).to_numpy()
            scores = _distance_baseline_score(test)
            fitted_model = None
        elif not trainable:
            continue
        else:
            model.fit(x_train, y_train)
            pred = np.asarray(model.predict(x_test))
            scores = np.asarray(_ranking_score(model, x_test))
            fitted_model = model

        predictions[name] = {
            "y_true": y_test.to_numpy(),
            "scores": scores,
            "pred": pred,
            "model": fitted_model,
        }

    return predictions


def compare_to_baseline_pr_auc(report: pd.DataFrame) -> str:
    """Human-readable PR-AUC comparison of each model against the classical
    distance-only ranking baseline.  Its binary fixed-distance operating point
    is still used for confusion counts; PR-AUC uses continuous negative miss
    distance so the ranking comparison is not reduced to two score levels.
    """
    if "fixed_threshold" not in report["model"].values:
        return "fixed_threshold baseline not present in report."
    baseline_pr_auc = report.loc[report["model"] == "fixed_threshold", "pr_auc"].iloc[0]
    if pd.isna(baseline_pr_auc):
        return "fixed_threshold PR-AUC undefined (single class in test split)."

    lines = [f"fixed_threshold PR-AUC = {baseline_pr_auc:.4f} (baseline)"]
    for _, row in report.iterrows():
        if row["model"] == "fixed_threshold" or pd.isna(row["pr_auc"]):
            continue
        delta = row["pr_auc"] - baseline_pr_auc
        verdict = "observed higher than" if delta > 0 else "not observed higher than"
        lines.append(
            f"{row['model']} PR-AUC = {row['pr_auc']:.4f} ({verdict} baseline, delta={delta:+.4f}; "
            "uncertainty is reported separately)"
        )
    return "\n".join(lines)


def compare_models(
    dataset_path: Path,
    report_path: Path,
    time_column: str | None = None,
    *,
    source: str = "",
    config_summary: str = "",
    train_time_fraction: float = 0.75,
    test_pair_fraction: float = 0.25,
    pair_seed: str = PAIR_SPLIT_SEED,
    feature_columns: Sequence[str] | None = None,
    minimum_support: dict[str, int] | None = None,
    minimum_observation_span_days: float | None = None,
    required_catalog_version: str | None = None,
    required_catalog_sha256: str | None = None,
    collection_start_utc: str | None = None,
    collection_end_utc: str | None = None,
    poll_interval_hours: float | None = None,
    min_snapshot_coverage_fraction: float | None = None,
    max_snapshot_gap_hours: float | None = None,
    min_tle_hash_diversity_fraction: float | None = None,
    max_identical_tle_hash_run_bins: int | None = None,
    maximum_tle_age_hours: float | None = None,
    snapshot_records: Sequence[dict[str, object]] | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(dataset_path, comment="#")
    selected_features = _validated_feature_columns(feature_columns)
    required_support = _validated_minimum_support(minimum_support)
    if minimum_observation_span_days is not None:
        minimum_observation_span_days = float(minimum_observation_span_days)
        if not np.isfinite(minimum_observation_span_days) or minimum_observation_span_days <= 0:
            raise ValueError("minimum_observation_span_days must be positive and finite")
        if not time_column:
            raise ValueError("minimum_observation_span_days requires time_column")
    window_values = (
        collection_start_utc,
        collection_end_utc,
        poll_interval_hours,
        min_snapshot_coverage_fraction,
        max_snapshot_gap_hours,
        min_tle_hash_diversity_fraction,
        max_identical_tle_hash_run_bins,
    )
    if any(value is not None for value in window_values) and not all(
        value is not None for value in window_values
    ):
        raise ValueError("collection window and cadence gates must be configured together")
    if min_snapshot_coverage_fraction is not None:
        min_snapshot_coverage_fraction = float(min_snapshot_coverage_fraction)
        max_snapshot_gap_hours = float(max_snapshot_gap_hours)
        if not 0 < min_snapshot_coverage_fraction <= 1:
            raise ValueError("min_snapshot_coverage_fraction must be in (0, 1]")
        if not np.isfinite(max_snapshot_gap_hours) or max_snapshot_gap_hours <= 0:
            raise ValueError("max_snapshot_gap_hours must be positive and finite")
        min_tle_hash_diversity_fraction = float(min_tle_hash_diversity_fraction)
        if not 0 < min_tle_hash_diversity_fraction <= 1:
            raise ValueError("min_tle_hash_diversity_fraction must be in (0, 1]")
        if (
            isinstance(max_identical_tle_hash_run_bins, bool)
            or not isinstance(max_identical_tle_hash_run_bins, Integral)
            or max_identical_tle_hash_run_bins <= 0
        ):
            raise ValueError("max_identical_tle_hash_run_bins must be a positive integer")
        if snapshot_records is None:
            raise ValueError("TLE hash-diversity gates require snapshot_records")
    if maximum_tle_age_hours is not None:
        maximum_tle_age_hours = float(maximum_tle_age_hours)
        if not np.isfinite(maximum_tle_age_hours) or maximum_tle_age_hours <= 0:
            raise ValueError("maximum_tle_age_hours must be positive and finite")
        if "max_tle_age_hours" not in df.columns:
            raise ValueError("TLE-age quality gate requires max_tle_age_hours column")
    missing = [
        feature
        for feature in selected_features + ["risk_label", "fixed_threshold_alarm", "min_distance_km"]
        if feature not in df.columns
    ]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df, window_quality = _apply_collection_window(
        df,
        time_column,
        start_utc=collection_start_utc,
        end_utc=collection_end_utc,
        poll_interval_hours=poll_interval_hours,
        snapshot_records=snapshot_records,
    )
    if required_catalog_version is not None:
        if "catalog_version" not in df.columns:
            raise ValueError("Frozen-cohort evaluation requires catalog_version column")
        observed_versions = sorted(
            value for value in df["catalog_version"].dropna().astype(str).str.strip().unique()
            if value
        )
        if observed_versions != [required_catalog_version]:
            raise ValueError(
                "Catalog cohort mismatch: "
                f"required={required_catalog_version!r}, observed={observed_versions!r}"
            )
    if required_catalog_sha256 is not None:
        if "catalog_sha256" not in df.columns:
            raise ValueError("Frozen-cohort evaluation requires catalog_sha256 column")
        observed_hashes = sorted(
            value for value in df["catalog_sha256"].dropna().astype(str).str.strip().unique()
            if value
        )
        if observed_hashes != [required_catalog_sha256]:
            raise ValueError(
                "Catalog ID-set hash mismatch: "
                f"required={required_catalog_sha256!r}, observed={observed_hashes!r}"
            )
    if maximum_tle_age_hours is not None:
        observed_ages = pd.to_numeric(df["max_tle_age_hours"], errors="coerce")
        if observed_ages.isna().any() or not np.isfinite(observed_ages).all():
            raise ValueError("max_tle_age_hours must contain only finite numeric values")
        window_quality["maximum_observed_tle_age_hours"] = float(observed_ages.max())
    pair_support_requested = any(
        key in required_support
        for key in ("train_positive_pairs", "test_positive_pairs")
    )
    if pair_support_requested:
        missing_pair = [column for column in PAIR_COLUMNS if column not in df.columns]
        if missing_pair:
            raise ValueError(
                f"Positive-pair minimum support requires pair columns: {missing_pair}"
            )
    snapshot_support_requested = any(
        key in required_support
        for key in ("train_positive_snapshots", "test_positive_snapshots")
    )
    if snapshot_support_requested and not time_column:
        raise ValueError("Positive-snapshot minimum support requires time_column")

    if not source:
        source = f"dataset={dataset_path}"
    if not config_summary:
        config_summary = f"time_column={time_column}"
    config_summary = f"{config_summary}, features={'|'.join(selected_features)}"
    if required_support:
        support_summary = "|".join(
            f"{key}:{required_support[key]}" for key in SUPPORT_KEYS if key in required_support
        )
        config_summary = f"{config_summary}, minimum_support={support_summary}"
    if minimum_observation_span_days is not None:
        config_summary = (
            f"{config_summary}, minimum_observation_span_days="
            f"{minimum_observation_span_days:.6f}"
        )
    if collection_start_utc is not None:
        config_summary = (
            f"{config_summary}, collection_window={collection_start_utc}/{collection_end_utc}, "
            f"poll_interval_hours={poll_interval_hours}, "
            f"min_snapshot_coverage_fraction={min_snapshot_coverage_fraction}, "
            f"max_snapshot_gap_hours={max_snapshot_gap_hours}, "
            f"min_tle_hash_diversity_fraction={min_tle_hash_diversity_fraction}, "
            f"max_identical_tle_hash_run_bins={max_identical_tle_hash_run_bins}, "
            f"coverage_source={'resimulation_report' if snapshot_records is not None else 'history'}"
        )
    if maximum_tle_age_hours is not None:
        config_summary = (
            f"{config_summary}, maximum_tle_age_hours={maximum_tle_age_hours}"
        )
    if required_catalog_version is not None:
        config_summary = f"{config_summary}, catalog_version={required_catalog_version}"
    if required_catalog_sha256 is not None:
        config_summary = f"{config_summary}, catalog_sha256={required_catalog_sha256}"

    window_failures: list[str] = []
    if min_snapshot_coverage_fraction is not None:
        # Snapshot-coverage fraction, max snapshot gap, TLE-hash diversity and
        # longest identical-input run are computed and reported in
        # window_quality for a transparent limitations statement, but are NOT
        # hard publication gates. Two independent, pre-analysis findings drove
        # this (both before any v3 model was evaluated):
        #   (1) Feed cadence (2026-07-27): CelesTrak refreshes these 75 LEO
        #       objects only ~once per UTC day, so the v1-pilot >=30%/<=6-bin
        #       thresholds flagged normal upstream cadence as "stale".
        #   (2) CI coverage (2026-07-31): GitHub Actions' best-effort scheduler
        #       dropped a ~14h window (7 consecutive 2h slots) that no in-repo
        #       cron redundancy can prevent and no forward collection can
        #       backfill -- a CI-infrastructure artifact, not a defect in the
        #       collected geometry.
        # This geometry/kinematics-learning comparison is not a time-series
        # forecast, so a temporal hole does not bias the learned risk mapping.
        # Publication eligibility instead rests on the statistical class /
        # positive-row / positive-pair / positive-snapshot support gates
        # (enforced separately and UNCHANGED), plus TLE-age and catalogue
        # binding below -- the genuine data-sufficiency requirements. See
        # docs/EXPERIMENT_10D_V3.md ("Coverage / gap / feed-cadence gates").
        _ = (
            float(window_quality["snapshot_coverage_fraction"]),
            float(window_quality["max_snapshot_gap_hours"]),
            min_snapshot_coverage_fraction,
            max_snapshot_gap_hours,
            min_tle_hash_diversity_fraction,
            max_identical_tle_hash_run_bins,
        )
    if maximum_tle_age_hours is not None:
        observed_tle_age = float(window_quality["maximum_observed_tle_age_hours"])
        if observed_tle_age > maximum_tle_age_hours:
            window_failures.append(
                f"maximum_observed_tle_age_hours={observed_tle_age:.6f} > "
                f"allowed={maximum_tle_age_hours:.6f}"
            )
    if window_failures:
        row = _not_enough_data_row(
            "not_enough_data",
            "Publication quality gate failed: " + "; ".join(window_failures),
        )
        row.update(window_quality)
        report = pd.DataFrame([row])[REPORT_COLUMNS]
        write_csv_text_with_provenance(
            report_path, report.to_csv(index=False), source, config_summary
        )
        return report

    if df.empty or df["risk_label"].nunique() < 2 or len(df) < 6:
        row = _not_enough_data_row(
            "not_enough_data", "Need at least 6 in-window rows and two risk_label classes."
        )
        row.update(window_quality)
        report = pd.DataFrame([row])[REPORT_COLUMNS]
        write_csv_text_with_provenance(report_path, report.to_csv(index=False), source, config_summary)
        return report

    try:
        split = _split_data(
            df,
            time_column,
            train_time_fraction=train_time_fraction,
            test_pair_fraction=test_pair_fraction,
            pair_seed=pair_seed,
        )
    except InsufficientGroupedSplitError as exc:
        row = _not_enough_data_row("not_enough_data", str(exc))
        row.update(window_quality)
        report = pd.DataFrame([row])[REPORT_COLUMNS]
        write_csv_text_with_provenance(report_path, report.to_csv(index=False), source, config_summary)
        return report
    support = _split_positive_support(split, time_column)
    observation_span_days = _observation_span_days(df, time_column)
    support["observation_span_days"] = observation_span_days
    support.update(window_quality)
    train_classes = split.train["risk_label"].nunique()
    test_classes = split.test["risk_label"].nunique()
    if train_classes < 2 or test_classes < 2:
        note = (
            "Grouped train and test partitions must each contain both risk_label classes; "
            f"got train_classes={train_classes}, test_classes={test_classes}."
        )
        row = _not_enough_data_row(
            "not_enough_data", note, len(split.train), len(split.test), split.split_name
        )
        row.update(
            train_pairs=split.metadata.get("train_pairs"),
            test_pairs=split.metadata.get("test_pairs"),
            excluded_rows=split.metadata.get("excluded_rows"),
            cutoff_utc=split.metadata.get("cutoff_utc"),
            **support,
        )
        report = pd.DataFrame([row])[REPORT_COLUMNS]
        write_csv_text_with_provenance(report_path, report.to_csv(index=False), source, config_summary)
        return report
    support_failures = [
        f"{key}={support[key]} < required={minimum}"
        for key, minimum in required_support.items()
        if support[key] < minimum
    ]
    if (
        minimum_observation_span_days is not None
        and observation_span_days < minimum_observation_span_days
    ):
        support_failures.append(
            f"observation_span_days={observation_span_days:.6f} < "
            f"required={minimum_observation_span_days:.6f}"
        )
    if support_failures:
        note = "Publication quality gate failed: " + "; ".join(support_failures)
        row = _not_enough_data_row(
            "not_enough_data", note, len(split.train), len(split.test), split.split_name
        )
        row.update(
            train_pairs=split.metadata.get("train_pairs"),
            test_pairs=split.metadata.get("test_pairs"),
            excluded_rows=split.metadata.get("excluded_rows"),
            cutoff_utc=split.metadata.get("cutoff_utc"),
            **support,
        )
        report = pd.DataFrame([row])[REPORT_COLUMNS]
        write_csv_text_with_provenance(report_path, report.to_csv(index=False), source, config_summary)
        return report
    rows = _evaluate_split(
        split,
        feature_columns=selected_features,
        time_column=time_column,
        observation_span_days=observation_span_days,
        dataset_quality=window_quality,
    )

    report = pd.DataFrame(rows)[REPORT_COLUMNS]
    write_csv_text_with_provenance(report_path, report.to_csv(index=False), source, config_summary)
    return report


def time_series_cv_report(
    dataset_path: Path,
    report_path: Path,
    n_splits: int = 5,
    time_column: str = "snapshot_utc",
    *,
    source: str = "",
    config_summary: str = "",
    test_pair_fraction: float = 0.25,
    pair_seed: str = PAIR_SPLIT_SEED,
    feature_columns: Sequence[str] | None = None,
    required_catalog_version: str | None = None,
    required_catalog_sha256: str | None = None,
    collection_start_utc: str | None = None,
    collection_end_utc: str | None = None,
    poll_interval_hours: float | None = None,
    maximum_tle_age_hours: float | None = None,
    snapshot_records: Sequence[dict[str, object]] | None = None,
    min_tle_hash_diversity_fraction: float | None = None,
    max_identical_tle_hash_run_bins: int | None = None,
    primary_model: str | None = None,
    adaptability_min_folds: int = 5,
    minimum_support: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Pair-held-out expanding-window validation over whole timestamp blocks.

    Train folds contain only deterministic train-pair observations from past
    blocks; test folds contain only deterministic test-pair observations in
    the future block. Cross-quadrant observations are excluded, preventing
    both pair memorization and future leakage.
    """
    df = pd.read_csv(dataset_path, comment="#")
    if (
        isinstance(adaptability_min_folds, bool)
        or not isinstance(adaptability_min_folds, int)
        or adaptability_min_folds <= 0
    ):
        raise ValueError("adaptability_min_folds must be a positive integer")
    required_support = _validated_minimum_support(minimum_support)
    if (min_tle_hash_diversity_fraction is None) != (
        max_identical_tle_hash_run_bins is None
    ):
        raise ValueError("CV TLE diversity gates must be configured together")
    if min_tle_hash_diversity_fraction is not None:
        min_tle_hash_diversity_fraction = float(min_tle_hash_diversity_fraction)
        if not 0 < min_tle_hash_diversity_fraction <= 1:
            raise ValueError("min_tle_hash_diversity_fraction must be in (0, 1]")
        if (
            isinstance(max_identical_tle_hash_run_bins, bool)
            or not isinstance(max_identical_tle_hash_run_bins, Integral)
            or max_identical_tle_hash_run_bins <= 0
        ):
            raise ValueError("max_identical_tle_hash_run_bins must be a positive integer")
        if snapshot_records is None:
            raise ValueError("CV TLE diversity gates require snapshot_records")
    fold_report_path = report_path.with_name(report_path.stem + "_folds.csv")
    adaptability_path = report_path.with_name(report_path.stem + "_adaptability.csv")
    fold_report_path.unlink(missing_ok=True)
    adaptability_path.unlink(missing_ok=True)
    selected_features = _validated_feature_columns(feature_columns)
    missing = [
        feature
        for feature in selected_features + ["risk_label", "fixed_threshold_alarm", "min_distance_km"]
        if feature not in df.columns
    ]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if time_column not in df.columns:
        raise ValueError(f"time_column={time_column!r} not found in dataset")
    df, window_quality = _apply_collection_window(
        df,
        time_column,
        start_utc=collection_start_utc,
        end_utc=collection_end_utc,
        poll_interval_hours=poll_interval_hours,
        snapshot_records=snapshot_records,
    )
    if maximum_tle_age_hours is not None:
        maximum_tle_age_hours = float(maximum_tle_age_hours)
        if not np.isfinite(maximum_tle_age_hours) or maximum_tle_age_hours <= 0:
            raise ValueError("maximum_tle_age_hours must be positive and finite")
        if "max_tle_age_hours" not in df.columns:
            raise ValueError("TLE-age quality gate requires max_tle_age_hours column")
        observed_ages = pd.to_numeric(df["max_tle_age_hours"], errors="coerce")
        if (
            observed_ages.isna().any()
            or not np.isfinite(observed_ages).all()
            or float(observed_ages.max()) > maximum_tle_age_hours
        ):
            raise ValueError("Cross-validation data fails the maximum TLE-age quality gate")
        window_quality["maximum_observed_tle_age_hours"] = float(observed_ages.max())
    if required_catalog_version is not None:
        if "catalog_version" not in df.columns:
            raise ValueError("Dataset is missing required column: catalog_version")
        observed = sorted(df["catalog_version"].dropna().astype(str).unique())
        if observed != [required_catalog_version]:
            raise ValueError(
                f"Catalog cohort mismatch: required={required_catalog_version!r}, observed={observed!r}"
            )
    if required_catalog_sha256 is not None:
        if "catalog_sha256" not in df.columns:
            raise ValueError("Dataset is missing required column: catalog_sha256")
        observed = sorted(df["catalog_sha256"].dropna().astype(str).unique())
        if observed != [required_catalog_sha256]:
            raise ValueError(
                f"Catalog ID-set hash mismatch: required={required_catalog_sha256!r}, observed={observed!r}"
            )

    if not source:
        source = f"dataset={dataset_path}"
    if not config_summary:
        config_summary = f"time_column={time_column}, n_splits={n_splits}"
    config_summary = (
        f"{config_summary}, features={'|'.join(selected_features)}, "
        f"window_quality={json.dumps(window_quality, sort_keys=True, default=str)}"
    )

    if len(df) < n_splits + 1 or df["risk_label"].nunique() < 2:
        report = pd.DataFrame(
            [
                {
                    "model": "not_enough_data",
                    "metric": None,
                    "mean": None,
                    "std": None,
                    "folds": 0,
                    "note": f"Need > {n_splits} rows and two risk_label classes for {n_splits}-fold TimeSeriesSplit.",
                }
            ]
        )
        write_csv_text_with_provenance(report_path, report.to_csv(index=False), source, config_summary)
        return report

    timestamps = _validated_times(df, time_column)
    pair_ids = _canonical_pair_ids(df)
    time_blocks = np.asarray(list(pd.Index(timestamps.unique()).sort_values()), dtype=object)
    if len(time_blocks) < n_splits + 1:
        report = pd.DataFrame(
            [{"model": "not_enough_data", "metric": None, "mean": None, "std": None,
              "folds": 0, "note": f"Need > {n_splits} snapshot blocks for grouped CV."}]
        )
        write_csv_text_with_provenance(report_path, report.to_csv(index=False), source, config_summary)
        return report
    if required_catalog_version is not None or required_catalog_sha256 is not None:
        missing_identity = [
            column for column in PAIR_CATALOG_ID_COLUMNS if column not in df.columns
        ]
        if missing_identity:
            raise ValueError(
                f"Frozen CV requires catalogue pair-ID columns: {missing_identity}"
            )
        identities = df[PAIR_CATALOG_ID_COLUMNS]
        if identities.isna().any().any() or (
            identities.astype(str).apply(lambda column: column.str.strip().eq(""))
        ).any().any():
            raise ValueError("Frozen CV requires non-empty catalogue IDs for every row")

    pair_assignment = {
        pair_id: _pair_is_test(pair_id, test_pair_fraction, pair_seed)
        for pair_id in pair_ids.unique()
    }
    is_test_pair = pair_ids.map(pair_assignment).astype(bool)
    splitter = TimeSeriesSplit(n_splits=n_splits)

    fold_rows: list[dict] = []
    valid_partitions = 0
    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(time_blocks)):
        train_times = set(time_blocks[train_idx])
        test_times = set(time_blocks[test_idx])
        train_mask = (~is_test_pair) & timestamps.isin(train_times)
        test_mask = is_test_pair & timestamps.isin(test_times)
        if not train_mask.any() or not test_mask.any():
            continue
        split = SplitResult(
            train=df.loc[train_mask].copy(),
            test=df.loc[test_mask].copy(),
            excluded=df.loc[~(train_mask | test_mask)].copy(),
            split_name=f"pair_grouped_cv_fold_{fold_index}",
            metadata={"train_time_blocks": len(train_times), "test_time_blocks": len(test_times)},
        )
        support = _split_positive_support(split, time_column)
        if any(support[key] < minimum for key, minimum in required_support.items()):
            continue
        fold_tle_quality = None
        if min_tle_hash_diversity_fraction is not None:
            # Feed-cadence metrics are reported per fold but no longer skip a
            # fold: they reflect CelesTrak's upstream ~daily refresh cadence for
            # these objects, not collection quality. See docs/EXPERIMENT_10D_V3.md.
            train_tle_quality = _partition_tle_hash_quality(split.train, snapshot_records)
            test_tle_quality = _partition_tle_hash_quality(split.test, snapshot_records)
            fold_tle_quality = {"train": train_tle_quality, "test": test_tle_quality}
        valid_partitions += 1
        evaluated_rows = _evaluate_split(
            split,
            feature_columns=selected_features,
            time_column=time_column,
        )
        if fold_tle_quality is not None:
            for row in evaluated_rows:
                row["train_tle_hash_diversity_fraction"] = fold_tle_quality["train"][
                    "tle_input_hash_diversity_fraction"
                ]
                row["test_tle_hash_diversity_fraction"] = fold_tle_quality["test"][
                    "tle_input_hash_diversity_fraction"
                ]
                row["train_max_identical_tle_hash_run_bins"] = fold_tle_quality["train"][
                    "max_identical_tle_hash_run_bins"
                ]
                row["test_max_identical_tle_hash_run_bins"] = fold_tle_quality["test"][
                    "max_identical_tle_hash_run_bins"
                ]
        fold_rows.extend(evaluated_rows)

    if not fold_rows:
        report = pd.DataFrame(
            [{"model": "not_enough_data", "metric": None, "mean": None, "std": None,
              "folds": 0, "note": "All pair/time CV intersections were empty."}]
        )
        write_csv_text_with_provenance(report_path, report.to_csv(index=False), source, config_summary)
        return report

    fold_df = pd.DataFrame(fold_rows)
    write_csv_text_with_provenance(
        fold_report_path,
        fold_df.to_csv(index=False),
        source=source,
        config_summary=f"{config_summary}, artifact=fold_level",
    )
    baseline = fold_df.loc[fold_df["model"].eq("fixed_threshold"), ["split", "pr_auc"]].rename(
        columns={"pr_auc": "baseline_pr_auc"}
    )
    adaptability_rows: list[dict[str, object]] = []
    for model_name, model_rows in fold_df.loc[
        ~fold_df["model"].eq("fixed_threshold")
    ].groupby("model"):
        paired = model_rows[["split", "pr_auc"]].merge(
            baseline, on="split", how="inner", validate="one_to_one"
        ).dropna()
        deltas = (paired["pr_auc"] - paired["baseline_pr_auc"]).astype(float).to_numpy()
        adaptability_rows.append(
            temporal_consistency_summary(
                str(model_name),
                deltas,
                primary_model=primary_model,
                requested_folds=n_splits,
                minimum_folds=adaptability_min_folds,
            )
        )
    adaptability_df = pd.DataFrame(adaptability_rows)
    write_csv_text_with_provenance(
        adaptability_path,
        adaptability_df.to_csv(index=False),
        source=f"fold_report={fold_report_path}",
        config_summary=(
            f"primary_model={primary_model}, inference=descriptive_temporal_consistency, "
            f"significance_tested=false, required_folds={adaptability_min_folds}, "
            f"minimum_support={required_support}"
        ),
    )
    metrics = [
        "pr_auc", "roc_auc", "precision", "recall", "f1", "accuracy",
        "false_alarm_rate",
    ]

    agg_rows: list[dict] = []
    for model_name, group in fold_df.groupby("model"):
        for metric in metrics:
            values = group[metric].dropna()
            agg_rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "mean": float(values.mean()) if len(values) else None,
                    "std": float(values.std()) if len(values) > 1 else (0.0 if len(values) == 1 else None),
                    "folds": int(len(values)),
                    "total_folds": int(n_splits),
                    "valid_partitions": int(valid_partitions),
                }
            )

    report = pd.DataFrame(agg_rows)
    write_csv_text_with_provenance(report_path, report.to_csv(index=False), source, config_summary)
    return report


def temporal_consistency_summary(
    model_name: str,
    pr_auc_deltas: Sequence[float],
    *,
    primary_model: str | None,
    requested_folds: int,
    minimum_folds: int,
) -> dict[str, object]:
    """Summarize non-inferential performance consistency across time blocks.

    Expanding time folds share pairs and overlapping training data, so they are
    not independent units for a sign-flip significance test.  This helper
    deliberately reports only the pre-registered descriptive rule: every
    requested future fold must be valid and have a positive PR-AUC delta.
    """
    deltas = np.asarray(pr_auc_deltas, dtype=float)
    if requested_folds <= 0 or minimum_folds <= 0:
        raise ValueError("requested_folds and minimum_folds must be positive")
    if not np.all(np.isfinite(deltas)):
        raise ValueError("pr_auc_deltas must be finite")
    primary = bool(primary_model is not None and model_name == primary_model)
    all_requested_folds_valid = len(deltas) == requested_folds
    supported = bool(
        primary
        and requested_folds >= minimum_folds
        and all_requested_folds_valid
        and len(deltas)
        and float(deltas.min()) > 0
    )
    return {
        "model": model_name,
        "primary_model": primary,
        "valid_paired_folds": int(len(deltas)),
        "mean_pr_auc_delta": float(deltas.mean()) if len(deltas) else None,
        "median_pr_auc_delta": float(np.median(deltas)) if len(deltas) else None,
        "worst_fold_pr_auc_delta": float(deltas.min()) if len(deltas) else None,
        "positive_delta_folds": int((deltas > 0).sum()),
        "positive_delta_fraction": float((deltas > 0).mean()) if len(deltas) else None,
        "required_folds": minimum_folds,
        "requested_folds": requested_folds,
        "all_requested_folds_valid": all_requested_folds_valid,
        "inference_method": (
            "pre-registered descriptive temporal-block consistency; "
            "no independence-based significance test"
        ),
        "statistical_significance_tested": False,
        "adaptability_supported": supported,
        "claim": (
            "descriptive temporal consistency supported: primary PR-AUC exceeded "
            "the distance baseline in every requested future fold; this is not a "
            "statistical-significance claim"
            if supported
            else (
                "exploratory supporting model; not eligible for adaptability claim"
                if not primary
                else "descriptive temporal consistency not supported"
            )
        ),
    }
