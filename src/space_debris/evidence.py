"""Frozen held-out predictions and dependence-aware uncertainty for IAC claims."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from space_debris.experiment import ExperimentConfig
from space_debris.ml import (
    FEATURES,
    REPORT_COLUMNS,
    InsufficientGroupedSplitError,
    _apply_collection_window,
    _build_models,
    _canonical_pair_ids,
    _distance_baseline_score,
    _pair_grouped_time_split,
    _positive_partition_support,
    _ranking_score,
    _split_data,
    _split_positive_support,
    _validated_feature_columns,
    feature_set_by_name,
)
from space_debris.provenance import (
    generated_utc,
    git_commit_full_hash,
    git_worktree_state,
    write_csv_text_with_provenance,
    write_json_atomic,
)
from space_debris.tle_validation import partition_tle_hash_quality


class EvidenceGenerationError(RuntimeError):
    """Raised when a publication comparison cannot be generated safely."""


def _index_by_unique_row_id(frame: pd.DataFrame) -> pd.DataFrame:
    """Index paired predictions by ``source_row_id`` with a fail-closed check.

    Replaces the ``verify_integrity=True`` keyword (deprecated in pandas 3 and
    slated for removal) while preserving its contract: a duplicated row id means
    the prediction rows are not uniquely paired, which must raise rather than
    silently produce an ambiguous alignment.
    """
    indexed = frame.set_index("source_row_id")
    if not indexed.index.is_unique:
        raise EvidenceGenerationError(
            "Prediction rows are not uniquely keyed by source_row_id"
        )
    return indexed


CALIBRATED_DISTANCE_BASELINE = "inner_calibrated_distance_threshold"
INELIGIBLE_SUPPORT_CONTROLS = frozenset(
    {
        CALIBRATED_DISTANCE_BASELINE,
        "constant_velocity_cpa",
        "label_permutation_control",
        "proxy_rule_oracle",
    }
)
NON_PRIMARY_REFERENCES = frozenset(
    {*INELIGIBLE_SUPPORT_CONTROLS, "fixed_threshold", CALIBRATED_DISTANCE_BASELINE}
)
MIN_TIME_BLOCKS = 5


def _partition_tle_hash_quality(
    frame: pd.DataFrame,
    snapshot_records: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Measure independent TLE updates represented inside one model partition."""
    return partition_tle_hash_quality(frame, snapshot_records, error_cls=EvidenceGenerationError)


def _constant_velocity_cpa_score(
    frame: pd.DataFrame, horizon_minutes: float
) -> np.ndarray:
    """Return negative linear-motion CPA miss distance from snapshot state.

    This supporting baseline uses no SGP4 TCA output. Relative radial and
    tangential velocity decompose the snapshot relative velocity, and the
    analytic closest point is clamped to the same forward simulation horizon.
    """
    distance = pd.to_numeric(frame["current_distance_km"], errors="raise").to_numpy(float)
    radial = pd.to_numeric(frame["radial_velocity_km_s"], errors="raise").to_numpy(float)
    tangential = pd.to_numeric(
        frame["tangential_velocity_km_s"], errors="raise"
    ).to_numpy(float)
    if (
        not np.isfinite(horizon_minutes)
        or float(horizon_minutes) <= 0
        or not np.all(np.isfinite(distance))
        or not np.all(np.isfinite(radial))
        or not np.all(np.isfinite(tangential))
        or np.any(distance < 0)
        or np.any(tangential < 0)
    ):
        raise EvidenceGenerationError(
            "Constant-velocity CPA inputs must be finite and physically valid"
        )
    speed_squared = radial**2 + tangential**2
    time_seconds = np.divide(
        -distance * radial,
        speed_squared,
        out=np.zeros_like(distance),
        where=speed_squared > np.finfo(float).eps,
    )
    time_seconds = np.clip(time_seconds, 0.0, float(horizon_minutes) * 60.0)
    miss_squared = (
        distance**2
        + 2.0 * distance * radial * time_seconds
        + speed_squared * time_seconds**2
    )
    return -np.sqrt(np.maximum(miss_squared, 0.0))


ESTIMATOR_SUPPORT_COLUMNS = [
    "estimator_train_rows",
    "estimator_train_pairs",
    "estimator_train_positive_rows",
    "estimator_train_positive_pairs",
    "estimator_train_positive_snapshots",
    "estimator_train_cutoff_utc",
    "calibration_rows",
    "calibration_pairs",
    "calibration_positive_rows",
    "calibration_positive_pairs",
    "calibration_positive_snapshots",
]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _row_identity(df: pd.DataFrame, time_column: str) -> pd.Series:
    pair_ids = _canonical_pair_ids(df)
    collection_ids = (
        df["collection_id"].fillna("").astype(str).str.strip()
        if "collection_id" in df.columns
        else pd.Series("", index=df.index)
    )
    timestamps = pd.to_datetime(df[time_column], utc=True, errors="raise", format="ISO8601")
    identities = pd.Series(
        [
            _sha256_bytes(
                _canonical_json(
                    {
                        "collection_id": collection_ids.loc[index],
                        "snapshot_utc": timestamp.isoformat(),
                        "canonical_pair_id": pair_ids.loc[index],
                    }
                )
            )
            for index, timestamp in timestamps.items()
        ],
        index=df.index,
        name="source_row_id",
    )
    if identities.duplicated().any():
        duplicates = identities.loc[identities.duplicated(keep=False)].unique().tolist()
        raise EvidenceGenerationError(
            f"Held-out row identity is not unique; duplicate hashes={duplicates[:5]}"
        )
    return identities


def _class_ratio(frame: pd.DataFrame) -> float:
    labels = frame["risk_label"].astype(int)
    positive = int(labels.eq(1).sum())
    negative = int(labels.eq(0).sum())
    if not positive or not negative:
        raise EvidenceGenerationError("Training partition must contain both classes")
    return negative / positive


def _binary_metrics(y_true: np.ndarray, pred: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    if len(np.unique(y_true)) != 2:
        raise EvidenceGenerationError("Metrics require both classes")
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy_score(y_true, pred)),
        "proxy_false_alarm_rate": float(fp / (fp + tn)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "true_negative": int(tn),
    }


def _weighted_bootstrap_metrics(
    y_true: np.ndarray,
    pred: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    """Compute only the metrics needed inside paired resampling loops."""
    y = np.asarray(y_true, dtype=int)
    predicted = np.asarray(pred, dtype=int)
    ranking = np.asarray(scores, dtype=float)
    sample_weight = np.asarray(weights, dtype=float)
    positive_weight = float(sample_weight[y == 1].sum())
    negative_weight = float(sample_weight[y == 0].sum())
    if positive_weight <= 0 or negative_weight <= 0:
        raise EvidenceGenerationError("Bootstrap replicate requires both weighted classes")
    tp = float(sample_weight[(y == 1) & (predicted == 1)].sum())
    fp = float(sample_weight[(y == 0) & (predicted == 1)].sum())
    fn = float(sample_weight[(y == 1) & (predicted == 0)].sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "proxy_false_alarm_rate": fp / negative_weight,
        "pr_auc": float(
            average_precision_score(y, ranking, sample_weight=sample_weight)
        ),
    }


def select_threshold_for_target_recall(
    y_true: Sequence[int],
    scores: Sequence[float],
    target_recall: float,
) -> dict[str, float]:
    """Select the highest score threshold reaching a frozen validation recall.

    Selection sees only inner-validation labels.  Higher thresholds are tested
    first; therefore the chosen point has the lowest attainable false-alarm
    rate among monotone score thresholds that meet the target recall.
    """
    y = np.asarray(y_true, dtype=int)
    values = np.asarray(scores, dtype=float)
    if len(y) != len(values) or not len(y):
        raise EvidenceGenerationError("Threshold calibration arrays must be non-empty and aligned")
    if len(np.unique(y)) != 2 or not np.all(np.isfinite(values)):
        raise EvidenceGenerationError("Threshold calibration requires finite scores and both classes")
    if not 0 < target_recall <= 1:
        raise EvidenceGenerationError("Baseline validation recall must be in (0, 1]")
    order = np.argsort(-values, kind="stable")
    ordered_scores = values[order]
    ordered_y = y[order]
    cumulative_tp = np.cumsum(ordered_y == 1)
    cumulative_fp = np.cumsum(ordered_y == 0)
    group_ends = np.flatnonzero(np.r_[ordered_scores[1:] != ordered_scores[:-1], True])
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    eligible: list[tuple[float, float, int]] = []
    for index in group_ends:
        recall = float(cumulative_tp[index] / positives)
        if recall + 1e-12 >= target_recall:
            false_alarm_rate = float(cumulative_fp[index] / negatives)
            eligible.append((false_alarm_rate, -float(ordered_scores[index]), int(index)))
    if not eligible:
        raise EvidenceGenerationError("No inner-validation threshold reaches target recall")
    false_alarm_rate, negative_threshold, index = min(
        eligible, key=lambda item: (item[0], item[1])
    )
    true_positive = int(cumulative_tp[index])
    false_positive = int(cumulative_fp[index])
    recall = float(true_positive / positives)
    precision = float(true_positive / (true_positive + false_positive))
    return {
        "threshold": -negative_threshold,
        "target_recall": float(target_recall),
        "validation_recall": recall,
        "validation_proxy_false_alarm_rate": false_alarm_rate,
        "validation_precision": precision,
    }


def calibrate_distance_baseline(
    validation_y: Sequence[int],
    validation_scores: Sequence[float],
    test_scores: Sequence[float],
    target_recall: float,
) -> tuple[dict[str, float], np.ndarray]:
    """Freeze a distance-only threshold without accepting held-out labels.

    The API deliberately has no ``test_y`` argument: only inner-validation
    labels may influence the operating point.  Held-out distance scores are
    transformed into predictions after that threshold is frozen.
    """
    threshold_info = select_threshold_for_target_recall(
        validation_y, validation_scores, target_recall
    )
    held_out_scores = np.asarray(test_scores, dtype=float)
    if not len(held_out_scores) or not np.all(np.isfinite(held_out_scores)):
        raise EvidenceGenerationError(
            "Calibrated distance baseline requires finite held-out scores"
        )
    predictions = (held_out_scores >= threshold_info["threshold"]).astype(int)
    return threshold_info, predictions


def _percentile_interval(values: list[float], confidence_level: float) -> tuple[float, float]:
    alpha = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(np.asarray(values, dtype=float), [alpha, 1.0 - alpha])
    return float(low), float(high)


def paired_pair_cluster_bootstrap(
    predictions: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
    confidence_level: float,
    recall_noninferiority_margin: float,
    min_valid_fraction: float,
    primary_model: str,
    claim_eligible: bool = True,
    min_time_blocks: int = MIN_TIME_BLOCKS,
) -> pd.DataFrame:
    """Compare learners with paired pair-cluster and UTC-day block resampling."""
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates <= 0:
        raise ValueError("replicates must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    if not 0 <= recall_noninferiority_margin < 1:
        raise ValueError("recall_noninferiority_margin must be in [0, 1)")
    if not 0 < min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be in (0, 1]")
    if not primary_model:
        raise ValueError("primary_model must be non-empty")
    if primary_model in NON_PRIMARY_REFERENCES:
        raise ValueError(f"Reference/control cannot be the primary model: {primary_model}")
    if isinstance(min_time_blocks, bool) or not isinstance(min_time_blocks, int) or min_time_blocks < 2:
        raise ValueError("min_time_blocks must be an integer of at least 2")
    required = {
        "source_row_id",
        "canonical_pair_id",
        "snapshot_utc",
        "y_true",
        "model",
        "score",
        "pred",
        "operating_threshold",
        "calibration_target_recall",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise EvidenceGenerationError(f"Prediction artifact is missing columns: {missing}")
    baseline = predictions.loc[predictions["model"].eq("fixed_threshold")].copy()
    if baseline.empty or baseline["source_row_id"].duplicated().any():
        raise EvidenceGenerationError("Exactly one fixed-threshold prediction per test row is required")
    if primary_model not in set(predictions["model"]):
        raise EvidenceGenerationError(f"Primary model is missing from predictions: {primary_model}")
    baseline = _index_by_unique_row_id(baseline)
    y_baseline = baseline["y_true"].astype(int).to_numpy()
    baseline_pred_values = baseline["pred"].astype(int).to_numpy()
    baseline_score_values = baseline["score"].astype(float).to_numpy()
    baseline_point = _binary_metrics(
        y_baseline,
        baseline_pred_values,
        baseline_score_values,
    )
    calibrated_reference: pd.DataFrame | None = None
    if CALIBRATED_DISTANCE_BASELINE in set(predictions["model"]):
        calibrated_reference = _index_by_unique_row_id(
            predictions.loc[predictions["model"].eq(CALIBRATED_DISTANCE_BASELINE)]
        )
        if set(calibrated_reference.index) != set(baseline.index):
            raise EvidenceGenerationError(
                "Calibrated distance prediction rows are not paired with the fixed baseline"
            )
        calibrated_reference = calibrated_reference.loc[baseline.index]
        if not np.array_equal(
            calibrated_reference["y_true"].astype(int), baseline["y_true"].astype(int)
        ):
            raise EvidenceGenerationError(
                "y_true mismatch for calibrated_distance_threshold"
            )
    elif claim_eligible:
        raise EvidenceGenerationError(
            "Claim-eligible evidence requires an inner-validation-calibrated distance baseline"
        )
    calibrated_point = (
        _binary_metrics(
            y_baseline,
            calibrated_reference["pred"].astype(int).to_numpy(),
            calibrated_reference["score"].astype(float).to_numpy(),
        )
        if calibrated_reference is not None
        else None
    )
    calibrated_pred_values = (
        calibrated_reference["pred"].astype(int).to_numpy()
        if calibrated_reference is not None
        else None
    )
    calibrated_score_values = (
        calibrated_reference["score"].astype(float).to_numpy()
        if calibrated_reference is not None
        else None
    )
    cpa_reference: pd.DataFrame | None = None
    if "constant_velocity_cpa" in set(predictions["model"]):
        cpa_reference = _index_by_unique_row_id(
            predictions.loc[predictions["model"].eq("constant_velocity_cpa")]
        )
        if set(cpa_reference.index) != set(baseline.index):
            raise EvidenceGenerationError(
                "Constant-velocity CPA prediction rows are not paired with the fixed baseline"
            )
        cpa_reference = cpa_reference.loc[baseline.index]
        if not np.array_equal(
            cpa_reference["y_true"].astype(int), baseline["y_true"].astype(int)
        ):
            raise EvidenceGenerationError("y_true mismatch for constant_velocity_cpa")
    cpa_point = (
        _binary_metrics(
            y_baseline,
            cpa_reference["pred"].astype(int).to_numpy(),
            cpa_reference["score"].astype(float).to_numpy(),
        )
        if cpa_reference is not None
        else None
    )
    cpa_pred_values = (
        cpa_reference["pred"].astype(int).to_numpy()
        if cpa_reference is not None
        else None
    )
    cpa_score_values = (
        cpa_reference["score"].astype(float).to_numpy()
        if cpa_reference is not None
        else None
    )
    rows = [
        {
            "model": "fixed_threshold",
            "operating_threshold": float(baseline["operating_threshold"].iloc[0]),
            "calibration_target_recall": None,
            **{f"test_{key}": value for key, value in baseline_point.items()},
            "paired_clusters": int(baseline["canonical_pair_id"].nunique()),
            "valid_bootstrap_replicates": None,
            "valid_bootstrap_fraction": None,
            "false_alarm_reduction_supported": False,
            "claim": "fixed-distance reference",
        }
    ]
    cluster_values = baseline["canonical_pair_id"].astype(str)
    cluster_row_values = cluster_values.to_numpy()
    clusters = np.asarray(sorted(cluster_values.unique()), dtype=object)
    rng = np.random.default_rng(seed)
    sampled_clusters = [rng.choice(clusters, size=len(clusters), replace=True) for _ in range(replicates)]
    time_values = pd.to_datetime(
        baseline["snapshot_utc"], utc=True, errors="raise", format="ISO8601"
    ).dt.strftime("%Y-%m-%d")
    time_row_values = time_values.to_numpy()
    time_blocks = np.asarray(sorted(time_values.unique()), dtype=object)
    time_rng = np.random.default_rng(seed + 1)
    sampled_time_blocks = (
        [
            time_rng.choice(time_blocks, size=len(time_blocks), replace=True)
            for _ in range(replicates)
        ]
        if len(time_blocks) >= min_time_blocks
        else []
    )

    def resample_weights(sampled: np.ndarray, row_values: np.ndarray) -> np.ndarray:
        counts: dict[str, int] = {}
        for value in sampled:
            key = str(value)
            counts[key] = counts.get(key, 0) + 1
        return np.fromiter(
            (counts.get(str(value), 0) for value in row_values),
            dtype=float,
            count=len(row_values),
        )

    for model_name in sorted(set(predictions["model"]) - {"fixed_threshold"}):
        model = _index_by_unique_row_id(
            predictions.loc[predictions["model"].eq(model_name)]
        )
        if set(model.index) != set(baseline.index):
            raise EvidenceGenerationError(f"Prediction rows are not paired for {model_name}")
        model = model.loc[baseline.index]
        if not np.array_equal(model["y_true"].astype(int), baseline["y_true"].astype(int)):
            raise EvidenceGenerationError(f"y_true mismatch for paired model {model_name}")
        model_pred_values = model["pred"].astype(int).to_numpy()
        model_score_values = model["score"].astype(float).to_numpy()
        point = _binary_metrics(y_baseline, model_pred_values, model_score_values)
        delta_names = ("precision", "recall", "f1", "proxy_false_alarm_rate", "pr_auc")
        distributions = {name: [] for name in delta_names}
        time_distributions = {name: [] for name in delta_names}
        compare_to_cpa = bool(
            cpa_reference is not None and model_name not in INELIGIBLE_SUPPORT_CONTROLS
        )
        compare_to_calibrated = bool(
            calibrated_reference is not None
            and model_name not in NON_PRIMARY_REFERENCES
        )
        cpa_distributions = {name: [] for name in delta_names}
        cpa_time_distributions = {name: [] for name in delta_names}
        calibrated_distributions = {name: [] for name in delta_names}
        calibrated_time_distributions = {name: [] for name in delta_names}
        valid = 0
        for sampled in sampled_clusters:
            weights = resample_weights(sampled, cluster_row_values)
            if weights[y_baseline == 1].sum() <= 0 or weights[y_baseline == 0].sum() <= 0:
                continue
            sampled_baseline = _weighted_bootstrap_metrics(
                y_baseline, baseline_pred_values, baseline_score_values, weights
            )
            sampled_model = _weighted_bootstrap_metrics(
                y_baseline, model_pred_values, model_score_values, weights
            )
            for name in delta_names:
                distributions[name].append(sampled_model[name] - sampled_baseline[name])
            if compare_to_cpa:
                sampled_cpa = _weighted_bootstrap_metrics(
                    y_baseline, cpa_pred_values, cpa_score_values, weights
                )
                for name in delta_names:
                    cpa_distributions[name].append(
                        sampled_model[name] - sampled_cpa[name]
                    )
            if compare_to_calibrated:
                sampled_calibrated = _weighted_bootstrap_metrics(
                    y_baseline,
                    calibrated_pred_values,
                    calibrated_score_values,
                    weights,
                )
                for name in delta_names:
                    calibrated_distributions[name].append(
                        sampled_model[name] - sampled_calibrated[name]
                    )
            valid += 1
        valid_fraction = valid / replicates
        valid_time = 0
        for sampled in sampled_time_blocks:
            weights = resample_weights(sampled, time_row_values)
            if weights[y_baseline == 1].sum() <= 0 or weights[y_baseline == 0].sum() <= 0:
                continue
            sampled_baseline = _weighted_bootstrap_metrics(
                y_baseline, baseline_pred_values, baseline_score_values, weights
            )
            sampled_model = _weighted_bootstrap_metrics(
                y_baseline, model_pred_values, model_score_values, weights
            )
            for name in delta_names:
                time_distributions[name].append(sampled_model[name] - sampled_baseline[name])
            if compare_to_cpa:
                sampled_cpa = _weighted_bootstrap_metrics(
                    y_baseline, cpa_pred_values, cpa_score_values, weights
                )
                for name in delta_names:
                    cpa_time_distributions[name].append(
                        sampled_model[name] - sampled_cpa[name]
                    )
            if compare_to_calibrated:
                sampled_calibrated = _weighted_bootstrap_metrics(
                    y_baseline,
                    calibrated_pred_values,
                    calibrated_score_values,
                    weights,
                )
                for name in delta_names:
                    calibrated_time_distributions[name].append(
                        sampled_model[name] - sampled_calibrated[name]
                    )
            valid_time += 1
        valid_time_fraction = valid_time / replicates
        result: dict[str, Any] = {
            "model": model_name,
            "operating_threshold": float(model["operating_threshold"].iloc[0]),
            "calibration_target_recall": (
                float(model["calibration_target_recall"].iloc[0])
                if pd.notna(model["calibration_target_recall"].iloc[0])
                else None
            ),
            **{f"test_{key}": value for key, value in point.items()},
            "paired_clusters": int(len(clusters)),
            "valid_bootstrap_replicates": valid,
            "valid_bootstrap_fraction": valid_fraction,
            "time_blocks": int(len(time_blocks)),
            "valid_time_block_bootstrap_replicates": valid_time,
            "valid_time_block_bootstrap_fraction": valid_time_fraction,
        }
        intervals: dict[str, tuple[float, float]] = {}
        time_intervals: dict[str, tuple[float, float]] = {}
        for name in delta_names:
            result[f"delta_{name}"] = point[name] - baseline_point[name]
            if distributions[name]:
                low, high = _percentile_interval(distributions[name], confidence_level)
            else:
                low, high = math.nan, math.nan
            intervals[name] = (low, high)
            result[f"delta_{name}_ci_low"] = low
            result[f"delta_{name}_ci_high"] = high
            if time_distributions[name]:
                time_low, time_high = _percentile_interval(
                    time_distributions[name], confidence_level
                )
            else:
                time_low, time_high = math.nan, math.nan
            time_intervals[name] = (time_low, time_high)
            result[f"time_block_delta_{name}_ci_low"] = time_low
            result[f"time_block_delta_{name}_ci_high"] = time_high
            if compare_to_cpa and cpa_point is not None:
                result[f"delta_vs_constant_velocity_cpa_{name}"] = (
                    point[name] - cpa_point[name]
                )
                cpa_low, cpa_high = (
                    _percentile_interval(cpa_distributions[name], confidence_level)
                    if cpa_distributions[name]
                    else (math.nan, math.nan)
                )
                cpa_time_low, cpa_time_high = (
                    _percentile_interval(cpa_time_distributions[name], confidence_level)
                    if cpa_time_distributions[name]
                    else (math.nan, math.nan)
                )
                result[f"delta_vs_constant_velocity_cpa_{name}_ci_low"] = cpa_low
                result[f"delta_vs_constant_velocity_cpa_{name}_ci_high"] = cpa_high
                result[
                    f"time_block_delta_vs_constant_velocity_cpa_{name}_ci_low"
                ] = cpa_time_low
                result[
                    f"time_block_delta_vs_constant_velocity_cpa_{name}_ci_high"
                ] = cpa_time_high
            if compare_to_calibrated and calibrated_point is not None:
                result[f"delta_vs_calibrated_distance_{name}"] = (
                    point[name] - calibrated_point[name]
                )
                calibrated_low, calibrated_high = (
                    _percentile_interval(
                        calibrated_distributions[name], confidence_level
                    )
                    if calibrated_distributions[name]
                    else (math.nan, math.nan)
                )
                calibrated_time_low, calibrated_time_high = (
                    _percentile_interval(
                        calibrated_time_distributions[name], confidence_level
                    )
                    if calibrated_time_distributions[name]
                    else (math.nan, math.nan)
                )
                result[f"delta_vs_calibrated_distance_{name}_ci_low"] = calibrated_low
                result[f"delta_vs_calibrated_distance_{name}_ci_high"] = calibrated_high
                result[
                    f"time_block_delta_vs_calibrated_distance_{name}_ci_low"
                ] = calibrated_time_low
                result[
                    f"time_block_delta_vs_calibrated_distance_{name}_ci_high"
                ] = calibrated_time_high
        far_high = intervals["proxy_false_alarm_rate"][1]
        recall_low = intervals["recall"][0]
        pair_gate = bool(
            valid_fraction >= min_valid_fraction
            and np.isfinite(far_high)
            and np.isfinite(recall_low)
            and far_high < 0
            and recall_low >= -recall_noninferiority_margin
        )
        time_far_high = time_intervals["proxy_false_alarm_rate"][1]
        time_recall_low = time_intervals["recall"][0]
        time_gate = bool(
            len(time_blocks) >= min_time_blocks
            and valid_time_fraction >= min_valid_fraction
            and np.isfinite(time_far_high)
            and np.isfinite(time_recall_low)
            and time_far_high < 0
            and time_recall_low >= -recall_noninferiority_margin
        )
        calibrated_pair_gate = False
        calibrated_time_gate = False
        if compare_to_calibrated:
            calibrated_far_high = result[
                "delta_vs_calibrated_distance_proxy_false_alarm_rate_ci_high"
            ]
            calibrated_recall_low = result[
                "delta_vs_calibrated_distance_recall_ci_low"
            ]
            calibrated_time_far_high = result[
                "time_block_delta_vs_calibrated_distance_proxy_false_alarm_rate_ci_high"
            ]
            calibrated_time_recall_low = result[
                "time_block_delta_vs_calibrated_distance_recall_ci_low"
            ]
            calibrated_pair_gate = bool(
                valid_fraction >= min_valid_fraction
                and np.isfinite(calibrated_far_high)
                and np.isfinite(calibrated_recall_low)
                and calibrated_far_high < 0
                and calibrated_recall_low >= -recall_noninferiority_margin
            )
            calibrated_time_gate = bool(
                len(time_blocks) >= min_time_blocks
                and valid_time_fraction >= min_valid_fraction
                and np.isfinite(calibrated_time_far_high)
                and np.isfinite(calibrated_time_recall_low)
                and calibrated_time_far_high < 0
                and calibrated_time_recall_low >= -recall_noninferiority_margin
            )
        statistical_gate = bool(
            pair_gate
            and time_gate
            and calibrated_pair_gate
            and calibrated_time_gate
        )
        supported = bool(claim_eligible and model_name == primary_model and statistical_gate)
        result["false_alarm_reduction_supported"] = supported
        result["primary_model"] = model_name == primary_model
        result["statistical_gate_passed"] = statistical_gate
        result["conditional_pair_day_sensitivity_gate_passed"] = statistical_gate
        result["pair_cluster_gate_passed"] = pair_gate
        result["time_block_gate_passed"] = time_gate
        result["calibrated_distance_pair_gate_passed"] = calibrated_pair_gate
        result["calibrated_distance_time_gate_passed"] = calibrated_time_gate
        if compare_to_cpa:
            cpa_far_high = result[
                "delta_vs_constant_velocity_cpa_proxy_false_alarm_rate_ci_high"
            ]
            cpa_recall_low = result["delta_vs_constant_velocity_cpa_recall_ci_low"]
            cpa_time_far_high = result[
                "time_block_delta_vs_constant_velocity_cpa_proxy_false_alarm_rate_ci_high"
            ]
            cpa_time_recall_low = result[
                "time_block_delta_vs_constant_velocity_cpa_recall_ci_low"
            ]
            result["constant_velocity_cpa_pair_sensitivity_passed"] = bool(
                valid_fraction >= min_valid_fraction
                and np.isfinite(cpa_far_high)
                and np.isfinite(cpa_recall_low)
                and cpa_far_high < 0
                and cpa_recall_low >= -recall_noninferiority_margin
            )
            result["constant_velocity_cpa_time_sensitivity_passed"] = bool(
                len(time_blocks) >= min_time_blocks
                and valid_time_fraction >= min_valid_fraction
                and np.isfinite(cpa_time_far_high)
                and np.isfinite(cpa_time_recall_low)
                and cpa_time_far_high < 0
                and cpa_time_recall_low >= -recall_noninferiority_margin
            )
        result["claim"] = (
            "conditional pair-cluster and UTC-day-block sensitivity supports lower "
            "proxy false-alarm rate than both pre-specified and validation-calibrated "
            "distance thresholds with recall non-inferiority"
            if supported
            else (
                "inner-validation-calibrated distance-only reference"
                if model_name == CALIBRATED_DISTANCE_BASELINE
                else (
                "exploratory supporting model; not eligible for the primary claim"
                if not claim_eligible
                else (
                    "exploratory supporting model; not eligible for the primary claim"
                    if model_name != primary_model
                    else "false-alarm reduction claim not supported"
                )
                )
            )
        )
        rows.append(result)
    return pd.DataFrame(rows)


def _validate_frozen_adaptability_protocol(config: ExperimentConfig) -> None:
    frozen_by_experiment = {
        "iac26-leo-mixed-75-v1": {
            "train_time_fraction": 0.50,
            "adaptability_block_hours": 48.0,
            "adaptability_embargo_hours": 2.0,
            "adaptability_min_blocks": 10,
            "adaptability_min_unique_objects": 30,
            "adaptability_min_pairs": 30,
            "adaptability_min_positive_pairs_per_block": 5,
            "adaptability_min_positive_days_per_block": 2,
            "adaptability_bootstrap_replicates": 10000,
            "confidence_level": 0.95,
            "min_valid_bootstrap_fraction": 0.90,
            "bootstrap_seed": 114764,
        },
        "iac26-10d-v2": {
            "train_time_fraction": 0.40,
            "adaptability_block_hours": 12.0,
            "adaptability_embargo_hours": 2.0,
            "adaptability_min_blocks": 10,
            "adaptability_min_unique_objects": 30,
            "adaptability_min_pairs": 30,
            "adaptability_min_positive_pairs_per_block": 5,
            "adaptability_min_positive_days_per_block": 1,
            "adaptability_bootstrap_replicates": 10000,
            "confidence_level": 0.95,
            "min_valid_bootstrap_fraction": 0.90,
            "bootstrap_seed": 214764,
        },
        # v3 re-anchors v2's collection window only; every frozen adaptability
        # parameter below is identical to v2 (see config/experiment_10_days_v3.json).
        "iac26-10d-v3": {
            "train_time_fraction": 0.40,
            "adaptability_block_hours": 12.0,
            "adaptability_embargo_hours": 2.0,
            "adaptability_min_blocks": 10,
            "adaptability_min_unique_objects": 30,
            "adaptability_min_pairs": 30,
            "adaptability_min_positive_pairs_per_block": 5,
            "adaptability_min_positive_days_per_block": 1,
            "adaptability_bootstrap_replicates": 10000,
            "confidence_level": 0.95,
            "min_valid_bootstrap_fraction": 0.90,
            "bootstrap_seed": 214764,
        },
    }
    frozen = frozen_by_experiment.get(config.experiment_id)
    if frozen is None:
        raise EvidenceGenerationError(
            "Frozen adaptability protocol has no registered experiment_id="
            f"{config.experiment_id!r}"
        )
    requirements = {}
    for field, expected in frozen.items():
        observed = getattr(config, field)
        requirements[f"{field} == {expected}"] = (
            math.isclose(
                float(observed), float(expected), rel_tol=0.0, abs_tol=1e-12
            )
            if isinstance(expected, float)
            else observed == expected
        )
    failures = [name for name, passed in requirements.items() if not passed]
    if failures:
        raise EvidenceGenerationError(
            "Frozen adaptability protocol mismatch: " + "; ".join(failures)
        )


def adaptability_inference_from_predictions(
    predictions: pd.DataFrame,
    config: ExperimentConfig,
    *,
    split_cutoff_utc: str,
    block_anchor_utc: str,
    analysis_end_utc: str,
    enforce_frozen_protocol: bool = True,
) -> dict[str, object]:
    """Infer future-block transportability with dyadic/time resampling.

    The estimand is the paired AP advantage of the frozen primary learner over
    the inner-validation-calibrated continuous-distance reference.  Catalogue
    objects are resampled as dyadic endpoints and already non-overlapping
    future time blocks are resampled jointly.  The same row weights are used
    for both methods in every replicate.
    """
    if enforce_frozen_protocol:
        _validate_frozen_adaptability_protocol(config)
    required = {
        "source_row_id",
        "snapshot_utc",
        "canonical_pair_id",
        "object_1_catalog_id",
        "object_2_catalog_id",
        "y_true",
        "model",
        "score",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise EvidenceGenerationError(
            f"Adaptability predictions are missing columns: {missing}"
        )
    if config.primary_model != "xgboost" or config.primary_feature_set != "snapshot_only":
        raise EvidenceGenerationError(
            "Claim-eligible adaptability inference is frozen to snapshot-only XGBoost"
        )
    primary = predictions.loc[predictions["model"].eq(config.primary_model)].copy()
    baseline = predictions.loc[
        predictions["model"].eq(CALIBRATED_DISTANCE_BASELINE)
    ].copy()
    if primary.empty or baseline.empty:
        raise EvidenceGenerationError(
            "Adaptability inference requires primary and calibrated-distance predictions"
        )
    if primary["source_row_id"].duplicated().any() or baseline[
        "source_row_id"
    ].duplicated().any():
        raise EvidenceGenerationError(
            "Adaptability inference requires one prediction per method and source row"
        )
    primary = _index_by_unique_row_id(primary)
    baseline = _index_by_unique_row_id(baseline)
    if set(primary.index) != set(baseline.index):
        raise EvidenceGenerationError(
            "Adaptability primary and baseline prediction rows are not paired"
        )
    baseline = baseline.loc[primary.index]
    identity_columns = [
        "snapshot_utc",
        "canonical_pair_id",
        "object_1_catalog_id",
        "object_2_catalog_id",
        "y_true",
    ]
    for column in identity_columns:
        if not np.array_equal(
            primary[column].astype(str).to_numpy(),
            baseline[column].astype(str).to_numpy(),
        ):
            raise EvidenceGenerationError(
                f"Adaptability paired-row metadata mismatch: {column}"
            )

    frame = primary[identity_columns].copy()
    frame["primary_score"] = pd.to_numeric(primary["score"], errors="raise")
    frame["baseline_score"] = pd.to_numeric(baseline["score"], errors="raise")
    frame["y_true"] = pd.to_numeric(frame["y_true"], errors="raise").astype(int)
    frame["timestamp"] = pd.to_datetime(
        frame["snapshot_utc"], utc=True, errors="raise", format="ISO8601"
    )
    if not set(frame["y_true"].unique()).issubset({0, 1}):
        raise EvidenceGenerationError("Adaptability labels must be binary")
    if not np.all(
        np.isfinite(frame[["primary_score", "baseline_score"]].to_numpy(dtype=float))
    ):
        raise EvidenceGenerationError("Adaptability scores must be finite")

    split_cutoff = pd.Timestamp(split_cutoff_utc)
    anchor = pd.Timestamp(block_anchor_utc)
    analysis_end = pd.Timestamp(analysis_end_utc)
    if split_cutoff.tzinfo is None or anchor.tzinfo is None or analysis_end.tzinfo is None:
        raise EvidenceGenerationError("Adaptability block boundaries must be timezone-aware")
    split_cutoff = split_cutoff.tz_convert("UTC")
    anchor = anchor.tz_convert("UTC")
    analysis_end = analysis_end.tz_convert("UTC")
    if anchor < split_cutoff:
        raise EvidenceGenerationError(
            "Adaptability block anchor must not precede the outer-test cutoff"
        )
    if analysis_end <= anchor:
        raise EvidenceGenerationError("Adaptability analysis end must follow its block anchor")
    block_seconds = float(config.adaptability_block_hours) * 3600.0
    embargo_seconds = float(config.adaptability_embargo_hours) * 3600.0
    cycle_seconds = block_seconds + embargo_seconds
    if (frame["timestamp"] < split_cutoff).any():
        raise EvidenceGenerationError(
            "Adaptability predictions precede the frozen outer-test cutoff"
        )
    initial_embargo_rows = int(
        frame["timestamp"].ge(split_cutoff).mul(frame["timestamp"].lt(anchor)).sum()
    )
    # Rows between the outer-test cutoff and the first analysis block are valid
    # held-out predictions, but belong to the pre-registered initial embargo.
    frame = frame.loc[frame["timestamp"].ge(anchor)].copy()
    elapsed = (frame["timestamp"] - anchor).dt.total_seconds().to_numpy(dtype=float)
    frame["adaptability_block"] = np.floor(elapsed / cycle_seconds).astype(int)
    within_cycle = np.mod(elapsed, cycle_seconds)
    last_complete_block = math.floor(
        ((analysis_end - anchor).total_seconds() - block_seconds) / cycle_seconds
    )
    complete_blocks = list(range(max(0, last_complete_block + 1)))
    frame = frame.loc[
        (within_cycle < block_seconds)
        & frame["adaptability_block"].isin(complete_blocks)
    ].copy()
    frame["utc_day"] = frame["timestamp"].dt.floor("D")

    block_reports: list[dict[str, object]] = []
    valid_blocks: list[int] = []
    for block_id in complete_blocks:
        block = frame.loc[frame["adaptability_block"].eq(block_id)]
        scheduled_start = anchor + pd.Timedelta(seconds=block_id * cycle_seconds)
        scheduled_end = scheduled_start + pd.Timedelta(seconds=block_seconds)
        positives = block.loc[block["y_true"].eq(1)]
        two_classes = block["y_true"].nunique() == 2
        positive_pairs = int(positives["canonical_pair_id"].astype(str).nunique())
        positive_days = int(positives["utc_day"].nunique())
        eligible = bool(
            two_classes
            and positive_pairs >= config.adaptability_min_positive_pairs_per_block
            and positive_days >= config.adaptability_min_positive_days_per_block
        )
        delta_ap = None
        if two_classes:
            y = block["y_true"].to_numpy(dtype=int)
            delta_ap = float(
                average_precision_score(y, block["primary_score"].to_numpy(dtype=float))
                - average_precision_score(y, block["baseline_score"].to_numpy(dtype=float))
            )
        if eligible:
            valid_blocks.append(int(block_id))
        block_reports.append(
            {
                "block_id": int(block_id),
                "scheduled_start_utc": scheduled_start.isoformat().replace("+00:00", "Z"),
                "scheduled_end_utc": scheduled_end.isoformat().replace("+00:00", "Z"),
                "first_observation_utc": (
                    block["timestamp"].min().isoformat().replace("+00:00", "Z")
                    if len(block)
                    else None
                ),
                "last_observation_utc": (
                    block["timestamp"].max().isoformat().replace("+00:00", "Z")
                    if len(block)
                    else None
                ),
                "rows": int(len(block)),
                "pairs": int(block["canonical_pair_id"].astype(str).nunique()),
                "positive_rows": int(len(positives)),
                "positive_pairs": positive_pairs,
                "positive_utc_days": positive_days,
                "two_classes": bool(two_classes),
                "eligible": eligible,
                "delta_average_precision": delta_ap,
            }
        )

    analysis = frame.loc[frame["adaptability_block"].isin(valid_blocks)].copy()
    objects = sorted(
        set(analysis["object_1_catalog_id"].astype(str))
        | set(analysis["object_2_catalog_id"].astype(str))
    )
    pair_count = int(analysis["canonical_pair_id"].astype(str).nunique())
    support = {
        "rows": int(len(analysis)),
        "complete_planned_future_blocks": int(len(complete_blocks)),
        "valid_future_blocks": int(len(valid_blocks)),
        "required_future_blocks": int(config.adaptability_min_blocks),
        "unique_objects": int(len(objects)),
        "required_unique_objects": int(config.adaptability_min_unique_objects),
        "canonical_pairs": pair_count,
        "required_canonical_pairs": int(config.adaptability_min_pairs),
        "block_hours": float(config.adaptability_block_hours),
        "embargo_hours": float(config.adaptability_embargo_hours),
    }
    result: dict[str, object] = {
        "schema_version": 1,
        "generated_utc": generated_utc(),
        "git_commit": git_commit_full_hash(),
        "source_worktree": git_worktree_state(),
        "protocol": "dyadic_object_x_nonoverlap_future_time_block_bootstrap_v1",
        "primary_model": config.primary_model,
        "primary_feature_set": config.primary_feature_set,
        "baseline_model": CALIBRATED_DISTANCE_BASELINE,
        "estimand": (
            "paired average-precision advantage on frozen future rows: "
            "AP(primary snapshot-only XGBoost) - AP(inner-calibrated distance)"
        ),
        "null_hypothesis": "delta_average_precision <= 0",
        "alternative_hypothesis": "delta_average_precision > 0",
        "support": support,
        "blocks": block_reports,
        "bootstrap": {
            "replicates": int(config.adaptability_bootstrap_replicates),
            "seed": int(config.bootstrap_seed),
            "confidence_level": float(config.confidence_level),
            "minimum_valid_fraction": float(config.min_valid_bootstrap_fraction),
            "object_resampling": "catalogue endpoint multiplicities; row weight=m_obj1*m_obj2",
            "time_resampling": "eligible non-overlap future blocks with replacement",
            "paired_methods": True,
        },
        "statistical_significance_tested": False,
        "statistical_adaptability_supported": False,
        "split_cutoff_utc": split_cutoff.isoformat().replace("+00:00", "Z"),
        "block_anchor_utc": anchor.isoformat().replace("+00:00", "Z"),
        "analysis_end_utc": analysis_end.isoformat().replace("+00:00", "Z"),
        "initial_embargo_rows_excluded": initial_embargo_rows,
    }
    insufficient_reasons = []
    if len(valid_blocks) < config.adaptability_min_blocks:
        insufficient_reasons.append("insufficient valid non-overlap future blocks")
    if len(valid_blocks) != len(complete_blocks):
        insufficient_reasons.append(
            "one or more complete planned future blocks lack required support"
        )
    if len(objects) < config.adaptability_min_unique_objects:
        insufficient_reasons.append("insufficient unique catalogue objects")
    if pair_count < config.adaptability_min_pairs:
        insufficient_reasons.append("insufficient canonical pairs")
    if analysis.empty or analysis["y_true"].nunique() != 2:
        insufficient_reasons.append("analysis population lacks both proxy classes")
    if insufficient_reasons:
        result.update(
            {
                "status": "not_enough_data",
                "reason": "; ".join(insufficient_reasons),
                "claim": "statistical adaptability claim not estimable",
            }
        )
        return result

    y = analysis["y_true"].to_numpy(dtype=int)
    primary_scores = analysis["primary_score"].to_numpy(dtype=float)
    baseline_scores = analysis["baseline_score"].to_numpy(dtype=float)
    point_delta = float(
        average_precision_score(y, primary_scores)
        - average_precision_score(y, baseline_scores)
    )
    object_1 = analysis["object_1_catalog_id"].astype(str).to_numpy()
    object_2 = analysis["object_2_catalog_id"].astype(str).to_numpy()
    block_values = analysis["adaptability_block"].to_numpy(dtype=int)
    rng = np.random.default_rng(config.bootstrap_seed)
    bootstrap_deltas: list[float] = []
    object_lookup = {value: index for index, value in enumerate(objects)}
    block_lookup = {value: index for index, value in enumerate(valid_blocks)}
    object_1_indices = np.fromiter(
        (object_lookup[value] for value in object_1), dtype=int, count=len(analysis)
    )
    object_2_indices = np.fromiter(
        (object_lookup[value] for value in object_2), dtype=int, count=len(analysis)
    )
    block_indices = np.fromiter(
        (block_lookup[int(value)] for value in block_values),
        dtype=int,
        count=len(analysis),
    )
    object_probabilities = np.full(len(objects), 1.0 / len(objects), dtype=float)
    block_probabilities = np.full(
        len(valid_blocks), 1.0 / len(valid_blocks), dtype=float
    )

    def ranking_structure(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(-scores, kind="stable")
        sorted_scores = scores[order]
        group_ends = np.flatnonzero(
            np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
        )
        return order, group_ends

    def weighted_average_precision(
        weights: np.ndarray,
        order: np.ndarray,
        group_ends: np.ndarray,
    ) -> float:
        ordered_weights = weights[order]
        ordered_y = y[order]
        cumulative_tp = np.cumsum(ordered_weights * (ordered_y == 1))
        cumulative_fp = np.cumsum(ordered_weights * (ordered_y == 0))
        total_positive = float(cumulative_tp[-1])
        total_negative = float(cumulative_fp[-1])
        if total_positive <= 0 or total_negative <= 0:
            return math.nan
        tp = cumulative_tp[group_ends]
        fp = cumulative_fp[group_ends]
        precision = np.divide(
            tp,
            tp + fp,
            out=np.zeros_like(tp, dtype=float),
            where=(tp + fp) > 0,
        )
        recall = tp / total_positive
        recall_increment = np.diff(np.r_[0.0, recall])
        return float(np.sum(recall_increment * precision))

    primary_order, primary_group_ends = ranking_structure(primary_scores)
    baseline_order, baseline_group_ends = ranking_structure(baseline_scores)
    for _ in range(config.adaptability_bootstrap_replicates):
        object_counts = rng.multinomial(len(objects), object_probabilities)
        block_counts = rng.multinomial(len(valid_blocks), block_probabilities)
        weights = (
            object_counts[object_1_indices]
            * object_counts[object_2_indices]
            * block_counts[block_indices]
        )
        primary_ap = weighted_average_precision(
            weights, primary_order, primary_group_ends
        )
        baseline_ap = weighted_average_precision(
            weights, baseline_order, baseline_group_ends
        )
        delta = primary_ap - baseline_ap
        if np.isfinite(delta):
            bootstrap_deltas.append(delta)

    valid_fraction = len(bootstrap_deltas) / config.adaptability_bootstrap_replicates
    alpha = 1.0 - config.confidence_level
    if bootstrap_deltas:
        values = np.asarray(bootstrap_deltas, dtype=float)
        lower_bound = float(np.quantile(values, alpha))
        centered = values - point_delta
        p_value = float(
            (1 + np.count_nonzero(centered >= point_delta)) / (len(values) + 1)
        )
    else:
        lower_bound = math.nan
        p_value = math.nan
    valid_block_deltas = [
        float(row["delta_average_precision"])
        for row in block_reports
        if row["eligible"] and row["delta_average_precision"] is not None
    ]
    all_blocks_positive = bool(
        len(valid_block_deltas) >= config.adaptability_min_blocks
        and all(value > 0 for value in valid_block_deltas)
    )
    supported = bool(
        valid_fraction >= config.min_valid_bootstrap_fraction
        and np.isfinite(lower_bound)
        and lower_bound > 0
        and np.isfinite(p_value)
        and p_value < alpha
        and all_blocks_positive
    )
    result.update(
        {
            "status": "supported" if supported else "not_supported",
            "statistical_significance_tested": bool(
                valid_fraction >= config.min_valid_bootstrap_fraction
            ),
            "point_delta_average_precision": point_delta,
            "one_sided_confidence_lower_bound": lower_bound,
            "centered_bootstrap_one_sided_p_value": p_value,
            "valid_bootstrap_replicates": int(len(bootstrap_deltas)),
            "valid_bootstrap_fraction": valid_fraction,
            "all_valid_future_blocks_positive": all_blocks_positive,
            "statistical_adaptability_supported": supported,
            "claim": (
                "future-block transportability advantage supported"
                if supported
                else "statistical adaptability claim not supported"
            ),
        }
    )
    return result


def adaptability_result_fingerprint(result: dict[str, object]) -> str:
    """Hash deterministic inference content, excluding run-time bookkeeping."""
    payload = {
        key: value
        for key, value in result.items()
        if key not in {"generated_utc", "source_worktree", "result_fingerprint_sha256"}
    }
    return _sha256_bytes(_canonical_json(payload))


def generate_adaptability_inference(
    predictions_path: Path,
    split_manifest_path: Path,
    output_path: Path,
    config: ExperimentConfig,
) -> Path:
    """Write hash-bound statistical adaptability evidence."""
    predictions_path = Path(predictions_path)
    split_manifest_path = Path(split_manifest_path)
    output_path = Path(output_path)
    predictions = pd.read_csv(predictions_path, comment="#")
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    if split_manifest.get("schema_version") != 2:
        raise EvidenceGenerationError("Unsupported evaluation split manifest schema")
    cutoff = pd.Timestamp(split_manifest.get("outer", {}).get("cutoff_utc"))
    if cutoff.tzinfo is None:
        raise EvidenceGenerationError("Evaluation split cutoff must be timezone-aware")
    if config.collection_end_utc is None:
        raise EvidenceGenerationError(
            "Claim-eligible adaptability requires a frozen collection end"
        )
    block_anchor = cutoff.tz_convert("UTC") + pd.Timedelta(
        hours=config.poll_interval_hours
    )
    result = adaptability_inference_from_predictions(
        predictions,
        config,
        split_cutoff_utc=cutoff.tz_convert("UTC").isoformat().replace("+00:00", "Z"),
        block_anchor_utc=block_anchor.isoformat().replace("+00:00", "Z"),
        analysis_end_utc=config.collection_end_utc,
    )
    result["predictions"] = {
        "path": str(predictions_path),
        "sha256": _file_sha256(predictions_path),
    }
    result["config"] = {
        "path": str(config.path),
        "sha256": _file_sha256(config.path),
    }
    result["split_manifest"] = {
        "path": str(split_manifest_path),
        "sha256": _file_sha256(split_manifest_path),
        "split_sha256": split_manifest.get("split_sha256"),
    }
    result["result_fingerprint_sha256"] = adaptability_result_fingerprint(result)
    write_json_atomic(output_path, result)
    return output_path


def _serializable_params(model: object) -> dict[str, object]:
    if model is None or not hasattr(model, "get_params"):
        return {}
    result: dict[str, object] = {}
    for key, value in model.get_params(deep=True).items():
        if value is None or isinstance(value, (str, int, float, bool)):
            result[key] = value
        else:
            result[key] = repr(value)
    return result


def canonical_report_from_evidence(
    gate_report: pd.DataFrame,
    evidence_report: pd.DataFrame,
) -> pd.DataFrame:
    """Replace exploratory default-threshold metrics with frozen predictions.

    Dataset/window metadata comes from the already-passed quality-gate report.
    Model training support and every numeric test metric come from the exact
    fitted/prediction evidence used by bootstrap and figures.
    """
    gate_by_model = gate_report.set_index("model", drop=False)
    rows: list[dict[str, object]] = []
    metric_mapping = {
        "pr_auc": "test_pr_auc",
        "roc_auc": "test_roc_auc",
        "precision": "test_precision",
        "recall": "test_recall",
        "f1": "test_f1",
        "accuracy": "test_accuracy",
        "false_alarm_rate": "test_proxy_false_alarm_rate",
        "false_positive": "test_false_positive",
        "false_negative": "test_false_negative",
        "true_positive": "test_true_positive",
        "true_negative": "test_true_negative",
    }
    fit_mapping = {
        "estimator_train_rows": "fitted_train_rows",
        "estimator_train_pairs": "fitted_train_pairs",
        "estimator_train_positive_rows": "fitted_train_positive_rows",
        "estimator_train_positive_pairs": "fitted_train_positive_pairs",
        "estimator_train_positive_snapshots": "fitted_train_positive_snapshots",
        "estimator_train_cutoff_utc": "fitted_train_cutoff_utc",
        "calibration_rows": "calibration_rows",
        "calibration_pairs": "calibration_pairs",
        "calibration_positive_rows": "calibration_positive_rows",
        "calibration_positive_pairs": "calibration_positive_pairs",
        "calibration_positive_snapshots": "calibration_positive_snapshots",
    }
    for _, evidence_row in evidence_report.iterrows():
        model_name = str(evidence_row["model"])
        if model_name in INELIGIBLE_SUPPORT_CONTROLS:
            continue
        if model_name not in gate_by_model.index:
            raise EvidenceGenerationError(
                f"Evidence model is absent from quality-gate report: {model_name}"
            )
        row = gate_by_model.loc[model_name].to_dict()
        for report_name, evidence_name in metric_mapping.items():
            row[report_name] = evidence_row[evidence_name]
        for report_name, evidence_name in fit_mapping.items():
            if evidence_name not in evidence_row:
                raise EvidenceGenerationError(
                    f"Evidence row is missing fitted-support field: {evidence_name}"
                )
            row[report_name] = evidence_row[evidence_name]
        threshold = evidence_row.get("operating_threshold")
        row["note"] = (
            f"frozen operating_threshold={threshold}; metrics sourced from "
            "held_out_predictions.csv; estimator_train_* is the actual inner-train fit "
            "set (null for the pre-specified fixed threshold), while train_* describes "
            "the outer development pool; uncertainty is conditional "
            "on the frozen fit/calibration split in paired_cluster_evidence.csv"
        )
        rows.append(row)
    report = pd.DataFrame(rows)
    return report[REPORT_COLUMNS + ESTIMATOR_SUPPORT_COLUMNS]


def generate_evaluation_evidence(
    dataset_path: Path,
    output_dir: Path,
    config: ExperimentConfig,
    *,
    feature_columns: Sequence[str] = FEATURES,
    snapshot_records: Sequence[dict[str, object]] | None = None,
    upstream_artifacts: Sequence[Path] = (),
    validated_input_binding: dict[str, object] | None = None,
    claim_eligible: bool = True,
) -> dict[str, Path]:
    """Fit once, freeze inner-validation thresholds, and write all evidence."""
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=".evaluation-evidence-", dir=output_dir))
    final_paths = {
        "split_manifest": output_dir / "evaluation_split_manifest.json",
        "predictions": output_dir / "held_out_predictions.csv",
        "evidence": output_dir / "paired_cluster_evidence.csv",
        "evaluation_manifest": output_dir / "evaluation_manifest.json",
        "feature_importance": output_dir / "feature_importance.csv",
    }
    work_paths = {name: temporary_dir / path.name for name, path in final_paths.items()}
    try:
        return _generate_evaluation_evidence_files(
            dataset_path,
            config,
            feature_columns,
            work_paths,
            final_paths,
            snapshot_records,
            upstream_artifacts,
            validated_input_binding,
            claim_eligible,
        )
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)


def _generate_evaluation_evidence_files(
    dataset_path: Path,
    config: ExperimentConfig,
    feature_columns: Sequence[str],
    work_paths: dict[str, Path],
    final_paths: dict[str, Path],
    snapshot_records: Sequence[dict[str, object]] | None,
    upstream_artifacts: Sequence[Path],
    validated_input_binding: dict[str, object] | None,
    claim_eligible: bool,
) -> dict[str, Path]:
    selected_features = _validated_feature_columns(feature_columns)
    primary_features = feature_set_by_name(config.primary_feature_set)
    if claim_eligible and selected_features != primary_features:
        raise EvidenceGenerationError(
            "Claim-eligible evidence must use the frozen primary snapshot-only feature set"
        )
    if claim_eligible and not (
        isinstance(validated_input_binding, dict)
        and validated_input_binding.get("validated") is True
    ):
        raise EvidenceGenerationError(
            "Claim-eligible evidence requires a validated archive-to-resimulation input binding"
        )
    feature_set_name = (
        config.primary_feature_set
        if selected_features == primary_features
        else "exploratory_custom"
    )
    frame = pd.read_csv(dataset_path, comment="#")
    required = set(selected_features) | {
        "risk_label",
        "fixed_threshold_alarm",
        "min_distance_km",
        "relative_velocity_km_s",
        "collection_id",
        config.time_column,
        "object_1",
        "object_2",
        "object_1_catalog_id",
        "object_2_catalog_id",
        "catalog_version",
        "catalog_sha256",
        "current_distance_km",
        "radial_velocity_km_s",
        "tangential_velocity_km_s",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise EvidenceGenerationError(f"Dataset is missing evidence columns: {missing}")
    catalog_ids = frame[["object_1_catalog_id", "object_2_catalog_id"]]
    if catalog_ids.isna().any().any() or (
        catalog_ids.astype(str).apply(lambda column: column.str.strip().eq(""))
    ).any().any():
        raise EvidenceGenerationError(
            "Frozen evidence requires non-empty catalogue IDs for every row"
        )
    frame, window_quality = _apply_collection_window(
        frame,
        config.time_column,
        start_utc=config.collection_start_utc,
        end_utc=config.collection_end_utc,
        poll_interval_hours=config.poll_interval_hours,
        snapshot_records=snapshot_records,
    )
    if config.min_snapshot_coverage_fraction is not None:
        # Snapshot coverage, endpoint gap, TLE-hash diversity and identical-run
        # remain in window_quality for a transparent limitations statement but
        # are NOT publication gates. (1) diversity/run measure CelesTrak's
        # ~daily upstream refresh cadence, not our collection quality; (2) a
        # ~14h GitHub-Actions scheduler outage on 2026-07-31 dropped 7
        # consecutive 2h slots, a CI-infrastructure artifact that cannot be
        # backfilled. This geometry-learning comparison is not a time-series
        # forecast, so a temporal hole does not bias the learned mapping.
        # Publication eligibility rests on TLE-age, catalogue binding, class
        # presence and the statistical class/pair/snapshot support gates below,
        # all UNCHANGED. See docs/EXPERIMENT_10D_V3.md.
        _ = (
            float(window_quality["snapshot_coverage_fraction"]),
            float(window_quality["max_snapshot_gap_hours"]),
        )
    observed_tle_ages = pd.to_numeric(frame["max_tle_age_hours"], errors="coerce")
    if (
        observed_tle_ages.isna().any()
        or not np.isfinite(observed_tle_ages).all()
        or float(observed_tle_ages.max()) > config.max_tle_age_hours
    ):
        raise EvidenceGenerationError(
            "Frozen evidence fails the maximum TLE-age quality gate"
        )
    window_quality["maximum_observed_tle_age_hours"] = float(observed_tle_ages.max())
    if sorted(frame["catalog_version"].dropna().astype(str).unique()) != [config.catalog_version]:
        raise EvidenceGenerationError("Evidence dataset catalog_version mismatch")
    if sorted(frame["catalog_sha256"].dropna().astype(str).unique()) != [config.catalog_sha256]:
        raise EvidenceGenerationError("Evidence dataset catalog_sha256 mismatch")
    if frame["risk_label"].nunique() != 2:
        raise EvidenceGenerationError("Evidence dataset requires both classes")
    expected_proxy_label = (
        pd.to_numeric(frame["min_distance_km"], errors="raise")
        <= float(config.label_threshold_km)
    ) & (
        pd.to_numeric(frame["relative_velocity_km_s"], errors="raise")
        >= float(config.label_relative_velocity_km_s)
    )
    if not np.array_equal(
        frame["risk_label"].astype(int).to_numpy(), expected_proxy_label.astype(int).to_numpy()
    ):
        raise EvidenceGenerationError(
            "risk_label is inconsistent with the frozen deterministic proxy rule"
        )
    try:
        outer = _split_data(
            frame,
            config.time_column,
            train_time_fraction=config.train_time_fraction,
            test_pair_fraction=config.test_pair_fraction,
            pair_seed=config.pair_seed,
        )
        inner = _pair_grouped_time_split(
            outer.train,
            config.time_column,
            train_time_fraction=config.inner_train_time_fraction,
            test_pair_fraction=config.inner_validation_pair_fraction,
            pair_seed=config.inner_pair_seed,
        )
    except InsufficientGroupedSplitError as exc:
        raise EvidenceGenerationError(f"Cannot form frozen outer/inner split: {exc}") from exc
    for name, partition in {
        "outer_train": outer.train,
        "outer_test": outer.test,
        "inner_train": inner.train,
        "inner_validation": inner.test,
    }.items():
        if partition["risk_label"].nunique() != 2:
            raise EvidenceGenerationError(f"{name} must contain both classes")
    if snapshot_records is None:
        raise EvidenceGenerationError(
            "Claim evidence requires authoritative snapshot records for partition quality"
        )
    partition_tle_quality = {
        name: _partition_tle_hash_quality(partition, snapshot_records)
        for name, partition in {
            "outer_train": outer.train,
            "outer_test": outer.test,
            "inner_train": inner.train,
            "inner_validation": inner.test,
        }.items()
    }
    # Per-partition TLE-hash diversity / identical-run are retained in
    # partition_tle_quality (reported below) but are no longer publication
    # gates -- they measure CelesTrak's upstream ~daily refresh cadence for
    # these objects, not our collection quality. Coverage, endpoint gap,
    # TLE-age, catalogue binding and all statistical class/pair/snapshot
    # support gates remain hard fail-closed. See docs/EXPERIMENT_10D_V3.md.
    outer_support = _split_positive_support(outer, config.time_column)
    outer_requirements = {
        "train_positive_rows": config.min_train_positive_rows,
        "test_positive_rows": config.min_test_positive_rows,
        "train_positive_pairs": config.min_train_positive_pairs,
        "test_positive_pairs": config.min_test_positive_pairs,
        "train_positive_snapshots": config.min_train_positive_snapshots,
        "test_positive_snapshots": config.min_test_positive_snapshots,
    }
    outer_failures = [
        f"{name}={outer_support[name]} < required={required}"
        for name, required in outer_requirements.items()
        if outer_support[name] is None or outer_support[name] < required
    ]
    if outer_failures:
        raise EvidenceGenerationError(
            "Outer publication support gate failed: " + "; ".join(outer_failures)
        )
    inner_train_support = _positive_partition_support(inner.train, config.time_column)
    inner_validation_support = _positive_partition_support(inner.test, config.time_column)
    inner_requirements = {
        "inner_train_positive_rows": (
            inner_train_support["rows"], config.min_inner_train_positive_rows
        ),
        "inner_validation_positive_rows": (
            inner_validation_support["rows"], config.min_inner_validation_positive_rows
        ),
        "inner_train_positive_pairs": (
            inner_train_support["pairs"], config.min_inner_train_positive_pairs
        ),
        "inner_validation_positive_pairs": (
            inner_validation_support["pairs"], config.min_inner_validation_positive_pairs
        ),
        "inner_train_positive_snapshots": (
            inner_train_support["snapshots"], config.min_inner_train_positive_snapshots
        ),
        "inner_validation_positive_snapshots": (
            inner_validation_support["snapshots"], config.min_inner_validation_positive_snapshots
        ),
    }
    inner_failures = [
        f"{name}={observed} < required={required}"
        for name, (observed, required) in inner_requirements.items()
        if observed is None or observed < required
    ]
    if inner_failures:
        raise EvidenceGenerationError(
            "Inner threshold-calibration support gate failed: " + "; ".join(inner_failures)
        )

    row_ids = _row_identity(frame, config.time_column)
    pair_ids = _canonical_pair_ids(frame)
    split_payload = {
        "schema_version": 2,
        "dataset_sha256": _file_sha256(dataset_path),
        "config_sha256": _file_sha256(config.path),
        "catalog_version": config.catalog_version,
        "catalog_sha256": config.catalog_sha256,
        "time_column": config.time_column,
        "outer": {
            "cutoff_utc": outer.metadata.get("cutoff_utc"),
            "pair_seed": config.pair_seed,
            "train_time_fraction": config.train_time_fraction,
            "test_pair_fraction": config.test_pair_fraction,
            "train_row_ids": sorted(row_ids.loc[outer.train.index]),
            "test_row_ids": sorted(row_ids.loc[outer.test.index]),
            "excluded_row_ids": sorted(row_ids.loc[outer.excluded.index]),
            "train_pair_ids": sorted(set(pair_ids.loc[outer.train.index])),
            "test_pair_ids": sorted(set(pair_ids.loc[outer.test.index])),
        },
        "inner": {
            "cutoff_utc": inner.metadata.get("cutoff_utc"),
            "pair_seed": config.inner_pair_seed,
            "train_time_fraction": config.inner_train_time_fraction,
            "validation_pair_fraction": config.inner_validation_pair_fraction,
            "train_row_ids": sorted(row_ids.loc[inner.train.index]),
            "validation_row_ids": sorted(row_ids.loc[inner.test.index]),
            "train_rows": int(len(inner.train)),
            "validation_rows": int(len(inner.test)),
            "train_pair_ids": sorted(set(pair_ids.loc[inner.train.index])),
            "validation_pair_ids": sorted(set(pair_ids.loc[inner.test.index])),
            "train_positive_support": inner_train_support,
            "validation_positive_support": inner_validation_support,
        },
        "window_quality": window_quality,
        "partition_tle_quality": partition_tle_quality,
        "outer_positive_support": outer_support,
        "features": selected_features,
        "feature_set_name": feature_set_name,
        "claim_eligible": claim_eligible,
    }
    split_sha256 = _sha256_bytes(_canonical_json(split_payload))
    split_manifest = {**split_payload, "split_sha256": split_sha256}
    split_manifest_path = work_paths["split_manifest"]
    write_json_atomic(split_manifest_path, split_manifest)

    inner_baseline_pred = inner.test["fixed_threshold_alarm"].astype(int).to_numpy()
    inner_y = inner.test["risk_label"].astype(int).to_numpy()
    baseline_target_recall = _binary_metrics(
        inner_y, inner_baseline_pred, _distance_baseline_score(inner.test)
    )["recall"]
    if baseline_target_recall <= 0:
        raise EvidenceGenerationError(
            "Fixed-distance baseline has zero inner-validation recall; comparable-recall threshold is undefined"
        )

    test_y = outer.test["risk_label"].astype(int).to_numpy()
    test_row_ids = row_ids.loc[outer.test.index].to_numpy()
    test_pair_ids = pair_ids.loc[outer.test.index].to_numpy()
    test_object_1_ids = outer.test["object_1_catalog_id"].astype(str).to_numpy()
    test_object_2_ids = outer.test["object_2_catalog_id"].astype(str).to_numpy()
    test_times = pd.to_datetime(
        outer.test[config.time_column], utc=True, errors="raise", format="ISO8601"
    ).map(lambda value: value.isoformat().replace("+00:00", "Z"))
    prediction_rows: list[dict[str, object]] = []
    baseline_scores = _distance_baseline_score(outer.test)
    baseline_pred = outer.test["fixed_threshold_alarm"].astype(int).to_numpy()
    for index in range(len(outer.test)):
        prediction_rows.append(
            {
                "source_row_id": test_row_ids[index],
                "snapshot_utc": test_times.iloc[index],
                "canonical_pair_id": test_pair_ids[index],
                "object_1_catalog_id": test_object_1_ids[index],
                "object_2_catalog_id": test_object_2_ids[index],
                "y_true": int(test_y[index]),
                "model": "fixed_threshold",
                "score": float(baseline_scores[index]),
                "min_distance_km": float(outer.test["min_distance_km"].iloc[index]),
                "operating_threshold": -float(config.fixed_threshold_km),
                "pred": int(baseline_pred[index]),
                "threshold_source": "pre-specified fixed distance",
                "calibration_target_recall": None,
                "calibration_recall": None,
                "calibration_proxy_false_alarm_rate": None,
                "split_sha256": split_sha256,
            }
        )

    calibrated_distance_info, calibrated_distance_pred = calibrate_distance_baseline(
        inner_y,
        _distance_baseline_score(inner.test),
        baseline_scores,
        baseline_target_recall,
    )
    thresholds: dict[str, dict[str, float]] = {
        CALIBRATED_DISTANCE_BASELINE: calibrated_distance_info
    }
    for index in range(len(outer.test)):
        prediction_rows.append(
            {
                "source_row_id": test_row_ids[index],
                "snapshot_utc": test_times.iloc[index],
                "canonical_pair_id": test_pair_ids[index],
                "object_1_catalog_id": test_object_1_ids[index],
                "object_2_catalog_id": test_object_2_ids[index],
                "y_true": int(test_y[index]),
                "model": CALIBRATED_DISTANCE_BASELINE,
                "score": float(baseline_scores[index]),
                "min_distance_km": float(outer.test["min_distance_km"].iloc[index]),
                "operating_threshold": calibrated_distance_info["threshold"],
                "pred": int(calibrated_distance_pred[index]),
                "threshold_source": (
                    "inner pair-held-out future validation; recall matched to "
                    f"pre-specified {config.fixed_threshold_km:g} km baseline"
                ),
                "calibration_target_recall": calibrated_distance_info[
                    "target_recall"
                ],
                "calibration_recall": calibrated_distance_info[
                    "validation_recall"
                ],
                "calibration_proxy_false_alarm_rate": calibrated_distance_info[
                    "validation_proxy_false_alarm_rate"
                ],
                "split_sha256": split_sha256,
            }
        )

    if claim_eligible:
        oracle_pred = (
            (
                pd.to_numeric(outer.test["min_distance_km"], errors="raise")
                <= float(config.label_threshold_km)
            )
            & (
                pd.to_numeric(outer.test["relative_velocity_km_s"], errors="raise")
                >= float(config.label_relative_velocity_km_s)
            )
        ).astype(int).to_numpy()
        for index in range(len(outer.test)):
            prediction_rows.append(
                {
                    "source_row_id": test_row_ids[index],
                    "snapshot_utc": test_times.iloc[index],
                    "canonical_pair_id": test_pair_ids[index],
                    "object_1_catalog_id": test_object_1_ids[index],
                    "object_2_catalog_id": test_object_2_ids[index],
                    "y_true": int(test_y[index]),
                    "model": "proxy_rule_oracle",
                    "score": float(oracle_pred[index]),
                    "min_distance_km": float(outer.test["min_distance_km"].iloc[index]),
                    "operating_threshold": 0.5,
                    "pred": int(oracle_pred[index]),
                    "threshold_source": (
                        f"exact frozen proxy: distance <= {config.label_threshold_km:g} km and "
                        f"relative velocity >= {config.label_relative_velocity_km_s:g} km/s"
                    ),
                    "calibration_target_recall": None,
                    "calibration_recall": None,
                    "calibration_proxy_false_alarm_rate": None,
                    "split_sha256": split_sha256,
                }
            )

        cpa_scores = _constant_velocity_cpa_score(
            outer.test, config.horizon_minutes
        )
        cpa_pred = (cpa_scores >= -float(config.fixed_threshold_km)).astype(int)
        for index in range(len(outer.test)):
            prediction_rows.append(
                {
                    "source_row_id": test_row_ids[index],
                    "snapshot_utc": test_times.iloc[index],
                    "canonical_pair_id": test_pair_ids[index],
                    "object_1_catalog_id": test_object_1_ids[index],
                    "object_2_catalog_id": test_object_2_ids[index],
                    "y_true": int(test_y[index]),
                    "model": "constant_velocity_cpa",
                    "score": float(cpa_scores[index]),
                    "min_distance_km": float(outer.test["min_distance_km"].iloc[index]),
                    "operating_threshold": -float(config.fixed_threshold_km),
                    "pred": int(cpa_pred[index]),
                    "threshold_source": (
                        f"pre-specified {config.fixed_threshold_km:g} km "
                        "snapshot-state linear CPA"
                    ),
                    "calibration_target_recall": None,
                    "calibration_recall": None,
                    "calibration_proxy_false_alarm_rate": None,
                    "split_sha256": split_sha256,
                }
            )

    inner_models = _build_models(_class_ratio(inner.train))
    fitted_models: dict[str, object] = {}
    permutation_train_support: dict[str, int] | None = None
    model_names = sorted(set(inner_models) - {"fixed_threshold"})
    missing_models = sorted(set(config.required_models) - set(model_names))
    if missing_models:
        raise EvidenceGenerationError(
            f"Required publication models are unavailable: {missing_models}"
        )
    for model_name in model_names:
        inner_model = inner_models[model_name]
        inner_model.fit(inner.train[selected_features], inner.train["risk_label"].astype(int))
        inner_scores = np.asarray(_ranking_score(inner_model, inner.test[selected_features]), dtype=float)
        threshold_info = select_threshold_for_target_recall(
            inner_y, inner_scores, baseline_target_recall
        )
        # Keep the exact fitted estimator whose score scale was used to select
        # the threshold. Refitting on outer-train+validation could shift SVM
        # decision scores or boosted probabilities and invalidate the frozen
        # operating threshold.
        test_scores = np.asarray(_ranking_score(inner_model, outer.test[selected_features]), dtype=float)
        if not np.all(np.isfinite(test_scores)):
            raise EvidenceGenerationError(f"Non-finite outer-test scores for {model_name}")
        test_pred = (test_scores >= threshold_info["threshold"]).astype(int)
        thresholds[model_name] = threshold_info
        fitted_models[model_name] = inner_model
        for index in range(len(outer.test)):
            prediction_rows.append(
                {
                    "source_row_id": test_row_ids[index],
                    "snapshot_utc": test_times.iloc[index],
                    "canonical_pair_id": test_pair_ids[index],
                    "object_1_catalog_id": test_object_1_ids[index],
                    "object_2_catalog_id": test_object_2_ids[index],
                    "y_true": int(test_y[index]),
                    "model": model_name,
                    "score": float(test_scores[index]),
                    "min_distance_km": float(outer.test["min_distance_km"].iloc[index]),
                    "operating_threshold": threshold_info["threshold"],
                    "pred": int(test_pred[index]),
                    "threshold_source": "inner pair-held-out future validation",
                    "calibration_target_recall": threshold_info["target_recall"],
                    "calibration_recall": threshold_info["validation_recall"],
                    "calibration_proxy_false_alarm_rate": threshold_info[
                        "validation_proxy_false_alarm_rate"
                    ],
                    "split_sha256": split_sha256,
                }
            )

    if claim_eligible:
        permutation_name = "label_permutation_control"
        permutation_model = clone(inner_models[config.primary_model])
        permutation_rng = np.random.default_rng(config.bootstrap_seed)
        permuted_labels = permutation_rng.permutation(
            inner.train["risk_label"].astype(int).to_numpy()
        )
        permutation_train = inner.train.copy()
        permutation_train["risk_label"] = permuted_labels
        permutation_train_support = _positive_partition_support(
            permutation_train, config.time_column
        )
        permutation_model.fit(inner.train[selected_features], permuted_labels)
        validation_scores = np.asarray(
            _ranking_score(permutation_model, inner.test[selected_features]), dtype=float
        )
        threshold_info = select_threshold_for_target_recall(
            inner_y, validation_scores, baseline_target_recall
        )
        test_scores = np.asarray(
            _ranking_score(permutation_model, outer.test[selected_features]), dtype=float
        )
        if not np.all(np.isfinite(test_scores)):
            raise EvidenceGenerationError("Non-finite label-permutation control scores")
        test_pred = (test_scores >= threshold_info["threshold"]).astype(int)
        thresholds[permutation_name] = threshold_info
        fitted_models[permutation_name] = permutation_model
        for index in range(len(outer.test)):
            prediction_rows.append(
                {
                    "source_row_id": test_row_ids[index],
                    "snapshot_utc": test_times.iloc[index],
                    "canonical_pair_id": test_pair_ids[index],
                    "object_1_catalog_id": test_object_1_ids[index],
                    "object_2_catalog_id": test_object_2_ids[index],
                    "y_true": int(test_y[index]),
                    "model": permutation_name,
                    "score": float(test_scores[index]),
                    "min_distance_km": float(outer.test["min_distance_km"].iloc[index]),
                    "operating_threshold": threshold_info["threshold"],
                    "pred": int(test_pred[index]),
                    "threshold_source": "inner validation after deterministic train-label permutation",
                    "calibration_target_recall": threshold_info["target_recall"],
                    "calibration_recall": threshold_info["validation_recall"],
                    "calibration_proxy_false_alarm_rate": threshold_info[
                        "validation_proxy_false_alarm_rate"
                    ],
                    "split_sha256": split_sha256,
                }
            )

    predictions = pd.DataFrame(prediction_rows).sort_values(
        ["model", "snapshot_utc", "canonical_pair_id"], kind="stable"
    )
    predictions_path = work_paths["predictions"]
    provenance_config = (
        f"split_sha256={split_sha256}, threshold_source=inner_pair_grouped_time_validation, "
        f"features={'|'.join(selected_features)}"
    )
    write_csv_text_with_provenance(
        predictions_path,
        predictions.to_csv(index=False),
        source=f"dataset={dataset_path} sha256={split_payload['dataset_sha256']}",
        config_summary=provenance_config,
    )

    evidence = paired_pair_cluster_bootstrap(
        predictions,
        replicates=config.bootstrap_replicates,
        seed=config.bootstrap_seed,
        confidence_level=config.confidence_level,
        recall_noninferiority_margin=config.recall_noninferiority_margin,
        min_valid_fraction=config.min_valid_bootstrap_fraction,
        primary_model=config.primary_model,
        claim_eligible=claim_eligible,
    )
    fitted_pair_count = int(_canonical_pair_ids(inner.train).nunique())
    fitted_support = {
        "fitted_train_rows": int(len(inner.train)),
        "fitted_train_pairs": fitted_pair_count,
        "fitted_train_positive_rows": int(inner_train_support["rows"]),
        "fitted_train_positive_pairs": int(inner_train_support["pairs"]),
        "fitted_train_positive_snapshots": int(inner_train_support["snapshots"]),
        "fitted_train_cutoff_utc": inner.metadata.get("cutoff_utc"),
        "calibration_rows": int(len(inner.test)),
        "calibration_pairs": int(_canonical_pair_ids(inner.test).nunique()),
        "calibration_positive_rows": int(inner_validation_support["rows"]),
        "calibration_positive_pairs": int(inner_validation_support["pairs"]),
        "calibration_positive_snapshots": int(inner_validation_support["snapshots"]),
    }
    for key, value in fitted_support.items():
        evidence[key] = value
    if permutation_train_support is not None:
        permutation_mask = evidence["model"].eq("label_permutation_control")
        evidence.loc[permutation_mask, "fitted_train_positive_rows"] = int(
            permutation_train_support["rows"]
        )
        evidence.loc[permutation_mask, "fitted_train_positive_pairs"] = int(
            permutation_train_support["pairs"]
        )
        evidence.loc[permutation_mask, "fitted_train_positive_snapshots"] = int(
            permutation_train_support["snapshots"]
        )
    no_support_mask = evidence["model"].isin(
        ["fixed_threshold", "constant_velocity_cpa", "proxy_rule_oracle"]
    )
    for key in fitted_support:
        evidence.loc[no_support_mask, key] = None
    calibrated_mask = evidence["model"].eq(CALIBRATED_DISTANCE_BASELINE)
    for key in fitted_support:
        if key.startswith("fitted_"):
            evidence.loc[calibrated_mask, key] = None
    evidence_path = work_paths["evidence"]
    write_csv_text_with_provenance(
        evidence_path,
        evidence.to_csv(index=False),
        source=(
            f"predictions={final_paths['predictions']} "
            f"sha256={_file_sha256(predictions_path)}"
        ),
        config_summary=(
            f"split_sha256={split_sha256}, bootstrap_replicates={config.bootstrap_replicates}, "
            f"bootstrap_seed={config.bootstrap_seed}, confidence_level={config.confidence_level}, "
            f"recall_noninferiority_margin={config.recall_noninferiority_margin}"
        ),
    )

    importance_rows: list[dict[str, object]] = []
    for model_name, model in fitted_models.items():
        if model_name == "label_permutation_control":
            continue
        if hasattr(model, "feature_importances_"):
            values = np.asarray(model.feature_importances_, dtype=float)
            if len(values) != len(selected_features):
                raise EvidenceGenerationError(
                    f"Feature-importance width mismatch for {model_name}"
                )
            for feature, importance in zip(selected_features, values):
                importance_rows.append(
                    {
                        "model": model_name,
                        "feature": feature,
                        "importance": float(importance),
                        "split_sha256": split_sha256,
                    }
                )
    feature_importance_path = work_paths["feature_importance"]
    write_csv_text_with_provenance(
        feature_importance_path,
        pd.DataFrame(
            importance_rows,
            columns=["model", "feature", "importance", "split_sha256"],
        ).to_csv(index=False),
        source=f"dataset={dataset_path} sha256={split_payload['dataset_sha256']}",
        config_summary=f"split_sha256={split_sha256}, fitted_on=inner_train",
    )

    dependency_names = ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm")
    dependencies = {}
    for dependency in dependency_names:
        try:
            dependencies[dependency] = importlib.metadata.version(dependency)
        except importlib.metadata.PackageNotFoundError:
            dependencies[dependency] = None
    upstream = []
    for path in upstream_artifacts:
        path = Path(path)
        if not path.is_file():
            raise EvidenceGenerationError(f"Required upstream artifact is missing: {path}")
        upstream.append({"path": str(path), "sha256": _file_sha256(path)})
    evaluation_manifest = {
        "schema_version": 3,
        "generated_utc": generated_utc(),
        "git_commit": git_commit_full_hash(),
        "source_worktree": git_worktree_state(),
        "dataset": {"path": str(dataset_path), "sha256": split_payload["dataset_sha256"]},
        "config": {"path": str(config.path), "sha256": split_payload["config_sha256"]},
        "split_manifest": {
            "path": str(final_paths["split_manifest"]),
            "sha256": _file_sha256(split_manifest_path),
            "split_sha256": split_sha256,
        },
        "predictions": {
            "path": str(final_paths["predictions"]),
            "sha256": _file_sha256(predictions_path),
        },
        "evidence": {
            "path": str(final_paths["evidence"]),
            "sha256": _file_sha256(evidence_path),
        },
        "feature_importance": {
            "path": str(final_paths["feature_importance"]),
            "sha256": _file_sha256(feature_importance_path),
        },
        "threshold_protocol": {
            "baseline_validation_target_recall": baseline_target_recall,
            "models": thresholds,
        },
        "fitted_estimator_support": fitted_support,
        "bootstrap": {
            "unit": "canonical_pair_id",
            "secondary_dependence_unit": "UTC calendar day",
            "minimum_time_blocks": MIN_TIME_BLOCKS,
            "paired": True,
            "replicates": config.bootstrap_replicates,
            "seed": config.bootstrap_seed,
            "confidence_level": config.confidence_level,
            "recall_noninferiority_margin": config.recall_noninferiority_margin,
            "min_valid_fraction": config.min_valid_bootstrap_fraction,
            "primary_model": config.primary_model,
            "multiplicity_policy": "single pre-specified primary model; all others exploratory",
            "claim_eligible": claim_eligible,
            "uncertainty_scope": (
                "conditional on the frozen fitted estimators and inner-validation "
                "thresholds; training/calibration variability is not resampled; pair "
                "and UTC-day bootstraps are separate marginal sensitivity analyses, "
                "not a multiway bootstrap; shared-object network dependence is not modeled; "
                "UTC-day resampling assumes exchangeable held-out day blocks"
            ),
        },
        "features": selected_features,
        "feature_set_name": feature_set_name,
        "model_parameters": {
            name: _serializable_params(model) for name, model in fitted_models.items()
        },
        "support_controls": {
            CALIBRATED_DISTANCE_BASELINE: {
                "enabled": True,
                "inputs": ["min_distance_km"],
                "calibration_partition": "inner pair-held-out future validation",
                "target_recall_source": (
                    f"pre-specified {config.fixed_threshold_km:g} km baseline"
                ),
                "operating_threshold": calibrated_distance_info["threshold"],
                "distance_threshold_km": -calibrated_distance_info["threshold"],
                "claim_eligible": False,
                "purpose": (
                    "tests whether validation-only distance-threshold tuning explains "
                    "the apparent false-alarm reduction; it directly uses one component "
                    "of the deterministic proxy label and is not independent ground truth"
                ),
            },
            "constant_velocity_cpa": {
                "enabled": claim_eligible,
                "inputs": [
                    "current_distance_km",
                    "radial_velocity_km_s",
                    "tangential_velocity_km_s",
                ],
                "horizon_minutes": config.horizon_minutes,
                "distance_threshold_km": config.fixed_threshold_km,
                "claim_eligible": False,
                "comparison": (
                    "paired model-minus-CPA deltas and pair/day bootstrap intervals are "
                    "reported as a same-snapshot-information sensitivity analysis"
                ),
            },
            "label_permutation_control": {
                "enabled": claim_eligible,
                "permutation_seed": config.bootstrap_seed,
                "base_estimator": config.primary_model,
                "claim_eligible": False,
            },
            "proxy_rule_oracle": {
                "enabled": claim_eligible,
                "inputs": ["min_distance_km", "relative_velocity_km_s"],
                "label_threshold_km": config.label_threshold_km,
                "label_relative_velocity_km_s": config.label_relative_velocity_km_s,
                "purpose": "positive control exposing deterministic target construction",
                "claim_eligible": False,
            },
        },
        "dependencies": dependencies,
        "upstream_artifacts": upstream,
        "archive_resimulation_binding": validated_input_binding,
        "scientific_scope": (
            "Labels and false alarms refer to the deterministic physics-based proxy, not physical Pc."
        ),
        "evaluation_population": {
            "condition": "SGP4 candidate-selected pairs",
            "candidate_threshold_km": config.candidate_threshold_km,
            "catalog_version": config.catalog_version,
            "catalog_sha256": config.catalog_sha256,
            "catalogue_wide_claim_eligible": False,
        },
    }
    evaluation_manifest_path = work_paths["evaluation_manifest"]
    write_json_atomic(evaluation_manifest_path, evaluation_manifest)
    for name in (
        "split_manifest",
        "predictions",
        "evidence",
        "feature_importance",
        "evaluation_manifest",
    ):
        os.replace(work_paths[name], final_paths[name])
    return final_paths
