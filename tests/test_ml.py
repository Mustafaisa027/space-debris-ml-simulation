"""Unit tests for space_debris.ml: imbalance-aware models, PR-AUC/ROC-AUC
reporting, and TimeSeriesSplit cross-validation (ROADMAP_YOL1.md GOREV 4).

Tests use a perfectly BALANCED synthetic dataset (50/50 risk_label), as
requested, so failures are attributable to the new metrics/weighting wiring
rather than to the severe class imbalance already covered by GOREV 3.

Run with:  PYTHONPATH=src python -m pytest tests/ -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from space_debris.ml import (
    FEATURES,
    REPORT_COLUMNS,
    _build_models,
    _ranking_metrics,
    compare_models,
    compare_to_baseline_pr_auc,
    time_series_cv_report,
)


def _synthetic_dataset(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Perfectly balanced and interleaved so every time-ordered prefix (used by
    # both the chronological split and TimeSeriesSplit folds) still contains
    # both classes.
    labels = np.array([i % 2 for i in range(n)])
    min_distance = np.where(labels == 1, rng.uniform(1.0, 15.0, n), rng.uniform(50.0, 500.0, n))
    relative_velocity = np.where(labels == 1, rng.uniform(8.0, 15.0, n), rng.uniform(0.5, 4.0, n))

    return pd.DataFrame(
        {
            "time_to_tca_min": rng.uniform(0.0, 720.0, n),
            "current_distance_km": min_distance + rng.uniform(0.0, 5.0, n),
            "min_distance_km": min_distance,
            "relative_velocity_km_s": relative_velocity,
            "altitude_difference_km": rng.uniform(0.0, 50.0, n),
            "max_tle_age_hours": rng.uniform(0.0, 100.0, n),
            "fixed_threshold_alarm": (min_distance <= 25.0).astype(int),
            "risk_label": labels,
            "snapshot_utc": pd.date_range("2026-01-01", periods=n, freq="h").astype(str),
        }
    )


def test_build_models_are_imbalance_aware():
    models = _build_models(scale_pos_weight=3.0)

    assert models["logistic_regression"].named_steps["logisticregression"].class_weight == "balanced"
    assert models["random_forest"].class_weight == "balanced"
    assert models["svm"].named_steps["svc"].class_weight == "balanced"
    if "xgboost" in models:
        assert models["xgboost"].get_params()["scale_pos_weight"] == 3.0


def test_ranking_metrics_matches_sklearn_directly():
    y_true = [0, 0, 1, 1, 0, 1]
    scores = [0.1, 0.4, 0.35, 0.8, 0.2, 0.9]
    pr_auc, roc_auc, note = _ranking_metrics(y_true, scores)

    assert note == ""
    assert pr_auc == pytest.approx(average_precision_score(y_true, scores))
    assert roc_auc == pytest.approx(roc_auc_score(y_true, scores))


def test_ranking_metrics_handles_single_class_test_split():
    pr_auc, roc_auc, note = _ranking_metrics([1, 1, 1], [0.2, 0.5, 0.9])
    assert pr_auc is None
    assert roc_auc is None
    assert "single class" in note


def test_compare_models_reports_pr_auc_and_roc_auc(tmp_path):
    df = _synthetic_dataset(200, seed=1)
    dataset_path = tmp_path / "dataset.csv"
    df.to_csv(dataset_path, index=False)
    report_path = tmp_path / "report.csv"

    report = compare_models(dataset_path, report_path, time_column=None)

    assert list(report.columns) == REPORT_COLUMNS
    assert {"pr_auc", "roc_auc", "precision", "recall", "f1", "accuracy"} <= set(report.columns)
    # With well-separated balanced classes every model (including the
    # degenerate fixed_threshold "score") should have a defined PR-AUC.
    assert report["pr_auc"].notna().all()
    assert report["roc_auc"].notna().all()
    assert report_path.exists()


def test_compare_models_time_split_uses_chronological_order(tmp_path):
    df = _synthetic_dataset(200, seed=2)
    dataset_path = tmp_path / "dataset.csv"
    df.to_csv(dataset_path, index=False)
    report_path = tmp_path / "report.csv"

    report = compare_models(dataset_path, report_path, time_column="snapshot_utc")

    assert (report["split"] == "time").all()
    assert report["pr_auc"].notna().any()


def test_compare_to_baseline_pr_auc_reports_deltas():
    report = pd.DataFrame(
        [
            {"model": "fixed_threshold", "pr_auc": 0.5},
            {"model": "random_forest", "pr_auc": 0.8},
            {"model": "logistic_regression", "pr_auc": 0.3},
        ]
    )
    text = compare_to_baseline_pr_auc(report)

    assert "fixed_threshold PR-AUC = 0.5000 (baseline)" in text
    assert "random_forest PR-AUC = 0.8000 (beats baseline, delta=+0.3000)" in text
    assert "logistic_regression PR-AUC = 0.3000 (does not beat baseline, delta=-0.2000)" in text


def test_compare_to_baseline_pr_auc_without_baseline_row():
    report = pd.DataFrame([{"model": "random_forest", "pr_auc": 0.8}])
    assert compare_to_baseline_pr_auc(report) == "fixed_threshold baseline not present in report."


def test_time_series_cv_report_aggregates_mean_and_std(tmp_path):
    df = _synthetic_dataset(60, seed=3)
    dataset_path = tmp_path / "dataset.csv"
    df.to_csv(dataset_path, index=False)
    report_path = tmp_path / "cv_report.csv"

    report = time_series_cv_report(dataset_path, report_path, n_splits=3, time_column="snapshot_utc")

    assert {"model", "metric", "mean", "std", "folds"} <= set(report.columns)
    pr_auc_rows = report[(report["model"] == "random_forest") & (report["metric"] == "pr_auc")]
    assert len(pr_auc_rows) == 1
    row = pr_auc_rows.iloc[0]
    assert row["folds"] == 3
    assert row["mean"] is not None
    assert row["std"] is not None
    assert report_path.exists()


def test_time_series_cv_report_handles_too_few_rows(tmp_path):
    df = _synthetic_dataset(4, seed=4)
    dataset_path = tmp_path / "dataset.csv"
    df.to_csv(dataset_path, index=False)
    report_path = tmp_path / "cv_report.csv"

    report = time_series_cv_report(dataset_path, report_path, n_splits=5, time_column="snapshot_utc")

    assert report["model"].iloc[0] == "not_enough_data"


def test_compare_models_missing_columns_raises(tmp_path):
    dataset_path = tmp_path / "bad.csv"
    pd.DataFrame({"risk_label": [0, 1, 0, 1, 0, 1]}).to_csv(dataset_path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        compare_models(dataset_path, tmp_path / "report.csv")


def test_features_still_excludes_leakage_columns():
    assert "risk_score" not in FEATURES
    assert "risk_label" not in FEATURES
    assert "fixed_threshold_alarm" not in FEATURES
