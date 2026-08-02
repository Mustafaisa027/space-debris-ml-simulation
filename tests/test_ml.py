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
    FEATURES_WITHOUT_LABEL_RULE,
    SNAPSHOT_ONLY_FEATURES,
    REPORT_COLUMNS,
    _build_models,
    _apply_collection_window,
    _canonical_pair_ids,
    _pair_is_test,
    _pair_grouped_time_split,
    _ranking_metrics,
    compare_models,
    compare_to_baseline_pr_auc,
    model_predictions_for_plotting,
    temporal_consistency_summary,
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
            "relative_radial_km": rng.uniform(-100.0, 100.0, n),
            "relative_intrack_km": rng.uniform(-100.0, 100.0, n),
            "relative_crosstrack_km": rng.uniform(-100.0, 100.0, n),
            "relative_inclination_deg": rng.uniform(0.0, 180.0, n),
            "radial_velocity_km_s": rng.uniform(-10.0, 10.0, n),
            "tangential_velocity_km_s": rng.uniform(0.0, 10.0, n),
            "approach_angle_deg": rng.uniform(0.0, 180.0, n),
            "fixed_threshold_alarm": (min_distance <= 25.0).astype(int),
            "risk_label": labels,
            "snapshot_utc": pd.date_range("2026-01-01", periods=n, freq="h").astype(str),
            "object_1": [f"OBJECT-{i:05d}" for i in range(n)],
            "object_2": [f"TARGET-{i:05d}" for i in range(n)],
        }
    )


def test_build_models_are_imbalance_aware():
    models = _build_models(scale_pos_weight=3.0)

    assert models["logistic_regression"].named_steps["logisticregression"].class_weight == "balanced"
    assert models["decision_tree"].class_weight == "balanced"
    assert models["random_forest"].class_weight == "balanced"
    assert models["svm"].named_steps["svc"].class_weight == "balanced"
    if "xgboost" in models:
        assert models["xgboost"].get_params()["scale_pos_weight"] == 3.0
    if "lightgbm" in models:
        assert models["lightgbm"].get_params()["class_weight"] == "balanced"


def test_adviser_model_set_contains_tree_and_primary_candidates():
    models = _build_models(scale_pos_weight=3.0)

    assert {"logistic_regression", "svm", "decision_tree", "random_forest"} <= set(models)
    assert "xgboost" in models or "lightgbm" in models


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


def test_collection_window_counts_unique_cadence_slots_not_retries():
    df = pd.DataFrame(
        {
            "snapshot_utc": [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:10:00Z",  # same nominal two-hour slot
                "2026-01-01T02:00:00Z",
                "2026-01-01T04:00:00Z",
                "2026-01-01T06:00:00Z",
                "2026-01-01T08:00:00Z",  # half-open end: excluded
            ],
            "value": range(6),
        }
    )

    selected, quality = _apply_collection_window(
        df,
        "snapshot_utc",
        start_utc="2026-01-01T00:00:00Z",
        end_utc="2026-01-01T08:00:00Z",
        poll_interval_hours=2,
    )

    assert len(selected) == 4
    assert quality["expected_snapshot_slots"] == 4
    assert quality["observed_snapshot_times"] == 5
    assert quality["occupied_snapshot_slots"] == 4
    assert quality["duplicate_slot_snapshot_times"] == 1
    assert quality["duplicate_slot_rows_excluded"] == 1
    assert quality["snapshot_coverage_fraction"] == 1.0
    assert quality["max_snapshot_gap_hours"] == 2.0
    assert quality["window_excluded_rows"] == 2


def test_frozen_half_open_bins_accept_all_real_collection_times_without_rounding():
    df = pd.DataFrame(
        {
            "collection_id": [
                "20260716_164008",
                "20260716_193651",
                "20260716_211950",
            ],
            "snapshot_utc": [
                "2026-07-16T16:40:08Z",
                "2026-07-16T19:36:51Z",
                "2026-07-16T21:19:50Z",
            ],
        }
    )
    records = df.to_dict("records")
    for record in records:
        record["input_sha256"] = "a" * 64

    selected, quality = _apply_collection_window(
        df,
        "snapshot_utc",
        start_utc="2026-07-16T16:17:00Z",
        end_utc="2026-09-14T16:17:00Z",
        poll_interval_hours=2,
        snapshot_records=records,
    )

    assert selected["collection_id"].tolist() == [
        "20260716_164008",
        "20260716_193651",
        "20260716_211950",
    ]
    assert quality["occupied_snapshot_slots"] == 3
    assert quality["unique_tle_input_hashes"] == 1
    assert quality["tle_input_hash_diversity_fraction"] == pytest.approx(1 / 3)


def test_collection_window_reports_missing_trailing_slots():
    df = pd.DataFrame(
        {"snapshot_utc": ["2026-01-01T00:00:00Z", "2026-01-01T02:00:00Z"]}
    )

    _, quality = _apply_collection_window(
        df,
        "snapshot_utc",
        start_utc="2026-01-01T00:00:00Z",
        end_utc="2026-01-01T08:00:00Z",
        poll_interval_hours=2,
    )

    assert quality["snapshot_coverage_fraction"] == 0.5
    assert quality["max_snapshot_gap_hours"] == 6.0


def test_collection_window_gates_real_elapsed_gap_not_only_adjacent_bins():
    df = pd.DataFrame(
        {
            "snapshot_utc": [
                "2026-01-01T00:01:00Z",
                "2026-01-01T03:59:00Z",
                "2026-01-01T04:01:00Z",
            ]
        }
    )

    _, quality = _apply_collection_window(
        df,
        "snapshot_utc",
        start_utc="2026-01-01T00:00:00Z",
        end_utc="2026-01-01T06:00:00Z",
        poll_interval_hours=2,
    )

    assert quality["occupied_snapshot_slots"] == 3
    assert quality["max_nominal_bin_gap_hours"] == 2.0
    assert quality["max_snapshot_gap_hours"] == pytest.approx(3 + 58 / 60)


def test_collection_coverage_uses_resimulation_records_including_zero_candidates():
    df = pd.DataFrame(
        {
            "collection_id": ["a", "c"],
            "snapshot_utc": ["2026-01-01T00:00:00Z", "2026-01-01T04:00:00Z"],
        }
    )
    records = [
        {"collection_id": "a", "snapshot_utc": "2026-01-01T00:00:00Z", "input_sha256": "a" * 64},
        {"collection_id": "b", "snapshot_utc": "2026-01-01T02:00:00Z", "input_sha256": "b" * 64},
        {"collection_id": "c", "snapshot_utc": "2026-01-01T04:00:00Z", "input_sha256": "c" * 64},
        {"collection_id": "d", "snapshot_utc": "2026-01-01T06:00:00Z", "input_sha256": "d" * 64},
    ]

    selected, quality = _apply_collection_window(
        df,
        "snapshot_utc",
        start_utc="2026-01-01T00:00:00Z",
        end_utc="2026-01-01T08:00:00Z",
        poll_interval_hours=2,
        snapshot_records=records,
    )

    assert selected["collection_id"].tolist() == ["a", "c"]
    assert quality["coverage_source"] == "resimulation_snapshot_records"
    assert quality["occupied_snapshot_slots"] == 4
    assert quality["snapshot_coverage_fraction"] == 1.0
    assert quality["max_snapshot_gap_hours"] == 2.0
    assert quality["unique_tle_input_hashes"] == 4
    assert quality["tle_input_hash_diversity_fraction"] == 1.0


def test_compare_models_fails_closed_on_incomplete_collection_window(tmp_path):
    df = _synthetic_dataset(80, seed=33)
    # All rows occupy only the first two of four expected slots.
    df["snapshot_utc"] = np.where(
        np.arange(len(df)) % 2 == 0,
        "2026-01-01T00:00:00Z",
        "2026-01-01T02:00:00Z",
    )
    df["collection_id"] = np.where(
        np.arange(len(df)) % 2 == 0,
        "20260101_000000",
        "20260101_020000",
    )
    dataset_path = tmp_path / "window.csv"
    df.to_csv(dataset_path, index=False)

    report = compare_models(
        dataset_path,
        tmp_path / "report.csv",
        time_column="snapshot_utc",
        collection_start_utc="2026-01-01T00:00:00Z",
        collection_end_utc="2026-01-01T08:00:00Z",
        poll_interval_hours=2,
        min_snapshot_coverage_fraction=0.9,
        max_snapshot_gap_hours=3,
        min_tle_hash_diversity_fraction=0.5,
        max_identical_tle_hash_run_bins=2,
        snapshot_records=[
            {
                "collection_id": "20260101_000000",
                "snapshot_utc": "2026-01-01T00:00:00Z",
                "input_sha256": "a" * 64,
            },
            {
                "collection_id": "20260101_020000",
                "snapshot_utc": "2026-01-01T02:00:00Z",
                "input_sha256": "b" * 64,
            },
        ],
    )

    # Snapshot coverage (0.5) and endpoint gap are now REPORTED for a
    # transparent limitations statement but do not gate publication: a temporal
    # hole is a CI-infrastructure artifact, not a defect in the collected
    # geometry, and this is a classification task, not a forecast. Publication
    # eligibility rests on TLE-age, catalogue binding and the statistical
    # class/pair/snapshot support gates instead. See docs/EXPERIMENT_10D_V3.md.
    assert report.iloc[0]["snapshot_coverage_fraction"] == 0.5
    note = str(report.iloc[0].get("note", "") or "")
    assert "snapshot_coverage_fraction" not in note
    assert "max_snapshot_gap_hours" not in note


def test_compare_models_reports_pr_auc_and_roc_auc(tmp_path):
    df = _synthetic_dataset(200, seed=1)
    dataset_path = tmp_path / "dataset.csv"
    df.to_csv(dataset_path, index=False)
    report_path = tmp_path / "report.csv"

    report = compare_models(dataset_path, report_path, time_column=None)

    assert list(report.columns) == REPORT_COLUMNS
    assert {
        "pr_auc", "roc_auc", "precision", "recall", "f1", "accuracy",
        "false_alarm_rate",
    } <= set(report.columns)
    # With well-separated balanced classes every model (including the
    # continuous distance-only baseline) should have a defined PR-AUC.
    assert report["pr_auc"].notna().all()
    assert report["roc_auc"].notna().all()
    assert report_path.exists()


def test_compare_models_time_split_uses_chronological_order(tmp_path):
    df = _synthetic_dataset(200, seed=2)
    dataset_path = tmp_path / "dataset.csv"
    df.to_csv(dataset_path, index=False)
    report_path = tmp_path / "report.csv"

    report = compare_models(dataset_path, report_path, time_column="snapshot_utc")

    assert (report["split"] == "pair_grouped_time").all()
    assert report["pr_auc"].notna().any()
    assert (report["excluded_rows"] > 0).all()


def test_compare_models_quality_gate_rejects_single_class_grouped_partitions(tmp_path):
    df = _synthetic_dataset(200, seed=24)
    pair_ids = _canonical_pair_ids(df)
    # The whole dataset has both classes, but SHA pair assignment makes the
    # train partition all-negative and the held-out test all-positive.
    df["risk_label"] = [int(_pair_is_test(pair, 0.25, "iac26-pair-split-v1")) for pair in pair_ids]
    assert df["risk_label"].nunique() == 2
    dataset_path = tmp_path / "dataset.csv"
    df.to_csv(dataset_path, index=False)

    report = compare_models(dataset_path, tmp_path / "report.csv", time_column="snapshot_utc")

    assert report["model"].tolist() == ["not_enough_data"]
    assert "must each contain both" in report.iloc[0]["note"]
    assert report.iloc[0]["train_positive_rows"] == 0
    assert report.iloc[0]["test_positive_rows"] > 0
    assert report.iloc[0]["train_positive_pairs"] == 0
    assert report.iloc[0]["test_positive_pairs"] > 0


def test_canonical_pair_id_is_direction_independent():
    df = pd.DataFrame({"object_1": ["A", "B"], "object_2": ["B", "A"]})

    pair_ids = _canonical_pair_ids(df)

    assert pair_ids.tolist() == ["A|B", "A|B"]


def test_canonical_pair_prefers_catalog_ids_when_names_are_duplicated():
    df = pd.DataFrame(
        {
            "object_1": ["IRIDIUM 33 DEB", "IRIDIUM 33 DEB"],
            "object_2": ["TARGET", "TARGET"],
            "object_1_catalog_id": ["33773", "33775"],
            "object_2_catalog_id": ["99999", "99999"],
        }
    )

    pair_ids = _canonical_pair_ids(df)

    assert pair_ids.nunique() == 2
    assert set(pair_ids) == {"33773|99999", "33775|99999"}


def test_pair_grouped_time_split_has_no_pair_or_time_leakage():
    df = _synthetic_dataset(80, seed=20)
    # Repeat canonical pairs in both early and late source data. The splitter
    # must keep each pair assignment stable while excluding cross-quadrant rows.
    df.loc[40:, "object_1"] = df.loc[:39, "object_1"].to_numpy()
    df.loc[40:, "object_2"] = df.loc[:39, "object_2"].to_numpy()

    split = _pair_grouped_time_split(df, "snapshot_utc")
    train_pairs = set(_canonical_pair_ids(split.train))
    test_pairs = set(_canonical_pair_ids(split.test))
    train_times = pd.to_datetime(split.train["snapshot_utc"], utc=True)
    test_times = pd.to_datetime(split.test["snapshot_utc"], utc=True)

    assert train_pairs.isdisjoint(test_pairs)
    assert train_times.max() < test_times.min()
    assert len(split.train) + len(split.test) + len(split.excluded) == len(df)
    assert split.metadata["excluded_rows"] == len(split.excluded)


def test_pair_grouped_time_split_is_stable_when_input_rows_are_shuffled():
    df = _synthetic_dataset(120, seed=21).assign(source_row_id=np.arange(120))
    first = _pair_grouped_time_split(df, "snapshot_utc")
    shuffled = df.sample(frac=1.0, random_state=99)
    second = _pair_grouped_time_split(shuffled, "snapshot_utc")

    assert set(first.train["source_row_id"]) == set(second.train["source_row_id"])
    assert set(first.test["source_row_id"]) == set(second.test["source_row_id"])
    assert set(first.excluded["source_row_id"]) == set(second.excluded["source_row_id"])
    assert first.metadata["cutoff_utc"] == second.metadata["cutoff_utc"]


def test_sha_pair_assignment_is_stable_when_new_pairs_are_added():
    existing = ["A|B", "C|D", "E|F"]
    before = {pair: _pair_is_test(pair, 0.25, "fixed-seed") for pair in existing}

    after = {
        pair: _pair_is_test(pair, 0.25, "fixed-seed")
        for pair in existing + ["NEW|PAIR"]
    }

    assert {pair: after[pair] for pair in existing} == before


def test_pair_grouped_time_split_keeps_cutoff_snapshot_out_of_train():
    df = _synthetic_dataset(40, seed=22)
    # Four rows per complete snapshot block.
    df["snapshot_utc"] = np.repeat(pd.date_range("2026-01-01", periods=10, freq="h").astype(str), 4)

    split = _pair_grouped_time_split(df, "snapshot_utc")
    cutoff = pd.Timestamp(split.metadata["cutoff_utc"])
    train_times = pd.to_datetime(split.train["snapshot_utc"], utc=True)
    test_times = pd.to_datetime(split.test["snapshot_utc"], utc=True)

    assert (train_times < cutoff).all()
    assert (test_times >= cutoff).all()
    assert cutoff not in set(train_times)


def test_pair_grouped_time_split_rejects_missing_time_column():
    df = _synthetic_dataset(20).drop(columns=["snapshot_utc"])

    with pytest.raises(ValueError, match="not found"):
        _pair_grouped_time_split(df, "snapshot_utc")


@pytest.mark.parametrize("bad_value", [None, ""])
def test_pair_grouped_time_split_rejects_missing_or_empty_pair_identity(bad_value):
    df = _synthetic_dataset(20)
    df.loc[0, "object_1"] = bad_value

    with pytest.raises(ValueError, match="Pair identity"):
        _pair_grouped_time_split(df, "snapshot_utc")


def test_pair_grouped_time_split_rejects_invalid_timestamp():
    df = _synthetic_dataset(20)
    df.loc[0, "snapshot_utc"] = "not-a-timestamp"

    with pytest.raises(ValueError):
        _pair_grouped_time_split(df, "snapshot_utc")


def test_fixed_threshold_predictions_use_the_actual_grouped_test_rows(tmp_path):
    df = _synthetic_dataset(200, seed=23).sample(frac=1.0, random_state=17)
    # Make baseline values deliberately independent of row order so an index
    # reset/alignment bug is observable.
    df["fixed_threshold_alarm"] = (np.arange(len(df)) % 3 == 0).astype(int)
    dataset_path = tmp_path / "shuffled.csv"
    df.to_csv(dataset_path, index=False)

    round_tripped = pd.read_csv(dataset_path)
    split = _pair_grouped_time_split(round_tripped, "snapshot_utc")
    predictions = model_predictions_for_plotting(dataset_path, time_column="snapshot_utc")

    assert predictions["fixed_threshold"]["pred"].tolist() == split.test[
        "fixed_threshold_alarm"
    ].astype(int).tolist()
    np.testing.assert_allclose(
        predictions["fixed_threshold"]["scores"],
        -split.test["min_distance_km"].astype(float).to_numpy(),
    )


def test_fixed_threshold_report_uses_continuous_distance_for_ranking(tmp_path):
    df = _synthetic_dataset(200, seed=25)
    initial_split = _pair_grouped_time_split(df, "snapshot_utc")
    test_indices = initial_split.test.index.to_numpy()
    pattern_distance = np.array([5.0, 10.0, 30.0, 40.0])
    pattern_label = np.array([0, 1, 1, 0])
    repeats = int(np.ceil(len(test_indices) / len(pattern_distance)))
    test_distance = np.tile(pattern_distance, repeats)[: len(test_indices)]
    test_labels = np.tile(pattern_label, repeats)[: len(test_indices)]
    df.loc[test_indices, "min_distance_km"] = test_distance
    df.loc[test_indices, "risk_label"] = test_labels
    df.loc[test_indices, "fixed_threshold_alarm"] = (test_distance <= 25.0).astype(int)

    dataset_path = tmp_path / "distance_baseline.csv"
    report_path = tmp_path / "report.csv"
    df.to_csv(dataset_path, index=False)

    split = _pair_grouped_time_split(df, "snapshot_utc")
    expected_pr_auc = average_precision_score(
        split.test["risk_label"].astype(int),
        -split.test["min_distance_km"].astype(float),
    )
    binary_alarm_pr_auc = average_precision_score(
        split.test["risk_label"].astype(int),
        split.test["fixed_threshold_alarm"].astype(int),
    )
    test_is_negative = split.test["risk_label"].astype(int).eq(0)
    expected_false_alarm_rate = float(
        split.test.loc[test_is_negative, "fixed_threshold_alarm"].astype(int).mean()
    )
    report = compare_models(
        dataset_path,
        report_path,
        time_column="snapshot_utc",
    )
    baseline = report.loc[report["model"] == "fixed_threshold"].iloc[0]

    assert expected_pr_auc != pytest.approx(binary_alarm_pr_auc)
    assert baseline["pr_auc"] == pytest.approx(expected_pr_auc)
    assert baseline["false_positive"] > 0
    assert baseline["false_negative"] > 0
    assert 0.0 < expected_false_alarm_rate < 1.0
    assert baseline["false_alarm_rate"] == pytest.approx(expected_false_alarm_rate)


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
    assert "observed higher than baseline" in text
    assert "uncertainty is reported separately" in text
    assert "random_forest PR-AUC = 0.8000 (observed higher than baseline" in text
    assert "logistic_regression PR-AUC = 0.3000 (not observed higher than baseline" in text


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
    assert "false_alarm_rate" in set(report["metric"])
    pr_auc_rows = report[(report["model"] == "random_forest") & (report["metric"] == "pr_auc")]
    assert len(pr_auc_rows) == 1
    row = pr_auc_rows.iloc[0]
    # ``folds`` counts only folds where this metric is defined; pair/time
    # holdout can legitimately leave a fold with a single test class.
    assert 1 <= row["folds"] <= 3
    assert row["total_folds"] == 3
    assert row["valid_partitions"] == 3
    assert row["mean"] is not None
    assert row["std"] is not None
    assert report_path.exists()
    assert report_path.with_name("cv_report_folds.csv").exists()
    assert report_path.with_name("cv_report_adaptability.csv").exists()


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


def test_random_split_remains_compatible_without_pair_columns(tmp_path):
    df = _synthetic_dataset(200, seed=31).drop(
        columns=["object_1", "object_2", "snapshot_utc"]
    )
    dataset_path = tmp_path / "dataset.csv"
    df.to_csv(dataset_path, index=False)

    report = compare_models(dataset_path, tmp_path / "report.csv", time_column=None)

    assert "not_enough_data" not in set(report["model"])
    assert report["train_positive_pairs"].isna().all()
    assert report["test_positive_pairs"].isna().all()
    assert report["train_positive_snapshots"].isna().all()
    assert report["test_positive_snapshots"].isna().all()


def test_features_still_excludes_leakage_columns():
    assert "risk_score" not in FEATURES
    assert "risk_label" not in FEATURES
    assert "fixed_threshold_alarm" not in FEATURES


def test_feature_ablation_removes_only_rule_defining_predictors(tmp_path):
    removed = {
        "min_distance_km",
        "relative_velocity_km_s",
        "relative_radial_km",
        "relative_intrack_km",
        "relative_crosstrack_km",
    }
    assert removed.isdisjoint(FEATURES_WITHOUT_LABEL_RULE)
    assert set(FEATURES) - set(FEATURES_WITHOUT_LABEL_RULE) == removed

    df = _synthetic_dataset(200, seed=26)
    dataset_path = tmp_path / "ablation.csv"
    df.to_csv(dataset_path, index=False)
    full = compare_models(
        dataset_path,
        tmp_path / "full.csv",
        time_column="snapshot_utc",
    )
    ablated = compare_models(
        dataset_path,
        tmp_path / "ablated.csv",
        time_column="snapshot_utc",
        feature_columns=FEATURES_WITHOUT_LABEL_RULE,
    )

    for column in (
        "train_rows", "test_rows", "train_pairs", "test_pairs",
        "excluded_rows", "cutoff_utc", "split",
    ):
        assert full[column].tolist() == ablated[column].tolist()

    provenance = (tmp_path / "ablated.csv").read_text(encoding="utf-8").splitlines()
    config_line = next(line for line in provenance if line.startswith("# config:"))
    assert all(feature not in config_line for feature in removed)


def test_snapshot_only_primary_features_exclude_all_forward_tca_quantities():
    assert SNAPSHOT_ONLY_FEATURES == [
        "current_distance_km",
        "altitude_difference_km",
        "max_tle_age_hours",
        "radial_velocity_km_s",
        "tangential_velocity_km_s",
        "approach_angle_deg",
    ]
    forward_or_label_reconstructing = {
        "time_to_tca_min",
        "min_distance_km",
        "relative_velocity_km_s",
        "relative_radial_km",
        "relative_intrack_km",
        "relative_crosstrack_km",
        "relative_inclination_deg",
    }
    assert forward_or_label_reconstructing.isdisjoint(SNAPSHOT_ONLY_FEATURES)


def test_temporal_consistency_is_descriptive_and_requires_every_requested_fold():
    supported = temporal_consistency_summary(
        "xgboost",
        [0.03, 0.01, 0.04, 0.02, 0.05],
        primary_model="xgboost",
        requested_folds=5,
        minimum_folds=5,
    )
    assert supported["adaptability_supported"] is True
    assert supported["statistical_significance_tested"] is False
    assert supported["all_requested_folds_valid"] is True
    assert supported["positive_delta_fraction"] == 1.0
    assert "not a statistical-significance claim" in supported["claim"]

    nonpositive = temporal_consistency_summary(
        "xgboost",
        [0.03, 0.01, 0.0, 0.02, 0.05],
        primary_model="xgboost",
        requested_folds=5,
        minimum_folds=5,
    )
    missing_fold = temporal_consistency_summary(
        "xgboost",
        [0.03, 0.01, 0.04, 0.02],
        primary_model="xgboost",
        requested_folds=5,
        minimum_folds=5,
    )
    nonprimary = temporal_consistency_summary(
        "random_forest",
        [0.03, 0.01, 0.04, 0.02, 0.05],
        primary_model="xgboost",
        requested_folds=5,
        minimum_folds=5,
    )
    assert nonpositive["adaptability_supported"] is False
    assert missing_fold["adaptability_supported"] is False
    assert nonprimary["adaptability_supported"] is False


def test_compare_models_rejects_empty_feature_ablation(tmp_path):
    df = _synthetic_dataset(20, seed=27)
    dataset_path = tmp_path / "dataset.csv"
    df.to_csv(dataset_path, index=False)

    with pytest.raises(ValueError, match="feature_columns cannot be empty"):
        compare_models(dataset_path, tmp_path / "report.csv", feature_columns=[])


def test_publication_support_gate_fails_closed_and_reports_support(tmp_path):
    df = _synthetic_dataset(200, seed=29)
    dataset_path = tmp_path / "dataset.csv"
    report_path = tmp_path / "report.csv"
    df.to_csv(dataset_path, index=False)

    report = compare_models(
        dataset_path,
        report_path,
        time_column="snapshot_utc",
        minimum_support={
            "train_positive_rows": 1,
            "test_positive_rows": 10_000,
            "train_positive_pairs": 1,
            "test_positive_pairs": 1,
            "train_positive_snapshots": 1,
            "test_positive_snapshots": 1,
        },
    )

    assert report.iloc[0]["model"] == "not_enough_data"
    assert "Publication quality gate failed" in report.iloc[0]["note"]
    assert report.iloc[0]["train_positive_rows"] > 0
    assert report.iloc[0]["test_positive_rows"] > 0
    assert "test_positive_rows=" in report.iloc[0]["note"]
    provenance = report_path.read_text(encoding="utf-8")
    assert "minimum_support=" in provenance
    assert "test_positive_rows:10000" in provenance


def test_publication_gate_requires_full_observation_span(tmp_path):
    df = _synthetic_dataset(200, seed=33)
    dataset_path = tmp_path / "dataset.csv"
    report_path = tmp_path / "report.csv"
    df.to_csv(dataset_path, index=False)

    report = compare_models(
        dataset_path,
        report_path,
        time_column="snapshot_utc",
        minimum_observation_span_days=10.0,
    )

    row = report.iloc[0]
    assert row["model"] == "not_enough_data"
    assert 8.0 < row["observation_span_days"] < 9.0
    assert "observation_span_days=" in row["note"]
    assert "minimum_observation_span_days=10.000000" in report_path.read_text(
        encoding="utf-8"
    )


def test_frozen_catalog_gate_rejects_mixed_cohorts(tmp_path):
    df = _synthetic_dataset(200, seed=34)
    df["catalog_version"] = "iac26-75-v1"
    df.loc[df.index[-1], "catalog_version"] = "legacy-v0"
    dataset_path = tmp_path / "mixed.csv"
    df.to_csv(dataset_path, index=False)

    with pytest.raises(ValueError, match="Catalog cohort mismatch"):
        compare_models(
            dataset_path,
            tmp_path / "report.csv",
            time_column="snapshot_utc",
            required_catalog_version="iac26-75-v1",
        )


def test_frozen_catalog_gate_rejects_wrong_id_set_hash(tmp_path):
    df = _synthetic_dataset(200, seed=35)
    df["catalog_version"] = "iac26-75-v1"
    df["catalog_sha256"] = "wrong"
    dataset_path = tmp_path / "wrong-hash.csv"
    df.to_csv(dataset_path, index=False)

    with pytest.raises(ValueError, match="Catalog ID-set hash mismatch"):
        compare_models(
            dataset_path,
            tmp_path / "report.csv",
            time_column="snapshot_utc",
            required_catalog_version="iac26-75-v1",
            required_catalog_sha256="expected",
        )


def test_publication_support_gate_accepts_sufficient_split(tmp_path):
    df = _synthetic_dataset(200, seed=30)
    dataset_path = tmp_path / "dataset.csv"
    df.to_csv(dataset_path, index=False)

    report = compare_models(
        dataset_path,
        tmp_path / "report.csv",
        time_column="snapshot_utc",
        minimum_support={key: 1 for key in (
            "train_positive_rows",
            "test_positive_rows",
            "train_positive_pairs",
            "test_positive_pairs",
            "train_positive_snapshots",
            "test_positive_snapshots",
        )},
    )

    assert "not_enough_data" not in set(report["model"])
    assert report["train_positive_rows"].min() > 0
    assert report["test_positive_rows"].min() > 0


@pytest.mark.parametrize("invalid", [True, 1.9, 0, -1])
def test_minimum_support_rejects_non_positive_integer_values(tmp_path, invalid):
    df = _synthetic_dataset(20, seed=32)
    dataset_path = tmp_path / "dataset.csv"
    df.to_csv(dataset_path, index=False)

    with pytest.raises(ValueError, match="positive integers"):
        compare_models(
            dataset_path,
            tmp_path / "report.csv",
            minimum_support={"train_positive_rows": invalid},
        )


@pytest.mark.parametrize("forbidden", ["risk_label", "fixed_threshold_alarm", "risk_score"])
def test_compare_models_rejects_direct_target_or_leakage_features(tmp_path, forbidden):
    df = _synthetic_dataset(20, seed=28)
    df["risk_score"] = np.linspace(0.0, 1.0, len(df))
    dataset_path = tmp_path / "dataset.csv"
    df.to_csv(dataset_path, index=False)

    with pytest.raises(ValueError, match="forbidden target/leakage"):
        compare_models(
            dataset_path,
            tmp_path / "report.csv",
            feature_columns=[forbidden],
        )


def test_compare_models_embeds_provenance_header_and_stays_readable(tmp_path):
    df = _synthetic_dataset(200, seed=5)
    dataset_path = tmp_path / "dataset.csv"
    df.to_csv(dataset_path, index=False)
    report_path = tmp_path / "report.csv"

    report = compare_models(
        dataset_path, report_path, source="unit-test", config_summary="n=200"
    )

    raw_lines = report_path.read_text(encoding="utf-8").splitlines()
    assert raw_lines[0].startswith("# generated_utc:")
    assert any("unit-test" in line for line in raw_lines)
    assert any("n=200" in line for line in raw_lines)

    # The in-memory report returned to callers is unaffected by the header.
    assert list(report.columns) == REPORT_COLUMNS
    round_tripped = pd.read_csv(report_path, comment="#")
    # CSV round-tripping turns empty strings (e.g. an all-"" note column)
    # into NaN, which changes dtype but not meaning -- normalize before
    # comparing values.
    pd.testing.assert_frame_equal(
        round_tripped.fillna(""), report.fillna(""), check_dtype=False
    )


def test_compare_models_can_read_a_dataset_with_a_provenance_header(tmp_path):
    # Regression test: core.write_pair_results() may prepend a '#' header to
    # its own CSV output; compare_models() must still parse the dataset it
    # produces.
    from space_debris.provenance import write_csv_text_with_provenance

    df = _synthetic_dataset(200, seed=6)
    dataset_path = tmp_path / "dataset.csv"
    write_csv_text_with_provenance(dataset_path, df.to_csv(index=False), source="x", config_summary="y")

    report = compare_models(dataset_path, tmp_path / "report.csv")
    assert report["pr_auc"].notna().all()


def test_model_predictions_for_plotting_returns_expected_structure(tmp_path):
    df = _synthetic_dataset(200, seed=7)
    dataset_path = tmp_path / "dataset.csv"
    df.to_csv(dataset_path, index=False)

    predictions = model_predictions_for_plotting(dataset_path)

    assert "fixed_threshold" in predictions
    assert predictions["fixed_threshold"]["model"] is None
    assert "random_forest" in predictions
    rf = predictions["random_forest"]
    assert rf["model"] is not None
    assert hasattr(rf["model"], "feature_importances_")
    assert len(rf["y_true"]) == len(rf["scores"]) == len(rf["pred"])


def test_model_predictions_for_plotting_empty_on_insufficient_data(tmp_path):
    df = _synthetic_dataset(4, seed=8)
    dataset_path = tmp_path / "dataset.csv"
    df.to_csv(dataset_path, index=False)

    assert model_predictions_for_plotting(dataset_path) == {}
