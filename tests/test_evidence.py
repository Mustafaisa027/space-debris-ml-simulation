from __future__ import annotations

import json
import hashlib
import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from space_debris import evidence
from space_debris.evidence import (
    CALIBRATED_DISTANCE_BASELINE,
    EvidenceGenerationError,
    _constant_velocity_cpa_score,
    adaptability_inference_from_predictions,
    _validate_frozen_adaptability_protocol,
    calibrate_distance_baseline,
    canonical_report_from_evidence,
    generate_evaluation_evidence,
    paired_pair_cluster_bootstrap,
    select_threshold_for_target_recall,
)
from space_debris.experiment import load_experiment_config
from space_debris.ml import REPORT_COLUMNS, SNAPSHOT_ONLY_FEATURES


def _snapshot_records(frame: pd.DataFrame) -> list[dict[str, str]]:
    records = frame[["collection_id", "snapshot_utc"]].drop_duplicates().to_dict("records")
    for record in records:
        record["input_sha256"] = hashlib.sha256(
            record["collection_id"].encode("utf-8")
        ).hexdigest()
    return records


def test_constant_velocity_cpa_uses_only_snapshot_state_and_forward_horizon():
    frame = pd.DataFrame(
        {
            "current_distance_km": [10.0, 10.0, 10.0],
            "radial_velocity_km_s": [-1.0, 1.0, -1.0],
            "tangential_velocity_km_s": [0.0, 0.0, 1.0],
        }
    )

    scores = _constant_velocity_cpa_score(frame, horizon_minutes=1.0)

    assert scores[0] == pytest.approx(0.0)
    assert scores[1] == pytest.approx(-10.0)
    assert scores[2] == pytest.approx(-10.0 / np.sqrt(2.0))


@pytest.mark.parametrize(
    ("column", "value", "horizon"),
    [
        ("current_distance_km", -1.0, 1.0),
        ("tangential_velocity_km_s", -1.0, 1.0),
        ("current_distance_km", 10.0, 0.0),
        ("current_distance_km", 10.0, float("nan")),
    ],
)
def test_constant_velocity_cpa_rejects_nonphysical_domain(column, value, horizon):
    frame = pd.DataFrame(
        {
            "current_distance_km": [10.0],
            "radial_velocity_km_s": [-1.0],
            "tangential_velocity_km_s": [0.0],
        }
    )
    frame.loc[0, column] = value

    with pytest.raises(EvidenceGenerationError, match="physically valid"):
        _constant_velocity_cpa_score(frame, horizon_minutes=horizon)


def _evidence_dataset(config, snapshots: int = 12, pairs: int = 40) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    for snapshot in range(snapshots):
        timestamp = start + pd.Timedelta(hours=2 * snapshot)
        for pair in range(pairs):
            positive = int((snapshot + pair) % 4 == 0)
            # Half of the proxy positives sit inside the 25 km baseline and
            # half between 25 and 50 km, giving non-zero validation recall.
            min_distance = (10.0 if (snapshot + pair) % 8 == 0 else 40.0) if positive else 80.0
            velocity = 12.0 if positive else 5.0
            row = {
                "collection_id": timestamp.strftime("%Y%m%d_%H%M%S"),
                "catalog_version": config.catalog_version,
                "catalog_sha256": config.catalog_sha256,
                "snapshot_utc": timestamp.isoformat().replace("+00:00", "Z"),
                "object_1": f"SAT-{pair:03d}",
                "object_2": f"DEB-{pair:03d}",
                "object_1_catalog_id": str(10000 + pair),
                "object_2_catalog_id": str(20000 + pair),
                "risk_label": positive,
                "fixed_threshold_alarm": int(min_distance <= 25.0),
                "min_distance_km": min_distance,
                "relative_velocity_km_s": velocity,
                "time_to_tca_min": float(snapshot * 10 + pair),
                "current_distance_km": min_distance + 5.0,
                "altitude_difference_km": float(pair % 20),
                "max_tle_age_hours": float(snapshot),
                "relative_radial_km": min_distance,
                "relative_intrack_km": 0.0,
                "relative_crosstrack_km": 0.0,
                "relative_inclination_deg": float(pair % 90),
                "radial_velocity_km_s": velocity / 2,
                "tangential_velocity_km_s": velocity,
                "approach_angle_deg": 45.0,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def test_threshold_selection_uses_highest_threshold_meeting_recall():
    selected = select_threshold_for_target_recall(
        [1, 1, 0, 0], [0.9, 0.6, 0.7, 0.1], target_recall=0.5
    )

    assert selected["threshold"] == 0.9
    assert selected["validation_recall"] == 0.5
    assert selected["validation_proxy_false_alarm_rate"] == 0.0


def test_threshold_selection_rejects_zero_baseline_recall():
    with pytest.raises(EvidenceGenerationError, match="must be in"):
        select_threshold_for_target_recall([1, 0], [0.8, 0.2], target_recall=0.0)


def test_calibrated_distance_threshold_cannot_see_held_out_labels():
    validation_y = [1, 1, 0, 0]
    validation_scores = [-10.0, -20.0, -15.0, -80.0]

    first_info, first_predictions = calibrate_distance_baseline(
        validation_y, validation_scores, [-5.0, -30.0], target_recall=0.5
    )
    second_info, second_predictions = calibrate_distance_baseline(
        validation_y, validation_scores, [-100.0, -1.0], target_recall=0.5
    )

    assert first_info == second_info
    assert first_info["threshold"] == -10.0
    assert first_predictions.tolist() == [1, 0]
    assert second_predictions.tolist() == [0, 1]


def test_canonical_report_metrics_come_from_frozen_evidence():
    gate_row = {column: None for column in REPORT_COLUMNS}
    gate_row.update(
        model="model",
        precision=0.1,
        recall=0.2,
        f1=0.1,
        false_alarm_rate=0.9,
        train_rows=100,
        test_rows=20,
        split="pair_grouped_time",
    )
    frozen = pd.DataFrame(
        [
            {
                "model": "model",
                "operating_threshold": 0.73,
                "test_pr_auc": 0.8,
                "test_roc_auc": 0.9,
                "test_precision": 0.7,
                "test_recall": 0.6,
                "test_f1": 0.65,
                "test_accuracy": 0.85,
                "test_proxy_false_alarm_rate": 0.1,
                "test_false_positive": 2,
                "test_false_negative": 4,
                "test_true_positive": 6,
                "test_true_negative": 18,
                "fitted_train_rows": 60,
                "fitted_train_pairs": 12,
                "fitted_train_positive_rows": 15,
                "fitted_train_positive_pairs": 5,
                "fitted_train_positive_snapshots": 4,
                "fitted_train_cutoff_utc": "2026-01-10T00:00:00Z",
                "calibration_rows": 20,
                "calibration_pairs": 4,
                "calibration_positive_rows": 5,
                "calibration_positive_pairs": 2,
                "calibration_positive_snapshots": 2,
            }
        ]
    )
    frozen = pd.concat(
        [
            frozen,
            frozen.assign(model="constant_velocity_cpa"),
            frozen.assign(model="label_permutation_control"),
            frozen.assign(model="proxy_rule_oracle"),
        ],
        ignore_index=True,
    )

    report = canonical_report_from_evidence(pd.DataFrame([gate_row]), frozen)

    assert report.iloc[0]["precision"] == 0.7
    assert report["model"].tolist() == ["model"]
    assert report.iloc[0]["recall"] == 0.6
    assert report.iloc[0]["false_alarm_rate"] == 0.1
    assert report.iloc[0]["train_rows"] == 100
    assert report.iloc[0]["estimator_train_rows"] == 60
    assert report.iloc[0]["calibration_rows"] == 20
    assert "held_out_predictions.csv" in report.iloc[0]["note"]


def test_paired_pair_cluster_bootstrap_supports_dominating_model():
    rows = []
    for pair in range(12):
        for day in range(6):
            for y_true in (1, 0, 0):
                row_id = f"{pair}-{day}-{y_true}-{len(rows)}"
                common = {
                    "source_row_id": row_id,
                    "canonical_pair_id": f"PAIR-{pair}",
                    "snapshot_utc": f"2026-01-{day + 1:02d}T00:00:00Z",
                    "y_true": y_true,
                    "operating_threshold": 0.5,
                    "calibration_target_recall": 1.0,
                }
                rows.append(
                    {**common, "model": "fixed_threshold", "score": float(y_true), "pred": 1}
                )
                rows.append(
                    {
                        **common,
                        "model": CALIBRATED_DISTANCE_BASELINE,
                        "score": float(y_true),
                        "pred": 1,
                    }
                )
                rows.append(
                    {
                        **common,
                        "model": "constant_velocity_cpa",
                        "score": float(y_true),
                        "pred": 1,
                    }
                )
                rows.append(
                    {**common, "model": "model", "score": float(y_true), "pred": y_true}
                )
    predictions = pd.DataFrame(rows)

    report = paired_pair_cluster_bootstrap(
        predictions,
        replicates=100,
        seed=114764,
        confidence_level=0.95,
        recall_noninferiority_margin=0.05,
        min_valid_fraction=0.9,
        primary_model="model",
    )

    model = report.loc[report["model"].eq("model")].iloc[0]
    assert model["delta_proxy_false_alarm_rate"] == -1.0
    assert model["delta_proxy_false_alarm_rate_ci_high"] < 0
    assert model["delta_recall_ci_low"] == 0.0
    assert model["time_block_delta_proxy_false_alarm_rate_ci_high"] < 0
    assert model["time_block_delta_recall_ci_low"] == 0.0
    assert bool(model["time_block_gate_passed"])
    assert bool(model["calibrated_distance_pair_gate_passed"])
    assert bool(model["calibrated_distance_time_gate_passed"])
    assert model["delta_vs_constant_velocity_cpa_proxy_false_alarm_rate"] == -1.0
    assert model["delta_vs_constant_velocity_cpa_proxy_false_alarm_rate_ci_high"] < 0
    assert (
        model[
            "time_block_delta_vs_constant_velocity_cpa_proxy_false_alarm_rate_ci_high"
        ]
        < 0
    )
    assert bool(model["constant_velocity_cpa_pair_sensitivity_passed"])
    assert bool(model["constant_velocity_cpa_time_sensitivity_passed"])
    assert bool(model["false_alarm_reduction_supported"])


def test_cpa_sensitivity_is_separate_from_fixed_threshold_primary_claim():
    rows = []
    for pair in range(8):
        for day in range(5):
            for y_true in (1, 0, 0):
                common = {
                    "source_row_id": f"{pair}-{day}-{y_true}-{len(rows)}",
                    "canonical_pair_id": f"PAIR-{pair}",
                    "snapshot_utc": f"2026-02-{day + 1:02d}T00:00:00Z",
                    "y_true": y_true,
                    "operating_threshold": 0.5,
                    "calibration_target_recall": 1.0,
                }
                rows.append(
                    {**common, "model": "fixed_threshold", "score": float(y_true), "pred": 1}
                )
                rows.append(
                    {
                        **common,
                        "model": CALIBRATED_DISTANCE_BASELINE,
                        "score": float(y_true),
                        "pred": 1,
                    }
                )
                rows.append(
                    {
                        **common,
                        "model": "constant_velocity_cpa",
                        "score": float(y_true),
                        "pred": y_true,
                    }
                )
                model_pred = y_true if y_true else int(pair % 2 == 0)
                rows.append(
                    {
                        **common,
                        "model": "model",
                        "score": float(y_true),
                        "pred": model_pred,
                    }
                )

    report = paired_pair_cluster_bootstrap(
        pd.DataFrame(rows),
        replicates=200,
        seed=114764,
        confidence_level=0.95,
        recall_noninferiority_margin=0.05,
        min_valid_fraction=0.9,
        primary_model="model",
    )
    model = report.loc[report["model"].eq("model")].iloc[0]

    assert bool(model["false_alarm_reduction_supported"])
    assert not bool(model["constant_velocity_cpa_pair_sensitivity_passed"])
    assert not bool(model["constant_velocity_cpa_time_sensitivity_passed"])


def test_calibrated_distance_reference_can_veto_fixed_threshold_claim():
    rows = []
    for pair in range(10):
        for day in range(5):
            for y_true in (1, 0, 0):
                common = {
                    "source_row_id": f"{pair}-{day}-{y_true}-{len(rows)}",
                    "canonical_pair_id": f"PAIR-{pair}",
                    "snapshot_utc": f"2026-03-{day + 1:02d}T00:00:00Z",
                    "y_true": y_true,
                    "operating_threshold": 0.5,
                    "calibration_target_recall": 1.0,
                }
                rows.append(
                    {**common, "model": "fixed_threshold", "score": float(y_true), "pred": 1}
                )
                rows.append(
                    {
                        **common,
                        "model": CALIBRATED_DISTANCE_BASELINE,
                        "score": float(y_true),
                        "pred": y_true,
                    }
                )
                model_pred = y_true if y_true else int(pair % 2 == 0)
                rows.append(
                    {**common, "model": "model", "score": float(y_true), "pred": model_pred}
                )

    report = paired_pair_cluster_bootstrap(
        pd.DataFrame(rows),
        replicates=200,
        seed=114764,
        confidence_level=0.95,
        recall_noninferiority_margin=0.05,
        min_valid_fraction=0.9,
        primary_model="model",
    )
    model = report.loc[report["model"].eq("model")].iloc[0]

    assert bool(model["pair_cluster_gate_passed"])
    assert bool(model["time_block_gate_passed"])
    assert not bool(model["calibrated_distance_pair_gate_passed"])
    assert not bool(model["false_alarm_reduction_supported"])


def _adaptability_predictions(blocks: int) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    for block in range(blocks):
        block_start = start + pd.Timedelta(hours=50 * block)
        for pair in range(40):
            y_true = int(pair < 10)
            timestamp = block_start + pd.Timedelta(hours=24 * (pair % 2))
            common = {
                "source_row_id": f"block-{block}-pair-{pair}",
                "snapshot_utc": timestamp.isoformat().replace("+00:00", "Z"),
                "canonical_pair_id": f"PAIR-{pair}",
                "object_1_catalog_id": str(10000 + pair),
                "object_2_catalog_id": str(10000 + ((pair + 1) % 40)),
                "y_true": y_true,
            }
            rows.append(
                {
                    **common,
                    "model": "xgboost",
                    "score": float(y_true),
                }
            )
            rows.append(
                {
                    **common,
                    "model": CALIBRATED_DISTANCE_BASELINE,
                    "score": 0.5,
                }
            )
    return pd.DataFrame(rows)


def test_adaptability_inference_uses_object_and_future_block_bootstrap():
    config = replace(
        load_experiment_config("config/experiment_60_days.json"),
        adaptability_bootstrap_replicates=200,
        min_valid_bootstrap_fraction=0.8,
    )

    report = adaptability_inference_from_predictions(
        _adaptability_predictions(10),
        config,
        split_cutoff_utc="2026-01-01T00:00:00Z",
        block_anchor_utc="2026-01-01T00:00:00Z",
        analysis_end_utc="2026-01-21T20:00:00Z",
        enforce_frozen_protocol=False,
    )

    assert report["status"] == "supported"
    assert report["statistical_significance_tested"] is True
    assert report["statistical_adaptability_supported"] is True
    assert report["support"]["valid_future_blocks"] == 10
    assert report["support"]["unique_objects"] == 40
    assert report["one_sided_confidence_lower_bound"] > 0
    assert report["centered_bootstrap_one_sided_p_value"] < 0.05
    assert report["all_valid_future_blocks_positive"] is True


def test_adaptability_inference_fails_closed_with_too_few_future_blocks():
    config = replace(
        load_experiment_config("config/experiment_60_days.json"),
        adaptability_bootstrap_replicates=20,
    )

    report = adaptability_inference_from_predictions(
        _adaptability_predictions(3),
        config,
        split_cutoff_utc="2026-01-01T00:00:00Z",
        block_anchor_utc="2026-01-01T00:00:00Z",
        analysis_end_utc="2026-01-07T06:00:00Z",
        enforce_frozen_protocol=False,
    )

    assert report["status"] == "not_enough_data"
    assert report["statistical_significance_tested"] is False
    assert report["statistical_adaptability_supported"] is False
    assert "future blocks" in report["reason"]


def test_adaptability_treats_initial_outer_test_rows_as_embargo_not_split_violation():
    config = replace(
        load_experiment_config("config/experiment_60_days.json"),
        adaptability_bootstrap_replicates=20,
    )
    predictions = _adaptability_predictions(3)
    embargo_rows = predictions.loc[
        predictions["source_row_id"].eq("block-0-pair-0")
    ].copy()
    embargo_rows["source_row_id"] = "initial-embargo-pair-0"
    embargo_rows["snapshot_utc"] = "2025-12-31T22:00:00Z"
    predictions = pd.concat([embargo_rows, predictions], ignore_index=True)

    report = adaptability_inference_from_predictions(
        predictions,
        config,
        split_cutoff_utc="2025-12-31T22:00:00Z",
        block_anchor_utc="2026-01-01T00:00:00Z",
        analysis_end_utc="2026-01-07T06:00:00Z",
        enforce_frozen_protocol=False,
    )

    assert report["initial_embargo_rows_excluded"] == 1
    assert report["support"]["rows"] == 120

    before_cutoff = predictions.copy()
    before_cutoff.loc[
        before_cutoff["source_row_id"].eq("initial-embargo-pair-0"), "snapshot_utc"
    ] = "2025-12-31T21:59:59Z"
    with pytest.raises(EvidenceGenerationError, match="outer-test cutoff"):
        adaptability_inference_from_predictions(
            before_cutoff,
            config,
            split_cutoff_utc="2025-12-31T22:00:00Z",
            block_anchor_utc="2026-01-01T00:00:00Z",
            analysis_end_utc="2026-01-07T06:00:00Z",
            enforce_frozen_protocol=False,
        )


def test_adaptability_inference_rejects_weakened_claim_protocol():
    config = replace(
        load_experiment_config("config/experiment_60_days.json"),
        adaptability_min_blocks=1,
    )

    with pytest.raises(EvidenceGenerationError, match="Frozen adaptability protocol"):
        adaptability_inference_from_predictions(
            _adaptability_predictions(1),
            config,
            split_cutoff_utc="2026-01-01T00:00:00Z",
            block_anchor_utc="2026-01-01T00:00:00Z",
            analysis_end_utc="2026-01-03T02:00:00Z",
        )


def test_v2_adaptability_protocol_is_registered_before_collection():
    config = load_experiment_config("config/experiment_10_days_v2.json")

    _validate_frozen_adaptability_protocol(config)


def test_complete_v2_window_can_supply_ten_frozen_adaptability_blocks():
    config = load_experiment_config("config/experiment_10_days_v2.json")
    start = pd.Timestamp(config.collection_start_utc)
    end = pd.Timestamp(config.collection_end_utc)
    expected_slots = int(
        (end - start).total_seconds() / (config.poll_interval_hours * 3600.0)
    )
    cutoff_index = math.ceil(expected_slots * config.train_time_fraction)
    cutoff = start + pd.Timedelta(hours=cutoff_index * config.poll_interval_hours)
    anchor = cutoff + pd.Timedelta(hours=config.poll_interval_hours)
    cycle_hours = (
        config.adaptability_block_hours + config.adaptability_embargo_hours
    )
    complete_blocks = (
        math.floor(
            (
                (end - anchor).total_seconds() / 3600.0
                - config.adaptability_block_hours
            )
            / cycle_hours
        )
        + 1
    )

    assert expected_slots == 120
    assert cutoff_index == 48
    assert complete_blocks == config.adaptability_min_blocks == 10


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("train_time_fraction", 0.50),
        ("adaptability_block_hours", 48.0),
        ("adaptability_min_positive_days_per_block", 2),
        ("bootstrap_seed", 114764),
        ("adaptability_min_blocks", 9),
    ],
)
def test_v2_adaptability_protocol_rejects_any_v1_or_weakened_value(field, value):
    config = replace(
        load_experiment_config("config/experiment_10_days_v2.json"),
        **{field: value},
    )

    with pytest.raises(EvidenceGenerationError, match=f"{field} =="):
        _validate_frozen_adaptability_protocol(config)


def test_adaptability_protocol_rejects_unregistered_experiment_id():
    config = replace(
        load_experiment_config("config/experiment_10_days_v2.json"),
        experiment_id="unregistered-follow-up",
    )

    with pytest.raises(EvidenceGenerationError, match="no registered experiment_id"):
        _validate_frozen_adaptability_protocol(config)


def test_adaptability_ignores_partial_final_cycle_and_requires_all_complete_blocks():
    config = replace(
        load_experiment_config("config/experiment_60_days.json"),
        adaptability_bootstrap_replicates=20,
    )
    predictions = _adaptability_predictions(3)
    # Remove the positives from the second complete block. The third block is
    # only partial at analysis_end and must not be counted at all.
    second_block_start = pd.Timestamp("2026-01-03T02:00:00Z")
    second_block_end = second_block_start + pd.Timedelta(hours=48)
    timestamps = pd.to_datetime(predictions["snapshot_utc"], utc=True)
    second_block = timestamps.ge(second_block_start) & timestamps.lt(second_block_end)
    predictions.loc[second_block & predictions["y_true"].eq(1), "y_true"] = 0

    report = adaptability_inference_from_predictions(
        predictions,
        config,
        split_cutoff_utc="2026-01-01T00:00:00Z",
        block_anchor_utc="2026-01-01T00:00:00Z",
        analysis_end_utc="2026-01-06T04:00:00Z",
        enforce_frozen_protocol=False,
    )

    assert report["support"]["complete_planned_future_blocks"] == 2
    assert report["support"]["valid_future_blocks"] == 1
    assert report["status"] == "not_enough_data"
    assert "complete planned future blocks" in report["reason"]


def test_support_control_cannot_be_promoted_to_primary_claim():
    predictions = pd.DataFrame(
        [
            {
                "source_row_id": "row-1",
                "canonical_pair_id": "pair-1",
                "y_true": 1,
                "model": "fixed_threshold",
                "score": 1.0,
                "pred": 1,
                "operating_threshold": 0.5,
                "calibration_target_recall": None,
            },
            {
                "source_row_id": "row-1",
                "canonical_pair_id": "pair-1",
                "y_true": 1,
                "model": "constant_velocity_cpa",
                "score": 1.0,
                "pred": 1,
                "operating_threshold": 0.5,
                "calibration_target_recall": None,
            },
        ]
    )

    with pytest.raises(ValueError, match="Reference/control cannot be the primary"):
        paired_pair_cluster_bootstrap(
            predictions,
            replicates=10,
            seed=1,
            confidence_level=0.95,
            recall_noninferiority_margin=0.05,
            min_valid_fraction=0.9,
            primary_model="constant_velocity_cpa",
        )


def test_paired_bootstrap_rejects_unpaired_prediction_rows():
    predictions = pd.DataFrame(
        [
            {"source_row_id": "a", "canonical_pair_id": "p", "y_true": 1,
             "snapshot_utc": "2026-01-01T00:00:00Z",
             "model": "fixed_threshold", "score": 1.0, "pred": 1,
             "operating_threshold": 0.0, "calibration_target_recall": None},
            {"source_row_id": "b", "canonical_pair_id": "q", "y_true": 0,
             "snapshot_utc": "2026-01-02T00:00:00Z",
             "model": "fixed_threshold", "score": 0.0, "pred": 0,
             "operating_threshold": 0.0, "calibration_target_recall": None},
            {"source_row_id": "a", "canonical_pair_id": "p", "y_true": 1,
             "snapshot_utc": "2026-01-01T00:00:00Z",
             "model": CALIBRATED_DISTANCE_BASELINE, "score": 1.0, "pred": 1,
             "operating_threshold": 0.0, "calibration_target_recall": 1.0},
            {"source_row_id": "b", "canonical_pair_id": "q", "y_true": 0,
             "snapshot_utc": "2026-01-02T00:00:00Z",
             "model": CALIBRATED_DISTANCE_BASELINE, "score": 0.0, "pred": 0,
             "operating_threshold": 0.0, "calibration_target_recall": 1.0},
            {"source_row_id": "a", "canonical_pair_id": "p", "y_true": 1,
             "snapshot_utc": "2026-01-01T00:00:00Z",
             "model": "model", "score": 1.0, "pred": 1,
             "operating_threshold": 0.5, "calibration_target_recall": 1.0},
        ]
    )

    with pytest.raises(EvidenceGenerationError, match="not paired"):
        paired_pair_cluster_bootstrap(
            predictions,
            replicates=10,
            seed=1,
            confidence_level=0.95,
            recall_noninferiority_margin=0.05,
            min_valid_fraction=0.9,
            primary_model="model",
        )


def test_generate_evaluation_evidence_writes_frozen_paired_artifacts(
    tmp_path, monkeypatch
):
    base = load_experiment_config("config/experiment_60_days.json")
    config = replace(
        base,
        collection_start_utc="2026-01-01T00:00:00Z",
        collection_end_utc="2026-01-02T00:00:00Z",
        poll_interval_hours=2,
        bootstrap_replicates=50,
        min_valid_bootstrap_fraction=0.8,
        primary_model="logistic_regression",
        required_models=("logistic_regression",),
        min_inner_train_positive_rows=2,
        min_inner_validation_positive_rows=2,
        min_inner_train_positive_pairs=2,
        min_inner_validation_positive_pairs=2,
        min_inner_train_positive_snapshots=1,
        min_inner_validation_positive_snapshots=1,
        min_train_positive_rows=2,
        min_test_positive_rows=2,
        min_train_positive_pairs=2,
        min_test_positive_pairs=2,
        min_train_positive_snapshots=1,
        min_test_positive_snapshots=1,
    )
    frame = _evidence_dataset(config)
    dataset_path = tmp_path / "history.csv"
    frame.to_csv(dataset_path, index=False)

    def models(_weight):
        return {
            "fixed_threshold": None,
            "logistic_regression": make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
            ),
        }

    monkeypatch.setattr(evidence, "_build_models", models)
    snapshot_records = _snapshot_records(frame)
    paths = generate_evaluation_evidence(
        dataset_path,
        tmp_path / "evidence",
        config,
        feature_columns=SNAPSHOT_ONLY_FEATURES,
        snapshot_records=snapshot_records,
        validated_input_binding={"validated": True, "collection_count": len(snapshot_records)},
    )

    assert set(paths) == {
        "split_manifest", "predictions", "evidence", "evaluation_manifest", "feature_importance"
    }
    split = json.loads(paths["split_manifest"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["evaluation_manifest"].read_text(encoding="utf-8"))
    predictions = pd.read_csv(paths["predictions"], comment="#")
    report = pd.read_csv(paths["evidence"], comment="#")
    assert set(predictions["model"]) == {
        "fixed_threshold",
        CALIBRATED_DISTANCE_BASELINE,
        "constant_velocity_cpa",
        "proxy_rule_oracle",
        "logistic_regression",
        "label_permutation_control",
    }
    assert predictions.groupby("model")["source_row_id"].nunique().nunique() == 1
    assert predictions["split_sha256"].nunique() == 1
    assert predictions["split_sha256"].iloc[0] == split["split_sha256"]
    assert set(split["outer"]["train_pair_ids"]).isdisjoint(split["outer"]["test_pair_ids"])
    assert manifest["split_manifest"]["split_sha256"] == split["split_sha256"]
    assert manifest["bootstrap"]["unit"] == "canonical_pair_id"
    assert "logistic_regression" in set(report["model"])
    assert set(SNAPSHOT_ONLY_FEATURES) == set(manifest["features"])
    assert manifest["schema_version"] == 3
    assert manifest["support_controls"]["constant_velocity_cpa"]["claim_eligible"] is False
    assert manifest["support_controls"][CALIBRATED_DISTANCE_BASELINE][
        "claim_eligible"
    ] is False
    assert manifest["support_controls"]["label_permutation_control"]["claim_eligible"] is False
    assert manifest["support_controls"]["proxy_rule_oracle"]["claim_eligible"] is False
    oracle = report.loc[report["model"].eq("proxy_rule_oracle")].iloc[0]
    assert oracle["test_accuracy"] == 1.0
    assert oracle["test_false_positive"] == 0
    assert oracle["test_false_negative"] == 0
    assert split["schema_version"] == 2
    assert split["inner"]["train_rows"] == len(split["inner"]["train_row_ids"])
    assert split["inner"]["validation_rows"] == len(
        split["inner"]["validation_row_ids"]
    )
    assert split["partition_tle_quality"]["outer_test"][
        "tle_input_hash_diversity_fraction"
    ] == 1.0
    learner = report.loc[report["model"].eq("logistic_regression")].iloc[0]
    baseline = report.loc[report["model"].eq("fixed_threshold")].iloc[0]
    calibrated = report.loc[report["model"].eq(CALIBRATED_DISTANCE_BASELINE)].iloc[0]
    assert learner["fitted_train_rows"] == split["inner"]["train_rows"]
    assert pd.notna(learner["delta_vs_constant_velocity_cpa_pr_auc"])
    assert pd.isna(baseline["fitted_train_rows"])
    assert pd.isna(calibrated["fitted_train_rows"])
    assert calibrated["calibration_rows"] == split["inner"]["validation_rows"]
    assert "training/calibration variability is not resampled" in manifest["bootstrap"][
        "uncertainty_scope"
    ]
    assert paths["feature_importance"].is_file()
    gate_report = pd.DataFrame(
        [
            {**{column: None for column in REPORT_COLUMNS}, "model": model_name}
            for model_name in ("fixed_threshold", "logistic_regression")
        ]
    )
    canonical = canonical_report_from_evidence(gate_report, report)
    assert set(canonical["model"]) == {"fixed_threshold", "logistic_regression"}


def test_frozen_evidence_rejects_missing_catalog_ids(tmp_path):
    base = load_experiment_config("config/experiment_60_days.json")
    config = replace(
        base,
        collection_start_utc="2026-01-01T00:00:00Z",
        collection_end_utc="2026-01-02T00:00:00Z",
        poll_interval_hours=2,
        primary_model="logistic_regression",
        required_models=("logistic_regression",),
    )
    frame = _evidence_dataset(config)
    frame.loc[0, "object_1_catalog_id"] = ""
    dataset_path = tmp_path / "missing-id.csv"
    frame.to_csv(dataset_path, index=False)

    with pytest.raises(EvidenceGenerationError, match="non-empty catalogue IDs"):
        generate_evaluation_evidence(
            dataset_path,
            tmp_path / "out",
            config,
            feature_columns=SNAPSHOT_ONLY_FEATURES,
            snapshot_records=_snapshot_records(frame),
            validated_input_binding={"validated": True},
        )


def test_frozen_evidence_rejects_stale_tle_rows(tmp_path):
    base = load_experiment_config("config/experiment_60_days.json")
    config = replace(
        base,
        collection_start_utc="2026-01-01T00:00:00Z",
        collection_end_utc="2026-01-02T00:00:00Z",
        poll_interval_hours=2,
        primary_model="logistic_regression",
        required_models=("logistic_regression",),
    )
    frame = _evidence_dataset(config)
    frame.loc[0, "max_tle_age_hours"] = config.max_tle_age_hours + 1
    dataset_path = tmp_path / "stale.csv"
    frame.to_csv(dataset_path, index=False)

    with pytest.raises(EvidenceGenerationError, match="maximum TLE-age"):
        generate_evaluation_evidence(
            dataset_path,
            tmp_path / "out",
            config,
            feature_columns=SNAPSHOT_ONLY_FEATURES,
            snapshot_records=_snapshot_records(frame),
            validated_input_binding={"validated": True},
        )


def test_frozen_evidence_rejects_low_tle_diversity_inside_test_partition(tmp_path):
    base = load_experiment_config("config/experiment_60_days.json")
    config = replace(
        base,
        collection_start_utc="2026-01-01T00:00:00Z",
        collection_end_utc="2026-01-02T00:00:00Z",
        poll_interval_hours=2,
        primary_model="logistic_regression",
        required_models=("logistic_regression",),
        min_train_positive_rows=2,
        min_test_positive_rows=2,
        min_train_positive_pairs=2,
        min_test_positive_pairs=2,
        min_train_positive_snapshots=1,
        min_test_positive_snapshots=1,
        min_inner_train_positive_rows=2,
        min_inner_validation_positive_rows=2,
        min_inner_train_positive_pairs=2,
        min_inner_validation_positive_pairs=2,
        min_inner_train_positive_snapshots=1,
        min_inner_validation_positive_snapshots=1,
    )
    frame = _evidence_dataset(config)
    dataset_path = tmp_path / "history.csv"
    frame.to_csv(dataset_path, index=False)
    records = _snapshot_records(frame)
    # With the frozen 50/50 time split the outer test contains the final six
    # snapshots. Collapse five of them so global diversity still passes while
    # the held-out partition fails its own update-diversity gate.
    for record in records[-5:]:
        record["input_sha256"] = "a" * 64

    with pytest.raises(EvidenceGenerationError, match="Partition TLE-update diversity"):
        generate_evaluation_evidence(
            dataset_path,
            tmp_path / "out",
            config,
            feature_columns=SNAPSHOT_ONLY_FEATURES,
            snapshot_records=records,
            validated_input_binding={"validated": True},
        )


def test_claim_eligible_evidence_rejects_rule_recovery_feature_set(tmp_path):
    config = load_experiment_config("config/experiment_60_days.json")
    dataset_path = tmp_path / "history.csv"
    _evidence_dataset(config).to_csv(dataset_path, index=False)

    with pytest.raises(EvidenceGenerationError, match="primary snapshot-only"):
        generate_evaluation_evidence(
            dataset_path,
            tmp_path / "out",
            config,
            feature_columns=["min_distance_km", "relative_velocity_km_s"],
            claim_eligible=True,
        )


def test_claim_eligible_evidence_requires_validated_archive_binding(tmp_path):
    config = load_experiment_config("config/experiment_60_days.json")
    dataset_path = tmp_path / "history.csv"
    _evidence_dataset(config).to_csv(dataset_path, index=False)

    with pytest.raises(EvidenceGenerationError, match="archive-to-resimulation"):
        generate_evaluation_evidence(
            dataset_path,
            tmp_path / "out",
            config,
            feature_columns=SNAPSHOT_ONLY_FEATURES,
            claim_eligible=True,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"replicates": 0},
        {"seed": -1},
        {"confidence_level": 1.0},
        {"recall_noninferiority_margin": -0.1},
        {"min_valid_fraction": 0.0},
    ],
)
def test_paired_bootstrap_validates_public_parameters(kwargs):
    defaults = {
        "predictions": pd.DataFrame(),
        "replicates": 10,
        "seed": 1,
        "confidence_level": 0.95,
        "recall_noninferiority_margin": 0.05,
        "min_valid_fraction": 0.9,
        "primary_model": "model",
    }
    defaults.update(kwargs)

    with pytest.raises(ValueError):
        paired_pair_cluster_bootstrap(**defaults)
