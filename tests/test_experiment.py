from __future__ import annotations

import json
import hashlib
from types import SimpleNamespace

import pandas as pd
import pytest

import collect_observations
import rebuild_history
import run_pipeline
import train_from_history
from space_debris.experiment import load_experiment_config


def _config() -> dict:
    return {
        "duration_days": 60,
        "poll_interval_hours": 2,
        "provider": "auto",
        "default_query": {
            "preset": "leo_mixed",
            "catalog_version": "iac26-leo-mixed-75-v1",
            "catalog_sha256": "64c3d2329281da0e246226cb552aafe3b961c553c9487d7cfd9c96374aaa1f4c",
            "max_objects": 75,
        },
        "simulation": {
            "leo_min_altitude_km": 160,
            "leo_max_altitude_km": 2000,
            "horizon_minutes": 720,
            "step_minutes": 5,
            "screening_step_seconds": 30,
            "candidate_threshold_km": 200,
            "fixed_threshold_km": 25,
            "label_threshold_km": 50,
            "label_relative_velocity_km_s": 10,
            "max_tle_age_hours": 336,
        },
        "evaluation": {
            "time_column": "snapshot_utc",
            "train_time_fraction": 0.75,
            "test_pair_fraction": 0.25,
            "pair_seed": "iac26-pair-split-v1",
            "collection_window": {
                "start_utc": "2026-01-01T00:00:00Z",
                "end_utc": "2026-03-02T00:00:00Z",
                "slot_anchor_basis": "github_actions_cron_17_even_utc",
                "slot_timestamp_field": "snapshot_utc",
                "slot_assignment": "half_open_snapshot_bins",
                "min_snapshot_coverage_fraction": 0.9,
                "max_snapshot_gap_hours": 6,
                "min_tle_hash_diversity_fraction": 0.5,
                "max_identical_tle_hash_run_bins": 12,
            },
            "evidence": {
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
                "min_valid_bootstrap_fraction": 0.9,
                "adaptability_min_folds": 5,
            },
            "quality_gate": {
                "min_train_positive_rows": 30,
                "min_test_positive_rows": 20,
                "min_train_positive_pairs": 10,
                "min_test_positive_pairs": 5,
                "min_train_positive_snapshots": 5,
                "min_test_positive_snapshots": 3,
            },
        },
        "outputs": {
            "snapshots": "data/tle_snapshots",
            "runs": "outputs/runs",
            "history": "outputs/history/conjunction_observations_v3.csv",
            "archive_import_report": "outputs/history/collection_archive_import_v3.json",
            "history_rebuild_report": "outputs/history/history_rebuild_report_v3.json",
            "resimulated_runs": "outputs/resimulated_runs_v3",
            "resimulated_history": "outputs/history/conjunction_observations_resimulated_v3.csv",
            "resimulation_report": "outputs/history/resimulation_report_v3.json",
            "time_split_report": "outputs/history/model_comparison_pair_grouped_time_split.csv",
        },
    }


def test_load_experiment_config_exposes_authoritative_defaults(tmp_path):
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(_config()), encoding="utf-8")

    config = load_experiment_config(path)

    assert config.preset == "leo_mixed"
    assert config.catalog_version == "iac26-leo-mixed-75-v1"
    assert config.catalog_sha256 == "64c3d2329281da0e246226cb552aafe3b961c553c9487d7cfd9c96374aaa1f4c"
    assert config.label_threshold_km == 50
    assert config.fixed_threshold_km == 25
    assert config.history.endswith("conjunction_observations_v3.csv")
    assert config.pair_seed == "iac26-pair-split-v1"
    assert config.screening_step_seconds == 30
    assert config.min_train_positive_rows == 30
    assert config.min_test_positive_rows == 20
    assert config.collection_start_utc == "2026-01-01T00:00:00Z"
    assert config.collection_end_utc == "2026-03-02T00:00:00Z"
    assert config.collection_slot_anchor_basis == "github_actions_cron_17_even_utc"
    assert config.collection_slot_timestamp_field == "snapshot_utc"
    assert config.collection_slot_assignment == "half_open_snapshot_bins"
    assert config.min_snapshot_coverage_fraction == 0.9
    assert config.max_snapshot_gap_hours == 6
    assert config.min_tle_hash_diversity_fraction == 0.5
    assert config.max_identical_tle_hash_run_bins == 12
    assert config.minimum_provider_poll_interval_hours == 2
    assert config.bootstrap_replicates == 2000
    assert config.bootstrap_seed == 114764
    assert config.recall_noninferiority_margin == 0.05
    assert config.primary_model == "xgboost"
    assert config.primary_feature_set == "snapshot_only"
    assert config.adaptability_min_folds == 5
    assert set(config.required_models) == {
        "logistic_regression", "random_forest", "svm", "xgboost"
    }


def test_experiment_config_rejects_candidate_radius_smaller_than_label_radius(tmp_path):
    raw = _config()
    raw["simulation"]["candidate_threshold_km"] = 20
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="must include"):
        load_experiment_config(path)


def test_experiment_config_rejects_nan_numeric_value(tmp_path):
    raw = _config()
    raw["simulation"]["candidate_threshold_km"] = float("nan")
    path = tmp_path / "nan.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="finite"):
        load_experiment_config(path)


def test_experiment_config_rejects_nonpositive_quality_gate(tmp_path):
    raw = _config()
    raw["evaluation"]["quality_gate"]["min_test_positive_rows"] = 0
    path = tmp_path / "invalid-gate.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="positive integer"):
        load_experiment_config(path)


@pytest.mark.parametrize("invalid", [True, 1.9])
def test_experiment_config_rejects_noninteger_quality_gate(tmp_path, invalid):
    raw = _config()
    raw["evaluation"]["quality_gate"]["min_test_positive_rows"] = invalid
    path = tmp_path / "invalid-gate.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="positive integer"):
        load_experiment_config(path)


def test_legacy_config_without_quality_gate_uses_documented_defaults(tmp_path):
    raw = _config()
    del raw["evaluation"]["quality_gate"]
    del raw["default_query"]["catalog_version"]
    del raw["default_query"]["catalog_sha256"]
    del raw["evaluation"]["collection_window"]
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    config = load_experiment_config(path)

    assert config.min_train_positive_rows == 30
    assert config.min_test_positive_rows == 20
    assert config.min_train_positive_pairs == 10
    assert config.min_test_positive_pairs == 5
    assert config.catalog_version == "leo_mixed-legacy-unspecified"
    assert config.catalog_sha256 == ""
    assert config.collection_start_utc is None


def test_experiment_config_rejects_partial_collection_window(tmp_path):
    raw = _config()
    del raw["evaluation"]["collection_window"]["end_utc"]
    path = tmp_path / "partial-window.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="configured together"):
        load_experiment_config(path)


def test_experiment_config_rejects_window_duration_different_from_protocol(tmp_path):
    raw = _config()
    raw["evaluation"]["collection_window"]["end_utc"] = "2026-02-01T00:00:00Z"
    path = tmp_path / "short-window.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="duration must equal"):
        load_experiment_config(path)


def test_experiment_config_rejects_provider_floor_above_scientific_slot(tmp_path):
    raw = _config()
    raw["minimum_provider_poll_interval_hours"] = 3
    path = tmp_path / "invalid-provider-floor.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="cannot exceed"):
        load_experiment_config(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("confidence_level", 1.0, "confidence_level"),
        ("recall_noninferiority_margin", -0.1, "recall_noninferiority_margin"),
        ("min_valid_bootstrap_fraction", 0.0, "min_valid_bootstrap_fraction"),
        ("bootstrap_replicates", 0, "bootstrap_replicates"),
        ("bootstrap_seed", 1.5, "bootstrap_seed"),
    ],
)
def test_experiment_config_rejects_invalid_evidence_protocol(
    tmp_path, field, value, message
):
    raw = _config()
    raw["evaluation"]["evidence"][field] = value
    path = tmp_path / "invalid-evidence.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_experiment_config(path)


def test_repository_experiment_config_is_valid():
    config = load_experiment_config("config/experiment_60_days.json")

    assert config.poll_interval_hours >= 2
    assert config.candidate_threshold_km >= config.label_threshold_km
    assert config.history.endswith("_iac26_75_v1.csv")
    assert config.snapshot_dir.endswith("tle_snapshots_iac26_75_v1")


def test_repository_v2_config_is_pre_registered_and_isolated():
    config = load_experiment_config("config/experiment_10_days_v2.json")

    assert config.duration_days == 10
    assert config.collection_start_utc == "2026-07-24T00:17:00Z"
    assert config.collection_end_utc == "2026-08-03T00:17:00Z"
    assert config.catalog_version == "iac26-leo-mixed-75-v2"
    assert config.archive_collections == "experiments/iac26-10d-v2/collections"
    assert config.snapshot_dir.endswith("tle_snapshots_iac26_75_v2")
    assert config.history.endswith("_iac26_75_v2.csv")
    assert config.train_time_fraction == 0.40
    assert config.adaptability_block_hours == 12
    assert config.adaptability_min_blocks == 10
    assert config.min_tle_hash_diversity_fraction == 0.30


def test_repository_v4_config_is_active_and_frozen_protocol_consistent():
    """v4 is independent while preserving the accepted scientific method."""
    from space_debris.experiment import ACTIVE_EXPERIMENT_CONFIG
    from space_debris.evidence import _validate_frozen_adaptability_protocol

    assert ACTIVE_EXPERIMENT_CONFIG.name == "experiment_15_days_v4.json"

    v4 = load_experiment_config("config/experiment_15_days_v4.json")
    v3 = load_experiment_config("config/experiment_10_days_v3.json")

    assert v4.experiment_id == "iac26-15d-v4"
    assert v4.duration_days == 15
    assert v4.poll_interval_hours == 3
    assert v4.minimum_provider_poll_interval_hours == 2
    assert v4.duration_days * 24 / v4.poll_interval_hours == 120
    assert v4.collection_start_utc == "2026-08-10T00:17:00Z"
    assert v4.collection_end_utc == "2026-08-25T00:17:00Z"
    assert v4.archive_collections == "experiments/iac26-15d-v4/collections"
    assert v4.snapshot_dir.endswith("tle_snapshots_iac26_75_v4")
    assert v4.history.endswith("_iac26_75_v4.csv")

    # Catalogue, method, gates and seeds are preserved; only prospective
    # acquisition mechanics and isolated output paths change.
    assert v4.catalog_version == v3.catalog_version
    assert v4.catalog_sha256 == v3.catalog_sha256
    frozen_fields = (
        "candidate_threshold_km",
        "fixed_threshold_km",
        "label_threshold_km",
        "label_relative_velocity_km_s",
        "max_tle_age_hours",
        "train_time_fraction",
        "test_pair_fraction",
        "pair_seed",
        "primary_model",
        "primary_feature_set",
        "required_models",
        "inner_pair_seed",
        "bootstrap_replicates",
        "bootstrap_seed",
        "recall_noninferiority_margin",
        "min_snapshot_coverage_fraction",
        "max_snapshot_gap_hours",
        "min_tle_hash_diversity_fraction",
        "max_identical_tle_hash_run_bins",
        "min_train_positive_rows",
        "min_test_positive_rows",
        "min_train_positive_pairs",
        "min_test_positive_pairs",
        "min_train_positive_snapshots",
        "min_test_positive_snapshots",
    )
    for field in frozen_fields:
        assert getattr(v4, field) == getattr(v3, field)

    _validate_frozen_adaptability_protocol(v4)


def test_experiment_config_rejects_unsafe_archive_collection_path(tmp_path):
    raw = _config()
    raw["outputs"]["archive_collections"] = "../collections"
    path = tmp_path / "unsafe-archive.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="archive_collections"):
        load_experiment_config(path)


def test_command_defaults_are_loaded_from_the_same_experiment_config(monkeypatch):
    monkeypatch.setattr("sys.argv", ["collect_observations.py"])
    collect_args = collect_observations.parse_args()
    monkeypatch.setattr("sys.argv", ["run_pipeline.py"])
    pipeline_args = run_pipeline.parse_args()
    monkeypatch.setattr("sys.argv", ["train_from_history.py"])
    train_args = train_from_history.parse_args()

    assert collect_args.label_threshold_km == pipeline_args.label_threshold_km == 50
    assert collect_args.candidate_threshold_km == pipeline_args.candidate_threshold_km == 200
    assert collect_args.step_minutes == pipeline_args.step_minutes == 5
    assert collect_args.screening_step_seconds == pipeline_args.screening_step_seconds == 30
    assert collect_args.history.endswith("conjunction_observations_iac26_75_v1.csv")
    assert train_args.history.endswith("conjunction_observations_resimulated_iac26_75_v1.csv")
    assert collect_args.history != train_args.history


def test_rebuild_defaults_follow_custom_config_paths(monkeypatch, tmp_path):
    raw = _config()
    raw["outputs"].update(
        snapshots=str(tmp_path / "snapshots"),
        runs=str(tmp_path / "runs"),
        history=str(tmp_path / "history" / "custom.csv"),
        history_rebuild_report=str(tmp_path / "history" / "custom-report.json"),
    )
    config_path = tmp_path / "custom.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["rebuild_history.py", "--config", str(config_path)])

    args = rebuild_history.parse_args()

    assert args.run_root == tmp_path / "runs"
    assert args.snapshot_dir == tmp_path / "snapshots"
    assert args.output == tmp_path / "history" / "custom.csv"
    assert args.report == tmp_path / "history" / "custom-report.json"


def test_quality_gate_cleanup_removes_only_known_stale_artifacts(tmp_path):
    report = tmp_path / "models.csv"
    keep = tmp_path / "unrelated.txt"
    keep.write_text("keep", encoding="utf-8")
    stale = [
        report.with_suffix(".png"),
        report.with_name("models_cv.csv"),
        report.with_name("models_adaptability_inference.json"),
        report.with_name("models_without_label_rule_features.csv"),
        report.with_name("models_snapshot_geometry_only.csv"),
        report.with_name("models_snapshot_kinematics_only.csv"),
        report.with_name("models_distance_only.csv"),
        report.with_name("models_proxy_rule_only.csv"),
        tmp_path / "pub_pr_curves.png",
        tmp_path / "pub_false_alarm_recall_curves.png",
        tmp_path / "pub_confusion_matrices.png",
        tmp_path / "pub_feature_importance.png",
        tmp_path / "pub_threshold_sensitivity.png",
        tmp_path / "evaluation_split_manifest.json",
        tmp_path / "held_out_predictions.csv",
        tmp_path / "paired_cluster_evidence.csv",
        tmp_path / "evaluation_manifest.json",
        tmp_path / "feature_importance.csv",
        tmp_path / "publication_manifest.json",
        tmp_path / "publication_incomplete.json",
        tmp_path / "model_comparison_time_split.csv",
        tmp_path / "model_comparison_time_split.png",
        tmp_path / "model_comparison_time_split_cv.csv",
        tmp_path / "model_comparison_time_split_without_label_rule_features.csv",
        tmp_path / "model_comparison_pair_grouped_time_split.csv",
        tmp_path / "model_comparison_pair_grouped_time_split.png",
        tmp_path / "model_comparison_pair_grouped_time_split_cv.csv",
        tmp_path / "model_comparison_pair_grouped_time_split_without_label_rule_features.csv",
    ]
    for path in stale:
        path.write_text("stale", encoding="utf-8")

    train_from_history._clear_stale_model_artifacts(report)

    assert all(not path.exists() for path in stale)
    assert keep.read_text(encoding="utf-8") == "keep"


def test_training_passes_configured_fixed_threshold_to_publication_plots(
    monkeypatch, tmp_path
):
    raw = _config()
    raw["outputs"]["resimulated_history"] = str(tmp_path / "history.csv")
    raw["outputs"]["time_split_report"] = str(tmp_path / "models.csv")
    raw["outputs"]["resimulation_report"] = str(tmp_path / "resimulation.json")
    raw["outputs"]["archive_import_report"] = str(tmp_path / "archive-import.json")
    archive_bundles = [
        {
            "name": "collection_1_1",
            "collection_id": "20260101_000000",
            "manifest_sha256": "b" * 64,
            "input_snapshot_sha256": "a" * 64,
            "input_sidecar_sha256": "e" * 64,
            "source_git_commit": "c" * 40,
            "runtime_provenance_status": "verified",
            "manifest_schema_version": 2,
        }
    ]
    archive_manifest_set = hashlib.sha256(
        json.dumps(archive_bundles, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    (tmp_path / "archive-import.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "archive_revision": "d" * 40,
                "manifest_set_sha256": archive_manifest_set,
                "catalog_version": raw["default_query"]["catalog_version"],
                "catalog_sha256": raw["default_query"]["catalog_sha256"],
                "verify_only": False,
                "clean_checkout_verified": True,
                "trusted_source_ref": "origin/main",
                "trusted_source_ref_commit": "9" * 40,
                "source_ancestry_verified": True,
                "bundles_verified": 1,
                "collections_verified": 1,
                "bundles": archive_bundles,
            }
        ),
        encoding="utf-8",
    )
    history_path = tmp_path / "history.csv"
    history_path.write_text("value\n1\n", encoding="utf-8")
    history_fingerprints = {"20260101_000000": "f" * 64}
    history_fingerprint_set = hashlib.sha256(
        json.dumps(
            history_fingerprints, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    (tmp_path / "resimulation.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "completed": [
                    {
                        "collection_id": "20260101_000000",
                        "snapshot_utc": "2026-01-01T00:00:00Z",
                        "input_sha256": "a" * 64,
                        "run_fingerprint": "f" * 64,
                        "fingerprint_inputs": {"sidecar_sha256": "e" * 64},
                    }
                ],
                "failures": [],
                "history": {
                    "output": str(history_path),
                    "output_sha256": hashlib.sha256(history_path.read_bytes()).hexdigest(),
                    "output_bytes": history_path.stat().st_size,
                    "rows_written": 1,
                    "included_runs": ["20260101_000000"],
                    "resimulation_fingerprints": history_fingerprints,
                    "resimulation_fingerprint_set_sha256": history_fingerprint_set,
                    "resimulation_integrity_required": True,
                },
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setattr(train_from_history, "DEFAULT_EXPERIMENT_CONFIG", config_path)
    captured: dict[str, object] = {"evidence_features": []}
    report = pd.DataFrame([{"model": "random_forest", "pr_auc": 0.5}])
    evidence_table = tmp_path / "evidence.csv"
    pd.DataFrame([{"model": "random_forest"}]).to_csv(evidence_table, index=False)

    monkeypatch.setattr(
        "sys.argv",
        ["train_from_history.py", "--config", str(config_path), "--cv-splits", "0"],
    )
    def capture_compare(*args, **kwargs):
        captured["gate_features"] = list(kwargs["feature_columns"])
        return report

    monkeypatch.setattr(train_from_history, "compare_models", capture_compare)
    monkeypatch.setattr(train_from_history, "plot_model_metrics", lambda *args, **kwargs: None)
    def capture_evidence(*args, **kwargs):
        captured["evidence_features"].append(list(kwargs["feature_columns"]))
        predictions_path = tmp_path / "predictions.csv"
        importance_path = tmp_path / "importance.csv"
        split_path = tmp_path / "split.json"
        predictions_path.write_text("model,score\nrandom_forest,0.5\n", encoding="utf-8")
        importance_path.write_text("model,feature\nrandom_forest,x\n", encoding="utf-8")
        split_path.write_text("{}\n", encoding="utf-8")
        return {
            "split_manifest": split_path,
            "predictions": predictions_path,
            "feature_importance": importance_path,
            "evidence": evidence_table,
            "evaluation_manifest": tmp_path / "evaluation-manifest.json",
        }

    monkeypatch.setattr(
        train_from_history, "generate_evaluation_evidence", capture_evidence
    )
    monkeypatch.setattr(
        train_from_history,
        "generate_adaptability_inference",
        lambda _predictions, _split, output, _config: output,
    )
    monkeypatch.setattr(
        train_from_history,
        "canonical_report_from_evidence",
        lambda *_args, **_kwargs: report,
    )
    monkeypatch.setattr(
        train_from_history,
        "compare_to_baseline_pr_auc",
        lambda *_args, **_kwargs: "comparison",
    )

    def capture_plots(**kwargs):
        captured["threshold"] = kwargs["current_threshold_km"]
        return []

    monkeypatch.setattr(train_from_history, "create_publication_plots", capture_plots)
    monkeypatch.setattr(
        train_from_history, "_write_publication_manifest", lambda *args, **kwargs: None
    )

    train_from_history.main()

    assert captured["threshold"] == 25.0
    primary = train_from_history.feature_set_by_name("snapshot_only")
    assert captured["gate_features"] == primary
    assert captured["evidence_features"][0] == primary
    assert captured["evidence_features"][1:] == [
        list(features)
        for features in train_from_history.FROZEN_ABLATION_FEATURES.values()
    ]


def test_frozen_ablation_name_cannot_be_bound_to_another_feature_set():
    expected = list(train_from_history.FROZEN_ABLATION_FEATURES["distance_only"])
    train_from_history._validate_frozen_ablation_features(
        "distance_only", {"features": expected}
    )

    swapped = list(
        train_from_history.FROZEN_ABLATION_FEATURES["snapshot_geometry_only"]
    )
    with pytest.raises(ValueError, match="frozen protocol"):
        train_from_history._validate_frozen_ablation_features(
            "distance_only", {"features": swapped}
        )


def test_publication_manifest_never_promotes_descriptive_adaptability_to_significance(
    tmp_path, monkeypatch
):
    artifact = tmp_path / "artifact.csv"
    artifact.write_text("value\n1\n", encoding="utf-8")
    adaptability = tmp_path / "models_cv_adaptability.csv"
    pd.DataFrame(
        [
            {
                "model": "xgboost",
                "adaptability_supported": True,
                "statistical_significance_tested": False,
                "claim": "descriptive only",
            }
        ]
    ).to_csv(adaptability, index=False)
    evidence_report = pd.DataFrame(
        [
            {
                "model": "xgboost",
                "false_alarm_reduction_supported": True,
                "pair_cluster_gate_passed": True,
                "time_block_gate_passed": True,
                "calibrated_distance_pair_gate_passed": True,
                "calibrated_distance_time_gate_passed": True,
                "claim": "conditional proxy FAR reduction",
                "test_pr_auc": 0.8,
            },
            {
                "model": "proxy_rule_oracle",
                "test_accuracy": 1.0,
                "test_false_positive": 0,
                "test_false_negative": 0,
                "test_pr_auc": 1.0,
            },
            {
                "model": "label_permutation_control",
                "test_pr_auc": 0.2,
            }
        ]
    )
    monkeypatch.setattr(
        train_from_history,
        "git_worktree_state",
        lambda: {"dirty": False, "status_sha256": "0" * 64},
    )
    monkeypatch.setattr(train_from_history, "git_commit_full_hash", lambda: "1" * 40)

    output = tmp_path / "publication_manifest.json"
    train_from_history._write_publication_manifest(
        output,
        [artifact, adaptability],
        primary_model="xgboost",
        primary_feature_set="snapshot_only",
        evidence_report=evidence_report,
        validated_input_binding={"validated": True, "collection_count": 1},
        require_evaluation_bundle=False,
    )

    manifest = json.loads(output.read_text(encoding="utf-8"))
    decision = manifest["claim_decision"]
    assert manifest["schema_version"] == 4
    assert decision["false_alarm_reduction_supported"] is True
    assert decision["adaptability_supported"] is True
    assert decision["statistical_adaptability_supported"] is False
    assert decision["overall_abstract_claim_supported"] is False
    assert decision["archive_resimulation_binding_validated"] is True
    assert decision["pipeline_integrity_controls_passed"] is True
    assert decision["label_permutation_diagnostic_favorable"] is True


def test_publication_manifest_fails_closed_on_invalid_sanity_control(
    tmp_path, monkeypatch
):
    artifact = tmp_path / "artifact.csv"
    artifact.write_text("value\n1\n", encoding="utf-8")
    evidence_report = pd.DataFrame(
        [
            {
                "model": "xgboost",
                "false_alarm_reduction_supported": True,
                "pair_cluster_gate_passed": True,
                "time_block_gate_passed": True,
                "calibrated_distance_pair_gate_passed": True,
                "calibrated_distance_time_gate_passed": True,
                "claim": "conditional proxy FAR reduction",
                "test_pr_auc": 0.8,
            },
            {
                "model": "proxy_rule_oracle",
                "test_accuracy": 0.99,
                "test_false_positive": 1,
                "test_false_negative": 0,
                "test_pr_auc": 0.99,
            },
            {
                "model": "label_permutation_control",
                "test_pr_auc": 0.8,
            },
        ]
    )
    monkeypatch.setattr(
        train_from_history,
        "git_worktree_state",
        lambda: {"dirty": False, "status_sha256": "0" * 64},
    )
    monkeypatch.setattr(train_from_history, "git_commit_full_hash", lambda: "1" * 40)

    output = tmp_path / "publication_manifest.json"
    train_from_history._write_publication_manifest(
        output,
        [artifact],
        primary_model="xgboost",
        primary_feature_set="snapshot_only",
        evidence_report=evidence_report,
        validated_input_binding={"validated": True, "collection_count": 1},
        require_evaluation_bundle=False,
    )

    decision = json.loads(output.read_text(encoding="utf-8"))["claim_decision"]
    assert decision["pipeline_integrity_controls_passed"] is False
    assert decision["false_alarm_reduction_supported"] is False
    assert decision["overall_abstract_claim_supported"] is False


def test_publication_manifest_rejects_self_asserted_adaptability_without_bundle(
    tmp_path, monkeypatch
):
    artifact = tmp_path / "artifact.csv"
    artifact.write_text("value\n1\n", encoding="utf-8")
    predictions = tmp_path / "held_out_predictions.csv"
    predictions.write_text("row\n1\n", encoding="utf-8")
    config = tmp_path / "experiment.json"
    config.write_text("{}\n", encoding="utf-8")
    inference = tmp_path / "models_adaptability_inference.json"
    inference.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "primary_model": "xgboost",
                "primary_feature_set": "snapshot_only",
                "status": "supported",
                "claim": "future-block transportability advantage supported",
                "source_worktree": {"dirty": False},
                "predictions": {
                    "path": str(predictions),
                    "sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
                },
                "config": {
                    "path": str(config),
                    "sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                },
                "support": {
                    "valid_future_blocks": 10,
                    "required_future_blocks": 10,
                    "unique_objects": 40,
                    "required_unique_objects": 30,
                    "canonical_pairs": 40,
                    "required_canonical_pairs": 30,
                },
                "bootstrap": {
                    "confidence_level": 0.95,
                    "minimum_valid_fraction": 0.9,
                },
                "statistical_significance_tested": True,
                "valid_bootstrap_fraction": 0.99,
                "one_sided_confidence_lower_bound": 0.1,
                "centered_bootstrap_one_sided_p_value": 0.01,
                "all_valid_future_blocks_positive": True,
                "statistical_adaptability_supported": True,
            }
        ),
        encoding="utf-8",
    )
    evidence_report = pd.DataFrame(
        [
            {
                "model": "xgboost",
                "false_alarm_reduction_supported": True,
                "pair_cluster_gate_passed": True,
                "time_block_gate_passed": True,
                "calibrated_distance_pair_gate_passed": True,
                "calibrated_distance_time_gate_passed": True,
                "claim": "conditional proxy FAR reduction",
                "test_pr_auc": 0.8,
            },
            {
                "model": "proxy_rule_oracle",
                "test_accuracy": 1.0,
                "test_false_positive": 0,
                "test_false_negative": 0,
                "test_pr_auc": 1.0,
            },
            {"model": "label_permutation_control", "test_pr_auc": 0.2},
        ]
    )
    monkeypatch.setattr(
        train_from_history,
        "git_worktree_state",
        lambda: {"dirty": False, "status_sha256": "0" * 64},
    )
    monkeypatch.setattr(train_from_history, "git_commit_full_hash", lambda: "1" * 40)

    output = tmp_path / "publication_manifest.json"
    with pytest.raises(ValueError, match="verified main evaluation bundle"):
        train_from_history._write_publication_manifest(
            output,
            [artifact, predictions, inference],
            primary_model="xgboost",
            primary_feature_set="snapshot_only",
            evidence_report=evidence_report,
            validated_input_binding={"validated": True, "collection_count": 1},
            require_evaluation_bundle=False,
        )


def test_publication_manifest_treats_permutation_as_diagnostic_not_claim_gate(
    tmp_path, monkeypatch
):
    artifact = tmp_path / "artifact.csv"
    artifact.write_text("value\n1\n", encoding="utf-8")
    evidence_report = pd.DataFrame(
        [
            {
                "model": "xgboost",
                "false_alarm_reduction_supported": True,
                "pair_cluster_gate_passed": True,
                "time_block_gate_passed": True,
                "calibrated_distance_pair_gate_passed": True,
                "calibrated_distance_time_gate_passed": True,
                "claim": "conditional proxy FAR reduction",
                "test_pr_auc": 0.8,
            },
            {
                "model": "proxy_rule_oracle",
                "test_accuracy": 1.0,
                "test_false_positive": 0,
                "test_false_negative": 0,
                "test_pr_auc": 1.0,
            },
            {"model": "label_permutation_control", "test_pr_auc": 0.8},
        ]
    )
    monkeypatch.setattr(
        train_from_history,
        "git_worktree_state",
        lambda: {"dirty": False, "status_sha256": "0" * 64},
    )
    monkeypatch.setattr(train_from_history, "git_commit_full_hash", lambda: "1" * 40)

    output = tmp_path / "publication_manifest.json"
    train_from_history._write_publication_manifest(
        output,
        [artifact],
        primary_model="xgboost",
        primary_feature_set="snapshot_only",
        evidence_report=evidence_report,
        validated_input_binding={"validated": True, "collection_count": 1},
        require_evaluation_bundle=False,
    )

    decision = json.loads(output.read_text(encoding="utf-8"))["claim_decision"]
    assert decision["label_permutation_diagnostic_favorable"] is False
    assert decision["pipeline_integrity_controls_passed"] is True
    assert decision["false_alarm_reduction_supported"] is True


def test_publication_evaluation_bundle_revalidates_recursive_hash_graph(
    tmp_path, monkeypatch
):
    commit = "1" * 40
    dataset = tmp_path / "history.csv"
    config = tmp_path / "experiment.json"
    split = tmp_path / "evaluation_split_manifest.json"
    predictions = tmp_path / "held_out_predictions.csv"
    evidence = tmp_path / "paired_cluster_evidence.csv"
    importance = tmp_path / "feature_importance.csv"
    manifest_path = tmp_path / "evaluation_manifest.json"
    dataset.write_text("risk_label\n0\n", encoding="utf-8")
    config.write_text("{}\n", encoding="utf-8")
    dataset_sha = hashlib.sha256(dataset.read_bytes()).hexdigest()
    config_sha = hashlib.sha256(config.read_bytes()).hexdigest()
    split_payload = {
        "schema_version": 2,
        "dataset_sha256": dataset_sha,
        "config_sha256": config_sha,
        "claim_eligible": True,
        "outer": {},
        "inner": {},
    }
    split_sha = hashlib.sha256(
        json.dumps(
            split_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    binding = {
        "validated": True,
        "collection_count": 1,
        "canonical_history": {"path": str(dataset), "sha256": dataset_sha},
    }
    split.write_text(
        json.dumps({**split_payload, "split_sha256": split_sha}),
        encoding="utf-8",
    )
    predictions.write_text(
        f"source_row_id,split_sha256\nrow-1,{split_sha}\n", encoding="utf-8"
    )
    evidence.write_text("model\nxgboost\n", encoding="utf-8")
    importance.write_text("model,feature\nxgboost,current_distance_km\n", encoding="utf-8")

    def entry(path):
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "git_commit": commit,
                "source_worktree": {"dirty": False},
                "archive_resimulation_binding": binding,
                "dataset": entry(dataset),
                "config": entry(config),
                "split_manifest": {**entry(split), "split_sha256": split_sha},
                "predictions": entry(predictions),
                "evidence": entry(evidence),
                "feature_importance": entry(importance),
                "bootstrap": {"claim_eligible": True},
                "upstream_artifacts": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(train_from_history, "git_commit_full_hash", lambda: commit)
    artifacts = [manifest_path, split, predictions, evidence, importance]

    verified = train_from_history._verify_publication_evaluation_bundle(
        manifest_path, artifacts, binding, canonical_config_path=config
    )
    assert verified["predictions_path"] == predictions

    predictions.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="predictions hash/path"):
        train_from_history._verify_publication_evaluation_bundle(
            manifest_path, artifacts, binding, canonical_config_path=config
        )


def test_publication_evaluation_bundle_recomputes_split_and_cross_bindings(
    tmp_path, monkeypatch
):
    commit = "1" * 40
    dataset = tmp_path / "history.csv"
    config = tmp_path / "experiment.json"
    split = tmp_path / "evaluation_split_manifest.json"
    predictions = tmp_path / "held_out_predictions.csv"
    evidence = tmp_path / "paired_cluster_evidence.csv"
    importance = tmp_path / "feature_importance.csv"
    manifest_path = tmp_path / "evaluation_manifest.json"
    dataset.write_text("risk_label\n0\n", encoding="utf-8")
    config.write_text("{}\n", encoding="utf-8")
    dataset_sha = hashlib.sha256(dataset.read_bytes()).hexdigest()
    config_sha = hashlib.sha256(config.read_bytes()).hexdigest()
    split_payload = {
        "schema_version": 2,
        "dataset_sha256": dataset_sha,
        "config_sha256": config_sha,
        "claim_eligible": True,
        "outer": {},
        "inner": {},
    }
    split_sha = train_from_history._canonical_payload_sha256(split_payload)
    split.write_text(json.dumps({**split_payload, "split_sha256": split_sha}), encoding="utf-8")
    predictions.write_text(
        f"source_row_id,split_sha256\nrow-1,{split_sha}\n", encoding="utf-8"
    )
    evidence.write_text("model\nxgboost\n", encoding="utf-8")
    importance.write_text("model,feature\nxgboost,current_distance_km\n", encoding="utf-8")

    def entry(path):
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def write_manifest():
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 3,
                    "git_commit": commit,
                    "source_worktree": {"dirty": False},
                    "archive_resimulation_binding": binding,
                    "dataset": entry(dataset),
                    "config": entry(config),
                    "split_manifest": {**entry(split), "split_sha256": split_sha},
                    "predictions": entry(predictions),
                    "evidence": entry(evidence),
                    "feature_importance": entry(importance),
                    "bootstrap": {"claim_eligible": True},
                    "upstream_artifacts": [],
                }
            ),
            encoding="utf-8",
        )

    binding = {
        "validated": True,
        "canonical_history": {"path": str(dataset), "sha256": dataset_sha},
    }
    monkeypatch.setattr(train_from_history, "git_commit_full_hash", lambda: commit)
    artifacts = [manifest_path, split, predictions, evidence, importance]
    write_manifest()

    tampered = json.loads(split.read_text(encoding="utf-8"))
    tampered["dataset_sha256"] = "f" * 64
    split.write_text(json.dumps(tampered), encoding="utf-8")
    write_manifest()
    with pytest.raises(ValueError, match="split SHA"):
        train_from_history._verify_publication_evaluation_bundle(
            manifest_path, artifacts, binding, canonical_config_path=config
        )

    split.write_text(json.dumps({**split_payload, "split_sha256": split_sha}), encoding="utf-8")
    write_manifest()
    wrong_binding = {
        **binding,
        "canonical_history": {"path": str(dataset), "sha256": "e" * 64},
    }
    binding = wrong_binding
    write_manifest()
    with pytest.raises(ValueError, match="validated canonical history"):
        train_from_history._verify_publication_evaluation_bundle(
            manifest_path, artifacts, wrong_binding, canonical_config_path=config
        )


def _archive_binding_report(raw: dict, *, input_sha256: str = "a" * 64) -> dict:
    bundles = [
        {
            "name": "collection_1_1",
            "collection_id": "20260101_000000",
            "manifest_sha256": "b" * 64,
            "input_snapshot_sha256": input_sha256,
            "input_sidecar_sha256": "e" * 64,
            "source_git_commit": "c" * 40,
            "runtime_provenance_status": "verified",
            "manifest_schema_version": 2,
        }
    ]
    return {
        "schema_version": 1,
        "archive_revision": "d" * 40,
        "manifest_set_sha256": hashlib.sha256(
            json.dumps(bundles, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "catalog_version": raw["default_query"]["catalog_version"],
        "catalog_sha256": raw["default_query"]["catalog_sha256"],
        "verify_only": False,
        "clean_checkout_verified": True,
        "trusted_source_ref": "origin/main",
        "trusted_source_ref_commit": "9" * 40,
        "source_ancestry_verified": True,
        "bundles_verified": 1,
        "collections_verified": 1,
        "bundles": bundles,
    }


def test_archive_resimulation_binding_requires_exact_collection_and_hash(tmp_path):
    raw = _config()
    report_path = tmp_path / "archive.json"
    report_path.write_text(json.dumps(_archive_binding_report(raw)), encoding="utf-8")
    snapshots = [
        {
            "collection_id": "20260101_000000",
            "snapshot_utc": "2026-01-01T00:00:00Z",
            "input_sha256": "a" * 64,
            "sidecar_sha256": "e" * 64,
            "run_fingerprint": "f" * 64,
        }
    ]

    binding = train_from_history._validate_archive_resimulation_binding(
        report_path,
        snapshots,
        catalog_version=raw["default_query"]["catalog_version"],
        catalog_sha256=raw["default_query"]["catalog_sha256"],
    )
    assert binding["validated"] is True
    assert binding["collection_count"] == 1

    report_path.write_text(
        json.dumps(_archive_binding_report(raw, input_sha256="e" * 64)),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="do not match exactly"):
        train_from_history._validate_archive_resimulation_binding(
            report_path,
            snapshots,
            catalog_version=raw["default_query"]["catalog_version"],
            catalog_sha256=raw["default_query"]["catalog_sha256"],
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda report: report.update({"verify_only": True}), "actual archive import"),
        (lambda report: report.update({"archive_revision": "unspecified"}), "clean committed"),
        (lambda report: report.update({"clean_checkout_verified": False}), "clean Git checkout"),
        (lambda report: report.update({"bundles": []}), "no verified bundles"),
    ],
)
def test_archive_resimulation_binding_rejects_nonpublication_reports(
    tmp_path, mutation, message
):
    raw = _config()
    report = _archive_binding_report(raw)
    mutation(report)
    report_path = tmp_path / "archive.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        train_from_history._validate_archive_resimulation_binding(
            report_path,
            [
                {
                    "collection_id": "20260101_000000",
                    "snapshot_utc": "2026-01-01T00:00:00Z",
                    "input_sha256": "a" * 64,
                    "sidecar_sha256": "e" * 64,
                    "run_fingerprint": "f" * 64,
                }
            ],
            catalog_version=raw["default_query"]["catalog_version"],
            catalog_sha256=raw["default_query"]["catalog_sha256"],
        )


def test_canonical_history_binding_rejects_tampered_history(tmp_path):
    history_path = tmp_path / "history.csv"
    history_path.write_text("value\n1\n", encoding="utf-8")
    records = [
        {
            "collection_id": "20260101_000000",
            "snapshot_utc": "2026-01-01T00:00:00Z",
            "input_sha256": "a" * 64,
            "sidecar_sha256": "e" * 64,
            "run_fingerprint": "f" * 64,
        }
    ]
    fingerprints = {"20260101_000000": "f" * 64}
    fingerprint_set = hashlib.sha256(
        json.dumps(fingerprints, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    report_path = tmp_path / "resimulation.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "completed": [],
                "failures": [],
                "history": {
                    "output": str(history_path),
                    "output_sha256": hashlib.sha256(history_path.read_bytes()).hexdigest(),
                    "output_bytes": history_path.stat().st_size,
                    "rows_written": 1,
                    "included_runs": ["20260101_000000"],
                    "resimulation_fingerprints": fingerprints,
                    "resimulation_fingerprint_set_sha256": fingerprint_set,
                    "resimulation_integrity_required": True,
                },
            }
        ),
        encoding="utf-8",
    )

    binding = train_from_history._validate_canonical_history_binding(
        report_path, history_path, records
    )
    assert binding["validated"] is True
    assert binding["rows"] == 1

    history_path.write_text("value\n2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        train_from_history._validate_canonical_history_binding(
            report_path, history_path, records
        )


def test_archive_binding_rejects_unapproved_schema_one_bundle(tmp_path):
    raw = _config()
    report = _archive_binding_report(raw)
    report["bundles"][0]["manifest_schema_version"] = 1
    report["bundles"][0]["runtime_provenance_status"] = "legacy_runtime_missing"
    report["manifest_set_sha256"] = hashlib.sha256(
        json.dumps(
            report["bundles"], sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    report_path = tmp_path / "archive.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="three frozen legacy bundles"):
        train_from_history._validate_archive_resimulation_binding(
            report_path,
            [
                {
                    "collection_id": "20260101_000000",
                    "snapshot_utc": "2026-01-01T00:00:00Z",
                    "input_sha256": "a" * 64,
                    "sidecar_sha256": "e" * 64,
                    "run_fingerprint": "f" * 64,
                }
            ],
            catalog_version=raw["default_query"]["catalog_version"],
            catalog_sha256=raw["default_query"]["catalog_sha256"],
        )


def test_pipeline_resolves_duplicate_debris_names_by_catalog_id():
    satellites = [
        SimpleNamespace(name="COSMOS 2251 DEB", model=SimpleNamespace(satnum=33757)),
        SimpleNamespace(name="COSMOS 2251 DEB", model=SimpleNamespace(satnum=33758)),
        SimpleNamespace(name="AURA", model=SimpleNamespace(satnum=28376)),
    ]
    pair = SimpleNamespace(
        object_1="AURA",
        object_2="COSMOS 2251 DEB",
        object_1_catalog_id="28376",
        object_2_catalog_id="33758",
    )

    first, second = run_pipeline._resolve_pair_satellites(satellites, pair)

    assert first.model.satnum == 28376
    assert second.model.satnum == 33758
