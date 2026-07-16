from __future__ import annotations

import json
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
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    config = load_experiment_config(path)

    assert config.min_train_positive_rows == 30
    assert config.min_test_positive_rows == 20
    assert config.min_train_positive_pairs == 10
    assert config.min_test_positive_pairs == 5
    assert config.catalog_version == "leo_mixed-legacy-unspecified"
    assert config.catalog_sha256 == ""


def test_repository_experiment_config_is_valid():
    config = load_experiment_config("config/experiment_60_days.json")

    assert config.poll_interval_hours >= 2
    assert config.candidate_threshold_km >= config.label_threshold_km
    assert config.history.endswith("_iac26_75_v1.csv")
    assert config.snapshot_dir.endswith("tle_snapshots_iac26_75_v1")


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
        report.with_name("models_without_label_rule_features.csv"),
        tmp_path / "pub_pr_curves.png",
        tmp_path / "pub_false_alarm_recall_curves.png",
        tmp_path / "pub_confusion_matrices.png",
        tmp_path / "pub_feature_importance.png",
        tmp_path / "pub_threshold_sensitivity.png",
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
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    captured: dict[str, float] = {}
    report = pd.DataFrame([{"model": "random_forest", "pr_auc": 0.5}])

    monkeypatch.setattr("sys.argv", ["train_from_history.py", "--config", str(config_path)])
    monkeypatch.setattr(train_from_history, "compare_models", lambda *args, **kwargs: report)
    monkeypatch.setattr(train_from_history, "plot_model_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        train_from_history,
        "compare_to_baseline_pr_auc",
        lambda *_args, **_kwargs: "comparison",
    )

    def capture_plots(**kwargs):
        captured["threshold"] = kwargs["current_threshold_km"]
        return []

    monkeypatch.setattr(train_from_history, "create_publication_plots", capture_plots)

    train_from_history.main()

    assert captured["threshold"] == 25.0


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
