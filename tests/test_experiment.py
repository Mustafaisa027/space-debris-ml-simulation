from __future__ import annotations

import json

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
        "default_query": {"preset": "leo_mixed", "max_objects": 75},
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
    assert config.label_threshold_km == 50
    assert config.fixed_threshold_km == 25
    assert config.history.endswith("conjunction_observations_v3.csv")
    assert config.pair_seed == "iac26-pair-split-v1"
    assert config.screening_step_seconds == 30


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


def test_repository_experiment_config_is_valid():
    config = load_experiment_config("config/experiment_60_days.json")

    assert config.poll_interval_hours >= 2
    assert config.candidate_threshold_km >= config.label_threshold_km
    assert config.history.endswith("_v3.csv")


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
    assert collect_args.history == train_args.history
    assert train_args.history.endswith("conjunction_observations_v3.csv")


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
