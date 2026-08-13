from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

import exploratory_compare
from space_debris.experiment import load_experiment_config


def test_exploratory_comparison_is_explicitly_non_claim_eligible(
    monkeypatch, tmp_path
):
    experiment = load_experiment_config("config/experiment_10_days_v3.json")
    history = tmp_path / "history.csv"
    history.write_text(
        "risk_label,snapshot_utc\n0,2026-07-26T00:17:00Z\n1,2026-07-27T00:17:00Z\n",
        encoding="utf-8",
    )
    experiment = replace(experiment, resimulated_history=str(history.resolve()))
    monkeypatch.setattr(
        exploratory_compare, "load_experiment_config", lambda _path: experiment
    )
    captured = {}

    def fake_compare(dataset_path, report_path, **kwargs):
        captured.update(kwargs)
        report = pd.DataFrame(
            [{"model": "xgboost", "pr_auc": 0.5, "precision": 0.4}]
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(report_path, index=False)
        return report

    def fake_plot(_report_path, output_path):
        output_path.write_bytes(b"png")

    monkeypatch.setattr(exploratory_compare, "compare_models", fake_compare)
    monkeypatch.setattr(exploratory_compare, "plot_model_metrics", fake_plot)
    monkeypatch.setattr(
        exploratory_compare, "git_worktree_state", lambda: {"dirty": False}
    )

    manifest = exploratory_compare.run_exploratory_comparison(
        history, tmp_path / "out", Path("config/experiment_10_days_v3.json")
    )

    assert manifest["claim_eligible"] is False
    assert manifest["publication_use_permitted"] is False
    assert manifest["cadence_publication_gates_enforced"] is False
    assert manifest["evaluation"]["best_pr_auc_model"] == "xgboost"
    assert captured["feature_columns"] == [
        "current_distance_km",
        "altitude_difference_km",
        "max_tle_age_hours",
        "radial_velocity_km_s",
        "tangential_velocity_km_s",
        "approach_angle_deg",
    ]
    assert "collection_start_utc" not in captured
    assert "minimum_support" not in captured
    written = json.loads(
        (tmp_path / "out" / "exploratory_manifest.json").read_text(encoding="utf-8")
    )
    assert written["publication_use_permitted"] is False
