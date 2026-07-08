"""Unit tests for analyze_distributions.py: percentile summaries and the
threshold-calibration mechanism (ROADMAP_YOL1.md GOREV 3).

Run with:  PYTHONPATH=src python -m pytest tests/ -q
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import analyze_distributions as ad


def _synthetic_df(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "min_distance_km": rng.uniform(5.0, 500.0, size=n),
            "relative_velocity_km_s": rng.uniform(0.5, 15.0, size=n),
        }
    )


def test_load_history_missing_columns_raises(tmp_path):
    path = tmp_path / "history.csv"
    pd.DataFrame({"foo": [1, 2, 3]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        ad.load_history(path)


def test_load_history_empty_raises(tmp_path):
    path = tmp_path / "history.csv"
    pd.DataFrame(columns=ad.REQUIRED_COLUMNS).to_csv(path, index=False)
    with pytest.raises(ValueError, match="no rows to analyze"):
        ad.load_history(path)


def test_percentile_summary_matches_numpy():
    series = pd.Series(range(1, 101))  # 1..100
    summary = ad.percentile_summary(series, percentiles=(10, 50, 90))
    assert summary["p50"] == pytest.approx(np.percentile(series, 50))
    assert summary["p10"] == pytest.approx(np.percentile(series, 10))
    assert summary["p90"] == pytest.approx(np.percentile(series, 90))


def test_summarize_distributions_reports_expected_shape():
    df = _synthetic_df(300, seed=1)
    summary = ad.summarize_distributions(df)
    assert summary["n_rows"] == 300
    assert summary["min_distance_km"]["min"] <= summary["min_distance_km"]["mean"] <= summary["min_distance_km"]["max"]
    assert "p50" in summary["relative_velocity_km_s"]["percentiles"]


def test_calibrate_thresholds_finds_combination_within_target_range():
    # Independent uniform distance/velocity: dp=25 (P(X<=d)=0.25) and
    # vp=50 (P(Y>=v)=0.50) gives an expected joint rate of ~12.5%, squarely
    # inside the default [2%, 15%] target -- with n=3000 the empirical rate
    # should land close enough to that for the search grid to find it.
    df = _synthetic_df(3000, seed=2)
    calibration = ad.calibrate_thresholds(df)

    assert calibration["reliable"] is True
    assert calibration["within_target"] is True
    assert 0.02 <= calibration["achieved_positive_rate"] <= 0.15
    assert calibration["label_threshold_km"] > 0
    assert calibration["label_relative_velocity_km_s"] > 0
    assert "rationale" in calibration and str(calibration["n_rows"]) in calibration["rationale"]


def test_calibrate_thresholds_flags_small_datasets_as_unreliable():
    df = _synthetic_df(10, seed=3)
    calibration = ad.calibrate_thresholds(df)

    assert calibration["n_rows"] == 10
    assert calibration["reliable"] is False
    assert "WARNING" in calibration["rationale"]


def test_calibrate_thresholds_handles_zero_achievable_positive_rate():
    # Anti-correlated by construction: the 50 "close" pairs are always slow
    # and the 50 "fast" pairs are always far, so no row is ever both
    # close AND fast -- distance<=d AND velocity>=v can never both hold for
    # any grid combination. Must not crash and must honestly report
    # within_target=False instead of pretending a threshold works.
    df = pd.DataFrame(
        {
            "min_distance_km": np.concatenate([np.full(50, 10.0), np.full(50, 1000.0)]),
            "relative_velocity_km_s": np.concatenate([np.full(50, 0.1), np.full(50, 20.0)]),
        }
    )
    calibration = ad.calibrate_thresholds(df, target_low=0.5, target_high=0.9)

    assert calibration["within_target"] is False
    assert calibration["achieved_positive_rate"] == pytest.approx(0.0)


def test_plot_distributions_writes_two_png_files(tmp_path):
    df = _synthetic_df(50, seed=4)
    calibration = ad.calibrate_thresholds(df)
    outputs = ad.plot_distributions(df, tmp_path / "plots", calibration)

    assert len(outputs) == 2
    for path in outputs:
        assert path.exists()
        assert path.stat().st_size > 0


def test_write_calibration_report_is_a_separate_regeneratable_artifact(tmp_path):
    df = _synthetic_df(50, seed=5)
    summary = ad.summarize_distributions(df)
    calibration = ad.calibrate_thresholds(df)
    report_path = tmp_path / "threshold_calibration.json"
    history_path = tmp_path / "history.csv"

    ad.write_calibration_report(report_path, history_path, summary, calibration)

    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert written["source_history"] == str(history_path)
    assert written["calibration"] == calibration
    assert written["distribution_summary"] == summary
    assert "generated_utc" in written and written["generated_utc"].endswith("Z")
    assert "how_to_apply" in written
    # The mechanism must never silently mutate the hand-maintained config.
    assert not (tmp_path / "config" / "experiment_60_days.json").exists()


def test_end_to_end_on_synthetic_demo_pairs(tmp_path):
    from make_demo_tles import generate
    from space_debris.core import TleObject, build_satellites, simulate_pairs, write_pair_results

    blocks = generate(count=20, seed=7)
    objects = []
    for block in blocks:
        name, line1, line2 = block.splitlines()
        objects.append(TleObject(name, line1, line2))
    satellites = build_satellites(objects)

    rows = simulate_pairs(
        satellites=satellites,
        start_utc=datetime(2026, 7, 1, tzinfo=timezone.utc),
        horizon_minutes=60,
        step_minutes=10,
        leo_min_altitude_km=160.0,
        leo_max_altitude_km=2000.0,
        fixed_threshold_km=25.0,
        label_threshold_km=20.0,
        label_relative_velocity_km_s=10.0,
    )
    assert rows, "synthetic demo catalogue should yield simulated pairs"

    dataset_path = tmp_path / "dataset.csv"
    write_pair_results(dataset_path, rows)

    df = ad.load_history(dataset_path)
    summary = ad.summarize_distributions(df)
    calibration = ad.calibrate_thresholds(df)
    plots = ad.plot_distributions(df, tmp_path / "plots", calibration)

    assert summary["n_rows"] == len(rows)
    assert "label_threshold_km" in calibration
    assert len(plots) == 2
    for path in plots:
        assert path.exists()
