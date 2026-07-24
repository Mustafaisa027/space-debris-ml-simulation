"""End-to-end smoke test for the one-shot pipeline entry point.

`run_pipeline.main()` is the first command the README shows, yet no test
actually executed it -- a wiring regression (bad import, renamed argument,
plotting break) could ship undetected. This test drives the real entry point
on a small deterministic synthetic catalogue and asserts the documented
artifacts are produced. It is intentionally an integration smoke test, not a
numerical check: the frozen scientific claims live in the evidence-chain tests.
"""

from __future__ import annotations

import pandas as pd
import pytest

import make_demo_tles
import run_pipeline


def _write_demo_catalogue(path, count=30, seed=20260703):
    blocks = make_demo_tles.generate(count, seed)
    path.write_text("\n".join(blocks) + "\n", encoding="utf-8")


@pytest.mark.parametrize("feature_set", ["snapshot_only", "full_rule_recovery"])
def test_run_pipeline_end_to_end(tmp_path, monkeypatch, feature_set):
    tle_path = tmp_path / "demo.txt"
    out_dir = tmp_path / "pipeline"
    _write_demo_catalogue(tle_path)

    argv = [
        "run_pipeline.py",
        "--tle", str(tle_path),
        "--outputs", str(out_dir),
        "--horizon-minutes", "120",
        "--step-minutes", "3",
        "--candidate-threshold-km", "500",
        "--fixed-threshold-km", "50",
        "--label-threshold-km", "200",
        "--label-relative-velocity-km-s", "5",
        # Suppress the stale-epoch warning for the deterministic synthetic demo.
        "--max-tle-age-hours", "100000",
        "--feature-set", feature_set,
    ]
    monkeypatch.setattr("sys.argv", argv)

    run_pipeline.main()

    # Documented CSV artifacts exist and are non-empty.
    for name in (
        "conjunction_dataset.csv",
        "identified_conjunctions.csv",
        "model_comparison.csv",
    ):
        artifact = out_dir / name
        assert artifact.exists(), f"missing {name}"
        assert artifact.stat().st_size > 0, f"empty {name}"

    # At least one scientific figure was rendered.
    pngs = list(out_dir.glob("*.png"))
    assert pngs, "no figures were produced"

    # The four abstract-required classifiers plus the fixed baseline are all
    # present in the comparison report.
    report = pd.read_csv(out_dir / "model_comparison.csv", comment="#")
    models = set(report["model"])
    required = {
        "fixed_threshold",
        "logistic_regression",
        "random_forest",
        "svm",
        "xgboost",
    }
    assert required.issubset(models), f"missing models: {required - models}"

    # Identified conjunctions must carry the abstract's three core risk metrics.
    conjunctions = pd.read_csv(out_dir / "identified_conjunctions.csv", comment="#")
    assert not conjunctions.empty
    for column in ("min_distance_km", "tca_utc", "relative_velocity_km_s"):
        assert column in conjunctions.columns, f"missing risk metric {column}"
