"""Unit tests for space_debris.plots: provenance-embedded PNGs and the four
publication-ready figures (ROADMAP_YOL1.md GOREV 6).

Run with:  PYTHONPATH=src python -m pytest tests/ -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from space_debris import plots
from space_debris.ml import compare_models


def _synthetic_dataset(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
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
        }
    )


def _png_info(path) -> dict:
    with Image.open(path) as img:
        return dict(img.info)


def test_savefig_embeds_provenance_metadata_and_dpi(tmp_path):
    import matplotlib.pyplot as plt

    path = tmp_path / "fig.png"
    plt.figure()
    plt.plot([1, 2, 3])
    plots._savefig(path, dpi=300)

    info = _png_info(path)
    assert "git_commit=" in info["Description"]
    assert info["Software"] == "space-debris-ml-simulation"
    dpi = info.get("dpi")
    assert dpi is not None and round(dpi[0]) == 300


def test_publication_plots_render_on_balanced_dataset(tmp_path):
    df = _synthetic_dataset(200, seed=1)
    dataset_path = tmp_path / "dataset.csv"
    df.to_csv(dataset_path, index=False)

    outputs = plots.create_publication_plots(dataset_path, tmp_path, current_threshold_km=25.0)

    assert len(outputs) == 4
    for path in outputs:
        assert path.exists()
        assert path.stat().st_size > 0
        dpi = _png_info(path).get("dpi")
        assert dpi is not None and round(dpi[0]) == 300


def test_publication_plots_skip_gracefully_on_single_class_dataset(tmp_path):
    df = _synthetic_dataset(20, seed=2)
    df["risk_label"] = 0  # force a single class -> nothing trainable/plottable
    dataset_path = tmp_path / "dataset.csv"
    df.to_csv(dataset_path, index=False)

    outputs = plots.create_publication_plots(dataset_path, tmp_path)

    # Must not crash; simply produces no publication figures for this dataset.
    assert outputs == []


def test_plot_model_metrics_reads_report_with_provenance_header(tmp_path):
    df = _synthetic_dataset(200, seed=3)
    dataset_path = tmp_path / "dataset.csv"
    df.to_csv(dataset_path, index=False)
    report_path = tmp_path / "report.csv"

    # compare_models() embeds a '#' provenance header in report_path; this is
    # a regression test that plot_model_metrics() still parses it correctly.
    compare_models(dataset_path, report_path, source="unit-test", config_summary="n=200")
    assert report_path.read_text(encoding="utf-8").splitlines()[0].startswith("#")

    output_path = tmp_path / "model_metrics.png"
    plots.plot_model_metrics(report_path, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
