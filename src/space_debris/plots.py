from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta, timezone

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from skyfield.api import load

from space_debris.ml import (
    FEATURES,
    InsufficientGroupedSplitError,
    _split_data,
    model_predictions_for_plotting,
)
from space_debris.provenance import png_provenance_metadata

EARTH_RADIUS_KM = 6378.137

# Every output CSV written by core.write_pair_results / ml.compare_models may
# carry a '#'-prefixed provenance header (ROADMAP_YOL1.md GOREV 6); pandas
# must be told to skip those lines rather than mis-parse one as the header row.
_CSV_KWARGS = {"comment": "#"}


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _savefig(path: Path, dpi: int = 180) -> None:
    """Save the current figure with embedded PNG provenance metadata
    (git commit + generation timestamp), then close it.
    """
    plt.savefig(path, dpi=dpi, metadata=png_provenance_metadata())
    plt.close()


def plot_risk_scatter(dataset_path: Path, output_path: Path) -> None:
    df = pd.read_csv(dataset_path, **_CSV_KWARGS)
    if df.empty:
        return

    _ensure_dir(output_path)
    colors = np.where(df["risk_label"].astype(int) == 1, "#b33a3a", "#2f5d8c")
    plt.figure(figsize=(9, 6))
    plt.scatter(
        df["min_distance_km"],
        df["relative_velocity_km_s"],
        c=colors,
        s=80,
        edgecolor="#222222",
        linewidth=0.6,
        alpha=0.88,
    )
    top_rows = df.sort_values("risk_score", ascending=False).head(3)
    for _, row in top_rows.iterrows():
        label = f"{row['object_1']} / {row['object_2']}"
        plt.annotate(
            label,
            (row["min_distance_km"], row["relative_velocity_km_s"]),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=8,
        )
    plt.xlabel("Minimum approach distance (km)")
    plt.ylabel("Relative velocity at TCA (km/s)")
    plt.title(f"Conjunction Risk Feature Space (n={len(df)})")
    plt.grid(True, alpha=0.35)
    risky = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#b33a3a", markeredgecolor="#222222", label="Proxy risky")
    non_risky = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2f5d8c", markeredgecolor="#222222", label="Proxy non-risky")
    plt.legend(handles=[risky, non_risky], loc="best")
    plt.tight_layout()
    _savefig(output_path, dpi=180)


def plot_tca_distance(dataset_path: Path, output_path: Path, candidate_threshold_km: float) -> None:
    df = pd.read_csv(dataset_path, **_CSV_KWARGS)
    if df.empty:
        return

    _ensure_dir(output_path)
    colors = np.where(df["risk_label"].astype(int) == 1, "#c0362c", "#2f6f9f")
    plt.figure(figsize=(9, 6))
    plt.scatter(
        df["time_to_tca_min"],
        df["min_distance_km"],
        c=colors,
        s=70,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )
    plt.axhline(candidate_threshold_km, color="#555555", linestyle="--", label="candidate threshold")
    plt.xlabel("Time to closest approach (min)")
    plt.ylabel("Minimum approach distance (km)")
    plt.title(f"TCA vs Minimum Approach Distance (n={len(df)})")
    plt.grid(True, alpha=0.35)
    risky = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#c0362c", markeredgecolor="#222222", label="Proxy risky")
    non_risky = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2f6f9f", markeredgecolor="#222222", label="Proxy non-risky")
    threshold = plt.Line2D([0], [0], color="#555555", linestyle="--", label="candidate threshold")
    plt.legend(handles=[threshold, risky, non_risky], loc="best")
    plt.tight_layout()
    _savefig(output_path, dpi=180)


def plot_altitude_distance(dataset_path: Path, output_path: Path) -> None:
    df = pd.read_csv(dataset_path, **_CSV_KWARGS)
    if df.empty:
        return

    _ensure_dir(output_path)
    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(
        df["altitude_difference_km"],
        df["min_distance_km"],
        c=df["relative_velocity_km_s"],
        s=70,
        cmap="plasma",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )
    plt.xlabel("Altitude difference at snapshot (km)")
    plt.ylabel("Minimum approach distance (km)")
    plt.title("Orbital Geometry: Altitude Difference vs Approach Distance")
    plt.grid(True, alpha=0.35)
    cbar = plt.colorbar(scatter)
    cbar.set_label("Relative velocity (km/s)")
    plt.tight_layout()
    _savefig(output_path, dpi=180)


def plot_risk_heatmap(dataset_path: Path, output_path: Path) -> None:
    df = pd.read_csv(dataset_path, **_CSV_KWARGS)
    if len(df) < 30:
        if output_path.exists():
            output_path.unlink()
        return

    _ensure_dir(output_path)
    x = df["time_to_tca_min"].to_numpy(dtype=float)
    y = df["min_distance_km"].to_numpy(dtype=float)
    weights = df["risk_score"].to_numpy(dtype=float)
    bins_x = min(12, max(3, len(df) // 2))
    bins_y = min(12, max(3, len(df) // 2))
    heat, x_edges, y_edges = np.histogram2d(x, y, bins=[bins_x, bins_y], weights=weights)
    counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    mean_heat = np.divide(heat, counts, out=np.zeros_like(heat), where=counts > 0)

    plt.figure(figsize=(9, 6))
    image = plt.imshow(
        mean_heat.T,
        origin="lower",
        aspect="auto",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        cmap="magma",
    )
    plt.xlabel("Time to closest approach (min)")
    plt.ylabel("Minimum approach distance (km)")
    plt.title("Mean Risk Score Density")
    cbar = plt.colorbar(image)
    cbar.set_label("Mean risk score")
    plt.tight_layout()
    _savefig(output_path, dpi=180)


def plot_risk_ranking(dataset_path: Path, output_path: Path, top_n: int = 10) -> None:
    df = pd.read_csv(dataset_path, **_CSV_KWARGS)
    if df.empty:
        return

    top = df.sort_values("risk_score", ascending=True).tail(top_n).copy()
    labels = [f"{row.object_1} / {row.object_2}" for row in top.itertuples()]
    _ensure_dir(output_path)
    plt.figure(figsize=(10, 6))
    bars = plt.barh(labels, top["risk_score"], color="#315f72", edgecolor="#1d3641")
    x_max = max(float(top["risk_score"].max()) * 1.35, 0.001)
    plt.xlim(0, x_max)
    for bar, row in zip(bars, top.itertuples()):
        text = f"d={row.min_distance_km:.0f} km, v={row.relative_velocity_km_s:.1f} km/s"
        plt.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            "  " + text,
            va="center",
            fontsize=8,
        )
    plt.xlabel("Proxy risk score")
    plt.title(f"Top Conjunction Candidates by Risk Score (top {len(top)})")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _savefig(output_path, dpi=180)


def plot_model_metrics(report_path: Path, output_path: Path) -> None:
    df = pd.read_csv(report_path, **_CSV_KWARGS)
    if df.empty or "model" not in df:
        return

    # A "not_enough_data" report has no numeric metrics to plot.
    if "note" in df.columns and df["model"].astype(str).eq("not_enough_data").any():
        _ensure_dir(output_path)
        plt.figure(figsize=(9, 5))
        plt.axis("off")
        note = str(df["note"].dropna().iloc[0]) if df["note"].notna().any() else \
            "Not enough data to train models."
        plt.text(
            0.5, 0.5,
            "Model comparison unavailable\n" + note,
            ha="center", va="center", fontsize=12, wrap=True,
        )
        plt.tight_layout()
        _savefig(output_path, dpi=180)
        return

    # pr_auc/roc_auc lead; accuracy trails since it is not the headline
    # metric under the dataset's severe class imbalance.
    metrics = ["pr_auc", "roc_auc", "precision", "recall", "f1", "accuracy"]
    available = [metric for metric in metrics if metric in df.columns]
    plot_df = df[["model"] + available].copy()
    for metric in available:
        plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce").fillna(0.0)

    _ensure_dir(output_path)
    x = np.arange(len(plot_df))
    width = 0.18
    train_rows = int(pd.to_numeric(df.get("train_rows", pd.Series([0])), errors="coerce").fillna(0).max() or 0)
    test_rows = int(pd.to_numeric(df.get("test_rows", pd.Series([0])), errors="coerce").fillna(0).max() or 0)
    plt.figure(figsize=(11, 6))
    for index, metric in enumerate(available):
        offset = (index - (len(available) - 1) / 2) * width
        plt.bar(x + offset, plot_df[metric], width=width, label=metric)

    plt.xticks(x, plot_df["model"], rotation=25, ha="right")
    plt.ylim(0, 1.08)
    plt.ylabel("Score")
    plt.title(f"Model Performance Comparison (preliminary, train={train_rows}, test={test_rows})")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(ncol=4)
    plt.tight_layout()
    _savefig(output_path, dpi=180)


def _to_skyfield_time(ts, dt: datetime):
    dt = dt.astimezone(timezone.utc)
    return ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)


def _norm_rows(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(values * values, axis=1))


def plot_top_pair_eci_trajectory(
    sat1,
    sat2,
    start_utc: datetime,
    horizon_minutes: int,
    step_minutes: int,
    output_path: Path,
) -> None:
    _ensure_dir(output_path)
    ts = load.timescale()
    offsets = np.arange(0, horizon_minutes + 1, step_minutes)
    p1 = []
    p2 = []
    distances = []
    for offset in offsets:
        dt = start_utc + timedelta(minutes=int(offset))
        t = _to_skyfield_time(ts, dt)
        pos1 = np.array(sat1.at(t).position.km)
        pos2 = np.array(sat2.at(t).position.km)
        p1.append(pos1)
        p2.append(pos2)
        distances.append(np.linalg.norm(pos1 - pos2))
    p1 = np.array(p1)
    p2 = np.array(p2)
    distances = np.array(distances)
    tca_idx = int(np.argmin(distances))

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    u = np.linspace(0, 2 * np.pi, 48)
    v = np.linspace(0, np.pi, 24)
    x = EARTH_RADIUS_KM * np.outer(np.cos(u), np.sin(v))
    y = EARTH_RADIUS_KM * np.outer(np.sin(u), np.sin(v))
    z = EARTH_RADIUS_KM * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color="#d8e4ea", alpha=0.45, linewidth=0)
    ax.plot(p1[:, 0], p1[:, 1], p1[:, 2], color="#2f5d8c", linewidth=1.8, label=sat1.name)
    ax.plot(p2[:, 0], p2[:, 1], p2[:, 2], color="#b33a3a", linewidth=1.8, label=sat2.name)
    ax.scatter(p1[tca_idx, 0], p1[tca_idx, 1], p1[tca_idx, 2], color="#2f5d8c", s=55)
    ax.scatter(p2[tca_idx, 0], p2[tca_idx, 1], p2[tca_idx, 2], color="#b33a3a", s=55)
    ax.plot(
        [p1[tca_idx, 0], p2[tca_idx, 0]],
        [p1[tca_idx, 1], p2[tca_idx, 1]],
        [p1[tca_idx, 2], p2[tca_idx, 2]],
        color="#222222",
        linestyle="--",
        linewidth=1.2,
        label="separation at TCA",
    )
    limit = max(np.max(np.abs(p1)), np.max(np.abs(p2)), EARTH_RADIUS_KM) * 1.08
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_xlabel("ECI x (km)")
    ax.set_ylabel("ECI y (km)")
    ax.set_zlabel("ECI z (km)")
    ax.set_title("Top Candidate ECI Trajectories")
    ax.legend(loc="upper left")
    ax.view_init(elev=24, azim=38)
    plt.tight_layout()
    _savefig(output_path, dpi=180)


def plot_top_pair_altitudes(
    sat1,
    sat2,
    start_utc: datetime,
    horizon_minutes: int,
    step_minutes: int,
    output_path: Path,
) -> None:
    _ensure_dir(output_path)
    ts = load.timescale()
    offsets = np.arange(0, horizon_minutes + 1, step_minutes)
    alt1 = []
    alt2 = []
    distances = []
    for offset in offsets:
        dt = start_utc + timedelta(minutes=int(offset))
        t = _to_skyfield_time(ts, dt)
        pos1 = np.array(sat1.at(t).position.km)
        pos2 = np.array(sat2.at(t).position.km)
        alt1.append(np.linalg.norm(pos1) - EARTH_RADIUS_KM)
        alt2.append(np.linalg.norm(pos2) - EARTH_RADIUS_KM)
        distances.append(np.linalg.norm(pos1 - pos2))
    tca_offset = offsets[int(np.argmin(distances))]

    plt.figure(figsize=(10, 5))
    plt.plot(offsets, alt1, color="#2f5d8c", linewidth=1.8, label=sat1.name)
    plt.plot(offsets, alt2, color="#b33a3a", linewidth=1.8, label=sat2.name)
    plt.axvline(tca_offset, color="#222222", linestyle="--", linewidth=1.1, label="TCA")
    plt.xlabel("Minutes from snapshot")
    plt.ylabel("Altitude (km)")
    plt.title("Top Candidate Altitude Evolution")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    _savefig(output_path, dpi=180)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def _relative_samples(sat1, sat2, start_utc: datetime, horizon_minutes: int, step_minutes: int):
    ts = load.timescale()
    offsets = np.arange(0, horizon_minutes + 1, step_minutes)
    p1 = []
    p2 = []
    v1 = []
    v2 = []
    for offset in offsets:
        dt = start_utc + timedelta(minutes=int(offset))
        t = _to_skyfield_time(ts, dt)
        state1 = sat1.at(t)
        state2 = sat2.at(t)
        p1.append(np.array(state1.position.km))
        p2.append(np.array(state2.position.km))
        v1.append(np.array(state1.velocity.km_per_s))
        v2.append(np.array(state2.velocity.km_per_s))
    p1 = np.array(p1)
    p2 = np.array(p2)
    v1 = np.array(v1)
    v2 = np.array(v2)
    rel_r = p2 - p1
    rel_v = v2 - v1
    distances = _norm_rows(rel_r)
    tca_idx = int(np.argmin(distances))
    return offsets, p1, p2, v1, v2, rel_r, rel_v, distances, tca_idx


def plot_encounter_plane(
    sat1,
    sat2,
    start_utc: datetime,
    horizon_minutes: int,
    step_minutes: int,
    output_path: Path,
    window_minutes: int = 120,
) -> None:
    _ensure_dir(output_path)
    offsets, p1, _p2, v1, _v2, rel_r, rel_v, distances, tca_idx = _relative_samples(
        sat1, sat2, start_utc, horizon_minutes, step_minutes
    )
    tca_offset = offsets[tca_idx]
    mask = np.abs(offsets - tca_offset) <= window_minutes

    v_hat = _unit(rel_v[tca_idx])
    radial_hat = _unit(p1[tca_idx])
    xi_hat = np.cross(v_hat, radial_hat)
    if np.linalg.norm(xi_hat) < 1e-9:
        xi_hat = np.cross(v_hat, _unit(v1[tca_idx]))
    xi_hat = _unit(xi_hat)
    zeta_hat = _unit(np.cross(v_hat, xi_hat))

    xi = rel_r[mask] @ xi_hat
    zeta = rel_r[mask] @ zeta_hat
    plot_offsets = offsets[mask] - tca_offset
    tca_xi = rel_r[tca_idx] @ xi_hat
    tca_zeta = rel_r[tca_idx] @ zeta_hat

    plt.figure(figsize=(8, 7))
    scatter = plt.scatter(
        xi,
        zeta,
        c=plot_offsets,
        cmap="coolwarm",
        s=42,
        edgecolor="#222222",
        linewidth=0.4,
        zorder=3,
    )
    plt.plot(xi, zeta, color="#555555", linewidth=1.0, alpha=0.7, zorder=2)
    plt.scatter([tca_xi], [tca_zeta], color="#111111", marker="x", s=90, label="TCA")
    plt.axhline(0, color="#aaaaaa", linewidth=0.8)
    plt.axvline(0, color="#aaaaaa", linewidth=0.8)
    plt.xlabel("Encounter-plane xi component (km)")
    plt.ylabel("Encounter-plane zeta component (km)")
    plt.title(f"Encounter Plane Projection (+/- {window_minutes} min around TCA)")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Minutes from TCA")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.legend(loc="best")
    plt.tight_layout()
    _savefig(output_path, dpi=180)


def plot_relative_ric_components(
    sat1,
    sat2,
    start_utc: datetime,
    horizon_minutes: int,
    step_minutes: int,
    output_path: Path,
    window_minutes: int = 120,
) -> None:
    _ensure_dir(output_path)
    offsets, p1, _p2, v1, _v2, rel_r, _rel_v, distances, tca_idx = _relative_samples(
        sat1, sat2, start_utc, horizon_minutes, step_minutes
    )
    tca_offset = offsets[tca_idx]
    mask = np.abs(offsets - tca_offset) <= window_minutes
    minutes_from_tca = offsets[mask] - tca_offset

    radial = []
    intrack = []
    crosstrack = []
    for idx in np.where(mask)[0]:
        r_hat = _unit(p1[idx])
        h_hat = _unit(np.cross(p1[idx], v1[idx]))
        i_hat = _unit(np.cross(h_hat, r_hat))
        radial.append(rel_r[idx] @ r_hat)
        intrack.append(rel_r[idx] @ i_hat)
        crosstrack.append(rel_r[idx] @ h_hat)

    plt.figure(figsize=(10, 6))
    plt.plot(minutes_from_tca, radial, label="Radial", color="#2f5d8c", linewidth=1.8)
    plt.plot(minutes_from_tca, intrack, label="In-track", color="#6b8f3e", linewidth=1.8)
    plt.plot(minutes_from_tca, crosstrack, label="Cross-track", color="#b33a3a", linewidth=1.8)
    plt.axvline(0, color="#222222", linestyle="--", linewidth=1.1, label="TCA")
    plt.xlabel("Minutes from TCA")
    plt.ylabel("Relative position component (km)")
    plt.title("Relative Position in Local RIC Frame")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    _savefig(output_path, dpi=180)


def create_top_pair_physical_plots(
    sat1,
    sat2,
    start_utc: datetime,
    horizon_minutes: int,
    step_minutes: int,
    output_dir: Path,
) -> list[Path]:
    outputs = [
        output_dir / "top_pair_eci_trajectory.png",
        output_dir / "top_pair_altitude_evolution.png",
        output_dir / "top_pair_encounter_plane.png",
        output_dir / "top_pair_relative_ric.png",
    ]
    plot_top_pair_eci_trajectory(sat1, sat2, start_utc, horizon_minutes, step_minutes, outputs[0])
    plot_top_pair_altitudes(sat1, sat2, start_utc, horizon_minutes, step_minutes, outputs[1])
    plot_encounter_plane(sat1, sat2, start_utc, horizon_minutes, step_minutes, outputs[2])
    plot_relative_ric_components(sat1, sat2, start_utc, horizon_minutes, step_minutes, outputs[3])
    return [path for path in outputs if path.exists()]


def create_pipeline_plots(
    dataset_path: Path,
    model_report_path: Path,
    output_dir: Path,
    candidate_threshold_km: float,
) -> list[Path]:
    outputs = [
        output_dir / "risk_feature_space.png",
        output_dir / "tca_distance_scatter.png",
        output_dir / "altitude_distance_geometry.png",
        output_dir / "risk_ranking.png",
        output_dir / "model_metrics.png",
        output_dir / "risk_density_heatmap.png",
    ]
    plot_risk_scatter(dataset_path, outputs[0])
    plot_tca_distance(dataset_path, outputs[1], candidate_threshold_km)
    plot_altitude_distance(dataset_path, outputs[2])
    plot_risk_ranking(dataset_path, outputs[3])
    plot_model_metrics(model_report_path, outputs[4])
    plot_risk_heatmap(dataset_path, outputs[5])
    return [path for path in outputs if path.exists()]


# --- Publication-ready figures (ROADMAP_YOL1.md GOREV 6) ---------------
# 300 DPI, explicit titles/axis labels, one clear question each: does ML
# beat the baseline (PR curves), how false alarms trade against recall
# (operating-characteristic curves), where do the errors land (confusion
# matrices), which features drive the tree models (feature importance), and
# how arbitrary is the classical fixed-distance operating point (threshold
# sensitivity)? These re-fit models purely for plotting via
# space_debris.ml.model_predictions_for_plotting(); compare_models() /
# time_series_cv_report() remain the source of truth for the numeric report.


def plot_pr_curves(
    dataset_path: Path, output_path: Path, time_column: str | None = None, **split_kwargs
) -> None:
    predictions = model_predictions_for_plotting(
        dataset_path, time_column=time_column, **split_kwargs
    )
    usable = {name: p for name, p in predictions.items() if len(np.unique(p["y_true"])) == 2}
    if not usable:
        return

    _ensure_dir(output_path)
    plt.figure(figsize=(8, 6))
    for name, p in usable.items():
        precision, recall, _ = precision_recall_curve(p["y_true"], p["scores"])
        ap = average_precision_score(p["y_true"], p["scores"])
        plt.plot(recall, precision, linewidth=1.8, label=f"{name} (AP={ap:.3f})")
    baseline_rate = float(np.mean(next(iter(usable.values()))["y_true"]))
    plt.axhline(
        baseline_rate, color="#888888", linestyle=":", linewidth=1.3,
        label=f"no-skill (positive rate={baseline_rate:.3f})",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves by Model (test split)")
    plt.xlim(0.0, 1.02)
    plt.ylim(0.0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    _savefig(output_path, dpi=300)


def plot_false_alarm_recall_curves(
    dataset_path: Path, output_path: Path, time_column: str | None = None, **split_kwargs
) -> None:
    """Plot recall against false alarm rate over every score threshold.

    This is the operating-characteristic view needed for a fair false-alarm
    claim: models can be compared at the same recall without choosing a new
    decision threshold from the held-out test labels.  The fixed-distance
    baseline contributes its continuous ``-min_distance_km`` ranking here;
    its configured binary operating point remains in the confusion matrices.
    """
    predictions = model_predictions_for_plotting(
        dataset_path, time_column=time_column, **split_kwargs
    )
    usable = {name: p for name, p in predictions.items() if len(np.unique(p["y_true"])) == 2}
    if not usable:
        return

    _ensure_dir(output_path)
    plt.figure(figsize=(8, 6))
    for name, p in usable.items():
        false_alarm_rate, recall, _ = roc_curve(p["y_true"], p["scores"])
        auc = roc_auc_score(p["y_true"], p["scores"])
        plt.plot(
            false_alarm_rate,
            recall,
            linewidth=1.8,
            label=f"{name} (AUC={auc:.3f})",
        )
    plt.plot([0.0, 1.0], [0.0, 1.0], color="#888888", linestyle=":", linewidth=1.3)
    plt.xlabel("False alarm rate, FP / (FP + TN)")
    plt.ylabel("Recall")
    plt.title("False Alarm Rate vs Recall (held-out test split)")
    plt.xlim(-0.01, 1.01)
    plt.ylim(-0.01, 1.01)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    _savefig(output_path, dpi=300)


def plot_confusion_matrices(
    dataset_path: Path, output_path: Path, time_column: str | None = None, **split_kwargs
) -> None:
    predictions = model_predictions_for_plotting(
        dataset_path, time_column=time_column, **split_kwargs
    )
    usable = {name: p for name, p in predictions.items() if len(np.unique(p["y_true"])) == 2}
    if not usable:
        return

    _ensure_dir(output_path)
    n = len(usable)
    cols = min(3, n)
    rows = -(-n // cols)  # ceil division
    fig, axes = plt.subplots(rows, cols, figsize=(4.3 * cols, 4.0 * rows), squeeze=False)
    class_labels = ["not risky", "risky"]
    for idx, (name, p) in enumerate(usable.items()):
        ax = axes[idx // cols][idx % cols]
        cm = confusion_matrix(p["y_true"], p["pred"], labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        peak = cm.max() if cm.max() > 0 else 1
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > peak / 2 else "black", fontsize=11,
                )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(class_labels)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(class_labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(name)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    fig.suptitle("Confusion Matrices by Model (test split)", fontsize=13)
    plt.tight_layout()
    _savefig(output_path, dpi=300)


def plot_feature_importance(
    dataset_path: Path, output_path: Path, time_column: str | None = None, **split_kwargs
) -> None:
    predictions = model_predictions_for_plotting(
        dataset_path, time_column=time_column, **split_kwargs
    )
    importances: dict[str, np.ndarray] = {}
    for name in ("decision_tree", "random_forest", "xgboost", "lightgbm"):
        entry = predictions.get(name)
        if entry and entry["model"] is not None and hasattr(entry["model"], "feature_importances_"):
            importances[name] = np.asarray(entry["model"].feature_importances_)
    if not importances:
        return

    _ensure_dir(output_path)
    n_features = len(FEATURES)
    y_pos = np.arange(n_features)
    n_models = len(importances)
    bar_height = 0.8 / n_models
    plt.figure(figsize=(9, 0.45 * n_features + 2))
    for i, (name, values) in enumerate(importances.items()):
        offset = (i - (n_models - 1) / 2) * bar_height
        plt.barh(y_pos + offset, values, height=bar_height, label=name)
    plt.yticks(y_pos, FEATURES)
    plt.gca().invert_yaxis()
    plt.xlabel("Feature importance (impurity/gain-based)")
    plt.title("Feature Importance: Tree-Based Models")
    plt.legend(loc="lower right")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _savefig(output_path, dpi=300)


def plot_threshold_sensitivity(
    dataset_path: Path,
    output_path: Path,
    current_threshold_km: float | None = None,
    time_column: str | None = None,
    **split_kwargs,
) -> None:
    df = pd.read_csv(dataset_path, **_CSV_KWARGS)
    if df.empty or "risk_label" not in df.columns:
        return
    try:
        df = _split_data(df, time_column, **split_kwargs).test
    except InsufficientGroupedSplitError:
        return
    y_true = df["risk_label"].astype(int)
    if y_true.nunique() < 2:
        return

    max_thresh = max(float(df["min_distance_km"].quantile(0.75)), 1.0)
    thresholds = np.linspace(0.0, max_thresh, 60)
    precisions, recalls, f1s = [], [], []
    for thresh in thresholds:
        y_pred = (df["min_distance_km"] <= thresh).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    _ensure_dir(output_path)
    plt.figure(figsize=(9, 6))
    plt.plot(thresholds, precisions, label="Precision", linewidth=1.8, color="#2f5d8c")
    plt.plot(thresholds, recalls, label="Recall", linewidth=1.8, color="#6b8f3e")
    plt.plot(thresholds, f1s, label="F1", linewidth=1.8, color="#b33a3a")
    if current_threshold_km is not None:
        plt.axvline(
            current_threshold_km, color="#222222", linestyle="--", linewidth=1.2,
            label=f"current fixed_threshold_km={current_threshold_km:.1f}",
        )
    plt.xlabel("Candidate fixed-distance threshold (km)")
    plt.ylabel("Score")
    plt.title("Fixed-Distance Baseline: Threshold Sensitivity")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    _savefig(output_path, dpi=300)


def create_publication_plots(
    dataset_path: Path,
    output_dir: Path,
    time_column: str | None = None,
    current_threshold_km: float | None = None,
    train_time_fraction: float = 0.75,
    test_pair_fraction: float = 0.25,
    pair_seed: str = "iac26-pair-split-v1",
) -> list[Path]:
    outputs = [
        output_dir / "pub_pr_curves.png",
        output_dir / "pub_false_alarm_recall_curves.png",
        output_dir / "pub_confusion_matrices.png",
        output_dir / "pub_feature_importance.png",
        output_dir / "pub_threshold_sensitivity.png",
    ]
    split_kwargs = {
        "train_time_fraction": train_time_fraction,
        "test_pair_fraction": test_pair_fraction,
        "pair_seed": pair_seed,
    }
    plot_pr_curves(dataset_path, outputs[0], time_column=time_column, **split_kwargs)
    plot_false_alarm_recall_curves(
        dataset_path, outputs[1], time_column=time_column, **split_kwargs
    )
    plot_confusion_matrices(dataset_path, outputs[2], time_column=time_column, **split_kwargs)
    plot_feature_importance(dataset_path, outputs[3], time_column=time_column, **split_kwargs)
    plot_threshold_sensitivity(
        dataset_path,
        outputs[4],
        current_threshold_km=current_threshold_km,
        time_column=time_column,
        **split_kwargs,
    )
    return [path for path in outputs if path.exists()]
