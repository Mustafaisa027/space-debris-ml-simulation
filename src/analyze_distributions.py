"""Analyze accumulated conjunction history and suggest physically-defensible
label thresholds (ROADMAP_YOL1.md GOREV 3).

This intentionally does NOT hardcode a threshold into
config/experiment_60_days.json. With only a few hours of leo_mixed
collection, conjunction counts are still ~0, so any threshold "calibrated"
today would be noise, not physics. Instead this script is a *mechanism*:
rerun it as collection history accumulates, inspect the generated report
(config/threshold_calibration.json by default), and only then hand-edit the
experiment config, citing the report's rationale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from space_debris.provenance import generated_utc, git_commit_hash

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["min_distance_km", "relative_velocity_km_s"]

# Grid searched by calibrate_thresholds(). Distance percentiles are read from
# the LOW end (closest approaches); velocity percentiles from the HIGH end
# (fastest approaches), since risk_label requires both close AND fast.
DEFAULT_DISTANCE_PERCENTILES = (1, 2, 5, 10, 15, 20, 25, 30, 40, 50)
DEFAULT_VELOCITY_PERCENTILES = (50, 60, 70, 75, 80, 85, 90, 95, 99)

# Below this many rows, a calibration is reported but flagged unreliable --
# it describes a statistical fluke, not the shape of the true distribution.
MIN_ROWS_FOR_RELIABLE_CALIBRATION = 200
DEFAULT_MAX_PHYSICAL_DISTANCE_KM = 50.0


def load_history(path: Path) -> pd.DataFrame:
    # comment="#" tolerates a provenance header (ROADMAP_YOL1.md GOREV 6,
    # space_debris.provenance) if one is ever present on this file.
    df = pd.read_csv(path, comment="#")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if df.empty:
        raise ValueError(f"{path} has no rows to analyze; run collect_observations.py first.")
    return df


def percentile_summary(series: pd.Series, percentiles=(1, 5, 10, 25, 50, 75, 90, 95, 99)) -> dict[str, float]:
    return {f"p{p}": float(np.percentile(series, p)) for p in percentiles}


def summarize_distributions(df: pd.DataFrame) -> dict:
    distance = df["min_distance_km"]
    velocity = df["relative_velocity_km_s"]
    return {
        "n_rows": int(len(df)),
        "min_distance_km": {
            "min": float(distance.min()),
            "max": float(distance.max()),
            "mean": float(distance.mean()),
            "std": float(distance.std()) if len(distance) > 1 else 0.0,
            "percentiles": percentile_summary(distance),
        },
        "relative_velocity_km_s": {
            "min": float(velocity.min()),
            "max": float(velocity.max()),
            "mean": float(velocity.mean()),
            "std": float(velocity.std()) if len(velocity) > 1 else 0.0,
            "percentiles": percentile_summary(velocity),
        },
    }


def calibrate_thresholds(
    df: pd.DataFrame,
    target_low: float = 0.02,
    target_high: float = 0.15,
    distance_percentiles=DEFAULT_DISTANCE_PERCENTILES,
    velocity_percentiles=DEFAULT_VELOCITY_PERCENTILES,
    max_distance_km: float = DEFAULT_MAX_PHYSICAL_DISTANCE_KM,
) -> dict:
    """Grid-search (label_threshold_km, label_relative_velocity_km_s) so that
    P(min_distance_km <= d AND relative_velocity_km_s >= v) lands inside
    [target_low, target_high] of THIS dataset.

    Both thresholds are derived from empirical percentiles of the observed
    mixed-orbit data rather than arbitrary constants -- that is what makes
    them physically/statistically defensible. No column here is the label
    itself, so this never touches risk_label or risk_score (no leakage).
    """
    n = len(df)
    distance = df["min_distance_km"].to_numpy()
    velocity = df["relative_velocity_km_s"].to_numpy()

    candidates = []
    for dp in distance_percentiles:
        d_thresh = min(float(np.percentile(distance, dp)), max_distance_km)
        for vp in velocity_percentiles:
            v_thresh = float(np.percentile(velocity, vp))
            positive_rate = float(np.mean((distance <= d_thresh) & (velocity >= v_thresh)))
            candidates.append(
                {
                    "distance_percentile": dp,
                    "velocity_percentile": vp,
                    "label_threshold_km": d_thresh,
                    "label_relative_velocity_km_s": v_thresh,
                    "positive_rate": positive_rate,
                }
            )

    midpoint = (target_low + target_high) / 2.0
    in_target = [c for c in candidates if target_low <= c["positive_rate"] <= target_high]
    if in_target:
        best = min(in_target, key=lambda c: abs(c["positive_rate"] - midpoint))
        within_target = True
    else:
        best = min(candidates, key=lambda c: abs(c["positive_rate"] - midpoint))
        within_target = False

    reliable = n >= MIN_ROWS_FOR_RELIABLE_CALIBRATION and within_target

    rationale = (
        f"label_threshold_km = distance percentile {best['distance_percentile']} of observed "
        f"min_distance_km ({best['label_threshold_km']:.3f} km); "
        f"label_relative_velocity_km_s = velocity percentile {best['velocity_percentile']} of observed "
        f"relative_velocity_km_s ({best['label_relative_velocity_km_s']:.3f} km/s). "
        f"Joint positive rate on n={n} pairs: {best['positive_rate'] * 100:.2f}% "
        f"(target [{target_low * 100:.0f}%, {target_high * 100:.0f}%]); "
        f"distance threshold capped at the physical screening limit of {max_distance_km:.3f} km."
    )
    if not within_target:
        rationale += (
            " No grid combination reached the target range; reporting the closest "
            "achievable combination instead."
        )
    if n < MIN_ROWS_FOR_RELIABLE_CALIBRATION:
        rationale += (
            f" WARNING: only {n} rows available (< {MIN_ROWS_FOR_RELIABLE_CALIBRATION}); "
            "treat as illustrative only and re-run once more history has accumulated."
        )
    elif not within_target:
        rationale += (
            " WARNING: row count is sufficient, but no physically bounded threshold "
            "produced the target positive rate; collect a denser/more representative "
            "catalogue instead of widening the distance definition."
        )

    return {
        "n_rows": n,
        "reliable": reliable,
        "within_target": within_target,
        "target_positive_rate_low": target_low,
        "target_positive_rate_high": target_high,
        "max_physical_distance_km": max_distance_km,
        "distance_percentile_used": best["distance_percentile"],
        "velocity_percentile_used": best["velocity_percentile"],
        "label_threshold_km": best["label_threshold_km"],
        "label_relative_velocity_km_s": best["label_relative_velocity_km_s"],
        "achieved_positive_rate": best["positive_rate"],
        "rationale": rationale,
    }


def plot_distributions(df: pd.DataFrame, output_dir: Path, calibration: dict | None = None) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    plt.figure(figsize=(8, 5))
    plt.hist(df["min_distance_km"], bins=min(40, max(len(df) // 2, 5)), color="#2f5d8c", edgecolor="black", alpha=0.85)
    if calibration:
        plt.axvline(
            calibration["label_threshold_km"],
            color="#b33a3a",
            linestyle="--",
            label=f"suggested threshold ({calibration['label_threshold_km']:.1f} km)",
        )
        plt.legend()
    plt.xlabel("Minimum approach distance (km)")
    plt.ylabel("Pair count")
    plt.title(f"min_distance_km distribution (n={len(df)})")
    plt.tight_layout()
    dist_path = output_dir / "distance_distribution.png"
    plt.savefig(dist_path, dpi=160)
    plt.close()
    outputs.append(dist_path)

    plt.figure(figsize=(8, 5))
    plt.hist(df["relative_velocity_km_s"], bins=min(40, max(len(df) // 2, 5)), color="#2f8c5d", edgecolor="black", alpha=0.85)
    if calibration:
        plt.axvline(
            calibration["label_relative_velocity_km_s"],
            color="#b33a3a",
            linestyle="--",
            label=f"suggested threshold ({calibration['label_relative_velocity_km_s']:.2f} km/s)",
        )
        plt.legend()
    plt.xlabel("Relative velocity at TCA (km/s)")
    plt.ylabel("Pair count")
    plt.title(f"relative_velocity_km_s distribution (n={len(df)})")
    plt.tight_layout()
    vel_path = output_dir / "velocity_distribution.png"
    plt.savefig(vel_path, dpi=160)
    plt.close()
    outputs.append(vel_path)

    return outputs


def write_calibration_report(path: Path, history_path: Path, summary: dict, calibration: dict) -> None:
    """Write a standalone, regenerate-able calibration report.

    This is deliberately a separate file from config/experiment_60_days.json:
    the pipeline's simulation thresholds stay a hand-edited, versioned config;
    this report is a disposable, timestamped artifact recomputed from however
    much history has accumulated. Adopting a new calibration always means a
    human reads this file and edits the experiment config -- it is never
    applied automatically.
    """
    report = {
        "generated_utc": generated_utc(),
        "git_commit": git_commit_hash(),
        "source_history": str(history_path),
        "distribution_summary": summary,
        "calibration": calibration,
        "how_to_apply": (
            "Re-run src/analyze_distributions.py as data accumulates. Only if "
            "calibration.reliable and calibration.within_target are both true, "
            "you may copy label_threshold_km / label_relative_velocity_km_s into "
            "config/experiment_60_days.json's `simulation` block by hand -- this "
            "file never modifies that config automatically."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze accumulated conjunction history and suggest physically-defensible label thresholds"
    )
    parser.add_argument("--history", default="outputs/history/conjunction_observations_v3.csv")
    parser.add_argument("--output", default="config/threshold_calibration.json")
    parser.add_argument("--plots-dir", default="outputs/history")
    parser.add_argument("--target-positive-rate-low", type=float, default=0.02)
    parser.add_argument("--target-positive-rate-high", type=float, default=0.15)
    parser.add_argument("--max-physical-distance-km", type=float, default=DEFAULT_MAX_PHYSICAL_DISTANCE_KM)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history_path = Path(args.history)
    df = load_history(history_path)

    summary = summarize_distributions(df)
    calibration = calibrate_thresholds(
        df,
        target_low=args.target_positive_rate_low,
        target_high=args.target_positive_rate_high,
        max_distance_km=args.max_physical_distance_km,
    )
    plots = plot_distributions(df, Path(args.plots_dir), calibration)
    write_calibration_report(Path(args.output), history_path, summary, calibration)

    print(f"n_rows={summary['n_rows']}")
    print(calibration["rationale"])
    print(f"Report -> {args.output}")
    for plot_path in plots:
        print(f"Plot -> {plot_path}")


if __name__ == "__main__":
    main()
