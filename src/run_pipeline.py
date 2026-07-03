from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from space_debris.core import (
    build_satellites,
    filter_conjunctions,
    read_tles,
    simulate_pairs,
    write_distance_timeseries,
    write_pair_results,
)
from space_debris.ml import compare_models
from space_debris.plots import create_pipeline_plots, create_top_pair_physical_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LEO conjunction simulation pipeline")
    parser.add_argument("--tle", default="data/sample_tles.txt", help="3-line TLE file")
    parser.add_argument("--outputs", default="outputs/pipeline", help="output directory")
    parser.add_argument("--horizon-minutes", type=int, default=720)
    parser.add_argument("--step-minutes", type=int, default=5)
    parser.add_argument("--leo-min-altitude-km", type=float, default=160.0)
    parser.add_argument("--leo-max-altitude-km", type=float, default=2000.0)
    # Physical conjunction-screening scales (km). Operational SSA screening
    # volumes are a few km; these are widened for a sparse public-TLE demo but
    # are still physically interpretable, unlike the previous 3000-10000 km.
    parser.add_argument("--candidate-threshold-km", type=float, default=50.0,
                        help="keep pairs whose closest approach is within this radius")
    parser.add_argument("--fixed-threshold-km", type=float, default=25.0,
                        help="classical baseline: alarm if min distance <= this")
    parser.add_argument("--label-threshold-km", type=float, default=20.0,
                        help="ground-truth screening radius for risk_label")
    parser.add_argument("--label-relative-velocity-km-s", type=float, default=10.0,
                        help="ground-truth: risky only if v_rel >= this at TCA")
    parser.add_argument("--max-tle-age-hours", type=float, default=336.0,
                        help="warn if any TLE epoch is older than this (default 14 days)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.outputs)
    output_dir.mkdir(parents=True, exist_ok=True)

    objects = read_tles(Path(args.tle))
    satellites = build_satellites(objects)
    start_utc = datetime.now(timezone.utc).replace(microsecond=0)

    rows = simulate_pairs(
        satellites=satellites,
        start_utc=start_utc,
        horizon_minutes=args.horizon_minutes,
        step_minutes=args.step_minutes,
        leo_min_altitude_km=args.leo_min_altitude_km,
        leo_max_altitude_km=args.leo_max_altitude_km,
        fixed_threshold_km=args.fixed_threshold_km,
        label_threshold_km=args.label_threshold_km,
        label_relative_velocity_km_s=args.label_relative_velocity_km_s,
        max_tle_age_hours=args.max_tle_age_hours,
    )
    conjunctions = filter_conjunctions(rows, args.candidate_threshold_km)

    dataset_path = output_dir / "conjunction_dataset.csv"
    conjunction_path = output_dir / "identified_conjunctions.csv"
    ml_report_path = output_dir / "model_comparison.csv"
    top_pair_timeseries_path = output_dir / "top_pair_distance_timeseries.csv"
    top_pair_plot_path = output_dir / "top_pair_distance.png"

    # ``dataset_path`` keeps every simulated pair for inspection, but — per the
    # paper — the classifier is trained on the *identified conjunctions* only.
    # This screening step is what makes the problem non-trivial: within the
    # candidate set, distance alone no longer separates risky from non-risky,
    # so a fixed-distance baseline starts making mistakes that the ML models
    # can avoid.
    write_pair_results(dataset_path, rows)
    write_pair_results(conjunction_path, conjunctions)
    report = compare_models(conjunction_path, ml_report_path)
    scientific_plots = create_pipeline_plots(
        dataset_path=conjunction_path,
        model_report_path=ml_report_path,
        output_dir=output_dir,
        candidate_threshold_km=args.candidate_threshold_km,
    )

    if rows:
        satellites_by_name = {sat.name: sat for sat in satellites}
        top = rows[0]
        physical_plots = create_top_pair_physical_plots(
            satellites_by_name[top.object_1],
            satellites_by_name[top.object_2],
            start_utc,
            args.horizon_minutes,
            args.step_minutes,
            output_dir,
        )
        write_distance_timeseries(
            top_pair_timeseries_path,
            satellites_by_name[top.object_1],
            satellites_by_name[top.object_2],
            start_utc,
            args.horizon_minutes,
            args.step_minutes,
        )
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import pandas as pd

            df_plot = pd.read_csv(top_pair_timeseries_path)
            plt.figure(figsize=(10, 5))
            plt.plot(df_plot["minute"], df_plot["distance_km"], label="distance")
            plt.axhline(args.candidate_threshold_km, linestyle="--", label="candidate threshold")
            plt.scatter([top.time_to_tca_min], [top.min_distance_km], marker="x", s=100, label="TCA")
            plt.xlabel("Minutes from start")
            plt.ylabel("Distance (km)")
            plt.title(f"{top.object_1} / {top.object_2} distance over time")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(top_pair_plot_path, dpi=160)
            plt.close()
        except Exception as exc:
            print(f"Plot skipped: {exc}")

    print("=== SPACE DEBRIS PIPELINE OK ===")
    print(f"TLE objects             : {len(objects)}")
    print(f"LEO pairs simulated     : {len(rows)}")
    print(f"Identified conjunctions : {len(conjunctions)}")
    print(f"Dataset                 : {dataset_path}")
    print(f"Conjunctions            : {conjunction_path}")
    print(f"Model report            : {ml_report_path}")
    if rows:
        print(f"Top pair timeseries     : {top_pair_timeseries_path}")
        print(f"Top pair plot           : {top_pair_plot_path}")
    for plot_path in scientific_plots:
        print(f"Scientific plot         : {plot_path}")
    if rows:
        for plot_path in physical_plots:
            print(f"Physical plot           : {plot_path}")
    if rows:
        top = rows[0]
        print()
        print("Top risk candidate:")
        print(
            f"{top.object_1} / {top.object_2} | TCA={top.tca_utc} | "
            f"d_min={top.min_distance_km:.1f} km | v_rel={top.relative_velocity_km_s:.2f} km/s"
        )
    print()
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
