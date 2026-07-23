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
from space_debris.experiment import DEFAULT_EXPERIMENT_CONFIG, load_experiment_config
from space_debris.ml import FEATURE_SETS, compare_models, compare_to_baseline_pr_auc, feature_set_by_name
from space_debris.plots import create_pipeline_plots, create_top_pair_physical_plots
from space_debris.provenance import png_provenance_metadata


def _resolve_pair_satellites(satellites, pair):
    """Resolve a result pair by NORAD ID; names are not unique for debris."""
    by_catalog_id = {str(satellite.model.satnum): satellite for satellite in satellites}
    if pair.object_1_catalog_id and pair.object_2_catalog_id:
        try:
            return (
                by_catalog_id[str(pair.object_1_catalog_id)],
                by_catalog_id[str(pair.object_2_catalog_id)],
            )
        except KeyError as exc:
            raise ValueError(f"Pair references unknown NORAD catalogue ID: {exc.args[0]}") from exc

    def unique_name(name):
        matches = [satellite for satellite in satellites if satellite.name == name]
        if len(matches) != 1:
            raise ValueError(
                f"Cannot resolve non-unique/missing satellite name {name!r}; catalogue IDs are required"
            )
        return matches[0]

    return unique_name(pair.object_1), unique_name(pair.object_2)


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=str(DEFAULT_EXPERIMENT_CONFIG))
    config_args, _ = config_parser.parse_known_args()
    config = load_experiment_config(config_args.config)

    parser = argparse.ArgumentParser(description="LEO conjunction simulation pipeline")
    parser.add_argument("--config", default=str(config.path), help="authoritative experiment JSON")
    parser.add_argument("--tle", default="data/sample_tles.txt", help="3-line TLE file")
    parser.add_argument("--outputs", default="outputs/pipeline", help="output directory")
    parser.add_argument("--horizon-minutes", type=int, default=config.horizon_minutes)
    parser.add_argument("--step-minutes", type=int, default=config.step_minutes)
    parser.add_argument("--screening-step-seconds", type=float, default=config.screening_step_seconds)
    parser.add_argument("--leo-min-altitude-km", type=float, default=config.leo_min_altitude_km)
    parser.add_argument("--leo-max-altitude-km", type=float, default=config.leo_max_altitude_km)
    # Physical conjunction-screening scales (km). Operational SSA screening
    # volumes are a few km; these are widened for a sparse public-TLE demo but
    # are still physically interpretable, unlike the previous 3000-10000 km.
    parser.add_argument("--candidate-threshold-km", type=float, default=config.candidate_threshold_km,
                        help="keep pairs whose closest approach is within this radius")
    parser.add_argument("--fixed-threshold-km", type=float, default=config.fixed_threshold_km,
                        help="classical baseline: alarm if min distance <= this")
    parser.add_argument("--label-threshold-km", type=float, default=config.label_threshold_km,
                        help="ground-truth screening radius for risk_label")
    parser.add_argument("--label-relative-velocity-km-s", type=float, default=config.label_relative_velocity_km_s,
                        help="ground-truth: risky only if v_rel >= this at TCA")
    parser.add_argument("--max-tle-age-hours", type=float, default=config.max_tle_age_hours,
                        help="warn if any TLE epoch is older than this (default 14 days)")
    parser.add_argument(
        "--feature-set", choices=sorted(FEATURE_SETS), default="snapshot_only",
        help=(
            "'snapshot_only' (default): predictors available before the TCA search, "
            "the practical predictive-power result. 'full_rule_recovery': also gives "
            "the model min_distance_km/relative_velocity_km_s, which define risk_label; "
            "this measures rule *recovery*, not predictive power, and is reported as a "
            "separate, clearly labelled diagnostic in the paper, never as the headline "
            "predictive-accuracy claim."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.outputs)
    output_dir.mkdir(parents=True, exist_ok=True)

    objects = read_tles(Path(args.tle))
    satellites = build_satellites(objects)
    start_utc = datetime.now(timezone.utc).replace(microsecond=0)

    screening_report: dict = {}
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
        candidate_screening_threshold_km=args.candidate_threshold_km,
        screening_step_seconds=args.screening_step_seconds,
        screening_report=screening_report,
    )
    conjunctions = filter_conjunctions(rows, args.candidate_threshold_km)
    screening_report["exact_candidates"] = len(conjunctions)

    dataset_path = output_dir / "conjunction_dataset.csv"
    conjunction_path = output_dir / "identified_conjunctions.csv"
    ml_report_path = output_dir / "model_comparison.csv"
    top_pair_timeseries_path = output_dir / "top_pair_distance_timeseries.csv"
    top_pair_plot_path = output_dir / "top_pair_distance.png"

    source = f"tle={args.tle}"
    config_summary = (
        f"horizon={args.horizon_minutes}min step={args.step_minutes}min "
        f"screening_step={args.screening_step_seconds}s "
        f"label_threshold_km={args.label_threshold_km} "
        f"label_relative_velocity_km_s={args.label_relative_velocity_km_s} "
        f"candidate_threshold_km={args.candidate_threshold_km} "
        f"fixed_threshold_km={args.fixed_threshold_km} "
        f"feature_set={args.feature_set}"
    )
    if args.feature_set == "full_rule_recovery":
        print(
            "NOTE: --feature-set=full_rule_recovery gives the model "
            "min_distance_km/relative_velocity_km_s, the two quantities that define "
            "risk_label. This measures how well the model recovers a known rule, NOT "
            "predictive power on unseen risk -- see README 'What is (and isn't) claimed'."
        )

    # ``dataset_path`` keeps every simulated pair for inspection, but — per the
    # paper — the classifier is trained on the *identified conjunctions* only.
    # This screening step is what makes the problem non-trivial: within the
    # candidate set, distance alone no longer separates risky from non-risky,
    # so a fixed-distance baseline starts making mistakes that the ML models
    # can avoid.
    write_pair_results(dataset_path, rows, source=source, config_summary=config_summary)
    write_pair_results(conjunction_path, conjunctions, source=source, config_summary=config_summary)
    report = compare_models(
        conjunction_path, ml_report_path, source=source, config_summary=config_summary,
        feature_columns=feature_set_by_name(args.feature_set),
    )
    scientific_plots = create_pipeline_plots(
        dataset_path=conjunction_path,
        model_report_path=ml_report_path,
        output_dir=output_dir,
        candidate_threshold_km=args.candidate_threshold_km,
    )
    # Publication-grade plots require the frozen evaluation manifest,
    # predictions, and feature-importance artifacts that only
    # `train_from_history.py` produces (see space_debris.plots.create_publication_plots);
    # this one-shot pipeline has no frozen evidence to hand it and does not
    # attempt to fake one.

    if rows:
        top = rows[0]
        top_satellite_1, top_satellite_2 = _resolve_pair_satellites(satellites, top)
        physical_plots = create_top_pair_physical_plots(
            top_satellite_1,
            top_satellite_2,
            start_utc,
            args.horizon_minutes,
            args.step_minutes,
            output_dir,
        )
        write_distance_timeseries(
            top_pair_timeseries_path,
            top_satellite_1,
            top_satellite_2,
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
            plt.savefig(top_pair_plot_path, dpi=160, metadata=png_provenance_metadata(source, config_summary))
            plt.close()
        except Exception as exc:
            print(f"Plot skipped: {exc}")

    print("=== SPACE DEBRIS PIPELINE OK ===")
    print(f"Screening -> {screening_report}")
    print(f"TLE objects             : {len(objects)}")
    print(f"Pairs exact-refined     : {len(rows)}")
    print(f"Identified conjunctions : {len(conjunctions)}")
    print(f"Dataset                 : {dataset_path}")
    print(f"Conjunctions            : {conjunction_path}")
    print(f"Feature set             : {args.feature_set}")
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
    print()
    print(compare_to_baseline_pr_auc(report))


if __name__ == "__main__":
    main()
