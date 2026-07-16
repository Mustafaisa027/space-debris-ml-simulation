from __future__ import annotations

import argparse
from pathlib import Path

from space_debris.ml import compare_models, compare_to_baseline_pr_auc, time_series_cv_report
from space_debris.experiment import DEFAULT_EXPERIMENT_CONFIG, load_experiment_config
from space_debris.plots import create_publication_plots, plot_model_metrics


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=str(DEFAULT_EXPERIMENT_CONFIG))
    config_args, _ = config_parser.parse_known_args()
    config = load_experiment_config(config_args.config)

    parser = argparse.ArgumentParser(description="Train/evaluate models from accumulated observation history")
    parser.add_argument("--config", default=str(config.path), help="authoritative experiment JSON")
    parser.add_argument("--history", default=config.history)
    parser.add_argument("--report", default=config.time_split_report)
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=None,
        help="also run pair-held-out expanding-time CV (mean +/- std per model/metric)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = load_experiment_config(args.config)
    source = f"history={args.history}"
    config_summary = (
        f"experiment_config={args.config}, time_column={experiment.time_column}, "
        f"train_time_fraction={experiment.train_time_fraction}, "
        f"test_pair_fraction={experiment.test_pair_fraction}, pair_seed={experiment.pair_seed}"
    )
    report = compare_models(
        Path(args.history), Path(args.report), time_column=experiment.time_column,
        source=source, config_summary=config_summary,
        train_time_fraction=experiment.train_time_fraction,
        test_pair_fraction=experiment.test_pair_fraction,
        pair_seed=experiment.pair_seed,
    )
    if len(report) == 1 and report.iloc[0]["model"] == "not_enough_data":
        print(f"DATA QUALITY GATE -> {report.iloc[0]['note']}")
        print("No model comparison or publication claims were produced.")
        return
    plot_model_metrics(Path(args.report), Path(args.report).with_suffix(".png"))
    print(f"OK -> {args.report}")
    print(f"Plot -> {Path(args.report).with_suffix('.png')}")
    print(report.to_string(index=False))
    print()
    print(compare_to_baseline_pr_auc(report))

    publication_plots = create_publication_plots(
        dataset_path=Path(args.history),
        output_dir=Path(args.report).parent,
        time_column=experiment.time_column,
        train_time_fraction=experiment.train_time_fraction,
        test_pair_fraction=experiment.test_pair_fraction,
        pair_seed=experiment.pair_seed,
    )
    for plot_path in publication_plots:
        print(f"Publication plot -> {plot_path}")

    if args.cv_splits:
        cv_report_path = Path(args.report).with_name(Path(args.report).stem + "_cv.csv")
        cv_report = time_series_cv_report(
            Path(args.history), cv_report_path, n_splits=args.cv_splits, time_column=experiment.time_column,
            source=source, config_summary=f"{config_summary}, n_splits={args.cv_splits}",
            test_pair_fraction=experiment.test_pair_fraction,
            pair_seed=experiment.pair_seed,
        )
        print()
        print(f"CV report ({args.cv_splits}-fold pair/time split) -> {cv_report_path}")
        print(cv_report.to_string(index=False))


if __name__ == "__main__":
    main()
