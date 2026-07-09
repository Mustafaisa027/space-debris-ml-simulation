from __future__ import annotations

import argparse
from pathlib import Path

from space_debris.ml import compare_models, compare_to_baseline_pr_auc, time_series_cv_report
from space_debris.plots import plot_model_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate models from accumulated observation history")
    parser.add_argument("--history", default="outputs/history/conjunction_observations.csv")
    parser.add_argument("--report", default="outputs/history/model_comparison_time_split.csv")
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=None,
        help="also run k-fold TimeSeriesSplit cross-validation (mean +/- std per model/metric)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare_models(Path(args.history), Path(args.report), time_column="snapshot_utc")
    plot_model_metrics(Path(args.report), Path(args.report).with_suffix(".png"))
    print(f"OK -> {args.report}")
    print(f"Plot -> {Path(args.report).with_suffix('.png')}")
    print(report.to_string(index=False))
    print()
    print(compare_to_baseline_pr_auc(report))

    if args.cv_splits:
        cv_report_path = Path(args.report).with_name(Path(args.report).stem + "_cv.csv")
        cv_report = time_series_cv_report(
            Path(args.history), cv_report_path, n_splits=args.cv_splits, time_column="snapshot_utc"
        )
        print()
        print(f"CV report ({args.cv_splits}-fold TimeSeriesSplit) -> {cv_report_path}")
        print(cv_report.to_string(index=False))


if __name__ == "__main__":
    main()
