from __future__ import annotations

import argparse
from pathlib import Path

from space_debris.ml import compare_models
from space_debris.plots import plot_model_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate models from accumulated observation history")
    parser.add_argument("--history", default="outputs/history/conjunction_observations.csv")
    parser.add_argument("--report", default="outputs/history/model_comparison_time_split.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare_models(Path(args.history), Path(args.report), time_column="snapshot_utc")
    plot_model_metrics(Path(args.report), Path(args.report).with_suffix(".png"))
    print(f"OK -> {args.report}")
    print(f"Plot -> {Path(args.report).with_suffix('.png')}")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
