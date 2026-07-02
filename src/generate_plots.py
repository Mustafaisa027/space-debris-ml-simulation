from __future__ import annotations

import argparse
from pathlib import Path

from space_debris.plots import create_pipeline_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate scientific plots from existing pipeline outputs")
    parser.add_argument("--outputs", default="outputs/verification_pipeline")
    parser.add_argument("--candidate-threshold-km", type=float, default=10000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.outputs)
    plots = create_pipeline_plots(
        dataset_path=output_dir / "conjunction_dataset.csv",
        model_report_path=output_dir / "model_comparison.csv",
        output_dir=output_dir,
        candidate_threshold_km=args.candidate_threshold_km,
    )
    print("Generated plots:")
    for path in plots:
        print(f"- {path}")


if __name__ == "__main__":
    main()
