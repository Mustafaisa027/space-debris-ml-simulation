from __future__ import annotations

import argparse
from pathlib import Path

from space_debris.ml import (
    FEATURES_WITHOUT_LABEL_RULE,
    compare_models,
    compare_to_baseline_pr_auc,
    time_series_cv_report,
)
from space_debris.experiment import DEFAULT_EXPERIMENT_CONFIG, load_experiment_config
from space_debris.plots import create_publication_plots, plot_model_metrics


def _clear_stale_model_artifacts(report_path: Path) -> None:
    """Remove only known derived claims when the current quality gate fails."""
    output_dir = report_path.parent
    stale_paths = [
        report_path.with_suffix(".png"),
        report_path.with_name(report_path.stem + "_cv.csv"),
        report_path.with_name(report_path.stem + "_without_label_rule_features.csv"),
        output_dir / "model_comparison_time_split.csv",
        output_dir / "model_comparison_time_split.png",
        output_dir / "model_comparison_time_split_cv.csv",
        output_dir / "model_comparison_time_split_without_label_rule_features.csv",
        output_dir / "model_comparison_pair_grouped_time_split.csv",
        output_dir / "model_comparison_pair_grouped_time_split.png",
        output_dir / "model_comparison_pair_grouped_time_split_cv.csv",
        output_dir / "model_comparison_pair_grouped_time_split_without_label_rule_features.csv",
        output_dir / "pub_pr_curves.png",
        output_dir / "pub_false_alarm_recall_curves.png",
        output_dir / "pub_confusion_matrices.png",
        output_dir / "pub_feature_importance.png",
        output_dir / "pub_threshold_sensitivity.png",
    ]
    for path in stale_paths:
        path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=str(DEFAULT_EXPERIMENT_CONFIG))
    config_args, _ = config_parser.parse_known_args()
    config = load_experiment_config(config_args.config)

    parser = argparse.ArgumentParser(description="Train/evaluate models from accumulated observation history")
    parser.add_argument("--config", default=str(config.path), help="authoritative experiment JSON")
    parser.add_argument(
        "--history",
        default=config.resimulated_history,
        help="corrected-TCA canonical history (defaults to outputs.resimulated_history)",
    )
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
    minimum_support = {
        "train_positive_rows": experiment.min_train_positive_rows,
        "test_positive_rows": experiment.min_test_positive_rows,
        "train_positive_pairs": experiment.min_train_positive_pairs,
        "test_positive_pairs": experiment.min_test_positive_pairs,
        "train_positive_snapshots": experiment.min_train_positive_snapshots,
        "test_positive_snapshots": experiment.min_test_positive_snapshots,
    }
    minimum_observation_span_days = (
        experiment.duration_days - experiment.poll_interval_hours / 24.0
    )
    report = compare_models(
        Path(args.history), Path(args.report), time_column=experiment.time_column,
        source=source, config_summary=config_summary,
        train_time_fraction=experiment.train_time_fraction,
        test_pair_fraction=experiment.test_pair_fraction,
        pair_seed=experiment.pair_seed,
        minimum_support=minimum_support,
        minimum_observation_span_days=minimum_observation_span_days,
        required_catalog_version=experiment.catalog_version,
        required_catalog_sha256=experiment.catalog_sha256,
    )
    if len(report) == 1 and report.iloc[0]["model"] == "not_enough_data":
        _clear_stale_model_artifacts(Path(args.report))
        print(f"DATA QUALITY GATE -> {report.iloc[0]['note']}")
        print("No model comparison or publication claims were produced.")
        return
    plot_model_metrics(Path(args.report), Path(args.report).with_suffix(".png"))
    print(f"OK -> {args.report}")
    print(f"Plot -> {Path(args.report).with_suffix('.png')}")
    print(report.to_string(index=False))
    print()
    print(compare_to_baseline_pr_auc(report))

    ablation_report_path = Path(args.report).with_name(
        Path(args.report).stem + "_without_label_rule_features.csv"
    )
    ablation_report = compare_models(
        Path(args.history),
        ablation_report_path,
        time_column=experiment.time_column,
        source=source,
        config_summary=(
            f"{config_summary}, "
            "feature_ablation=without_label_rule_and_reconstructing_ric_position"
        ),
        train_time_fraction=experiment.train_time_fraction,
        test_pair_fraction=experiment.test_pair_fraction,
        pair_seed=experiment.pair_seed,
        feature_columns=FEATURES_WITHOUT_LABEL_RULE,
        minimum_support=minimum_support,
        minimum_observation_span_days=minimum_observation_span_days,
        required_catalog_version=experiment.catalog_version,
        required_catalog_sha256=experiment.catalog_sha256,
    )
    print()
    print(f"Feature ablation report -> {ablation_report_path}")
    print(ablation_report.to_string(index=False))

    publication_plots = create_publication_plots(
        dataset_path=Path(args.history),
        output_dir=Path(args.report).parent,
        time_column=experiment.time_column,
        current_threshold_km=experiment.fixed_threshold_km,
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
