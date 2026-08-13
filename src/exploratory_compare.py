"""Run a clearly non-claim-eligible model comparison on a failed experiment.

The canonical ``train_from_history.py`` entry point must remain fail-closed when
any frozen publication gate fails.  This companion entry point intentionally
omits only the collection-cadence publication gates so that already collected
data can still be inspected for model-development purposes.  It preserves the
frozen catalogue, labels, feature set, chronological cutoff and held-out pair
assignment.  Every artifact is marked exploratory and must not be used as
publication evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from space_debris.experiment import DEFAULT_EXPERIMENT_CONFIG, load_experiment_config
from space_debris.ml import compare_models, feature_set_by_name
from space_debris.plots import plot_model_metrics
from space_debris.provenance import (
    generated_utc,
    git_commit_full_hash,
    git_worktree_state,
    write_json_atomic,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_exploratory_comparison(
    history_path: Path,
    output_dir: Path,
    config_path: Path,
) -> dict[str, object]:
    """Evaluate models without promoting a failed collection to evidence."""
    experiment = load_experiment_config(config_path)
    if history_path.resolve() != Path(experiment.resimulated_history).resolve():
        raise ValueError(
            "Exploratory comparison still requires the canonical corrected-TCA history"
        )
    if not history_path.is_file():
        raise FileNotFoundError(
            f"Corrected-TCA history is missing: {history_path}; run resimulate_snapshots.py first"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "model_comparison_exploratory.csv"
    plot_path = output_dir / "model_metrics_exploratory.png"
    manifest_path = output_dir / "exploratory_manifest.json"
    features = feature_set_by_name(experiment.primary_feature_set)
    source = f"canonical_resimulated_history={history_path}; sha256={_sha256(history_path)}"
    config_summary = (
        f"experiment_config={config_path}, experiment_id={experiment.experiment_id}, "
        "claim_eligible=false, cadence_publication_gates_bypassed_for_exploration=true, "
        f"time_column={experiment.time_column}, "
        f"train_time_fraction={experiment.train_time_fraction}, "
        f"test_pair_fraction={experiment.test_pair_fraction}, pair_seed={experiment.pair_seed}, "
        f"primary_feature_set={experiment.primary_feature_set}"
    )

    # Deliberately do not pass the collection-window cadence gates or minimum
    # publication-support thresholds.  All identity, age, label, feature,
    # chronological and pair-holdout protections remain active.
    report = compare_models(
        history_path,
        report_path,
        time_column=experiment.time_column,
        source=source,
        config_summary=config_summary,
        train_time_fraction=experiment.train_time_fraction,
        test_pair_fraction=experiment.test_pair_fraction,
        pair_seed=experiment.pair_seed,
        feature_columns=features,
        required_catalog_version=experiment.catalog_version,
        required_catalog_sha256=experiment.catalog_sha256,
        maximum_tle_age_hours=experiment.max_tle_age_hours,
    )
    plot_model_metrics(report_path, plot_path)

    history = pd.read_csv(history_path, comment="#")
    model_rows = report.loc[report["model"].ne("not_enough_data")]
    metric_columns = [
        "model",
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "false_alarm_rate",
        "false_positive",
        "false_negative",
        "true_positive",
        "true_negative",
    ]
    available_metric_columns = [
        column for column in metric_columns if column in model_rows.columns
    ]
    metrics = model_rows[available_metric_columns].where(pd.notna(model_rows), None)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "generated_utc": generated_utc(),
        "git_commit": git_commit_full_hash(),
        "git_worktree": git_worktree_state(),
        "experiment_id": experiment.experiment_id,
        "claim_eligible": False,
        "publication_use_permitted": False,
        "cadence_publication_gates_enforced": False,
        "interpretation": (
            "Exploratory model-development result only. The frozen experiment's "
            "failed cadence gates remain authoritative and this run cannot open a claim."
        ),
        "history": {
            "path": str(history_path),
            "sha256": _sha256(history_path),
            "rows": int(len(history)),
            "positive_rows": int(pd.to_numeric(history["risk_label"]).eq(1).sum()),
            "snapshots": int(history[experiment.time_column].nunique()),
        },
        "evaluation": {
            "feature_set": experiment.primary_feature_set,
            "features": features,
            "train_time_fraction": experiment.train_time_fraction,
            "test_pair_fraction": experiment.test_pair_fraction,
            "pair_seed": experiment.pair_seed,
            "models_compared": model_rows["model"].astype(str).tolist(),
            "metrics": metrics.to_dict(orient="records"),
            "best_pr_auc_model": (
                str(model_rows.loc[model_rows["pr_auc"].astype(float).idxmax(), "model"])
                if not model_rows.empty and model_rows["pr_auc"].notna().any()
                else None
            ),
            "status": (
                "not_enough_data"
                if report["model"].astype(str).eq("not_enough_data").any()
                else "complete"
            ),
        },
        "artifacts": {
            "report": str(report_path),
            "report_sha256": _sha256(report_path),
            "plot": str(plot_path),
            "plot_sha256": _sha256(plot_path),
        },
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=str(DEFAULT_EXPERIMENT_CONFIG))
    config_args, _ = config_parser.parse_known_args()
    experiment = load_experiment_config(config_args.config)

    parser = argparse.ArgumentParser(
        description="Run a non-claim-eligible comparison on canonical resimulated history"
    )
    parser.add_argument("--config", default=str(experiment.path))
    parser.add_argument("--history", default=experiment.resimulated_history)
    parser.add_argument(
        "--output-dir",
        default=f"outputs/exploratory_{experiment.experiment_id.replace('-', '_')}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_exploratory_comparison(
        Path(args.history), Path(args.output_dir), Path(args.config)
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
