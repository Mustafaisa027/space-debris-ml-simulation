from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import pandas as pd
from PIL import Image

from space_debris.ml import (
    FEATURES,
    FEATURES_WITHOUT_LABEL_RULE,
    compare_models,
    compare_to_baseline_pr_auc,
    feature_set_by_name,
    time_series_cv_report,
)
from space_debris.evidence import (
    adaptability_inference_from_predictions,
    adaptability_result_fingerprint,
    canonical_report_from_evidence,
    generate_adaptability_inference,
    generate_evaluation_evidence,
)
from space_debris.experiment import DEFAULT_EXPERIMENT_CONFIG, load_experiment_config
from space_debris.plots import create_publication_plots, plot_model_metrics
from space_debris.archive import file_sha256
from space_debris.provenance import (
    generated_utc,
    git_commit_full_hash,
    git_worktree_state,
    write_csv_text_with_provenance,
)


LEGACY_SCHEMA1_ALLOWLIST = frozenset(
    {
        (
            "github-run-29516502619-attempt-1",
            "20260716_164008",
            "1d7e2720eb584ebd2ce43de8c9a77cc54378fbc3",
        ),
        (
            "github-run-29528673738-attempt-1",
            "20260716_193651",
            "1d7e2720eb584ebd2ce43de8c9a77cc54378fbc3",
        ),
        (
            "github-run-29535594505-attempt-1",
            "20260716_211950",
            "1d7e2720eb584ebd2ce43de8c9a77cc54378fbc3",
        ),
    }
)

FROZEN_ABLATION_FEATURES = {
    "snapshot_geometry_only": (
        "current_distance_km",
        "altitude_difference_km",
        "max_tle_age_hours",
    ),
    "snapshot_kinematics_only": (
        "radial_velocity_km_s",
        "tangential_velocity_km_s",
        "approach_angle_deg",
        "max_tle_age_hours",
    ),
    "full_rule_recovery": tuple(FEATURES),
    "without_label_rule_features": tuple(FEATURES_WITHOUT_LABEL_RULE),
    "distance_only": ("min_distance_km",),
    "proxy_rule_only": ("min_distance_km", "relative_velocity_km_s"),
}
FROZEN_ABLATION_NAMES = frozenset(FROZEN_ABLATION_FEATURES)


def _clear_stale_model_artifacts(report_path: Path) -> None:
    """Remove only known derived claims when the current quality gate fails."""
    output_dir = report_path.parent
    stale_paths = [
        report_path.with_suffix(".png"),
        report_path.with_name(report_path.stem + "_cv.csv"),
        report_path.with_name(report_path.stem + "_cv_folds.csv"),
        report_path.with_name(report_path.stem + "_cv_adaptability.csv"),
        report_path.with_name(report_path.stem + "_adaptability_inference.json"),
        report_path.with_name(report_path.stem + "_without_label_rule_features.csv"),
        report_path.with_name(report_path.stem + "_full_rule_recovery.csv"),
        report_path.with_name(report_path.stem + "_snapshot_geometry_only.csv"),
        report_path.with_name(report_path.stem + "_snapshot_kinematics_only.csv"),
        report_path.with_name(report_path.stem + "_distance_only.csv"),
        report_path.with_name(report_path.stem + "_proxy_rule_only.csv"),
        output_dir / "model_comparison_time_split.csv",
        output_dir / "model_comparison_time_split.png",
        output_dir / "model_comparison_time_split_cv.csv",
        output_dir / "model_comparison_time_split_cv_folds.csv",
        output_dir / "model_comparison_time_split_cv_adaptability.csv",
        output_dir / "model_comparison_time_split_adaptability_inference.json",
        output_dir / "model_comparison_time_split_snapshot_geometry_only.csv",
        output_dir / "model_comparison_time_split_snapshot_kinematics_only.csv",
        output_dir / "model_comparison_time_split_without_label_rule_features.csv",
        output_dir / "model_comparison_pair_grouped_time_split.csv",
        output_dir / "model_comparison_pair_grouped_time_split.png",
        output_dir / "model_comparison_pair_grouped_time_split_cv.csv",
        output_dir / "model_comparison_pair_grouped_time_split_cv_folds.csv",
        output_dir / "model_comparison_pair_grouped_time_split_cv_adaptability.csv",
        output_dir / "model_comparison_pair_grouped_time_split_adaptability_inference.json",
        output_dir / "model_comparison_pair_grouped_time_split_snapshot_geometry_only.csv",
        output_dir / "model_comparison_pair_grouped_time_split_snapshot_kinematics_only.csv",
        output_dir / "model_comparison_pair_grouped_time_split_without_label_rule_features.csv",
        output_dir / "pub_pr_curves.png",
        output_dir / "pub_false_alarm_recall_curves.png",
        output_dir / "pub_confusion_matrices.png",
        output_dir / "pub_feature_importance.png",
        output_dir / "pub_threshold_sensitivity.png",
        output_dir / "evaluation_split_manifest.json",
        output_dir / "held_out_predictions.csv",
        output_dir / "paired_cluster_evidence.csv",
        output_dir / "evaluation_manifest.json",
        output_dir / "feature_importance.csv",
        output_dir / "publication_manifest.json",
        output_dir / "publication_incomplete.json",
    ]
    for path in stale_paths:
        path.unlink(missing_ok=True)
    shutil.rmtree(output_dir / "ablations", ignore_errors=True)


def _cleanup_failed_publication(report_path: Path) -> None:
    """Remove every known claim-bearing output after an interrupted build."""
    Path(report_path).unlink(missing_ok=True)
    _clear_stale_model_artifacts(Path(report_path))


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _strict_report_bool(value: object) -> bool:
    """Parse CSV/JSON booleans without treating NaN or non-empty text as true."""
    if value is True:
        return True
    if value is False or value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value == 1)


def _canonical_payload_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_embedded_hash(path: Path, source_path: Path, *, label: str) -> None:
    if path.suffix.lower() == ".png":
        with Image.open(path) as image:
            source_metadata = str(image.info.get("Description", ""))
    else:
        source_metadata = ""
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("# source: "):
                    source_metadata = line.removeprefix("# source: ").strip()
                    break
                if not line.startswith("#"):
                    break
    expected = f"sha256={file_sha256(source_path)}"
    if expected not in source_metadata:
        raise ValueError(f"{label} does not embed its verified source hash")


def _validate_frozen_ablation_features(
    ablation_name: str, evaluation_manifest: dict[str, object]
) -> None:
    expected = FROZEN_ABLATION_FEATURES.get(ablation_name)
    if expected is None or tuple(evaluation_manifest.get("features", ())) != tuple(expected):
        raise ValueError(
            f"Ablation feature set differs from frozen protocol: {ablation_name}"
        )


def _verified_file_binding(
    entry: object,
    *,
    label: str,
    required_artifacts: set[Path] | None = None,
) -> Path:
    if not isinstance(entry, dict):
        raise ValueError(f"Evaluation manifest is missing {label} binding")
    path = Path(str(entry.get("path", "")))
    if not path.is_file() or file_sha256(path) != entry.get("sha256"):
        raise ValueError(f"Evaluation manifest {label} hash/path binding is invalid")
    if required_artifacts is not None and path.resolve() not in required_artifacts:
        raise ValueError(f"Evaluation manifest {label} is absent from publication artifacts")
    return path


def _verify_publication_evaluation_bundle(
    evaluation_manifest_path: Path,
    artifacts: list[Path],
    validated_input_binding: dict[str, object],
    *,
    expected_claim_eligible: bool = True,
    canonical_config_path: Path | None = None,
) -> dict[str, object]:
    """Revalidate an evaluation hash graph before any publication manifest."""
    artifact_set = {path.resolve() for path in artifacts}
    if evaluation_manifest_path.resolve() not in artifact_set:
        raise ValueError("Evaluation manifest is absent from publication artifacts")
    evaluation = json.loads(evaluation_manifest_path.read_text(encoding="utf-8"))
    current_commit = git_commit_full_hash()
    if evaluation.get("schema_version") != 3:
        raise ValueError("Unsupported evaluation manifest schema")
    if (
        evaluation.get("git_commit") != current_commit
        or evaluation.get("source_worktree", {}).get("dirty") is not False
    ):
        raise ValueError("Evaluation was not generated from the current clean commit")
    if evaluation.get("archive_resimulation_binding") != validated_input_binding:
        raise ValueError("Evaluation archive/resimulation binding mismatch")
    dataset_path = _verified_file_binding(evaluation.get("dataset"), label="dataset")
    config_path = _verified_file_binding(evaluation.get("config"), label="config")
    expected_config_path = Path(
        DEFAULT_EXPERIMENT_CONFIG if canonical_config_path is None else canonical_config_path
    )
    if config_path.resolve() != expected_config_path.resolve():
        raise ValueError("Evaluation is not bound to the frozen canonical experiment config")
    canonical_history = validated_input_binding.get("canonical_history")
    if not isinstance(canonical_history, dict):
        raise ValueError("Validated input binding is missing canonical history")
    if (
        dataset_path.resolve() != Path(str(canonical_history.get("path", ""))).resolve()
        or file_sha256(dataset_path) != canonical_history.get("sha256")
    ):
        raise ValueError("Evaluation dataset is not the validated canonical history")
    split_path = _verified_file_binding(
        evaluation.get("split_manifest"),
        label="split manifest",
        required_artifacts=artifact_set,
    )
    predictions_path = _verified_file_binding(
        evaluation.get("predictions"),
        label="predictions",
        required_artifacts=artifact_set,
    )
    evidence_path = _verified_file_binding(
        evaluation.get("evidence"),
        label="paired evidence",
        required_artifacts=artifact_set,
    )
    importance_path = _verified_file_binding(
        evaluation.get("feature_importance"),
        label="feature importance",
        required_artifacts=artifact_set,
    )
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("schema_version") != 2:
        raise ValueError("Unsupported evaluation split manifest schema")
    expected_split_sha = evaluation.get("split_manifest", {}).get("split_sha256")
    split_payload = dict(split)
    asserted_split_sha = split_payload.pop("split_sha256", None)
    recomputed_split_sha = _canonical_payload_sha256(split_payload)
    if asserted_split_sha != recomputed_split_sha or asserted_split_sha != expected_split_sha:
        raise ValueError("Evaluation split SHA mismatch")
    dataset_sha = file_sha256(dataset_path)
    config_sha = file_sha256(config_path)
    if (
        split.get("dataset_sha256") != dataset_sha
        or evaluation.get("dataset", {}).get("sha256") != dataset_sha
    ):
        raise ValueError("Evaluation split/dataset binding mismatch")
    if (
        split.get("config_sha256") != config_sha
        or evaluation.get("config", {}).get("sha256") != config_sha
    ):
        raise ValueError("Evaluation split/config binding mismatch")
    if (
        split.get("claim_eligible") is not expected_claim_eligible
        or evaluation.get("bootstrap", {}).get("claim_eligible")
        is not expected_claim_eligible
    ):
        raise ValueError("Evaluation claim-eligibility role mismatch")
    predictions = pd.read_csv(predictions_path, comment="#")
    if (
        "split_sha256" not in predictions
        or predictions["split_sha256"].nunique(dropna=False) != 1
        or str(predictions["split_sha256"].iloc[0]) != expected_split_sha
    ):
        raise ValueError("Prediction rows are not bound to the main evaluation split")
    for upstream in evaluation.get("upstream_artifacts", []):
        _verified_file_binding(upstream, label="upstream artifact")
    return {
        "manifest": evaluation,
        "dataset_path": dataset_path,
        "config_path": config_path,
        "split_path": split_path,
        "split": split,
        "predictions_path": predictions_path,
        "predictions": predictions,
        "evidence_path": evidence_path,
        "importance_path": importance_path,
    }


def _write_publication_manifest(
    output_path: Path,
    artifacts: list[Path],
    *,
    primary_model: str,
    primary_feature_set: str,
    evidence_report: pd.DataFrame,
    validated_input_binding: dict[str, object],
    require_evaluation_bundle: bool = True,
    canonical_report_path: Path | None = None,
) -> None:
    unique_artifacts = sorted({Path(path) for path in artifacts}, key=str)
    missing = [str(path) for path in unique_artifacts if not path.is_file()]
    if missing:
        raise ValueError(f"Publication artifacts are missing: {missing}")
    evaluation_bundle = None
    if require_evaluation_bundle:
        evaluation_bundle = _verify_publication_evaluation_bundle(
            output_path.parent / "evaluation_manifest.json",
            unique_artifacts,
            validated_input_binding,
        )
        if canonical_report_path is None:
            raise ValueError("Publication requires the canonical model report path")
        canonical_report_path = Path(canonical_report_path)
        if canonical_report_path.resolve() not in {
            path.resolve() for path in unique_artifacts
        }:
            raise ValueError("Canonical model report is absent from publication artifacts")
        _require_embedded_hash(
            canonical_report_path,
            evaluation_bundle["predictions_path"],
            label="Canonical model report",
        )
        _require_embedded_hash(
            canonical_report_path,
            evaluation_bundle["evidence_path"],
            label="Canonical model report",
        )
        canonical_report_plot = canonical_report_path.with_suffix(".png")
        if canonical_report_plot.resolve() not in {
            path.resolve() for path in unique_artifacts
        }:
            raise ValueError("Canonical model report plot is absent from publication artifacts")
        _require_embedded_hash(
            canonical_report_plot,
            canonical_report_path,
            label="Canonical model report plot",
        )
        ablation_manifest_paths = sorted(
            path
            for path in unique_artifacts
            if path.name == "evaluation_manifest.json"
            and path.resolve() != (output_path.parent / "evaluation_manifest.json").resolve()
        )
        observed_ablation_names = {path.parent.name for path in ablation_manifest_paths}
        if observed_ablation_names != FROZEN_ABLATION_NAMES:
            raise ValueError(
                "Publication ablation bundles do not match the frozen protocol: "
                f"observed={sorted(observed_ablation_names)}"
            )
        for ablation_manifest_path in ablation_manifest_paths:
            ablation_bundle = _verify_publication_evaluation_bundle(
                ablation_manifest_path,
                unique_artifacts,
                validated_input_binding,
                expected_claim_eligible=False,
            )
            if (
                ablation_bundle["split"].get("outer")
                != evaluation_bundle["split"].get("outer")
                or ablation_bundle["split"].get("inner")
                != evaluation_bundle["split"].get("inner")
            ):
                raise ValueError(
                    f"Ablation split differs from the main frozen split: "
                    f"{ablation_manifest_path.parent.name}"
                )
            ablation_name = ablation_manifest_path.parent.name
            _validate_frozen_ablation_features(
                ablation_name, ablation_bundle["manifest"]
            )
            ablation_report = canonical_report_path.with_name(
                canonical_report_path.stem + f"_{ablation_name}.csv"
            )
            if ablation_report.resolve() not in {
                path.resolve() for path in unique_artifacts
            }:
                raise ValueError(f"Ablation report is missing: {ablation_name}")
            _require_embedded_hash(
                ablation_report,
                ablation_bundle["predictions_path"],
                label=f"Ablation report {ablation_name}",
            )
            _require_embedded_hash(
                ablation_report,
                ablation_bundle["evidence_path"],
                label=f"Ablation report {ablation_name}",
            )
        prediction_plots = {
            "pub_pr_curves.png",
            "pub_false_alarm_recall_curves.png",
            "pub_confusion_matrices.png",
            "pub_threshold_sensitivity.png",
        }
        for plot_name in prediction_plots:
            plot_path = output_path.parent / plot_name
            if plot_path.resolve() not in {path.resolve() for path in unique_artifacts}:
                raise ValueError(f"Publication plot is missing: {plot_name}")
            _require_embedded_hash(
                plot_path,
                evaluation_bundle["predictions_path"],
                label=f"Publication plot {plot_name}",
            )
        importance_plot = output_path.parent / "pub_feature_importance.png"
        if importance_plot.resolve() not in {path.resolve() for path in unique_artifacts}:
            raise ValueError("Publication feature-importance plot is missing")
        _require_embedded_hash(
            importance_plot,
            evaluation_bundle["importance_path"],
            label="Publication feature-importance plot",
        )
        history_sha = str(validated_input_binding["canonical_history"]["sha256"])
        for path in unique_artifacts:
            if path.suffix.lower() == ".csv" and (
                path.name.endswith("_cv.csv")
                or path.name.endswith("_cv_folds.csv")
                or path.name.endswith("_cv_adaptability.csv")
            ):
                source_line = next(
                    (
                        line.removeprefix("# source: ").strip()
                        for line in path.read_text(encoding="utf-8").splitlines()
                        if line.startswith("# source: ")
                    ),
                    "",
                )
                if f"sha256={history_sha}" not in source_line:
                    raise ValueError(
                        f"Descriptive CV artifact lacks canonical-history binding: {path}"
                    )
        verified_evidence_report = pd.read_csv(
            evaluation_bundle["evidence_path"], comment="#"
        )
        if not verified_evidence_report.reset_index(drop=True).equals(
            evidence_report.reset_index(drop=True)
        ):
            raise ValueError("Publication evidence table differs from the verified bundle")
        evidence_report = verified_evidence_report
    primary = evidence_report.loc[evidence_report["model"].eq(primary_model)]
    if len(primary) != 1:
        raise ValueError(f"Evidence must contain exactly one primary model row: {primary_model}")
    primary_row = primary.iloc[0]
    oracle = evidence_report.loc[evidence_report["model"].eq("proxy_rule_oracle")]
    permutation = evidence_report.loc[
        evidence_report["model"].eq("label_permutation_control")
    ]
    oracle_passed = bool(
        len(oracle) == 1
        and float(oracle.iloc[0].get("test_accuracy", float("nan"))) == 1.0
        and int(oracle.iloc[0].get("test_false_positive", -1)) == 0
        and int(oracle.iloc[0].get("test_false_negative", -1)) == 0
    )
    permutation_diagnostic_favorable = bool(
        len(permutation) == 1
        and float(permutation.iloc[0].get("test_pr_auc", float("nan")))
        < float(primary_row.get("test_pr_auc", float("nan")))
    )
    pipeline_integrity_controls_passed = oracle_passed
    adaptability_decision = {
        "adaptability_supported": False,
        "statistical_adaptability_supported": False,
        "adaptability_claim": "adaptability report not available",
    }
    adaptability_paths = [path for path in unique_artifacts if path.name.endswith("_adaptability.csv")]
    if len(adaptability_paths) > 1:
        raise ValueError("Publication artifacts contain multiple descriptive adaptability reports")
    if adaptability_paths:
        adaptability = pd.read_csv(adaptability_paths[0], comment="#")
        primary_adaptability = adaptability.loc[adaptability["model"].eq(primary_model)]
        if len(primary_adaptability) == 1:
            adaptability_row = primary_adaptability.iloc[0]
            adaptability_decision = {
                "adaptability_supported": _strict_report_bool(
                    adaptability_row["adaptability_supported"]
                ),
                "statistical_adaptability_supported": bool(
                    _strict_report_bool(
                        adaptability_row.get("statistical_significance_tested", False)
                    )
                    and _strict_report_bool(adaptability_row["adaptability_supported"])
                ),
                "adaptability_claim": str(adaptability_row["claim"]),
            }
    inference_paths = [
        path for path in unique_artifacts if path.name.endswith("_adaptability_inference.json")
    ]
    if len(inference_paths) > 1:
        raise ValueError("Publication artifacts contain multiple adaptability inference reports")
    if inference_paths:
        if evaluation_bundle is None:
            raise ValueError(
                "Adaptability inference requires a verified main evaluation bundle"
            )
        inference = json.loads(inference_paths[0].read_text(encoding="utf-8"))
        if (
            inference.get("primary_model") != primary_model
            or inference.get("primary_feature_set") != primary_feature_set
        ):
            raise ValueError("Adaptability inference does not match the frozen primary experiment")
        current_commit = git_commit_full_hash()
        if (
            inference.get("schema_version") != 1
            or inference.get("git_commit") != current_commit
            or inference.get("source_worktree", {}).get("dirty") is not False
        ):
            raise ValueError("Adaptability inference was not generated by the current clean commit")
        expected_bindings = {
            "predictions": {
                "path": str(evaluation_bundle["predictions_path"]),
                "sha256": file_sha256(evaluation_bundle["predictions_path"]),
            },
            "config": {
                "path": str(evaluation_bundle["config_path"]),
                "sha256": file_sha256(evaluation_bundle["config_path"]),
            },
            "split_manifest": {
                "path": str(evaluation_bundle["split_path"]),
                "sha256": file_sha256(evaluation_bundle["split_path"]),
                "split_sha256": evaluation_bundle["split"].get("split_sha256"),
            },
        }
        if any(inference.get(key) != value for key, value in expected_bindings.items()):
            raise ValueError("Adaptability inference cross-artifact binding mismatch")
        experiment = load_experiment_config(evaluation_bundle["config_path"])
        cutoff = pd.Timestamp(evaluation_bundle["split"].get("outer", {}).get("cutoff_utc"))
        if cutoff.tzinfo is None or experiment.collection_end_utc is None:
            raise ValueError("Adaptability inference lacks frozen time boundaries")
        anchor = cutoff.tz_convert("UTC") + pd.Timedelta(
            hours=experiment.poll_interval_hours
        )
        recomputed = adaptability_inference_from_predictions(
            evaluation_bundle["predictions"],
            experiment,
            split_cutoff_utc=cutoff.tz_convert("UTC").isoformat().replace("+00:00", "Z"),
            block_anchor_utc=anchor.isoformat().replace("+00:00", "Z"),
            analysis_end_utc=experiment.collection_end_utc,
        )
        recomputed.update(expected_bindings)
        recomputed_fingerprint = adaptability_result_fingerprint(recomputed)
        if (
            inference.get("result_fingerprint_sha256")
            != adaptability_result_fingerprint(inference)
            or inference.get("result_fingerprint_sha256") != recomputed_fingerprint
        ):
            raise ValueError("Adaptability inference result does not reproduce")
        statistical_gate = bool(
            recomputed.get("status") == "supported"
            and recomputed.get("statistical_significance_tested") is True
            and recomputed.get("statistical_adaptability_supported") is True
        )
        adaptability_decision.update(
            {
                "statistical_adaptability_supported": statistical_gate,
                "adaptability_inference_status": str(inference.get("status", "invalid")),
                "adaptability_inference_claim": str(inference.get("claim", "missing")),
            }
        )
    else:
        adaptability_decision.update(
            {
                "adaptability_inference_status": "missing",
                "adaptability_inference_claim": "statistical adaptability report not available",
            }
        )
    source_worktree = git_worktree_state()
    combined_distance_statistical_gate_passed = _strict_report_bool(
        primary_row["false_alarm_reduction_supported"]
    )
    fixed_25_pair_gate_passed = _strict_report_bool(
        primary_row.get("pair_cluster_gate_passed", False)
    )
    fixed_25_time_gate_passed = _strict_report_bool(
        primary_row.get("time_block_gate_passed", False)
    )
    fixed_25_statistical_gate_passed = bool(
        fixed_25_pair_gate_passed and fixed_25_time_gate_passed
    )
    calibrated_distance_pair_gate_passed = _strict_report_bool(
        primary_row.get("calibrated_distance_pair_gate_passed", False)
    )
    calibrated_distance_time_gate_passed = _strict_report_bool(
        primary_row.get("calibrated_distance_time_gate_passed", False)
    )
    calibrated_distance_statistical_gate_passed = bool(
        calibrated_distance_pair_gate_passed
        and calibrated_distance_time_gate_passed
    )
    statistical_adaptability_supported = bool(
        adaptability_decision["statistical_adaptability_supported"]
    )
    clean_committed_source = source_worktree.get("dirty") is False
    input_binding_validated = bool(validated_input_binding.get("validated") is True)
    false_alarm_supported = bool(
        combined_distance_statistical_gate_passed
        and fixed_25_statistical_gate_passed
        and calibrated_distance_statistical_gate_passed
        and pipeline_integrity_controls_passed
        and clean_committed_source
        and input_binding_validated
    )
    cpa_pair_sensitivity_passed = _strict_report_bool(
        primary_row.get("constant_velocity_cpa_pair_sensitivity_passed", False)
    )
    cpa_time_sensitivity_passed = _strict_report_bool(
        primary_row.get("constant_velocity_cpa_time_sensitivity_passed", False)
    )
    cpa_robustness_sensitivity_passed = bool(
        cpa_pair_sensitivity_passed and cpa_time_sensitivity_passed
    )
    manifest = {
        "schema_version": 4,
        "generated_utc": generated_utc(),
        "git_commit": git_commit_full_hash(),
        "source_worktree": source_worktree,
        "primary_model": primary_model,
        "primary_feature_set": primary_feature_set,
        "claim_decision": {
            "false_alarm_reduction_supported": false_alarm_supported,
            "combined_distance_statistical_gate_passed": (
                combined_distance_statistical_gate_passed
            ),
            "fixed_25_pair_gate_passed": fixed_25_pair_gate_passed,
            "fixed_25_time_gate_passed": fixed_25_time_gate_passed,
            "fixed_25_statistical_gate_passed": (
                fixed_25_statistical_gate_passed
            ),
            "calibrated_distance_pair_gate_passed": (
                calibrated_distance_pair_gate_passed
            ),
            "calibrated_distance_time_gate_passed": (
                calibrated_distance_time_gate_passed
            ),
            "calibrated_distance_statistical_gate_passed": (
                calibrated_distance_statistical_gate_passed
            ),
            "false_alarm_statistical_result": str(primary_row["claim"]),
            "false_alarm_claim": (
                str(primary_row["claim"])
                if false_alarm_supported
                else "publication false-alarm claim not supported after provenance/integrity gates"
            ),
            **adaptability_decision,
            "clean_committed_source": clean_committed_source,
            "archive_resimulation_binding_validated": input_binding_validated,
            "pipeline_integrity_controls_passed": pipeline_integrity_controls_passed,
            "constant_velocity_cpa_pair_sensitivity_passed": (
                cpa_pair_sensitivity_passed
            ),
            "constant_velocity_cpa_time_sensitivity_passed": (
                cpa_time_sensitivity_passed
            ),
            "constant_velocity_cpa_robustness_sensitivity_passed": (
                cpa_robustness_sensitivity_passed
            ),
            "label_permutation_diagnostic_favorable": (
                permutation_diagnostic_favorable
            ),
            "overall_abstract_claim_supported": bool(
                false_alarm_supported
                and statistical_adaptability_supported
            ),
            "scope": (
                "candidate-selected deterministic proxy-risk classification; "
                "not physical Pc or catalogue-wide collision prediction"
            ),
        },
        "archive_resimulation_binding": validated_input_binding,
        "sanity_controls": {
            "proxy_rule_oracle_exact_pipeline_wiring": oracle_passed,
            "label_permutation_pr_auc_below_primary_diagnostic": (
                permutation_diagnostic_favorable
            ),
            "label_permutation_is_claim_gate": False,
        },
        "artifacts": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in unique_artifacts
        ],
    }
    _write_json_atomic(output_path, manifest)


def _load_resimulation_snapshot_records(path: Path) -> list[dict[str, str]]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read canonical resimulation report {path}: {exc}") from exc
    if report.get("failures"):
        raise ValueError("Canonical resimulation report contains failures")
    completed = report.get("completed")
    if not isinstance(completed, list) or not completed:
        raise ValueError("Canonical resimulation report has no completed snapshots")
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in completed:
        if not isinstance(item, dict):
            raise ValueError("Malformed completed entry in resimulation report")
        collection_id = str(item.get("collection_id", "")).strip()
        snapshot_utc = str(item.get("snapshot_utc", "")).strip()
        input_sha256 = str(item.get("input_sha256", "")).strip().lower()
        run_fingerprint = str(item.get("run_fingerprint", "")).strip().lower()
        fingerprint_inputs = item.get("fingerprint_inputs")
        sidecar_sha256 = (
            str(fingerprint_inputs.get("sidecar_sha256", "")).strip().lower()
            if isinstance(fingerprint_inputs, dict)
            else ""
        )
        if (
            not collection_id
            or not snapshot_utc
            or collection_id in seen
            or len(input_sha256) != 64
            or any(character not in "0123456789abcdef" for character in input_sha256)
            or not re.fullmatch(r"[0-9a-f]{64}", sidecar_sha256)
            or not re.fullmatch(r"[0-9a-f]{64}", run_fingerprint)
        ):
            raise ValueError(
                "Resimulation snapshot IDs/timestamps/hashes must be valid and unique"
            )
        seen.add(collection_id)
        records.append(
            {
                "collection_id": collection_id,
                "snapshot_utc": snapshot_utc,
                "input_sha256": input_sha256,
                "sidecar_sha256": sidecar_sha256,
                "run_fingerprint": run_fingerprint,
            }
        )
    return records


def _validate_archive_resimulation_binding(
    archive_report_path: Path,
    snapshot_records: list[dict[str, str]],
    *,
    catalog_version: str,
    catalog_sha256: str,
) -> dict[str, object]:
    """Bind an actual clean-checkout archive import to every resimulated TLE input."""
    try:
        report = json.loads(archive_report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read archive import report {archive_report_path}: {exc}") from exc
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ValueError("Archive import report must use schema_version 1")
    if report.get("verify_only") is not False:
        raise ValueError("Publication requires an actual archive import, not verify-only validation")
    if report.get("clean_checkout_verified") is not True:
        raise ValueError("Publication archive import must verify a clean Git checkout")
    if (
        report.get("source_ancestry_verified") is not True
        or report.get("trusted_source_ref") != "origin/main"
        or not re.fullmatch(
            r"[0-9a-f]{40}", str(report.get("trusted_source_ref_commit", ""))
        )
    ):
        raise ValueError(
            "Publication archive import must verify bundle source ancestry against origin/main"
        )
    revision = str(report.get("archive_revision", "")).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Archive import report must identify a clean committed archive revision")
    manifest_set_sha256 = str(report.get("manifest_set_sha256", "")).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", manifest_set_sha256):
        raise ValueError("Archive import report has an invalid manifest-set SHA-256")
    if (
        report.get("catalog_version") != catalog_version
        or report.get("catalog_sha256") != catalog_sha256
    ):
        raise ValueError("Archive import report does not match the frozen catalogue")

    bundles = report.get("bundles")
    if not isinstance(bundles, list) or not bundles:
        raise ValueError("Archive import report contains no verified bundles")
    if report.get("bundles_verified") != len(bundles):
        raise ValueError("Archive import report bundle count is inconsistent")
    expected_manifest_set = hashlib.sha256(
        json.dumps(bundles, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if expected_manifest_set != manifest_set_sha256:
        raise ValueError("Archive import report manifest-set digest is inconsistent")

    archive_inputs: set[tuple[str, str, str]] = set()
    runtime_statuses: dict[str, int] = {}
    for bundle in bundles:
        if not isinstance(bundle, dict):
            raise ValueError("Archive import report contains a malformed bundle")
        collection_id = str(bundle.get("collection_id", "")).strip()
        bundle_name = str(bundle.get("name", "")).strip()
        input_sha256 = str(bundle.get("input_snapshot_sha256", "")).strip().lower()
        sidecar_sha256 = str(bundle.get("input_sidecar_sha256", "")).strip().lower()
        manifest_sha256 = str(bundle.get("manifest_sha256", "")).strip().lower()
        source_commit = str(bundle.get("source_git_commit", "")).strip().lower()
        runtime_status = str(bundle.get("runtime_provenance_status", "")).strip()
        manifest_schema_version = bundle.get("manifest_schema_version")
        key = (collection_id, input_sha256, sidecar_sha256)
        if (
            not re.fullmatch(r"[0-9]{8}_[0-9]{6}", collection_id)
            or not re.fullmatch(r"[0-9a-f]{64}", input_sha256)
            or not re.fullmatch(r"[0-9a-f]{64}", sidecar_sha256)
            or not re.fullmatch(r"[0-9a-f]{64}", manifest_sha256)
            or not re.fullmatch(r"[0-9a-f]{40}", source_commit)
            or runtime_status
            not in {"verified", "verified_legacy", "legacy_runtime_missing"}
            or manifest_schema_version not in {1, 2}
            or key in archive_inputs
        ):
            raise ValueError("Archive import report bundle provenance is invalid or duplicated")
        archive_inputs.add(key)
        if manifest_schema_version == 2 and runtime_status != "verified":
            raise ValueError("Schema-2 publication bundles require verified runtime provenance")
        if manifest_schema_version == 1 and (
            bundle_name,
            collection_id,
            source_commit,
        ) not in LEGACY_SCHEMA1_ALLOWLIST:
            raise ValueError(
                "Schema-1 publication compatibility is restricted to the three frozen legacy bundles"
            )
        runtime_statuses[runtime_status] = runtime_statuses.get(runtime_status, 0) + 1
    if report.get("collections_verified") != len(archive_inputs):
        raise ValueError("Archive import report collection count is inconsistent")

    resimulation_inputs = {
        (record["collection_id"], record["input_sha256"], record["sidecar_sha256"])
        for record in snapshot_records
    }
    if archive_inputs != resimulation_inputs:
        missing = sorted(resimulation_inputs - archive_inputs)
        extra = sorted(archive_inputs - resimulation_inputs)
        raise ValueError(
            "Archive import and resimulation inputs do not match exactly: "
            f"missing_from_archive={missing}, extra_in_archive={extra}"
        )
    return {
        "validated": True,
        "archive_revision": revision,
        "manifest_set_sha256": manifest_set_sha256,
        "collection_count": len(archive_inputs),
        "runtime_provenance_status_counts": runtime_statuses,
        "binding_key": "collection_id+input_snapshot_sha256+input_sidecar_sha256",
    }


def _validate_canonical_history_binding(
    resimulation_report_path: Path,
    history_path: Path,
    snapshot_records: list[dict[str, str]],
) -> dict[str, object]:
    """Bind the exact rebuilt history bytes to completed resimulation fingerprints."""
    try:
        report = json.loads(resimulation_report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Cannot read canonical resimulation report {resimulation_report_path}: {exc}"
        ) from exc
    if report.get("schema_version") != 2 or report.get("failures"):
        raise ValueError("Canonical resimulation report schema/status is invalid")
    history = report.get("history")
    if not isinstance(history, dict):
        raise ValueError("Canonical resimulation report has no bound history record")
    if history.get("resimulation_integrity_required") is not True:
        raise ValueError("Canonical history was not rebuilt from hash-verified resimulation outputs")
    if not history_path.is_file():
        raise ValueError(f"Canonical history is missing: {history_path}")
    reported_output = Path(str(history.get("output", "")))
    if reported_output.resolve() != history_path.resolve():
        raise ValueError("Resimulation report is bound to a different history path")
    observed_sha256 = file_sha256(history_path)
    if history.get("output_sha256") != observed_sha256:
        raise ValueError("Canonical history SHA-256 does not match the resimulation report")
    if history.get("output_bytes") != history_path.stat().st_size:
        raise ValueError("Canonical history byte count does not match the resimulation report")
    try:
        observed_rows = len(pd.read_csv(history_path, comment="#"))
    except Exception as exc:
        raise ValueError(f"Cannot parse canonical history {history_path}: {exc}") from exc
    if history.get("rows_written") != observed_rows:
        raise ValueError("Canonical history row count does not match the resimulation report")

    expected_fingerprints = {
        record["collection_id"]: record["run_fingerprint"] for record in snapshot_records
    }
    included_runs = history.get("included_runs")
    reported_fingerprints = history.get("resimulation_fingerprints")
    if (
        not isinstance(included_runs, list)
        or set(map(str, included_runs)) != set(expected_fingerprints)
        or reported_fingerprints != expected_fingerprints
    ):
        raise ValueError(
            "Canonical history collections/fingerprints do not match completed resimulations"
        )
    expected_fingerprint_set_sha256 = hashlib.sha256(
        json.dumps(
            expected_fingerprints, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if history.get("resimulation_fingerprint_set_sha256") != expected_fingerprint_set_sha256:
        raise ValueError("Canonical history fingerprint-set digest is inconsistent")
    return {
        "validated": True,
        "path": str(history_path),
        "sha256": observed_sha256,
        "bytes": history_path.stat().st_size,
        "rows": observed_rows,
        "collection_count": len(expected_fingerprints),
        "resimulation_fingerprint_set_sha256": expected_fingerprint_set_sha256,
    }


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
        default=config.adaptability_min_folds,
        help="also run pair-held-out expanding-time CV (mean +/- std per model/metric)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if Path(args.config).resolve() != Path(DEFAULT_EXPERIMENT_CONFIG).resolve():
        raise ValueError(
            "Claim-eligible training requires config/experiment_60_days.json; "
            "a changed protocol requires a new versioned experiment entry point"
        )
    experiment = load_experiment_config(args.config)
    history_path = Path(args.history)
    if history_path.resolve() != Path(experiment.resimulated_history).resolve():
        raise ValueError(
            "Claim-eligible training requires the canonical resimulated history path"
        )
    primary_features = feature_set_by_name(experiment.primary_feature_set)
    snapshot_records = _load_resimulation_snapshot_records(
        Path(experiment.resimulation_report)
    )
    input_binding = _validate_archive_resimulation_binding(
        Path(experiment.archive_import_report),
        snapshot_records,
        catalog_version=experiment.catalog_version,
        catalog_sha256=experiment.catalog_sha256,
    )
    history_binding = _validate_canonical_history_binding(
        Path(experiment.resimulation_report), history_path, snapshot_records
    )
    input_binding["canonical_history"] = history_binding
    source = f"history={args.history}; sha256={history_binding['sha256']}"
    config_summary = (
        f"experiment_config={args.config}, time_column={experiment.time_column}, "
        f"train_time_fraction={experiment.train_time_fraction}, "
        f"test_pair_fraction={experiment.test_pair_fraction}, pair_seed={experiment.pair_seed}"
        f", primary_feature_set={experiment.primary_feature_set}, "
        f"primary_features={'|'.join(primary_features)}"
        f", collection_slot_anchor_basis={experiment.collection_slot_anchor_basis}, "
        f"collection_slot_timestamp_field={experiment.collection_slot_timestamp_field}, "
        f"collection_slot_assignment={experiment.collection_slot_assignment}"
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
        None
        if experiment.collection_start_utc is not None
        else experiment.duration_days - experiment.poll_interval_hours / 24.0
    )
    report_path = Path(args.report)
    gate_report_path = report_path.with_name(f".{report_path.name}.quality-gate.tmp")
    _clear_stale_model_artifacts(report_path)
    report_path.unlink(missing_ok=True)
    gate_report_path.unlink(missing_ok=True)
    gate_report = compare_models(
        Path(args.history), gate_report_path, time_column=experiment.time_column,
        source=source, config_summary=config_summary,
        train_time_fraction=experiment.train_time_fraction,
        test_pair_fraction=experiment.test_pair_fraction,
        pair_seed=experiment.pair_seed,
        minimum_support=minimum_support,
        minimum_observation_span_days=minimum_observation_span_days,
        required_catalog_version=experiment.catalog_version,
        required_catalog_sha256=experiment.catalog_sha256,
        collection_start_utc=experiment.collection_start_utc,
        collection_end_utc=experiment.collection_end_utc,
        poll_interval_hours=experiment.poll_interval_hours,
        min_snapshot_coverage_fraction=experiment.min_snapshot_coverage_fraction,
        max_snapshot_gap_hours=experiment.max_snapshot_gap_hours,
        min_tle_hash_diversity_fraction=experiment.min_tle_hash_diversity_fraction,
        max_identical_tle_hash_run_bins=experiment.max_identical_tle_hash_run_bins,
        maximum_tle_age_hours=experiment.max_tle_age_hours,
        snapshot_records=snapshot_records,
        feature_columns=primary_features,
    )
    if len(gate_report) == 1 and gate_report.iloc[0]["model"] == "not_enough_data":
        gate_report_path.replace(report_path)
        print(f"DATA QUALITY GATE -> {gate_report.iloc[0]['note']}")
        print("No model comparison or publication claims were produced.")
        return

    publication_incomplete_path = report_path.parent / "publication_incomplete.json"
    _write_json_atomic(
        publication_incomplete_path,
        {
            "schema_version": 1,
            "status": "publication_incomplete",
            "generated_utc": generated_utc(),
            "git_commit": git_commit_full_hash(),
            "message": (
                "Publication artifacts are non-authoritative until this sentinel is removed "
                "and publication_manifest.json is written successfully."
            ),
        },
    )

    try:
        evidence_paths = generate_evaluation_evidence(
            Path(args.history), report_path.parent, experiment,
            feature_columns=primary_features,
            snapshot_records=snapshot_records,
            upstream_artifacts=(
                Path(experiment.archive_import_report),
                Path(experiment.resimulation_report),
            ),
            validated_input_binding=input_binding,
        )
        evidence_report = pd.read_csv(evidence_paths["evidence"], comment="#")
        report = canonical_report_from_evidence(gate_report, evidence_report)
        write_csv_text_with_provenance(
            report_path,
            report.to_csv(index=False),
            source=(
                f"held_out_predictions={evidence_paths['predictions']}; "
                f"sha256={file_sha256(evidence_paths['predictions'])}; "
                f"paired_evidence={evidence_paths['evidence']}; "
                f"sha256={file_sha256(evidence_paths['evidence'])}"
            ),
            config_summary=f"{config_summary}, operating_point=inner_validation_frozen",
        )
    except Exception:
        report_path.unlink(missing_ok=True)
        _clear_stale_model_artifacts(report_path)
        raise
    finally:
        gate_report_path.unlink(missing_ok=True)

    plot_model_metrics(report_path, report_path.with_suffix(".png"))
    print(f"OK -> {args.report}")
    print(f"Plot -> {Path(args.report).with_suffix('.png')}")
    print(report.to_string(index=False))
    print()
    print(compare_to_baseline_pr_auc(report))

    ablation_paths: list[Path] = []
    ablation_artifacts: list[Path] = []
    for ablation_name, ablation_features in FROZEN_ABLATION_FEATURES.items():
        staged_path = report_path.with_name(
            report_path.stem + f"_{ablation_name}.csv"
        )
        arm_paths = generate_evaluation_evidence(
            Path(args.history),
            report_path.parent / "ablations" / ablation_name,
            experiment,
            feature_columns=ablation_features,
            snapshot_records=snapshot_records,
            upstream_artifacts=(
                Path(experiment.archive_import_report),
                Path(experiment.resimulation_report),
            ),
            validated_input_binding=input_binding,
            claim_eligible=False,
        )
        arm_evidence = pd.read_csv(arm_paths["evidence"], comment="#")
        staged_report = canonical_report_from_evidence(gate_report, arm_evidence)
        write_csv_text_with_provenance(
            staged_path,
            staged_report.to_csv(index=False),
            source=(
                f"ablation_predictions={arm_paths['predictions']}; "
                f"sha256={file_sha256(arm_paths['predictions'])}; "
                f"ablation_evidence={arm_paths['evidence']}; "
                f"sha256={file_sha256(arm_paths['evidence'])}"
            ),
            config_summary=(
                f"{config_summary}, feature_ablation={ablation_name}, "
                "claim_eligible=false"
            ),
        )
        print()
        print(f"Feature ablation report ({ablation_name}) -> {staged_path}")
        print(staged_report.to_string(index=False))
        ablation_paths.append(staged_path)
        ablation_artifacts.extend(arm_paths.values())

    for evidence_name, evidence_path in evidence_paths.items():
        print(f"Evaluation evidence ({evidence_name}) -> {evidence_path}")

    adaptability_inference_path = report_path.with_name(
        report_path.stem + "_adaptability_inference.json"
    )
    generate_adaptability_inference(
        evidence_paths["predictions"],
        evidence_paths["split_manifest"],
        adaptability_inference_path,
        experiment,
    )
    print(f"Statistical adaptability evidence -> {adaptability_inference_path}")

    publication_plots = create_publication_plots(
        dataset_path=Path(args.history),
        output_dir=report_path.parent,
        time_column=experiment.time_column,
        current_threshold_km=experiment.fixed_threshold_km,
        train_time_fraction=experiment.train_time_fraction,
        test_pair_fraction=experiment.test_pair_fraction,
        pair_seed=experiment.pair_seed,
        predictions_path=evidence_paths["predictions"],
        feature_importance_path=evidence_paths["feature_importance"],
        evaluation_manifest_path=evidence_paths["evaluation_manifest"],
    )
    for plot_path in publication_plots:
        print(f"Publication plot -> {plot_path}")

    cv_artifacts: list[Path] = []
    if args.cv_splits:
        cv_report_path = Path(args.report).with_name(Path(args.report).stem + "_cv.csv")
        cv_report = time_series_cv_report(
            Path(args.history), cv_report_path, n_splits=args.cv_splits, time_column=experiment.time_column,
            source=source, config_summary=f"{config_summary}, n_splits={args.cv_splits}",
            test_pair_fraction=experiment.test_pair_fraction,
            pair_seed=experiment.pair_seed,
            required_catalog_version=experiment.catalog_version,
            required_catalog_sha256=experiment.catalog_sha256,
            collection_start_utc=experiment.collection_start_utc,
            collection_end_utc=experiment.collection_end_utc,
            poll_interval_hours=experiment.poll_interval_hours,
            maximum_tle_age_hours=experiment.max_tle_age_hours,
            snapshot_records=snapshot_records,
            min_tle_hash_diversity_fraction=experiment.min_tle_hash_diversity_fraction,
            max_identical_tle_hash_run_bins=experiment.max_identical_tle_hash_run_bins,
            feature_columns=primary_features,
            primary_model=experiment.primary_model,
            adaptability_min_folds=experiment.adaptability_min_folds,
            minimum_support=minimum_support,
        )
        print()
        print(f"CV report ({args.cv_splits}-fold pair/time split) -> {cv_report_path}")
        print(cv_report.to_string(index=False))
        cv_artifacts.append(cv_report_path)
        fold_path = cv_report_path.with_name(cv_report_path.stem + "_folds.csv")
        adaptability_path = cv_report_path.with_name(
            cv_report_path.stem + "_adaptability.csv"
        )
        cv_artifacts.extend(
            path for path in (fold_path, adaptability_path) if path.is_file()
        )

    publication_manifest_path = report_path.parent / "publication_manifest.json"
    _write_publication_manifest(
        publication_manifest_path,
        [
            report_path,
            report_path.with_suffix(".png"),
            *ablation_paths,
            *ablation_artifacts,
            *evidence_paths.values(),
            adaptability_inference_path,
            *publication_plots,
            *cv_artifacts,
        ],
        primary_model=experiment.primary_model,
        primary_feature_set=experiment.primary_feature_set,
        evidence_report=evidence_report,
        validated_input_binding=input_binding,
        canonical_report_path=report_path,
    )
    publication_incomplete_path.unlink(missing_ok=True)
    print(f"Publication manifest -> {publication_manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        try:
            failed_args = parse_args()
            failed_config = load_experiment_config(failed_args.config)
            failed_report = Path(failed_args.report or failed_config.time_split_report)
            _cleanup_failed_publication(failed_report)
        except BaseException:
            pass
        raise
