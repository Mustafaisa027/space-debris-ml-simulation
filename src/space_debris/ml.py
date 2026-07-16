from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from space_debris.provenance import write_csv_text_with_provenance


# NOTE: ``risk_score`` and the raw label-defining columns are intentionally
# absent. The ground-truth label (risk_label) is a function of min_distance_km
# and relative_velocity_km_s, so a learner is given those physical predictors
# and must recover the decision boundary itself. ``risk_score`` is excluded
# because it is a monotone transform of the same quantities and would leak the
# target. This is what lets the ML models be compared fairly against the
# fixed-distance baseline.
#
# The relative_*/radial_velocity/tangential_velocity/approach_angle features
# (ROADMAP_YOL1.md GOREV 5) are geometric decompositions computed from the
# position/velocity VECTORS, not rescalings of the label-defining scalars:
# the RIC components describe orientation (radial/in-track/cross-track), not
# magnitude; radial/tangential velocity and approach_angle are evaluated at
# snapshot time rather than at TCA; and relative_inclination_deg is pure
# orbital-plane geometry. None of them is a duplicate or monotone transform
# of min_distance_km / relative_velocity_km_s.
FEATURES = [
    "time_to_tca_min",
    "current_distance_km",
    "min_distance_km",
    "relative_velocity_km_s",
    "altitude_difference_km",
    "max_tle_age_hours",
    "relative_radial_km",
    "relative_intrack_km",
    "relative_crosstrack_km",
    "relative_inclination_deg",
    "radial_velocity_km_s",
    "tangential_velocity_km_s",
    "approach_angle_deg",
]

# Reported for every model. pr_auc/roc_auc lead the report and are the
# metrics the paper leans on; accuracy is kept for context but is
# deliberately NOT the headline -- with rare positives, a model that always
# predicts "not risky" still scores >99% accuracy while missing every
# conjunction (see ROADMAP_YOL1.md observations G5/G6).
REPORT_COLUMNS = [
    "model",
    "pr_auc",
    "roc_auc",
    "precision",
    "recall",
    "f1",
    "accuracy",
    "false_positive",
    "false_negative",
    "true_positive",
    "true_negative",
    "train_rows",
    "test_rows",
    "train_pairs",
    "test_pairs",
    "excluded_rows",
    "cutoff_utc",
    "split",
    "note",
]


PAIR_COLUMNS = ["object_1", "object_2"]
PAIR_CATALOG_ID_COLUMNS = ["object_1_catalog_id", "object_2_catalog_id"]
PAIR_SPLIT_SEED = "iac26-pair-split-v1"


class InsufficientGroupedSplitError(ValueError):
    """Raised when independent pair cohorts cannot form train and test sets."""


@dataclass(frozen=True)
class SplitResult:
    train: pd.DataFrame
    test: pd.DataFrame
    excluded: pd.DataFrame
    split_name: str
    metadata: dict[str, object]


def _canonical_pair_ids(df: pd.DataFrame) -> pd.Series:
    missing = [column for column in PAIR_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Pair-grouped split requires columns: {missing}")
    if df[PAIR_COLUMNS].isna().any().any():
        raise ValueError("Pair identity columns cannot contain missing values")
    use_catalog_ids = all(column in df.columns for column in PAIR_CATALOG_ID_COLUMNS)
    if use_catalog_ids:
        catalog = df[PAIR_CATALOG_ID_COLUMNS]
        use_catalog_ids = not catalog.isna().any().any() and not (
            catalog.astype(str).apply(lambda column: column.str.strip().eq("")).any().any()
        )
    identity_columns = PAIR_CATALOG_ID_COLUMNS if use_catalog_ids else PAIR_COLUMNS
    left = df[identity_columns[0]].astype(str).str.strip()
    right = df[identity_columns[1]].astype(str).str.strip()
    if (left == "").any() or (right == "").any():
        raise ValueError("Pair identity columns cannot contain empty values")
    return pd.Series(
        np.where(left <= right, left + "|" + right, right + "|" + left),
        index=df.index,
        name="canonical_pair_id",
    )


def _validated_times(df: pd.DataFrame, time_column: str) -> pd.Series:
    if time_column not in df.columns:
        raise ValueError(f"time_column={time_column!r} not found in dataset")
    return pd.to_datetime(df[time_column], utc=True, errors="raise", format="ISO8601")


def _pair_is_test(pair_id: str, test_pair_fraction: float, pair_seed: str) -> bool:
    digest = hashlib.sha256(f"{pair_seed}|{pair_id}".encode("utf-8")).digest()
    score = int.from_bytes(digest[:8], "big") / 2**64
    return score < test_pair_fraction


def _pair_grouped_time_split(
    df: pd.DataFrame,
    time_column: str,
    train_time_fraction: float = 0.75,
    test_pair_fraction: float = 0.25,
    pair_seed: str = PAIR_SPLIT_SEED,
) -> SplitResult:
    """Hold out deterministic pairs in a strictly future snapshot window.

    SHA-256 assigns every canonical pair permanently to the train or test
    group. The cutoff lies between complete timestamp blocks. Training uses
    only train-pair observations before the cutoff; testing uses only test-pair
    observations at/after it. Cross-quadrant rows are retained as ``excluded``
    metadata rather than silently leaking into the opposite partition.
    """
    if not 0.0 < train_time_fraction < 1.0:
        raise ValueError("train_time_fraction must be between 0 and 1")
    if not 0.0 < test_pair_fraction < 1.0:
        raise ValueError("test_pair_fraction must be between 0 and 1")
    timestamps = _validated_times(df, time_column)
    pair_ids = _canonical_pair_ids(df)
    unique_times = pd.Index(timestamps.unique()).sort_values()
    if len(unique_times) < 2:
        raise InsufficientGroupedSplitError("Need at least two distinct snapshot timestamps.")
    cutoff_index = min(max(1, int(np.ceil(len(unique_times) * train_time_fraction))), len(unique_times) - 1)
    cutoff = unique_times[cutoff_index]

    pair_assignment = {
        pair_id: _pair_is_test(pair_id, test_pair_fraction, pair_seed)
        for pair_id in pair_ids.unique()
    }
    is_test_pair = pair_ids.map(pair_assignment).astype(bool)
    train_mask = (~is_test_pair) & (timestamps < cutoff)
    test_mask = is_test_pair & (timestamps >= cutoff)
    excluded_mask = ~(train_mask | test_mask)
    train = df.loc[train_mask].copy()
    test = df.loc[test_mask].copy()
    excluded = df.loc[excluded_mask].copy()
    if train.empty or test.empty:
        raise InsufficientGroupedSplitError(
            "Pair/time intersection produced an empty train or test partition."
        )

    train_pairs = set(pair_ids[train_mask])
    test_pairs = set(pair_ids[test_mask])
    if train_pairs & test_pairs:
        raise AssertionError("Canonical pair leakage detected in grouped split")
    train_times = timestamps[train_mask]
    test_times = timestamps[test_mask]
    if not train_times.max() < test_times.min():
        raise AssertionError("Chronological leakage detected in grouped split")

    metadata = {
        "cutoff_utc": cutoff.isoformat(),
        "pair_seed": pair_seed,
        "test_pair_fraction": test_pair_fraction,
        "train_time_fraction": train_time_fraction,
        "train_pairs": len(train_pairs),
        "test_pairs": len(test_pairs),
        "train_rows": len(train),
        "test_rows": len(test),
        "excluded_rows": len(excluded),
    }
    return SplitResult(train, test, excluded, "pair_grouped_time", metadata)


def _split_data(
    df: pd.DataFrame,
    time_column: str | None,
    *,
    train_time_fraction: float = 0.75,
    test_pair_fraction: float = 0.25,
    pair_seed: str = PAIR_SPLIT_SEED,
) -> SplitResult:
    y = df["risk_label"].astype(int)
    if time_column:
        return _pair_grouped_time_split(
            df, time_column, train_time_fraction, test_pair_fraction, pair_seed
        )

    stratify = y if y.value_counts().min() >= 2 else None
    train, test = train_test_split(
        df, test_size=0.35, random_state=42, stratify=stratify
    )
    return SplitResult(
        train.copy(), test.copy(), df.iloc[0:0].copy(), "random",
        {"train_rows": len(train), "test_rows": len(test), "excluded_rows": 0},
    )


def _ranking_score(model, x_test):
    """Continuous score for ranking-based metrics (PR-AUC/ROC-AUC).

    Prefers predict_proba (probability of the positive class), falls back to
    decision_function (e.g. an SVC without probability calibration enabled),
    and finally to hard 0/1 predictions if neither is available.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(x_test)
    return model.predict(x_test)


def _ranking_metrics(y_true, scores) -> tuple[float | None, float | None, str]:
    """average_precision_score / roc_auc_score are undefined with a single
    class in y_true; report None with a note instead of letting sklearn raise.
    """
    if pd.Series(y_true).nunique() < 2:
        return None, None, "single class in test split; PR-AUC/ROC-AUC undefined"
    return (
        float(average_precision_score(y_true, scores)),
        float(roc_auc_score(y_true, scores)),
        "",
    )


def _build_models(scale_pos_weight: float) -> dict:
    """Every learner is imbalance-aware: class_weight='balanced' re-weights
    the rare positive class for LogReg/RF/SVM, and XGBoost's scale_pos_weight
    (n_negative / n_positive of the TRAINING split) plays the same role.
    fixed_threshold is not a learner (None) -- it needs no weighting.
    """
    models: dict = {
        "fixed_threshold": None,
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
        ),
        "decision_tree": DecisionTreeClassifier(
            random_state=42, class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"
        ),
        "svm": make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced")),
    }
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            eval_metric="logloss", random_state=42, scale_pos_weight=scale_pos_weight
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = LGBMClassifier(
            random_state=42,
            class_weight="balanced",
            verbosity=-1,
        )
    except Exception:
        pass
    return models


def _not_enough_data_row(model: str, note: str, train_rows=None, test_rows=None, split=None) -> dict:
    return {
        "model": model,
        "pr_auc": None,
        "roc_auc": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "accuracy": None,
        "false_positive": None,
        "false_negative": None,
        "true_positive": None,
        "true_negative": None,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "train_pairs": None,
        "test_pairs": None,
        "excluded_rows": None,
        "cutoff_utc": None,
        "split": split,
        "note": note,
    }


def _evaluate_split(split: SplitResult, split_name: str | None = None) -> list[dict]:
    """Fit every model on one train/test split and score it. Shared by the
    single chronological split (compare_models) and each fold of
    TimeSeriesSplit cross-validation (time_series_cv_report).
    """
    train = split.train
    test = split.test
    x_train, x_test = train[FEATURES], test[FEATURES]
    y_train = train["risk_label"].astype(int)
    y_test = test["risk_label"].astype(int)
    split_name = split_name or split.split_name
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    models = _build_models(scale_pos_weight)
    trainable = y_train.nunique() >= 2

    rows: list[dict] = []
    for name, model in models.items():
        if name == "fixed_threshold":
            # Purely distance-based baseline: no training required. Its
            # binary alarm doubles as a (degenerate, two-level) ranking
            # score so PR-AUC/ROC-AUC stay comparable across all models.
            pred = test["fixed_threshold_alarm"].astype(int).to_numpy()
            scores = pred
        elif not trainable:
            rows.append(
                _not_enough_data_row(
                    name,
                    "Training split had a single class; model skipped.",
                    train_rows=int(len(y_train)),
                    test_rows=int(len(y_test)),
                    split=split_name,
                )
            )
            continue
        else:
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            scores = _ranking_score(model, x_test)

        pr_auc, roc_auc, ranking_note = _ranking_metrics(y_test, scores)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, pred, average="binary", zero_division=0
        )
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "model": name,
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy_score(y_test, pred),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
                "true_negative": int(tn),
                "train_rows": int(len(y_train)),
                "test_rows": int(len(y_test)),
                "train_pairs": split.metadata.get("train_pairs"),
                "test_pairs": split.metadata.get("test_pairs"),
                "excluded_rows": split.metadata.get("excluded_rows", len(split.excluded)),
                "cutoff_utc": split.metadata.get("cutoff_utc"),
                "split": split_name,
                "note": ranking_note,
            }
        )

    if "xgboost" not in models:
        rows.append(
            _not_enough_data_row(
                "xgboost",
                "xgboost is not installed.",
                train_rows=int(len(y_train)),
                test_rows=int(len(y_test)),
                split=split_name,
            )
        )
    if "lightgbm" not in models:
        rows.append(
            _not_enough_data_row(
                "lightgbm",
                "lightgbm is not installed.",
                train_rows=int(len(y_train)),
                test_rows=int(len(y_test)),
                split=split_name,
            )
        )

    return rows


def model_predictions_for_plotting(
    dataset_path: Path,
    time_column: str | None = None,
    *,
    train_time_fraction: float = 0.75,
    test_pair_fraction: float = 0.25,
    pair_seed: str = PAIR_SPLIT_SEED,
) -> dict:
    """Per-model test-split predictions/scores/fitted estimator, for
    PUBLICATION FIGURES ONLY (PR curves, confusion matrices, feature
    importance in space_debris.plots) -- compare_models()/
    time_series_cv_report() remain the source of truth for the numeric report.

    Returns {model_name: {"y_true", "scores", "pred", "model"}}; empty dict if
    there is not enough data to form a meaningful split (mirrors
    compare_models()'s "not enough data" guard).
    """
    df = pd.read_csv(dataset_path, comment="#")
    missing = [feature for feature in FEATURES + ["risk_label", "fixed_threshold_alarm"] if feature not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if df.empty or df["risk_label"].nunique() < 2 or len(df) < 6:
        return {}

    try:
        split = _split_data(
            df,
            time_column,
            train_time_fraction=train_time_fraction,
            test_pair_fraction=test_pair_fraction,
            pair_seed=pair_seed,
        )
    except InsufficientGroupedSplitError:
        return {}
    train, test = split.train, split.test
    if train["risk_label"].nunique() < 2 or test["risk_label"].nunique() < 2:
        return {}
    x_train, x_test = train[FEATURES], test[FEATURES]
    y_train, y_test = train["risk_label"].astype(int), test["risk_label"].astype(int)
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    models = _build_models(scale_pos_weight)
    trainable = y_train.nunique() >= 2

    predictions: dict = {}
    for name, model in models.items():
        if name == "fixed_threshold":
            pred = test["fixed_threshold_alarm"].astype(int).to_numpy()
            scores = pred
            fitted_model = None
        elif not trainable:
            continue
        else:
            model.fit(x_train, y_train)
            pred = np.asarray(model.predict(x_test))
            scores = np.asarray(_ranking_score(model, x_test))
            fitted_model = model

        predictions[name] = {
            "y_true": y_test.to_numpy(),
            "scores": scores,
            "pred": pred,
            "model": fitted_model,
        }

    return predictions


def compare_to_baseline_pr_auc(report: pd.DataFrame) -> str:
    """Human-readable PR-AUC comparison of each model against the classical
    fixed-distance baseline -- the paper's central question is whether ML
    beats a naive threshold on the imbalanced screening set, and PR-AUC (not
    accuracy) is the metric that actually answers it under class imbalance.
    """
    if "fixed_threshold" not in report["model"].values:
        return "fixed_threshold baseline not present in report."
    baseline_pr_auc = report.loc[report["model"] == "fixed_threshold", "pr_auc"].iloc[0]
    if pd.isna(baseline_pr_auc):
        return "fixed_threshold PR-AUC undefined (single class in test split)."

    lines = [f"fixed_threshold PR-AUC = {baseline_pr_auc:.4f} (baseline)"]
    for _, row in report.iterrows():
        if row["model"] == "fixed_threshold" or pd.isna(row["pr_auc"]):
            continue
        delta = row["pr_auc"] - baseline_pr_auc
        verdict = "beats" if delta > 0 else "does not beat"
        lines.append(
            f"{row['model']} PR-AUC = {row['pr_auc']:.4f} ({verdict} baseline, delta={delta:+.4f})"
        )
    return "\n".join(lines)


def compare_models(
    dataset_path: Path,
    report_path: Path,
    time_column: str | None = None,
    *,
    source: str = "",
    config_summary: str = "",
    train_time_fraction: float = 0.75,
    test_pair_fraction: float = 0.25,
    pair_seed: str = PAIR_SPLIT_SEED,
) -> pd.DataFrame:
    df = pd.read_csv(dataset_path, comment="#")
    missing = [feature for feature in FEATURES + ["risk_label", "fixed_threshold_alarm"] if feature not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    if not source:
        source = f"dataset={dataset_path}"
    if not config_summary:
        config_summary = f"time_column={time_column}"

    if df.empty or df["risk_label"].nunique() < 2 or len(df) < 6:
        report = pd.DataFrame(
            [_not_enough_data_row("not_enough_data", "Need at least 6 rows and two risk_label classes.")]
        )
        write_csv_text_with_provenance(report_path, report.to_csv(index=False), source, config_summary)
        return report

    try:
        split = _split_data(
            df,
            time_column,
            train_time_fraction=train_time_fraction,
            test_pair_fraction=test_pair_fraction,
            pair_seed=pair_seed,
        )
    except InsufficientGroupedSplitError as exc:
        report = pd.DataFrame([_not_enough_data_row("not_enough_data", str(exc))])
        write_csv_text_with_provenance(report_path, report.to_csv(index=False), source, config_summary)
        return report
    train_classes = split.train["risk_label"].nunique()
    test_classes = split.test["risk_label"].nunique()
    if train_classes < 2 or test_classes < 2:
        note = (
            "Grouped train and test partitions must each contain both risk_label classes; "
            f"got train_classes={train_classes}, test_classes={test_classes}."
        )
        row = _not_enough_data_row(
            "not_enough_data", note, len(split.train), len(split.test), split.split_name
        )
        row.update(
            train_pairs=split.metadata.get("train_pairs"),
            test_pairs=split.metadata.get("test_pairs"),
            excluded_rows=split.metadata.get("excluded_rows"),
            cutoff_utc=split.metadata.get("cutoff_utc"),
        )
        report = pd.DataFrame([row])[REPORT_COLUMNS]
        write_csv_text_with_provenance(report_path, report.to_csv(index=False), source, config_summary)
        return report
    rows = _evaluate_split(split)

    report = pd.DataFrame(rows)[REPORT_COLUMNS]
    write_csv_text_with_provenance(report_path, report.to_csv(index=False), source, config_summary)
    return report


def time_series_cv_report(
    dataset_path: Path,
    report_path: Path,
    n_splits: int = 5,
    time_column: str = "snapshot_utc",
    *,
    source: str = "",
    config_summary: str = "",
    test_pair_fraction: float = 0.25,
    pair_seed: str = PAIR_SPLIT_SEED,
) -> pd.DataFrame:
    """Pair-held-out expanding-window validation over whole timestamp blocks.

    Train folds contain only deterministic train-pair observations from past
    blocks; test folds contain only deterministic test-pair observations in
    the future block. Cross-quadrant observations are excluded, preventing
    both pair memorization and future leakage.
    """
    df = pd.read_csv(dataset_path, comment="#")
    missing = [feature for feature in FEATURES + ["risk_label", "fixed_threshold_alarm"] if feature not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if time_column not in df.columns:
        raise ValueError(f"time_column={time_column!r} not found in dataset")

    if not source:
        source = f"dataset={dataset_path}"
    if not config_summary:
        config_summary = f"time_column={time_column}, n_splits={n_splits}"

    if len(df) < n_splits + 1 or df["risk_label"].nunique() < 2:
        report = pd.DataFrame(
            [
                {
                    "model": "not_enough_data",
                    "metric": None,
                    "mean": None,
                    "std": None,
                    "folds": 0,
                    "note": f"Need > {n_splits} rows and two risk_label classes for {n_splits}-fold TimeSeriesSplit.",
                }
            ]
        )
        write_csv_text_with_provenance(report_path, report.to_csv(index=False), source, config_summary)
        return report

    timestamps = _validated_times(df, time_column)
    pair_ids = _canonical_pair_ids(df)
    time_blocks = np.asarray(list(pd.Index(timestamps.unique()).sort_values()), dtype=object)
    if len(time_blocks) < n_splits + 1:
        report = pd.DataFrame(
            [{"model": "not_enough_data", "metric": None, "mean": None, "std": None,
              "folds": 0, "note": f"Need > {n_splits} snapshot blocks for grouped CV."}]
        )
        write_csv_text_with_provenance(report_path, report.to_csv(index=False), source, config_summary)
        return report
    pair_assignment = {
        pair_id: _pair_is_test(pair_id, test_pair_fraction, pair_seed)
        for pair_id in pair_ids.unique()
    }
    is_test_pair = pair_ids.map(pair_assignment).astype(bool)
    splitter = TimeSeriesSplit(n_splits=n_splits)

    fold_rows: list[dict] = []
    valid_partitions = 0
    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(time_blocks)):
        train_times = set(time_blocks[train_idx])
        test_times = set(time_blocks[test_idx])
        train_mask = (~is_test_pair) & timestamps.isin(train_times)
        test_mask = is_test_pair & timestamps.isin(test_times)
        if not train_mask.any() or not test_mask.any():
            continue
        split = SplitResult(
            train=df.loc[train_mask].copy(),
            test=df.loc[test_mask].copy(),
            excluded=df.loc[~(train_mask | test_mask)].copy(),
            split_name=f"pair_grouped_cv_fold_{fold_index}",
            metadata={"train_time_blocks": len(train_times), "test_time_blocks": len(test_times)},
        )
        valid_partitions += 1
        fold_rows.extend(
            _evaluate_split(split)
        )

    if not fold_rows:
        report = pd.DataFrame(
            [{"model": "not_enough_data", "metric": None, "mean": None, "std": None,
              "folds": 0, "note": "All pair/time CV intersections were empty."}]
        )
        write_csv_text_with_provenance(report_path, report.to_csv(index=False), source, config_summary)
        return report

    fold_df = pd.DataFrame(fold_rows)
    metrics = ["pr_auc", "roc_auc", "precision", "recall", "f1", "accuracy"]

    agg_rows: list[dict] = []
    for model_name, group in fold_df.groupby("model"):
        for metric in metrics:
            values = group[metric].dropna()
            agg_rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "mean": float(values.mean()) if len(values) else None,
                    "std": float(values.std()) if len(values) > 1 else (0.0 if len(values) == 1 else None),
                    "folds": int(len(values)),
                    "total_folds": int(n_splits),
                    "valid_partitions": int(valid_partitions),
                }
            )

    report = pd.DataFrame(agg_rows)
    write_csv_text_with_provenance(report_path, report.to_csv(index=False), source, config_summary)
    return report
