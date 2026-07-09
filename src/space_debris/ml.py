from __future__ import annotations

from pathlib import Path

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


# NOTE: ``risk_score`` and the raw label-defining columns are intentionally
# absent. The ground-truth label (risk_label) is a function of min_distance_km
# and relative_velocity_km_s, so a learner is given those physical predictors
# and must recover the decision boundary itself. ``risk_score`` is excluded
# because it is a monotone transform of the same quantities and would leak the
# target. This is what lets the ML models be compared fairly against the
# fixed-distance baseline.
FEATURES = [
    "time_to_tca_min",
    "current_distance_km",
    "min_distance_km",
    "relative_velocity_km_s",
    "altitude_difference_km",
    "max_tle_age_hours",
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
    "split",
    "note",
]


def _split_data(df: pd.DataFrame, time_column: str | None):
    x = df[FEATURES]
    y = df["risk_label"].astype(int)
    if time_column and time_column in df.columns:
        ordered = df.sort_values(time_column).reset_index(drop=True)
        split_at = max(1, int(len(ordered) * 0.75))
        split_at = min(split_at, len(ordered) - 1)
        train = ordered.iloc[:split_at]
        test = ordered.iloc[split_at:]
        return train[FEATURES], test[FEATURES], train["risk_label"].astype(int), test["risk_label"].astype(int), "time"

    stratify = y if y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.35, random_state=42, stratify=stratify
    )
    return x_train, x_test, y_train, y_test, "random"


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
        "split": split,
        "note": note,
    }


def _evaluate_split(df: pd.DataFrame, x_train, x_test, y_train, y_test, split_name: str) -> list[dict]:
    """Fit every model on one train/test split and score it. Shared by the
    single chronological split (compare_models) and each fold of
    TimeSeriesSplit cross-validation (time_series_cv_report).
    """
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
            pred = df.loc[x_test.index, "fixed_threshold_alarm"].astype(int)
            scores = pred.to_numpy()
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

    return rows


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


def compare_models(dataset_path: Path, report_path: Path, time_column: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    missing = [feature for feature in FEATURES + ["risk_label", "fixed_threshold_alarm"] if feature not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    if df.empty or df["risk_label"].nunique() < 2 or len(df) < 6:
        report = pd.DataFrame(
            [_not_enough_data_row("not_enough_data", "Need at least 6 rows and two risk_label classes.")]
        )
        report.to_csv(report_path, index=False)
        return report

    x_train, x_test, y_train, y_test, split_name = _split_data(df, time_column)
    rows = _evaluate_split(df, x_train, x_test, y_train, y_test, split_name)

    report = pd.DataFrame(rows)[REPORT_COLUMNS]
    report.to_csv(report_path, index=False)
    return report


def time_series_cv_report(
    dataset_path: Path,
    report_path: Path,
    n_splits: int = 5,
    time_column: str = "snapshot_utc",
) -> pd.DataFrame:
    """TimeSeriesSplit cross-validation, aggregated as mean +/- std per
    model/metric. This is an OPTION alongside (not a replacement for) the
    single chronological split in compare_models(): more folds give a less
    noisy read of how stable each model's PR-AUC/ROC-AUC is across the
    accumulated history, at the cost of smaller/earlier training folds.
    """
    df = pd.read_csv(dataset_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    missing = [feature for feature in FEATURES + ["risk_label", "fixed_threshold_alarm"] if feature not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if time_column not in df.columns:
        raise ValueError(f"time_column={time_column!r} not found in dataset")

    ordered = df.sort_values(time_column).reset_index(drop=True)

    if len(ordered) < n_splits + 1 or ordered["risk_label"].nunique() < 2:
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
        report.to_csv(report_path, index=False)
        return report

    x = ordered[FEATURES]
    y = ordered["risk_label"].astype(int)
    splitter = TimeSeriesSplit(n_splits=n_splits)

    fold_rows: list[dict] = []
    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(x)):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        fold_rows.extend(
            _evaluate_split(ordered, x_train, x_test, y_train, y_test, split_name=f"cv_fold_{fold_index}")
        )

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
                }
            )

    report = pd.DataFrame(agg_rows)
    report.to_csv(report_path, index=False)
    return report
