from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
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


def compare_models(dataset_path: Path, report_path: Path, time_column: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    missing = [feature for feature in FEATURES + ["risk_label", "fixed_threshold_alarm"] if feature not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    if df.empty or df["risk_label"].nunique() < 2 or len(df) < 6:
        report = pd.DataFrame(
            [
                {
                    "model": "not_enough_data",
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "accuracy": None,
                    "false_positive": None,
                    "false_negative": None,
                    "true_positive": None,
                    "true_negative": None,
                    "train_rows": None,
                    "test_rows": None,
                    "split": None,
                    "note": "Need at least 6 rows and two risk_label classes.",
                }
            ]
        )
        report.to_csv(report_path, index=False)
        return report

    x_train, x_test, y_train, y_test, split_name = _split_data(df, time_column)

    # A learner needs both classes in the training split. With rare conjunction
    # events and a time-ordered split this is not guaranteed, so we report the
    # situation instead of letting scikit-learn raise.
    trainable = y_train.nunique() >= 2

    models = {
        "fixed_threshold": None,
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "svm": make_pipeline(StandardScaler(), SVC(kernel="rbf")),
    }

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(eval_metric="logloss", random_state=42)
    except Exception:
        pass

    rows = []
    for name, model in models.items():
        note = ""
        if name == "fixed_threshold":
            # Purely distance-based baseline: no training required.
            pred = df.loc[x_test.index, "fixed_threshold_alarm"].astype(int)
        elif not trainable:
            rows.append(
                {
                    "model": name,
                    "precision": None, "recall": None, "f1": None, "accuracy": None,
                    "false_positive": None, "false_negative": None,
                    "true_positive": None, "true_negative": None,
                    "train_rows": int(len(y_train)), "test_rows": int(len(y_test)),
                    "split": split_name,
                    "note": "Training split had a single class; model skipped.",
                }
            )
            continue
        else:
            model.fit(x_train, y_train)
            pred = model.predict(x_test)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, pred, average="binary", zero_division=0
        )
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "model": name,
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
                "note": "",
            }
        )

    if "xgboost" not in models:
        rows.append(
            {
                "model": "xgboost",
                "precision": None,
                "recall": None,
                "f1": None,
                "accuracy": None,
                "false_positive": None,
                "false_negative": None,
                "true_positive": None,
                "true_negative": None,
                "train_rows": int(len(y_train)),
                "test_rows": int(len(y_test)),
                "split": split_name,
                "note": "xgboost is not installed.",
            }
        )

    report = pd.DataFrame(rows)
    report.to_csv(report_path, index=False)
    return report
