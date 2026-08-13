"""Train and compare churn models for the IBM Telco Customer Churn dataset.

Example
-------
python -m churn_prediction.train --data data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from math import ceil
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOGGER = logging.getLogger(__name__)
RANDOM_STATE = 42
TARGET = "Churn"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train telecom churn classifiers.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"),
        help="Path to the IBM Telco Customer Churn CSV file.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction of data reserved for final evaluation (default: 0.20).",
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.10,
        help="Highest-risk fraction for the retention targeting analysis (default: 0.10).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Directory where the selected model is saved.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports"),
        help="Directory where metrics, predictions, and charts are saved.",
    )
    return parser.parse_args()


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load data and return features, binary target, and customer identifiers."""
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}. Run `python scripts/download_data.py` first "
            "or pass --data with the CSV location."
        )

    data = pd.read_csv(path)
    required_columns = {"customerID", "TotalCharges", TARGET}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"The data file is missing expected columns: {sorted(missing_columns)}")

    # The raw IBM file stores blank total charges for customers with no tenure.
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    target = data[TARGET].map({"Yes": 1, "No": 0})
    if target.isna().any():
        raise ValueError("Churn must contain only 'Yes' and 'No' values.")

    customer_ids = data["customerID"].copy()
    features = data.drop(columns=[TARGET, "customerID"])
    return features, target.astype(int), customer_ids


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing that is fitted inside each estimator pipeline."""
    numeric_features = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [column for column in features.columns if column not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def build_models(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    """Create comparable pipelines with imbalance-aware classifiers."""
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2_000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }
    # A transformer is cloned when Pipeline is fitted, keeping every candidate isolated.
    return {
        name: Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        for name, estimator in models.items()
    }


def score_model(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    top_fraction: float,
) -> tuple[dict[str, float | int], pd.Series]:
    """Score a fitted model, including the retention team's top-risk segment."""
    probabilities = pd.Series(model.predict_proba(x_test)[:, 1], index=x_test.index, name="churn_risk")
    predictions = (probabilities >= 0.50).astype(int)
    target_count = max(1, ceil(len(y_test) * top_fraction))
    highest_risk_indexes = probabilities.nlargest(target_count).index
    captured_churners = int(y_test.loc[highest_risk_indexes].sum())
    total_churners = int(y_test.sum())

    metrics: dict[str, float | int] = {
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "average_precision": round(float(average_precision_score(y_test, probabilities)), 4),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision_at_0_50": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall_at_0_50": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "top_risk_fraction": top_fraction,
        "customers_targeted": target_count,
        "churners_captured": captured_churners,
        "total_test_churners": total_churners,
        "churner_capture_rate": round(captured_churners / total_churners, 4) if total_churners else 0.0,
        "top_segment_precision": round(float(y_test.loc[highest_risk_indexes].mean()), 4),
    }
    return metrics, probabilities


def save_evaluation_charts(
    y_test: pd.Series,
    model_scores: dict[str, pd.Series],
    report_dir: Path,
) -> None:
    """Save ROC and precision-recall comparison charts."""
    sns.set_theme(style="whitegrid", context="notebook")

    fig, axis = plt.subplots(figsize=(8, 6))
    for name, scores in model_scores.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, scores)
        axis.plot(false_positive_rate, true_positive_rate, label=name.replace("_", " ").title())
    axis.plot([0, 1], [0, 1], "--", color="gray", label="No-skill baseline")
    axis.set(title="ROC Curve: Churn Models", xlabel="False Positive Rate", ylabel="True Positive Rate")
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(report_dir / "roc_curve_comparison.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 6))
    prevalence = y_test.mean()
    for name, scores in model_scores.items():
        precision, recall, _ = precision_recall_curve(y_test, scores)
        axis.plot(recall, precision, label=name.replace("_", " ").title())
    axis.axhline(prevalence, linestyle="--", color="gray", label="No-skill baseline")
    axis.set(title="Precision-Recall Curve: Churn Models", xlabel="Recall", ylabel="Precision")
    axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(report_dir / "precision_recall_comparison.png", dpi=180)
    plt.close(fig)


def select_best_model(results: pd.DataFrame) -> str:
    """Select the model with the highest ROC-AUC, then average precision."""
    return results.sort_values(["roc_auc", "average_precision"], ascending=False).index[0]


def main() -> None:
    """Train both candidates, report performance, and save the winner."""
    args = parse_args()
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")
    if not 0 < args.top_fraction <= 1:
        raise ValueError("--top-fraction must be in (0, 1].")

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    features, target, customer_ids = load_data(args.data)
    x_train, x_test, y_train, y_test, _, ids_test = train_test_split(
        features,
        target,
        customer_ids,
        test_size=args.test_size,
        stratify=target,
        random_state=RANDOM_STATE,
    )
    LOGGER.info("Training on %d records; evaluating on %d records.", len(x_train), len(x_test))

    preprocessor = build_preprocessor(x_train)
    models = build_models(preprocessor)
    metrics_by_model: dict[str, dict[str, float | int]] = {}
    scores_by_model: dict[str, pd.Series] = {}
    fitted_models: dict[str, Pipeline] = {}

    for name, pipeline in models.items():
        LOGGER.info("Fitting %s...", name)
        pipeline.fit(x_train, y_train)
        metrics, probabilities = score_model(pipeline, x_test, y_test, args.top_fraction)
        metrics_by_model[name] = metrics
        scores_by_model[name] = probabilities
        fitted_models[name] = pipeline

    results = pd.DataFrame.from_dict(metrics_by_model, orient="index")
    results.index.name = "model"
    results = results.sort_values(["roc_auc", "average_precision"], ascending=False)
    best_name = select_best_model(results)
    best_model = fitted_models[best_name]

    joblib.dump(best_model, args.model_dir / "best_churn_model.joblib")
    results.to_csv(args.report_dir / "model_metrics.csv")
    with (args.report_dir / "model_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "dataset": {"records": len(features), "test_records": len(x_test), "test_churn_rate": round(float(y_test.mean()), 4)},
                "selection_metric": "roc_auc",
                "selected_model": best_name,
                "models": results.to_dict(orient="index"),
            },
            file,
            indent=2,
        )

    best_predictions = pd.DataFrame(
        {
            "customerID": ids_test,
            "actual_churn": y_test,
            "churn_risk": scores_by_model[best_name],
        }
    ).sort_values("churn_risk", ascending=False)
    best_predictions["risk_rank"] = range(1, len(best_predictions) + 1)
    best_predictions["retention_target"] = best_predictions["risk_rank"] <= max(
        1, ceil(len(best_predictions) * args.top_fraction)
    )
    best_predictions.to_csv(args.report_dir / "test_set_risk_scores.csv", index=False)
    save_evaluation_charts(y_test, scores_by_model, args.report_dir)

    print("\nModel comparison (test set)")
    print(results.to_string())
    print(f"\nSelected model: {best_name}")
    print(f"Saved model: {args.model_dir / 'best_churn_model.joblib'}")
    print(f"Saved reports: {args.report_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
