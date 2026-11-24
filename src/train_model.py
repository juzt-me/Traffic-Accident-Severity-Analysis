"""Train, compare, evaluate, and save severity classification pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.data_loader import load_dataset
from src.evaluate_model import evaluate_model, save_confusion_matrix, save_feature_importance
from src.preprocessing import prepare_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: Any | None = None,
    random_state: int = 42,
) -> dict[str, Pipeline]:
    """Fit three models; each preprocessing step learns from training data only."""
    if X_train.empty or len(y_train) == 0:
        raise ValueError("Training data cannot be empty.")
    if preprocessor is None:
        prepared = prepare_data(
            pd.concat([X_train, y_train.rename("Severity")], axis=1),
            test_size=0.2,
            random_state=42,
        )
        preprocessor = prepared["preprocessor"]

    classifiers = {
        "Logistic Regression": LogisticRegression(
            max_iter=500, class_weight="balanced", random_state=random_state
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, class_weight="balanced", random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=18,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    fitted: dict[str, Pipeline] = {}
    for name, classifier in classifiers.items():
        pipeline = Pipeline(
            steps=[("preprocessor", clone(preprocessor)), ("classifier", classifier)]
        )
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
    return fitted


def compare_models(
    models: dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Evaluate every model on the same held-out test data."""
    rows = []
    details = {}
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        details[name] = metrics
        rows.append(
            {
                "Model": name,
                "Accuracy": metrics["accuracy"],
                "Precision_Macro": metrics["precision_macro"],
                "Recall_Macro": metrics["recall_macro"],
                "F1_Macro": metrics["f1_macro"],
            }
        )
    comparison = pd.DataFrame(rows).sort_values(
        ["F1_Macro", "Recall_Macro", "Accuracy"], ascending=False
    ).reset_index(drop=True)
    return comparison, details


def run_training(
    df: pd.DataFrame | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run the full training workflow and write model and evaluation artifacts."""
    if df is None:
        df = load_dataset()
    prepared = prepare_data(df, test_size=test_size, random_state=random_state)
    models = train_models(
        prepared["X_train"],
        prepared["y_train"],
        prepared["preprocessor"],
        random_state=random_state,
    )
    comparison, details = compare_models(models, prepared["X_test"], prepared["y_test"])
    best_name = str(comparison.iloc[0]["Model"])
    best_model = models[best_name]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    best_metrics = details[best_name]
    pd.DataFrame(
        best_metrics["confusion_matrix"],
        index=[str(label) for label in best_metrics["labels"]],
        columns=[str(label) for label in best_metrics["labels"]],
    ).to_csv(REPORTS_DIR / "confusion_matrix.csv")
    save_confusion_matrix(
        best_metrics["confusion_matrix"],
        best_metrics["labels"],
        PLOTS_DIR / "confusion_matrix.png",
    )
    importance = save_feature_importance(
        models["Random Forest"], REPORTS_DIR / "feature_importance.csv"
    )

    metadata = {
        "best_model": best_name,
        "selection_criterion": "Highest macro F1-score; then macro recall; then accuracy.",
        "feature_columns": prepared["feature_columns"],
        "numerical_features": prepared["numerical_features"],
        "categorical_features": prepared["categorical_features"],
        "skipped_features": prepared["skipped_features"],
        "class_distribution": {
            str(label): int(count) for label, count in prepared["class_distribution"].items()
        },
        "training_rows": int(len(prepared["X_train"])),
        "test_rows": int(len(prepared["X_test"])),
        "random_state": random_state,
        "test_size": test_size,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    report_lines = [
        "MODEL EVALUATION — HELD-OUT TEST DATA",
        "=" * 42,
        f"Selected model: {best_name}",
        "Selection criterion: highest macro F1-score (balanced across severity classes).",
        f"Training rows: {len(prepared['X_train']):,}",
        f"Test rows: {len(prepared['X_test']):,}",
        "",
        "MODEL COMPARISON",
        comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "",
        "CLASSIFICATION REPORT FOR SELECTED MODEL",
        best_metrics["classification_report"],
        "",
        "Limitations: evaluation is from one random held-out split. Results do not "
        "prove causation and are not suitable for real-world emergency decisions.",
    ]
    (REPORTS_DIR / "evaluation_report.txt").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSelected by macro F1: {best_name}")
    print(f"Saved trained pipeline: {MODEL_PATH}")
    if not importance.empty:
        print(f"Saved Random Forest feature importance ({len(importance)} features).")
    return {
        "best_model_name": best_name,
        "best_model": best_model,
        "comparison": comparison,
        "details": details,
        "prepared": prepared,
        "feature_importance": importance,
    }


if __name__ == "__main__":
    run_training()
