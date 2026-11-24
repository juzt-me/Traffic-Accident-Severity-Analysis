"""Evaluation metrics and output helpers for trained severity classifiers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Calculate classification metrics and a confusion matrix."""
    predictions = model.predict(X_test)
    labels = sorted(set(y_test) | set(predictions), key=str)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_macro": float(
            precision_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_test, predictions, labels=labels, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels),
        "labels": labels,
        "predictions": predictions,
    }


def save_confusion_matrix(
    matrix: Any, labels: list[Any], filepath: str | Path
) -> Path:
    """Save a readable confusion-matrix heatmap."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[str(label) for label in labels],
        yticklabels=[str(label) for label in labels],
        ax=ax,
    )
    ax.set(title="Confusion Matrix", xlabel="Predicted severity", ylabel="Actual severity")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_feature_importance(
    pipeline: Any, filepath: str | Path, top_n: int = 25
) -> pd.DataFrame:
    """Save tree-model feature importance when the selected classifier supports it."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    classifier = pipeline.named_steps["classifier"]
    importances = getattr(classifier, "feature_importances_", None)
    if importances is None:
        return pd.DataFrame(columns=["feature", "importance"])

    preprocessor = pipeline.named_steps["preprocessor"]
    names = preprocessor.get_feature_names_out()
    result = pd.DataFrame({"feature": names, "importance": importances})
    result = result.sort_values("importance", ascending=False).head(top_n)
    result.to_csv(path, index=False)

    plot_path = path.parent.parent / "plots" / "feature_importance.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, max(5, len(result) * 0.32)))
    display = result.sort_values("importance")
    ax.barh(display["feature"], display["importance"], color="#3674a5")
    ax.set(title="Random Forest Feature Importance", xlabel="Importance", ylabel="Feature")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return result
