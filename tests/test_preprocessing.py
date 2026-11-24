"""Tests for real-data loading, preprocessing, and model training."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_loader import inspect_dataset, load_dataset, validate_dataset
from src.preprocessing import (
    clean_data,
    extract_time_features,
    make_preprocessor,
    prepare_data,
)


def sample_accidents(rows: int = 120) -> pd.DataFrame:
    """Create a small test fixture with the shape of the Kaggle CSV."""
    rng = np.random.default_rng(42)
    times = pd.date_range("2023-01-01", periods=rows, freq="h")
    return pd.DataFrame(
        {
            "ID": [f"test-{index}" for index in range(rows)],
            "Severity": np.resize([1, 2, 3, 4], rows),
            "Start_Time": times.astype(str),
            "Weather_Condition": rng.choice(["Clear", "Rain", "Cloudy"], rows),
            "Temperature(F)": rng.normal(60, 15, rows),
            "Visibility(mi)": rng.uniform(0.5, 10, rows),
            "Junction": rng.choice([True, False], rows),
        }
    )


def test_load_dataset_uniform_subset_and_inspect(tmp_path):
    source = tmp_path / "accidents.csv"
    sample_accidents(37).to_csv(source, index=False)
    loaded = load_dataset(source, max_rows=12, chunk_size=10, random_state=10)
    summary = inspect_dataset(loaded)
    assert len(loaded) == 12
    assert summary["shape"][1] == 7
    assert summary["severity_counts"]
    assert "Start_Time" in summary["columns"]


def test_missing_dataset_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Kaggle dataset"):
        load_dataset(tmp_path / "missing.csv")


def test_invalid_or_missing_target_is_rejected():
    with pytest.raises(ValueError, match="Severity"):
        validate_dataset(pd.DataFrame({"weather": ["rain"]}))
    with pytest.raises(ValueError, match="Severity"):
        clean_data(pd.DataFrame({"weather": ["rain"]}))


def test_cleaning_drops_duplicate_and_missing_target_rows():
    df = sample_accidents(8)
    df.loc[0, "Severity"] = np.nan
    cleaned = clean_data(pd.concat([df, df.iloc[[1]]], ignore_index=True))
    assert len(cleaned) == 7
    assert cleaned["Severity"].notna().all()


def test_time_features_are_extracted_from_start_time():
    frame = pd.DataFrame({"Start_Time": ["2023-03-04 23:15:00", "2023-03-06 08:00:00"]})
    result = extract_time_features(frame)
    assert result.loc[0, "Hour"] == 23
    assert result.loc[0, "Weekday"] == "Saturday"
    assert result.loc[0, "Month"] == 3
    assert result.loc[0, "Is_Weekend"] == 1
    assert result.loc[1, "Is_Weekend"] == 0


def test_preprocessor_imputes_missing_values_and_encodes_categories():
    frame = pd.DataFrame(
        {"temperature": [20.0, np.nan, 30.0], "weather": ["Clear", None, "Rain"]}
    )
    transformer = make_preprocessor(["temperature"], ["weather"])
    transformed = transformer.fit_transform(frame)
    assert transformed.shape[0] == 3
    dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    assert np.isfinite(np.asarray(dense)).all()


def test_prepare_data_splits_stratified_and_excludes_identifiers():
    result = prepare_data(sample_accidents(120), random_state=7)
    assert len(result["X_train"]) == 96
    assert len(result["X_test"]) == 24
    assert "ID" not in result["feature_columns"]
    assert "Severity" not in result["feature_columns"]
    assert set(result["y_train"].unique()) == {1, 2, 3, 4}
    assert result["preprocessor"].transformers


def test_all_three_models_train_and_predict_on_same_test_split():
    from src.train_model import compare_models, train_models

    prepared = prepare_data(sample_accidents(120), random_state=17)
    models = train_models(
        prepared["X_train"], prepared["y_train"], prepared["preprocessor"]
    )
    comparison, details = compare_models(
        models, prepared["X_test"], prepared["y_test"]
    )
    assert set(models) == {"Logistic Regression", "Decision Tree", "Random Forest"}
    assert len(comparison) == 3
    assert comparison["F1_Macro"].between(0, 1).all()
    assert all(len(details[name]["predictions"]) == len(prepared["y_test"]) for name in models)


def test_training_workflow_saves_a_predictable_pipeline(tmp_path, monkeypatch):
    import joblib
    import src.train_model as training

    monkeypatch.setattr(training, "MODEL_PATH", tmp_path / "models" / "best_model.joblib")
    monkeypatch.setattr(training, "METADATA_PATH", tmp_path / "models" / "metadata.json")
    monkeypatch.setattr(training, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(training, "PLOTS_DIR", tmp_path / "plots")
    result = training.run_training(sample_accidents(80), random_state=9)
    saved = joblib.load(training.MODEL_PATH)
    probe = result["prepared"]["X_test"].head(1)

    assert training.MODEL_PATH.is_file()
    assert (tmp_path / "reports" / "model_comparison.csv").is_file()
    assert (tmp_path / "reports" / "evaluation_report.txt").is_file()
    assert len(saved.predict(probe)) == 1
