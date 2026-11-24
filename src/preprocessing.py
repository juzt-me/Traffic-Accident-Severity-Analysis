"""Data cleaning, feature extraction, and leakage-safe preprocessing."""

from __future__ import annotations

from math import ceil
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "Severity"
DERIVED_FEATURES = ("Hour", "Weekday", "Month", "Is_Weekend")
EXCLUDED_COLUMNS = {
    "id",
    "source",
    "severity",
    "start_time",
    "end_time",
    "weather_timestamp",
    "description",
    "start_lat",
    "start_lng",
    "end_lat",
    "end_lng",
    "street",
    "city",
    "county",
    "state",
    "zipcode",
    "country",
    "timezone",
    "airport_code",
    "number_of_casualties",
}
MAX_CATEGORICAL_LEVELS = 200


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names, remove exact duplicates, and discard missing targets."""
    if df is None or df.empty:
        raise ValueError("The input dataset is empty.")
    result = df.copy()
    result.columns = [str(column).strip() for column in result.columns]
    if TARGET_COLUMN not in result.columns:
        raise ValueError(
            f"Required target column '{TARGET_COLUMN}' is missing. "
            f"Found columns: {', '.join(map(str, result.columns))}"
        )
    result = result.drop_duplicates()
    result = result.loc[result[TARGET_COLUMN].notna()].copy()
    if result.empty:
        raise ValueError(f"No rows remain with a valid '{TARGET_COLUMN}' value.")
    return result.reset_index(drop=True)


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, weekday, month, and weekend features from available date columns."""
    result = df.copy()
    timestamp = None
    for column in ("Start_Time", "StartTime", "Date"):
        if column in result.columns:
            timestamp = pd.to_datetime(result[column], errors="coerce", format="mixed")
            break

    if timestamp is not None:
        if "Hour" not in result.columns:
            result["Hour"] = timestamp.dt.hour
        if "Weekday" not in result.columns:
            result["Weekday"] = timestamp.dt.day_name()
        if "Month" not in result.columns:
            result["Month"] = timestamp.dt.month
        if "Is_Weekend" not in result.columns:
            result["Is_Weekend"] = timestamp.dt.dayofweek.isin([5, 6]).astype("Int64")
    return result


def get_feature_columns(
    df: pd.DataFrame, target_column: str = TARGET_COLUMN
) -> tuple[list[str], list[str], list[str]]:
    """Select usable numeric and low-cardinality categorical features."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is missing.")

    excluded = {name.casefold() for name in EXCLUDED_COLUMNS}
    numerical: list[str] = []
    categorical: list[str] = []
    skipped: list[str] = []

    for column in df.columns:
        if column == target_column or str(column).casefold() in excluded:
            continue
        series = df[column]
        if series.isna().all() or series.nunique(dropna=True) <= 1:
            skipped.append(str(column))
            continue
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            numerical.append(str(column))
        elif series.nunique(dropna=True) <= MAX_CATEGORICAL_LEVELS:
            categorical.append(str(column))
        else:
            skipped.append(str(column))

    return numerical, categorical, skipped


def make_preprocessor(
    numerical_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """Create imputation, scaling, and one-hot encoding transformations."""
    transformers: list[tuple[str, Any, list[str]]] = []
    if numerical_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numerical_features))
    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))
    if not transformers:
        raise ValueError(
            "No usable input features were found. Check the dataset columns and values."
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def prepare_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> dict[str, Any]:
    """Clean, engineer features, split once, and prepare training-only transforms.

    The returned X data remains in its raw DataFrame form. Each model pipeline
    fits its own cloned preprocessor on the training partition only.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    prepared = extract_time_features(clean_data(df))
    numerical, categorical, skipped = get_feature_columns(prepared)
    feature_columns = numerical + categorical
    if not feature_columns:
        raise ValueError("No usable model features remain after column selection.")

    X = prepared.loc[:, feature_columns].copy()
    y = prepared[TARGET_COLUMN].copy()
    class_counts = y.value_counts()
    test_rows = ceil(len(y) * test_size)
    can_stratify = (
        len(class_counts) > 1
        and class_counts.min() >= 2
        and test_rows >= len(class_counts)
        and len(y) - test_rows >= len(class_counts)
    )
    stratify = y if can_stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": make_preprocessor(numerical, categorical),
        "feature_columns": feature_columns,
        "numerical_features": numerical,
        "categorical_features": categorical,
        "skipped_features": skipped,
        "class_distribution": class_counts.to_dict(),
        "cleaned_data": prepared,
    }


def main() -> None:
    from src.data_loader import load_dataset

    data = load_dataset()
    prepared = prepare_data(data)
    print(
        f"Prepared {len(prepared['cleaned_data']):,} records; "
        f"{len(prepared['feature_columns'])} model features."
    )
    print(f"Training rows: {len(prepared['X_train']):,}")
    print(f"Test rows: {len(prepared['X_test']):,}")
    if prepared["skipped_features"]:
        print("Excluded columns:", ", ".join(prepared["skipped_features"]))


if __name__ == "__main__":
    main()
