"""Exploratory analysis plots and a data-derived text report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.preprocessing import extract_time_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
SEVERITY = "Severity"
ROAD_FEATURES = (
    "Road_Type",
    "Junction",
    "Traffic_Signal",
    "Crossing",
    "Stop",
    "Railway",
    "Roundabout",
    "Bump",
    "Traffic_Calming",
    "Give_Way",
    "No_Exit",
    "Turning_Loop",
    "Amenity",
    "Station",
)


def _finish(fig: plt.Figure, filename: str, output_dir: str | Path) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _placeholder(ax: Any, title: str, reason: str) -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, reason, ha="center", va="center", wrap=True)
    ax.set_axis_off()


def _countplot(
    df: pd.DataFrame,
    feature: str | None,
    title: str,
    filename: str,
    output_dir: str | Path,
    top_n: int = 12,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    if feature is None or feature not in df.columns:
        _placeholder(ax, title, "This feature is not available in the supplied CSV.")
    else:
        data = df[[feature, SEVERITY]].dropna().copy()
        top_values = data[feature].value_counts().head(top_n).index
        data[feature] = data[feature].where(data[feature].isin(top_values), "Other")
        sns.countplot(
            data=data,
            x=feature,
            hue=SEVERITY,
            order=data[feature].value_counts().index,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel(feature.replace("_", " "))
        ax.set_ylabel("Accident records")
        ax.tick_params(axis="x", rotation=35)
        ax.legend(title="Severity", bbox_to_anchor=(1.02, 1), loc="upper left")
    return _finish(fig, filename, output_dir)


def _plot_severity_distribution(df: pd.DataFrame, output_dir: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df[SEVERITY].value_counts(dropna=False).sort_index(key=lambda values: values.astype(str))
    sns.barplot(x=counts.index.astype(str), y=counts.values, color="#4c78a8", ax=ax)
    ax.set(title="Distribution of Accident Severity", xlabel="Recorded severity", ylabel="Records")
    return _finish(fig, "severity_distribution.png", output_dir)


def _plot_hour(df: pd.DataFrame, output_dir: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    if "Hour" not in df:
        _placeholder(ax, "Accidents by Hour of Day", "Start_Time is not available.")
    else:
        counts = df["Hour"].dropna().astype(int).value_counts().reindex(range(24), fill_value=0)
        sns.lineplot(x=counts.index, y=counts.values, marker="o", ax=ax)
        ax.set(title="Accidents by Hour of Day", xlabel="Hour (0–23)", ylabel="Accident records")
        ax.set_xticks(range(24))
    return _finish(fig, "accidents_by_hour.png", output_dir)


def _plot_weekday(df: pd.DataFrame, output_dir: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    if "Weekday" not in df:
        _placeholder(ax, "Accidents by Day of Week", "Start_Time is not available.")
    else:
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        counts = df["Weekday"].value_counts().reindex(order, fill_value=0)
        sns.barplot(x=counts.index, y=counts.values, color="#4c78a8", ax=ax)
        ax.set(title="Accidents by Day of Week", xlabel="Day", ylabel="Accident records")
        ax.tick_params(axis="x", rotation=25)
    return _finish(fig, "accidents_by_weekday.png", output_dir)


def _plot_speed(df: pd.DataFrame, output_dir: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    feature = next(
        (name for name in ("Speed_Limit", "Speed_Limit(mph)", "speed_limit") if name in df),
        None,
    )
    if feature is None:
        _placeholder(ax, "Speed Limit by Accident Severity", "No speed-limit column is available.")
    else:
        sns.boxplot(data=df, x=SEVERITY, y=feature, ax=ax)
        ax.set(title="Speed Limit by Accident Severity", xlabel="Recorded severity", ylabel=feature)
    return _finish(fig, "speed_limit_by_severity.png", output_dir)


def _plot_correlation(df: pd.DataFrame, output_dir: str | Path) -> Path:
    numeric = df.select_dtypes(include="number").copy()
    if SEVERITY in df and pd.api.types.is_numeric_dtype(df[SEVERITY]):
        numeric[SEVERITY] = df[SEVERITY]
    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1].iloc[:, :15]
    fig, ax = plt.subplots(figsize=(11, 8))
    if numeric.shape[1] < 2:
        _placeholder(ax, "Correlation Heatmap", "Fewer than two varying numeric columns are available.")
    else:
        sns.heatmap(numeric.corr(), cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation Heatmap of Numeric Features")
    return _finish(fig, "correlation_heatmap.png", output_dir)


def _plot_month(df: pd.DataFrame, output_dir: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    if "Month" not in df:
        _placeholder(ax, "Accident Severity by Month", "A usable start date is not available.")
    else:
        month_data = df[["Month", SEVERITY]].dropna().copy()
        sns.countplot(data=month_data, x="Month", hue=SEVERITY, order=range(1, 13), ax=ax)
        ax.set(title="Accident Severity by Month", xlabel="Month (1 = January)", ylabel="Records")
        ax.legend(title="Severity", bbox_to_anchor=(1.02, 1), loc="upper left")
    return _finish(fig, "severity_by_month.png", output_dir)


def _plot_environment(df: pd.DataFrame, output_dir: str | Path) -> Path:
    features = [name for name in ("Temperature(F)", "Visibility(mi)") if name in df]
    fig, axes = plt.subplots(1, max(1, len(features)), figsize=(7 * max(1, len(features)), 5))
    axes_list = [axes] if not isinstance(axes, (list, tuple, np.ndarray)) else list(axes)
    if not features:
        _placeholder(axes_list[0], "Temperature and Visibility by Severity", "No temperature or visibility column is available.")
    else:
        for ax, feature in zip(axes_list, features):
            sns.boxplot(data=df, x=SEVERITY, y=feature, showfliers=False, ax=ax)
            ax.set_title(f"{feature} by Severity")
            ax.set_xlabel("Recorded severity")
    return _finish(fig, "temperature_visibility_by_severity.png", output_dir)


def generate_all_charts(
    df: pd.DataFrame, output_dir: str | Path = PLOTS_DIR
) -> dict[str, Path]:
    """Generate charts from the provided real data; unavailable features are labeled."""
    if df is None or df.empty:
        raise ValueError("The dataset is empty or None.")
    if SEVERITY not in df:
        raise ValueError(f"Required target column '{SEVERITY}' is missing.")
    data = extract_time_features(df)

    road_feature = next((name for name in ROAD_FEATURES if name in data), None)
    weather_feature = next(
        (name for name in ("Weather_Condition", "Weather_Conditions") if name in data),
        None,
    )
    lighting_feature = next(
        (
            name
            for name in ("Sunrise_Sunset", "Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight")
            if name in data
        ),
        None,
    )
    charts = {
        "severity_distribution": _plot_severity_distribution(data, output_dir),
        "severity_by_road_feature": _countplot(
            data, road_feature, "Accident Severity by Road Feature",
            "severity_by_road_feature.png", output_dir,
        ),
        "severity_by_weather": _countplot(
            data, weather_feature, "Accident Severity by Weather Condition",
            "severity_by_weather.png", output_dir,
        ),
        "severity_by_lighting": _countplot(
            data, lighting_feature, "Accident Severity by Lighting Condition",
            "severity_by_lighting.png", output_dir,
        ),
        "accidents_by_hour": _plot_hour(data, output_dir),
        "accidents_by_weekday": _plot_weekday(data, output_dir),
        "correlation_heatmap": _plot_correlation(data, output_dir),
        "severity_by_month": _plot_month(data, output_dir),
        "temperature_visibility": _plot_environment(data, output_dir),
    }
    if any(name in data for name in ("Speed_Limit", "Speed_Limit(mph)", "speed_limit")):
        charts["speed_limit_by_severity"] = _plot_speed(data, output_dir)
    return charts


def generate_eda_report(
    df: pd.DataFrame,
    output_dir: str | Path = REPORTS_DIR,
) -> str:
    """Write a concise report whose statistics come only from the supplied data."""
    if df is None or df.empty:
        raise ValueError("The dataset is empty or None.")
    if SEVERITY not in df:
        raise ValueError(f"Required target column '{SEVERITY}' is missing.")
    data = extract_time_features(df)
    distribution = data[SEVERITY].value_counts(dropna=False)
    lines = [
        "US ACCIDENTS DATASET — EXPLORATORY DATA ANALYSIS",
        "=" * 58,
        f"Records analyzed: {len(data):,}",
        f"Total source records: {int(data.attrs.get('source_rows', len(data))):,}",
        f"Uniform sample used: {'yes' if data.attrs.get('sampled', False) else 'no'}",
        f"Columns: {len(data.columns):,}",
        f"Duplicate rows: {int(data.duplicated().sum()):,}",
        f"Columns with missing values: {int((data.isna().sum() > 0).sum()):,}",
        "",
        "SEVERITY DISTRIBUTION",
        "-" * 24,
    ]
    for label, count in distribution.items():
        lines.append(f"{label}: {count:,} ({count / len(data):.1%})")
    nonzero_counts = distribution[distribution > 0]
    if len(nonzero_counts) > 1:
        imbalance_ratio = nonzero_counts.max() / nonzero_counts.min()
        lines.append(
            f"Most-to-least common class ratio: {imbalance_ratio:.2f}:1 "
            "(a larger ratio indicates stronger class imbalance)."
        )
    lines.extend(["", "AVAILABLE ANALYSIS FIELDS", "-" * 25])
    requested = {
        "Road conditions": list(ROAD_FEATURES),
        "Weather": ["Weather_Condition", "Weather_Conditions"],
        "Lighting": ["Sunrise_Sunset", "Civil_Twilight", "Nautical_Twilight"],
        "Speed limit": ["Speed_Limit", "Speed_Limit(mph)", "speed_limit"],
        "Temperature": ["Temperature(F)"],
        "Visibility": ["Visibility(mi)"],
        "Start timestamp": ["Start_Time", "StartTime", "Date"],
    }
    for label, candidates in requested.items():
        found = [column for column in candidates if column in data]
        lines.append(f"{label}: {', '.join(found) if found else 'not available in this CSV'}")
    if "Hour" in data and data["Hour"].notna().any():
        hour_counts = data["Hour"].value_counts()
        lines.extend(["", f"Hour with the most records in this sample: {int(hour_counts.idxmax()):02d}:00 ({int(hour_counts.max()):,} records)"])
    if "Weather_Condition" in data and data["Weather_Condition"].notna().any():
        top_weather = data["Weather_Condition"].value_counts().head(5)
        lines.extend(["", "Most frequent recorded weather conditions:"])
        lines.extend(f"- {label}: {count:,}" for label, count in top_weather.items())
    if "Temperature(F)" in data:
        lines.extend(["", "Temperature summary (Fahrenheit):"])
        lines.append(data["Temperature(F)"].describe().round(2).to_string())
    if "Visibility(mi)" in data:
        lines.extend(["", "Visibility summary (miles):"])
        lines.append(data["Visibility(mi)"].describe().round(2).to_string())
    lines.extend(
        [
            "",
            "INTERPRETATION",
            "-" * 14,
            "These are descriptive statistics for the loaded dataset/sample. "
            "Observed associations do not establish causation.",
        ]
    )
    report = "\n".join(lines)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "eda_report.txt").write_text(report, encoding="utf-8")
    return report


def run_eda(df: pd.DataFrame | None = None) -> dict[str, Any]:
    """Load the real dataset if needed, save charts, and write the EDA report."""
    if df is None:
        from src.data_loader import load_dataset

        df = load_dataset()
    charts = generate_all_charts(df)
    report = generate_eda_report(df)
    print(f"Generated {len(charts)} charts in {PLOTS_DIR}")
    print(f"Saved EDA summary to {REPORTS_DIR / 'eda_report.txt'}")
    return {"charts": charts, "report": report}


if __name__ == "__main__":
    run_eda()
