"""TRAFFIC INSIGHT: interactive US accident analysis and prediction platform."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import DEFAULT_SAMPLE_SIZE, load_dataset
from src.preprocessing import extract_time_features

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PLOTS_DIR = ROOT / "outputs" / "plots"
REPORTS_DIR = ROOT / "outputs" / "reports"
MODEL_PATH = ROOT / "models" / "best_model.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
REPORT_PATH = REPORTS_DIR / "evaluation_report.txt"
EDA_REPORT_PATH = REPORTS_DIR / "eda_report.txt"
CONFUSION_PATH = REPORTS_DIR / "confusion_matrix.csv"
IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"
SOURCE_URL = "https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents"
PAGES = [
    "Home",
    "Explore Data",
    "Analytics",
    "Severity Prediction",
    "Model Performance",
    "Reports",
]
PAGE_ICONS = {
    "Home": "⌂",
    "Explore Data": "▦",
    "Analytics": "◷",
    "Severity Prediction": "◇",
    "Model Performance": "▥",
    "Reports": "⇩",
}
SEVERITY_COLORS = {
    "1": "#16A34A",
    "2": "#2563EB",
    "3": "#F59E0B",
    "4": "#DC2626",
}
PLOT_TEMPLATE = "plotly_white"
ROAD_FEATURES = [
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
    "Amenity",
    "Station",
]
LIGHTING_FEATURES = [
    "Sunrise_Sunset",
    "Civil_Twilight",
    "Nautical_Twilight",
    "Astronomical_Twilight",
]
MONTH_LABELS = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}

st.set_page_config(
    page_title="TRAFFIC INSIGHT | Accident Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
      --navy: #0F172A;
      --navy-soft: #1E293B;
      --blue: #2563EB;
      --blue-dark: #1D4ED8;
      --ink: #0F172A;
      --muted: #64748B;
      --line: #E2E8F0;
      --surface: #FFFFFF;
      --page: #F8FAFC;
    }
    .stApp { background: var(--page); color: var(--ink); }
    [data-testid="stAppViewContainer"] .main .block-container {
      max-width: 1440px; padding: 1.15rem 2rem 2.5rem;
    }
    [data-testid="stSidebar"] { display: none; }
    header[data-testid="stHeader"], div[data-testid="stToolbar"] { display: none; }
    h1 { color: var(--ink); letter-spacing: -.04em; font-weight: 760; }
    h2, h3 { color: var(--ink); letter-spacing: -.025em; }
    p, li, label, .stCaption { color: var(--muted); }
    [data-testid="stCaptionContainer"] { color: #64748B !important; }
    [data-testid="stMetric"] {
      background: var(--surface); border: 1px solid var(--line);
      border-radius: 14px; padding: 1rem 1.15rem;
      box-shadow: 0 4px 14px rgba(15, 23, 42, .045);
    }
    [data-testid="stMetricLabel"] { color: #64748B !important; }
    [data-testid="stMetricValue"] { color: #17283F !important; font-weight: 720; }
    [data-testid="stMetricDelta"] { color: #3978C5 !important; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
      background: var(--surface); border: 1px solid var(--line);
      border-radius: 14px;
    }
    .st-key-top-navigation {
      position: sticky; top: 0; z-index: 100; background: rgba(248,250,252,.96);
      border-bottom: 1px solid var(--line); padding: .45rem .8rem .35rem;
      margin: -.5rem -.8rem 1.8rem; backdrop-filter: blur(12px);
    }
    .ti-brand {
      font-size: 1.12rem; font-weight: 820; letter-spacing: .06em;
      color: var(--navy); padding: .55rem 0 .1rem; white-space: nowrap;
    }
    .ti-brand-mark {
      display: inline-grid; place-items: center; width: 30px; height: 30px;
      margin-right: .38rem; border-radius: 9px; color: white; background: var(--blue);
      font-size: 1rem; vertical-align: middle;
    }
    .st-key-nav-mobile { display: none; }
    .st-key-nav-desktop [data-testid="stRadio"] > div {
      display: flex; justify-content: flex-end; gap: .05rem; flex-wrap: nowrap;
      overflow-x: auto; scrollbar-width: none;
    }
    .st-key-nav-desktop [data-testid="stRadio"] input[type="radio"] { display: none !important; }
    .st-key-nav-desktop [data-testid="stRadioOption"] > div > div:first-child {
      display: none !important;
    }
    .st-key-nav-desktop [data-testid="stRadio"] label {
      border: 1px solid transparent; border-radius: 9px; padding: .34rem .42rem;
      color: #475569; font-size: .8rem; font-weight: 650; white-space: nowrap;
      transition: all .16s ease;
    }
    .st-key-nav-desktop [data-testid="stRadio"] label:hover {
      background: #EFF6FF; color: var(--blue);
    }
        .st-key-nav-desktop [data-testid="stRadio"] label:has(input:checked) {
          color: var(--blue); background: #EFF6FF; border-color: #BFDBFE;
        }
    .ti-eyebrow {
      color: var(--blue); text-transform: uppercase; letter-spacing: .12em;
      font-size: .72rem; font-weight: 750; margin-bottom: .35rem;
    }
    .ti-hero {
      background: linear-gradient(115deg, #FFFFFF 8%, #EFF6FF 100%);
      border: 1px solid #DBEAFE; border-radius: 20px;
      padding: 2rem 2.15rem; margin-bottom: 1.3rem;
      box-shadow: 0 12px 32px rgba(37,99,235,.06);
    }
    .ti-hero h1 { margin: 0 0 .55rem; font-size: clamp(2.2rem, 4vw, 3.45rem); line-height: 1.06; }
    .ti-hero p { margin: .35rem 0 1.1rem; max-width: 610px; line-height: 1.7; font-size: 1rem; }
    .ti-tagline { color: #475569; font-size: .82rem; font-weight: 700; letter-spacing: .025em; }
    .ti-hero-visual {
      background: linear-gradient(145deg,#EFF6FF,#DBEAFE); border: 1px solid #BFDBFE;
      border-radius: 18px; min-height: 265px; display: grid; place-items: center;
      overflow: hidden; box-shadow: inset 0 1px #fff;
    }
    .ti-feature {
      background: #fff; border: 1px solid var(--line); border-radius: 14px;
      padding: 1rem 1.05rem; min-height: 155px;
      box-shadow: 0 4px 14px rgba(15,23,42,.035); transition: transform .16s ease, box-shadow .16s ease;
    }
    .ti-feature:hover { transform: translateY(-2px); box-shadow: 0 9px 22px rgba(15,23,42,.08); }
    .ti-feature-icon { font-size: 1.25rem; margin-bottom: .55rem; }
    .ti-feature-title { color: var(--navy); font-size: .98rem; font-weight: 720; margin-bottom: .28rem; }
    .ti-feature-copy { color: var(--muted); font-size: .82rem; min-height: 2.2rem; line-height: 1.45; }
    .ti-download-card {
      border: 1px solid var(--line); background: #fff; border-radius: 12px;
      padding: .9rem 1rem; margin: .25rem 0 .65rem;
    }
    .ti-download-title { color: var(--navy); font-size: .94rem; font-weight: 720; }
    .ti-download-meta { color: var(--muted); font-size: .78rem; margin-top: .2rem; }
    .ti-section-title { color: var(--navy); font-weight: 740; font-size: 1.18rem; margin: .25rem 0 .2rem; }
    .ti-section-copy { color: var(--muted); font-size: .86rem; margin-bottom: .8rem; }
    .ti-card {
      background: #FFFFFF; border: 1px solid var(--line);
      border-radius: 14px; padding: 1rem 1.15rem;
      box-shadow: 0 2px 8px rgba(27, 43, 71, .045);
      min-height: 112px;
    }
    .ti-card-label { color: #69798F; font-size: .82rem; font-weight: 600; }
    .ti-card-value { color: #182A42; font-size: 1.45rem; font-weight: 760; margin-top: .35rem; }
    .ti-card-detail { color: #78879A; font-size: .78rem; margin-top: .24rem; }
    .ti-note {
      background: #EEF4FC; border-left: 4px solid var(--blue);
      border-radius: 0 10px 10px 0; padding: .85rem 1rem; color: #31445E;
    }
    .ti-warning {
      background: #FFF7E9; border: 1px solid #F2DFB6; border-radius: 10px;
      padding: .8rem 1rem; color: #654B20;
    }
    div.stButton > button[kind="primary"] {
      background: var(--blue); border: 1px solid var(--blue); color: #FFFFFF;
      border-radius: 9px; font-weight: 650;
    }
    div.stButton > button[kind="primary"]:hover {
      background: var(--blue-dark); border-color: var(--blue-dark); color: #FFFFFF;
    }
    [data-testid="stDownloadButton"] button {
      border-radius: 9px; border-color: #C9D7E8; color: #24496E;
      background: #FFFFFF;
    }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 10px; }
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
      border-radius: 8px;
    }
    .st-key-prediction-result {
      position: sticky; top: 5.2rem; align-self: flex-start;
    }
    @media (max-width: 760px) {
      [data-testid="stAppViewContainer"] .main .block-container { padding: .8rem .85rem 1.5rem; }
      .ti-hero { padding: 1.25rem; }
      .st-key-nav-desktop { display: none !important; }
      .st-key-nav-mobile { display: block !important; }
      .st-key-top-navigation { padding: .35rem .4rem; margin: -.35rem -.4rem 1.2rem; }
      .ti-hero-visual { min-height: 210px; }
      .st-key-prediction-result { position: static; }
    }
    @media (min-width: 761px) {
      .st-key-nav-mobile { display: none !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    """Load the cached 100,000-row sample and engineer its date features once."""
    data = load_dataset(max_rows=DEFAULT_SAMPLE_SIZE)
    return extract_time_features(data)


@st.cache_resource(show_spinner=False)
def get_model() -> Any | None:
    """Load the saved complete preprocessing/classification pipeline once."""
    if not MODEL_PATH.is_file():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def read_comparison() -> pd.DataFrame:
    if not COMPARISON_PATH.is_file():
        return pd.DataFrame()
    return pd.read_csv(COMPARISON_PATH)


@st.cache_data(show_spinner=False)
def read_text_file(path_string: str) -> str:
    path = Path(path_string)
    return path.read_text(encoding="utf-8") if path.is_file() else ""


@st.cache_data(show_spinner=False)
def read_csv_file(path_string: str, index_col: int | None = 0) -> pd.DataFrame:
    path = Path(path_string)
    return pd.read_csv(path, index_col=index_col) if path.is_file() else pd.DataFrame()


def _label(value: Any) -> str:
    return str(value).replace("_", " ").strip()


def _severity_palette(values: list[Any]) -> dict[str, str]:
    return {str(value): SEVERITY_COLORS.get(str(value), "#718198") for value in values}


def _chart_layout(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        template=PLOT_TEMPLATE,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=height,
        margin=dict(l=18, r=18, t=55, b=24),
        font=dict(family="Arial, sans-serif", size=12, color="#475569"),
        title=dict(font=dict(size=16, color="#1D2A3A")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="#FFFFFF", font_color="#1D2A3A"),
    )
    return fig


def _section(title: str, subtitle: str | None = None) -> None:
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)


def _plotly(fig: go.Figure, key: str, height: int | None = None) -> None:
    if height is not None:
        fig.update_layout(height=height)
    st.plotly_chart(
        fig,
        width="stretch",
        theme=None,
        config={"displayModeBar": False, "responsive": True},
        key=f"{key}_{id(fig)}",
    )


def _metric_card(label: str, value: str, detail: str = "", icon: str = "◈") -> None:
    st.markdown(
        f"<div class='ti-card'><div class='ti-card-label'><span style='color:#2563EB;"
        f"font-size:1.05rem;margin-right:.35rem'>{icon}</span>{label}</div>"
        f"<div class='ti-card-value'>{value}</div>"
        f"<div class='ti-card-detail'>{detail}</div></div>",
        unsafe_allow_html=True,
    )


def _available(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((column for column in candidates if column in df.columns), None)


def _sync_desktop_navigation() -> None:
    st.session_state["traffic_insight_mobile_nav"] = st.session_state[
        "traffic_insight_page"
    ]


def _sync_mobile_navigation() -> None:
    st.session_state["traffic_insight_page"] = st.session_state[
        "traffic_insight_mobile_nav"
    ]


def _navigation(df: pd.DataFrame | None, data_error: Exception | None) -> str:
    """Render accessible desktop and compact mobile navigation for the app pages."""
    st.session_state.setdefault("traffic_insight_page", "Home")
    st.session_state.setdefault("traffic_insight_mobile_nav", "Home")
    if st.session_state["traffic_insight_page"] not in PAGES:
        st.session_state["traffic_insight_page"] = "Home"
    if st.session_state["traffic_insight_mobile_nav"] not in PAGES:
        st.session_state["traffic_insight_mobile_nav"] = st.session_state[
            "traffic_insight_page"
        ]
    with st.container():
        logo, links = st.columns([1.25, 4.2], vertical_alignment="center")
        with logo:
            st.markdown(
                "<div class='ti-brand'><span class='ti-brand-mark'>◈</span>"
                "TRAFFIC INSIGHT</div>",
                unsafe_allow_html=True,
            )
        with links:
            with st.container(key="nav-desktop"):
                st.radio(
                    "Main navigation",
                    PAGES,
                    key="traffic_insight_page",
                    horizontal=True,
                    label_visibility="collapsed",
                    on_change=_sync_desktop_navigation,
                )
        with st.container(key="nav-mobile"):
            st.selectbox(
                "Navigate",
                PAGES,
                key="traffic_insight_mobile_nav",
                format_func=lambda page: f"{PAGE_ICONS[page]}  {page}",
                on_change=_sync_mobile_navigation,
            )
    if df is not None:
        st.caption(
            f"{len(df):,} sampled records · {df.shape[1]:,} source columns · "
            f"Dataset: Kaggle US Accidents"
        )
    elif data_error:
        st.caption(f"Dataset not available: {data_error}")
    return st.session_state["traffic_insight_page"]


def _go_to(page: str) -> None:
    st.session_state["traffic_insight_page"] = page
    st.session_state["traffic_insight_mobile_nav"] = page


def _top_header(eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        f"<div class='ti-eyebrow'>{eyebrow}</div><h1>{title}</h1>"
        f"<p style='color:#65748A;margin-top:-.55rem;margin-bottom:1.35rem'>"
        f"{description}</p>",
        unsafe_allow_html=True,
    )


def _filtered_data(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    """Render compact filters for fields that actually exist in this dataset."""
    filtered = df
    controls = st.columns(5)
    if "Severity" in filtered:
        levels = sorted(filtered["Severity"].dropna().unique(), key=str)
        chosen = controls[0].multiselect(
            "Severity class", levels, default=levels, key=f"{key_prefix}_severity"
        )
        filtered = filtered[filtered["Severity"].isin(chosen)]
    if "Month" in filtered:
        months = sorted(pd.to_numeric(filtered["Month"], errors="coerce").dropna().unique())
        chosen_months = controls[1].multiselect(
            "Month",
            months,
            default=months,
            format_func=lambda value: MONTH_LABELS.get(int(value), str(value)),
            key=f"{key_prefix}_month",
        )
        filtered = filtered[filtered["Month"].isin(chosen_months)]
    weather_col = _available(filtered, ["Weather_Condition", "Weather_Conditions"])
    if weather_col:
        weather = filtered[weather_col].dropna().value_counts().head(18).index.tolist()
        selected_weather = controls[2].multiselect(
            "Weather",
            weather,
            default=weather,
            key=f"{key_prefix}_weather",
        )
        filtered = filtered[filtered[weather_col].isin(selected_weather)]
    if "Hour" in filtered:
        hours = controls[3].slider(
            "Hour of day",
            0,
            23,
            (0, 23),
            key=f"{key_prefix}_hour",
        )
        filtered = filtered[filtered["Hour"].between(hours[0], hours[1])]
    state_col = _available(filtered, ["State"])
    if state_col:
        states = sorted(filtered[state_col].dropna().astype(str).unique().tolist())
        chosen_states = controls[4].multiselect(
            "State", states, default=states, key=f"{key_prefix}_state"
        )
        filtered = filtered[filtered[state_col].astype(str).isin(chosen_states)]
    st.caption(f"Showing **{len(filtered):,}** of {len(df):,} sampled records.")
    return filtered


def _severity_bar(df: pd.DataFrame, title: str) -> go.Figure:
    counts = df["Severity"].value_counts().sort_index()
    fig = px.bar(
        x=counts.index.astype(str),
        y=counts.values,
        color=counts.index.astype(str),
        color_discrete_map=_severity_palette(list(counts.index)),
        labels={"x": "Recorded severity class", "y": "Accident records", "color": "Severity"},
        title=title,
        text=counts.values,
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
    return _chart_layout(fig, 340)


def _hour_chart(df: pd.DataFrame, title: str = "Accidents by hour") -> go.Figure:
    if "Hour" not in df or df["Hour"].dropna().empty:
        return _chart_layout(go.Figure().add_annotation(text="Start time unavailable", showarrow=False))
    summary = (
        df.dropna(subset=["Hour"])
        .assign(Hour=lambda frame: pd.to_numeric(frame["Hour"], errors="coerce"))
        .groupby("Hour", observed=True)
        .size()
        .reindex(range(24), fill_value=0)
        .rename("Records")
        .reset_index()
    )
    fig = px.area(
        summary,
        x="Hour",
        y="Records",
        title=title,
        labels={"Hour": "Hour of day (0–23)", "Records": "Accident records"},
        markers=True,
    )
    fig.update_traces(line_color="#3978C5", fillcolor="rgba(57,120,197,.16)")
    fig.update_xaxes(dtick=2)
    return _chart_layout(fig, 340)


def _weather_chart(df: pd.DataFrame, title: str = "Accidents by weather") -> go.Figure:
    weather = _available(df, ["Weather_Condition", "Weather_Conditions"])
    if not weather:
        return _chart_layout(go.Figure().add_annotation(text="Weather field unavailable", showarrow=False))
    summary = df[weather].dropna().value_counts().head(10).sort_values().reset_index()
    summary.columns = [weather, "Records"]
    fig = px.bar(
        summary,
        x="Records",
        y=weather,
        orientation="h",
        title=title,
        labels={"Records": "Accident records", weather: "Weather condition"},
        color_discrete_sequence=["#3978C5"],
    )
    return _chart_layout(fig, 365)


def _severity_feature_chart(
    df: pd.DataFrame, feature: str | None, title: str, normalize: bool = False
) -> go.Figure:
    if not feature or feature not in df:
        return _chart_layout(go.Figure().add_annotation(text="This field is unavailable", showarrow=False))
    summary = df[[feature, "Severity"]].dropna().copy()
    if summary.empty:
        return _chart_layout(go.Figure().add_annotation(text="No matching observations", showarrow=False))
    summary[feature] = summary[feature].astype(str)
    if summary[feature].nunique() > 14:
        top = summary[feature].value_counts().head(13).index
        summary[feature] = summary[feature].where(summary[feature].isin(top), "Other")
    counts = summary.groupby([feature, "Severity"], observed=True).size().rename("Records")
    table = counts.reset_index()
    if normalize:
        table["Percent"] = table["Records"] / table.groupby(feature)["Records"].transform("sum") * 100
        y_col, y_label = "Percent", "Percent within category"
    else:
        y_col, y_label = "Records", "Accident records"
    fig = px.bar(
        table,
        x=feature,
        y=y_col,
        color="Severity",
        barmode="stack",
        title=title,
        labels={feature: _label(feature), y_col: y_label, "Severity": "Recorded class"},
        color_discrete_map=_severity_palette(table["Severity"].unique().tolist()),
        hover_data={"Records": True, y_col: ":.2f" if normalize else True},
    )
    fig.update_xaxes(tickangle=-25)
    return _chart_layout(fig, 380)


def _month_chart(df: pd.DataFrame, title: str = "Monthly accident trends") -> go.Figure:
    if "Month" not in df:
        return _chart_layout(go.Figure().add_annotation(text="Month unavailable", showarrow=False))
    table = (
        df.dropna(subset=["Month", "Severity"])
        .groupby(["Month", "Severity"], observed=True)
        .size()
        .rename("Records")
        .reset_index()
    )
    table["Month name"] = table["Month"].map(MONTH_LABELS)
    fig = px.line(
        table,
        x="Month name",
        y="Records",
        color="Severity",
        markers=True,
        title=title,
        labels={"Month name": "Month", "Records": "Accident records", "Severity": "Severity"},
        color_discrete_map=_severity_palette(table["Severity"].unique().tolist()),
        category_orders={"Month name": list(MONTH_LABELS.values())},
    )
    return _chart_layout(fig, 365)


def _weekday_chart(df: pd.DataFrame) -> go.Figure:
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    if "Weekday" not in df:
        return _chart_layout(go.Figure().add_annotation(text="Start time unavailable", showarrow=False))
    table = (
        df.dropna(subset=["Weekday", "Severity"])
        .groupby(["Weekday", "Severity"], observed=True)
        .size()
        .rename("Records")
        .reset_index()
    )
    fig = px.bar(
        table,
        x="Weekday",
        y="Records",
        color="Severity",
        barmode="group",
        title="Accidents by weekday",
        category_orders={"Weekday": order},
        labels={"Weekday": "Day", "Records": "Accident records", "Severity": "Severity"},
        color_discrete_map=_severity_palette(table["Severity"].unique().tolist()),
    )
    return _chart_layout(fig, 365)


def _environment_chart(df: pd.DataFrame, feature: str) -> go.Figure:
    if feature not in df:
        return _chart_layout(go.Figure().add_annotation(text=f"{feature} unavailable", showarrow=False))
    data = df[["Severity", feature]].copy()
    data[feature] = pd.to_numeric(data[feature], errors="coerce")
    data = data.dropna()
    if data.empty:
        return _chart_layout(go.Figure().add_annotation(text="No numeric observations available", showarrow=False))
    fig = px.box(
        data,
        x="Severity",
        y=feature,
        color="Severity",
        points=False,
        title=f"{_label(feature)} by recorded severity",
        labels={"Severity": "Recorded severity", feature: _label(feature)},
        color_discrete_map=_severity_palette(data["Severity"].unique().tolist()),
    )
    return _chart_layout(fig, 350)


def _correlation_chart(df: pd.DataFrame) -> go.Figure:
    numeric = df.select_dtypes(include="number").copy()
    if "Severity" in numeric:
        numeric["Severity"] = df["Severity"]
    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]
    numeric = numeric.iloc[:, :14]
    if numeric.shape[1] < 2:
        return _chart_layout(go.Figure().add_annotation(text="Not enough numeric columns", showarrow=False))
    correlation = numeric.corr()
    fig = px.imshow(
        correlation,
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Numeric feature correlation",
        labels={"color": "Correlation"},
    )
    return _chart_layout(fig, max(390, 30 * len(correlation.columns)))


def _class_report_frame(report: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for line in report.splitlines():
        match = re.match(
            r"^\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)\s*$",
            line,
        )
        if match:
            label, precision, recall, f1, support = match.groups()
            records.append(
                {
                    "Severity": label,
                    "Precision": float(precision),
                    "Recall": float(recall),
                    "F1-score": float(f1),
                    "Support": int(support),
                }
            )
    return pd.DataFrame(records)


def _load_metadata() -> dict[str, Any]:
    if not METADATA_PATH.is_file():
        return {}
    try:
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _render_home(df: pd.DataFrame | None, data_error: Exception | None) -> None:
    if df is None:
        st.markdown(
            "<div class='ti-hero'><div class='ti-eyebrow'>TRAFFIC ACCIDENT INTELLIGENCE PLATFORM</div>"
            "<h1>Understand the Patterns Behind Every Accident.</h1>"
            "<p>Explore real-world traffic accident data and discover patterns with transparent analytics.</p></div>",
            unsafe_allow_html=True,
        )
        st.error(f"Dataset unavailable: {data_error}")
        return
    class_counts = df["Severity"].value_counts()
    comparison = read_comparison()
    best_f1 = "—"
    if not comparison.empty and {"Model", "F1_Macro"} <= set(comparison.columns):
        best_row = comparison.sort_values("F1_Macro", ascending=False).iloc[0]
        best_f1 = f"{float(best_row['F1_Macro']):.3f}"
    source_rows = int(df.attrs.get("source_rows", len(df)))
    sampled_label = f"sampled from {source_rows:,} source rows" if source_rows > len(df) else "loaded dataset rows"

    hero_copy, hero_art = st.columns([1.25, .75], vertical_alignment="center")
    with hero_copy:
        st.markdown(
            "<div class='ti-hero'><div class='ti-eyebrow'>TRAFFIC ACCIDENT INTELLIGENCE PLATFORM</div>"
            "<h1>Understand the Patterns Behind Every Accident.</h1>"
            "<div class='ti-tagline'>Understand Accidents. Discover Patterns. Improve Road Safety.</div>"
            "<p>Explore real-world traffic accident data, investigate weather and road conditions, "
            "uncover severity patterns, and evaluate machine learning predictions through interactive analytics.</p></div>",
            unsafe_allow_html=True,
        )
        action_left, action_right = st.columns(2)
        action_left.button(
            "Explore Accident Data",
            type="primary",
            key="home_explore_data",
            on_click=_go_to,
            args=("Explore Data",),
            width="stretch",
        )
        action_right.button(
            "Analyze Severity",
            key="home_analyze_severity",
            on_click=_go_to,
            args=("Analytics",),
            width="stretch",
        )
    with hero_art:
        st.markdown(
            """
            <div class="ti-hero-visual" aria-label="Illustration of a road and traffic analytics chart">
              <svg viewBox="0 0 480 300" width="100%" role="img" aria-label="Road analytics illustration">
                <defs>
                  <linearGradient id="road" x1="0" x2="1"><stop offset="0" stop-color="#334155"/>
                    <stop offset="1" stop-color="#0F172A"/></linearGradient>
                  <filter id="shadow"><feDropShadow dx="0" dy="8" stdDeviation="8" flood-opacity=".15"/></filter>
                </defs>
                <rect x="18" y="18" width="444" height="264" rx="18" fill="#EFF6FF"/>
                <path d="M75 279 C143 221 160 185 223 149 S337 82 405 25" fill="none" stroke="#BFDBFE" stroke-width="76"/>
                <path d="M75 279 C143 221 160 185 223 149 S337 82 405 25" fill="none" stroke="url(#road)" stroke-width="62"/>
                <path d="M75 279 C143 221 160 185 223 149 S337 82 405 25" fill="none" stroke="#E2E8F0" stroke-width="3" stroke-dasharray="15 14"/>
                <g filter="url(#shadow)">
                  <rect x="236" y="42" width="184" height="113" rx="13" fill="#fff" stroke="#DBEAFE"/>
                  <text x="253" y="67" font-family="Arial" font-size="12" font-weight="700" fill="#475569">SEVERITY PATTERNS</text>
                  <path d="M255 129 L282 113 L305 119 L332 88 L359 101 L396 77" fill="none" stroke="#2563EB" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
                  <circle cx="332" cy="88" r="5" fill="#F59E0B"/>
                  <circle cx="396" cy="77" r="5" fill="#16A34A"/>
                  <rect x="256" y="137" width="28" height="5" rx="2" fill="#DBEAFE"/>
                  <rect x="290" y="137" width="47" height="5" rx="2" fill="#BFDBFE"/>
                </g>
                <circle cx="128" cy="225" r="22" fill="#fff" stroke="#2563EB" stroke-width="3"/>
                <path d="M128 211 L128 227 M128 237 L128 238" stroke="#2563EB" stroke-width="4" stroke-linecap="round"/>
                <circle cx="372" cy="192" r="7" fill="#F59E0B"/>
                <circle cx="374" cy="191" r="16" fill="none" stroke="#F59E0B" stroke-opacity=".35" stroke-width="5"/>
              </svg>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        f"<div class='ti-note'>Data source: <a href='{SOURCE_URL}' target='_blank'>Kaggle US Accidents</a> · "
        f"{len(df):,} records analyzed ({sampled_label}). Descriptive patterns are not evidence of causation.</div>",
        unsafe_allow_html=True,
    )
    _section("A clear view of the data", "Live statistics calculated from the loaded dataset and saved evaluation.")
    metrics = st.columns(4)
    cards = [
        ("Accident records analyzed", f"{len(df):,}", sampled_label, "▦"),
        ("Available features", f"{df.shape[1] - 1:,}", "excluding the Severity target", "◈"),
        ("Severity categories", f"{df['Severity'].nunique():,}", "recorded target classes", "◷"),
        ("Best model · macro F1", best_f1, "saved held-out test result", "▥"),
    ]
    for column, (label, value, detail, icon) in zip(metrics, cards):
        with column:
            _metric_card(label, value, detail, icon)

    _section("Accident patterns at a glance", "Use the charts below to explore the sampled records.")
    left, right = st.columns(2)
    with left:
        _plotly(_severity_bar(df, "Accident severity distribution"), "home_severity")
    with right:
        _plotly(_hour_chart(df, "Accidents by hour of day"), "home_hour")
    left, right = st.columns(2)
    with left:
        _plotly(_weather_chart(df, "Most common weather conditions"), "home_weather")
    with right:
        _plotly(_month_chart(df, "Monthly accident trends"), "home_month")

    weather_col = _available(df, ["Weather_Condition", "Weather_Conditions"])
    weather_top = (
        str(df[weather_col].dropna().value_counts().index[0])
        if weather_col and not df[weather_col].dropna().empty
        else "unavailable"
    )
    hour_top = "unavailable"
    if "Hour" in df and not df["Hour"].dropna().empty:
        hour_top = f"{int(df['Hour'].dropna().value_counts().idxmax()):02d}:00"
    month_top = "not available"
    if "Month" in df and not df["Month"].dropna().empty:
        top_month = int(df["Month"].dropna().value_counts().idxmax())
        month_top = MONTH_LABELS.get(top_month, str(top_month))
    _section("What stands out in this sample", "Calculated directly from the currently loaded records.")
    st.markdown(
        f"<div class='ti-note'>The most frequent recorded severity is class <b>{class_counts.idxmax()}</b> "
        f"({class_counts.max():,} records). The most frequent hour is <b>{hour_top}</b>; "
        f"the most common weather label is <b>{weather_top}</b>; and the busiest calendar month is "
        f"<b>{month_top}</b>. These are descriptive patterns and do not establish causes.</div>",
        unsafe_allow_html=True,
    )

    _section("Explore the platform", "Choose a workspace to continue.")
    feature_specs = [
        ("▦", "Explore the dataset", "Search, filter, and inspect accident records.", "Explore Data"),
        ("◷", "Discover accident patterns", "Analyze accidents by time, weather, and road conditions.", "Analytics"),
        ("◇", "Predict recorded severity", "Experiment with the saved model and its real feature schema.", "Severity Prediction"),
        ("▥", "Evaluate models", "Compare held-out classification metrics and feature importance.", "Model Performance"),
    ]
    feature_columns = st.columns(4)
    for column, (icon, title, copy, destination) in zip(feature_columns, feature_specs):
        with column:
            st.markdown(
                f"<div class='ti-feature'><div class='ti-feature-icon'>{icon}</div>"
                f"<div class='ti-feature-title'>{title}</div>"
                f"<div class='ti-feature-copy'>{copy}</div></div>",
                unsafe_allow_html=True,
            )
            st.button(
                f"Open {title}",
                key=f"home_feature_{destination}",
                on_click=_go_to,
                args=(destination,),
                width="stretch",
            )


def _dataset_filters(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    filtered = df
    controls = st.columns(5)
    if "Severity" in filtered:
        classes = sorted(filtered["Severity"].dropna().unique(), key=str)
        selected = controls[0].multiselect(
            "Severity", classes, default=classes, key=f"{prefix}_sev"
        )
        filtered = filtered[filtered["Severity"].isin(selected)]
    weather = _available(filtered, ["Weather_Condition", "Weather_Conditions"])
    if weather:
        choices = filtered[weather].dropna().value_counts().head(20).index.tolist()
        selected_weather = controls[1].multiselect(
            "Weather", choices, default=choices, key=f"{prefix}_wx"
        )
        filtered = filtered[filtered[weather].isin(selected_weather)]
    if "Month" in filtered:
        months = sorted(pd.to_numeric(filtered["Month"], errors="coerce").dropna().unique())
        chosen_months = controls[2].multiselect(
            "Month",
            months,
            default=months,
            format_func=lambda value: MONTH_LABELS.get(int(value), str(value)),
            key=f"{prefix}_month",
        )
        filtered = filtered[filtered["Month"].isin(chosen_months)]
    if "Hour" in filtered:
        hour_range = controls[3].slider(
            "Hour range", 0, 23, (0, 23), key=f"{prefix}_hr"
        )
        filtered = filtered[filtered["Hour"].between(*hour_range)]
    state_col = _available(filtered, ["State"])
    if state_col:
        states = sorted(filtered[state_col].dropna().astype(str).unique().tolist())
        chosen_states = controls[4].multiselect(
            "State", states, default=states, key=f"{prefix}_state"
        )
        filtered = filtered[filtered[state_col].astype(str).isin(chosen_states)]
    return filtered


def _reset_explorer_filters() -> None:
    for key in (
        "explorer_sev",
        "explorer_wx",
        "explorer_month",
        "explorer_hr",
        "explorer_state",
        "explorer_query",
        "explorer_search_column",
        "explorer_page_size",
        "explorer_page",
    ):
        st.session_state.pop(key, None)


def _render_explorer(df: pd.DataFrame) -> None:
    _top_header(
        "DATA EXPLORATION",
        "Explore Accident Data",
        "Investigate the dataset using interactive filters and searchable records.",
    )
    cols = st.columns(4)
    cols[0].metric("Rows in sample", f"{len(df):,}")
    cols[1].metric("Columns", f"{len(df.columns):,}")
    cols[2].metric("Exact duplicate rows", f"{int(df.duplicated().sum()):,}")
    cols[3].metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    filter_heading, reset_column = st.columns([4, 1])
    with filter_heading:
        _section("Filter records", "Only controls for fields present in the dataset are shown.")
    with reset_column:
        st.button(
            "Reset filters",
            key="explorer_reset_filters",
            on_click=_reset_explorer_filters,
            width="stretch",
        )
    filtered = _dataset_filters(df, "explorer")
    st.caption(
        f"Active view: **{len(filtered):,} records** · "
        f"{'filtered' if len(filtered) != len(df) else 'no filters applied'}"
    )
    search_columns = st.columns([1, 2, 1])
    searchable = [column for column in filtered.columns if filtered[column].nunique(dropna=True) <= 500]
    search_column = search_columns[0].selectbox(
        "Search column",
        ["All text columns"] + searchable,
        key="explorer_search_column",
    )
    query = search_columns[1].text_input(
        "Search records", placeholder="Type a value to narrow the preview…", key="explorer_query"
    )
    if query.strip():
        if search_column == "All text columns":
            text_columns = filtered.select_dtypes(include=["object", "string"]).columns.tolist()
            mask = pd.Series(False, index=filtered.index)
            for column in text_columns:
                mask |= filtered[column].astype("string").str.contains(
                    query, case=False, na=False, regex=False
                )
            filtered = filtered[mask]
        else:
            filtered = filtered[
                filtered[search_column].astype("string").str.contains(
                    query, case=False, na=False, regex=False
                )
            ]
    row_limit = search_columns[2].selectbox(
        "Rows per page", [25, 50, 100, 250], index=1, key="explorer_page_size"
    )
    default_columns = list(filtered.columns[: min(12, len(filtered.columns))])
    selected_columns = st.multiselect(
        "Columns to display",
        options=list(filtered.columns),
        default=default_columns,
        key="explorer_columns",
    )
    if not selected_columns:
        st.info("Select at least one column to preview records.")
    else:
        total_pages = max(1, (len(filtered) + row_limit - 1) // row_limit)
        page_number = st.number_input(
            "Preview page",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1,
            key="explorer_page",
        )
        start = (int(page_number) - 1) * row_limit
        st.dataframe(
            filtered.iloc[start : start + row_limit][selected_columns],
            width="stretch",
            hide_index=True,
        )
        st.caption(f"Page {page_number} of {total_pages} · {len(filtered):,} filtered records")
    download_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered records",
        data=download_bytes,
        file_name="traffic_accidents_filtered.csv",
        mime="text/csv",
        key="explorer_download",
    )
    with st.expander("Column types and missing values"):
        column_table = pd.DataFrame(
            {
                "Column": df.columns,
                "Data type": [str(dtype) for dtype in df.dtypes],
                "Missing": df.isna().sum().values,
                "Missing %": (df.isna().mean().values * 100).round(2),
                "Unique": [df[column].nunique(dropna=True) for column in df.columns],
            }
        ).sort_values("Missing", ascending=False)
        st.dataframe(column_table, width="stretch", hide_index=True)
    with st.expander("Descriptive statistics"):
        description = df.describe(include="all").transpose()
        mixed_columns = description.select_dtypes(include=["object"]).columns
        description[mixed_columns] = description[mixed_columns].astype("string")
        st.dataframe(description, width="stretch")
    with st.expander("Severity class distribution"):
        counts = df["Severity"].value_counts().sort_index().rename_axis("Severity").reset_index(name="Records")
        counts["Percent"] = counts["Records"] / len(df) * 100
        st.dataframe(counts, width="stretch", hide_index=True)
        _plotly(_severity_bar(df, "Severity classes"), "explorer_severity")


def _render_eda(df: pd.DataFrame) -> None:
    _top_header(
        "INTERACTIVE ANALYTICS",
        "Accident Analytics",
        "Explore the conditions and time patterns associated with recorded accident severity.",
    )
    filtered = _filtered_data(df, "eda")
    if filtered.empty:
        st.warning("No records match the selected filters. Adjust the selections above.")
        return
    counts = filtered["Severity"].value_counts()
    peak_class = counts.idxmax()
    overview = st.columns(3)
    overview[0].metric("Filtered accident records", f"{len(filtered):,}")
    overview[1].metric("Most common severity", str(peak_class), f"{counts.loc[peak_class]:,} records")
    overview[2].metric("Severity categories", f"{filtered['Severity'].nunique():,}")
    st.markdown(
        f"<div class='ti-note'>In the filtered sample, class <b>{peak_class}</b> is most common "
        f"({counts.loc[peak_class]:,} of {len(filtered):,} records, "
        f"{counts.loc[peak_class] / len(filtered):.1%}). This describes the sample only "
        "and does not explain causes.</div>",
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    with left:
        _plotly(_severity_bar(filtered, "Accident severity distribution"), "eda_severity")
    with right:
        _plotly(_weather_chart(filtered, "Weather conditions by frequency"), "eda_weather")
    road_feature = _available(filtered, ROAD_FEATURES)
    lighting_feature = _available(filtered, LIGHTING_FEATURES)
    left, right = st.columns(2)
    with left:
        options = [feature for feature in ROAD_FEATURES if feature in filtered]
        selected_road = st.selectbox(
            "Road feature",
            options,
            key="eda_road_feature",
        ) if options else None
        _plotly(
            _severity_feature_chart(
                filtered, selected_road or road_feature, "Severity by road feature"
            ),
            "eda_road",
        )
    with right:
        if lighting_feature:
            lighting_options = [feature for feature in LIGHTING_FEATURES if feature in filtered]
            selected_lighting = st.selectbox(
                "Lighting / twilight feature", lighting_options, key="eda_lighting_feature"
            )
        else:
            selected_lighting = None
            st.caption("Lighting fields are not available in this dataset.")
        _plotly(
            _severity_feature_chart(
                filtered,
                selected_lighting,
                "Severity by lighting / twilight",
                normalize=True,
            ),
            "eda_lighting",
        )
    left, right = st.columns(2)
    with left:
        _plotly(_hour_chart(filtered, "Accidents by hour of day"), "eda_hour")
    with right:
        _plotly(_weekday_chart(filtered), "eda_weekday")
    left, right = st.columns(2)
    with left:
        _plotly(_month_chart(filtered, "Monthly accidents by severity"), "eda_month")
    with right:
        _plotly(_correlation_chart(filtered), "eda_correlation")
    left, right = st.columns(2)
    with left:
        _plotly(_environment_chart(filtered, "Temperature(F)"), "eda_temperature")
    with right:
        _plotly(_environment_chart(filtered, "Visibility(mi)"), "eda_visibility")
    speed_feature = _available(filtered, ["Speed_Limit", "Speed_Limit(mph)", "speed_limit"])
    if speed_feature:
        _plotly(
            _environment_chart(filtered, speed_feature),
            "eda_speed_limit",
        )
    else:
        st.info("A speed-limit field is unavailable in this dataset, so no speed-limit chart is shown.")
    _render_geographic_distribution(filtered)


def _render_geographic_distribution(filtered: pd.DataFrame) -> None:
    _section("Geographic distribution", "Coordinates are shown only when valid latitude and longitude are available.")
    lat_col = _available(filtered, ["Start_Lat", "Start_Latitude", "Latitude"])
    lon_col = _available(filtered, ["Start_Lng", "Start_Lon", "Longitude"])
    if not lat_col or not lon_col:
        st.info("Latitude / longitude columns are unavailable; no location map can be shown.")
        return
    geo = filtered[[lat_col, lon_col, "Severity"]].copy()
    geo[lat_col] = pd.to_numeric(geo[lat_col], errors="coerce")
    geo[lon_col] = pd.to_numeric(geo[lon_col], errors="coerce")
    geo = geo.dropna()
    geo = geo[geo[lat_col].between(-90, 90) & geo[lon_col].between(-180, 180)]
    if geo.empty:
        st.info("No valid geographic coordinates are available for this filter selection.")
        return
    geo_sample = geo.sample(n=min(2500, len(geo)), random_state=42)
    geo_sample = geo_sample.rename(columns={lat_col: "latitude", lon_col: "longitude"})
    map_fig = px.scatter_map(
        geo_sample,
        lat="latitude",
        lon="longitude",
        color=geo_sample["Severity"].astype(str),
        color_discrete_map=_severity_palette(geo_sample["Severity"].unique().tolist()),
        hover_data={"Severity": True, "latitude": False, "longitude": False},
        zoom=3,
        title="Accident coordinates in the filtered sample",
        map_style="open-street-map",
    )
    _plotly(_chart_layout(map_fig, 500), "analytics_map")
    st.caption(f"Map displays up to 2,500 sampled coordinates from {len(geo):,} matching records.")


def _numeric_default(df: pd.DataFrame, feature: str) -> tuple[float, float, float]:
    values = pd.to_numeric(df[feature], errors="coerce").dropna()
    if values.empty:
        return 0.0, -1_000_000.0, 1_000_000.0
    default = float(values.median())
    low, high = float(values.min()), float(values.max())
    if low == high:
        low -= 1.0
        high += 1.0
    if feature == "Hour":
        low, high = 0.0, 23.0
    elif feature == "Month":
        low, high = 1.0, 12.0
    elif feature == "Is_Weekend":
        low, high = 0.0, 1.0
    return default, low, high


def _prediction_preset(df: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
    preset: dict[str, Any] = {}
    categorical = set()
    model = get_model()
    if model is not None:
        for name, _, columns in model.named_steps["preprocessor"].transformers_:
            if name == "categorical":
                categorical.update(columns)
    for feature in feature_columns:
        if feature in categorical:
            values = df[feature].dropna().value_counts()
            preset[feature] = values.index[0] if not values.empty else "Unknown"
        elif feature in df and pd.api.types.is_bool_dtype(df[feature]):
            preset[feature] = bool(df[feature].mode().iloc[0]) if not df[feature].mode().empty else False
        elif feature in df:
            preset[feature] = _numeric_default(df, feature)[0]
        else:
            preset[feature] = 0
    return preset


def _reset_prediction_values(preset: dict[str, Any]) -> None:
    for feature, value in preset.items():
        st.session_state[f"prediction_{feature}"] = value
    st.session_state.pop("prediction_result", None)


def _render_prediction(df: pd.DataFrame) -> None:
    _top_header(
        "INTERACTIVE MACHINE LEARNING",
        "Accident Severity Prediction",
        "Explore how environmental and road-related features relate to a predicted recorded category.",
    )
    model = get_model()
    if model is None:
        st.error("The trained model is missing. Run `python -m src.train_model` first.")
        return
    try:
        preprocessor = model.named_steps["preprocessor"]
        feature_columns = list(preprocessor.feature_names_in_)
        categorical_features: set[str] = set()
        for name, _, columns in preprocessor.transformers_:
            if name == "categorical":
                categorical_features.update(columns)
    except (AttributeError, KeyError, TypeError) as exc:
        st.error(f"The saved model does not expose a usable fitted feature schema: {exc}")
        return
    st.caption(
        f"Input schema loaded from the fitted pipeline: {len(feature_columns)} features "
        f"({len(feature_columns) - len(categorical_features)} numeric / boolean, "
        f"{len(categorical_features)} categorical). No model retraining occurs on this page."
    )
    with st.expander("Features expected by the saved pipeline"):
        st.code(", ".join(feature_columns), language="text")

    preset = _prediction_preset(df, feature_columns)
    group_map = {
        "Time context": {"Hour", "Month", "Is_Weekend", "Weekday"},
        "Weather & visibility": {
            "Temperature(F)",
            "Wind_Chill(F)",
            "Humidity(%)",
            "Pressure(in)",
            "Visibility(mi)",
            "Wind_Speed(mph)",
            "Precipitation(in)",
            "Wind_Direction",
            "Weather_Condition",
            "Sunrise_Sunset",
            "Civil_Twilight",
            "Nautical_Twilight",
            "Astronomical_Twilight",
        },
        "Road context": set(feature_columns),
    }
    groups: dict[str, list[str]] = {
        label: [] for label in ("Time context", "Weather & visibility", "Road context")
    }
    assigned: set[str] = set()
    for feature in feature_columns:
        if feature in group_map["Time context"]:
            groups["Time context"].append(feature)
            assigned.add(feature)
        elif feature in group_map["Weather & visibility"]:
            groups["Weather & visibility"].append(feature)
            assigned.add(feature)
        else:
            groups["Road context"].append(feature)
            assigned.add(feature)
    for feature in feature_columns:
        if feature not in assigned:
            groups["Road context"].append(feature)

    entered: dict[str, Any] = {}
    form_col, result_col = st.columns([1.25, 1], gap="large")
    with form_col:
        _section("Accident conditions", "Use values from the supported feature set; defaults are typical sample values.")
        reset_col, _ = st.columns([1, 3])
        reset_col.button(
            "Reset to typical",
            key="prediction_reset",
            on_click=_reset_prediction_values,
            args=(preset,),
        )
        with st.form("severity_prediction_form", border=True):
            for group_name, features in groups.items():
                if not features:
                    continue
                _section(group_name)
                cols = st.columns(2)
                for index, feature in enumerate(features):
                    column = cols[index % 2]
                    display_label = _label(feature)
                    state_key = f"prediction_{feature}"
                    with column:
                        if feature in categorical_features:
                            choices = (
                                df[feature].dropna().value_counts().head(40).index.tolist()
                                if feature in df
                                else []
                            )
                            if not choices:
                                choices = ["Unknown"]
                            default_value = st.session_state.get(state_key, preset[feature])
                            if default_value not in choices:
                                choices.insert(0, default_value)
                            entered[feature] = st.selectbox(
                                display_label,
                                options=choices,
                                index=choices.index(default_value),
                                key=state_key,
                            )
                        elif feature in df and pd.api.types.is_bool_dtype(df[feature]):
                            entered[feature] = st.checkbox(
                                display_label,
                                value=bool(st.session_state.get(state_key, preset[feature])),
                                key=state_key,
                            )
                        else:
                            default_value, minimum, maximum = (
                                _numeric_default(df, feature)
                                if feature in df
                                else (0.0, -1_000_000.0, 1_000_000.0)
                            )
                            default_value = float(st.session_state.get(state_key, preset[feature]))
                            default_value = min(max(default_value, minimum), maximum)
                            entered[feature] = st.number_input(
                                display_label,
                                min_value=float(minimum),
                                max_value=float(maximum),
                                value=float(default_value),
                                step=1.0 if feature in {"Hour", "Month", "Is_Weekend"} else 0.1,
                                key=state_key,
                            )
            submitted = st.form_submit_button(
                "Predict severity", type="primary", width="stretch"
            )
    prediction_error = None
    if submitted:
        try:
            input_frame = pd.DataFrame(
                [{feature: entered[feature] for feature in feature_columns}],
                columns=feature_columns,
            )
            if list(input_frame.columns) != feature_columns or input_frame.isna().any().any():
                raise ValueError("All expected model fields must have valid values.")
            prediction = model.predict(input_frame)[0]
            probabilities = (
                model.predict_proba(input_frame)[0].tolist()
                if hasattr(model, "predict_proba")
                else None
            )
            st.session_state["prediction_result"] = {
                "prediction": str(prediction),
                "probabilities": probabilities,
            }
        except (ValueError, TypeError, KeyError) as exc:
            prediction_error = str(exc)
    with result_col:
        with st.container(key="prediction-result", border=True):
            _section("Prediction result", "Output from the saved preprocessing and classification pipeline.")
            if prediction_error:
                st.error(f"Could not calculate a prediction. Check the entered features: {prediction_error}")
            else:
                result = st.session_state.get("prediction_result")
                if result:
                    severity_key = result["prediction"]
                    color = SEVERITY_COLORS.get(severity_key, "#2563EB")
                    st.markdown(
                        f"<div style='background:#F8FAFC;border:1px solid #E2E8F0;border-left:6px solid {color};"
                        "border-radius:12px;padding:1.2rem;margin:.5rem 0 1rem'>"
                        "<div style='color:#64748B;font-size:.78rem;font-weight:700'>PREDICTED RECORDED CLASS</div>"
                        f"<div style='color:#0F172A;font-size:2rem;font-weight:800;margin-top:.25rem'>"
                        f"Severity {severity_key}</div></div>",
                        unsafe_allow_html=True,
                    )
                    probabilities = result.get("probabilities")
                    if probabilities is not None:
                        probability_df = pd.DataFrame(
                            {
                                "Severity class": model.classes_.astype(str),
                                "Probability": probabilities,
                            }
                        ).sort_values("Probability", ascending=False)
                        _plotly(
                            px.bar(
                                probability_df,
                                x="Probability",
                                y="Severity class",
                                orientation="h",
                                color="Severity class",
                                color_discrete_map=_severity_palette(
                                    probability_df["Severity class"].tolist()
                                ),
                                text=probability_df["Probability"].map(lambda value: f"{value:.1%}"),
                                labels={"Probability": "Model probability"},
                                title="Predicted class probabilities",
                            ).update_layout(yaxis={"categoryorder": "total ascending"}),
                            "prediction_probabilities",
                            300,
                        )
                else:
                    st.info("Enter the conditions on the left and select **Predict severity** to see a model estimate.")
            st.markdown(
                "<div class='ti-warning'><b>Experimental estimate only.</b> This model is for learning and "
                "historical data exploration. It is not intended for emergency, insurance, legal, or real-world "
                "road-safety decisions.</div>",
                unsafe_allow_html=True,
            )


def _render_ml_lab() -> None:
    _top_header(
        "MODEL PERFORMANCE",
        "Machine Learning Performance",
        "Review actual saved scores from the held-out test split and inspect per-class performance.",
    )
    comparison = read_comparison()
    report = read_text_file(str(REPORT_PATH))
    if comparison.empty:
        st.info("No saved model comparison is available. Run `python -m src.train_model`.")
        return
    required = {"Model", "Accuracy", "Precision_Macro", "Recall_Macro", "F1_Macro"}
    if not required.issubset(comparison.columns):
        st.error("The saved comparison file is missing required metric columns.")
        return
    metadata = _load_metadata()
    model_rows = int(metadata.get("training_rows", 0))
    test_rows = int(metadata.get("test_rows", 0))
    best_row = comparison.sort_values(
        ["F1_Macro", "Recall_Macro", "Accuracy"], ascending=False
    ).iloc[0]
    stored_selection = str(metadata.get("best_model", ""))
    is_confirmed_best = str(best_row["Model"]) == stored_selection
    metrics = st.columns(4)
    metrics[0].metric("Training records", f"{model_rows:,}" if model_rows else "—")
    metrics[1].metric("Held-out test records", f"{test_rows:,}" if test_rows else "—")
    metrics[2].metric("Selected by macro F1", stored_selection or "—")
    metrics[3].metric("Selected model macro F1", f"{float(best_row['F1_Macro']):.3f}")
    if is_confirmed_best:
        st.success(
            f"{stored_selection} is the top saved model by macro F1 "
            "(ties are resolved by macro recall and then accuracy)."
        )
    else:
        st.warning(
            "The saved best-model metadata does not match the top row in the comparison. "
            "Review the model artifacts before relying on the selection."
        )
    st.info(
        f"These results were trained on {model_rows:,} rows and evaluated on "
        f"{test_rows:,} held-out rows from the configured sample. Accuracy is the "
        "overall correct fraction; macro F1 averages the F1 score of each class equally, "
        "so rare classes affect it as much as common classes."
    )
    display = comparison.rename(
        columns={
            "Precision_Macro": "Macro precision",
            "Recall_Macro": "Macro recall",
            "F1_Macro": "Macro F1",
        }
    )
    st.dataframe(
        display.style.format(
            {
                "Accuracy": "{:.3f}",
                "Macro precision": "{:.3f}",
                "Macro recall": "{:.3f}",
                "Macro F1": "{:.3f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    bar_data = comparison.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision_Macro", "Recall_Macro", "F1_Macro"],
        var_name="Metric",
        value_name="Score",
    )
    bar_data["Metric"] = bar_data["Metric"].replace(
        {
            "Precision_Macro": "Macro precision",
            "Recall_Macro": "Macro recall",
            "F1_Macro": "Macro F1",
        }
    )
    _plotly(
        px.bar(
            bar_data,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            title="Actual held-out model comparison",
            range_y=[0, 1],
            labels={"Score": "Score (0–1)"},
        ),
        "ml_comparison",
        390,
    )
    confusion = read_csv_file(str(CONFUSION_PATH))
    class_report = _class_report_frame(report)
    left, right = st.columns(2)
    with left:
        _section("Confusion matrix", "Rows are actual classes; columns are predicted classes.")
        if not confusion.empty:
            _plotly(
                px.imshow(
                    confusion,
                    text_auto=True,
                    color_continuous_scale="Blues",
                    aspect="auto",
                    labels={"x": "Predicted class", "y": "Actual class", "color": "Records"},
                    title="Selected model · test set",
                ),
                "ml_confusion",
                410,
            )
        else:
            st.info("The confusion matrix has not been saved.")
    with right:
        _section("Per-class metrics")
        if not class_report.empty:
            st.dataframe(
                class_report.style.format(
                    {"Precision": "{:.2f}", "Recall": "{:.2f}", "F1-score": "{:.2f}"}
                ),
                width="stretch",
                hide_index=True,
            )
        elif report:
            st.code(report, language="text")
        else:
            st.info("The classification report has not been saved.")
    importance = read_csv_file(str(IMPORTANCE_PATH), index_col=None)
    if not importance.empty and {"feature", "importance"} <= set(importance.columns):
        _section("Random Forest feature importance", "Relative split-importance values from the saved Random Forest.")
        top = importance.head(20).sort_values("importance")
        _plotly(
            px.bar(
                top,
                x="importance",
                y="feature",
                orientation="h",
                title="Top Random Forest features",
                labels={"importance": "Importance", "feature": "Model feature"},
                color_discrete_sequence=["#3978C5"],
            ),
            "ml_importance",
            max(360, 20 * len(top)),
        )
    if report:
        with st.expander("Full classification report"):
            st.code(report, language="text")


def _render_reports(df: pd.DataFrame | None) -> None:
    _top_header(
        "PROJECT OUTPUTS",
        "Reports & Downloads",
        "Access the generated analysis and machine learning evaluation outputs.",
    )
    if df is not None:
        _section("Dataset summary")
        counts = df["Severity"].value_counts().sort_index()
        summary_cols = st.columns(4)
        summary_cols[0].metric("Analyzed records", f"{len(df):,}")
        summary_cols[1].metric("Available columns", f"{len(df.columns):,}")
        summary_cols[2].metric("Severity categories", f"{df['Severity'].nunique():,}")
        summary_cols[3].metric("Rows with missing values", f"{int(df.isna().any(axis=1).sum()):,}")
        st.dataframe(
            pd.DataFrame(
                {
                    "Severity": counts.index.astype(str),
                    "Records": counts.values,
                    "Percent": (counts.values / len(df) * 100).round(2),
                }
            ),
            width="stretch",
            hide_index=True,
        )
        _section("Filtered dataset export")
        filtered = _dataset_filters(df, "reports")
        _download_card(
            "Filtered dataset sample",
            "traffic_accidents_filtered.csv",
            f"{len(filtered):,} currently filtered rows from the local sample.",
        )
        st.download_button(
            "Download filtered sample records",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="traffic_accidents_filtered.csv",
            mime="text/csv",
            key="reports_data_download",
        )
    eda_text = read_text_file(str(EDA_REPORT_PATH))
    if eda_text:
        _section("EDA findings from the analyzed sample")
        _download_card("Exploratory analysis report", EDA_REPORT_PATH.name, "Calculated summary of the actual dataset sample.")
        st.text(eda_text)
        st.download_button(
            "Download EDA report",
            data=eda_text.encode("utf-8"),
            file_name="eda_report.txt",
            mime="text/plain",
            key="download_eda_report",
        )
    else:
        st.info("EDA report is not available yet. Run `python -m src.eda`.")
    comparison = read_comparison()
    if not comparison.empty:
        _section("Model comparison")
        _download_card("Held-out model scores", COMPARISON_PATH.name, "Accuracy and macro-averaged classification metrics.")
        st.dataframe(comparison, width="stretch", hide_index=True)
        st.download_button(
            "Download model comparison CSV",
            data=COMPARISON_PATH.read_bytes(),
            file_name="model_comparison.csv",
            mime="text/csv",
            key="download_comparison",
        )
    else:
        st.info("Model comparison report is not available yet. Run `python -m src.train_model`.")
    report_text = read_text_file(str(REPORT_PATH))
    if report_text:
        _section("Evaluation report")
        _download_card("Selected model evaluation", REPORT_PATH.name, "Per-class scores and selected model details.")
        st.code(report_text, language="text")
        st.download_button(
            "Download classification / evaluation report",
            data=report_text.encode("utf-8"),
            file_name="evaluation_report.txt",
            mime="text/plain",
            key="download_evaluation",
        )
    confusion = read_csv_file(str(CONFUSION_PATH))
    if not confusion.empty:
        _section("Confusion matrix")
        _download_card("Confusion matrix", CONFUSION_PATH.name, "Actual and predicted class counts for the held-out test set.")
        st.dataframe(confusion, width="stretch")
        st.download_button(
            "Download confusion matrix CSV",
            data=CONFUSION_PATH.read_bytes(),
            file_name="confusion_matrix.csv",
            mime="text/csv",
            key="download_confusion",
        )
    chart_paths = sorted(path for path in PLOTS_DIR.glob("*.png") if path.is_file())
    if chart_paths:
        _section("Generated visualizations")
        selected_chart_name = st.selectbox(
            "Choose a chart", [path.name for path in chart_paths], key="report_chart_select"
        )
        selected_chart = PLOTS_DIR / selected_chart_name
        _download_card("Generated chart", selected_chart.name, "A saved visualization produced from the dataset sample.")
        st.image(str(selected_chart), caption=selected_chart.stem.replace("_", " ").title(), width="stretch")
        st.download_button(
            "Download selected chart",
            data=selected_chart.read_bytes(),
            file_name=selected_chart.name,
            mime="image/png",
            key="download_selected_chart",
        )
    else:
        st.info("No chart images have been generated yet. Run `python -m src.eda`.")


def _download_card(title: str, filename: str, description: str) -> None:
    st.markdown(
        f"<div class='ti-download-card'><div class='ti-download-title'>⇩ &nbsp;{title}</div>"
        f"<div class='ti-download-meta'>{filename} · {description}</div></div>",
        unsafe_allow_html=True,
    )


try:
    with st.spinner("Loading the cached accident sample…"):
        dataset = get_dataset()
    dataset_error: Exception | None = None
except (FileNotFoundError, ValueError, OSError, pd.errors.ParserError) as exc:
    dataset = None
    dataset_error = exc

current_page = _navigation(dataset, dataset_error)

if current_page == "Home":
    _render_home(dataset, dataset_error)
elif current_page == "Explore Data":
    _top_header(
        "DATA EXPLORATION",
        "Explore Accident Data",
        "Explore the sampled records and inspect dataset quality.",
    ) if dataset is None else None
    if dataset is None:
        st.error(f"Dataset unavailable: {dataset_error}")
    else:
        _render_explorer(dataset)
elif current_page == "Analytics":
    if dataset is None:
        st.error(f"Dataset unavailable: {dataset_error}")
    else:
        _render_eda(dataset)
elif current_page == "Severity Prediction":
    if dataset is None:
        st.error(f"Dataset unavailable: {dataset_error}")
    else:
        _render_prediction(dataset)
elif current_page == "Model Performance":
    _render_ml_lab()
elif current_page == "Reports":
    _render_reports(dataset)
