"""Tests ensuring EDA outputs are derived from supplied data."""

import pandas as pd
import pytest

from src.eda import generate_all_charts, generate_eda_report
from tests.test_preprocessing import sample_accidents


def test_eda_creates_charts_and_report_from_actual_input(tmp_path):
    data = sample_accidents(32)
    charts = generate_all_charts(data, tmp_path / "plots")
    report = generate_eda_report(data, tmp_path / "reports")
    assert len(charts) >= 8
    assert all(path.is_file() and path.stat().st_size > 0 for path in charts.values())
    assert "Records analyzed: 32" in report
    assert "SEVERITY DISTRIBUTION" in report
    assert (tmp_path / "reports" / "eda_report.txt").is_file()


def test_eda_requires_severity_and_nonempty_data(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        generate_all_charts(pd.DataFrame(), tmp_path)
    with pytest.raises(ValueError, match="Severity"):
        generate_eda_report(pd.DataFrame({"weather": ["Clear"]}), tmp_path)
