"""Tests for M16: Bass Diffusion Model + Prophet Changepoint Detection.

RED phase: Tests define the expected interface for the combined
forecasting module that supplements the existing ARIMA-based m12.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))


@pytest.fixture
def papers_df():
    """Create a synthetic publication time series for forecasting tests.

    Simulates an S-curve growth pattern (suitable for Bass diffusion).
    """
    np.random.seed(42)
    # Simulate accelerating publication growth
    years = list(range(2019, 2026))
    counts = [1, 6, 18, 38, 57, 124, 413]

    rows = []
    for year, count in zip(years, counts):
        for i in range(count):
            rows.append({
                "id": f"{year}_{i}",
                "year": year,
                "publication_date": f"{year}-{np.random.randint(1,13):02d}-15",
                "cluster_id": np.random.choice([0, 1, 2, 3]),
                "cluster_label": f"Cluster {np.random.choice([0, 1, 2, 3])}",
            })

    return pd.DataFrame(rows)


@pytest.fixture
def cfg():
    """Minimal config dict."""
    return {
        "project": {"current_year": 2026},
        "variables": {
            "theme_a": {"name": "is_governance", "label": "Governance"},
            "theme_b": {"name": "is_capability", "label": "Technical"},
        },
    }


@pytest.fixture
def out_dir(tmp_path):
    d = tmp_path / "scisci_test_output"
    d.mkdir()
    return str(d)


class TestBassModel:
    """Verify Bass Diffusion Model component."""

    def test_module_importable(self):
        """m16_bass_prophet should be importable with a run() function."""
        from scisci.m16_bass_prophet import run
        assert callable(run)

    def test_bass_parameters_returned(self, papers_df, cfg, out_dir):
        """run() should return dict containing Bass model parameters (p, q, M)."""
        from scisci.m16_bass_prophet import run
        result = run(papers_df, cfg, out_dir)
        assert "bass" in result
        bass = result["bass"]
        assert "p" in bass  # innovation coefficient
        assert "q" in bass  # imitation coefficient
        assert "M" in bass  # saturation ceiling
        assert bass["p"] > 0
        assert bass["q"] > 0
        assert bass["M"] > sum([1, 6, 18, 38, 57, 124, 413])  # M > current total

    def test_bass_table_created(self, papers_df, cfg, out_dir):
        """Should produce table_bass_diffusion.md."""
        from scisci.m16_bass_prophet import run
        run(papers_df, cfg, out_dir)
        assert os.path.exists(os.path.join(out_dir, "table_bass_diffusion.md"))

    def test_bass_figure_created(self, papers_df, cfg, out_dir):
        """Should produce fig_bass_adoption.png."""
        from scisci.m16_bass_prophet import run
        run(papers_df, cfg, out_dir)
        fig_path = os.path.join(out_dir, "fig_bass_adoption.png")
        assert os.path.exists(fig_path)
        assert os.path.getsize(fig_path) > 1000


class TestProphetChangepoints:
    """Verify Prophet Changepoint Detection component."""

    def test_prophet_results_returned(self, papers_df, cfg, out_dir):
        """run() should return dict containing prophet changepoint data."""
        from scisci.m16_bass_prophet import run
        result = run(papers_df, cfg, out_dir)
        assert "prophet" in result
        prophet = result["prophet"]
        assert "changepoints" in prophet
        assert isinstance(prophet["changepoints"], list)

    def test_prophet_table_created(self, papers_df, cfg, out_dir):
        """Should produce table_prophet_changepoints.md."""
        from scisci.m16_bass_prophet import run
        run(papers_df, cfg, out_dir)
        assert os.path.exists(os.path.join(out_dir, "table_prophet_changepoints.md"))

    def test_prophet_figure_created(self, papers_df, cfg, out_dir):
        """Should produce fig_prophet_decomposition.png."""
        from scisci.m16_bass_prophet import run
        run(papers_df, cfg, out_dir)
        fig_path = os.path.join(out_dir, "fig_prophet_decomposition.png")
        assert os.path.exists(fig_path)
        assert os.path.getsize(fig_path) > 1000


class TestBassFit:
    """Verify the Bass model fitting function directly."""

    def test_fit_bass_returns_parameters(self):
        """fit_bass() should return (p, q, M, r_squared)."""
        from scisci.m16_bass_prophet import fit_bass
        # Cumulative: [1, 7, 25, 63, 120, 244, 657]
        cumulative = np.array([1, 7, 25, 63, 120, 244, 657], dtype=float)
        t = np.arange(len(cumulative), dtype=float)
        p, q, M, r2 = fit_bass(t, cumulative)
        assert p > 0
        assert q > 0
        assert M > cumulative[-1]
        assert 0 <= r2 <= 1.0

    def test_fit_bass_with_too_few_points(self):
        """fit_bass() should handle gracefully with < 4 data points."""
        from scisci.m16_bass_prophet import fit_bass
        cumulative = np.array([1, 7], dtype=float)
        t = np.arange(len(cumulative), dtype=float)
        p, q, M, r2 = fit_bass(t, cumulative)
        # Should return defaults, not crash
        assert M >= 0


class TestTerminalYearCheck:
    """Verify terminal year truncation diagnostic."""

    def test_check_terminal_year_returns_flag(self):
        """check_terminal_year should return a boolean flag."""
        from scisci.data_loader import check_terminal_year
        df = pd.DataFrame({"year": [2023] * 100 + [2024] * 150 + [2025] * 400 + [2026] * 50})
        is_truncated = check_terminal_year(df)
        assert isinstance(is_truncated, bool)

    def test_detects_truncated_year(self):
        """Should detect 2026 as truncated if count < 75% of 2025."""
        from scisci.data_loader import check_terminal_year
        df = pd.DataFrame({"year": [2024] * 100 + [2025] * 400 + [2026] * 50})
        assert check_terminal_year(df) is True

    def test_no_truncation_if_full_year(self):
        """Should return False if terminal year is similar to penultimate."""
        from scisci.data_loader import check_terminal_year
        df = pd.DataFrame({"year": [2024] * 100 + [2025] * 400 + [2026] * 380})
        assert check_terminal_year(df) is False
