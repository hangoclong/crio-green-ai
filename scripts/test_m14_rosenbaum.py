"""Tests for M14: Rosenbaum Bounds Sensitivity Analysis.

RED phase: These tests define the expected interface and behavior
before implementation exists.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

# Ensure scisci package is importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))


@pytest.fixture
def matched_sample():
    """Create a minimal PSM matched sample for Rosenbaum bounds testing.

    Simulates 40 treated + 40 control papers with known outcome differences.
    """
    np.random.seed(42)
    n_per_group = 40

    treated = pd.DataFrame({
        "is_boundary": 1,
        "log_citations": np.random.normal(3.5, 1.0, n_per_group),
        "propensity_score": np.random.uniform(0.3, 0.7, n_per_group),
        "paper_age": np.random.randint(1, 7, n_per_group),
        "author_count": np.random.randint(1, 6, n_per_group),
        "degree": np.random.randint(0, 20, n_per_group),
        "author_prestige": np.random.randint(1, 10, n_per_group),
        "is_review": np.random.choice([0, 1], n_per_group, p=[0.85, 0.15]),
    })

    control = pd.DataFrame({
        "is_boundary": 0,
        "log_citations": np.random.normal(3.0, 1.0, n_per_group),
        "propensity_score": np.random.uniform(0.3, 0.7, n_per_group),
        "paper_age": np.random.randint(1, 7, n_per_group),
        "author_count": np.random.randint(1, 6, n_per_group),
        "degree": np.random.randint(0, 20, n_per_group),
        "author_prestige": np.random.randint(1, 10, n_per_group),
        "is_review": np.random.choice([0, 1], n_per_group, p=[0.85, 0.15]),
    })

    return pd.concat([treated, control], ignore_index=True)


@pytest.fixture
def out_dir(tmp_path):
    """Temporary output directory for test artifacts."""
    d = tmp_path / "scisci_test_output"
    d.mkdir()
    return str(d)


class TestRosenbaumBounds:
    """Verify Rosenbaum bounds sensitivity analysis module."""

    def test_module_importable(self):
        """m14_rosenbaum should be importable with a run() function."""
        from scisci.m14_rosenbaum import run
        assert callable(run)

    def test_gamma_star_returned(self, matched_sample, out_dir):
        """run() should return a dict containing gamma_star (Γ*)."""
        from scisci.m14_rosenbaum import run
        result = run(matched_sample, out_dir)
        assert "gamma_star" in result
        assert isinstance(result["gamma_star"], float)
        assert result["gamma_star"] >= 1.0  # Γ* is always >= 1.0

    def test_p_bounds_table_created(self, matched_sample, out_dir):
        """Should produce a markdown table of p-value bounds per Γ."""
        from scisci.m14_rosenbaum import run
        run(matched_sample, out_dir)
        table_path = os.path.join(out_dir, "table_rosenbaum_bounds.md")
        assert os.path.exists(table_path)
        with open(table_path) as f:
            content = f.read()
        assert "Γ" in content or "Gamma" in content
        assert "p-value" in content.lower() or "p_upper" in content.lower()

    def test_sensitivity_figure_created(self, matched_sample, out_dir):
        """Should produce a sensitivity plot (fig_rosenbaum_sensitivity.png)."""
        from scisci.m14_rosenbaum import run
        run(matched_sample, out_dir)
        fig_path = os.path.join(out_dir, "fig_rosenbaum_sensitivity.png")
        assert os.path.exists(fig_path)
        assert os.path.getsize(fig_path) > 1000  # not an empty file

    def test_gamma_star_coherent_with_data(self, matched_sample, out_dir):
        """With a small treatment effect, Γ* should not be astronomically high."""
        from scisci.m14_rosenbaum import run
        result = run(matched_sample, out_dir)
        # For our synthetic data with moderate effect, Γ* should be reasonable
        assert 1.0 <= result["gamma_star"] <= 10.0

    def test_handles_empty_matched_sample(self, out_dir):
        """Should handle gracefully if matched sample is tiny."""
        from scisci.m14_rosenbaum import run
        tiny = pd.DataFrame({
            "is_boundary": [1, 0],
            "log_citations": [3.0, 2.5],
            "propensity_score": [0.5, 0.5],
        })
        result = run(tiny, out_dir)
        assert "gamma_star" in result
