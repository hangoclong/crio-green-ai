"""Tests for M15: Semantic Structural Break Detection.

RED phase: These tests define the expected interface for the semantic
break module. The module must compute SPECTER2 embeddings, calculate
yearly centroid drift, and detect semantic regime shifts.
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
    """Create a synthetic corpus with known semantic shift.

    Papers from 2019-2022 are about topic A (hardware efficiency).
    Papers from 2023-2025 are about topic B (LLM carbon footprint).
    This should produce a detectable semantic centroid drift.
    """
    np.random.seed(42)
    years_early = [2019] * 5 + [2020] * 10 + [2021] * 15 + [2022] * 20
    years_late = [2023] * 40 + [2024] * 60 + [2025] * 80

    titles_early = [
        f"Energy-efficient edge computing architecture {i}"
        for i in range(len(years_early))
    ]
    abstracts_early = [
        "This paper proposes a novel FPGA-based accelerator for deep learning "
        "inference that reduces energy consumption by 40% compared to GPU baselines. "
        "We evaluate power usage effectiveness on edge devices."
        for _ in range(len(years_early))
    ]

    titles_late = [
        f"Carbon footprint of large language models and generative AI {i}"
        for i in range(len(years_late))
    ]
    abstracts_late = [
        "We measure the environmental impact of training GPT-4 class models, "
        "finding that a single training run emits 500 tons of CO2. We propose "
        "carbon pricing mechanisms for responsible AI deployment."
        for _ in range(len(years_late))
    ]

    df = pd.DataFrame({
        "id": list(range(len(years_early) + len(years_late))),
        "year": years_early + years_late,
        "title": titles_early + titles_late,
        "abstract": abstracts_early + abstracts_late,
    })
    return df


@pytest.fixture
def out_dir(tmp_path):
    d = tmp_path / "scisci_test_output"
    d.mkdir()
    return str(d)


class TestSemanticBreak:
    """Verify semantic structural break detection module."""

    def test_module_importable(self):
        """m15_semantic_break should be importable with a run() function."""
        from scisci.m15_semantic_break import run
        assert callable(run)

    def test_run_returns_expected_keys(self, papers_df, out_dir):
        """run() should return dict with drift series and break results."""
        from scisci.m15_semantic_break import run
        result = run(papers_df, {}, out_dir)
        assert "max_drift_year" in result
        assert "max_drift_value" in result
        assert "drift_series" in result

    def test_drift_table_created(self, papers_df, out_dir):
        """Should produce table_semantic_break.md."""
        from scisci.m15_semantic_break import run
        run(papers_df, {}, out_dir)
        table_path = os.path.join(out_dir, "table_semantic_break.md")
        assert os.path.exists(table_path)
        with open(table_path) as f:
            content = f.read()
        assert "drift" in content.lower() or "cosine" in content.lower()

    def test_drift_figure_created(self, papers_df, out_dir):
        """Should produce fig_semantic_drift.png."""
        from scisci.m15_semantic_break import run
        run(papers_df, {}, out_dir)
        fig_path = os.path.join(out_dir, "fig_semantic_drift.png")
        assert os.path.exists(fig_path)
        assert os.path.getsize(fig_path) > 1000

    def test_detects_shift_in_synthetic_data(self, papers_df, out_dir):
        """With a clear topic shift at 2023, max drift should be around 2022-2023."""
        from scisci.m15_semantic_break import run
        result = run(papers_df, {}, out_dir)
        # The max drift should occur around the transition year
        assert result["max_drift_year"] in [2022, 2023]
        # Drift value should be non-trivial (> 0.01)
        assert result["max_drift_value"] > 0.01

    def test_drift_series_has_correct_length(self, papers_df, out_dir):
        """Drift series length should equal (n_years - 1)."""
        from scisci.m15_semantic_break import run
        result = run(papers_df, {}, out_dir)
        unique_years = sorted(papers_df["year"].unique())
        assert len(result["drift_series"]) == len(unique_years) - 1


class TestEmbeddingFunction:
    """Verify the SPECTER2 embedding helper works correctly."""

    def test_embed_papers_returns_correct_shape(self):
        """Embedding function should return (n_papers, dim) array."""
        from scisci.m15_semantic_break import embed_papers
        df = pd.DataFrame({
            "title": ["Test paper one", "Test paper two", "Test paper three"],
            "abstract": ["Abstract one.", "Abstract two.", "Abstract three."],
        })
        embeddings = embed_papers(df)
        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] > 0  # Embedding dimension (768 for SPECTER2)

    def test_embeddings_are_l2_normalized(self):
        """All embedding vectors should be L2-normalized."""
        from scisci.m15_semantic_break import embed_papers
        df = pd.DataFrame({
            "title": ["Test paper one", "Test paper two"],
            "abstract": ["Abstract one.", "Abstract two."],
        })
        embeddings = embed_papers(df)
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=0.01)
