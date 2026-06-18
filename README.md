# CRIO Replication Package

**Paper:** Carbon-Return Integrated Optimization (CRIO): Reframing the AI-Climate Paradox for Sustainable Economic Decision-Making

**Repository:** [https://github.com/hangoclong/crio-green-ai](https://github.com/hangoclong/crio-green-ai)

---

## Overview

This repository contains the complete dataset and analysis pipeline for reproducing all quantitative results reported in the manuscript. The pipeline implements 17 analytical modules covering descriptive bibliometrics, quasi-experimental causal inference (PSM, R-Learner), sensitivity analysis (Rosenbaum bounds, placebo/permutation tests), growth modeling (Bass diffusion, Prophet changepoint detection), semantic drift analysis (SPECTER2), and stochastic Monte Carlo simulation of the CRIO decision framework.

## Directory Structure

```
crio-green-ai/
├── README.md                  # This file
├── config/
│   └── scisci_config.yaml     # Hypothesis definitions, variable specifications, paths
├── data/
│   └── dataset_a_enriched.csv # 946-paper bibliometric dataset (Scopus + OpenAlex)
└── scripts/
    ├── run_all.py             # Master orchestrator — runs all modules sequentially
    ├── data_loader.py         # Data ingestion and feature engineering
    ├── style.py               # Publication-quality figure styling
    ├── m01_descriptive.py     # Descriptive statistics and bibliometric laws
    ├── m02_chi_square.py      # Chi-square silo structure test (H1)
    ├── m03_citation_diag.py   # Citation distribution diagnostics
    ├── m04_ols_logistic.py    # OLS and logistic regression baselines
    ├── m05_neg_binomial.py    # Negative binomial count model
    ├── m06_structural_break.py# Structural break detection
    ├── m07_psm.py             # Propensity Score Matching (ATT estimation)
    ├── m08_causal_forest.py   # R-Learner causal forest (ATE/HTE)
    ├── m09_triple_helix.py    # Triple Helix collaboration analysis
    ├── m10_network_mod.py     # Network centrality moderation
    ├── m11_shap_analysis.py   # SHAP feature importance
    ├── m12_topic_forecast.py  # Topic drift forecasting
    ├── m13_bertopic_evo.py    # BERTopic temporal evolution
    ├── m14_rosenbaum.py       # Rosenbaum sensitivity bounds
    ├── m15_semantic_break.py  # SPECTER2 semantic structural break
    ├── m16_bass_prophet.py    # Bass diffusion + Prophet changepoints
    ├── m17_monte_carlo_crio.py# Monte Carlo CRIO simulation (10,000 draws)
    ├── global_sensitivity_crio.py  # Global sensitivity analysis
    ├── placebo_permutation_test.py # Placebo/permutation falsification tests
    ├── fetch_openalex_dates.py     # OpenAlex date enrichment utility
    ├── test_m14_rosenbaum.py       # Unit tests for Rosenbaum module
    ├── test_m15_semantic_break.py  # Unit tests for semantic break module
    └── test_m16_bass_prophet.py    # Unit tests for Bass/Prophet module
```

## Dataset Description

**File:** `data/dataset_a_enriched.csv` (946 rows × 30+ columns)

| Column | Description |
|--------|-------------|
| `title` | Paper title |
| `authors` | Author list |
| `year` | Publication year |
| `journal` | Source journal |
| `doi` | Digital Object Identifier |
| `abstract` | Paper abstract |
| `citations` | Citation count (Scopus) |
| `affiliations` | Author affiliations |
| `cluster_id` | Leiden community cluster assignment (0–3) |
| `cluster_label` | Semantic cluster label from BERTopic |
| `openalex_topic` | OpenAlex topic classification |
| `openalex_subfield` | OpenAlex subfield |
| `openalex_field` | OpenAlex field |
| `keywords_normalized` | Normalized keyword set |

**Source:** Scopus database (search executed 2026-05-26), enriched with OpenAlex metadata via API.

## Requirements

```
Python >= 3.10
pandas >= 2.0
numpy >= 1.24
scikit-learn >= 1.3
statsmodels >= 0.14
scipy >= 1.11
matplotlib >= 3.7
seaborn >= 0.12
PyYAML >= 6.0
```

Optional (for specific modules):
```
shap >= 0.43           # m11_shap_analysis.py
bertopic >= 0.16       # m13_bertopic_evo.py
sentence-transformers  # m15_semantic_break.py (SPECTER2)
prophet >= 1.1         # m16_bass_prophet.py
```

## Reproducibility

All stochastic algorithms use a fixed random seed (`random_state=42`) for full reproducibility. The complete parameter registry is documented in S1 Supporting Information, Table S3.

### Quick Start

```bash
# Clone the repository
git clone https://github.com/hangoclong/crio-green-ai.git
cd crio-green-ai

# Install dependencies
pip install pandas numpy scikit-learn statsmodels scipy matplotlib seaborn pyyaml

# Run the full pipeline
python scripts/run_all.py --config config/scisci_config.yaml

# Run individual modules
python scripts/m07_psm.py           # Propensity Score Matching
python scripts/m08_causal_forest.py # R-Learner causal forest
python scripts/m17_monte_carlo_crio.py  # Monte Carlo CRIO simulation

# Run unit tests
python -m pytest scripts/test_*.py -v
```

### Key Outputs

Running the pipeline produces:
- **17 publication-quality figures** (PNG, 300 DPI) in the `output/figures/` directory
- **18 markdown tables** with statistical results
- Console output with all reported statistics (ATT, ATE, χ², p-values, R², etc.)

## Mapping Modules to Manuscript Sections

| Module | Manuscript Section | Key Result |
|--------|--------------------|------------|
| `m01` | §4.1–4.4 | Descriptive bibliometrics |
| `m02` | §4.7 | χ² = 133.92, p < 0.001 (H1) |
| `m07` | §4.8 | ATT = 0.144, p = 0.266 |
| `m08` | §4.8 | ATE = 0.329 ± 0.025 log-citations |
| `m14` | §4.8 | Γ* = 1.0, p_upper = 0.624 |
| `m15` | §4.1 | Δcos = 0.103, z = 2.42 |
| `m16` | §4.10 | Bass R² = 0.996, 5% saturation |
| `m17` | §5.1 | Median CRIO = 3.226 (EU ETS) |

## License

This dataset and code are provided for academic reproducibility. Please cite the manuscript if you use this material.

## Contact

N. Long Ha, Ph.D.  
University of Economics, Hue University  
Email: hnlong@hueuni.edu.vn
