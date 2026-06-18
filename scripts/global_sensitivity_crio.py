#!/usr/bin/env python3
"""Global Sensitivity Analysis for CRIO Monte Carlo.

Computes Spearman rank-order correlations between each input parameter
and the CRIO output across 10,000 Monte Carlo draws, identifying which
parameters drive the variance in economic viability.

This analysis re-uses the EXACT same sampling logic and CRIO formula as
m17_monte_carlo_crio.py to ensure consistency.

Usage:
    uv run python papers/6b-crio/scripts/scisci/global_sensitivity_crio.py

Output:
    - experiments/results/figures/10-scisci/table_crio_gsa.md
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

# ── Physical constants (same as m17) ────────────────────────────────────────
F_ELEC = 0.85
PUE = 1.10
LAMBDA_DECAY = 0.02
T = 5
r = 0.08
N = 10_000

# ── Project paths ───────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
OUT_DIR = os.path.join(PROJECT_ROOT, "experiments", "results", "figures", "10-scisci")


def crio_npv(V_e, C_f, C_i, p_c, E_m_series, T, r):
    """CRIO formula — identical to m17."""
    if C_i <= 0:
        return 0.0
    discounted_sum = 0.0
    for t in range(1, T + 1):
        discount = (1 + r) ** (-t)
        E_mt = E_m_series[t - 1] if t - 1 < len(E_m_series) else E_m_series[-1]
        net_value = V_e - (C_f + p_c * E_mt)
        discounted_sum += discount * net_value
    return discounted_sum / C_i


def compute_emission_series(alpha_0, C_i, epsilon, T):
    """Emission model — identical to m17."""
    t_arr = np.arange(1, T + 1)
    # Scale by f_elec × PUE to get total facility energy draw
    E_m = (alpha_0 * np.exp(-LAMBDA_DECAY * t_arr)
           * (C_i ** epsilon) * F_ELEC * PUE) / 1000.0
    return E_m


def main():
    print("=" * 60)
    print("  Global Sensitivity Analysis — CRIO Monte Carlo")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # ── Draw parameters (identical to m17) ──────────────────────────────────
    C_i_draws = rng.lognormal(mean=12.2, sigma=0.7, size=N)
    Ve_ratio_draws = rng.uniform(0.3, 1.5, size=N)
    V_e_draws = Ve_ratio_draws * C_i_draws
    alpha_0_draws = rng.uniform(0.2, 0.7, size=N)
    epsilon_draws = rng.uniform(0.6, 0.9, size=N)
    C_f_draws = 0.10 * C_i_draws

    # Use EU ETS regime as baseline for GSA
    p_c_draws = rng.uniform(50, 80, size=N)

    # ── Compute CRIO for all draws ──────────────────────────────────────────
    crio_vals = np.zeros(N)
    for i in range(N):
        E_m_series = compute_emission_series(
            alpha_0_draws[i], C_i_draws[i], epsilon_draws[i], T
        )
        crio_vals[i] = crio_npv(
            V_e_draws[i], C_f_draws[i], C_i_draws[i],
            p_c_draws[i], E_m_series, T, r
        )

    # ── Spearman rank-order correlations ────────────────────────────────────
    params = {
        "V_e/C_i (Economic Return Ratio)": Ve_ratio_draws,
        "C_i (Capital Investment)": C_i_draws,
        "α₀ (Grid Carbon Intensity)": alpha_0_draws,
        "p_c (Carbon Price)": p_c_draws,
        "ε (Capital-to-Compute Elasticity)": epsilon_draws,
    }

    print(f"\n  Spearman Rank Correlations with CRIO (N={N:,}):")
    print(f"  {'Parameter':<40} {'ρ':>8} {'p-value':>12}")
    print("  " + "-" * 62)

    results = []
    for name, values in params.items():
        rho, p_val = stats.spearmanr(values, crio_vals)
        results.append({"Parameter": name, "ρ": rho, "p-value": p_val})
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        print(f"  {name:<40} {rho:>8.4f} {p_val:>12.2e}  {sig}")

    results_df = pd.DataFrame(results).sort_values("ρ", key=abs, ascending=False)

    # ── Write output table ──────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    lines = [
        "## Global Sensitivity Analysis — Spearman Rank Correlations\n",
        f"**N = {N:,} Monte Carlo draws | EU ETS regime ($50–$80/tCO₂e)**\n",
        "| Parameter | Spearman ρ | p-value | Interpretation |",
        "|:---|---:|---:|:---|",
    ]

    for _, row in results_df.iterrows():
        rho = row["ρ"]
        p = row["p-value"]
        if abs(rho) > 0.8:
            interp = "**Dominant driver**"
        elif abs(rho) > 0.3:
            interp = "Moderate driver"
        elif abs(rho) > 0.1:
            interp = "Weak driver"
        else:
            interp = "Negligible"
        lines.append(f"| {row['Parameter']} | {rho:.4f} | {p:.2e} | {interp} |")

    out_path = os.path.join(OUT_DIR, "table_crio_gsa.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n  ✅ GSA table saved: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
