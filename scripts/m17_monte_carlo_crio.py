"""M17: Monte Carlo CRIO Simulation — Carbon-Return Integrated Optimization.

Implements the NPV-discounted CRIO formula from manuscript §5.1 with:
- Table 9: Deterministic sensitivity across three carbon pricing regimes.
- Table 10: Stochastic Monte Carlo (10⁴ draws) with structural emission model.
- BLOOM retrospective case + "Dirty BLOOM" counterfactual.
- Distribution histogram with regime overlays.

References:
    Strubell et al. (2019) — BLOOM energy reporting baseline.
    Stern & Stiglitz (2017) — Social cost of carbon benchmarks.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from scisci.style import PAL, save_fig


# ── Physical constants for emission modelling ────────────────────────────────
F_ELEC = 0.85       # Fraction of total energy that is electrical (GPU workload)
R_ELEC = 3.6e6      # J per kWh conversion factor (unused in simplified model)
PUE = 1.10          # Power Usage Effectiveness (hyperscale average)
LAMBDA_DECAY = 0.02 # Annual grid decarbonization rate


def crio_npv(V_e: float, C_f: float, C_i: float, p_c: float,
             E_m_series: np.ndarray, T: int, r: float) -> float:
    """Compute the NPV-discounted CRIO index.

    CRIO = (1/C_i) * Σ_{t=1}^{T} (1+r)^{-t} * [V_e - (C_f + p_c * E_m,t)]

    Args:
        V_e: Annual expected economic value ($).
        C_f: Annual baseline operational cost ($).
        C_i: Initial capital investment ($).
        p_c: Carbon price per tonne CO₂e ($/tCO₂e).
        E_m_series: Array of annual emissions [t=1..T] in tCO₂e.
        T: Deployment horizon (years).
        r: Discount rate.

    Returns:
        CRIO index (dimensionless ratio).
    """
    if C_i <= 0:
        return 0.0
    discounted_sum = 0.0
    for t in range(1, T + 1):
        discount = (1 + r) ** (-t)
        E_mt = E_m_series[t - 1] if t - 1 < len(E_m_series) else E_m_series[-1]
        net_value = V_e - (C_f + p_c * E_mt)
        discounted_sum += discount * net_value
    return discounted_sum / C_i


def compute_emission_series(alpha_0: float, C_i: float, epsilon: float,
                            T: int) -> np.ndarray:
    """Compute annual emissions with grid decarbonization and capital elasticity.

    E_{m,t} = α₀ * e^{-λt} * (C_i^ε * f_elec) * PUE

    The structural model captures:
    - α₀: initial grid carbon intensity (kgCO₂e/kWh → converted to tCO₂e)
    - λ: annual grid decarbonization rate
    - ε: capital-to-compute elasticity (sub-linear: 2x capital ≠ 2x energy)
    - PUE: data centre overhead

    Returns:
        Array of shape (T,) with annual emissions in tCO₂e.
    """
    t_arr = np.arange(1, T + 1)
    # α₀ is in kgCO₂e/kWh; C_i^ε yields a compute-proportional energy proxy
    # Scale by f_elec × PUE to get total facility energy draw
    # Divide by 1000 to convert kg → tonnes
    E_m = (alpha_0 * np.exp(-LAMBDA_DECAY * t_arr)
           * (C_i ** epsilon) * F_ELEC * PUE) / 1000.0
    return E_m


def run(df: pd.DataFrame, cfg: dict, out_dir: str) -> dict:
    """Run Monte Carlo CRIO simulation.

    Args:
        df: Dataset DataFrame (unused — CRIO is a normative simulation).
        cfg: Config dict from scisci_config.yaml.
        out_dir: Output directory for tables and figures.

    Returns:
        Dict with summary statistics.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(42)

    # ══════════════════════════════════════════════════════════════════════════
    # TABLE 9: DETERMINISTIC SENSITIVITY
    # ══════════════════════════════════════════════════════════════════════════
    V_e = 100_000.0   # $/year expected economic value
    C_f = 20_000.0    # $/year baseline operational cost
    C_i = 200_000.0   # $ initial capital investment
    E_m_fixed = 100.0 # tCO₂e/year (constant — no decay in deterministic)
    T = 5             # years
    r = 0.08          # discount rate

    carbon_prices = [
        ("Low (Voluntary)", 20),
        ("Mid (EU ETS)", 65),
        ("High (Net-Zero)", 150),
    ]

    det_lines = [
        "## Table 9: Deterministic CRIO Sensitivity Analysis\n",
        f"**Parameters:** $V_e$ = \\${V_e:,.0f}/yr, $C_f$ = \\${C_f:,.0f}/yr, "
        f"$C_i$ = \\${C_i:,.0f}, $E_m$ = {E_m_fixed:.0f} tCO₂e/yr, "
        f"$T$ = {T} yr, $r$ = {r}\n",
        "| Carbon Regime | Price ($/tCO₂e) | Annual Carbon Cost | CRIO | NPV ($) |",
        "|:---|---:|---:|---:|---:|",
    ]

    raw_crio = crio_npv(V_e, C_f, C_i, 0.0,
                        np.full(T, E_m_fixed), T, r)
    det_results = []
    for name, p_c in carbon_prices:
        E_m_series = np.full(T, E_m_fixed)
        crio_val = crio_npv(V_e, C_f, C_i, p_c, E_m_series, T, r)
        carbon_cost = p_c * E_m_fixed
        npv = crio_val * C_i
        det_results.append((name, p_c, carbon_cost, crio_val, npv))
        det_lines.append(
            f"| {name} | ${p_c} | ${carbon_cost:,.0f} | "
            f"**{crio_val:.3f}** | ${npv:,.0f} |"
        )

    det_lines.append(f"\n*Baseline (no carbon): CRIO = {raw_crio:.3f}*\n")

    with open(os.path.join(out_dir, "table_crio_deterministic.md"), "w") as f:
        f.write("\n".join(det_lines) + "\n")
    print(f"  ✅ Table 9 (deterministic): CRIO range = "
          f"{det_results[0][3]:.3f}–{det_results[-1][3]:.3f}")

    # ══════════════════════════════════════════════════════════════════════════
    # TABLE 10: STOCHASTIC MONTE CARLO
    # ══════════════════════════════════════════════════════════════════════════
    N = 10_000

    # Draw parameters
    C_i_draws = rng.lognormal(mean=12.2, sigma=0.7, size=N)  # ~$200k median
    Ve_ratio = rng.uniform(0.3, 1.5, size=N)  # V_e / C_i ratio
    V_e_draws = Ve_ratio * C_i_draws
    alpha_0_draws = rng.uniform(0.2, 0.7, size=N)  # kgCO₂e/kWh
    epsilon_draws = rng.uniform(0.6, 0.9, size=N)   # capital-to-compute elasticity
    C_f_ratio = 0.10  # operational cost = 10% of C_i
    C_f_draws = C_f_ratio * C_i_draws

    # Three carbon pricing regimes
    regimes = {
        "EU ETS": (50, 80),
        "Stern-Stiglitz": (50, 100),
        "Net-Zero": (150, 250),
    }

    mc_results = {}
    for regime_name, (p_lo, p_hi) in regimes.items():
        p_c_draws = rng.uniform(p_lo, p_hi, size=N)
        crio_vals = np.zeros(N)
        for i in range(N):
            E_m_series = compute_emission_series(
                alpha_0_draws[i], C_i_draws[i], epsilon_draws[i], T
            )
            crio_vals[i] = crio_npv(
                V_e_draws[i], C_f_draws[i], C_i_draws[i],
                p_c_draws[i], E_m_series, T, r
            )
        mc_results[regime_name] = crio_vals
        pct_negative = 100 * np.mean(crio_vals < 0)
        print(f"  📊 {regime_name}: median CRIO = {np.median(crio_vals):.3f}, "
              f"P(CRIO<0) = {pct_negative:.1f}%")

    # Write Table 10
    mc_lines = [
        "## Table 10: Monte Carlo CRIO Simulation ($10^4$ draws)\n",
        f"**Stochastic Parameters:**\n"
        f"- $C_i \\sim \\text{{LogNormal}}(\\mu=12.2, \\sigma=0.7)$\n"
        f"- $V_e/C_i \\sim U[0.3, 1.5]$\n"
        f"- $\\alpha_0 \\sim U[0.2, 0.7]$ kgCO₂e/kWh\n"
        f"- $\\varepsilon \\sim U(0.6, 0.9)$ (capital-to-compute elasticity)\n"
        f"- $\\lambda = {LAMBDA_DECAY}$ (grid decarbonization rate)\n"
        f"- $T = {T}$, $r = {r}$\n",
        "| Regime | Price Range | Median CRIO | Mean CRIO | SD | "
        "P(CRIO < 0) | 5th pctile | 95th pctile |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|",
    ]

    for regime_name, vals in mc_results.items():
        p_lo, p_hi = regimes[regime_name]
        mc_lines.append(
            f"| {regime_name} | ${p_lo}–${p_hi} | "
            f"{np.median(vals):.3f} | {np.mean(vals):.3f} | "
            f"{np.std(vals):.3f} | {100 * np.mean(vals < 0):.1f}% | "
            f"{np.percentile(vals, 5):.3f} | {np.percentile(vals, 95):.3f} |"
        )

    with open(os.path.join(out_dir, "table_crio_monte_carlo.md"), "w") as f:
        f.write("\n".join(mc_lines) + "\n")

    # ══════════════════════════════════════════════════════════════════════════
    # BLOOM RETROSPECTIVE CASE STUDY
    # ══════════════════════════════════════════════════════════════════════════
    # BLOOM actual: 176B parameters (Luccioni et al., 2023)
    #   Training: ~25 tCO₂e, Energy: 433 MWh
    #   Deployed on Jean Zay (French nuclear grid: α ≈ 0.057 kgCO₂e/kWh)
    # We use empirical emissions directly (not the structural model) for
    # the case study, since known figures are available.
    bloom_C_i = 5_000_000.0    # ~$5M estimated total compute cost
    bloom_V_e = 2_000_000.0    # Estimated annual research value
    bloom_C_f = 500_000.0      # Annual operational/maintenance
    bloom_T = 5
    bloom_r = 0.08

    # Empirical: Training 25 tCO₂e amortised over 5yr = 5 tCO₂e/yr
    # + Annual inference/serving ~20 tCO₂e/yr (estimated from energy reports)
    # With λ=0.02 grid decarbonization
    bloom_base_annual = 25.0  # tCO₂e/yr (training amortised + inference)
    bloom_Em = np.array([bloom_base_annual * np.exp(-LAMBDA_DECAY * t)
                         for t in range(1, bloom_T + 1)])
    bloom_p_c = 65.0  # EU ETS mid-range

    bloom_crio = crio_npv(bloom_V_e, bloom_C_f, bloom_C_i,
                          bloom_p_c, bloom_Em, bloom_T, bloom_r)

    # "Dirty BLOOM" counterfactual: trained on coal-heavy grid (α ≈ 0.60)
    # Scale emissions by grid intensity ratio: 0.60 / 0.057 ≈ 10.53x
    dirty_alpha = 0.60
    clean_alpha = 0.057
    grid_ratio = dirty_alpha / clean_alpha
    dirty_Em = bloom_Em * grid_ratio  # ~263 tCO₂e/yr
    dirty_crio = crio_npv(bloom_V_e, bloom_C_f, bloom_C_i,
                          bloom_p_c, dirty_Em, bloom_T, bloom_r)

    bloom_lines = [
        "## BLOOM Retrospective Case Study\n",
        "### Actual BLOOM (Jean Zay — Nuclear Grid)\n",
        f"- **Capital Investment:** ${bloom_C_i:,.0f}",
        f"- **Grid Carbon Intensity (α₀):** {clean_alpha} kgCO₂e/kWh (French nuclear)",
        f"- **Year-1 Emissions:** {bloom_Em[0]:.1f} tCO₂e",
        f"- **Year-5 Emissions:** {bloom_Em[-1]:.1f} tCO₂e (with λ={LAMBDA_DECAY} decay)",
        f"- **Carbon Price:** ${bloom_p_c}/tCO₂e (EU ETS)",
        f"- **CRIO:** **{bloom_crio:.3f}**\n",
        "### \"Dirty BLOOM\" Counterfactual (Coal Grid)\n",
        f"- **Grid Carbon Intensity (α₀):** {dirty_alpha} kgCO₂e/kWh (coal-heavy)",
        f"- **Grid Intensity Ratio:** {grid_ratio:.1f}× nuclear baseline",
        f"- **Year-1 Emissions:** {dirty_Em[0]:.1f} tCO₂e",
        f"- **Year-5 Emissions:** {dirty_Em[-1]:.1f} tCO₂e",
        f"- **CRIO:** **{dirty_crio:.3f}**\n",
        f"**CRIO Degradation:** {abs(bloom_crio - dirty_crio):.3f} "
        f"({abs(bloom_crio - dirty_crio) / abs(bloom_crio) * 100:.1f}% "
        f"reduction due to grid carbon intensity alone)\n",
    ]

    with open(os.path.join(out_dir, "table_crio_bloom_case.md"), "w") as f:
        f.write("\n".join(bloom_lines) + "\n")
    print(f"  🌸 BLOOM: CRIO = {bloom_crio:.3f} | "
          f"Dirty BLOOM: CRIO = {dirty_crio:.3f}")

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE: MONTE CARLO DISTRIBUTION HISTOGRAM
    # ══════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 6))

    regime_colors = {
        "EU ETS": PAL["primary"],
        "Stern-Stiglitz": PAL["secondary"],
        "Net-Zero": PAL["accent"],
    }

    for regime_name, vals in mc_results.items():
        ax.hist(vals, bins=60, alpha=0.45, color=regime_colors[regime_name],
                edgecolor="white", linewidth=0.3, label=regime_name, density=True)
        ax.axvline(np.median(vals), color=regime_colors[regime_name],
                   ls="--", lw=1.5, alpha=0.8)

    ax.axvline(0, color="red", ls=":", lw=2, label="Break-even (CRIO = 0)")

    # Add BLOOM markers
    ax.axvline(bloom_crio, color="#FFB300", ls="-", lw=2,
               label=f"BLOOM (nuclear): {bloom_crio:.3f}")
    ax.axvline(dirty_crio, color="#D32F2F", ls="-", lw=2,
               label=f"Dirty BLOOM (coal): {dirty_crio:.3f}")

    ax.set_xlabel("CRIO Index (NPV-Discounted Carbon-Return Ratio)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Monte Carlo CRIO Distribution ($10^4$ Simulations)\n"
                 "with Grid Decarbonization (λ=0.02) and Capital Elasticity (ε~U[0.6,0.9])",
                 fontweight="bold", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    save_fig(fig, "fig_crio_monte_carlo.png", out_dir)

    # ── Summary ──────────────────────────────────────────────────────────────
    eu_vals = mc_results["EU ETS"]
    return {
        "det_crio_low": det_results[0][3],
        "det_crio_high": det_results[-1][3],
        "mc_eu_median": float(np.median(eu_vals)),
        "mc_eu_p_negative": float(100 * np.mean(eu_vals < 0)),
        "bloom_crio": bloom_crio,
        "dirty_bloom_crio": dirty_crio,
        "n_simulations": N,
    }
