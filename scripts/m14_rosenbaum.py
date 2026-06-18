"""M14: Rosenbaum Bounds Sensitivity Analysis for PSM.

Computes Γ* — the minimum hidden bias (Rosenbaum Gamma) at which the
PSM treatment effect becomes statistically insignificant. This is the
gold-standard sensitivity analysis for observational causal inference
(Rosenbaum, 2002).

Methodology:
    For each Γ from 1.0 to 3.0 (step 0.1), inflate the treatment assignment
    odds ratio bounds, then compute the upper-bound p-value from a Wilcoxon
    signed-rank test on matched pair outcome differences. Γ* is the smallest
    Γ where p_upper > 0.05.

Interpretation:
    - Γ* > 2.0 → Robust: an unobserved confounder would need to more than
      double treatment odds to nullify the finding.
    - 1.5 < Γ* < 2.0 → Moderate sensitivity.
    - Γ* < 1.5 → Fragile: small hidden biases could invalidate the result.

References:
    Rosenbaum, P. R. (2002). Observational Studies (2nd ed.). Springer.
    DiPrete, T. A., & Gangl, M. (2004). Assessing Bias in the Estimation
    of Causal Effects. Sociological Methodology, 34(1), 271–310.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from scisci.style import PAL, save_fig


def _wilcoxon_upper_p(differences: np.ndarray, gamma: float) -> float:
    """Compute Rosenbaum upper-bound p-value under hidden bias Γ.

    Under the null hypothesis of no treatment effect, if an unobserved
    confounder could inflate treatment odds by factor Γ, the Wilcoxon
    signed-rank statistic is adjusted by shifting the mean of the test
    statistic distribution.

    Args:
        differences: Outcome differences (treated - control) for matched pairs.
        gamma: Rosenbaum sensitivity parameter (Γ >= 1.0).

    Returns:
        Upper-bound p-value (one-sided).
    """
    n = len(differences)
    if n < 2:
        return 1.0

    # Compute Wilcoxon signed-rank statistic
    abs_diff = np.abs(differences)
    ranks = stats.rankdata(abs_diff)
    # Sum of ranks for positive differences
    t_plus = np.sum(ranks[differences > 0])

    # Under Γ, the upper bound on E[T+] and Var[T+]
    # Each pair i has probability Γ/(1+Γ) of being assigned to treatment
    p_upper = gamma / (1.0 + gamma)

    # Expected value and variance under Γ-inflated assignment
    n_ranks = len(ranks)
    e_t = p_upper * n_ranks * (n_ranks + 1) / 2.0
    var_t = p_upper * (1 - p_upper) * n_ranks * (n_ranks + 1) * (2 * n_ranks + 1) / 6.0

    if var_t <= 0:
        return 1.0

    # Standardized test statistic
    z = (t_plus - e_t) / np.sqrt(var_t)

    # One-sided p-value (upper tail: does treatment help?)
    p_val = 1.0 - stats.norm.cdf(z)
    return float(p_val)


def run(df: pd.DataFrame, cfg_or_out_dir=None, out_dir: str = None) -> dict:
    """Run Rosenbaum bounds sensitivity analysis.

    Supports two calling conventions:
        1. Pipeline: run(df, cfg, out_dir)  — called by run_all.py
        2. Direct:   run(matched_df, out_dir) — called by tests

    Args:
        df: DataFrame (full corpus or pre-matched sample).
        cfg_or_out_dir: Config dict (pipeline) or output dir string (direct).
        out_dir: Output directory (pipeline mode only).

    Returns:
        Dict with gamma_star, n_pairs, and the full sensitivity table.
    """
    # Resolve calling convention
    if isinstance(cfg_or_out_dir, str) and out_dir is None:
        # Direct call: run(matched_df, out_dir)
        out_dir = cfg_or_out_dir
        matched_df = df
    elif isinstance(cfg_or_out_dir, dict):
        # Pipeline call: run(df, cfg, out_dir)
        # Replicate PSM matching from m07 to get the matched sample
        cfg = cfg_or_out_dir
        df = df.copy()
        # Create treatment indicator (same as m07_psm.py)
        if "interaction" in df.columns:
            df["is_boundary"] = df["interaction"].astype(int)
        # Use m07's covariate list
        covariates = ["paper_age", "author_count", "degree", "author_prestige", "is_review"]
        covariates = [c for c in covariates if c in df.columns]
        if "is_boundary" in df.columns and covariates and df["is_boundary"].sum() >= 3:
            from scisci.m07_psm import _match
            _, matched_df = _match(df, "is_boundary", covariates)
            if len(matched_df) == 0:
                matched_df = df
            print(f"  ℹ️  Rosenbaum: Using PSM matched sample (N={len(matched_df)})")
        else:
            matched_df = df
    else:
        matched_df = df
        if out_dir is None:
            out_dir = "."

    os.makedirs(out_dir, exist_ok=True)

    treatment_col = "is_boundary"
    outcome_col = "log_citations"

    # Ensure required columns exist
    if treatment_col not in matched_df.columns:
        matched_df = matched_df.copy()
        if "interaction" in matched_df.columns:
            matched_df[treatment_col] = matched_df["interaction"].astype(int)
        else:
            matched_df[treatment_col] = 0

    if outcome_col not in matched_df.columns:
        matched_df = matched_df.copy()
        matched_df[outcome_col] = np.log1p(matched_df.get("citations", pd.Series([0])))

    treated = matched_df[matched_df[treatment_col] == 1].reset_index(drop=True)
    control = matched_df[matched_df[treatment_col] == 0].reset_index(drop=True)

    # Pair by index (PSM matched order)
    n_pairs = min(len(treated), len(control))

    if n_pairs < 2:
        # Insufficient data — report gracefully
        _write_fallback(out_dir)
        return {"gamma_star": 1.0, "n_pairs": n_pairs, "robust": False}

    differences = treated[outcome_col].values[:n_pairs] - control[outcome_col].values[:n_pairs]

    # Iterate Γ from 1.0 to 3.0
    gammas = np.arange(1.0, 3.05, 0.1)
    results = []
    gamma_star = gammas[-1]  # Default: robust up to max Γ tested
    found_star = False

    for g in gammas:
        p_upper = _wilcoxon_upper_p(differences, g)
        significant = p_upper < 0.05
        results.append({"gamma": round(g, 1), "p_upper": p_upper, "significant": significant})

        if not significant and not found_star:
            gamma_star = round(g, 1)
            found_star = True

    # If never crossed 0.05, gamma_star = max tested
    if not found_star:
        gamma_star = float(gammas[-1])

    # Robustness assessment
    if gamma_star >= 2.0:
        robustness = "ROBUST"
        assessment = "An unobserved confounder would need to more than double treatment odds."
    elif gamma_star >= 1.5:
        robustness = "MODERATE"
        assessment = "Moderate sensitivity to unobserved confounders."
    else:
        robustness = "FRAGILE"
        assessment = "Small hidden biases could invalidate the treatment effect."

    # ── Save table ────────────────────────────────────────────────────────
    lines = [
        "## Rosenbaum Bounds Sensitivity Analysis\n",
        f"- **N (matched pairs):** {n_pairs}",
        f"- **Γ\\* (critical threshold):** {gamma_star:.1f}",
        f"- **Robustness:** {robustness}",
        f"- **Assessment:** {assessment}\n",
        "| Γ | p_upper | Significant (α=0.05)? |",
        "|---:|---:|:---|",
    ]
    for r in results:
        sig = "✓ Yes" if r["significant"] else "No"
        marker = " **← Γ\\***" if r["gamma"] == gamma_star and found_star else ""
        lines.append(f"| {r['gamma']:.1f} | {r['p_upper']:.4f} | {sig}{marker} |")

    with open(os.path.join(out_dir, "table_rosenbaum_bounds.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ── Save figure ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    g_vals = [r["gamma"] for r in results]
    p_vals = [r["p_upper"] for r in results]

    ax.plot(g_vals, p_vals, "o-", color=PAL["primary"], linewidth=2, markersize=6)
    ax.axhline(0.05, color="red", ls="--", lw=1.5, label="α = 0.05 threshold")
    ax.axvline(gamma_star, color=PAL["accent"], ls=":", lw=2,
               label=f"Γ* = {gamma_star:.1f} ({robustness})")

    ax.fill_between(g_vals, 0, p_vals, alpha=0.1, color=PAL["primary"])
    ax.set_xlabel("Rosenbaum Γ (Hidden Bias Factor)", fontsize=12)
    ax.set_ylabel("Upper Bound p-value", fontsize=12)
    ax.set_title(
        f"Rosenbaum Bounds Sensitivity (N={n_pairs} pairs, Γ*={gamma_star:.1f})",
        fontweight="bold", fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.set_ylim(-0.02, max(p_vals) * 1.1 + 0.05)
    fig.tight_layout()
    save_fig(fig, "fig_rosenbaum_sensitivity.png", out_dir)

    return {
        "gamma_star": gamma_star,
        "n_pairs": n_pairs,
        "robust": robustness,
        "assessment": assessment,
        "sensitivity_table": results,
    }


def _write_fallback(out_dir: str) -> None:
    """Write fallback files when matched sample is too small."""
    with open(os.path.join(out_dir, "table_rosenbaum_bounds.md"), "w") as f:
        f.write("## Rosenbaum Bounds\n\nInsufficient matched pairs for sensitivity analysis.\n")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", fontsize=14)
    ax.set_title("Rosenbaum Bounds", fontweight="bold")
    save_fig(fig, "fig_rosenbaum_sensitivity.png", out_dir)
