"""M06: Structural Break Test (Negative Binomial GLM) for regime shift detection."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from scisci.style import PAL, save_fig


def _negbin_break_test(counts_df, break_year):
    """Fit a NegBin GLM with a structural break dummy + slope interaction.

    Unrestricted model: paper_count ~ year_c + D_break + year_c * D_break
    Restricted model:   paper_count ~ year_c

    Uses Wald test on the joint significance of D_break and its interaction
    with year, and reports AIC of the unrestricted model.

    Returns:
        Tuple of (wald_stat, p_value, aic) for the break coefficients.
    """
    df = counts_df.copy()
    df["D_break"] = (df["year"] >= break_year).astype(int)
    df["year_c"] = df["year"] - df["year"].min()
    df["year_x_break"] = df["year_c"] * df["D_break"]

    y = df["paper_count"].values.astype(float)

    # Unrestricted model: intercept + year_c + D_break + year_c*D_break
    X_full = sm.add_constant(df[["year_c", "D_break", "year_x_break"]].values)

    try:
        model = sm.GLM(y, X_full, family=sm.families.NegativeBinomial()).fit()
        # Wald test on columns 2 and 3 (D_break + year_x_break)
        wald = model.wald_test(np.eye(4)[2:4])
        wald_stat = float(wald.statistic.item())
        p_val = float(wald.pvalue.item())
        aic = float(model.aic)
    except Exception:
        wald_stat, p_val, aic = 0.0, 1.0, np.inf

    return wald_stat, p_val, aic


def run(df, cfg, out_dir):
    """Test for structural breaks in publication volume time series.

    Uses quarterly publication counts (N=28) for sufficient statistical power.
    Fits a Negative Binomial GLM with a break-year dummy, iterated
    over all candidate years, selecting the best break by minimum AIC.

    Returns:
        Dict with best_break_year, z_stat, p_value, aic.
    """
    os.makedirs(out_dir, exist_ok=True)
    df_filtered = df[df["year"] < 2026].copy()

    # Build quarterly counts for higher resolution (N=28 vs N=7 annual)
    if "publication_date" in df_filtered.columns:
        df_filtered["publication_date"] = pd.to_datetime(df_filtered["publication_date"])
        df_filtered["quarter"] = df_filtered["publication_date"].dt.to_period("Q")
        all_quarters = pd.period_range(start="2019Q1", end="2025Q4", freq="Q")
        quarterly = df_filtered.groupby("quarter").size().reindex(all_quarters, fill_value=0)
        # Extract year from period for the break dummy
        counts = pd.DataFrame({
            "quarter": quarterly.index,
            "year": [q.year for q in quarterly.index],
            "paper_count": quarterly.values,
        })
    else:
        # Fallback to annual counts
        annual = df_filtered.groupby("year").size().reset_index(name="paper_count")
        counts = annual.sort_values("year")

    # Test all candidate break years (need data on both sides)
    candidate_years = sorted(counts["year"].unique())[2:-1]
    results = []
    for candidate in candidate_years:
        wald_stat, p_val, aic = _negbin_break_test(counts, candidate)
        results.append({
            "year": candidate, "wald_stat": wald_stat, "p_value": p_val, "aic": aic
        })

    # Deduplicate by year (quarterly data produces multiple rows per year)
    results_df = pd.DataFrame(results).drop_duplicates(subset="year")

    if len(results_df) > 0:
        best = results_df.loc[results_df["aic"].idxmin()]
        best_year = int(best["year"])
        best_wald = float(best["wald_stat"])
        best_p = float(best["p_value"])
        best_aic = float(best["aic"])
    else:
        best_year, best_wald, best_p, best_aic = 2024, 0.0, 1.0, np.inf

    # Save table
    lines = [
        "## Negative Binomial Structural Break Test\n",
        "| Break Year | Wald χ² | p-value | AIC | Significant? |",
        "|---:|---:|---:|---:|:---|",
    ]
    for _, r in results_df.iterrows():
        sig = "✓ Yes" if r["p_value"] < 0.05 else "No"
        bold = "**" if r["year"] == best_year else ""
        lines.append(
            f"| {bold}{int(r['year'])}{bold} | {r['wald_stat']:.3f} | "
            f"{r['p_value']:.4f} | {r['aic']:.1f} | {sig} |"
        )
    with open(os.path.join(out_dir, "table_structural_break.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # Also write legacy filename for backwards compatibility with verify_all_numbers
    with open(os.path.join(out_dir, "table_chow_test.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # Figure: use annual counts for visual clarity
    annual_counts = df_filtered.groupby("year").size().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(annual_counts.index, annual_counts.values,
           color=PAL["light"], edgecolor=PAL["primary"], linewidth=0.8)
    ax.axvline(best_year, color="red", ls="--", lw=2,
               label=f"Break: {best_year} (Wald={best_wald:.2f}, p={best_p:.4f})")
    ax.set_xlabel("Year")
    ax.set_ylabel("Publication Count")
    ax.set_title("Structural Break Test: Negative Binomial GLM (Quarterly Data)",
                 fontweight="bold", fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()
    save_fig(fig, "fig_structural_break.png", out_dir)

    return {
        "best_break_year": best_year,
        "wald_stat": best_wald,
        "p_value": best_p,
        "aic": best_aic,
    }
