"""M07: Propensity Score Matching (PSM) for causal inference."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import statsmodels.formula.api as smf

from scisci.style import PAL, save_fig


def _match(df, treatment_col, covariates, caliper=0.25):
    """Nearest-neighbor PSM matching with caliper."""
    # [Jun 17, 2026] WARNING: fillna(0) treats missing 'paper_age' as brand new (age 0), 
    # and missing 'author_prestige' as lowest prestige. In the current CRIO dataset (946 papers)
    # sourced from Scopus and enhanced by OpenAlex, missingness was verified at 0%, so this 
    # is safe here. For future datasets with high missingness, replace with median-imputation 
    # or dropna() to avoid systemic bias.
    X = df[covariates].fillna(0).values
    y = df[treatment_col].values

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X, y)
    ps = lr.predict_proba(X)[:, 1]
    df = df.copy()
    df["propensity_score"] = ps

    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]

    if len(control) == 0 or len(treated) == 0:
        return df, pd.DataFrame()

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(control[["propensity_score"]].values)
    distances, indices = nn.kneighbors(treated[["propensity_score"]].values)

    matched_control_idx = []
    matched_treated_idx = []
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        if dist[0] <= caliper:
            matched_treated_idx.append(treated.index[i])
            matched_control_idx.append(control.index[idx[0]])

    matched = pd.concat([
        df.loc[matched_treated_idx],
        df.loc[matched_control_idx],
    ])
    return df, matched


def run(df, cfg, out_dir):
    """Run PSM to estimate the Average Treatment Effect on the Treated (ATT).

    Treatment = "Boundary Spanner" (interaction == 1).

    Returns:
        Dict with att, att_se, n_matched.
    """
    os.makedirs(out_dir, exist_ok=True)
    ta = cfg["variables"]["theme_a"]["name"]
    tb = cfg["variables"]["theme_b"]["name"]

    # Treatment: boundary spanner (papers that bridge both themes)
    df = df.copy()
    df["is_boundary"] = df["interaction"].astype(int)

    covariates = ["paper_age", "author_count", "degree", "author_prestige", "is_review"]
    covariates = [c for c in covariates if c in df.columns]

    if not covariates or df["is_boundary"].sum() < 3:
        # Not enough boundary spanners; return empty results
        with open(os.path.join(out_dir, "table_psm_results.md"), "w") as f:
            f.write("## PSM Results\n\nInsufficient boundary-spanning papers for matching.\n")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", fontsize=14)
        ax.set_title("PSM Balance Check", fontweight="bold")
        save_fig(fig, "fig_psm_balance.png", out_dir)
        return {"att": 0.0, "att_se": 0.0, "n_matched": 0}

    df_full, matched = _match(df, "is_boundary", covariates)

    if len(matched) < 4:
        with open(os.path.join(out_dir, "table_psm_results.md"), "w") as f:
            f.write("## PSM Results\n\nMatched sample too small for regression.\n")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Matched sample too small", ha="center", va="center", fontsize=14)
        save_fig(fig, "fig_psm_balance.png", out_dir)
        return {"att": 0.0, "att_se": 0.0, "n_matched": len(matched)}

    # ATT via OLS on matched sample
    formula = "log_citations ~ is_boundary + " + " + ".join(covariates)
    ols_matched = smf.ols(formula, data=matched).fit()
    att = float(ols_matched.params.get("is_boundary", 0))
    att_se = float(ols_matched.bse.get("is_boundary", 0))
    att_p = float(ols_matched.pvalues.get("is_boundary", 1))

    # Save table
    with open(os.path.join(out_dir, "table_psm_results.md"), "w") as f:
        f.write("## Propensity Score Matching: ATT of Boundary Spanning\n\n")
        f.write(f"- **ATT (β):** {att:.4f}\n")
        f.write(f"- **Std. Error:** {att_se:.4f}\n")
        f.write(f"- **p-value:** {att_p:.4f}\n")
        f.write(f"- **N (matched):** {len(matched)}\n\n```\n")
        f.write(ols_matched.summary().as_text())
        f.write("\n```\n")

    # Balance plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, cov in enumerate(covariates):
        treated_mean = matched[matched["is_boundary"] == 1][cov].mean()
        control_mean = matched[matched["is_boundary"] == 0][cov].mean()
        smd = (treated_mean - control_mean) / max(matched[cov].std(), 1e-10)
        color = PAL["accent"] if abs(smd) < 0.1 else PAL["secondary"]
        ax.barh(i, smd, color=color, edgecolor="white", height=0.6)
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(-0.1, color="red", ls="--", alpha=0.5)
    ax.axvline(0.1, color="red", ls="--", alpha=0.5, label="±0.1 SMD threshold")
    ax.set_yticks(range(len(covariates)))
    ax.set_yticklabels(covariates)
    ax.set_xlabel("Standardized Mean Difference (SMD)")
    ax.set_title(f"PSM Covariate Balance (N={len(matched)})", fontweight="bold", fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_fig(fig, "fig_psm_balance.png", out_dir)

    return {"att": att, "att_se": att_se, "att_p": att_p, "n_matched": len(matched)}
