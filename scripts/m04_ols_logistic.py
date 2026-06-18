"""M04: OLS + Logistic Regression (Legacy H2+H3)."""

import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from scisci.data_loader import get_labels, get_formula
from scisci.style import PAL, save_fig


def run(df, cfg, out_dir):
    """Run OLS (H2) and Logistic (H3) regressions with interaction term.

    Returns:
        Dict with ols_r2, logit_pseudo_r2, interaction_p.
    """
    os.makedirs(out_dir, exist_ok=True)
    labels = get_labels(cfg)
    formula_ols = get_formula(cfg, "log_citations")
    formula_logit = get_formula(cfg, "is_top_10")

    ols = smf.ols(formula_ols, data=df).fit(
        cov_type='cluster', cov_kwds={'groups': df['cluster_id']})
    logit = smf.logit(formula_logit, data=df).fit(
        disp=0, cov_type='cluster', cov_kwds={'groups': df['cluster_id']})

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 6))

    # Panel (a): OLS coefficients
    oc = ols.conf_int()
    oc["coef"] = ols.params
    oc.columns = ["lo", "hi", "coef"]
    oc = oc.drop("Intercept")
    oc.index = [labels.get(x, x) for x in oc.index]
    err = [oc["coef"] - oc["lo"], oc["hi"] - oc["coef"]]
    yp = np.arange(len(oc))
    a1.errorbar(oc["coef"], yp, xerr=err, fmt="o", color=PAL["primary"],
                ecolor=PAL["tertiary"], capsize=5, ms=8)
    a1.axvline(0, color="red", ls="--", alpha=0.5)
    a1.set_yticks(yp)
    a1.set_yticklabels(oc.index)
    a1.set_xlabel("Coefficient (Impact on log(citations))")
    a1.set_title("(a) OLS Regression Coefficients", fontweight="bold", fontsize=12)
    for i in range(len(oc)):
        p = ols.pvalues.iloc[i + 1]
        sig = "n.s." if p > 0.05 else ("***" if p < 0.001 else ("**" if p < 0.01 else "*"))
        a1.annotate(sig, (oc["coef"].iloc[i], i), xytext=(0, 10),
                    textcoords="offset points", ha="center",
                    color="red" if sig == "n.s." else "black", fontweight="bold")

    # Panel (b): Odds ratios
    lc = np.exp(logit.conf_int())
    lc["or"] = np.exp(logit.params)
    lc.columns = ["lo", "hi", "or"]
    lc = lc.drop("Intercept")
    lc.index = [labels.get(x, x) for x in lc.index]
    elo = lc["or"] - lc["lo"]
    ehi = lc["hi"] - lc["or"]
    yp2 = np.arange(len(lc))
    a2.errorbar(lc["or"], yp2, xerr=[elo, ehi], fmt="D", color=PAL["secondary"],
                ecolor=PAL["tertiary"], capsize=5, ms=8)
    a2.axvline(1, color="red", ls="--", alpha=0.5)
    a2.set_yticks(yp2)
    a2.set_yticklabels(lc.index)
    a2.set_xlabel("Odds Ratio (Likelihood of Breakthrough Impact)")
    a2.set_title("(b) Logistic Regression Odds Ratios", fontweight="bold", fontsize=12)
    for i, (_, row) in enumerate(lc.iterrows()):
        p = logit.pvalues.iloc[i + 1]
        sig = "n.s." if p > 0.05 else ("***" if p < 0.001 else ("**" if p < 0.01 else "*"))
        col = "red" if sig == "n.s." else "black"
        a2.annotate(f"OR={row['or']:.2f} {sig}", (row["or"], i), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=9, fontweight="bold", color=col)

    fig.suptitle("Econometric Testing: Integration Penalty (H2) and Breakthrough Predictors (H3)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "fig_interaction_penalty.png", out_dir)

    # Save regression tables
    with open(os.path.join(out_dir, "table_regression_results.md"), "w") as f:
        f.write("## OLS Regression: log(citations+1)\n\n```\n")
        f.write(ols.summary().as_text())
        f.write("\n```\n\n## Logistic Regression: Breakthrough Impact (Top 10%)\n\n```\n")
        f.write(logit.summary().as_text())
        f.write("\n```\n")

    return {
        "ols_r2": float(ols.rsquared),
        "ols_r2_adj": float(ols.rsquared_adj),
        "interaction_p": float(ols.pvalues.get("interaction", 1)),
        "logit_pseudo_r2": float(logit.prsquared),
    }
