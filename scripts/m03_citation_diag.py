"""M03: Citation Distribution Diagnostic (Histogram + Q-Q)."""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.formula.api as smf

from scisci.data_loader import get_formula
from scisci.style import PAL, save_fig


def run(df, cfg, out_dir):
    """Produce 2-panel citation diagnostic: histogram + Q-Q plot.

    Returns:
        Dict with lognorm_sigma, qq_r2, shapiro_w.
    """
    os.makedirs(out_dir, exist_ok=True)
    formula = get_formula(cfg, "log_citations")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 6))

    cit = df["citations"].values
    cit_nz = cit[cit > 0]
    mx = max(int(np.percentile(cit, 99)), 1)
    a1.hist(cit[cit <= mx], bins=min(60, mx), density=True, color=PAL["light"],
            edgecolor=PAL["primary"], linewidth=0.5, alpha=0.8, label="Observed")
    if len(cit_nz) > 2:
        shape, loc, scale = stats.lognorm.fit(cit_nz, floc=0)
        xf = np.linspace(1, mx, 500)
        frac = len(cit_nz) / max(len(cit), 1)
        a1.plot(xf, stats.lognorm.pdf(xf, shape, loc, scale) * frac,
                color=PAL["secondary"], lw=2, label=f"Lognormal (σ={shape:.2f})")
    else:
        shape = 0.0
    nz = int((cit == 0).sum())
    pz = nz / max(len(cit), 1) * 100
    a1.annotate(f"Zero-cit: {nz} ({pz:.1f}%)", xy=(0.45, 0.65), xycoords="axes fraction",
                fontsize=9, fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PAL["tertiary"], alpha=0.9))
    a1.set_xlabel("Raw Citation Count")
    a1.set_ylabel("Density")
    a1.set_title("(a) Citation Distribution", fontweight="bold", fontsize=12)
    a1.legend(loc="upper right", fontsize=9)

    try:
        mdl = smf.ols(formula, data=df).fit()
        res = mdl.resid
        (osm, osr), (sl, ic, rv) = stats.probplot(res, dist="norm")
        a2.scatter(osm, osr, s=8, alpha=0.4, color=PAL["primary"], edgecolors="none")
        xl = np.array([osm.min(), osm.max()])
        a2.plot(xl, sl * xl + ic, color=PAL["secondary"], lw=2, label=f"R²={rv**2:.4f}")
        nsw = min(len(res), 5000)
        sw, sp = stats.shapiro(res[:nsw])
        a2.annotate(f"Shapiro-Wilk: W={sw:.4f}\nn={nsw}, p={sp:.4f}",
                    xy=(0.05, 0.85), xycoords="axes fraction", fontsize=8, color=PAL["tertiary"],
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PAL["tertiary"], alpha=0.9))
        qq_r2 = rv**2
    except Exception:
        qq_r2 = 0.0
        sw = 0.0

    a2.set_xlabel("Theoretical Quantiles")
    a2.set_ylabel("Residual Quantiles")
    a2.set_title("(b) OLS Residual Q-Q Plot", fontweight="bold", fontsize=12)
    a2.legend(loc="lower right", fontsize=9)
    fig.suptitle("Regression Diagnostic: Validating log(citations+1) Transform",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "fig_citation_diagnostic.png", out_dir)

    return {"lognorm_sigma": float(shape), "qq_r2": float(qq_r2), "shapiro_w": float(sw)}
