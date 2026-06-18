"""M05: Negative Binomial GLM (Robustness check for over-dispersed citations)."""

import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from scisci.data_loader import get_labels, get_model_vars
from scisci.style import PAL, save_fig


def run(df, cfg, out_dir):
    """Run Negative Binomial regression on raw citation counts.

    Returns:
        Dict with nb_pseudo_r2, nb_aic, interaction_p.
    """
    os.makedirs(out_dir, exist_ok=True)
    labels = get_labels(cfg)
    ta = cfg["variables"]["theme_a"]["name"]
    tb = cfg["variables"]["theme_b"]["name"]
    ctrl_cols = [c["col"] for c in cfg["variables"].get("controls", [])]
    ivs = [ta, tb, "interaction"] + ctrl_cols
    ivs = [v for v in ivs if v in df.columns]

    X = df[ivs].copy()
    X.insert(0, "const", 1)
    y = df["citations"].values.astype(float)

    try:
        nb_model = sm.NegativeBinomial(y, X, loglike_method="nb2").fit(
            disp=0, maxiter=200,
            cov_type='cluster', cov_kwds={'groups': df['cluster_id']})
        pseudo_r2 = float(1 - nb_model.llf / nb_model.llnull)
        aic = float(nb_model.aic)
        interaction_p = float(nb_model.pvalues.get("interaction", nb_model.pvalues.iloc[-1]))
    except Exception:
        # Fallback to GLM NegBin
        nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit(
            cov_type='cluster', cov_kwds={'groups': df['cluster_id']})
        pseudo_r2 = float(1 - nb_model.deviance / nb_model.null_deviance)
        aic = float(nb_model.aic)
        interaction_p = float(nb_model.pvalues.get("interaction", nb_model.pvalues.iloc[-1]))

    # Save table
    with open(os.path.join(out_dir, "table_negbin_results.md"), "w") as f:
        f.write("## Negative Binomial Regression: Citations (Raw Count)\n\n```\n")
        f.write(nb_model.summary().as_text())
        f.write("\n```\n")

    # Figure: coefficient plot
    params = nb_model.params.drop("const") if "const" in nb_model.params.index else nb_model.params
    ci = nb_model.conf_int()
    if "const" in ci.index:
        ci = ci.drop("const")
    ci.columns = ["lo", "hi"]
    ci["coef"] = params

    fig, ax = plt.subplots(figsize=(8, 5))
    yp = np.arange(len(ci))
    err = [ci["coef"] - ci["lo"], ci["hi"] - ci["coef"]]
    ax.errorbar(ci["coef"], yp, xerr=err, fmt="s", color=PAL["accent"],
                ecolor=PAL["tertiary"], capsize=5, ms=8)
    ax.axvline(0, color="red", ls="--", alpha=0.5)
    ax.set_yticks(yp)
    ax.set_yticklabels([labels.get(x, x) for x in ci.index])
    ax.set_xlabel("NegBin Coefficient")
    ax.set_title("Negative Binomial Regression Coefficients\n(Robustness Check: Over-dispersed Count Data)",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    save_fig(fig, "fig_negbin_coefficients.png", out_dir)

    return {"nb_pseudo_r2": pseudo_r2, "nb_aic": aic, "interaction_p": interaction_p}
