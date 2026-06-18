"""M10: Network Centrality Moderation (Triple Interaction)."""

import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from scisci.data_loader import get_labels
from scisci.style import PAL, save_fig


def run(df, cfg, out_dir):
    """Test if network centrality moderates the boundary-spanning effect.

    Adds a triple interaction: Gov × Tech × Centrality.

    Returns:
        Dict with triple_interaction_p, triple_interaction_beta.
    """
    os.makedirs(out_dir, exist_ok=True)
    ta = cfg["variables"]["theme_a"]["name"]
    tb = cfg["variables"]["theme_b"]["name"]
    labels = get_labels(cfg)

    df = df.copy()
    df["triple_interaction"] = df[ta] * df[tb] * df["norm_degree_z"]

    ctrl_cols = [c["col"] for c in cfg["variables"].get("controls", [])]
    ivs = [ta, tb, "interaction", "triple_interaction"] + ctrl_cols
    ivs = [v for v in ivs if v in df.columns]
    formula = "log_citations ~ " + " + ".join(ivs)

    ols = smf.ols(formula, data=df).fit()

    triple_beta = float(ols.params.get("triple_interaction", 0))
    triple_p = float(ols.pvalues.get("triple_interaction", 1))

    # Save table
    with open(os.path.join(out_dir, "table_network_moderation.md"), "w") as f:
        f.write("## Network Centrality Moderation (Triple Interaction)\n\n")
        f.write(f"- **Triple Interaction β:** {triple_beta:.4f}\n")
        f.write(f"- **p-value:** {triple_p:.4f}\n")
        f.write(f"- **R² (adj):** {ols.rsquared_adj:.4f}\n\n```\n")
        f.write(ols.summary().as_text())
        f.write("\n```\n")

    # Figure: coefficient plot with triple interaction highlighted
    params = ols.params.drop("Intercept")
    ci = ols.conf_int().drop("Intercept")
    ci.columns = ["lo", "hi"]
    ci["coef"] = params

    labels_ext = {**labels, "triple_interaction": f"{labels.get(ta, ta)} × {labels.get(tb, tb)} × Centrality"}
    ci.index = [labels_ext.get(x, x) for x in ci.index]

    fig, ax = plt.subplots(figsize=(9, 6))
    yp = np.arange(len(ci))
    err = [ci["coef"] - ci["lo"], ci["hi"] - ci["coef"]]
    colors = [PAL["secondary"] if "Centrality" in idx else PAL["primary"] for idx in ci.index]
    ax.errorbar(ci["coef"], yp, xerr=err, fmt="o", color=PAL["primary"],
                ecolor=PAL["tertiary"], capsize=5, ms=8)
    ax.axvline(0, color="red", ls="--", alpha=0.5)
    ax.set_yticks(yp)
    ax.set_yticklabels(ci.index)
    ax.set_xlabel("Coefficient")
    ax.set_title(f"Network Moderation: Triple Interaction\n"
                 f"(β={triple_beta:.4f}, p={triple_p:.4f})",
                 fontweight="bold", fontsize=12)

    # Annotate significance
    for i in range(len(ci)):
        p = ols.pvalues.iloc[i + 1]
        sig = "n.s." if p > 0.05 else ("***" if p < 0.001 else ("**" if p < 0.01 else "*"))
        ax.annotate(sig, (ci["coef"].iloc[i], i), xytext=(0, 10),
                    textcoords="offset points", ha="center",
                    color="red" if sig == "n.s." else "black", fontweight="bold")

    fig.tight_layout()
    save_fig(fig, "fig_network_moderation.png", out_dir)

    return {"triple_interaction_beta": triple_beta, "triple_interaction_p": triple_p, "r2_adj": float(ols.rsquared_adj)}
