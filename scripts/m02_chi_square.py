"""M02: Chi-Square Silo Test (H1)."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

from scisci.style import PAL, save_fig


def run(df: pd.DataFrame, cfg: dict, out_dir: str) -> dict:
    """Test if governance papers are non-uniformly distributed across clusters.

    Returns:
        Dict with chi2, df, p values.
    """
    os.makedirs(out_dir, exist_ok=True)
    h1 = cfg["hypotheses"]["h1_silo"]
    theme_col = cfg["variables"][h1["theme"]]["name"]
    theme_label = cfg["variables"][h1["theme"]]["label"]

    gov_dist = df[df[theme_col] == 1]["cluster_id"].value_counts()
    total = df["cluster_id"].value_counts()
    rate = gov_dist.sum() / total.sum()

    labels, obs, exp = [], [], []
    for c in sorted(total.index):
        lbl = df.loc[df["cluster_id"] == c, "_cname"].iloc[0] if "_cname" in df.columns else f"C{c}"
        labels.append(lbl)
        obs.append(gov_dist.get(c, 0))
        exp.append(total.get(c, 0) * rate)

    oa, ea = np.array(obs), np.array(exp)
    # Avoid division by zero
    ea_safe = np.where(ea == 0, 1e-10, ea)
    chi2 = float(np.sum((oa - ea_safe) ** 2 / ea_safe))
    dof = max(len(labels) - 1, 1)
    p_val = 1 - stats.chi2.cdf(chi2, dof)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, obs, w, label=f"Observed {theme_label} Papers", color=PAL["secondary"])
    ax.bar(x + w / 2, exp, w, label="Expected (uniform rate)", color=PAL["primary"], alpha=0.6)
    ax.set_ylabel("Number of Papers")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend()
    ax.set_title(
        f"H1: {theme_label}-Paper Distribution Across Clusters\n"
        f"χ²={chi2:.2f}, df={dof}, p={p_val:.4f}",
        fontweight="bold",
        fontsize=12,
    )
    fig.tight_layout()
    save_fig(fig, "fig_chi_square_silo.png", out_dir)

    # Save data table
    tbl_lines = [
        f"## Chi-Square Silo Test (H1: {theme_label})\n",
        f"- **χ²:** {chi2:.2f}",
        f"- **df:** {dof}",
        f"- **p-value:** {p_val:.4f}",
        f"- **Significant (α=0.05):** {'✓ Yes' if p_val < 0.05 else 'No'}\n",
        "| Cluster | Observed | Expected | Residual |",
        "|:---|---:|---:|---:|",
    ]
    for lbl, o, e in zip(labels, obs, exp):
        residual = (o - e) / max(e**0.5, 1e-10)
        tbl_lines.append(f"| {lbl} | {o} | {e:.1f} | {residual:+.2f} |")
    with open(os.path.join(out_dir, "table_chi_square_silo.md"), "w") as f:
        f.write("\n".join(tbl_lines) + "\n")

    return {"chi2": chi2, "df": dof, "p": p_val}
