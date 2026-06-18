"""M09: Triple Helix Institutional Analysis (University / Industry / Government)."""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scisci.style import PAL, save_fig

# Heuristic patterns for institutional classification
_INDUSTRY_PATTERNS = re.compile(
    r"\b(inc\.?|corp\.?|ltd\.?|llc|gmbh|company|google|microsoft|ibm|amazon|meta|"
    r"nvidia|apple|samsung|huawei|tencent|alibaba|intel|oracle|salesforce|"
    r"siemens|bosch|shell|bp|exxon|tesla|openai)\b",
    re.IGNORECASE,
)
_GOV_PATTERNS = re.compile(
    r"\b(ministry|government|national lab|national institute|department of|"
    r"agency|commission|bureau|federal|nasa|nist|doe|epa|ipcc|undp|unep|"
    r"world bank|european commission)\b",
    re.IGNORECASE,
)


def _classify_affiliation(aff_str):
    """Classify an affiliation string into Academic, Industry, or Government."""
    if pd.isna(aff_str) or not str(aff_str).strip():
        return "Unknown"
    aff = str(aff_str).lower()
    has_industry = bool(_INDUSTRY_PATTERNS.search(aff))
    has_gov = bool(_GOV_PATTERNS.search(aff))
    if has_industry and has_gov:
        return "Industry-Gov"
    elif has_industry:
        return "Industry"
    elif has_gov:
        return "Government"
    else:
        return "Academic"


def run(df, cfg, out_dir):
    """Classify papers by institutional sector and compare citation impact.

    Returns:
        Dict with academic_pct, industry_pct, gov_pct, and citation means.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = df.copy()
    df["sector"] = df["affiliations"].apply(_classify_affiliation)

    sector_counts = df["sector"].value_counts()
    total = len(df)
    pcts = {s: round(c / total * 100, 1) for s, c in sector_counts.items()}

    # Citation comparison by sector
    sector_stats = df.groupby("sector")["citations"].agg(["count", "mean", "median", "std"])
    sector_stats = sector_stats.sort_values("mean", ascending=False)

    # Save table
    lines = ["| Sector | N | % | Mean Citations | Median | SD |", "|:---|---:|---:|---:|---:|---:|"]
    for sector, row in sector_stats.iterrows():
        pct = pcts.get(sector, 0)
        lines.append(f"| {sector} | {int(row['count'])} | {pct}% | {row['mean']:.2f} | {row['median']:.1f} | {row['std']:.2f} |")
    with open(os.path.join(out_dir, "table_institutional_impact.md"), "w") as f:
        f.write("## Triple Helix Institutional Analysis\n\n" + "\n".join(lines) + "\n")

    # Figure: grouped bar chart
    sectors = sector_stats.index.tolist()
    means = sector_stats["mean"].values
    counts = sector_stats["count"].values

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 6))

    colors = [PAL["primary"], PAL["secondary"], PAL["accent"], PAL["tertiary"], PAL["light"]]
    a1.bar(range(len(sectors)), counts, color=colors[:len(sectors)], edgecolor="white")
    a1.set_xticks(range(len(sectors)))
    a1.set_xticklabels(sectors, rotation=15, ha="right")
    a1.set_ylabel("Number of Papers")
    a1.set_title("(a) Papers by Institutional Sector", fontweight="bold", fontsize=11)

    a2.bar(range(len(sectors)), means, color=colors[:len(sectors)], edgecolor="white")
    a2.set_xticks(range(len(sectors)))
    a2.set_xticklabels(sectors, rotation=15, ha="right")
    a2.set_ylabel("Mean Citations")
    a2.set_title("(b) Citation Impact by Sector", fontweight="bold", fontsize=11)

    fig.suptitle("Triple Helix: Institutional Distribution of Green AI Research",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "fig_triple_helix.png", out_dir)

    return {
        "academic_pct": float(pcts.get("Academic", 0)),
        "industry_pct": float(pcts.get("Industry", 0)),
        "gov_pct": float(pcts.get("Government", 0)),
        "academic_mean_cit": float(sector_stats.loc["Academic", "mean"]) if "Academic" in sector_stats.index else 0.0,
    }
