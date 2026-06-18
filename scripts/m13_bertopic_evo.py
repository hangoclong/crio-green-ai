"""M13: BERTopic Strategic Map Evolution."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scisci.style import PAL, save_fig


def run(df, cfg, out_dir):
    """Visualize the BERTopic strategic map evolution over time slices.

    Uses the BERTopic topic assignments in df to create a Callon-style
    centrality × density strategic map per time period.

    Returns:
        Dict with summary stats.
    """
    os.makedirs(out_dir, exist_ok=True)

    if "topic_id_bert" not in df.columns or "year" not in df.columns:
        _write_empty(out_dir)
        return {"n_topics": 0}

    # Create time slices
    df = df.copy()
    bins = [2018, 2020, 2022, 2024, 2027]
    labels = ["2019-2020", "2021-2022", "2023-2024", "2025-2026"]
    df["time_slice"] = pd.cut(df["year"], bins=bins, labels=labels, right=True)

    # For each time slice, compute topic centrality and density
    records = []
    for ts in labels:
        sub = df[df["time_slice"] == ts]
        if len(sub) < 3:
            continue
        for tid in sub["topic_id_bert"].dropna().unique():
            topic_papers = sub[sub["topic_id_bert"] == tid]
            n = len(topic_papers)
            # Centrality: proportion of total papers
            centrality = n / max(len(sub), 1)
            # Density: mean citation impact within topic
            density = topic_papers["norm_citations"].mean() if "norm_citations" in topic_papers.columns else 0
            kw = topic_papers["topic_keywords_bert"].mode()
            kw_str = kw.iloc[0] if len(kw) > 0 else f"Topic {tid}"
            records.append({
                "time_slice": ts,
                "topic_id": tid,
                "label": str(kw_str)[:40],
                "centrality": centrality,
                "density": density,
                "paper_count": n,
            })

    if not records:
        _write_empty(out_dir)
        return {"n_topics": 0}

    map_df = pd.DataFrame(records)
    slices = [s for s in labels if s in map_df["time_slice"].values]

    n_slices = len(slices)
    if n_slices == 0:
        _write_empty(out_dir)
        return {"n_topics": 0}

    n_cols = 2 if n_slices > 2 else n_slices
    n_rows = (n_slices + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)

    colors = [PAL["primary"], PAL["secondary"], PAL["accent"], PAL["tertiary"]]

    for i, ts in enumerate(slices[:4]):
        ax = axes[i // n_cols][i % n_cols]
        sub = map_df[map_df["time_slice"] == ts]
        for j, (_, row) in enumerate(sub.iterrows()):
            ax.scatter(row["centrality"], row["density"],
                       s=row["paper_count"] * 3,
                       color=colors[j % len(colors)],
                       alpha=0.7, edgecolors="black", linewidths=0.5)
            ax.annotate(row["label"][:20], (row["centrality"], row["density"]),
                        fontsize=7, ha="center", va="bottom")

        ax.set_xlabel("Centrality (share)")
        ax.set_ylabel("Density (impact)")
        ax.set_title(ts, fontweight="bold", fontsize=11)
        ax.axhline(sub["density"].median(), color="gray", ls=":", alpha=0.5)
        ax.axvline(sub["centrality"].median(), color="gray", ls=":", alpha=0.5)

    # Hide unused subplots
    for idx in range(len(slices[:4]), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle("BERTopic Strategic Map Evolution (Centrality × Density)",
                 fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "fig_bertopic_strategic_map.png", out_dir)

    # Save data table
    tbl_lines = [
        "## BERTopic Strategic Map Data\n",
        f"- **Total topic-slice entries:** {len(map_df)}",
        f"- **Time slices:** {n_slices}\n",
        "| Time Slice | Topic ID | Label | Centrality | Density | Papers |",
        "|:---|---:|:---|---:|---:|---:|",
    ]
    for _, row in map_df.sort_values(["time_slice", "centrality"], ascending=[True, False]).iterrows():
        tbl_lines.append(
            f"| {row['time_slice']} | {row['topic_id']} | {row['label']} "
            f"| {row['centrality']:.4f} | {row['density']:.3f} | {row['paper_count']} |"
        )
    with open(os.path.join(out_dir, "table_bertopic_strategic_map.md"), "w") as f:
        f.write("\n".join(tbl_lines) + "\n")

    return {"n_topics": len(map_df), "n_slices": n_slices}


def _write_empty(out_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, "No BERTopic data available", ha="center", va="center", fontsize=11)
    ax.set_title("BERTopic Strategic Map", fontweight="bold")
    save_fig(fig, "fig_bertopic_strategic_map.png", out_dir)
