"""M15: Semantic Structural Break Detection via Embedding Centroid Drift.

Computes SPECTER2 embeddings for all papers, calculates yearly semantic
centroids, and measures cosine drift year-over-year. Tests for a semantic
regime shift that volume-based break tests cannot detect.

Core insight (Lessons-Learned Cookbook §13.1):
    "You cannot conjure a temporal break from a smooth curve."
    Volume grows exponentially, but the *meaning* of the literature may
    shift abruptly. This module detects that semantic shift.

Architecture follows the biblio-engine `get_specter2_embeddings()` pattern:
    1. AutoAdapterModel + proximity adapter (NOT sentence-transformers)
    2. CLS token extraction (NOT mean pooling)
    3. L2 normalization (required for cosine metrics)
    4. Batched inference (prevents OOM)
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

from scisci.style import PAL, save_fig


def embed_papers(
    df: pd.DataFrame,
    batch_size: int = 16,
) -> np.ndarray:
    """Embed papers using SPECTER2 with proximity adapter.

    Follows the exact architecture from biblio-engine's
    `get_specter2_embeddings()`: AutoAdapterModel + CLS + L2 norm.

    Args:
        df: DataFrame with 'title' and optionally 'abstract' columns.
        batch_size: Inference batch size.

    Returns:
        L2-normalized embeddings of shape (n_papers, 768).
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer
    from adapters import AutoAdapterModel

    if "title" not in df.columns:
        raise ValueError("DataFrame must have 'title' column")

    # Select device: CUDA > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"  📦 Loading SPECTER2 base model + proximity adapter (device={device})...")

    # Step 1: Load tokenizer and adapter-enabled model
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")

    # Step 2: Load and activate the proximity adapter
    model.load_adapter(
        "allenai/specter2", source="hf",
        load_as="proximity", set_active=True,
    )
    model.to(device)
    model.eval()
    print("  ✅ SPECTER2 proximity adapter loaded and activated")

    # Step 3: Prepare texts — title [SEP] abstract
    texts = [
        f"{t}{tokenizer.sep_token}{a}"
        for t, a in zip(
            df["title"].fillna(""),
            df["abstract"].fillna("") if "abstract" in df.columns else [""] * len(df),
        )
    ]

    # Step 4: Batched inference with CLS token extraction + L2 normalization
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=False,
                max_length=512,
            ).to(device)

            output = model(**inputs)

            # CLS token = first token of last hidden state
            cls_embeddings = output.last_hidden_state[:, 0, :]

            # L2 normalize — critical for cosine distance metrics
            normalized = F.normalize(cls_embeddings, p=2, dim=1)
            all_embeddings.append(normalized.cpu().numpy())

            if (i // batch_size) % 10 == 0 and i > 0:
                print(f"    Embedded {i}/{len(texts)} papers...")

    embeddings = np.vstack(all_embeddings)
    print(f"  ✅ Embedded {len(texts)} papers → shape {embeddings.shape}")
    return embeddings


def _compute_centroid_drift(
    embeddings: np.ndarray,
    years: np.ndarray,
) -> tuple[list[int], list[float]]:
    """Compute yearly centroid drift using cosine distance.

    For each consecutive year pair (t, t+1), calculate:
        drift_t = 1 - cosine(μ_t, μ_{t+1})

    where μ_t is the mean embedding vector of all papers published in year t.

    Args:
        embeddings: L2-normalized embeddings (n_papers, dim).
        years: Year labels for each paper.

    Returns:
        Tuple of (transition_years, drift_values).
    """
    unique_years = sorted(set(years))
    centroids = {}
    for y in unique_years:
        mask = years == y
        if mask.sum() > 0:
            centroid = embeddings[mask].mean(axis=0)
            # Re-normalize the mean centroid
            centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
            centroids[y] = centroid

    transition_years = []
    drift_values = []
    sorted_years = sorted(centroids.keys())
    for i in range(len(sorted_years) - 1):
        y1, y2 = sorted_years[i], sorted_years[i + 1]
        c1, c2 = centroids[y1], centroids[y2]
        sim = cosine_similarity(c1.reshape(1, -1), c2.reshape(1, -1))[0, 0]
        drift = 1.0 - sim
        transition_years.append(y2)
        drift_values.append(float(drift))

    return transition_years, drift_values


def run(df: pd.DataFrame, cfg: dict, out_dir: str) -> dict:
    """Run semantic structural break analysis.

    1. Compute SPECTER2 embeddings for all papers.
    2. Calculate yearly semantic centroids.
    3. Compute cosine drift series.
    4. Identify max drift year (candidate semantic break point).

    Args:
        df: DataFrame with 'year', 'title', and optionally 'abstract'.
        cfg: Config dict (unused but kept for pipeline compatibility).
        out_dir: Output directory.

    Returns:
        Dict with max_drift_year, max_drift_value, and drift_series.
    """
    os.makedirs(out_dir, exist_ok=True)

    if "title" not in df.columns or "year" not in df.columns:
        _write_fallback(out_dir)
        return {"max_drift_year": 0, "max_drift_value": 0.0, "drift_series": []}

    df = df.copy()

    # Compute embeddings
    print("  🔬 Computing SPECTER2 embeddings for semantic break analysis...")
    embeddings = embed_papers(df)

    # Compute centroid drift
    years = df["year"].values
    transition_years, drift_values = _compute_centroid_drift(embeddings, years)

    if not drift_values:
        _write_fallback(out_dir)
        return {"max_drift_year": 0, "max_drift_value": 0.0, "drift_series": []}

    # Find maximum drift
    max_idx = np.argmax(drift_values)
    max_drift_year = transition_years[max_idx]
    max_drift_value = drift_values[max_idx]

    # Statistical significance: is the max drift an outlier?
    mean_drift = np.mean(drift_values)
    std_drift = np.std(drift_values) if len(drift_values) > 1 else 1e-10
    z_score = (max_drift_value - mean_drift) / std_drift if std_drift > 0 else 0.0
    is_significant = abs(z_score) > 1.96  # 95% CI

    # ── Save table ────────────────────────────────────────────────────────
    lines = [
        "## Semantic Structural Break Analysis (SPECTER2 Centroid Drift)\n",
        f"- **Max drift year:** {max_drift_year}",
        f"- **Max drift value:** {max_drift_value:.4f}",
        f"- **Mean drift:** {mean_drift:.4f}",
        f"- **Z-score:** {z_score:.2f}",
        f"- **Significant (|z| > 1.96):** {'✓ Yes' if is_significant else 'No'}\n",
        "| Transition | Cosine Drift | Z-score | Significant? |",
        "|:---|---:|---:|:---|",
    ]
    for ty, dv in zip(transition_years, drift_values):
        z = (dv - mean_drift) / std_drift if std_drift > 0 else 0.0
        sig = "✓ Yes" if abs(z) > 1.96 else "No"
        marker = " **← MAX**" if ty == max_drift_year else ""
        lines.append(f"| {ty-1}→{ty} | {dv:.4f} | {z:.2f} | {sig}{marker} |")

    with open(os.path.join(out_dir, "table_semantic_break.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ── Save figure ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    # Bar chart of drift per transition
    labels = [f"{ty-1}→{ty}" for ty in transition_years]
    colors = [PAL["secondary"] if ty == max_drift_year else PAL["primary"]
              for ty in transition_years]

    bars = ax.bar(labels, drift_values, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(mean_drift, color=PAL["tertiary"], ls="--", lw=1.5,
               label=f"Mean drift = {mean_drift:.4f}")

    if is_significant:
        # Mark the break point
        ax.annotate(
            f"Semantic Break\n(z={z_score:.1f})",
            xy=(max_idx, max_drift_value),
            xytext=(max_idx, max_drift_value * 1.15),
            ha="center", fontsize=10, fontweight="bold", color="red",
            arrowprops=dict(arrowstyle="->", color="red"),
        )

    ax.set_xlabel("Year Transition", fontsize=12)
    ax.set_ylabel("Cosine Drift", fontsize=12)
    ax.set_title(
        "Semantic Centroid Drift — SPECTER2 Embedding Analysis",
        fontweight="bold", fontsize=13,
    )
    ax.legend(fontsize=10)
    fig.tight_layout()
    save_fig(fig, "fig_semantic_drift.png", out_dir)

    return {
        "max_drift_year": int(max_drift_year),
        "max_drift_value": float(max_drift_value),
        "drift_series": list(zip(transition_years, drift_values)),
        "z_score": float(z_score),
        "is_significant": bool(is_significant),
    }


def _write_fallback(out_dir: str) -> None:
    """Write fallback files when data is insufficient."""
    with open(os.path.join(out_dir, "table_semantic_break.md"), "w") as f:
        f.write("## Semantic Break\n\nInsufficient data for semantic analysis.\n")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", fontsize=14)
    ax.set_title("Semantic Drift", fontweight="bold")
    save_fig(fig, "fig_semantic_drift.png", out_dir)
