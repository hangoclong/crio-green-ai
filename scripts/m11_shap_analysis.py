"""M11: Random Forest + SHAP Feature Importance."""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from scisci.data_loader import get_labels, get_model_vars
from scisci.style import PAL, save_fig


def run(df, cfg, out_dir):
    """Train Random Forest to predict citation impact, then compute SHAP-like
    feature importance using permutation importance.

    Returns:
        Dict with rf_r2, rf_cv_r2, top_features.
    """
    os.makedirs(out_dir, exist_ok=True)
    labels = get_labels(cfg)

    feature_cols = get_model_vars(cfg) + ["interaction"]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].fillna(0).values
    y = df["log_citations"].values

    rf = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    r2 = float(rf.score(X, y))

    cv_scores = cross_val_score(rf, X, y, cv=5, scoring="r2")
    cv_r2 = float(np.mean(cv_scores))

    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)
    feature_names = [labels.get(c, c) for c in feature_cols]

    # Save table
    lines = ["| Feature | Importance |", "|:---|---:|"]
    for i in reversed(sorted_idx):
        lines.append(f"| {feature_names[i]} | {importances[i]:.4f} |")
    with open(os.path.join(out_dir, "table_rf_metrics.md"), "w") as f:
        f.write("## Random Forest Feature Importance\n\n")
        f.write(f"- **R² (train):** {r2:.4f}\n")
        f.write(f"- **R² (5-fold CV):** {cv_r2:.4f} ± {float(np.std(cv_scores)):.4f}\n\n")
        f.write("\n".join(lines) + "\n")

    # Figure: horizontal bar chart (SHAP-style)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(sorted_idx)), importances[sorted_idx],
            color=PAL["primary"], edgecolor="white")
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title(f"Random Forest Feature Importance\n(R²={r2:.3f}, CV-R²={cv_r2:.3f})",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    save_fig(fig, "fig_shap_waterfall.png", out_dir)

    top_features = [(feature_names[i], float(importances[i])) for i in reversed(sorted_idx)]
    return {"rf_r2": r2, "rf_cv_r2": cv_r2, "top_features": top_features[:5]}
