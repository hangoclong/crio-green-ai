"""M08: R-Learner (Robinson's Transformation) for heterogeneous treatment effects.

Replaces the prior Poisson T-Learner with a statistically valid continuous
R-Learner on log-transformed citations ln(y+1).

Robinson's orthogonalization:
    1. Estimate propensity ê(X) = P(boundary=1|X) via GradientBoostingClassifier.
    2. Estimate nuisance m̂(X) = E[Y|X] via GradientBoostingRegressor.
    3. Residualize: Ỹ = Y - m̂(X), T̃ = T - ê(X).
    4. Estimate CATE: τ̂(X) = argmin Σ [Ỹ_i - τ(X_i) T̃_i]².
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict

from scisci.style import PAL, save_fig


def run(df: pd.DataFrame, cfg: dict, out_dir: str) -> dict:
    """Estimate heterogeneous treatment effects via Robinson's R-Learner.

    Outcome = log-transformed citations ln(citations + 1).
    Treatment = boundary spanning (interaction indicator).
    Covariates = paper_age, author_count, degree, author_prestige, is_review.

    Returns:
        Dict with ATE, ATE std, sample sizes, and propensity diagnostics.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = df.copy()
    df["is_boundary"] = df["interaction"].astype(int)

    covariates = ["paper_age", "author_count", "degree", "author_prestige", "is_review"]
    covariates = [c for c in covariates if c in df.columns]

    if not covariates or df["is_boundary"].sum() < 5:
        _write_empty(out_dir, "Insufficient boundary-spanning papers for R-Learner.")
        return {"ate": 0.0, "ate_std": 0.0, "n_treated": 0, "n_control": 0}

    # Outcome: log-transformed citations (continuous)
    X = df[covariates].fillna(0).values
    Y = np.log1p(df["citations"].values).astype(float)  # ln(y + 1)
    T = df["is_boundary"].values.astype(float)

    n_treated = int(T.sum())
    n_control = int((1 - T).sum())

    if n_treated < 5 or n_control < 5:
        _write_empty(out_dir, "Too few samples in treatment or control group.")
        return {"ate": 0.0, "ate_std": 0.0, "n_treated": n_treated, "n_control": n_control}

    print(f"🔬 R-Learner (Robinson's Transformation) on {len(df)} papers...")
    print(f"   Treated: {n_treated} | Control: {n_control}")

    # ── Step 1: Propensity model ê(X) ────────────────────────────────────
    propensity_model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
    )
    # Cross-validated propensity scores to avoid overfitting
    e_hat = cross_val_predict(propensity_model, X, T, cv=5, method="predict_proba")[:, 1]

    # Clip propensity scores for stability (standard practice)
    e_hat = np.clip(e_hat, 0.05, 0.95)

    print(f"   Propensity scores: mean = {e_hat.mean():.3f}, "
          f"std = {e_hat.std():.3f}, "
          f"[{e_hat.min():.3f}, {e_hat.max():.3f}]")

    # ── Step 2: Outcome nuisance model m̂(X) ─────────────────────────────
    outcome_model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
    )
    m_hat = cross_val_predict(outcome_model, X, Y, cv=5)

    # ── Step 3: Robinson's orthogonalization ─────────────────────────────
    Y_tilde = Y - m_hat       # Residualized outcome
    T_tilde = T - e_hat       # Residualized treatment

    # ── Step 4: Estimate CATE via weighted regression ────────────────────
    # τ̂(X) = argmin Σ [Ỹ_i - τ(X_i) T̃_i]²
    # For heterogeneous effects, we weight by T̃² and regress Ỹ/T̃ on X
    # Using Ridge for regularization

    # Avoid division by near-zero residualized treatment
    valid_mask = np.abs(T_tilde) > 0.01
    X_valid = X[valid_mask]
    pseudo_outcome = Y_tilde[valid_mask] / T_tilde[valid_mask]
    weights = T_tilde[valid_mask] ** 2

    cate_model = Ridge(alpha=1.0)
    cate_model.fit(X_valid, pseudo_outcome, sample_weight=weights)

    # Predict individual treatment effects for all observations
    tau_hat = cate_model.predict(X)

    # ATE = mean of CATE estimates
    ate_mean = float(np.mean(tau_hat))
    ate_std = float(np.std(tau_hat) / np.sqrt(len(tau_hat)))  # SE of mean

    # Alternative simple ATE via moment condition
    # ATE_simple = Σ (Ỹ_i * T̃_i) / Σ (T̃_i²)
    ate_simple = float(np.sum(Y_tilde * T_tilde) / np.sum(T_tilde ** 2))

    print(f"   R-Learner ATE (Ridge CATE): {ate_mean:.4f} ± {ate_std:.4f}")
    print(f"   R-Learner ATE (moment):     {ate_simple:.4f}")

    # ── Save table report ────────────────────────────────────────────────
    lines = [
        "## R-Learner Causal Analysis: Heterogeneous Treatment Effects\n",
        "### Robinson's Transformation (Orthogonalized Estimator)\n",
        f"- **Outcome:** log-transformed citations $\\ln(y+1)$",
        f"- **Treatment:** Boundary spanning (governance × capability)",
        f"- **Propensity model:** GradientBoostingClassifier (5-fold CV)",
        f"- **Outcome model:** GradientBoostingRegressor (5-fold CV)",
        f"- **CATE model:** Ridge regression (α=1.0)\n",
        f"- **Average Treatment Effect (ATE):** {ate_mean:.4f} ± {ate_std:.4f}",
        f"- **ATE (moment estimator):** {ate_simple:.4f}",
        f"- **N (treated):** {n_treated}",
        f"- **N (control):** {n_control}",
        "",
        "### Propensity Score Diagnostics",
        f"- Mean propensity: {e_hat.mean():.3f}",
        f"- Std propensity: {e_hat.std():.3f}",
        f"- Min/Max: [{e_hat.min():.3f}, {e_hat.max():.3f}]",
        f"- Treated overlap [10th, 90th]: "
        f"[{np.percentile(e_hat[T == 1], 10):.3f}, "
        f"{np.percentile(e_hat[T == 1], 90):.3f}]",
        f"- Control overlap [10th, 90th]: "
        f"[{np.percentile(e_hat[T == 0], 10):.3f}, "
        f"{np.percentile(e_hat[T == 0], 90):.3f}]",
        "",
        "### CATE Distribution Summary",
        "| Metric | Value |",
        "|:---|---:|",
        f"| Mean (ATE) | {ate_mean:.4f} |",
        f"| Std. Dev. | {float(np.std(tau_hat)):.4f} |",
        f"| Min | {float(np.min(tau_hat)):.4f} |",
        f"| Median | {float(np.median(tau_hat)):.4f} |",
        f"| Max | {float(np.max(tau_hat)):.4f} |",
        "",
        "### Feature Coefficients (CATE Model)",
        "| Feature | Coefficient |",
        "|:---|---:|",
    ]
    for feat, coef in zip(covariates, cate_model.coef_):
        lines.append(f"| {feat} | {coef:.4f} |")

    with open(os.path.join(out_dir, "table_causal_forest.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ── Figures ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel (a): Propensity score overlap diagnostic
    ax1 = axes[0]
    ax1.hist(e_hat[T == 0], bins=30, alpha=0.6, color=PAL["primary"],
             label=f"Control (n={n_control})", density=True, edgecolor="white")
    ax1.hist(e_hat[T == 1], bins=30, alpha=0.6, color=PAL["secondary"],
             label=f"Treated (n={n_treated})", density=True, edgecolor="white")
    ax1.set_xlabel("Propensity Score ê(X)")
    ax1.set_ylabel("Density")
    ax1.set_title("(a) Propensity Score Overlap", fontweight="bold", fontsize=11)
    ax1.legend(fontsize=9)

    # Panel (b): CATE distribution
    ax2 = axes[1]
    ax2.hist(tau_hat, bins=40, color=PAL["light"], edgecolor=PAL["primary"],
             alpha=0.8)
    ax2.axvline(ate_mean, color=PAL["secondary"], ls="--", lw=2,
                label=f"ATE = {ate_mean:.3f}")
    ax2.axvline(0, color="red", ls=":", lw=1.5)
    ax2.set_xlabel("Conditional Average Treatment Effect (CATE)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("(b) CATE Distribution (R-Learner)", fontweight="bold", fontsize=11)
    ax2.legend(fontsize=9)

    # Panel (c): Feature coefficients
    ax3 = axes[2]
    coefs = cate_model.coef_
    sorted_idx = np.argsort(np.abs(coefs))
    ax3.barh(range(len(covariates)), coefs[sorted_idx], color=PAL["accent"])
    ax3.set_yticks(range(len(covariates)))
    ax3.set_yticklabels([covariates[i] for i in sorted_idx])
    ax3.axvline(0, color="gray", ls="-", lw=0.5)
    ax3.set_xlabel("CATE Coefficient")
    ax3.set_title("(c) Treatment Effect Moderators", fontweight="bold", fontsize=11)

    fig.suptitle("R-Learner Causal Analysis: Boundary Spanning × Citation Impact",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, "fig_causal_forest_hte.png", out_dir)

    print(f"✅ R-Learner completed: ATE = {ate_mean:.4f} ± {ate_std:.4f}\n")
    return {
        "ate": ate_mean,
        "ate_std": ate_std,
        "ate_simple": ate_simple,
        "n_treated": n_treated,
        "n_control": n_control,
        "propensity_mean": float(e_hat.mean()),
        "propensity_std": float(e_hat.std()),
    }


def _write_empty(out_dir: str, msg: str) -> None:
    """Write placeholder outputs when insufficient data exists."""
    with open(os.path.join(out_dir, "table_causal_forest.md"), "w") as f:
        f.write(f"## R-Learner Causal Analysis\n\n{msg}\n")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=11)
    ax.set_title("R-Learner", fontweight="bold")
    save_fig(fig, "fig_causal_forest_hte.png", out_dir)
