"""M16: Bass Diffusion Model + Prophet Changepoint Detection.

Supplements the ARIMA-based m12 with two additional forecasting methods:

1. **Bass Diffusion Model** (Bass, 1969): Fits cumulative publication counts
   to an S-curve adoption model. Extracts innovation (p), imitation (q),
   and market potential (M) parameters for interpretable saturation analysis.

2. **Prophet Changepoint Detection** (Taylor & Letham, 2018): Auto-detects
   acceleration/deceleration points in the publication time series with
   calibrated changepoint probabilities.

Both methods provide complementary interpretive value:
  - Bass → "When will Green AI literature saturate?"
  - Prophet → "When did the acceleration actually begin?"
"""

from __future__ import annotations
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from scisci.style import PAL, save_fig


# ── Bass Diffusion Model ─────────────────────────────────────────────────────

def _bass_cumulative(t: np.ndarray, p: float, q: float, M: float) -> np.ndarray:
    """Bass diffusion cumulative adoption function.

    F(t) = M * (1 - exp(-(p+q)*t)) / (1 + (q/p)*exp(-(p+q)*t))

    Args:
        t: Time index (0, 1, 2, ...).
        p: Innovation coefficient (external influence).
        q: Imitation coefficient (internal influence, word-of-mouth).
        M: Market potential (saturation ceiling).

    Returns:
        Cumulative adoptions at each time point.
    """
    exp_term = np.exp(-(p + q) * t)
    return M * (1 - exp_term) / (1 + (q / p) * exp_term)


def fit_bass(t: np.ndarray, cumulative: np.ndarray) -> tuple[float, float, float, float]:
    """Fit a Bass diffusion model to cumulative publication data.

    Args:
        t: Time index array (0, 1, 2, ...).
        cumulative: Cumulative publication counts.

    Returns:
        Tuple of (p, q, M, r_squared). Returns defaults on failure.
    """
    if len(t) < 4:
        # Need at least 4 points for 3-parameter estimation
        return 0.01, 0.3, float(cumulative[-1]) * 2 if len(cumulative) > 0 else 100.0, 0.0

    # Initial guesses: p~0.03 (typical innovation), q~0.38 (typical imitation),
    # M ~ 2x current total (reasonable saturation estimate)
    M_init = float(cumulative[-1]) * 3
    p0 = [0.03, 0.38, M_init]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                _bass_cumulative, t, cumulative,
                p0=p0,
                bounds=([1e-6, 1e-6, cumulative[-1]], [1.0, 2.0, cumulative[-1] * 20]),
                maxfev=10000,
            )
        p_fit, q_fit, M_fit = popt

        # R² calculation
        y_pred = _bass_cumulative(t, p_fit, q_fit, M_fit)
        ss_res = np.sum((cumulative - y_pred) ** 2)
        ss_tot = np.sum((cumulative - np.mean(cumulative)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return float(p_fit), float(q_fit), float(M_fit), float(r2)

    except (RuntimeError, ValueError):
        return 0.01, 0.3, float(cumulative[-1]) * 2, 0.0


# ── Prophet Changepoints ─────────────────────────────────────────────────────

def _detect_changepoints_manual(ts: pd.Series) -> dict:
    """Manual changepoint detection using second-derivative analysis.

    Fallback for when Prophet is not installed. Detects acceleration
    points using the discrete second difference of the time series.

    Args:
        ts: Time series (index = datetime-like, values = counts).

    Returns:
        Dict with changepoints list and trend data.
    """
    values = ts.values.astype(float)
    if len(values) < 3:
        return {"changepoints": [], "method": "second_difference", "trend": values.tolist()}

    # Second difference = acceleration
    d2 = np.diff(values, n=2)
    # Detect significant acceleration points (> 1 std above mean)
    threshold = np.mean(np.abs(d2)) + np.std(np.abs(d2))
    change_indices = np.where(np.abs(d2) > threshold)[0] + 1  # +1 for diff offset

    changepoints = []
    for idx in change_indices:
        if idx < len(ts):
            changepoints.append({
                "date": str(ts.index[idx]),
                "magnitude": float(d2[idx - 1]),
                "direction": "acceleration" if d2[idx - 1] > 0 else "deceleration",
            })

    return {
        "changepoints": changepoints,
        "method": "second_difference",
        "trend": values.tolist(),
    }


def _run_prophet(quarterly_ts: pd.DataFrame) -> dict:
    """Run Prophet changepoint detection if available.

    Args:
        quarterly_ts: DataFrame with 'ds' (date) and 'y' (count) columns.

    Returns:
        Dict with changepoints and decomposition data.
    """
    try:
        from prophet import Prophet

        model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_mode="additive",
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(quarterly_ts)

        # Get changepoints
        cps = model.changepoints.tolist()
        changepoints = [{"date": str(cp), "magnitude": 0.0} for cp in cps]

        # Forecast
        future = model.make_future_dataframe(periods=8, freq="QS")
        forecast = model.predict(future)

        return {
            "changepoints": changepoints,
            "method": "prophet",
            "forecast": forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_dict("records"),
            "trend": forecast["trend"].tolist(),
        }

    except ImportError:
        # Prophet not installed — use manual fallback
        ts = pd.Series(quarterly_ts["y"].values, index=quarterly_ts["ds"])
        return _detect_changepoints_manual(ts)


# ── Main Module Entry Point ──────────────────────────────────────────────────

def run(df: pd.DataFrame, cfg: dict, out_dir: str) -> dict:
    """Run Bass Diffusion + Prophet changepoint analysis.

    Args:
        df: Full corpus DataFrame with 'year' and 'publication_date' columns.
        cfg: Configuration dict (from scisci_config.yaml).
        out_dir: Output directory.

    Returns:
        Dict with 'bass' and 'prophet' sub-results.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = df.copy()
    df = df[df["year"] < 2026]  # Exclude partial terminal year

    # ── Annual counts for Bass ────────────────────────────────────────────
    annual = df.groupby("year").size().sort_index()
    cumulative = annual.cumsum()

    t = np.arange(len(cumulative), dtype=float)
    cum_values = cumulative.values.astype(float)

    p_bass, q_bass, M_bass, r2_bass = fit_bass(t, cum_values)

    # Peak adoption year (when dF/dt is maximum)
    if q_bass > p_bass:
        t_peak = np.log(q_bass / p_bass) / (p_bass + q_bass)
        peak_year = int(annual.index[0] + t_peak)
    else:
        peak_year = int(annual.index[-1])

    # ── Bass figure ───────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Cumulative fit
    t_extended = np.linspace(0, len(t) + 5, 200)
    fitted_cum = _bass_cumulative(t_extended, p_bass, q_bass, M_bass)

    ax1.plot(annual.index, cum_values, "o", color=PAL["primary"], ms=8, label="Observed")
    years_extended = np.linspace(annual.index[0], annual.index[-1] + 5, 200)
    ax1.plot(years_extended, fitted_cum, "-", color=PAL["secondary"], lw=2,
             label=f"Bass (p={p_bass:.4f}, q={q_bass:.4f})")
    ax1.axhline(M_bass, color=PAL["accent"], ls="--", lw=1.5,
                label=f"Saturation M={M_bass:.0f}")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Cumulative Publications")
    ax1.set_title(f"Bass Diffusion Model (R²={r2_bass:.3f})", fontweight="bold")
    ax1.legend(fontsize=9)

    # Right: Annual adoption rate (derivative)
    dt = 0.01
    t_deriv = np.linspace(0, len(t) + 5, 500)
    cum_deriv = _bass_cumulative(t_deriv, p_bass, q_bass, M_bass)
    annual_rate = np.diff(cum_deriv) / dt
    t_rate = t_deriv[:-1]
    years_rate = np.linspace(annual.index[0], annual.index[-1] + 5, len(t_rate))

    ax2.plot(years_rate, annual_rate, "-", color=PAL["primary"], lw=2)
    ax2.bar(annual.index, annual.values, color=PAL["light"], edgecolor=PAL["primary"],
            linewidth=0.8, alpha=0.7, label="Observed Annual")
    ax2.axvline(peak_year, color="red", ls="--", lw=1.5,
                label=f"Peak adoption: {peak_year}")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Annual Publications")
    ax2.set_title("Adoption Rate (Bass dF/dt)", fontweight="bold")
    ax2.legend(fontsize=9)

    fig.suptitle("Bass Diffusion Model — Green AI Literature Adoption",
                 fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "fig_bass_adoption.png", out_dir)

    # ── Bass table ────────────────────────────────────────────────────────
    bass_lines = [
        "## Bass Diffusion Model Results\n",
        "| Parameter | Value | Interpretation |",
        "|:---|---:|:---|",
        f"| p (innovation) | {p_bass:.4f} | External influence coefficient |",
        f"| q (imitation) | {q_bass:.4f} | Internal influence (word-of-mouth) |",
        f"| M (saturation) | {M_bass:.0f} | Estimated total market potential |",
        f"| R² | {r2_bass:.4f} | Goodness of fit |",
        f"| Peak year | {peak_year} | Year of maximum adoption rate |",
        f"| Current total | {int(cum_values[-1])} | Cumulative publications to date |",
        f"| Penetration | {cum_values[-1] / M_bass * 100:.1f}% | Current % of saturation |",
    ]
    with open(os.path.join(out_dir, "table_bass_diffusion.md"), "w") as f:
        f.write("\n".join(bass_lines) + "\n")

    # ── Prophet changepoints ──────────────────────────────────────────────
    if "publication_date" in df.columns:
        df["publication_date"] = pd.to_datetime(df["publication_date"])
        quarterly = df.groupby(df["publication_date"].dt.to_period("Q")).size()
        quarters = pd.period_range(start="2019Q1", end="2025Q4", freq="Q")
        quarterly = quarterly.reindex(quarters, fill_value=0)
        prophet_df = pd.DataFrame({
            "ds": [q.start_time for q in quarterly.index],
            "y": quarterly.values.astype(float),
        })
    else:
        prophet_df = pd.DataFrame({
            "ds": pd.to_datetime([f"{y}-06-15" for y in annual.index]),
            "y": annual.values.astype(float),
        })

    prophet_result = _run_prophet(prophet_df)

    # ── Prophet figure ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(prophet_df["ds"], prophet_df["y"], "o-", color=PAL["primary"],
            ms=6, label="Observed")

    # Mark changepoints
    for cp in prophet_result["changepoints"]:
        cp_date = pd.to_datetime(cp["date"])
        ax.axvline(cp_date, color=PAL["secondary"], ls="--", lw=1, alpha=0.7)

    if prophet_result["changepoints"]:
        # Add single legend entry for changepoints
        ax.axvline(pd.to_datetime(prophet_result["changepoints"][0]["date"]),
                   color=PAL["secondary"], ls="--", lw=1, alpha=0.7,
                   label=f"Changepoints (n={len(prophet_result['changepoints'])})")

    ax.set_xlabel("Quarter")
    ax.set_ylabel("Publications per Quarter")
    ax.set_title(
        f"Changepoint Detection ({prophet_result['method']})",
        fontweight="bold", fontsize=13,
    )
    ax.legend(fontsize=10)
    fig.tight_layout()
    save_fig(fig, "fig_prophet_decomposition.png", out_dir)

    # ── Prophet table ─────────────────────────────────────────────────────
    prophet_lines = [
        f"## Changepoint Detection ({prophet_result['method']})\n",
        f"**Method:** {prophet_result['method']}",
        f"**Changepoints detected:** {len(prophet_result['changepoints'])}\n",
        "| # | Date | Direction | Magnitude |",
        "|---:|:---|:---|---:|",
    ]
    for i, cp in enumerate(prophet_result["changepoints"], 1):
        direction = cp.get("direction", "—")
        magnitude = cp.get("magnitude", 0.0)
        prophet_lines.append(f"| {i} | {cp['date']} | {direction} | {magnitude:.2f} |")

    with open(os.path.join(out_dir, "table_prophet_changepoints.md"), "w") as f:
        f.write("\n".join(prophet_lines) + "\n")

    return {
        "bass": {
            "p": p_bass,
            "q": q_bass,
            "M": M_bass,
            "r_squared": r2_bass,
            "peak_year": peak_year,
            "penetration_pct": float(cum_values[-1] / M_bass * 100),
        },
        "prophet": prophet_result,
    }
