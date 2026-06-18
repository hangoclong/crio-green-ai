"""M12: Topic Drift Forecasting (ARIMA on cluster density time series at quarterly resolution)."""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from pmdarima.arima import ADFTest, KPSSTest
import scipy.stats as stats

from scisci.style import PAL, save_fig

def diebold_mariano_hac(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray, h: int = 2) -> tuple[float, float]:
    """Compute the Diebold-Mariano test statistic with lag-1 HAC standard errors.

    Loss function: Squared Error.
    Model 1: optimized ARIMA.
    Model 2: naive Random Walk baseline.
    """
    e1 = y_true - y_pred1
    e2 = y_true - y_pred2
    d = e1**2 - e2**2  # loss differential (squared error)
    
    d_bar = np.mean(d)
    T = len(d)
    if T <= 2:
        return 0.0, 1.0
        
    # Autocovariances
    gamma0 = np.var(d, ddof=1)
    # Manual lag-1 covariance
    d_centered = d - d_bar
    gamma1 = np.sum(d_centered[1:] * d_centered[:-1]) / (T - 1)
        
    # HAC variance for h=2 steps ahead
    var_d_bar = (gamma0 + 2 * (1.0 - 0.5) * gamma1) / T
    if var_d_bar <= 0:
        return 0.0, 1.0
        
    dm_stat = d_bar / np.sqrt(var_d_bar)
    p_val = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    return float(dm_stat), float(p_val)

def run(df: pd.DataFrame, cfg: dict, out_dir: str) -> dict:
    """Forecast topic density for the next 2 years (8 quarters) using AIC-optimized ARIMA.

    Uses quarterly publication counts per cluster as the time series (N=28).
    Evaluates out-of-sample forecast accuracy using the Diebold-Mariano test.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = df.copy()

    # Build quarterly cluster counts
    if "cluster_id" not in df.columns or "publication_date" not in df.columns:
        _write_empty(out_dir)
        return {"forecast": "no data"}

    # Exclude partial 2026 data for clean temporal baseline (2019-2025 = 28 quarters)
    df["publication_date"] = pd.to_datetime(df["publication_date"])
    df = df[df["publication_date"].dt.year < 2026]

    df["quarter"] = df["publication_date"].dt.to_period("Q")
    
    # Fill in complete quarter grid to prevent missing time segments
    all_quarters = pd.period_range(start="2019Q1", end="2025Q4", freq="Q")
    
    # Build complete pivot table
    pivot = df.groupby(["quarter", "cluster_id"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(all_quarters, fill_value=0)
    
    clusters = pivot.columns.tolist()

    # Get cluster labels
    label_map = {}
    if "_cname" in df.columns:
        for cid in clusters:
            sub = df[df["cluster_id"] == cid]
            if len(sub) > 0 and "_cname" in sub.columns:
                label_map[cid] = sub["_cname"].iloc[0]
            else:
                label_map[cid] = f"Cluster {cid}"
    else:
        label_map = {c: f"Cluster {c}" for c in clusters}

    # Forecast quarters: 2026Q1 to 2027Q4 (8 quarters)
    forecast_quarters = pd.period_range(start="2026Q1", end="2027Q4", freq="Q")
    forecast_labels = [str(q) for q in forecast_quarters]

    forecasts = {}
    dm_results = {}
    n_clusters = min(len(clusters), 4)
    n_cols = 2 if n_clusters > 2 else n_clusters
    n_rows = (n_clusters + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)

    for i, cid in enumerate(clusters[:4]):
        ts = pivot[cid].sort_index()
        ax = axes[i // n_cols][i % n_cols]
        label = label_map.get(cid, f"C{cid}")

        # Plot observed quarterly data (2019-2025)
        ax.plot([str(q) for q in ts.index], ts.values, "o-", color=PAL["primary"], label="Observed")

        # Log-transform to satisfy ARIMA distributional assumptions
        ts_raw = ts.values.astype(float)
        ts_log = np.log1p(ts_raw)

        # Stationarity testing (ADF and KPSS) on log-transformed data
        try:
            adf_test = ADFTest(alpha=0.05)
            adf_pval, adf_diff = adf_test.should_diff(ts_log)
            
            kpss_test = KPSSTest(alpha=0.05)
            kpss_pval, kpss_diff = kpss_test.should_diff(ts_log)
            
            print(f"  [Stationarity Test] Cluster: {label} (log-transformed)")
            print(f"    ADF Test:  p-value = {adf_pval:.4f}, should difference = {adf_diff}")
            print(f"    KPSS Test: p-value = {kpss_pval:.4f}, should difference = {kpss_diff}")
        except Exception as se:
            print(f"  [Stationarity Warning] Failed to compute stationarity tests for {label}: {se}")

        try:
            # 1. Automated pmdarima AIC grid search on log-transformed series
            auto_model = pm.auto_arima(
                ts_log,
                seasonal=False,
                start_p=0, start_q=0,
                max_p=3, max_q=3,
                max_d=1,
                information_criterion="aic",
                error_action="ignore",
                suppress_warnings=True
            )
            order = auto_model.order
            
            # 2. Fit ARIMA in log-space, then back-transform forecasts
            model = ARIMA(ts_log, order=order)
            fitted = model.fit()
            fc_log = fitted.forecast(steps=8)
            fc = np.expm1(fc_log)  # back-transform: exp(x) - 1
            fc = np.maximum(fc, 0)  # ensure non-negative counts
            forecasts[label] = fc.tolist()

            # Plot forecasts (in natural scale)
            x_fc = [str(q) for q in forecast_quarters]
            ax.plot(x_fc, fc, "s--", color=PAL["secondary"], ms=8, label=f"ARIMA{order}")
            ax.fill_between(x_fc, fc * 0.7, fc * 1.3, alpha=0.15, color=PAL["secondary"])

            # 3. Walk-forward validation (DM test on back-transformed predictions)
            y_val = ts_raw[-8:]  # actual raw counts
            arima_preds = []
            naive_preds = []
            
            for step in range(8):
                idx = len(ts) - 8 + step
                train_log = ts_log[:idx]
                
                # Fit ARIMA in log-space, back-transform prediction
                try:
                    m_sub = ARIMA(train_log, order=order).fit()
                    pred_log = m_sub.forecast(steps=1)[0]
                    arima_preds.append(max(np.expm1(pred_log), 0))
                except Exception:
                    arima_preds.append(ts_raw[idx - 1])  # fallback
                    
                # Naive baseline: last raw value
                naive_preds.append(ts_raw[idx - 1])

            # Run Diebold-Mariano test on natural-scale predictions
            dm_stat, p_val = diebold_mariano_hac(y_val, np.array(arima_preds), np.array(naive_preds), h=2)
            dm_results[label] = {"stat": dm_stat, "p_value": p_val, "order": order}

        except Exception as e:
            print(f"  [ARIMA Warning] Forecasting failed for {label}: {e}")
            forecasts[label] = [0] * 8
            dm_results[label] = {"stat": 0.0, "p_value": 1.0, "order": (1, 1, 0)}

        ax.set_title(f"{label[:25]} ARIMA{dm_results[label]['order']}", fontweight="bold", fontsize=10)
        ax.set_xlabel("Quarter")
        ax.set_ylabel("Papers")
        
        # Style x-axis to be readable
        ticks = ax.get_xticks()
        ax.set_xticks(ticks[::4]) # Show every 4th quarter (annual markers)
        ax.tick_params(axis='x', rotation=30)
        ax.legend(fontsize=8)

    # Hide unused subplots
    for idx in range(n_clusters, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle("Topic Drift Forecast (Quarterly resampled ARIMA)", fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "fig_topic_forecast.png", out_dir)

    # Save table
    lines = [
        "| Cluster | Optimal ARIMA Order | DM Stat | DM p-value | Significant? | 2027 Total Forecast |",
        "|:---|:---:|---:|---:|:---|---:|"
    ]
    for label, fc in forecasts.items():
        res = dm_results.get(label, {"stat": 0.0, "p_value": 1.0, "order": (1,1,0)})
        sig = "✓ Yes" if res["p_value"] < 0.05 else "No"
        # Total forecasted papers for 2027 (quarters 4 to 7 in fc, corresponding to 2027Q1-Q4)
        total_2027 = sum(fc[4:8])
        lines.append(f"| {label} | ARIMA{res['order']} | {res['stat']:.3f} | {res['p_value']:.4f} | {sig} | {total_2027:.0f} |")
        
    with open(os.path.join(out_dir, "table_topic_forecast.md"), "w") as f:
        f.write("## Topic Drift Forecast (Quarterly ARIMA with Diebold-Mariano Validation)\n\n")
        f.write("\n".join(lines) + "\n")

    return {"forecasts": forecasts, "dm_results": dm_results}

def _write_empty(out_dir: str) -> None:
    with open(os.path.join(out_dir, "table_topic_forecast.md"), "w") as f:
        f.write("## Topic Forecast\n\nNo cluster data available.\n")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, "No data", ha="center", va="center")
    save_fig(fig, "fig_topic_forecast.png", out_dir)
