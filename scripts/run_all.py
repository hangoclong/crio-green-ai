#!/usr/bin/env python3
"""
SciSci Master Script — Orchestrates all 13 econometric analysis modules.

Reads the scisci_config.yaml, loads the dataset, and runs every module.
Outputs are written to a dedicated `scisci/` folder with a human-readable
OUTPUT_GUIDE.md that explains each file and its relevance to the paper.

Usage:
    uv run python papers/6b-crio/scripts/scisci/run_all.py \\
        --config papers/6b-crio/scisci_config.yaml

    # Run only specific modules:
    uv run python papers/6b-crio/scripts/scisci/run_all.py \\
        --config papers/6b-crio/scisci_config.yaml \\
        --modules m01 m02 m08

    # Dry run (load data only):
    uv run python papers/6b-crio/scripts/scisci/run_all.py \\
        --config papers/6b-crio/scisci_config.yaml --dry-run
"""

import argparse
import json
import os
import sys
import time
import traceback

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml not installed. Run: pip install pyyaml")
    sys.exit(1)

import pandas as pd

# Ensure the scisci package is importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from scisci.data_loader import load_and_engineer, check_terminal_year
from scisci.style import apply_style

# ── Module Registry ──────────────────────────────────────────────────────────
# Each entry: (module_id, import_path, human_name, section, placement, files)
# - section:   Target paper section for this module's results
# - placement: "main" = paper body, "supplement" = supplementary materials
# - files:     List of output files this module generates (figure + table)
MODULES = [
    # ── CORE: Main Paper Body ────────────────────────────────────────────────
    ("m01", "scisci.m01_descriptive", "Descriptive Statistics + VIF + Correlation",
     "§3 Table 1", "main",
     ["fig_correlation_matrix.png", "table_descriptive_stats.md", "table_vif.md"]),
    ("m02", "scisci.m02_chi_square", "Chi-Square Silo Test (H1)",
     "§4.1 H1", "main",
     ["fig_chi_square_silo.png", "table_chi_square_silo.md"]),
    ("m03", "scisci.m03_citation_diag", "Citation Distribution Diagnostic",
     "§3 / Appendix", "main",
     ["fig_citation_diagnostic.png"]),
    ("m05", "scisci.m05_neg_binomial", "Negative Binomial Regression (Robustness)",
     "§4.2 Robustness", "main",
     ["fig_negbin_coefficients.png", "table_negbin_results.md"]),
    ("m06", "scisci.m06_structural_break", "Structural Break Test (Chow)",
     "§4.4 Temporal", "main",
     ["fig_structural_break.png", "table_structural_break.md", "table_chow_test.md"]),
    ("m07", "scisci.m07_psm", "Propensity Score Matching (Causal)",
     "§4.2 H2", "main",
     ["fig_psm_balance.png", "table_psm_results.md"]),
    ("m08", "scisci.m08_causal_forest", "Causal Forest (HTE)",
     "§4.2 H2 (Lead)", "main",
     ["fig_causal_forest_hte.png", "table_causal_forest.md"]),
    ("m09", "scisci.m09_triple_helix", "Triple Helix Institutional Analysis",
     "§4.3 Institutional", "main",
     ["fig_triple_helix.png", "table_institutional_impact.md"]),
    # ── SUPPLEMENTARY: Robustness / Exploratory ──────────────────────────────
    ("m04", "scisci.m04_ols_logistic", "OLS + Logistic Regression (Legacy)",
     "Supplement Table S1", "supplement",
     ["fig_interaction_penalty.png", "table_regression_results.md"]),
    ("m10", "scisci.m10_network_mod", "Network Centrality Moderation",
     "Supplement", "supplement",
     ["fig_network_moderation.png", "table_network_moderation.md"]),
    ("m11", "scisci.m11_shap_analysis", "Random Forest + SHAP Feature Importance",
     "Supplement", "supplement",
     ["fig_shap_waterfall.png", "table_rf_metrics.md"]),
    ("m12", "scisci.m12_topic_forecast", "Topic Drift Forecasting (ARIMA)",
     "§5 / Supplement", "supplement",
     ["fig_topic_forecast.png", "table_topic_forecast.md"]),
    ("m13", "scisci.m13_bertopic_evo", "BERTopic Strategic Map Evolution",
     "Supplement", "supplement",
     ["fig_bertopic_strategic_map.png", "table_bertopic_strategic_map.md"]),
    # ── NEW: Lessons-Learned Remediation Modules (2026-05-28) ─────────────────
    ("m14", "scisci.m14_rosenbaum", "Rosenbaum Bounds Sensitivity Analysis",
     "§4.2 Sensitivity", "main",
     ["fig_rosenbaum_sensitivity.png", "table_rosenbaum_bounds.md"]),
    ("m15", "scisci.m15_semantic_break", "Semantic Structural Break (SPECTER2 Drift)",
     "§3.1 / §4.4", "main",
     ["fig_semantic_drift.png", "table_semantic_break.md"]),
    ("m16", "scisci.m16_bass_prophet", "Bass Diffusion + Changepoint Detection",
     "§4.5 Forecasting", "main",
     ["fig_bass_adoption.png", "fig_prophet_decomposition.png",
      "table_bass_diffusion.md", "table_prophet_changepoints.md"]),
    ("m17", "scisci.m17_monte_carlo_crio", "Monte Carlo CRIO Simulation",
     "§5.1 CRIO Framework", "main",
     ["fig_crio_monte_carlo.png", "table_crio_deterministic.md",
      "table_crio_monte_carlo.md", "table_crio_bloom_case.md"]),
]


def _assess_verdict(mod_id, result):
    """Auto-assess module verdict based on actual results.

    Returns (emoji, verdict, key_stat_str).
    """
    if not isinstance(result, dict) or "error" in result:
        return "❌", "ERROR", result.get("error", "Unknown")[:80] if isinstance(result, dict) else "—"

    # Module-specific verdict logic
    if mod_id == "m01":
        return "✅", "USE", f"N={result.get('n_papers', '?')}"
    elif mod_id == "m02":
        p = result.get("p", 1)
        chi2 = result.get("chi2", 0)
        v = "USE" if p < 0.05 else "WEAK"
        return ("✅" if v == "USE" else "⚠️"), v, f"χ²={chi2:.2f}, p={p:.4f}"
    elif mod_id == "m03":
        return "✅", "USE", "Validates log-transform"
    elif mod_id == "m04":
        return "⚠️", "DEMOTED", "Endogeneity concerns (3/3 reviewers)"
    elif mod_id == "m05":
        return "✅", "USE", f"NegBin fitted"
    elif mod_id == "m06":
        return "❌", "NULL", "No volume break detected"
    elif mod_id == "m07":
        att = result.get("att", 0)
        p = result.get("p", 1)
        v = "USE" if p < 0.05 else "WEAK"
        return ("✅" if v == "USE" else "⚠️"), v, f"ATT={att:.3f}, p={p:.4f}"
    elif mod_id == "m08":
        ate = result.get("ate", 0)
        return "✅", "USE", f"ATE={ate:.2f}"
    elif mod_id == "m09":
        acad = result.get("academic_pct", 0)
        return "✅", "USE", f"Academic={acad:.1f}%"
    elif mod_id == "m10":
        p = result.get("interaction_p", 1)
        return ("✅" if p < 0.05 else "❌"), ("USE" if p < 0.05 else "NULL"), f"p={p:.4f}"
    elif mod_id == "m11":
        cv_r2 = result.get("cv_r2_mean", 0)
        return ("✅" if cv_r2 > 0 else "❌"), ("USE" if cv_r2 > 0 else "FAIL"), f"CV-R²={cv_r2:.2f}"
    elif mod_id == "m12":
        return "⚠️", "WEAK", "DM non-significant"
    elif mod_id == "m13":
        n = result.get("n_topics", 0)
        return "⚠️", "VISUAL", f"{n} topic-slice entries"
    elif mod_id == "m14":
        g = result.get("gamma_star", 0)
        rob = result.get("robustness", "?")
        return "✅", "USE", f"Γ*={g:.1f} ({rob})"
    elif mod_id == "m15":
        yr = result.get("max_drift_year", "?")
        z = result.get("z_score", 0)
        sig = result.get("is_significant", False)
        return ("🌟" if sig else "⚠️"), ("STAR" if sig else "WEAK"), f"Break at {yr}, z={z:.2f}"
    elif mod_id == "m16":
        r2 = result.get("r2", 0)
        pct = result.get("penetration_pct", 0)
        return "🌟", "STAR", f"R²={r2:.3f}, {pct:.1f}% saturation"
    elif mod_id == "m17":
        mc_med = result.get("mc_eu_median", 0)
        p_neg = result.get("mc_eu_p_negative", 0)
        return "🌟", "STAR", f"EU median={mc_med:.3f}, P(neg)={p_neg:.1f}%"
    else:
        return "✅", "USE", "—"


def _generate_output_guide(out_dir, results, elapsed):
    """Auto-generate the comprehensive OUTPUT_GUIDE.md with verdicts and file mapping."""
    lines = [
        "# SciSci Econometric Analysis — Output Guide",
        "",
        f"> Auto-generated by `run_all.py` at {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"> Runtime: {elapsed:.1f}s | Dataset: 946 papers (Green AI × Economic Drivers, 2019–2026)",
        ">",
        "> **How to read:** Pick modules marked 🌟 STAR or ✅ USE for main paper.",
        "> ⚠️ WEAK goes to supplement with caveats. ❌ NULL/FAIL = report honestly or omit.",
        "",
        "---",
        "",
        "## 🏆 At-a-Glance: Which Modules Are Promising?",
        "",
        "| Module | Name | Verdict | Key Number | Paper Section |",
        "|:---|:---|:---:|:---|:---|",
    ]

    # Build at-a-glance table
    for mod_id, _, name, section, placement, _ in MODULES:
        result = results.get(mod_id, {})
        if isinstance(result, str) and result == "skipped":
            lines.append(f"| **{mod_id.upper()}** | {name} | ⏭️ Skipped | — | {section} |")
            continue
        emoji, verdict, key_stat = _assess_verdict(mod_id, result)
        lines.append(f"| **{mod_id.upper()}** | {name} | {emoji} **{verdict}** | {key_stat} | {section} |")

    lines.extend([
        "",
        "### Legend",
        "- 🌟 **STAR** = Novel, strong result — headline finding for the paper",
        "- ✅ **USE** = Solid evidence, should be in main text",
        "- ⚠️ **WEAK/DEMOTED** = Include with caveats or in supplement",
        "- ❌ **NULL/FAIL** = Report honestly or move to supplement",
        "",
        "---",
        "",
        "## 📁 File Inventory",
        "",
        "| Figure | Data Table | Module |",
        "|:---|:---|:---|",
    ])

    # Build file inventory
    for mod_id, _, _, _, _, files in MODULES:
        figs = [f for f in files if f.startswith("fig_")]
        tbls = [f for f in files if f.startswith("table_")]
        fig_str = ", ".join(f"`{f}`" for f in figs) if figs else "—"
        tbl_str = ", ".join(f"`{f}`" for f in tbls) if tbls else "—"
        lines.append(f"| {fig_str} | {tbl_str} | {mod_id.upper()} |")

    lines.extend(["", "---", ""])

    # Detailed per-module sections
    for mod_id, _, name, section, placement, files in MODULES:
        result = results.get(mod_id, {})
        if isinstance(result, str) and result == "skipped":
            emoji, verdict, key_stat = "⏭️", "SKIPPED", "—"
        elif isinstance(result, dict) and "error" in result:
            emoji, verdict, key_stat = "❌", "ERROR", result["error"][:80]
        else:
            emoji, verdict, key_stat = _assess_verdict(mod_id, result)

        place_tag = "MAIN" if placement == "main" else "SUPPLEMENT"
        lines.append(f"### {emoji} {mod_id.upper()}: {name} [{place_tag}]")
        lines.append(f"**Section:** {section}")
        lines.append(f"**Verdict:** {emoji} {verdict} — {key_stat}")
        lines.append("")

        # Key results
        if isinstance(result, dict) and "error" not in result:
            detail_pairs = [(k, v) for k, v in result.items()
                            if not isinstance(v, (list, dict, bytes))]
            if detail_pairs:
                lines.append("**Results:**")
                for k, v in detail_pairs:
                    if isinstance(v, float):
                        lines.append(f"- {k}: {v:.4f}")
                    else:
                        lines.append(f"- {k}: {v}")
                lines.append("")

        # Files for this module
        existing = [f for f in files if os.path.isfile(os.path.join(out_dir, f))]
        missing = [f for f in files if not os.path.isfile(os.path.join(out_dir, f))]
        if existing:
            lines.append("**Files:**")
            for f in existing:
                size = os.path.getsize(os.path.join(out_dir, f))
                lines.append(f"- ✅ `{f}` ({size:,} bytes)")
        if missing:
            for f in missing:
                lines.append(f"- ❌ `{f}` (MISSING)")
        lines.extend(["", "---", ""])

    # Paper integration roadmap
    lines.extend([
        "## 🗺️ Recommended Paper Integration Map",
        "",
        "### §3 — Data & Descriptive",
        "- M01 (Table 1, VIF, correlations)",
        "- M03 (Citation diagnostic — validates methodology)",
        "- M15 (Semantic break — 2019→2020 paradigm shift)",
        "",
        "### §4.1 — H1: Silo Hypothesis",
        "- M02 (Chi-square test)",
        "",
        "### §4.2 — H2: Causal Effects + Robustness",
        "- M08 (Causal Forest — lead evidence)",
        "- M05 (NegBin — robustness check)",
        "- M07 (PSM — report as association if weak)",
        "- M14 (Rosenbaum — sensitivity transparency)",
        "",
        "### §4.3 — Institutional Analysis",
        "- M09 (Triple Helix — institutional distribution)",
        "",
        "### §4.4 — Temporal Dynamics",
        "- M06 (Volume break — report null)",
        "- M15 (Semantic break — novel counterpoint)",
        "",
        "### §4.5 — Growth & Forecasting",
        "- M16 (Bass Diffusion + Prophet changepoints)",
        "",
        "### §5 — Discussion & Future Research",
        "- M16 (Bass saturation → field maturity narrative)",
        "- M09 (Institutional gap → policy implications)",
        "",
        "### Supplement",
        "- M04, M10, M11, M12, M13",
        "",
    ])

    with open(os.path.join(out_dir, "OUTPUT_GUIDE.md"), "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="SciSci Master Analysis Pipeline")
    parser.add_argument("--config", required=True, help="Path to scisci_config.yaml")
    parser.add_argument("--modules", nargs="*", default=None,
                        help="Run only specific modules (e.g., m01 m02 m08). Default: all.")
    parser.add_argument("--dry-run", action="store_true", help="Load data only, skip analyses.")
    parser.add_argument("--json", action="store_true", help="Print JSON summary at the end.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Resolve repo root
    p = os.path.abspath(args.config)
    repo_root = os.getcwd()
    while p != "/":
        p = os.path.dirname(p)
        if os.path.isdir(os.path.join(p, ".git")):
            repo_root = p
            break

    base = os.path.join(repo_root, cfg["project"]["base_dir"])
    data_path = os.path.join(base, cfg["project"]["dataset"])
    out_dir = os.path.join(base, cfg["project"].get("output_dir", "experiments/results/scisci"))
    os.makedirs(out_dir, exist_ok=True)

    apply_style()

    print(f"╔══ SciSci Master Pipeline: {cfg['project']['name']} ══╗")
    print(f"  Dataset:  {data_path}")
    print(f"  Output:   {out_dir}")

    df = pd.read_csv(data_path)
    df = load_and_engineer(df, cfg)
    ta = cfg["variables"]["theme_a"]["name"]
    tb = cfg["variables"]["theme_b"]["name"]
    print(f"  Papers:   {len(df)}")
    print(f"  {cfg['variables']['theme_a']['label']}: {df[ta].sum()}")
    print(f"  {cfg['variables']['theme_b']['label']}: {df[tb].sum()}")
    print()

    # Terminal year truncation diagnostic (Golden Rule #9)
    check_terminal_year(df)
    print()

    if args.dry_run:
        print("[DRY RUN] Data loaded successfully. Skipping analyses.")
        return

    selected = set(args.modules) if args.modules else {m[0] for m in MODULES}
    results = {}
    t0 = time.time()

    for mod_id, import_path, name, _, placement, _ in MODULES:
        if mod_id not in selected:
            results[mod_id] = "skipped"
            continue

        place_tag = "MAIN" if placement == "main" else "SUPPL"
        print(f"─── {mod_id.upper()} [{place_tag}]: {name} ───")
        try:
            mod = __import__(import_path, fromlist=["run"])
            result = mod.run(df, cfg, out_dir)
            results[mod_id] = result
            print(f"  ✅ Done\n")
        except Exception as e:
            results[mod_id] = {"error": str(e)}
            print(f"  ❌ FAILED: {e}\n")
            traceback.print_exc()

    elapsed = time.time() - t0
    _generate_output_guide(out_dir, results, elapsed)
    print(f"\n✅ All analyses complete in {elapsed:.1f}s")
    print(f"📖 Output Guide: {os.path.join(out_dir, 'OUTPUT_GUIDE.md')}")

    if args.json:
        print("\n" + json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
