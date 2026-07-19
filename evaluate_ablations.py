"""
evaluate_ablations.py
======================

Runs and tabulates the three-way comparison that is the core experimental
result of the paper:

    1. Raw baseline        – degraded video straight into rPPG, no enhancement
    2. Fixed-best baseline – single best recipe (from labeling_harness.py
                             diagnostics) applied uniformly to all inputs
    3. Adaptive router     – our method: degradation-aware per-input recipe
                             selection via the trained classifier

Produces:
    - A LaTeX-ready results table
    - A bar chart comparing HR MAE, Pulse SNR, and SQI across baselines
    - A scatter plot showing fidelity (SSIM) vs. physiology (HR MAE) to
      visualize where the two objectives diverge
    - Per-degradation-type and per-severity breakdown tables

All figures are saved as publication-quality PNGs (300 DPI).

Author: Signal & Image Processing project (NMIMS Mumbai), MoSICom 2026.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger("evaluate_ablations")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10


# --------------------------------------------------------------------------- #
# Table generation
# --------------------------------------------------------------------------- #

def generate_results_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean ± std for HR MAE, Pulse SNR, and SQI per baseline."""
    rows = []
    for baseline in ["raw", "fixed_best", "adaptive"]:
        bdf = results_df[results_df["baseline_name"] == baseline]
        if len(bdf) == 0:
            continue
        mae_vals = bdf[bdf["hr_mae_vs_truth"].notna()]["hr_mae_vs_truth"]
        rows.append({
            "Baseline": baseline.replace("_", " ").title(),
            "N clips": len(bdf),
            "HR MAE (bpm)": f"{mae_vals.mean():.2f} ± {mae_vals.std():.2f}" if len(mae_vals) > 0 else "n/a",
            "Pulse SNR (dB)": f"{bdf['pulse_snr_db'].mean():.2f} ± {bdf['pulse_snr_db'].std():.2f}",
            "SQI": f"{bdf['sqi'].mean():.3f} ± {bdf['sqi'].std():.3f}",
            "Face Det. Rate": f"{bdf['face_detection_rate'].mean():.2%}",
        })
    return pd.DataFrame(rows)


def generate_per_severity_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Break down results by degradation severity and baseline."""
    if "degradation_severity" not in results_df.columns:
        return pd.DataFrame()
    rows = []
    for severity in ["clean", "mild", "moderate", "severe"]:
        for baseline in ["raw", "fixed_best", "adaptive"]:
            sdf = results_df[(results_df["degradation_severity"] == severity) &
                             (results_df["baseline_name"] == baseline)]
            if len(sdf) == 0:
                continue
            mae_vals = sdf[sdf["hr_mae_vs_truth"].notna()]["hr_mae_vs_truth"]
            rows.append({
                "Severity": severity.title(),
                "Baseline": baseline.replace("_", " ").title(),
                "N": len(sdf),
                "HR MAE": f"{mae_vals.mean():.2f}" if len(mae_vals) > 0 else "n/a",
                "Pulse SNR": f"{sdf['pulse_snr_db'].mean():.2f}",
                "SQI": f"{sdf['sqi'].mean():.3f}",
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figure 1: Baseline comparison bar chart
# --------------------------------------------------------------------------- #

def plot_baseline_comparison(results_df: pd.DataFrame, output_path: Path) -> None:
    """Bar chart: HR MAE, Pulse SNR, SQI across the three baselines."""
    baselines = ["raw", "fixed_best", "adaptive"]
    labels = ["Raw", "Fixed Best", "Adaptive"]
    colors = ["#e74c3c", "#f39c12", "#27ae60"]

    mae_means, mae_stds = [], []
    snr_means, snr_stds = [], []
    sqi_means, sqi_stds = [], []

    for b in baselines:
        bdf = results_df[results_df["baseline_name"] == b]
        mae_vals = bdf[bdf["hr_mae_vs_truth"].notna()]["hr_mae_vs_truth"]
        mae_means.append(mae_vals.mean() if len(mae_vals) > 0 else 0)
        mae_stds.append(mae_vals.std() if len(mae_vals) > 1 else 0)
        snr_means.append(bdf["pulse_snr_db"].mean())
        snr_stds.append(bdf["pulse_snr_db"].std())
        sqi_means.append(bdf["sqi"].mean())
        sqi_stds.append(bdf["sqi"].std())

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # HR MAE (lower is better)
    ax = axes[0]
    bars = ax.bar(labels, mae_means, yerr=mae_stds, color=colors, capsize=5, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("HR MAE (bpm)")
    ax.set_title("Heart Rate Error")
    ax.set_ylim(bottom=0)
    for bar, val in zip(bars, mae_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    # Pulse SNR (higher is better)
    ax = axes[1]
    bars = ax.bar(labels, snr_means, yerr=snr_stds, color=colors, capsize=5, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Pulse SNR (dB)")
    ax.set_title("Pulse Signal Quality")
    for bar, val in zip(bars, snr_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    # SQI (higher is better)
    ax = axes[2]
    bars = ax.bar(labels, sqi_means, yerr=sqi_stds, color=colors, capsize=5, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("SQI")
    ax.set_title("Signal Quality Index")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, sqi_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Three-Way Baseline Comparison", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved baseline comparison figure -> %s", output_path)


# --------------------------------------------------------------------------- #
# Figure 2: Fidelity vs. Physiology scatter
# --------------------------------------------------------------------------- #

def plot_fidelity_vs_physiology(
    results_df: pd.DataFrame,
    fidelity_csv: Optional[Path],
    output_path: Path,
) -> None:
    """Scatter plot: SSIM (fidelity) vs. HR MAE (physiology), colored by
    baseline. Shows where fidelity-optimal and physiology-optimal diverge."""
    if fidelity_csv is None or not fidelity_csv.exists():
        logger.warning("No fidelity CSV provided; skipping fidelity-vs-physiology plot.")
        return

    fidelity_df = pd.read_csv(fidelity_csv)
    # Merge on degraded_path ≈ video_path
    # This is a simplified merge; exact matching depends on path conventions
    merged = results_df.copy()
    merged["ssim"] = np.nan

    for _, frow in fidelity_df.iterrows():
        degraded_name = Path(frow["degraded_path"]).name
        mask = merged["video_path"].str.contains(degraded_name, regex=False, na=False)
        merged.loc[mask, "ssim"] = frow.get("ssim_mean", np.nan)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"raw": "#e74c3c", "fixed_best": "#f39c12", "adaptive": "#27ae60"}
    markers = {"raw": "o", "fixed_best": "s", "adaptive": "D"}

    for baseline in ["raw", "fixed_best", "adaptive"]:
        bdf = merged[merged["baseline_name"] == baseline]
        valid = bdf[bdf["ssim"].notna() & bdf["hr_mae_vs_truth"].notna()]
        if len(valid) == 0:
            continue
        ax.scatter(valid["ssim"], valid["hr_mae_vs_truth"],
                   c=colors.get(baseline, "gray"),
                   marker=markers.get(baseline, "o"),
                   label=baseline.replace("_", " ").title(),
                   alpha=0.6, s=40, edgecolors="black", linewidth=0.3)

    ax.set_xlabel("SSIM (vs. clean reference)")
    ax.set_ylabel("HR MAE (bpm)")
    ax.set_title("Fidelity vs. Physiological Accuracy")
    ax.legend(title="Baseline")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved fidelity-vs-physiology figure -> %s", output_path)


# --------------------------------------------------------------------------- #
# Figure 3: Fairness gap comparison
# --------------------------------------------------------------------------- #

def plot_fairness_gap(fairness_report: dict, output_path: Path) -> None:
    """Bar chart comparing fairness gaps (ΔMAE) across baselines."""
    baselines = ["raw", "fixed_best", "adaptive"]
    labels = ["Raw", "Fixed Best", "Adaptive"]
    colors = ["#e74c3c", "#f39c12", "#27ae60"]

    gaps = []
    for b in baselines:
        gap = fairness_report.get(b, {}).get("fairness_gap_mae_fitzpatrick")
        gaps.append(gap if gap is not None else 0)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, gaps, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Fairness Gap ΔMAE (bpm)")
    ax.set_title("Fairness Gap Across Skin-Tone Groups")
    ax.set_ylim(bottom=0)
    for bar, val in zip(bars, gaps):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved fairness gap figure -> %s", output_path)


# --------------------------------------------------------------------------- #
# Figure 4: Per-degradation-type breakdown
# --------------------------------------------------------------------------- #

def plot_per_degradation_type(results_df: pd.DataFrame, output_path: Path) -> None:
    """Grouped bar chart showing HR MAE broken down by which degradation
    type was dominant in the clip, per baseline."""
    # Determine dominant degradation from features
    score_cols = ["noise_score", "blur_score", "compression_score",
                  "illumination_score", "motion_score"]
    available = [c for c in score_cols if c in results_df.columns]
    if not available:
        logger.warning("No degradation feature columns found; skipping per-type plot.")
        return

    results_df = results_df.copy()
    results_df["dominant_degradation"] = results_df[available].idxmax(axis=1).str.replace("_score", "")

    deg_types = ["noise", "blur", "compression", "illumination", "motion"]
    baselines = ["raw", "fixed_best", "adaptive"]
    colors = {"raw": "#e74c3c", "fixed_best": "#f39c12", "adaptive": "#27ae60"}

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(deg_types))
    width = 0.25

    for i, baseline in enumerate(baselines):
        mae_by_type = []
        for deg in deg_types:
            subset = results_df[(results_df["baseline_name"] == baseline) &
                                (results_df["dominant_degradation"] == deg)]
            mae_vals = subset[subset["hr_mae_vs_truth"].notna()]["hr_mae_vs_truth"]
            mae_by_type.append(mae_vals.mean() if len(mae_vals) > 0 else 0)
        ax.bar(x + i * width, mae_by_type, width, label=baseline.replace("_", " ").title(),
               color=colors[baseline], edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Dominant Degradation Type")
    ax.set_ylabel("HR MAE (bpm)")
    ax.set_title("HR Error by Dominant Degradation Type")
    ax.set_xticks(x + width)
    ax.set_xticklabels([d.title() for d in deg_types])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved per-degradation-type figure -> %s", output_path)


# --------------------------------------------------------------------------- #
# Main evaluation orchestrator
# --------------------------------------------------------------------------- #

def run_full_evaluation(
    results_csv: Path,
    output_dir: Path,
    fidelity_csv: Optional[Path] = None,
    fairness_report_json: Optional[Path] = None,
) -> Path:
    """Generate all tables and figures for the paper's results section."""
    results_csv = Path(results_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.read_csv(results_csv)

    # Parse features if stored as JSON strings
    if "features" in results_df.columns and isinstance(results_df["features"].iloc[0], str):
        features_df = pd.json_normalize(results_df["features"].apply(json.loads))
        for col in features_df.columns:
            results_df[col] = features_df[col].values

    # 1. Main results table
    main_table = generate_results_table(results_df)
    main_table.to_csv(output_dir / "table_main_results.csv", index=False)
    main_table.to_latex(output_dir / "table_main_results.tex", index=False, float_format="%.3f")
    logger.info("Main results table:
%s", main_table.to_string(index=False))

    # 2. Per-severity table
    sev_table = generate_per_severity_table(results_df)
    if not sev_table.empty:
        sev_table.to_csv(output_dir / "table_per_severity.csv", index=False)
        sev_table.to_latex(output_dir / "table_per_severity.tex", index=False)
        logger.info("Per-severity table:
%s", sev_table.to_string(index=False))

    # 3. Figures
    plot_baseline_comparison(results_df, output_dir / "fig_baseline_comparison.png")
    plot_fidelity_vs_physiology(results_df, fidelity_csv, output_dir / "fig_fidelity_vs_physiology.png")
    plot_per_degradation_type(results_df, output_dir / "fig_per_degradation_type.png")

    # 4. Fairness gap figure
    if fairness_report_json is not None and fairness_report_json.exists():
        with open(fairness_report_json) as f:
            fairness_report = json.load(f)
        plot_fairness_gap(fairness_report, output_dir / "fig_fairness_gap.png")

    logger.info("All evaluation outputs written to %s", output_dir)
    return output_dir


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate paper results tables and figures from pipeline outputs."
    )
    parser.add_argument("--results-csv", required=True, type=Path,
                        help="Pipeline results CSV from pipeline_runner.py")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fidelity-csv", type=Path, default=None,
                        help="Optional fidelity metrics CSV for fidelity-vs-physiology plot")
    parser.add_argument("--fairness-report", type=Path, default=None,
                        help="Optional fairness_report.json for fairness gap plot")
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    run_full_evaluation(
        args.results_csv, args.output_dir,
        fidelity_csv=args.fidelity_csv,
        fairness_report_json=args.fairness_report,
    )


if __name__ == "__main__":
    main()
