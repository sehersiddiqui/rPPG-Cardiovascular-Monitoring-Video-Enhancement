"""
fairness_eval.py
=================

Aggregates rPPG pipeline results by Fitzpatrick skin-tone group and
degradation severity bin, computing the fairness-gap (ΔMAE) metric that
is the core fairness claim of the paper.

The fairness-gap is defined as:
    ΔMAE = max_{group} MAE(group) - min_{group} MAE(group)

A smaller ΔMAE means the system performs more equitably across skin tones.
We report ΔMAE for each baseline (raw, fixed_best, adaptive) to show that
the adaptive router narrows the gap relative to both naive baselines.

Also reports per-group:
    - Mean HR MAE
    - Mean Pulse SNR
    - Mean SQI
    - Sample count

Author: Signal & Image Processing project (NMIMS Mumbai), MoSICom 2026.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("fairness_eval")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


# --------------------------------------------------------------------------- #
# Degradation severity binning
# --------------------------------------------------------------------------- #

def bin_degradation_severity(features_df: pd.DataFrame) -> pd.Series:
    """Assign each clip to a degradation severity bin based on the maximum
    of its five degradation scores. Bins: clean, mild, moderate, severe."""
    score_cols = ["noise_score", "blur_score", "compression_score",
                  "illumination_score", "motion_score"]
    available = [c for c in score_cols if c in features_df.columns]
    if not available:
        return pd.Series(["unknown"] * len(features_df), index=features_df.index)

    max_score = features_df[available].max(axis=1)
    bins = pd.cut(max_score, bins=[-0.1, 0.15, 0.4, 0.7, 1.1],
                  labels=["clean", "mild", "moderate", "severe"])
    return bins


# --------------------------------------------------------------------------- #
# Per-group aggregation
# --------------------------------------------------------------------------- #

@dataclass
class GroupMetrics:
    group_label: str
    n_samples: int
    mean_hr_mae: float
    std_hr_mae: float
    mean_pulse_snr_db: float
    mean_sqi: float
    mean_face_detection_rate: float

    def to_dict(self) -> dict:
        return {
            "group_label": self.group_label,
            "n_samples": self.n_samples,
            "mean_hr_mae": self.mean_hr_mae,
            "std_hr_mae": self.std_hr_mae,
            "mean_pulse_snr_db": self.mean_pulse_snr_db,
            "mean_sqi": self.mean_sqi,
            "mean_face_detection_rate": self.mean_face_detection_rate,
        }


def aggregate_by_group(
    results_df: pd.DataFrame,
    group_col: str,
) -> list[GroupMetrics]:
    """Aggregate metrics by a grouping column (e.g. fitzpatrick or severity)."""
    groups = []
    for label, gdf in results_df.groupby(group_col):
        finite_mae = gdf[gdf["hr_mae_vs_truth"].notna()]["hr_mae_vs_truth"]
        groups.append(GroupMetrics(
            group_label=str(label),
            n_samples=len(gdf),
            mean_hr_mae=float(np.mean(finite_mae)) if len(finite_mae) > 0 else float("nan"),
            std_hr_mae=float(np.std(finite_mae)) if len(finite_mae) > 1 else 0.0,
            mean_pulse_snr_db=float(np.mean(gdf["pulse_snr_db"])),
            mean_sqi=float(np.mean(gdf["sqi"])),
            mean_face_detection_rate=float(np.mean(gdf["face_detection_rate"])),
        ))
    return groups


# --------------------------------------------------------------------------- #
# Fairness gap computation
# --------------------------------------------------------------------------- #

def compute_fairness_gap(group_metrics: list[GroupMetrics]) -> Optional[float]:
    """ΔMAE = max(group MAE) - min(group MAE). Returns None if any group
    has no valid MAE."""
    maes = [g.mean_hr_mae for g in group_metrics if not np.isnan(g.mean_hr_mae)]
    if len(maes) < 2:
        return None
    return float(max(maes) - min(maes))


# --------------------------------------------------------------------------- #
# Full fairness evaluation
# --------------------------------------------------------------------------- #

def run_fairness_evaluation(
    results_csv: Path,
    output_dir: Path,
) -> dict:
    """Main entry point: reads pipeline results, aggregates by Fitzpatrick
    skin-tone and by degradation severity, computes fairness gaps, and
    writes report JSON + CSVs."""
    results_csv = Path(results_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_csv)

    # Parse features JSON if stored as string
    if "features" in df.columns and isinstance(df["features"].iloc[0], str):
        features_df = pd.json_normalize(df["features"].apply(json.loads))
        for col in features_df.columns:
            df[col] = features_df[col].values

    # Add degradation severity bin
    df["degradation_severity"] = bin_degradation_severity(df)

    report = {}

    for baseline in df["baseline_name"].unique():
        baseline_df = df[df["baseline_name"] == baseline]
        baseline_report = {}

        # By Fitzpatrick skin-tone
        if "fitzpatrick" in baseline_df.columns and baseline_df["fitzpatrick"].notna().any():
            fp_groups = aggregate_by_group(baseline_df, "fitzpatrick")
            baseline_report["by_fitzpatrick"] = [g.to_dict() for g in fp_groups]
            baseline_report["fairness_gap_mae_fitzpatrick"] = compute_fairness_gap(fp_groups)
        else:
            baseline_report["by_fitzpatrick"] = []
            baseline_report["fairness_gap_mae_fitzpatrick"] = None

        # By degradation severity
        sev_groups = aggregate_by_group(baseline_df, "degradation_severity")
        baseline_report["by_severity"] = [g.to_dict() for g in sev_groups]
        baseline_report["fairness_gap_mae_severity"] = compute_fairness_gap(sev_groups)

        # Overall
        finite_mae = baseline_df[baseline_df["hr_mae_vs_truth"].notna()]["hr_mae_vs_truth"]
        baseline_report["overall"] = {
            "n_samples": len(baseline_df),
            "mean_hr_mae": float(np.mean(finite_mae)) if len(finite_mae) > 0 else None,
            "mean_pulse_snr_db": float(np.mean(baseline_df["pulse_snr_db"])),
            "mean_sqi": float(np.mean(baseline_df["sqi"])),
        }

        report[baseline] = baseline_report

    # Write report
    report_json = output_dir / "fairness_report.json"
    with open(report_json, "w") as f:
        json.dump(report, f, indent=2)

    # Write per-baseline CSVs
    for baseline in report:
        if report[baseline]["by_fitzpatrick"]:
            pd.DataFrame(report[baseline]["by_fitzpatrick"]).to_csv(
                output_dir / f"fairness_{baseline}_by_fitzpatrick.csv", index=False
            )
        pd.DataFrame(report[baseline]["by_severity"]).to_csv(
            output_dir / f"fairness_{baseline}_by_severity.csv", index=False
        )

    logger.info("Fairness evaluation complete -> %s", output_dir)
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate fairness metrics across skin-tone groups and degradation severities."
    )
    parser.add_argument("--results-csv", required=True, type=Path,
                        help="Pipeline results CSV from pipeline_runner.py")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    report = run_fairness_evaluation(args.results_csv, args.output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
