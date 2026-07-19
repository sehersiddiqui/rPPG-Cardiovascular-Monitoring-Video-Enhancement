"""
run_full_pipeline.py
=====================

Master orchestration script that runs the entire adaptive rPPG
preprocessing pipeline from start to finish:

    1. Build Track A synthetic dataset (degradation_injector.py)
    2. Compute fidelity metrics (fidelity_metrics.py)
    3. Run labeling harness to get physiology-grounded labels (labeling_harness.py)
    4. Train the router classifier (router_classifier.py)
    5. Run end-to-end pipeline with 3 baselines (pipeline_runner.py)
    6. Evaluate fairness (fairness_eval.py)
    7. Generate paper tables and figures (evaluate_ablations.py)

Usage:
    python run_full_pipeline.py --config config.json

Author: Signal & Image Processing project (NMIMS Mumbai), MoSICom 2026.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("run_full_pipeline")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def run_step(name: str, cmd: list[str]) -> None:
    """Run a pipeline step via subprocess, logging output."""
    logger.info("=" * 60)
    logger.info("STEP: %s", name)
    logger.info("Command: %s", " ".join(cmd))
    logger.info("=" * 60)
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        logger.error("Step '%s' failed with exit code %d", name, result.returncode)
        sys.exit(result.returncode)
    logger.info("Step '%s' completed successfully.\n", name)


def main():
    parser = argparse.ArgumentParser(description="Run the full adaptive rPPG pipeline.")
    parser.add_argument("--config", type=Path, default=None,
                        help="JSON config file with paths and parameters")
    parser.add_argument("--clean-video-dir", type=Path, default=Path("data/clean"))
    parser.add_argument("--track-a-output", type=Path, default=Path("outputs/track_a"))
    parser.add_argument("--labeling-output", type=Path, default=Path("outputs/labeling"))
    parser.add_argument("--router-model", type=Path, default=Path("outputs/router_model.joblib"))
    parser.add_argument("--pipeline-output", type=Path, default=Path("outputs/pipeline"))
    parser.add_argument("--paper-output", type=Path, default=Path("outputs/paper"))
    parser.add_argument("--skip-track-a", action="store_true",
                        help="Skip Track A generation (use existing)")
    parser.add_argument("--skip-labeling", action="store_true",
                        help="Skip labeling harness (use existing training table)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip router training (use existing model)")
    args = parser.parse_args()

    if args.config and args.config.exists():
        with open(args.config) as f:
            config = json.load(f)
        # Override args with config values
        for key, val in config.items():
            if hasattr(args, key):
                setattr(args, key, Path(val) if isinstance(val, str) and "/" in val else val)

    # Step 1: Build Track A synthetic dataset
    if not args.skip_track_a:
        run_step("Track A Dataset Generation", [
            sys.executable, "degradation_injector.py",
            "sweep",
            "--input-dir", str(args.clean_video_dir),
            "--output-dir", str(args.track_a_output),
        ])

    # Step 2: Compute fidelity metrics
    manifest = args.track_a_output / "track_a_manifest.csv"
    if manifest.exists():
        run_step("Fidelity Metrics", [
            sys.executable, "fidelity_metrics.py",
            "batch",
            "--manifest", str(manifest),
            "--output-csv", str(args.track_a_output / "fidelity_metrics.csv"),
        ])

    # Step 3: Labeling harness (brute-force recipes → physiology labels)
    if not args.skip_labeling:
        # First, create manifest from Track A outputs
        # (In practice, you'd also include MMPD real-world data here)
        run_step("Labeling Harness", [
            sys.executable, "labeling_harness.py",
            "run",
            "--manifest", str(manifest),
            "--output-dir", str(args.labeling_output),
            "--rppg-method", "POS",
        ])

    # Step 4: Train router classifier
    training_table = args.labeling_output / "router_training_table.csv"
    if not args.skip_training and training_table.exists():
        run_step("Router Training", [
            sys.executable, "router_classifier.py",
            "train",
            "--train-csv", str(training_table),
            "--model-type", "random_forest",
            "--output-model", str(args.router_model),
            "--report-dir", str(args.labeling_output / "router_report"),
        ])

    # Step 5: Run pipeline with 3 baselines
    if manifest.exists() and args.router_model.exists():
        run_step("Pipeline Evaluation", [
            sys.executable, "pipeline_runner.py",
            "batch",
            "--manifest", str(manifest),
            "--output-dir", str(args.pipeline_output),
            "--router-model", str(args.router_model),
            "--fixed-recipe", "R1",
            "--rppg-method", "POS",
        ])

    # Step 6: Fairness evaluation
    results_csv = args.pipeline_output / "pipeline_results.csv"
    if results_csv.exists():
        run_step("Fairness Evaluation", [
            sys.executable, "fairness_eval.py",
            "--results-csv", str(results_csv),
            "--output-dir", str(args.pipeline_output / "fairness"),
        ])

    # Step 7: Generate paper figures and tables
    fairness_report = args.pipeline_output / "fairness" / "fairness_report.json"
    fidelity_csv = args.track_a_output / "fidelity_metrics.csv"
    if results_csv.exists():
        cmd = [
            sys.executable, "evaluate_ablations.py",
            "--results-csv", str(results_csv),
            "--output-dir", str(args.paper_output),
        ]
        if fidelity_csv.exists():
            cmd.extend(["--fidelity-csv", str(fidelity_csv)])
        if fairness_report.exists():
            cmd.extend(["--fairness-report", str(fairness_report)])
        run_step("Paper Figures & Tables", cmd)

    logger.info("=" * 60)
    logger.info("FULL PIPELINE COMPLETE")
    logger.info("Outputs in: %s", args.paper_output)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
