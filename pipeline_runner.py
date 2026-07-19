"""
pipeline_runner.py
===================

End-to-end orchestrator for the adaptive rPPG preprocessing pipeline.

Pipeline flow:
    1. Input video (raw / degraded / real-world)
    2. Degradation estimation (no-reference, 5-axis feature vector)
    3. Router classification (predict best recipe from feature vector)
    4. Recipe application (enhance frames using selected recipe)
    5. rPPG extraction (face ROI → POS/CHROM → pulse waveform)
    6. Pulse metrics (HR, HRV, Pulse SNR, SQI)
    7. Optional: fidelity metrics (Track A only, if clean reference exists)

Also runs the three evaluation baselines:
    - Baseline 1: raw video, no enhancement (R0)
    - Baseline 2: fixed "globally best" recipe applied to everything
    - Baseline 3: adaptive router (our method)

Output: a JSON results file with all metrics for all baselines, plus
per-recipe diagnostics.

Author: Signal & Image Processing project (NMIMS Mumbai), MoSICom 2026.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from degradation_estimator import DegradationEstimator, FEATURE_NAMES
from degradation_injector import iter_frames, read_video_properties
from enhancement_recipes import RECIPE_IDS, apply_recipe_to_frame, get_recipe
from pulse_metrics import PulseAssessment, assess_pulse_signal
from rppg_extraction import extract_pulse_signal_from_frames

logger = logging.getLogger("pipeline_runner")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


# --------------------------------------------------------------------------- #
# Per-run result container
# --------------------------------------------------------------------------- #

@dataclass
class PipelineResult:
    video_path: str
    baseline_name: str          # "raw", "fixed_best", "adaptive"
    recipe_id: str
    features: dict
    hr_bpm: float
    hrv_rmssd_ms: Optional[float]
    pulse_snr_db: float
    sqi: float
    face_detection_rate: float
    hr_mae_vs_truth: Optional[float] = None
    processing_time_sec: float = 0.0

    def to_dict(self) -> dict:
        return {
            "video_path": self.video_path,
            "baseline_name": self.baseline_name,
            "recipe_id": self.recipe_id,
            "features": self.features,
            "hr_bpm": self.hr_bpm,
            "hrv_rmssd_ms": self.hrv_rmssd_ms,
            "pulse_snr_db": self.pulse_snr_db,
            "sqi": self.sqi,
            "face_detection_rate": self.face_detection_rate,
            "hr_mae_vs_truth": self.hr_mae_vs_truth,
            "processing_time_sec": self.processing_time_sec,
        }


# --------------------------------------------------------------------------- #
# Core pipeline step
# --------------------------------------------------------------------------- #

def run_single_pipeline(
    video_path: Path,
    recipe_id: str,
    baseline_name: str,
    true_hr_bpm: Optional[float] = None,
    rppg_method: str = "POS",
    estimator: Optional[DegradationEstimator] = None,
) -> PipelineResult:
    """Run the full pipeline for a single video with a specified recipe.
    If estimator is None, a new one is created."""
    import time
    t0 = time.time()

    video_path = Path(video_path)
    props = read_video_properties(video_path)
    frames = list(iter_frames(video_path))

    # 1. Degradation estimation (on RAW frames)
    if estimator is None:
        estimator = DegradationEstimator()
    features = estimator.estimate(video_path).to_dict()

    # 2. Apply recipe
    recipe = get_recipe(recipe_id)
    enhanced_frames = [apply_recipe_to_frame(f, recipe) for f in frames]

    # 3. rPPG extraction
    extraction = extract_pulse_signal_from_frames(enhanced_frames, props.fps, method=rppg_method)

    # 4. Pulse metrics
    assessment = assess_pulse_signal(
        extraction.pulse_signal, extraction.fps, extraction.face_detection_rate,
        true_hr_bpm=true_hr_bpm,
    )

    elapsed = time.time() - t0

    return PipelineResult(
        video_path=str(video_path),
        baseline_name=baseline_name,
        recipe_id=recipe_id,
        features=features,
        hr_bpm=assessment.hr_bpm,
        hrv_rmssd_ms=assessment.hrv_rmssd_ms,
        pulse_snr_db=assessment.pulse_snr_db,
        sqi=assessment.sqi,
        face_detection_rate=extraction.face_detection_rate,
        hr_mae_vs_truth=assessment.hr_mae_vs_truth,
        processing_time_sec=elapsed,
    )


# --------------------------------------------------------------------------- #
# Three-baseline comparison on one video
# --------------------------------------------------------------------------- #

def run_three_baselines(
    video_path: Path,
    router_model_path: Optional[Path] = None,
    fixed_recipe_id: str = "R1",
    true_hr_bpm: Optional[float] = None,
    rppg_method: str = "POS",
) -> list[PipelineResult]:
    """Run all three baselines on a single video:
        1. Raw (R0, no enhancement)
        2. Fixed-best recipe (user-specified or default)
        3. Adaptive router (if model provided)
    """
    from router_classifier import RouterClassifier

    video_path = Path(video_path)
    estimator = DegradationEstimator()
    results = []

    # Baseline 1: Raw
    logger.info("--- Baseline 1: RAW (R0) ---")
    results.append(run_single_pipeline(
        video_path, "R0", "raw",
        true_hr_bpm=true_hr_bpm, rppg_method=rppg_method, estimator=estimator,
    ))

    # Baseline 2: Fixed best
    logger.info("--- Baseline 2: FIXED BEST (%s) ---", fixed_recipe_id)
    results.append(run_single_pipeline(
        video_path, fixed_recipe_id, "fixed_best",
        true_hr_bpm=true_hr_bpm, rppg_method=rppg_method, estimator=estimator,
    ))

    # Baseline 3: Adaptive router
    if router_model_path is not None and router_model_path.exists():
        logger.info("--- Baseline 3: ADAPTIVE ROUTER ---")
        router = RouterClassifier.load(router_model_path)
        feature_vec = np.array([results[0].features[name] for name in FEATURE_NAMES], dtype=np.float64)
        predicted_recipe = router.predict(feature_vec)
        logger.info("Router predicted recipe: %s", predicted_recipe)
        results.append(run_single_pipeline(
            video_path, predicted_recipe, "adaptive",
            true_hr_bpm=true_hr_bpm, rppg_method=rppg_method, estimator=estimator,
        ))
    else:
        logger.warning("No router model provided; skipping adaptive baseline.")

    return results


# --------------------------------------------------------------------------- #
# Batch processing from manifest
# --------------------------------------------------------------------------- #

def run_batch_from_manifest(
    manifest_csv: Path,
    output_dir: Path,
    router_model_path: Optional[Path] = None,
    fixed_recipe_id: str = "R1",
    rppg_method: str = "POS",
) -> Path:
    """Process every video in a manifest through all three baselines.
    Writes results CSV and a summary JSON."""
    from dataset_loader import load_labeling_manifest

    manifest_csv = Path(manifest_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = load_labeling_manifest(manifest_csv)
    all_results = []

    for _, row in manifest_df.iterrows():
        video_path = Path(row["video_path"])
        true_hr = float(row["true_hr_bpm"]) if pd.notna(row.get("true_hr_bpm")) else None

        logger.info("Processing %s (true_hr=%s)", video_path.name,
                    f"{true_hr:.1f}" if true_hr is not None else "n/a")

        try:
            results = run_three_baselines(
                video_path,
                router_model_path=router_model_path,
                fixed_recipe_id=fixed_recipe_id,
                true_hr_bpm=true_hr,
                rppg_method=rppg_method,
            )
            all_results.extend(results)
        except Exception as e:
            logger.error("Failed on %s: %s", video_path, e)

    # Write results
    results_csv = output_dir / "pipeline_results.csv"
    if all_results:
        pd.DataFrame([r.to_dict() for r in all_results]).to_csv(results_csv, index=False)

    # Summary JSON
    summary = _compute_summary(all_results)
    summary_json = output_dir / "pipeline_summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Batch complete: %d results -> %s", len(all_results), output_dir)
    return results_csv


def _compute_summary(results: list[PipelineResult]) -> dict:
    """Aggregate summary statistics per baseline."""
    summary = {}
    for baseline in ["raw", "fixed_best", "adaptive"]:
        baseline_results = [r for r in results if r.baseline_name == baseline]
        if not baseline_results:
            continue
        mae_vals = [r.hr_mae_vs_truth for r in baseline_results if r.hr_mae_vs_truth is not None]
        snr_vals = [r.pulse_snr_db for r in baseline_results]
        sqi_vals = [r.sqi for r in baseline_results]
        summary[baseline] = {
            "n_clips": len(baseline_results),
            "mean_hr_mae": float(np.mean(mae_vals)) if mae_vals else None,
            "std_hr_mae": float(np.std(mae_vals)) if mae_vals else None,
            "mean_pulse_snr_db": float(np.mean(snr_vals)),
            "mean_sqi": float(np.mean(sqi_vals)),
            "mean_processing_time_sec": float(np.mean([r.processing_time_sec for r in baseline_results])),
        }
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="End-to-end adaptive rPPG preprocessing pipeline."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    single = sub.add_parser("single", help="Process one video through all baselines.")
    single.add_argument("--input", required=True, type=Path)
    single.add_argument("--router-model", type=Path, default=None)
    single.add_argument("--fixed-recipe", default="R1", choices=RECIPE_IDS)
    single.add_argument("--true-hr", type=float, default=None)
    single.add_argument("--rppg-method", choices=["POS", "CHROM"], default="POS")
    single.add_argument("--json-out", type=Path, default=None)

    batch = sub.add_parser("batch", help="Process a full manifest.")
    batch.add_argument("--manifest", required=True, type=Path)
    batch.add_argument("--output-dir", required=True, type=Path)
    batch.add_argument("--router-model", type=Path, default=None)
    batch.add_argument("--fixed-recipe", default="R1", choices=RECIPE_IDS)
    batch.add_argument("--rppg-method", choices=["POS", "CHROM"], default="POS")

    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.mode == "single":
        results = run_three_baselines(
            args.input,
            router_model_path=args.router_model,
            fixed_recipe_id=args.fixed_recipe,
            true_hr_bpm=args.true_hr,
            rppg_method=args.rppg_method,
        )
        output = [r.to_dict() for r in results]
        print(json.dumps(output, indent=2))
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump(output, f, indent=2)

    elif args.mode == "batch":
        run_batch_from_manifest(
            args.manifest, args.output_dir,
            router_model_path=args.router_model,
            fixed_recipe_id=args.fixed_recipe,
            rppg_method=args.rppg_method,
        )


if __name__ == "__main__":
    main()
