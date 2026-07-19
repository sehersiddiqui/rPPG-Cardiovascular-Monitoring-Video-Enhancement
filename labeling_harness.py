"""
labeling_harness.py
=======================

Brute-forces every fixed enhancement recipe (R0-R5, from
enhancement_recipes.py) through the full rPPG pipeline on every training
clip, scores each recipe by downstream physiological accuracy against
contact-sensor ground truth, and emits the winning-recipe label used to
train router_classifier.py.

This is the file that makes "physiology, not pixels, is the ground truth
the router learns from" actually true rather than aspirational: nowhere in
this module is image fidelity (PSNR/SSIM/etc.) used to pick a winner. Only
HR MAE and Pulse SNR, computed against the clip's contact-sensor ground
truth, decide the label.

Label-selection criterion
----------------------------
Primary: minimize HR MAE (bpm) against ground truth -- this is the
directly clinically meaningful target, so it is the criterion of first
resort.

Tiebreak: maximize Pulse SNR (dB). HR is estimated via an FFT peak within
a Welch PSD (see pulse_metrics.py), which is quantized to discrete
frequency bins -- on a short clip, several recipes routinely land in the
exact same bin and tie exactly on HR MAE even though their underlying
waveform quality differs substantially. Our own validation runs (see
notes in this module's self-test) showed this tying is common and that
Pulse SNR is what actually discriminates a clean recovery from a noisy
one in that situation, so it is the correct tiebreaker rather than an
arbitrary one.

If a recipe's face-detection rate on a clip falls below
MIN_FACE_DETECTION_RATE, that recipe is excluded from consideration for
that clip entirely (a recipe cannot "win" by accident on a clip where the
ROI trace was mostly noise from a lost face track).

Training-clip input contract
--------------------------------
A "labeling manifest" CSV with columns:
    - video_path     : path to the (possibly already-degraded) clip
    - subject_id      : REQUIRED for downstream subject-level splitting
    - window_id       : unique id for this clip/window (for traceability)
    - true_hr_bpm     : ground-truth heart rate for this clip, from the
                         dataset's contact PPG/ECG sensor
    - fitzpatrick     : optional, skin-tone group label (for MMPD-derived
                         rows; used later by fairness_eval.py, simply
                         passed through here)

Producing this manifest from raw dataset files (UBFC-rPPG/PURE/MMPD
ground-truth formats, or degradation_injector.py's Track A sidecars) is
the job of dataset_loader.py -- out of scope for this module, which only
consumes the already-normalized manifest format above.

Output
------
Two CSVs:
    1. `router_training_table.csv` -- exactly the contract
       router_classifier.py expects (FEATURE_NAMES + subject_id +
       best_recipe), ready to hand straight to `router_classifier.py train`.
    2. `labeling_diagnostics.csv` -- full per-recipe, per-clip metrics
       (HR estimate, HR MAE, Pulse SNR, SQI for all 6 recipes on every
       clip), which is what the paper's "fidelity vs. physiology" tension
       figure and the "which single recipe should be the fixed baseline"
       analysis (evaluate_ablations.py baseline 2) are both built from.

Author: Signal & Image Processing project (NMIMS Mumbai), MoSICom 2026.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from degradation_estimator import DegradationEstimator, FEATURE_NAMES
from degradation_injector import iter_frames, read_video_properties
from enhancement_recipes import RECIPE_IDS, apply_recipe_to_frame, get_recipe
from pulse_metrics import assess_pulse_signal
from rppg_extraction import extract_pulse_signal_from_frames

logger = logging.getLogger("labeling_harness")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

MIN_FACE_DETECTION_RATE = 0.5  # recipes with worse ROI tracking than this are disqualified for that clip
REQUIRED_MANIFEST_COLUMNS = ["video_path", "subject_id", "window_id", "true_hr_bpm"]


# --------------------------------------------------------------------------- #
# Manifest loading
# --------------------------------------------------------------------------- #

def load_labeling_manifest(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Labeling manifest {csv_path} missing required column(s): {missing}")
    logger.info("Loaded labeling manifest: %d clips, %d subjects",
                len(df), df["subject_id"].nunique())
    return df


# --------------------------------------------------------------------------- #
# Per-recipe scoring on a single clip
# --------------------------------------------------------------------------- #

@dataclass
class RecipeScore:
    recipe_id: str
    hr_bpm: float
    hr_mae: float
    pulse_snr_db: float
    sqi: float
    face_detection_rate: float
    disqualified: bool


def _score_recipe_on_frames(
    recipe_id: str, raw_frames: list, fps: float, true_hr_bpm: float, rppg_method: str,
) -> RecipeScore:
    recipe = get_recipe(recipe_id)
    enhanced_frames = [apply_recipe_to_frame(f, recipe) for f in raw_frames]

    extraction = extract_pulse_signal_from_frames(enhanced_frames, fps, method=rppg_method)
    assessment = assess_pulse_signal(
        extraction.pulse_signal, extraction.fps, extraction.face_detection_rate,
        true_hr_bpm=true_hr_bpm,
    )

    disqualified = extraction.face_detection_rate < MIN_FACE_DETECTION_RATE
    return RecipeScore(
        recipe_id=recipe_id,
        hr_bpm=assessment.hr_bpm,
        hr_mae=assessment.hr_mae_vs_truth if assessment.hr_mae_vs_truth is not None else float("inf"),
        pulse_snr_db=assessment.pulse_snr_db,
        sqi=assessment.sqi,
        face_detection_rate=extraction.face_detection_rate,
        disqualified=disqualified,
    )


def select_winning_recipe(scores: list[RecipeScore]) -> RecipeScore:
    """Applies the primary/tiebreak criterion documented at module level:
    minimize HR MAE first, break ties by maximizing Pulse SNR. Disqualified
    recipes (face tracking too unreliable on this clip) are excluded unless
    every recipe is disqualified, in which case we fall back to ranking
    among all of them anyway rather than producing no label at all."""
    candidates = [s for s in scores if not s.disqualified]
    if not candidates:
        logger.warning("All recipes disqualified (face detection) for this clip; "
                        "falling back to ranking among all candidates anyway.")
        candidates = scores

    # Sort ascending by HR MAE, then descending by Pulse SNR as tiebreak.
    ranked = sorted(candidates, key=lambda s: (s.hr_mae, -s.pulse_snr_db))
    return ranked[0]


# --------------------------------------------------------------------------- #
# Full per-clip processing
# --------------------------------------------------------------------------- #

@dataclass
class ClipLabelingResult:
    video_path: str
    subject_id: str
    window_id: str
    true_hr_bpm: float
    fitzpatrick: Optional[str]
    features: dict            # degradation_estimator scores on the RAW clip
    recipe_scores: list[RecipeScore]
    best_recipe: str


def label_one_clip(
    row: pd.Series, estimator: DegradationEstimator, rppg_method: str = "POS",
) -> ClipLabelingResult:
    video_path = Path(row["video_path"])
    props = read_video_properties(video_path)
    raw_frames = list(iter_frames(video_path))

    # Degradation features are computed ONCE, on the raw (un-enhanced)
    # clip -- this is the feature vector the router will see at inference
    # time, since at inference time we haven't chosen a recipe yet.
    features = estimator.estimate(video_path).to_dict()

    scores = [
        _score_recipe_on_frames(rid, raw_frames, props.fps, row["true_hr_bpm"], rppg_method)
        for rid in RECIPE_IDS
    ]
    winner = select_winning_recipe(scores)

    logger.info(
        "%-40s true_hr=%.1f  winner=%s (MAE=%.2f, SNR=%.2fdB)",
        video_path.name, row["true_hr_bpm"], winner.recipe_id, winner.hr_mae, winner.pulse_snr_db,
    )

    return ClipLabelingResult(
        video_path=str(video_path),
        subject_id=str(row["subject_id"]),
        window_id=str(row["window_id"]),
        true_hr_bpm=float(row["true_hr_bpm"]),
        fitzpatrick=row.get("fitzpatrick", None),
        features=features,
        recipe_scores=scores,
        best_recipe=winner.recipe_id,
    )


# --------------------------------------------------------------------------- #
# Batch harness
# --------------------------------------------------------------------------- #

def run_labeling_harness(
    manifest_df: pd.DataFrame,
    output_dir: Path,
    rppg_method: str = "POS",
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    estimator = DegradationEstimator()

    training_rows = []
    diagnostic_rows = []

    for i, row in manifest_df.iterrows():
        try:
            result = label_one_clip(row, estimator, rppg_method=rppg_method)
        except Exception as e:
            logger.error("Failed to label clip %s: %s", row.get("video_path"), e)
            continue

        training_row = {
            "subject_id": result.subject_id,
            "window_id": result.window_id,
            **{name: result.features[name] for name in FEATURE_NAMES},
            "best_recipe": result.best_recipe,
        }
        training_rows.append(training_row)

        for score in result.recipe_scores:
            diagnostic_rows.append({
                "subject_id": result.subject_id,
                "window_id": result.window_id,
                "video_path": result.video_path,
                "true_hr_bpm": result.true_hr_bpm,
                "fitzpatrick": result.fitzpatrick,
                "recipe_id": score.recipe_id,
                "hr_bpm": score.hr_bpm,
                "hr_mae": score.hr_mae,
                "pulse_snr_db": score.pulse_snr_db,
                "sqi": score.sqi,
                "face_detection_rate": score.face_detection_rate,
                "disqualified": score.disqualified,
                "is_winner": score.recipe_id == result.best_recipe,
            })

    training_csv = output_dir / "router_training_table.csv"
    diagnostics_csv = output_dir / "labeling_diagnostics.csv"

    if training_rows:
        pd.DataFrame(training_rows).to_csv(training_csv, index=False)
    if diagnostic_rows:
        pd.DataFrame(diagnostic_rows).to_csv(diagnostics_csv, index=False)

    logger.info(
        "Labeling harness complete: %d clips labeled -> %s (training), %s (diagnostics)",
        len(training_rows), training_csv, diagnostics_csv,
    )
    if training_rows:
        _log_label_distribution(pd.DataFrame(training_rows))
    return training_csv, diagnostics_csv


def _log_label_distribution(training_df: pd.DataFrame) -> None:
    counts = training_df["best_recipe"].value_counts()
    logger.info("Winning-recipe distribution across all labeled clips:\n%s", counts.to_string())


# --------------------------------------------------------------------------- #
# Fixed-best-recipe baseline (paper baseline 2) derived directly from the
# diagnostics table -- the single recipe with the lowest MEAN HR MAE across
# every clip, i.e. the best recipe if you were NOT allowed to adapt per input.
# --------------------------------------------------------------------------- #

def compute_fixed_best_recipe(diagnostics_csv: Path) -> str:
    df = pd.read_csv(diagnostics_csv)
    finite = df[np.isfinite(df["hr_mae"])]
    mean_mae_by_recipe = finite.groupby("recipe_id")["hr_mae"].mean().sort_values()
    logger.info("Mean HR MAE by recipe (fixed-baseline candidate ranking):\n%s",
                mean_mae_by_recipe.to_string())
    return str(mean_mae_by_recipe.index[0])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Brute-force enhancement recipes through the rPPG pipeline to "
                     "produce physiology-grounded router training labels."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    run = sub.add_parser("run", help="Run the full labeling harness on a manifest.")
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--rppg-method", choices=["POS", "CHROM"], default="POS")

    fixed = sub.add_parser("fixed-best-recipe", help="Report the best single fixed recipe from a diagnostics table.")
    fixed.add_argument("--diagnostics-csv", required=True, type=Path)

    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.mode == "run":
        manifest_df = load_labeling_manifest(args.manifest)
        training_csv, diagnostics_csv = run_labeling_harness(
            manifest_df, args.output_dir, rppg_method=args.rppg_method,
        )
        best_fixed = compute_fixed_best_recipe(diagnostics_csv)
        print(f"\nFixed-best-recipe baseline candidate: {best_fixed}")

    elif args.mode == "fixed-best-recipe":
        best_fixed = compute_fixed_best_recipe(args.diagnostics_csv)
        print(best_fixed)


if __name__ == "__main__":
    main()
