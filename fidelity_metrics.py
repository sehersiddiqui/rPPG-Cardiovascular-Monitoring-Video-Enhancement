"""
fidelity_metrics.py
====================

Reference-based image/video fidelity metrics for Track A (synthetic
degradation) evaluation. These metrics require a clean reference video to
compare against, so they are ONLY applicable to the synthetically-degraded
clips produced by degradation_injector.py -- not to real-world MMPD footage
where no clean twin exists.

Metrics computed:
    - PSNR (Peak Signal-to-Noise Ratio)
    - SSIM (Structural Similarity Index Measure, multiscale)
    - Image SNR (signal power / noise power, in dB)
    - MSE (Mean Squared Error)
    - MAE (Mean Absolute Error)
    - Δbrightness (mean luma difference)
    - Contrast ratio (std-dev ratio)

These are computed frame-by-frame and averaged across the clip. They exist
purely as a diagnostic layer to visualize where fidelity-optimal enhancement
and physiology-optimal enhancement diverge -- they are NEVER used as
optimization targets for the router or the recipes.

Author: Signal & Image Processing project (NMIMS Mumbai), MoSICom 2026.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from degradation_injector import iter_frames, read_video_properties

logger = logging.getLogger("fidelity_metrics")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


# --------------------------------------------------------------------------- #
# Per-frame metric primitives
# --------------------------------------------------------------------------- #

def compute_psnr(clean: np.ndarray, degraded: np.ndarray, max_val: float = 255.0) -> float:
    """Peak Signal-to-Noise Ratio between two uint8 frames."""
    mse = float(np.mean((clean.astype(np.float64) - degraded.astype(np.float64)) ** 2))
    if mse < 1e-10:
        return float("inf")
    return float(20.0 * np.log10(max_val / np.sqrt(mse)))


def compute_ssim(clean: np.ndarray, degraded: np.ndarray) -> float:
    """Multiscale SSIM between two BGR uint8 frames.
    skimage's SSIM operates on grayscale or multichannel; we use
    multichannel=True for the 3-channel BGR input."""
    return float(ssim(
        clean, degraded,
        data_range=255,
        channel_axis=-1,
        multichannel=True,
    ))


def compute_mse(clean: np.ndarray, degraded: np.ndarray) -> float:
    return float(np.mean((clean.astype(np.float64) - degraded.astype(np.float64)) ** 2))


def compute_mae(clean: np.ndarray, degraded: np.ndarray) -> float:
    return float(np.mean(np.abs(clean.astype(np.float64) - degraded.astype(np.float64))))


def compute_image_snr(clean: np.ndarray, degraded: np.ndarray) -> float:
    """Signal-to-noise ratio in dB: 10*log10(signal_power / noise_power).
    Signal power is approximated by the variance of the clean frame;
    noise power by the MSE between clean and degraded."""
    signal_power = float(np.var(clean.astype(np.float64)))
    noise_power = compute_mse(clean, degraded)
    if noise_power < 1e-10:
        return float("inf")
    return float(10.0 * np.log10(signal_power / noise_power))


def compute_delta_brightness(clean: np.ndarray, degraded: np.ndarray) -> float:
    """Signed difference in mean Y-channel luminance (YCrCb space).
    Positive = degraded is brighter; negative = degraded is darker."""
    clean_y = cv2.cvtColor(clean, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    deg_y = cv2.cvtColor(degraded, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    return float(np.mean(deg_y) - np.mean(clean_y))


def compute_contrast_ratio(clean: np.ndarray, degraded: np.ndarray) -> float:
    """Ratio of standard deviations in Y-channel luminance.
    > 1 means degraded has higher contrast; < 1 means lower."""
    clean_y = cv2.cvtColor(clean, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    deg_y = cv2.cvtColor(degraded, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    clean_std = float(np.std(clean_y))
    deg_std = float(np.std(deg_y))
    if clean_std < 1e-6:
        return 1.0
    return deg_std / clean_std


# --------------------------------------------------------------------------- #
# Aggregate per-clip results
# --------------------------------------------------------------------------- #

@dataclass
class FidelityResult:
    clean_path: str
    degraded_path: str
    n_frames: int
    psnr_mean: float
    psnr_std: float
    ssim_mean: float
    ssim_std: float
    image_snr_mean: float
    image_snr_std: float
    mse_mean: float
    mse_std: float
    mae_mean: float
    mae_std: float
    delta_brightness_mean: float
    delta_brightness_std: float
    contrast_ratio_mean: float
    contrast_ratio_std: float

    def to_dict(self) -> dict:
        return {
            "clean_path": self.clean_path,
            "degraded_path": self.degraded_path,
            "n_frames": self.n_frames,
            "psnr_mean": self.psnr_mean,
            "psnr_std": self.psnr_std,
            "ssim_mean": self.ssim_mean,
            "ssim_std": self.ssim_std,
            "image_snr_mean": self.image_snr_mean,
            "image_snr_std": self.image_snr_std,
            "mse_mean": self.mse_mean,
            "mse_std": self.mse_std,
            "mae_mean": self.mae_mean,
            "mae_std": self.mae_std,
            "delta_brightness_mean": self.delta_brightness_mean,
            "delta_brightness_std": self.delta_brightness_std,
            "contrast_ratio_mean": self.contrast_ratio_mean,
            "contrast_ratio_std": self.contrast_ratio_std,
        }


def compute_fidelity_metrics(clean_path: Path, degraded_path: Path) -> FidelityResult:
    """Computes all fidelity metrics frame-by-frame between a clean reference
    and its degraded counterpart, returning aggregate statistics."""
    clean_path, degraded_path = Path(clean_path), Path(degraded_path)

    clean_frames = list(iter_frames(clean_path))
    degraded_frames = list(iter_frames(degraded_path))

    if len(clean_frames) != len(degraded_frames):
        logger.warning(
            "Frame count mismatch: clean=%d vs degraded=%d for %s / %s",
            len(clean_frames), len(degraded_frames), clean_path.name, degraded_path.name,
        )
        n = min(len(clean_frames), len(degraded_frames))
        clean_frames = clean_frames[:n]
        degraded_frames = degraded_frames[:n]

    psnrs, ssims, image_snrs, mses, maes = [], [], [], [], []
    delta_brights, contrast_ratios = [], []

    for c_frame, d_frame in zip(clean_frames, degraded_frames):
        # Ensure same shape
        if c_frame.shape != d_frame.shape:
            d_frame = cv2.resize(d_frame, (c_frame.shape[1], c_frame.shape[0]))

        psnrs.append(compute_psnr(c_frame, d_frame))
        ssims.append(compute_ssim(c_frame, d_frame))
        image_snrs.append(compute_image_snr(c_frame, d_frame))
        mses.append(compute_mse(c_frame, d_frame))
        maes.append(compute_mae(c_frame, d_frame))
        delta_brights.append(compute_delta_brightness(c_frame, d_frame))
        contrast_ratios.append(compute_contrast_ratio(c_frame, d_frame))

    def _mean_std(vals: list[float]) -> tuple[float, float]:
        arr = np.array(vals)
        return float(np.mean(arr)), float(np.std(arr))

    return FidelityResult(
        clean_path=str(clean_path),
        degraded_path=str(degraded_path),
        n_frames=len(clean_frames),
        psnr_mean=_mean_std(psnrs)[0], psnr_std=_mean_std(psnrs)[1],
        ssim_mean=_mean_std(ssims)[0], ssim_std=_mean_std(ssims)[1],
        image_snr_mean=_mean_std(image_snrs)[0], image_snr_std=_mean_std(image_snrs)[1],
        mse_mean=_mean_std(mses)[0], mse_std=_mean_std(mses)[1],
        mae_mean=_mean_std(maes)[0], mae_std=_mean_std(maes)[1],
        delta_brightness_mean=_mean_std(delta_brights)[0],
        delta_brightness_std=_mean_std(delta_brights)[1],
        contrast_ratio_mean=_mean_std(contrast_ratios)[0],
        contrast_ratio_std=_mean_std(contrast_ratios)[1],
    )


# --------------------------------------------------------------------------- #
# Batch processing from Track A manifest
# --------------------------------------------------------------------------- #

def batch_compute_from_manifest(manifest_csv: Path, output_csv: Path) -> Path:
    """Reads a Track A manifest (clean_source, degraded_output columns) and
    computes fidelity metrics for every pair, writing results to CSV."""
    import pandas as pd

    manifest_csv = Path(manifest_csv)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_csv)
    if "clean_source" not in df.columns or "degraded_output" not in df.columns:
        raise ValueError(f"Manifest {manifest_csv} missing 'clean_source' or 'degraded_output' columns")

    rows = []
    for _, row in df.iterrows():
        clean = Path(row["clean_source"])
        degraded = Path(row["degraded_output"])
        if not clean.exists():
            logger.warning("Clean video not found: %s", clean)
            continue
        if not degraded.exists():
            logger.warning("Degraded video not found: %s", degraded)
            continue
        try:
            result = compute_fidelity_metrics(clean, degraded)
            d = result.to_dict()
            # merge manifest metadata if present
            for key in ["spec_name", "active_types", "seed"]:
                if key in row:
                    d[key] = row[key]
            rows.append(d)
            logger.info("Fidelity: %s -> PSNR=%.2f SSIM=%.3f", degraded.name, d["psnr_mean"], d["ssim_mean"])
        except Exception as e:
            logger.error("Failed on %s / %s: %s", clean.name, degraded.name, e)

    if rows:
        pd.DataFrame(rows).to_csv(output_csv, index=False)
    logger.info("Wrote fidelity metrics for %d pairs -> %s", len(rows), output_csv)
    return output_csv


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute reference-based fidelity metrics (Track A only)."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    single = sub.add_parser("single", help="Compare one clean/degraded pair.")
    single.add_argument("--clean", required=True, type=Path)
    single.add_argument("--degraded", required=True, type=Path)
    single.add_argument("--json-out", type=Path, default=None)

    batch = sub.add_parser("batch", help="Process a full Track A manifest.")
    batch.add_argument("--manifest", required=True, type=Path)
    batch.add_argument("--output-csv", required=True, type=Path)

    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.mode == "single":
        result = compute_fidelity_metrics(args.clean, args.degraded)
        d = result.to_dict()
        print(json.dumps(d, indent=2))
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump(d, f, indent=2)

    elif args.mode == "batch":
        batch_compute_from_manifest(args.manifest, args.output_csv)


if __name__ == "__main__":
    main()
