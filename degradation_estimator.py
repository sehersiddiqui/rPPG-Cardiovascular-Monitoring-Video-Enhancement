"""
degradation_estimator.py
==========================

No-reference (blind) degradation scoring module. Inspects a raw video --
synthetic (Track A) or real-world (Track B / MMPD) -- and produces a
compact, fixed-order feature vector describing five independent
degradation axes, WITHOUT requiring a clean reference to compare against.
This is the core requirement: at inference time on real MMPD footage, or
on any video a deployed system would encounter, no clean twin exists, so
every estimator in this module must work from the degraded video alone.

The five axes, and the no-reference technique used for each:

    1. Noise level        - Immerkaer's fast Laplacian-based noise-sigma
                             estimator (Immerkaer, 1996).
    2. Blur / sharpness    - Variance-of-Laplacian sharpness measure.
    3. Compression severity- Blind blockiness measure (Wang et al., 2000
                             style: boundary vs. interior gradient energy at
                             the 8-pixel macroblock grid).
    4. Illumination quality- Histogram-based exposure analysis (mean luma
                             deviation from mid-gray + clipped-pixel
                             fraction), signed for under/over-exposure.
    5. Motion magnitude    - Dense optical flow (Farneback) magnitude,
                             averaged over short bursts of consecutive
                             frames sampled across the clip.

None of these five techniques is novel in isolation -- each is a standard,
well-established blind image quality primitive. The contribution of this
module is (a) combining all five into one coherent, fixed-order feature
vector specifically tailored to what the enhancement router
(router_classifier.py) needs to decide between recipes, and (b) validating
each axis's output against KNOWN ground-truth severity from
degradation_injector.py's sidecar records, which is the only reason we can
claim the estimator is measuring what we think it's measuring rather than
just producing plausible-looking numbers.

Output contract
----------------
`DegradationEstimator.estimate(video_path)` returns a `DegradationFeatures`
object whose `.to_array()` method yields a fixed-order 5-element numpy
vector -- this exact ordering (FEATURE_NAMES) is what router_classifier.py
trains and predicts on. Additional diagnostic sub-fields (e.g. exposure
direction, per-burst motion variance) are retained on the object for
analysis and paper figures but are NOT part of the classifier's input
vector, to keep the router's feature space small and interpretable given
our data scale.

Author: Signal & Image Processing project (NMIMS Mumbai), MoSICom 2026.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from degradation_injector import iter_frames, read_video_properties

logger = logging.getLogger("degradation_estimator")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# Fixed feature ordering consumed by router_classifier.py. Changing this
# order (or adding/removing an entry) requires retraining the router.
FEATURE_NAMES: list[str] = [
    "noise_score",
    "blur_score",
    "compression_score",
    "illumination_score",
    "motion_score",
]


# --------------------------------------------------------------------------- #
# Frame sampling helpers
# --------------------------------------------------------------------------- #

def _load_sampled_frames(video_path: Path, n_samples: int = 24) -> list[np.ndarray]:
    """Evenly-spaced frame sampling across the whole clip, used by the
    per-frame static estimators (noise, blur, compression, illumination).
    A full sequential decode is used (rather than CAP_PROP_POS_FRAMES
    seeking) because seek accuracy is unreliable on inter-frame-coded
    (B/P-frame) compressed streams -- exactly the kind of video this
    module has to handle."""
    all_indices_frames = []
    for idx, frame in enumerate(iter_frames(video_path)):
        all_indices_frames.append((idx, frame))
    if not all_indices_frames:
        raise IOError(f"No frames could be read from {video_path}")

    total = len(all_indices_frames)
    if total <= n_samples:
        return [f for _, f in all_indices_frames]

    sample_idx = np.linspace(0, total - 1, n_samples).astype(int)
    sample_idx = sorted(set(sample_idx.tolist()))
    return [all_indices_frames[i][1] for i in sample_idx]


def _load_motion_bursts(
    video_path: Path, n_bursts: int = 4, burst_length: int = 6
) -> list[list[np.ndarray]]:
    """Samples several short bursts of CONSECUTIVE frames spread across the
    clip (rather than isolated evenly-spaced single frames), because
    optical flow needs frame-to-frame continuity within a burst."""
    frames = list(iter_frames(video_path))
    total = len(frames)
    if total < 2:
        return []
    burst_length = min(burst_length, total)
    if total <= burst_length:
        return [frames]

    starts = np.linspace(0, total - burst_length, n_bursts).astype(int)
    starts = sorted(set(starts.tolist()))
    return [frames[s:s + burst_length] for s in starts]


# --------------------------------------------------------------------------- #
# 1. Noise estimation -- Immerkaer's fast noise-sigma estimator
# --------------------------------------------------------------------------- #

_IMMERKAER_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)


@dataclass
class NoiseEstimate:
    sigma_est: float           # estimated additive Gaussian noise std-dev, 0-255 units
    score: float                # calibrated 0-1 severity score fed to the router


def _estimate_noise_sigma_single_frame(gray: np.ndarray) -> float:
    """Immerkaer (1996), 'Fast Noise Variance Estimation': convolves the
    image with a Laplacian-like mask designed to be insensitive to image
    structure (its response to any smooth or edge region is theoretically
    zero, so its response is dominated by noise), then normalizes by a
    closed-form constant derived from the kernel's own second moment."""
    h, w = gray.shape
    conv = cv2.filter2D(gray.astype(np.float64), -1, _IMMERKAER_KERNEL, borderType=cv2.BORDER_REPLICATE)
    sigma = np.sum(np.abs(conv)) * math.sqrt(math.pi / 2) / (6.0 * (w - 2) * (h - 2))
    return float(sigma)


def estimate_noise(frames: list[np.ndarray]) -> NoiseEstimate:
    sigmas = []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        sigmas.append(_estimate_noise_sigma_single_frame(gray))
    sigma_est = float(np.median(sigmas))  # median across frames: robust to any single outlier frame

    # Calibration: sigma_est of ~0-3 is typical of a clean webcam/phone
    # capture; our injector's mild/moderate/severe presets are 8/18/32.
    # Map through a soft-saturating curve anchored at those points so the
    # score is roughly linear over the operating range we actually inject,
    # and saturates gracefully beyond it rather than exploding.
    score = float(np.clip(sigma_est / 32.0, 0.0, 1.0))
    return NoiseEstimate(sigma_est=sigma_est, score=score)


# --------------------------------------------------------------------------- #
# 2. Blur / sharpness estimation -- variance of Laplacian
# --------------------------------------------------------------------------- #

@dataclass
class BlurEstimate:
    laplacian_variance: float   # raw sharpness statistic; HIGHER = sharper
    score: float                  # calibrated 0-1 severity score; HIGHER = more blurred


def estimate_blur(frames: list[np.ndarray]) -> BlurEstimate:
    variances = []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        variances.append(float(lap.var()))
    lap_var = float(np.median(variances))

    # Calibration: a reasonably sharp 480p-ish face video typically has
    # Laplacian variance in the low hundreds to low thousands depending on
    # resolution and texture; heavy blur (our 'severe' kernel=19 preset)
    # drives this toward single digits. We use a log-scale inverse mapping
    # so the score is well-behaved across this multi-order-of-magnitude
    # range instead of being dominated by a few very sharp outlier frames.
    REFERENCE_SHARP_VAR = 800.0  # calibration anchor; refine empirically per-dataset
    ratio = lap_var / REFERENCE_SHARP_VAR
    sharpness_norm = float(np.clip(ratio, 1e-4, 10.0))
    score = float(np.clip(1.0 - (math.log10(sharpness_norm) + 4) / 5.0, 0.0, 1.0))
    return BlurEstimate(laplacian_variance=lap_var, score=score)


# --------------------------------------------------------------------------- #
# 3. Compression severity -- blind blockiness measure
# --------------------------------------------------------------------------- #

@dataclass
class CompressionEstimate:
    blockiness_raw: float
    score: float


def _blockiness_single_frame(gray: np.ndarray, block_size: int = 8) -> float:
    """Simplified blind blockiness measure in the spirit of Wang, Sheikh &
    Bovik (2000): H.264/MPEG-family codecs code independently-quantized
    blocks on a fixed grid (typically 8x8 or 16x16 macroblocks), which
    introduces small but systematic intensity discontinuities exactly at
    block boundaries that are not present in natural, uncompressed image
    gradients. We compare the mean absolute gradient AT the block-boundary
    columns/rows against the mean absolute gradient at non-boundary
    columns/rows; a positive gap indicates blocking artefacts."""
    gray = gray.astype(np.float64)
    h, w = gray.shape

    # Horizontal gradient (column-to-column differences)
    d_h = np.abs(np.diff(gray, axis=1))  # shape (h, w-1)
    col_idx = np.arange(d_h.shape[1])
    boundary_cols = (col_idx % block_size) == (block_size - 1)
    if boundary_cols.sum() == 0 or (~boundary_cols).sum() == 0:
        h_gap = 0.0
    else:
        h_gap = float(d_h[:, boundary_cols].mean() - d_h[:, ~boundary_cols].mean())

    # Vertical gradient (row-to-row differences)
    d_v = np.abs(np.diff(gray, axis=0))  # shape (h-1, w)
    row_idx = np.arange(d_v.shape[0])
    boundary_rows = (row_idx % block_size) == (block_size - 1)
    if boundary_rows.sum() == 0 or (~boundary_rows).sum() == 0:
        v_gap = 0.0
    else:
        v_gap = float(d_v[boundary_rows, :].mean() - d_v[~boundary_rows, :].mean())

    return max(0.0, (h_gap + v_gap) / 2.0)


def estimate_compression(frames: list[np.ndarray]) -> CompressionEstimate:
    scores = []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        scores.append(_blockiness_single_frame(gray))
    blockiness_raw = float(np.median(scores))

    # Calibration: blockiness_raw is typically < 0.3 for CRF ~18-23 (mild),
    # and climbs toward 2-4+ for CRF 35-42 (severe) at typical resolutions.
    # These anchors should be re-derived per-resolution once real footage
    # is available (see `validate` CLI mode below), since blockiness scales
    # with how large a block is relative to frame size.
    score = float(np.clip(blockiness_raw / 3.0, 0.0, 1.0))
    return CompressionEstimate(blockiness_raw=blockiness_raw, score=score)


# --------------------------------------------------------------------------- #
# 4. Illumination quality -- histogram-based exposure analysis
# --------------------------------------------------------------------------- #

@dataclass
class IlluminationEstimate:
    mean_luma: float
    clipped_low_fraction: float   # fraction of near-black pixels (luma < 5)
    clipped_high_fraction: float  # fraction of near-white pixels (luma > 250)
    direction: float               # -1 (severely underexposed) .. +1 (severely overexposed)
    score: float                    # 0 (ideal exposure) .. 1 (severe under- or over-exposure)


def estimate_illumination(frames: list[np.ndarray]) -> IlluminationEstimate:
    lumas, low_fracs, high_fracs = [], [], []
    for f in frames:
        y = cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
        lumas.append(float(y.mean()))
        low_fracs.append(float(np.mean(y < 5)))
        high_fracs.append(float(np.mean(y > 250)))

    mean_luma = float(np.median(lumas))
    clipped_low = float(np.median(low_fracs))
    clipped_high = float(np.median(high_fracs))

    deviation = abs(mean_luma - 128.0) / 128.0  # 0 = ideal mid-gray exposure
    direction = float(np.clip((128.0 - mean_luma) / 128.0, -1.0, 1.0))  # + = underexposed

    score = float(np.clip(deviation + 2.0 * (clipped_low + clipped_high), 0.0, 1.0))
    return IlluminationEstimate(
        mean_luma=mean_luma, clipped_low_fraction=clipped_low,
        clipped_high_fraction=clipped_high, direction=direction, score=score,
    )


# --------------------------------------------------------------------------- #
# 5. Motion magnitude -- dense optical flow over short bursts
# --------------------------------------------------------------------------- #

@dataclass
class MotionEstimate:
    mean_flow_px: float          # mean per-frame optical-flow magnitude, in pixels
    max_flow_px: float
    score: float


_FLOW_RESIZE_WIDTH = 240  # downsample for speed; motion magnitude is rescaled back to full-res pixels


def _flow_magnitude_for_burst(burst: list[np.ndarray]) -> list[float]:
    if len(burst) < 2:
        return []
    h0, w0 = burst[0].shape[:2]
    scale = _FLOW_RESIZE_WIDTH / w0
    resized = [
        cv2.cvtColor(cv2.resize(f, (int(w0 * scale), int(h0 * scale))), cv2.COLOR_BGR2GRAY)
        for f in burst
    ]
    mags = []
    for i in range(len(resized) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            resized[i], resized[i + 1], None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0,
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mean_mag_px = float(np.mean(mag)) / scale  # rescale back to original-resolution pixels
        mags.append(mean_mag_px)
    return mags


def estimate_motion(bursts: list[list[np.ndarray]]) -> MotionEstimate:
    all_mags: list[float] = []
    for burst in bursts:
        all_mags.extend(_flow_magnitude_for_burst(burst))

    if not all_mags:
        return MotionEstimate(mean_flow_px=0.0, max_flow_px=0.0, score=0.0)

    mean_flow = float(np.mean(all_mags))
    max_flow = float(np.max(all_mags))

    # Calibration: our injector's mild/moderate/severe motion presets are
    # bounded at 3/8/16 px of translation drift per frame; static / minimal
    # real handheld footage is typically well under 1px of Farneback flow
    # at this downsample.
    score = float(np.clip(mean_flow / 10.0, 0.0, 1.0))
    return MotionEstimate(mean_flow_px=mean_flow, max_flow_px=max_flow, score=score)


# --------------------------------------------------------------------------- #
# Composite feature vector
# --------------------------------------------------------------------------- #

@dataclass
class DegradationFeatures:
    video_path: str
    noise: NoiseEstimate
    blur: BlurEstimate
    compression: CompressionEstimate
    illumination: IlluminationEstimate
    motion: MotionEstimate

    def to_array(self) -> np.ndarray:
        """Fixed-order 5-element feature vector, matching FEATURE_NAMES,
        consumed directly by router_classifier.py."""
        return np.array([
            self.noise.score,
            self.blur.score,
            self.compression.score,
            self.illumination.score,
            self.motion.score,
        ], dtype=np.float64)

    def to_dict(self) -> dict:
        return {
            "video_path": self.video_path,
            "noise_score": self.noise.score, "noise_sigma_est": self.noise.sigma_est,
            "blur_score": self.blur.score, "blur_laplacian_variance": self.blur.laplacian_variance,
            "compression_score": self.compression.score, "compression_blockiness_raw": self.compression.blockiness_raw,
            "illumination_score": self.illumination.score, "illumination_mean_luma": self.illumination.mean_luma,
            "illumination_direction": self.illumination.direction,
            "illumination_clipped_low": self.illumination.clipped_low_fraction,
            "illumination_clipped_high": self.illumination.clipped_high_fraction,
            "motion_score": self.motion.score, "motion_mean_flow_px": self.motion.mean_flow_px,
            "motion_max_flow_px": self.motion.max_flow_px,
        }


class DegradationEstimator:
    """Top-level entry point: runs all five no-reference axis estimators
    over a video and returns the combined feature vector."""

    def __init__(
        self,
        n_static_samples: int = 24,
        n_motion_bursts: int = 4,
        motion_burst_length: int = 6,
    ):
        self.n_static_samples = n_static_samples
        self.n_motion_bursts = n_motion_bursts
        self.motion_burst_length = motion_burst_length

    def estimate(self, video_path: Path) -> DegradationFeatures:
        video_path = Path(video_path)
        props = read_video_properties(video_path)
        logger.info(
            "Estimating degradation for '%s' [%dx%d, %d frames]",
            video_path.name, props.width, props.height, props.n_frames,
        )

        static_frames = _load_sampled_frames(video_path, self.n_static_samples)
        noise = estimate_noise(static_frames)
        blur = estimate_blur(static_frames)
        compression = estimate_compression(static_frames)
        illumination = estimate_illumination(static_frames)

        bursts = _load_motion_bursts(video_path, self.n_motion_bursts, self.motion_burst_length)
        motion = estimate_motion(bursts)

        features = DegradationFeatures(
            video_path=str(video_path),
            noise=noise, blur=blur, compression=compression,
            illumination=illumination, motion=motion,
        )
        logger.info(
            "  noise=%.3f blur=%.3f compression=%.3f illumination=%.3f motion=%.3f",
            noise.score, blur.score, compression.score, illumination.score, motion.score,
        )
        return features


# --------------------------------------------------------------------------- #
# Batch feature-table builder (feeds labeling_harness.py / router training)
# --------------------------------------------------------------------------- #

def build_feature_table(video_paths: list[Path], output_csv: Path) -> Path:
    estimator = DegradationEstimator()
    rows = []
    for vp in video_paths:
        try:
            feats = estimator.estimate(vp)
            rows.append(feats.to_dict())
        except Exception as e:
            logger.error("Failed to estimate degradation for %s: %s", vp, e)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    logger.info("Wrote feature table for %d videos -> %s", len(rows), output_csv)
    return output_csv


# --------------------------------------------------------------------------- #
# Validation against known ground truth (degradation_injector sidecar JSON)
#
# This is what lets us claim the no-reference estimator is actually
# tracking real degradation severity rather than producing plausible noise:
# we run it on our own synthetically-degraded clips, whose exact injected
# parameters we already know, and check agreement.
# --------------------------------------------------------------------------- #

def _true_severity_proxies(sidecar: dict) -> dict:
    """Extracts a comparable 'ground truth' scalar per axis from a
    degradation_injector sidecar record. These are in different physical
    units than our estimator's scores (that's expected -- we are checking
    RANK correlation / trend agreement, not unit-for-unit equality)."""
    params = sidecar["params"]
    return {
        "true_noise_sigma": params["noise"]["sigma"],
        "true_blur_kernel": params["blur"]["kernel_size"],
        "true_compression_crf": params["compression"]["crf"],
        "true_illumination_severity": abs(params["illumination"]["gamma"] - 1.0)
        + abs(params["illumination"]["brightness_delta"]) / 50.0,
        "true_motion_px": params["motion"]["max_translation_px"],
    }


def validate_against_injector(degraded_dir: Path) -> Path:
    """Scans a directory produced by degradation_injector.py's `sweep` /
    build_track_a_dataset (i.e. containing <clip>.mp4 + <clip>.json sidecar
    pairs), runs the estimator on every clip, and writes a comparison CSV
    of estimated-score vs. known-true-severity-proxy per axis, plus prints
    Pearson correlation per axis -- the sanity-check table for the paper's
    methodology section confirming the no-reference estimator tracks known
    injected severity even though it never sees the clean reference."""
    degraded_dir = Path(degraded_dir)
    sidecars = sorted(degraded_dir.rglob("*.json"))
    if not sidecars:
        raise FileNotFoundError(f"No sidecar JSON files found under {degraded_dir}")

    estimator = DegradationEstimator()
    rows = []
    for sidecar_path in sidecars:
        with open(sidecar_path) as f:
            sidecar = json.load(f)
        video_path = Path(sidecar["degraded_output"])
        if not video_path.exists():
            video_path = sidecar_path.with_suffix(".mp4")
        if not video_path.exists():
            logger.warning("Skipping %s: video not found", sidecar_path)
            continue

        feats = estimator.estimate(video_path)
        row = feats.to_dict()
        row["spec_name"] = sidecar["spec_name"]
        row.update(_true_severity_proxies(sidecar))
        rows.append(row)

    out_csv = degraded_dir / "estimator_validation.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote validation table (%d clips) -> %s", len(rows), out_csv)
    _print_axis_correlations(rows)
    return out_csv


def _print_axis_correlations(rows: list[dict]) -> None:
    import numpy as np
    pairs = [
        ("noise_sigma_est", "true_noise_sigma"),
        ("blur_score", "true_blur_kernel"),        # inverse relationship expected (higher kernel -> higher blur_score)
        ("compression_score", "true_compression_crf"),
        ("illumination_score", "true_illumination_severity"),
        ("motion_mean_flow_px", "true_motion_px"),
    ]
    print("\n--- Estimator vs. known ground truth (Pearson r) ---")
    for est_key, true_key in pairs:
        est_vals = np.array([r[est_key] for r in rows], dtype=float)
        true_vals = np.array([r[true_key] for r in rows], dtype=float)
        if np.std(est_vals) < 1e-9 or np.std(true_vals) < 1e-9:
            print(f"  {est_key:24s} vs {true_key:26s}: r=  n/a (no variance in this subset)")
            continue
        r = float(np.corrcoef(est_vals, true_vals)[0, 1])
        print(f"  {est_key:24s} vs {true_key:26s}: r={r:+.3f}")
    print("-----------------------------------------------------\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="No-reference degradation estimation for the adaptive rPPG preprocessing pipeline."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    single = sub.add_parser("estimate", help="Estimate degradation features for one video.")
    single.add_argument("--input", required=True, type=Path)
    single.add_argument("--json-out", type=Path, default=None)

    batch = sub.add_parser("batch", help="Estimate degradation features for every video in a directory.")
    batch.add_argument("--input-dir", required=True, type=Path)
    batch.add_argument("--pattern", default="*.mp4")
    batch.add_argument("--output-csv", required=True, type=Path)

    validate = sub.add_parser(
        "validate",
        help="Validate estimator output against degradation_injector's known ground truth "
             "on a directory of degraded clips + sidecar JSON files.",
    )
    validate.add_argument("--degraded-dir", required=True, type=Path)

    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.mode == "estimate":
        estimator = DegradationEstimator()
        feats = estimator.estimate(args.input)
        result = feats.to_dict()
        print(json.dumps(result, indent=2))
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            with open(args.json_out, "w") as f:
                json.dump(result, f, indent=2)

    elif args.mode == "batch":
        videos = sorted(args.input_dir.glob(args.pattern))
        build_feature_table(videos, args.output_csv)

    elif args.mode == "validate":
        validate_against_injector(args.degraded_dir)


if __name__ == "__main__":
    main()
