"""
degradation_injector.py
========================

Track A synthetic-degradation generator for the adaptive rPPG preprocessing
pipeline (MoSICom 2026 submission).

Purpose
-------
Given a *clean* source video (e.g. a UBFC-rPPG or PURE clip, which ships with
synchronized contact-PPG ground truth), this module synthesizes controlled,
parametrically-known degradations along the five axes the degradation
estimator is designed to detect:

    1. sensor noise            (Gaussian, additive, per-channel)
    2. blur / defocus          (Gaussian blur, kernel-size controlled)
    3. compression artefacts   (real H.264 encoding via ffmpeg, CRF sweep)
    4. illumination shift      (gamma + brightness, over/under-exposure)
    5. motion jitter           (smoothed random-walk affine perturbation,
                                 simulating handheld camera shake rather than
                                 frame-independent flicker)

Because every degradation is applied by *this* module with known parameters,
the original clean video is always available as a reference. This is what
makes Track A (as opposed to Track B / MMPD real-world footage) suitable for
reference-based fidelity metrics (PSNR, SSIM, MSE, MAE, Delta-brightness,
contrast ratio) in fidelity_metrics.py, in addition to downstream rPPG
accuracy metrics.

This module does NOT decide what is "good" or "bad" -- it only manufactures
controlled degraded/clean pairs and records the exact ground-truth parameters
used, so that:
    - degradation_estimator.py can be validated against known-truth severity
      (does the no-reference estimator's noise score track the actual sigma
      we injected?), and
    - labeling_harness.py has a large, systematically varied set of degraded
      clips to brute-force through candidate enhancement recipes.

Design notes
------------
- All frame-level degradations (noise, blur, illumination, motion) are
  applied in linear sequence directly on decoded frames and written out
  through a *near-lossless* intermediate encode, so they are not
  contaminated by incidental compression artefacts.
- Compression is applied last and separately, via a real ffmpeg libx264
  encode at a specified CRF -- this is not simulated, it is the actual
  codec, because compression artefacts (blockiness, ringing, chroma
  subsampling loss) are not well approximated by simple pixel filters and
  we want the degradation estimator's "compression severity" axis to be
  trained/validated against ground truth CRF, not a proxy.
- Motion jitter uses a low-pass-filtered random walk (not i.i.d. per-frame
  noise) so consecutive frames drift smoothly, matching how real handheld
  motion looks, which matters for rPPG since abrupt frame-to-frame jumps
  break face tracking in a qualitatively different way than smooth drift.
- Every degraded output is written with a sidecar JSON record containing the
  exact parameters used, a reproducibility seed, and the path to its clean
  reference, so every experiment in the paper is exactly reproducible.

Author: Signal & Image Processing project (NMIMS Mumbai), MoSICom 2026.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

logger = logging.getLogger("degradation_injector")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


# --------------------------------------------------------------------------- #
# Degradation taxonomy
# --------------------------------------------------------------------------- #

class DegradationType(str, Enum):
    NOISE = "noise"
    BLUR = "blur"
    COMPRESSION = "compression"
    ILLUMINATION = "illumination"
    MOTION = "motion"


class Severity(str, Enum):
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


# --------------------------------------------------------------------------- #
# Per-axis parameter dataclasses
# --------------------------------------------------------------------------- #

@dataclass
class NoiseParams:
    """Additive Gaussian sensor noise, applied per-channel in float space."""
    sigma: float = 0.0          # std-dev in 0-255 pixel intensity units
    mean: float = 0.0

    def is_active(self) -> bool:
        return self.sigma > 0.0


@dataclass
class BlurParams:
    """Isotropic Gaussian blur, simulating defocus / low-quality optics."""
    kernel_size: int = 0        # must be odd; 0 = disabled
    sigma: Optional[float] = None  # None -> derived from kernel_size by OpenCV

    def is_active(self) -> bool:
        return self.kernel_size >= 3

    def __post_init__(self):
        if self.kernel_size and self.kernel_size % 2 == 0:
            raise ValueError(f"BlurParams.kernel_size must be odd, got {self.kernel_size}")


@dataclass
class CompressionParams:
    """Real H.264 (libx264) re-encode at a target Constant Rate Factor (CRF).

    CRF scale (libx264 convention): lower = higher quality/bitrate,
    higher = more aggressive compression / more artefacts.
        18  ~ visually near-lossless
        23  ~ default / mild
        28  ~ noticeable blocking, typical of poor video-call bandwidth
        35  ~ heavy artefacting
        40+ ~ severe, near-unusable
    """
    crf: int = 0                 # 0 = disabled (no re-encode)
    preset: str = "medium"       # ffmpeg -preset

    def is_active(self) -> bool:
        return self.crf > 0


@dataclass
class IlluminationParams:
    """Gamma + brightness-offset exposure shift.

    gamma < 1.0  -> brightens (simulates gain boost in low light)
    gamma > 1.0  -> darkens   (simulates under-exposure)
    brightness_delta is a signed additive offset in 0-255 units, applied
    after gamma correction.
    """
    gamma: float = 1.0
    brightness_delta: float = 0.0

    def is_active(self) -> bool:
        return not np.isclose(self.gamma, 1.0) or abs(self.brightness_delta) > 1e-6


@dataclass
class MotionParams:
    """Smoothed random-walk affine jitter simulating handheld camera shake.

    max_translation_px : per-axis translation drift bound, in pixels
    max_rotation_deg    : rotation drift bound, in degrees
    smoothness          : 0-1, higher = slower/smoother drift (low-pass filter
                           coefficient). 0 would be i.i.d. per-frame jitter
                           (unrealistic flicker); values close to 1 give slow,
                           realistic handheld sway.
    """
    max_translation_px: float = 0.0
    max_rotation_deg: float = 0.0
    smoothness: float = 0.85

    def is_active(self) -> bool:
        return self.max_translation_px > 0.0 or self.max_rotation_deg > 0.0


# --------------------------------------------------------------------------- #
# Severity presets
#
# These are the physical parameter ranges each qualitative severity level
# maps to. They are intentionally centralized here (rather than scattered as
# magic numbers) so the whole degradation grid used across the paper's
# experiments is defined in exactly one place.
# --------------------------------------------------------------------------- #

NOISE_PRESETS = {
    Severity.MILD:     NoiseParams(sigma=8.0),
    Severity.MODERATE: NoiseParams(sigma=18.0),
    Severity.SEVERE:   NoiseParams(sigma=32.0),
}

BLUR_PRESETS = {
    Severity.MILD:     BlurParams(kernel_size=5),
    Severity.MODERATE: BlurParams(kernel_size=11),
    Severity.SEVERE:   BlurParams(kernel_size=19),
}

COMPRESSION_PRESETS = {
    Severity.MILD:     CompressionParams(crf=23),
    Severity.MODERATE: CompressionParams(crf=32),
    Severity.SEVERE:   CompressionParams(crf=42),
}

ILLUMINATION_PRESETS = {
    # underexposed variants (dim room / poor lighting)
    "underexposed_mild":     IlluminationParams(gamma=1.4, brightness_delta=-15),
    "underexposed_moderate": IlluminationParams(gamma=1.9, brightness_delta=-35),
    "underexposed_severe":   IlluminationParams(gamma=2.6, brightness_delta=-55),
    # overexposed variants (harsh backlight / webcam auto-gain overshoot)
    "overexposed_mild":      IlluminationParams(gamma=0.75, brightness_delta=15),
    "overexposed_moderate":  IlluminationParams(gamma=0.55, brightness_delta=35),
    "overexposed_severe":    IlluminationParams(gamma=0.35, brightness_delta=55),
}

MOTION_PRESETS = {
    Severity.MILD:     MotionParams(max_translation_px=3.0, max_rotation_deg=0.8),
    Severity.MODERATE: MotionParams(max_translation_px=8.0, max_rotation_deg=2.0),
    Severity.SEVERE:   MotionParams(max_translation_px=16.0, max_rotation_deg=4.0),
}


# --------------------------------------------------------------------------- #
# Composite degradation specification
# --------------------------------------------------------------------------- #

@dataclass
class DegradationSpec:
    """A named, composite degradation configuration.

    Any subset of the five axes may be active simultaneously (e.g. a
    "realistic low-bandwidth telehealth call" composite combines noise +
    illumination + compression). Axes left at their default/inactive
    parameters are simply no-ops.
    """
    name: str
    noise: NoiseParams = field(default_factory=NoiseParams)
    blur: BlurParams = field(default_factory=BlurParams)
    compression: CompressionParams = field(default_factory=CompressionParams)
    illumination: IlluminationParams = field(default_factory=IlluminationParams)
    motion: MotionParams = field(default_factory=MotionParams)
    seed: int = 0

    def active_types(self) -> list[DegradationType]:
        active = []
        if self.noise.is_active():
            active.append(DegradationType.NOISE)
        if self.blur.is_active():
            active.append(DegradationType.BLUR)
        if self.compression.is_active():
            active.append(DegradationType.COMPRESSION)
        if self.illumination.is_active():
            active.append(DegradationType.ILLUMINATION)
        if self.motion.is_active():
            active.append(DegradationType.MOTION)
        return active

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# A handful of single-axis sweeps (for isolating the degradation estimator's
# sensitivity to one axis at a time) plus realistic composites (for the main
# rPPG-accuracy experiments) that the paper's results are built from.
def build_standard_spec_grid() -> list[DegradationSpec]:
    specs: list[DegradationSpec] = []

    # Single-axis sweeps
    for sev, p in NOISE_PRESETS.items():
        specs.append(DegradationSpec(name=f"noise_{sev.value}", noise=p))
    for sev, p in BLUR_PRESETS.items():
        specs.append(DegradationSpec(name=f"blur_{sev.value}", blur=p))
    for sev, p in COMPRESSION_PRESETS.items():
        specs.append(DegradationSpec(name=f"compression_{sev.value}", compression=p))
    for label, p in ILLUMINATION_PRESETS.items():
        specs.append(DegradationSpec(name=f"illumination_{label}", illumination=p))
    for sev, p in MOTION_PRESETS.items():
        specs.append(DegradationSpec(name=f"motion_{sev.value}", motion=p))

    # Realistic composites (mirroring plausible real-world capture conditions)
    specs.append(DegradationSpec(
        name="composite_lowlight_call",
        noise=NOISE_PRESETS[Severity.MODERATE],
        illumination=ILLUMINATION_PRESETS["underexposed_moderate"],
        compression=COMPRESSION_PRESETS[Severity.MODERATE],
    ))
    specs.append(DegradationSpec(
        name="composite_handheld_outdoor",
        motion=MOTION_PRESETS[Severity.MODERATE],
        illumination=ILLUMINATION_PRESETS["overexposed_mild"],
        compression=COMPRESSION_PRESETS[Severity.MILD],
    ))
    specs.append(DegradationSpec(
        name="composite_worst_case",
        noise=NOISE_PRESETS[Severity.SEVERE],
        blur=BLUR_PRESETS[Severity.MILD],
        illumination=ILLUMINATION_PRESETS["underexposed_severe"],
        compression=COMPRESSION_PRESETS[Severity.SEVERE],
        motion=MOTION_PRESETS[Severity.MILD],
    ))
    return specs


# --------------------------------------------------------------------------- #
# Frame-level degradation operators
# --------------------------------------------------------------------------- #

def inject_gaussian_noise(frame: np.ndarray, params: NoiseParams, rng: np.random.Generator) -> np.ndarray:
    """Add i.i.d. Gaussian noise per-pixel, per-channel, in float space."""
    if not params.is_active():
        return frame
    noise = rng.normal(params.mean, params.sigma, frame.shape).astype(np.float32)
    noisy = frame.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def inject_blur(frame: np.ndarray, params: BlurParams) -> np.ndarray:
    """Isotropic Gaussian blur."""
    if not params.is_active():
        return frame
    ksize = (params.kernel_size, params.kernel_size)
    sigma = params.sigma if params.sigma is not None else 0  # 0 -> OpenCV derives from ksize
    return cv2.GaussianBlur(frame, ksize, sigma)


def inject_illumination_shift(frame: np.ndarray, params: IlluminationParams) -> np.ndarray:
    """Gamma correction followed by an additive brightness offset."""
    if not params.is_active():
        return frame
    normalized = frame.astype(np.float32) / 255.0
    gamma_corrected = np.power(np.clip(normalized, 0, 1), params.gamma) * 255.0
    shifted = gamma_corrected + params.brightness_delta
    return np.clip(shifted, 0, 255).astype(np.uint8)


class MotionJitterGenerator:
    """Generates a smooth, low-pass-filtered random-walk trajectory of
    affine perturbations (dx, dy, d-theta) across a whole clip, then applies
    the per-frame transform to each frame in sequence.

    Using a random walk with an exponential low-pass filter (rather than
    independent per-frame random transforms) is what makes the resulting
    "shake" look like real handheld motion instead of flicker: consecutive
    frames drift continuously rather than jumping discontinuously.
    """

    def __init__(self, params: MotionParams, n_frames: int, rng: np.random.Generator):
        self.params = params
        self.n_frames = n_frames
        self.rng = rng
        self._trajectory = self._generate_trajectory() if params.is_active() else None

    def _generate_trajectory(self) -> np.ndarray:
        p = self.params
        alpha = 1.0 - p.smoothness  # low-pass filter coefficient
        raw = self.rng.normal(0, 1, size=(self.n_frames, 3))  # columns: dx, dy, dtheta
        smoothed = np.zeros_like(raw)
        smoothed[0] = raw[0]
        for t in range(1, self.n_frames):
            smoothed[t] = (1 - alpha) * smoothed[t - 1] + alpha * raw[t]
        # normalize each channel to its bound
        for col, bound in enumerate((p.max_translation_px, p.max_translation_px, p.max_rotation_deg)):
            peak = np.max(np.abs(smoothed[:, col])) + 1e-8
            smoothed[:, col] = smoothed[:, col] / peak * bound
        return smoothed  # shape (n_frames, 3) -> dx, dy, dtheta_degrees

    def apply(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        if self._trajectory is None:
            return frame
        h, w = frame.shape[:2]
        dx, dy, dtheta = self._trajectory[min(frame_idx, self.n_frames - 1)]
        center = (w / 2.0, h / 2.0)
        rot_mat = cv2.getRotationMatrix2D(center, dtheta, 1.0)
        rot_mat[0, 2] += dx
        rot_mat[1, 2] += dy
        return cv2.warpAffine(frame, rot_mat, (w, h), borderMode=cv2.BORDER_REFLECT101)


# --------------------------------------------------------------------------- #
# Video I/O helpers
# --------------------------------------------------------------------------- #

@dataclass
class VideoProps:
    fps: float
    width: int
    height: int
    n_frames: int


def read_video_properties(path: Path) -> VideoProps:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    props = VideoProps(
        fps=cap.get(cv2.CAP_PROP_FPS),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return props


def iter_frames(path: Path) -> Iterable[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()


def write_near_lossless(frames: Iterable[np.ndarray], out_path: Path, fps: float, width: int, height: int) -> None:
    """Write frames through a genuinely near-lossless encode so pixel-level
    degradations already baked into the frames are preserved without any
    additional, uncontrolled compression contamination.

    IMPORTANT: an earlier version of this function used cv2.VideoWriter
    with the 'mp4v' fourcc. That was found (via degradation_estimator.py's
    blind blockiness measure, cross-checked against the original untouched
    source video) to silently introduce SEVERE compression artefacts of
    its own -- worse, in fact, than a deliberate real CRF-32 ffmpeg encode
    -- because cv2.VideoWriter's mp4v path uses an unspecified, low
    default bitrate. That contaminated every non-compression-axis output
    (noise/blur/illumination/motion clips) with uncontrolled blocking
    artefacts, which would have quietly corrupted both the compression
    axis's ground truth and every Track A PSNR/SSIM fidelity comparison.

    Fixed by piping raw frames directly into ffmpeg's libx264 encoder at
    qp=0 (mathematically lossless mode for x264), which is the same real
    encoder used for the deliberate compression stage, just configured for
    (near-)lossless rather than lossy output. 4:2:0 chroma subsampling is
    still applied (standard for yuv420p), matching how virtually all real
    camera/webcam video is captured, so this does not itself introduce an
    unrealistic idealization.
    """
    if shutil.which("ffmpeg") is None:
        raise EnvironmentError("ffmpeg not found on PATH; required for near-lossless encoding.")

    out_path = Path(out_path)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-qp", "0", "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in frames:
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            proc.stdin.write(frame.astype(np.uint8).tobytes())
    finally:
        proc.stdin.close()
        stderr = proc.stderr.read()
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg near-lossless encode failed:\n{stderr.decode(errors='replace')}")


def compress_with_ffmpeg(input_path: Path, output_path: Path, params: CompressionParams) -> None:
    """Re-encode a video through real libx264 at the given CRF, baking in
    genuine compression artefacts (as opposed to a hand-rolled blockiness
    filter)."""
    if shutil.which("ffmpeg") is None:
        raise EnvironmentError("ffmpeg not found on PATH; required for compression degradation.")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-crf", str(params.crf),
        "-preset", params.preset,
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    logger.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg compression failed (crf={params.crf}):\n{result.stderr}")


# --------------------------------------------------------------------------- #
# Degradation record (sidecar metadata written alongside every degraded clip)
# --------------------------------------------------------------------------- #

@dataclass
class DegradationRecord:
    clean_source: str
    degraded_output: str
    spec_name: str
    seed: int
    active_types: list[str]
    params: dict
    source_video_hash: str
    fps: float
    width: int
    height: int
    n_frames: int

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=2)


def _file_hash(path: Path, block_size: int = 65536) -> str:
    """Short SHA-256 prefix of the source file, so every degraded clip's
    sidecar record can be traced back to an exact, verifiable clean source
    even if filenames are later reorganized."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block_size):
            h.update(chunk)
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Core injector
# --------------------------------------------------------------------------- #

class DegradationInjector:
    """Applies a DegradationSpec to a clean source video, producing a
    degraded output video plus a JSON sidecar recording exactly how it was
    produced."""

    def __init__(self, spec: DegradationSpec):
        self.spec = spec
        self.rng = np.random.default_rng(spec.seed)

    def run(self, clean_path: Path, output_path: Path) -> DegradationRecord:
        clean_path = Path(clean_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        props = read_video_properties(clean_path)
        logger.info(
            "Degrading '%s' -> '%s' [spec=%s, %dx%d @ %.2ffps, %d frames]",
            clean_path.name, output_path.name, self.spec.name,
            props.width, props.height, props.fps, props.n_frames,
        )

        motion_gen = MotionJitterGenerator(self.spec.motion, props.n_frames, self.rng)

        def degraded_frame_stream():
            for idx, frame in enumerate(iter_frames(clean_path)):
                f = inject_illumination_shift(frame, self.spec.illumination)
                f = inject_blur(f, self.spec.blur)
                f = inject_gaussian_noise(f, self.spec.noise, self.rng)
                f = motion_gen.apply(f, idx)
                yield f

        needs_compression = self.spec.compression.is_active()

        if needs_compression:
            # Stage 1: write pixel-degraded frames to a temp near-lossless file
            with tempfile.TemporaryDirectory() as tmp_dir:
                intermediate = Path(tmp_dir) / "pre_compression.mp4"
                write_near_lossless(degraded_frame_stream(), intermediate, props.fps, props.width, props.height)
                # Stage 2: real ffmpeg CRF re-encode -> final output
                compress_with_ffmpeg(intermediate, output_path, self.spec.compression)
        else:
            write_near_lossless(degraded_frame_stream(), output_path, props.fps, props.width, props.height)

        record = DegradationRecord(
            clean_source=str(clean_path),
            degraded_output=str(output_path),
            spec_name=self.spec.name,
            seed=self.spec.seed,
            active_types=[t.value for t in self.spec.active_types()],
            params=self.spec.to_dict(),
            source_video_hash=_file_hash(clean_path),
            fps=props.fps, width=props.width, height=props.height, n_frames=props.n_frames,
        )
        sidecar_path = output_path.with_suffix(".json")
        record.save(sidecar_path)
        logger.info("Wrote degraded clip and sidecar: %s, %s", output_path, sidecar_path)
        return record


# --------------------------------------------------------------------------- #
# Batch dataset builder (Track A generation)
# --------------------------------------------------------------------------- #

def build_track_a_dataset(
    clean_video_paths: Iterable[Path],
    output_dir: Path,
    specs: Optional[list[DegradationSpec]] = None,
    base_seed: int = 42,
) -> Path:
    """Runs the full degradation grid over every clean input video, writing
    all degraded outputs plus a manifest CSV (consumed by
    labeling_harness.py and fidelity_metrics.py) mapping every
    (clean, degraded, spec) triple.

    Returns the path to the manifest CSV.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = specs or build_standard_spec_grid()

    manifest_rows = []
    for clean_path in clean_video_paths:
        clean_path = Path(clean_path)
        subject_dir = output_dir / clean_path.stem
        subject_dir.mkdir(parents=True, exist_ok=True)
        for i, spec in enumerate(specs):
            # Give every (video, spec) pair a distinct, reproducible seed.
            spec_i = dataclasses.replace(spec, seed=base_seed + i)
            out_path = subject_dir / f"{clean_path.stem}__{spec.name}.mp4"
            try:
                record = DegradationInjector(spec_i).run(clean_path, out_path)
                manifest_rows.append({
                    "clean_source": record.clean_source,
                    "degraded_output": record.degraded_output,
                    "spec_name": record.spec_name,
                    "active_types": ";".join(record.active_types),
                    "seed": record.seed,
                })
            except Exception as e:
                logger.error("Failed on %s / %s: %s", clean_path.name, spec.name, e)

    manifest_path = output_dir / "track_a_manifest.csv"
    _write_manifest_csv(manifest_rows, manifest_path)
    logger.info("Track A dataset built: %d clips -> manifest at %s", len(manifest_rows), manifest_path)
    return manifest_path


def _write_manifest_csv(rows: list[dict], path: Path) -> None:
    import csv
    if not rows:
        logger.warning("No manifest rows to write.")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Track A synthetic degraded/clean video pairs for the "
                     "adaptive rPPG preprocessing pipeline."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    single = sub.add_parser("single", help="Apply one named preset spec to one video.")
    single.add_argument("--input", required=True, type=Path)
    single.add_argument("--output", required=True, type=Path)
    single.add_argument("--spec", required=True, choices=[s.name for s in build_standard_spec_grid()])
    single.add_argument("--seed", type=int, default=42)

    sweep = sub.add_parser("sweep", help="Apply the full degradation grid to every video in a directory.")
    sweep.add_argument("--input-dir", required=True, type=Path)
    sweep.add_argument("--output-dir", required=True, type=Path)
    sweep.add_argument("--pattern", default="*.mp4")
    sweep.add_argument("--seed", type=int, default=42)

    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.mode == "single":
        specs = {s.name: s for s in build_standard_spec_grid()}
        spec = dataclasses.replace(specs[args.spec], seed=args.seed)
        DegradationInjector(spec).run(args.input, args.output)

    elif args.mode == "sweep":
        clean_videos = sorted(args.input_dir.glob(args.pattern))
        if not clean_videos:
            logger.error("No videos matching '%s' found in %s", args.pattern, args.input_dir)
            return
        build_track_a_dataset(clean_videos, args.output_dir, base_seed=args.seed)


if __name__ == "__main__":
    main()
