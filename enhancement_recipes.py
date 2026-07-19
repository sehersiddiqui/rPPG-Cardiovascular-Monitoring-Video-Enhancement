"""
enhancement_recipes.py
========================

Plain function library of classical image/video enhancement techniques,
refactored out of the earlier Streamlit demo (video_enhancer_streamlit.py)
into a UI-free, importable module, plus a fixed, ordered set of composite
"recipes" (R0-R5) that the routing classifier selects between.

Purpose in the architecture
----------------------------
None of the individual techniques below (CLAHE, bilateral filtering, unsharp
masking, gray-world white balance, etc.) are novel -- they are standard,
well-established classical image processing operations. What this project
contributes is NOT these techniques; it is:

    (a) constraining them into a small set of fixed, ORDERED recipes rather
        than an intractable free combination search (order matters -- e.g.
        sharpening before denoising amplifies noise rather than removing
        it), and
    (b) letting the routing classifier (router_classifier.py) pick between
        these fixed recipes based on the degradation estimator's feature
        vector, with the recipe LABELS supplied by labeling_harness.py
        determined by downstream rPPG physiological accuracy (Pulse SNR,
        HR MAE) -- NOT by image fidelity (PSNR/SSIM). This module only
        supplies the enhancement mechanics; it has no opinion about which
        recipe is "best" for a given input. That decision is made
        elsewhere, on physiological grounds.

R0 is a mandatory pass-through / no-enhancement control, used as one of the
three evaluation baselines (raw / fixed-best-recipe / adaptive-router).

All functions operate on a single BGR uint8 frame (OpenCV convention,
shape (H, W, 3), dtype uint8) and return a frame of the same shape/dtype,
so they compose cleanly and can be applied frame-by-frame across a video or
directly to a single image.

Author: Signal & Image Processing project (NMIMS Mumbai), MoSICom 2026.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import cv2
import numpy as np

# Reuse the video I/O primitives already built and tested in
# degradation_injector.py, rather than duplicating frame read/write logic.
from degradation_injector import (
    iter_frames,
    read_video_properties,
    write_near_lossless,
)

logger = logging.getLogger("enhancement_recipes")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    """Clip and cast back to the canonical uint8 BGR representation every
    technique below must return, regardless of intermediate float math."""
    return np.clip(frame, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# 1. Exposure correction
# --------------------------------------------------------------------------- #

def correct_exposure(
    frame: np.ndarray,
    target_mean: float = 128.0,
    gamma_bounds: tuple[float, float] = (0.3, 3.0),
) -> np.ndarray:
    """Adaptive gamma correction: estimates the gamma that pushes the
    frame's mean luminance toward `target_mean`, then applies it. This is
    the auto-exposure counterpart to the manual gamma degradation applied
    in degradation_injector.py -- here the gamma is *solved for* from the
    input rather than specified, since at inference time we don't know the
    true exposure error in advance.
    """
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0].astype(np.float32)
    current_mean = float(np.mean(y)) + 1e-6

    # Solve gamma such that ((current_mean/255)^gamma) * 255 == target_mean
    # => gamma = log(target_mean/255) / log(current_mean/255)
    ratio_target = np.clip(target_mean / 255.0, 1e-3, 0.999)
    ratio_current = np.clip(current_mean / 255.0, 1e-3, 0.999)
    gamma = float(np.log(ratio_target) / np.log(ratio_current))
    gamma = float(np.clip(gamma, gamma_bounds[0], gamma_bounds[1]))

    y_norm = y / 255.0
    y_corrected = np.power(y_norm, gamma) * 255.0
    ycrcb[:, :, 0] = _to_uint8(y_corrected)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


# --------------------------------------------------------------------------- #
# 2. Contrast enhancement
# --------------------------------------------------------------------------- #

def apply_clahe(
    frame: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization, applied to the L
    (lightness) channel in LAB space so chroma is left untouched -- this
    matters for rPPG downstream, since we do not want to distort the
    chrominance channels that carry pulse-related color information any
    more than necessary."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)
    merged = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def apply_histogram_equalization(frame: np.ndarray) -> np.ndarray:
    """Global histogram equalization on the Y (luma) channel in YCrCb
    space. More aggressive and less locally adaptive than CLAHE; kept as a
    separate technique since the two behave very differently under
    non-uniform illumination."""
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    merged = cv2.merge((y_eq, cr, cb))
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


def apply_contrast_stretch(
    frame: np.ndarray,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
) -> np.ndarray:
    """Per-channel percentile-based linear contrast stretch: maps the
    [lower_percentile, upper_percentile] intensity range of each channel to
    [0, 255]. Percentile clipping (rather than min/max) avoids a single
    outlier pixel collapsing the whole stretch."""
    out = np.empty_like(frame, dtype=np.float32)
    for c in range(3):
        channel = frame[:, :, c].astype(np.float32)
        lo, hi = np.percentile(channel, [lower_percentile, upper_percentile])
        if hi - lo < 1e-6:
            out[:, :, c] = channel
            continue
        stretched = (channel - lo) * (255.0 / (hi - lo))
        out[:, :, c] = stretched
    return _to_uint8(out)


# --------------------------------------------------------------------------- #
# 3. Denoising
# --------------------------------------------------------------------------- #

def denoise_median(frame: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Median filter -- effective against impulsive/salt-and-pepper noise,
    edge-preserving but can erode fine texture at larger kernel sizes."""
    if ksize % 2 == 0:
        ksize += 1
    return cv2.medianBlur(frame, ksize)


def denoise_bilateral(
    frame: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """Bilateral filter -- edge-preserving smoothing; averages over spatial
    neighbours only when their intensity is also similar, so it suppresses
    Gaussian sensor noise while keeping strong edges (e.g. face/hair
    boundaries) comparatively sharp."""
    return cv2.bilateralFilter(frame, d, sigma_color, sigma_space)


def denoise_gaussian(frame: np.ndarray, ksize: int = 5, sigma: float = 0.0) -> np.ndarray:
    """Plain isotropic Gaussian blur used as a denoising step. The
    cheapest option; no edge preservation, so it is the most likely of the
    denoisers to also blur out the subtle skin-color fluctuations rPPG
    depends on -- included for completeness and as a deliberately "bad"
    ablation case in the results."""
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(frame, (ksize, ksize), sigma)


def denoise_nlm(
    frame: np.ndarray,
    h: float = 8.0,
    h_color: float = 8.0,
    template_window_size: int = 7,
    search_window_size: int = 21,
) -> np.ndarray:
    """Non-Local Means denoising -- averages over similar patches across
    the whole search window rather than just local neighbours; generally
    the strongest denoiser here, but also the most computationally
    expensive and the one most prone to smoothing away weak periodic
    signals if `h` is set too high."""
    return cv2.fastNlMeansDenoisingColored(
        frame, None, h, h_color, template_window_size, search_window_size
    )


# --------------------------------------------------------------------------- #
# 4. Color correction
# --------------------------------------------------------------------------- #

def white_balance_gray_world(frame: np.ndarray) -> np.ndarray:
    """Gray-world white balance: assumes the average scene reflectance is
    achromatic (gray), so it rescales each color channel by the ratio of
    the overall mean intensity to that channel's mean, correcting color
    casts from mixed/incorrect illumination -- directly relevant to rPPG,
    since the pulse signal is extracted from RGB channel ratios and a
    color cast can bias those ratios systematically."""
    b, g, r = cv2.split(frame.astype(np.float32))
    mean_b, mean_g, mean_r = np.mean(b), np.mean(g), np.mean(r)
    mean_gray = (mean_b + mean_g + mean_r) / 3.0 + 1e-6

    b_bal = b * (mean_gray / (mean_b + 1e-6))
    g_bal = g * (mean_gray / (mean_g + 1e-6))
    r_bal = r * (mean_gray / (mean_r + 1e-6))

    return _to_uint8(cv2.merge((b_bal, g_bal, r_bal)))


# --------------------------------------------------------------------------- #
# 5. Sharpening
# --------------------------------------------------------------------------- #

def sharpen_unsharp_mask(
    frame: np.ndarray,
    ksize: int = 5,
    sigma: float = 1.0,
    amount: float = 1.5,
    threshold: int = 0,
) -> np.ndarray:
    """Classic unsharp masking: sharpened = original + amount * (original -
    blurred). A `threshold` on the difference magnitude can be used to
    avoid amplifying low-contrast noise; 0 disables thresholding."""
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(frame, (ksize, ksize), sigma)
    diff = frame.astype(np.float32) - blurred.astype(np.float32)
    if threshold > 0:
        mask = (np.abs(diff) >= threshold).astype(np.float32)
        diff = diff * mask
    sharpened = frame.astype(np.float32) + amount * diff
    return _to_uint8(sharpened)


def sharpen_laplacian(frame: np.ndarray, ksize: int = 3, scale: float = 1.0) -> np.ndarray:
    """Laplacian sharpening: adds back a scaled Laplacian (second
    derivative) of the image, boosting high-frequency edge content
    directly rather than via a blur-and-subtract construction."""
    gray_like = frame.astype(np.float32)
    channels = []
    for c in range(3):
        lap = cv2.Laplacian(gray_like[:, :, c], cv2.CV_32F, ksize=ksize)
        channels.append(gray_like[:, :, c] - scale * lap)
    sharpened = np.stack(channels, axis=-1)
    return _to_uint8(sharpened)


def sharpen_highboost(
    frame: np.ndarray,
    ksize: int = 5,
    sigma: float = 1.0,
    boost_factor: float = 2.2,
) -> np.ndarray:
    """High-boost filtering:
        highboost = A * original - lowpass(original)

    Because Gaussian blurring approximately preserves mean intensity,
    mean(highboost) approx (A - 1) * mean(original): the output is
    brightness-NEUTRAL at A = 2.0, darkens progressively for A < 2, and
    brightens progressively for A > 2. A == 1 reduces to pure high-pass
    edge extraction (mean approx 0), which is almost never what you want
    as a "sharpen" step. The default of 2.2 sits just above the neutral
    point, giving mild edge emphasis without the large unintended exposure
    shift that a naively-chosen A < 2 (e.g. 1.5) would introduce -- an
    early version of this function used A = 1.5 and was found, by testing,
    to darken frames by roughly half; callers should not set boost_factor
    below ~1.8 unless a darkening effect is specifically desired."""
    if ksize % 2 == 0:
        ksize += 1
    low_pass = cv2.GaussianBlur(frame, (ksize, ksize), sigma).astype(np.float32)
    original = frame.astype(np.float32)
    highboost = boost_factor * original - low_pass
    return _to_uint8(highboost)


def overlay_edge_detail(frame: np.ndarray, ksize: int = 3, alpha: float = 0.3) -> np.ndarray:
    """Edge-detail overlay: extracts a Sobel gradient-magnitude edge map
    and additively blends it back into the original at weight `alpha`.
    Unlike unsharp/Laplacian/highboost (which sharpen the whole frequency
    band above a cutoff), this explicitly isolates edges first and adds
    them back, giving finer control over how much structural emphasis is
    introduced without a full-image contrast change."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    edge_mag = cv2.magnitude(grad_x, grad_y)
    edge_mag = edge_mag / (edge_mag.max() + 1e-6) * 255.0
    edge_3ch = np.stack([edge_mag] * 3, axis=-1)
    # Additive overlay: keep the full original base image and add a scaled
    # edge map on top, rather than blending it in (which would also darken
    # flat, low-gradient regions).
    blended = frame.astype(np.float32) + alpha * edge_3ch
    return _to_uint8(blended)


# --------------------------------------------------------------------------- #
# Step registry
#
# Every technique above is registered under a short string key so that
# Recipe definitions (below) and any future automated search can refer to
# steps by name + kwargs, rather than importing functions directly.
# --------------------------------------------------------------------------- #

StepFn = Callable[..., np.ndarray]

STEP_REGISTRY: dict[str, StepFn] = {
    "exposure_correction": correct_exposure,
    "clahe": apply_clahe,
    "histogram_equalization": apply_histogram_equalization,
    "contrast_stretch": apply_contrast_stretch,
    "denoise_median": denoise_median,
    "denoise_bilateral": denoise_bilateral,
    "denoise_gaussian": denoise_gaussian,
    "denoise_nlm": denoise_nlm,
    "white_balance_gray_world": white_balance_gray_world,
    "sharpen_unsharp_mask": sharpen_unsharp_mask,
    "sharpen_laplacian": sharpen_laplacian,
    "sharpen_highboost": sharpen_highboost,
    "overlay_edge_detail": overlay_edge_detail,
}


def list_available_steps() -> list[str]:
    return sorted(STEP_REGISTRY.keys())


# --------------------------------------------------------------------------- #
# Recipe definitions: fixed, ORDERED chains of steps
# --------------------------------------------------------------------------- #

@dataclass
class RecipeStep:
    """One stage in a recipe: a registry key plus the kwargs to call it
    with. Kept as plain data (not a bound function) so recipes can be
    serialized to JSON alongside experiment results."""
    op: str
    params: dict = field(default_factory=dict)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        fn = STEP_REGISTRY.get(self.op)
        if fn is None:
            raise KeyError(f"Unknown enhancement op '{self.op}'. Available: {list_available_steps()}")
        return fn(frame, **self.params)


@dataclass
class Recipe:
    """A fixed, ordered enhancement chain. The router classifier's output
    space is exactly the set of `id`s in RECIPES below -- it never
    constructs a chain freely, since order matters (sharpening before
    denoising amplifies noise instead of removing it) and free
    combinatorial search over 5 technique categories is intractable for a
    lightweight classifier trained on a modest sample count."""
    id: str
    name: str
    description: str
    steps: list[RecipeStep] = field(default_factory=list)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        out = frame
        for step in self.steps:
            out = step.apply(out)
        return out

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ---- The fixed recipe set (R0 - R5) ---- #
#
# R0 is the mandatory pass-through control (baseline 1 in the paper's
# 3-way ablation: raw / fixed-best / adaptive-router).
#
# R1-R5 are deliberately varied in character so the routing classifier has
# meaningfully different options to choose between across the degradation
# feature space, rather than five near-identical variants:
#   R1 - denoise-led, for noisy/compressed input
#   R2 - exposure-led, for under/over-exposed input
#   R3 - color-correction-led + strong denoise, for real-world (MMPD-style)
#        mixed color-cast + noise conditions
#   R4 - contrast+sharpen-led, for blurry/low-detail but otherwise clean input
#   R5 - gentle multi-stage, a lower-aggression alternative to R3 for
#        moderate (not severe) real-world degradation

RECIPES: dict[str, Recipe] = {
    "R0": Recipe(
        id="R0", name="pass_through",
        description="No enhancement. Internal control / raw baseline.",
        steps=[],
    ),
    "R1": Recipe(
        id="R1", name="denoise_then_mild_clahe",
        description="Bilateral denoise followed by mild CLAHE. Targets sensor-noise "
                     "and compression-dominated degradation without over-sharpening "
                     "already-noisy content.",
        steps=[
            RecipeStep("denoise_bilateral", {"d": 9, "sigma_color": 60, "sigma_space": 60}),
            RecipeStep("clahe", {"clip_limit": 1.5, "tile_grid_size": (8, 8)}),
        ],
    ),
    "R2": Recipe(
        id="R2", name="exposure_then_contrast_stretch",
        description="Adaptive exposure correction followed by percentile contrast "
                     "stretch. Targets under/over-exposed illumination-dominated "
                     "degradation.",
        steps=[
            RecipeStep("exposure_correction", {"target_mean": 128.0}),
            RecipeStep("contrast_stretch", {"lower_percentile": 2.0, "upper_percentile": 98.0}),
        ],
    ),
    "R3": Recipe(
        id="R3", name="white_balance_then_nlm",
        description="Gray-world white balance followed by Non-Local Means denoise. "
                     "Targets real-world mixed color-cast + noise conditions "
                     "(e.g. MMPD natural-light / incandescent scenes).",
        steps=[
            RecipeStep("white_balance_gray_world", {}),
            RecipeStep("denoise_nlm", {"h": 7.0, "h_color": 7.0}),
        ],
    ),
    "R4": Recipe(
        id="R4", name="clahe_then_unsharp",
        description="CLAHE followed by unsharp masking. Targets blurry / low-detail "
                     "but otherwise low-noise input (defocus-dominated degradation).",
        steps=[
            RecipeStep("clahe", {"clip_limit": 2.0, "tile_grid_size": (8, 8)}),
            RecipeStep("sharpen_unsharp_mask", {"ksize": 5, "sigma": 1.0, "amount": 1.2}),
        ],
    ),
    "R5": Recipe(
        id="R5", name="gentle_wb_bilateral_highboost",
        description="Gray-world white balance, gentle bilateral denoise, mild "
                     "high-boost sharpening. A lower-aggression alternative to R3 "
                     "for moderate (not severe) real-world degradation, where R3's "
                     "stronger NLM denoise risks over-smoothing the pulse signal.",
        steps=[
            RecipeStep("white_balance_gray_world", {}),
            RecipeStep("denoise_bilateral", {"d": 7, "sigma_color": 40, "sigma_space": 40}),
            RecipeStep("sharpen_highboost", {"ksize": 5, "sigma": 1.0, "boost_factor": 2.1}),
        ],
    ),
}

RECIPE_IDS: list[str] = list(RECIPES.keys())  # fixed order; used as the router classifier's label space


def get_recipe(recipe_id: str) -> Recipe:
    if recipe_id not in RECIPES:
        raise KeyError(f"Unknown recipe id '{recipe_id}'. Available: {RECIPE_IDS}")
    return RECIPES[recipe_id]


def list_recipes() -> list[str]:
    return list(RECIPE_IDS)


# --------------------------------------------------------------------------- #
# Application to a single frame / full video
# --------------------------------------------------------------------------- #

def apply_recipe_to_frame(frame: np.ndarray, recipe: Recipe) -> np.ndarray:
    return recipe.apply(frame)


def apply_recipe_to_video(input_path: Path, output_path: Path, recipe: Recipe) -> None:
    """Applies a recipe to every frame of a video, writing the result via
    the same near-lossless encode used by degradation_injector.py, so the
    enhanced output is not contaminated by an unrelated secondary
    compression pass."""
    input_path, output_path = Path(input_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    props = read_video_properties(input_path)

    logger.info(
        "Applying recipe '%s' (%s) to '%s' -> '%s' [%d frames]",
        recipe.id, recipe.name, input_path.name, output_path.name, props.n_frames,
    )

    def enhanced_frame_stream():
        for frame in iter_frames(input_path):
            yield apply_recipe_to_frame(frame, recipe)

    write_near_lossless(enhanced_frame_stream(), output_path, props.fps, props.width, props.height)
    logger.info("Wrote enhanced video: %s", output_path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply a fixed enhancement recipe (R0-R5) to a video."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--recipe", required=True, choices=RECIPE_IDS)
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    recipe = get_recipe(args.recipe)
    apply_recipe_to_video(args.input, args.output, recipe)


if __name__ == "__main__":
    main()
