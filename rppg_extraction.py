"""
rppg_extraction.py
===================

Face ROI detection/tracking and POS/CHROM pulse-signal extraction from a
frame sequence. Built on top of standard OpenCV face detection (Haar + DNN
fallback) and the classical POS/CHROM rPPG methods, which are well
documented in the literature and implemented here from first principles
rather than requiring the full rPPG-Toolbox dependency (which is heavy and
brings in PyTorch, etc.).

This module is consumed by:
    - labeling_harness.py (brute-force recipe scoring)
    - pipeline_runner.py  (end-to-end inference)

Output contract
----------------
`extract_pulse_signal_from_frames()` returns an `ExtractionResult` containing:
    - pulse_signal      : 1-D numpy array of normalized pulse waveform
    - fps               : frame rate (Hz)
    - timestamps        : 1-D array of frame timestamps (seconds)
    - face_detection_rate : fraction of frames where a face was detected
    - roi_trace         : raw RGB mean trace before POS/CHROM projection
    - roi_boxes         : bounding box per frame (or None if no face)

The pulse_signal is bandpass-filtered to [0.7, 4.0] Hz (≈ 42–240 bpm) and
normalized to zero mean, unit variance for downstream metric computation.

Author: Signal & Image Processing project (NMIMS Mumbai), MoSICom 2026.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy.signal import butter, filtfilt

from degradation_injector import iter_frames, read_video_properties

logger = logging.getLogger("rppg_extraction")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# --------------------------------------------------------------------------- #
# Face detection
# --------------------------------------------------------------------------- #

# Try to load OpenCV DNN face detector; fall back to Haar if unavailable
_FACE_DETECTOR_DNN = None
_FACE_DETECTOR_HAAR = None


def _init_face_detectors():
    global _FACE_DETECTOR_DNN, _FACE_DETECTOR_HAAR
    if _FACE_DETECTOR_DNN is None:
        # OpenCV DNN face detector (ResNet-SSD based)
        prototxt = cv2.data.haarcascades + "deploy.prototxt"
        model = cv2.data.haarcascades + "res10_300x300_ssd_iter_140000.caffemodel"
        try:
            _FACE_DETECTOR_DNN = cv2.dnn.readNetFromCaffe(prototxt, model)
        except Exception:
            _FACE_DETECTOR_DNN = None
    if _FACE_DETECTOR_HAAR is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _FACE_DETECTOR_HAAR = cv2.CascadeClassifier(cascade_path)


def detect_face_dnn(frame: np.ndarray, confidence_threshold: float = 0.5) -> Optional[tuple[int, int, int, int]]:
    """Returns (x, y, w, h) bounding box or None."""
    if _FACE_DETECTOR_DNN is None:
        return None
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    _FACE_DETECTOR_DNN.setInput(blob)
    detections = _FACE_DETECTOR_DNN.forward()
    best_box = None
    best_conf = 0.0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold and confidence > best_conf:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            best_box = (x1, y1, x2 - x1, y2 - y1)
            best_conf = confidence
    return best_box


def detect_face_haar(frame: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """Returns (x, y, w, h) bounding box or None."""
    if _FACE_DETECTOR_HAAR is None:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _FACE_DETECTOR_HAAR.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if len(faces) == 0:
        return None
    # Pick largest face
    best = max(faces, key=lambda r: r[2] * r[3])
    return tuple(best)


def detect_face(frame: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """Try DNN first, fall back to Haar."""
    _init_face_detectors()
    box = detect_face_dnn(frame)
    if box is not None:
        return box
    return detect_face_haar(frame)


# --------------------------------------------------------------------------- #
# ROI tracking with simple temporal smoothing
# --------------------------------------------------------------------------- #

def track_face_roi(
    frames: list[np.ndarray],
    smooth_alpha: float = 0.3,
) -> list[Optional[tuple[int, int, int, int]]]:
    """Detects face in every frame, with temporal smoothing of the bounding
    box to reduce jitter. If a face is lost in a frame, the previous box
    is carried forward for up to `max_missing` frames."""
    boxes: list[Optional[tuple[int, int, int, int]]] = []
    prev_box: Optional[tuple[float, float, float, float]] = None
    max_missing = 5
    missing_count = 0

    for frame in frames:
        det = detect_face(frame)
        if det is not None:
            x, y, w, h = det
            # Shrink box to focus on forehead/cheeks (exclude hair/mouth)
            margin_x = int(w * 0.15)
            margin_y = int(h * 0.25)
            x += margin_x
            y += margin_y
            w -= 2 * margin_x
            h -= 2 * margin_y
            det = (max(0, x), max(0, y), max(1, w), max(1, h))

            if prev_box is not None:
                # Exponential smoothing
                px, py, pw, ph = prev_box
                sx = smooth_alpha * det[0] + (1 - smooth_alpha) * px
                sy = smooth_alpha * det[1] + (1 - smooth_alpha) * py
                sw = smooth_alpha * det[2] + (1 - smooth_alpha) * pw
                sh = smooth_alpha * det[3] + (1 - smooth_alpha) * ph
                prev_box = (sx, sy, sw, sh)
            else:
                prev_box = (float(det[0]), float(det[1]), float(det[2]), float(det[3]))
            boxes.append((int(prev_box[0]), int(prev_box[1]), int(prev_box[2]), int(prev_box[3])))
            missing_count = 0
        else:
            missing_count += 1
            if prev_box is not None and missing_count <= max_missing:
                boxes.append((int(prev_box[0]), int(prev_box[1]), int(prev_box[2]), int(prev_box[3])))
            else:
                boxes.append(None)

    return boxes


# --------------------------------------------------------------------------- #
# Signal preprocessing
# --------------------------------------------------------------------------- #

def bandpass_filter(signal: np.ndarray, fps: float, low_hz: float = 0.7, high_hz: float = 4.0) -> np.ndarray:
    """Butterworth bandpass filter to isolate plausible pulse band."""
    nyq = fps / 2.0
    low = low_hz / nyq
    high = high_hz / nyq
    if low <= 0 or high >= 1 or low >= high:
        return signal
    b, a = butter(4, [low, high], btype="band")
    return filtfilt(b, a, signal)


def detrend_signal(signal: np.ndarray, lambda_: float = 10.0) -> np.ndarray:
    """Smoothness priors detrending (Tarvainen et al., 2002), adapted for
    1-D signals. Removes slow baseline drift while preserving pulse
    oscillations."""
    n = len(signal)
    if n < 3:
        return signal
    # Second-difference matrix
    I = np.eye(n)
    D2 = np.diff(I, n=2, axis=0)
    # Regularized least squares: (I + lambda^2 * D2^T D2) x = signal
    H = I + lambda_ ** 2 * (D2.T @ D2)
    try:
        trend = np.linalg.solve(H, signal)
    except np.linalg.LinAlgError:
        trend = signal
    return signal - trend


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalization."""
    std = float(np.std(signal))
    if std < 1e-10:
        return signal - np.mean(signal)
    return (signal - np.mean(signal)) / std


# --------------------------------------------------------------------------- #
# POS (Plane-Orthogonal-to-Skin) method
# --------------------------------------------------------------------------- #

def _pos_projection(rgb_trace: np.ndarray, fps: float, window_sec: float = 1.6) -> np.ndarray:
    """POS method: project RGB onto a plane orthogonal to the skin-tone
    direction, then temporally filter. Based on Wang et al. (2016).

    rgb_trace: shape (n_frames, 3), columns [R, G, B] means over ROI.
    """
    n_frames = len(rgb_trace)
    window_size = int(window_sec * fps)
    if window_size < 2:
        window_size = n_frames

    pulse = np.zeros(n_frames)
    for start in range(0, n_frames, window_size):
        end = min(start + window_size, n_frames)
        block = rgb_trace[start:end]
        if len(block) < 2:
            pulse[start:end] = block[:, 1] if len(block) > 0 else 0
            continue
        # Temporal normalization within window
        mean_block = np.mean(block, axis=0)
        std_block = np.std(block, axis=0) + 1e-6
        norm_block = (block - mean_block) / std_block

        # Projection: S = 3*G - 2*R - B (orthogonal to typical skin tone)
        s = 3.0 * norm_block[:, 1] - 2.0 * norm_block[:, 0] - norm_block[:, 2]
        pulse[start:end] = s

    return pulse


# --------------------------------------------------------------------------- #
# CHROM (CHRominance-based) method
# --------------------------------------------------------------------------- #

def _chrom_projection(rgb_trace: np.ndarray, fps: float, window_sec: float = 1.6) -> np.ndarray:
    """CHROM method: use chrominance signals X and Y, then combine.
    Based on De Haan & Jeanne (2013).

    rgb_trace: shape (n_frames, 3), columns [R, G, B].
    """
    n_frames = len(rgb_trace)
    window_size = int(window_sec * fps)
    if window_size < 2:
        window_size = n_frames

    pulse = np.zeros(n_frames)
    for start in range(0, n_frames, window_size):
        end = min(start + window_size, n_frames)
        block = rgb_trace[start:end]
        if len(block) < 2:
            pulse[start:end] = block[:, 1] if len(block) > 0 else 0
            continue

        # Temporal normalization
        mean_block = np.mean(block, axis=0)
        std_block = np.std(block, axis=0) + 1e-6
        norm_block = (block - mean_block) / std_block

        r, g, b = norm_block[:, 0], norm_block[:, 1], norm_block[:, 2]
        x = 3.0 * r - 2.0 * g
        y = 1.5 * r + g - 1.5 * b

        # Standard deviation ratio for alpha
        std_x = np.std(x) + 1e-6
        std_y = np.std(y) + 1e-6
        alpha = std_x / std_y

        pulse[start:end] = x - alpha * y

    return pulse


# --------------------------------------------------------------------------- #
# Main extraction entry point
# --------------------------------------------------------------------------- #

@dataclass
class ExtractionResult:
    pulse_signal: np.ndarray
    fps: float
    timestamps: np.ndarray
    face_detection_rate: float
    roi_trace: np.ndarray          # raw (n_frames, 3) RGB mean over ROI
    roi_boxes: list[Optional[tuple[int, int, int, int]]]
    method: str


def extract_pulse_signal_from_frames(
    frames: list[np.ndarray],
    fps: float,
    method: str = "POS",
) -> ExtractionResult:
    """Full pipeline: detect/track face → extract RGB trace → POS/CHROM
    projection → bandpass → detrend → normalize."""
    _init_face_detectors()
    n_frames = len(frames)
    if n_frames == 0:
        raise ValueError("No frames provided")

    logger.info("Extracting pulse signal from %d frames @ %.2f fps (method=%s)", n_frames, fps, method)

    # 1. Face tracking
    boxes = track_face_roi(frames)
    detected_count = sum(1 for b in boxes if b is not None)
    face_detection_rate = detected_count / n_frames if n_frames > 0 else 0.0
    logger.info("Face detection rate: %.1f%% (%d/%d)", face_detection_rate * 100, detected_count, n_frames)

    if detected_count == 0:
        logger.warning("No faces detected in any frame; returning zero signal.")
        timestamps = np.arange(n_frames) / fps
        return ExtractionResult(
            pulse_signal=np.zeros(n_frames),
            fps=fps,
            timestamps=timestamps,
            face_detection_rate=0.0,
            roi_trace=np.zeros((n_frames, 3)),
            roi_boxes=boxes,
            method=method,
        )

    # 2. Extract RGB mean over ROI per frame
    rgb_trace = np.zeros((n_frames, 3))
    for i, (frame, box) in enumerate(zip(frames, boxes)):
        if box is not None:
            x, y, w, h = box
            h_img, w_img = frame.shape[:2]
            x = max(0, min(x, w_img - 1))
            y = max(0, min(y, h_img - 1))
            w = max(1, min(w, w_img - x))
            h = max(1, min(h, h_img - y))
            roi = frame[y:y+h, x:x+w]
            rgb_trace[i] = np.mean(roi, axis=(0, 1))
        else:
            # Carry forward last valid value
            if i > 0:
                rgb_trace[i] = rgb_trace[i - 1]

    # 3. POS or CHROM projection
    if method.upper() == "POS":
        raw_pulse = _pos_projection(rgb_trace, fps)
    elif method.upper() == "CHROM":
        raw_pulse = _chrom_projection(rgb_trace, fps)
    else:
        raise ValueError(f"Unknown rPPG method '{method}'. Use 'POS' or 'CHROM'.")

    # 4. Post-processing
    pulse = bandpass_filter(raw_pulse, fps)
    pulse = detrend_signal(pulse)
    pulse = normalize_signal(pulse)

    timestamps = np.arange(n_frames) / fps

    return ExtractionResult(
        pulse_signal=pulse,
        fps=fps,
        timestamps=timestamps,
        face_detection_rate=face_detection_rate,
        roi_trace=rgb_trace,
        roi_boxes=boxes,
        method=method,
    )


def extract_pulse_signal_from_video(video_path: Path, method: str = "POS") -> ExtractionResult:
    """Convenience wrapper that reads a video file then extracts pulse."""
    video_path = Path(video_path)
    props = read_video_properties(video_path)
    frames = list(iter_frames(video_path))
    return extract_pulse_signal_from_frames(frames, props.fps, method=method)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract rPPG pulse signal from a video using POS or CHROM."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--method", choices=["POS", "CHROM"], default="POS")
    parser.add_argument("--output-npy", type=Path, default=None,
                        help="Save pulse signal as .npy array")
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    result = extract_pulse_signal_from_video(args.input, method=args.method)
    print(f"Extracted {len(result.pulse_signal)} samples @ {result.fps:.2f} fps")
    print(f"Face detection rate: {result.face_detection_rate:.2%}")
    if args.output_npy:
        np.save(args.output_npy, result.pulse_signal)
        print(f"Saved pulse signal -> {args.output_npy}")


if __name__ == "__main__":
    main()
