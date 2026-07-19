"""
pulse_metrics.py
=================

Converts an extracted rPPG pulse waveform into clinically meaningful
metrics: heart rate (HR), heart-rate variability (HRV), Pulse SNR, and
Signal Quality Index (SQI). Compares estimated HR against contact-sensor
ground truth when available.

HR estimation uses Welch's method for power spectral density (PSD), then
picks the dominant peak within the physiologically plausible band
[0.7, 4.0] Hz (≈ 42–240 bpm). This is more robust than simple FFT peak
picking on short clips.

HRV is computed from the inter-peak intervals of the detrended pulse
signal, reported as RMSSD (root mean square of successive differences).

Pulse SNR measures how much the pulse band energy dominates over
out-of-band noise, giving a dB-scale quality score.

SQI (Signal Quality Index) is a composite 0–1 score combining:
    - pulse SNR (higher = better)
    - face detection rate (higher = better)
    - pulse waveform kurtosis (closer to Gaussian = more plausible)

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
from scipy.signal import find_peaks, welch
from scipy.stats import kurtosis

logger = logging.getLogger("pulse_metrics")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# Physiological pulse band
PULSE_LOW_HZ = 0.7   # ≈ 42 bpm
PULSE_HIGH_HZ = 4.0  # ≈ 240 bpm


# --------------------------------------------------------------------------- #
# Heart rate estimation
# --------------------------------------------------------------------------- #

def estimate_hr_from_pulse(
    pulse_signal: np.ndarray,
    fps: float,
    nperseg: int = 256,
) -> float:
    """Estimate heart rate (bpm) from pulse waveform via Welch PSD peak
    detection in the physiologically plausible band."""
    if len(pulse_signal) < nperseg:
        nperseg = max(8, len(pulse_signal) // 2)

    freqs, psd = welch(pulse_signal, fs=fps, nperseg=nperseg, noverlap=nperseg // 2)

    # Mask to pulse band
    band_mask = (freqs >= PULSE_LOW_HZ) & (freqs <= PULSE_HIGH_HZ)
    if not band_mask.any():
        return 0.0

    band_freqs = freqs[band_mask]
    band_psd = psd[band_mask]

    # Peak frequency -> bpm
    peak_idx = int(np.argmax(band_psd))
    peak_freq = band_freqs[peak_idx]
    hr_bpm = peak_freq * 60.0
    return float(np.clip(hr_bpm, 42.0, 240.0))


# --------------------------------------------------------------------------- #
# HRV computation
# --------------------------------------------------------------------------- #

def compute_hrv_rmssd(pulse_signal: np.ndarray, fps: float) -> Optional[float]:
    """Compute HRV as RMSSD (ms) from inter-peak intervals of the pulse
    signal. Returns None if too few peaks are found."""
    # Normalize and find peaks
    sig = pulse_signal - np.mean(pulse_signal)
    if np.std(sig) > 1e-6:
        sig = sig / np.std(sig)

    # Minimum distance between peaks: ~300 ms (200 bpm max)
    min_distance = int(fps * 0.3)
    if min_distance < 1:
        min_distance = 1

    peaks, _ = find_peaks(sig, distance=min_distance, prominence=0.3)
    if len(peaks) < 3:
        return None

    # Inter-peak intervals in seconds
    ipis = np.diff(peaks) / fps
    if len(ipis) < 2:
        return None

    # RMSSD in milliseconds
    rmssd = float(np.sqrt(np.mean(np.diff(ipis) ** 2)) * 1000.0)
    return rmssd


# --------------------------------------------------------------------------- #
# Pulse SNR
# --------------------------------------------------------------------------- #

def compute_pulse_snr(pulse_signal: np.ndarray, fps: float, nperseg: int = 256) -> float:
    """Pulse SNR in dB: ratio of in-band (pulse) PSD energy to out-of-band
    noise PSD energy. Higher = cleaner pulse signal."""
    if len(pulse_signal) < nperseg:
        nperseg = max(8, len(pulse_signal) // 2)

    freqs, psd = welch(pulse_signal, fs=fps, nperseg=nperseg, noverlap=nperseg // 2)

    band_mask = (freqs >= PULSE_LOW_HZ) & (freqs <= PULSE_HIGH_HZ)
    if not band_mask.any():
        return -99.0

    signal_power = float(np.sum(psd[band_mask]))
    noise_power = float(np.sum(psd[~band_mask])) + 1e-12

    if signal_power <= 0:
        return -99.0
    snr_db = float(10.0 * np.log10(signal_power / noise_power))
    return snr_db


# --------------------------------------------------------------------------- #
# Signal Quality Index (SQI)
# --------------------------------------------------------------------------- #

def compute_sqi(
    pulse_signal: np.ndarray,
    fps: float,
    face_detection_rate: float,
) -> float:
    """Composite Signal Quality Index, 0–1 scale.

    Combines:
        - pulse SNR (normalized to ~0–1 over typical operating range)
        - face detection rate (directly)
        - waveform kurtosis penalty (excess kurtosis far from 0 suggests
          artifact / non-physiological signal)
    """
    snr_db = compute_pulse_snr(pulse_signal, fps)
    # Map SNR: -10 dB -> 0, 20 dB -> 1 (sigmoid-like clipping)
    snr_score = float(np.clip((snr_db + 10.0) / 30.0, 0.0, 1.0))

    # Kurtosis: ideal Gaussian = 3 (excess kurtosis = 0)
    k = float(kurtosis(pulse_signal, fisher=True))  # excess kurtosis
    kurt_score = float(np.clip(1.0 - abs(k) / 5.0, 0.0, 1.0))

    # Weighted combination
    sqi = 0.5 * snr_score + 0.3 * face_detection_rate + 0.2 * kurt_score
    return float(np.clip(sqi, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Ground-truth comparison
# --------------------------------------------------------------------------- #

def compute_hr_mae(estimated_hr: float, true_hr: float) -> float:
    """Absolute error in bpm."""
    return abs(estimated_hr - true_hr)


def compute_hr_mape(estimated_hr: float, true_hr: float) -> float:
    """Mean absolute percentage error."""
    if true_hr < 1e-6:
        return float("inf")
    return abs(estimated_hr - true_hr) / true_hr * 100.0


# --------------------------------------------------------------------------- #
# Aggregate assessment result
# --------------------------------------------------------------------------- #

@dataclass
class PulseAssessment:
    hr_bpm: float
    hrv_rmssd_ms: Optional[float]
    pulse_snr_db: float
    sqi: float
    hr_mae_vs_truth: Optional[float]  # None if no ground truth provided
    hr_mape_vs_truth: Optional[float]

    def to_dict(self) -> dict:
        return {
            "hr_bpm": self.hr_bpm,
            "hrv_rmssd_ms": self.hrv_rmssd_ms,
            "pulse_snr_db": self.pulse_snr_db,
            "sqi": self.sqi,
            "hr_mae_vs_truth": self.hr_mae_vs_truth,
            "hr_mape_vs_truth": self.hr_mape_vs_truth,
        }


def assess_pulse_signal(
    pulse_signal: np.ndarray,
    fps: float,
    face_detection_rate: float,
    true_hr_bpm: Optional[float] = None,
) -> PulseAssessment:
    """Full assessment of a pulse signal: HR, HRV, SNR, SQI, and optional
    ground-truth comparison."""
    hr = estimate_hr_from_pulse(pulse_signal, fps)
    hrv = compute_hrv_rmssd(pulse_signal, fps)
    snr = compute_pulse_snr(pulse_signal, fps)
    sqi = compute_sqi(pulse_signal, fps, face_detection_rate)

    mae = compute_hr_mae(hr, true_hr_bpm) if true_hr_bpm is not None else None
    mape = compute_hr_mape(hr, true_hr_bpm) if true_hr_bpm is not None else None

    logger.info(
        "Pulse assessment: HR=%.1f bpm  SNR=%.2f dB  SQI=%.3f  MAE=%s",
        hr, snr, sqi,
        f"{mae:.2f}" if mae is not None else "n/a",
    )

    return PulseAssessment(
        hr_bpm=hr,
        hrv_rmssd_ms=hrv,
        pulse_snr_db=snr,
        sqi=sqi,
        hr_mae_vs_truth=mae,
        hr_mape_vs_truth=mape,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assess pulse signal quality and estimate HR/HRV."
    )
    parser.add_argument("--pulse-npy", required=True, type=Path,
                        help="Path to .npy file containing pulse signal")
    parser.add_argument("--fps", required=True, type=float)
    parser.add_argument("--face-detection-rate", type=float, default=1.0)
    parser.add_argument("--true-hr", type=float, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    pulse = np.load(args.pulse_npy)
    assessment = assess_pulse_signal(
        pulse, args.fps, args.face_detection_rate, true_hr_bpm=args.true_hr,
    )
    d = assessment.to_dict()
    print(json.dumps(d, indent=2))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(d, f, indent=2)


if __name__ == "__main__":
    main()
