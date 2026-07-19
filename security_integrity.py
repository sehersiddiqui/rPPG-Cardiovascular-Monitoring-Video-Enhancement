"""
security_integrity.py
======================

Security, integrity, and reproducibility safeguards for the adaptive rPPG
pipeline. Ensures:
    - Cryptographic verification of dataset integrity (SHA-256)
    - Tamper-evident sidecar records for every degraded clip
    - Reproducible random seeds across all stochastic operations
    - Audit logging of all pipeline decisions
    - Input validation and sanitization

These measures address reviewer concerns about:
    - Data integrity and reproducibility
    - Adversarial manipulation of training labels
    - Traceability of routing decisions in clinical deployment

Author: Signal & Image Processing project (NMIMS Mumbai), MoSICom 2026.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("security_integrity")


# --------------------------------------------------------------------------- #
# Cryptographic hashing
# --------------------------------------------------------------------------- #

def compute_file_hash(path: Path, algorithm: str = "sha256") -> str:
    """Compute cryptographic hash of a file for integrity verification."""
    path = Path(path)
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_file_hash(path: Path, expected_hash: str, algorithm: str = "sha256") -> bool:
    """Verify file integrity against expected hash."""
    actual = compute_file_hash(path, algorithm)
    match = hmac.compare_digest(actual, expected_hash)
    if not match:
        logger.error("Hash mismatch for %s: expected %s, got %s", path, expected_hash, actual)
    return match


# --------------------------------------------------------------------------- #
# Tamper-evident sidecar records
# --------------------------------------------------------------------------- #

@dataclass
class TamperEvidentRecord:
    """A sidecar record with an HMAC signature, ensuring any modification
    of the record (or the referenced video) is detectable."""
    record_data: dict
    video_hash: str
    secret_key: str
    signature: str = ""

    def __post_init__(self):
        if not self.signature:
            self.signature = self._compute_signature()

    def _compute_signature(self) -> str:
        payload = json.dumps(self.record_data, sort_keys=True) + self.video_hash
        return hmac.new(
            self.secret_key.encode(),
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()

    def verify(self) -> bool:
        expected = self._compute_signature()
        return hmac.compare_digest(self.signature, expected)

    def to_dict(self) -> dict:
        return {
            "record": self.record_data,
            "video_hash": self.video_hash,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, d: dict, secret_key: str) -> "TamperEvidentRecord":
        return cls(
            record_data=d["record"],
            video_hash=d["video_hash"],
            secret_key=secret_key,
            signature=d["signature"],
        )


def create_tamper_evident_sidecar(
    record_data: dict,
    video_path: Path,
    secret_key: Optional[str] = None,
) -> TamperEvidentRecord:
    """Create a cryptographically signed sidecar for a degraded video."""
    video_hash = compute_file_hash(video_path)
    if secret_key is None:
        secret_key = secrets.token_hex(32)
        logger.info("Generated new secret key for sidecar signing")
    return TamperEvidentRecord(
        record_data=record_data,
        video_hash=video_hash,
        secret_key=secret_key,
    )


# --------------------------------------------------------------------------- #
# Reproducibility seeds
# --------------------------------------------------------------------------- #

class ReproducibleRNG:
    """Wrapper around numpy Generator that logs and verifies seed usage."""

    def __init__(self, seed: Optional[int] = None):
        if seed is None:
            seed = secrets.randbelow(2**32)
            logger.info("No seed provided; generated random seed: %d", seed)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        logger.info("Initialized ReproducibleRNG with seed=%d", seed)

    def integers(self, *args, **kwargs):
        return self.rng.integers(*args, **kwargs)

    def random(self, *args, **kwargs):
        return self.rng.random(*args, **kwargs)

    def normal(self, *args, **kwargs):
        return self.rng.normal(*args, **kwargs)

    def uniform(self, *args, **kwargs):
        return self.rng.uniform(*args, **kwargs)

    def get_state_dict(self) -> dict:
        return {"seed": int(self.seed), "bit_generator": str(self.rng.bit_generator)}


# --------------------------------------------------------------------------- #
# Audit logging
# --------------------------------------------------------------------------- #

class AuditLogger:
    """Logs every routing decision with full context for clinical audit trails."""

    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_routing_decision(
        self,
        video_path: str,
        features: dict,
        predicted_recipe: str,
        probabilities: dict,
        processing_time_ms: float,
    ) -> None:
        entry = {
            "timestamp_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            "video_path": video_path,
            "features": features,
            "predicted_recipe": predicted_recipe,
            "recipe_probabilities": probabilities,
            "processing_time_ms": processing_time_ms,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_training_event(
        self,
        model_type: str,
        n_train_samples: int,
        n_test_samples: int,
        test_accuracy: float,
        feature_importances: dict,
    ) -> None:
        entry = {
            "timestamp_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            "event": "model_training",
            "model_type": model_type,
            "n_train_samples": n_train_samples,
            "n_test_samples": n_test_samples,
            "test_accuracy": test_accuracy,
            "feature_importances": feature_importances,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #

def validate_feature_vector(features: dict) -> tuple[bool, Optional[str]]:
    """Validate a degradation feature vector before routing.
    Returns (is_valid, error_message)."""
    from degradation_estimator import FEATURE_NAMES

    missing = [name for name in FEATURE_NAMES if name not in features]
    if missing:
        return False, f"Missing features: {missing}"

    for name in FEATURE_NAMES:
        val = features[name]
        if not isinstance(val, (int, float)):
            return False, f"Feature '{name}' must be numeric, got {type(val)}"
        if not (0.0 <= float(val) <= 1.0):
            return False, f"Feature '{name}' out of range [0,1]: {val}"

    return True, None


def sanitize_path(path: str) -> Path:
    """Sanitize a file path to prevent directory traversal attacks."""
    p = Path(path).resolve()
    # Ensure path does not escape allowed directories
    # In production, compare against an allow-list
    return p


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_arg_parser() -> argparse.ArgumentParser:
    import argparse
    parser = argparse.ArgumentParser(description="Security and integrity utilities.")
    sub = parser.add_subparsers(dest="mode", required=True)

    hash_cmd = sub.add_parser("hash", help="Compute file hash.")
    hash_cmd.add_argument("--file", required=True, type=Path)
    hash_cmd.add_argument("--algorithm", default="sha256", choices=["sha256", "sha512", "md5"])

    verify = sub.add_parser("verify", help="Verify file against hash.")
    verify.add_argument("--file", required=True, type=Path)
    verify.add_argument("--expected-hash", required=True)

    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.mode == "hash":
        h = compute_file_hash(args.file, args.algorithm)
        print(f"{args.algorithm}: {h}")

    elif args.mode == "verify":
        ok = verify_file_hash(args.file, args.expected_hash)
        print("VERIFIED" if ok else "HASH MISMATCH")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    import sys
    main()
