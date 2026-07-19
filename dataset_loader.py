"""
dataset_loader.py
==================

Loads UBFC-rPPG, PURE, and MMPD datasets (videos + ground-truth
annotations + skin-tone metadata) and produces the normalized
"labeling manifest" CSV that labeling_harness.py and
router_classifier.py consume.

Supported datasets:
    - UBFC-rPPG: video files + groundTruth.txt (BVP + HR)
    - PURE: video folders + json-signals-groundtruth.json (BVP + HR)
    - MMPD: video files + labels.csv (HR + Fitzpatrick skin-tone)

All datasets are normalized to a common manifest format with columns:
    video_path, subject_id, window_id, true_hr_bpm, fitzpatrick

Subject-level (leakage-safe) train/test splitting is performed here,
ensuring no subject appears in both splits. This is the ONLY place in
the codebase where dataset-specific parsing happens; everything
downstream works on the normalized manifest.

Author: Signal & Image Processing project (NMIMS Mumbai), MoSICom 2026.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

logger = logging.getLogger("dataset_loader")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


# --------------------------------------------------------------------------- #
# UBFC-rPPG loader
# --------------------------------------------------------------------------- #

def load_ubfc_rppg(dataset_dir: Path) -> pd.DataFrame:
    """Load UBFC-rPPG dataset.

    Expected structure:
        dataset_dir/
            subject1/
                vid.avi
                groundTruth.txt
            subject2/
                ...

    groundTruth.txt format: two columns [BVP_sample, HR_bpm], one per frame.
    We average the HR column to get a single ground-truth HR per video.
    """
    dataset_dir = Path(dataset_dir)
    rows = []
    for subject_dir in sorted(dataset_dir.iterdir()):
        if not subject_dir.is_dir():
            continue
        video_file = subject_dir / "vid.avi"
        gt_file = subject_dir / "groundTruth.txt"
        if not video_file.exists() or not gt_file.exists():
            logger.warning("Skipping %s: missing vid.avi or groundTruth.txt", subject_dir.name)
            continue
        try:
            gt = np.loadtxt(gt_file)
            if gt.ndim == 2 and gt.shape[1] >= 2:
                hr_values = gt[:, 1]
            elif gt.ndim == 1:
                hr_values = gt
            else:
                logger.warning("Unexpected groundTruth shape in %s: %s", subject_dir.name, gt.shape)
                continue
            true_hr = float(np.median(hr_values))
            rows.append({
                "video_path": str(video_file),
                "subject_id": subject_dir.name,
                "window_id": f"{subject_dir.name}_full",
                "true_hr_bpm": true_hr,
                "fitzpatrick": None,
            })
        except Exception as e:
            logger.error("Failed to load %s: %s", subject_dir.name, e)

    logger.info("UBFC-rPPG: loaded %d clips from %s", len(rows), dataset_dir)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# PURE loader
# --------------------------------------------------------------------------- #

def load_pure(dataset_dir: Path) -> pd.DataFrame:
    """Load PURE dataset.

    Expected structure:
        dataset_dir/
            01-01/
                01-01.avi
                01-01.json
            01-02/
                ...

    JSON format (simplified): {"/FullPackage": [{"Pulse": hr_bpm, ...}, ...]}
    We extract the median HR from the JSON.
    """
    dataset_dir = Path(dataset_dir)
    rows = []
    for subject_dir in sorted(dataset_dir.iterdir()):
        if not subject_dir.is_dir():
            continue
        video_files = list(subject_dir.glob("*.avi")) + list(subject_dir.glob("*.mp4"))
        json_files = list(subject_dir.glob("*.json"))
        if not video_files or not json_files:
            logger.warning("Skipping %s: missing video or JSON", subject_dir.name)
            continue
        video_file = video_files[0]
        json_file = json_files[0]
        try:
            with open(json_file) as f:
                data = json.load(f)
            # PURE JSON structure varies; try common keys
            hr_values = []
            if "/FullPackage" in data:
                for entry in data["/FullPackage"]:
                    if "Pulse" in entry:
                        hr_values.append(float(entry["Pulse"]))
            elif "pulse" in data:
                hr_values = [float(v) for v in data["pulse"]]
            if not hr_values:
                logger.warning("No HR values found in %s", json_file)
                continue
            true_hr = float(np.median(hr_values))
            rows.append({
                "video_path": str(video_file),
                "subject_id": subject_dir.name.split("-")[0],  # e.g. "01" from "01-01"
                "window_id": subject_dir.name,
                "true_hr_bpm": true_hr,
                "fitzpatrick": None,
            })
        except Exception as e:
            logger.error("Failed to load %s: %s", subject_dir.name, e)

    logger.info("PURE: loaded %d clips from %s", len(rows), dataset_dir)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# MMPD loader
# --------------------------------------------------------------------------- #

def load_mmpd(dataset_dir: Path) -> pd.DataFrame:
    """Load MMPD dataset.

    Expected structure:
        dataset_dir/
            videos/
                subject_001.mp4
                subject_002.mp4
                ...
            labels.csv
                video_file,hr_bpm,fitzpatrick,...

    The labels CSV must contain at minimum: video_file, hr_bpm columns.
    fitzpatrick is optional but strongly recommended for fairness analysis.
    """
    dataset_dir = Path(dataset_dir)
    videos_dir = dataset_dir / "videos"
    labels_csv = dataset_dir / "labels.csv"

    if not labels_csv.exists():
        # Try alternative names
        for alt in ["label.csv", "annotations.csv", "gt.csv"]:
            if (dataset_dir / alt).exists():
                labels_csv = dataset_dir / alt
                break

    if not labels_csv.exists():
        raise FileNotFoundError(f"MMPD labels CSV not found in {dataset_dir}")

    labels_df = pd.read_csv(labels_csv)
    # Normalize column names
    labels_df.columns = [c.lower().strip() for c in labels_df.columns]

    # Map common column name variants
    col_map = {}
    for c in labels_df.columns:
        if c in ["video_file", "video", "filename", "file", "video_path"]:
            col_map[c] = "video_file"
        elif c in ["hr", "hr_bpm", "heart_rate", "bpm", "true_hr"]:
            col_map[c] = "hr_bpm"
        elif c in ["fitzpatrick", "skin_tone", "skintone", "fp_type"]:
            col_map[c] = "fitzpatrick"

    labels_df = labels_df.rename(columns=col_map)

    if "video_file" not in labels_df.columns or "hr_bpm" not in labels_df.columns:
        raise ValueError(f"MMPD labels CSV missing required columns. Found: {list(labels_df.columns)}")

    rows = []
    for _, row in labels_df.iterrows():
        video_file = videos_dir / row["video_file"] if videos_dir.exists() else dataset_dir / row["video_file"]
        if not video_file.exists():
            # Try without videos/ prefix
            video_file = dataset_dir / row["video_file"]
        subject_id = Path(row["video_file"]).stem
        rows.append({
            "video_path": str(video_file),
            "subject_id": subject_id,
            "window_id": subject_id,
            "true_hr_bpm": float(row["hr_bpm"]),
            "fitzpatrick": row.get("fitzpatrick", None),
        })

    logger.info("MMPD: loaded %d clips from %s", len(rows), dataset_dir)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Generic loader dispatcher
# --------------------------------------------------------------------------- #

def load_dataset(dataset_dir: Path, dataset_type: str) -> pd.DataFrame:
    """Load any supported dataset by type."""
    dataset_type = dataset_type.lower()
    if dataset_type in ("ubfc", "ubfc-rppg", "ubfc_rppg"):
        return load_ubfc_rppg(dataset_dir)
    elif dataset_type == "pure":
        return load_pure(dataset_dir)
    elif dataset_type == "mmpd":
        return load_mmpd(dataset_dir)
    else:
        raise ValueError(f"Unknown dataset type '{dataset_type}'. Supported: ubfc, pure, mmpd")


# --------------------------------------------------------------------------- #
# Sliding-window segmentation (for long videos)
# --------------------------------------------------------------------------- #

def segment_into_windows(
    manifest_df: pd.DataFrame,
    window_sec: float = 10.0,
    stride_sec: float = 5.0,
    fps: float = 30.0,
) -> pd.DataFrame:
    """Split long videos into overlapping sliding windows. Each window gets
    its own row in the manifest with a unique window_id."""
    from degradation_injector import read_video_properties

    rows = []
    for _, row in manifest_df.iterrows():
        video_path = Path(row["video_path"])
        try:
            props = read_video_properties(video_path)
            actual_fps = props.fps
            n_frames = props.n_frames
        except Exception:
            actual_fps = fps
            n_frames = int(fps * 60)  # guess 60s if we can't read

        window_frames = int(window_sec * actual_fps)
        stride_frames = int(stride_sec * actual_fps)

        if n_frames <= window_frames:
            rows.append(row.to_dict())
            continue

        for start in range(0, n_frames - window_frames + 1, stride_frames):
            end = start + window_frames
            window_id = f"{row['window_id']}_w{start:06d}_{end:06d}"
            rows.append({
                "video_path": row["video_path"],
                "subject_id": row["subject_id"],
                "window_id": window_id,
                "true_hr_bpm": row["true_hr_bpm"],
                "fitzpatrick": row.get("fitzpatrick", None),
                "window_start_frame": start,
                "window_end_frame": end,
            })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Subject-level train/test split
# --------------------------------------------------------------------------- #

def subject_level_split_manifest(
    manifest_df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split manifest rows into train/test with no subject leakage."""
    n_subjects = manifest_df["subject_id"].nunique()
    if n_subjects < 2:
        raise ValueError(f"Need at least 2 subjects for splitting, found {n_subjects}")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(manifest_df, groups=manifest_df["subject_id"]))
    train_df = manifest_df.iloc[train_idx].reset_index(drop=True)
    test_df = manifest_df.iloc[test_idx].reset_index(drop=True)

    overlap = set(train_df["subject_id"]) & set(test_df["subject_id"])
    assert not overlap, f"Subject leakage detected: {overlap}"

    logger.info(
        "Split: %d train rows (%d subjects), %d test rows (%d subjects)",
        len(train_df), train_df["subject_id"].nunique(),
        len(test_df), test_df["subject_id"].nunique(),
    )
    return train_df, test_df


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load a dataset and produce a normalized labeling manifest."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    load = sub.add_parser("load", help="Load a dataset and write manifest CSV.")
    load.add_argument("--dataset-dir", required=True, type=Path)
    load.add_argument("--dataset-type", required=True, choices=["ubfc", "pure", "mmpd"])
    load.add_argument("--output-csv", required=True, type=Path)
    load.add_argument("--window-sec", type=float, default=0.0,
                       help="If > 0, segment videos into sliding windows of this duration")
    load.add_argument("--stride-sec", type=float, default=5.0)
    load.add_argument("--fps", type=float, default=30.0)

    split = sub.add_parser("split", help="Split an existing manifest into train/test.")
    split.add_argument("--manifest", required=True, type=Path)
    split.add_argument("--test-size", type=float, default=0.2)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--train-csv", required=True, type=Path)
    split.add_argument("--test-csv", required=True, type=Path)

    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.mode == "load":
        df = load_dataset(args.dataset_dir, args.dataset_type)
        if args.window_sec > 0:
            df = segment_into_windows(df, window_sec=args.window_sec,
                                       stride_sec=args.stride_sec, fps=args.fps)
        df.to_csv(args.output_csv, index=False)
        logger.info("Wrote manifest (%d rows) -> %s", len(df), args.output_csv)

    elif args.mode == "split":
        df = pd.read_csv(args.manifest)
        train_df, test_df = subject_level_split_manifest(df, test_size=args.test_size, seed=args.seed)
        train_df.to_csv(args.train_csv, index=False)
        test_df.to_csv(args.test_csv, index=False)
        logger.info("Wrote train (%d rows) -> %s", len(train_df), args.train_csv)
        logger.info("Wrote test  (%d rows) -> %s", len(test_df), args.test_csv)


if __name__ == "__main__":
    main()
