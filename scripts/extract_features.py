#!/usr/bin/env python3
"""
Offline preprocessing and ROI feature extraction.

Processes raw synchronized thermal+RGB video pairs into pre-extracted
ROI arrays ready for training. This is a one-time preprocessing step.

For each subject/session, produces:
  - rgb_rois.npy:     (N_frames, 5, 3, 128, 128) uint8
  - thermal_rois.npy: (N_frames, 5, 1, 128, 128) float32 (z-normalised)
  - labels.csv:       timestamp, nrs_score columns
  - sync_stats.json:  synchronisation statistics

Usage:
    python scripts/extract_features.py \\
        --data_root /path/to/raw_data \\
        --output_dir /path/to/features \\
        --dataset dataset1 \\
        --homography_path calibration/H.npy \\
        --num_workers 4
"""

import argparse
import json
import logging
import multiprocessing
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import (
    Preprocessor,
    ThermalNormaliser,
    ROI_NAMES,
)
from src.utils.sync import FrameSynchroniser, ThermalVideoReader
from src.utils.logging_utils import setup_logging

logger = logging.getLogger("csaf.extract")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract ROI features from raw video.")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root of raw data (contains subject_XXX/ directories).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for extracted features.")
    parser.add_argument("--dataset", type=str, default="dataset1",
                        choices=["dataset1", "dataset2"])
    parser.add_argument("--homography_path", type=str, default=None,
                        help="Path to pre-computed homography .npy file.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Specific subject IDs to process (default: all).")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--baseline_seconds", type=float, default=180.0,
                        help="Baseline duration for thermal normalisation (s).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing features.")
    return parser.parse_args()


def process_session(
    session_dir: Path,
    output_session_dir: Path,
    preprocessor: Preprocessor,
    fps: float = 30.0,
    baseline_seconds: float = 180.0,
    overwrite: bool = False,
) -> Optional[dict]:
    """
    Process a single recording session.

    Expected session_dir layout:
        session_dir/
            rgb/          # RGB frames as PNG (or rgb_video.mp4)
            thermal/      # thermal frames as .npy (or thermal.npy stack)
            labels.csv    # frame-level NRS labels
            sync_log.csv  # DAQ timestamp log (optional)
    """
    output_session_dir.mkdir(parents=True, exist_ok=True)

    rgb_rois_path = output_session_dir / "rgb_rois.npy"
    if rgb_rois_path.exists() and not overwrite:
        logger.info(f"Skipping {session_dir.name} (already processed).")
        return None

    # ── Load RGB video ─────────────────────────────────────────────────────
    rgb_dir = session_dir / "rgb"
    rgb_video = session_dir / "rgb_video.mp4"

    if rgb_dir.exists():
        frame_paths = sorted(rgb_dir.glob("*.png")) + sorted(rgb_dir.glob("*.jpg"))
        rgb_frames = [cv2.imread(str(p)) for p in frame_paths]
    elif rgb_video.exists():
        cap = cv2.VideoCapture(str(rgb_video))
        rgb_frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frames.append(frame)
        cap.release()
    else:
        logger.error(f"No RGB data found in {session_dir}")
        return None

    # ── Load thermal data ──────────────────────────────────────────────────
    thermal_npy = session_dir / "thermal.npy"
    thermal_reader = ThermalVideoReader(str(thermal_npy))
    thermal_frames = thermal_reader.load()   # (N_thermal, 480, 640) float32

    # ── Synchronise ────────────────────────────────────────────────────────
    sync_log = session_dir / "sync_log.csv"
    if sync_log.exists():
        synchroniser = FrameSynchroniser()
        rgb_ts, th_ts = synchroniser.load_daq_log(str(sync_log))
        rgb_idx, thermal_idx, sync_stats = synchroniser.align_from_timestamps(
            rgb_ts, th_ts
        )
        rgb_frames = [rgb_frames[i] for i in rgb_idx if i < len(rgb_frames)]
        thermal_frames = thermal_frames[thermal_idx[thermal_idx < len(thermal_frames)]]
        sync_info = {
            "n_pairs": sync_stats.n_frame_pairs,
            "median_offset_ms": sync_stats.median_offset_ms,
            "max_offset_ms": sync_stats.max_offset_ms,
        }
    else:
        # Assume already synchronised; truncate to shorter sequence
        n = min(len(rgb_frames), len(thermal_frames))
        rgb_frames = rgb_frames[:n]
        thermal_frames = thermal_frames[:n]
        sync_info = {"n_pairs": n, "note": "assumed_pre_synced"}

    n_frames = min(len(rgb_frames), len(thermal_frames))
    logger.info(f"{session_dir.name}: {n_frames} synchronised frame pairs")

    # ── Thermal normaliser fit on baseline ────────────────────────────────
    normaliser = ThermalNormaliser(
        baseline_duration_s=baseline_seconds, fps=fps
    )
    n_baseline = int(baseline_seconds * fps)

    # Collect baseline ROI crops for fitting
    baseline_rois: dict = {roi: [] for roi in ROI_NAMES}
    for frame_idx in range(min(n_baseline, n_frames)):
        rois = preprocessor.process_frame(
            rgb_frames[frame_idx], thermal_frames[frame_idx]
        )
        if rois is not None:
            for roi_name in ROI_NAMES:
                baseline_rois[roi_name].append(rois[roi_name]["thermal"])

    for roi_name in ROI_NAMES:
        if baseline_rois[roi_name]:
            normaliser.fit({roi_name: baseline_rois[roi_name]})

    # ── Extract all frames ─────────────────────────────────────────────────
    all_rgb_rois = []    # (N, 5, 3, 128, 128) uint8
    all_thermal_rois = []  # (N, 5, 1, 128, 128) float32
    valid_frame_indices = []

    for frame_idx in tqdm(range(n_frames), desc=session_dir.name, leave=False):
        rois = preprocessor.process_frame(
            rgb_frames[frame_idx], thermal_frames[frame_idx]
        )
        if rois is None:
            continue

        rgb_stack = np.stack(
            [rois[r]["rgb"] for r in ROI_NAMES], axis=0
        )  # (5, 128, 128, 3) uint8
        rgb_stack = rgb_stack.transpose(0, 3, 1, 2)   # (5, 3, 128, 128)

        th_stack = np.stack(
            [normaliser.transform(rois[r]["thermal"], r) for r in ROI_NAMES],
            axis=0
        )  # (5, 128, 128)
        th_stack = th_stack[:, np.newaxis, :, :]       # (5, 1, 128, 128)

        all_rgb_rois.append(rgb_stack)
        all_thermal_rois.append(th_stack)
        valid_frame_indices.append(frame_idx)

    if not all_rgb_rois:
        logger.error(f"No valid frames extracted from {session_dir.name}")
        return None

    rgb_array = np.stack(all_rgb_rois, axis=0).astype(np.uint8)    # (N, 5, 3, 128, 128)
    th_array = np.stack(all_thermal_rois, axis=0).astype(np.float32)  # (N, 5, 1, 128, 128)

    # ── Save features ──────────────────────────────────────────────────────
    np.save(output_session_dir / "rgb_rois.npy", rgb_array)
    np.save(output_session_dir / "thermal_rois.npy", th_array)

    # Copy and align labels
    labels_path = session_dir / "labels.csv"
    if labels_path.exists():
        labels_df = pd.read_csv(labels_path)
        labels_df = labels_df.iloc[valid_frame_indices].reset_index(drop=True)
        labels_df.to_csv(output_session_dir / "labels.csv", index=False)

    # Save sync stats
    detection_rate = preprocessor.detection_rate
    stats = {
        "n_frames": len(rgb_array),
        "face_detection_rate": detection_rate,
        "sync": sync_info,
    }
    with open(output_session_dir / "sync_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(
        f"  Saved: {len(rgb_array)} frames, "
        f"detection rate = {detection_rate:.3f}, "
        f"shape = {rgb_array.shape}"
    )
    return stats


def main() -> None:
    args = parse_args()
    setup_logging(args.output_dir, run_name="extract")
    data_root = Path(args.data_root)
    output_root = Path(args.output_dir)

    preprocessor = Preprocessor(
        homography_path=args.homography_path,
        use_gpu=True,
    )

    # Discover subjects
    if args.subjects:
        subject_dirs = [data_root / s for s in args.subjects]
    else:
        subject_dirs = sorted(data_root.glob("subject_*")) + \
                       sorted(data_root.glob("patient_*"))

    logger.info(f"Found {len(subject_dirs)} subjects to process.")
    total_stats = []

    for subject_dir in subject_dirs:
        if not subject_dir.is_dir():
            continue
        subject_id = subject_dir.name
        logger.info(f"Processing subject: {subject_id}")

        session_dirs = sorted(subject_dir.glob("session_*"))
        if not session_dirs:
            # Treat subject dir itself as a single session
            session_dirs = [subject_dir]

        for session_dir in session_dirs:
            output_session = output_root / subject_id / session_dir.name
            stats = process_session(
                session_dir=session_dir,
                output_session_dir=output_session,
                preprocessor=preprocessor,
                fps=args.fps,
                baseline_seconds=args.baseline_seconds,
                overwrite=args.overwrite,
            )
            if stats:
                stats["subject_id"] = subject_id
                total_stats.append(stats)

    # Summary
    if total_stats:
        total_frames = sum(s.get("n_frames", 0) for s in total_stats)
        avg_detection = np.mean([s.get("face_detection_rate", 0) for s in total_stats])
        logger.info("=" * 60)
        logger.info(f"Extraction complete: {len(total_stats)} sessions processed")
        logger.info(f"Total frames: {total_frames:,}")
        logger.info(f"Avg face detection rate: {avg_detection:.4f}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
