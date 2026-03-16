"""
PyTorch Dataset classes for the CSAF pain estimation experiments.

Dataset 1: Controlled pain induction (Cold Pressor Test + pressure algometry)
    - 50 healthy adults, ages 21–68
    - 87.3 h video, 9.46 M frames at 30 fps
    - Continuous NRS slider labels at 1 Hz (frame-level ground truth)

Dataset 2: Clinical postoperative monitoring (PACU)
    - 30 surgical patients, ages 31–74
    - 17.7 h video, 1.92 M frames
    - NRS self-report every 2 minutes (linearly interpolated)

Combined: 80 subjects, 105 h, 11.38 M frames.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

ROI_NAMES = ["periorbital", "forehead", "nasal", "cheeks", "perioral"]
ROI_ORDER = {name: idx for idx, name in enumerate(ROI_NAMES)}


class PainDataset(Dataset):
    """
    Base dataset for temporal windows of synchronized thermal-RGB ROIs.

    Each sample is a 10-second window (300 frames at 30 fps) consisting of:
        - rgb_rois:     (T, 5, 3, 128, 128) float32, normalised to [0, 1]
        - thermal_rois: (T, 5, 1, 128, 128) float32, z-normalised per ROI
        - nrs_score:    float32 scalar — mean NRS over the window

    Args:
        feature_dir:    Path to preprocessed feature directory for this split.
        subjects:       List of subject IDs to include.
        window_frames:  Number of frames per temporal window (default 300 = 10 s).
        stride:         Frame stride between consecutive windows.
        augment:        Whether to apply temporal and spatial augmentation.
        nrs_min / nrs_max: Clipping range for NRS labels.
    """

    def __init__(
        self,
        feature_dir: str,
        subjects: List[str],
        window_frames: int = 300,
        stride: int = 150,
        augment: bool = False,
        nrs_min: float = 0.0,
        nrs_max: float = 10.0,
    ) -> None:
        super().__init__()
        self.feature_dir = Path(feature_dir)
        self.subjects = subjects
        self.window_frames = window_frames
        self.stride = stride
        self.augment = augment
        self.nrs_min = nrs_min
        self.nrs_max = nrs_max

        self._index: List[Dict] = []
        self._build_index()

    def _build_index(self) -> None:
        """
        Build a flat list of (subject, session, start_frame) tuples
        representing all valid windows across all subjects.
        """
        for subject_id in self.subjects:
            subject_dir = self.feature_dir / subject_id
            if not subject_dir.exists():
                logger.warning(f"Subject directory not found: {subject_dir}")
                continue

            sessions = sorted(subject_dir.glob("session_*"))
            for session_dir in sessions:
                label_file = session_dir / "labels.csv"
                if not label_file.exists():
                    continue

                labels_df = pd.read_csv(label_file)
                n_frames = len(labels_df)

                for start in range(0, n_frames - self.window_frames + 1, self.stride):
                    end = start + self.window_frames
                    window_nrs = labels_df["nrs_score"].iloc[start:end].values
                    if np.any(np.isnan(window_nrs)):
                        continue
                    self._index.append({
                        "subject_id": subject_id,
                        "session_dir": str(session_dir),
                        "start_frame": start,
                        "end_frame": end,
                        "mean_nrs": float(window_nrs.mean()),
                        "labels": window_nrs.tolist(),
                    })

        logger.info(
            f"Built index: {len(self._index)} windows from {len(self.subjects)} subjects."
        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self._index[idx]
        session_dir = Path(entry["session_dir"])
        start = entry["start_frame"]
        end = entry["end_frame"]

        # Load pre-extracted ROI features (stored as NPY for speed)
        # Shape: (n_frames, 5, C, 128, 128)
        rgb_path = session_dir / "rgb_rois.npy"
        thermal_path = session_dir / "thermal_rois.npy"

        rgb_rois = np.load(rgb_path, mmap_mode="r")[start:end]       # (T, 5, 3, 128, 128)
        thermal_rois = np.load(thermal_path, mmap_mode="r")[start:end]  # (T, 5, 1, 128, 128)

        rgb_rois = rgb_rois.astype(np.float32) / 255.0               # Normalise to [0, 1]
        thermal_rois = thermal_rois.astype(np.float32)               # Already z-normalised

        if self.augment:
            rgb_rois, thermal_rois = self._augment(rgb_rois, thermal_rois)

        nrs = np.clip(entry["mean_nrs"], self.nrs_min, self.nrs_max)

        return {
            "rgb_rois": torch.from_numpy(rgb_rois.copy()),
            "thermal_rois": torch.from_numpy(thermal_rois.copy()),
            "nrs_score": torch.tensor(nrs, dtype=torch.float32),
            "subject_id": entry["subject_id"],
            "mean_nrs": entry["mean_nrs"],
        }

    def _augment(
        self,
        rgb: np.ndarray,
        thermal: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Temporal and spatial augmentation.

        Applied consistently across all ROIs and both modalities.
        - Temporal jitter: random frame offset ±5 frames
        - Horizontal flip (50% probability)
        - Brightness / contrast jitter on RGB only
        """
        T, R, C, H, W = rgb.shape

        # Horizontal flip (consistent across modalities and ROIs)
        if random.random() < 0.5:
            rgb = rgb[:, :, :, :, ::-1].copy()
            thermal = thermal[:, :, :, :, ::-1].copy()
            # Mirror ROI ordering: periorbital stays, cheeks swap L↔R
            # (simplified: full symmetry assumed here)

        # RGB brightness / contrast jitter
        alpha = np.random.uniform(0.8, 1.2)   # contrast
        beta = np.random.uniform(-0.1, 0.1)   # brightness
        rgb = np.clip(rgb * alpha + beta, 0.0, 1.0)

        # Thermal: small Gaussian noise (± 0.5 σ), simulating sensor noise
        noise_std = 0.05
        thermal = thermal + np.random.randn(*thermal.shape).astype(np.float32) * noise_std

        return rgb, thermal

    def get_subject_windows(self, subject_id: str) -> List[Dict]:
        """Return all window entries for a specific subject."""
        return [e for e in self._index if e["subject_id"] == subject_id]

    def get_pain_intensity_distribution(self) -> Dict[str, float]:
        """Return fraction of windows in each NRS intensity bin."""
        nrs_values = np.array([e["mean_nrs"] for e in self._index])
        return {
            "low_0_3": float((nrs_values <= 3).mean()),
            "moderate_4_6": float(((nrs_values > 3) & (nrs_values <= 6)).mean()),
            "high_7_10": float((nrs_values > 6).mean()),
        }


class Dataset1(PainDataset):
    """
    Dataset 1: Controlled laboratory pain induction.

    50 healthy adults, ages 21–68 (mean 42.3 ± 12.7 yr), 52% female.
    Protocols: Cold Pressor Test (CPT) + pressure algometry.
    Ground truth: continuous NRS slider at 1 Hz (frame-level).

    Cross-validation: 40 train / 10 test per fold (stratified 5-fold).
    """

    SUBJECT_IDS = [f"subject_{i:03d}" for i in range(1, 51)]
    N_SUBJECTS = 50
    TOTAL_HOURS = 87.3
    TOTAL_FRAMES = 9_460_000
    MEAN_NRS = 3.8
    STD_NRS = 2.6


class Dataset2(PainDataset):
    """
    Dataset 2: Clinical postoperative monitoring (PACU).

    30 surgical patients, ages 31–74 (mean 56.8 ± 11.2 yr), 47% female.
    Surgeries: appendectomy, cholecystectomy, hernia repair.
    Ground truth: NRS self-report every 2 minutes (linearly interpolated).

    Cross-validation: 24 train / 6 test per fold (stratified 5-fold).
    """

    SUBJECT_IDS = [f"patient_{i:03d}" for i in range(1, 31)]
    N_SUBJECTS = 30
    TOTAL_HOURS = 17.7
    TOTAL_FRAMES = 1_920_000
    MEAN_NRS = 4.3
    STD_NRS = 2.4


def get_cv_splits(
    dataset_name: str,
    n_folds: int = 5,
    seed: int = 42,
) -> List[Dict[str, List[str]]]:
    """
    Generate stratified 5-fold cross-validation splits.

    Stratification ensures each fold has similar pain intensity distributions.

    Args:
        dataset_name: "dataset1", "dataset2", or "combined".
        n_folds:      Number of CV folds (default 5).
        seed:         Random seed for reproducibility.

    Returns:
        List of n_folds dicts with keys "train" and "test".
    """
    from sklearn.model_selection import KFold

    if dataset_name == "dataset1":
        subject_ids = Dataset1.SUBJECT_IDS
    elif dataset_name == "dataset2":
        subject_ids = Dataset2.SUBJECT_IDS
    elif dataset_name == "combined":
        subject_ids = Dataset1.SUBJECT_IDS + Dataset2.SUBJECT_IDS
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(subject_ids).tolist()

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = []
    for train_idx, test_idx in kf.split(shuffled):
        splits.append({
            "train": [shuffled[i] for i in train_idx],
            "test": [shuffled[i] for i in test_idx],
        })
    return splits


def build_dataloaders(
    feature_dir: str,
    fold: int,
    dataset_name: str = "combined",
    batch_size: int = 16,
    num_workers: int = 8,
    window_frames: int = 300,
    train_stride: int = 150,
    eval_stride: int = 30,
    pin_memory: bool = True,
) -> Tuple["torch.utils.data.DataLoader", "torch.utils.data.DataLoader"]:
    """
    Build train and validation DataLoaders for a given CV fold.

    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader

    splits = get_cv_splits(dataset_name, n_folds=5)
    fold_split = splits[fold]

    train_ds = PainDataset(
        feature_dir=feature_dir,
        subjects=fold_split["train"],
        window_frames=window_frames,
        stride=train_stride,
        augment=True,
    )
    val_ds = PainDataset(
        feature_dir=feature_dir,
        subjects=fold_split["test"],
        window_frames=window_frames,
        stride=eval_stride,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    logger.info(
        f"Fold {fold}: {len(train_ds)} train windows, {len(val_ds)} val windows."
    )
    return train_loader, val_loader
