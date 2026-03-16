"""
Unit tests for data pipeline and preprocessing.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import (
    ROIExtractor, ThermalNormaliser, ROI_NAMES, ROI_BBOXES, ROI_SIZE,
    CANONICAL_LANDMARKS,
)
from src.data.dataset import get_cv_splits, PainDataset


class TestROIExtractor:

    @pytest.fixture
    def extractor(self):
        return ROIExtractor()

    def test_extract_rois_returns_five(self, extractor):
        canonical_rgb = np.zeros((512, 512, 3), dtype=np.uint8)
        canonical_thermal = np.zeros((512, 512), dtype=np.float32)
        rois = extractor.extract_rois(canonical_rgb, canonical_thermal)
        assert len(rois) == 5
        for roi_name in ROI_NAMES:
            assert roi_name in rois

    def test_roi_shapes(self, extractor):
        canonical_rgb = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        canonical_thermal = np.random.randn(512, 512).astype(np.float32)
        rois = extractor.extract_rois(canonical_rgb, canonical_thermal)
        for roi_name, roi_data in rois.items():
            assert roi_data["rgb"].shape == (ROI_SIZE, ROI_SIZE, 3), roi_name
            assert roi_data["thermal"].shape == (ROI_SIZE, ROI_SIZE), roi_name

    def test_roi_bbox_within_canvas(self):
        for roi_name, (x1, y1, x2, y2) in ROI_BBOXES.items():
            assert x1 >= 0 and y1 >= 0
            assert x2 <= 512 and y2 <= 512
            assert x2 > x1 and y2 > y1, f"Invalid bbox for {roi_name}"


class TestThermalNormaliser:

    def test_fit_and_transform(self):
        normaliser = ThermalNormaliser(baseline_duration_s=3.0, fps=10.0)
        # 30 baseline frames + 10 more
        n_baseline = 30
        frames = [np.random.randn(ROI_SIZE, ROI_SIZE).astype(np.float32) + 35.0
                  for _ in range(n_baseline + 10)]
        baseline = {"nasal": frames[:n_baseline]}
        normaliser.fit(baseline)

        test_frame = np.ones((ROI_SIZE, ROI_SIZE), dtype=np.float32) * 35.0
        norm = normaliser.transform(test_frame, "nasal")
        assert norm.shape == (ROI_SIZE, ROI_SIZE)
        # Mean of baseline is ~35°C; frame at 35°C should be ~0 after normalisation
        assert np.abs(norm.mean()) < 2.0

    def test_transform_without_fit_raises(self):
        normaliser = ThermalNormaliser()
        frame = np.ones((ROI_SIZE, ROI_SIZE), dtype=np.float32)
        with pytest.raises(RuntimeError):
            normaliser.transform(frame, "nasal")


class TestCVSplits:

    def test_five_folds(self):
        splits = get_cv_splits("dataset1", n_folds=5)
        assert len(splits) == 5

    def test_no_overlap_between_train_test(self):
        splits = get_cv_splits("dataset1", n_folds=5)
        for fold in splits:
            train_set = set(fold["train"])
            test_set = set(fold["test"])
            assert len(train_set & test_set) == 0

    def test_all_subjects_appear_in_test(self):
        splits = get_cv_splits("dataset1", n_folds=5)
        all_test = set()
        for fold in splits:
            all_test.update(fold["test"])
        from src.data.dataset import Dataset1
        assert all_test == set(Dataset1.SUBJECT_IDS)

    def test_combined_dataset(self):
        splits = get_cv_splits("combined", n_folds=5)
        assert len(splits) == 5
        # Combined should have more subjects per split
        from src.data.dataset import Dataset1, Dataset2
        total = len(Dataset1.SUBJECT_IDS) + len(Dataset2.SUBJECT_IDS)
        subjects_in_fold = len(splits[0]["train"]) + len(splits[0]["test"])
        assert subjects_in_fold == total

    def test_seed_reproducibility(self):
        splits1 = get_cv_splits("dataset1", seed=42)
        splits2 = get_cv_splits("dataset1", seed=42)
        for f1, f2 in zip(splits1, splits2):
            assert f1["train"] == f2["train"]
            assert f1["test"] == f2["test"]

    def test_different_seeds_differ(self):
        splits1 = get_cv_splits("dataset1", seed=42)
        splits2 = get_cv_splits("dataset1", seed=99)
        any_diff = any(f1["train"] != f2["train"]
                       for f1, f2 in zip(splits1, splits2))
        assert any_diff


class TestPainDatasetDummy:
    """
    Tests using a synthetically created feature directory.
    Avoids requiring real data for CI/CD.
    """

    @pytest.fixture
    def synthetic_feature_dir(self, tmp_path):
        """Create a minimal synthetic feature directory."""
        subject_dir = tmp_path / "subject_001" / "session_001"
        subject_dir.mkdir(parents=True)

        n_frames = 600   # 20 seconds at 30 fps
        n_rois = 5

        # rgb_rois: (N, 5, 3, 128, 128) uint8
        rgb = np.random.randint(0, 255, (n_frames, n_rois, 3, 128, 128),
                                dtype=np.uint8)
        np.save(subject_dir / "rgb_rois.npy", rgb)

        # thermal_rois: (N, 5, 1, 128, 128) float32
        thermal = np.random.randn(n_frames, n_rois, 1, 128, 128).astype(np.float32)
        np.save(subject_dir / "thermal_rois.npy", thermal)

        # labels
        nrs_values = np.random.uniform(0, 10, n_frames)
        df = pd.DataFrame({"timestamp_ms": np.arange(n_frames) * 33.3,
                           "nrs_score": nrs_values})
        df.to_csv(subject_dir / "labels.csv", index=False)

        return str(tmp_path)

    def test_dataset_builds_index(self, synthetic_feature_dir):
        ds = PainDataset(
            feature_dir=synthetic_feature_dir,
            subjects=["subject_001"],
            window_frames=300,
            stride=150,
        )
        assert len(ds) > 0

    def test_getitem_shapes(self, synthetic_feature_dir):
        ds = PainDataset(
            feature_dir=synthetic_feature_dir,
            subjects=["subject_001"],
            window_frames=300,
            stride=300,
        )
        sample = ds[0]
        assert sample["rgb_rois"].shape == (300, 5, 3, 128, 128)
        assert sample["thermal_rois"].shape == (300, 5, 1, 128, 128)
        assert sample["nrs_score"].shape == ()

    def test_nrs_in_valid_range(self, synthetic_feature_dir):
        ds = PainDataset(
            feature_dir=synthetic_feature_dir,
            subjects=["subject_001"],
            window_frames=300,
            stride=300,
        )
        for i in range(len(ds)):
            sample = ds[i]
            nrs = sample["nrs_score"].item()
            assert 0.0 <= nrs <= 10.0
