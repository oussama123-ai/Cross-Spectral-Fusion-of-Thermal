"""
Preprocessing Pipeline — Spatial Registration and ROI Extraction.

Implements Algorithm 1 from the paper:
  1. Apply homography to align thermal → RGB coordinate frame
  2. Detect face and landmarks with RetinaFace (99.1% detection rate)
  3. Compute similarity transform to canonical face alignment
  4. Extract 5 facial ROIs (periorbital, forehead, nasal, cheeks, perioral)
  5. Normalise thermal ROIs by per-subject, per-ROI baseline statistics

Camera specifications:
  - RGB:     Canon EOS 90D, 1920×1080, 30 fps
  - Thermal: FLIR A655sc, 640×480, 7.5–14 µm, < 0.04°C sensitivity
  - Hardware-triggered sync: NI USB-6001 DAQ, TTL 30 Hz, offset < 10 ms
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── ROI bounding boxes in canonical 512×512 face space ────────────────────────
# Table in S1 Algorithm of paper (xmin, ymin, xmax, ymax)
ROI_BBOXES: Dict[str, Tuple[int, int, int, int]] = {
    "periorbital": (80, 140, 432, 260),   # eyes, AU6/7
    "forehead":    (100, 60, 412, 155),   # brow, AU4
    "nasal":       (170, 240, 342, 370),  # nose, perinasal (key for vasoconstriction)
    "cheeks":      (50, 250, 462, 380),   # bilateral zygomatic
    "perioral":    (130, 350, 382, 460),  # mouth, AU9/10
}
ROI_NAMES = list(ROI_BBOXES.keys())
ROI_SIZE = 128   # each ROI resized to 128×128

# Canonical 5-point landmarks for similarity transform
# (left eye, right eye, nose tip, left mouth corner, right mouth corner)
CANONICAL_LANDMARKS = np.array([
    [192.0, 239.0],  # left eye
    [320.0, 239.0],  # right eye
    [256.0, 285.0],  # nose tip
    [175.0, 360.0],  # left mouth corner
    [337.0, 360.0],  # right mouth corner
], dtype=np.float32)


@dataclass
class PreprocessingConfig:
    canonical_size: int = 512
    roi_size: int = 128
    thermal_sensitivity_threshold: float = 0.04   # °C (FLIR A655sc spec)
    min_thermal_change: float = 0.3               # minimum pain-related ΔT
    baseline_duration_seconds: float = 180.0      # 3-minute baseline for normalization
    environment_temp_range: Tuple[float, float] = (21.0, 23.0)  # controlled room °C


class FaceDetector:
    """
    Wrapper around RetinaFace for face detection and 5-point landmark localisation.

    Uses insightface for RetinaFace inference.
    Falls back to OpenCV Haar cascade if insightface is unavailable.
    """

    def __init__(self, det_thresh: float = 0.5, use_gpu: bool = True) -> None:
        self.det_thresh = det_thresh
        self._model = None
        self._use_insightface = False
        self._setup(use_gpu)

    def _setup(self, use_gpu: bool) -> None:
        try:
            import insightface
            from insightface.app import FaceAnalysis
            ctx_id = 0 if use_gpu else -1
            app = FaceAnalysis(
                name="buffalo_l",
                allowed_modules=["detection"],
                providers=["CUDAExecutionProvider"] if use_gpu else ["CPUExecutionProvider"],
            )
            app.prepare(ctx_id=ctx_id, det_thresh=self.det_thresh, det_size=(640, 640))
            self._model = app
            self._use_insightface = True
            logger.info("RetinaFace (insightface) loaded successfully.")
        except Exception as e:
            logger.warning(f"insightface not available ({e}). Using OpenCV fallback.")
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._model = cv2.CascadeClassifier(cascade_path)

    def detect(
        self, frame_bgr: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Detect the most confident face in frame.

        Args:
            frame_bgr: HxWx3 uint8 BGR image.

        Returns:
            bbox: (4,) [x1, y1, x2, y2] in pixel coords.
            landmarks: (5, 2) float32 landmarks (eyes, nose, mouth corners).
            None if no face detected.
        """
        if self._use_insightface:
            faces = self._model.get(frame_bgr)
            if not faces:
                return None
            # Take highest-confidence face
            face = max(faces, key=lambda f: f.det_score)
            bbox = face.bbox.astype(np.float32)
            landmarks = face.kps.astype(np.float32)
            return bbox, landmarks
        else:
            # OpenCV fallback — returns bbox only, landmarks estimated
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self._model.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
            if len(faces) == 0:
                return None
            x, y, w, h = faces[0]
            bbox = np.array([x, y, x + w, y + h], dtype=np.float32)
            # Approximate landmarks from bbox geometry
            cx, cy = x + w / 2, y + h / 2
            landmarks = np.array([
                [cx - w * 0.18, cy - h * 0.1],
                [cx + w * 0.18, cy - h * 0.1],
                [cx, cy + h * 0.05],
                [cx - w * 0.15, cy + h * 0.22],
                [cx + w * 0.15, cy + h * 0.22],
            ], dtype=np.float32)
            return bbox, landmarks


class SpatialRegistration:
    """
    Aligns thermal images to RGB coordinate frame using pre-computed homography H.

    H is computed offline from a calibration checkerboard (10×10, 25 mm squares)
    visible in both spectra. Cameras are separated by 15 cm horizontally.

    Paper reports mean registration error: 1.8 pixels.
    """

    def __init__(self, homography: Optional[np.ndarray] = None) -> None:
        self._H = homography   # 3×3 homography matrix

    def load_homography(self, path: str) -> None:
        """Load pre-computed homography from .npy file."""
        self._H = np.load(path)
        logger.info(f"Loaded homography from {path}: shape {self._H.shape}")

    def align_thermal(
        self,
        thermal: np.ndarray,
        rgb_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Warp thermal image into RGB coordinate frame.

        Args:
            thermal:   HxW float32 thermal frame (°C or normalised).
            rgb_shape: (H, W) of the RGB frame.

        Returns:
            Aligned thermal image of shape rgb_shape.
        """
        if self._H is None:
            raise RuntimeError("Homography not loaded. Call load_homography() first.")
        h, w = rgb_shape
        aligned = cv2.warpPerspective(thermal, self._H, (w, h),
                                      flags=cv2.INTER_LINEAR)
        return aligned

    @staticmethod
    def compute_homography_from_checkerboard(
        rgb_corners: np.ndarray,
        thermal_corners: np.ndarray,
    ) -> np.ndarray:
        """
        Compute homography from corresponding checkerboard corner pairs.

        Args:
            rgb_corners:     (N, 2) corners in RGB image.
            thermal_corners: (N, 2) corresponding corners in thermal image.

        Returns:
            H: (3, 3) homography matrix.
        """
        H, mask = cv2.findHomography(thermal_corners, rgb_corners,
                                     method=cv2.RANSAC, ransacReprojThreshold=3.0)
        inlier_ratio = mask.sum() / len(mask)
        logger.info(f"Homography inlier ratio: {inlier_ratio:.3f}")
        return H


class ROIExtractor:
    """
    Extracts the 5 facial ROIs from a canonically-aligned face image.

    Implements the face alignment and ROI extraction in Algorithm 1 (paper).
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None) -> None:
        self.config = config or PreprocessingConfig()

    def compute_similarity_transform(
        self, src_landmarks: np.ndarray
    ) -> np.ndarray:
        """
        Compute similarity transform from detected landmarks to canonical positions.

        Args:
            src_landmarks: (5, 2) detected landmark coordinates.

        Returns:
            M: (2, 3) affine transformation matrix.
        """
        M, _ = cv2.estimateAffinePartial2D(
            src_landmarks,
            CANONICAL_LANDMARKS,
            method=cv2.LMEDS,
        )
        return M

    def warp_to_canonical(
        self,
        image: np.ndarray,
        transform: np.ndarray,
        size: int = 512,
    ) -> np.ndarray:
        """
        Warp image to canonical 512×512 face space.

        Args:
            image:     Input image (any channels).
            transform: (2, 3) affine transform from compute_similarity_transform.
            size:      Output canvas size (default 512).

        Returns:
            Warped image of shape (size, size, C) or (size, size).
        """
        return cv2.warpAffine(image, transform, (size, size),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=0)

    def extract_rois(
        self,
        canonical_rgb: np.ndarray,
        canonical_thermal: np.ndarray,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract 5 facial ROIs from canonically-aligned images.

        Args:
            canonical_rgb:     (512, 512, 3) uint8.
            canonical_thermal: (512, 512) float32 normalised thermal.

        Returns:
            Dict[roi_name → {"rgb": (128,128,3), "thermal": (128,128)}]
        """
        rois = {}
        for roi_name, (x1, y1, x2, y2) in ROI_BBOXES.items():
            rgb_crop = canonical_rgb[y1:y2, x1:x2]
            th_crop = canonical_thermal[y1:y2, x1:x2]

            roi_rgb = cv2.resize(rgb_crop, (ROI_SIZE, ROI_SIZE),
                                 interpolation=cv2.INTER_LINEAR)
            roi_th = cv2.resize(th_crop, (ROI_SIZE, ROI_SIZE),
                                interpolation=cv2.INTER_LINEAR)
            rois[roi_name] = {"rgb": roi_rgb, "thermal": roi_th}

        return rois


class ThermalNormaliser:
    """
    Normalises thermal ROIs per subject per ROI using a baseline window.

    Normalisation: (T - μ_baseline) / σ_baseline
    Baseline: first 3 minutes of recording (pre-pain period).

    This removes inter-subject thermoregulatory differences and focuses
    on pain-related deviations from each individual's baseline.
    """

    def __init__(self, baseline_duration_s: float = 180.0, fps: float = 30.0) -> None:
        self.baseline_frames = int(baseline_duration_s * fps)
        self._means: Dict[str, float] = {}
        self._stds: Dict[str, float] = {}

    def fit(self, baseline_frames: Dict[str, List[np.ndarray]]) -> None:
        """
        Compute per-ROI baseline statistics.

        Args:
            baseline_frames: Dict[roi_name → list of thermal ROI arrays].
        """
        for roi_name, frames in baseline_frames.items():
            arr = np.stack(frames, axis=0)  # (T, H, W)
            self._means[roi_name] = float(arr.mean())
            std = float(arr.std())
            self._stds[roi_name] = max(std, 1e-6)  # avoid division by zero

    def transform(self, roi_thermal: np.ndarray, roi_name: str) -> np.ndarray:
        """
        Normalise a single thermal ROI.

        Args:
            roi_thermal: (H, W) float32 thermal values.
            roi_name:    One of the 5 ROI names.

        Returns:
            Normalised thermal ROI (H, W) float32.
        """
        if roi_name not in self._means:
            raise RuntimeError(
                f"Normaliser not fitted for ROI '{roi_name}'. Call fit() first."
            )
        return (roi_thermal - self._means[roi_name]) / self._stds[roi_name]

    def fit_transform(
        self,
        all_frames: Dict[str, List[np.ndarray]],
        roi_name: str,
    ) -> List[np.ndarray]:
        """Fit on first baseline_frames and transform all."""
        baseline = {roi_name: all_frames[roi_name][: self.baseline_frames]}
        self.fit(baseline)
        return [self.transform(f, roi_name) for f in all_frames[roi_name]]


class Preprocessor:
    """
    Full preprocessing pipeline combining all components.

    Usage:
        preprocessor = Preprocessor(homography_path="calibration/H.npy")
        preprocessor.face_detector = FaceDetector()
        result = preprocessor.process_frame(rgb_frame, thermal_frame)
    """

    def __init__(
        self,
        homography_path: Optional[str] = None,
        config: Optional[PreprocessingConfig] = None,
        use_gpu: bool = True,
    ) -> None:
        self.config = config or PreprocessingConfig()
        self.face_detector = FaceDetector(use_gpu=use_gpu)
        self.registration = SpatialRegistration()
        self.roi_extractor = ROIExtractor(self.config)
        self.normaliser = ThermalNormaliser()
        self._frames_processed = 0
        self._frames_dropped = 0

        if homography_path and Path(homography_path).exists():
            self.registration.load_homography(homography_path)

    def process_frame(
        self,
        rgb: np.ndarray,
        thermal: np.ndarray,
    ) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
        """
        Full pipeline for a single synchronised frame pair.

        Args:
            rgb:     (H, W, 3) uint8 BGR.
            thermal: (H_th, W_th) float32 temperature in °C.

        Returns:
            Dict of 5 ROIs with 'rgb' and 'thermal' keys, or None if face not detected.
        """
        self._frames_processed += 1
        h, w = rgb.shape[:2]

        # Step 1: Align thermal to RGB
        thermal_aligned = self.registration.align_thermal(thermal, (h, w))

        # Step 2: Face detection
        detection = self.face_detector.detect(rgb)
        if detection is None:
            self._frames_dropped += 1
            return None
        _, landmarks = detection

        # Step 3: Compute similarity transform
        transform = self.roi_extractor.compute_similarity_transform(landmarks)

        # Step 4: Warp both images to canonical space
        rgb_canonical = self.roi_extractor.warp_to_canonical(rgb, transform)
        thermal_canonical = self.roi_extractor.warp_to_canonical(
            thermal_aligned, transform
        )

        # Step 5: Extract ROIs
        rois = self.roi_extractor.extract_rois(rgb_canonical, thermal_canonical)

        return rois

    @property
    def detection_rate(self) -> float:
        if self._frames_processed == 0:
            return 0.0
        return 1.0 - self._frames_dropped / self._frames_processed
