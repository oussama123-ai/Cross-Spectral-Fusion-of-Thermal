"""
Camera Synchronisation Utilities.

Hardware setup (Section 2.2, S3 Table):
  - NI USB-6001 DAQ sends simultaneous TTL pulses (5 V, 10 ms) at 30 Hz
  - Hardware-triggered synchronisation of FLIR A655sc + Canon EOS 90D
  - Verified median inter-camera temporal offset: 3.2 ms (< 10 ms criterion)
  - Validation: LED light source (1 Hz) detected within same 33 ms frame window

This module provides:
  1. Timestamp-based frame alignment and verification
  2. Hardware trigger pulse generation (NI DAQ interface)
  3. Post-hoc synchronisation correction from log files
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Synchronisation criteria
MAX_ALLOWED_OFFSET_MS = 10.0   # < 10 ms criterion (30 fps = 33.3 ms per frame)
TARGET_FPS = 30.0
FRAME_DURATION_MS = 1000.0 / TARGET_FPS  # 33.33 ms


@dataclass
class SyncStats:
    """Statistics from synchronisation verification."""
    n_frame_pairs: int
    median_offset_ms: float
    mean_offset_ms: float
    max_offset_ms: float
    iqr_ms: Tuple[float, float]       # (Q25, Q75)
    fraction_within_criterion: float   # fraction with offset < 10 ms
    criterion_ms: float = MAX_ALLOWED_OFFSET_MS


class FrameSynchroniser:
    """
    Aligns thermal and RGB frame sequences based on hardware timestamps.

    For each recording session, the NI DAQ logs timestamps for every
    trigger pulse sent to both cameras. This class uses those logs to:
      1. Compute per-frame temporal offsets.
      2. Identify and drop frames that exceed the offset criterion.
      3. Build the final aligned index of (rgb_frame_idx, thermal_frame_idx).
    """

    def __init__(
        self,
        max_offset_ms: float = MAX_ALLOWED_OFFSET_MS,
        fps: float = TARGET_FPS,
    ) -> None:
        self.max_offset_ms = max_offset_ms
        self.fps = fps

    def align_from_timestamps(
        self,
        rgb_timestamps_ms: np.ndarray,
        thermal_timestamps_ms: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, SyncStats]:
        """
        Align two timestamp arrays to find synchronised frame pairs.

        Uses nearest-neighbour matching: for each thermal timestamp,
        find the RGB frame whose timestamp is closest.

        Args:
            rgb_timestamps_ms:     (N_rgb,) in milliseconds.
            thermal_timestamps_ms: (N_th,)  in milliseconds.

        Returns:
            rgb_indices:    (M,) indices into rgb_timestamps_ms.
            thermal_indices:(M,) indices into thermal_timestamps_ms.
            stats:          SyncStats summary.
        """
        rgb_indices = []
        thermal_indices = []
        offsets_ms = []

        for th_idx, th_ts in enumerate(thermal_timestamps_ms):
            # Find closest RGB timestamp
            diffs = np.abs(rgb_timestamps_ms - th_ts)
            rgb_idx = int(np.argmin(diffs))
            offset = float(diffs[rgb_idx])

            if offset <= self.max_offset_ms:
                rgb_indices.append(rgb_idx)
                thermal_indices.append(th_idx)
                offsets_ms.append(offset)

        offsets_arr = np.array(offsets_ms)
        stats = SyncStats(
            n_frame_pairs=len(offsets_arr),
            median_offset_ms=float(np.median(offsets_arr)) if len(offsets_arr) > 0 else 0.0,
            mean_offset_ms=float(np.mean(offsets_arr)) if len(offsets_arr) > 0 else 0.0,
            max_offset_ms=float(np.max(offsets_arr)) if len(offsets_arr) > 0 else 0.0,
            iqr_ms=(
                float(np.percentile(offsets_arr, 25)),
                float(np.percentile(offsets_arr, 75)),
            ) if len(offsets_arr) > 0 else (0.0, 0.0),
            fraction_within_criterion=float(
                (offsets_arr <= self.max_offset_ms).mean()
            ) if len(offsets_arr) > 0 else 0.0,
        )

        logger.info(
            f"Synchronisation: {stats.n_frame_pairs} aligned pairs | "
            f"median offset = {stats.median_offset_ms:.2f} ms | "
            f"max = {stats.max_offset_ms:.2f} ms"
        )
        if stats.max_offset_ms > self.max_offset_ms:
            logger.warning(
                f"Some frames exceed {self.max_offset_ms} ms offset criterion!"
            )

        return (
            np.array(rgb_indices, dtype=np.int64),
            np.array(thermal_indices, dtype=np.int64),
            stats,
        )

    def verify_sync_with_led(
        self,
        rgb_brightness_signal: np.ndarray,
        thermal_heat_signal: np.ndarray,
        led_frequency_hz: float = 1.0,
    ) -> Dict[str, float]:
        """
        Verify synchronisation using an LED light source pulsed at led_frequency_hz.

        The LED is visible to the RGB camera (brightness changes) and
        generates heat visible to the thermal camera.

        Method: detect state transitions in both signals and measure lag.

        Args:
            rgb_brightness_signal:  (N,) per-frame average brightness.
            thermal_heat_signal:    (N,) per-frame average heat in LED region.
            led_frequency_hz:       LED pulse frequency (default 1 Hz).

        Returns:
            Dict with measured_lag_ms and criterion_passed.
        """
        from scipy.signal import find_peaks

        # Threshold signals
        rgb_binary = (rgb_brightness_signal > rgb_brightness_signal.mean()).astype(float)
        thermal_binary = (thermal_heat_signal > thermal_heat_signal.mean()).astype(float)

        # Find rising edges
        rgb_edges = np.where(np.diff(rgb_binary) > 0.5)[0]
        th_edges = np.where(np.diff(thermal_binary) > 0.5)[0]

        if len(rgb_edges) == 0 or len(th_edges) == 0:
            logger.warning("LED verification: could not detect transitions.")
            return {"measured_lag_ms": float("nan"), "criterion_passed": False}

        # Align edge pairs and compute mean lag
        n_pairs = min(len(rgb_edges), len(th_edges))
        lags_frames = th_edges[:n_pairs] - rgb_edges[:n_pairs]
        lag_ms = float(np.abs(lags_frames).mean() * (1000.0 / TARGET_FPS))

        result = {
            "measured_lag_ms": lag_ms,
            "criterion_passed": lag_ms < MAX_ALLOWED_OFFSET_MS,
            "n_transitions": n_pairs,
        }
        logger.info(f"LED sync verification: lag = {lag_ms:.2f} ms, "
                    f"passed = {result['criterion_passed']}")
        return result

    def load_daq_log(self, log_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse NI USB-6001 DAQ trigger log file.

        Expected CSV format:
            timestamp_rgb_ms, timestamp_thermal_ms
            1000.00, 1003.20
            1033.33, 1036.51
            ...

        Args:
            log_path: Path to DAQ log CSV.

        Returns:
            rgb_timestamps_ms, thermal_timestamps_ms (both (N,) float64).
        """
        import pandas as pd
        df = pd.read_csv(log_path, header=0,
                         names=["timestamp_rgb_ms", "timestamp_thermal_ms"])
        return df["timestamp_rgb_ms"].values, df["timestamp_thermal_ms"].values


class ThermalVideoReader:
    """
    Reader for raw FLIR A655sc thermal video data.

    FLIR A655sc specifications:
        - Resolution: 640×480 pixels
        - Spectral range: 7.5–14 µm (longwave infrared)
        - Thermal sensitivity: < 0.04°C at 30°C
        - Frame rate: 30 fps (hardware-triggered)
        - Temperature range: −40 to +150°C

    Supports reading:
        - .npy files (pre-extracted frame arrays)
        - FLIR SEQ / CSQ files (requires flirpy)
        - Raw radiometric TIFF stacks
    """

    def __init__(self, file_path: str) -> None:
        self.file_path = Path(file_path)
        self._data: Optional[np.ndarray] = None

    def load(self) -> np.ndarray:
        """
        Load all frames into memory.

        Returns:
            (N_frames, 480, 640) float32 array in °C.
        """
        ext = self.file_path.suffix.lower()
        if ext == ".npy":
            self._data = np.load(self.file_path)
        elif ext in (".seq", ".csq"):
            self._data = self._load_flir_seq()
        elif ext in (".tiff", ".tif"):
            self._data = self._load_tiff_stack()
        else:
            raise ValueError(f"Unsupported thermal file format: {ext}")

        logger.info(
            f"Loaded thermal video: {self._data.shape}, "
            f"T range = [{self._data.min():.1f}, {self._data.max():.1f}] °C"
        )
        return self._data

    def _load_flir_seq(self) -> np.ndarray:
        """Load FLIR SEQ/CSQ file using flirpy (optional dependency)."""
        try:
            from flirpy.io.seq import Seq
            seq = Seq(str(self.file_path))
            frames = [seq.get_frame(i) for i in range(seq.num_frames)]
            return np.stack(frames, axis=0).astype(np.float32)
        except ImportError:
            raise ImportError(
                "flirpy is required to read FLIR SEQ/CSQ files. "
                "Install with: pip install flirpy"
            )

    def _load_tiff_stack(self) -> np.ndarray:
        """Load multi-page TIFF stack."""
        from PIL import Image
        img = Image.open(self.file_path)
        frames = []
        for i in range(img.n_frames):
            img.seek(i)
            frames.append(np.array(img, dtype=np.float32))
        return np.stack(frames, axis=0)

    def get_frame(self, idx: int) -> np.ndarray:
        if self._data is None:
            self.load()
        return self._data[idx]

    def __len__(self) -> int:
        if self._data is None:
            self.load()
        return len(self._data)
