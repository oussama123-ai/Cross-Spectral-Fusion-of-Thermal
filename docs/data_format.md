# Data Format Specification

## Overview

This document describes the expected data format for the CSAF pain estimation pipeline.

---

## Raw Data Layout (Input to `scripts/extract_features.py`)

```
data/raw/
├── dataset1/                    # Controlled lab (CPT + pressure algometry)
│   ├── subject_001/
│   │   ├── session_001/         # CPT session
│   │   │   ├── rgb/             # RGB frames as PNG (1920×1080)
│   │   │   │   ├── frame_000001.png
│   │   │   │   ├── frame_000002.png
│   │   │   │   └── ...
│   │   │   ├── thermal.npy      # Thermal stack: (N, 480, 640) float32 in °C
│   │   │   ├── labels.csv       # Frame-level NRS labels (see below)
│   │   │   └── sync_log.csv     # DAQ timestamp log (optional)
│   │   └── session_002/         # Pressure algometry session
│   │       └── ...
│   └── subject_002/
│       └── ...
└── dataset2/                    # Clinical PACU
    ├── patient_001/
    │   └── session_001/
    │       ├── rgb_video.mp4    # (alternative to rgb/ directory)
    │       ├── thermal.npy
    │       ├── labels.csv
    │       └── sync_log.csv
    └── ...
```

### labels.csv

| Column | Type | Description |
|--------|------|-------------|
| `timestamp_ms` | float | Frame timestamp in milliseconds |
| `nrs_score` | float | NRS pain score 0–10 |

For Dataset 1: NRS sampled at 1 Hz from continuous slider (frame-level interpolated).  
For Dataset 2: NRS self-report every 2 minutes (linearly interpolated between reports).

### sync_log.csv (optional, from NI USB-6001 DAQ)

| Column | Type | Description |
|--------|------|-------------|
| `timestamp_rgb_ms` | float | RGB camera frame timestamp (ms) |
| `timestamp_thermal_ms` | float | Thermal camera frame timestamp (ms) |

---

## Preprocessed Feature Layout (Output of `scripts/extract_features.py`)

```
data/features/
├── subject_001/
│   └── session_001/
│       ├── rgb_rois.npy         # (N, 5, 3, 128, 128) uint8
│       ├── thermal_rois.npy     # (N, 5, 1, 128, 128) float32 (z-normalised)
│       ├── labels.csv           # aligned labels for valid frames only
│       └── sync_stats.json      # synchronisation statistics
└── ...
```

### rgb_rois.npy

Shape: `(N_frames, 5, 3, 128, 128)` — uint8  
- Axis 0: frame index  
- Axis 1: ROI index (see ROI order below)  
- Axis 2: RGB channels  
- Axes 3–4: 128×128 spatial  

### thermal_rois.npy

Shape: `(N_frames, 5, 1, 128, 128)` — float32  
- z-normalised by 3-minute per-subject, per-ROI baseline statistics  
- `(T - μ_baseline) / σ_baseline`  

### ROI Order

| Index | Name | Anatomical Region | Relevance |
|-------|------|-------------------|-----------|
| 0 | periorbital | Eyes, brow | AU4, AU6/7 (expression) |
| 1 | forehead | Brow ridge | AU4, stress cooling |
| 2 | nasal | Nose, perinasal | Sympathetic vasoconstriction |
| 3 | cheeks | Bilateral zygomatic | Vascular tone |
| 4 | perioral | Mouth area | AU9/10 (expression) |

Bounding boxes in canonical 512×512 face space:

| ROI | xmin | ymin | xmax | ymax |
|-----|------|------|------|------|
| periorbital | 80 | 140 | 432 | 260 |
| forehead | 100 | 60 | 412 | 155 |
| nasal | 170 | 240 | 342 | 370 |
| cheeks | 50 | 250 | 462 | 380 |
| perioral | 130 | 350 | 382 | 460 |

### sync_stats.json

```json
{
  "n_frames": 9464,
  "face_detection_rate": 0.991,
  "sync": {
    "n_pairs": 9464,
    "median_offset_ms": 3.2,
    "max_offset_ms": 9.7
  }
}
```

---

## Camera Specifications

| Parameter | RGB (Canon EOS 90D) | Thermal (FLIR A655sc) |
|-----------|--------------------|-----------------------|
| Resolution | 1920×1080 | 640×480 |
| Frame rate | 30 fps | 30 fps |
| Spectral range | 400–700 nm | 7.5–14 µm |
| Thermal sensitivity | — | < 0.04°C |
| Sync method | Hardware TTL | Hardware TTL |

---

## NRS Label Conventions

- Scale: 0 (no pain) to 10 (worst imaginable pain)
- Intensity bins: Low = 0–3, Moderate = 4–6, High = 7–10
- Minimal Clinically Important Difference (MCID): 1.5 NRS points
- Our MAE = 0.87 is well below MCID on the combined dataset
