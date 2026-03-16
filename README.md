# Cross-Spectral Fusion of Thermal and RGB Imaging for Objective Pain Estimation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Official implementation of the paper:

> **Cross-Spectral Fusion of Thermal and RGB Imaging for Objective Pain Estimation**  
> Oussama El Othmani, Sami Naouali  
> *PLOS Digital Health*, 2026

---

## Overview

This repository contains the full implementation of the **Cross-Spectral Attention Fusion (CSAF)** model, which integrates synchronized thermal infrared and RGB facial video for continuous, objective pain intensity estimation on a 0–10 Numeric Rating Scale (NRS).

**Key Results:**
- **MAE = 0.87** on combined dataset (n=80 subjects, 105 h video)
- **29.3% improvement** over best RGB-only baseline (RGB-Transformer, MAE=1.23)
- **38.5% improvement** at high pain intensities (NRS 7–10)
- **37.6% improvement** for low-expressor patients
- Thermal signals precede visible expressions by **1.2 ± 0.3 seconds**

---

## Architecture

```
Synchronized Thermal + RGB Video
        │
        ▼
┌─────────────────────────────────┐
│     Preprocessing Pipeline      │
│  Face Detection → ROI Extraction│
│  (5 regions × 128×128 pixels)   │
└─────────────┬───────────────────┘
              │
     ┌────────┴────────┐
     ▼                 ▼
┌─────────┐       ┌─────────┐
│ResNet-50│       │ResNet-50│
│  (RGB)  │       │(Thermal)│
└────┬────┘       └────┬────┘
     └────────┬─────────┘
              ▼
   ┌───────────────────────┐
   │  Cross-Spectral        │
   │  Attention Fusion      │
   │  (CSAF) — Bidirectional│
   │  + Adaptive Gating     │
   └──────────┬────────────┘
              ▼
   ┌───────────────────────┐
   │  Temporal Transformer  │
   │  6 layers, 8 heads     │
   │  10-second windows     │
   └──────────┬────────────┘
              ▼
      Pain Score (0–10 NRS)
```

---

## Repository Structure

```
Cross-Spectral-Fusion-of-Thermal/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── csaf.py              # Cross-Spectral Attention Fusion module
│   │   ├── encoders.py          # ResNet-50 modal encoders
│   │   ├── temporal_transformer.py  # Temporal Transformer
│   │   └── pain_estimator.py    # Full pipeline model
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py           # Dataset classes (Dataset1 & Dataset2)
│   │   ├── preprocessing.py     # Face detection, ROI extraction, registration
│   │   └── augmentation.py      # Data augmentation strategies
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py           # Three-stage training logic
│   │   └── losses.py            # MAE + smooth + ordinal loss
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py           # MAE, PCC, ICC, accuracy
│   │   └── visualisation.py     # Attention maps, thermal patterns
│   └── utils/
│       ├── __init__.py
│       ├── sync.py              # Camera synchronisation utilities
│       └── logging_utils.py     # Experiment logging
├── configs/
│   ├── default.yaml             # Default hyperparameters
│   ├── dataset1.yaml            # Controlled pain dataset config
│   └── dataset2.yaml            # Clinical postoperative config
├── scripts/
│   ├── train.py                 # Main training script
│   ├── evaluate.py              # Evaluation / inference script
│   ├── extract_features.py      # Offline feature extraction
│   └── visualise_attention.py   # Attention map visualisation
├── tests/
│   ├── test_models.py
│   ├── test_data.py
│   └── test_metrics.py
├── docs/
│   └── data_format.md           # Data format specification
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

---

## Installation

### Requirements
- Python ≥ 3.9
- PyTorch ≥ 2.0
- CUDA ≥ 11.7 (for GPU training)

```bash
git clone https://github.com/oussama123-ai/Cross-Spectral-Fusion-of-Thermal.git
cd Cross-Spectral-Fusion-of-Thermal
pip install -r requirements.txt
pip install -e .
```

---

## Data Preparation

### Dataset Format

Organize your data as follows:

```
data/
├── dataset1/          # Controlled lab (Cold Pressor Test + pressure algometry)
│   ├── subject_001/
│   │   ├── rgb/       # RGB frames (PNG, 1920×1080)
│   │   ├── thermal/   # Thermal frames (NPY float32, 640×480)
│   │   └── labels.csv # timestamp, nrs_score columns
│   └── ...
└── dataset2/          # Clinical postoperative (PACU)
    ├── patient_001/
    │   ├── rgb/
    │   ├── thermal/
    │   └── labels.csv
    └── ...
```

See [`docs/data_format.md`](docs/data_format.md) for detailed specifications.

### Preprocessing

```bash
python scripts/extract_features.py \
    --data_root /path/to/data \
    --output_dir /path/to/features \
    --dataset dataset1
```

---

## Training

### Three-Stage Training (recommended)

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --data_root /path/to/features \
    --output_dir experiments/csaf_run1 \
    --gpus 4
```

### Single-stage (end-to-end only)

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --training_strategy end_to_end \
    --data_root /path/to/features \
    --output_dir experiments/csaf_e2e
```

### Configuration

Key hyperparameters in `configs/default.yaml`:

| Parameter | Value |
|-----------|-------|
| Batch size | 16 |
| Learning rate | 1e-4 |
| Optimizer | AdamW |
| LR schedule | Cosine annealing |
| Gradient clipping | max_norm=1.0 |
| Stage 1 epochs | 20 |
| Stage 2 epochs | 30 |
| Stage 3 epochs | 50 |
| Temporal window | 300 frames (10 s) |
| Transformer layers | 6 |
| Attention heads | 8 |
| Model dimension | 512 |

---

## Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint experiments/csaf_run1/best_model.pth \
    --data_root /path/to/features \
    --dataset combined \
    --output_dir results/
```

Outputs: MAE, RMSE, PCC, ICC, 3-class accuracy, per-fold results, subgroup analysis.

---

## Visualisation

### Attention Maps

```bash
python scripts/visualise_attention.py \
    --checkpoint experiments/csaf_run1/best_model.pth \
    --video_rgb /path/to/rgb_video.mp4 \
    --video_thermal /path/to/thermal.npy \
    --output_dir results/attention_maps/
```

### Temporal Pain Dynamics

```python
from src.evaluation.visualisation import plot_temporal_dynamics

plot_temporal_dynamics(
    predictions=model_output,
    ground_truth=nrs_labels,
    rgb_weights=lambda_rgb,
    thermal_weights=lambda_thermal,
    save_path="results/temporal_dynamics.png"
)
```

---

## Pre-trained Models

Model weights are available upon reasonable request from the corresponding author, subject to institutional approval and execution of a data use agreement.

Contact: **salnawali@kfu.edu.sa**

---

## Data Availability

- **Anonymized processed features & metadata**: Available as Supporting Information (S2–S3 Tables) with the published article.
- **Raw data sample**: Publicly deposited at [Zenodo](https://zenodo.org/records/18991937) (DOI: 10.5281/zenodo.18991937).
- **Full raw video**: Restricted due to IRB requirements (IRB-MRC-MHT-2022-001). Requests: ethics.mrc@mht.tn

---

## Citation

If you use this code, please cite:

```bibtex
@article{elothmani2026crossspectral,
  title   = {Cross-Spectral Fusion of Thermal and {RGB} Imaging for Objective Pain Estimation},
  author  = {El Othmani, Oussama and Naouali, Sami},
  journal = {PLOS Digital Health},
  year    = {2026},
  doi     = {10.1371/journal.pdig.XXXXXXX}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

We thank the volunteers and patients who participated in this study, the nursing staff for clinical data collection support, and the anonymous reviewers for their constructive feedback.
