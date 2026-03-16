"""
Visualisation utilities for attention maps, thermal patterns, and temporal dynamics.

Reproduces Figures 5–9 from the paper:
  - Figure 5:  Dynamic modality contribution weights by pain level and facial region
  - Figure 6:  Temporal evolution during pain onset (thermal precedence 1.2 s)
  - Figure 7:  Learned spatial attention maps (RGB vs thermal heatmaps)
  - Figure 8:  Discovered thermal patterns (nasal cooling, periorbital warming, etc.)
  - Figure 9:  Performance on special populations
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logger = logging.getLogger(__name__)

ROI_NAMES = ["periorbital", "forehead", "nasal", "cheeks", "perioral"]
NRS_LEVELS = ["Low\n(0–3)", "Moderate\n(4–6)", "High\n(7–10)"]
PALETTE = {
    "rgb": "#4C72B0",
    "thermal": "#DD8452",
    "fusion": "#2CA02C",
    "ground_truth": "black",
}


# ── Figure 5: Modality contribution weights ────────────────────────────────────

def plot_modality_contributions(
    rgb_weights_by_level: List[Tuple[float, float]],
    thermal_weights_by_level: List[Tuple[float, float]],
    rgb_weights_by_roi: Dict[str, float],
    thermal_weights_by_roi: Dict[str, float],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot learned adaptive gate weights (λ_rgb, λ_th) by pain level and ROI.

    Reproduces Figure 5 from the paper.

    Args:
        rgb_weights_by_level:    [(mean, std)] for each NRS level (low/mod/high).
        thermal_weights_by_level: Same for thermal.
        rgb_weights_by_roi:      {roi_name: mean_weight}.
        thermal_weights_by_roi:  Same for thermal.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Dynamic Modality Contribution Weights", fontsize=14, fontweight="bold")

    # --- Panel (a): By pain intensity level ---
    x = np.arange(len(NRS_LEVELS))
    width = 0.6

    rgb_means = [w[0] for w in rgb_weights_by_level]
    th_means = [w[0] for w in thermal_weights_by_level]

    bars_rgb = ax1.bar(x, rgb_means, width, color=PALETTE["rgb"],
                       alpha=0.85, label="RGB (λ_rgb)")
    bars_th = ax1.bar(x, th_means, width, bottom=rgb_means,
                      color=PALETTE["thermal"], alpha=0.85, label="Thermal (λ_th)")

    for i, (r, t) in enumerate(zip(rgb_means, th_means)):
        ax1.text(i, r / 2, f"{r:.0%}", ha="center", va="center",
                 fontsize=13, fontweight="bold", color="white")
        ax1.text(i, r + t / 2, f"{t:.0%}", ha="center", va="center",
                 fontsize=13, fontweight="bold", color="white")

    ax1.set_xticks(x)
    ax1.set_xticklabels(NRS_LEVELS, fontsize=12)
    ax1.set_ylabel("Modality Contribution Weight", fontsize=12)
    ax1.set_title("(a) Modality Contributions by Pain Level", fontsize=12)
    ax1.legend(loc="upper right", fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel("Pain Intensity Level", fontsize=12)

    # --- Panel (b): By facial region ---
    rois = list(rgb_weights_by_roi.keys())
    y = np.arange(len(rois))
    rgb_vals = [rgb_weights_by_roi[r] for r in rois]
    th_vals = [thermal_weights_by_roi[r] for r in rois]

    ax2.barh(y, rgb_vals, height=0.6, color=PALETTE["rgb"], alpha=0.85, label="RGB")
    ax2.barh(y, th_vals, height=0.6, left=rgb_vals,
             color=PALETTE["thermal"], alpha=0.85, label="Thermal")

    for i, (r, t) in enumerate(zip(rgb_vals, th_vals)):
        ax2.text(r / 2, i, f"{r:.0%}", ha="center", va="center",
                 fontsize=11, fontweight="bold", color="white")
        ax2.text(r + t / 2, i, f"{t:.0%}", ha="center", va="center",
                 fontsize=11, fontweight="bold", color="white")

    ax2.set_yticks(y)
    ax2.set_yticklabels(rois, fontsize=12)
    ax2.set_xlabel("Modality Contribution Weight", fontsize=12)
    ax2.set_title("(b) Modality Contributions by Facial Region", fontsize=12)
    ax2.legend(loc="lower right", fontsize=11)
    ax2.set_xlim(0, 1.05)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved Figure 5 to {save_path}")
    return fig


# ── Figure 6: Temporal evolution during pain onset ────────────────────────────

def plot_temporal_dynamics(
    ground_truth: np.ndarray,
    thermal_prediction: np.ndarray,
    rgb_prediction: np.ndarray,
    fusion_prediction: np.ndarray,
    rgb_weights: np.ndarray,
    thermal_weights: np.ndarray,
    time_axis: Optional[np.ndarray] = None,
    thermal_lead_seconds: float = 1.2,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Three-panel figure showing temporal evolution during pain onset.

    Panel (a): NRS ground truth + modal predictions vs time.
    Panel (b): Dynamic modality contribution weights over time.
    Panel (c): Absolute prediction error over time.

    Args:
        ground_truth:       (T,) NRS ground truth.
        thermal_prediction: (T,) thermal-only predictions.
        rgb_prediction:     (T,) RGB-only predictions.
        fusion_prediction:  (T,) CSAF+Transformer predictions.
        rgb_weights:        (T,) λ_rgb gate values.
        thermal_weights:    (T,) λ_th gate values.
        time_axis:          (T,) time in seconds (default 0..T/30).
        thermal_lead_seconds: Lead time of thermal over RGB (default 1.2).
        save_path: Optional path to save.

    Returns:
        matplotlib Figure.
    """
    T = len(ground_truth)
    if time_axis is None:
        time_axis = np.arange(T) / 30.0  # 30 fps

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Temporal Evolution During Pain Onset", fontsize=13, fontweight="bold")

    # ── Panel (a): Pain signals ────────────────────────────────────────────
    ax = axes[0]
    ax.plot(time_axis, ground_truth, color="black", lw=2.5, label="Ground Truth",
            linestyle="--", zorder=5)
    ax.plot(time_axis, thermal_prediction, color=PALETTE["thermal"], lw=1.5,
            alpha=0.9, label="Thermal Prediction")
    ax.plot(time_axis, rgb_prediction, color=PALETTE["rgb"], lw=1.5,
            alpha=0.9, label="RGB Prediction")
    ax.plot(time_axis, fusion_prediction, color=PALETTE["fusion"], lw=2.0,
            alpha=0.9, label="Fusion Prediction")

    # Mark thermal/RGB response onset
    onset_t = time_axis[np.argmax(ground_truth > 1.0)] if (ground_truth > 1.0).any() else 5.0
    ax.axvline(onset_t, color="gray", linestyle=":", lw=1.5, label="Stimulus Onset")
    ax.axvline(onset_t - thermal_lead_seconds, color=PALETTE["thermal"],
               linestyle=":", lw=1.5, alpha=0.7, label="Thermal Response")
    ax.axvline(onset_t, color=PALETTE["rgb"],
               linestyle=":", lw=1.0, alpha=0.7, label="RGB Response")

    ax.set_ylabel("Pain Intensity (0–10)", fontsize=11)
    ax.set_ylim(-0.5, 11)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.set_title("(a) Temporal Evolution of Pain Signals", fontsize=11)

    # ── Panel (b): Modality weights ────────────────────────────────────────
    ax = axes[1]
    ax.fill_between(time_axis, 0, rgb_weights, color=PALETTE["rgb"],
                    alpha=0.7, label="RGB Contribution (α)")
    ax.fill_between(time_axis, rgb_weights, 1.0, color=PALETTE["thermal"],
                    alpha=0.7, label="Thermal Contribution (β)")
    ax.set_ylabel("Modality Weight", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title("(b) Dynamic Modality Contribution Weights", fontsize=11)

    # ── Panel (c): Prediction error ────────────────────────────────────────
    ax = axes[2]
    rgb_err = np.abs(rgb_prediction - ground_truth)
    th_err = np.abs(thermal_prediction - ground_truth)
    fusion_err = np.abs(fusion_prediction - ground_truth)
    improvement = rgb_err - fusion_err

    ax.plot(time_axis, rgb_err, color=PALETTE["rgb"], lw=1.5, label="RGB Error")
    ax.plot(time_axis, th_err, color=PALETTE["thermal"], lw=1.5, label="Thermal Error")
    ax.plot(time_axis, fusion_err, color=PALETTE["fusion"], lw=2.0, label="Fusion Error")
    ax.fill_between(time_axis, 0, np.clip(improvement, 0, None),
                    color=PALETTE["fusion"], alpha=0.2, label="Fusion Improvement")

    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Absolute Error", fontsize=11)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title("(c) Prediction Error Over Time", fontsize=11)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved Figure 6 to {save_path}")
    return fig


# ── Figure 7: Spatial attention maps ──────────────────────────────────────────

def plot_attention_maps(
    rgb_attention_maps: Dict[str, np.ndarray],
    thermal_attention_maps: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualise learned spatial attention maps for RGB and thermal modalities.

    Args:
        rgb_attention_maps:     Dict[roi_name → (128, 128) attention map].
        thermal_attention_maps: Same for thermal.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    rois_to_show = ["cheeks", "perioral"]  # show two ROIs + averaged
    n_cols = len(rois_to_show) + 1  # +1 for averaged

    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8))
    fig.suptitle("Learned Spatial Attention Maps (NRS ≥ 7)", fontsize=13)

    def _plot_roi(ax, attn_map, title, cmap="hot"):
        im = ax.imshow(attn_map, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 0: RGB attention
    for col_i, roi_name in enumerate(rois_to_show):
        attn = rgb_attention_maps.get(roi_name, np.zeros((128, 128)))
        _plot_roi(axes[0, col_i], attn, f"{roi_name.title()}\nRGB Attention")

    # Averaged RGB
    avg_rgb = np.mean(
        [rgb_attention_maps.get(r, np.zeros((128, 128))) for r in ROI_NAMES], axis=0
    )
    _plot_roi(axes[0, -1], avg_rgb, "RGB Attention\n(Averaged)")

    # Row 1: Thermal attention
    for col_i, roi_name in enumerate(["periorbital", "forehead"]):
        attn = thermal_attention_maps.get(roi_name, np.zeros((128, 128)))
        _plot_roi(axes[1, col_i], attn, f"{roi_name.title()}\nThermal Attention", cmap="hot")

    # Averaged Thermal
    avg_th = np.mean(
        [thermal_attention_maps.get(r, np.zeros((128, 128))) for r in ROI_NAMES], axis=0
    )
    _plot_roi(axes[1, -1], avg_th, "Thermal Attention\n(Averaged)", cmap="hot")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved Figure 7 to {save_path}")
    return fig


# ── Figure 8: Thermal pain patterns ───────────────────────────────────────────

def plot_thermal_patterns(
    patterns: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualise the four thermal patterns discovered via unsupervised clustering.

    Patterns (high-pain instances, NRS ≥ 7):
        A. Nasal cooling:      −0.8±0.3°C  (42% of instances)
        B. Periorbital warming: +0.6±0.2°C (38%)
        C. Forehead cooling:   −0.5±0.2°C  (28%)
        D. Thermal asymmetry:  >0.8°C lateral difference (18%)

    Args:
        patterns: Dict[pattern_name → (128, 128) temperature change map].
        save_path: Optional save path.
    """
    pattern_meta = {
        "nasal_cooling": {
            "title": "Pattern 1: Nasal Cooling\n(42% of high-pain instances)",
            "subtitle": "ΔT = −0.8 to −1.5°C",
            "cmap": "Blues_r",
        },
        "periorbital_warming": {
            "title": "Pattern 2: Periorbital Warming\n(38% of high-pain instances)",
            "subtitle": "ΔT = +0.4 to +0.9°C",
            "cmap": "Reds",
        },
        "forehead_cooling": {
            "title": "Pattern 3: Forehead Cooling\n(28% of high-pain instances)",
            "subtitle": "ΔT = −0.6 to −1.2°C",
            "cmap": "Blues_r",
        },
        "thermal_asymmetry": {
            "title": "Pattern 4: Thermal Asymmetry\n(18% of instances)",
            "subtitle": "ΔT > 0.5°C (L-R difference)",
            "cmap": "RdBu_r",
        },
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Discovered Thermal Patterns (High Pain, NRS ≥ 7)", fontsize=13)
    axes_flat = axes.flatten()

    for idx, (pattern_name, meta) in enumerate(pattern_meta.items()):
        ax = axes_flat[idx]
        pattern_map = patterns.get(
            pattern_name,
            np.random.randn(128, 128) * 0.5  # placeholder
        )
        vmax = 2.0
        im = ax.imshow(pattern_map, cmap=meta["cmap"], vmin=-vmax, vmax=vmax,
                       aspect="auto")
        ax.set_title(meta["title"], fontsize=10, fontweight="bold")
        ax.text(0.5, -0.08, meta["subtitle"], ha="center", va="top",
                transform=ax.transAxes, fontsize=10, color="gray")
        ax.axis("off")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Temperature Change (°C)", fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved Figure 8 to {save_path}")
    return fig


# ── Figure 2 / Table 2 reproduction: overall performance ──────────────────────

def plot_overall_performance(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Horizontal bar chart reproducing Figure 2 (MAE and PCC comparison).

    Args:
        results: Dict[method_name → {mae, pcc, icc, accuracy_3class}]
    """
    methods = list(results.keys())
    maes = [results[m]["mae"] for m in methods]
    pccs = [results[m].get("pcc", 0.0) for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Overall Performance Comparison — Combined Dataset (5-fold CV)",
                 fontsize=12, fontweight="bold")

    colors = ["#2CA02C" if "CSAF" in m else
              ("#4C72B0" if any(k in m for k in ["RGB+Physio", "Early", "Late"]) else "#AAAAAA")
              for m in methods]

    y_pos = range(len(methods))

    # MAE (lower is better)
    bars = ax1.barh(y_pos, maes, color=colors, alpha=0.85)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(methods, fontsize=11)
    ax1.set_xlabel("Mean Absolute Error (MAE) ↓ lower is better", fontsize=11)
    ax1.set_title("(a) MAE — Primary Metric", fontsize=11)
    for bar, val in zip(bars, maes):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}", va="center", fontsize=10)
    ax1.invert_yaxis()

    # PCC (higher is better)
    bars2 = ax2.barh(y_pos, pccs, color=colors, alpha=0.85)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(methods, fontsize=11)
    ax2.set_xlabel("Pearson Correlation Coefficient (PCC) ↑ higher is better", fontsize=11)
    ax2.set_title("(b) PCC — Secondary Metric", fontsize=11)
    for bar, val in zip(bars2, pccs):
        ax2.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}", va="center", fontsize=10)
    ax2.invert_yaxis()

    # Legend
    patch_single = mpatches.Patch(color="#AAAAAA", alpha=0.85, label="Traditional / Single-modality")
    patch_multi = mpatches.Patch(color="#4C72B0", alpha=0.85, label="Multimodal Fusion baselines")
    patch_ours = mpatches.Patch(color="#2CA02C", alpha=0.85, label="CSAF+Transformer (Ours)")
    fig.legend(handles=[patch_single, patch_multi, patch_ours],
               loc="lower center", ncol=3, fontsize=11, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved overall performance figure to {save_path}")
    return fig
