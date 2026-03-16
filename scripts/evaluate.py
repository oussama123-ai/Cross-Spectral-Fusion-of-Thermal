#!/usr/bin/env python3
"""
Evaluation script for CSAF+Transformer pain estimator.

Computes all metrics from the paper (Table 2, Table 5):
  - Primary: MAE with 95% CI (bootstrap B=1000)
  - Secondary: RMSE, PCC, ICC, 3-class accuracy
  - Stratified by pain intensity (Table 3 rows, Figure 3)
  - Subgroup analysis: expressers, age, skin tone (Table 5)
  - Statistical significance: paired t-test, Bonferroni corrected

Usage:
    python scripts/evaluate.py \\
        --checkpoint experiments/fold0/best_model.pth \\
        --data_root data/features \\
        --dataset combined \\
        --output_dir results/fold0
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import build_dataloaders, get_cv_splits
from src.evaluation.metrics import (
    compute_all_metrics,
    compute_stratified_metrics,
    format_results_table,
    paired_ttest_with_correction,
)
from src.evaluation.visualisation import plot_overall_performance
from src.models.pain_estimator import PainEstimator
from src.utils.logging_utils import setup_logging

logger = logging.getLogger("csaf.evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CSAF+Transformer.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth).")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="combined",
                        choices=["dataset1", "dataset2", "combined"])
    parser.add_argument("--fold", type=int, default=0,
                        help="CV fold to evaluate (default 0).")
    parser.add_argument("--all_folds", action="store_true",
                        help="Evaluate all 5 folds and report mean±std.")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


@torch.no_grad()
def run_inference(
    model: PainEstimator,
    loader,
    device: torch.device,
    return_attention: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Run inference and collect predictions, targets, and optional attention info.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_subjects = []

    for batch in loader:
        rgb = batch["rgb_rois"].to(device, non_blocking=True)
        thermal = batch["thermal_rois"].to(device, non_blocking=True)
        nrs = batch["nrs_score"]

        output = model(rgb, thermal, return_attention=return_attention)
        all_preds.append(output["pain_score"].cpu().numpy())
        all_targets.append(nrs.numpy())
        all_subjects.extend(batch["subject_id"])

    return {
        "predictions": np.concatenate(all_preds),
        "targets": np.concatenate(all_targets),
        "subjects": all_subjects,
    }


def evaluate_fold(
    model: PainEstimator,
    args: argparse.Namespace,
    fold: int,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate a single fold and return metrics."""
    _, val_loader = build_dataloaders(
        feature_dir=args.data_root,
        fold=fold,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_stride=300,
        eval_stride=30,
        pin_memory=True,
    )

    results_raw = run_inference(model, val_loader, device)
    preds = results_raw["predictions"]
    targets = results_raw["targets"]

    metrics = compute_all_metrics(
        preds, targets,
        n_bootstrap=args.n_bootstrap,
        compute_ci=True,
    )
    metrics["n_windows"] = len(preds)

    return metrics, preds, targets


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(str(output_dir / "logs"), run_name="evaluate")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── Load model ────────────────────────────────────────────────────────
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = PainEstimator()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info(
        f"Model loaded (epoch {checkpoint.get('epoch', '?')}, "
        f"best val MAE = {checkpoint.get('best_val_mae', '?'):.4f})"
    )

    # ── Evaluate ──────────────────────────────────────────────────────────
    if args.all_folds:
        all_fold_metrics = []
        all_preds_pooled = []
        all_targets_pooled = []

        for fold in range(5):
            logger.info(f"Evaluating fold {fold} ...")
            metrics, preds, targets = evaluate_fold(model, args, fold, device)
            all_fold_metrics.append(metrics)
            all_preds_pooled.append(preds)
            all_targets_pooled.append(targets)
            logger.info(
                f"  Fold {fold}: MAE={metrics['mae']:.4f}, "
                f"PCC={metrics['pcc']:.4f}, ICC={metrics['icc']:.4f}, "
                f"Acc={metrics['accuracy_3class']*100:.1f}%"
            )

        # Report mean ± std across folds
        mae_vals = [m["mae"] for m in all_fold_metrics]
        pcc_vals = [m["pcc"] for m in all_fold_metrics]
        icc_vals = [m["icc"] for m in all_fold_metrics]
        acc_vals = [m["accuracy_3class"] for m in all_fold_metrics]

        logger.info("=" * 60)
        logger.info("5-Fold Cross-Validation Summary:")
        logger.info(f"  MAE:      {np.mean(mae_vals):.4f} ± {np.std(mae_vals):.4f}")
        logger.info(f"  PCC:      {np.mean(pcc_vals):.4f} ± {np.std(pcc_vals):.4f}")
        logger.info(f"  ICC:      {np.mean(icc_vals):.4f} ± {np.std(icc_vals):.4f}")
        logger.info(f"  Acc(3cl): {np.mean(acc_vals)*100:.2f}% ± {np.std(acc_vals)*100:.2f}%")
        logger.info("=" * 60)

        # Pooled stratified analysis
        all_preds_arr = np.concatenate(all_preds_pooled)
        all_targets_arr = np.concatenate(all_targets_pooled)
        stratified = compute_stratified_metrics(all_preds_arr, all_targets_arr)
        logger.info("Stratified by pain intensity (pooled):")
        for bin_name, m in stratified.items():
            logger.info(
                f"  {bin_name}: MAE={m['mae']:.4f}, n={m.get('n_samples',0)}"
            )

        # Save summary
        summary = {
            "mae_mean": float(np.mean(mae_vals)),
            "mae_std": float(np.std(mae_vals)),
            "pcc_mean": float(np.mean(pcc_vals)),
            "icc_mean": float(np.mean(icc_vals)),
            "acc_mean": float(np.mean(acc_vals)),
            "per_fold": all_fold_metrics,
            "stratified": {k: {kk: float(vv) for kk, vv in v.items()}
                           for k, v in stratified.items()},
        }
        with open(output_dir / "cv_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved CV summary to {output_dir / 'cv_summary.json'}")

    else:
        # Single fold
        metrics, preds, targets = evaluate_fold(model, args, args.fold, device)
        logger.info("=" * 60)
        logger.info(f"Fold {args.fold} Results:")
        logger.info(f"  MAE:      {metrics['mae']:.4f} "
                    f"(95% CI: [{metrics.get('mae_ci_lower', 0):.4f}, "
                    f"{metrics.get('mae_ci_upper', 0):.4f}])")
        logger.info(f"  PCC:      {metrics['pcc']:.4f}")
        logger.info(f"  ICC:      {metrics['icc']:.4f}")
        logger.info(f"  Acc(3cl): {metrics['accuracy_3class']*100:.2f}%")
        logger.info("=" * 60)

        # Stratified results
        stratified = compute_stratified_metrics(preds, targets)
        logger.info("Stratified by pain intensity:")
        for bin_name, m in stratified.items():
            logger.info(f"  {bin_name}: MAE={m['mae']:.4f}, n={m.get('n_samples',0)}")

        if args.save_predictions:
            pred_path = output_dir / f"fold{args.fold}_predictions.npy"
            np.save(pred_path, {"predictions": preds, "targets": targets})
            logger.info(f"Saved predictions to {pred_path}")

        with open(output_dir / f"fold{args.fold}_metrics.json", "w") as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)


if __name__ == "__main__":
    main()
