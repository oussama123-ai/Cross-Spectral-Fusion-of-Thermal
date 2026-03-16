#!/usr/bin/env python3
"""
Main training script for CSAF+Transformer pain estimation.

Supports:
  - Three-stage or end-to-end training
  - Multi-GPU via torch.distributed (DDP)
  - 5-fold cross-validation
  - Resume from checkpoint

Usage examples:

  # Single-GPU training, fold 0
  python scripts/train.py --config configs/default.yaml \\
      --data_root data/features --output_dir experiments/run1

  # 4-GPU DDP training
  torchrun --nproc_per_node=4 scripts/train.py \\
      --config configs/default.yaml --data_root data/features \\
      --output_dir experiments/run1 --gpus 4

  # Run all 5 folds
  for fold in 0 1 2 3 4; do
      python scripts/train.py --config configs/default.yaml \\
          --data_root data/features --fold $fold \\
          --output_dir experiments/fold${fold}
  done
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.distributed as dist
import yaml
from omegaconf import OmegaConf

from src.data.dataset import build_dataloaders
from src.models.pain_estimator import PainEstimator, PainEstimatorConfig
from src.training.trainer import Trainer
from src.utils.logging_utils import setup_logging, ExperimentLogger

logger = logging.getLogger("csaf.train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CSAF+Transformer pain estimator."
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file.")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of preprocessed features.")
    parser.add_argument("--output_dir", type=str, default="experiments/csaf_run",
                        help="Directory for checkpoints and logs.")
    parser.add_argument("--fold", type=int, default=0,
                        help="Cross-validation fold index (0–4).")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Override dataset name: dataset1|dataset2|combined.")
    parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from.")
    parser.add_argument("--training_strategy", type=str, default=None,
                        help="Override: three_stage | end_to_end.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def setup_distributed(local_rank: int) -> None:
    """Initialise NCCL process group for DDP."""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)


def main() -> None:
    args = parse_args()

    # ── Load config ───────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg = OmegaConf.create(cfg)

    # Apply CLI overrides
    if args.dataset:
        cfg.data.dataset = args.dataset
    if args.training_strategy:
        cfg.training.strategy = args.training_strategy

    # ── Distributed setup ─────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = local_rank == 0

    if world_size > 1:
        setup_distributed(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Logging ───────────────────────────────────────────────────────────
    if is_main:
        setup_logging(
            log_dir=os.path.join(args.output_dir, "logs"),
            run_name=f"fold{args.fold}",
        )
        exp_logger = ExperimentLogger(args.output_dir, run_name=f"fold{args.fold}")
        exp_logger.log_config(OmegaConf.to_container(cfg, resolve=True))

    # ── Reproducibility ───────────────────────────────────────────────────
    torch.manual_seed(args.seed + args.fold)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + args.fold)

    # ── Data ──────────────────────────────────────────────────────────────
    if is_main:
        logger.info(
            f"Building dataloaders for fold {args.fold}, "
            f"dataset={cfg.data.dataset}"
        )
    train_loader, val_loader = build_dataloaders(
        feature_dir=args.data_root,
        fold=args.fold,
        dataset_name=cfg.data.dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        window_frames=cfg.data.window_frames,
        train_stride=cfg.data.window_stride,
        eval_stride=cfg.data.window_stride_eval,
        pin_memory=cfg.data.pin_memory,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model_cfg = PainEstimatorConfig(
        rgb_pretrained=cfg.model.rgb_pretrained,
        thermal_pretrained=cfg.model.thermal_pretrained,
        csaf_d_model=cfg.model.csaf.d_key,
        csaf_dropout=cfg.model.csaf.dropout,
        csaf_bidirectional=cfg.model.csaf.bidirectional,
        csaf_adaptive_gating=cfg.model.csaf.adaptive_gating,
        transformer_d_model=cfg.model.temporal_transformer.d_model,
        transformer_n_layers=cfg.model.temporal_transformer.n_layers,
        transformer_n_heads=cfg.model.temporal_transformer.n_heads,
        transformer_d_ff=cfg.model.temporal_transformer.d_ff,
        transformer_max_seq_len=cfg.model.temporal_transformer.max_seq_len,
        transformer_dropout=cfg.model.temporal_transformer.dropout,
        head_hidden_dims=list(cfg.model.regression_head.hidden_dims),
        head_dropout=cfg.model.regression_head.dropout,
    )
    model = PainEstimator(model_cfg).to(device)

    if is_main:
        n_params = model.count_parameters()
        logger.info(f"Model: {n_params / 1e6:.1f}M trainable parameters")

    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

    # ── Trainer ───────────────────────────────────────────────────────────
    fold_output_dir = os.path.join(args.output_dir, f"fold{args.fold}")
    trainer = Trainer(
        model=model.module if world_size > 1 else model,
        output_dir=fold_output_dir,
        device=str(device),
        local_rank=local_rank,
        use_amp=cfg.training.amp,
        grad_clip_max_norm=cfg.training.grad_clip_max_norm,
        save_every_n_epochs=cfg.training.save_every_n_epochs,
        log_interval=cfg.logging.log_interval,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    # ── Train ─────────────────────────────────────────────────────────────
    optimizer_cfg = {
        "lr_stage1": cfg.training.stage1.lr,
        "lr_stage2": cfg.training.stage2.lr,
        "lr_stage3": cfg.training.stage3.lr,
        "weight_decay": cfg.training.weight_decay,
    }

    strategy = cfg.training.strategy
    if strategy == "three_stage":
        trainer.train_three_stages(
            train_loader=train_loader,
            val_loader=val_loader,
            stage1_epochs=cfg.training.stage1.epochs,
            stage2_epochs=cfg.training.stage2.epochs,
            stage3_epochs=cfg.training.stage3.epochs,
            optimizer_cfg=optimizer_cfg,
        )
    elif strategy == "end_to_end":
        import torch.optim as optim
        from src.training.trainer import build_optimizer, build_scheduler
        total_epochs = (
            cfg.training.stage1.epochs
            + cfg.training.stage2.epochs
            + cfg.training.stage3.epochs
        )
        optimizer = build_optimizer(model, lr=optimizer_cfg["lr_stage1"],
                                    weight_decay=optimizer_cfg["weight_decay"])
        scheduler = build_scheduler(
            optimizer, total_steps=total_epochs * len(train_loader)
        )
        trainer._run_stage(train_loader, val_loader, total_epochs,
                           optimizer, scheduler, stage_name="end_to_end")
    else:
        raise ValueError(f"Unknown training strategy: {strategy}")

    if is_main:
        logger.info(f"Training complete. Best val MAE = {trainer.best_val_mae:.4f}")
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
