"""
Three-Stage Training Strategy for CSAF+Transformer.

Stage 1 (20 epochs): Independent modal encoder pre-training.
    - RGB encoder fine-tuned on NRS regression (ImageNet+VGGFace2 init).
    - Thermal encoder trained from scratch.
    - Each encoder optimised independently with MAE loss.

Stage 2 (30 epochs): CSAF fusion training with frozen encoders.
    - Encoders frozen; only CSAF module + temporal transformer trained.
    - Allows CSAF to learn optimal cross-modal attention patterns.

Stage 3 (50 epochs): End-to-end fine-tuning.
    - All parameters unfrozen and jointly optimised.
    - Lower learning rate to preserve Stage 1/2 representations.

Total: ~48 h on 4×NVIDIA A100 GPUs.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.pain_estimator import PainEstimator
from .losses import CombinedPainLoss

logger = logging.getLogger(__name__)


def build_optimizer(
    model: nn.Module,
    optimizer_name: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
) -> optim.Optimizer:
    """Build optimizer, separating weight_decay from bias/norm parameters."""
    no_decay_keys = {"bias", "LayerNorm.weight", "bn.weight", "bn.bias"}
    param_groups = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and not any(k in n for k in no_decay_keys)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and any(k in n for k in no_decay_keys)
            ],
            "weight_decay": 0.0,
        },
    ]
    if optimizer_name == "adamw":
        return optim.AdamW(param_groups, lr=lr)
    elif optimizer_name == "adam":
        return optim.Adam(param_groups, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def build_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str = "cosine",
    total_steps: int = 1000,
    warmup_steps: int = 100,
) -> optim.lr_scheduler.LRScheduler:
    """Build learning rate scheduler with optional warmup."""
    if scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=1e-7
        )
    elif scheduler_name == "linear":
        return optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0,
                                           total_iters=total_steps)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


class Trainer:
    """
    Handles the three-stage training loop for CSAF+Transformer.

    Supports:
    - Distributed Data Parallel (DDP) across 4 GPUs
    - Automatic mixed precision (AMP)
    - Gradient clipping (max_norm = 1.0)
    - TensorBoard logging
    - Checkpoint saving and resumption
    """

    def __init__(
        self,
        model: PainEstimator,
        output_dir: str,
        device: str = "cuda",
        local_rank: int = 0,
        use_amp: bool = True,
        grad_clip_max_norm: float = 1.0,
        save_every_n_epochs: int = 10,
        log_interval: int = 50,
    ) -> None:
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.local_rank = local_rank
        self.use_amp = use_amp
        self.grad_clip_max_norm = grad_clip_max_norm
        self.save_every_n_epochs = save_every_n_epochs
        self.log_interval = log_interval

        self.scaler = GradScaler(enabled=use_amp)
        self.loss_fn = CombinedPainLoss(
            mae_weight=1.0,
            smooth_weight=0.1,
            ordinal_weight=0.05,
        )

        self._writer = None
        if local_rank == 0:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
            except ImportError:
                logger.warning("TensorBoard not available.")

        self.global_step = 0
        self.best_val_mae = float("inf")

    # ── Three-Stage Training ──────────────────────────────────────────────────

    def train_three_stages(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        stage1_epochs: int = 20,
        stage2_epochs: int = 30,
        stage3_epochs: int = 50,
        optimizer_cfg: Optional[Dict] = None,
    ) -> None:
        """Run the full three-stage training protocol."""
        cfg = optimizer_cfg or {}

        # ── Stage 1: Modal encoder pre-training ───────────────────────────
        logger.info("=" * 60)
        logger.info("STAGE 1: Modal encoder pre-training")
        logger.info("=" * 60)
        optimizer = build_optimizer(self.model, lr=cfg.get("lr_stage1", 1e-4),
                                    weight_decay=cfg.get("weight_decay", 1e-4))
        scheduler = build_scheduler(
            optimizer, total_steps=stage1_epochs * len(train_loader),
            warmup_steps=cfg.get("warmup_steps", 100)
        )
        self._run_stage(train_loader, val_loader, stage1_epochs,
                        optimizer, scheduler, stage_name="stage1")

        # ── Stage 2: CSAF fusion (encoders frozen) ─────────────────────────
        logger.info("=" * 60)
        logger.info("STAGE 2: CSAF fusion training (encoders frozen)")
        logger.info("=" * 60)
        self.model.freeze_encoders()
        optimizer = build_optimizer(self.model, lr=cfg.get("lr_stage2", 1e-4),
                                    weight_decay=cfg.get("weight_decay", 1e-4))
        scheduler = build_scheduler(
            optimizer, total_steps=stage2_epochs * len(train_loader),
            warmup_steps=cfg.get("warmup_steps", 100)
        )
        self._run_stage(train_loader, val_loader, stage2_epochs,
                        optimizer, scheduler, stage_name="stage2")

        # ── Stage 3: End-to-end fine-tuning ────────────────────────────────
        logger.info("=" * 60)
        logger.info("STAGE 3: End-to-end fine-tuning")
        logger.info("=" * 60)
        self.model.unfreeze_encoders()
        optimizer = build_optimizer(self.model, lr=cfg.get("lr_stage3", 1e-5),
                                    weight_decay=cfg.get("weight_decay", 1e-4))
        scheduler = build_scheduler(
            optimizer, total_steps=stage3_epochs * len(train_loader),
            warmup_steps=cfg.get("warmup_steps", 50)
        )
        self._run_stage(train_loader, val_loader, stage3_epochs,
                        optimizer, scheduler, stage_name="stage3")

        logger.info("Three-stage training complete.")

    # ── Single-stage training loop ────────────────────────────────────────────

    def _run_stage(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int,
        optimizer: optim.Optimizer,
        scheduler: Any,
        stage_name: str = "stage",
    ) -> None:
        """Run training for n_epochs, evaluating after each epoch."""
        self.model.to(self.device)

        for epoch in range(1, n_epochs + 1):
            train_metrics = self._train_epoch(train_loader, optimizer, scheduler, epoch)
            val_metrics = self._val_epoch(val_loader, epoch)

            if self.local_rank == 0:
                logger.info(
                    f"[{stage_name}] Epoch {epoch}/{n_epochs} | "
                    f"Train MAE: {train_metrics['mae']:.4f} | "
                    f"Val MAE: {val_metrics['mae']:.4f}"
                )
                self._log_metrics(train_metrics, val_metrics, epoch, stage_name)

                if val_metrics["mae"] < self.best_val_mae:
                    self.best_val_mae = val_metrics["mae"]
                    self.save_checkpoint("best_model.pth", epoch, val_metrics)
                    logger.info(f"  ↳ New best val MAE: {self.best_val_mae:.4f}")

                if epoch % self.save_every_n_epochs == 0:
                    self.save_checkpoint(f"{stage_name}_epoch{epoch}.pth", epoch, val_metrics)

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: Any,
        epoch: int,
    ) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"Train Epoch {epoch}", disable=self.local_rank != 0)
        for batch in pbar:
            rgb = batch["rgb_rois"].to(self.device, non_blocking=True)
            thermal = batch["thermal_rois"].to(self.device, non_blocking=True)
            nrs = batch["nrs_score"].to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                output = self.model(rgb, thermal)
                pred = output["pain_score"]
                loss, loss_dict = self.loss_fn(pred, nrs)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip_max_norm
            )
            self.scaler.step(optimizer)
            self.scaler.update()
            scheduler.step()

            mae = torch.abs(pred.detach() - nrs).mean().item()
            total_loss += loss.item()
            total_mae += mae
            n_batches += 1
            self.global_step += 1

            if self.global_step % self.log_interval == 0 and self._writer:
                self._writer.add_scalar("train/loss", loss.item(), self.global_step)
                self._writer.add_scalar("train/mae", mae, self.global_step)
                for k, v in loss_dict.items():
                    self._writer.add_scalar(f"train/{k}", v, self.global_step)
                current_lr = optimizer.param_groups[0]["lr"]
                self._writer.add_scalar("train/lr", current_lr, self.global_step)

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "mae": f"{mae:.4f}"})

        return {"loss": total_loss / n_batches, "mae": total_mae / n_batches}

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Run one validation epoch."""
        self.model.eval()
        all_preds = []
        all_targets = []

        for batch in tqdm(loader, desc=f"Val Epoch {epoch}", disable=self.local_rank != 0):
            rgb = batch["rgb_rois"].to(self.device, non_blocking=True)
            thermal = batch["thermal_rois"].to(self.device, non_blocking=True)
            nrs = batch["nrs_score"].to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                output = self.model(rgb, thermal)
                pred = output["pain_score"]

            all_preds.append(pred.cpu())
            all_targets.append(nrs.cpu())

        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        mae = torch.abs(preds - targets).mean().item()

        if self._writer:
            self._writer.add_scalar("val/mae", mae, epoch)

        return {"mae": mae}

    # ── Checkpoint management ─────────────────────────────────────────────────

    def save_checkpoint(
        self,
        filename: str,
        epoch: int,
        metrics: Dict[str, float],
    ) -> None:
        """Save model checkpoint."""
        path = self.output_dir / filename
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "best_val_mae": self.best_val_mae,
            "metrics": metrics,
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str) -> int:
        """Load checkpoint, return epoch number."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_mae = checkpoint.get("best_val_mae", float("inf"))
        epoch = checkpoint.get("epoch", 0)
        logger.info(f"Loaded checkpoint from epoch {epoch}: {path}")
        return epoch

    def _log_metrics(
        self,
        train_metrics: Dict,
        val_metrics: Dict,
        epoch: int,
        stage: str,
    ) -> None:
        if self._writer:
            for k, v in val_metrics.items():
                self._writer.add_scalar(f"{stage}/val_{k}", v, epoch)
            for k, v in train_metrics.items():
                self._writer.add_scalar(f"{stage}/train_{k}", v, epoch)
