"""
Loss functions for the CSAF pain estimation model.

Combined loss (Section 2.6 of paper):
    L = L_MAE + 0.1 * L_smooth + 0.05 * L_ordinal

Components:
    L_MAE:     Primary mean absolute error — direct NRS regression.
    L_smooth:  Temporal smoothness regularisation — penalises abrupt
               prediction changes within a window (pain evolves gradually).
    L_ordinal: Ordinal ranking loss — enforces monotonicity between
               windows at different pain intensities.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MAELoss(nn.Module):
    """Mean Absolute Error — primary training loss and evaluation metric."""

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred:   (B,) predicted NRS scores.
            target: (B,) ground-truth NRS scores.
            weight: Optional (B,) sample weights.

        Returns:
            Scalar loss.
        """
        loss = torch.abs(pred - target)
        if weight is not None:
            loss = loss * weight
        return loss.mean()


class SmoothnessLoss(nn.Module):
    """
    Temporal smoothness regularisation.

    Penalises large frame-to-frame prediction differences within a window.
    Pain intensity changes continuously; abrupt jumps likely indicate errors.

    L_smooth = mean(|ŷ_t - ŷ_{t-1}|) over consecutive frames in window.

    Used when model outputs per-frame predictions (not just window-level).
    For window-level output, applied across batch samples as proxy.
    """

    def forward(self, pred_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_sequence: (B, T) per-frame pain predictions over a window.

        Returns:
            Scalar smoothness loss.
        """
        if pred_sequence.dim() != 2 or pred_sequence.size(1) < 2:
            return torch.tensor(0.0, device=pred_sequence.device)
        diffs = torch.abs(pred_sequence[:, 1:] - pred_sequence[:, :-1])
        return diffs.mean()


class OrdinalLoss(nn.Module):
    """
    Ordinal ranking loss.

    For any pair of samples (i, j) where y_i > y_j + margin,
    penalises the model if it does not also predict ŷ_i > ŷ_j.

    This encourages correct pain intensity ordering, which is
    clinically important (distinguishing low vs high pain).

    L_ordinal = mean(max(0, margin + ŷ_j - ŷ_i)) over violating pairs.

    Args:
        margin: Minimum desired prediction difference (default 0.5 NRS points).
    """

    def __init__(self, margin: float = 0.5) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred:   (B,) predicted NRS scores.
            target: (B,) ground-truth NRS scores.

        Returns:
            Scalar ordinal loss.
        """
        B = pred.size(0)
        if B < 2:
            return torch.tensor(0.0, device=pred.device)

        # Pairwise differences: (B, B)
        pred_i = pred.unsqueeze(1).expand(B, B)      # row = i
        pred_j = pred.unsqueeze(0).expand(B, B)      # col = j
        target_i = target.unsqueeze(1).expand(B, B)
        target_j = target.unsqueeze(0).expand(B, B)

        # Pairs where target_i > target_j + margin (i is truly higher pain)
        valid_pairs = (target_i - target_j) > self.margin

        if not valid_pairs.any():
            return torch.tensor(0.0, device=pred.device)

        # Penalise if prediction does not respect ordering
        violation = F.relu(self.margin + pred_j - pred_i)
        loss = violation[valid_pairs].mean()
        return loss


class CombinedPainLoss(nn.Module):
    """
    Combined loss for CSAF+Transformer training.

    L = λ_mae * L_MAE + λ_smooth * L_smooth + λ_ordinal * L_ordinal

    Paper values: λ_mae=1.0, λ_smooth=0.1, λ_ordinal=0.05.

    Args:
        mae_weight:     Weight for MAE loss (default 1.0).
        smooth_weight:  Weight for smoothness loss (default 0.1).
        ordinal_weight: Weight for ordinal loss (default 0.05).
        ordinal_margin: Margin for ordinal loss (default 0.5).
    """

    def __init__(
        self,
        mae_weight: float = 1.0,
        smooth_weight: float = 0.1,
        ordinal_weight: float = 0.05,
        ordinal_margin: float = 0.5,
    ) -> None:
        super().__init__()
        self.mae_weight = mae_weight
        self.smooth_weight = smooth_weight
        self.ordinal_weight = ordinal_weight

        self.mae_loss = MAELoss()
        self.smooth_loss = SmoothnessLoss()
        self.ordinal_loss = OrdinalLoss(margin=ordinal_margin)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_sequence: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred:           (B,) window-level pain predictions.
            target:         (B,) window-level NRS ground truth.
            pred_sequence:  Optional (B, T) per-frame predictions for smoothness.
            sample_weight:  Optional (B,) sample importance weights.

        Returns:
            total_loss: Scalar combined loss.
            loss_dict:  Dict with individual loss components (for logging).
        """
        l_mae = self.mae_loss(pred, target, weight=sample_weight)
        l_ord = self.ordinal_loss(pred, target)

        if pred_sequence is not None:
            l_smooth = self.smooth_loss(pred_sequence)
        else:
            l_smooth = torch.tensor(0.0, device=pred.device)

        total = (
            self.mae_weight * l_mae
            + self.smooth_weight * l_smooth
            + self.ordinal_weight * l_ord
        )

        loss_dict = {
            "loss_total": total.item(),
            "loss_mae": l_mae.item(),
            "loss_smooth": l_smooth.item(),
            "loss_ordinal": l_ord.item(),
        }

        return total, loss_dict
