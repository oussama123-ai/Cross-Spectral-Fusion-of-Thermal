"""
Cross-Spectral Attention Fusion (CSAF) Module.

Implements bidirectional cross-attention between RGB and thermal features,
followed by adaptive gating to produce fused representations for each
of the 5 facial ROIs.

Mathematical formulation (per ROI r):

    F_rgb←th = Attention(Q=F_rgb, K=F_th, V=F_th)
    F_th←rgb = Attention(Q=F_th, K=F_rgb, V=F_rgb)

    where Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    λ_rgb, λ_th = σ(W_gate [F_rgb; F_th] + b_gate)

    F_fused = F_rgb + λ_rgb · F_rgb←th + λ_th · F_th←rgb

This residual formulation preserves the unimodal RGB features while
enriching them with thermally-attended context (and vice versa).

The adaptive gates λ learn to rely more on thermal signals when
facial expressions are suppressed (confirmed in Section 4.5 of the paper).
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """
    Single-direction scaled dot-product cross-attention.

    Query comes from modality A; Keys and Values come from modality B.
    The output enriches modality A with information from modality B.

    Args:
        d_model: Feature dimension (must match encoder output).
        dropout: Attention dropout probability.
    """

    def __init__(self, d_model: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.scale = math.sqrt(d_model)

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in [self.W_Q, self.W_K, self.W_V]:
            nn.init.xavier_uniform_(module.weight)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query:  (B, d_model) — features whose context is being enriched
            key:    (B, d_model) — features providing context
            value:  (B, d_model) — features providing context (same as key here)

        Returns:
            attended: (B, d_model) — enriched query features
            attn_weights: (B, d_model) — attention weight vector (for interpretability)

        Note:
            With 1D feature vectors (no spatial dimension), attention reduces to a
            learned feature-dimension weighting. This is the formulation used in the
            paper for ROI-level cross-modal attention.
        """
        Q = self.W_Q(query)   # (B, d_model)
        K = self.W_K(key)     # (B, d_model)
        V = self.W_V(value)   # (B, d_model)

        # Scaled dot-product: (B,) scalar similarity per sample
        # Expand to per-dimension attention over feature channels
        attn_scores = (Q * K) / self.scale        # (B, d_model) element-wise
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, d_model)
        attn_weights = self.dropout(attn_weights)

        attended = attn_weights * V               # (B, d_model)
        return attended, attn_weights


class AdaptiveGate(nn.Module):
    """
    Per-ROI learned modality weighting gate.

    Takes concatenated RGB and thermal features and outputs
    sigmoid-activated weights λ_rgb, λ_th ∈ [0, 1].

    Args:
        d_model: Feature dimension for each modality.
    """

    def __init__(self, d_model: int = 2048) -> None:
        super().__init__()
        # Input: [F_rgb; F_th] concatenated → 2 * d_model
        self.gate = nn.Linear(2 * d_model, 2, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(
        self, f_rgb: torch.Tensor, f_th: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            f_rgb: (B, d_model)
            f_th:  (B, d_model)

        Returns:
            lambda_rgb: (B, 1) — weight for thermal→RGB cross-attention
            lambda_th:  (B, 1) — weight for RGB→thermal cross-attention
        """
        concat = torch.cat([f_rgb, f_th], dim=-1)   # (B, 2*d_model)
        gates = torch.sigmoid(self.gate(concat))     # (B, 2)
        lambda_rgb = gates[:, 0:1]                   # (B, 1)
        lambda_th = gates[:, 1:2]                    # (B, 1)
        return lambda_rgb, lambda_th


class CrossSpectralAttentionFusion(nn.Module):
    """
    Full CSAF module operating over all 5 facial ROIs.

    For each ROI r, performs:
        1. Thermal-informed RGB: F_rgb←th via CrossAttention(Q=rgb, K=th, V=th)
        2. RGB-informed thermal: F_th←rgb via CrossAttention(Q=th, K=rgb, V=rgb)
        3. Adaptive gating: λ_rgb, λ_th learned from [F_rgb; F_th]
        4. Residual fusion: F_fused = F_rgb + λ_rgb·F_rgb←th + λ_th·F_th←rgb

    Args:
        d_model:     Feature dimension (must match encoder output, default 2048).
        num_rois:    Number of facial ROIs (default 5).
        dropout:     Attention dropout probability.
        bidirectional: If True, compute both directions of cross-attention.
        adaptive_gating: If True, use learned gates; else use equal 0.5 weights.
    """

    def __init__(
        self,
        d_model: int = 2048,
        num_rois: int = 5,
        dropout: float = 0.1,
        bidirectional: bool = True,
        adaptive_gating: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_rois = num_rois
        self.bidirectional = bidirectional
        self.adaptive_gating = adaptive_gating

        # One set of attention modules per ROI (shared weights is an option,
        # but per-ROI weights allow specialization, e.g., periorbital vs nasal)
        self.rgb_to_thermal_attn = nn.ModuleList(
            [CrossAttention(d_model, dropout) for _ in range(num_rois)]
        )
        self.thermal_to_rgb_attn = nn.ModuleList(
            [CrossAttention(d_model, dropout) for _ in range(num_rois)]
        )

        if adaptive_gating:
            self.gates = nn.ModuleList(
                [AdaptiveGate(d_model) for _ in range(num_rois)]
            )

        # Layer norm for stability
        self.norm = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_rois)]
        )

    def forward(
        self,
        rgb_features: List[torch.Tensor],
        thermal_features: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], Dict[str, List[torch.Tensor]]]:
        """
        Args:
            rgb_features:     List of num_rois tensors, each (B, d_model).
            thermal_features: List of num_rois tensors, each (B, d_model).

        Returns:
            fused_features: List of num_rois tensors, each (B, d_model).
            attention_info: Dict containing attention weights and gate values
                            for interpretability (Section 4.5 & 4.7 of paper).
        """
        assert len(rgb_features) == self.num_rois
        assert len(thermal_features) == self.num_rois

        fused_features = []
        attn_rgb_to_th = []
        attn_th_to_rgb = []
        gate_rgb_vals = []
        gate_th_vals = []

        for r in range(self.num_rois):
            f_rgb = rgb_features[r]       # (B, d_model)
            f_th = thermal_features[r]    # (B, d_model)

            # Direction 1: thermal context enriches RGB
            f_rgb_enriched, alpha_rgb_th = self.thermal_to_rgb_attn[r](
                query=f_rgb, key=f_th, value=f_th
            )

            # Direction 2: RGB context enriches thermal (bidirectional)
            if self.bidirectional:
                f_th_enriched, alpha_th_rgb = self.rgb_to_thermal_attn[r](
                    query=f_th, key=f_rgb, value=f_rgb
                )
            else:
                f_th_enriched = torch.zeros_like(f_th)
                alpha_th_rgb = torch.zeros_like(f_th)

            # Adaptive gating
            if self.adaptive_gating:
                lambda_rgb, lambda_th = self.gates[r](f_rgb, f_th)
            else:
                lambda_rgb = torch.full(
                    (f_rgb.size(0), 1), 0.5, device=f_rgb.device
                )
                lambda_th = lambda_rgb.clone()

            # Residual fusion (Eq. 9 in paper's Algorithm 2)
            f_fused = f_rgb + lambda_rgb * f_rgb_enriched + lambda_th * f_th_enriched
            f_fused = self.norm[r](f_fused)

            fused_features.append(f_fused)
            attn_rgb_to_th.append(alpha_rgb_th)
            attn_th_to_rgb.append(alpha_th_rgb)
            gate_rgb_vals.append(lambda_rgb)
            gate_th_vals.append(lambda_th)

        attention_info = {
            "attn_rgb_to_thermal": attn_rgb_to_th,    # list of (B, d_model)
            "attn_thermal_to_rgb": attn_th_to_rgb,    # list of (B, d_model)
            "lambda_rgb": gate_rgb_vals,               # list of (B, 1)
            "lambda_thermal": gate_th_vals,            # list of (B, 1)
        }

        return fused_features, attention_info

    @torch.no_grad()
    def get_modality_weights(
        self,
        rgb_features: List[torch.Tensor],
        thermal_features: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Convenience method to extract scalar modality contribution weights
        averaged over all ROIs. Used for the analyses in Table 4 and Figure 5.

        Returns:
            Dict with keys "rgb_weight" and "thermal_weight", each (B,).
        """
        _, attention_info = self.forward(rgb_features, thermal_features)
        # Stack per-ROI gate values: (B, num_rois)
        lambda_rgb = torch.cat(attention_info["lambda_rgb"], dim=-1)   # (B, num_rois)
        lambda_th = torch.cat(attention_info["lambda_thermal"], dim=-1)

        return {
            "rgb_weight": lambda_rgb.mean(dim=-1),      # (B,)
            "thermal_weight": lambda_th.mean(dim=-1),   # (B,)
        }
