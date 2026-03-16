"""
Full CSAF+Transformer Pain Estimator.

End-to-end pipeline combining:
  1. ModalEncoder (RGB) — ResNet-50
  2. ModalEncoder (Thermal) — ResNet-50
  3. CrossSpectralAttentionFusion (CSAF)
  4. TemporalTransformer
  5. Regression head → continuous pain score ŷ ∈ [0, 10]

Total parameters: ~87M
Training hardware: 4×NVIDIA A100 (40 GB), ~48 h total

Paper: Cross-Spectral Fusion of Thermal and RGB Imaging
       for Objective Pain Estimation (El Othmani & Naouali, 2026)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .csaf import CrossSpectralAttentionFusion
from .encoders import ModalEncoder
from .temporal_transformer import TemporalTransformer


@dataclass
class PainEstimatorConfig:
    """Configuration for the full PainEstimator model."""

    # Encoders
    encoder_arch: str = "resnet50"
    encoder_feature_dim: int = 2048
    rgb_pretrained: str = "imagenet+vggface2"
    thermal_pretrained: str = "none"
    rgb_channels: int = 3
    thermal_channels: int = 1
    encoder_dropout: float = 0.1

    # CSAF
    csaf_d_model: int = 2048
    csaf_dropout: float = 0.1
    csaf_bidirectional: bool = True
    csaf_adaptive_gating: bool = True

    # Temporal Transformer
    transformer_d_model: int = 512
    transformer_n_layers: int = 6
    transformer_n_heads: int = 8
    transformer_d_ff: int = 2048
    transformer_max_seq_len: int = 300
    transformer_dropout: float = 0.1

    # Regression Head
    head_hidden_dims: List[int] = field(default_factory=lambda: [512, 512])
    head_dropout: float = 0.2

    # ROIs
    num_rois: int = 5

    # NRS range
    nrs_min: float = 0.0
    nrs_max: float = 10.0


class RegressionHead(nn.Module):
    """
    MLP regression head mapping pooled temporal features to pain score.

    Architecture: Linear → GELU → Dropout → Linear → GELU → Dropout → Linear(1) → Sigmoid × 10
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.2,
        nrs_max: float = 10.0,
    ) -> None:
        super().__init__()
        self.nrs_max = nrs_max

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev_dim = h_dim

        layers += [nn.Linear(prev_dim, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim)
        Returns:
            pain_score: (B,) in [0, nrs_max]
        """
        out = self.mlp(x)                    # (B, 1)
        out = torch.sigmoid(out)             # (B, 1) → [0, 1]
        return out.squeeze(-1) * self.nrs_max  # (B,) → [0, 10]


class PainEstimator(nn.Module):
    """
    Complete CSAF+Transformer pain estimation model.

    Forward pass for a single temporal window:
        rgb_rois:     (B, T, num_rois, 3, 128, 128)
        thermal_rois: (B, T, num_rois, 1, 128, 128)
        → pain_score: (B,) ∈ [0, 10]

    Args:
        config: PainEstimatorConfig dataclass.
    """

    def __init__(self, config: Optional[PainEstimatorConfig] = None) -> None:
        super().__init__()
        if config is None:
            config = PainEstimatorConfig()
        self.config = config

        # ── 1. Modal Encoders ─────────────────────────────────────────────
        self.rgb_encoder = ModalEncoder(
            modality="rgb",
            pretrained=config.rgb_pretrained,
            feature_dim=config.encoder_feature_dim,
            num_rois=config.num_rois,
            input_channels=config.rgb_channels,
            dropout=config.encoder_dropout,
        )
        self.thermal_encoder = ModalEncoder(
            modality="thermal",
            pretrained=config.thermal_pretrained,
            feature_dim=config.encoder_feature_dim,
            num_rois=config.num_rois,
            input_channels=config.thermal_channels,
            dropout=config.encoder_dropout,
        )

        # ── 2. CSAF Fusion ────────────────────────────────────────────────
        self.csaf = CrossSpectralAttentionFusion(
            d_model=config.csaf_d_model,
            num_rois=config.num_rois,
            dropout=config.csaf_dropout,
            bidirectional=config.csaf_bidirectional,
            adaptive_gating=config.csaf_adaptive_gating,
        )

        # ROI pooling: average fused features over all ROIs → single frame vector
        self.roi_pool = nn.Linear(config.num_rois * config.encoder_feature_dim,
                                  config.transformer_d_model)

        # ── 3. Temporal Transformer ───────────────────────────────────────
        self.temporal_transformer = TemporalTransformer(
            input_dim=config.transformer_d_model,
            d_model=config.transformer_d_model,
            n_layers=config.transformer_n_layers,
            n_heads=config.transformer_n_heads,
            d_ff=config.transformer_d_ff,
            max_seq_len=config.transformer_max_seq_len,
            dropout=config.transformer_dropout,
        )

        # ── 4. Regression Head ────────────────────────────────────────────
        pooled_dim = 2 * config.transformer_d_model  # mean + max pooling
        self.regression_head = RegressionHead(
            input_dim=pooled_dim,
            hidden_dims=config.head_hidden_dims,
            dropout=config.head_dropout,
            nrs_max=config.nrs_max,
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        rgb_rois: torch.Tensor,
        thermal_rois: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            rgb_rois:     (B, T, num_rois, 3, H, W)
            thermal_rois: (B, T, num_rois, 1, H, W)
            padding_mask: (B, T) bool — True = padded frame (for variable-length seqs)
            return_attention: If True, include attention maps in output dict.

        Returns:
            dict with:
                "pain_score":   (B,) ∈ [0, 10]
                "lambda_rgb":   (B, T, num_rois) — modality weights (if return_attention)
                "lambda_thermal": (B, T, num_rois)
                "temporal_attn": (B, T, T) — transformer attention (if return_attention)
                "attn_rgb_th":  list of per-ROI attention maps (if return_attention)
        """
        B, T, R, *_ = rgb_rois.shape
        assert R == self.config.num_rois

        # ── Encode each frame ─────────────────────────────────────────────
        # Merge batch and time dimensions for parallel processing
        rgb_flat = rgb_rois.view(B * T, R, *rgb_rois.shape[3:])       # (B*T, R, 3, H, W)
        th_flat = thermal_rois.view(B * T, R, *thermal_rois.shape[3:]) # (B*T, R, 1, H, W)

        rgb_feats = self.rgb_encoder(rgb_flat)         # list of R × (B*T, d)
        th_feats = self.thermal_encoder(th_flat)       # list of R × (B*T, d)

        # ── CSAF fusion ───────────────────────────────────────────────────
        fused_feats, attn_info = self.csaf(rgb_feats, th_feats)
        # fused_feats: list of R × (B*T, d)

        # Pool over ROIs: stack → (B*T, R, d) → reshape → (B*T, R*d) → project
        fused_stacked = torch.stack(fused_feats, dim=1)  # (B*T, R, d)
        frame_feat = self.roi_pool(
            fused_stacked.view(B * T, -1)                # (B*T, R*d)
        )                                                # (B*T, d_model)

        # ── Temporal Transformer ──────────────────────────────────────────
        frame_feat = frame_feat.view(B, T, -1)           # (B, T, d_model)
        pooled, temporal_attn = self.temporal_transformer(frame_feat, padding_mask)
        # pooled: (B, 2*d_model)

        # ── Regression ────────────────────────────────────────────────────
        pain_score = self.regression_head(pooled)        # (B,)

        output = {"pain_score": pain_score}

        if return_attention:
            # Reshape gate values to (B, T, num_rois) for interpretability
            def _reshape_gates(gate_list):
                # gate_list: list of R tensors each (B*T, 1)
                stacked = torch.cat(gate_list, dim=-1)         # (B*T, R)
                return stacked.view(B, T, R)                    # (B, T, R)

            output["lambda_rgb"] = _reshape_gates(attn_info["lambda_rgb"])
            output["lambda_thermal"] = _reshape_gates(attn_info["lambda_thermal"])
            output["temporal_attn"] = temporal_attn
            output["attn_rgb_to_thermal"] = attn_info["attn_rgb_to_thermal"]
            output["attn_thermal_to_rgb"] = attn_info["attn_thermal_to_rgb"]

        return output

    # ── Utilities ─────────────────────────────────────────────────────────────

    def freeze_encoders(self) -> None:
        """Freeze modal encoders (used in Stage 2 training)."""
        for param in self.rgb_encoder.parameters():
            param.requires_grad = False
        for param in self.thermal_encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoders(self) -> None:
        """Unfreeze modal encoders (used in Stage 3 fine-tuning)."""
        for param in self.rgb_encoder.parameters():
            param.requires_grad = True
        for param in self.thermal_encoder.parameters():
            param.requires_grad = True

    def count_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config_file(cls, config_path: str) -> "PainEstimator":
        """Load model from a YAML config file."""
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        model_cfg = cfg.get("model", {})
        config = PainEstimatorConfig(**model_cfg)
        return cls(config)
