"""
Modal-specific feature encoders for RGB and thermal modalities.

Both use ResNet-50 backbones:
  - RGB encoder: ImageNet + VGGFace2 pretraining
  - Thermal encoder: trained from scratch

Each produces 2048-dimensional regional features for the 5 facial ROIs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models
from typing import List, Optional


class ModalEncoder(nn.Module):
    """
    ResNet-50 encoder for a single modality (RGB or thermal).

    Processes each of the 5 facial ROIs independently and returns
    per-ROI 2048-dimensional feature vectors.

    Args:
        modality:       "rgb" or "thermal".
        pretrained:     "imagenet", "imagenet+vggface2", or "none".
        feature_dim:    Output feature dimension (default 2048, ResNet-50 GAP).
        num_rois:       Number of facial ROIs (default 5).
        input_channels: Input channels (3 for RGB, 1 for thermal).
        dropout:        Dropout probability applied after GAP.
    """

    def __init__(
        self,
        modality: str = "rgb",
        pretrained: str = "imagenet",
        feature_dim: int = 2048,
        num_rois: int = 5,
        input_channels: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert modality in ("rgb", "thermal"), f"Unknown modality: {modality}"
        self.modality = modality
        self.feature_dim = feature_dim
        self.num_rois = num_rois

        # ── Build backbone ────────────────────────────────────────────────
        use_pretrained = pretrained != "none" and modality == "rgb"
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if use_pretrained else None
        backbone = tv_models.resnet50(weights=weights)

        # Adapt first conv for thermal (1-channel) input
        if input_channels != 3:
            orig = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                input_channels,
                orig.out_channels,
                kernel_size=orig.kernel_size,
                stride=orig.stride,
                padding=orig.padding,
                bias=False,
            )
            if use_pretrained:
                # Average pretrained RGB weights across channel dim
                with torch.no_grad():
                    backbone.conv1.weight.copy_(
                        orig.weight.mean(dim=1, keepdim=True)
                    )

        # Remove the final FC layer; keep everything up to global average pool
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # → (B, 2048, 1, 1)
        self.dropout = nn.Dropout(p=dropout)

        # Optional projection to a different output dimension
        self.proj: Optional[nn.Linear] = None
        if feature_dim != 2048:
            self.proj = nn.Linear(2048, feature_dim)

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, rois: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            rois: Tensor of shape (B, num_rois, C, H, W).
                  B = batch, C = channels, H = W = 128.

        Returns:
            List of `num_rois` tensors each of shape (B, feature_dim).
        """
        B, R, C, H, W = rois.shape
        assert R == self.num_rois, (
            f"Expected {self.num_rois} ROIs, got {R}"
        )

        # Process all ROIs in parallel by merging batch and ROI dims
        x = rois.view(B * R, C, H, W)          # (B*R, C, H, W)
        x = self.features(x)                    # (B*R, 2048, h', w')
        x = self.gap(x).flatten(1)              # (B*R, 2048)
        x = self.dropout(x)
        if self.proj is not None:
            x = self.proj(x)                    # (B*R, feature_dim)

        # Split back into per-ROI list
        x = x.view(B, R, self.feature_dim)      # (B, R, feature_dim)
        return [x[:, r, :] for r in range(R)]   # list of R × (B, feature_dim)

    def encode_single_roi(self, roi: torch.Tensor) -> torch.Tensor:
        """
        Encode a single ROI.

        Args:
            roi: (B, C, H, W)
        Returns:
            (B, feature_dim)
        """
        x = self.features(roi)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        if self.proj is not None:
            x = self.proj(x)
        return x
