"""
Temporal Transformer for modelling pain dynamics across time.

Architecture:
  - 6-layer, 8-head transformer (d_model = 512)
  - Sinusoidal positional encodings
  - Operates over 300-frame (10-second) windows at 30 fps
  - Mean + max temporal pooling before regression head
  - Learns to exploit the 1.2 s thermal precedence over RGB expressions
    (H3: Temporal Dynamics Hypothesis)

The transformer aggregates fused frame-level features from the CSAF module
to produce a window-level pain score.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding.

    PE(pos, 2i)   = sin(pos / 10000^{2i / d_model})
    PE(pos, 2i+1) = cos(pos / 10000^{2i / d_model})
    """

    def __init__(self, d_model: int = 512, max_len: int = 300, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                   # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model) with positional encoding added.
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer with pre-norm (more stable).

    pre-LN: LayerNorm → Multi-Head Self-Attention → residual
            LayerNorm → FFN → residual
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, d_model)
            src_key_padding_mask: (B, T) boolean mask (True = ignore)

        Returns:
            x: (B, T, d_model)
            attn_weights: (B, T, T)
        """
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        x, attn_weights = self.self_attn(
            x, x, x,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        x = residual + x

        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x, attn_weights


class TemporalTransformer(nn.Module):
    """
    Temporal transformer for pain dynamics modelling.

    Takes a sequence of per-frame fused features (after CSAF and ROI pooling)
    and aggregates them into a single window-level representation for regression.

    Args:
        input_dim:  Input feature dimension (sum of ROI dims after CSAF).
        d_model:    Internal transformer dimension (512 in paper).
        n_layers:   Number of transformer layers (6 in paper).
        n_heads:    Number of attention heads (8 in paper).
        d_ff:       Feed-forward hidden dimension (2048 in paper).
        max_seq_len: Maximum temporal window length (300 frames / 10 s).
        dropout:    Dropout probability.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        max_seq_len: int = 300,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Project input features to d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, input_dim) — sequence of per-frame fused features.
            padding_mask: (B, T) boolean tensor; True = padded frame.

        Returns:
            pooled: (B, 2 * d_model) — mean+max temporal pooling concatenated.
            last_attn: (B, T, T) — attention weights from the final layer
                       (used for temporal interpretability, Figure 6).
        """
        x = self.input_proj(x)     # (B, T, d_model)
        x = self.pos_enc(x)

        last_attn = None
        for layer in self.layers:
            x, last_attn = layer(x, src_key_padding_mask=padding_mask)

        x = self.norm(x)

        # Temporal pooling: mask padded frames before mean
        if padding_mask is not None:
            mask = (~padding_mask).float().unsqueeze(-1)  # (B, T, 1)
            mean_pool = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            x_masked = x.masked_fill(padding_mask.unsqueeze(-1), float("-inf"))
            max_pool = x_masked.max(dim=1).values
        else:
            mean_pool = x.mean(dim=1)   # (B, d_model)
            max_pool = x.max(dim=1).values

        pooled = torch.cat([mean_pool, max_pool], dim=-1)   # (B, 2*d_model)
        return pooled, last_attn
