"""
Unit tests for CSAF model components.

Tests cover:
  - ModalEncoder: forward pass shapes, modality variants
  - CrossSpectralAttentionFusion: shapes, bidirectionality, gating
  - TemporalTransformer: sequence handling, pooling
  - PainEstimator: full pipeline end-to-end
"""

import pytest
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.csaf import CrossSpectralAttentionFusion, CrossAttention, AdaptiveGate
from src.models.encoders import ModalEncoder
from src.models.temporal_transformer import TemporalTransformer, SinusoidalPositionalEncoding
from src.models.pain_estimator import PainEstimator, PainEstimatorConfig


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def batch_size():
    return 2

@pytest.fixture
def n_rois():
    return 5

@pytest.fixture
def d_model():
    return 256   # use smaller dimension for fast testing

@pytest.fixture
def seq_len():
    return 30    # shorter than 300 for speed

@pytest.fixture
def roi_inputs(batch_size, n_rois, d_model):
    """Dummy per-ROI feature tensors."""
    return [torch.randn(batch_size, d_model) for _ in range(n_rois)]


# ── ModalEncoder tests ─────────────────────────────────────────────────────────

class TestModalEncoder:

    def test_rgb_forward_shape(self, batch_size, n_rois):
        encoder = ModalEncoder(modality="rgb", pretrained="none",
                               feature_dim=256, num_rois=n_rois,
                               input_channels=3)
        rois = torch.randn(batch_size, n_rois, 3, 128, 128)
        out = encoder(rois)
        assert isinstance(out, list)
        assert len(out) == n_rois
        assert out[0].shape == (batch_size, 256)

    def test_thermal_forward_shape(self, batch_size, n_rois):
        encoder = ModalEncoder(modality="thermal", pretrained="none",
                               feature_dim=256, num_rois=n_rois,
                               input_channels=1)
        rois = torch.randn(batch_size, n_rois, 1, 128, 128)
        out = encoder(rois)
        assert len(out) == n_rois
        assert out[0].shape == (batch_size, 256)

    def test_wrong_num_rois_raises(self, batch_size, n_rois):
        encoder = ModalEncoder(modality="rgb", pretrained="none",
                               feature_dim=256, num_rois=n_rois,
                               input_channels=3)
        rois = torch.randn(batch_size, n_rois + 1, 3, 128, 128)
        with pytest.raises(AssertionError):
            encoder(rois)


# ── CrossAttention tests ───────────────────────────────────────────────────────

class TestCrossAttention:

    def test_output_shape(self, batch_size, d_model):
        attn = CrossAttention(d_model=d_model, dropout=0.0)
        q = torch.randn(batch_size, d_model)
        k = torch.randn(batch_size, d_model)
        v = torch.randn(batch_size, d_model)
        out, weights = attn(q, k, v)
        assert out.shape == (batch_size, d_model)
        assert weights.shape == (batch_size, d_model)

    def test_attention_weights_sum_to_one(self, batch_size, d_model):
        attn = CrossAttention(d_model=d_model, dropout=0.0)
        q = torch.randn(batch_size, d_model)
        k = torch.randn(batch_size, d_model)
        v = torch.randn(batch_size, d_model)
        _, weights = attn(q, k, v)
        # Softmax over d_model dim should sum to 1
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(batch_size), atol=1e-5)


# ── AdaptiveGate tests ─────────────────────────────────────────────────────────

class TestAdaptiveGate:

    def test_gates_in_range(self, batch_size, d_model):
        gate = AdaptiveGate(d_model=d_model)
        f_rgb = torch.randn(batch_size, d_model)
        f_th = torch.randn(batch_size, d_model)
        lam_rgb, lam_th = gate(f_rgb, f_th)
        assert lam_rgb.shape == (batch_size, 1)
        assert lam_th.shape == (batch_size, 1)
        assert (lam_rgb >= 0).all() and (lam_rgb <= 1).all()
        assert (lam_th >= 0).all() and (lam_th <= 1).all()


# ── CSAF tests ─────────────────────────────────────────────────────────────────

class TestCSAF:

    def test_output_shape(self, batch_size, n_rois, d_model, roi_inputs):
        csaf = CrossSpectralAttentionFusion(d_model=d_model, num_rois=n_rois, dropout=0.0)
        rgb_feats = roi_inputs
        th_feats = [torch.randn(batch_size, d_model) for _ in range(n_rois)]
        fused, attn_info = csaf(rgb_feats, th_feats)
        assert len(fused) == n_rois
        assert fused[0].shape == (batch_size, d_model)

    def test_attn_info_keys(self, batch_size, n_rois, d_model, roi_inputs):
        csaf = CrossSpectralAttentionFusion(d_model=d_model, num_rois=n_rois, dropout=0.0)
        th_feats = [torch.randn(batch_size, d_model) for _ in range(n_rois)]
        _, attn_info = csaf(roi_inputs, th_feats)
        assert "lambda_rgb" in attn_info
        assert "lambda_thermal" in attn_info
        assert len(attn_info["lambda_rgb"]) == n_rois

    def test_unidirectional_mode(self, batch_size, n_rois, d_model, roi_inputs):
        csaf = CrossSpectralAttentionFusion(
            d_model=d_model, num_rois=n_rois, dropout=0.0, bidirectional=False
        )
        th_feats = [torch.randn(batch_size, d_model) for _ in range(n_rois)]
        fused, _ = csaf(roi_inputs, th_feats)
        assert len(fused) == n_rois

    def test_no_gating_mode(self, batch_size, n_rois, d_model, roi_inputs):
        csaf = CrossSpectralAttentionFusion(
            d_model=d_model, num_rois=n_rois, dropout=0.0, adaptive_gating=False
        )
        th_feats = [torch.randn(batch_size, d_model) for _ in range(n_rois)]
        fused, _ = csaf(roi_inputs, th_feats)
        assert len(fused) == n_rois

    def test_gradient_flow(self, batch_size, n_rois, d_model):
        """Verify gradients flow through all CSAF parameters."""
        csaf = CrossSpectralAttentionFusion(d_model=d_model, num_rois=n_rois)
        rgb_feats = [torch.randn(batch_size, d_model, requires_grad=True)
                     for _ in range(n_rois)]
        th_feats = [torch.randn(batch_size, d_model, requires_grad=True)
                    for _ in range(n_rois)]
        fused, _ = csaf(rgb_feats, th_feats)
        loss = sum(f.sum() for f in fused)
        loss.backward()
        for f in rgb_feats + th_feats:
            assert f.grad is not None


# ── TemporalTransformer tests ──────────────────────────────────────────────────

class TestTemporalTransformer:

    def test_output_shape(self, batch_size, seq_len):
        transformer = TemporalTransformer(
            input_dim=128, d_model=64, n_layers=2, n_heads=4,
            d_ff=128, max_seq_len=100
        )
        x = torch.randn(batch_size, seq_len, 128)
        pooled, attn = transformer(x)
        assert pooled.shape == (batch_size, 128)   # 2 * d_model
        assert attn is not None

    def test_padding_mask(self, batch_size, seq_len):
        transformer = TemporalTransformer(
            input_dim=128, d_model=64, n_layers=2, n_heads=4, max_seq_len=100
        )
        x = torch.randn(batch_size, seq_len, 128)
        # Mask last 5 frames of each sequence
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask[:, -5:] = True
        pooled_masked, _ = transformer(x, padding_mask=mask)
        pooled_unmasked, _ = transformer(x)
        # Outputs should differ with masking
        assert not torch.allclose(pooled_masked, pooled_unmasked)

    def test_positional_encoding_shape(self, batch_size, seq_len):
        pe = SinusoidalPositionalEncoding(d_model=64, max_len=100, dropout=0.0)
        x = torch.randn(batch_size, seq_len, 64)
        out = pe(x)
        assert out.shape == x.shape


# ── PainEstimator end-to-end tests ─────────────────────────────────────────────

class TestPainEstimator:

    @pytest.fixture
    def small_config(self):
        return PainEstimatorConfig(
            rgb_pretrained="none",
            thermal_pretrained="none",
            encoder_feature_dim=256,
            csaf_d_model=256,
            transformer_d_model=64,
            transformer_n_layers=2,
            transformer_n_heads=4,
            transformer_d_ff=128,
            transformer_max_seq_len=30,
            head_hidden_dims=[64],
            num_rois=5,
        )

    def test_forward_shape(self, small_config):
        model = PainEstimator(small_config)
        B, T = 2, 15
        rgb = torch.randn(B, T, 5, 3, 128, 128)
        thermal = torch.randn(B, T, 5, 1, 128, 128)
        output = model(rgb, thermal)
        assert "pain_score" in output
        assert output["pain_score"].shape == (B,)

    def test_output_range(self, small_config):
        """Pain score should be in [0, 10]."""
        model = PainEstimator(small_config)
        B, T = 2, 15
        rgb = torch.randn(B, T, 5, 3, 128, 128)
        thermal = torch.randn(B, T, 5, 1, 128, 128)
        output = model(rgb, thermal)
        scores = output["pain_score"]
        assert (scores >= 0).all() and (scores <= 10).all()

    def test_attention_output(self, small_config):
        model = PainEstimator(small_config)
        B, T = 2, 15
        rgb = torch.randn(B, T, 5, 3, 128, 128)
        thermal = torch.randn(B, T, 5, 1, 128, 128)
        output = model(rgb, thermal, return_attention=True)
        assert "lambda_rgb" in output
        assert "lambda_thermal" in output
        assert output["lambda_rgb"].shape == (B, T, 5)

    def test_freeze_unfreeze_encoders(self, small_config):
        model = PainEstimator(small_config)
        model.freeze_encoders()
        for p in model.rgb_encoder.parameters():
            assert not p.requires_grad
        model.unfreeze_encoders()
        for p in model.rgb_encoder.parameters():
            assert p.requires_grad

    def test_parameter_count(self, small_config):
        model = PainEstimator(small_config)
        n_params = model.count_parameters()
        assert n_params > 0
        # Verify it's a positive integer
        assert isinstance(n_params, int)

    def test_backward_pass(self, small_config):
        """Verify loss.backward() works without errors."""
        model = PainEstimator(small_config)
        B, T = 2, 10
        rgb = torch.randn(B, T, 5, 3, 128, 128)
        thermal = torch.randn(B, T, 5, 1, 128, 128)
        target = torch.rand(B) * 10  # random NRS in [0, 10]
        output = model(rgb, thermal)
        loss = torch.abs(output["pain_score"] - target).mean()
        loss.backward()
        # Check some gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN grad in {name}"
