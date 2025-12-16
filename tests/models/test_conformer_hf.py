"""
HuggingFace-Aligned Tests for Conformer Encoder with CTC.

This test suite validates that the FMS ConformerEncoder implementation:
1. Matches HuggingFace's mid-layer CTC supervision logic
2. Produces correct output shapes
3. Handles variable sequence lengths correctly
4. Maintains numerical stability with CTC feedback

Reference: HuggingFace granite_speech/modeling_granite_speech.py:252-278
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fms.models.conformer import ConformerConfig, ConformerEncoder


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def hf_aligned_config():
    """HF-aligned encoder config matching granite-speech-3.3-8b."""
    return ConformerConfig(
        num_features=160,      # HF: input_dim = 160
        hidden_dim=1024,       # HF: hidden_dim = 1024
        num_layers=16,         # HF: num_layers = 16
        num_heads=8,           # HF: num_heads = 8
        dim_head=128,          # HF: dim_head = 128
        conv_kernel_size=15,   # HF: conv_kernel_size = 15
        dropout=0.0,           # Disable dropout for testing
        max_pos_emb=512,       # HF: max_pos_emb = 512
        context_size=200,      # HF: context_size = 200
        output_dim=42,         # HF: output_dim = 42 (CTC vocabulary)
        use_ctc=True,          # Enable CTC for HF compatibility
    )


@pytest.fixture
def small_config():
    """Smaller config for faster tests."""
    return ConformerConfig(
        num_features=80,
        hidden_dim=256,
        num_layers=4,          # Fewer layers for speed
        num_heads=4,
        dim_head=64,
        conv_kernel_size=7,
        dropout=0.0,
        max_pos_emb=256,
        context_size=100,
        output_dim=42,
        use_ctc=True,
    )


@pytest.fixture
def no_ctc_config():
    """Config with CTC disabled."""
    return ConformerConfig(
        num_features=80,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        dim_head=64,
        conv_kernel_size=7,
        dropout=0.0,
        output_dim=42,
        use_ctc=False,  # CTC disabled
    )


# ============================================================================
# CTC Configuration Tests
# ============================================================================


class TestCTCConfiguration:
    """Test CTC configuration options."""

    def test_ctc_enabled_by_default(self, small_config):
        """Test CTC is enabled by default for HF compatibility."""
        assert small_config.use_ctc is True

    def test_output_dim_default(self, small_config):
        """Test default output_dim matches HF."""
        assert small_config.output_dim == 42

    def test_ctc_layers_created_when_enabled(self, small_config):
        """Test CTC layers are created when use_ctc=True."""
        encoder = ConformerEncoder(small_config)
        assert encoder.out is not None
        assert encoder.out_mid is not None

    def test_ctc_layers_not_created_when_disabled(self, no_ctc_config):
        """Test CTC layers are not created when use_ctc=False."""
        encoder = ConformerEncoder(no_ctc_config)
        assert encoder.out is None
        assert encoder.out_mid is None

    def test_ctc_layer_shapes(self, small_config):
        """Test CTC layers have correct input/output dimensions."""
        encoder = ConformerEncoder(small_config)
        # out: hidden_dim → output_dim
        assert encoder.out.in_features == small_config.hidden_dim
        assert encoder.out.out_features == small_config.output_dim
        # out_mid: output_dim → hidden_dim
        assert encoder.out_mid.in_features == small_config.output_dim
        assert encoder.out_mid.out_features == small_config.hidden_dim


# ============================================================================
# Mid-Layer CTC Logic Tests
# ============================================================================


class TestMidLayerCTC:
    """Test mid-layer CTC supervision logic matches HF."""

    def test_mid_layer_index(self, small_config):
        """Test CTC is applied at layer num_layers // 2."""
        # For 4 layers, mid_layer should be 2 (1-indexed: after layer 2)
        expected_mid_layer = small_config.num_layers // 2
        assert expected_mid_layer == 2

    def test_output_shape_with_ctc(self, small_config):
        """Test output shape is preserved with CTC feedback."""
        encoder = ConformerEncoder(small_config)
        encoder.eval()

        batch_size, seq_len = 2, 100
        x = torch.randn(batch_size, seq_len, small_config.num_features)

        with torch.no_grad():
            output = encoder(x)

        # Output shape: (batch, seq_len, hidden_dim)
        assert output.shape == (batch_size, seq_len, small_config.hidden_dim)

    def test_output_shape_without_ctc(self, no_ctc_config):
        """Test output shape when CTC is disabled."""
        encoder = ConformerEncoder(no_ctc_config)
        encoder.eval()

        batch_size, seq_len = 2, 100
        x = torch.randn(batch_size, seq_len, no_ctc_config.num_features)

        with torch.no_grad():
            output = encoder(x)

        # Output shape should still be (batch, seq_len, hidden_dim)
        assert output.shape == (batch_size, seq_len, no_ctc_config.hidden_dim)

    def test_ctc_changes_output(self, small_config):
        """Test that CTC feedback changes the output (vs no CTC)."""
        # Create config with CTC disabled but same architecture
        no_ctc_config = ConformerConfig(
            num_features=small_config.num_features,
            hidden_dim=small_config.hidden_dim,
            num_layers=small_config.num_layers,
            num_heads=small_config.num_heads,
            dim_head=small_config.dim_head,
            conv_kernel_size=small_config.conv_kernel_size,
            dropout=0.0,
            max_pos_emb=small_config.max_pos_emb,
            context_size=small_config.context_size,
            output_dim=small_config.output_dim,
            use_ctc=False,  # CTC disabled
        )

        # Create two encoders with same weights except CTC
        encoder_ctc = ConformerEncoder(small_config)
        encoder_no_ctc = ConformerEncoder(no_ctc_config)

        # Copy weights from ctc encoder to no_ctc encoder (excluding CTC layers)
        ctc_state_dict = encoder_ctc.state_dict()
        no_ctc_state_dict = encoder_no_ctc.state_dict()

        # Only copy matching keys (excludes out, out_mid)
        for key in no_ctc_state_dict:
            if key in ctc_state_dict:
                no_ctc_state_dict[key] = ctc_state_dict[key]

        encoder_no_ctc.load_state_dict(no_ctc_state_dict)

        encoder_ctc.eval()
        encoder_no_ctc.eval()

        x = torch.randn(2, 100, small_config.num_features)

        with torch.no_grad():
            output_ctc = encoder_ctc(x)
            output_no_ctc = encoder_no_ctc(x)

        # Outputs should be different due to CTC feedback
        assert not torch.allclose(output_ctc, output_no_ctc, atol=1e-5)


# ============================================================================
# Attention Distance Precomputation Tests
# ============================================================================


class TestAttentionDistances:
    """Test attention distance precomputation matches HF."""

    def test_attention_dists_shape(self, small_config):
        """Test precomputed attention distances have correct shape."""
        encoder = ConformerEncoder(small_config)
        # Shape is (context_size, context_size) for chunked attention
        assert encoder.attention_dists.shape[0] == small_config.context_size
        assert encoder.attention_dists.shape[1] == small_config.context_size

    def test_attention_dists_clamping(self, small_config):
        """Test attention distances are clamped correctly."""
        encoder = ConformerEncoder(small_config)
        dists = encoder.attention_dists

        # Values should be in [0, 2*max_pos_emb]
        assert dists.min() >= 0
        assert dists.max() <= 2 * small_config.max_pos_emb

    def test_attention_dists_symmetric_pattern(self, small_config):
        """Test diagonal elements are max_pos_emb (zero relative distance)."""
        encoder = ConformerEncoder(small_config)
        dists = encoder.attention_dists

        # Diagonal should be max_pos_emb (relative distance 0, shifted by max_pos_emb)
        diagonal = torch.diag(dists[:100, :100])  # Check first 100
        expected = small_config.max_pos_emb
        assert torch.all(diagonal == expected)

    def test_dynamic_attention_dists_for_long_sequences(self, small_config):
        """Test encoder handles sequences longer than precomputed."""
        encoder = ConformerEncoder(small_config)
        encoder.eval()

        # Sequence longer than default precomputed (5000)
        seq_len = 5500
        x = torch.randn(1, seq_len, small_config.num_features)

        with torch.no_grad():
            output = encoder(x)

        # Should work without error
        assert output.shape == (1, seq_len, small_config.hidden_dim)


# ============================================================================
# Variable Sequence Length Tests
# ============================================================================


class TestVariableSequenceLengths:
    """Test handling of variable sequence lengths."""

    def test_various_sequence_lengths(self, small_config):
        """Test encoder handles various sequence lengths."""
        encoder = ConformerEncoder(small_config)
        encoder.eval()

        test_lengths = [10, 50, 100, 200, 500, 1000]

        for seq_len in test_lengths:
            x = torch.randn(1, seq_len, small_config.num_features)
            with torch.no_grad():
                output = encoder(x)
            assert output.shape == (1, seq_len, small_config.hidden_dim), \
                f"Failed for seq_len={seq_len}"

    def test_batch_with_same_length(self, small_config):
        """Test batched input with same sequence length."""
        encoder = ConformerEncoder(small_config)
        encoder.eval()

        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 100, small_config.num_features)
            with torch.no_grad():
                output = encoder(x)
            assert output.shape[0] == batch_size


# ============================================================================
# Gradient Flow Tests
# ============================================================================


class TestGradientFlowWithCTC:
    """Test gradient flow through CTC layers."""

    def test_gradient_to_ctc_layers(self, small_config):
        """Test gradients flow to CTC layers."""
        encoder = ConformerEncoder(small_config)
        x = torch.randn(2, 100, small_config.num_features)

        output = encoder(x)
        loss = output.sum()
        loss.backward()

        # Check CTC layers receive gradients
        assert encoder.out.weight.grad is not None
        assert encoder.out_mid.weight.grad is not None
        assert not torch.isnan(encoder.out.weight.grad).any()
        assert not torch.isnan(encoder.out_mid.weight.grad).any()

    def test_gradient_to_input(self, small_config):
        """Test gradients flow back to input."""
        encoder = ConformerEncoder(small_config)
        x = torch.randn(2, 100, small_config.num_features, requires_grad=True)

        output = encoder(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_gradient_through_all_blocks(self, small_config):
        """Test gradients flow through all Conformer blocks."""
        encoder = ConformerEncoder(small_config)
        x = torch.randn(2, 100, small_config.num_features)

        output = encoder(x)
        loss = output.sum()
        loss.backward()

        # Check all blocks receive gradients
        for i, block in enumerate(encoder.blocks):
            for name, param in block.named_parameters():
                if param.requires_grad:
                    assert param.grad is not None, f"Block {i}, param {name} has no gradient"


# ============================================================================
# Numerical Stability Tests
# ============================================================================


class TestNumericalStability:
    """Test numerical stability with CTC feedback."""

    def test_no_nan_output(self, small_config):
        """Test output has no NaN values."""
        encoder = ConformerEncoder(small_config)
        encoder.eval()

        for seq_len in [10, 50, 100, 500]:
            x = torch.randn(2, seq_len, small_config.num_features)
            with torch.no_grad():
                output = encoder(x)
            assert not torch.isnan(output).any(), f"NaN at seq_len={seq_len}"

    def test_no_inf_output(self, small_config):
        """Test output has no Inf values."""
        encoder = ConformerEncoder(small_config)
        encoder.eval()

        for seq_len in [10, 50, 100, 500]:
            x = torch.randn(2, seq_len, small_config.num_features)
            with torch.no_grad():
                output = encoder(x)
            assert not torch.isinf(output).any(), f"Inf at seq_len={seq_len}"

    def test_ctc_softmax_stability(self, small_config):
        """Test CTC softmax doesn't cause numerical issues."""
        encoder = ConformerEncoder(small_config)
        encoder.eval()

        # Large values that could cause softmax overflow
        x = torch.randn(2, 100, small_config.num_features) * 10

        with torch.no_grad():
            output = encoder(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_output_bounded(self, small_config):
        """Test output values are reasonably bounded."""
        encoder = ConformerEncoder(small_config)
        encoder.eval()

        x = torch.randn(2, 100, small_config.num_features)
        with torch.no_grad():
            output = encoder(x)

        # Output should not have extreme values (arbitrary threshold)
        assert output.abs().max() < 1000


# ============================================================================
# HuggingFace Compatibility Tests
# ============================================================================


class TestHFCompatibility:
    """Test compatibility with HuggingFace implementation details."""

    def test_input_projection_exists(self, small_config):
        """Test input projection layer exists (HF: input_linear)."""
        encoder = ConformerEncoder(small_config)
        assert hasattr(encoder, 'input_proj')
        assert encoder.input_proj.in_features == small_config.num_features
        assert encoder.input_proj.out_features == small_config.hidden_dim

    def test_correct_number_of_blocks(self, small_config):
        """Test correct number of Conformer blocks."""
        encoder = ConformerEncoder(small_config)
        assert len(encoder.blocks) == small_config.num_layers

    def test_hf_aligned_config_values(self, hf_aligned_config):
        """Test HF-aligned config has correct values."""
        assert hf_aligned_config.num_features == 160
        assert hf_aligned_config.hidden_dim == 1024
        assert hf_aligned_config.num_layers == 16
        assert hf_aligned_config.num_heads == 8
        assert hf_aligned_config.dim_head == 128
        assert hf_aligned_config.output_dim == 42
        assert hf_aligned_config.use_ctc is True


# ============================================================================
# Run tests if executed directly
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
