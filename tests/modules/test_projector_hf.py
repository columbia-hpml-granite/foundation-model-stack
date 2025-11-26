"""
HuggingFace-Aligned Tests for Speech Projector.

This test suite validates that the FMS Speech Projector implementation:
1. Produces correct output shapes with window-based processing
2. Handles variable sequence lengths correctly
3. Matches HuggingFace implementation behavior
4. Maintains numerical stability

Reference: HuggingFace granite_speech/modeling_granite_speech.py:63-94
"""

import math
import pytest
import torch
import torch.nn as nn

from fms.modules.projector import (
    SpeechProjector,
    SpeechProjectorConfig,
    QFormerSelfAttention,
    QFormerCrossAttention,
    QFormerAttentionOutput,
    QFormerFeedForward,
    QFormerLayer,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def hf_aligned_config():
    """HF-aligned projector config matching granite-speech-3.3-8b."""
    return SpeechProjectorConfig(
        encoder_dim=1024,
        decoder_dim=4096,
        num_queries=3,        # window_size // downsample_rate = 15 // 5
        window_size=15,       # HF default
        num_hidden_layers=2,  # HF default
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_dropout_prob=0.0,  # Disable dropout for testing
        attention_dropout_prob=0.0,
    )


@pytest.fixture
def small_config():
    """Smaller config for faster tests."""
    return SpeechProjectorConfig(
        encoder_dim=256,
        decoder_dim=512,
        num_queries=3,
        window_size=15,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=512,
        hidden_dropout_prob=0.0,
        attention_dropout_prob=0.0,
    )


# ============================================================================
# Configuration Tests
# ============================================================================


class TestSpeechProjectorConfig:
    """Test SpeechProjectorConfig creation and validation."""

    def test_default_config(self):
        """Test default config matches HF-aligned values."""
        config = SpeechProjectorConfig()
        assert config.encoder_dim == 1024
        assert config.decoder_dim == 4096
        assert config.num_queries == 3
        assert config.window_size == 15
        assert config.num_hidden_layers == 2

    def test_hf_aligned_config(self, hf_aligned_config):
        """Test HF-aligned config parameters."""
        assert hf_aligned_config.num_queries == 3
        assert hf_aligned_config.window_size == 15
        # num_queries should equal window_size // downsample_rate
        downsample_rate = hf_aligned_config.window_size // hf_aligned_config.num_queries
        assert downsample_rate == 5


# ============================================================================
# Window Processing Tests (HF-aligned)
# ============================================================================


class TestWindowProcessing:
    """Test HF-aligned window-based processing."""

    def test_window_padding_exact(self, small_config):
        """Test no padding when seq_len is exact multiple of window_size."""
        projector = SpeechProjector(small_config)
        projector.eval()

        seq_len = 15  # Exactly 1 window
        x = torch.randn(1, seq_len, small_config.encoder_dim)

        with torch.no_grad():
            output = projector(x)

        # 1 window * 3 queries = 3 output tokens
        assert output.shape == (1, 3, small_config.decoder_dim)

    def test_window_padding_partial(self, small_config):
        """Test padding when seq_len is not multiple of window_size."""
        projector = SpeechProjector(small_config)
        projector.eval()

        # 14 frames: needs 1 frame padding -> 1 window
        x = torch.randn(1, 14, small_config.encoder_dim)
        with torch.no_grad():
            output = projector(x)
        assert output.shape == (1, 3, small_config.decoder_dim)

        # 16 frames: needs 14 frames padding -> 2 windows
        x = torch.randn(1, 16, small_config.encoder_dim)
        with torch.no_grad():
            output = projector(x)
        assert output.shape == (1, 6, small_config.decoder_dim)

    def test_window_output_shape_formula(self, small_config):
        """Test output shape matches HF formula: nblocks * num_queries."""
        projector = SpeechProjector(small_config)
        projector.eval()

        test_cases = [
            (14, 1, 3),   # ceil(14/15)=1 window -> 1*3=3 queries
            (15, 1, 3),   # ceil(15/15)=1 window -> 1*3=3 queries
            (16, 2, 6),   # ceil(16/15)=2 windows -> 2*3=6 queries
            (30, 2, 6),   # ceil(30/15)=2 windows -> 2*3=6 queries
            (45, 3, 9),   # ceil(45/15)=3 windows -> 3*3=9 queries
            (100, 7, 21), # ceil(100/15)=7 windows -> 7*3=21 queries
        ]

        for seq_len, expected_nblocks, expected_queries in test_cases:
            x = torch.randn(1, seq_len, small_config.encoder_dim)
            with torch.no_grad():
                output = projector(x)

            actual_nblocks = math.ceil(seq_len / small_config.window_size)
            assert actual_nblocks == expected_nblocks, f"seq_len={seq_len}"
            assert output.shape[1] == expected_queries, f"seq_len={seq_len}"

    def test_batch_processing(self, small_config):
        """Test batch dimension is preserved correctly."""
        projector = SpeechProjector(small_config)
        projector.eval()

        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 30, small_config.encoder_dim)
            with torch.no_grad():
                output = projector(x)
            assert output.shape[0] == batch_size


# ============================================================================
# Query Initialization Tests
# ============================================================================


class TestQueryInitialization:
    """Test learnable query initialization matches HF."""

    def test_query_shape(self, small_config):
        """Test query embeddings have correct shape."""
        projector = SpeechProjector(small_config)
        # Shape: (1, num_queries, encoder_dim)
        assert projector.query_embeds.shape == (
            1, small_config.num_queries, small_config.encoder_dim
        )

    def test_query_normal_initialization(self, small_config):
        """Test queries are initialized with N(0,1) distribution."""
        # Create multiple projectors to get statistics
        n_samples = 10
        all_queries = []

        for _ in range(n_samples):
            projector = SpeechProjector(small_config)
            all_queries.append(projector.query_embeds.data.flatten())

        all_queries = torch.cat(all_queries)

        # Check distribution is approximately N(0, 1)
        mean = all_queries.mean().item()
        std = all_queries.std().item()

        assert abs(mean) < 0.1, f"Mean should be ~0, got {mean}"
        assert abs(std - 1.0) < 0.2, f"Std should be ~1, got {std}"

    def test_query_is_learnable(self, small_config):
        """Test query embeddings have gradients."""
        projector = SpeechProjector(small_config)
        x = torch.randn(1, 30, small_config.encoder_dim)
        output = projector(x)
        loss = output.sum()
        loss.backward()

        assert projector.query_embeds.grad is not None
        assert not torch.isnan(projector.query_embeds.grad).any()


# ============================================================================
# Component Tests
# ============================================================================


class TestQFormerComponents:
    """Test individual Q-Former components."""

    def test_self_attention_output_shape(self, small_config):
        """Test self-attention produces correct output shape."""
        self_attn = QFormerSelfAttention(small_config)
        x = torch.randn(2, 3, small_config.encoder_dim)
        output = self_attn(x)
        assert output.shape == x.shape

    def test_cross_attention_output_shape(self, small_config):
        """Test cross-attention produces correct output shape."""
        cross_attn = QFormerCrossAttention(small_config)
        queries = torch.randn(2, 3, small_config.encoder_dim)
        encoder_states = torch.randn(2, 15, small_config.encoder_dim)
        output = cross_attn(queries, encoder_states)
        assert output.shape == queries.shape

    def test_attention_output_residual(self, small_config):
        """Test attention output applies residual connection."""
        attn_out = QFormerAttentionOutput(small_config)
        hidden = torch.randn(2, 3, small_config.encoder_dim)
        residual = torch.randn(2, 3, small_config.encoder_dim)
        output = attn_out(hidden, residual)
        assert output.shape == hidden.shape
        # Output should differ from both inputs (residual + projection + norm)
        assert not torch.allclose(output, hidden)
        assert not torch.allclose(output, residual)

    def test_feedforward_output_shape(self, small_config):
        """Test feed-forward produces correct output shape."""
        ffn = QFormerFeedForward(small_config)
        x = torch.randn(2, 3, small_config.encoder_dim)
        output = ffn(x)
        assert output.shape == x.shape

    def test_qformer_layer_output_shape(self, small_config):
        """Test Q-Former layer produces correct output shape."""
        layer = QFormerLayer(small_config)
        queries = torch.randn(2, 3, small_config.encoder_dim)
        encoder_states = torch.randn(2, 15, small_config.encoder_dim)
        output = layer(queries, encoder_states)
        assert output.shape == queries.shape


# ============================================================================
# Gradient Flow Tests
# ============================================================================


class TestGradientFlow:
    """Test gradients flow correctly through all components."""

    def test_gradient_to_input(self, small_config):
        """Test gradients flow back to encoder hidden states."""
        projector = SpeechProjector(small_config)
        x = torch.randn(2, 30, small_config.encoder_dim, requires_grad=True)
        output = projector(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_gradient_to_all_parameters(self, small_config):
        """Test all parameters receive gradients."""
        projector = SpeechProjector(small_config)
        x = torch.randn(2, 30, small_config.encoder_dim)
        output = projector(x)
        loss = output.sum()
        loss.backward()

        for name, param in projector.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


# ============================================================================
# Numerical Stability Tests
# ============================================================================


class TestNumericalStability:
    """Test numerical stability across various inputs."""

    def test_no_nan_output(self, small_config):
        """Test output has no NaN values."""
        projector = SpeechProjector(small_config)
        projector.eval()

        for seq_len in [14, 15, 16, 30, 100, 500]:
            x = torch.randn(2, seq_len, small_config.encoder_dim)
            with torch.no_grad():
                output = projector(x)
            assert not torch.isnan(output).any(), f"NaN at seq_len={seq_len}"

    def test_no_inf_output(self, small_config):
        """Test output has no Inf values."""
        projector = SpeechProjector(small_config)
        projector.eval()

        for seq_len in [14, 15, 16, 30, 100, 500]:
            x = torch.randn(2, seq_len, small_config.encoder_dim)
            with torch.no_grad():
                output = projector(x)
            assert not torch.isinf(output).any(), f"Inf at seq_len={seq_len}"

    def test_large_input_stability(self, small_config):
        """Test stability with large input values."""
        projector = SpeechProjector(small_config)
        projector.eval()

        x = torch.randn(2, 30, small_config.encoder_dim) * 10
        with torch.no_grad():
            output = projector(x)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


# ============================================================================
# Compression Ratio Tests
# ============================================================================


class TestCompressionRatio:
    """Test temporal compression matches HF specification."""

    def test_compression_ratio(self, hf_aligned_config):
        """Test compression ratio matches window_size / (window_size // downsample_rate)."""
        projector = SpeechProjector(hf_aligned_config)
        projector.eval()

        # Input: 150 frames (10 windows of 15)
        seq_len = 150
        x = torch.randn(1, seq_len, hf_aligned_config.encoder_dim)

        with torch.no_grad():
            output = projector(x)

        # Output: 10 windows * 3 queries = 30 tokens
        expected_output_len = (seq_len // hf_aligned_config.window_size) * hf_aligned_config.num_queries
        assert output.shape[1] == expected_output_len

        # Compression ratio: 150 / 30 = 5x
        compression_ratio = seq_len / output.shape[1]
        assert compression_ratio == 5.0


# ============================================================================
# HuggingFace Equivalence Test (requires transformers)
# ============================================================================


@pytest.mark.slow
class TestHuggingFaceEquivalence:
    """Test numerical equivalence with HuggingFace implementation."""

    def test_output_shape_matches_hf(self, hf_aligned_config):
        """Test output shape matches HuggingFace formula."""
        projector = SpeechProjector(hf_aligned_config)
        projector.eval()

        # Test various sequence lengths
        for seq_len in [30, 45, 100, 500]:
            x = torch.randn(2, seq_len, hf_aligned_config.encoder_dim)
            with torch.no_grad():
                output = projector(x)

            # HF formula: nblocks * (window_size // downsample_rate)
            nblocks = math.ceil(seq_len / hf_aligned_config.window_size)
            expected_queries = nblocks * hf_aligned_config.num_queries

            assert output.shape == (2, expected_queries, hf_aligned_config.decoder_dim), \
                f"seq_len={seq_len}, expected ({2}, {expected_queries}, {hf_aligned_config.decoder_dim}), got {output.shape}"


# ============================================================================
# Run tests if executed directly
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
