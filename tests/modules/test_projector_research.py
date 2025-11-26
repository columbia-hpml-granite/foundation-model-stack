"""
Test suite for Speech Projector implementation.

Following the FMS testing standards established in test_conformer.py:
1. Architectural accuracy tests (shapes, components, initialization)
2. Research accuracy tests (representation quality, discriminability)
3. Integration tests (FMS patterns, serialization)
4. Component tests (individual modules)

The Speech Projector bridges Conformer encoder → Language decoder using Q-Former architecture.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fms.modules.projector import (
    SpeechProjectorConfig,
    QFormerSelfAttention,
    QFormerCrossAttention,
    QFormerAttentionOutput,
    QFormerFeedForward,
    QFormerLayer,
    SpeechProjector,
)


# ============================================================================
# Configuration Tests
# ============================================================================


class TestSpeechProjectorConfig:
    """Test configuration following FMS patterns."""

    def test_config_initialization(self):
        """Test that config initializes with correct defaults."""
        config = SpeechProjectorConfig()

        # Check all required attributes exist
        assert hasattr(config, "encoder_dim")
        assert hasattr(config, "decoder_dim")
        assert hasattr(config, "num_queries")
        assert hasattr(config, "num_hidden_layers")
        assert hasattr(config, "num_attention_heads")
        assert hasattr(config, "intermediate_size")

        # Check default values match specification
        assert config.encoder_dim == 1024
        assert config.decoder_dim == 2048
        assert config.num_queries == 32
        assert config.num_hidden_layers == 6

    def test_config_follows_fms_pattern(self):
        """Test that config follows FMS ModelConfig pattern."""
        from fms.utils.config import ModelConfig

        config = SpeechProjectorConfig()
        assert isinstance(config, ModelConfig)

    def test_config_custom_values(self):
        """Test that config accepts custom values."""
        config = SpeechProjectorConfig(
            encoder_dim=512,
            decoder_dim=1024,
            num_queries=16,
            num_hidden_layers=4,
        )

        assert config.encoder_dim == 512
        assert config.decoder_dim == 1024
        assert config.num_queries == 16
        assert config.num_hidden_layers == 4


# ============================================================================
# Component Tests (Individual Modules)
# ============================================================================


class TestQFormerSelfAttention:
    """Test self-attention component."""

    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return SpeechProjectorConfig(
            encoder_dim=256,
            num_attention_heads=4,  # 256 / 4 = 64 per head
            attention_dropout_prob=0.1,
        )

    @pytest.fixture
    def self_attn(self, config):
        """Provide self-attention module."""
        return QFormerSelfAttention(config)

    def test_self_attention_initialization(self, self_attn, config):
        """Test that self-attention initializes correctly."""
        assert isinstance(self_attn, nn.Module)
        assert self_attn.num_attention_heads == config.num_attention_heads
        assert hasattr(self_attn, "attention_head_size")

    def test_self_attention_forward_shape(self, self_attn, config):
        """Test that forward pass produces correct output shape."""
        batch_size = 2
        seq_len = 32
        hidden_dim = config.encoder_dim

        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        output = self_attn(hidden_states)

        # Output should match input shape
        assert output.shape == hidden_states.shape, \
            f"Expected {hidden_states.shape}, got {output.shape}"

    def test_self_attention_no_nan(self, self_attn):
        """Test that output doesn't contain NaN values."""
        hidden_states = torch.randn(2, 32, 256)
        output = self_attn(hidden_states)

        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"


class TestQFormerCrossAttention:
    """Test cross-attention component."""

    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return SpeechProjectorConfig(
            encoder_dim=256,
            num_attention_heads=4,
            attention_dropout_prob=0.1,
        )

    @pytest.fixture
    def cross_attn(self, config):
        """Provide cross-attention module."""
        return QFormerCrossAttention(config)

    def test_cross_attention_initialization(self, cross_attn, config):
        """Test that cross-attention initializes correctly."""
        assert isinstance(cross_attn, nn.Module)
        assert cross_attn.num_attention_heads == config.num_attention_heads

    def test_cross_attention_forward_shape(self, cross_attn, config):
        """Test that forward pass produces correct output shape."""
        batch_size = 2
        num_queries = 32
        audio_seq_len = 500
        hidden_dim = config.encoder_dim

        query_states = torch.randn(batch_size, num_queries, hidden_dim)
        encoder_hidden_states = torch.randn(batch_size, audio_seq_len, hidden_dim)

        output = cross_attn(query_states, encoder_hidden_states)

        # Output should match query shape (temporal downsampling happens here!)
        assert output.shape == query_states.shape, \
            f"Expected {query_states.shape}, got {output.shape}"

    def test_cross_attention_temporal_reduction(self, cross_attn):
        """Test that cross-attention achieves temporal downsampling."""
        batch_size = 1
        num_queries = 32
        audio_seq_len = 500
        hidden_dim = 256

        query_states = torch.randn(batch_size, num_queries, hidden_dim)
        encoder_hidden_states = torch.randn(batch_size, audio_seq_len, hidden_dim)

        output = cross_attn(query_states, encoder_hidden_states)

        # Verify downsampling: 500 → 32
        assert output.shape[1] == num_queries
        compression_ratio = audio_seq_len / num_queries
        assert compression_ratio == 15.625


class TestQFormerFeedForward:
    """Test feed-forward network component."""

    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return SpeechProjectorConfig(
            encoder_dim=256,
            intermediate_size=1024,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
        )

    @pytest.fixture
    def ffn(self, config):
        """Provide feed-forward module."""
        return QFormerFeedForward(config)

    def test_ffn_initialization(self, ffn):
        """Test that FFN initializes correctly."""
        assert isinstance(ffn, nn.Module)

    def test_ffn_forward_shape(self, ffn, config):
        """Test that FFN maintains input/output shape."""
        batch_size = 2
        seq_len = 32
        hidden_dim = config.encoder_dim

        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        output = ffn(hidden_states)

        assert output.shape == hidden_states.shape

    def test_ffn_no_nan(self, ffn):
        """Test that FFN produces valid outputs."""
        hidden_states = torch.randn(2, 32, 256)
        output = ffn(hidden_states)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestQFormerLayer:
    """Test complete Q-Former layer."""

    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return SpeechProjectorConfig(
            encoder_dim=256,
            num_attention_heads=4,
            intermediate_size=1024,
        )

    @pytest.fixture
    def layer(self, config):
        """Provide Q-Former layer."""
        return QFormerLayer(config)

    def test_layer_initialization(self, layer):
        """Test that Q-Former layer initializes correctly."""
        assert isinstance(layer, nn.Module)

    def test_layer_forward_shape(self, layer, config):
        """Test that layer forward maintains query shape."""
        batch_size = 2
        num_queries = 32
        audio_seq_len = 500
        hidden_dim = config.encoder_dim

        query_states = torch.randn(batch_size, num_queries, hidden_dim)
        encoder_hidden_states = torch.randn(batch_size, audio_seq_len, hidden_dim)

        output = layer(query_states, encoder_hidden_states)

        assert output.shape == query_states.shape

    def test_layer_forward_no_nan(self, layer):
        """Test that layer doesn't produce NaN values."""
        query_states = torch.randn(2, 32, 256)
        encoder_hidden_states = torch.randn(2, 500, 256)

        output = layer(query_states, encoder_hidden_states)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


# ============================================================================
# Main Projector Tests (Architectural Accuracy)
# ============================================================================


class TestSpeechProjector:
    """Test main Speech Projector module (following test_conformer.py patterns)."""

    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return SpeechProjectorConfig(
            encoder_dim=1024,
            decoder_dim=2048,
            num_queries=32,
            num_hidden_layers=6,
            num_attention_heads=8,
        )

    @pytest.fixture
    def projector(self, config):
        """Provide Speech Projector."""
        return SpeechProjector(config)

    def test_projector_initialization(self, projector, config):
        """Test that projector initializes correctly."""
        assert isinstance(projector, nn.Module)

        # Check that config is stored
        assert hasattr(projector, "config")
        assert projector.config == config

        # Check that learnable queries exist
        assert hasattr(projector, "query_embeds"), "Should have learnable queries"
        assert isinstance(projector.query_embeds, nn.Parameter), \
            "Queries should be parameters"

        # Check query shape
        expected_shape = (config.num_queries, config.encoder_dim)
        assert projector.query_embeds.shape == expected_shape

    def test_projector_forward_shape(self, projector, config):
        """Test that forward pass produces correct output shape."""
        batch_size = 2
        audio_seq_len = 500
        encoder_dim = config.encoder_dim

        encoder_hidden_states = torch.randn(batch_size, audio_seq_len, encoder_dim)
        output = projector(encoder_hidden_states)

        # Expected: (batch, num_queries, decoder_dim)
        expected_shape = (batch_size, config.num_queries, config.decoder_dim)
        assert output.shape == expected_shape, \
            f"Expected {expected_shape}, got {output.shape}"

    def test_projector_variable_sequence_length(self, projector, config):
        """Test that projector handles variable sequence lengths."""
        batch_size = 2
        encoder_dim = config.encoder_dim

        # Test different sequence lengths (like test_conformer.py)
        for seq_len in [100, 250, 500, 1000, 1500]:
            encoder_hidden_states = torch.randn(batch_size, seq_len, encoder_dim)
            output = projector(encoder_hidden_states)

            # Output should always be (batch, num_queries, decoder_dim)
            expected_shape = (batch_size, config.num_queries, config.decoder_dim)
            assert output.shape == expected_shape, \
                f"Failed for seq_len={seq_len}: expected {expected_shape}, got {output.shape}"

    def test_projector_output_no_nan(self, projector):
        """Test that projector doesn't produce NaN values."""
        encoder_hidden_states = torch.randn(2, 500, 1024)
        output = projector(encoder_hidden_states)

        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"

    def test_projector_deterministic_with_dropout_disabled(self, config):
        """Test that projector produces deterministic output when dropout is disabled."""
        config.hidden_dropout_prob = 0.0
        config.attention_dropout_prob = 0.0
        projector = SpeechProjector(config)
        projector.eval()

        encoder_hidden_states = torch.randn(2, 500, 1024)

        # Run twice
        output1 = projector(encoder_hidden_states)
        output2 = projector(encoder_hidden_states)

        # Should be identical
        assert torch.allclose(output1, output2), "Output should be deterministic"

    def test_projector_batch_independence(self, projector):
        """Test that samples in batch are processed independently."""
        projector.eval()

        audio_seq_len = 500
        encoder_dim = 1024

        # Create batch
        input_batch = torch.randn(4, audio_seq_len, encoder_dim)

        # Process full batch
        output_batch = projector(input_batch)

        # Process samples individually
        outputs_individual = [
            projector(input_batch[i:i+1]) for i in range(4)
        ]
        outputs_stacked = torch.cat(outputs_individual, dim=0)

        # Should be identical (within floating point precision)
        assert torch.allclose(output_batch, outputs_stacked, atol=1e-5)

    def test_projector_gradient_flow(self, projector):
        """Test that gradients flow through projector."""
        projector.train()

        encoder_hidden_states = torch.randn(2, 500, 1024, requires_grad=True)

        # Forward pass
        output = projector(encoder_hidden_states)

        # Compute dummy loss
        loss = output.sum()
        loss.backward()

        # Check that input has gradients
        assert encoder_hidden_states.grad is not None, "Gradients should flow to input"
        assert not torch.isnan(encoder_hidden_states.grad).any(), \
            "Gradients should not be NaN"

        # Check that query embeddings have gradients (learnable!)
        assert projector.query_embeds.grad is not None, \
            "Gradients should flow to query embeddings"

    def test_projector_no_temporal_expansion(self, projector, config):
        """Test that projector always outputs fixed number of queries."""
        # Unlike Conformer (preserves length), projector COMPRESSES to fixed size
        for seq_len in [100, 500, 1000]:
            encoder_hidden_states = torch.randn(1, seq_len, 1024)
            output = projector(encoder_hidden_states)

            # Output length should ALWAYS be num_queries (32)
            assert output.shape[1] == config.num_queries, \
                f"Output length should always be {config.num_queries}, got {output.shape[1]}"


# ============================================================================
# Integration Tests
# ============================================================================


class TestSpeechProjectorIntegration:
    """Integration tests following FMS patterns."""

    def test_projector_follows_fms_pattern(self):
        """Test that projector follows FMS architectural patterns."""
        config = SpeechProjectorConfig()
        projector = SpeechProjector(config)

        # Should be an nn.Module
        assert isinstance(projector, nn.Module)

        # Should have a config attribute
        assert hasattr(projector, "config")

        # Config should be a ModelConfig subclass
        from fms.utils.config import ModelConfig
        assert isinstance(config, ModelConfig)

    def test_projector_serialization_compatibility(self):
        """Test that projector can be saved/loaded with torch."""
        config = SpeechProjectorConfig(num_hidden_layers=2)
        projector = SpeechProjector(config)

        # Get initial parameters
        initial_state = projector.state_dict()

        # Create new projector and load state
        projector_new = SpeechProjector(config)
        projector_new.load_state_dict(initial_state)

        # Should produce identical outputs
        encoder_hidden_states = torch.randn(1, 500, 1024)

        projector.eval()
        projector_new.eval()

        output1 = projector(encoder_hidden_states)
        output2 = projector_new(encoder_hidden_states)

        assert torch.allclose(output1, output2, atol=1e-6), \
            "Loaded model should produce identical outputs"

    def test_projector_with_conformer_output(self):
        """Test projector with realistic Conformer output."""
        # Simulate Conformer output shape
        batch_size = 2
        audio_seq_len = 500
        encoder_dim = 1024

        config = SpeechProjectorConfig(
            encoder_dim=1024,
            decoder_dim=2048,
            num_queries=32,
        )
        projector = SpeechProjector(config)
        projector.eval()

        # Create dummy Conformer output
        conformer_output = torch.randn(batch_size, audio_seq_len, encoder_dim)

        # Forward pass
        decoder_input = projector(conformer_output)

        # Verify output for decoder
        assert decoder_input.shape == (batch_size, 32, 2048)
        assert not torch.isnan(decoder_input).any()


# ============================================================================
# Research Accuracy Tests (Representation Quality)
# ============================================================================


class TestSpeechProjectorRepresentations:
    """
    Research accuracy tests following test_conformer.py patterns.

    These tests validate that the projector learns meaningful representations
    and exhibits expected properties.
    """

    @pytest.fixture
    def small_projector(self):
        """Small projector for faster research tests."""
        config = SpeechProjectorConfig(
            encoder_dim=256,
            decoder_dim=512,
            num_queries=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            hidden_dropout_prob=0.0,
            attention_dropout_prob=0.0,
        )
        return SpeechProjector(config)

    def test_representation_collapse_detection(self, small_projector):
        """
        CRITICAL: Test that projector doesn't collapse all inputs to same representation.

        A collapsed projector outputs nearly identical representations for different inputs.
        """
        small_projector.eval()

        # Generate two different random inputs
        x1 = torch.randn(1, 200, 256)
        x2 = torch.randn(1, 200, 256)

        # Project both
        h1 = small_projector(x1)  # (1, 16, 512)
        h2 = small_projector(x2)  # (1, 16, 512)

        # Pool over query dimension
        h1_pooled = h1.mean(dim=1)  # (1, 512)
        h2_pooled = h2.mean(dim=1)  # (1, 512)

        # Compute cosine similarity
        cos_sim = F.cosine_similarity(h1_pooled, h2_pooled, dim=-1)

        # Different inputs should produce different representations
        assert cos_sim.item() < 0.95, (
            f"Representation collapse detected! Cosine similarity: {cos_sim.item():.3f}. "
            "Different inputs produce nearly identical representations."
        )

        # Also check L2 distance
        l2_dist = torch.norm(h1_pooled - h2_pooled, p=2)
        assert l2_dist.item() > 0.1, (
            f"Representations too similar (L2 dist: {l2_dist.item():.3f})"
        )

    def test_query_independence(self, small_projector):
        """
        Test that each query learns different patterns (not duplicates).

        Queries should specialize to extract different information.
        """
        small_projector.eval()

        # Create input
        x = torch.randn(1, 200, 256)
        output = small_projector(x)  # (1, 16, 512)

        # Each query should produce different output
        num_queries = output.shape[1]
        for i in range(num_queries - 1):
            assert not torch.allclose(output[0, i], output[0, i+1], atol=1e-3), \
                f"Query {i} and {i+1} should produce different outputs"

    def test_length_generalization(self, small_projector):
        """
        Test that projector generalizes to different sequence lengths.

        Projector must handle variable-length audio inputs.
        """
        small_projector.eval()

        # Test on various lengths
        test_lengths = [50, 100, 200, 500]

        outputs = []
        for seq_len in test_lengths:
            x = torch.randn(1, seq_len, 256)
            output = small_projector(x)

            # Check output statistics remain stable
            mean_val = output.mean().item()
            std_val = output.std().item()

            outputs.append((seq_len, mean_val, std_val))

            # Sanity checks
            assert not torch.isnan(output).any(), f"NaN for length {seq_len}"
            assert not torch.isinf(output).any(), f"Inf for length {seq_len}"

        # Check that statistics don't vary wildly across lengths
        means = [m for _, m, _ in outputs]
        stds = [s for _, _, s in outputs]

        mean_range = max(means) - min(means)
        std_range = max(stds) - min(stds)

        assert mean_range < 2.0, f"Mean varies too much across lengths: {mean_range:.3f}"
        assert std_range < 2.0, f"Std varies too much across lengths: {std_range:.3f}"

    def test_representation_discriminability(self, small_projector):
        """
        Test that projector produces discriminable representations for distinct inputs.

        Key property: If inputs are different, representations should be distinguishable.
        """
        small_projector.eval()

        seq_len = 200

        # Pattern 1: High values
        x1 = torch.ones(1, seq_len, 256) * 2.0

        # Pattern 2: Low values
        x2 = torch.ones(1, seq_len, 256) * (-2.0)

        # Pattern 3: Zeros
        x3 = torch.zeros(1, seq_len, 256)

        # Project all
        h1 = small_projector(x1).mean(dim=1)  # Pool over queries
        h2 = small_projector(x2).mean(dim=1)
        h3 = small_projector(x3).mean(dim=1)

        # Compute pairwise distances
        d12 = torch.norm(h1 - h2, p=2).item()
        d13 = torch.norm(h1 - h3, p=2).item()
        d23 = torch.norm(h2 - h3, p=2).item()

        # All pairs should be distinguishable
        assert d12 > 0.1, f"Patterns 1 and 2 too similar: {d12:.4f}"
        assert d13 > 0.1, f"Patterns 1 and 3 too similar: {d13:.4f}"
        assert d23 > 0.1, f"Patterns 2 and 3 too similar: {d23:.4f}"

    def test_temporal_compression_quality(self, small_projector):
        """
        Test that temporal compression preserves information.

        The projector should compress 200 frames → 16 queries while
        maintaining discriminability.
        """
        small_projector.eval()

        # Create two inputs that differ in a specific region
        seq_len = 200
        x_base = torch.randn(1, seq_len, 256)

        # Modify first half vs second half
        x1 = x_base.clone()
        x1[:, :100, :] += 2.0  # Boost first half

        x2 = x_base.clone()
        x2[:, 100:, :] += 2.0  # Boost second half

        # Project both
        h1 = small_projector(x1)
        h2 = small_projector(x2)

        # Outputs should be different (compression should preserve temporal info)
        assert not torch.allclose(h1, h2, atol=0.1), \
            "Temporal compression lost information about temporal structure"

        # Check that difference is substantial
        l2_dist = torch.norm(h1 - h2, p=2)
        assert l2_dist > 0.5, \
            f"Temporal patterns not sufficiently distinguished: {l2_dist:.4f}"

    def test_learnable_queries_are_learned(self, small_projector):
        """
        Test that learnable queries are actual parameters (not constants).

        This is THE key innovation: queries are learned, not fixed.
        """
        # Check that query_embeds is a parameter
        assert isinstance(small_projector.query_embeds, nn.Parameter)

        # Check that queries have requires_grad=True
        assert small_projector.query_embeds.requires_grad

        # Check that queries are in parameters()
        param_names = [name for name, _ in small_projector.named_parameters()]
        assert "query_embeds" in param_names

        # Check that gradient flows to queries
        small_projector.train()
        x = torch.randn(1, 200, 256)
        output = small_projector(x)
        loss = output.sum()
        loss.backward()

        assert small_projector.query_embeds.grad is not None, \
            "Queries should receive gradients (they are learnable!)"


# ============================================================================
# Performance and Numerical Stability Tests
# ============================================================================


class TestSpeechProjectorPerformance:
    """Performance tests following test_conformer_simple.py patterns."""

    def test_projector_parameter_count(self):
        """Test that projector has expected parameter count."""
        config = SpeechProjectorConfig()
        projector = SpeechProjector(config)

        total_params = sum(p.numel() for p in projector.parameters())

        # Learnable queries: 32 × 1024 = 32,768 parameters
        query_params = projector.query_embeds.numel()
        assert query_params == config.num_queries * config.encoder_dim

        # Total should be reasonable (depends on architecture)
        assert total_params > query_params, "Should have more than just query parameters"

        # Query parameters should be small relative to total
        query_ratio = query_params / total_params
        assert query_ratio < 0.01, \
            f"Query parameters should be <1% of total, got {query_ratio*100:.2f}%"

    def test_projector_forward_pass_performance(self):
        """Test basic forward pass performance."""
        config = SpeechProjectorConfig()
        projector = SpeechProjector(config)
        projector.eval()

        # Test input
        batch_size = 2
        seq_length = 500
        encoder_hidden_states = torch.randn(batch_size, seq_length, 1024)

        # Forward pass should complete without errors
        with torch.no_grad():
            output = projector(encoder_hidden_states)

        assert output.shape == (batch_size, 32, 2048)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
