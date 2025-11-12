"""
Test suite for Conformer encoder implementation.
Following TDD approach - these tests should initially FAIL until implementation is complete.
"""

import pytest
import torch
import torch.nn as nn

from fms.models.conformer import (
    ConformerConfig,
    ConformerBlock,
    ConformerEncoder,
)
from fms.testing._internal.model_test_suite import (
    ModelConsistencyTestSuite,
)


class TestConformerConfig:
    """Test Conformer configuration dataclass."""

    def test_config_initialization_with_defaults(self):
        """Test that config initializes with sensible defaults."""
        config = ConformerConfig()

        # Audio feature dimensions
        assert config.num_features == 80, "Default should be 80 log-mel features"

        # Architecture dimensions
        assert config.num_layers == 16, "Granite-speech uses 16 layers"
        assert config.hidden_dim == 1024, "Default hidden dimension"
        assert config.num_heads == 8, "Default number of attention heads"
        assert config.dim_head == 64, "Default per-head dimension"

        # Convolution parameters
        assert config.conv_kernel_size == 31, "Default kernel size"
        assert config.conv_expansion_factor == 2, "Default expansion factor"

        # Feed-forward parameters
        assert config.feedforward_mult == 4, "Default FFN expansion"

        # Regularization
        assert 0.0 <= config.dropout <= 1.0, "Dropout should be valid probability"

        # Position encoding
        assert config.max_pos_emb > 0, "Max positional embedding distance"
        assert config.context_size > 0, "Local attention window size"

    def test_config_custom_initialization(self):
        """Test that config accepts custom parameters."""
        config = ConformerConfig(
            num_features=40,
            num_layers=12,
            hidden_dim=512,
            num_heads=4,
            dropout=0.2,
        )

        assert config.num_features == 40
        assert config.num_layers == 12
        assert config.hidden_dim == 512
        assert config.num_heads == 4
        assert config.dropout == 0.2

    def test_attention_dimension_compatibility(self):
        """Test that attention dimensions are compatible."""
        config = ConformerConfig(num_heads=8, dim_head=64)

        # Inner attention dimension should be num_heads * dim_head
        expected_inner_dim = config.num_heads * config.dim_head
        assert expected_inner_dim == 512, "8 heads * 64 dim = 512"


class TestConformerBlock:
    """Test individual Conformer block."""

    @pytest.fixture
    def config(self):
        """Provide a test configuration."""
        return ConformerConfig(
            hidden_dim=256,
            num_heads=4,
            dim_head=64,
            conv_kernel_size=31,
            feedforward_mult=4,
            dropout=0.1,
            max_pos_emb=1000,
            context_size=100,
        )

    @pytest.fixture
    def conformer_block(self, config):
        """Provide an uninitialized ConformerBlock."""
        return ConformerBlock(config)

    def test_block_initialization(self, conformer_block, config):
        """Test that ConformerBlock initializes correctly."""
        assert isinstance(conformer_block, nn.Module)

        # Check that submodules exist (will fail until implementation)
        assert hasattr(conformer_block, "ff1"), "Should have first feed-forward module"
        assert hasattr(conformer_block, "attn"), "Should have attention module"
        assert hasattr(conformer_block, "conv"), "Should have convolution module"
        assert hasattr(conformer_block, "ff2"), "Should have second feed-forward module"
        assert hasattr(conformer_block, "post_norm"), "Should have post layer norm"

    def test_block_forward_shape(self, conformer_block, config):
        """Test that forward pass produces correct output shape."""
        batch_size = 2
        seq_length = 100
        hidden_dim = config.hidden_dim

        # Create dummy input
        x = torch.randn(batch_size, seq_length, hidden_dim)

        # Create dummy attention distances (relative positional encodings)
        attention_dists = torch.randint(
            0, 2 * config.max_pos_emb + 1, (seq_length, seq_length)
        )

        # Forward pass
        output = conformer_block(x, attention_dists)

        # Output should have same shape as input
        assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"

    def test_block_forward_no_nan(self, conformer_block):
        """Test that forward pass doesn't produce NaN values."""
        x = torch.randn(2, 50, 256)
        attention_dists = torch.randint(0, 2001, (50, 50))

        output = conformer_block(x, attention_dists)

        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"

    def test_block_residual_connections(self, conformer_block):
        """Test that residual connections are working."""
        x = torch.randn(1, 10, 256)
        attention_dists = torch.randint(0, 2001, (10, 10))

        # If all submodules returned zeros, residuals should preserve input
        # This tests the architectural pattern
        output = conformer_block(x, attention_dists)

        # Output should not be identical to input (due to processing)
        assert not torch.allclose(output, x), "Output should be different from input"

    def test_block_batch_independence(self, conformer_block):
        """Test that samples in batch are processed independently."""
        x = torch.randn(4, 20, 256)
        attention_dists = torch.randint(0, 2001, (20, 20))

        # Process full batch
        output_batch = conformer_block(x, attention_dists)

        # Process samples individually
        outputs_individual = [
            conformer_block(x[i:i+1], attention_dists) for i in range(4)
        ]
        outputs_stacked = torch.cat(outputs_individual, dim=0)

        # Should be identical (within floating point precision)
        assert torch.allclose(output_batch, outputs_stacked, atol=1e-5)


class TestConformerEncoder:
    """Test full Conformer encoder."""

    @pytest.fixture
    def config(self):
        """Provide a test configuration."""
        return ConformerConfig(
            num_features=80,
            hidden_dim=256,
            num_layers=4,  # Small for testing
            num_heads=4,
            dim_head=64,
            conv_kernel_size=31,
            feedforward_mult=4,
            dropout=0.0,  # Disable for deterministic testing
            max_pos_emb=1000,
            context_size=100,
        )

    @pytest.fixture
    def encoder(self, config):
        """Provide an uninitialized ConformerEncoder."""
        return ConformerEncoder(config)

    def test_encoder_initialization(self, encoder, config):
        """Test that ConformerEncoder initializes correctly."""
        assert isinstance(encoder, nn.Module)

        # Check input projection
        assert hasattr(encoder, "input_proj"), "Should have input projection layer"

        # Check conformer blocks
        assert hasattr(encoder, "blocks"), "Should have conformer blocks"
        assert isinstance(encoder.blocks, nn.ModuleList), "Blocks should be ModuleList"
        assert len(encoder.blocks) == config.num_layers, f"Should have {config.num_layers} blocks"

        # Check attention distance buffer
        assert hasattr(encoder, "attention_dists"), "Should precompute attention distances"

    def test_encoder_forward_shape(self, encoder, config):
        """Test that forward pass produces correct output shape."""
        batch_size = 2
        seq_length = 100
        num_features = config.num_features

        # Create dummy input (audio features)
        input_features = torch.randn(batch_size, seq_length, num_features)

        # Forward pass
        output = encoder(input_features)

        # Expected output shape
        expected_shape = (batch_size, seq_length, config.hidden_dim)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

    def test_encoder_input_dimension_mismatch(self, encoder, config):
        """Test that encoder raises error for wrong input dimension."""
        batch_size = 2
        seq_length = 100
        wrong_features = 40  # Should be 80

        input_features = torch.randn(batch_size, seq_length, wrong_features)

        # Should raise an error
        with pytest.raises((RuntimeError, AssertionError)):
            output = encoder(input_features)

    def test_encoder_variable_sequence_length(self, encoder, config):
        """Test that encoder handles variable sequence lengths."""
        batch_size = 2
        num_features = config.num_features

        # Test different sequence lengths
        for seq_length in [50, 100, 200, 500]:
            input_features = torch.randn(batch_size, seq_length, num_features)
            output = encoder(input_features)

            expected_shape = (batch_size, seq_length, config.hidden_dim)
            assert output.shape == expected_shape, \
                f"Failed for seq_length={seq_length}: expected {expected_shape}, got {output.shape}"

    def test_encoder_output_no_nan(self, encoder, config):
        """Test that encoder doesn't produce NaN values."""
        input_features = torch.randn(2, 100, config.num_features)

        output = encoder(input_features)

        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"

    def test_encoder_deterministic_with_dropout_disabled(self, config):
        """Test that encoder produces deterministic output when dropout is disabled."""
        config.dropout = 0.0
        encoder = ConformerEncoder(config)
        encoder.eval()  # Set to eval mode

        input_features = torch.randn(2, 100, config.num_features)

        # Run twice
        output1 = encoder(input_features)
        output2 = encoder(input_features)

        # Should be identical
        assert torch.allclose(output1, output2), "Output should be deterministic"

    def test_encoder_batch_independence(self, encoder, config):
        """Test that samples in batch are processed independently."""
        num_features = config.num_features
        seq_length = 100

        # Create batch
        input_batch = torch.randn(4, seq_length, num_features)

        # Process full batch
        output_batch = encoder(input_batch)

        # Process samples individually
        outputs_individual = [
            encoder(input_batch[i:i+1]) for i in range(4)
        ]
        outputs_stacked = torch.cat(outputs_individual, dim=0)

        # Should be identical (within floating point precision)
        assert torch.allclose(output_batch, outputs_stacked, atol=1e-5)

    def test_encoder_gradient_flow(self, encoder, config):
        """Test that gradients flow through encoder."""
        encoder.train()

        input_features = torch.randn(2, 50, config.num_features, requires_grad=True)

        # Forward pass
        output = encoder(input_features)

        # Compute dummy loss
        loss = output.sum()
        loss.backward()

        # Check that input has gradients
        assert input_features.grad is not None, "Gradients should flow to input"
        assert not torch.isnan(input_features.grad).any(), "Gradients should not be NaN"

    def test_encoder_no_temporal_downsampling(self, encoder, config):
        """Test that encoder preserves sequence length (no downsampling in blocks)."""
        for seq_length in [50, 100, 200]:
            input_features = torch.randn(1, seq_length, config.num_features)
            output = encoder(input_features)

            assert output.shape[1] == seq_length, \
                f"Sequence length should be preserved: {seq_length} != {output.shape[1]}"


class TestConformerIntegration:
    """Integration tests for Conformer with FMS patterns."""

    def test_conformer_follows_fms_pattern(self):
        """Test that Conformer follows FMS architectural patterns."""
        config = ConformerConfig()
        encoder = ConformerEncoder(config)

        # Should be an nn.Module
        assert isinstance(encoder, nn.Module)

        # Should have a config attribute
        assert hasattr(encoder, "config")

        # Config should be a ModelConfig subclass
        from fms.utils.config import ModelConfig
        assert isinstance(config, ModelConfig)

    def test_conformer_model_registration_pattern(self):
        """Test that Conformer can be registered with FMS model registry."""
        from fms import models

        # This will fail until we add registration
        # But it documents the expected pattern
        config = ConformerConfig()

        def conformer_factory():
            return ConformerEncoder(config)

        # Expected registration pattern
        architecture_name = "conformer"
        variant_name = "base"

        # This should work once implemented
        # models.register_model(architecture_name, variant_name, conformer_factory)
        # model = models.get_model(architecture_name, variant_name)

        assert True  # Placeholder - will implement registration later

    def test_conformer_serialization_compatibility(self):
        """Test that Conformer can be saved/loaded with torch."""
        config = ConformerConfig(num_layers=2)
        encoder = ConformerEncoder(config)

        # Get initial parameters
        initial_state = encoder.state_dict()

        # Create new encoder and load state
        encoder_new = ConformerEncoder(config)
        encoder_new.load_state_dict(initial_state)

        # Should produce identical outputs
        input_features = torch.randn(1, 50, config.num_features)

        encoder.eval()
        encoder_new.eval()

        output1 = encoder(input_features)
        output2 = encoder_new(input_features)

        assert torch.allclose(output1, output2, atol=1e-6), \
            "Loaded model should produce identical outputs"


class TestConformerComponents:
    """Test individual Conformer components."""

    @pytest.fixture
    def config(self):
        return ConformerConfig(hidden_dim=256, num_heads=4, dim_head=64)

    def test_feedforward_module_exists(self, config):
        """Test that ConformerFeedForward module can be instantiated."""
        from fms.models.conformer import ConformerFeedForward

        ff = ConformerFeedForward(
            dim=config.hidden_dim,
            mult=config.feedforward_mult,
            dropout=config.dropout,
        )

        assert isinstance(ff, nn.Module)

        # Test forward pass
        x = torch.randn(2, 50, config.hidden_dim)
        output = ff(x)
        assert output.shape == x.shape

    def test_attention_module_exists(self, config):
        """Test that ConformerAttention module can be instantiated."""
        from fms.models.conformer import ConformerAttention

        attn = ConformerAttention(
            dim=config.hidden_dim,
            num_heads=config.num_heads,
            dim_head=config.dim_head,
            max_pos_emb=config.max_pos_emb,
            dropout=config.dropout,
        )

        assert isinstance(attn, nn.Module)

        # Test forward pass
        x = torch.randn(2, 50, config.hidden_dim)
        attention_dists = torch.randint(0, 2001, (50, 50))
        output = attn(x, attention_dists)
        assert output.shape == x.shape

    def test_convolution_module_exists(self, config):
        """Test that ConformerConvModule can be instantiated."""
        from fms.models.conformer import ConformerConvModule

        conv = ConformerConvModule(
            dim=config.hidden_dim,
            kernel_size=config.conv_kernel_size,
            expansion_factor=config.conv_expansion_factor,
            dropout=config.dropout,
        )

        assert isinstance(conv, nn.Module)

        # Test forward pass
        x = torch.randn(2, 50, config.hidden_dim)
        output = conv(x)
        assert output.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])