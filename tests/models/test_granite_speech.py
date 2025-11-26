"""
End-to-End Tests for Granite Speech Model.

This test suite validates the complete Granite Speech implementation:
1. Model instantiation and configuration
2. Forward pass with audio and text inputs
3. Gradient flow through all components
4. Weight conversion from HuggingFace to FMS format
5. Integration of encoder, projector, and decoder

Reference: HuggingFace granite_speech/modeling_granite_speech.py
"""

import math
import pytest
import torch
import torch.nn as nn

from fms.models.granite_speech import (
    GraniteSpeech,
    GraniteSpeechConfig,
    _hf_to_fms_names,
    _split_kv_weights,
)
from fms.models.conformer import ConformerConfig
from fms.models.granite import GraniteConfig
from fms.modules.projector import SpeechProjectorConfig


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def small_config():
    """Small config for fast testing."""
    encoder = ConformerConfig(
        num_features=80,
        hidden_dim=128,
        num_layers=2,
        num_heads=2,
        dim_head=64,
        conv_kernel_size=3,
        dropout=0.0,
        max_pos_emb=128,
        context_size=50,
        output_dim=42,
        use_ctc=True,
    )

    projector = SpeechProjectorConfig(
        encoder_dim=128,
        decoder_dim=256,
        num_queries=3,
        window_size=15,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=256,
        hidden_dropout_prob=0.0,
        attention_dropout_prob=0.0,
    )

    decoder = GraniteConfig(
        src_vocab_size=1000,
        emb_dim=256,
        norm_eps=1e-5,
        nheads=4,
        head_dim=64,
        kvheads=2,
        nlayers=2,
        hidden_grow_factor=2.0,
        max_expected_seq_len=512,
        rope_theta=10000.0,
        pad_id=0,
        p_dropout=0.0,
    )

    return GraniteSpeechConfig(
        encoder_config=encoder,
        projector_config=projector,
        decoder_config=decoder,
        audio_token_index=999,
    )


# ============================================================================
# Configuration Tests
# ============================================================================


class TestGraniteSpeechConfig:
    """Test GraniteSpeechConfig creation and validation."""

    def test_default_config(self):
        """Test default config is created correctly."""
        config = GraniteSpeechConfig()
        assert config.encoder_config is not None
        assert config.projector_config is not None
        assert config.decoder_config is not None
        assert config.audio_token_index == 49155  # Default

    def test_small_config(self, small_config):
        """Test small config fixture."""
        assert small_config.encoder_config.hidden_dim == 128
        assert small_config.projector_config.decoder_dim == 256
        assert small_config.decoder_config.emb_dim == 256

    def test_config_nesting(self, small_config):
        """Test nested configs are properly accessible."""
        assert hasattr(small_config.encoder_config, 'num_layers')
        assert hasattr(small_config.projector_config, 'num_queries')
        assert hasattr(small_config.decoder_config, 'nlayers')


# ============================================================================
# Model Instantiation Tests
# ============================================================================


class TestModelInstantiation:
    """Test GraniteSpeech model creation."""

    def test_model_creation(self, small_config):
        """Test model can be instantiated."""
        model = GraniteSpeech(small_config)
        assert model is not None
        assert model.encoder is not None
        assert model.projector is not None
        assert model.decoder is not None
        assert model.lm_head is not None

    def test_model_from_config(self, small_config):
        """Test model creation from config class method."""
        model = GraniteSpeech.from_config(small_config)
        assert model is not None

    def test_get_config(self, small_config):
        """Test get_config returns config."""
        model = GraniteSpeech(small_config)
        assert model.get_config() == small_config

    def test_embedding_access(self, small_config):
        """Test embeddings are accessible."""
        model = GraniteSpeech(small_config)
        assert model.get_input_embeddings() is not None
        assert model.get_output_embeddings() is not None


# ============================================================================
# Forward Pass Tests
# ============================================================================


class TestForwardPass:
    """Test forward pass through the model."""

    def test_audio_features_extraction(self, small_config):
        """Test get_audio_features produces correct output shape."""
        model = GraniteSpeech(small_config)
        model.eval()

        batch_size = 2
        audio_len = 100  # Will produce 7 windows * 3 queries = 21 tokens

        audio = torch.randn(batch_size, audio_len, 80)

        with torch.no_grad():
            features = model.get_audio_features(audio)

        # Expected: 7 windows * 3 queries = 21
        expected_queries = math.ceil(audio_len / 15) * 3
        assert features.shape == (batch_size, expected_queries, 256)

    def test_text_only_forward(self, small_config):
        """Test text-only forward pass."""
        model = GraniteSpeech(small_config)
        model.eval()

        batch_size = 2
        seq_len = 20
        input_ids = torch.randint(0, 998, (batch_size, seq_len))

        with torch.no_grad():
            logits, loss = model(input_ids=input_ids)

        assert logits.shape == (batch_size, seq_len, 1000)
        assert loss is None

    def test_multimodal_forward(self, small_config):
        """Test multimodal forward with audio + text."""
        model = GraniteSpeech(small_config)
        model.eval()

        batch_size = 2
        audio_len = 100  # 21 audio tokens
        text_len = 30
        total_len = 21 + text_len

        audio = torch.randn(batch_size, audio_len, 80)
        input_ids = torch.randint(0, 998, (batch_size, total_len))
        input_ids[:, :21] = 999  # Audio token placeholders

        with torch.no_grad():
            logits, loss = model(
                input_ids=input_ids,
                input_features=audio,
            )

        assert logits.shape == (batch_size, total_len, 1000)
        assert loss is None

    def test_forward_with_labels(self, small_config):
        """Test forward pass computes loss with labels."""
        model = GraniteSpeech(small_config)

        batch_size = 2
        audio_len = 100
        total_len = 51

        audio = torch.randn(batch_size, audio_len, 80)
        input_ids = torch.randint(0, 998, (batch_size, total_len))
        input_ids[:, :21] = 999
        labels = torch.randint(0, 998, (batch_size, total_len))

        logits, loss = model(
            input_ids=input_ids,
            input_features=audio,
            labels=labels,
        )

        assert logits.shape == (batch_size, total_len, 1000)
        assert loss is not None
        assert loss.item() > 0

    def test_forward_with_cache(self, small_config):
        """Test forward pass with KV caching."""
        model = GraniteSpeech(small_config)
        model.eval()

        batch_size = 2
        input_ids = torch.randint(0, 998, (batch_size, 10))

        with torch.no_grad():
            logits, loss, cache = model(
                input_ids=input_ids,
                use_cache=True,
            )

        assert logits.shape == (batch_size, 10, 1000)
        assert cache is not None
        assert len(cache) == small_config.decoder_config.nlayers


# ============================================================================
# Gradient Flow Tests
# ============================================================================


class TestGradientFlow:
    """Test gradient flow through the model."""

    def test_gradient_flow_all_components(self, small_config):
        """Test gradients flow through encoder, projector, decoder, and lm_head."""
        model = GraniteSpeech(small_config)

        batch_size = 2
        audio_len = 100
        total_len = 51

        audio = torch.randn(batch_size, audio_len, 80)
        input_ids = torch.randint(0, 998, (batch_size, total_len))
        input_ids[:, :21] = 999
        labels = torch.randint(0, 998, (batch_size, total_len))

        logits, loss = model(
            input_ids=input_ids,
            input_features=audio,
            labels=labels,
        )

        loss.backward()

        # Check gradients on all components
        encoder_has_grad = any(p.grad is not None for p in model.encoder.parameters())
        projector_has_grad = any(p.grad is not None for p in model.projector.parameters())
        decoder_has_grad = any(p.grad is not None for p in model.decoder.parameters())
        lm_head_has_grad = model.lm_head.weight.grad is not None

        assert encoder_has_grad, "Encoder should have gradients"
        assert projector_has_grad, "Projector should have gradients"
        assert decoder_has_grad, "Decoder should have gradients"
        assert lm_head_has_grad, "LM head should have gradients"

    def test_gradient_flow_text_only(self, small_config):
        """Test gradients flow in text-only mode."""
        model = GraniteSpeech(small_config)

        input_ids = torch.randint(0, 998, (2, 20))
        labels = torch.randint(0, 998, (2, 20))

        logits, loss = model(input_ids=input_ids, labels=labels)
        loss.backward()

        # Decoder and lm_head should have gradients
        decoder_has_grad = any(p.grad is not None for p in model.decoder.parameters())
        lm_head_has_grad = model.lm_head.weight.grad is not None

        assert decoder_has_grad
        assert lm_head_has_grad


# ============================================================================
# Freeze Tests
# ============================================================================


class TestFreeze:
    """Test freezing components."""

    def test_freeze_encoder(self, small_config):
        """Test encoder can be frozen."""
        small_config.freeze_encoder = True
        model = GraniteSpeech(small_config)

        for param in model.encoder.parameters():
            assert not param.requires_grad

    def test_freeze_decoder(self, small_config):
        """Test decoder can be frozen."""
        small_config.freeze_decoder = True
        model = GraniteSpeech(small_config)

        for param in model.decoder.parameters():
            assert not param.requires_grad
        for param in model.lm_head.parameters():
            assert not param.requires_grad


# ============================================================================
# Weight Conversion Tests
# ============================================================================


class TestWeightConversion:
    """Test HuggingFace to FMS weight conversion."""

    def test_encoder_name_conversion(self):
        """Test encoder weight names are converted correctly."""
        test_sd = {
            "encoder.input_linear.weight": torch.randn(128, 80),
            "encoder.input_linear.bias": torch.randn(128),
            "encoder.layers.0.ff1.pre_norm.weight": torch.randn(128),
            "encoder.layers.0.ff1.up_proj.weight": torch.randn(512, 128),
            "encoder.layers.0.ff1.down_proj.weight": torch.randn(128, 512),
            "encoder.layers.0.attn.pre_norm.weight": torch.randn(128),
            "encoder.layers.0.attn.to_q.weight": torch.randn(128, 128),
            "encoder.layers.0.attn.to_kv.weight": torch.randn(256, 128),
            "encoder.layers.0.attn.rel_pos_emb.weight": torch.randn(1025, 64),
        }

        converted = _hf_to_fms_names(test_sd)

        assert "encoder.input_proj.weight" in converted
        assert "encoder.input_proj.bias" in converted
        assert "encoder.blocks.0.ff1.norm.weight" in converted
        assert "encoder.blocks.0.ff1.fc1.weight" in converted
        assert "encoder.blocks.0.ff1.fc2.weight" in converted
        assert "encoder.blocks.0.attn.norm.weight" in converted
        assert "encoder.blocks.0.attn.to_q.weight" in converted
        assert "encoder.blocks.0.attn.to_kv.weight" in converted
        assert "encoder.blocks.0.attn.pos_emb.weight" in converted

    def test_kv_split(self):
        """Test combined K-V weights are split correctly."""
        kv_weight = torch.randn(256, 128)  # Combined K and V
        test_sd = {
            "encoder.blocks.0.attn.to_kv.weight": kv_weight,
            "encoder.blocks.0.attn.to_q.weight": torch.randn(128, 128),
        }

        converted = _split_kv_weights(test_sd)

        assert "encoder.blocks.0.attn.to_k.weight" in converted
        assert "encoder.blocks.0.attn.to_v.weight" in converted
        assert "encoder.blocks.0.attn.to_kv.weight" not in converted

        # Check split is correct
        k = converted["encoder.blocks.0.attn.to_k.weight"]
        v = converted["encoder.blocks.0.attn.to_v.weight"]
        assert k.shape == (128, 128)
        assert v.shape == (128, 128)
        assert torch.allclose(torch.cat([k, v], dim=0), kv_weight)

    def test_projector_name_conversion(self):
        """Test projector weight names are converted correctly."""
        test_sd = {
            "projector.query": torch.randn(1, 3, 128),
            "projector.qformer.encoder.layer.0.attention.attention.query.weight": torch.randn(128, 128),
            "projector.qformer.encoder.layer.0.crossattention.attention.query.weight": torch.randn(128, 128),
            "projector.qformer.encoder.layer.0.intermediate_query.dense.weight": torch.randn(256, 128),
            "projector.qformer.encoder.layer.0.output_query.dense.weight": torch.randn(128, 256),
            "projector.linear.weight": torch.randn(256, 128),
        }

        converted = _hf_to_fms_names(test_sd)

        assert "projector.query_embeds" in converted
        assert "projector.layers.0.self_attention.query.weight" in converted
        assert "projector.layers.0.cross_attention.query.weight" in converted
        assert "projector.layers.0.feed_forward.dense_in.weight" in converted
        assert "projector.layers.0.feed_forward.dense_out.weight" in converted
        assert "projector.output_proj.weight" in converted

    def test_decoder_name_conversion(self):
        """Test decoder weight names are converted correctly."""
        test_sd = {
            "language_model.lm_head.weight": torch.randn(1000, 256),
            "language_model.model.embed_tokens.weight": torch.randn(1000, 256),
            "language_model.model.norm.weight": torch.randn(256),
            "language_model.model.layers.0.self_attn.q_proj.weight": torch.randn(256, 256),
            "language_model.model.layers.0.mlp.gate_proj.weight": torch.randn(512, 256),
        }

        converted = _hf_to_fms_names(test_sd)

        assert "lm_head.weight" in converted
        assert "decoder.embedding.weight" in converted
        assert "decoder.dec_norm.weight" in converted
        assert "decoder.layers.0.attn.in_proj.query.weight" in converted
        assert "decoder.layers.0.ff_sub_layer.wg.weight" in converted


# ============================================================================
# Numerical Stability Tests
# ============================================================================


class TestNumericalStability:
    """Test numerical stability of the model."""

    def test_no_nan_output(self, small_config):
        """Test output has no NaN values."""
        model = GraniteSpeech(small_config)
        model.eval()

        batch_size = 2
        audio = torch.randn(batch_size, 100, 80)
        input_ids = torch.randint(0, 998, (batch_size, 51))
        input_ids[:, :21] = 999

        with torch.no_grad():
            logits, _ = model(
                input_ids=input_ids,
                input_features=audio,
            )

        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_large_batch_stability(self, small_config):
        """Test stability with larger batch."""
        model = GraniteSpeech(small_config)
        model.eval()

        batch_size = 8
        audio = torch.randn(batch_size, 100, 80)
        input_ids = torch.randint(0, 998, (batch_size, 51))
        input_ids[:, :21] = 999

        with torch.no_grad():
            logits, _ = model(
                input_ids=input_ids,
                input_features=audio,
            )

        assert not torch.isnan(logits).any()


# ============================================================================
# Run tests if executed directly
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
