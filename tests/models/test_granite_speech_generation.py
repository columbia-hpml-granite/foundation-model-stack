"""
Tests for prepare_inputs_for_generation() hook in GraniteSpeech.

This test suite validates the generation hook functionality:
1. Direct hook invocation (iteration 0 with audio)
2. Direct hook invocation (iteration > 0 with cache)
3. Full generation with the hook
4. Generation without audio (text-only)
"""

import pytest
import torch

from fms.models.granite_speech import GraniteSpeech, GraniteSpeechConfig
from fms.models.conformer import ConformerConfig
from fms.models.granite import GraniteConfig
from fms.modules.projector import SpeechProjectorConfig
from fms.utils.generation import generate


@pytest.fixture
def small_config():
    """Small config for fast testing."""
    encoder = ConformerConfig(
        num_features=160,
        hidden_dim=128,
        num_layers=2,
        num_heads=2,
        dim_head=64,
        conv_kernel_size=3,
        dropout=0.0,
        max_pos_emb=128,
        context_size=50,
        output_dim=42,
        use_ctc=False,  # Disable CTC for simpler testing
    )

    projector = SpeechProjectorConfig(
        encoder_dim=128,
        decoder_dim=256,
        num_queries=3,
        window_size=15,
        downsample_rate=5,
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
        tie_heads=False,
        fused_weights=False,
    )

    return GraniteSpeechConfig(
        encoder_config=encoder,
        projector_config=projector,
        decoder_config=decoder,
        audio_token_index=999,  # Use 999 as audio token
    )


class TestPrepareInputsForGeneration:
    """Test prepare_inputs_for_generation() hook."""

    def test_hook_iteration_0_with_audio(self, small_config):
        """Test hook on first iteration (iteration=0) with audio."""
        model = GraniteSpeech(small_config)
        model.eval()

        # Create dummy inputs
        batch_size = 1
        audio_len = 30  # Short audio for testing
        seq_len = 20

        input_ids = torch.randint(0, 998, (batch_size, seq_len))
        # Place audio tokens at the beginning
        num_audio_tokens = (audio_len // small_config.window_size) * small_config.num_queries
        input_ids[0, :num_audio_tokens] = small_config.audio_token_index

        input_features = torch.randn(batch_size, audio_len, 160)
        input_features_mask = torch.ones(batch_size, audio_len, dtype=torch.bool)

        kwargs = {
            "use_cache": True,
            "input_features": input_features,
            "input_features_mask": input_features_mask,
        }

        # Call hook directly
        with torch.no_grad():
            returned_input_ids, returned_kwargs = model.prepare_inputs_for_generation(
                iteration=0,
                input_ids=input_ids,
                kwargs=kwargs.copy(),
            )

        # Verify hook behavior
        assert returned_input_ids is None, "Hook should return None for input_ids when audio is present"
        assert "inputs_embeds" in returned_kwargs, "Hook should add inputs_embeds to kwargs"
        assert "input_features" not in returned_kwargs, "Hook should remove input_features from kwargs"
        assert returned_kwargs["inputs_embeds"].shape[0] == batch_size
        assert returned_kwargs["inputs_embeds"].shape[1] == seq_len

        # Test that forward works with the hook output
        with torch.no_grad():
            logits, loss, cache = model(
                input_ids=returned_input_ids,
                use_cache=True,
                **returned_kwargs
            )

        assert logits.shape == (batch_size, seq_len, small_config.decoder_config.src_vocab_size)
        assert cache is not None

    def test_hook_iteration_1_with_cache(self, small_config):
        """Test hook on subsequent iteration (iteration > 0) with cache."""
        model = GraniteSpeech(small_config)
        model.eval()

        # Create dummy inputs
        batch_size = 1
        seq_len = 1  # Only one token for cached decoding

        input_ids = torch.randint(0, 998, (batch_size, seq_len))

        kwargs = {
            "use_cache": True,
            "past_key_value_states": None,  # Would be set by generation loop
        }

        # Call hook directly
        with torch.no_grad():
            returned_input_ids, returned_kwargs = model.prepare_inputs_for_generation(
                iteration=1,  # > 0, so should skip audio processing
                input_ids=input_ids,
                kwargs=kwargs.copy(),
            )

        # Verify hook behavior
        assert torch.equal(returned_input_ids, input_ids), "Hook should return input_ids unchanged for cached steps"
        assert "inputs_embeds" not in returned_kwargs, "Hook should not add inputs_embeds for cached steps"

    def test_generation_with_audio(self, small_config):
        """Test full generation with the hook and audio."""
        model = GraniteSpeech(small_config)
        model.eval()

        # Create dummy inputs
        batch_size = 1
        audio_len = 30
        seq_len = 10

        input_ids = torch.randint(0, 998, (batch_size, seq_len))
        # Place audio tokens
        num_audio_tokens = (audio_len // small_config.window_size) * small_config.num_queries
        input_ids[0, :num_audio_tokens] = small_config.audio_token_index

        input_features = torch.randn(batch_size, audio_len, 160)
        input_features_mask = torch.ones(batch_size, audio_len, dtype=torch.bool)

        # Test generation with hook
        with torch.no_grad():
            output_ids = generate(
                model,
                input_ids,
                max_new_tokens=5,
                use_cache=True,
                do_sample=False,  # Greedy decoding for deterministic results
                prepare_model_inputs_hook=model.prepare_inputs_for_generation,
                extra_kwargs={
                    "input_features": input_features,
                    "input_features_mask": input_features_mask,
                }
            )

            assert output_ids.shape == (batch_size, seq_len + 5), \
                f"Expected shape ({batch_size}, {seq_len + 5}), got {output_ids.shape}"

    def test_generation_without_audio(self, small_config):
        """Test generation without audio (text-only)."""
        model = GraniteSpeech(small_config)
        model.eval()

        # Create text-only inputs
        batch_size = 1
        seq_len = 10

        input_ids = torch.randint(0, 998, (batch_size, seq_len))

        # Test generation without audio
        with torch.no_grad():
            output_ids = generate(
                model,
                input_ids,
                max_new_tokens=5,
                use_cache=True,
                do_sample=False,
                prepare_model_inputs_hook=model.prepare_inputs_for_generation,
                # No extra_kwargs with audio
            )

            assert output_ids.shape == (batch_size, seq_len + 5), \
                f"Expected shape ({batch_size}, {seq_len + 5}), got {output_ids.shape}"

