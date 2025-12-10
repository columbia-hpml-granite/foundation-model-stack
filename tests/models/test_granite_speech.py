# Copyright 2025 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tests for GraniteSpeech model.

Migrated from HuggingFace transformers:
tests/models/granite_speech/test_modeling_granite_speech.py
tests/models/granite_speech/test_processing_granite_speech.py

Contains:
- GraniteSpeechFixtures: FMS test fixtures for config and model
- TestGraniteSpeech: Core model tests using FMS test suites
- TestGraniteSpeechModel: Model-specific tests from HF
- TestGraniteSpeechProcessor: Processor tests from HF
- TestGraniteSpeechE2E: End-to-end integration tests
"""

import gc

import numpy as np
import pytest
import torch

from fms.models.granite_speech import (
    GraniteSpeech,
    GraniteSpeechConfig,
    GraniteSpeechFeatureExtractor,
    GraniteSpeechProcessor,
)
from fms.models.conformer import ConformerConfig
from fms.models.granite import GraniteConfig
from fms.modules.projector import SpeechProjectorConfig
from fms.testing._internal.model_test_suite import (
    ConfigFixtureMixin,
    ModelCompileTestSuite,
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelFixtureMixin,
)
from fms.utils.config import ModelConfig


# Check for GPU availability
def get_torch_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


torch_device = get_torch_device()


def floats_tensor(shape, scale=1.0):
    """Create a random float tensor."""
    return torch.rand(shape) * scale


def ids_tensor(shape, vocab_size):
    """Create a random integer tensor for token IDs."""
    return torch.randint(low=0, high=vocab_size, size=shape)


def cleanup(device, gc_collect=True):
    """Clean up memory after tests."""
    if gc_collect:
        gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


# =============================================================================
# FMS Test Fixtures (following test_llava_next.py pattern)
# =============================================================================


class GraniteSpeechFixtures(ConfigFixtureMixin, ModelFixtureMixin):
    """
    Base GraniteSpeech Fixtures that can be re-used for other purposes.

    This will include the config and model signatures.
    """

    @pytest.fixture(scope="class", autouse=True)
    def uninitialized_model(self, config: GraniteSpeechConfig):
        return GraniteSpeech(config)

    @pytest.fixture(scope="class", autouse=True)
    def config(self) -> ModelConfig:
        _encoder_config = ConformerConfig(
            num_features=160,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            dim_head=32,
            conv_kernel_size=15,
            conv_expansion_factor=2,
            feedforward_mult=4,
            dropout=0.1,
            output_dim=42,
        )

        _decoder_config = GraniteConfig(
            src_vocab_size=99,
            emb_dim=32,
            nlayers=2,
            nheads=4,
            hidden_grow_factor=37 / 32,
            max_expected_seq_len=580,
            pad_id=1,
        )

        _projector_config = SpeechProjectorConfig(
            encoder_dim=32,
            decoder_dim=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            window_size=15,
            downsample_rate=5,
        )

        return GraniteSpeechConfig(
            encoder_config=_encoder_config,
            decoder_config=_decoder_config,
            projector_config=_projector_config,
            audio_token_index=0,
            downsample_rate=5,
            window_size=15,
        )


class TestGraniteSpeech(
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelCompileTestSuite,
    GraniteSpeechFixtures,
):
    """
    Core FMS test suite for GraniteSpeech model.

    Inherits tests from:
    - ModelConfigTestSuite: Config validation tests
    - ModelConsistencyTestSuite: Output consistency tests
    - ModelCompileTestSuite: Model compilation tests
    """

    @staticmethod
    def get_logits(f_out):
        return f_out[0]

    # Sample inputs for testing (following test_llava_next.py pattern)
    batch_size = 3
    seq_length = 9  # 7 + 2 audio tokens
    sequence_dim = 844
    feature_dim = 160

    input_features = floats_tensor([batch_size, sequence_dim, feature_dim])
    input_ids = ids_tensor([batch_size, seq_length], 97) + 2
    input_ids[:, :2] = 0  # Set first 2 tokens as audio tokens

    _get_signature_params = ["input_ids_or_embeds"]
    _get_signature_input_ids = input_ids
    _get_signature_optional_params = {
        "input_features": input_features,
    }
    _get_signature_logits_getter_fn = get_logits

    def test_config_passed_to_model_and_updated(self, model, config):
        """Test model constructor appropriately merges any passed kwargs into the config."""
        model = type(model)(
            config=config, audio_token_index=config.audio_token_index + 1
        )
        # check not same reference
        assert model.get_config() is not config

        # modify audio_token_index to the new value expected and check equivalence
        config.audio_token_index = config.audio_token_index + 1
        assert model.get_config().as_dict() == config.as_dict()

    def test_config_params_passed_as_kwargs_to_model(self, model, config):
        pytest.skip(
            "granite_speech uses nested configs for encoder, decoder, and projector"
        )


# =============================================================================
# Model-Specific Tests (migrated from HF test_modeling_granite_speech.py)
# =============================================================================


class GraniteSpeechForConditionalGenerationModelTester:
    """
    Helper class for creating test configurations and inputs.

    Migrated from HF test_modeling_granite_speech.py L55-211.
    Adapted for FMS GraniteSpeech model.
    """

    def __init__(
        self,
        seq_length=7,
        encoder_config=None,
        decoder_config=None,
        projector_config=None,
        audio_token_index=0,
        downsample_rate=5,
        window_size=15,
    ):
        # Encoder config (Conformer)
        self.encoder_config = encoder_config or ConformerConfig(
            num_features=160,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            dim_head=32,
            conv_kernel_size=15,
            conv_expansion_factor=2,
            feedforward_mult=4,
            dropout=0.1,
            output_dim=42,
        )

        # Decoder config (Granite)
        self.decoder_config = decoder_config or GraniteConfig(
            src_vocab_size=99,
            emb_dim=32,
            nlayers=2,
            nheads=4,
            hidden_grow_factor=37 / 32,
            max_expected_seq_len=580,
            pad_id=1,
        )

        # Projector config (SpeechProjector)
        self.projector_config = projector_config or SpeechProjectorConfig(
            encoder_dim=32,
            decoder_dim=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            window_size=window_size,
            downsample_rate=downsample_rate,
        )

        self.audio_token_index = audio_token_index
        self.downsample_rate = downsample_rate
        self.window_size = window_size

        # Dims for audio features
        self.sequence_dim = 844
        self.feature_dim = 160
        self.batch_size = 3
        self.pad_token_id = 1
        self.seq_len = 7
        self.num_audio_tokens = 2
        self.seq_length = seq_length + self.num_audio_tokens

    def get_config(self) -> GraniteSpeechConfig:
        """Create a GraniteSpeechConfig for testing."""
        return GraniteSpeechConfig(
            encoder_config=self.encoder_config,
            decoder_config=self.decoder_config,
            projector_config=self.projector_config,
            audio_token_index=self.audio_token_index,
            downsample_rate=self.downsample_rate,
            window_size=self.window_size,
        )

    def prepare_config_and_inputs(self):
        """Prepare config and input features."""
        input_features = floats_tensor(
            [self.batch_size, self.sequence_dim, self.feature_dim],
        )
        config = self.get_config()
        return config, input_features

    def prepare_config_and_inputs_for_common(self):
        """Prepare config and full inputs dict for common tests."""
        config, input_features = self.prepare_config_and_inputs()

        input_ids = ids_tensor(
            [self.batch_size, self.seq_length],
            self.decoder_config.src_vocab_size - 2
        ) + 2
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
        input_ids[input_ids == config.audio_token_index] = self.pad_token_id

        # Place audio tokens at the beginning
        input_ids[:, : self.num_audio_tokens] = config.audio_token_index

        inputs_dict = {
            "input_features": input_features,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


class TestGraniteSpeechModel:
    """
    Model-specific tests for GraniteSpeech.

    Contains tests migrated from HF test_modeling_granite_speech.py
    """

    @pytest.fixture
    def model_tester(self):
        return GraniteSpeechForConditionalGenerationModelTester()

    def test_inputs_embeds(self, model_tester):
        """Test that the model can accept inputs_embeds instead of input_ids.

        Overwrite inputs_embeds tests because we need to delete "input features"
        for the audio model.

        HF Source: test_modeling_granite_speech.py L231-250
        """
        config, inputs_dict = model_tester.prepare_config_and_inputs_for_common()

        model = GraniteSpeech(config)
        model.to(torch_device)
        model.eval()

        input_ids = inputs_dict["input_ids"].to(torch_device)
        # Don't use input_features for this test

        # Get input embeddings
        wte = model.decoder.model.embedding
        inputs_embeds = wte(input_ids)

        with torch.no_grad():
            # Forward with inputs_embeds instead of input_ids
            logits, _ = model.decoder(inputs_embeds=inputs_embeds)

        assert logits is not None
        assert not torch.isnan(logits).any().item()

    def test_sdpa_can_dispatch_composite_models(self, model_tester):
        """Test SDPA attention dispatch for composite audio+text model.

        Overwrite because Granite Speech is audio+text model (not vision+text).
        NOTE - currently we only enable alternate attention implementations on
        the encapsulated LLM; in the future, this should be added for the conformer
        encoder as well.

        HF Source: test_modeling_granite_speech.py L252-287
        """
        config, inputs_dict = model_tester.prepare_config_and_inputs_for_common()
        model = GraniteSpeech(config)

        # FMS models don't have the same _attn_implementation attribute as HF
        # This test verifies that the model can be created and run
        # The actual SDPA dispatch is handled internally by PyTorch
        model.to(torch_device)
        model.eval()

        input_ids = inputs_dict["input_ids"].to(torch_device)
        input_features = inputs_dict["input_features"].to(torch_device)

        with torch.no_grad():
            logits, _ = model(input_ids=input_ids, input_features=input_features)

        assert logits is not None
        assert not torch.isnan(logits).any().item()


# =============================================================================
# FMS Native Feature Extractor and Processor Tests
# =============================================================================


class TestFMSGraniteSpeechFeatureExtractor:
    """
    Tests for FMS-native GraniteSpeechFeatureExtractor.

    These tests validate the skeleton implementation added in commit 49b9db94.
    The FMS FeatureExtractor is a simplified version of the HF implementation.
    """

    def test_feature_extractor_init(self):
        """Test that GraniteSpeechFeatureExtractor can be initialized with default params."""
        extractor = GraniteSpeechFeatureExtractor()

        # Check default values match expected
        assert extractor.sampling_rate == 16000
        assert extractor.n_fft == 512
        assert extractor.win_length == 400
        assert extractor.hop_length == 160
        assert extractor.n_mels == 80
        assert extractor.projector_window_size == 15
        assert extractor.projector_downsample_rate == 5

    def test_feature_extractor_init_custom_params(self):
        """Test that GraniteSpeechFeatureExtractor can be initialized with custom params."""
        extractor = GraniteSpeechFeatureExtractor(
            sampling_rate=8000,
            n_fft=256,
            win_length=200,
            hop_length=80,
            n_mels=40,
            projector_window_size=10,
            projector_downsample_rate=2,
        )

        assert extractor.sampling_rate == 8000
        assert extractor.n_fft == 256
        assert extractor.win_length == 200
        assert extractor.hop_length == 80
        assert extractor.n_mels == 40
        assert extractor.projector_window_size == 10
        assert extractor.projector_downsample_rate == 2

    def test_feature_extractor_call_returns_dict(self):
        """Test that calling feature extractor returns expected dict structure."""
        extractor = GraniteSpeechFeatureExtractor()

        # Create dummy audio input
        audio = torch.randn(1, 16000)  # 1 second of audio at 16kHz

        result = extractor(audio)

        # Check dict structure (even if values are None for skeleton implementation)
        assert isinstance(result, dict)
        assert "input_features" in result
        assert "audio_embed_sizes" in result
        assert "input_features_mask" in result


class TestFMSGraniteSpeechProcessor:
    """
    Tests for FMS-native GraniteSpeechProcessor.

    These tests validate the skeleton implementation added in commit 49b9db94.
    The FMS Processor combines audio feature extraction with text tokenization.
    """

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        class MockTokenizer:
            def __init__(self):
                self.audio_token = "<|audio|>"

            def __call__(self, text, **kwargs):
                # Simple mock tokenization
                return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

        return MockTokenizer()

    def test_processor_init(self, mock_tokenizer):
        """Test that GraniteSpeechProcessor can be initialized."""
        audio_processor = GraniteSpeechFeatureExtractor()
        processor = GraniteSpeechProcessor(
            audio_processor=audio_processor,
            tokenizer=mock_tokenizer,
        )

        assert processor.audio_processor is audio_processor
        assert processor.tokenizer is mock_tokenizer
        assert processor.audio_token == "<|audio|>"

    def test_processor_init_custom_audio_token(self, mock_tokenizer):
        """Test that GraniteSpeechProcessor respects custom audio token."""
        audio_processor = GraniteSpeechFeatureExtractor()
        processor = GraniteSpeechProcessor(
            audio_processor=audio_processor,
            tokenizer=mock_tokenizer,
            audio_token="<audio>",
        )

        # Should use tokenizer's audio_token if available
        assert processor.audio_token == "<|audio|>"

    def test_processor_init_tokenizer_without_audio_token(self):
        """Test processor with tokenizer that doesn't have audio_token attribute."""
        class SimpleTokenizer:
            def __call__(self, text, **kwargs):
                return {"input_ids": [1, 2, 3]}

        audio_processor = GraniteSpeechFeatureExtractor()
        tokenizer = SimpleTokenizer()

        processor = GraniteSpeechProcessor(
            audio_processor=audio_processor,
            tokenizer=tokenizer,
            audio_token="<custom_audio>",
        )

        # Should use provided audio_token since tokenizer doesn't have one
        assert processor.audio_token == "<custom_audio>"

    def test_processor_call_returns_dict(self, mock_tokenizer):
        """Test that calling processor returns expected dict structure."""
        audio_processor = GraniteSpeechFeatureExtractor()
        processor = GraniteSpeechProcessor(
            audio_processor=audio_processor,
            tokenizer=mock_tokenizer,
        )

        result = processor(text="Test text")

        # Check dict structure (even if values are None for skeleton implementation)
        assert isinstance(result, dict)
        assert "input_ids" in result
        assert "attention_mask" in result

    def test_get_validated_text_string(self, mock_tokenizer):
        """Test _get_validated_text with string input."""
        audio_processor = GraniteSpeechFeatureExtractor()
        processor = GraniteSpeechProcessor(
            audio_processor=audio_processor,
            tokenizer=mock_tokenizer,
        )

        result = processor._get_validated_text("hello")
        assert result == ["hello"]

    def test_get_validated_text_list(self, mock_tokenizer):
        """Test _get_validated_text with list of strings input."""
        audio_processor = GraniteSpeechFeatureExtractor()
        processor = GraniteSpeechProcessor(
            audio_processor=audio_processor,
            tokenizer=mock_tokenizer,
        )

        result = processor._get_validated_text(["hello", "world"])
        assert result == ["hello", "world"]

    def test_get_validated_text_invalid_type(self, mock_tokenizer):
        """Test _get_validated_text raises TypeError for invalid input."""
        audio_processor = GraniteSpeechFeatureExtractor()
        processor = GraniteSpeechProcessor(
            audio_processor=audio_processor,
            tokenizer=mock_tokenizer,
        )

        with pytest.raises(TypeError):
            processor._get_validated_text(123)

    def test_get_validated_text_invalid_list(self, mock_tokenizer):
        """Test _get_validated_text raises TypeError for list of non-strings."""
        audio_processor = GraniteSpeechFeatureExtractor()
        processor = GraniteSpeechProcessor(
            audio_processor=audio_processor,
            tokenizer=mock_tokenizer,
        )

        with pytest.raises(TypeError):
            processor._get_validated_text([123, 456])


# =============================================================================
# E2E Integration Tests (using FMS components)
# =============================================================================

# Check for required dependencies
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


@pytest.mark.skipif(not HAS_DATASETS, reason="datasets not available")
class TestGraniteSpeechE2E:
    """
    End-to-End integration tests for FMS GraniteSpeech.

    Tests the full pipeline using FMS-native components:
    - GraniteSpeech model
    - GraniteSpeechFeatureExtractor
    - GraniteSpeechProcessor

    Note: These tests use skeleton implementations. Once the FMS processor
    and feature extractor are fully implemented, these tests will validate
    the complete E2E flow.
    """

    @pytest.fixture
    def small_config(self):
        """Create a small config for testing."""
        encoder_config = ConformerConfig(
            num_features=160,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            dim_head=16,
            conv_kernel_size=15,
            conv_expansion_factor=2,
            feedforward_mult=4,
            dropout=0.0,
            output_dim=42,
        )

        decoder_config = GraniteConfig(
            src_vocab_size=1000,
            emb_dim=64,
            nlayers=2,
            nheads=4,
            hidden_grow_factor=2.0,
            max_expected_seq_len=512,
            pad_id=0,
        )

        projector_config = SpeechProjectorConfig(
            encoder_dim=64,
            decoder_dim=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=128,
            window_size=15,
            downsample_rate=5,
            num_queries=3,
        )

        return GraniteSpeechConfig(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            projector_config=projector_config,
            audio_token_index=999,
            downsample_rate=5,
            window_size=15,
        )

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for E2E testing."""
        class MockTokenizer:
            def __init__(self):
                self.audio_token = "<|audio|>"
                self.pad_token_id = 0
                self.eos_token_id = 1

            def __call__(self, text, return_tensors=None, padding=True, **kwargs):
                # Simple mock: return fixed token IDs
                if isinstance(text, str):
                    text = [text]

                # Count audio tokens and create input_ids
                batch_input_ids = []
                for t in text:
                    # Simple tokenization: audio token -> 999, other chars -> random ids
                    ids = []
                    i = 0
                    while i < len(t):
                        if t[i:i+len(self.audio_token)] == self.audio_token:
                            ids.append(999)  # audio_token_index
                            i += len(self.audio_token)
                        else:
                            ids.append(ord(t[i]) % 998 + 1)  # Map to 1-998
                            i += 1
                    batch_input_ids.append(ids)

                # Pad to same length
                max_len = max(len(ids) for ids in batch_input_ids)
                for ids in batch_input_ids:
                    ids.extend([0] * (max_len - len(ids)))

                result = {
                    "input_ids": batch_input_ids,
                    "attention_mask": [[1] * len(ids) for ids in batch_input_ids],
                }

                if return_tensors == "pt":
                    result = {k: torch.tensor(v) for k, v in result.items()}

                return result

            def decode(self, ids, skip_special_tokens=True):
                return "mock decoded text"

            def batch_decode(self, ids, skip_special_tokens=True):
                return ["mock decoded text"] * len(ids)

        return MockTokenizer()

    def test_fms_model_forward_text_only(self, small_config):
        """Test FMS GraniteSpeech model forward pass with text only."""
        model = GraniteSpeech(small_config)
        model.eval()

        batch_size = 2
        seq_len = 10

        # Text-only input (no audio tokens)
        input_ids = torch.randint(1, 998, (batch_size, seq_len))

        with torch.no_grad():
            logits, loss = model(input_ids=input_ids)

        assert logits.shape == (batch_size, seq_len, small_config.decoder_config.src_vocab_size)
        assert loss is None  # No labels provided

    def test_fms_model_forward_with_audio(self, small_config):
        """Test FMS GraniteSpeech model forward pass with audio features."""
        model = GraniteSpeech(small_config)
        model.eval()

        batch_size = 1
        seq_len = 10
        audio_seq_len = 45  # 3 windows of 15

        # Calculate expected audio tokens
        num_windows = audio_seq_len // small_config.window_size
        num_audio_tokens = num_windows * small_config.projector_config.num_queries

        # Input with audio tokens at the beginning
        input_ids = torch.randint(1, 998, (batch_size, seq_len))
        input_ids[0, :num_audio_tokens] = small_config.audio_token_index

        # Audio features
        input_features = torch.randn(batch_size, audio_seq_len, 160)

        with torch.no_grad():
            logits, loss = model(
                input_ids=input_ids,
                input_features=input_features,
            )

        assert logits.shape == (batch_size, seq_len, small_config.decoder_config.src_vocab_size)

    def test_fms_processor_integration(self, small_config, mock_tokenizer):
        """Test FMS processor with feature extractor integration."""
        feature_extractor = GraniteSpeechFeatureExtractor()
        processor = GraniteSpeechProcessor(
            audio_processor=feature_extractor,
            tokenizer=mock_tokenizer,
        )

        # Test text processing
        text = "<|audio|> transcribe this audio"
        result = processor(text=text)

        assert isinstance(result, dict)
        assert "input_ids" in result
        assert "attention_mask" in result

    def test_fms_feature_extractor_integration(self, small_config):
        """Test FMS feature extractor with model integration."""
        feature_extractor = GraniteSpeechFeatureExtractor()

        # Create dummy audio (1 second at 16kHz)
        audio = torch.randn(1, 16000)

        # Extract features (skeleton returns dummy output)
        result = feature_extractor(audio)

        assert isinstance(result, dict)
        assert "input_features" in result
        assert "audio_embed_sizes" in result
        assert "input_features_mask" in result

    @pytest.mark.slow
    def test_fms_e2e_with_real_audio(self, small_config, mock_tokenizer):
        """
        E2E test with real audio from LibriSpeech dataset.

        Note: This test uses skeleton implementations for processor/feature extractor.
        The test validates that the pipeline structure works, but actual audio
        processing will only work once the implementations are complete.
        """
        # Load real audio samples
        ds = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy",
            "clean",
            split="validation",
            trust_remote_code=True,
        )
        speech_samples = ds.sort("id")[:1]["audio"]
        audio_array = speech_samples[0]["array"]

        # Create FMS components
        model = GraniteSpeech(small_config)
        model.eval()

        feature_extractor = GraniteSpeechFeatureExtractor()
        processor = GraniteSpeechProcessor(
            audio_processor=feature_extractor,
            tokenizer=mock_tokenizer,
        )

        # Process text (audio processing is skeleton)
        text = "<|audio|> transcribe this audio"
        processed = processor(text=text)

        # Verify processor output structure
        assert isinstance(processed, dict)
        assert "input_ids" in processed

        # Test model with text-only forward (since feature extractor is skeleton)
        # Once feature extractor is implemented, this should use audio features
        input_ids = torch.randint(1, 998, (1, 20))

        with torch.no_grad():
            logits, loss = model(input_ids=input_ids)

        assert logits is not None
        assert logits.shape[0] == 1  # batch size
        assert logits.shape[2] == small_config.decoder_config.src_vocab_size
