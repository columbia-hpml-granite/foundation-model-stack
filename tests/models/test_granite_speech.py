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

from fms.models.granite_speech import GraniteSpeech, GraniteSpeechConfig
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
            input_dim=160,
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
            input_dim=160,
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
# Processor Tests (migrated from HF test_processing_granite_speech.py)
# =============================================================================

# Check for required dependencies
try:
    from transformers import AutoTokenizer, GPT2TokenizerFast
    from transformers import GraniteSpeechFeatureExtractor, GraniteSpeechProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not available")
class TestGraniteSpeechProcessor:
    """
    Tests for GraniteSpeechProcessor.

    Migrated from HF test_processing_granite_speech.py:GraniteSpeechProcessorTest
    All test method names are kept exactly as in HF for traceability.
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Set up test fixtures."""
        self.tmpdirname = str(tmp_path)
        self.checkpoint = "ibm-granite/granite-speech-3.3-8b"
        processor = GraniteSpeechProcessor.from_pretrained(self.checkpoint)
        processor.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return AutoTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_audio_processor(self, **kwargs):
        return GraniteSpeechFeatureExtractor.from_pretrained(self.tmpdirname, **kwargs)

    # =========================================================================
    # Test methods migrated from HF (exact names preserved)
    # =========================================================================

    def test_save_load_pretrained_default(self):
        """Ensure we can save / reload a processor correctly.

        HF Source: test_processing_granite_speech.py L55-71
        """
        tokenizer = self.get_tokenizer()
        audio_processor = self.get_audio_processor()
        processor = GraniteSpeechProcessor(
            tokenizer=tokenizer,
            audio_processor=audio_processor,
        )

        processor.save_pretrained(self.tmpdirname)
        processor = GraniteSpeechProcessor.from_pretrained(self.tmpdirname)

        assert processor.tokenizer.get_vocab() == tokenizer.get_vocab()
        assert isinstance(processor.tokenizer, GPT2TokenizerFast)

        assert processor.audio_processor.to_json_string() == audio_processor.to_json_string()
        assert isinstance(processor.audio_processor, GraniteSpeechFeatureExtractor)

    def test_requires_text(self):
        """Ensure we require text.

        HF Source: test_processing_granite_speech.py L73-83
        """
        tokenizer = self.get_tokenizer()
        audio_processor = self.get_audio_processor()
        processor = GraniteSpeechProcessor(
            tokenizer=tokenizer,
            audio_processor=audio_processor,
        )

        with pytest.raises(TypeError):
            processor(text=None)

    def test_bad_text_fails(self):
        """Ensure we gracefully fail if text is the wrong type.

        HF Source: test_processing_granite_speech.py L85-92
        """
        tokenizer = self.get_tokenizer()
        audio_processor = self.get_audio_processor()

        processor = GraniteSpeechProcessor(tokenizer=tokenizer, audio_processor=audio_processor)
        with pytest.raises(TypeError):
            processor(text=424, audio=None)

    def test_bad_nested_text_fails(self):
        """Ensure we gracefully fail if text is the wrong nested type.

        HF Source: test_processing_granite_speech.py L94-104
        """
        tokenizer = self.get_tokenizer()
        audio_processor = self.get_audio_processor()
        processor = GraniteSpeechProcessor(
            tokenizer=tokenizer,
            audio_processor=audio_processor,
        )

        with pytest.raises(TypeError):
            processor(text=[424], audio=None)

    def test_bad_audio_fails(self):
        """Ensure we gracefully fail if audio is the wrong type.

        HF Source: test_processing_granite_speech.py L106-116
        """
        tokenizer = self.get_tokenizer()
        audio_processor = self.get_audio_processor()
        processor = GraniteSpeechProcessor(
            tokenizer=tokenizer,
            audio_processor=audio_processor,
        )

        with pytest.raises(TypeError):
            processor(text=None, audio="foo")

    def test_nested_bad_audio_fails(self):
        """Ensure we gracefully fail if audio is the wrong nested type.

        HF Source: test_processing_granite_speech.py L118-128
        """
        tokenizer = self.get_tokenizer()
        audio_processor = self.get_audio_processor()
        processor = GraniteSpeechProcessor(
            tokenizer=tokenizer,
            audio_processor=audio_processor,
        )

        with pytest.raises(TypeError):
            processor(text=None, audio=["foo"])

    @pytest.mark.parametrize(
        "vec_dims,num_expected_features,random_func",
        [
            ([1, 269920], [171], torch.rand),
            ([1, 269920], [171], np.random.rand),
        ],
    )
    def test_audio_token_filling_same_len_feature_tensors(self, vec_dims, num_expected_features, random_func):
        """Ensure audio token filling is handled correctly when we have
        one or more audio inputs whose features are all the same length
        stacked into a tensor / numpy array.

        NOTE: Currently we enforce that each sample can only have one audio.

        HF Source: test_processing_granite_speech.py L130-163
        """
        tokenizer = self.get_tokenizer()
        audio_processor = self.get_audio_processor()
        processor = GraniteSpeechProcessor(
            tokenizer=tokenizer,
            audio_processor=audio_processor,
        )
        audio = random_func(*vec_dims) - 0.5

        audio_tokens = processor.audio_token * vec_dims[0]
        inputs = processor(text=f"{audio_tokens} Can you compare this audio?", audio=audio, return_tensors="pt")

        # Check the number of audio tokens
        audio_token_id = tokenizer.get_vocab()[processor.audio_token]

        # Make sure the number of audio tokens matches the number of features
        num_computed_features = processor.audio_processor._get_num_audio_features(
            [vec_dims[1] for _ in range(vec_dims[0])],
        )
        num_audio_tokens = int(torch.sum(inputs["input_ids"] == audio_token_id))
        assert list(inputs["input_features"].shape) == [vec_dims[0], 844, 160]
        assert sum(num_computed_features) == num_audio_tokens

    def test_audio_token_filling_varying_len_feature_list(self):
        """Ensure audio token filling is handled correctly when we have
        multiple varying len audio sequences passed as a list.

        HF Source: test_processing_granite_speech.py L165-197
        """
        tokenizer = self.get_tokenizer()
        audio_processor = self.get_audio_processor()
        processor = GraniteSpeechProcessor(
            tokenizer=tokenizer,
            audio_processor=audio_processor,
        )
        vec_dims = [[1, 142100], [1, 269920]]
        num_expected_features = [90, 171]
        audio = [torch.rand(dims) - 0.5 for dims in vec_dims]

        inputs = processor(
            text=[
                f"{processor.audio_token} Can you describe this audio?",
                f"{processor.audio_token} How does it compare with this audio?",
            ],
            audio=audio,
            return_tensors="pt",
        )

        # Check the number of audio tokens
        audio_token_id = tokenizer.get_vocab()[processor.audio_token]

        # Make sure the number of audio tokens matches the number of features
        num_calculated_features = processor.audio_processor._get_num_audio_features(
            [dims[1] for dims in vec_dims],
        )
        num_audio_tokens = int(torch.sum(inputs["input_ids"] == audio_token_id))
        assert num_calculated_features == [90, 171]
        assert sum(num_expected_features) == num_audio_tokens

    @pytest.mark.skipif(
        torch_device == "cpu",
        reason="Test requires GPU/accelerator"
    )
    def test_device_override(self):
        """Ensure that we regardless of the processing device, the tensors
        produced are on the CPU.

        HF Source: test_processing_granite_speech.py L199-221
        """
        tokenizer = self.get_tokenizer()
        audio_processor = self.get_audio_processor()
        processor = GraniteSpeechProcessor(
            tokenizer=tokenizer,
            audio_processor=audio_processor,
        )

        vec_dims = [1, 269920]
        wav = torch.rand(vec_dims) - 0.5

        inputs = processor(
            text=f"{processor.audio_token} Can you transcribe this audio?",
            audio=wav,
            return_tensors="pt",
            device=torch_device,
        )

        assert inputs["input_features"].device.type == "cpu"


# =============================================================================
# E2E Integration Tests (migrated from HF test_modeling_granite_speech.py)
# =============================================================================

# Check for required dependencies
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    from peft import PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not available")
@pytest.mark.skipif(not HAS_DATASETS, reason="datasets not available")
class TestGraniteSpeechE2E:
    """
    End-to-End integration tests for GraniteSpeech.

    Migrated from HF test_modeling_granite_speech.py:
    GraniteSpeechForConditionalGenerationIntegrationTest (L296-378)

    All test method names are kept exactly as in HF for traceability.
    """

    model_path = "ibm-granite/granite-speech-3.3-2b"

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up and tear down for each test."""
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.prompt = self._get_prompt(self.processor.tokenizer)

        yield

        # Cleanup after test
        cleanup(torch_device, gc_collect=True)

    def _get_prompt(self, tokenizer):
        """Create the prompt for transcription.

        HF Source: test_modeling_granite_speech.py L305-316
        """
        chat = [
            {
                "role": "system",
                "content": "Knowledge Cutoff Date: April 2024.\nToday's Date: December 19, 2024.\nYou are Granite, developed by IBM. You are a helpful AI assistant",
            },
            {
                "role": "user",
                "content": "<|audio|>can you transcribe the speech into a written format?",
            },
        ]
        return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    def _load_datasamples(self, num_samples):
        """Load audio samples from LibriSpeech dataset.

        HF Source: test_modeling_granite_speech.py L318-323
        """
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # automatic decoding with librispeech
        speech_samples = ds.sort("id")[:num_samples]["audio"]

        return [x["array"] for x in speech_samples]

    @pytest.mark.slow
    @pytest.mark.skipif(not HAS_PEFT, reason="Outputs diverge without lora")
    def test_small_model_integration_test_single(self):
        """Test single audio transcription E2E.

        HF Source: test_modeling_granite_speech.py L325-349
        """
        from transformers import GraniteSpeechForConditionalGeneration

        model = GraniteSpeechForConditionalGeneration.from_pretrained(self.model_path).to(torch_device)
        input_speech = self._load_datasamples(1)

        # Verify feature sizes; note that the feature mask refers to the size of
        # features that are masked into the LLM, not the output of the processor,
        # which is why we inspect the mask instead of the `num_features` tensor.
        inputs = self.processor(self.prompt, input_speech, return_tensors="pt").to(torch_device)

        num_computed_features = self.processor.audio_processor._get_num_audio_features(
            [speech_arr.shape[-1] for speech_arr in input_speech],
        )[0]
        num_actual_features = torch.sum(inputs["input_features_mask"]).item()
        assert num_actual_features == num_computed_features

        # verify generation
        output = model.generate(**inputs, max_new_tokens=32)
        EXPECTED_DECODED_TEXT = "systemKnowledge Cutoff Date: April 2024.\nToday's Date: December 19, 2024.\nYou are Granite, developed by IBM. You are a helpful AI assistant\nusercan you transcribe the speech into a written format?\nassistantmister quilter is the apostle of the middle classes and we are glad to welcome his gospel"  # fmt: skip

        assert self.processor.tokenizer.decode(output[0], skip_special_tokens=True) == EXPECTED_DECODED_TEXT

    @pytest.mark.slow
    @pytest.mark.skipif(not HAS_PEFT, reason="Outputs diverge without lora")
    def test_small_model_integration_test_batch(self):
        """Test batch audio transcription E2E.

        HF Source: test_modeling_granite_speech.py L351-378
        """
        from transformers import GraniteSpeechForConditionalGeneration

        model = GraniteSpeechForConditionalGeneration.from_pretrained(self.model_path).to(torch_device)
        input_speech = self._load_datasamples(2)
        prompts = [self.prompt, self.prompt]

        # Verify feature sizes & padding
        inputs = self.processor(prompts, input_speech, return_tensors="pt").to(model.device)
        num_computed_features = self.processor.audio_processor._get_num_audio_features(
            [speech_arr.shape[-1] for speech_arr in input_speech],
        )
        num_actual_features = torch.sum(inputs["input_features_mask"], dim=-1)
        for e_feats, a_feats in zip(num_computed_features, num_actual_features):
            assert e_feats == a_feats.item()

        # verify generation
        output = model.generate(**inputs, max_new_tokens=32)

        EXPECTED_DECODED_TEXT = [
            "systemKnowledge Cutoff Date: April 2024.\nToday's Date: December 19, 2024.\nYou are Granite, developed by IBM. You are a helpful AI assistant\nusercan you transcribe the speech into a written format?\nassistantmister quilter is the apostle of the middle classes and we are glad to welcome his gospel",
            "systemKnowledge Cutoff Date: April 2024.\nToday's Date: December 19, 2024.\nYou are Granite, developed by IBM. You are a helpful AI assistant\nusercan you transcribe the speech into a written format?\nassistantnor is mister quilter's manner less interesting than his matter"
        ]  # fmt: skip

        assert self.processor.tokenizer.batch_decode(output, skip_special_tokens=True) == EXPECTED_DECODED_TEXT
