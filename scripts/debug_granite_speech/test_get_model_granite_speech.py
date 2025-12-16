#!/usr/bin/env python
"""
Comprehensive get_model() test script for GraniteSpeech.

This script tests various input parameter combinations for get_model()
when loading the granite-speech model, including end-to-end inference.

Tests cover:
1. Architecture/variant combinations
2. Device types (cpu, cuda)
3. Data types (float32, float16, bfloat16)
4. Source loading (hf, hf_pretrained, hf_configured)
5. End-to-end audio-to-text inference

Usage:
    python test_get_model_granite_speech.py

Note: This is a standalone test script (not pytest) for flexibility in debugging.
"""

import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class TestResult:
    """Result of a single test case."""
    name: str
    passed: bool
    duration: float
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class TestRunner:
    """Simple test runner for get_model tests."""

    def __init__(self):
        self.results: List[TestResult] = []

    def run_test(self, name: str, test_fn, **kwargs):
        """Run a single test and record the result."""
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print('='*60)

        start_time = time.time()
        try:
            details = test_fn(**kwargs)
            duration = time.time() - start_time
            result = TestResult(
                name=name,
                passed=True,
                duration=duration,
                details=details
            )
            print(f"PASSED ({duration:.2f}s)")
        except Exception as e:
            duration = time.time() - start_time
            result = TestResult(
                name=name,
                passed=False,
                duration=duration,
                error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            )
            print(f"FAILED ({duration:.2f}s): {e}")

        self.results.append(result)
        return result

    def print_summary(self):
        """Print summary of all test results."""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)

        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total = len(self.results)

        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  [{status}] {r.name} ({r.duration:.2f}s)")
            if not r.passed and r.error:
                # Print first line of error
                first_line = r.error.split('\n')[0]
                print(f"         Error: {first_line}")

        print(f"\nTotal: {passed}/{total} passed, {failed} failed")
        return failed == 0


# =============================================================================
# Test Functions
# =============================================================================


def test_list_models_and_variants():
    """Test that granite_speech is registered with correct variants."""
    from fms.models import list_models, list_variants

    models = list_models()
    assert "granite_speech" in models, f"granite_speech not in {models}"

    variants = list_variants("granite_speech")
    expected_variants = ["3.3-8b", "3.2-8b", "3.3-2b"]
    for v in expected_variants:
        assert v in variants, f"Variant {v} not in {variants}"

    return {"models": models, "variants": variants}


def test_get_model_architecture_variant(variant: str = "3.3-2b"):
    """Test get_model with architecture + variant (random init, no weights)."""
    from fms.models import get_model
    from fms.models.granite_speech import GraniteSpeech

    model = get_model(
        architecture="granite_speech",
        variant=variant,
        device_type="cpu",
    )

    assert model is not None, "Model is None"
    assert isinstance(model, GraniteSpeech), f"Expected GraniteSpeech, got {type(model)}"

    # Check config
    config = model.get_config()
    return {
        "variant": variant,
        "audio_token_index": config.audio_token_index,
        "encoder_layers": config.encoder_config.num_layers,
        "decoder_layers": config.decoder_config.nlayers,
    }


def test_get_model_hf_configured():
    """Test get_model with hf_configured (downloads config only, random init)."""
    from fms.models import get_model
    from fms.models.granite_speech import GraniteSpeech

    model_id = "ibm-granite/granite-speech-3.3-2b"

    model = get_model(
        architecture="hf_configured",
        variant=model_id,
        device_type="cpu",
    )

    assert model is not None, "Model is None"
    assert isinstance(model, GraniteSpeech), f"Expected GraniteSpeech, got {type(model)}"

    config = model.get_config()
    return {
        "model_id": model_id,
        "audio_token_index": config.audio_token_index,
        "vocab_size": config.decoder_config.src_vocab_size,
    }


def test_get_model_hf_pretrained_variant():
    """Test get_model with hf_pretrained + variant (downloads weights from HF cache)."""
    from fms.models import get_model
    from fms.models.granite_speech import GraniteSpeech

    model_id = "ibm-granite/granite-speech-3.3-2b"

    # This downloads weights if not cached
    model = get_model(
        architecture="hf_pretrained",
        variant=model_id,
        device_type="cpu",
        data_type=torch.float32,
    )

    assert model is not None, "Model is None"
    assert isinstance(model, GraniteSpeech), f"Expected GraniteSpeech, got {type(model)}"

    # Verify weights are loaded (not random)
    # Check that embedding weights have reasonable magnitude
    embed_weight = model.get_input_embeddings().weight
    assert embed_weight.abs().mean() > 0.001, "Weights appear to be zero (not loaded)"

    config = model.get_config()
    return {
        "model_id": model_id,
        "vocab_size": config.decoder_config.src_vocab_size,
        "embed_weight_mean": float(embed_weight.abs().mean()),
    }


def test_get_model_hf_pretrained_model_path():
    """Test get_model with hf_pretrained + model_path (local path)."""
    from huggingface_hub import snapshot_download
    from fms.models import get_model
    from fms.models.granite_speech import GraniteSpeech

    model_id = "ibm-granite/granite-speech-3.3-2b"

    # Download to local path (with workaround for old checkpoint files)
    local_path = snapshot_download(
        model_id,
        ignore_patterns=["*-of-00003.safetensors"],
    )

    model = get_model(
        architecture="hf_pretrained",
        model_path=local_path,
        device_type="cpu",
        data_type=torch.float32,
    )

    assert model is not None, "Model is None"
    assert isinstance(model, GraniteSpeech), f"Expected GraniteSpeech, got {type(model)}"

    return {
        "local_path": local_path,
        "model_type": type(model).__name__,
    }


def test_get_model_data_types():
    """Test get_model with different data types."""
    from fms.models import get_model

    results = {}
    data_types = [
        ("float32", torch.float32),
        ("float16", torch.float16),
        ("bfloat16", torch.bfloat16),
    ]

    for dtype_name, dtype in data_types:
        try:
            model = get_model(
                architecture="granite_speech",
                variant="3.3-2b",
                device_type="cpu",
                data_type=dtype,
            )

            # Check actual dtype of parameters
            param = next(model.parameters())
            actual_dtype = param.dtype
            results[dtype_name] = {
                "requested": str(dtype),
                "actual": str(actual_dtype),
                "match": dtype == actual_dtype,
            }
        except Exception as e:
            results[dtype_name] = {"error": str(e)}

    # All should succeed
    for dtype_name, result in results.items():
        if "error" in result:
            raise AssertionError(f"Failed for {dtype_name}: {result['error']}")
        assert result["match"], f"Dtype mismatch for {dtype_name}"

    return results


def test_get_model_device_cpu():
    """Test get_model with CPU device."""
    from fms.models import get_model

    model = get_model(
        architecture="granite_speech",
        variant="3.3-2b",
        device_type="cpu",
    )

    # Check device
    param = next(model.parameters())
    assert param.device.type == "cpu", f"Expected cpu, got {param.device}"

    return {"device": str(param.device)}


def test_get_model_device_cuda():
    """Test get_model with CUDA device."""
    if not torch.cuda.is_available():
        return {"skipped": "CUDA not available"}

    from fms.models import get_model

    model = get_model(
        architecture="granite_speech",
        variant="3.3-2b",
        device_type="cuda",
        data_type=torch.float32,
    )

    # Check device
    param = next(model.parameters())
    assert param.device.type == "cuda", f"Expected cuda, got {param.device}"

    return {"device": str(param.device)}


def test_get_model_with_data_type_string():
    """Test get_model with data_type as string (not torch.dtype)."""
    from fms.models import get_model

    model = get_model(
        architecture="granite_speech",
        variant="3.3-2b",
        device_type="cpu",
        data_type="float32",  # String instead of torch.float32
    )

    param = next(model.parameters())
    assert param.dtype == torch.float32, f"Expected float32, got {param.dtype}"

    return {"dtype": str(param.dtype)}


def test_forward_pass_random_weights():
    """Test forward pass with randomly initialized model."""
    from fms.models import get_model

    model = get_model(
        architecture="granite_speech",
        variant="3.3-2b",
        device_type="cpu",
        data_type=torch.float32,
    )
    model.eval()

    # Create dummy inputs
    batch_size = 1
    seq_len = 20
    audio_len = 45  # 3 windows of 15

    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    # Don't use audio for simplicity (text-only)

    with torch.no_grad():
        logits, loss = model(input_ids=input_ids)

    assert logits is not None, "Logits is None"
    assert logits.shape[0] == batch_size, f"Batch size mismatch"
    assert logits.shape[1] == seq_len, f"Seq len mismatch"
    assert loss is None, "Loss should be None (no labels)"

    return {
        "logits_shape": list(logits.shape),
        "logits_mean": float(logits.mean()),
    }


def test_forward_pass_with_audio_random_weights():
    """Test forward pass with audio features (random weights)."""
    from fms.models import get_model
    import math

    model = get_model(
        architecture="granite_speech",
        variant="3.3-2b",
        device_type="cpu",
        data_type=torch.float32,
    )
    model.eval()

    config = model.get_config()

    # Create inputs with audio
    batch_size = 1
    audio_len = 45  # Multiple of window_size (15)
    num_windows = audio_len // config.window_size
    num_audio_tokens = num_windows * config.projector_config.num_queries

    seq_len = num_audio_tokens + 10  # audio tokens + text tokens

    input_ids = torch.randint(1, 1000, (batch_size, seq_len))
    input_ids[0, :num_audio_tokens] = config.audio_token_index

    input_features = torch.randn(batch_size, audio_len, 160)

    with torch.no_grad():
        logits, loss = model(
            input_ids=input_ids,
            input_features=input_features,
        )

    assert logits is not None, "Logits is None"
    assert logits.shape[0] == batch_size
    assert logits.shape[1] == seq_len

    return {
        "logits_shape": list(logits.shape),
        "num_audio_tokens": num_audio_tokens,
    }


# =============================================================================
# End-to-End Tests (require pretrained weights)
# =============================================================================


def test_e2e_hf_pretrained_forward():
    """End-to-end test: Load pretrained model and run forward pass."""
    from fms.models import get_model

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_model(
        architecture="hf_pretrained",
        variant="ibm-granite/granite-speech-3.3-2b",
        device_type=device,
        data_type=torch.float32,
    )
    model.eval()

    config = model.get_config()

    # Simple text-only forward
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)

    with torch.no_grad():
        logits, loss = model(input_ids=input_ids)

    assert logits is not None
    assert logits.shape == (1, 5, config.decoder_config.src_vocab_size)

    return {
        "device": device,
        "vocab_size": config.decoder_config.src_vocab_size,
        "logits_shape": list(logits.shape),
    }


def test_e2e_audio_to_text_with_hf_processor():
    """
    End-to-end test: Audio file -> Text transcription.

    Uses HF processor for audio preprocessing, FMS model for inference.
    This is the most comprehensive test of get_model functionality.
    """
    from datasets import load_dataset
    from transformers import GraniteSpeechProcessor
    from fms.models import get_model
    from fms.utils.generation import generate

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "ibm-granite/granite-speech-3.3-2b"

    # Load processor and audio
    print("  Loading processor and audio...")
    processor = GraniteSpeechProcessor.from_pretrained(model_id)

    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy",
        "clean",
        split="validation"
    )
    sample = ds[0]
    audio = sample["audio"]["array"]
    expected_text_contains = "mister"  # LibriSpeech sample should contain this

    # Process inputs
    text = ["Transcribe the following audio: <|audio|>"]
    inputs = processor(text=text, audio=audio, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print(f"  Input shapes: input_ids={inputs['input_ids'].shape}, "
          f"input_features={inputs['input_features'].shape}")

    # Load FMS model
    print("  Loading FMS model...")
    model = get_model(
        architecture="hf_pretrained",
        variant=model_id,
        device_type=device,
        data_type=torch.float32,
    )
    model.eval()

    # Generate transcription
    print("  Generating transcription...")
    with torch.no_grad():
        output_ids = generate(
            model,
            inputs["input_ids"],
            max_new_tokens=100,
            do_sample=False,
            use_cache=True,
            extra_kwargs={
                "input_features": inputs["input_features"],
                "input_features_mask": inputs.get("input_features_mask"),
                "attention_mask": inputs.get("attention_mask"),
            },
        )

    transcription = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    print(f"  Transcription: {transcription}")

    # Verify transcription quality
    assert len(transcription) > 10, f"Transcription too short: {transcription}"

    return {
        "device": device,
        "transcription": transcription,
        "transcription_length": len(transcription),
    }


def test_e2e_generation_equivalence():
    """
    End-to-end test: Verify FMS generation matches HF generation.

    This is the ultimate equivalence test for get_model.
    """
    from datasets import load_dataset
    from transformers import (
        GraniteSpeechProcessor,
        GraniteSpeechForConditionalGeneration
    )
    from fms.models import get_model
    from fms.utils.generation import generate

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "ibm-granite/granite-speech-3.3-2b"

    # Load processor and audio
    print("  Loading processor and audio...")
    processor = GraniteSpeechProcessor.from_pretrained(model_id)

    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy",
        "clean",
        split="validation"
    )
    audio = ds[0]["audio"]["array"]

    text = ["Transcribe the following audio: <|audio|>"]
    inputs = processor(text=text, audio=audio, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Load HF model
    print("  Loading HF model...")
    hf_model = GraniteSpeechForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map=device,
    )
    hf_model.eval()

    # HF generation
    print("  HF generation...")
    with torch.no_grad():
        hf_output = hf_model.generate(**inputs, max_new_tokens=50, do_sample=False)
    hf_text = processor.batch_decode(hf_output, skip_special_tokens=True)[0]

    # Load FMS model
    print("  Loading FMS model...")
    fms_model = get_model(
        architecture="hf_pretrained",
        variant=model_id,
        device_type=device,
        data_type=torch.float32,
    )
    fms_model.eval()

    # FMS generation
    print("  FMS generation...")
    with torch.no_grad():
        fms_output = generate(
            fms_model,
            inputs["input_ids"],
            max_new_tokens=50,
            do_sample=False,
            use_cache=True,
            extra_kwargs={
                "input_features": inputs["input_features"],
                "input_features_mask": inputs.get("input_features_mask"),
                "attention_mask": inputs.get("attention_mask"),
            },
        )
    fms_text = processor.batch_decode(fms_output, skip_special_tokens=True)[0]

    print(f"  HF:  {hf_text}")
    print(f"  FMS: {fms_text}")

    # Compare
    match = hf_text == fms_text

    return {
        "device": device,
        "hf_transcription": hf_text,
        "fms_transcription": fms_text,
        "match": match,
    }


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all tests."""
    runner = TestRunner()

    print("\n" + "="*60)
    print("GET_MODEL TEST SUITE FOR GRANITE-SPEECH")
    print("="*60)

    # Basic registration tests
    runner.run_test("list_models_and_variants", test_list_models_and_variants)

    # Architecture/variant tests (random init)
    runner.run_test("get_model_architecture_variant_2b",
                    test_get_model_architecture_variant, variant="3.3-2b")
    runner.run_test("get_model_architecture_variant_8b",
                    test_get_model_architecture_variant, variant="3.3-8b")

    # hf_configured test (downloads config only)
    runner.run_test("get_model_hf_configured", test_get_model_hf_configured)

    # Data type tests
    runner.run_test("get_model_data_types", test_get_model_data_types)
    runner.run_test("get_model_data_type_string", test_get_model_with_data_type_string)

    # Device tests
    runner.run_test("get_model_device_cpu", test_get_model_device_cpu)
    runner.run_test("get_model_device_cuda", test_get_model_device_cuda)

    # Forward pass tests (random weights)
    runner.run_test("forward_pass_random_weights", test_forward_pass_random_weights)
    runner.run_test("forward_pass_with_audio_random_weights",
                    test_forward_pass_with_audio_random_weights)

    # E2E tests (require pretrained weights - slower)
    print("\n" + "-"*60)
    print("RUNNING E2E TESTS (requires model download)")
    print("-"*60)

    runner.run_test("get_model_hf_pretrained_variant",
                    test_get_model_hf_pretrained_variant)
    runner.run_test("get_model_hf_pretrained_model_path",
                    test_get_model_hf_pretrained_model_path)
    runner.run_test("e2e_hf_pretrained_forward", test_e2e_hf_pretrained_forward)
    runner.run_test("e2e_audio_to_text", test_e2e_audio_to_text_with_hf_processor)
    runner.run_test("e2e_generation_equivalence", test_e2e_generation_equivalence)

    # Print summary
    success = runner.print_summary()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
