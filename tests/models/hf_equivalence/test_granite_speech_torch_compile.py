import time

import pytest
import torch

from fms.models import get_model

device = "cuda"
torch.set_default_dtype(torch.float32)

MODEL_PATH = "ibm-granite/granite-speech-3.3-2b"


def _get_inputs(processor):
    """Get sample audio inputs from LibriSpeech dummy dataset."""
    from datasets import load_dataset

    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    )
    sample = ds[0]
    audio = sample["audio"]["array"]
    prompt = "Transcribe the following audio: <|audio|>"

    inputs = processor(
        audio=audio,
        text=prompt,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs


def _get_hf_model(model_path):
    """Load HF model."""
    from transformers import GraniteSpeechForConditionalGeneration

    model = GraniteSpeechForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map=device,
    )
    model.eval()
    return model


def _get_fms_model(model_path):
    """Load FMS model using hf_pretrained."""
    model = get_model(
        "hf_pretrained",
        model_path,
        data_type=torch.float32,
        device_type=device,
    )
    model.eval()
    return model


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Granite Speech torch.compile test requires CUDA"
)
def test_granite_speech_compile_logits_equivalence():
    """
    Tests torch.compile compatibility and performance for GraniteSpeech.
    Compares eager mode and compiled mode outputs between HF and FMS.
    """
    from transformers import GraniteSpeechProcessor

    processor = GraniteSpeechProcessor.from_pretrained(MODEL_PATH)
    inputs = _get_inputs(processor)

    hf_model = _get_hf_model(MODEL_PATH)
    fms_model = _get_fms_model(MODEL_PATH)

    # Helpers that match HF and FMS signatures
    def hf_fn(input_ids, input_features, input_features_mask, attention_mask):
        return hf_model(
            input_ids=input_ids,
            input_features=input_features,
            input_features_mask=input_features_mask,
            attention_mask=attention_mask,
        ).logits

    def fms_fn(input_ids, input_features, input_features_mask, attention_mask):
        logits, _ = fms_model(
            input_ids=input_ids,
            input_features=input_features,
            input_features_mask=input_features_mask,
            attention_mask=attention_mask,
        )
        return logits

    args = (
        inputs["input_ids"],
        inputs["input_features"],
        inputs.get("input_features_mask", None),
        inputs.get("attention_mask", None),
    )

    # Eager sanity run
    with torch.no_grad():
        hf_logits_eager = hf_fn(*args).detach().cpu()
        fms_logits_eager = fms_fn(*args).detach().cpu()

    # Quick eager comparison
    torch.testing.assert_close(fms_logits_eager, hf_logits_eager, atol=1e-3, rtol=1e-3)
    print(f"Eager max diff: {(hf_logits_eager - fms_logits_eager).abs().max().item()}")

    # Compile both functions
    compiled_hf_fn = torch.compile(hf_fn, mode="max-autotune")
    compiled_fms_fn = torch.compile(fms_fn, mode="max-autotune")

    # Warmup and measure
    def bench(fn, label, n_warmup=2, n_runs=5):
        for _ in range(n_warmup):
            with torch.no_grad():
                _ = fn(*args)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_runs):
            with torch.no_grad():
                _ = fn(*args)
        torch.cuda.synchronize()
        end = time.perf_counter()

        ms = (end - start) * 1000 / n_runs
        print(f"{label}: {ms:.2f} ms / run")
        return ms

    print("=== Timing (compiled) ===")
    hf_ms = bench(compiled_hf_fn, "HF (compiled)")
    fms_ms = bench(compiled_fms_fn, "FMS (compiled)")

    # Compare compiled outputs
    with torch.no_grad():
        hf_logits_compiled = compiled_hf_fn(*args).detach().cpu()
        fms_logits_compiled = compiled_fms_fn(*args).detach().cpu()

    torch.testing.assert_close(fms_logits_compiled, hf_logits_compiled, atol=1e-3, rtol=1e-3)
    print(f"Compiled max diff: {(hf_logits_compiled - fms_logits_compiled).abs().max().item()}")


if __name__ == "__main__":
    test_granite_speech_compile_logits_equivalence()
