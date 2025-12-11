import time

import pytest
import torch

from fms.models import get_model
from fms.utils import serialization


MODEL_ID = "ibm-granite/granite-speech-3.3-2b"  # Using 3.3-2b as 3.2-8b uses unregistered granite_speech_qformer
FMS_VARIANT = "3.3-2b"                          # must match your FMS registry


def _require_deps():
    try:
        from transformers import GraniteSpeechProcessor, GraniteSpeechForConditionalGeneration  # noqa: F401
        from datasets import load_dataset  # noqa: F401
    except ImportError as e:
        pytest.skip(f"Missing dependency for HF Granite Speech test: {e}")


def _load_hf_components(device: str):
    from transformers import GraniteSpeechProcessor, GraniteSpeechForConditionalGeneration

    processor = GraniteSpeechProcessor.from_pretrained(MODEL_ID)
    hf_model = GraniteSpeechForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map=device,
    ).eval()

    return processor, hf_model


def _build_inputs(processor, device: str):
    from datasets import load_dataset

    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    sample = ds[0]
    audio = sample["audio"]["array"]
    prompt = "Transcribe the following audio: <|audio|>"

    inputs = processor(
        audios=[audio],
        text=[prompt],
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs


def _load_fms_with_hf_weights(device: str):
    _, hf_model = _load_hf_components(device)
    hf_state = hf_model.state_dict()

    fms_model = get_model("granite_speech", FMS_VARIANT)
    fms_model.to(device)
    fms_model.eval()

    fms_state = serialization.apply_adapter("granite_speech", "hf", hf_state)
    missing, unexpected = fms_model.load_state_dict(fms_state, strict=False)

    if missing:
        print("Missing keys when loading HF -> FMS:", missing)
    if unexpected:
        print("Unexpected keys when loading HF -> FMS:", unexpected)

    return fms_model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Granite Speech torch.compile test requires CUDA")
def test_granite_speech_compile_logits_equivalence():
    _require_deps()

    device = "cuda"
    torch.set_default_dtype(torch.float32)

    processor, hf_model = _load_hf_components(device)
    inputs = _build_inputs(processor, device)
    fms_model = _load_fms_with_hf_weights(device)

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

    # Quick eager comparison before compile
    assert hf_logits_eager.shape == fms_logits_eager.shape
    eager_diff = (hf_logits_eager - fms_logits_eager).abs()
    print("Eager max abs diff:", eager_diff.max().item())
    print("Eager mean abs diff:", eager_diff.mean().item())

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

    assert hf_ms > 0
    assert fms_ms > 0

    # Compare compiled outputs
    with torch.no_grad():
        hf_logits_compiled = compiled_hf_fn(*args).detach().cpu()
        fms_logits_compiled = compiled_fms_fn(*args).detach().cpu()

    assert hf_logits_compiled.shape == fms_logits_compiled.shape

    diff = (hf_logits_compiled - fms_logits_compiled).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print("Compiled max abs diff:", max_diff)
    print("Compiled mean abs diff:", mean_diff)

    # If this fails, pytest will show the printed diffs above
    assert torch.allclose(hf_logits_compiled, fms_logits_compiled, atol=1e-3, rtol=1e-3), (
        f"Compiled logits differ: max={max_diff}, mean={mean_diff}"
    )
