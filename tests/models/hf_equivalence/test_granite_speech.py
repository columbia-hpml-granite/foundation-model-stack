import pytest
import torch

from fms.models import get_model

device = "cuda"
torch.set_default_dtype(torch.float32)


def _get_inputs(processor):
    """Get sample audio inputs from LibriSpeech dummy dataset."""
    from datasets import load_dataset

    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    )
    sample = ds[0]
    audio = sample["audio"]["array"]
    text = ["Transcribe the following audio: <|audio|>"]

    inputs = processor(
        text=text,
        audio=audio,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs


def _get_hf_model_output(model_path, inputs):
    """Run HF model and return logits."""
    from transformers import GraniteSpeechForConditionalGeneration

    model = GraniteSpeechForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map=device,
    )
    model.eval()
    with torch.no_grad():
        output = model(**inputs)
    return output.logits


def _get_fms_model_output(model_path, inputs):
    """Run FMS model and return logits."""
    model = get_model(
        "hf_pretrained",
        model_path,
        data_type=torch.float32,
        device_type=device,
    )
    model.eval()

    with torch.no_grad():
        logits, _ = model(
            input_ids=inputs["input_ids"],
            input_features=inputs["input_features"],
            input_features_mask=inputs.get("input_features_mask", None),
            attention_mask=inputs.get("attention_mask", None),
        )
    return logits


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Granite Speech HF equivalence test requires CUDA"
)
def test_granite_speech_3_3_2b_equivalence():
    """Tests GraniteSpeech 2B equivalence with HuggingFace implementation."""
    from transformers import GraniteSpeechProcessor

    model_path = "ibm-granite/granite-speech-3.3-2b"
    processor = GraniteSpeechProcessor.from_pretrained(model_path)
    inputs = _get_inputs(processor)

    hf_logits = _get_hf_model_output(model_path, inputs)
    fms_logits = _get_fms_model_output(model_path, inputs)

    # Compare outputs
    torch.testing.assert_close(fms_logits, hf_logits, atol=1e-3, rtol=1e-3)
    print(f"Shape: {fms_logits.shape}")
    print(f"Max diff: {(fms_logits - hf_logits).abs().max().item()}")


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Granite Speech HF equivalence test requires CUDA"
)
def test_granite_speech_3_3_8b_equivalence():
    """Tests GraniteSpeech 8B equivalence with HuggingFace implementation."""
    from transformers import GraniteSpeechProcessor

    model_path = "ibm-granite/granite-speech-3.3-8b"
    processor = GraniteSpeechProcessor.from_pretrained(model_path)
    inputs = _get_inputs(processor)

    hf_logits = _get_hf_model_output(model_path, inputs)
    fms_logits = _get_fms_model_output(model_path, inputs)

    # Compare outputs
    torch.testing.assert_close(fms_logits, hf_logits, atol=1e-3, rtol=1e-3)
    print(f"Shape: {fms_logits.shape}")
    print(f"Max diff: {(fms_logits - hf_logits).abs().max().item()}")


if __name__ == "__main__":
    test_granite_speech_3_3_2b_equivalence()
