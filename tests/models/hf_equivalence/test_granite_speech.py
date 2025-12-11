import pytest
import torch

from fms.models import get_model
from fms.utils import serialization


"""
HF <-> FMS Granite Speech equivalence test.

This test is meant to:

1. Load the official Hugging Face GraniteSpeechForConditionalGeneration
   (using ibm-granite/granite-speech-*).
2. Build the corresponding FMS GraniteSpeech model.
3. Use the registered HF -> FMS adapter to map weights into the FMS model.
4. Compare logits on the same (audio, text) input.
"""


def _get_hf_components(model_id: str, device: str = "cuda"):
    """
    Load HF GraniteSpeechProcessor + GraniteSpeechForConditionalGeneration.
    """
    from transformers import GraniteSpeechProcessor, GraniteSpeechForConditionalGeneration

    processor = GraniteSpeechProcessor.from_pretrained(model_id)
    hf_model = GraniteSpeechForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map=device,
    ).eval()

    return processor, hf_model


def _get_sample_inputs(processor, device: str = "cuda"):
    """
    Get a small (audio, text) pair to drive the model.

    We follow HF docs and use a single short audio clip plus a prompt
    containing one <|audio|> token.

    We can swap this out with whichever dataset we want.
    """
    from datasets import load_dataset

    # Tiny dummy dataset just for testing purposes.
    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    )
    sample = ds[0]
    audio = sample["audio"]["array"]

    # Prompt must contain exactly one audio token for now
    text = ["Transcribe the following audio: <|audio|>"]

    inputs = processor(
        text=text,
        audio=audio,
        return_tensors="pt",
    )

    # Move tensors to the right device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    return inputs


def _load_fms_granite_speech(model_id: str, device: str = "cuda"):
    """
    Build FMS GraniteSpeech and load HF weights into it.

    This relies on the adapter we registered at the bottom of
    fms/models/granite_speech.py via serialization.register_adapter(...).
    """
    # Grab HF model to extract config and state dict
    from transformers import GraniteSpeechForConditionalGeneration

    hf_model = GraniteSpeechForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map=device,
    ).eval()

    # Extract audio_token_index from HF config
    # Different model versions use different values:
    # - granite-speech-3.2-8b: 49155
    # - granite-speech-3.3-2b: 49159
    # - granite-speech-3.3-8b: 49159
    audio_token_index = getattr(hf_model.config, "audio_token_index", 49155)

    # NOTE: The variant name ("3.3-2b") should match how we registered
    # the model in granite_speech.py. Adjust if the variant name differs.
    # Pass audio_token_index to ensure FMS uses the same value as HF model
    fms_model = get_model(
        "granite_speech",
        "3.3-2b",
        audio_token_index=audio_token_index,
    )
    fms_model.to(device)
    fms_model.eval()

    hf_state = hf_model.state_dict()

    # Apply HF -> FMS adapter (registered in granite_speech.py)
    # This will:
    #   - rename keys
    #   - split KV weights if needed
    #   - fuse/unfuse any special parameters
    fms_state = serialization.get_adapted("granite_speech", "hf", hf_state, {})

    missing, unexpected = fms_model.load_state_dict(fms_state, strict=False)
    if missing:
        print("Missing keys when loading HF -> FMS:", missing)
    if unexpected:
        print("Unexpected keys when loading HF -> FMS:", unexpected)

    return fms_model


def _run_hf(model_id: str, device: str = "cuda"):
    """
    Run the HF Granite Speech model and return logits.
    """
    processor, hf_model = _get_hf_components(model_id, device=device)
    inputs = _get_sample_inputs(processor, device=device)

    with torch.no_grad():
        # HF forward returns a ModelOutput; logits field is (B, T, V)
        outputs = hf_model(**inputs)
        logits = outputs.logits

    return logits, processor, inputs


def _run_fms(model_id: str, device: str = "cuda"):
    """
    Run the FMS Granite Speech model (with HF weights loaded) and return logits.
    """
    # We reuse the processor + inputs from HF to guarantee identical preprocessing.
    processor, hf_model = _get_hf_components(model_id, device=device)
    inputs = _get_sample_inputs(processor, device=device)

    fms_model = _load_fms_granite_speech(model_id, device=device)

    # FMS GraniteSpeech.forward is expected to accept:
    #   - input_ids
    #   - input_features
    #   - input_features_mask
    #   - attention_mask
    with torch.no_grad():
        logits, _ = fms_model(
            input_ids=inputs["input_ids"],
            input_features=inputs["input_features"],
            input_features_mask=inputs.get("input_features_mask", None),
            attention_mask=inputs.get("attention_mask", None),
        )

    return logits, processor, inputs


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Granite Speech HF equivalence test requires CUDA"
)
def test_granite_speech_3_3_2b_equivalence():
    """
    Main equivalence test:

    - Loads HF model + FMS model
    - Runs both on the same inputs
    - Asserts logits are numerically close
    """
    device = "cuda"
    torch.set_default_dtype(torch.float32)

    # Using 3.3-2b as 3.2-8b uses granite_speech_qformer which is not registered in transformers yet
    model_id = "ibm-granite/granite-speech-3.3-2b"

    hf_logits, _, _ = _run_hf(model_id, device=device)
    fms_logits, _, _ = _run_fms(model_id, device=device)

    # Basic sanity checks
    assert hf_logits.shape == fms_logits.shape

    # Allow small numerical differences
    torch.testing.assert_close(
        fms_logits, hf_logits, atol=1e-3, rtol=1e-3
    )


if __name__ == "__main__":
    # Allow running as a standalone script for quick debugging.
    test_granite_speech_3_3_2b_equivalence()
