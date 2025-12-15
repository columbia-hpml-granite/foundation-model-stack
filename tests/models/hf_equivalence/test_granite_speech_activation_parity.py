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


def _attach_hooks(model, modules, prefix=""):
    """Attach forward hooks to capture intermediate activations."""
    acts = {}
    handles = []

    def make_hook(name):
        def hook(_m, _inp, out):
            if isinstance(out, (tuple, list)):
                tensor = out[0]
            else:
                tensor = out
            acts[name] = tensor.detach().cpu()
        return hook

    for name, module in modules.items():
        full = f"{prefix}{name}"
        h = module.register_forward_hook(make_hook(full))
        handles.append(h)

    return acts, handles


def _remove_hooks(handles):
    """Remove all registered hooks."""
    for h in handles:
        h.remove()


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Granite Speech activation parity test requires CUDA"
)
def test_granite_speech_activation_parity():
    """
    Compares HF vs FMS at several internal points:
    - encoder output
    - projector output
    - first decoder block output
    - final logits
    """
    from transformers import GraniteSpeechProcessor

    processor = GraniteSpeechProcessor.from_pretrained(MODEL_PATH)
    inputs = _get_inputs(processor)

    hf_model = _get_hf_model(MODEL_PATH)
    fms_model = _get_fms_model(MODEL_PATH)

    # Modules to hook in HF
    hf_modules = {
        "encoder": hf_model.encoder,
        "projector": hf_model.projector,
        "decoder_block0": hf_model.language_model.model.layers[0],
    }
    hf_acts, hf_handles = _attach_hooks(hf_model, hf_modules, prefix="hf:")

    # Modules to hook in FMS
    fms_modules = {
        "encoder": fms_model.encoder,
        "projector": fms_model.projector,
        "decoder_block0": fms_model.decoder.layers[0],
    }
    fms_acts, fms_handles = _attach_hooks(fms_model, fms_modules, prefix="fms:")

    # HF forward
    with torch.no_grad():
        hf_out = hf_model(**inputs)
    hf_logits = hf_out.logits.detach().cpu()

    # FMS forward
    with torch.no_grad():
        fms_logits, _ = fms_model(
            input_ids=inputs["input_ids"],
            input_features=inputs["input_features"],
            input_features_mask=inputs.get("input_features_mask", None),
            attention_mask=inputs.get("attention_mask", None),
        )
    fms_logits = fms_logits.detach().cpu()

    _remove_hooks(hf_handles)
    _remove_hooks(fms_handles)

    # Compare logits
    torch.testing.assert_close(fms_logits, hf_logits, atol=1e-3, rtol=1e-3)
    print(f"logits max diff: {(fms_logits - hf_logits).abs().max().item()}")

    # Compare internal activations
    for key in ["encoder", "projector", "decoder_block0"]:
        hf_act = hf_acts[f"hf:{key}"]
        fms_act = fms_acts[f"fms:{key}"]
        torch.testing.assert_close(fms_act, hf_act, atol=1e-3, rtol=1e-3)
        print(f"{key} max diff: {(fms_act - hf_act).abs().max().item()}")


if __name__ == "__main__":
    test_granite_speech_activation_parity()
