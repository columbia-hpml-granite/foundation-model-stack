import pytest
import torch

from fms.models import get_model
from fms.utils import serialization


MODEL_ID = "ibm-granite/granite-speech-3.2-8b"  # adjust if needed
FMS_VARIANT = "3.2-8b" # must match the FMS registry


def _require_deps():
    try:
        from transformers import GraniteSpeechProcessor, GraniteSpeechForConditionalGeneration 
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


def _attach_module_hooks(model, modules, prefix=""):
    acts = {}
    handles = []

    def make_hook(name):
        def hook(_m, _inp, out):
            # Handling tuple outputs (e.g. (hidden_states, ...))
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
    for h in handles:
        h.remove()


def _compare_tensor_pair(a, b, name, atol=1e-3, rtol=1e-3):
    print(f"{name}:")
    print("HF shape:", tuple(a.shape), "| FMS shape:", tuple(b.shape))
    assert a.shape == b.shape, f"Shape mismatch for {name}: {a.shape} vs {b.shape}"

    diff = (a - b).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print("max abs diff:", max_diff)
    print("mean abs diff:", mean_diff)

    assert torch.allclose(a, b, atol=atol, rtol=rtol), (
        f"{name} not close enough: max={max_diff}, mean={mean_diff}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Granite Speech activation parity test requires CUDA")
def test_granite_speech_activation_parity():
    """
    Compares HF vs FMS at several internal points:

    - encoder output
    - projector output
    - first decoder block output
    - final logits

    Uses nn.Module.forward hooks for component-level parity.
    """
    _require_deps()

    device = "cuda"
    torch.set_default_dtype(torch.float32)

    processor, hf_model = _load_hf_components(device)
    inputs = _build_inputs(processor, device)
    fms_model = _load_fms_with_hf_weights(device)

    # Modules to hook in HF
    hf_modules = {
        "encoder": hf_model.encoder,
        "projector": hf_model.projector,
        "decoder_block0": hf_model.language_model.model.layers[0],
    }
    hf_acts, hf_handles = _attach_module_hooks(hf_model, hf_modules, prefix="hf:")

    # Modules to hook in FMS
    fms_modules = {
        "encoder": fms_model.encoder,
        "projector": fms_model.projector,
        "decoder_block0": fms_model.decoder.layers[0],
    }
    fms_acts, fms_handles = _attach_module_hooks(fms_model, fms_modules, prefix="fms:")

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

    # Compare logits first
    _compare_tensor_pair(hf_logits, fms_logits, "logits")

    # Compare internal activations
    for key in ["encoder", "projector", "decoder_block0"]:
        hf_key = f"hf:{key}"
        fms_key = f"fms:{key}"
        assert hf_key in hf_acts, f"Missing HF activation for {hf_key}"
        assert fms_key in fms_acts, f"Missing FMS activation for {fms_key}"
        _compare_tensor_pair(hf_acts[hf_key], fms_acts[fms_key], key)
