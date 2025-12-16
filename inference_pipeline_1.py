"""
GraniteSpeech: Full pipeline demonstration (text + audio -> text)

This script showcases the granite-speech model's speech-to-text capabilities
using audio samples from the LibriSpeech dataset on HuggingFace.

Usage:
    python scripts/granite_speech_inference.py
"""

import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from fms.models import get_model
from fms.models.granite_speech import GraniteSpeechFeatureExtractor, GraniteSpeechProcessor
from fms.utils.generation import generate


def main():
    # Configuration
    model_id = "ibm-granite/granite-speech-3.3-8b"  # 8B model (cleaner checkpoint, no mixed shards)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32  # bf16 saves VRAM

    print("=" * 70)
    print("GraniteSpeech Pipeline Demo: text + audio -> text")
    print("=" * 70)

    # Load model and tokenizer
    print(f"\n[1/4] Loading model: {model_id} ({dtype})")
    model_path = snapshot_download(model_id)
    model = get_model("granite_speech", "3.3-8b", model_path=model_path, source="hf", device_type=device, data_type=dtype)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load audio from HuggingFace dataset
    print("[2/4] Loading audio from LibriSpeech dataset...")
    dataset = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy",
        "clean",
        split="validation",
        trust_remote_code=True,
    )
    sample = dataset[0]
    audio = torch.tensor(sample["audio"]["array"], dtype=torch.float32)

    print(f"       Audio duration: {len(audio) / 16000:.2f}s @ 16kHz")
    print(f"       Ground truth: \"{sample['text']}\"")

    # Process inputs (text prompt + audio)
    print("[3/4] Processing inputs...")
    processor = GraniteSpeechProcessor(GraniteSpeechFeatureExtractor(), tokenizer)
    prompt = "Transcribe the following audio: <|audio|>"
    inputs = processor(text=[prompt], audio=audio, return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Generate transcription
    print("[4/4] Generating transcription...")
    with torch.no_grad():
        output_ids = generate(
            model,
            inputs["input_ids"],
            max_new_tokens=100,
            do_sample=False,
            use_cache=True,
            prepare_model_inputs_hook=model.prepare_inputs_for_generation,
            extra_kwargs={
                "input_features": inputs["input_features"],
                "input_features_mask": inputs.get("input_features_mask"),
                "attention_mask": inputs.get("attention_mask"),
            },
        )

    transcription = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Prompt:        {prompt}")
    print(f"Ground truth:  {sample['text']}")
    print(f"Transcription: {transcription}")
    print("=" * 70)


if __name__ == "__main__":
    main()

