#!/usr/bin/env python3
"""Test script for Granite Speech model with a sample audio file."""

import torch
import torchaudio
from transformers import AutoProcessor, GraniteSpeechForConditionalGeneration


def main():
    # Path to audio file
    audio_path = "/Users/garfield/Downloads/harvard.wav"
    model_id = "ibm-granite/granite-speech-3.3-2b"

    print(f"Loading model: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = GraniteSpeechForConditionalGeneration.from_pretrained(model_id)

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()

    # Load audio file
    print(f"Loading audio: {audio_path}")
    audio, sr = torchaudio.load(audio_path)

    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # Resample to 16kHz if needed
    if sr != 16000:
        print(f"Resampling from {sr}Hz to 16000Hz")
        audio = torchaudio.transforms.Resample(sr, 16000)(audio)

    # Squeeze to 1D tensor
    audio = audio.squeeze(0)

    print(f"Audio shape: {audio.shape}, duration: {audio.shape[0] / 16000:.2f}s")

    # Prepare the prompt with audio placeholder
    chat = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant.",
        },
        {
            "role": "user",
            "content": "<|audio|>Can you transcribe this speech into a written format?",
        },
    ]
    prompt = processor.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    # Process inputs
    print("Processing inputs...")
    inputs = processor(prompt, audio, return_tensors="pt").to(device)

    print(f"Input features shape: {inputs['input_features'].shape}")
    print(f"Input IDs shape: {inputs['input_ids'].shape}")

    # Generate transcription
    print("Generating transcription...")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256)

    # Decode and print
    transcription = processor.tokenizer.decode(output[0], skip_special_tokens=True)

    print("\n" + "=" * 60)
    print("FULL OUTPUT:")
    print("=" * 60)
    print(transcription)

    # Extract just the assistant's response
    if "assistant" in transcription.lower():
        assistant_response = transcription.split("assistant")[-1].strip()
        print("\n" + "=" * 60)
        print("TRANSCRIPTION:")
        print("=" * 60)
        print(assistant_response)


if __name__ == "__main__":
    main()
