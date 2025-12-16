#!/usr/bin/env python
"""Debug script to find where FMS and HF granite-speech outputs diverge."""

import torch
from datasets import load_dataset
from transformers import GraniteSpeechProcessor, GraniteSpeechForConditionalGeneration
from fms.models import get_model
from huggingface_hub import snapshot_download

device = "cuda"
torch.set_default_dtype(torch.float32)

def main():
    model_path = "ibm-granite/granite-speech-3.3-2b"

    # Load processor and inputs
    print("Loading processor and inputs...")
    processor = GraniteSpeechProcessor.from_pretrained(model_path)
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    sample = ds[0]
    audio = sample["audio"]["array"]
    text = ["Transcribe the following audio: <|audio|>"]

    inputs = processor(text=text, audio=audio, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print(f"Input shapes:")
    for k, v in inputs.items():
        print(f"  {k}: {v.shape}")

    # Load HF model
    print("\nLoading HF model...")
    hf_model = GraniteSpeechForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float32, device_map=device
    )
    hf_model.eval()

    # Load FMS model
    print("Loading FMS model...")
    fms_model_path = snapshot_download(
        model_path,
        ignore_patterns=["*-of-00003.safetensors"],
    )
    fms_model = get_model(
        "hf_pretrained",
        fms_model_path,
        data_type=torch.float32,
        device_type=device,
    )
    fms_model.eval()

    print("\n" + "="*60)
    print("COMPARISON 1: Text Embeddings (no audio)")
    print("="*60)

    # Compare text embeddings without audio token
    test_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)

    with torch.no_grad():
        hf_embed = hf_model.language_model.model.embed_tokens(test_ids)
        fms_embed = fms_model.get_input_embeddings()(test_ids)

    print(f"HF embed shape: {hf_embed.shape}, FMS embed shape: {fms_embed.shape}")
    print(f"HF embed stats: min={hf_embed.min():.4f}, max={hf_embed.max():.4f}, mean={hf_embed.mean():.4f}")
    print(f"FMS embed stats: min={fms_embed.min():.4f}, max={fms_embed.max():.4f}, mean={fms_embed.mean():.4f}")
    print(f"Embedding diff: {(hf_embed - fms_embed).abs().max():.6f}")

    print("\n" + "="*60)
    print("COMPARISON 2: Audio Encoder Output")
    print("="*60)

    with torch.no_grad():
        # HF: encoder (GraniteSpeechCTCEncoder)
        hf_encoder_out = hf_model.encoder(inputs["input_features"])
        print(f"HF encoder output shape: {hf_encoder_out.shape}")
        print(f"HF encoder stats: min={hf_encoder_out.min():.4f}, max={hf_encoder_out.max():.4f}, mean={hf_encoder_out.mean():.4f}")

        # FMS: encoder
        fms_encoder_out = fms_model.encoder(inputs["input_features"])
        print(f"FMS encoder output shape: {fms_encoder_out.shape}")
        print(f"FMS encoder stats: min={fms_encoder_out.min():.4f}, max={fms_encoder_out.max():.4f}, mean={fms_encoder_out.mean():.4f}")

        print(f"Encoder diff: {(hf_encoder_out - fms_encoder_out).abs().max():.6f}")

    print("\n" + "="*60)
    print("COMPARISON 3: Projector Output")
    print("="*60)

    with torch.no_grad():
        # HF: projector (GraniteSpeechEncoderProjector)
        hf_proj_out = hf_model.projector(hf_encoder_out)
        print(f"HF projector output shape: {hf_proj_out.shape}")
        print(f"HF projector stats: min={hf_proj_out.min():.4f}, max={hf_proj_out.max():.4f}, mean={hf_proj_out.mean():.4f}")

        # FMS: projector
        fms_proj_out = fms_model.projector(fms_encoder_out)
        print(f"FMS projector output shape: {fms_proj_out.shape}")
        print(f"FMS projector stats: min={fms_proj_out.min():.4f}, max={fms_proj_out.max():.4f}, mean={fms_proj_out.mean():.4f}")

        print(f"Projector diff: {(hf_proj_out - fms_proj_out).abs().max():.6f}")

    print("\n" + "="*60)
    print("COMPARISON 4: Merged Embeddings (audio into text)")
    print("="*60)

    with torch.no_grad():
        # Get audio features for merging
        hf_audio_embeds = hf_model.projector(hf_model.encoder(inputs["input_features"]))
        fms_audio_embeds = fms_model.projector(fms_model.encoder(inputs["input_features"]))

        # Check FMS merged embeddings
        input_features_mask = inputs.get("input_features_mask")
        if input_features_mask is None:
            input_features_mask = fms_audio_embeds.new_ones(fms_audio_embeds.shape[:2], dtype=torch.bool)
        input_features_mask = input_features_mask.to(device=fms_audio_embeds.device, dtype=torch.bool)

        fms_merged = fms_model.get_merged_audio_embeddings(
            input_ids=inputs["input_ids"],
            audio_features=fms_audio_embeds,
            input_features_mask=input_features_mask,
        )
        print(f"FMS merged embeds shape: {fms_merged.shape}")
        print(f"FMS merged embeds stats: min={fms_merged.min():.4f}, max={fms_merged.max():.4f}, mean={fms_merged.mean():.4f}")

    print("\n" + "="*60)
    print("COMPARISON 5: Decoder Output (before lm_head)")
    print("="*60)

    with torch.no_grad():
        # FMS decoder forward
        fms_dec_out, _ = fms_model.decoder(
            x_in=fms_merged,
            attention_mask=inputs.get("attention_mask"),
            use_cache=False,
        )
        print(f"FMS decoder output shape: {fms_dec_out.shape}")
        print(f"FMS decoder output stats: min={fms_dec_out.min():.4f}, max={fms_dec_out.max():.4f}, mean={fms_dec_out.mean():.4f}")

        # FMS lm_head
        fms_logits_manual = fms_model.lm_head(fms_dec_out)
        print(f"FMS logits (manual) stats: min={fms_logits_manual.min():.4f}, max={fms_logits_manual.max():.4f}, mean={fms_logits_manual.mean():.4f}")

    print("\n" + "="*60)
    print("COMPARISON 6: Full Forward Pass")
    print("="*60)

    with torch.no_grad():
        # HF forward
        hf_output = hf_model(
            input_ids=inputs["input_ids"],
            input_features=inputs["input_features"],
            input_features_mask=inputs.get("input_features_mask"),
            attention_mask=inputs.get("attention_mask"),
            return_dict=True,
        )
        hf_logits = hf_output.logits

        # FMS forward
        fms_output = fms_model(
            input_ids=inputs["input_ids"],
            input_features=inputs["input_features"],
            input_features_mask=inputs.get("input_features_mask"),
            attention_mask=inputs.get("attention_mask"),
        )
        fms_logits = fms_output[0]  # (logits, cache/loss)

    print(f"HF logits shape: {hf_logits.shape}")
    print(f"FMS logits shape: {fms_logits.shape}")
    print(f"HF logits stats: min={hf_logits.min():.4f}, max={hf_logits.max():.4f}, mean={hf_logits.mean():.4f}")
    print(f"FMS logits stats: min={fms_logits.min():.4f}, max={fms_logits.max():.4f}, mean={fms_logits.mean():.4f}")

    # Signature computation (same as test)
    hf_sig = (hf_logits.max(2)[0] - hf_logits.min(2)[0]).squeeze()
    fms_sig = (fms_logits.max(2)[0] - fms_logits.min(2)[0]).squeeze()

    print(f"\nHF signature (first 5): {(hf_sig - hf_sig.min())[:5].tolist()}")
    print(f"FMS signature (first 5): {(fms_sig - fms_sig.min())[:5].tolist()}")

    print(f"\nLogits diff: {(hf_logits - fms_logits).abs().max():.6f}")

    # Compare with manual computation
    print(f"\nManual vs forward logits match: {torch.allclose(fms_logits_manual, fms_logits, atol=1e-5)}")
    print(f"Manual vs HF diff: {(fms_logits_manual - hf_logits).abs().max():.6f}")

if __name__ == "__main__":
    main()
