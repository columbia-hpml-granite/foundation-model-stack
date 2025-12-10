"""
Granite Speech End-to-End Test Script

Steps:
- Step 2: FMS shape test (no weights)
- Step 3: Load HF weights into FMS
- Step 4: Compare FMS output with HF output

Prerequisites:
- pip install transformers torch torchaudio librosa huggingface_hub

Usage:
    python test_granite_speech_e2e.py --audio test.wav
    python test_granite_speech_e2e.py --step 2  # Shape test only
    python test_granite_speech_e2e.py --step 3  # Weight loading test
    python test_granite_speech_e2e.py --step 4 --audio test.wav  # Full comparison
"""

import argparse
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add FMS to path
sys.path.insert(0, str(Path(__file__).parent))


# ============================================================================
# Step 2: FMS Shape Test (No Weights)
# ============================================================================

def test_step2_shapes():
    """Test FMS model shapes without loading weights."""
    print("\n" + "=" * 60)
    print("STEP 2: FMS Shape Test (No Weights)")
    print("=" * 60)

    from fms.models.granite_speech import GraniteSpeech, GraniteSpeechConfig

    # Create model with default config
    config = GraniteSpeechConfig()
    model = GraniteSpeech(config)
    model.eval()

    print(f"\n[Config]")
    print(f"  Encoder: {config.encoder_config.num_layers} layers, {config.encoder_config.hidden_dim} dim")
    print(f"  Projector: window_size={config.window_size}, downsample_rate={config.downsample_rate}")
    print(f"  Decoder: {config.decoder_config.nlayers} layers, {config.decoder_config.emb_dim} dim")

    # Test with dummy audio input
    batch_size = 2
    audio_seq_len = 500  # ~5 seconds
    num_features = 160

    audio_features = torch.randn(batch_size, audio_seq_len, num_features)

    print(f"\n[Input]")
    print(f"  audio_features: {audio_features.shape}")

    # Test encoder
    print(f"\n[Encoder Output]")
    with torch.no_grad():
        encoder_out = model.encoder(audio_features)
    print(f"  encoder_out: {encoder_out.shape}")
    assert encoder_out.shape == (batch_size, audio_seq_len, 1024), f"Expected (2, 500, 1024), got {encoder_out.shape}"
    print(f"  ✓ Shape correct: (batch, seq_len, 1024)")

    # Test projector
    print(f"\n[Projector Output]")
    with torch.no_grad():
        projected = model.projector(encoder_out)
    nblocks = (audio_seq_len + config.window_size - 1) // config.window_size  # ceil division
    expected_tokens = nblocks * (config.window_size // config.downsample_rate)
    print(f"  projected: {projected.shape}")
    print(f"  Expected tokens: ceil({audio_seq_len}/{config.window_size}) × {config.window_size // config.downsample_rate} = {expected_tokens}")
    assert projected.shape == (batch_size, expected_tokens, 4096), f"Expected (2, {expected_tokens}, 4096), got {projected.shape}"
    print(f"  ✓ Shape correct: (batch, {expected_tokens}, 4096)")

    # Test get_audio_features (encoder + projector)
    print(f"\n[get_audio_features Output]")
    with torch.no_grad():
        audio_embeds = model.get_audio_features(audio_features)
    print(f"  audio_embeds: {audio_embeds.shape}")
    assert audio_embeds.shape == projected.shape
    print(f"  ✓ Shape correct")

    print(f"\n✓ STEP 2 PASSED: All shapes are correct!")
    return True


# ============================================================================
# Step 3: Load HF Weights into FMS
# ============================================================================

def test_step3_weight_loading(model_id: str = "ibm-granite/granite-speech-3.3-8b"):
    """Test loading HuggingFace weights into FMS model."""
    print("\n" + "=" * 60)
    print("STEP 3: Weight Loading Test")
    print("=" * 60)

    from huggingface_hub import hf_hub_download, list_repo_files
    from safetensors.torch import load_file
    import re

    from fms.models.granite_speech import GraniteSpeech, GraniteSpeechConfig

    # List files in repo
    print(f"\n[Checking HuggingFace repo: {model_id}]")
    try:
        files = list_repo_files(model_id)
        weight_files = [f for f in files if f.endswith(('.safetensors', '.bin'))]
        print(f"  Weight files found: {weight_files[:5]}...")  # Show first 5
    except Exception as e:
        print(f"  Error listing files: {e}")
        print(f"  Make sure you have access to {model_id}")
        return False

    # Download weights
    print(f"\n[Downloading weights]")
    try:
        # Try safetensors first
        if any('safetensors' in f for f in weight_files):
            weight_path = hf_hub_download(model_id, "model.safetensors")
            hf_state_dict = load_file(weight_path)
        else:
            weight_path = hf_hub_download(model_id, "pytorch_model.bin")
            hf_state_dict = torch.load(weight_path, map_location="cpu")
        print(f"  Downloaded: {weight_path}")
        print(f"  Number of tensors: {len(hf_state_dict)}")
    except Exception as e:
        print(f"  Error downloading: {e}")
        return False

    # Show sample weight names
    print(f"\n[HF Weight Names (sample)]")
    for i, name in enumerate(list(hf_state_dict.keys())[:10]):
        print(f"  {name}: {hf_state_dict[name].shape}")

    # Create FMS model
    print(f"\n[Creating FMS model]")
    config = GraniteSpeechConfig()
    fms_model = GraniteSpeech(config)
    fms_state_dict = fms_model.state_dict()
    print(f"  FMS model created with {len(fms_state_dict)} parameters")

    # Show sample FMS weight names
    print(f"\n[FMS Weight Names (sample)]")
    for i, name in enumerate(list(fms_state_dict.keys())[:10]):
        print(f"  {name}: {fms_state_dict[name].shape}")

    # Convert HF weights to FMS format
    print(f"\n[Converting HF weights to FMS format]")

    def convert_hf_to_fms(hf_sd):
        """Convert HuggingFace state dict to FMS format."""
        fms_sd = {}

        # Encoder mappings
        encoder_map = [
            (r"^encoder\.input_linear\.", "encoder.input_proj."),
            (r"^encoder\.layers\.(\d+)\.", r"encoder.blocks.\1."),
            (r"\.ff1\.pre_norm\.", ".ff1.norm."),
            (r"\.ff1\.up_proj\.", ".ff1.fc1."),
            (r"\.ff1\.down_proj\.", ".ff1.fc2."),
            (r"\.attn\.pre_norm\.", ".attn.norm."),
            (r"\.attn\.to_q\.", ".attn.to_q."),
            (r"\.attn\.to_k\.", ".attn.to_k."),
            (r"\.attn\.to_v\.", ".attn.to_v."),
            (r"\.attn\.to_out\.", ".attn.to_out."),
            (r"\.attn\.rel_pos_emb\.", ".attn.pos_emb."),
            (r"\.conv\.pre_norm\.", ".conv.norm."),
            (r"\.conv\.up_conv\.", ".conv.pointwise_conv1."),
            (r"\.conv\.depth_conv\.conv\.", ".conv.depthwise_conv."),
            (r"\.conv\.depth_conv\.batch_norm\.", ".conv.batch_norm."),
            (r"\.conv\.down_conv\.", ".conv.pointwise_conv2."),
            (r"\.ff2\.pre_norm\.", ".ff2.norm."),
            (r"\.ff2\.up_proj\.", ".ff2.fc1."),
            (r"\.ff2\.down_proj\.", ".ff2.fc2."),
            (r"\.post_norm\.", ".post_norm."),
        ]

        # Projector mappings
        projector_map = [
            (r"^projector\.query$", "projector.query_embeds"),
            (r"^projector\.qformer\.encoder\.layer\.(\d+)\.attention\.attention\.query\.", r"projector.layers.\1.self_attention.query."),
            (r"^projector\.qformer\.encoder\.layer\.(\d+)\.attention\.attention\.key\.", r"projector.layers.\1.self_attention.key."),
            (r"^projector\.qformer\.encoder\.layer\.(\d+)\.attention\.attention\.value\.", r"projector.layers.\1.self_attention.value."),
            (r"^projector\.qformer\.encoder\.layer\.(\d+)\.attention\.output\.dense\.", r"projector.layers.\1.self_attention_output.dense."),
            (r"^projector\.qformer\.encoder\.layer\.(\d+)\.attention\.output\.LayerNorm\.", r"projector.layers.\1.self_attention_output.LayerNorm."),
            (r"^projector\.qformer\.encoder\.layer\.(\d+)\.crossattention\.attention\.query\.", r"projector.layers.\1.cross_attention.query."),
            (r"^projector\.qformer\.encoder\.layer\.(\d+)\.crossattention\.attention\.key\.", r"projector.layers.\1.cross_attention.key."),
            (r"^projector\.qformer\.encoder\.layer\.(\d+)\.crossattention\.attention\.value\.", r"projector.layers.\1.cross_attention.value."),
            (r"^projector\.qformer\.encoder\.layer\.(\d+)\.crossattention\.output\.dense\.", r"projector.layers.\1.cross_attention_output.dense."),
            (r"^projector\.qformer\.encoder\.layer\.(\d+)\.crossattention\.output\.LayerNorm\.", r"projector.layers.\1.cross_attention_output.LayerNorm."),
            (r"^projector\.qformer\.encoder\.layer\.(\d+)\.intermediate_query\.dense\.", r"projector.layers.\1.feed_forward.dense_in."),
            (r"^projector\.qformer\.encoder\.layer\.(\d+)\.output_query\.dense\.", r"projector.layers.\1.feed_forward.dense_out."),
            (r"^projector\.qformer\.encoder\.layer\.(\d+)\.output_query\.LayerNorm\.", r"projector.layers.\1.feed_forward.LayerNorm."),
            (r"^projector\.linear\.", "projector.output_proj."),
        ]

        # Decoder mappings
        decoder_map = [
            (r"^language_model\.lm_head\.", "lm_head."),
            (r"^language_model\.model\.embed_tokens\.", "decoder.embedding."),
            (r"^language_model\.model\.norm\.", "decoder.dec_norm."),
            (r"^language_model\.model\.layers\.", "decoder.layers."),
            (r"\.self_attn\.q_proj\.", ".attn.in_proj.query."),
            (r"\.self_attn\.k_proj\.", ".attn.in_proj.key."),
            (r"\.self_attn\.v_proj\.", ".attn.in_proj.value."),
            (r"\.self_attn\.o_proj\.", ".attn.dense."),
            (r"\.mlp\.gate_proj\.", ".ff_sub_layer.wg."),
            (r"\.mlp\.up_proj\.", ".ff_sub_layer.w1."),
            (r"\.mlp\.down_proj\.", ".ff_sub_layer.w2."),
            (r"\.input_layernorm\.", ".ln."),
            (r"\.post_attention_layernorm\.", ".ff_ln."),
        ]

        all_maps = encoder_map + projector_map + decoder_map

        converted_count = 0
        skipped = []

        for hf_name, tensor in hf_sd.items():
            fms_name = hf_name
            for pattern, replacement in all_maps:
                fms_name = re.sub(pattern, replacement, fms_name)

            if fms_name != hf_name:
                converted_count += 1

            fms_sd[fms_name] = tensor

        print(f"  Converted {converted_count} weight names")
        return fms_sd

    converted_sd = convert_hf_to_fms(hf_state_dict)

    # Check which weights match
    print(f"\n[Checking weight compatibility]")
    matched = 0
    missing_in_fms = []
    missing_in_hf = []
    shape_mismatch = []

    for name in fms_state_dict.keys():
        if name in converted_sd:
            if fms_state_dict[name].shape == converted_sd[name].shape:
                matched += 1
            else:
                shape_mismatch.append((name, fms_state_dict[name].shape, converted_sd[name].shape))
        else:
            missing_in_hf.append(name)

    for name in converted_sd.keys():
        if name not in fms_state_dict:
            missing_in_fms.append(name)

    print(f"  Matched: {matched}/{len(fms_state_dict)}")

    if shape_mismatch:
        print(f"\n  Shape mismatches ({len(shape_mismatch)}):")
        for name, fms_shape, hf_shape in shape_mismatch[:5]:
            print(f"    {name}: FMS {fms_shape} vs HF {hf_shape}")

    if missing_in_hf:
        print(f"\n  Missing in HF ({len(missing_in_hf)}):")
        for name in missing_in_hf[:5]:
            print(f"    {name}")

    if missing_in_fms:
        print(f"\n  Extra in HF (not in FMS) ({len(missing_in_fms)}):")
        for name in missing_in_fms[:5]:
            print(f"    {name}")

    # Try loading weights
    print(f"\n[Loading weights into FMS model]")
    try:
        # Load with strict=False to see what loads
        result = fms_model.load_state_dict(converted_sd, strict=False)
        print(f"  Missing keys: {len(result.missing_keys)}")
        print(f"  Unexpected keys: {len(result.unexpected_keys)}")

        if result.missing_keys:
            print(f"  Sample missing: {result.missing_keys[:3]}")
        if result.unexpected_keys:
            print(f"  Sample unexpected: {result.unexpected_keys[:3]}")

        if matched > len(fms_state_dict) * 0.5:  # At least 50% matched
            print(f"\n✓ STEP 3 PARTIALLY PASSED: {matched}/{len(fms_state_dict)} weights loaded")
            return fms_model
        else:
            print(f"\n✗ STEP 3 FAILED: Only {matched}/{len(fms_state_dict)} weights matched")
            return None
    except Exception as e:
        print(f"  Error loading weights: {e}")
        return None


# ============================================================================
# Step 4: Compare FMS Output with HF
# ============================================================================

def test_step4_comparison(audio_path: str, model_id: str = "ibm-granite/granite-speech-3.3-8b"):
    """Compare FMS output with HuggingFace output."""
    print("\n" + "=" * 60)
    print("STEP 4: FMS vs HF Output Comparison")
    print("=" * 60)

    import librosa
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

    # Load audio
    print(f"\n[Loading audio: {audio_path}]")
    try:
        audio, sr = librosa.load(audio_path, sr=16000)
        print(f"  Duration: {len(audio)/sr:.2f}s, Sample rate: {sr}Hz")
    except Exception as e:
        print(f"  Error loading audio: {e}")
        return False

    # Load HF model
    print(f"\n[Loading HuggingFace model]")
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        hf_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.float32,  # Use float32 for comparison
            device_map="cpu",
        )
        hf_model.eval()
        print(f"  HF model loaded")
    except Exception as e:
        print(f"  Error loading HF model: {e}")
        return False

    # Process audio with HF
    print(f"\n[Processing with HuggingFace]")
    inputs = processor(audio=audio, sampling_rate=16000, return_tensors="pt")
    print(f"  input_features shape: {inputs['input_features'].shape}")

    # Get HF encoder output
    with torch.no_grad():
        hf_encoder_out = hf_model.encoder(inputs['input_features'])
        if hasattr(hf_encoder_out, 'last_hidden_state'):
            hf_encoder_out = hf_encoder_out.last_hidden_state
        print(f"  HF encoder output: {hf_encoder_out.shape}")

    # Load FMS model with weights
    print(f"\n[Loading FMS model with weights]")
    fms_model = test_step3_weight_loading(model_id)
    if fms_model is None:
        print("  Failed to load FMS model")
        return False
    fms_model.eval()

    # Get FMS encoder output
    print(f"\n[Processing with FMS]")
    with torch.no_grad():
        fms_encoder_out = fms_model.encoder(inputs['input_features'])
        print(f"  FMS encoder output: {fms_encoder_out.shape}")

    # Compare encoder outputs
    print(f"\n[Comparing Encoder Outputs]")
    if hf_encoder_out.shape == fms_encoder_out.shape:
        diff = (hf_encoder_out - fms_encoder_out).abs()
        print(f"  Shape match: ✓")
        print(f"  Max difference: {diff.max().item():.6f}")
        print(f"  Mean difference: {diff.mean().item():.6f}")

        if diff.max().item() < 1e-3:
            print(f"  ✓ Encoder outputs match!")
        else:
            print(f"  ✗ Encoder outputs differ significantly")
    else:
        print(f"  Shape mismatch: HF {hf_encoder_out.shape} vs FMS {fms_encoder_out.shape}")

    # Compare projector outputs
    print(f"\n[Comparing Projector Outputs]")
    with torch.no_grad():
        hf_projected = hf_model.projector(hf_encoder_out)
        if hasattr(hf_projected, 'last_hidden_state'):
            hf_projected = hf_projected.last_hidden_state
        fms_projected = fms_model.projector(fms_encoder_out)

        print(f"  HF projector output: {hf_projected.shape}")
        print(f"  FMS projector output: {fms_projected.shape}")

        if hf_projected.shape == fms_projected.shape:
            diff = (hf_projected - fms_projected).abs()
            print(f"  Max difference: {diff.max().item():.6f}")
            print(f"  Mean difference: {diff.mean().item():.6f}")

    # Generate text with HF
    print(f"\n[Generating text with HuggingFace]")
    with torch.no_grad():
        generated_ids = hf_model.generate(**inputs, max_new_tokens=50)
        hf_transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"  HF output: {hf_transcription}")

    print(f"\n✓ STEP 4 COMPLETED")
    return True


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Granite Speech E2E Test")
    parser.add_argument("--step", type=int, choices=[2, 3, 4], help="Run specific step")
    parser.add_argument("--audio", type=str, help="Path to audio file (required for step 4)")
    parser.add_argument("--model", type=str, default="ibm-granite/granite-speech-3.3-8b", help="HF model ID")
    args = parser.parse_args()

    if args.step == 2 or args.step is None:
        test_step2_shapes()

    if args.step == 3 or args.step is None:
        test_step3_weight_loading(args.model)

    if args.step == 4:
        if not args.audio:
            print("Error: --audio is required for step 4")
            return
        test_step4_comparison(args.audio, args.model)

    if args.step is None:
        print("\n" + "=" * 60)
        print("To run full comparison with audio:")
        print(f"  python {__file__} --step 4 --audio your_audio.wav")
        print("=" * 60)


if __name__ == "__main__":
    main()
