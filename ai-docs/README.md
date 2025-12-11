# AI Documentation for Granite-Speech Migration

## Purpose
This directory contains comparison documents for validating the migration of `granite-speech` model from HuggingFace Transformers to FMS (Foundation Model Stack).

## Documents

| File | Description | Status |
|------|-------------|--------|
| `conformer_attention_comparison.md` | ConformerAttention vs GraniteSpeechConformerAttention | NO DISCREPANCIES |
| `conformer_feedforward_comparison.md` | ConformerFeedForward vs GraniteSpeechConformerFeedForward | NO DISCREPANCIES |
| `conformer_conv_comparison.md` | ConformerConvModule vs GraniteSpeechConformerConvModule | NO DISCREPANCIES |
| `conformer_block_comparison.md` | ConformerBlock vs GraniteSpeechConformerBlock | NO DISCREPANCIES |
| `conformer_encoder_comparison.md` | ConformerEncoder vs GraniteSpeechCTCEncoder | DISCREPANCIES FOUND |
| `projector_comparison.md` | SpeechProjector vs GraniteSpeechEncoderProjector (Blip2QFormer) | **CRITICAL DISCREPANCIES** |
| `granite_speech_model_comparison.md` | GraniteSpeech vs GraniteSpeechForConditionalGeneration | DISCREPANCIES FOUND |
| `processor_comparison.md` | GraniteSpeechProcessor vs GraniteSpeechProcessor | MINOR DISCREPANCIES |
| `feature_extractor_comparison.md` | GraniteSpeechFeatureExtractor vs GraniteSpeechFeatureExtractor | MINOR DISCREPANCIES |
| `configuration_comparison.md` | All Config classes (Encoder, Projector, Main) | **CRITICAL DISCREPANCIES** |

## Reference Paths

### FMS (Target)
- Base: `~/src/foundation-model-stack/`
- Conformer: `fms/models/conformer.py`
- Projector: `fms/modules/projector.py`
- Granite Speech: `fms/models/granite_speech.py`

### HuggingFace (Source)
- Base: `~/src/transformers/`
- Modeling: `src/transformers/models/granite_speech/modeling_granite_speech.py`
- Config: `src/transformers/models/granite_speech/configuration_granite_speech.py`
- Blip2 Q-Former: `src/transformers/models/blip_2/modeling_blip_2.py`

---

## Migration Validation Summary

### Components Checked

| Component | FMS Location | HF Location | Result |
|-----------|--------------|-------------|--------|
| ConformerAttention | `conformer.py:146-282` | `modeling_granite_speech.py:119-182` | IDENTICAL |
| ConformerFeedForward | `conformer.py:81-143` | `modeling_granite_speech.py:99-116` | IDENTICAL |
| ConformerConvModule | `conformer.py:285-414` | `modeling_granite_speech.py:185-230` | IDENTICAL |
| ConformerBlock | `conformer.py:417-511` | `modeling_granite_speech.py:233-250` | IDENTICAL |
| ConformerEncoder | `conformer.py:514-646` | `modeling_granite_speech.py:253-279` | DISCREPANCIES |
| SpeechProjector | `projector.py:396-587` | `modeling_granite_speech.py:64-95` + `blip_2/` | **CRITICAL** |
| GraniteSpeech | `granite_speech.py:173-623` | `modeling_granite_speech.py:303-544` | DISCREPANCIES |
| GraniteSpeechProcessor | `granite_speech.py:1165-1314` | `processing_granite_speech.py:32-96` | MINOR |
| GraniteSpeechFeatureExtractor | `granite_speech.py:935-1163` | `feature_extraction_granite_speech.py:38-186` | MINOR |
| ConformerConfig | `conformer.py:22-79` | `configuration_granite_speech.py:21-105` | **CRITICAL** |
| SpeechProjectorConfig | `projector.py:31-96` | `Blip2QFormerConfig` (blip_2) | **CRITICAL** |
| GraniteSpeechConfig | `granite_speech.py:115-166` | `configuration_granite_speech.py:107-196` | **CRITICAL** |

---

## Discrepancy Summary

### ConformerEncoder (MEDIUM)

| Issue | Severity | Description |
|-------|----------|-------------|
| Clone in mid-layer CTC | MEDIUM | HF clones hidden_states before CTC, FMS does not |
| Buffer persistence | LOW | FMS saves attention_dists in state_dict, HF does not |
| CTC conditional flag | LOW | FMS has use_ctc flag, HF always applies CTC |

### SpeechProjector (CRITICAL)

| Issue | Severity | Description |
|-------|----------|-------------|
| Cross-attention K/V dimension | **CRITICAL** | HF uses `encoder_hidden_size` for K/V, FMS uses `encoder_dim` |
| Cross-attention frequency | **CRITICAL** | HF has configurable frequency, FMS has every layer |
| Query-specific FFN | **CRITICAL** | HF has separate `intermediate_query`/`output_query`, FMS has single FFN |

### GraniteSpeech Full Model (MEDIUM)

| Issue | Severity | Description |
|-------|----------|-------------|
| Decoder structure | MEDIUM | HF uses AutoModelForCausalLM, FMS uses GraniteHeadless + lm_head |
| Config naming | LOW | `audio_token_id` (HF) vs `audio_token_index` (FMS) |
| Loss computation | MEDIUM | HF handles attention mask in loss, FMS doesn't |
| Forward return type | MEDIUM | HF returns dataclass, FMS returns tuple |

### GraniteSpeechProcessor (MINOR)

| Issue | Severity | Description |
|-------|----------|-------------|
| No ProcessorMixin | MEDIUM | FMS doesn't inherit from ProcessorMixin (missing save/load methods) |
| Return type | LOW | HF returns `BatchFeature`, FMS returns plain `dict` |
| Missing chat_template | LOW | FMS doesn't support chat_template parameter |

### GraniteSpeechFeatureExtractor (MINOR)

| Issue | Severity | Description |
|-------|----------|-------------|
| No FeatureExtractionMixin | MEDIUM | FMS missing save/load/serialization methods |
| Return type | LOW | HF returns `BatchFeature`, FMS returns plain `dict` |
| Core algorithms | NONE | Mel extraction, length calc, mask creation are **IDENTICAL** |

### Configuration Classes (CRITICAL)

| Issue | Severity | Description |
|-------|----------|-------------|
| `num_layers` default | **CRITICAL** | HF=10, FMS=16 - model architecture mismatch |
| `audio_token_index` default | **CRITICAL** | HF=49155, FMS=49159 - will break inference |
| `input_dim` vs `num_features` | MEDIUM | Different parameter naming |
| `text_config` vs `decoder_config` | MEDIUM | Different parameter naming |
| Missing projector params | **CRITICAL** | `cross_attention_frequency`, `encoder_hidden_size` not in FMS |

---

## Priority Actions

### HIGH PRIORITY - Projector Architectural Changes Required

1. **Add `encoder_hidden_size` config** for cross-attention K/V projections
2. **Add `cross_attention_frequency` config** and conditional cross-attention creation
3. **Add separate `intermediate_query` and `output_query`** FFN modules for query tokens
4. **Update weight name mapping** in `_hf_to_fms_names`

### HIGH PRIORITY - Configuration Fixes Required

1. **Fix `num_layers` default** in ConformerConfig (16 -> 10 to match HF default, or make model-specific)
2. **Fix `audio_token_index` default** in GraniteSpeechConfig (49159 -> 49155 to match HF)
3. **Rename `num_features` to `input_dim`** in ConformerConfig for consistency
4. **Rename `decoder_config` to `text_config`** in GraniteSpeechConfig for consistency

### MEDIUM PRIORITY - Encoder Fixes

1. **Add clone in mid-layer CTC** for gradient flow consistency
2. **Set buffer persistence=False** to match HF serialization

### LOW PRIORITY - Full Model Alignment

1. Config parameter naming alignment
2. Forward return type standardization (optional)

---

## Usage for AI Agents

When continuing this migration validation:

1. Read the relevant comparison document first
2. The Projector (`projector_comparison.md`) is the highest priority issue
3. Use the same format as existing documents for consistency
4. Update this README when adding new comparison documents

### Quick Reference: What's Identical vs Different

**IDENTICAL (no changes needed):**
- ConformerAttention
- ConformerFeedForward
- ConformerConvModule
- ConformerBlock
- GraniteSpeechProcessor (core logic identical, minor API differences)
- GraniteSpeechFeatureExtractor (mel extraction/length calc identical, minor API differences)

**NEEDS FIXES:**
- ConformerEncoder (minor fixes)
- SpeechProjector (major architectural changes)
- GraniteSpeech full model (medium fixes, mostly config/API)
