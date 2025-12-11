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

---

## Priority Actions

### HIGH PRIORITY - Projector Architectural Changes Required

1. **Add `encoder_hidden_size` config** for cross-attention K/V projections
2. **Add `cross_attention_frequency` config** and conditional cross-attention creation
3. **Add separate `intermediate_query` and `output_query`** FFN modules for query tokens
4. **Update weight name mapping** in `_hf_to_fms_names`

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

**NEEDS FIXES:**
- ConformerEncoder (minor fixes)
- SpeechProjector (major architectural changes)
- GraniteSpeech full model (medium fixes, mostly config/API)
