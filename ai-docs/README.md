# AI Documentation for Granite-Speech Migration

## Purpose
This directory contains comparison documents for validating the migration of `granite-speech` model from HuggingFace Transformers to FMS (Foundation Model Stack).

## Latest Status

**MIGRATION STATUS: COMPLETE** (as of latest code update)

The FMS implementation is now architecturally aligned with HF. See `full_comparison_report.md` for comprehensive details.

## Documents

| File | Description | Status |
|------|-------------|--------|
| **`full_comparison_report.md`** | **LATEST: Complete re-comparison after code updates** | **SEE THIS FIRST** |
| `conformer_attention_comparison.md` | ConformerAttention vs GraniteSpeechConformerAttention | NO DISCREPANCIES |
| `conformer_feedforward_comparison.md` | ConformerFeedForward vs GraniteSpeechConformerFeedForward | NO DISCREPANCIES |
| `conformer_conv_comparison.md` | ConformerConvModule vs GraniteSpeechConformerConvModule | NO DISCREPANCIES |
| `conformer_block_comparison.md` | ConformerBlock vs GraniteSpeechConformerBlock | NO DISCREPANCIES |
| `conformer_encoder_comparison.md` | ConformerEncoder vs GraniteSpeechCTCEncoder | MINOR (clone, buffer) |
| `projector_comparison.md` | SpeechProjector vs GraniteSpeechEncoderProjector | **FIXED** |
| `granite_speech_model_comparison.md` | GraniteSpeech vs GraniteSpeechForConditionalGeneration | MINOR |
| `processor_comparison.md` | GraniteSpeechProcessor vs GraniteSpeechProcessor | MINOR |
| `feature_extractor_comparison.md` | GraniteSpeechFeatureExtractor vs GraniteSpeechFeatureExtractor | MINOR |
| `configuration_comparison.md` | All Config classes | **FIXED** |

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
- Feature Extraction: `src/transformers/models/granite_speech/feature_extraction_granite_speech.py`
- Processing: `src/transformers/models/granite_speech/processing_granite_speech.py`
- Blip2 Q-Former: `src/transformers/models/blip_2/modeling_blip_2.py`

---

## Summary After Code Updates

### What's Now Identical (No Discrepancies):
- ConformerAttention
- ConformerFeedForward
- ConformerConvModule
- ConformerBlock
- SpeechProjector (Q-Former) - including:
  - Cross-attention frequency
  - encoder_hidden_size for K/V projections
  - Separate intermediate_query/output_query FFN
- GraniteSpeechFeatureExtractor (core algorithms)
- GraniteSpeechProcessor (core logic)
- Configuration defaults (audio_token_index=49155)

### Minor Remaining Differences (Acceptable):

| Issue | Component | Impact |
|-------|-----------|--------|
| Clone in CTC | Encoder | None for inference |
| Loss attention_mask | Model | Use labels=-100 |
| Return types | All | dict vs BatchFeature |
| Buffer persistence | Encoder | Serialization only |
| Missing HF mixins | Processor/Extractor | No save/load methods |

### For Inference:
Models produce **identical outputs** given the same weights.

### For Training:
Use `labels=-100` for padding/ignored positions instead of attention_mask filtering.

---

## Components Checked

| Component | FMS Location | HF Location | Result |
|-----------|--------------|-------------|--------|
| ConformerAttention | `conformer.py` | `modeling_granite_speech.py:119-182` | IDENTICAL |
| ConformerFeedForward | `conformer.py` | `modeling_granite_speech.py:99-116` | IDENTICAL |
| ConformerConvModule | `conformer.py` | `modeling_granite_speech.py:185-230` | IDENTICAL |
| ConformerBlock | `conformer.py` | `modeling_granite_speech.py:233-250` | IDENTICAL |
| ConformerEncoder | `conformer.py` | `modeling_granite_speech.py:253-279` | MINOR |
| SpeechProjector | `projector.py` | `modeling_granite_speech.py:64-95` + `blip_2/` | **FIXED** |
| GraniteSpeech | `granite_speech.py` | `modeling_granite_speech.py:303-544` | MINOR |
| GraniteSpeechProcessor | `granite_speech.py` | `processing_granite_speech.py` | IDENTICAL (core) |
| GraniteSpeechFeatureExtractor | `granite_speech.py` | `feature_extraction_granite_speech.py` | IDENTICAL (core) |
| GraniteSpeechConfig | `granite_speech.py` | `configuration_granite_speech.py` | **FIXED** |
| SpeechProjectorConfig | `projector.py` | `Blip2QFormerConfig` | **FIXED** |

---

## Usage for AI Agents

When working on this migration:

1. **Start with `full_comparison_report.md`** - most comprehensive and up-to-date
2. Individual comparison files are kept for reference but may be outdated
3. Weight mapping is documented in `full_comparison_report.md` under "Weight Mapping (HF -> FMS)"
4. The `_hf_to_fms_names` function in `granite_speech.py` handles all name conversions

### Quick Reference: What's Identical vs Different

**IDENTICAL (no changes needed):**
- ConformerAttention
- ConformerFeedForward
- ConformerConvModule
- ConformerBlock
- SpeechProjector (Q-Former) - **FIXED**
- GraniteSpeechProcessor (core logic)
- GraniteSpeechFeatureExtractor (mel extraction/length calc)
- Configuration defaults - **FIXED**

**MINOR DIFFERENCES (acceptable):**
- ConformerEncoder (clone behavior, buffer persistence)
- GraniteSpeech full model (return types, loss handling)
- Return types (dict vs BatchFeature throughout)
