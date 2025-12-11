# Full Granite-Speech Migration Comparison Report

## Overview

This document provides a comprehensive comparison after code updates between HF and FMS implementations.

**Files Compared:**
- HF: `modeling_granite_speech.py`, `configuration_granite_speech.py`, `feature_extraction_granite_speech.py`, `processing_granite_speech.py`
- HF (dependency): `blip_2/modeling_blip_2.py` (Blip2QFormerModel)
- FMS: `conformer.py`, `projector.py`, `granite_speech.py`

---

## 1. Configuration Classes

### 1.1 GraniteSpeechEncoderConfig (HF) vs ConformerConfig (FMS)

| Parameter | HF Name | HF Default | FMS Name | FMS Default | Status |
|-----------|---------|------------|----------|-------------|--------|
| Input dimension | `input_dim` | 160 | `num_features` | 160 | **NAME DIFFERS** |
| Number of layers | `num_layers` | 10 | `num_layers` | 16 | **DEFAULT DIFFERS** |
| Hidden dimension | `hidden_dim` | 1024 | `hidden_dim` | 1024 | OK |
| FF multiplier | `feedforward_mult` | 4 | `feedforward_mult` | 4 | OK |
| Attention heads | `num_heads` | 8 | `num_heads` | 8 | OK |
| Head dimension | `dim_head` | 128 | `dim_head` | 128 | OK |
| CTC output dim | `output_dim` | 42 | `output_dim` | 42 | OK |
| Context size | `context_size` | 200 | `context_size` | 200 | OK |
| Max pos embedding | `max_pos_emb` | 512 | `max_pos_emb` | 512 | OK |
| Dropout | `dropout` | 0.1 | `dropout` | 0.1 | OK |
| Conv kernel size | `conv_kernel_size` | 15 | `conv_kernel_size` | 15 | OK |
| Conv expansion | `conv_expansion_factor` | 2 | `conv_expansion_factor` | 2 | OK |

**DISCREPANCIES:**
1. `input_dim` vs `num_features` - naming difference
2. `num_layers` default: HF=10, FMS=16 (but FMS `_default_encoder_config` uses 16)

### 1.2 GraniteSpeechConfig (HF) vs GraniteSpeechConfig (FMS)

| Parameter | HF Name | HF Default | FMS Name | FMS Default | Status |
|-----------|---------|------------|----------|-------------|--------|
| Text config | `text_config` | GraniteConfig | `decoder_config` | GraniteConfig | **NAME DIFFERS** |
| Encoder config | `encoder_config` | GraniteSpeechEncoderConfig | `encoder_config` | ConformerConfig | OK |
| Projector config | `projector_config` | Blip2QFormerConfig | `projector_config` | SpeechProjectorConfig | OK |
| Audio token ID | `audio_token_index` | 49155 | `audio_token_index` | 49155 | **OK (FIXED)** |
| Has LoRA adapter | `has_lora_adapter` | True | `has_lora_adapter` | True | OK |
| Downsample rate | `downsample_rate` | 5 | `downsample_rate` | 5 | OK |
| Window size | `window_size` | 15 | `window_size` | 15 | OK |
| Initializer range | `initializer_range` | 0.02 | `initializer_range` | 0.02 | OK |

**DISCREPANCIES:**
1. `text_config` vs `decoder_config` - naming difference (intentional FMS convention)

---

## 2. Encoder (Conformer)

### 2.1 GraniteSpeechCTCEncoder (HF) vs ConformerEncoder (FMS)

#### Layer Structure

| Component | HF Name | FMS Name | Status |
|-----------|---------|----------|--------|
| Input projection | `input_linear` | `input_proj` | NAME DIFFERS |
| Conformer blocks | `layers` | `blocks` | NAME DIFFERS |
| CTC output | `out` | `out` | OK |
| CTC feedback | `out_mid` | `out_mid` | OK |
| Attention distances buffer | `attention_dists` | `attention_dists` | OK |

#### Forward Pass Comparison

| Step | HF (lines 270-279) | FMS | Status |
|------|---------------------|-----|--------|
| 1. Input projection | `hidden_states = self.input_linear(hidden_states)` | `x = self.input_proj(input_features)` | OK |
| 2. Layer loop | `for idx, layer in enumerate(self.layers, start=1)` | `for idx, block in enumerate(self.blocks, start=1)` | OK |
| 3. Mid-layer check | `if idx == self.num_layers // 2` | `if self.config.use_ctc and self.out is not None and idx == mid_layer` | **EXTRA CONDITION** |
| 4. Clone | `hidden_states_mid = hidden_states.clone()` | No clone | **DISCREPANCY** |
| 5. CTC projection | `hidden_states_mid = self.out(hidden_states_mid)` | `x_mid = self.out(x)` | OK |
| 6. Softmax + feedback | `hidden_states += self.out_mid(nn.Softmax(dim=-1)(hidden_states_mid))` | `x = x + self.out_mid(F.softmax(x_mid, dim=-1))` | OK |

**REMAINING DISCREPANCIES:**
1. **Clone behavior**: HF clones before CTC, FMS does not
   - Impact: None for inference (mathematically equivalent)
2. **use_ctc conditional**: FMS has extra flag check, HF always applies CTC
3. **Buffer persistence**: HF uses `persistent=False`, FMS uses default (True)

### 2.2 Conformer Sub-components

#### GraniteSpeechConformerAttention vs ConformerAttention

| Aspect | Status |
|--------|--------|
| Layer structure | IDENTICAL |
| Q/K/V projections | IDENTICAL |
| Positional embedding | IDENTICAL |
| Shaw's relative attention | IDENTICAL |
| Masking logic | IDENTICAL |

**STATUS: NO DISCREPANCIES**

#### GraniteSpeechConformerFeedForward vs ConformerFeedForward

| Aspect | Status |
|--------|--------|
| Layer structure | IDENTICAL |
| Forward computation | IDENTICAL |

**STATUS: NO DISCREPANCIES**

#### GraniteSpeechConformerConvModule vs ConformerConvModule

| Aspect | Status |
|--------|--------|
| Layer structure | IDENTICAL |
| Padding logic | IDENTICAL (for default kernel_size=15) |
| GLU implementation | EQUIVALENT (nn.GLU vs manual chunk+sigmoid) |

**STATUS: NO DISCREPANCIES**

#### GraniteSpeechConformerBlock vs ConformerBlock

| Aspect | Status |
|--------|--------|
| Layer order | IDENTICAL |
| Residual scaling | IDENTICAL (0.5 for FF, 1.0 for attn/conv) |
| Post normalization | IDENTICAL |

**STATUS: NO DISCREPANCIES**

---

## 3. Projector (Q-Former)

### 3.1 GraniteSpeechEncoderProjector (HF) vs SpeechProjector (FMS)

HF uses `AutoModel.from_config(config.projector_config)` which loads `Blip2QFormerModel`.

#### Architecture Comparison

| Component | HF (via Blip2QFormer) | FMS | Status |
|-----------|----------------------|-----|--------|
| Query embeddings | `self.query` (Parameter) | `self.query_embeds` (Parameter) | NAME OK |
| Query shape | `(1, num_queries, hidden_size)` | `(1, num_queries, encoder_dim)` | OK |
| Query init | `N(0, 1)` | `N(0, 1)` | OK |
| Input LayerNorm | `self.qformer.layernorm` | `self.input_layernorm` | OK |
| Input Dropout | `self.qformer.dropout` | `self.input_dropout` | OK |
| Encoder layers | `self.qformer.encoder.layer` | `self.layers` | OK |
| Output projection | `self.linear` | `self.output_proj` | NAME OK |

#### Q-Former Layer Architecture

| Component | HF (Blip2QFormerLayer) | FMS (QFormerLayer) | Status |
|-----------|------------------------|---------------------|--------|
| Self-attention | `self.attention` | `self.self_attention` | OK |
| Self-attention output | `self.attention.output` | `self.self_attention_output` | OK |
| Cross-attention | `self.crossattention` (conditional) | `self.cross_attention` (conditional) | OK |
| Cross-attention output | `self.crossattention.output` | `self.cross_attention_output` | OK |
| Cross-attention frequency | `layer_idx % config.cross_attention_frequency == 0` | Same logic | **OK (FIXED)** |
| Query FFN intermediate | `self.intermediate_query` | `self.intermediate_query` | **OK (FIXED)** |
| Query FFN output | `self.output_query` | `self.output_query` | **OK (FIXED)** |

#### Cross-Attention K/V Dimensions

| Aspect | HF | FMS | Status |
|--------|-----|-----|--------|
| K/V input dimension | `config.encoder_hidden_size` | `config.encoder_hidden_size or config.encoder_dim` | **OK (FIXED)** |

**FMS now has `encoder_hidden_size` parameter in SpeechProjectorConfig.**

#### Forward Pass

| Step | HF | FMS | Status |
|------|-----|-----|--------|
| 1. Pad to window_size | `nn.functional.pad(...)` | `F.pad(...)` | OK |
| 2. Reshape to windows | `view(batch * nblocks, window_size, dim)` | Same | OK |
| 3. Apply LayerNorm | `self.layernorm(query_embeds)` | `self.input_layernorm(query_states)` | OK |
| 4. Apply dropout | `self.dropout(...)` | `self.input_dropout(...)` | OK |
| 5. Q-Former layers | `self.encoder(...)` | `for layer in self.layers: ...` | OK |
| 6. Reshape output | `view(batch, nblocks * num_queries, -1)` | Same | OK |
| 7. Output projection | `self.linear(...)` | `self.output_proj(...)` | OK |

**STATUS: NO REMAINING DISCREPANCIES** (all previously identified issues fixed)

---

## 4. Main Model

### 4.1 GraniteSpeechForConditionalGeneration (HF) vs GraniteSpeech (FMS)

#### Model Structure

| Component | HF | FMS | Status |
|-----------|-----|-----|--------|
| Language model | `AutoModelForCausalLM` | `GraniteHeadless` + `lm_head` | STRUCTURAL DIFF |
| Encoder | `GraniteSpeechCTCEncoder` | `ConformerEncoder` | OK |
| Projector | `GraniteSpeechEncoderProjector` | `SpeechProjector` | OK |

#### Forward Signature

| Parameter | HF | FMS | Status |
|-----------|-----|-----|--------|
| input_ids | Yes | Yes | OK |
| input_features | Yes | Yes | OK |
| input_features_mask | Yes | Yes | OK |
| attention_mask | Yes | Yes | OK |
| position_ids | Yes | Yes | OK |
| past_key_values | Yes | Yes | OK |
| inputs_embeds | Yes | Yes | OK |
| labels | Yes | Yes | OK |
| use_cache | Yes | Yes | OK |
| output_attentions | Yes | No | FMS MISSING |
| output_hidden_states | Yes | No | FMS MISSING |
| return_dict | Yes | No | FMS MISSING |
| cache_position | Yes | No | FMS MISSING |
| logits_to_keep | Yes | No | FMS MISSING |

**These are intentionally missing in FMS (FMS convention returns tuples)**

#### get_merged_audio_embeddings Comparison

| Step | HF | FMS | Status |
|------|-----|-----|--------|
| 1. Find audio positions | `is_audio_idx = input_ids == self.config.audio_token_id` | `audio_pos = (input_ids == self.audio_token_index)` | OK |
| 2. Safe input_ids | `llm_input_ids = input_ids.clone(); llm_input_ids[is_audio_idx] = 0` | `safe_ids = torch.where(audio_pos, input_ids.new_zeros(()), input_ids)` | EQUIVALENT |
| 3. Get embeddings | `inputs_embeds = self.language_model.get_input_embeddings()(llm_input_ids)` | `token_embeds = self.get_input_embeddings()(safe_ids)` | OK |
| 4. Create mask | `special_audio_mask = is_audio_idx.unsqueeze(-1)` | `mask = audio_pos.unsqueeze(-1)` | OK |
| 5. Scatter audio | `inputs_embeds.masked_scatter(special_audio_mask, audio_features)` | `merged = token_embeds.masked_scatter(mask, audio_flat)` | OK |

**STATUS: EQUIVALENT LOGIC**

#### Loss Computation

| Aspect | HF | FMS | Status |
|--------|-----|-----|--------|
| Shift logits/labels | Yes | Yes | OK |
| Attention mask filtering | Yes (if attention_mask provided) | No | **DISCREPANCY** |
| NaN handling | No | Yes (`torch.nan_to_num`) | FMS EXTRA |
| Loss function | `nn.CrossEntropyLoss()` | Same | OK |

**REMAINING DISCREPANCY:**
- HF filters loss computation by attention_mask
- FMS relies on `labels=-100` for ignore (standard practice)
- **Impact**: FMS requires proper label masking; HF can use attention_mask

---

## 5. Feature Extractor

### 5.1 GraniteSpeechFeatureExtractor Comparison

| Aspect | HF | FMS | Status |
|--------|-----|-----|--------|
| Constructor params | All defaults identical | All defaults identical | OK |
| Mel extraction algorithm | Identical | Identical | OK |
| Length calculation | Identical formula | Identical formula | OK |
| Mask creation | `torch.arange(...) < torch.tensor(...)` | Same | OK |
| Return type | `BatchFeature` | `dict` | MINOR |
| Base class | `FeatureExtractionMixin` | None | MINOR |

**STATUS: FUNCTIONALLY IDENTICAL**

---

## 6. Processor

### 6.1 GraniteSpeechProcessor Comparison

| Aspect | HF | FMS | Status |
|--------|-----|-----|--------|
| Audio token expansion | Identical algorithm | Identical algorithm | OK |
| Text validation | `isinstance(text[0], str)` | `len(text) > 0 and isinstance(text[0], str)` | FMS MORE ROBUST |
| Return type | `BatchFeature` | `dict` | MINOR |
| Base class | `ProcessorMixin` | None | MINOR |
| chat_template | Supported | Not supported | MINOR |

**STATUS: FUNCTIONALLY IDENTICAL** for core processing

---

## Summary of Remaining Discrepancies

### Critical (Must Fix for Training Parity)

None remaining after updates.

### Medium (May Affect Behavior)

| Issue | Component | Description | Impact |
|-------|-----------|-------------|--------|
| Clone in CTC | Encoder | HF clones before CTC, FMS doesn't | None for inference; minor gradient difference in training |
| Loss attention_mask | Main Model | HF uses attention_mask in loss, FMS doesn't | Use `labels=-100` in FMS for same behavior |

### Low (Naming/Convention)

| Issue | Component | Description |
|-------|-----------|-------------|
| `input_dim` vs `num_features` | Config | Parameter naming |
| `text_config` vs `decoder_config` | Config | Parameter naming |
| `input_linear` vs `input_proj` | Encoder | Layer naming |
| `layers` vs `blocks` | Encoder | Layer naming |
| Return types | All | `BatchFeature` vs `dict` |
| Base classes | Processor/Extractor | Missing HF mixins |

### FMS-Only Features

| Feature | Component | Description |
|---------|-----------|-------------|
| `use_ctc` flag | Encoder Config | Optional CTC disabling |
| `freeze_encoder/decoder` | Main Config | Training controls |
| NaN handling in loss | Main Model | `torch.nan_to_num` for stability |

---

## Weight Mapping (HF -> FMS)

The `_hf_to_fms_names` function in FMS handles all name conversions:

### Encoder Mappings
```
encoder.input_linear -> encoder.input_proj
encoder.layers.{i} -> encoder.blocks.{i}
.ff1.pre_norm. -> .ff1.norm.
.ff1.up_proj. -> .ff1.fc1.
.ff1.down_proj. -> .ff1.fc2.
.attn.pre_norm. -> .attn.norm.
.attn.rel_pos_emb. -> .attn.pos_emb.
.conv.up_conv. -> .conv.pointwise_conv1.
.conv.depth_conv.conv. -> .conv.depthwise_conv.
.conv.down_conv. -> .conv.pointwise_conv2.
.ff2.pre_norm. -> .ff2.norm.
.ff2.up_proj. -> .ff2.fc1.
.ff2.down_proj. -> .ff2.fc2.
```

### Projector Mappings
```
projector.query -> projector.query_embeds
projector.qformer.layernorm. -> projector.input_layernorm.
projector.qformer.encoder.layer.{i} -> projector.layers.{i}
.attention.attention.query. -> .self_attention.query.
.attention.attention.key. -> .self_attention.key.
.attention.attention.value. -> .self_attention.value.
.attention.output.dense. -> .self_attention_output.dense.
.attention.output.LayerNorm. -> .self_attention_output.LayerNorm.
.crossattention.attention.query. -> .cross_attention.query.
.crossattention.attention.key. -> .cross_attention.key.
.crossattention.attention.value. -> .cross_attention.value.
.crossattention.output.dense. -> .cross_attention_output.dense.
.crossattention.output.LayerNorm. -> .cross_attention_output.LayerNorm.
.intermediate_query.dense. -> .intermediate_query.dense.
.output_query.dense. -> .output_query.dense.
.output_query.LayerNorm. -> .output_query.LayerNorm.
projector.linear. -> projector.output_proj.
```

### Decoder Mappings
```
language_model.lm_head.weight -> lm_head.weight
language_model.model.embed_tokens.weight -> decoder.embedding.weight
language_model.model.norm -> decoder.dec_norm
language_model.model.layers -> decoder.layers
self_attn.k_proj -> attn.in_proj.key
self_attn.v_proj -> attn.in_proj.value
self_attn.q_proj -> attn.in_proj.query
self_attn.o_proj -> attn.dense
mlp.gate_proj -> ff_sub_layer.wg
mlp.up_proj -> ff_sub_layer.w1
mlp.down_proj -> ff_sub_layer.w2
(decoder.layers.{i}.)input_layernorm -> \1ln
(decoder.layers.{i}.)post_attention_layernorm -> \1ff_ln
```

---

## Conclusion

**The FMS implementation is now architecturally aligned with HF** after the updates.

### What's Identical:
- Conformer encoder architecture (all sub-components)
- Projector (Q-Former) architecture including:
  - Cross-attention frequency
  - encoder_hidden_size for K/V
  - Separate intermediate_query/output_query FFN
- Feature extraction (mel-spectrogram computation)
- Processor (audio token expansion)
- Configuration defaults (audio_token_index=49155)

### Minor Remaining Differences:
- Clone in mid-layer CTC (inference equivalent)
- Loss attention_mask handling (use labels=-100 in FMS)
- Return types (dict vs BatchFeature)
- Missing HF base classes (no save/load methods)

### For Inference:
Models should produce **identical outputs** given the same weights.

### For Training:
Ensure:
1. Use `labels=-100` for padding/ignored positions
2. The clone difference in CTC shouldn't significantly affect gradients
