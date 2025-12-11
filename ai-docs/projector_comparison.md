# Projector Component Comparison

## Purpose
This document compares `SpeechProjector` (FMS) with `GraniteSpeechEncoderProjector` (HF) for migration validation.

## File Locations
- **FMS**: `fms/modules/projector.py` (SpeechProjector class, lines 396-587)
- **HF**: `modeling_granite_speech.py` lines 64-95 (GraniteSpeechEncoderProjector)
- **HF Q-Former**: `models/blip_2/modeling_blip_2.py` (Blip2QFormerModel, used via AutoModel)

## Architecture Overview

### HF Architecture
HF uses a **thin wrapper** around `Blip2QFormerModel`:
```python
class GraniteSpeechEncoderProjector(nn.Module):
    self.query = nn.Parameter(...)  # learnable queries
    self.qformer = AutoModel.from_config(config.projector_config)  # Blip2QFormerModel
    self.linear = nn.Linear(...)  # output projection
```

### FMS Architecture
FMS implements a **custom Q-Former** from scratch:
```python
class SpeechProjector(nn.Module):
    self.query_embeds = nn.Parameter(...)  # learnable queries
    self.input_layernorm = nn.LayerNorm(...)
    self.input_dropout = nn.Dropout(...)
    self.layers = nn.ModuleList([QFormerLayer(...)])  # custom Q-Former layers
    self.output_proj = nn.Linear(...)  # output projection
```

---

## CRITICAL DISCREPANCIES

### 1. Cross-Attention Key/Value Dimension (CRITICAL)

| Aspect | FMS | HF (Blip2QFormer) |
|--------|-----|-------------------|
| Cross-attn K/V input | `encoder_dim` (1024) | `encoder_hidden_size` (configurable, typically different) |

**HF Blip2QFormerMultiHeadAttention (lines 536-541):**
```python
if is_cross_attention:
    self.key = nn.Linear(config.encoder_hidden_size, self.all_head_size)
    self.value = nn.Linear(config.encoder_hidden_size, self.all_head_size)
else:
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
```

**FMS QFormerCrossAttention (lines 209-211):**
```python
self.query = nn.Linear(config.encoder_dim, self.all_head_size)
self.key = nn.Linear(config.encoder_dim, self.all_head_size)
self.value = nn.Linear(config.encoder_dim, self.all_head_size)
```

**Impact**:
- HF uses `encoder_hidden_size` for cross-attention K/V projections
- FMS uses `encoder_dim` for all projections
- If `encoder_hidden_size != encoder_dim`, weight shapes will mismatch

**ACTION REQUIRED**: Verify `encoder_hidden_size` in HF config matches FMS `encoder_dim`. If not, FMS cross-attention K/V Linear layers need to use a separate `encoder_hidden_size` parameter.

---

### 2. Cross-Attention Frequency (CRITICAL)

| Aspect | FMS | HF (Blip2QFormer) |
|--------|-----|-------------------|
| Cross-attention | Every layer | Configurable via `cross_attention_frequency` |

**HF Blip2QFormerLayer (lines 700-704):**
```python
if layer_idx % config.cross_attention_frequency == 0:
    self.crossattention = Blip2QFormerAttention(config, is_cross_attention=True)
    self.has_cross_attention = True
else:
    self.has_cross_attention = False
```

**FMS QFormerLayer:**
```python
# Cross-attention is always present
self.cross_attention = QFormerCrossAttention(config)
```

**Impact**:
- HF only adds cross-attention to every Nth layer (default N=2)
- FMS adds cross-attention to every layer
- Different number of cross-attention layers = different parameters

**ACTION REQUIRED**: Check HF `cross_attention_frequency` config. If != 1, FMS needs conditional cross-attention creation matching HF pattern.

---

### 3. Separate Feed-Forward for Queries (CRITICAL)

| Aspect | FMS | HF (Blip2QFormer) |
|--------|-----|-------------------|
| FFN for queries | Single shared FFN | Separate `intermediate_query` + `output_query` |

**HF Blip2QFormerLayer (lines 710-711):**
```python
self.intermediate_query = Blip2QFormerIntermediate(config)
self.output_query = Blip2QFormerOutput(config)
```

**HF Forward (lines 771-774):**
```python
def feed_forward_chunk_query(self, attention_output):
    intermediate_output = self.intermediate_query(attention_output)
    layer_output = self.output_query(intermediate_output, attention_output)
    return layer_output
```

**FMS QFormerLayer:**
```python
# Only one FFN shared for all
self.feed_forward = QFormerFeedForward(config)
```

**Impact**:
- HF has separate FFN parameters for query tokens (`intermediate_query`, `output_query`)
- FMS uses single shared FFN
- Different parameter count and weight names

**ACTION REQUIRED**: Add separate `intermediate_query` and `output_query` modules to FMS QFormerLayer matching HF structure.

---

### 4. Self-Attention Output Structure

| Aspect | FMS | HF (Blip2QFormer) |
|--------|-----|-------------------|
| Self-attn output | `QFormerAttentionOutput` | `Blip2QFormerSelfOutput` (same structure but different class) |

Both have identical structure:
```python
self.dense = nn.Linear(hidden_size, hidden_size)
self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
self.dropout = nn.Dropout(hidden_dropout_prob)
```

**Status**: IDENTICAL structure, different class names.

---

### 5. Q-Former Layer Order (DISCREPANCY)

| Step | FMS | HF |
|------|-----|-----|
| 1 | Self-attention | Self-attention (attention) |
| 2 | Self-attention output (dense + LN + residual) | Self-attention output (dense + dropout + LN + residual) |
| 3 | Cross-attention | Cross-attention (if has_cross_attention) |
| 4 | Cross-attention output | Cross-attention output |
| 5 | Feed-forward (dense_in + act + dense_out + dropout + LN + residual) | Feed-forward via chunking |

**FMS QFormerFeedForward (lines 316-331):**
```python
def forward(self, hidden_states):
    residual = hidden_states
    hidden_states = self.dense_in(hidden_states)
    hidden_states = self.activation(hidden_states)
    hidden_states = self.dense_out(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.LayerNorm(hidden_states + residual)
    return hidden_states
```

**HF Blip2QFormerIntermediate + Blip2QFormerOutput:**
```python
# Intermediate
hidden_states = self.dense(hidden_states)  # dense_in
hidden_states = self.intermediate_act_fn(hidden_states)  # activation
return hidden_states

# Output
hidden_states = self.dense(hidden_states)  # dense_out
hidden_states = self.dropout(hidden_states)
hidden_states = self.LayerNorm(hidden_states + input_tensor)  # residual
return hidden_states
```

**Status**: Structure is equivalent but split into two classes in HF.

---

### 6. Query Embedding Shape

| Aspect | FMS | HF |
|--------|-----|-----|
| Query shape | `(1, num_queries, encoder_dim)` | `(1, num_queries, hidden_size)` |
| Query initialization | `N(0, 1)` | `N(0, 1)` |

**FMS (line 473-477):**
```python
self.query_embeds = nn.Parameter(
    torch.zeros(1, self.num_queries, config.encoder_dim)
)
nn.init.normal_(self.query_embeds, mean=0.0, std=1.0)
```

**HF (lines 72-73):**
```python
self.query = nn.Parameter(torch.zeros(1, self.num_queries, config.projector_config.hidden_size))
self.query.data.normal_(mean=0.0, std=1.0)
```

**Note**: HF uses `projector_config.hidden_size` which should equal FMS `encoder_dim`. Verify config mapping.

---

### 7. Input LayerNorm Application

| Aspect | FMS | HF |
|--------|-----|-----|
| LayerNorm on queries | Applied in SpeechProjector.forward() | Applied in Blip2QFormerModel.forward() |

**FMS (lines 568-570):**
```python
query_states = self.input_layernorm(query_states)
query_states = self.input_dropout(query_states)
```

**HF Blip2QFormerModel (lines 965-967):**
```python
query_embeds = query_embeds.to(self.layernorm.weight.dtype)
embedding_output = self.layernorm(query_embeds)
embedding_output = self.dropout(embedding_output)
```

**Status**: IDENTICAL behavior.

---

### 8. Output Projection

| Aspect | FMS | HF |
|--------|-----|-----|
| Output projection | `nn.Linear(encoder_dim, decoder_dim)` | `nn.Linear(hidden_size, text_hidden_size)` |

**FMS (line 490):**
```python
self.output_proj = nn.Linear(config.encoder_dim, config.decoder_dim)
```

**HF (line 77):**
```python
self.linear = nn.Linear(config.projector_config.hidden_size, config.text_config.hidden_size)
```

**Status**: IDENTICAL structure, different naming.

---

### 9. Forward Pass Window Processing

| Step | FMS | HF |
|------|-----|-----|
| 1 | Calculate nblocks | Calculate nblocks |
| 2 | Pad to window_size multiple | Pad to window_size multiple |
| 3 | Reshape to (batch*nblocks, window_size, dim) | Reshape to (batch*nblocks, window_size, dim) |
| 4 | Expand queries | Pass query to qformer |
| 5 | Apply input LN + dropout | qformer applies LN + dropout |
| 6 | Pass through Q-Former layers | qformer.encoder processes |
| 7 | Reshape output | Reshape output |
| 8 | Apply output projection | Apply output projection |

**FMS (lines 551-586):**
```python
nblocks = math.ceil(seq_len / self.window_size)
pad = nblocks * self.window_size - seq_len
encoder_hidden_states = F.pad(encoder_hidden_states, (0, 0, 0, pad), "constant", 0)
encoder_hidden_states = encoder_hidden_states.view(batch_size * nblocks, self.window_size, dim)
# ... process through layers ...
query_states = query_states.view(batch_size, nblocks * self.num_queries, -1)
projected_states = self.output_proj(query_states)
```

**HF (lines 79-95):**
```python
nblocks = math.ceil(seq_len / self.window_size)
pad = nblocks * self.window_size - seq_len
hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad), "constant", 0)
hidden_states = hidden_states.view(batch_size * nblocks, self.window_size, dim)
query_output = self.qformer(query_embeds=self.query, encoder_hidden_states=hidden_states, ...)
query_proj = self.linear(query_output.last_hidden_state.view(batch_size, nblocks * self.window_size // self.downsample_rate, -1))
```

**Status**: Window processing logic is IDENTICAL.

---

## Summary of Discrepancies

| Discrepancy | Severity | Impact |
|-------------|----------|--------|
| Cross-attention K/V dimension | **CRITICAL** | Weight shape mismatch if encoder_hidden_size != hidden_size |
| Cross-attention frequency | **CRITICAL** | Different number of cross-attention layers |
| Separate query FFN | **CRITICAL** | Missing `intermediate_query` and `output_query` modules |
| Config parameter mapping | MEDIUM | Verify encoder_dim = hidden_size = encoder_hidden_size |

---

## Weight Name Mapping

| HF Name | FMS Name | Notes |
|---------|----------|-------|
| `projector.query` | `projector.query_embeds` | |
| `projector.qformer.layernorm` | `projector.input_layernorm` | |
| `projector.qformer.encoder.layer.{i}.attention` | `projector.layers.{i}.self_attention` | |
| `projector.qformer.encoder.layer.{i}.attention.output` | `projector.layers.{i}.self_attention_output` | |
| `projector.qformer.encoder.layer.{i}.crossattention` | `projector.layers.{i}.cross_attention` | Only if has_cross_attention |
| `projector.qformer.encoder.layer.{i}.crossattention.output` | `projector.layers.{i}.cross_attention_output` | |
| `projector.qformer.encoder.layer.{i}.intermediate_query` | **MISSING** | FMS needs this |
| `projector.qformer.encoder.layer.{i}.output_query` | **MISSING** | FMS needs this |
| `projector.linear` | `projector.output_proj` | |

---

## Conclusion

**CRITICAL DISCREPANCIES FOUND - REQUIRES CODE CHANGES**

The FMS SpeechProjector implementation has significant architectural differences from HF's Blip2QFormer-based projector:

1. **Cross-attention K/V dimension**: FMS uses same dim for Q/K/V, HF uses separate `encoder_hidden_size` for K/V in cross-attention
2. **Cross-attention frequency**: FMS has cross-attention in every layer, HF has configurable frequency
3. **Query-specific FFN**: HF has separate `intermediate_query` and `output_query` modules that FMS lacks

These differences mean:
- Weight loading will fail or produce incorrect results
- Model outputs will differ even with correct weights
- FMS projector needs architectural changes to match HF

**RECOMMENDED ACTIONS**:
1. Add `encoder_hidden_size` config parameter for cross-attention K/V
2. Add `cross_attention_frequency` config and conditional cross-attention
3. Add `intermediate_query` and `output_query` separate FFN modules
4. Update weight name mapping in `_hf_to_fms_names`
