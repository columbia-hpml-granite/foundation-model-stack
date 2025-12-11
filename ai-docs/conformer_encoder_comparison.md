# Conformer Encoder Component Comparison

## Purpose
This document compares `ConformerEncoder` (FMS) with `GraniteSpeechCTCEncoder` (HF) for migration validation.

## File Locations
- **FMS**: `fms/models/conformer.py` lines 514-646
- **HF**: `src/transformers/models/granite_speech/modeling_granite_speech.py` lines 253-279

## Architecture Summary

### Layer Structure (IDENTICAL)

| Component | FMS Name | HF Name | Shape | Bias |
|-----------|----------|---------|-------|------|
| Input projection | `self.input_proj` | `self.input_linear` | `(input_dim, hidden_dim)` | Yes |
| Conformer blocks | `self.blocks` | `self.layers` | `num_layers` blocks | N/A |
| CTC output | `self.out` | `self.out` | `(hidden_dim, output_dim)` | Yes |
| CTC feedback | `self.out_mid` | `self.out_mid` | `(output_dim, hidden_dim)` | Yes |
| Attention distances | `self.attention_dists` | `self.attention_dists` | `(context_size, context_size)` | N/A |

### Attention Distance Computation (IDENTICAL)

```python
seq = torch.arange(context_size)
relpos_dist = seq.view(-1, 1) - seq.view(1, -1)
attention_dists = torch.clamp(relpos_dist, -context_size, context_size) + max_pos_emb
```

### Forward Pass Comparison

| Step | FMS | HF | Match |
|------|-----|-----|-------|
| Input projection | `x = self.input_proj(input_features)` | `hidden_states = self.input_linear(hidden_states)` | YES |
| Loop | `for idx, block in enumerate(self.blocks, start=1)` | `for idx, layer in enumerate(self.layers, start=1)` | YES |
| Block forward | `x = block(x, attention_dists)` | `hidden_states = layer(hidden_states, attention_dists=self.attention_dists)` | YES |
| Mid-layer index | `idx == len(self.blocks) // 2` | `idx == self.num_layers // 2` | YES |
| CTC softmax | `F.softmax(x_mid, dim=-1)` | `nn.Softmax(dim=-1)(hidden_states_mid)` | YES |
| Feedback add | `x = x + self.out_mid(...)` | `hidden_states += self.out_mid(...)` | YES |

---

## DISCREPANCIES FOUND

### 1. Clone in Mid-Layer CTC (BEHAVIORAL DIFFERENCE)

| Aspect | FMS | HF |
|--------|-----|-----|
| Code | `x_mid = self.out(x)` | `hidden_states_mid = hidden_states.clone(); hidden_states_mid = self.out(hidden_states_mid)` |

**FMS Code (lines 641-643):**
```python
if self.config.use_ctc and self.out is not None and idx == mid_layer:
    x_mid = self.out(x)  # No clone
    x = x + self.out_mid(F.softmax(x_mid, dim=-1))
```

**HF Code (lines 275-278):**
```python
if idx == self.num_layers // 2:
    hidden_states_mid = hidden_states.clone()  # Clones first
    hidden_states_mid = self.out(hidden_states_mid)
    hidden_states += self.out_mid(nn.Softmax(dim=-1)(hidden_states_mid))
```

**Impact**:
- Forward inference: Mathematically equivalent
- Backward pass (training): Different gradient computation graph

**ACTION REQUIRED**: Verify if this affects training or if inference-only is acceptable.

---

### 2. Buffer Persistence (SERIALIZATION DIFFERENCE)

| Aspect | FMS | HF |
|--------|-----|-----|
| Code | `register_buffer("attention_dists", attention_dists)` | `register_buffer("attention_dists", attention_dists, persistent=False)` |

**Impact**:
- FMS: `attention_dists` is saved in `state_dict`
- HF: `attention_dists` is NOT saved in `state_dict`

**ACTION REQUIRED**: When loading HF weights into FMS, the `attention_dists` buffer will be missing from HF checkpoint. FMS recomputes it in `__init__`, so this should work, but verify weight loading doesn't fail.

---

### 3. CTC Conditional Flag (BEHAVIORAL DIFFERENCE)

| Aspect | FMS | HF |
|--------|-----|-----|
| Code | `if self.config.use_ctc and self.out is not None and idx == mid_layer` | `if idx == self.num_layers // 2` |

**Impact**:
- FMS: Can optionally disable mid-layer CTC via `use_ctc=False`
- HF: Always applies mid-layer CTC (no conditional)

**ACTION REQUIRED**: Ensure `use_ctc=True` is always set in FMS config when migrating from HF to maintain identical behavior.

---

### 4. Input Validation (FMS-ONLY FEATURE)

| Aspect | FMS | HF |
|--------|-----|-----|
| Code | Has `assert` for input dimension | No validation |

**Impact**: None on output, FMS is more defensive.

---

### 5. Dtype Conversion for attention_dists (MINOR)

| Aspect | FMS | HF |
|--------|-----|-----|
| Code | `attention_dists.long()` explicit conversion | No explicit conversion |

**Impact**: Should be equivalent since embedding lookup expects LongTensor, but explicit is safer.

---

## Validation Status

| Check | Status | Notes |
|-------|--------|-------|
| Layer count | PASS | |
| Layer types | PASS | |
| Parameter shapes | PASS | |
| Bias configuration | PASS | |
| Forward computation | REVIEW | Clone behavior differs |
| Mid-layer CTC logic | REVIEW | Conditional flag differs |
| Attention distances | REVIEW | Persistence differs |

---

## Conclusion

**DISCREPANCIES FOUND - REVIEW REQUIRED**

| Discrepancy | Severity | Action |
|-------------|----------|--------|
| Clone in mid-layer CTC | MEDIUM | Verify training equivalence or add clone to FMS |
| Buffer persistence | LOW | Verify weight loading works |
| CTC conditional flag | LOW | Ensure `use_ctc=True` in config |
| Dtype conversion | LOW | No action needed |

The implementations produce identical outputs for inference when `use_ctc=True`, but there are differences that may affect:
1. Training (gradient flow due to clone)
2. Serialization (buffer persistence)
3. Configuration flexibility (use_ctc flag)
