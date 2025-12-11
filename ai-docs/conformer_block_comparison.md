# Conformer Block Component Comparison

## Purpose
This document compares `ConformerBlock` (FMS) with `GraniteSpeechConformerBlock` (HF).

## File Locations
- **FMS**: `fms/models/conformer.py` lines 417-511
- **HF**: `modeling_granite_speech.py` lines 233-250

---

## Layer Structure Comparison

| Component | FMS Name | HF Name | Type |
|-----------|----------|---------|------|
| Feed-forward 1 | `self.ff1` | `self.ff1` | ConformerFeedForward |
| Attention | `self.attn` | `self.attn` | ConformerAttention |
| Convolution | `self.conv` | `self.conv` | ConformerConvModule |
| Feed-forward 2 | `self.ff2` | `self.ff2` | ConformerFeedForward |
| Post LayerNorm | `self.post_norm` | `self.post_norm` | nn.LayerNorm |

**Status**: IDENTICAL layer structure.

---

## Forward Pass Comparison

### FMS (lines 496-509):
```python
x = x + 0.5 * self.ff1(x)
x = x + self.attn(x, attention_dists)
x = x + self.conv(x)
x = x + 0.5 * self.ff2(x)
x = self.post_norm(x)
return x
```

### HF (lines 244-250):
```python
hidden_states = 0.5 * self.ff1(hidden_states) + hidden_states
hidden_states = self.attn(hidden_states, attention_dists=attention_dists) + hidden_states
hidden_states = self.conv(hidden_states) + hidden_states
hidden_states = 0.5 * self.ff2(hidden_states) + hidden_states
hidden_states = self.post_norm(hidden_states)
return hidden_states
```

| Step | FMS | HF | Match |
|------|-----|-----|-------|
| 1. FF1 residual | `x + 0.5 * ff1(x)` | `0.5 * ff1(x) + x` | YES (commutative) |
| 2. Attention residual | `x + attn(x, attention_dists)` | `attn(x, attention_dists) + x` | YES (commutative) |
| 3. Conv residual | `x + conv(x)` | `conv(x) + x` | YES (commutative) |
| 4. FF2 residual | `x + 0.5 * ff2(x)` | `0.5 * ff2(x) + x` | YES (commutative) |
| 5. Post norm | `post_norm(x)` | `post_norm(x)` | YES |

---

## Residual Scaling

| Module | FMS Scale | HF Scale | Match |
|--------|-----------|----------|-------|
| FF1 | 0.5 | 0.5 | YES |
| Attention | 1.0 | 1.0 | YES |
| Convolution | 1.0 | 1.0 | YES |
| FF2 | 0.5 | 0.5 | YES |

---

## Validation Status

| Check | Status |
|-------|--------|
| Layer count | PASS |
| Layer types | PASS |
| Layer names | PASS (identical) |
| Residual connections | PASS |
| Residual scaling | PASS |
| Post normalization | PASS |

---

## Conclusion

**NO DISCREPANCIES** - Implementations are identical.

The only difference is operand order in addition (e.g., `x + 0.5*ff1(x)` vs `0.5*ff1(x) + x`), which is mathematically equivalent due to commutativity.
