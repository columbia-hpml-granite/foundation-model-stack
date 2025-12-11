# Conformer FeedForward Component Comparison

## Purpose
This document compares `ConformerFeedForward` (FMS) with `GraniteSpeechConformerFeedForward` (HF).

## File Locations
- **FMS**: `fms/models/conformer.py` lines 81-143
- **HF**: `modeling_granite_speech.py` lines 99-116

## Architecture Comparison

### Layer Structure

| Component | FMS Name | HF Name | Shape | Bias |
|-----------|----------|---------|-------|------|
| LayerNorm | `self.norm` | `self.pre_norm` | `(hidden_dim,)` | Yes |
| Up projection | `self.fc1` | `self.up_proj` | `(hidden_dim, hidden_dim * mult)` | Yes |
| Activation | `self.activation` (str_to_activation) | `self.silu` (nn.SiLU) | N/A | N/A |
| Dropout 1 | `self.dropout1` | `self.dropout` | N/A | N/A |
| Down projection | `self.fc2` | `self.down_proj` | `(hidden_dim * mult, hidden_dim)` | Yes |
| Dropout 2 | `self.dropout2` | `self.dropout` (reused) | N/A | N/A |

### Forward Pass Comparison

| Step | FMS | HF | Match |
|------|-----|-----|-------|
| 1 | `x = self.norm(x)` | `hidden_states = self.pre_norm(hidden_states)` | YES |
| 2 | `x = self.fc1(x)` | `hidden_states = self.up_proj(hidden_states)` | YES |
| 3 | `x = self.activation(x)` | (combined with step 4) | See below |
| 4 | `x = self.dropout1(x)` | `hidden_states = self.dropout(self.silu(hidden_states))` | **DIFFERENT ORDER** |
| 5 | `x = self.fc2(x)` | `hidden_states = self.down_proj(hidden_states)` | YES |
| 6 | `x = self.dropout2(x)` | `hidden_states = self.dropout(hidden_states)` | YES |

---

## DISCREPANCY FOUND

### Dropout Order After Activation

| Aspect | FMS | HF |
|--------|-----|-----|
| Order | `activation(x)` → `dropout1(x)` | `dropout(silu(x))` - same dropout instance |

**FMS (lines 139-140):**
```python
x = self.activation(x)
x = self.dropout1(x)
```

**HF (line 113):**
```python
hidden_states = self.dropout(self.silu(hidden_states))
```

**Impact**: Functionally IDENTICAL - both apply activation then dropout. The difference is:
- FMS uses two separate dropout instances (`dropout1`, `dropout2`)
- HF reuses the same dropout instance (`dropout`) for both positions

Since dropout layers are stateless (only differ in probability), this has **NO functional impact**.

---

### Dropout Instance Reuse

| Aspect | FMS | HF |
|--------|-----|-----|
| Dropout instances | Two: `self.dropout1`, `self.dropout2` | One: `self.dropout` (reused) |

**FMS (lines 118, 124):**
```python
self.dropout1 = nn.Dropout(dropout)
self.dropout2 = nn.Dropout(dropout)
```

**HF (line 107):**
```python
self.dropout = nn.Dropout(config.dropout)
```

**Impact**: **NO functional impact** - dropout is stateless, same probability used.

---

## Validation Status

| Check | Status | Notes |
|-------|--------|-------|
| Layer count | PASS | Same layers |
| Layer types | PASS | Same types |
| Parameter shapes | PASS | Same shapes |
| Bias configuration | PASS | All have bias |
| Forward computation | PASS | Same order |
| Activation function | PASS | Both use SiLU |

---

## Conclusion

**NO DISCREPANCIES** - Implementations are functionally identical.

Minor differences:
- Naming conventions (`norm` vs `pre_norm`, `fc1/fc2` vs `up_proj/down_proj`)
- FMS uses two dropout instances, HF reuses one (no functional difference)
