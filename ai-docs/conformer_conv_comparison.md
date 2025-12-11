# Conformer Convolution Module Comparison

## Purpose
This document compares `ConformerConvModule` (FMS) with `GraniteSpeechConformerConvModule` (HF), including the depthwise conv wrapper.

## File Locations
- **FMS**: `fms/models/conformer.py` lines 285-414
- **HF**: `modeling_granite_speech.py` lines 185-230

---

## DepthWise Conv1d Wrapper Comparison

### HF `GraniteSpeechConformerDepthWiseConv1d` (lines 185-199)

HF uses a **separate wrapper class** for depthwise convolution with manual padding:

```python
class GraniteSpeechConformerDepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size):
        pad = kernel_size // 2
        pad_offset = (kernel_size + 1) % 2
        self.padding = (pad, pad - pad_offset)
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in, bias=False)

    def forward(self, hidden_states):
        hidden_states = F.pad(hidden_states, self.padding)
        return self.conv(hidden_states)
```

### FMS `ConformerConvModule` (inline depthwise conv)

FMS uses **inline Conv1d** with built-in padding:

```python
self.depthwise_conv = nn.Conv1d(
    dim * expansion_factor,
    dim * expansion_factor,
    kernel_size=kernel_size,
    stride=1,
    padding=(kernel_size - 1) // 2,
    groups=dim * expansion_factor,
    bias=False,
)
```

---

## DISCREPANCY: Padding Calculation

### Padding Formula Comparison

| Kernel Size | HF Padding (left, right) | FMS Padding (symmetric) |
|-------------|--------------------------|-------------------------|
| 15 (odd) | `pad=7, pad_offset=0` → `(7, 7)` | `(15-1)//2 = 7` |
| 16 (even) | `pad=8, pad_offset=1` → `(8, 7)` | `(16-1)//2 = 7` |
| 31 (odd) | `pad=15, pad_offset=0` → `(15, 15)` | `(31-1)//2 = 15` |

**HF Padding Logic (asymmetric for even kernels):**
```python
pad = kernel_size // 2           # e.g., 15 // 2 = 7
pad_offset = (kernel_size + 1) % 2  # e.g., (15 + 1) % 2 = 0
self.padding = (pad, pad - pad_offset)  # e.g., (7, 7)
```

**FMS Padding Logic (symmetric):**
```python
padding=(kernel_size - 1) // 2  # e.g., (15 - 1) // 2 = 7
```

**For odd kernel sizes (default is 15)**: Both produce same effective padding.
**For even kernel sizes**: HF uses asymmetric padding, FMS uses symmetric.

**Impact**: For the default `kernel_size=15` (odd), **NO DISCREPANCY**. For even kernel sizes, output dimensions may differ.

---

## ConvModule Layer Structure Comparison

| Component | FMS Name | HF Name | Shape | Notes |
|-----------|----------|---------|-------|-------|
| LayerNorm | `self.norm` | `self.norm` | `(hidden_dim,)` | IDENTICAL |
| Pointwise conv up | `self.pointwise_conv1` | `self.up_conv` | `Conv1d(dim, dim*exp*2, k=1)` | IDENTICAL |
| GLU | Manual: `x.chunk(2) + sigmoid` | `self.glu = nn.GLU(dim=1)` | N/A | See below |
| Depthwise conv | `self.depthwise_conv` | `self.depth_conv` (wrapper) | `Conv1d(inner, inner, k=15, groups=inner)` | See padding |
| BatchNorm | `self.batch_norm` | `self.batch_norm` | `(inner_dim,)` | IDENTICAL |
| Activation | `self.activation` | `self.silu` | N/A | IDENTICAL (SiLU) |
| Pointwise conv down | `self.pointwise_conv2` | `self.down_conv` | `Conv1d(inner, dim, k=1)` | IDENTICAL |
| Dropout | `self.dropout_layer` | `self.dropout` | N/A | IDENTICAL |

---

## DISCREPANCY: GLU Implementation

### HF (line 211, 225):
```python
self.glu = nn.GLU(dim=1)
# ...
hidden_states = self.glu(hidden_states)
```

### FMS (lines 393-394):
```python
x, gate = x.chunk(2, dim=1)
x = x * torch.sigmoid(gate)
```

**Mathematical Equivalence**:
- `nn.GLU(dim=1)` does exactly: `chunk(2, dim=1)` then `a * sigmoid(b)`
- Both are **functionally identical**

**Impact**: **NO DISCREPANCY** - mathematically equivalent implementations.

---

## Forward Pass Comparison

| Step | FMS | HF | Match |
|------|-----|-----|-------|
| 1 | `x = self.norm(x)` | `hidden_states = self.norm(hidden_states)` | YES |
| 2 | `x = x.transpose(1, 2)` | `hidden_states.permute(0, 2, 1)` | YES |
| 3 | `x = self.pointwise_conv1(x)` | `hidden_states = self.up_conv(...)` | YES |
| 4 | `x, gate = x.chunk(2, dim=1); x = x * sigmoid(gate)` | `hidden_states = self.glu(hidden_states)` | YES |
| 5 | `x = self.depthwise_conv(x)` | `hidden_states = self.depth_conv(hidden_states)` | YES* |
| 6 | `x = self.batch_norm(x)` | (combined with silu) | See below |
| 7 | `x = self.activation(x)` | `hidden_states = self.silu(self.batch_norm(hidden_states))` | YES |
| 8 | `x = self.pointwise_conv2(x)` | `hidden_states = self.down_conv(hidden_states)` | YES |
| 9 | `x = x.transpose(1, 2)` | `.permute(0, 2, 1)` | YES |
| 10 | `x = self.dropout_layer(x)` | `hidden_states = self.dropout(hidden_states)` | YES |

*Step 5: Padding differs for even kernel sizes only.

---

## Validation Status

| Check | Status | Notes |
|-------|--------|-------|
| Layer count | PASS | Same layers |
| Layer types | PASS | Same types |
| Parameter shapes | PASS | Same shapes |
| Depthwise conv padding | PASS* | *Same for odd kernels (default=15) |
| GLU implementation | PASS | Mathematically equivalent |
| Forward order | PASS | Same computation |

---

## Conclusion

**NO DISCREPANCIES** for default configuration (kernel_size=15).

Minor implementation differences:
- HF uses wrapper class for depthwise conv with manual F.pad, FMS uses built-in padding
- HF uses nn.GLU module, FMS uses manual chunk+sigmoid (equivalent)
- Naming differences

**Potential issue**: If even kernel sizes are ever used, padding behavior differs.
