# Conformer Attention Component Comparison

## Purpose
This document compares `ConformerAttention` (FMS) with `GraniteSpeechConformerAttention` (HF) for migration validation.

## File Locations
- **FMS**: `fms/models/conformer.py` lines 146-282
- **HF**: `src/transformers/models/granite_speech/modeling_granite_speech.py` lines 119-182

## Architecture Summary

### Layer Structure (IDENTICAL)

| Component | FMS Name | HF Name | Shape | Bias |
|-----------|----------|---------|-------|------|
| Layer Norm | `self.norm` | `self.pre_norm` | `(hidden_dim,)` | Yes |
| Query projection | `self.to_q` | `self.to_q` | `(hidden_dim, inner_dim)` | No |
| Key-Value projection | `self.to_kv` | `self.to_kv` | `(hidden_dim, inner_dim * 2)` | No |
| Output projection | `self.to_out` | `self.to_out` | `(inner_dim, hidden_dim)` | Yes |
| Positional embedding | `self.pos_emb` | `self.rel_pos_emb` | `(2 * max_pos_emb + 1, dim_head)` | N/A |
| Dropout | `self.dropout` | `self.dropout` | N/A | N/A |

Where `inner_dim = num_heads * dim_head`

### Forward Pass (IDENTICAL)

1. Apply LayerNorm
2. Compute `num_blocks = ceil(seq_len / context_size)`
3. Pad sequence if `seq_len % context_size != 0`
4. Project Q, K, V
5. Reshape to `(batch, num_blocks, num_heads, context_size, dim_head)`
6. Compute Shaw's relative positional attention: `einsum("b m h c d, c r d -> b m h c r", Q, rel_pos_emb) * scale`
7. Apply mask for padded positions in last block
8. Call `scaled_dot_product_attention` with MATH backend
9. Reshape output and apply output projection
10. Apply dropout

## Validation Status

| Check | Status |
|-------|--------|
| Layer count | PASS |
| Layer types | PASS |
| Parameter shapes | PASS |
| Bias configuration | PASS |
| Forward computation | PASS |
| Positional encoding | PASS |
| Masking logic | PASS |

## Conclusion

**NO DISCREPANCIES FOUND**

The implementations are functionally equivalent. Differences are cosmetic only:
- Variable naming (`x` vs `hidden_states`)
- Layer naming (`norm` vs `pre_norm`, `pos_emb` vs `rel_pos_emb`)
- Boolean assignment style (`False` vs `0`)
