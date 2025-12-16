# GraniteSpeech Debug Scripts

Debug scripts for validating FMS GraniteSpeech implementation against HuggingFace reference.

## Scripts Overview

| Script | Purpose | Test Level |
|--------|---------|------------|
| `debug_equivalence.py` | Layer-by-layer output comparison | L1 - Activation Parity |
| `debug_adapter.py` | Weight conversion debugging | L0 - Weight Loading |
| `debug_shape.py` | Shape mismatch detection | L0 - Weight Loading |
| `debug_config.py` | Config propagation verification | L0 - Config |
| `test_get_model_granite_speech.py` | Model loading via `get_model()` | L0 - Integration |

---

## debug_equivalence.py

**Purpose:** Compare FMS and HF GraniteSpeech outputs at each layer to identify where divergence occurs.

**Test Points:**
```
COMPARISON 1: Text Embeddings      - Embedding layer (no audio)
COMPARISON 2: Encoder Output       - Conformer encoder
COMPARISON 3: Projector Output     - Q-Former projector
COMPARISON 4: Merged Embeddings    - Audio injection into text
COMPARISON 5: Decoder Output       - Granite decoder (before lm_head)
COMPARISON 6: Full Forward Pass    - End-to-end logits comparison
```

**Usage:**
```bash
python scripts/debug_granite_speech/debug_equivalence.py
```

**Expected Output:**
```
Encoder diff: 0.000000
Projector diff: 0.000000
Logits diff: 0.000XXX  (< 1e-3 acceptable)
```

---

## debug_adapter.py

**Purpose:** Debug the HF-to-FMS weight adapter to verify key conversions.

**What It Tests:**
- Patches `load_state_dict_into_model` to intercept weight loading
- Traces `encoder.out` weight conversions (CTC output layer)
- Verifies `output_dim` config propagation

**Usage:**
```bash
python scripts/debug_granite_speech/debug_adapter.py
```

---

## debug_shape.py

**Purpose:** Detect shape mismatches between checkpoint weights and model parameters.

**What It Tests:**
- Patches `_load_partial_state_dict` to compare shapes
- Reports any parameter where `checkpoint.shape != model.shape`

**Usage:**
```bash
python scripts/debug_granite_speech/debug_shape.py
```

**Expected Output (success):**
```
(no SHAPE MISMATCH output)
```

---

## debug_config.py

**Purpose:** Verify model config is correctly propagated during weight loading.

**What It Tests:**
- `encoder_config.output_dim` value
- `encoder.out.weight.shape` matches config

**Usage:**
```bash
python scripts/debug_granite_speech/debug_config.py
```

---

## test_get_model_granite_speech.py

**Purpose:** Test that GraniteSpeech loads correctly via `get_model()` API.

**Usage:**
```bash
python scripts/debug_granite_speech/test_get_model_granite_speech.py
```

---

## Testing Levels Reference

| Level | Method | What It Validates |
|-------|--------|-------------------|
| **L0** | Weight/Config | Weights loaded correctly, configs match |
| **L1** | Activation Parity | Per-layer numerical equivalence (atol=1e-3) |
| **L2** | Output Signature | Final logits distribution match |
| **L3** | Generation | Decoded text output matches exactly |

---

## Formal Test Suite

For CI/regression testing, use the formal pytest tests instead:

```bash
# Activation parity test (L1)
pytest tests/models/hf_equivalence/test_granite_speech_activation_parity.py -v --runslow

# Signature equivalence test (L2)
pytest tests/models/hf_equivalence/test_granite_speech.py::test_granite_speech_2b_signature_equivalence -v --runslow

# Generation equivalence test (L3)
pytest tests/models/hf_equivalence/test_granite_speech.py::test_granite_speech_2b_generation_equivalence -v --runslow
```
