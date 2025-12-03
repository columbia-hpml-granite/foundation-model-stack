# Granite Speech: HuggingFace → FMS Migration Gap Analysis

---

## Goal

**Migrate and sync features of Granite Speech from HuggingFace to FMS.**

✅ **Assessment: Correctly identified.**

The project aims to port IBM's Granite Speech 3.3 8B model from the HuggingFace transformers library to the Foundation Model Stack (FMS), enabling production features like tensor parallelism, quantization, and custom attention backends.

---

## Outcome

**Run end-to-end tests (input .wav → text) and get identical results on both repos.**

✅ **Assessment: Correctly identified, but requires clarification.**

To achieve identical E2E results, we need parity at **three levels**:

| Level | Description | Current Status |
|-------|-------------|----------------|
| **1. Architecture** | Same neural network structure | ✅ Complete |
| **2. Weights** | Same parameter values loaded | ✅ Complete |
| **3. Pipeline** | Same preprocessing + generation | ❌ Incomplete |

**The gap is at Level 3 (Pipeline)**, not the model itself.

---

## Done ✅

### 1. Neural Network Architecture

| Component | HuggingFace | FMS | Status |
|-----------|-------------|-----|--------|
| Conformer Encoder | `GraniteSpeechCTCEncoder` | `ConformerEncoder` | ✅ Matched |
| Q-Former Projector | `GraniteSpeechEncoderProjector` | `SpeechProjector` | ✅ Matched |
| Granite Decoder | `GraniteForCausalLM` | `GraniteHeadless` | ✅ Matched |
| LM Head | `lm_head` | `lm_head` | ✅ Matched |

**All 14 architectural parameters verified matching** (see `analyzation.md` Section 3.1).

### 2. Weight Conversion

| Adapter | Function | Status |
|---------|----------|--------|
| `_hf_to_fms_names` | Rename HF weight keys to FMS format | ✅ Implemented |
| `_split_kv_weights` | Split combined K-V weights | ✅ Implemented |
| `_granite_speech_weight_fusion` | Handle fused/unfused weights | ✅ Implemented |

**Registered as:** `serialization.register_adapter("granite_speech", "hf", [...])`

### 3. Core Model Methods

| Method | Purpose | Status |
|--------|---------|--------|
| `get_audio_features()` | Encoder → Projector pipeline | ✅ Implemented |
| `get_merged_audio_embeddings()` | Replace `<\|audio\|>` tokens with audio embeddings | ✅ Implemented |
| `forward()` | Full multimodal forward pass | ✅ Implemented |
| Input validation | Raise `ValueError` on audio/token mismatch | ✅ Implemented |

### 4. Test Infrastructure

| Test Suite | Lines | Coverage |
|------------|-------|----------|
| `tests/models/test_granite_speech.py` | 486 | Model unit tests |
| `tests/models/test_conformer_hf.py` | 434 | Encoder tests |
| `tests/modules/test_projector_hf.py` | 401 | Projector tests |
| `tests/models/hf_equivalence/test_granite_speech.py` | 183 | HF ↔ FMS comparison |

**Result:** 68 tests passing.

---

## TODO ❌

### Critical Path to E2E Success

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     E2E PIPELINE: What's Missing                             │
└─────────────────────────────────────────────────────────────────────────────┘

  Input: audio.wav
       │
       ▼
  ┌─────────────────────────────────────┐
  │  ❌ GraniteSpeechFeatureExtractor   │  ← TODO #1
  │     wav → mel spectrogram (160-dim)  │
  └─────────────────┬───────────────────┘
                    │
       input_features (batch, seq_len, 160)
                    │
                    ▼
  ┌─────────────────────────────────────┐
  │  ❌ GraniteSpeechProcessor          │  ← TODO #2
  │     • Tokenize prompt                │
  │     • Expand <|audio|> tokens        │
  │     • Create input_features_mask     │
  └─────────────────┬───────────────────┘
                    │
       input_ids, input_features, mask
                    │
                    ▼
  ┌─────────────────────────────────────┐
  │  ✅ GraniteSpeech.forward()         │  ← DONE
  │     Encoder → Projector → Decoder    │
  └─────────────────┬───────────────────┘
                    │
       logits (single forward pass)
                    │
                    ▼
  ┌─────────────────────────────────────┐
  │  ❌ prepare_inputs_for_generation   │  ← TODO #3
  │     Skip audio reprocessing in cache │
  └─────────────────┬───────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────┐
  │  ⚠️ fms.utils.generation.generate() │  ← TODO #4 (integration)
  │     Autoregressive text generation   │
  └─────────────────┬───────────────────┘
                    │
       output_ids
                    │
                    ▼
  Output: "Transcribed text..."
```

---

### TODO #1: GraniteSpeechFeatureExtractor

**Priority:** P2 (Medium)
**Effort:** Medium
**Impact:** HIGH - Users cannot process raw audio without this

**What HF provides:**
```python
from transformers import GraniteSpeechFeatureExtractor

extractor = GraniteSpeechFeatureExtractor()
features = extractor(audio_waveform, sampling_rate=16000, return_tensors="pt")
# Returns: input_features, input_features_mask, audio_embedding_sizes
```

**HF Implementation Details:**
- Sampling rate: 16,000 Hz
- FFT: `n_fft=512`, `win_length=400`, `hop_length=160`
- Mel bands: 80
- Log scaling: `(log10(mel) - 8.0) / 4 + 1`
- Frame pairing: Stack consecutive frames → 160 features

**What FMS needs:**
```python
# fms/preprocessing/granite_speech.py (new file)
class GraniteSpeechFeatureExtractor:
    def __call__(self, audio, sampling_rate=16000):
        # 1. Compute mel spectrogram
        # 2. Apply log scaling
        # 3. Pair consecutive frames
        # 4. Return (input_features, mask, sizes)
```

**Workaround (current):** Use HF's extractor or manual librosa code.

---

### TODO #2: GraniteSpeechProcessor

**Priority:** P0 (Critical)
**Effort:** Medium
**Impact:** CRITICAL - Token mismatch causes `ValueError`

**What HF provides:**
```python
from transformers import GraniteSpeechProcessor

processor = GraniteSpeechProcessor.from_pretrained("ibm-granite/granite-speech-3.3-8b")
inputs = processor(text="Transcribe: <|audio|>", audio=audio, return_tensors="pt")
# Automatically expands <|audio|> to ceil(seq_len/15) × 3 tokens
```

**Critical Logic:**
```python
def get_num_audio_tokens(audio_seq_len):
    num_windows = math.ceil(audio_seq_len / 15)  # window_size = 15
    tokens_per_window = 3  # 15 // 5 (downsample_rate)
    return num_windows * tokens_per_window
```

**What FMS needs:**
```python
# fms/preprocessing/granite_speech.py
class GraniteSpeechProcessor:
    def __call__(self, text, audio_features, tokenizer):
        # 1. Calculate required audio tokens from audio_features.shape[1]
        # 2. Expand <|audio|> in text to correct count
        # 3. Tokenize
        # 4. Create input_features_mask
        # 5. Return BatchFeature-like dict
```

**Risk without this:** `get_merged_audio_embeddings()` raises:
```
ValueError: Mismatch: 150 audio positions but 147 audio vectors provided.
```

---

### TODO #3: prepare_inputs_for_generation

**Priority:** P1 (High)
**Effort:** Low
**Impact:** MEDIUM - Inefficient generation without this

**What HF provides:**
```python
def prepare_inputs_for_generation(self, input_ids, past_key_values=None, ...):
    if past_key_values is not None:
        # Skip audio processing after first iteration
        input_ids = input_ids[:, -1:]
    model_inputs = {"input_ids": input_ids}
    if past_key_values is None and input_features is not None:
        model_inputs["input_features"] = input_features
    return model_inputs
```

**What FMS needs:**
```python
# In fms/models/granite_speech.py
def prepare_inputs_for_generation(self, iteration, input_ids, kwargs):
    """Hook for fms.utils.generation.generate()"""
    if kwargs["use_cache"] and iteration > 0:
        # Cached decoding - skip audio
        return input_ids, kwargs

    if iteration == 0 and "input_features" in kwargs:
        # First iteration - merge audio embeddings
        audio_embeds = self.get_audio_features(kwargs.pop("input_features"))
        inputs_embeds = self.get_merged_audio_embeddings(
            input_ids, audio_embeds, kwargs.pop("input_features_mask")
        )
        return inputs_embeds, kwargs

    return input_ids, kwargs
```

**Reference:** FMS LlavaNext already has this pattern (`fms/models/llava_next.py:384-426`).

---

### TODO #4: Generation Integration

**Priority:** P1 (High)
**Effort:** Medium
**Impact:** MEDIUM - Users need wrapper code without this

**What HF provides:**
```python
outputs = model.generate(**inputs, max_new_tokens=256)
text = processor.batch_decode(outputs, skip_special_tokens=True)
```

**What FMS needs:**
```python
# Either add generate() method to GraniteSpeech, or document usage:

from fms.utils.generation import generate

output_ids = generate(
    model,  # or wrapped model_forward function
    input_ids,
    max_new_tokens=256,
    use_cache=True,
    prepare_model_inputs_hook=model.prepare_inputs_for_generation,
    extra_kwargs={"input_features": features, "input_features_mask": mask}
)
```

---

### TODO #5: LoRA Adapter Support (Optional)

**Priority:** P3 (Low for E2E, needed for quality parity)
**Effort:** High
**Impact:** MEDIUM - Output quality difference

**What HF provides:**
- `has_lora_adapter: True` in config
- Conditional activation: LoRA enabled when audio present, disabled for text-only
- Requires PEFT library

**FMS Status:** Not implemented. Model outputs will differ from HF on same inputs.

**Note:** This is optional for basic E2E functionality but required for exact output matching.

---

## Summary: Path to E2E Success

### Minimum Viable E2E (Identical Outputs)

| # | Component | Priority | Effort | Required for E2E? |
|---|-----------|----------|--------|-------------------|
| 1 | FeatureExtractor | P2 | Medium | ⚠️ Can use HF's |
| 2 | **Processor** | **P0** | **Medium** | **✅ Yes - Critical** |
| 3 | prepare_inputs_for_generation | P1 | Low | ✅ Yes - Efficiency |
| 4 | Generation integration | P1 | Medium | ✅ Yes - Usability |
| 5 | LoRA adapters | P3 | High | ⚠️ For exact match |

### Recommended Implementation Order

```
Week 1: TODO #2 (Processor) + TODO #3 (prepare_inputs hook)
        → Enables correct forward pass and efficient generation

Week 2: TODO #4 (Generation integration)
        → Clean API for users

Week 3: TODO #1 (FeatureExtractor)
        → Complete standalone pipeline

Later:  TODO #5 (LoRA)
        → Quality parity with HF
```

### Success Criteria

```python
# E2E Test: Identical outputs
def test_e2e_equivalence():
    audio = load_audio("test.wav")

    # HuggingFace
    hf_processor = GraniteSpeechProcessor.from_pretrained(MODEL_ID)
    hf_model = GraniteSpeechForConditionalGeneration.from_pretrained(MODEL_ID)
    hf_inputs = hf_processor(text="Transcribe: <|audio|>", audio=audio, return_tensors="pt")
    hf_outputs = hf_model.generate(**hf_inputs, max_new_tokens=256)
    hf_text = hf_processor.batch_decode(hf_outputs, skip_special_tokens=True)[0]

    # FMS (after TODOs completed)
    fms_processor = FMSGraniteSpeechProcessor(...)
    fms_model = GraniteSpeech(config)
    fms_model.load_state_dict(convert_hf_weights(hf_model.state_dict()))
    fms_inputs = fms_processor(text="Transcribe: <|audio|>", audio=audio)
    fms_outputs = fms_model.generate(**fms_inputs, max_new_tokens=256)
    fms_text = tokenizer.decode(fms_outputs, skip_special_tokens=True)

    # Must match
    assert hf_text == fms_text, f"HF: {hf_text}\nFMS: {fms_text}"
```

---

## Current Workaround

Until TODOs are completed, users can achieve E2E with a hybrid approach:

```python
from transformers import GraniteSpeechProcessor
from fms.models.granite_speech import GraniteSpeech
from fms.utils.generation import generate

# Use HF for preprocessing
processor = GraniteSpeechProcessor.from_pretrained("ibm-granite/granite-speech-3.3-8b")
inputs = processor(text="Transcribe: <|audio|>", audio=audio, return_tensors="pt")

# Use FMS for inference
fms_model = GraniteSpeech(config)
fms_model.load_state_dict(converted_weights)

# Manual generation loop with hook
def prepare_hook(iteration, input_ids, kwargs):
    if iteration == 0:
        audio_embeds = fms_model.get_audio_features(kwargs.pop("input_features"))
        inputs_embeds = fms_model.get_merged_audio_embeddings(
            input_ids, audio_embeds, kwargs.pop("input_features_mask")
        )
        return inputs_embeds, kwargs
    return input_ids, kwargs

output_ids = generate(
    fms_model,
    inputs["input_ids"],
    use_cache=True,
    prepare_model_inputs_hook=prepare_hook,
    extra_kwargs={"input_features": inputs["input_features"], ...}
)
```

**Lines of code:** ~50 (vs 5 with native FMS support)
