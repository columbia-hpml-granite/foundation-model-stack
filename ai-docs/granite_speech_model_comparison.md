# GraniteSpeech Full Model Comparison

## Purpose
This document compares `GraniteSpeech` (FMS) with `GraniteSpeechForConditionalGeneration` (HF).

## File Locations
- **FMS**: `fms/models/granite_speech.py` (GraniteSpeech class, lines 173-623)
- **HF**: `modeling_granite_speech.py` lines 303-544

---

## Architecture Overview

### HF `GraniteSpeechForConditionalGeneration`:
```python
self.language_model = AutoModelForCausalLM.from_config(config.text_config)
self.encoder = GraniteSpeechCTCEncoder(config.encoder_config)
self.projector = GraniteSpeechEncoderProjector(config)
```

### FMS `GraniteSpeech`:
```python
self.encoder = ConformerEncoder(self.config.encoder_config)
self.projector = SpeechProjector(self.config.projector_config, ...)
self.decoder = GraniteHeadless(self.config.decoder_config)
self.lm_head = nn.Linear(...)
```

---

## DISCREPANCIES

### 1. Language Model Structure (ARCHITECTURAL)

| Aspect | FMS | HF |
|--------|-----|-----|
| Decoder | `GraniteHeadless` + separate `lm_head` | `AutoModelForCausalLM` (includes lm_head) |
| LM Head | Separate `nn.Linear` layer | Part of `language_model` |

**FMS (lines 233-241):**
```python
self.decoder = GraniteHeadless(self.config.decoder_config)
self.lm_head = nn.Linear(
    self.config.decoder_config.emb_dim,
    self.config.decoder_config.src_vocab_size,
    bias=False,
)
```

**HF (line 310):**
```python
self.language_model = AutoModelForCausalLM.from_config(config.text_config)
```

**Impact**: Weight naming differs - FMS has `decoder.*` and `lm_head.*`, HF has `language_model.*`.

---

### 2. Forward Signature Differences

| Parameter | FMS | HF | Notes |
|-----------|-----|-----|-------|
| `input_ids` | Optional | Optional | Same |
| `input_features` | Optional | Optional | Same |
| `input_features_mask` | Optional | Optional | Same |
| `attention_mask` | Optional | Optional | Same |
| `position_ids` | Optional | Optional | Same |
| `past_key_values` | Optional | Optional | Same |
| `inputs_embeds` | Optional | Optional | Same |
| `labels` | Optional | Optional | Same |
| `use_cache` | Optional | Optional | Same |
| `output_attentions` | N/A | Optional | **FMS MISSING** |
| `output_hidden_states` | N/A | Optional | **FMS MISSING** |
| `return_dict` | N/A | Optional | **FMS MISSING** |
| `cache_position` | N/A | Optional | **FMS MISSING** |
| `logits_to_keep` | N/A | Optional | **FMS MISSING** |

**Impact**: FMS doesn't support some HF-specific features like returning attention weights or hidden states.

---

### 3. Forward Return Value

| Aspect | FMS | HF |
|--------|-----|-----|
| Return type | `Tuple[Tensor, ...]` | `GraniteSpeechCausalLMOutputWithPast` or tuple |
| With cache | `(logits, cache)` | `GraniteSpeechCausalLMOutputWithPast` |
| With loss | `(logits, loss)` | `GraniteSpeechCausalLMOutputWithPast` |

**FMS (lines 535-540):**
```python
if use_cache:
    return logits, cache
return logits, loss
```

**HF (lines 444-450):**
```python
return GraniteSpeechCausalLMOutputWithPast(
    loss=loss,
    logits=logits,
    past_key_values=outputs.past_key_values,
    hidden_states=outputs.hidden_states,
    attentions=outputs.attentions,
)
```

**Impact**: Different return structure - FMS uses tuples, HF uses dataclass.

---

### 4. Audio Token ID Config Name

| Aspect | FMS | HF |
|--------|-----|-----|
| Config attribute | `audio_token_index` | `audio_token_id` |

**FMS (line 218):**
```python
self.audio_token_index = self.config.audio_token_index
```

**HF (line 389):**
```python
is_audio_idx = input_ids == self.config.audio_token_id
```

**Impact**: Config parameter naming differs - need to map during weight conversion.

---

### 5. `get_merged_audio_embeddings` Logic

Both implementations have similar logic but with differences:

**FMS (lines 368-423):**
```python
audio_pos = (input_ids == self.audio_token_index)
safe_ids = torch.where(audio_pos, input_ids.new_zeros(()), input_ids)
token_embeds = self.get_input_embeddings()(safe_ids)
# ... mask handling ...
merged = token_embeds.masked_scatter(mask, audio_flat)
```

**HF (lines 482-515):**
```python
is_audio_index = input_ids == self.config.audio_token_id
llm_input_ids = torch.where(is_audio_index, 0, input_ids)
inputs_embeds = self.language_model.get_input_embeddings()(llm_input_ids)
# ... mask handling ...
inputs_embeds = inputs_embeds.masked_scatter(special_audio_mask, audio_features)
```

**Status**: Logic is equivalent, just different variable names.

---

### 6. Loss Computation

**FMS (lines 522-533):**
```python
if labels is not None:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_logits = torch.nan_to_num(shift_logits)  # Extra NaN handling
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
```

**HF (lines 422-438):**
```python
if labels is not None:
    if attention_mask is not None:
        # Attention mask handling for shift
        shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1):].to(logits.device)
        shift_logits = logits[..., :-1, :][shift_attention_mask != 0].contiguous()
        shift_labels = labels[..., 1:][shift_attention_mask != 0].contiguous()
    else:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device))
```

| Difference | FMS | HF |
|------------|-----|-----|
| Attention mask handling | None | Applies mask to select valid positions |
| NaN handling | `torch.nan_to_num()` | None |

**Impact**:
- FMS doesn't handle attention mask in loss computation
- FMS has extra NaN protection

---

### 7. `prepare_inputs_for_generation` Signature

**FMS (lines 542-622):**
```python
def prepare_inputs_for_generation(
    self,
    iteration: int,  # FMS uses iteration counter
    input_ids: torch.Tensor,
    kwargs: dict,  # kwargs as single dict
) -> Tuple[torch.Tensor, dict]:
```

**HF (lines 452-480):**
```python
def prepare_inputs_for_generation(
    self,
    input_ids,
    past_key_values=None,
    inputs_embeds=None,
    input_features=None,
    attention_mask=None,
    cache_position=None,
    logits_to_keep=None,
    **kwargs,
):
```

**Impact**: FMS uses FMS-style generation interface, HF uses HF-style. Not directly comparable.

---

### 8. PEFT/LoRA Adapter Handling

**FMS (lines 267-279, 285-297):**
```python
def load_adapter(self, adapter_path: str):
    from peft import PeftModel
    self.decoder = PeftModel.from_pretrained(self.decoder, adapter_path)
    self._peft_adapter_loaded = True

def _maybe_toggle_adapters(self, input_features):
    if input_features is not None:
        self.decoder.enable_adapters()
    else:
        self.decoder.disable_adapters()
```

**HF (lines 517-529):**
```python
def generate(self, *args, **kwargs) -> torch.LongTensor:
    input_features = kwargs.pop("input_features", None)
    if is_peft_available and self._hf_peft_config_loaded:
        if input_features is not None:
            self.enable_adapters()
        else:
            self.disable_adapters()
    return super().generate(*args, input_features=input_features, **kwargs)
```

**Status**: Similar logic, different integration patterns.

---

## Summary of Discrepancies

| Discrepancy | Severity | Impact |
|-------------|----------|--------|
| Decoder structure | MEDIUM | Weight naming differs |
| Config naming (`audio_token_index` vs `audio_token_id`) | LOW | Config mapping needed |
| Forward return type | MEDIUM | API difference |
| Missing forward params | LOW | Feature parity |
| Loss attention mask handling | MEDIUM | Training behavior differs |
| Generation interface | MEDIUM | Different patterns |

---

## Weight Name Mapping

| HF Name | FMS Name |
|---------|----------|
| `language_model.model.embed_tokens` | `decoder.embedding` |
| `language_model.model.layers.*` | `decoder.layers.*` |
| `language_model.model.norm` | `decoder.dec_norm` |
| `language_model.lm_head` | `lm_head` |

---

## Conclusion

**DISCREPANCIES FOUND - MEDIUM SEVERITY**

The implementations are architecturally similar but have differences in:
1. Decoder instantiation (AutoModelForCausalLM vs GraniteHeadless + lm_head)
2. Config parameter naming
3. Forward return types
4. Loss computation (attention mask handling)
5. Generation interface patterns

For inference with proper weight loading, the models should produce equivalent outputs. Training behavior may differ due to loss computation differences.
