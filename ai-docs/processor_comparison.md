# GraniteSpeechProcessor Comparison

## Purpose
This document compares `GraniteSpeechProcessor` (FMS) with `GraniteSpeechProcessor` (HF).

## File Locations
- **FMS**: `fms/models/granite_speech.py` lines 1165-1314
- **HF**: `processing_granite_speech.py` lines 32-96

---

## Class Structure

### HF `GraniteSpeechProcessor`
```python
class GraniteSpeechProcessor(ProcessorMixin):
    # Inherits from ProcessorMixin
```

### FMS `GraniteSpeechProcessor`
```python
class GraniteSpeechProcessor:
    # Standalone class, no inheritance
```

**DISCREPANCY**: FMS doesn't inherit from `ProcessorMixin`.

---

## Constructor (`__init__`) Comparison

### HF (lines 33-41):
```python
def __init__(
    self,
    audio_processor,
    tokenizer,
    audio_token="<|audio|>",
    chat_template=None,
):
    self.audio_token = tokenizer.audio_token if hasattr(tokenizer, "audio_token") else audio_token
    super().__init__(audio_processor, tokenizer, chat_template=chat_template)
```

### FMS (lines 1184-1199):
```python
def __init__(
    self,
    audio_processor: GraniteSpeechFeatureExtractor,
    tokenizer: Any,
    audio_token: str = "<|audio|>",
    **kwargs,
):
    self.audio_processor = audio_processor
    self.tokenizer = tokenizer
    self.audio_token = (
        tokenizer.audio_token
        if hasattr(tokenizer, "audio_token")
        else audio_token
    )
```

### Comparison Table

| Aspect | HF | FMS | Match |
|--------|-----|-----|-------|
| `audio_processor` param | Yes | Yes | YES |
| `tokenizer` param | Yes | Yes | YES |
| `audio_token` param | Yes (default: `"<|audio|>"`) | Yes (default: `"<|audio|>"`) | YES |
| `chat_template` param | Yes | No | **NO** |
| `**kwargs` param | No | Yes | **NO** |
| Store `audio_processor` | Via `super().__init__()` | `self.audio_processor = audio_processor` | Equivalent |
| Store `tokenizer` | Via `super().__init__()` | `self.tokenizer = tokenizer` | Equivalent |
| `audio_token` logic | `tokenizer.audio_token if hasattr(...) else audio_token` | Same logic | YES |

### Discrepancies in `__init__`:

| Issue | Severity | Description |
|-------|----------|-------------|
| Missing `chat_template` | LOW | FMS doesn't support `chat_template` parameter |
| No `ProcessorMixin` inheritance | MEDIUM | FMS doesn't get base class methods |

---

## `__call__` Method Comparison

### HF (lines 43-88):
```python
def __call__(
    self,
    text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]],
    audio: Union["torch.Tensor", list["torch.Tensor"]] = None,
    device: str = "cpu",
    **kwargs,
) -> BatchFeature:
    requires_backends(self, ["torch"])

    text = self._get_validated_text(text)
    prompt_strings = text

    if audio is not None:
        audio_inputs = self.audio_processor(audio, device=device)
        audio_embed_sizes = audio_inputs.pop("audio_embed_sizes")

        prompt_strings = []
        num_replaced = 0
        for sample in text:
            while self.audio_token in sample:
                sample = sample.replace(
                    self.audio_token,
                    "<placeholder>" * audio_embed_sizes[num_replaced],
                    1,
                )
                num_replaced += 1
            prompt_strings.append(sample)

        prompt_strings = [sample.replace("<placeholder>", self.audio_token) for sample in prompt_strings]
    else:
        audio_inputs = {}

    if "padding" not in kwargs:
        kwargs["padding"] = True
    text_inputs = self.tokenizer(prompt_strings, **kwargs)
    return BatchFeature(data={**text_inputs, **audio_inputs})
```

### FMS (lines 1201-1250):
```python
def __call__(
    self,
    text: Union[str, list[str]],
    audio: Optional[Union[torch.Tensor, list[torch.Tensor], np.ndarray, list[np.ndarray]]] = None,
    device: str = "cpu",
    **kwargs,
) -> dict:
    text = self._get_validated_text(text)
    prompt_strings = text

    if audio is not None:
        audio_inputs = self.audio_processor(audio, device=device)
        audio_embed_sizes = audio_inputs.pop("audio_embed_sizes")
        prompt_strings = self._expand_audio_tokens(text, audio_embed_sizes)
    else:
        audio_inputs = {}

    if "padding" not in kwargs:
        kwargs["padding"] = True
    text_inputs = self.tokenizer(prompt_strings, **kwargs)

    return {**text_inputs, **audio_inputs}
```

### Step-by-Step Comparison

| Step | HF | FMS | Match |
|------|-----|-----|-------|
| 1. Backend check | `requires_backends(self, ["torch"])` | None | **NO** |
| 2. Validate text | `self._get_validated_text(text)` | `self._get_validated_text(text)` | YES |
| 3. Init prompt_strings | `prompt_strings = text` | `prompt_strings = text` | YES |
| 4. Process audio | `self.audio_processor(audio, device=device)` | `self.audio_processor(audio, device=device)` | YES |
| 5. Pop embed sizes | `audio_inputs.pop("audio_embed_sizes")` | `audio_inputs.pop("audio_embed_sizes")` | YES |
| 6. Expand tokens | Inline loop | `self._expand_audio_tokens(text, audio_embed_sizes)` | Equivalent |
| 7. Default padding | `kwargs["padding"] = True` | `kwargs["padding"] = True` | YES |
| 8. Tokenize | `self.tokenizer(prompt_strings, **kwargs)` | `self.tokenizer(prompt_strings, **kwargs)` | YES |
| 9. Return type | `BatchFeature(data={...})` | `dict({...})` | **NO** |

### Discrepancies in `__call__`:

| Issue | Severity | Description |
|-------|----------|-------------|
| Missing `requires_backends` | LOW | FMS doesn't check for torch availability |
| Return type | MEDIUM | HF returns `BatchFeature`, FMS returns `dict` |
| Type hints | LOW | FMS accepts `np.ndarray` for audio, HF only `torch.Tensor` |

---

## Audio Token Expansion Logic

### HF (inline in `__call__`, lines 69-81):
```python
prompt_strings = []
num_replaced = 0
for sample in text:
    while self.audio_token in sample:
        sample = sample.replace(
            self.audio_token,
            "<placeholder>" * audio_embed_sizes[num_replaced],
            1,
        )
        num_replaced += 1
    prompt_strings.append(sample)

prompt_strings = [sample.replace("<placeholder>", self.audio_token) for sample in prompt_strings]
```

### FMS (`_expand_audio_tokens` method, lines 1252-1291):
```python
def _expand_audio_tokens(self, text: list[str], audio_embed_sizes: Sequence[int]) -> list[str]:
    prompt_strings = []
    num_replaced = 0

    for sample in text:
        while self.audio_token in sample:
            sample = sample.replace(
                self.audio_token,
                "<placeholder>" * audio_embed_sizes[num_replaced],
                1,
            )
            num_replaced += 1
        prompt_strings.append(sample)

    prompt_strings = [s.replace("<placeholder>", self.audio_token) for s in prompt_strings]
    return prompt_strings
```

### Comparison

| Aspect | HF | FMS | Match |
|--------|-----|-----|-------|
| Algorithm | While loop per sample | While loop per sample | YES |
| Placeholder string | `"<placeholder>"` | `"<placeholder>"` | YES |
| Multiplication | `"<placeholder>" * audio_embed_sizes[num_replaced]` | Same | YES |
| Replace count | `1` (first occurrence only) | `1` (first occurrence only) | YES |
| Counter increment | `num_replaced += 1` | `num_replaced += 1` | YES |
| Final replacement | `sample.replace("<placeholder>", self.audio_token)` | Same | YES |

**Status**: Logic is **IDENTICAL**. FMS just extracts it into a separate method.

---

## `_get_validated_text` Method Comparison

### HF (lines 90-95):
```python
def _get_validated_text(self, text: Union[str, list]) -> list[str]:
    if isinstance(text, str):
        return [text]
    elif isinstance(text, list) and isinstance(text[0], str):
        return text
    raise TypeError("Invalid text provided! Text should be a string or list of strings.")
```

### FMS (lines 1293-1313):
```python
def _get_validated_text(self, text: Union[str, list[str]]) -> list[str]:
    if isinstance(text, str):
        return [text]
    elif isinstance(text, list) and len(text) > 0 and isinstance(text[0], str):
        return text
    raise TypeError("Invalid text provided! Text should be a string or list of strings.")
```

### Comparison

| Aspect | HF | FMS | Match |
|--------|-----|-----|-------|
| String check | `isinstance(text, str)` | Same | YES |
| List check | `isinstance(text, list)` | Same | YES |
| First element check | `isinstance(text[0], str)` | `len(text) > 0 and isinstance(text[0], str)` | **DIFFERENT** |
| Error message | Same | Same | YES |

### DISCREPANCY: Empty List Handling

| Input | HF Behavior | FMS Behavior |
|-------|-------------|--------------|
| `[]` (empty list) | Raises `IndexError` (accesses `text[0]`) | Raises `TypeError` (len check fails) |

**FMS is more robust** - it checks `len(text) > 0` before accessing `text[0]`, avoiding potential `IndexError` on empty lists.

---

## Missing Methods in FMS

HF's `ProcessorMixin` provides additional methods that FMS doesn't have:

| Method | Description | In FMS? |
|--------|-------------|---------|
| `save_pretrained()` | Save processor to disk | NO |
| `from_pretrained()` | Load processor from disk | NO |
| `push_to_hub()` | Upload to HuggingFace Hub | NO |
| `apply_chat_template()` | Format messages with chat template | NO |
| `batch_decode()` | Decode token IDs to text | NO |
| `decode()` | Decode single sequence | NO |

---

## Instance Variables Comparison

| Variable | HF | FMS | Notes |
|----------|-----|-----|-------|
| `audio_processor` | Via ProcessorMixin | `self.audio_processor` | Equivalent |
| `tokenizer` | Via ProcessorMixin | `self.tokenizer` | Equivalent |
| `audio_token` | `self.audio_token` | `self.audio_token` | IDENTICAL |
| `chat_template` | Via ProcessorMixin | None | **MISSING** |

---

## Summary of Discrepancies

| Issue | Severity | Description |
|-------|----------|-------------|
| No `ProcessorMixin` inheritance | MEDIUM | Missing base class methods (save/load, etc.) |
| Missing `chat_template` | LOW | FMS doesn't support chat templates |
| Return type | MEDIUM | HF returns `BatchFeature`, FMS returns plain `dict` |
| Missing `requires_backends` | LOW | No torch availability check |
| Empty list handling | LOW | FMS is actually more robust (checks len > 0) |
| Type hints for audio | LOW | FMS accepts numpy arrays too |

---

## Validation Status

| Check | Status | Notes |
|-------|--------|-------|
| `__init__` parameters | PASS* | *Missing chat_template |
| `audio_token` logic | PASS | Identical |
| Audio processing call | PASS | Identical |
| Token expansion logic | PASS | Identical algorithm |
| Text validation | PASS* | *FMS more robust |
| Tokenization | PASS | Identical |
| Return structure | PARTIAL | Dict vs BatchFeature |

---

## Conclusion

**MINOR DISCREPANCIES** - Implementations are functionally equivalent for core processing.

Key differences:
1. **Return type**: HF uses `BatchFeature`, FMS uses plain `dict` (both dictionary-like)
2. **Inheritance**: FMS doesn't inherit from `ProcessorMixin`, missing utility methods
3. **chat_template**: Not supported in FMS

For inference purposes, the processors should behave identically. The audio token expansion logic is exactly the same.
