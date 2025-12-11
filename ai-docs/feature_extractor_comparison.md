# GraniteSpeechFeatureExtractor Comparison

## Purpose
This document compares `GraniteSpeechFeatureExtractor` (FMS) with `GraniteSpeechFeatureExtractor` (HF).

## File Locations
- **FMS**: `fms/models/granite_speech.py` lines 935-1163
- **HF**: `feature_extraction_granite_speech.py` lines 38-186

---

## Class Structure

### HF
```python
class GraniteSpeechFeatureExtractor(FeatureExtractionMixin):
    model_input_names = ["input_features"]
```

### FMS
```python
class GraniteSpeechFeatureExtractor:
    # Standalone class, no inheritance
```

**DISCREPANCY**: FMS doesn't inherit from `FeatureExtractionMixin`, missing `model_input_names` class attribute.

---

## Constructor (`__init__`) Comparison

### HF (lines 41-64):
```python
def __init__(
    self,
    sampling_rate: int = 16000,
    n_fft: int = 512,
    win_length: int = 400,
    hop_length: int = 160,
    n_mels: int = 80,
    projector_window_size: int = 15,
    projector_downsample_rate: int = 5,
    **kwargs,
):
    super().__init__(**kwargs)
    self.sampling_rate = sampling_rate
    self.melspec_kwargs = {
        "sample_rate": sampling_rate,
        "n_fft": n_fft,
        "win_length": win_length,
        "hop_length": hop_length,
        "n_mels": n_mels,
    }
    requires_backends(self, ["torchaudio"])
    self.mel_filters = torchaudio.transforms.MelSpectrogram(**self.melspec_kwargs)
    self.projector_window_size = projector_window_size
    self.projector_downsample_rate = projector_downsample_rate
```

### FMS (lines 954-987):
```python
def __init__(
    self,
    sampling_rate: int = 16000,
    n_fft: int = 512,
    win_length: int = 400,
    hop_length: int = 160,
    n_mels: int = 80,
    projector_window_size: int = 15,
    projector_downsample_rate: int = 5,
    **kwargs,
):
    if not TORCHAUDIO_AVAILABLE:
        raise ImportError(...)

    self.sampling_rate = sampling_rate
    self.n_fft = n_fft
    self.win_length = win_length
    self.hop_length = hop_length
    self.n_mels = n_mels
    self.projector_window_size = projector_window_size
    self.projector_downsample_rate = projector_downsample_rate

    self.melspec_kwargs = {
        "sample_rate": sampling_rate,
        "n_fft": n_fft,
        "win_length": win_length,
        "hop_length": hop_length,
        "n_mels": n_mels,
    }
    self.mel_filters = torchaudio.transforms.MelSpectrogram(**self.melspec_kwargs)
```

### Parameter Comparison

| Parameter | HF Default | FMS Default | Match |
|-----------|------------|-------------|-------|
| `sampling_rate` | 16000 | 16000 | YES |
| `n_fft` | 512 | 512 | YES |
| `win_length` | 400 | 400 | YES |
| `hop_length` | 160 | 160 | YES |
| `n_mels` | 80 | 80 | YES |
| `projector_window_size` | 15 | 15 | YES |
| `projector_downsample_rate` | 5 | 5 | YES |

### Instance Variable Comparison

| Variable | HF | FMS | Notes |
|----------|-----|-----|-------|
| `sampling_rate` | YES | YES | IDENTICAL |
| `n_fft` | NO (only in melspec_kwargs) | YES | FMS stores separately |
| `win_length` | NO (only in melspec_kwargs) | YES | FMS stores separately |
| `hop_length` | NO (only in melspec_kwargs) | YES | FMS stores separately |
| `n_mels` | NO (only in melspec_kwargs) | YES | FMS stores separately |
| `melspec_kwargs` | YES | YES | IDENTICAL structure |
| `mel_filters` | YES | YES | IDENTICAL |
| `projector_window_size` | YES | YES | IDENTICAL |
| `projector_downsample_rate` | YES | YES | IDENTICAL |

**Note**: FMS stores individual mel params as instance variables in addition to `melspec_kwargs`. This is a minor difference - FMS provides easier access to individual parameters.

---

## `__call__` Method Comparison

### HF (lines 66-93):
```python
def __call__(
    self,
    audios: AudioInput,
    device: Optional[str] = "cpu",
) -> BatchFeature:
    requires_backends(self, ["torchaudio"])

    speech_inputs = {}
    batched_audio, audio_lengths = self._get_audios_and_audio_lengths(audios)
    speech_inputs["input_features"] = self._extract_mel_spectrograms(
        batched_audio,
        device=device,
    )
    audio_embed_sizes = self._get_num_audio_features(audio_lengths)
    speech_inputs["audio_embed_sizes"] = audio_embed_sizes
    speech_inputs["input_features_mask"] = torch.arange(max(audio_embed_sizes)).view(1, -1) < torch.tensor(
        audio_embed_sizes
    ).view(-1, 1)
    return BatchFeature(data=speech_inputs)
```

### FMS (lines 989-1030):
```python
def __call__(
    self,
    audios: Union[torch.Tensor, Sequence[torch.Tensor], np.ndarray, Sequence[np.ndarray]],
    device: Optional[str] = "cpu",
    **kwargs,
) -> dict:
    speech_inputs = {}

    batched_audio, audio_lengths = self._get_audios_and_audio_lengths(audios)

    speech_inputs["input_features"] = self._extract_mel_spectrograms(
        batched_audio,
        device=device,
    )

    audio_embed_sizes = self._get_num_audio_features(audio_lengths)
    speech_inputs["audio_embed_sizes"] = audio_embed_sizes

    speech_inputs["input_features_mask"] = torch.arange(max(audio_embed_sizes)).view(1, -1) < torch.tensor(
        audio_embed_sizes
    ).view(-1, 1)

    return speech_inputs
```

### Step-by-Step Comparison

| Step | HF | FMS | Match |
|------|-----|-----|-------|
| 1. Backend check | `requires_backends(self, ["torchaudio"])` | None (checked in `__init__`) | Different timing |
| 2. Get audio/lengths | `self._get_audios_and_audio_lengths(audios)` | Same | YES |
| 3. Extract mel | `self._extract_mel_spectrograms(batched_audio, device=device)` | Same | YES |
| 4. Get embed sizes | `self._get_num_audio_features(audio_lengths)` | Same | YES |
| 5. Create mask | `torch.arange(max(...)).view(1, -1) < torch.tensor(...).view(-1, 1)` | **IDENTICAL** | YES |
| 6. Return type | `BatchFeature(data=speech_inputs)` | `speech_inputs` (dict) | **DIFFERENT** |

### Mask Creation Logic (IDENTICAL)

```python
# Both HF and FMS:
torch.arange(max(audio_embed_sizes)).view(1, -1) < torch.tensor(audio_embed_sizes).view(-1, 1)
```

This creates a boolean mask of shape `(batch_size, max_embed_size)` where `True` indicates valid positions.

---

## `_extract_mel_spectrograms` Method Comparison

### HF (lines 95-120):
```python
def _extract_mel_spectrograms(self, audio: "torch.Tensor", device="cpu"):
    requires_backends(self, ["torchaudio"])
    if device is not None:
        melspec = self.mel_filters.to(device)
        audio = audio.to(device)
    else:
        melspec = self.mel_filters

    bsz = audio.shape[0]
    with torch.no_grad():
        mel = melspec(audio.float())
        logmel = mel.transpose(-1, -2).clip_(min=1e-10).log10_()
        mx = logmel.amax(dim=(-2, -1), keepdim=True)
        logmel = torch.maximum(logmel, mx - 8.0).div_(4).add_(1)
        if logmel.shape[1] % 2 == 1:
            logmel = logmel[:, :-1]
        audio = logmel.reshape(bsz, -1, 2 * logmel.shape[-1])

    return audio
```

### FMS (lines 1032-1075):
```python
def _extract_mel_spectrograms(self, audio: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    if device is not None:
        melspec = self.mel_filters.to(device)
        audio = audio.to(device)
    else:
        melspec = self.mel_filters

    bsz = audio.shape[0]

    with torch.no_grad():
        mel = melspec(audio.float())
        logmel = mel.transpose(-1, -2).clip_(min=1e-10).log10_()
        mx = logmel.amax(dim=(-2, -1), keepdim=True)
        logmel = torch.maximum(logmel, mx - 8.0).div_(4).add_(1)
        if logmel.shape[1] % 2 == 1:
            logmel = logmel[:, :-1]
        audio_features = logmel.reshape(bsz, -1, 2 * logmel.shape[-1])

    return audio_features
```

### Step-by-Step Comparison

| Step | HF | FMS | Match |
|------|-----|-----|-------|
| 1. Device handling | `melspec.to(device); audio.to(device)` | Same | YES |
| 2. Get batch size | `bsz = audio.shape[0]` | Same | YES |
| 3. Compute mel | `mel = melspec(audio.float())` | Same | YES |
| 4. Transpose | `mel.transpose(-1, -2)` | Same | YES |
| 5. Clip min | `.clip_(min=1e-10)` | Same | YES |
| 6. Log10 | `.log10_()` | Same | YES |
| 7. Get max | `logmel.amax(dim=(-2, -1), keepdim=True)` | Same | YES |
| 8. Dynamic range | `torch.maximum(logmel, mx - 8.0)` | Same | YES |
| 9. Normalize | `.div_(4).add_(1)` | Same | YES |
| 10. Remove odd frame | `if logmel.shape[1] % 2 == 1: logmel = logmel[:, :-1]` | Same | YES |
| 11. Stack frames | `logmel.reshape(bsz, -1, 2 * logmel.shape[-1])` | Same | YES |

**Status**: **IDENTICAL** mel-spectrogram extraction logic.

### Mel-Spectrogram Normalization Formula

Both implementations use the same normalization:
```python
# 1. Clip to prevent log(0)
logmel = mel.clip_(min=1e-10).log10_()

# 2. Dynamic range compression (relative to max)
mx = logmel.amax(dim=(-2, -1), keepdim=True)
logmel = torch.maximum(logmel, mx - 8.0)  # Limit dynamic range to 8 orders of magnitude

# 3. Normalize to approximately [0, 2] range
logmel = logmel.div_(4).add_(1)
```

---

## `_get_num_audio_features` Method Comparison

### HF (lines 122-145):
```python
def _get_num_audio_features(self, audio_lengths: Sequence[int]) -> Sequence[int]:
    hop_length = self.melspec_kwargs["hop_length"]
    effective_window_size = self.projector_window_size // self.projector_downsample_rate

    projector_lengths = []
    for raw_length in audio_lengths:
        mel_length = raw_length // hop_length + 1
        encoder_length = mel_length // 2
        nblocks = math.ceil(encoder_length / self.projector_window_size)
        projector_length = nblocks * effective_window_size
        projector_lengths.append(projector_length)

    return projector_lengths
```

### FMS (lines 1077-1111):
```python
def _get_num_audio_features(self, audio_lengths: Sequence[int]) -> list[int]:
    effective_window_size = self.projector_window_size // self.projector_downsample_rate

    projector_lengths = []
    for raw_length in audio_lengths:
        mel_length = raw_length // self.hop_length + 1
        encoder_length = mel_length // 2
        nblocks = math.ceil(encoder_length / self.projector_window_size)
        projector_length = nblocks * effective_window_size
        projector_lengths.append(projector_length)

    return projector_lengths
```

### Comparison

| Step | HF | FMS | Match |
|------|-----|-----|-------|
| Get hop_length | `self.melspec_kwargs["hop_length"]` | `self.hop_length` | Equivalent |
| effective_window_size | `projector_window_size // projector_downsample_rate` | Same | YES |
| mel_length | `raw_length // hop_length + 1` | Same | YES |
| encoder_length | `mel_length // 2` | Same | YES |
| nblocks | `math.ceil(encoder_length / projector_window_size)` | Same | YES |
| projector_length | `nblocks * effective_window_size` | Same | YES |

**Status**: **IDENTICAL** computation logic. Only difference is how `hop_length` is accessed (dict lookup vs instance variable).

### Length Calculation Formula

```
mel_length = raw_audio_samples // hop_length + 1
encoder_length = mel_length // 2  (due to 2x stacking)
nblocks = ceil(encoder_length / projector_window_size)
projector_length = nblocks * (projector_window_size // projector_downsample_rate)
```

---

## `_get_audios_and_audio_lengths` Method Comparison

### HF (lines 147-183):
```python
def _get_audios_and_audio_lengths(self, audios: AudioInput) -> Sequence["torch.Tensor", Sequence[int]]:
    requires_backends(self, ["torch"])

    # Coerce numpy to torch
    if isinstance(audios, np.ndarray):
        audios = torch.from_numpy(audios)
    elif isinstance(audios, Sequence) and isinstance(audios[0], np.ndarray):
        audios = [torch.from_numpy(arr) for arr in audios]

    if isinstance(audios, torch.Tensor):
        if audios.ndim == 1:
            audios = audios.unsqueeze(0)
        if not torch.is_floating_point(audios):
            raise ValueError("Invalid audio provided. Audio should be a floating point between 0 and 1")

        if audios.shape[0] > 1:
            logger.warning("Audio samples are already collated; assuming they all have the same length")
        lengths = [audios.shape[-1]] * audios.shape[0]
        return audios, lengths

    elif isinstance(audios, Sequence) and isinstance(audios[0], torch.Tensor):
        if not torch.is_floating_point(audios[0]):
            raise ValueError("Invalid audio provided. Audio should be a floating point between 0 and 1")
        lengths = [audio.shape[-1] for audio in audios]
        audios = [audio.squeeze(0) for audio in audios]
        audios = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True, padding_value=0.0)
        return audios, lengths

    raise TypeError("Invalid audio provided. Audio should be a one or more torch tensors or numpy arrays")
```

### FMS (lines 1113-1162):
```python
def _get_audios_and_audio_lengths(
    self,
    audios: Union[torch.Tensor, Sequence[torch.Tensor], np.ndarray, Sequence[np.ndarray]]
) -> Tuple[torch.Tensor, list[int]]:
    # Coerce numpy to torch
    if isinstance(audios, np.ndarray):
        audios = torch.from_numpy(audios)
    elif isinstance(audios, Sequence) and len(audios) > 0 and isinstance(audios[0], np.ndarray):
        audios = [torch.from_numpy(arr) for arr in audios]

    if isinstance(audios, torch.Tensor):
        if audios.ndim == 1:
            audios = audios.unsqueeze(0)
        if not torch.is_floating_point(audios):
            raise ValueError("Invalid audio provided. Audio should be a floating point tensor")

        if audios.shape[0] > 1:
            logger.warning("Audio samples are already collated; assuming they all have the same length")
        lengths = [audios.shape[-1]] * audios.shape[0]
        return audios, lengths

    elif isinstance(audios, Sequence) and len(audios) > 0 and isinstance(audios[0], torch.Tensor):
        if not torch.is_floating_point(audios[0]):
            raise ValueError("Invalid audio provided. Audio should be a floating point tensor")

        lengths = [audio.shape[-1] for audio in audios]
        audios = [audio.squeeze(0) if audio.ndim > 1 else audio for audio in audios]
        audios = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True, padding_value=0.0)
        return audios, lengths

    raise TypeError("Invalid audio provided. Audio should be one or more torch tensors or numpy arrays")
```

### Comparison

| Step | HF | FMS | Match |
|------|-----|-----|-------|
| Backend check | `requires_backends(self, ["torch"])` | None | Different |
| Numpy single | `torch.from_numpy(audios)` | Same | YES |
| Numpy list check | `isinstance(audios[0], np.ndarray)` | `len(audios) > 0 and isinstance(audios[0], np.ndarray)` | **FMS MORE ROBUST** |
| 1D unsqueeze | `audios.unsqueeze(0)` | Same | YES |
| Float check | `torch.is_floating_point(audios)` | Same | YES |
| Multi-batch warning | `logger.warning(...)` | Same message | YES |
| Lengths from tensor | `[audios.shape[-1]] * audios.shape[0]` | Same | YES |
| List tensor check | `isinstance(audios[0], torch.Tensor)` | `len(audios) > 0 and isinstance(audios[0], torch.Tensor)` | **FMS MORE ROBUST** |
| Squeeze logic | `audio.squeeze(0)` | `audio.squeeze(0) if audio.ndim > 1 else audio` | **FMS MORE ROBUST** |
| Pad sequence | `pad_sequence(audios, batch_first=True, padding_value=0.0)` | Same | YES |
| Error message | Same | Same | YES |

### Discrepancies

| Issue | Severity | Description |
|-------|----------|-------------|
| Empty list check | LOW | FMS checks `len(audios) > 0` before accessing `audios[0]`, preventing IndexError |
| Squeeze logic | LOW | FMS only squeezes if `ndim > 1`, preventing errors on 1D tensors |
| Error message | TRIVIAL | Slightly different wording ("between 0 and 1" vs "tensor") |

**FMS is more robust** - handles edge cases better (empty lists, already-1D tensors).

---

## Missing Features in FMS

### From FeatureExtractionMixin

| Method | Description | In FMS? |
|--------|-------------|---------|
| `save_pretrained()` | Save config to disk | NO |
| `from_pretrained()` | Load config from disk | NO |
| `push_to_hub()` | Upload to HuggingFace Hub | NO |
| `to_dict()` | Convert config to dict | NO |
| `to_json_string()` | Convert config to JSON | NO |

### Class Attributes

| Attribute | HF | FMS |
|-----------|-----|-----|
| `model_input_names` | `["input_features"]` | None |

---

## Summary of Discrepancies

| Issue | Severity | Description |
|-------|----------|-------------|
| No `FeatureExtractionMixin` | MEDIUM | Missing save/load/serialization methods |
| Return type | LOW | HF returns `BatchFeature`, FMS returns `dict` |
| Missing `model_input_names` | LOW | Missing class attribute |
| Extra instance variables | NONE | FMS stores mel params separately (more convenient) |
| Empty list handling | NONE | FMS is more robust |
| Squeeze edge case | NONE | FMS handles 1D tensors better |

---

## Validation Status

| Check | Status | Notes |
|-------|--------|-------|
| Constructor parameters | PASS | All defaults identical |
| `melspec_kwargs` | PASS | Identical structure |
| `mel_filters` creation | PASS | Identical |
| Mel extraction algorithm | PASS | **IDENTICAL** |
| Length calculation | PASS | **IDENTICAL** |
| Audio validation | PASS | FMS more robust |
| Mask creation | PASS | **IDENTICAL** |
| Return structure | PASS* | *Dict vs BatchFeature |

---

## Conclusion

**MINOR DISCREPANCIES** - Core feature extraction logic is **IDENTICAL**.

Key findings:
1. **Mel-spectrogram extraction**: Identical algorithm (transpose, clip, log10, normalize, stack)
2. **Length calculation**: Identical formula
3. **Mask creation**: Identical logic
4. **Audio validation**: FMS is actually more robust (empty list checks, squeeze edge cases)

The only differences are:
- Return type (`dict` vs `BatchFeature`)
- Missing base class inheritance (affects serialization utilities)
- FMS stores individual mel params as instance variables

For inference, both implementations will produce **identical outputs**.
