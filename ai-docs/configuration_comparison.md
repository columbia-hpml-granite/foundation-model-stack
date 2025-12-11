# Configuration Classes Comparison

## Purpose
This document compares configuration classes between FMS and HF for Granite Speech.

## File Locations
- **HF**: `configuration_granite_speech.py` lines 21-198
- **FMS Encoder**: `fms/models/conformer.py` lines 22-79 (ConformerConfig)
- **FMS Projector**: `fms/modules/projector.py` lines 31-96 (SpeechProjectorConfig)
- **FMS Main**: `fms/models/granite_speech.py` lines 115-166 (GraniteSpeechConfig)

---

## Class Mapping

| HF Class | FMS Class | Notes |
|----------|-----------|-------|
| `GraniteSpeechEncoderConfig` | `ConformerConfig` | Encoder configuration |
| `Blip2QFormerConfig` (via AutoConfig) | `SpeechProjectorConfig` | Projector configuration |
| `GraniteSpeechConfig` | `GraniteSpeechConfig` | Main composite config |

---

## 1. Encoder Configuration Comparison

### HF `GraniteSpeechEncoderConfig` vs FMS `ConformerConfig`

#### Parameter Comparison

| Parameter | HF Name | HF Default | FMS Name | FMS Default | Match |
|-----------|---------|------------|----------|-------------|-------|
| Input dimension | `input_dim` | 160 | `num_features` | 160 | **DIFFERENT NAME** |
| Number of layers | `num_layers` | 10 | `num_layers` | 16 | **DIFFERENT DEFAULT** |
| Hidden dimension | `hidden_dim` | 1024 | `hidden_dim` | 1024 | YES |
| FF multiplier | `feedforward_mult` | 4 | `feedforward_mult` | 4 | YES |
| Attention heads | `num_heads` | 8 | `num_heads` | 8 | YES |
| Head dimension | `dim_head` | 128 | `dim_head` | 128 | YES |
| CTC output dim | `output_dim` | 42 | `output_dim` | 42 | YES |
| Context size | `context_size` | 200 | `context_size` | 200 | YES |
| Max pos embedding | `max_pos_emb` | 512 | `max_pos_emb` | 512 | YES |
| Dropout | `dropout` | 0.1 | `dropout` | 0.1 | YES |
| Conv kernel size | `conv_kernel_size` | 15 | `conv_kernel_size` | 15 | YES |
| Conv expansion | `conv_expansion_factor` | 2 | `conv_expansion_factor` | 2 | YES |
| Use CTC | N/A | N/A | `use_ctc` | True | **FMS ONLY** |
| Activation | N/A | N/A | `activation` | "silu" | **FMS ONLY** |
| Linear config | N/A | N/A | `linear_config` | None | **FMS ONLY** |

#### Discrepancies

| Issue | Severity | Description |
|-------|----------|-------------|
| Parameter name: `input_dim` vs `num_features` | MEDIUM | Different naming for same concept |
| Default `num_layers`: 10 vs 16 | **HIGH** | HF defaults to 10, FMS defaults to 16 |
| Missing `use_ctc` in HF | LOW | FMS has optional CTC flag |
| Missing `activation` in HF | LOW | HF hardcodes SiLU |

---

## 2. Projector Configuration Comparison

### HF `Blip2QFormerConfig` vs FMS `SpeechProjectorConfig`

HF uses `Blip2QFormerConfig` from the BLIP-2 model via `AutoConfig`. Let me compare key parameters:

#### Parameter Comparison

| Parameter | HF (Blip2QFormer) | FMS (SpeechProjectorConfig) | Match |
|-----------|-------------------|----------------------------|-------|
| Hidden size | `hidden_size` | `encoder_dim` | **DIFFERENT NAME** |
| Output dimension | Via parent config | `decoder_dim` | **DIFFERENT STRUCTURE** |
| Window size | Via parent `GraniteSpeechConfig` | `window_size` | **DIFFERENT LOCATION** |
| Downsample rate | Via parent `GraniteSpeechConfig` | `downsample_rate` | **DIFFERENT LOCATION** |
| Num queries | Computed at runtime | `num_queries` | **FMS STORES EXPLICITLY** |
| Num layers | `num_hidden_layers` | `num_hidden_layers` | YES |
| Attention heads | `num_attention_heads` | `num_attention_heads` | YES |
| FFN size | `intermediate_size` | `intermediate_size` | YES |
| Hidden dropout | `hidden_dropout_prob` | `hidden_dropout_prob` | YES |
| Attention dropout | `attention_probs_dropout_prob` | `attention_dropout_prob` | **DIFFERENT NAME** |
| Layer norm eps | `layer_norm_eps` | `layer_norm_eps` | YES |
| Initializer range | `initializer_range` | `initializer_range` | YES |
| Cross-attention freq | `cross_attention_frequency` | N/A | **MISSING IN FMS** |
| Encoder hidden size | `encoder_hidden_size` | N/A | **MISSING IN FMS** |

#### Critical Missing Parameters in FMS

| Parameter | HF Default | Description | Impact |
|-----------|------------|-------------|--------|
| `cross_attention_frequency` | 2 | How often to add cross-attention | **CRITICAL** - affects layer structure |
| `encoder_hidden_size` | varies | K/V projection dimension in cross-attention | **CRITICAL** - affects weight shapes |
| `use_qformer_text_input` | False | Whether to use text FFN | May affect architecture |

---

## 3. Main Configuration Comparison

### HF `GraniteSpeechConfig` vs FMS `GraniteSpeechConfig`

#### Parameter Comparison

| Parameter | HF Name | HF Default | FMS Name | FMS Default | Match |
|-----------|---------|------------|----------|-------------|-------|
| Text config | `text_config` | `GraniteConfig()` | `decoder_config` | `GraniteConfig()` | **DIFFERENT NAME** |
| Encoder config | `encoder_config` | `GraniteSpeechEncoderConfig()` | `encoder_config` | `ConformerConfig()` | YES (type differs) |
| Projector config | `projector_config` | `Blip2QFormerConfig()` | `projector_config` | `SpeechProjectorConfig()` | YES (type differs) |
| Audio token ID | `audio_token_index` | 49155 | `audio_token_index` | 49159 | **DIFFERENT DEFAULT** |
| Has LoRA adapter | `has_lora_adapter` | True | `has_lora_adapter` | True | YES |
| Downsample rate | `downsample_rate` | 5 | `downsample_rate` | 5 | YES |
| Window size | `window_size` | 15 | `window_size` | 15 | YES |
| Initializer range | `initializer_range` | 0.02 | `initializer_range` | 0.02 | YES |
| Freeze encoder | N/A | N/A | `freeze_encoder` | False | **FMS ONLY** |
| Freeze decoder | N/A | N/A | `freeze_decoder` | False | **FMS ONLY** |

#### Class Attributes

| Attribute | HF | FMS | Notes |
|-----------|-----|-----|-------|
| `model_type` | `"granite_speech"` | N/A (dataclass) | HF-specific |
| `attribute_map` | `{"audio_token_id": "audio_token_index"}` | N/A | HF-specific alias |
| `sub_configs` | Dict mapping | N/A | HF-specific |

#### Discrepancies

| Issue | Severity | Description |
|-------|----------|-------------|
| `text_config` vs `decoder_config` | MEDIUM | Different naming |
| `audio_token_index`: 49155 vs 49159 | **HIGH** | Different defaults - will affect inference! |
| Missing `freeze_*` in HF | LOW | FMS has additional training controls |

---

## 4. Config Instantiation Logic

### HF Config Instantiation

```python
# HF GraniteSpeechConfig.__init__
if isinstance(text_config, dict):
    text_config["model_type"] = text_config.get("model_type", "granite")
    text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
elif text_config is None:
    text_config = CONFIG_MAPPING["granite"]()

if isinstance(projector_config, dict):
    projector_config["model_type"] = projector_config.get("model_type", "blip_2_qformer")
    projector_config = CONFIG_MAPPING[projector_config["model_type"]](**projector_config)
elif projector_config is None:
    projector_config = CONFIG_MAPPING["blip_2_qformer"]()

if not isinstance(encoder_config, GraniteSpeechEncoderConfig):
    encoder_config = {} if encoder_config is None else encoder_config
    encoder_config = GraniteSpeechEncoderConfig(**encoder_config)
```

### FMS Config Instantiation

FMS uses dataclass `field(default_factory=...)` for lazy initialization:

```python
encoder_config: ConformerConfig = field(default_factory=lambda: _default_encoder_config)
projector_config: SpeechProjectorConfig = field(default_factory=lambda: _default_projector_config)
decoder_config: GraniteConfig = field(default_factory=lambda: _default_decoder_config)
```

**Difference**: HF dynamically resolves config types via `CONFIG_MAPPING`, FMS uses fixed types.

---

## 5. Inheritance

| Class | HF Base | FMS Base |
|-------|---------|----------|
| Encoder Config | `PreTrainedConfig` | `ModelConfig` (dataclass) |
| Projector Config | `PreTrainedConfig` | `ModelConfig` (dataclass) |
| Main Config | `PreTrainedConfig` | `ModelConfig` (dataclass) |

**Impact**: HF configs have serialization methods (save/load), FMS configs are plain dataclasses.

---

## Summary of All Discrepancies

### Critical (Affects Model Behavior)

| Issue | Location | Description |
|-------|----------|-------------|
| `num_layers` default | Encoder | HF=10, FMS=16 |
| `audio_token_index` default | Main | HF=49155, FMS=49159 |
| Missing `cross_attention_frequency` | Projector | FMS Q-Former has cross-attention every layer |
| Missing `encoder_hidden_size` | Projector | FMS uses same dim for Q/K/V in cross-attention |

### Medium (Naming/API Differences)

| Issue | Location | Description |
|-------|----------|-------------|
| `input_dim` vs `num_features` | Encoder | Different parameter name |
| `text_config` vs `decoder_config` | Main | Different parameter name |
| `attention_probs_dropout_prob` vs `attention_dropout_prob` | Projector | Different parameter name |

### Low (Extra Features)

| Issue | Location | Description |
|-------|----------|-------------|
| `use_ctc` | Encoder | FMS-only parameter |
| `activation` | Encoder | FMS-only parameter |
| `freeze_encoder/decoder` | Main | FMS-only parameters |

---

## Weight Loading Implications

When loading HF weights into FMS, the following config mappings are needed:

```python
# Encoder config mapping
hf_encoder_config.input_dim -> fms_conformer_config.num_features
hf_encoder_config.num_layers -> fms_conformer_config.num_layers  # CHECK VALUE!

# Main config mapping
hf_config.text_config -> fms_config.decoder_config
hf_config.audio_token_index -> fms_config.audio_token_index  # CHECK VALUE!

# Projector config mapping (CRITICAL - architecture differences)
hf_config.projector_config.cross_attention_frequency -> NOT SUPPORTED
hf_config.projector_config.encoder_hidden_size -> NOT SUPPORTED
```

---

## Conclusion

**DISCREPANCIES FOUND - SOME CRITICAL**

### Must Fix:
1. **`num_layers` default**: FMS defaults to 16, HF to 10 - must match model checkpoint
2. **`audio_token_index` default**: FMS=49159, HF=49155 - will cause inference errors
3. **Projector config**: Missing `cross_attention_frequency` and `encoder_hidden_size` (see projector_comparison.md)

### Should Align:
1. Parameter naming (`input_dim`/`num_features`, `text_config`/`decoder_config`)
2. Config inheritance (dataclass vs PreTrainedConfig)

### Optional:
1. Extra FMS parameters (`use_ctc`, `freeze_*`) are fine as additions
