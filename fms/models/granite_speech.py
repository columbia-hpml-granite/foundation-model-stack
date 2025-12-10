"""
Granite Speech Model for FMS.

This module implements the Granite Speech multimodal model that combines:
- Conformer encoder (speech/audio processing)
- Q-Former projector (temporal downsampling and cross-modal alignment)
- Granite decoder (language model)

Reference: HuggingFace granite_speech implementation
"""

import logging
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fms import models
from fms.distributed.strategy import DistributedStrategy, NoOpStrategy
from fms.models.conformer import ConformerConfig, ConformerEncoder
from fms.models.granite import Granite, GraniteConfig, GraniteHeadless
from fms.modules.projector import SpeechProjector, SpeechProjectorConfig
from fms.utils.config import ModelConfig

# Optional torchaudio import for mel-spectrogram
try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False


logger = logging.getLogger(__name__)


# ============================================================================
# Default Configurations (matching HuggingFace granite-speech-3.3-8b)
# ============================================================================

# Encoder config matching HF GraniteSpeechEncoderConfig defaults
_default_encoder_config = ConformerConfig(
    num_features=160,         # 80 log-mel * 2 channels
    hidden_dim=1024,          # HF: hidden_dim
    num_layers=16,            # HF: num_layers
    num_heads=8,              # HF: num_heads
    dim_head=128,             # HF: dim_head
    conv_kernel_size=15,      # HF: conv_kernel_size
    conv_expansion_factor=2,  # HF: conv_expansion_factor
    feedforward_mult=4,       # HF: feedforward_mult
    dropout=0.1,              # HF: dropout
    max_pos_emb=512,          # HF: max_pos_emb
    context_size=200,         # HF: context_size
    output_dim=42,            # HF: output_dim (CTC output dimension)
)

# Projector config matching HF Blip2QFormerConfig used in granite_speech
# Note: window_size and downsample_rate are in GraniteSpeechConfig, not here
_default_projector_config = SpeechProjectorConfig(
    encoder_dim=1024,           # Conformer output dimension
    decoder_dim=4096,           # Granite 8B hidden size
    num_hidden_layers=2,        # HF uses 2-layer Q-Former
    num_attention_heads=16,     # HF Blip2QFormer heads
    intermediate_size=4096,     # FFN intermediate size
    hidden_dropout_prob=0.1,
    attention_dropout_prob=0.1,
    hidden_act="gelu",
    layer_norm_eps=1e-12,
    initializer_range=0.02,
)

# Decoder config for Granite 8B (simplified - actual values from model)
_default_decoder_config = GraniteConfig(
    src_vocab_size=49160,
    emb_dim=4096,
    norm_eps=1e-5,
    nheads=32,
    head_dim=128,
    kvheads=8,
    nlayers=40,
    hidden_grow_factor=14336 / 4096,  # ~3.5
    max_expected_seq_len=8192,
    rope_theta=10000.0,
    pad_id=0,
    p_dropout=0.0,
    tie_heads=False,
    fused_weights=True,
)


# ============================================================================
# Configuration Class
# ============================================================================


@dataclass
class GraniteSpeechConfig(ModelConfig):
    """
    Configuration for Granite Speech multimodal model.

    Combines configurations for:
    - Conformer encoder (audio processing)
    - Q-Former projector (temporal downsampling)
    - Granite decoder (language model)

    Reference: HuggingFace granite_speech/configuration_granite_speech.py

    Args:
        encoder_config: ConformerConfig for the audio encoder
        projector_config: SpeechProjectorConfig for Q-Former projector
        decoder_config: GraniteConfig for the language decoder
        audio_token_index: Special token ID for audio placeholder (default: 49155)
        downsample_rate: Temporal downsampling rate in projector (default: 5)
        window_size: Window size for projector's windowed attention (default: 15)
        initializer_range: Std for weight initialization (default: 0.02)
    """
    # Nested sub-configs (FMS pattern from llava_next.py)
    encoder_config: ConformerConfig = field(
        default_factory=lambda: _default_encoder_config
    )
    projector_config: SpeechProjectorConfig = field(
        default_factory=lambda: _default_projector_config
    )
    decoder_config: GraniteConfig = field(
        default_factory=lambda: _default_decoder_config
    )

    # Audio token settings (from HF GraniteSpeechConfig)
    audio_token_index: int = 49155

    # Projector window settings (from HF)
    # These control temporal compression:
    # - window_size: frames per window
    # - downsample_rate: compression ratio per window
    # - num_queries = window_size // downsample_rate = 3
    downsample_rate: int = 5
    window_size: int = 15

    # Initialization
    initializer_range: float = 0.02

    # Training settings
    freeze_encoder: bool = False
    freeze_decoder: bool = False


# ============================================================================
# Model Class (Placeholder for Step 6)
# ============================================================================


class GraniteSpeech(nn.Module):
    """
    Granite Speech multimodal model.

    Combines Conformer encoder, Q-Former projector, and Granite decoder
    for speech-to-text generation.

    Architecture:
        Audio Features (batch, seq_len, 160)
            V
        Conformer Encoder
            V
        Encoder Output (batch, seq_len, 1024)
            V
        Q-Former Projector (windowed)
            V
        Projected Embeddings (batch, num_queries, 4096)
            V
        Merge with Text Embeddings
            V
        Granite Decoder
            V
        Logits
    """

    def __init__(
        self,
        config: Optional[GraniteSpeechConfig] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super(GraniteSpeech, self).__init__()

        # Initialize config
        if config is not None:
            self.config = config
        else:
            self.config = GraniteSpeechConfig()

        # Allow runtime parameter overrides
        self.config = self.config.updated(**kwargs) if kwargs else self.config
        self.distributed_strategy = distributed_strategy

        # Store commonly accessed config values
        self.audio_token_index = self.config.audio_token_index
        self.window_size = self.config.window_size
        self.downsample_rate = self.config.downsample_rate

        # Initialize sub-modules
        # 1. Conformer Encoder
        self.encoder = ConformerEncoder(self.config.encoder_config)

        # 2. Q-Former Projector
        self.projector = SpeechProjector(
            self.config.projector_config,
            window_size=self.config.window_size,
            downsample_rate=self.config.downsample_rate,
        )

        # 3. Granite Decoder
        self.decoder = GraniteHeadless(self.config.decoder_config)

        # 4. LM Head
        self.lm_head = nn.Linear(
            self.config.decoder_config.emb_dim,
            self.config.decoder_config.src_vocab_size,
            bias=False,
        )

        # Freeze modules if configured
        if self.config.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if self.config.freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False
            for param in self.lm_head.parameters():
                param.requires_grad = False

    @classmethod
    def from_config(cls, config: GraniteSpeechConfig) -> "GraniteSpeech":
        """Factory method to construct from config."""
        return cls(config)

    def get_config(self) -> GraniteSpeechConfig:
        """Return current config."""
        return self.config

    def reset_parameters(self):
        """Initialize all trainable parameters."""
        # LM head initialization
        nn.init.normal_(self.lm_head.weight, std=self.config.initializer_range)

    def post_init(self):
        """Post-initialization hook after model is on correct device."""
        # Hook left available for optional weight tying or device-specific setup.
        return

    def get_input_embeddings(self):
        """Get input embeddings from the decoder."""
        return self.decoder.embedding

    def get_output_embeddings(self):
        """Get output embeddings (LM head)."""
        return self.lm_head

    def get_audio_features(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Get audio features projected to decoder embedding space.

        Reference: HF modeling_granite_speech.py:350-354

        Args:
            input_features: Audio features of shape (batch, seq_len, num_features)

        Returns:
            Projected embeddings of shape (batch, num_queries, decoder_dim)
        """
        # 1. Encode audio with Conformer
        encoder_embeds = self.encoder(input_features)

        # 2. Project to decoder embedding space with Q-Former
        projected_embeds = self.projector(encoder_embeds)

        return projected_embeds

    def get_merged_audio_embeddings(
        self,
        input_ids: torch.Tensor,
        audio_features: torch.Tensor,
        input_features_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Merge audio features into text embeddings at audio token positions.

        Reference: HF modeling_granite_speech.py:495-528

        Args:
            input_ids: Input token IDs containing audio token placeholders
            audio_features: Projected audio features from get_audio_features()
            input_features_mask: Optional mask for audio features

        Returns:
            Merged embeddings of shape (batch, seq_len, decoder_dim)
        """
        # Finding positions of audio placeholder tokens
        audio_pos = (input_ids == self.audio_token_index)
    
        # Replacing audio tokens with a safe id for embedding lookup
        safe_ids = torch.where(audio_pos, input_ids.new_zeros(()), input_ids)
    
        # Getting base token embeddings
        token_embeds = self.get_input_embeddings()(safe_ids)
    
        # Audio features come in (B, N_audio, d). Selecting valid features
        audio_features = audio_features.to(token_embeds.device, token_embeds.dtype)
        # Align mask shape with projected audio features; fallback to full mask
        audio_mask = input_features_mask
        if audio_mask.shape != audio_features.shape[:2]:
            logger.warning(
                "input_features_mask shape %s does not match projected audio shape %s; using full mask",
                audio_mask.shape,
                audio_features.shape[:2],
            )
            audio_mask = torch.ones(
                audio_features.shape[:2],
                device=audio_features.device,
                dtype=torch.bool,
            )
        audio_flat = audio_features[audio_mask]  # (K, d)
    
        # Expecting exact match between placeholders and audio vectors
        expected = audio_pos.sum().item()
        if expected * token_embeds.size(-1) != audio_flat.numel():
            raise ValueError(
                f"Mismatch: {expected} audio positions but "
                f"{audio_flat.shape[0]} audio vectors provided."
            )
    
        # Scattering audio vectors into placeholder positions
        mask = audio_pos.unsqueeze(-1)  # (B, T, 1)
        merged = token_embeds.masked_scatter(mask, audio_flat)
    
        return merged

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        input_features_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        past_key_values: Optional[Tuple] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass for Granite Speech.

        Reference: HF modeling_granite_speech.py:357-463

        Args:
            input_ids: Token IDs of shape (batch, seq_len), may contain audio token placeholders
            input_features: Audio features of shape (batch, audio_len, num_features)
            input_features_mask: Optional mask for audio features
            attention_mask: Attention mask for the decoder
            position_ids: Optional position IDs
            inputs_embeds: Pre-computed embeddings (mutually exclusive with input_ids)
            labels: Labels for language modeling loss
            use_cache: Whether to use KV cache
            past_key_values: Cached key/value states

        Returns:
            Tuple of (logits, optional_loss, optional_past_key_values)
        """
        # Check if inputs_embeds is provided via kwargs (from prepare_inputs_for_generation hook)
        # This allows the generation hook to pass pre-computed embeddings
        # Note: If inputs_embeds is in kwargs, Python may have already extracted it to the parameter
        # So we check both the parameter and kwargs
        if inputs_embeds is None and "inputs_embeds" in kwargs:
            inputs_embeds = kwargs.pop("inputs_embeds")
            # If inputs_embeds is provided via hook, ignore input_ids to avoid conflict
            if input_ids is not None:
                input_ids = None
        
        # Input validation
        # Note: input_ids can be None when inputs_embeds is provided (from generation hook)
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Specify input_ids or inputs_embeds.")
        # Only raise error if both are provided and neither is None
        # Allow None input_ids when inputs_embeds is provided (from generation hook)
        # This is the normal case when using prepare_inputs_for_generation hook
        if input_ids is not None and inputs_embeds is not None:
            # Debug: This shouldn't happen when hook returns None for input_ids
            raise ValueError("input_ids and inputs_embeds are mutually exclusive.")
        if input_features is not None and inputs_embeds is not None:
            raise ValueError("input_features and inputs_embeds cannot be used together.")
    
        # Building embeddings
        if inputs_embeds is None:
            if input_features is not None:
                if input_ids is None:
                    raise ValueError("input_ids are required when using input_features.")
                if input_features_mask is None:
                    input_features_mask = input_features.new_ones(
                        input_features.shape[:2], dtype=torch.bool
                    )
                input_features_mask = input_features_mask.to(
                    device=input_features.device, dtype=torch.bool
                )
                if input_features_mask.shape != input_features.shape[:2]:
                    raise ValueError(
                        "input_features_mask must match input_features shape "
                        f"{input_features.shape[:2]}, got {input_features_mask.shape}"
                    )
                # Extracting audio features
                audio_embeds = self.get_audio_features(input_features)
                # Injecting audio into token stream
                inputs_embeds = self.get_merged_audio_embeddings(
                    input_ids=input_ids,
                    audio_features=audio_embeds,
                    input_features_mask=input_features_mask,
                )
            else:
                # Text-only
                inputs_embeds = self.get_input_embeddings()(input_ids)
    
        # Decoder forward
        dec_out, cache = self.decoder(
            x_in=inputs_embeds,
            position_ids=position_ids,
            past_key_value_states=past_key_values,
            attention_mask=attention_mask,
            use_cache=bool(use_cache),
        )
    
        # LM head
        logits = self.lm_head(dec_out)
    
        # Loss
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Avoid NaNs/Infs in loss computation
            shift_logits = torch.nan_to_num(shift_logits)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
    
        if use_cache:
            # Generation expects (logits, cache) when use_cache=True
            # Return cache only, loss is handled separately if needed
            return logits, cache
    
        return logits, loss

    def prepare_inputs_for_generation(
        self,
        iteration: int,
        input_ids: torch.Tensor,
        kwargs: dict,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Prepare inputs for generation with audio features.
        
        This hook is used by fms.utils.generation.generate() to handle audio
        processing during autoregressive generation. On the first iteration,
        audio features are processed and merged into token embeddings. On
        subsequent iterations with caching enabled, audio processing is skipped.
        
        Reference: fms/models/llava_next.py:384-426
        
        Args:
            iteration: Current generation iteration (0 for first/prefill step)
            input_ids: Token IDs of shape (batch, seq_len)
            kwargs: Dictionary containing generation parameters including:
                - use_cache: Whether KV caching is enabled
                - input_features: Audio features (batch, audio_len, num_features)
                - input_features_mask: Optional mask for audio features
                - Other forward pass parameters (attention_mask, position_ids, etc.)
        
        Returns:
            Tuple of (input_ids, updated_kwargs):
                - input_ids: Token IDs for cached steps, or None for first step
                  with audio (inputs_embeds will be in kwargs instead)
                - updated_kwargs: kwargs with audio-related keys removed and
                  inputs_embeds added (for first iteration with audio)
        """
        # Skip audio processing for cached decoding steps
        if kwargs.get("use_cache", False) and iteration > 0:
            # No need to process audio data again in cached decoding stage
            # Remove inputs_embeds from kwargs if it's still there from previous iteration
            kwargs.pop("inputs_embeds", None)
            return input_ids, kwargs
        
        # Extract audio features from kwargs
        input_features = kwargs.pop("input_features", None)
        input_features_mask = kwargs.pop("input_features_mask", None)
        
        # No audio data to process
        if input_features is None:
            return input_ids, kwargs
        
        # First iteration with audio: process and merge audio embeddings
        # This matches the logic in forward() method
        if input_features_mask is None:
            input_features_mask = input_features.new_ones(
                input_features.shape[:2], dtype=torch.bool
            )
        input_features_mask = input_features_mask.to(
            device=input_features.device, dtype=torch.bool
        )
        
        if input_features_mask.shape != input_features.shape[:2]:
            raise ValueError(
                "input_features_mask must match input_features shape "
                f"{input_features.shape[:2]}, got {input_features_mask.shape}"
            )
        
        # Get audio embeddings from encoder + projector
        audio_embeds = self.get_audio_features(input_features)
        
        # Merge audio embeddings into token embeddings at audio token positions
        inputs_embeds = self.get_merged_audio_embeddings(
            input_ids=input_ids,
            audio_features=audio_embeds,
            input_features_mask=input_features_mask,
        )
        
        # Put inputs_embeds in kwargs and return None for input_ids
        # The forward() method will use inputs_embeds from kwargs when input_ids is None
        kwargs["inputs_embeds"] = inputs_embeds
        return None, kwargs


# ============================================================================
# Model Registration (for FMS registry)
# ============================================================================


# Factory function pattern (from granite.py)
def _granite_speech_factory_factory(config: GraniteSpeechConfig):
    """Create factory function for model registration."""
    def factory(**kwargs):
        return GraniteSpeech(config, **kwargs)
    return factory


# Model architecture name
_architecture_name = "granite_speech"

# Register model variants so get_model() can instantiate Granite Speech
_granite_speech_default = GraniteSpeechConfig()

_granite_speech_2b = GraniteSpeechConfig(
    encoder_config=_default_encoder_config,
    projector_config=_default_projector_config.updated(decoder_dim=3072),
    decoder_config=GraniteConfig(
        src_vocab_size=49160,
        emb_dim=3072,
        norm_eps=1e-5,
        nheads=24,
        head_dim=128,
        kvheads=8,
        nlayers=32,
        hidden_grow_factor=11008 / 3072,
        max_expected_seq_len=8192,
        rope_theta=10000.0,
        pad_id=0,
        p_dropout=0.0,
        tie_heads=False,
        fused_weights=True,
    ),
)

models.register_model(
    _architecture_name, "3.3-8b", _granite_speech_factory_factory(_granite_speech_default)
)
# Alias for HF 3.2 naming used in tests/docs
models.register_model(
    _architecture_name, "3.2-8b", _granite_speech_factory_factory(_granite_speech_default)
)
models.register_model(
    _architecture_name, "3.3-2b", _granite_speech_factory_factory(_granite_speech_2b)
)


# ============================================================================
# HuggingFace -> FMS Weight Conversion
# ============================================================================


def _split_kv_weights(
    input_sd: Mapping[str, Any], **kwargs
) -> Mapping[str, Any]:
    """
    Split HF's combined key-value weights into separate FMS key and value weights.

    HF uses: encoder.layers.{i}.attn.to_kv (combined K and V)
    FMS uses: encoder.blocks.{i}.attn.to_k and encoder.blocks.{i}.attn.to_v (separate)
    """
    new_sd = {}

    kv_pattern = re.compile(r"^encoder\.blocks\.(\d+)\.attn\.to_kv\.weight$")

    for name, param in input_sd.items():
        match = kv_pattern.match(name)
        if match:
            layer_idx = match.group(1)
            # Split along the first dimension (output features)
            k, v = param.chunk(2, dim=0)
            new_sd[f"encoder.blocks.{layer_idx}.attn.to_k.weight"] = k
            new_sd[f"encoder.blocks.{layer_idx}.attn.to_v.weight"] = v
        else:
            new_sd[name] = param

    return new_sd


def _hf_to_fms_names(
    input_sd: Mapping[str, Any], **kwargs
) -> Mapping[str, Any]:
    """
    Convert HuggingFace granite_speech weight names to FMS format.

    Maps encoder, projector, and decoder weights from HF naming convention
    to FMS naming convention.
    """
    # Encoder weight mappings
    encoder_replacements = [
        # Input projection
        (r"^encoder\.input_linear", "encoder.input_proj"),
        # Layer index: layers -> blocks
        (r"^encoder\.layers\.(\d+)", r"encoder.blocks.\1"),
        # Feed-forward 1
        (r"\.ff1\.pre_norm\.", ".ff1.norm."),
        (r"\.ff1\.up_proj\.", ".ff1.fc1."),
        (r"\.ff1\.down_proj\.", ".ff1.fc2."),
        # Attention
        (r"\.attn\.pre_norm\.", ".attn.norm."),
        (r"\.attn\.to_kv\.", ".attn.to_kv."),  # Will be split later
        (r"\.attn\.rel_pos_emb\.", ".attn.pos_emb."),
        # Convolution module
        (r"\.conv\.up_conv\.", ".conv.pointwise_conv1."),
        (r"\.conv\.depth_conv\.conv\.", ".conv.depthwise_conv."),
        (r"\.conv\.down_conv\.", ".conv.pointwise_conv2."),
        # Feed-forward 2
        (r"\.ff2\.pre_norm\.", ".ff2.norm."),
        (r"\.ff2\.up_proj\.", ".ff2.fc1."),
        (r"\.ff2\.down_proj\.", ".ff2.fc2."),
    ]

    # Projector weight mappings (from blip_2_qformer)
    projector_replacements = [
        # Query embeddings
        (r"^projector\.query$", "projector.query_embeds"),
        # QFormer encoder layers
        (r"^projector\.qformer\.encoder\.layer\.(\d+)", r"projector.layers.\1"),
        # Self-attention
        (r"\.attention\.attention\.query\.", ".self_attention.query."),
        (r"\.attention\.attention\.key\.", ".self_attention.key."),
        (r"\.attention\.attention\.value\.", ".self_attention.value."),
        (r"\.attention\.output\.dense\.", ".self_attention_output.dense."),
        (r"\.attention\.output\.LayerNorm\.", ".self_attention_output.LayerNorm."),
        # Cross-attention
        (r"\.crossattention\.attention\.query\.", ".cross_attention.query."),
        (r"\.crossattention\.attention\.key\.", ".cross_attention.key."),
        (r"\.crossattention\.attention\.value\.", ".cross_attention.value."),
        (r"\.crossattention\.output\.dense\.", ".cross_attention_output.dense."),
        (r"\.crossattention\.output\.LayerNorm\.", ".cross_attention_output.LayerNorm."),
        # Feed-forward
        (r"\.intermediate_query\.dense\.", ".feed_forward.dense_in."),
        (r"\.output_query\.dense\.", ".feed_forward.dense_out."),
        (r"\.output_query\.LayerNorm\.", ".feed_forward.LayerNorm."),
        # Output projection
        (r"^projector\.linear\.", "projector.output_proj."),
    ]

    # Decoder weight mappings (same as Granite)
    decoder_replacements = [
        (r"^language_model\.lm_head\.weight", "lm_head.weight"),
        (r"^language_model\.model\.embed_tokens\.weight", "decoder.embedding.weight"),
        (r"^language_model\.model\.norm", "decoder.dec_norm"),
        (r"^language_model\.model\.layers", "decoder.layers"),
        (r"self_attn\.k_proj", "attn.in_proj.key"),
        (r"self_attn\.v_proj", "attn.in_proj.value"),
        (r"self_attn\.q_proj", "attn.in_proj.query"),
        (r"self_attn\.o_proj", "attn.dense"),
        (r"mlp\.gate_proj", "ff_sub_layer.wg"),
        (r"mlp\.up_proj", "ff_sub_layer.w1"),
        (r"mlp\.down_proj", "ff_sub_layer.w2"),
        (r"input_layernorm", "ln"),
        (r"post_attention_layernorm", "ff_ln"),
    ]

    # Combine all replacements
    all_replacements = encoder_replacements + projector_replacements + decoder_replacements

    new_sd = {}
    for name, param in input_sd.items():
        new_name = name
        for pattern, repl in all_replacements:
            new_name = re.sub(pattern, repl, new_name)
        new_sd[new_name] = param

    return new_sd


def _granite_speech_weight_fusion(
    input_sd: Mapping[str, Any],
    model_config: Optional[GraniteSpeechConfig] = None,
    **kwargs
) -> Mapping[str, Any]:
    """
    Apply weight fusion for the decoder (same as Granite).

    Note: Encoder and projector don't use fused weights.
    """
    from fms.utils import serialization

    # Only apply fusion to decoder weights
    decoder_sd = {k: v for k, v in input_sd.items() if k.startswith("decoder.")}
    other_sd = {k: v for k, v in input_sd.items() if not k.startswith("decoder.")}

    has_fused_weights = True
    if model_config and model_config.decoder_config:
        if not model_config.decoder_config.fused_weights:
            has_fused_weights = False

    if has_fused_weights and decoder_sd:
        decoder_sd = serialization._mlp_glu_unfused_to_fused_adapter_step(
            serialization._attn_unfused_to_fused_step(decoder_sd)
        )

    # Merge back
    new_sd = {**other_sd, **decoder_sd}
    return new_sd


# Register adapter steps
try:
    from fms.utils import serialization

    serialization.register_adapter_step(
        _architecture_name, "hf_to_fms_names", _hf_to_fms_names
    )

    serialization.register_adapter_step(
        _architecture_name, "split_kv_weights", _split_kv_weights
    )

    serialization.register_adapter_step(
        _architecture_name, "weight_fusion", _granite_speech_weight_fusion
    )

    # Register complete HF adapter
    serialization.register_adapter(
        _architecture_name,
        "hf",
        [
            "hf_to_fms_names",
            "split_kv_weights",
            "weight_fusion",
        ],
    )
except ImportError:
    logger.warning("Could not register serialization adapters - fms.serialization not available")


# ============================================================================
# Feature Extractor and Processor (for Audio Preprocessing)
# ============================================================================


class GraniteSpeechFeatureExtractor:
    """
    Feature extractor for Granite Speech model.

    Converts raw audio waveforms to mel-spectrogram features that can be
    processed by the Conformer encoder.

    Reference: HuggingFace GraniteSpeechFeatureExtractor

    Args:
        sampling_rate: Audio sampling rate in Hz (default: 16000)
        n_fft: FFT window size (default: 512)
        win_length: Window length for STFT (default: 400)
        hop_length: Hop length for STFT (default: 160)
        n_mels: Number of mel filterbanks (default: 80)
        projector_window_size: Window size for projector (default: 15)
        projector_downsample_rate: Downsample rate for projector (default: 5)
    """

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
            raise ImportError(
                "torchaudio is required for GraniteSpeechFeatureExtractor. "
                "Please install it with: pip install torchaudio"
            )

        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.projector_window_size = projector_window_size
        self.projector_downsample_rate = projector_downsample_rate

        # Initialize mel filterbank transform
        self.melspec_kwargs = {
            "sample_rate": sampling_rate,
            "n_fft": n_fft,
            "win_length": win_length,
            "hop_length": hop_length,
            "n_mels": n_mels,
        }
        self.mel_filters = torchaudio.transforms.MelSpectrogram(**self.melspec_kwargs)

    def __call__(
        self,
        audios: Union[torch.Tensor, Sequence[torch.Tensor], np.ndarray, Sequence[np.ndarray]],
        device: Optional[str] = "cpu",
        **kwargs,
    ) -> dict:
        """
        Extract mel-spectrogram features from raw audio.

        Args:
            audios: Raw audio waveforms as tensors, numpy arrays, or sequence thereof
            device: Device to place tensors on (default: "cpu")

        Returns:
            Dictionary containing:
                - input_features: Mel-spectrogram features (batch, seq_len, num_features=160)
                - audio_embed_sizes: List of audio embedding sizes after projection
                - input_features_mask: Boolean mask for valid audio features (batch, max_embed_size)
        """
        speech_inputs = {}

        # Step 1: Validate and batch audio inputs
        batched_audio, audio_lengths = self._get_audios_and_audio_lengths(audios)

        # Step 2: Extract mel-spectrogram features
        speech_inputs["input_features"] = self._extract_mel_spectrograms(
            batched_audio,
            device=device,
        )

        # Step 3: Calculate audio embedding sizes after projection
        audio_embed_sizes = self._get_num_audio_features(audio_lengths)
        speech_inputs["audio_embed_sizes"] = audio_embed_sizes

        # Step 4: Create attention mask for audio features
        # Note: mask shape is (batch, max_embed_size) which differs from input_features shape
        # This mask indicates valid positions in the projected embeddings
        speech_inputs["input_features_mask"] = torch.arange(max(audio_embed_sizes)).view(1, -1) < torch.tensor(
            audio_embed_sizes
        ).view(-1, 1)

        return speech_inputs

    def _extract_mel_spectrograms(
        self,
        audio: torch.Tensor,
        device: str = "cpu"
    ) -> torch.Tensor:
        """
        Compute mel-spectrogram features from raw audio.

        Reference: HF GraniteSpeechFeatureExtractor._extract_mel_spectrograms

        Args:
            audio: Batched audio tensor of shape (batch, audio_len)
            device: Device to place output on

        Returns:
            Mel features of shape (batch, mel_seq_len, num_features=160)
            Note: num_features = n_mels * 2 due to stacking
        """
        # Move mel filters and audio to device
        if device is not None:
            melspec = self.mel_filters.to(device)
            audio = audio.to(device)
        else:
            melspec = self.mel_filters

        bsz = audio.shape[0]

        with torch.no_grad():
            # Step 1: Apply mel filterbank transform
            mel = melspec(audio.float())

            # Step 2: Convert to log-mel and normalize
            logmel = mel.transpose(-1, -2).clip_(min=1e-10).log10_()
            mx = logmel.amax(dim=(-2, -1), keepdim=True)
            logmel = torch.maximum(logmel, mx - 8.0).div_(4).add_(1)

            # Step 3: Remove last frame if odd number of frames
            if logmel.shape[1] % 2 == 1:
                logmel = logmel[:, :-1]

            # Step 4: Stack and skip by 2 (creates 160-dim features from 80 mel bins)
            audio_features = logmel.reshape(bsz, -1, 2 * logmel.shape[-1])

        return audio_features

    def _get_num_audio_features(
        self,
        audio_lengths: Sequence[int]
    ) -> list[int]:
        """
        Calculate number of audio features after projection.

        This accounts for:
        1. Mel-spectrogram downsampling (hop_length)
        2. Encoder frame stacking (2x)
        3. Projector windowing and downsampling

        Reference: HF GraniteSpeechFeatureExtractor._get_num_audio_features

        Args:
            audio_lengths: Sequence of raw audio lengths (in samples)

        Returns:
            List of projected feature lengths
        """
        effective_window_size = self.projector_window_size // self.projector_downsample_rate

        projector_lengths = []
        for raw_length in audio_lengths:
            # Mel sequence length computation
            mel_length = raw_length // self.hop_length + 1
            # Encoder frame takes two mel features
            encoder_length = mel_length // 2
            # Number of blocks in projector
            nblocks = math.ceil(encoder_length / self.projector_window_size)
            # Projector output length
            projector_length = nblocks * effective_window_size
            projector_lengths.append(projector_length)

        return projector_lengths

    def _get_audios_and_audio_lengths(
        self,
        audios: Union[torch.Tensor, Sequence[torch.Tensor], np.ndarray, Sequence[np.ndarray]]
    ) -> Tuple[torch.Tensor, list[int]]:
        """
        Validate and batch audio inputs, extracting lengths.

        Reference: HF GraniteSpeechFeatureExtractor._get_audios_and_audio_lengths

        Args:
            audios: Raw audio as tensor(s) or numpy array(s)

        Returns:
            Tuple of (batched_audio, audio_lengths)
                - batched_audio: Padded audio tensor of shape (batch, max_audio_len)
                - audio_lengths: List of original audio lengths
        """
        # Step 1: Coerce to PyTorch tensors if we have numpy arrays
        if isinstance(audios, np.ndarray):
            audios = torch.from_numpy(audios)
        elif isinstance(audios, Sequence) and len(audios) > 0 and isinstance(audios[0], np.ndarray):
            audios = [torch.from_numpy(arr) for arr in audios]

        # Step 2: Handle different tensor formats
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

            # Step 3: Extract audio lengths before padding
            lengths = [audio.shape[-1] for audio in audios]

            # Squeeze audio to 1D if needed
            audios = [audio.squeeze(0) if audio.ndim > 1 else audio for audio in audios]

            # Step 4: Pad sequences to same length
            audios = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True, padding_value=0.0)
            return audios, lengths

        raise TypeError("Invalid audio provided. Audio should be one or more torch tensors or numpy arrays")


class GraniteSpeechProcessor:
    """
    Processor for Granite Speech model.

    Handles both text tokenization and audio feature extraction, and merges
    them by expanding audio token placeholders in the text.

    Similar to how LlavaNext processes images, this expands the special
    audio token (<|audio|>) in prompts to align with the number of acoustic
    embeddings output from the Projector.

    Reference: HuggingFace GraniteSpeechProcessor

    Args:
        audio_processor: GraniteSpeechFeatureExtractor instance
        tokenizer: Text tokenizer (should have audio_token attribute)
        audio_token: Special token for audio placeholder (default: "<|audio|>")
    """

    def __init__(
        self,
        audio_processor: GraniteSpeechFeatureExtractor,
        tokenizer: Any,  # Typically a HF tokenizer or FMS equivalent
        audio_token: str = "<|audio|>",
        **kwargs,
    ):
        self.audio_processor = audio_processor
        self.tokenizer = tokenizer

        # Use tokenizer's audio_token if available, otherwise use provided default
        self.audio_token = (
            tokenizer.audio_token
            if hasattr(tokenizer, "audio_token")
            else audio_token
        )

    def __call__(
        self,
        text: Union[str, list[str]],
        audio: Optional[Union[torch.Tensor, list[torch.Tensor], np.ndarray, list[np.ndarray]]] = None,
        device: str = "cpu",
        **kwargs,
    ) -> dict:
        """
        Process text and audio inputs for the model.

        Args:
            text: Text prompt(s) containing audio token placeholders
            audio: Optional raw audio waveforms (tensor(s) or numpy array(s))
            device: Device to place tensors on
            **kwargs: Additional arguments passed to tokenizer (e.g., padding, truncation)

        Returns:
            Dictionary containing tokenized text and audio features:
                - input_ids: Tokenized text with expanded audio tokens
                - attention_mask: Attention mask for text
                - input_features: Mel-spectrogram features (if audio provided)
                - input_features_mask: Mask for audio features (if audio provided)
                - audio_embed_sizes: List of audio embedding sizes (if audio provided)
        """
        # Step 1: Validate text input
        text = self._get_validated_text(text)
        prompt_strings = text

        # Step 2: Process audio if provided
        if audio is not None:
            # Extract audio features
            audio_inputs = self.audio_processor(audio, device=device)

            # Get audio embedding sizes (number of tokens after projection)
            audio_embed_sizes = audio_inputs.pop("audio_embed_sizes")

            # Step 3: Expand audio placeholders in text
            # Replace each <|audio|> with N copies of the token,
            # where N = number of embeddings from projector for that audio
            prompt_strings = self._expand_audio_tokens(text, audio_embed_sizes)
        else:
            audio_inputs = {}

        # Step 4: Tokenize text (with expanded audio tokens)
        if "padding" not in kwargs:
            kwargs["padding"] = True
        text_inputs = self.tokenizer(prompt_strings, **kwargs)

        # Step 5: Combine text and audio inputs
        return {**text_inputs, **audio_inputs}

    def _expand_audio_tokens(
        self,
        text: list[str],
        audio_embed_sizes: Sequence[int]
    ) -> list[str]:
        """
        Expand audio token placeholders to match projected feature dimensions.

        This is similar to how LlavaNext expands image tokens. Each <|audio|>
        placeholder is replaced with N copies of the token, where N is the
        number of embeddings that will be output by the projector for that audio.

        Reference: HF GraniteSpeechProcessor.__call__ (lines with placeholder logic)

        Args:
            text: List of text prompts containing <|audio|> tokens
            audio_embed_sizes: Number of embeddings per audio sample

        Returns:
            List of text prompts with expanded audio tokens
        """
        prompt_strings = []
        num_replaced = 0

        for sample in text:
            # Replace each <|audio|> token with the appropriate number of placeholders
            while self.audio_token in sample:
                # Replace one occurrence at a time
                sample = sample.replace(
                    self.audio_token,
                    "<placeholder>" * audio_embed_sizes[num_replaced],
                    1,  # Replace only first occurrence
                )
                num_replaced += 1
            prompt_strings.append(sample)

        # Replace all placeholders with the actual audio token
        prompt_strings = [s.replace("<placeholder>", self.audio_token) for s in prompt_strings]

        return prompt_strings

    def _get_validated_text(
        self,
        text: Union[str, list[str]]
    ) -> list[str]:
        """
        Validate and normalize text input to list of strings.

        Args:
            text: Single string or list of strings

        Returns:
            List of strings

        Raises:
            TypeError: If text is not string or list of strings
        """
        if isinstance(text, str):
            return [text]
        elif isinstance(text, list) and len(text) > 0 and isinstance(text[0], str):
            return text
        raise TypeError("Invalid text provided! Text should be a string or list of strings.")


__all__ = [
    "GraniteSpeech",
    "GraniteSpeechConfig",
    "GraniteSpeechFeatureExtractor",
    "GraniteSpeechProcessor",
]
