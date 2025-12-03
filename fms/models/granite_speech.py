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
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fms.distributed.strategy import DistributedStrategy, NoOpStrategy
from fms.models.conformer import ConformerConfig, ConformerEncoder
from fms.models.granite import Granite, GraniteConfig, GraniteHeadless
from fms.modules.projector import SpeechProjector, SpeechProjectorConfig
from fms.utils.config import ModelConfig


logger = logging.getLogger(__name__)


# ============================================================================
# Default Configurations (matching HuggingFace granite-speech-3.3-8b)
# ============================================================================

# Encoder config matching HF GraniteSpeechEncoderConfig defaults
_default_encoder_config = ConformerConfig(
    num_features=160,       # 80 log-mel * 2 channels
    hidden_dim=1024,        # HF: hidden_dim
    num_layers=16,          # HF: num_layers (actual model uses 16, HF default is 10)
    num_heads=8,            # HF: num_heads
    dim_head=128,           # HF: dim_head
    conv_kernel_size=15,    # HF: conv_kernel_size
    conv_expansion_factor=2,  # HF: conv_expansion_factor
    feedforward_mult=4,     # HF: feedforward_mult
    dropout=0.1,            # HF: dropout
    max_pos_emb=512,        # HF: max_pos_emb
    context_size=200,       # HF: context_size
    output_dim=42,          # HF: output_dim (CTC output dimension)
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
            ↓
        Conformer Encoder
            ↓
        Encoder Output (batch, seq_len, 1024)
            ↓
        Q-Former Projector (windowed)
            ↓
        Projected Embeddings (batch, num_queries, 4096)
            ↓
        Merge with Text Embeddings
            ↓
        Granite Decoder
            ↓
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
        audio_flat = audio_features[input_features_mask]  # (K, d)
    
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
        # Input validation
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Specify input_ids or inputs_embeds.")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("input_ids and inputs_embeds are mutually exclusive.")
        if input_features is not None and inputs_embeds is not None:
            raise ValueError("input_features and inputs_embeds cannot be used together.")
    
        # Building embeddings
        if inputs_embeds is None:
            if input_features is not None:
                if input_features_mask is None:
                    raise ValueError("input_features_mask is required with input_features.")
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
    
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
    
        if use_cache:
            return logits, loss, cache
    
        return logits, loss


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


# ============================================================================
# HuggingFace → FMS Weight Conversion
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
