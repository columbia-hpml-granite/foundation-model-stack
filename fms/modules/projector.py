"""
Speech Projector Module for Granite Speech Model.

This module implements a Q-Former-style projector that bridges the acoustic encoder
(Conformer) and the language decoder. The projector performs:
1. Temporal downsampling to reduce sequence length
2. Cross-modal alignment between acoustic and text representations
3. Dimension projection to match decoder input size

Architecture follows the HuggingFace Granite Speech implementation with window-based
processing for efficiency.

Reference: HuggingFace granite_speech/modeling_granite_speech.py:63-94
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fms.distributed.strategy import DistributedStrategy, NoOpStrategy
from fms.utils.config import ModelConfig
from fms.utils.activation import str_to_activation


logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class SpeechProjectorConfig(ModelConfig):
    """
    Configuration class for Speech Projector (Q-Former style).

    This projector connects the Conformer encoder output to the language decoder,
    performing temporal downsampling and cross-modal alignment using window-based
    processing.

    Architecture (HF-aligned):
        Conformer output (batch, audio_seq_len, encoder_dim)
        → Window-based chunking (nblocks windows of size window_size)
        → Q-Former with learnable queries (cross-attention)
        → Projected output (batch, nblocks * num_queries, decoder_dim)

    Args:
        encoder_dim: Input dimension from Conformer encoder (default: 1024)
        decoder_dim: Output dimension for language decoder (default: 4096)
        num_queries: Number of learnable queries per window (default: 3)
                    This equals window_size // downsample_rate (15 // 5 = 3)

        # Q-Former architecture parameters
        num_hidden_layers: Number of transformer layers in Q-Former (default: 2)
        num_attention_heads: Number of attention heads (default: 16)
        intermediate_size: Hidden size in feed-forward network (default: 4096)

        # Window-based processing parameters (HF-aligned)
        window_size: Window size for chunking (default: 15, from HF granite_speech)
                    Each window of 15 frames → 3 queries

        # Regularization
        hidden_dropout_prob: Dropout probability for hidden layers (default: 0.1)
        attention_dropout_prob: Dropout probability for attention (default: 0.1)

        # Activation
        hidden_act: Activation function (default: "gelu")

        # Layer normalization
        layer_norm_eps: Epsilon for layer normalization (default: 1e-12)

        # Initialization
        initializer_range: Standard deviation for weight initialization (default: 0.02)
    """

    # Input/Output dimensions
    encoder_dim: int = 1024  # Conformer output dimension
    decoder_dim: int = 4096  # Language decoder input dimension (Granite 8B)
    num_queries: int = 3     # Queries per window = window_size // downsample_rate

    # Q-Former architecture
    num_hidden_layers: int = 2   # HF Granite Speech uses 2 layers
    num_attention_heads: int = 16  # HF Blip2QFormer default
    intermediate_size: int = 4096  # FFN hidden size

    # Window-based processing (HF-aligned)
    window_size: int = 15  # Window size from HF granite_speech config

    # Regularization
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1

    # Activation
    hidden_act: str = "gelu"

    # Layer normalization
    layer_norm_eps: float = 1e-12

    # Initialization
    initializer_range: float = 0.02


# ============================================================================
# Q-Former Components
# ============================================================================


class QFormerSelfAttention(nn.Module):
    """
    Self-attention module for Q-Former.

    Implements multi-head self-attention with optional causal masking.
    Used in the self-attention sublayer of Q-Former transformer blocks.

    Args:
        config: SpeechProjectorConfig with attention parameters
    """

    def __init__(self, config: SpeechProjectorConfig):
        super().__init__()
        self.config = config

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.encoder_dim // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Q, K, V projection layers
        self.query = nn.Linear(config.encoder_dim, self.all_head_size)
        self.key = nn.Linear(config.encoder_dim, self.all_head_size)
        self.value = nn.Linear(config.encoder_dim, self.all_head_size)

        # Attention dropout
        self.dropout = nn.Dropout(config.attention_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transpose and reshape tensor for multi-head attention computation.

        Args:
            x: Input tensor of shape (batch, seq_len, all_head_size)

        Returns:
            Reshaped tensor of shape (batch, num_heads, seq_len, head_size)
        """
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for self-attention.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, encoder_dim)
            attention_mask: Optional mask of shape (batch, 1, 1, seq_len)
                          with 0 for valid positions and -inf for masked positions

        Returns:
            Attention output of shape (batch, seq_len, encoder_dim)
        """
        # Project to Q, K, V
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Apply softmax and dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)

        # Reshape back
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)

        return context_layer


class QFormerCrossAttention(nn.Module):
    """
    Cross-attention module for Q-Former.

    Implements multi-head cross-attention between learnable queries and encoder outputs.
    This is the key mechanism for temporal downsampling and cross-modal alignment.

    Queries attend to encoder outputs (keys/values from Conformer).

    Args:
        config: SpeechProjectorConfig with attention parameters
    """

    def __init__(self, config: SpeechProjectorConfig):
        super().__init__()
        self.config = config

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.encoder_dim // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Q projection (from queries)
        self.query = nn.Linear(config.encoder_dim, self.all_head_size)

        # K, V projections (from encoder outputs)
        self.key = nn.Linear(config.encoder_dim, self.all_head_size)
        self.value = nn.Linear(config.encoder_dim, self.all_head_size)

        # Attention dropout
        self.dropout = nn.Dropout(config.attention_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose and reshape tensor for multi-head attention computation."""
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        query_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for cross-attention.

        Args:
            query_states: Learnable queries of shape (batch, num_queries, encoder_dim)
            encoder_hidden_states: Encoder outputs of shape (batch, window_size, encoder_dim)
            encoder_attention_mask: Optional mask for encoder outputs

        Returns:
            Cross-attention output of shape (batch, num_queries, encoder_dim)
        """
        # Project queries (Q from query_states)
        query_layer = self.transpose_for_scores(self.query(query_states))

        # Project encoder outputs (K, V from encoder_hidden_states)
        key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
        value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))

        # Compute attention scores: (batch, heads, num_queries, window_size)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply encoder attention mask if provided
        if encoder_attention_mask is not None:
            attention_scores = attention_scores + encoder_attention_mask

        # Apply softmax and dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)

        # Reshape back
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)

        return context_layer


class QFormerAttentionOutput(nn.Module):
    """
    Output projection and residual connection for attention layers.

    Applies output projection, dropout, and residual connection with layer normalization.

    Args:
        config: SpeechProjectorConfig with projection parameters
    """

    def __init__(self, config: SpeechProjectorConfig):
        super().__init__()
        # Output projection
        self.dense = nn.Linear(config.encoder_dim, config.encoder_dim)

        # Layer normalization
        self.LayerNorm = nn.LayerNorm(config.encoder_dim, eps=config.layer_norm_eps)

        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply output projection with residual connection.

        Args:
            hidden_states: Attention output
            input_tensor: Input to attention layer (for residual connection)

        Returns:
            Output with residual connection and layer norm
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class QFormerFeedForward(nn.Module):
    """
    Feed-forward network for Q-Former transformer block.

    Architecture:
        hidden_states → Linear (encoder_dim → intermediate_size) →
        Activation (GELU) → Linear (intermediate_size → encoder_dim) →
        Dropout → LayerNorm (with residual)

    Args:
        config: SpeechProjectorConfig with FFN parameters
    """

    def __init__(self, config: SpeechProjectorConfig):
        super().__init__()
        # Feed-forward layers
        self.dense_in = nn.Linear(config.encoder_dim, config.intermediate_size)
        self.activation = str_to_activation(config.hidden_act)
        self.dense_out = nn.Linear(config.intermediate_size, config.encoder_dim)
        self.LayerNorm = nn.LayerNorm(config.encoder_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feed-forward network with residual connection.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, encoder_dim)

        Returns:
            Output tensor of shape (batch, seq_len, encoder_dim)
        """
        residual = hidden_states
        hidden_states = self.dense_in(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dense_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + residual)
        return hidden_states


class QFormerLayer(nn.Module):
    """
    Single transformer layer in Q-Former.

    Architecture:
        1. Self-Attention (queries attend to themselves)
        2. Cross-Attention (queries attend to encoder outputs)
        3. Feed-Forward Network

    Each sublayer has residual connections and layer normalization.

    Args:
        config: SpeechProjectorConfig with layer parameters
    """

    def __init__(self, config: SpeechProjectorConfig):
        super().__init__()
        # Q-Former layer components
        self.self_attention = QFormerSelfAttention(config)
        self.self_attention_output = QFormerAttentionOutput(config)
        self.cross_attention = QFormerCrossAttention(config)
        self.cross_attention_output = QFormerAttentionOutput(config)
        self.feed_forward = QFormerFeedForward(config)

    def forward(
        self,
        query_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        query_attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through Q-Former layer.

        Args:
            query_states: Learnable queries of shape (batch, num_queries, encoder_dim)
            encoder_hidden_states: Encoder outputs of shape (batch, window_size, encoder_dim)
            query_attention_mask: Optional mask for query self-attention
            encoder_attention_mask: Optional mask for encoder outputs

        Returns:
            Updated query states of shape (batch, num_queries, encoder_dim)
        """
        # 1. Self-attention on queries
        self_attn_output = self.self_attention(query_states, query_attention_mask)
        query_states = self.self_attention_output(self_attn_output, query_states)

        # 2. Cross-attention between queries and encoder outputs
        cross_attn_output = self.cross_attention(
            query_states, encoder_hidden_states, encoder_attention_mask
        )
        query_states = self.cross_attention_output(cross_attn_output, query_states)

        # 3. Feed-forward network
        query_states = self.feed_forward(query_states)

        return query_states


# ============================================================================
# Speech Projector (Q-Former with Window Processing)
# ============================================================================


class SpeechProjector(nn.Module):
    """
    Speech Projector module for Granite Speech.

    This module bridges the Conformer encoder and language decoder using a Q-Former
    architecture with learnable queries and window-based processing.

    **HF-Aligned Window-Based Processing:**
    Following HuggingFace granite_speech implementation:
    1. Pad input sequence to multiple of window_size
    2. Reshape to (batch * nblocks, window_size, dim)
    3. Apply Q-Former to each window
    4. Reshape output to (batch, nblocks * num_queries, decoder_dim)

    Key Features:
    1. **Temporal Downsampling**: Reduces audio sequence length
       - Input: (batch, audio_seq_len, encoder_dim)
       - Output: (batch, nblocks * num_queries, decoder_dim)
       - Each window of 15 frames → 3 queries (5x downsampling)

    2. **Cross-Modal Alignment**: Learnable queries attend to encoder outputs
       - Queries learn to extract relevant acoustic information
       - Cross-attention mechanism enables flexible information aggregation

    3. **Dimension Projection**: Matches decoder input dimension
       - Projects from encoder_dim (1024) to decoder_dim (4096)

    Architecture:
        Conformer output (batch, audio_seq_len, encoder_dim)
        → Window padding and chunking
        → Learnable Queries (1, num_queries, encoder_dim)
        → Q-Former Layers (self-attention + cross-attention + FFN) × num_layers
        → LayerNorm
        → Linear Projection (encoder_dim → decoder_dim)
        → Reshape to (batch, nblocks * num_queries, decoder_dim)

    Args:
        config: SpeechProjectorConfig with all hyperparameters
        distributed_strategy: Strategy for distributed training (default: NoOpStrategy)

    Reference: HuggingFace granite_speech/modeling_granite_speech.py:63-94
    """

    def __init__(
        self,
        config: SpeechProjectorConfig,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
    ):
        super().__init__()
        self.config = config
        self.distributed_strategy = distributed_strategy

        # Store config values
        self.window_size = config.window_size
        self.num_queries = config.num_queries

        # Learnable queries (HF: initialized with N(0,1))
        # Shape: (1, num_queries, encoder_dim) - will be broadcast to batch size
        self.query_embeds = nn.Parameter(
            torch.zeros(1, config.num_queries, config.encoder_dim)
        )
        # Initialize with normal distribution (matching HF)
        self.query_embeds.data.normal_(mean=0.0, std=1.0)

        # Stack Q-Former layers
        self.layers = nn.ModuleList([
            QFormerLayer(config) for _ in range(config.num_hidden_layers)
        ])

        # Layer normalization before output projection
        self.layer_norm = nn.LayerNorm(config.encoder_dim, eps=config.layer_norm_eps)

        # Output projection layer (encoder_dim → decoder_dim)
        self.output_proj = nn.Linear(config.encoder_dim, config.decoder_dim)

    def reset_parameters(self):
        """Initialize all trainable parameters."""
        # Re-initialize query embeddings with N(0,1)
        self.query_embeds.data.normal_(mean=0.0, std=1.0)

        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=self.config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through Speech Projector with window-based processing.

        Args:
            encoder_hidden_states: Conformer output of shape (batch, audio_seq_len, encoder_dim)
            attention_mask: Optional mask for encoder outputs (not used in HF implementation)

        Returns:
            projected_states: Embeddings for decoder of shape
                            (batch, nblocks * num_queries, decoder_dim)

        Example:
            >>> config = SpeechProjectorConfig(
            ...     encoder_dim=1024,
            ...     decoder_dim=4096,
            ...     num_queries=3,
            ...     window_size=15,
            ...     num_hidden_layers=2
            ... )
            >>> projector = SpeechProjector(config)
            >>> encoder_output = torch.randn(2, 30, 1024)  # batch=2, seq_len=30
            >>> decoder_input = projector(encoder_output)
            >>> decoder_input.shape  # 30/15 = 2 windows, 2*3 = 6 queries
            torch.Size([2, 6, 4096])
        """
        batch_size, seq_len, dim = encoder_hidden_states.size()

        # === Window-based processing (HF-aligned) ===

        # 1. Calculate number of windows and padding
        nblocks = math.ceil(seq_len / self.window_size)
        pad = nblocks * self.window_size - seq_len

        # 2. Pad input to multiple of window_size
        if pad > 0:
            encoder_hidden_states = F.pad(
                encoder_hidden_states, (0, 0, 0, pad), mode="constant", value=0
            )

        # 3. Reshape to (batch * nblocks, window_size, dim)
        encoder_hidden_states = encoder_hidden_states.view(
            batch_size * nblocks, self.window_size, dim
        )

        # 4. Expand learnable queries to match batch * nblocks
        # query_embeds: (1, num_queries, encoder_dim) → (batch * nblocks, num_queries, encoder_dim)
        query_states = self.query_embeds.expand(batch_size * nblocks, -1, -1)

        # 5. Pass through Q-Former layers
        for layer in self.layers:
            query_states = layer(
                query_states=query_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=None,  # HF doesn't use attention mask
            )

        # 6. Apply layer normalization
        query_states = self.layer_norm(query_states)

        # 7. Reshape output to (batch, nblocks * num_queries, encoder_dim)
        query_states = query_states.view(
            batch_size, nblocks * self.num_queries, -1
        )

        # 8. Project to decoder dimension
        projected_states = self.output_proj(query_states)

        return projected_states

    def _prepare_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Convert attention mask to format expected by attention layers.

        Args:
            attention_mask: Binary mask of shape (batch, seq_len)
                          with 1 for valid positions, 0 for padding

        Returns:
            Prepared mask of shape (batch, 1, 1, seq_len)
            with 0 for valid positions, -inf for masked positions
        """
        extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_mask = extended_mask.to(dtype=torch.float32)
        extended_mask = (1.0 - extended_mask) * torch.finfo(torch.float32).min
        return extended_mask


# ============================================================================
# Factory Functions and Model Registration
# ============================================================================


def _create_speech_projector_config(
    encoder_dim: int = 1024,
    decoder_dim: int = 4096,
    num_queries: int = 3,
    window_size: int = 15,
    num_hidden_layers: int = 2,
    **kwargs,
) -> SpeechProjectorConfig:
    """
    Factory function for creating Speech Projector configurations.

    Args:
        encoder_dim: Input dimension from encoder
        decoder_dim: Output dimension for decoder
        num_queries: Number of learnable queries per window
        window_size: Window size for chunking
        num_hidden_layers: Number of Q-Former layers
        **kwargs: Additional config parameters

    Returns:
        SpeechProjectorConfig instance
    """
    return SpeechProjectorConfig(
        encoder_dim=encoder_dim,
        decoder_dim=decoder_dim,
        num_queries=num_queries,
        window_size=window_size,
        num_hidden_layers=num_hidden_layers,
        **kwargs,
    )


# Predefined configurations for common use cases
SPEECH_PROJECTOR_CONFIGS = {
    "granite_speech_default": _create_speech_projector_config(
        encoder_dim=1024,
        decoder_dim=4096,
        num_queries=3,      # window_size // downsample_rate = 15 // 5
        window_size=15,     # HF granite_speech default
        num_hidden_layers=2,
        num_attention_heads=16,
        intermediate_size=4096,
    ),
    "projector_small": _create_speech_projector_config(
        encoder_dim=512,
        decoder_dim=1024,
        num_queries=3,
        window_size=15,
        num_hidden_layers=2,
        num_attention_heads=8,
        intermediate_size=2048,
    ),
}
