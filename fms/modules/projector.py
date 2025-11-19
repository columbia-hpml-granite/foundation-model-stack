"""
Speech Projector Module for Granite Speech Model.

This module implements a Q-Former-style projector that bridges the acoustic encoder
(Conformer) and the language decoder. The projector performs:
1. Temporal downsampling to reduce sequence length
2. Cross-modal alignment between acoustic and text representations
3. Dimension projection to match decoder input size

Architecture follows the Granite Speech paper's Q-Former design with learnable queries
and cross-attention mechanisms.
"""

import logging
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
    performing temporal downsampling and cross-modal alignment.

    Architecture:
        Conformer output (batch, audio_seq_len, encoder_dim) �
        Q-Former with learnable queries �
        Projected output (batch, num_queries, decoder_dim)

    Args:
        encoder_dim: Input dimension from Conformer encoder (default: 1024)
        decoder_dim: Output dimension for language decoder (default: 2048)
        num_queries: Number of learnable queries for downsampling (default: 32)
                    This controls the temporal downsampling ratio.
                    Example: 500 audio frames � 32 queries (15.6x downsampling)

        # Q-Former architecture parameters
        num_hidden_layers: Number of transformer layers in Q-Former (default: 2)
        num_attention_heads: Number of attention heads (default: 8)
        intermediate_size: Hidden size in feed-forward network (default: 4096)

        # Window attention parameters
        window_size: Optional local attention window size (default: None for full attention)
                    If set, enables windowed cross-attention for efficiency

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
    decoder_dim: int = 2048  # Language decoder input dimension
    num_queries: int = 32  # Number of learnable queries (controls downsampling)

    # Q-Former architecture
    num_hidden_layers: int = 2  # Number of transformer layers (Granite Speech uses 2-layer window query transformer)
    num_attention_heads: int = 8  # Number of attention heads
    intermediate_size: int = 4096  # FFN hidden size

    # Window attention parameters
    window_size: Optional[int] = None  # Local attention window size for cross-attention
                                       # If None, use full attention (all queries attend to all encoder outputs)
                                       # If set (e.g., 128), each query attends to a local window of encoder outputs
                                       # Reduces computation from O(num_queries * audio_seq_len) to O(num_queries * window_size)

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

        # TODO: Validate that hidden size is divisible by num_attention_heads
        # assert config.encoder_dim % config.num_attention_heads == 0

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.encoder_dim // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # TODO: Implement Q, K, V projection layers
        # self.query = nn.Linear(config.encoder_dim, self.all_head_size)
        # self.key = nn.Linear(config.encoder_dim, self.all_head_size)
        # self.value = nn.Linear(config.encoder_dim, self.all_head_size)

        # TODO: Implement attention dropout
        # self.dropout = nn.Dropout(config.attention_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transpose and reshape tensor for multi-head attention computation.

        Args:
            x: Input tensor of shape (batch, seq_len, all_head_size)

        Returns:
            Reshaped tensor of shape (batch, num_heads, seq_len, head_size)
        """
        # TODO: Implement reshaping for multi-head attention
        # new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # x = x.view(*new_shape)
        # return x.permute(0, 2, 1, 3)
        raise NotImplementedError

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
        # TODO: Implement self-attention forward pass
        # 1. Project to Q, K, V
        # 2. Reshape for multi-head attention
        # 3. Compute attention scores (Q @ K^T / sqrt(d_k))
        # 4. Apply attention mask if provided
        # 5. Apply softmax and dropout
        # 6. Apply attention to values (attn_weights @ V)
        # 7. Reshape and return
        raise NotImplementedError


class QFormerCrossAttention(nn.Module):
    """
    Cross-attention module for Q-Former.

    Implements multi-head cross-attention between learnable queries and encoder outputs.
    This is the key mechanism for temporal downsampling and cross-modal alignment.

    Queries attend to encoder outputs (keys/values from Conformer).

    Window Attention (Optional):
        If config.window_size is set, implements local windowed attention where each query
        attends to a local window of encoder outputs, reducing computation from
        O(num_queries * audio_seq_len) to O(num_queries * window_size).

    Args:
        config: SpeechProjectorConfig with attention parameters
    """

    def __init__(self, config: SpeechProjectorConfig):
        super().__init__()
        self.config = config
        self.window_size = config.window_size

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.encoder_dim // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # TODO: Implement Q projection (from queries)
        # self.query = nn.Linear(config.encoder_dim, self.all_head_size)

        # TODO: Implement K, V projections (from encoder outputs)
        # self.key = nn.Linear(config.encoder_dim, self.all_head_size)
        # self.value = nn.Linear(config.encoder_dim, self.all_head_size)

        # TODO: Implement attention dropout
        # self.dropout = nn.Dropout(config.attention_dropout_prob)

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
            encoder_hidden_states: Encoder outputs of shape (batch, audio_seq_len, encoder_dim)
            encoder_attention_mask: Optional mask for encoder outputs

        Returns:
            Cross-attention output of shape (batch, num_queries, encoder_dim)
        """
        # TODO: Implement cross-attention forward pass
        # 1. Project queries (Q from query_states)
        # 2. Project encoder outputs (K, V from encoder_hidden_states)
        # 3. Reshape for multi-head attention
        # 4. Compute attention scores (Q @ K^T / sqrt(d_k))
        # 5. If self.window_size is set:
        #    - Apply windowed attention: each query attends to a local window
        #    - Window center for query i: (i / num_queries) * audio_seq_len
        #    - Window range: [center - window_size//2, center + window_size//2]
        # 6. Apply encoder attention mask if provided
        # 7. Apply softmax and dropout
        # 8. Apply attention to values (attn_weights @ V)
        # 9. Reshape and return
        raise NotImplementedError


class QFormerAttentionOutput(nn.Module):
    """
    Output projection and residual connection for attention layers.

    Applies output projection, dropout, and residual connection with layer normalization.

    Args:
        config: SpeechProjectorConfig with projection parameters
    """

    def __init__(self, config: SpeechProjectorConfig):
        super().__init__()
        # TODO: Implement output projection
        # self.dense = nn.Linear(config.encoder_dim, config.encoder_dim)

        # TODO: Implement layer normalization
        # self.LayerNorm = nn.LayerNorm(config.encoder_dim, eps=config.layer_norm_eps)

        # TODO: Implement dropout
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        pass

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply output projection with residual connection.

        Args:
            hidden_states: Attention output
            input_tensor: Input to attention layer (for residual connection)

        Returns:
            Output with residual connection and layer norm
        """
        # TODO: Implement residual connection with layer norm
        # hidden_states = self.dense(hidden_states)
        # hidden_states = self.dropout(hidden_states)
        # hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # return hidden_states
        raise NotImplementedError


class QFormerFeedForward(nn.Module):
    """
    Feed-forward network for Q-Former transformer block.

    Architecture:
        hidden_states � Linear (encoder_dim � intermediate_size) �
        Activation (GELU) � Linear (intermediate_size � encoder_dim) �
        Dropout � LayerNorm (with residual)

    Args:
        config: SpeechProjectorConfig with FFN parameters
    """

    def __init__(self, config: SpeechProjectorConfig):
        super().__init__()
        # TODO: Implement feed-forward layers
        # self.dense_in = nn.Linear(config.encoder_dim, config.intermediate_size)
        # self.activation = str_to_activation(config.hidden_act)
        # self.dense_out = nn.Linear(config.intermediate_size, config.encoder_dim)
        # self.LayerNorm = nn.LayerNorm(config.encoder_dim, eps=config.layer_norm_eps)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        pass

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feed-forward network with residual connection.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, encoder_dim)

        Returns:
            Output tensor of shape (batch, seq_len, encoder_dim)
        """
        # TODO: Implement FFN forward pass with residual connection
        # residual = hidden_states
        # hidden_states = self.dense_in(hidden_states)
        # hidden_states = self.activation(hidden_states)
        # hidden_states = self.dense_out(hidden_states)
        # hidden_states = self.dropout(hidden_states)
        # hidden_states = self.LayerNorm(hidden_states + residual)
        # return hidden_states
        raise NotImplementedError


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
        # TODO: Implement Q-Former layer components
        # self.self_attention = QFormerSelfAttention(config)
        # self.self_attention_output = QFormerAttentionOutput(config)
        # self.cross_attention = QFormerCrossAttention(config)
        # self.cross_attention_output = QFormerAttentionOutput(config)
        # self.feed_forward = QFormerFeedForward(config)
        pass

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
            encoder_hidden_states: Encoder outputs of shape (batch, audio_seq_len, encoder_dim)
            query_attention_mask: Optional mask for query self-attention
            encoder_attention_mask: Optional mask for encoder outputs

        Returns:
            Updated query states of shape (batch, num_queries, encoder_dim)
        """
        # TODO: Implement Q-Former layer forward pass
        # 1. Self-attention on queries
        # 2. Cross-attention between queries and encoder outputs
        # 3. Feed-forward network
        # Each step includes residual connections and layer normalization
        raise NotImplementedError


# ============================================================================
# Speech Projector (Q-Former)
# ============================================================================


class SpeechProjector(nn.Module):
    """
    Speech Projector module for Granite Speech.

    This module bridges the Conformer encoder and language decoder using a Q-Former
    architecture with learnable queries. Key features:

    1. **Temporal Downsampling**: Reduces audio sequence length
       - Input: (batch, audio_seq_len, encoder_dim) - e.g., 500 frames
       - Output: (batch, num_queries, decoder_dim) - e.g., 32 queries
       - Downsampling ratio: audio_seq_len / num_queries (e.g., 15.6x)

    2. **Cross-Modal Alignment**: Learnable queries attend to encoder outputs
       - Queries learn to extract relevant acoustic information
       - Cross-attention mechanism enables flexible information aggregation

    3. **Dimension Projection**: Matches decoder input dimension
       - Projects from encoder_dim (1024) to decoder_dim (2048)

    Architecture:
        Conformer output (batch, audio_seq_len, encoder_dim) �
        Learnable Queries (batch, num_queries, encoder_dim) �
        Q-Former Layers (self-attention + cross-attention + FFN) � num_layers �
        Output Projection (encoder_dim � decoder_dim) �
        Projected output (batch, num_queries, decoder_dim)

    Args:
        config: SpeechProjectorConfig with all hyperparameters
        distributed_strategy: Strategy for distributed training (default: NoOpStrategy)

    Input:
        encoder_hidden_states: Conformer output of shape (batch, audio_seq_len, encoder_dim)
        attention_mask: Optional mask for encoder outputs

    Output:
        projected_states: Embeddings for decoder of shape (batch, num_queries, decoder_dim)
    """

    def __init__(
        self,
        config: SpeechProjectorConfig,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
    ):
        super().__init__()
        self.config = config
        self.distributed_strategy = distributed_strategy

        # TODO: Initialize learnable queries
        # Shape: (num_queries, encoder_dim)
        # These are learned parameters that attend to encoder outputs
        # self.query_embeds = nn.Parameter(
        #     torch.zeros(config.num_queries, config.encoder_dim)
        # )

        # TODO: Stack Q-Former layers
        # self.layers = nn.ModuleList([
        #     QFormerLayer(config) for _ in range(config.num_hidden_layers)
        # ])

        # TODO: Output projection layer (encoder_dim � decoder_dim)
        # self.output_proj = nn.Linear(config.encoder_dim, config.decoder_dim)

        # TODO: Layer normalization before output projection
        # self.layer_norm = nn.LayerNorm(config.encoder_dim, eps=config.layer_norm_eps)

    def _expand_query_embeds(self, batch_size: int) -> torch.Tensor:
        """
        Expand learnable queries to batch size.

        Args:
            batch_size: Batch size for current forward pass

        Returns:
            Expanded queries of shape (batch, num_queries, encoder_dim)
        """
        # TODO: Expand query embeddings to match batch size
        # return self.query_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        raise NotImplementedError

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through Speech Projector.

        Args:
            encoder_hidden_states: Conformer output of shape (batch, audio_seq_len, encoder_dim)
            attention_mask: Optional mask for encoder outputs
                          Shape: (batch, audio_seq_len) with 1 for valid, 0 for padding

        Returns:
            projected_states: Embeddings for decoder of shape (batch, num_queries, decoder_dim)

        Example:
            >>> config = SpeechProjectorConfig(
            ...     encoder_dim=1024,
            ...     decoder_dim=2048,
            ...     num_queries=32,
            ...     num_hidden_layers=6
            ... )
            >>> projector = SpeechProjector(config)
            >>> encoder_output = torch.randn(2, 500, 1024)  # batch=2, seq_len=500
            >>> decoder_input = projector(encoder_output)
            >>> decoder_input.shape
            torch.Size([2, 32, 2048])  # batch=2, queries=32, dim=2048
        """
        batch_size = encoder_hidden_states.shape[0]

        # TODO: Implement forward pass
        # 1. Expand learnable queries to batch size
        # query_states = self._expand_query_embeds(batch_size)

        # 2. Prepare attention mask if provided
        # if attention_mask is not None:
        #     # Convert (batch, seq_len) to (batch, 1, 1, seq_len) for broadcasting
        #     # Set masked positions to large negative value
        #     encoder_attention_mask = self._prepare_attention_mask(attention_mask)
        # else:
        #     encoder_attention_mask = None

        # 3. Pass through Q-Former layers
        # for layer in self.layers:
        #     query_states = layer(
        #         query_states=query_states,
        #         encoder_hidden_states=encoder_hidden_states,
        #         encoder_attention_mask=encoder_attention_mask,
        #     )

        # 4. Apply layer normalization
        # query_states = self.layer_norm(query_states)

        # 5. Project to decoder dimension
        # projected_states = self.output_proj(query_states)

        # return projected_states
        raise NotImplementedError

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
        # TODO: Implement attention mask preparation
        # extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # extended_mask = (1.0 - extended_mask) * torch.finfo(attention_mask.dtype).min
        # return extended_mask
        raise NotImplementedError


# ============================================================================
# Factory Functions and Model Registration
# ============================================================================


def _create_speech_projector_config(
    encoder_dim: int = 1024,
    decoder_dim: int = 2048,
    num_queries: int = 32,
    num_hidden_layers: int = 6,
    **kwargs,
) -> SpeechProjectorConfig:
    """
    Factory function for creating Speech Projector configurations.

    Args:
        encoder_dim: Input dimension from encoder
        decoder_dim: Output dimension for decoder
        num_queries: Number of learnable queries
        num_hidden_layers: Number of Q-Former layers
        **kwargs: Additional config parameters

    Returns:
        SpeechProjectorConfig instance
    """
    return SpeechProjectorConfig(
        encoder_dim=encoder_dim,
        decoder_dim=decoder_dim,
        num_queries=num_queries,
        num_hidden_layers=num_hidden_layers,
        **kwargs,
    )


# Predefined configurations for common use cases
SPEECH_PROJECTOR_CONFIGS = {
    "granite_speech_default": _create_speech_projector_config(
        encoder_dim=1024,
        decoder_dim=2048,
        num_queries=32,
        num_hidden_layers=2,  # Granite Speech uses 2-layer window query transformer
        num_attention_heads=8,
        intermediate_size=4096,
    ),
    "projector_small": _create_speech_projector_config(
        encoder_dim=512,
        decoder_dim=1024,
        num_queries=16,
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=2048,
    ),
}


# TODO: Register with FMS model registry if needed
# from fms import models
# _module_name = "speech_projector"
# for variant_name, config in SPEECH_PROJECTOR_CONFIGS.items():
#     models.register_model(
#         _module_name,
#         variant_name,
#         lambda c=config: SpeechProjector(c)
#     )
