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
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fms.distributed.strategy import DistributedStrategy, NoOpStrategy
from fms.utils.activation import str_to_activation
from fms.utils.config import ModelConfig


logger = logging.getLogger(__name__)


@dataclass
class SpeechProjectorConfig(ModelConfig):
    """
    Configuration class for Speech Projector (Q-Former style).

    This projector connects the Conformer encoder output to the language decoder,
    performing temporal downsampling and cross-modal alignment.

    Architecture:
        Conformer output (batch, audio_seq_len, encoder_dim) ->
        Q-Former with learnable queries ->
        Projected output (batch, num_queries, decoder_dim)

    Args:
        encoder_dim: Input dimension from Conformer encoder (default: 1024)
        decoder_dim: Output dimension for language decoder (default: 2048)
        window_size: Window size for temporal downsampling (default: 15)
        downsample_rate: Downsampling rate (default: 5)
        num_queries: Number of learnable queries per window (default: 3)
                    Derived as window_size // downsample_rate

        # Q-Former architecture parameters
        num_hidden_layers: Number of transformer layers in Q-Former (default: 2)
        num_attention_heads: Number of attention heads (default: 16)
        intermediate_size: Hidden size in feed-forward network (default: 4096)

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
    encoder_dim: int = 1024 # Conformer output dimension
    decoder_dim: int = 2048 # Language decoder input dimension

    # Window/downsampling (HF Granite Speech)
    window_size: int = 15
    downsample_rate: int = 5
    num_queries: int = 3 # Derived as window_size // downsample_rate in HF

    # Q-Former architecture
    num_hidden_layers: int = 2 # Number of transformer layers (HF default)
    num_attention_heads: int = 16 # Number of attention heads (HF Blip2QFormer)
    intermediate_size: int = 4096 # FFN hidden size

    # Regularization
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1

    # Activation
    hidden_act: str = "gelu"

    # Layer normalization
    layer_norm_eps: float = 1e-12

    # Initialization
    initializer_range: float = 0.02


class QFormerSelfAttention(nn.Module):
    """
    Self-attention module for Q-Former.

    Implements multi-head self-attention with optional masking.
    Used in the self-attention sublayer of Q-Former transformer blocks.

    Args:
        config: SpeechProjectorConfig with attention parameters
    """

    def __init__(self, config: SpeechProjectorConfig):
        super().__init__()
        self.config = config

        assert (
            config.encoder_dim % config.num_attention_heads == 0
        ), "encoder_dim must be divisible by num_attention_heads"

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.encoder_dim // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.encoder_dim, self.all_head_size)
        self.key = nn.Linear(config.encoder_dim, self.all_head_size)
        self.value = nn.Linear(config.encoder_dim, self.all_head_size)

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
        x = x.view(*new_shape)  # (B, S, H, Dh)
        return x.permute(0, 2, 1, 3)  # (B, H, S, Dh)

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
        # Projections
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Scaled dot-product attention
        attn_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attn_scores = attn_scores / (self.attention_head_size ** 0.5)

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask  # broadcast

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Applies attention to values
        context_layer = torch.matmul(attn_probs, value_layer)  # (B,H,S,Dh)

        # Reshape back to (B, S, D)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_shape)
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

        assert (
            config.encoder_dim % config.num_attention_heads == 0
        ), "encoder_dim must be divisible by num_attention_heads"

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.encoder_dim // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Note: queries from queries; keys/values from encoder
        self.query = nn.Linear(config.encoder_dim, self.all_head_size)
        self.key = nn.Linear(config.encoder_dim, self.all_head_size)
        self.value = nn.Linear(config.encoder_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_dropout_prob)

    def _transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
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
            encoder_hidden_states: Encoder outputs of shape (batch, audio_seq_len, encoder_dim)
            encoder_attention_mask: Optional mask of shape (batch, 1, 1, audio_seq_len)

        Returns:
            Cross-attention output of shape (batch, num_queries, encoder_dim)
        """
        # Project queries, keys, values
        query_layer = self._transpose_for_scores(self.query(query_states))
        key_layer = self._transpose_for_scores(self.key(encoder_hidden_states))
        value_layer = self._transpose_for_scores(self.value(encoder_hidden_states))

        # Attention scores: (B, H, Q, S_enc)
        attn_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attn_scores = attn_scores / (self.attention_head_size ** 0.5)

        if encoder_attention_mask is not None:
            attn_scores = attn_scores + encoder_attention_mask

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Context: (B, H, Q, Dh)
        context_layer = torch.matmul(attn_probs, value_layer)

        # Reshape back to (B, Q, D)
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
        self.dense = nn.Linear(config.encoder_dim, config.encoder_dim)
        self.LayerNorm = nn.LayerNorm(config.encoder_dim, eps=config.layer_norm_eps)
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
        hidden_states -> Linear (encoder_dim -> intermediate_size) ->
        Activation (GELU) -> Linear (intermediate_size -> encoder_dim) ->
        Dropout -> LayerNorm (with residual)

    Args:
        config: SpeechProjectorConfig with FFN parameters
    """

    def __init__(self, config: SpeechProjectorConfig):
        super().__init__()
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
            encoder_hidden_states: Encoder outputs of shape (batch, audio_seq_len, encoder_dim)
            query_attention_mask: Optional mask for query self-attention
                                  Shape: (batch, 1, 1, num_queries)
            encoder_attention_mask: Optional mask for encoder outputs
                                    Shape: (batch, 1, 1, audio_seq_len)

        Returns:
            Updated query states of shape (batch, num_queries, encoder_dim)
        """
        # 1. Self-attention on queries
        sa_output = self.self_attention(query_states, attention_mask=query_attention_mask)
        query_states = self.self_attention_output(sa_output, query_states)

        # 2. Cross-attention: queries attend over encoder outputs
        ca_output = self.cross_attention(
            query_states=query_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        query_states = self.cross_attention_output(ca_output, query_states)

        # 3. Feed-forward network
        query_states = self.feed_forward(query_states)
        return query_states


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

    Architecture (matching HuggingFace BLIP-2 Q-Former):
        Conformer output (batch, audio_seq_len, encoder_dim) ->
        Learnable Queries (batch, num_queries, encoder_dim) ->
        Input LayerNorm + Dropout ->
        Q-Former Layers (self-attention + cross-attention + FFN) × num_layers ->
        Output Projection (encoder_dim -> decoder_dim) ->
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
        window_size: Optional[int] = None,
        downsample_rate: Optional[int] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
    ):
        super().__init__()
        self.config = config
        self.distributed_strategy = distributed_strategy

        # Window-based processing parameters (prefer explicit args, fallback to config)
        self.window_size = (
            window_size
            if window_size is not None
            else getattr(config, "window_size", None)
        )
        self.downsample_rate = (
            downsample_rate
            if downsample_rate is not None
            else getattr(config, "downsample_rate", None)
        )
        if self.window_size is None or self.downsample_rate is None:
            raise ValueError("window_size and downsample_rate must be provided via args or config.")
        # num_queries can be set in config; otherwise derive from window/downsample
        self.num_queries = getattr(config, "num_queries", None)
        if self.num_queries is None:
            self.num_queries = self.window_size // self.downsample_rate
        else:
            expected = self.window_size // self.downsample_rate
            if self.num_queries != expected:
                logger.warning(
                    "num_queries (%s) does not match window_size // downsample_rate (%s); using num_queries",
                    self.num_queries,
                    expected,
                )

        # Learnable queries: (1, num_queries, encoder_dim) - matches HF shape
        self.query_embeds = nn.Parameter(
            torch.zeros(1, self.num_queries, config.encoder_dim)
        )
        # Random initialization matching HuggingFace: N(0, 1)
        nn.init.normal_(self.query_embeds, mean=0.0, std=1.0)

        # Input layer norm and dropout (applied to query embeddings before Q-Former)
        # This matches HuggingFace BLIP-2 Q-Former architecture
        self.input_layernorm = nn.LayerNorm(config.encoder_dim, eps=config.layer_norm_eps)
        self.input_dropout = nn.Dropout(config.hidden_dropout_prob)

        # Stack of Q-Former layers
        self.layers = nn.ModuleList(
            [QFormerLayer(config) for _ in range(config.num_hidden_layers)]
        )

        # Output projection to decoder dimension
        self.output_proj = nn.Linear(config.encoder_dim, config.decoder_dim)

        # Initialize parameters
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0, std=self.config.initializer_range)
            if getattr(module, "bias", None) is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1)

    def _expand_query_embeds(self, num_windows: int, device: torch.device) -> torch.Tensor:
        """
        Expand learnable queries to number of windows.

        Args:
            num_windows: batch_size * nblocks (total number of windows)
            device: device for the expanded tensor

        Returns:
            Expanded queries of shape (num_windows, num_queries, encoder_dim)
        """
        return self.query_embeds.expand(num_windows, -1, -1).to(device)

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
        # (B, S) -> (B, 1, 1, S)
        extended_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype=torch.float32)
        extended_mask = (1 - extended_mask) * torch.finfo(torch.float32).min
        return extended_mask

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Window-based forward pass through Speech Projector (matching HuggingFace).

        Args:
            encoder_hidden_states: Conformer output of shape (batch, audio_seq_len, encoder_dim)
            attention_mask: Optional mask for encoder outputs
                          Shape: (batch, audio_seq_len) with 1 for valid, 0 for padding

        Returns:
            projected_states: Embeddings for decoder of shape (batch, nblocks * num_queries, decoder_dim)
                            where nblocks = ceil(audio_seq_len / window_size)
        """
        batch_size, seq_len, dim = encoder_hidden_states.shape
        device = encoder_hidden_states.device

        # 1. Calculate number of windows and pad to multiple of window_size
        nblocks = math.ceil(seq_len / self.window_size)
        pad = nblocks * self.window_size - seq_len
        if pad > 0:
            encoder_hidden_states = F.pad(encoder_hidden_states, (0, 0, 0, pad), "constant", 0)

        # 2. Reshape to (batch * nblocks, window_size, dim)
        encoder_hidden_states = encoder_hidden_states.view(
            batch_size * nblocks, self.window_size, dim
        )

        # 3. Expand queries for all windows: (1, Q, D) -> (batch * nblocks, Q, D)
        query_states = self._expand_query_embeds(batch_size * nblocks, device=device)

        # 4. Apply input LayerNorm and dropout (matching HuggingFace BLIP-2 Q-Former)
        query_states = self.input_layernorm(query_states)
        query_states = self.input_dropout(query_states)

        # 5. Pass through all Q-Former layers
        for layer in self.layers:
            query_states = layer(
                query_states=query_states,
                encoder_hidden_states=encoder_hidden_states,
                query_attention_mask=None,
                encoder_attention_mask=None,
            )

        # 6. Reshape back to (batch, nblocks * num_queries, dim)
        query_states = query_states.view(batch_size, nblocks * self.num_queries, -1)

        # 7. Project to decoder_dim
        projected_states = self.output_proj(query_states)
        return projected_states
