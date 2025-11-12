"""
Conformer: Convolution-augmented Transformer for Speech Recognition

This module implements the Conformer architecture for acoustic modeling.
Based on: "Conformer: Convolution-augmented Transformer for Speech Recognition"
          (Gulati et al., 2020) https://arxiv.org/abs/2005.08100

Architecture Overview:
    Audio Features (80 log-mel) → Input Projection →
    Conformer Block 1 → ... → Conformer Block N →
    Acoustic Embeddings (hidden_dim)

Conformer Block Structure:
    x → LayerNorm → FeedForward1 (0.5x) →
    LayerNorm → Attention →
    LayerNorm → Convolution →
    LayerNorm → FeedForward2 (0.5x) →
    Post LayerNorm → output

Reference Implementation:
    - HuggingFace transformers: granite_speech/modeling_granite_speech.py
    - Uses Shaw's relative positional embeddings for attention
    - Depthwise separable convolution with GLU activation
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
class ConformerConfig(ModelConfig):
    """
    Configuration class for Conformer encoder.

    Args:
        num_features: Number of input audio features (default: 80 log-mel channels)
        hidden_dim: Hidden dimension for encoder layers
        num_layers: Number of Conformer blocks (Granite-speech uses 16)
        num_heads: Number of attention heads in multi-head attention
        dim_head: Dimension per attention head

        conv_kernel_size: Kernel size for depthwise convolution (default: 31)
        conv_expansion_factor: Expansion factor for convolution module (default: 2)

        feedforward_mult: Expansion multiplier for feed-forward networks (default: 4)

        dropout: Dropout probability applied throughout the model

        max_pos_emb: Maximum positional embedding distance for relative attention
        context_size: Local attention window size (sequence positions are clamped to +/- context_size)

        activation: Activation function name (default: "silu" for SiLU/Swish)
    """

    num_features: int = 80  # Input: 80 log-mel filterbank features
    hidden_dim: int = 1024  # Encoder hidden dimension
    num_layers: int = 16  # Number of conformer blocks (Granite-speech default)

    # Multi-head attention parameters
    num_heads: int = 8
    dim_head: int = 64  # Per-head dimension (num_heads * dim_head = inner_dim)

    # Convolution module parameters
    conv_kernel_size: int = 31
    conv_expansion_factor: int = 2

    # Feed-forward parameters
    feedforward_mult: int = 4  # FFN expansion: hidden_dim * feedforward_mult

    # Regularization
    dropout: float = 0.1

    # Positional encoding parameters
    max_pos_emb: int = 1000  # Maximum relative position distance
    context_size: int = 100  # Local attention window (clamping range)

    # Activation function
    activation: str = "silu"  # SiLU (Swish) activation


# ============================================================================
# Component Modules (Abstract Interfaces)
# ============================================================================


class ConformerFeedForward(nn.Module):
    """
    Feed-forward module for Conformer block.

    Architecture:
        x → LayerNorm → Linear (dim → dim * mult) → Activation → Dropout →
        Linear (dim * mult → dim) → Dropout → output

    Args:
        dim: Input and output dimension
        mult: Expansion multiplier (default: 4)
        dropout: Dropout probability
        activation: Activation function name
    """

    def __init__(
        self,
        dim: int,
        mult: int = 4,
        dropout: float = 0.1,
        activation: str = "silu",
    ):
        super().__init__()
        self.dim = dim
        self.mult = mult
        self.dropout = dropout

        # TODO: Implement feed-forward layers
        # - Layer normalization
        # - Linear projection: dim → dim * mult
        # - Activation function (SiLU)
        # - Dropout
        # - Linear projection: dim * mult → dim
        # - Dropout
        raise NotImplementedError("ConformerFeedForward not yet implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feed-forward network.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)

        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        # TODO: Implement forward pass
        raise NotImplementedError("ConformerFeedForward.forward() not yet implemented")


class ConformerAttention(nn.Module):
    """
    Multi-head self-attention with Shaw's relative positional embeddings.

    Uses relative positional encodings instead of absolute positions:
        - Computes attention scores with relative position bias
        - Enables better generalization to variable-length sequences

    Reference: "Self-Attention with Relative Position Representations" (Shaw et al., 2018)

    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        dim_head: Dimension per head
        max_pos_emb: Maximum relative position distance
        dropout: Dropout probability
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dim_head: int = 64,
        max_pos_emb: int = 1000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.inner_dim = num_heads * dim_head
        self.max_pos_emb = max_pos_emb
        self.dropout = dropout

        # TODO: Implement attention components
        # - Query, Key, Value projections
        # - Relative positional embeddings (learnable)
        # - Output projection
        # - Dropout layers
        raise NotImplementedError("ConformerAttention not yet implemented")

    def forward(
        self,
        x: torch.Tensor,
        attention_dists: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with relative positional attention.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            attention_dists: Precomputed relative position indices
                             of shape (seq_len, seq_len)
                             Values are in range [0, 2*max_pos_emb]

        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        # TODO: Implement forward pass
        # 1. Project to Q, K, V
        # 2. Reshape for multi-head attention
        # 3. Compute attention scores with relative positional bias
        # 4. Apply softmax and dropout
        # 5. Apply attention to values
        # 6. Reshape and project output
        raise NotImplementedError("ConformerAttention.forward() not yet implemented")


class ConformerConvModule(nn.Module):
    """
    Convolution module for Conformer block.

    Architecture (from HuggingFace implementation):
        x → LayerNorm →
        Pointwise Conv (expansion) →
        GLU →
        Depthwise Conv →
        BatchNorm →
        Activation (SiLU) →
        Pointwise Conv (compression) →
        Dropout → output

    Uses depthwise separable convolution for efficiency:
        - Pointwise conv for channel mixing
        - Depthwise conv for spatial/temporal patterns
        - Gated Linear Unit (GLU) for gating mechanism

    Args:
        dim: Input and output dimension
        kernel_size: Kernel size for depthwise convolution
        expansion_factor: Channel expansion factor
        dropout: Dropout probability
        activation: Activation function name
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 31,
        expansion_factor: int = 2,
        dropout: float = 0.1,
        activation: str = "silu",
    ):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.expansion_factor = expansion_factor
        self.dropout = dropout

        # TODO: Implement convolution module
        # - Layer normalization
        # - Pointwise convolution (expansion): dim → dim * expansion_factor * 2 (for GLU)
        # - GLU activation
        # - Depthwise convolution with padding
        # - Batch normalization
        # - Activation (SiLU)
        # - Pointwise convolution (compression): dim * expansion_factor → dim
        # - Dropout
        raise NotImplementedError("ConformerConvModule not yet implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through convolution module.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)

        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        # TODO: Implement forward pass
        # Note: Need to transpose for Conv1d: (batch, dim, seq_len)
        # Then transpose back: (batch, seq_len, dim)
        raise NotImplementedError("ConformerConvModule.forward() not yet implemented")


# ============================================================================
# Conformer Block
# ============================================================================


class ConformerBlock(nn.Module):
    """
    Single Conformer block combining feed-forward, attention, and convolution.

    Architecture (Conformer paper):
        x → [FF1 with 0.5x residual] →
        [Attention with 1.0x residual] →
        [Conv with 1.0x residual] →
        [FF2 with 0.5x residual] →
        LayerNorm → output

    Key features:
        - Half-step residual connections for feed-forward modules (0.5x scaling)
        - Full residual connections for attention and convolution
        - Pre-normalization (LayerNorm before each module)
        - Post-normalization (LayerNorm at the end)

    Args:
        config: ConformerConfig with all hyperparameters
    """

    def __init__(self, config: ConformerConfig):
        super().__init__()
        self.config = config

        # TODO: Implement Conformer block components
        # 1. First feed-forward module (with 0.5x scaling)
        # 2. Multi-head attention module
        # 3. Convolution module
        # 4. Second feed-forward module (with 0.5x scaling)
        # 5. Post layer normalization
        raise NotImplementedError("ConformerBlock not yet implemented")

    def forward(
        self,
        x: torch.Tensor,
        attention_dists: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through Conformer block.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)
            attention_dists: Precomputed relative position indices
                           of shape (seq_len, seq_len)

        Returns:
            Output tensor of shape (batch, seq_len, hidden_dim)
        """
        # TODO: Implement forward pass with residual connections
        # 1. x = x + 0.5 * ff1(x)
        # 2. x = x + attn(x, attention_dists)
        # 3. x = x + conv(x)
        # 4. x = x + 0.5 * ff2(x)
        # 5. x = post_norm(x)
        raise NotImplementedError("ConformerBlock.forward() not yet implemented")


# ============================================================================
# Conformer Encoder (Full Model)
# ============================================================================


class ConformerEncoder(nn.Module):
    """
    Full Conformer encoder for acoustic modeling.

    Architecture:
        Audio Features (batch, seq_len, num_features=80) →
        Input Projection (80 → hidden_dim) →
        Conformer Block 1 → ... → Conformer Block N →
        Output (batch, seq_len, hidden_dim)

    Key features:
        - Processes variable-length audio sequences
        - No temporal downsampling in Conformer blocks (Q-Former handles downsampling)
        - Precomputes relative position distances for all blocks
        - Shaw's relative positional embeddings for attention

    Args:
        config: ConformerConfig with all hyperparameters
        distributed_strategy: Strategy for distributed training (default: NoOpStrategy)

    Input:
        input_features: Audio features of shape (batch, seq_len, num_features)
                       Typically 80 log-mel filterbank features

    Output:
        hidden_states: Acoustic embeddings of shape (batch, seq_len, hidden_dim)
    """

    def __init__(
        self,
        config: ConformerConfig,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
    ):
        super().__init__()
        self.config = config
        self.distributed_strategy = distributed_strategy

        # TODO: Implement encoder components
        # 1. Input projection: num_features → hidden_dim
        # 2. Stack of Conformer blocks (num_layers)
        # 3. Register buffer for attention_dists (precomputed relative positions)
        raise NotImplementedError("ConformerEncoder not yet implemented")

    def _precompute_attention_dists(self, max_seq_len: int = 5000) -> torch.Tensor:
        """
        Precompute relative position distance matrix for attention.

        Computes pairwise distances between sequence positions:
            dist[i, j] = clamp(i - j, -context_size, context_size) + max_pos_emb

        This converts relative distances to indices for embedding lookup.

        Args:
            max_seq_len: Maximum sequence length to precompute

        Returns:
            Distance matrix of shape (max_seq_len, max_seq_len) with values in [0, 2*max_pos_emb]
        """
        # TODO: Implement attention distance precomputation
        # 1. Create position indices: [0, 1, 2, ..., max_seq_len-1]
        # 2. Compute pairwise differences: pos[i] - pos[j]
        # 3. Clamp to [-context_size, context_size]
        # 4. Shift by max_pos_emb to get indices in [0, 2*max_pos_emb]
        raise NotImplementedError("_precompute_attention_dists() not yet implemented")

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Conformer encoder.

        Args:
            input_features: Audio features of shape (batch, seq_len, num_features)
                          Expected: (batch, seq_len, 80) for 80 log-mel features

        Returns:
            hidden_states: Acoustic embeddings of shape (batch, seq_len, hidden_dim)

        Raises:
            AssertionError: If input_features dimension doesn't match config.num_features
        """
        # TODO: Implement forward pass
        # 1. Validate input shape
        # 2. Project input: (batch, seq_len, num_features) → (batch, seq_len, hidden_dim)
        # 3. Extract or pad attention_dists for current sequence length
        # 4. Pass through all Conformer blocks
        # 5. Return final hidden states
        raise NotImplementedError("ConformerEncoder.forward() not yet implemented")


# ============================================================================
# Model Registration (for FMS model registry)
# ============================================================================


def _create_conformer_config(
    num_features: int = 80,
    hidden_dim: int = 1024,
    num_layers: int = 16,
    num_heads: int = 8,
    dim_head: int = 64,
    **kwargs,
) -> ConformerConfig:
    """
    Factory function for creating Conformer configurations.

    Args:
        num_features: Number of input audio features
        hidden_dim: Hidden dimension
        num_layers: Number of Conformer blocks
        num_heads: Number of attention heads
        dim_head: Dimension per attention head
        **kwargs: Additional config parameters

    Returns:
        ConformerConfig instance
    """
    return ConformerConfig(
        num_features=num_features,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dim_head=dim_head,
        **kwargs,
    )


# Predefined configurations for common variants
CONFORMER_CONFIGS = {
    "granite_speech_16L_1024H": _create_conformer_config(
        num_features=80,
        hidden_dim=1024,
        num_layers=16,
        num_heads=8,
        dim_head=64,
        conv_kernel_size=31,
        feedforward_mult=4,
        dropout=0.1,
    ),
    "conformer_small_12L_512H": _create_conformer_config(
        num_features=80,
        hidden_dim=512,
        num_layers=12,
        num_heads=8,
        dim_head=64,
        conv_kernel_size=31,
        feedforward_mult=4,
        dropout=0.1,
    ),
}


# TODO: Register with FMS model registry
# from fms import models
# _architecture_name = "conformer"
# for variant_name, config in CONFORMER_CONFIGS.items():
#     models.register_model(
#         _architecture_name,
#         variant_name,
#         lambda c=config: ConformerEncoder(c)
#     )
