import logging
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fms import models
from fms.distributed.strategy import DistributedStrategy, NoOpStrategy
from fms.utils import serialization
from fms.utils.activation import str_to_activation
from fms.utils.config import ModelConfig


logger = logging.getLogger(__name__)


@dataclass
class ConformerConfig(ModelConfig):
    """
    Configuration class for Conformer encoder.

    Default values are set to match Granite-Speech-3.3-8b specifications.

    Args:
        num_features: Number of input audio features (default: 160 = 80 log-mel * 2 channels)
        hidden_dim: Hidden dimension for encoder layers
        num_layers: Number of Conformer blocks (default: 16)
        num_heads: Number of attention heads in multi-head attention
        dim_head: Dimension per attention head
        conv_kernel_size: Kernel size for depthwise convolution
        conv_expansion_factor: Expansion factor for convolution module
        feedforward_mult: Expansion multiplier for feed-forward networks
        dropout: Dropout probability applied throughout the model
        max_pos_emb: Maximum positional embedding distance for relative attention
        context_size: Local attention window size (sequence positions are clamped to +/- context_size)
        output_dim: CTC output dimension for mid-layer supervision
        use_ctc: Enable/disable mid-layer CTC output
        activation: Activation function name (default: "silu" for SiLU/Swish)
        linear_config: Configuration for linear module selection
    """

    num_features: int = 160  # Input: 80 log-mel * 2 channels
    hidden_dim: int = 1024  # Encoder hidden dimension
    num_layers: int = 16  # Number of Conformer blocks

    # Multi-head attention parameters
    num_heads: int = 8
    dim_head: int = 128  # Per-head dimension (8 heads * 128 = 1024 inner_dim)

    # Convolution module parameters
    conv_kernel_size: int = 15
    conv_expansion_factor: int = 2

    # Feed-forward parameters
    feedforward_mult: int = 4  # FFN expansion: hidden_dim * feedforward_mult

    # Regularization
    dropout: float = 0.1

    # Positional encoding parameters
    max_pos_emb: int = 512  # Maximum relative position distance
    context_size: int = 200  # Local attention window

    # CTC output dimension for mid-layer supervision
    output_dim: int = 42  # CTC vocabulary size

    # Enable/disable mid-layer CTC output
    use_ctc: bool = True

    # Activation function
    activation: str = "silu"  # SiLU (Swish) activation

    # Linear module configuration
    linear_config: Optional[Mapping[str, Any]] = None


class ConformerFeedForward(nn.Module):
    """
    Feed-forward module for Conformer block.

    Architecture:
        x -> LayerNorm -> Linear (dim -> dim * mult) -> Activation -> Dropout ->
        Linear (dim * mult -> dim) -> Dropout -> output

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

        # Layer normalization
        self.norm = nn.LayerNorm(dim)

        # Linear projection: dim -> dim * mult
        self.fc1 = nn.Linear(dim, dim * mult)

        # Activation function
        self.activation = str_to_activation(activation)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)

        # Linear projection: dim * mult -> dim
        self.fc2 = nn.Linear(dim * mult, dim)

        # Dropout
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feed-forward network.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)

        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        # x -> LayerNorm -> Linear -> Activation -> Dropout -> Linear -> Dropout
        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


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
        self.dropout_prob = dropout
        self.scale = dim_head ** -0.5

        # Layer normalization
        self.norm = nn.LayerNorm(dim)

        # Query, Key, Value projections
        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=False)

        # Relative positional embeddings (learnable)
        # Embedding size: 2*max_pos_emb + 1 (for positions from -max_pos_emb to +max_pos_emb)
        self.pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)

        # Output projection
        self.to_out = nn.Linear(self.inner_dim, dim)

        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

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
        batch, seq_len, _ = x.shape

        # Apply layer normalization
        x = self.norm(x)

        # 1. Project to Q, K, V
        q = self.to_q(x)  # (batch, seq_len, inner_dim)
        k = self.to_k(x)  # (batch, seq_len, inner_dim)
        v = self.to_v(x)  # (batch, seq_len, inner_dim)

        # 2. Reshape for multi-head attention
        q = q.view(batch, seq_len, self.num_heads, self.dim_head).transpose(1, 2)  # (batch, heads, seq_len, dim_head)
        k = k.view(batch, seq_len, self.num_heads, self.dim_head).transpose(1, 2)  # (batch, heads, seq_len, dim_head)
        v = v.view(batch, seq_len, self.num_heads, self.dim_head).transpose(1, 2)  # (batch, heads, seq_len, dim_head)

        # 3. Compute attention scores with relative positional bias
        # Standard attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch, heads, seq_len, seq_len)

        # Add relative positional bias
        # Get positional embeddings for the current sequence
        pos_emb = self.pos_emb(attention_dists[:seq_len, :seq_len])  # (seq_len, seq_len, dim_head)

        # Compute positional bias: Q * pos_emb^T
        # q shape: (batch, heads, seq_len, dim_head)
        # pos_emb shape: (seq_len, seq_len, dim_head)
        # We want: (batch, heads, seq_len, seq_len)
        pos_bias = torch.einsum('bhid,ijd->bhij', q, pos_emb) * self.scale

        attn_scores = attn_scores + pos_bias

        # 4. Apply softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 5. Apply attention to values
        out = torch.matmul(attn_weights, v)  # (batch, heads, seq_len, dim_head)

        # 6. Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.inner_dim)  # (batch, seq_len, inner_dim)
        out = self.to_out(out)  # (batch, seq_len, dim)
        out = self.out_dropout(out)

        return out


class ConformerConvModule(nn.Module):
    """
    Convolution module for Conformer block.

    Architecture (from HuggingFace implementation):
        x -> LayerNorm ->
        Pointwise Conv (expansion) ->
        GLU ->
        Depthwise Conv ->
        BatchNorm ->
        Activation (SiLU) ->
        Pointwise Conv (compression) ->
        Dropout -> output

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

        # Layer normalization
        self.norm = nn.LayerNorm(dim)

        # Pointwise convolution (expansion): dim -> dim * expansion_factor * 2 (for GLU)
        # We use *2 because GLU splits the channels in half
        self.pointwise_conv1 = nn.Conv1d(
            dim,
            dim * expansion_factor * 2,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # GLU activation (applied in forward pass)
        # GLU splits channels and applies sigmoid gating

        # Depthwise convolution with padding
        # Padding = (kernel_size - 1) // 2 for 'same' padding
        # Note: bias=False to match HF granite_speech encoder
        self.depthwise_conv = nn.Conv1d(
            dim * expansion_factor,
            dim * expansion_factor,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=dim * expansion_factor,  # Depthwise: each input channel convolved separately
            bias=False,
        )

        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(dim * expansion_factor)

        # Activation (SiLU)
        self.activation = str_to_activation(activation)

        # Pointwise convolution (compression): dim * expansion_factor -> dim
        self.pointwise_conv2 = nn.Conv1d(
            dim * expansion_factor,
            dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through convolution module.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)

        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        # Apply layer normalization
        x = self.norm(x)

        # Transpose for Conv1d: (batch, seq_len, dim) -> (batch, dim, seq_len)
        x = x.transpose(1, 2)

        # Pointwise expansion
        x = self.pointwise_conv1(x)  # (batch, dim * expansion_factor * 2, seq_len)

        # GLU activation: split channels and apply gating
        x, gate = x.chunk(2, dim=1)  # Each: (batch, dim * expansion_factor, seq_len)
        x = x * torch.sigmoid(gate)

        # Depthwise convolution
        x = self.depthwise_conv(x)  # (batch, dim * expansion_factor, seq_len)

        # Batch normalization
        x = self.batch_norm(x)

        # Activation
        x = self.activation(x)

        # Pointwise compression
        x = self.pointwise_conv2(x)  # (batch, dim, seq_len)

        # Transpose back: (batch, dim, seq_len) -> (batch, seq_len, dim)
        x = x.transpose(1, 2)

        # Dropout
        x = self.dropout_layer(x)

        return x


class ConformerBlock(nn.Module):
    """
    Single Conformer block combining feed-forward, attention, and convolution.

    Architecture (Conformer paper):
        x -> [FF1 with 0.5x residual] ->
        [Attention with 1.0x residual] ->
        [Conv with 1.0x residual] ->
        [FF2 with 0.5x residual] ->
        LayerNorm -> output

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

        # 1. First feed-forward module (with 0.5x scaling)
        self.ff1 = ConformerFeedForward(
            dim=config.hidden_dim,
            mult=config.feedforward_mult,
            dropout=config.dropout,
            activation=config.activation,
        )

        # 2. Multi-head attention module
        self.attn = ConformerAttention(
            dim=config.hidden_dim,
            num_heads=config.num_heads,
            dim_head=config.dim_head,
            max_pos_emb=config.max_pos_emb,
            dropout=config.dropout,
        )

        # 3. Convolution module
        self.conv = ConformerConvModule(
            dim=config.hidden_dim,
            kernel_size=config.conv_kernel_size,
            expansion_factor=config.conv_expansion_factor,
            dropout=config.dropout,
            activation=config.activation,
        )

        # 4. Second feed-forward module (with 0.5x scaling)
        self.ff2 = ConformerFeedForward(
            dim=config.hidden_dim,
            mult=config.feedforward_mult,
            dropout=config.dropout,
            activation=config.activation,
        )

        # 5. Post layer normalization
        self.post_norm = nn.LayerNorm(config.hidden_dim)

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
        # 1. x = x + 0.5 * ff1(x) - Half-step residual for feed-forward
        x = x + 0.5 * self.ff1(x)

        # 2. x = x + attn(x, attention_dists) - Full residual for attention
        x = x + self.attn(x, attention_dists)

        # 3. x = x + conv(x) - Full residual for convolution
        x = x + self.conv(x)

        # 4. x = x + 0.5 * ff2(x) - Half-step residual for feed-forward
        x = x + 0.5 * self.ff2(x)

        # 5. x = post_norm(x) - Post layer normalization
        x = self.post_norm(x)

        return x


class ConformerEncoder(nn.Module):
    """
    Full Conformer encoder for acoustic modeling.

    Processes audio features through a stack of Conformer blocks with optional mid-layer
    CTC supervision. Uses Shaw's relative positional embeddings for attention.

    Args:
        config: ConformerConfig with all hyperparameters
        distributed_strategy: Strategy for distributed training

    Input:
        input_features: Audio features of shape (batch, seq_len, num_features)

    Output:
        hidden_states: Acoustic embeddings of shape (batch, seq_len, hidden_dim)
    """

    def __init__(
        self,
        config: Optional[ConformerConfig] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super().__init__()
        if config is not None:
            self.config = config
        else:
            self.config = ConformerConfig()
        self.config = self.config.updated(**kwargs)
        self.distributed_strategy = distributed_strategy

        # Input projection: num_features -> hidden_dim
        self.input_proj = nn.Linear(self.config.num_features, self.config.hidden_dim)

        # Stack of Conformer blocks
        self.blocks = nn.ModuleList([
            ConformerBlock(self.config) for _ in range(self.config.num_layers)
        ])

        # CTC output layers for mid-layer supervision
        if self.config.use_ctc:
            # out: projects hidden_dim -> output_dim (CTC logits)
            self.out = nn.Linear(self.config.hidden_dim, self.config.output_dim)
            # out_mid: projects output_dim -> hidden_dim (feedback to encoder)
            self.out_mid = nn.Linear(self.config.output_dim, self.config.hidden_dim)
        else:
            self.out = None
            self.out_mid = None

        # Precompute relative position distances for attention
        attention_dists = self._precompute_attention_dists(max_seq_len=5000)
        self.register_buffer("attention_dists", attention_dists)

    @classmethod
    def from_config(cls, config: ConformerConfig) -> "ConformerEncoder":
        return cls(config)

    def get_config(self) -> ConformerConfig:
        return self.config

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
        # 1. Create position indices: [0, 1, 2, ..., max_seq_len-1]
        positions = torch.arange(max_seq_len)

        # 2. Compute pairwise differences: pos[i] - pos[j]
        # Using broadcasting: (max_seq_len, 1) - (1, max_seq_len) = (max_seq_len, max_seq_len)
        relative_dists = positions.unsqueeze(0) - positions.unsqueeze(1)

        # 3. Clamp to [-context_size, context_size]
        relative_dists = torch.clamp(
            relative_dists,
            min=-self.config.context_size,
            max=self.config.context_size
        )

        # 4. Shift by max_pos_emb to get indices in [0, 2*max_pos_emb]
        attention_dists = relative_dists + self.config.max_pos_emb

        return attention_dists.long()

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
        # 1. Validate input shape
        batch_size, seq_len, num_features = input_features.shape
        assert num_features == self.config.num_features, (
            f"Input features dimension {num_features} doesn't match "
            f"config.num_features {self.config.num_features}"
        )

        # 2. Project input: (batch, seq_len, num_features) -> (batch, seq_len, hidden_dim)
        x = self.input_proj(input_features)

        # 3. Extract attention_dists for current sequence length
        # If sequence is longer than precomputed, we'll handle it
        if seq_len > self.attention_dists.size(0):
            # Dynamically compute for longer sequences
            attention_dists = self._precompute_attention_dists(max_seq_len=seq_len)
            attention_dists = attention_dists.to(input_features.device)
        else:
            attention_dists = self.attention_dists

        # 4. Pass through all Conformer blocks with optional mid-layer CTC
        # Reference: HF modeling_granite_speech.py:270-278
        mid_layer = len(self.blocks) // 2
        for idx, block in enumerate(self.blocks, start=1):
            x = block(x, attention_dists)

            # Mid-layer CTC feedback (HF-aligned)
            # At the middle layer, compute CTC output and feed back into encoder
            if self.config.use_ctc and self.out is not None and idx == mid_layer:
                x_mid = self.out(x)  # (batch, seq_len, output_dim)
                x = x + self.out_mid(F.softmax(x_mid, dim=-1))  # Feedback to encoder

        # 5. Return final hidden states
        return x


_architecture_name = "conformer"
_default_config = ConformerConfig()


def _conformer_factory_factory(config):
    def factory(**kwargs):
        return ConformerEncoder(config, **kwargs)
    return factory


models.register_model(
    _architecture_name,
    "granite_speech",
    _conformer_factory_factory(_default_config),
)


def _hf_to_fms_names(input_sd: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
    """
    Convert HuggingFace Conformer weight names to FMS format.

    Maps encoder weights from HF naming convention to FMS naming convention.
    This enables loading pretrained HF Conformer checkpoints into FMS models.

    HF Structure (from granite-speech encoder):
        encoder.input_linear -> input projection
        encoder.layers.{i} -> conformer blocks
        encoder.layers.{i}.ff1 -> first feed-forward
        encoder.layers.{i}.attn -> attention module
        encoder.layers.{i}.conv -> convolution module
        encoder.layers.{i}.ff2 -> second feed-forward
        encoder.out -> CTC output layer
        encoder.out_mid -> CTC feedback layer

    FMS Structure:
        input_proj -> input projection
        blocks.{i} -> conformer blocks
        blocks.{i}.ff1 -> first feed-forward
        blocks.{i}.attn -> attention module
        blocks.{i}.conv -> convolution module
        blocks.{i}.ff2 -> second feed-forward
        out -> CTC output layer
        out_mid -> CTC feedback layer

    Args:
        input_sd: Input state dict with HF weight names
        **kwargs: Additional arguments (unused)

    Returns:
        State dict with FMS weight names
    """
    replacements = [
        # Input projection layer
        (r"^encoder\.input_linear\.", "input_proj."),

        # Layer index: HF uses 'layers', FMS uses 'blocks'
        (r"^encoder\.layers\.(\d+)\.", r"blocks.\1."),

        # Feed-forward modules (ff1 and ff2)
        (r"\.ff1\.pre_norm\.", ".ff1.norm."),
        (r"\.ff1\.up_proj\.", ".ff1.fc1."),
        (r"\.ff1\.down_proj\.", ".ff1.fc2."),
        (r"\.ff2\.pre_norm\.", ".ff2.norm."),
        (r"\.ff2\.up_proj\.", ".ff2.fc1."),
        (r"\.ff2\.down_proj\.", ".ff2.fc2."),

        # Attention module
        (r"\.attn\.pre_norm\.", ".attn.norm."),
        (r"\.attn\.to_q\.", ".attn.to_q."),
        (r"\.attn\.to_k\.", ".attn.to_k."),
        (r"\.attn\.to_v\.", ".attn.to_v."),
        (r"\.attn\.to_out\.", ".attn.to_out."),
        (r"\.attn\.rel_pos_emb\.", ".attn.pos_emb."),

        # Convolution module
        (r"\.conv\.pre_norm\.", ".conv.norm."),
        (r"\.conv\.up_conv\.", ".conv.pointwise_conv1."),
        (r"\.conv\.depth_conv\.conv\.", ".conv.depthwise_conv."),
        (r"\.conv\.down_conv\.", ".conv.pointwise_conv2."),
        (r"\.conv\.batch_norm\.", ".conv.batch_norm."),

        # Post normalization
        (r"\.post_norm\.", ".post_norm."),

        # CTC layers (mid-layer supervision)
        (r"^encoder\.out\.", "out."),
        (r"^encoder\.out_mid\.", "out_mid."),
    ]

    new_sd = {}
    for name, param in input_sd.items():
        new_name = name
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)
        new_sd[new_name] = param

    return new_sd


def _weight_fusion(
    input_sd: Mapping[str, Any],
    model_config: Optional[ConformerConfig] = None,
    **kwargs
) -> Mapping[str, Any]:
    """
    Weight fusion adapter for Conformer.

    Note: Conformer doesn't use fused attention/MLP weights like decoder models
    (e.g., Granite, LLaMA). The Conformer architecture uses separate Q, K, V
    projections and doesn't benefit from the same fusion optimizations.

    This is a pass-through function for consistency with the FMS adapter pattern.

    Args:
        input_sd: Input state dict
        model_config: Optional Conformer config (unused)
        **kwargs: Additional arguments (unused)

    Returns:
        Unmodified state dict
    """
    # Conformer uses standard attention without weight fusion
    return input_sd


# Register serialization adapter steps
serialization.register_adapter_step(
    _architecture_name, "hf_to_fms_names", _hf_to_fms_names
)

serialization.register_adapter_step(
    _architecture_name, "weight_fusion", _weight_fusion
)

# Register complete HF adapter (defines the pipeline: name conversion -> fusion)
serialization.register_adapter(
    _architecture_name,
    "hf",
    ["hf_to_fms_names", "weight_fusion"],
)
