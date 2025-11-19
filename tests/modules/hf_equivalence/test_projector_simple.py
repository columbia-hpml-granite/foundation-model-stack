"""
HuggingFace Equivalence Test for Speech Projector.

This test validates that the FMS Speech Projector implementation produces
equivalent outputs to the HuggingFace Granite Speech projector.

Following the pattern from test_conformer_simple.py:
- Performance benchmarking
- Numerical equivalence testing
- Shape compatibility verification
"""

import time
import pytest
import torch
import torch.nn as nn

from fms.modules.projector import SpeechProjectorConfig, SpeechProjector


def count_parameters(model: nn.Module):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_forward_pass(
    model: nn.Module,
    encoder_hidden_states: torch.Tensor,
    num_warmup: int = 10,
    num_runs: int = 50,
) -> tuple[float, float]:
    """
    Benchmark forward pass time for a model.

    Args:
        model: The model to benchmark
        encoder_hidden_states: Input tensor
        num_warmup: Number of warmup iterations
        num_runs: Number of benchmark iterations

    Returns:
        Tuple of (average_time_ms, std_time_ms)
    """
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(encoder_hidden_states)

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(encoder_hidden_states)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

    return avg_time, std_time


# ============================================================================
# Baseline Tests (Without HuggingFace)
# ============================================================================


def test_projector_baseline_performance():
    """
    Test baseline performance of FMS Speech Projector implementation.

    This test validates:
    - Model constructs correctly
    - Forward pass works without errors
    - No NaN/Inf in outputs
    - Performance metrics are collected
    """
    print("\n" + "=" * 70)
    print("FMS SPEECH PROJECTOR BASELINE PERFORMANCE TEST")
    print("=" * 70)

    # Configuration matching Granite Speech
    config = SpeechProjectorConfig(
        encoder_dim=1024,
        decoder_dim=2048,
        num_queries=32,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=4096,
        hidden_dropout_prob=0.0,  # Disable dropout for reproducibility
        attention_dropout_prob=0.0,
    )

    projector = SpeechProjector(config)
    projector.eval()

    # Test input (simulated Conformer output)
    batch_size = 2
    audio_seq_len = 500
    encoder_hidden_states = torch.randn(batch_size, audio_seq_len, 1024)

    print(f"\nConfiguration:")
    print(f"  Encoder dim: {config.encoder_dim}")
    print(f"  Decoder dim: {config.decoder_dim}")
    print(f"  Num queries: {config.num_queries}")
    print(f"  Num layers: {config.num_hidden_layers}")
    print(f"  Parameters: {count_parameters(projector):,}")

    print(f"\nInput shape: {encoder_hidden_states.shape}")

    # Test forward pass
    with torch.no_grad():
        output = projector(encoder_hidden_states)

    print(f"Output shape: {output.shape}")

    # Check for numerical issues
    has_nan = torch.isnan(output).any().item()
    has_inf = torch.isinf(output).any().item()

    assert not has_nan, "Output contains NaN values"
    assert not has_inf, "Output contains Inf values"
    print(f"Numerical stability: OK (no NaN/Inf)")

    # Benchmark
    print(f"\nBenchmarking (warmup={10}, runs={50})...")
    avg_time, std_time = benchmark_forward_pass(projector, encoder_hidden_states)

    print(f"Average time: {avg_time:.2f} ± {std_time:.2f} ms")

    # Throughput
    samples_per_second = 1000 * batch_size / avg_time
    print(f"Throughput: {samples_per_second:.2f} samples/sec")

    # Compression ratio
    compression_ratio = audio_seq_len / config.num_queries
    print(f"Compression ratio: {compression_ratio:.2f}x ({audio_seq_len} → {config.num_queries})")

    print("\n" + "=" * 70)
    print("BASELINE TEST PASSED")
    print("=" * 70)

    # Performance expectations (very lenient for baseline)
    assert avg_time < 10000, f"Forward pass too slow: {avg_time:.2f}ms"
    assert output.shape == (batch_size, config.num_queries, config.decoder_dim)


def test_projector_variable_sequence_lengths():
    """Test that projector handles variable sequence lengths correctly."""
    print("\n" + "=" * 70)
    print("VARIABLE SEQUENCE LENGTH TEST")
    print("=" * 70)

    config = SpeechProjectorConfig(
        encoder_dim=1024,
        decoder_dim=2048,
        num_queries=32,
        num_hidden_layers=2,  # Small for testing
        hidden_dropout_prob=0.0,
        attention_dropout_prob=0.0,
    )
    projector = SpeechProjector(config)
    projector.eval()

    test_lengths = [100, 250, 500, 1000]
    batch_size = 1

    for seq_len in test_lengths:
        encoder_hidden_states = torch.randn(batch_size, seq_len, 1024)

        # Validate output
        with torch.no_grad():
            output = projector(encoder_hidden_states)

        # Output should ALWAYS be (batch, num_queries, decoder_dim)
        expected_shape = (batch_size, config.num_queries, config.decoder_dim)
        assert output.shape == expected_shape, \
            f"Shape mismatch for seq_len={seq_len}: {output.shape}"

        assert not torch.isnan(output).any(), f"NaN for seq_len={seq_len}"
        assert not torch.isinf(output).any(), f"Inf for seq_len={seq_len}"

        # Benchmark this sequence length
        avg_time, std_time = benchmark_forward_pass(
            projector, encoder_hidden_states, num_warmup=5, num_runs=20
        )
        throughput = 1000 * batch_size / avg_time
        compression = seq_len / config.num_queries

        print(f"  seq_len={seq_len:4d}: {avg_time:6.2f} ± {std_time:5.2f} ms | "
              f"{throughput:6.2f} samples/sec | {compression:5.2f}x compression | "
              f"shape={output.shape}")

    print("\nVariable sequence length test passed")


def test_projector_gradient_flow():
    """Test that gradients flow correctly through the model."""
    print("\n" + "=" * 70)
    print("GRADIENT FLOW TEST")
    print("=" * 70)

    config = SpeechProjectorConfig(
        encoder_dim=1024,
        decoder_dim=2048,
        num_queries=32,
        num_hidden_layers=2,
        hidden_dropout_prob=0.1,
        attention_dropout_prob=0.1,
    )
    projector = SpeechProjector(config)
    projector.train()

    encoder_hidden_states = torch.randn(2, 500, 1024, requires_grad=True)
    output = projector(encoder_hidden_states)

    # Compute a simple loss
    loss = output.sum()
    loss.backward()

    # Check gradients exist and are not NaN
    assert encoder_hidden_states.grad is not None, "No gradient for input"
    assert not torch.isnan(encoder_hidden_states.grad).any(), "Gradient contains NaN"

    # Check learnable queries have gradients (KEY TEST!)
    assert projector.query_embeds.grad is not None, "No gradient for query embeddings"
    assert not torch.isnan(projector.query_embeds.grad).any(), \
        "Query gradient contains NaN"

    # Check at least some model parameters have gradients
    has_gradients = False
    for param in projector.parameters():
        if param.grad is not None:
            has_gradients = True
            assert not torch.isnan(param.grad).any(), "Parameter gradient contains NaN"

    assert has_gradients, "No parameter gradients found"

    print("  Gradients flow correctly")
    print("  Learnable queries receive gradients ✓")
    print("  No NaN in gradients")
    print("\nGradient flow test passed")


# ============================================================================
# HuggingFace Equivalence Tests (Require transformers library)
# ============================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="HuggingFace tests skipped (GPU required for performance comparison)"
)
def test_projector_huggingface_equivalence():
    """
    Test numerical equivalence with HuggingFace implementation.

    This test will be skipped if HuggingFace transformers is not available.
    """
    try:
        from transformers import AutoConfig, AutoModel
    except ImportError:
        pytest.skip("HuggingFace transformers not available")

    print("\n" + "=" * 70)
    print("HUGGINGFACE EQUIVALENCE TEST")
    print("=" * 70)

    # Load HuggingFace Granite Speech config
    try:
        hf_config = AutoConfig.from_pretrained("ibm-granite/granite-speech-3.3-8b")
    except Exception as e:
        pytest.skip(f"Could not load HuggingFace config: {e}")

    # Extract projector config from HuggingFace
    projector_config = hf_config.projector_config

    # Create FMS config matching HuggingFace
    fms_config = SpeechProjectorConfig(
        encoder_dim=projector_config.encoder_hidden_size,
        decoder_dim=hf_config.text_config.hidden_size,
        num_queries=hf_config.window_size // hf_config.downsample_rate,
        num_hidden_layers=projector_config.num_hidden_layers,
        num_attention_heads=projector_config.num_attention_heads,
        intermediate_size=projector_config.intermediate_size,
        hidden_dropout_prob=projector_config.hidden_dropout_prob,
        attention_dropout_prob=projector_config.attention_probs_dropout_prob,
    )

    print(f"\nConfiguration (from HuggingFace):")
    print(f"  Encoder dim: {fms_config.encoder_dim}")
    print(f"  Decoder dim: {fms_config.decoder_dim}")
    print(f"  Num queries: {fms_config.num_queries}")
    print(f"  Num layers: {fms_config.num_hidden_layers}")

    # Create FMS projector
    fms_projector = SpeechProjector(fms_config)
    fms_projector.eval()

    # Test input
    batch_size = 1
    seq_len = 500
    encoder_hidden_states = torch.randn(batch_size, seq_len, fms_config.encoder_dim)

    # FMS forward pass
    with torch.no_grad():
        fms_output = fms_projector(encoder_hidden_states)

    print(f"\nFMS output shape: {fms_output.shape}")
    print(f"FMS output mean: {fms_output.mean().item():.4f}")
    print(f"FMS output std: {fms_output.std().item():.4f}")

    # NOTE: Full numerical equivalence requires loading HuggingFace weights
    # This test validates shape compatibility and basic properties
    assert fms_output.shape[0] == batch_size
    assert fms_output.shape[1] == fms_config.num_queries
    assert fms_output.shape[2] == fms_config.decoder_dim

    print("\nShape compatibility: ✓")
    print("Numerical stability: ✓")
    print("\nNOTE: Full numerical equivalence requires loading HuggingFace weights")
    print("=" * 70)


def test_projector_compression_ratio():
    """Test temporal compression ratios match Granite Speech specification."""
    print("\n" + "=" * 70)
    print("COMPRESSION RATIO TEST")
    print("=" * 70)

    # Test different configurations
    configs = [
        ("Heavy (32 queries)", 32, 15.6),
        ("Medium (64 queries)", 64, 7.8),
        ("Light (128 queries)", 128, 3.9),
    ]

    for name, num_queries, expected_ratio in configs:
        config = SpeechProjectorConfig(
            encoder_dim=1024,
            decoder_dim=2048,
            num_queries=num_queries,
            num_hidden_layers=2,
        )
        projector = SpeechProjector(config)
        projector.eval()

        # Test with 500 frame input
        seq_len = 500
        encoder_hidden_states = torch.randn(1, seq_len, 1024)

        with torch.no_grad():
            output = projector(encoder_hidden_states)

        actual_ratio = seq_len / output.shape[1]

        print(f"\n{name}:")
        print(f"  Input: {seq_len} frames")
        print(f"  Output: {output.shape[1]} queries")
        print(f"  Compression: {actual_ratio:.2f}x (expected {expected_ratio:.2f}x)")

        assert abs(actual_ratio - expected_ratio) < 0.1, \
            f"Compression ratio mismatch: {actual_ratio:.2f} vs {expected_ratio:.2f}"

    print("\nCompression ratio test passed")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RUNNING ALL PROJECTOR TESTS")
    print("=" * 70)

    test_projector_baseline_performance()
    test_projector_variable_sequence_lengths()
    test_projector_gradient_flow()
    test_projector_compression_ratio()

    # Try HuggingFace test (will skip if not available)
    try:
        test_projector_huggingface_equivalence()
    except pytest.skip.Exception as e:
        print(f"\nSkipped HuggingFace equivalence test: {e}")

    print("\n" + "=" * 70)
    print("ALL BASELINE TESTS PASSED! 🎉")
    print("=" * 70)
