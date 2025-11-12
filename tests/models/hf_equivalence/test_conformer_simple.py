"""
Simple Conformer Performance Test (without HuggingFace dependencies)

This test benchmarks the FMS Conformer implementation without requiring
a working HuggingFace transformers installation.
"""

import time
import torch
import torch.nn as nn

from fms.models.conformer import ConformerConfig, ConformerEncoder


def count_parameters(model: nn.Module):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_forward_pass(
    model: nn.Module,
    input_features: torch.Tensor,
    num_warmup: int = 10,
    num_runs: int = 50,
) -> tuple[float, float]:
    """
    Benchmark forward pass time for a model.

    Args:
        model: The model to benchmark
        input_features: Input tensor
        num_warmup: Number of warmup iterations
        num_runs: Number of benchmark iterations

    Returns:
        Tuple of (average_time_ms, std_time_ms)
    """
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_features)

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(input_features)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

    return avg_time, std_time


def test_conformer_baseline_performance():
    """
    Test baseline performance of FMS Conformer implementation.

    This test validates:
    - Model constructs correctly
    - Forward pass works without errors
    - No NaN/Inf in outputs
    - Performance metrics are collected
    """
    print("\n" + "=" * 70)
    print("FMS CONFORMER BASELINE PERFORMANCE TEST")
    print("=" * 70)

    # Small config for quick testing
    config = ConformerConfig(
        num_features=80,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        dim_head=64,
        dropout=0.0,  # Disable dropout for reproducibility
    )

    encoder = ConformerEncoder(config)
    encoder.eval()

    # Test input
    batch_size = 2
    seq_length = 100
    input_features = torch.randn(batch_size, seq_length, 80)

    print(f"\nConfiguration:")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Heads: {config.num_heads}")
    print(f"  Parameters: {count_parameters(encoder):,}")

    print(f"\nInput shape: {input_features.shape}")

    # Test forward pass
    with torch.no_grad():
        output = encoder(input_features)

    print(f"Output shape: {output.shape}")

    # Check for numerical issues
    has_nan = torch.isnan(output).any().item()
    has_inf = torch.isinf(output).any().item()

    assert not has_nan, "Output contains NaN values"
    assert not has_inf, "Output contains Inf values"
    print(f"Numerical stability: OK (no NaN/Inf)")

    # Benchmark
    print(f"\nBenchmarking (warmup={10}, runs={50})...")
    avg_time, std_time = benchmark_forward_pass(encoder, input_features)

    print(f"Average time: {avg_time:.2f} ± {std_time:.2f} ms")

    # Throughput
    samples_per_second = 1000 * batch_size / avg_time
    print(f"Throughput: {samples_per_second:.2f} samples/sec")

    print("\n" + "=" * 70)
    print("BASELINE TEST PASSED")
    print("=" * 70)

    # Performance expectations (very lenient for baseline)
    assert avg_time < 5000, f"Forward pass too slow: {avg_time:.2f}ms"
    assert output.shape == (batch_size, seq_length, config.hidden_dim)


def test_conformer_variable_sequence_lengths():
    """Test that Conformer handles variable sequence lengths correctly."""
    print("\n" + "=" * 70)
    print("VARIABLE SEQUENCE LENGTH TEST")
    print("=" * 70)

    config = ConformerConfig(
        num_features=80,
        hidden_dim=256,
        num_layers=2,
        dropout=0.0,
    )
    encoder = ConformerEncoder(config)
    encoder.eval()

    test_lengths = [50, 100, 200, 500]
    batch_size = 1

    for seq_len in test_lengths:
        input_features = torch.randn(batch_size, seq_len, 80)

        # Validate output
        with torch.no_grad():
            output = encoder(input_features)

        assert output.shape == (batch_size, seq_len, config.hidden_dim), \
            f"Shape mismatch for seq_len={seq_len}: {output.shape}"

        assert not torch.isnan(output).any(), f"NaN for seq_len={seq_len}"
        assert not torch.isinf(output).any(), f"Inf for seq_len={seq_len}"

        # Benchmark this sequence length
        avg_time, std_time = benchmark_forward_pass(
            encoder, input_features, num_warmup=5, num_runs=20
        )
        throughput = 1000 * batch_size / avg_time

        print(f"  seq_len={seq_len:3d}: {avg_time:6.2f} ± {std_time:5.2f} ms | {throughput:6.2f} samples/sec | shape={output.shape}")

    print("\nVariable sequence length test passed")


def test_conformer_gradient_flow():
    """Test that gradients flow correctly through the model."""
    print("\n" + "=" * 70)
    print("GRADIENT FLOW TEST")
    print("=" * 70)

    config = ConformerConfig(
        num_features=80,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
    )
    encoder = ConformerEncoder(config)
    encoder.train()

    input_features = torch.randn(2, 50, 80, requires_grad=True)
    output = encoder(input_features)

    # Compute a simple loss
    loss = output.sum()
    loss.backward()

    # Check gradients exist and are not NaN
    assert input_features.grad is not None, "No gradient for input"
    assert not torch.isnan(input_features.grad).any(), "Gradient contains NaN"

    # Check at least some model parameters have gradients
    has_gradients = False
    for param in encoder.parameters():
        if param.grad is not None:
            has_gradients = True
            assert not torch.isnan(param.grad).any(), "Parameter gradient contains NaN"

    assert has_gradients, "No parameter gradients found"

    print("  Gradients flow correctly")
    print("  No NaN in gradients")
    print("\nGradient flow test passed")


if __name__ == "__main__":
    test_conformer_baseline_performance()
    test_conformer_variable_sequence_lengths()
    test_conformer_gradient_flow()
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! 🎉")
    print("=" * 70)
