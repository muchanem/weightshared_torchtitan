#!/usr/bin/env python3
"""
Test the integrated batched FFN in weight_sharing.py
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, "/local/scratch/muchane_680186/weightshared_torchtitan")

from torchtitan.models.qwen3.model.weight_sharing import SharedFeedForward


def test_batched_vs_sequential():
    """Compare batched vs sequential FFN implementations."""
    print("=" * 60)
    print("Testing SharedFeedForward: batched vs sequential")
    print("=" * 60)

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    batch, seq, dim = 4, 128, 256
    hidden_dim = 512
    rank = 32

    # Create shared weights
    shared_weights = {
        "w1": nn.Linear(dim, hidden_dim, bias=False).to(device).to(dtype),
        "w2": nn.Linear(hidden_dim, dim, bias=False).to(device).to(dtype),
        "w3": nn.Linear(dim, hidden_dim, bias=False).to(device).to(dtype),
    }

    # Create batched FFN (default now)
    ffn_batched = SharedFeedForward(
        shared_weights, dim, hidden_dim, rank, use_batched_lora=True
    ).to(device).to(dtype)

    # Create sequential FFN
    ffn_sequential = SharedFeedForward(
        shared_weights, dim, hidden_dim, rank, use_batched_lora=False
    ).to(device).to(dtype)

    # Copy weights from sequential to batched for fair comparison
    with torch.no_grad():
        # w1 LoRA
        ffn_batched.w13_lora.w_a[0].copy_(ffn_sequential.w1.lora.w_a.weight)
        ffn_batched.w13_lora.w_b[0].copy_(ffn_sequential.w1.lora.w_b.weight)
        # w3 LoRA
        ffn_batched.w13_lora.w_a[1].copy_(ffn_sequential.w3.lora.w_a.weight)
        ffn_batched.w13_lora.w_b[1].copy_(ffn_sequential.w3.lora.w_b.weight)
        # w2 LoRA
        ffn_batched.w2_lora.w_a.weight.copy_(ffn_sequential.w2.lora.w_a.weight)
        ffn_batched.w2_lora.w_b.weight.copy_(ffn_sequential.w2.lora.w_b.weight)

    # Test forward
    x = torch.randn(batch, seq, dim, device=device, dtype=dtype)

    out_batched = ffn_batched(x)
    out_sequential = ffn_sequential(x)

    diff = (out_batched - out_sequential).abs().max().item()
    print(f"Max diff batched vs sequential: {diff:.6f}")
    assert diff < 1e-2, f"Mismatch: {diff}"
    print("PASSED: Correctness test\n")


def test_backward_pass():
    """Test that gradients flow correctly."""
    print("=" * 60)
    print("Testing backward pass")
    print("=" * 60)

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    batch, seq, dim = 4, 64, 128
    hidden_dim = 256
    rank = 16

    shared_weights = {
        "w1": nn.Linear(dim, hidden_dim, bias=False).to(device).to(dtype),
        "w2": nn.Linear(hidden_dim, dim, bias=False).to(device).to(dtype),
        "w3": nn.Linear(dim, hidden_dim, bias=False).to(device).to(dtype),
    }

    ffn = SharedFeedForward(
        shared_weights, dim, hidden_dim, rank, use_batched_lora=True
    ).to(device).to(dtype)

    x = torch.randn(batch, seq, dim, device=device, dtype=dtype, requires_grad=True)

    out = ffn(x)
    loss = out.sum()
    loss.backward()

    # Check gradients exist
    assert x.grad is not None, "No gradient for input"
    assert ffn.w13_lora.w_a.grad is not None, "No gradient for w13_lora.w_a"
    assert ffn.w13_lora.w_b.grad is not None, "No gradient for w13_lora.w_b"
    assert ffn.w2_lora.w_a.weight.grad is not None, "No gradient for w2_lora.w_a"
    assert ffn.w2_lora.w_b.weight.grad is not None, "No gradient for w2_lora.w_b"

    print("All gradients computed successfully")
    print("PASSED: Backward pass\n")


def benchmark():
    """Benchmark batched vs sequential FFN."""
    print("=" * 60)
    print("Benchmarking FFN performance")
    print("=" * 60)

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    batch, seq, dim = 4, 2048, 1024
    hidden_dim = 2816
    rank = 256

    shared_weights = {
        "w1": nn.Linear(dim, hidden_dim, bias=False).to(device).to(dtype),
        "w2": nn.Linear(hidden_dim, dim, bias=False).to(device).to(dtype),
        "w3": nn.Linear(dim, hidden_dim, bias=False).to(device).to(dtype),
    }

    ffn_batched = SharedFeedForward(
        shared_weights, dim, hidden_dim, rank, use_batched_lora=True
    ).to(device).to(dtype)

    ffn_sequential = SharedFeedForward(
        shared_weights, dim, hidden_dim, rank, use_batched_lora=False
    ).to(device).to(dtype)

    x = torch.randn(batch, seq, dim, device=device, dtype=dtype)

    # Warmup
    for _ in range(10):
        _ = ffn_batched(x)
        _ = ffn_sequential(x)
    torch.cuda.synchronize()

    import time

    # Benchmark sequential
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        _ = ffn_sequential(x)
    torch.cuda.synchronize()
    t_seq = (time.perf_counter() - start) / 100

    # Benchmark batched
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        _ = ffn_batched(x)
    torch.cuda.synchronize()
    t_batched = (time.perf_counter() - start) / 100

    print(f"Sequential FFN: {t_seq*1000:.3f} ms")
    print(f"Batched FFN:    {t_batched*1000:.3f} ms")
    print(f"Speedup:        {t_seq/t_batched:.2f}x\n")

    # Profile kernel launches
    print("Profiling kernel launches...")

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        for _ in range(5):
            _ = ffn_sequential(x)
        torch.cuda.synchronize()

    seq_events = prof.key_averages()
    seq_kernel_calls = sum(e.count for e in seq_events if 'nvjet' in e.key.lower() or 'gemm' in e.key.lower())

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        for _ in range(5):
            _ = ffn_batched(x)
        torch.cuda.synchronize()

    batched_events = prof.key_averages()
    batched_kernel_calls = sum(e.count for e in batched_events if 'nvjet' in e.key.lower() or 'gemm' in e.key.lower())

    print(f"Sequential GEMM kernel calls: {seq_kernel_calls}")
    print(f"Batched GEMM kernel calls:    {batched_kernel_calls}")
    print(f"Kernel reduction:             {seq_kernel_calls - batched_kernel_calls}")


if __name__ == "__main__":
    test_batched_vs_sequential()
    test_backward_pass()
    benchmark()
