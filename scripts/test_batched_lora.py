#!/usr/bin/env python3
"""
Test and benchmark the batched LoRA implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, "/local/scratch/muchane_680186/weightshared_torchtitan")

from torchtitan.models.qwen3.model.batched_lora import (
    BatchedLoRAModule,
    BatchedFFNLoRA,
)
from torchtitan.models.qwen3.model.weight_sharing import (
    ScaledLoRAModule,
    SharedLinearWithLoRA,
    SharedFeedForward,
)


def test_batched_lora_correctness():
    """Test that BatchedLoRAModule produces same results as individual modules."""
    print("=" * 60)
    print("Testing BatchedLoRAModule correctness")
    print("=" * 60)

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    batch, seq, dim = 4, 128, 256
    out_dim = 512
    rank = 32
    n_loras = 2

    # Create individual LoRA modules
    lora1 = ScaledLoRAModule(dim, out_dim, rank=rank).to(device).to(dtype)
    lora2 = ScaledLoRAModule(dim, out_dim, rank=rank).to(device).to(dtype)

    # Create batched module
    batched = BatchedLoRAModule(n_loras, dim, out_dim, rank).to(device).to(dtype)

    # Copy weights from individual to batched
    with torch.no_grad():
        batched.w_a[0].copy_(lora1.w_a.weight)
        batched.w_a[1].copy_(lora2.w_a.weight)
        batched.w_b[0].copy_(lora1.w_b.weight)
        batched.w_b[1].copy_(lora2.w_b.weight)

    # Test forward
    x = torch.randn(batch * seq, dim, device=device, dtype=dtype)

    out1 = lora1(x)
    out2 = lora2(x)
    out_batched = batched(x)

    # Compare
    diff1 = (out1 - out_batched[0]).abs().max().item()
    diff2 = (out2 - out_batched[1]).abs().max().item()

    print(f"Max diff lora1 vs batched[0]: {diff1:.6f}")
    print(f"Max diff lora2 vs batched[1]: {diff2:.6f}")

    assert diff1 < 1e-3, f"LoRA 1 mismatch: {diff1}"
    assert diff2 < 1e-3, f"LoRA 2 mismatch: {diff2}"
    print("PASSED: BatchedLoRAModule correctness\n")


def test_batched_ffn_correctness():
    """Test that BatchedFFNLoRA produces same results as SharedFeedForward."""
    print("=" * 60)
    print("Testing BatchedFFNLoRA correctness")
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

    # Create sequential FFN (baseline)
    ffn_seq = SharedFeedForward(
        shared_weights, dim, hidden_dim, rank, use_grouped_mm=False
    ).to(device).to(dtype)

    # Create batched FFN
    ffn_batched = BatchedFFNLoRA(
        shared_weights["w1"],
        shared_weights["w2"],
        shared_weights["w3"],
        dim, hidden_dim, rank,
    ).to(device).to(dtype)

    # Copy weights from sequential to batched
    with torch.no_grad():
        # w1 LoRA
        ffn_batched.w13_lora.w_a[0].copy_(ffn_seq.w1.lora.w_a.weight)
        ffn_batched.w13_lora.w_b[0].copy_(ffn_seq.w1.lora.w_b.weight)
        # w3 LoRA
        ffn_batched.w13_lora.w_a[1].copy_(ffn_seq.w3.lora.w_a.weight)
        ffn_batched.w13_lora.w_b[1].copy_(ffn_seq.w3.lora.w_b.weight)
        # w2 LoRA
        ffn_batched.w2_lora_a.weight.copy_(ffn_seq.w2.lora.w_a.weight)
        ffn_batched.w2_lora_b.weight.copy_(ffn_seq.w2.lora.w_b.weight)

    # Test forward
    x = torch.randn(batch, seq, dim, device=device, dtype=dtype)

    out_seq = ffn_seq(x)
    out_batched = ffn_batched(x)

    diff = (out_seq - out_batched).abs().max().item()
    print(f"Max diff: {diff:.6f}")

    assert diff < 1e-2, f"FFN mismatch: {diff}"
    print("PASSED: BatchedFFNLoRA correctness\n")


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

    ffn_batched = BatchedFFNLoRA(
        shared_weights["w1"],
        shared_weights["w2"],
        shared_weights["w3"],
        dim, hidden_dim, rank,
    ).to(device).to(dtype)

    x = torch.randn(batch, seq, dim, device=device, dtype=dtype, requires_grad=True)

    out = ffn_batched(x)
    loss = out.sum()
    loss.backward()

    # Check gradients exist
    assert x.grad is not None, "No gradient for input"
    assert ffn_batched.w13_lora.w_a.grad is not None, "No gradient for w13_lora.w_a"
    assert ffn_batched.w13_lora.w_b.grad is not None, "No gradient for w13_lora.w_b"
    assert ffn_batched.w2_lora_a.weight.grad is not None, "No gradient for w2_lora_a"
    assert ffn_batched.w2_lora_b.weight.grad is not None, "No gradient for w2_lora_b"

    print("All gradients computed successfully")
    print("PASSED: Backward pass\n")


def benchmark_ffn():
    """Benchmark sequential vs batched FFN."""
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

    ffn_seq = SharedFeedForward(
        shared_weights, dim, hidden_dim, rank, use_grouped_mm=False
    ).to(device).to(dtype)

    ffn_batched = BatchedFFNLoRA(
        shared_weights["w1"],
        shared_weights["w2"],
        shared_weights["w3"],
        dim, hidden_dim, rank,
    ).to(device).to(dtype)

    x = torch.randn(batch, seq, dim, device=device, dtype=dtype)

    # Warmup
    for _ in range(10):
        _ = ffn_seq(x)
        _ = ffn_batched(x)
    torch.cuda.synchronize()

    import time

    # Benchmark sequential
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        _ = ffn_seq(x)
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
            _ = ffn_seq(x)
        torch.cuda.synchronize()

    events_seq = len(prof.key_averages())

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        for _ in range(5):
            _ = ffn_batched(x)
        torch.cuda.synchronize()

    events_batched = len(prof.key_averages())

    print(f"Sequential CUDA events: {events_seq}")
    print(f"Batched CUDA events:    {events_batched}")
    print(f"Event reduction:        {events_seq - events_batched}")


def profile_detailed():
    """Profile detailed kernel breakdown."""
    print("\n" + "=" * 60)
    print("Detailed kernel profiling")
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

    ffn_seq = SharedFeedForward(
        shared_weights, dim, hidden_dim, rank, use_grouped_mm=False
    ).to(device).to(dtype)

    ffn_batched = BatchedFFNLoRA(
        shared_weights["w1"],
        shared_weights["w2"],
        shared_weights["w3"],
        dim, hidden_dim, rank,
    ).to(device).to(dtype)

    x = torch.randn(batch, seq, dim, device=device, dtype=dtype)

    # Warmup
    for _ in range(3):
        _ = ffn_seq(x)
        _ = ffn_batched(x)
    torch.cuda.synchronize()

    print("\n--- Sequential FFN ---")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(5):
            _ = ffn_seq(x)
        torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    print("\n--- Batched FFN ---")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(5):
            _ = ffn_batched(x)
        torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))


if __name__ == "__main__":
    test_batched_lora_correctness()
    test_batched_ffn_correctness()
    test_backward_pass()
    benchmark_ffn()
    profile_detailed()
