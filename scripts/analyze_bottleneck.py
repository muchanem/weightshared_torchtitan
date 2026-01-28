#!/usr/bin/env python3
"""
Deep analysis: Where is the time actually going?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, "/local/scratch/muchane_680186/weightshared_torchtitan")

from torchtitan.models.qwen3.model.weight_sharing import (
    SharedFeedForward,
    ScaledLoRAModule,
)


def analyze_ffn_breakdown():
    """Break down where time goes in SharedFeedForward."""
    print("=" * 70)
    print("DETAILED TIME BREAKDOWN: SharedFeedForward")
    print("=" * 70)

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

    ffn = SharedFeedForward(
        shared_weights, dim, hidden_dim, rank, use_batched_lora=False
    ).to(device).to(dtype)

    x = torch.randn(batch, seq, dim, device=device, dtype=dtype)

    # Warmup
    for _ in range(5):
        _ = ffn(x)
    torch.cuda.synchronize()

    # Profile with detailed breakdown
    print("\n--- Full FFN Forward ---")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(10):
            _ = ffn(x)
        torch.cuda.synchronize()

    # Categorize kernels
    gemm_time = 0
    elementwise_time = 0
    other_time = 0
    gemm_count = 0
    elementwise_count = 0

    for e in prof.key_averages():
        key_lower = e.key.lower()
        time_us = e.self_device_time_total

        if 'nvjet' in key_lower or 'gemm' in key_lower or 'mm' in key_lower:
            gemm_time += time_us
            gemm_count += e.count
        elif 'elementwise' in key_lower or 'vectorized' in key_lower or 'add' in key_lower or 'mul' in key_lower or 'silu' in key_lower:
            elementwise_time += time_us
            elementwise_count += e.count
        else:
            other_time += time_us

    total_time = gemm_time + elementwise_time + other_time

    print(f"\nTime breakdown (10 iterations):")
    print(f"  GEMM kernels:       {gemm_time/1000:.2f} ms ({gemm_time/total_time*100:.1f}%) - {gemm_count} launches")
    print(f"  Elementwise:        {elementwise_time/1000:.2f} ms ({elementwise_time/total_time*100:.1f}%) - {elementwise_count} launches")
    print(f"  Other:              {other_time/1000:.2f} ms ({other_time/total_time*100:.1f}%)")
    print(f"  TOTAL:              {total_time/1000:.2f} ms")

    # Now let's see the actual kernel table
    print("\nDetailed kernel breakdown:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


def compare_with_without_lora():
    """Compare FFN with and without LoRA to see exact overhead."""
    print("\n" + "=" * 70)
    print("COMPARISON: FFN with LoRA vs without LoRA")
    print("=" * 70)

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    batch, seq, dim = 4, 2048, 1024
    hidden_dim = 2816
    rank = 256

    # FFN WITHOUT LoRA (just the base linear layers)
    w1 = nn.Linear(dim, hidden_dim, bias=False).to(device).to(dtype)
    w2 = nn.Linear(hidden_dim, dim, bias=False).to(device).to(dtype)
    w3 = nn.Linear(dim, hidden_dim, bias=False).to(device).to(dtype)

    def ffn_no_lora(x):
        return w2(F.silu(w1(x)) * w3(x))

    # FFN WITH LoRA
    shared_weights = {"w1": w1, "w2": w2, "w3": w3}
    ffn_with_lora = SharedFeedForward(
        shared_weights, dim, hidden_dim, rank, use_batched_lora=False
    ).to(device).to(dtype)

    x = torch.randn(batch, seq, dim, device=device, dtype=dtype)

    # Warmup
    for _ in range(5):
        _ = ffn_no_lora(x)
        _ = ffn_with_lora(x)
    torch.cuda.synchronize()

    import time

    # Benchmark without LoRA
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        _ = ffn_no_lora(x)
    torch.cuda.synchronize()
    t_no_lora = (time.perf_counter() - start) / 100

    # Benchmark with LoRA
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        _ = ffn_with_lora(x)
    torch.cuda.synchronize()
    t_with_lora = (time.perf_counter() - start) / 100

    print(f"\nTiming (100 iterations):")
    print(f"  FFN without LoRA: {t_no_lora*1000:.3f} ms")
    print(f"  FFN with LoRA:    {t_with_lora*1000:.3f} ms")
    print(f"  LoRA overhead:    {(t_with_lora - t_no_lora)*1000:.3f} ms ({(t_with_lora/t_no_lora - 1)*100:.1f}%)")

    # Profile both
    print("\n--- FFN WITHOUT LoRA ---")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        for _ in range(10):
            _ = ffn_no_lora(x)
        torch.cuda.synchronize()

    events_no_lora = prof.key_averages()
    gemm_no_lora = sum(e.count for e in events_no_lora if 'nvjet' in e.key.lower())
    elem_no_lora = sum(e.count for e in events_no_lora if 'elementwise' in e.key.lower() or 'vectorized' in e.key.lower())

    print(f"  GEMM launches: {gemm_no_lora}")
    print(f"  Elementwise launches: {elem_no_lora}")

    print("\n--- FFN WITH LoRA ---")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        for _ in range(10):
            _ = ffn_with_lora(x)
        torch.cuda.synchronize()

    events_with_lora = prof.key_averages()
    gemm_with_lora = sum(e.count for e in events_with_lora if 'nvjet' in e.key.lower())
    elem_with_lora = sum(e.count for e in events_with_lora if 'elementwise' in e.key.lower() or 'vectorized' in e.key.lower())

    print(f"  GEMM launches: {gemm_with_lora}")
    print(f"  Elementwise launches: {elem_with_lora}")

    print(f"\nDifference from LoRA:")
    print(f"  Extra GEMM launches: {gemm_with_lora - gemm_no_lora}")
    print(f"  Extra elementwise launches: {elem_with_lora - elem_no_lora}")


def analyze_lora_components():
    """Break down LoRA into its component operations."""
    print("\n" + "=" * 70)
    print("LORA COMPONENT BREAKDOWN")
    print("=" * 70)

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    M = 4 * 2048  # batch * seq
    dim = 1024
    out_dim = 2816
    rank = 256

    # Individual operations
    x = torch.randn(M, dim, device=device, dtype=dtype)
    W = torch.randn(out_dim, dim, device=device, dtype=dtype)
    A = torch.randn(rank, dim, device=device, dtype=dtype)
    B = torch.randn(out_dim, rank, device=device, dtype=dtype)

    # Warmup
    for _ in range(5):
        base = x @ W.t()
        lora_h = x @ A.t()
        lora_out = lora_h @ B.t()
        result = base + lora_out
    torch.cuda.synchronize()

    import time

    # Time each component separately
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        base = x @ W.t()
    torch.cuda.synchronize()
    t_base = (time.perf_counter() - start) / 100

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        lora_h = x @ A.t()
    torch.cuda.synchronize()
    t_lora_a = (time.perf_counter() - start) / 100

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        lora_out = lora_h @ B.t()
    torch.cuda.synchronize()
    t_lora_b = (time.perf_counter() - start) / 100

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        result = base + lora_out
    torch.cuda.synchronize()
    t_add = (time.perf_counter() - start) / 100

    total_lora = t_lora_a + t_lora_b + t_add

    print(f"\nPer-operation timing:")
    print(f"  Base matmul (x @ W.T):     {t_base*1000:.3f} ms")
    print(f"  LoRA A matmul (x @ A.T):   {t_lora_a*1000:.3f} ms")
    print(f"  LoRA B matmul (h @ B.T):   {t_lora_b*1000:.3f} ms")
    print(f"  Add (base + lora):         {t_add*1000:.3f} ms")
    print(f"  --------------------------------")
    print(f"  Total LoRA overhead:       {total_lora*1000:.3f} ms ({total_lora/t_base*100:.1f}% of base)")

    # Profile each
    print("\nKernel launches per operation:")

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            base = x @ W.t()
        torch.cuda.synchronize()
    print(f"  Base matmul: {sum(e.count for e in prof.key_averages() if 'nvjet' in e.key.lower())} GEMM kernels")

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            lora_h = x @ A.t()
        torch.cuda.synchronize()
    print(f"  LoRA A:      {sum(e.count for e in prof.key_averages() if 'nvjet' in e.key.lower())} GEMM kernels")

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            lora_out = lora_h @ B.t()
        torch.cuda.synchronize()
    print(f"  LoRA B:      {sum(e.count for e in prof.key_averages() if 'nvjet' in e.key.lower())} GEMM kernels")

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            result = base + lora_out
        torch.cuda.synchronize()
    elem = sum(e.count for e in prof.key_averages() if 'elementwise' in e.key.lower() or 'vectorized' in e.key.lower() or 'add' in e.key.lower())
    print(f"  Add:         {elem} elementwise kernels")


if __name__ == "__main__":
    analyze_ffn_breakdown()
    compare_with_without_lora()
    analyze_lora_components()
