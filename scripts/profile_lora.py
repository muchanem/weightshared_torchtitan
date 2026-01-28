#!/usr/bin/env python3
"""
Profile LoRA operations to understand kernel launches.
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, "/local/scratch/muchane_680186/weightshared_torchtitan")

from torchtitan.models.qwen3.model.weight_sharing import (
    ScaledLoRAModule,
    SharedLinearWithLoRA,
    SharedFeedForward,
)


def profile_sequential_vs_grouped():
    """Profile kernel launches for sequential vs grouped LoRA."""
    print("=" * 60)
    print("Profiling Sequential vs Grouped LoRA")
    print("=" * 60)

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    batch, seq, dim = 4, 2048, 1024
    hidden_dim = 2816
    lora_rank = 256

    # Create shared weights
    shared_weights = {
        "w1": nn.Linear(dim, hidden_dim, bias=False).to(device).to(dtype),
        "w2": nn.Linear(hidden_dim, dim, bias=False).to(device).to(dtype),
        "w3": nn.Linear(dim, hidden_dim, bias=False).to(device).to(dtype),
    }

    ffn_sequential = SharedFeedForward(
        shared_weights, dim, hidden_dim, lora_rank, use_grouped_mm=False
    ).to(device).to(dtype)

    ffn_grouped = SharedFeedForward(
        shared_weights, dim, hidden_dim, lora_rank, use_grouped_mm=True
    ).to(device).to(dtype)

    # Copy weights
    ffn_grouped.w1.lora.w_a.weight.data.copy_(ffn_sequential.w1.lora.w_a.weight.data)
    ffn_grouped.w1.lora.w_b.weight.data.copy_(ffn_sequential.w1.lora.w_b.weight.data)
    ffn_grouped.w2.lora.w_a.weight.data.copy_(ffn_sequential.w2.lora.w_a.weight.data)
    ffn_grouped.w2.lora.w_b.weight.data.copy_(ffn_sequential.w2.lora.w_b.weight.data)
    ffn_grouped.w3.lora.w_a.weight.data.copy_(ffn_sequential.w3.lora.w_a.weight.data)
    ffn_grouped.w3.lora.w_b.weight.data.copy_(ffn_sequential.w3.lora.w_b.weight.data)

    x = torch.randn(batch, seq, dim, device=device, dtype=dtype)

    # Warmup
    for _ in range(3):
        _ = ffn_sequential(x)
        _ = ffn_grouped(x)
    torch.cuda.synchronize()

    # Profile sequential
    print("\n--- Sequential FFN (use_grouped_mm=False) ---")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for _ in range(5):
            _ = ffn_sequential(x)
        torch.cuda.synchronize()

    # Print kernel summary
    events = prof.key_averages()
    print(f"Total CUDA events: {len(events)}")

    # Filter for gemm/matmul kernels
    gemm_events = [e for e in events if 'gemm' in e.key.lower() or 'mm' in e.key.lower() or 'matmul' in e.key.lower() or 'nvjet' in e.key.lower()]
    print(f"GEMM-related events: {len(gemm_events)}")
    for e in sorted(gemm_events, key=lambda x: -x.self_device_time_total):
        print(f"  {e.key}: count={e.count}, cuda_time={e.self_device_time_total/1000:.3f}ms")

    # Print full table for top kernels
    print("\nTop 10 kernels by CUDA time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Profile grouped
    print("\n--- Grouped FFN (use_grouped_mm=True) ---")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for _ in range(5):
            _ = ffn_grouped(x)
        torch.cuda.synchronize()

    events = prof.key_averages()
    print(f"Total CUDA events: {len(events)}")

    gemm_events = [e for e in events if 'gemm' in e.key.lower() or 'mm' in e.key.lower() or 'matmul' in e.key.lower() or 'nvjet' in e.key.lower()]
    print(f"GEMM-related events: {len(gemm_events)}")
    for e in sorted(gemm_events, key=lambda x: -x.self_device_time_total):
        print(f"  {e.key}: count={e.count}, cuda_time={e.self_device_time_total/1000:.3f}ms")

    print("\nTop 10 kernels by CUDA time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


def profile_isolated_lora():
    """Profile isolated LoRA module calls."""
    print("\n" + "=" * 60)
    print("Profiling Isolated LoRA Operations")
    print("=" * 60)

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    batch, seq, dim = 4, 2048, 1024
    rank = 256
    out_dim = 2816

    # Create 3 LoRA modules (like w1, w3, and another)
    loras = [
        ScaledLoRAModule(dim, out_dim, rank=rank).to(device).to(dtype)
        for _ in range(3)
    ]

    x = torch.randn(batch, seq, dim, device=device, dtype=dtype)

    # Warmup
    for _ in range(3):
        for lora in loras:
            _ = lora(x)
    torch.cuda.synchronize()

    # Profile sequential LoRA calls
    print("\n--- 3 Sequential LoRA calls ---")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(5):
            for lora in loras:
                _ = lora(x)
        torch.cuda.synchronize()

    events = prof.key_averages()
    gemm_events = [e for e in events if 'gemm' in e.key.lower() or 'mm' in e.key.lower()]
    print(f"Total CUDA events: {len(events)}, GEMM events: {len(gemm_events)}")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Profile BMM approach
    print("\n--- BMM Batched LoRA ---")
    x_flat = x.view(-1, dim)
    w_a_stack = torch.stack([l.w_a.weight for l in loras], dim=0)  # (3, rank, dim)
    w_b_stack = torch.stack([l.w_b.weight for l in loras], dim=0)  # (3, out_dim, rank)
    x_expanded = x_flat.unsqueeze(0).expand(3, -1, -1)  # (3, M, dim)

    # Warmup
    for _ in range(3):
        h = torch.bmm(x_expanded.bfloat16(), w_a_stack.bfloat16().transpose(-2, -1))
        out = torch.bmm(h, w_b_stack.bfloat16().transpose(-2, -1))
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(5):
            h = torch.bmm(x_expanded.bfloat16(), w_a_stack.bfloat16().transpose(-2, -1))
            out = torch.bmm(h, w_b_stack.bfloat16().transpose(-2, -1))
        torch.cuda.synchronize()

    events = prof.key_averages()
    gemm_events = [e for e in events if 'gemm' in e.key.lower() or 'mm' in e.key.lower()]
    print(f"Total CUDA events: {len(events)}, GEMM events: {len(gemm_events)}")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


if __name__ == "__main__":
    profile_sequential_vs_grouped()
    profile_isolated_lora()
