#!/usr/bin/env python3
"""
Analyze what TorchInductor is and isn't fusing in the weight-shared model.

This script profiles the compiled model to identify:
1. Which operations are being fused by Inductor
2. What head offset adds remain unfused
3. The critical path bottlenecks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


def count_kernels_detailed(prof):
    """Count CUDA kernels by type with detailed breakdown."""
    events = prof.key_averages()
    stats = defaultdict(lambda: {'count': 0, 'time_us': 0.0, 'kernels': []})

    for e in events:
        if e.self_device_time_total == 0:
            continue
        key_lower = e.key.lower()
        time_us = e.self_device_time_total

        # Classify kernel
        if 'nvjet' in key_lower or 'gemm' in key_lower or 'cutlass' in key_lower:
            cat = 'GEMM'
        elif 'addmm' in key_lower:
            cat = 'addmm'
        elif 'triton' in key_lower:
            cat = 'Triton (fused)'
        elif 'fused' in key_lower:
            cat = 'Other fused'
        elif 'memcpy' in key_lower or 'copy' in key_lower:
            cat = 'Memcpy'
        elif 'add' in key_lower:
            cat = 'Elementwise Add'
        elif 'mul' in key_lower:
            cat = 'Elementwise Mul'
        elif 'silu' in key_lower or 'activation' in key_lower:
            cat = 'Activation'
        elif 'rms' in key_lower or 'norm' in key_lower or 'layernorm' in key_lower:
            cat = 'Normalization'
        elif 'softmax' in key_lower:
            cat = 'Softmax'
        elif 'rope' in key_lower or 'rotary' in key_lower:
            cat = 'RoPE'
        else:
            cat = 'Other'

        stats[cat]['count'] += e.count
        stats[cat]['time_us'] += time_us
        stats[cat]['kernels'].append((e.key, e.count, time_us))

    return dict(stats)


def profile_module(module, x, name, warmup=10, iters=3):
    """Profile a module and return kernel breakdown."""
    # Warmup
    for _ in range(warmup):
        _ = module(x)
    torch.cuda.synchronize()

    # Profile
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(iters):
            _ = module(x)
        torch.cuda.synchronize()

    stats = count_kernels_detailed(prof)
    return stats


def print_stats(stats, title):
    """Print kernel statistics."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

    total_time = sum(s['time_us'] for s in stats.values())
    total_count = sum(s['count'] for s in stats.values())

    # Sort by time
    sorted_cats = sorted(stats.items(), key=lambda x: -x[1]['time_us'])

    print(f"\n{'Category':<25} {'Count':>8} {'Time (us)':>12} {'%':>8}")
    print("-" * 55)
    for cat, data in sorted_cats:
        pct = 100 * data['time_us'] / total_time if total_time > 0 else 0
        print(f"{cat:<25} {data['count']:>8} {data['time_us']:>12.1f} {pct:>7.1f}%")
    print("-" * 55)
    print(f"{'TOTAL':<25} {total_count:>8} {total_time:>12.1f}")

    return total_time, total_count


def main():
    from torchtitan.models.qwen3.model.weight_sharing import (
        ScaledLoRAModule,
        SharedLinearWithLoRA,
        SharedFeedForward,
        CombinedSharingAttention,
    )
    from torchtitan.models.qwen3.model.args import Qwen3ModelArgs
    from dataclasses import dataclass

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    # Model config (250M shared)
    dim = 1024
    hidden_dim = 2816
    layer_lora_rank = 256
    head_lora_rank = 192
    n_heads = 16
    n_kv_heads = 8
    head_dim = 64

    batch, seq = 4, 2048
    x = torch.randn(batch, seq, dim, device=device, dtype=dtype)

    print("=" * 60)
    print("TORCH INDUCTOR FUSION ANALYSIS")
    print("=" * 60)
    print(f"Input: batch={batch}, seq={seq}, dim={dim}")
    print(f"Layer LoRA rank: {layer_lora_rank}, Head LoRA rank: {head_lora_rank}")

    # ============================================================
    # Test 1: ScaledLoRAModule (head offset LoRA)
    # ============================================================
    print("\n\n--- Test 1: ScaledLoRAModule (head offset, no fusion opportunity) ---")

    head_lora = ScaledLoRAModule(dim, n_heads * head_dim, rank=head_lora_rank).to(device).to(dtype)

    # Eager
    stats_eager = profile_module(head_lora, x, "ScaledLoRAModule Eager")
    print_stats(stats_eager, "ScaledLoRAModule (Eager)")

    # Compiled
    head_lora_compiled = torch.compile(head_lora)
    stats_compiled = profile_module(head_lora_compiled, x, "ScaledLoRAModule Compiled")
    print_stats(stats_compiled, "ScaledLoRAModule (torch.compile)")

    # ============================================================
    # Test 2: SharedLinearWithLoRA (layer LoRA with addmm)
    # ============================================================
    print("\n\n--- Test 2: SharedLinearWithLoRA (layer LoRA, addmm fusion) ---")

    shared_linear = nn.Linear(dim, hidden_dim, bias=False).to(device).to(dtype)
    layer_lora = SharedLinearWithLoRA(shared_linear, dim, hidden_dim, layer_lora_rank).to(device).to(dtype)

    # Eager
    stats_eager = profile_module(layer_lora, x, "SharedLinearWithLoRA Eager")
    print_stats(stats_eager, "SharedLinearWithLoRA (Eager)")

    # Compiled
    layer_lora_compiled = torch.compile(layer_lora)
    stats_compiled = profile_module(layer_lora_compiled, x, "SharedLinearWithLoRA Compiled")
    print_stats(stats_compiled, "SharedLinearWithLoRA (torch.compile)")

    # ============================================================
    # Test 3: Head offset add pattern
    # ============================================================
    print("\n\n--- Test 3: Head offset add pattern (base + offset) ---")

    class HeadOffsetPattern(nn.Module):
        """Simulates: xq = layer_lora(x) + head_lora(x)"""
        def __init__(self, shared_linear, dim, out_dim, layer_rank, head_rank):
            super().__init__()
            self.layer_lora = SharedLinearWithLoRA(shared_linear, dim, out_dim, layer_rank)
            self.head_lora = ScaledLoRAModule(dim, out_dim, rank=head_rank)

        def forward(self, x):
            base = self.layer_lora(x)  # Uses addmm
            offset = self.head_lora(x)  # 2 matmuls
            return base + offset  # Is this fused?

    shared_wq = nn.Linear(dim, n_heads * head_dim, bias=False).to(device).to(dtype)
    head_pattern = HeadOffsetPattern(shared_wq, dim, n_heads * head_dim, layer_lora_rank, head_lora_rank).to(device).to(dtype)

    # Eager
    stats_eager = profile_module(head_pattern, x, "HeadOffsetPattern Eager")
    t_eager, _ = print_stats(stats_eager, "HeadOffsetPattern (Eager)")

    # Compiled
    head_pattern_compiled = torch.compile(head_pattern)
    stats_compiled = profile_module(head_pattern_compiled, x, "HeadOffsetPattern Compiled")
    t_compiled, _ = print_stats(stats_compiled, "HeadOffsetPattern (torch.compile)")

    print(f"\nSpeedup from compile: {t_eager/t_compiled:.2f}x")

    # ============================================================
    # Test 4: Full attention pattern (Q, K, V with head offsets)
    # ============================================================
    print("\n\n--- Test 4: Full attention pattern (Q, K, V with head offsets) ---")

    class FullAttentionLoRAPattern(nn.Module):
        """Simulates CombinedSharingAttention LoRA operations only."""
        def __init__(self, dim, n_heads, n_kv_heads, head_dim, layer_rank, head_rank):
            super().__init__()
            # Shared weights
            self.wq_shared = nn.Linear(dim, n_heads * head_dim, bias=False)
            self.wk_shared = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
            self.wv_shared = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
            self.wo_shared = nn.Linear(n_heads * head_dim, dim, bias=False)

            # Layer LoRAs
            self.wq = SharedLinearWithLoRA(self.wq_shared, dim, n_heads * head_dim, layer_rank)
            self.wk = SharedLinearWithLoRA(self.wk_shared, dim, n_kv_heads * head_dim, layer_rank)
            self.wv = SharedLinearWithLoRA(self.wv_shared, dim, n_kv_heads * head_dim, layer_rank)
            self.wo = SharedLinearWithLoRA(self.wo_shared, n_heads * head_dim, dim, layer_rank)

            # Head offset LoRAs
            self.wq_head_offset = ScaledLoRAModule(dim, n_heads * head_dim, rank=head_rank)
            self.wk_head_offset = ScaledLoRAModule(dim, n_kv_heads * head_dim, rank=head_rank)
            self.wv_head_offset = ScaledLoRAModule(dim, n_kv_heads * head_dim, rank=head_rank)

        def forward(self, x):
            # Layer projections (addmm fused)
            xq = self.wq(x)
            xk = self.wk(x)
            xv = self.wv(x)

            # Head offsets (NOT fused with addmm - separate matmuls + adds)
            xq = xq + self.wq_head_offset(x)
            xk = xk + self.wk_head_offset(x)
            xv = xv + self.wv_head_offset(x)

            # Output projection
            bsz, seq_len, _ = x.shape
            output = xq.view(bsz, seq_len, -1)  # Simplified
            return self.wo(output)

    attn_pattern = FullAttentionLoRAPattern(
        dim, n_heads, n_kv_heads, head_dim, layer_lora_rank, head_lora_rank
    ).to(device).to(dtype)

    # Eager
    stats_eager = profile_module(attn_pattern, x, "FullAttentionLoRA Eager")
    t_eager, c_eager = print_stats(stats_eager, "FullAttentionLoRA (Eager)")

    # Compiled
    attn_pattern_compiled = torch.compile(attn_pattern)
    stats_compiled = profile_module(attn_pattern_compiled, x, "FullAttentionLoRA Compiled")
    t_compiled, c_compiled = print_stats(stats_compiled, "FullAttentionLoRA (torch.compile)")

    print(f"\nSpeedup from compile: {t_eager/t_compiled:.2f}x")
    print(f"Kernel reduction: {c_eager} -> {c_compiled} ({(1 - c_compiled/c_eager)*100:.1f}% fewer)")

    # ============================================================
    # Test 5: What if we fuse head offsets INTO the layer projection?
    # ============================================================
    print("\n\n--- Test 5: Proposed optimization - fuse head offset into layer LoRA ---")

    class FusedLayerHeadLoRA(nn.Module):
        """
        Proposed: Instead of:
            base = shared(x) + layer_lora(x)  # addmm
            output = base + head_offset(x)    # separate add

        Do:
            combined_lora = layer_lora(x) + head_offset(x)  # Can be fused?
            output = addmm(combined_lora, x, shared.weight.t())
        """
        def __init__(self, shared_linear, dim, out_dim, layer_rank, head_rank):
            super().__init__()
            self.shared_linear = shared_linear
            self.layer_lora = ScaledLoRAModule(dim, out_dim, rank=layer_rank)
            self.head_lora = ScaledLoRAModule(dim, out_dim, rank=head_rank)

        def forward(self, x):
            # Combine both LoRAs first
            combined_lora = self.layer_lora(x) + self.head_lora(x)

            # Fused base + combined LoRA
            x_2d = x.view(-1, x.shape[-1])
            lora_2d = combined_lora.view(-1, combined_lora.shape[-1])
            y_2d = torch.addmm(lora_2d, x_2d, self.shared_linear.weight.t())
            return y_2d.view(*x.shape[:-1], -1)

    shared_wq2 = nn.Linear(dim, n_heads * head_dim, bias=False).to(device).to(dtype)
    fused_pattern = FusedLayerHeadLoRA(shared_wq2, dim, n_heads * head_dim, layer_lora_rank, head_lora_rank).to(device).to(dtype)

    # Eager
    stats_eager = profile_module(fused_pattern, x, "FusedLayerHeadLoRA Eager")
    t_eager_new, c_eager_new = print_stats(stats_eager, "FusedLayerHeadLoRA (Eager)")

    # Compiled
    fused_pattern_compiled = torch.compile(fused_pattern)
    stats_compiled = profile_module(fused_pattern_compiled, x, "FusedLayerHeadLoRA Compiled")
    t_compiled_new, c_compiled_new = print_stats(stats_compiled, "FusedLayerHeadLoRA (torch.compile)")

    print(f"\nSpeedup from compile: {t_eager_new/t_compiled_new:.2f}x")

    # ============================================================
    # Summary
    # ============================================================
    print("\n\n" + "=" * 60)
    print("SUMMARY: Kernel counts per Q projection (eager)")
    print("=" * 60)
    print("""
Current implementation (SharedLinearWithLoRA + separate head_offset):
  Layer LoRA path:  2 matmuls -> 1 addmm  (FUSED)
  Head LoRA path:   2 matmuls + 1 add     (NOT FUSED)
  Total: 3 matmuls + 1 add per Q/K/V (+ shared base inside addmm)

Proposed (combine LoRAs first, then single addmm):
  Combined LoRA:    4 matmuls + 1 add
  Base + combined:  1 addmm
  Total: 5 matmuls (could be batched) + 1 add + 1 addmm

Key insight: TorchInductor may fuse the lora1 + lora2 add, but NOT with addmm!
""")


if __name__ == "__main__":
    main()
