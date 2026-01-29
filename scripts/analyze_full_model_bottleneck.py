#!/usr/bin/env python3
"""
Analyze the full weight-shared model to identify the true bottleneck.

This script:
1. Profiles a full CombinedSharingTransformerBlock
2. Compares eager vs compiled
3. Counts operations per layer
4. Identifies what's NOT being fused
"""

import torch
import torch.nn as nn
from collections import defaultdict


def count_kernels_by_type(prof):
    """Detailed kernel breakdown."""
    events = prof.key_averages()
    stats = defaultdict(lambda: {'count': 0, 'time_us': 0.0, 'examples': []})

    for e in events:
        if e.self_device_time_total == 0:
            continue
        key_lower = e.key.lower()
        time_us = e.self_device_time_total

        # Classify
        if 'nvjet' in key_lower or 'gemm' in key_lower or 'cutlass' in key_lower:
            if '_badd_' in key_lower:
                cat = 'GEMM+add (fused)'
            elif '_bz_' in key_lower:
                cat = 'GEMM (beta=0)'
            else:
                cat = 'GEMM (other)'
        elif 'addmm' in key_lower:
            cat = 'addmm (fused)'
        elif 'triton' in key_lower:
            cat = 'Triton (fused)'
        elif 'memcpy' in key_lower or 'copy' in key_lower:
            cat = 'Memcpy'
        elif 'add' in key_lower:
            cat = 'Add (unfused)'
        elif 'mul' in key_lower:
            cat = 'Mul (unfused)'
        elif 'silu' in key_lower or 'gelu' in key_lower:
            cat = 'Activation'
        elif 'rms' in key_lower or 'norm' in key_lower or 'layernorm' in key_lower:
            cat = 'Normalization'
        elif 'softmax' in key_lower or 'flash' in key_lower or 'attention' in key_lower:
            cat = 'Attention core'
        elif 'rope' in key_lower or 'rotary' in key_lower:
            cat = 'RoPE'
        elif 'transpose' in key_lower or 'permute' in key_lower or 'view' in key_lower:
            cat = 'Reshape'
        else:
            cat = 'Other'

        stats[cat]['count'] += e.count
        stats[cat]['time_us'] += time_us
        if len(stats[cat]['examples']) < 3:
            stats[cat]['examples'].append((e.key[:60], time_us / e.count if e.count > 0 else 0))

    return dict(stats)


def print_detailed_stats(stats, title):
    """Print kernel statistics with examples."""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")

    total_time = sum(s['time_us'] for s in stats.values())
    total_count = sum(s['count'] for s in stats.values())

    sorted_cats = sorted(stats.items(), key=lambda x: -x[1]['time_us'])

    print(f"\n{'Category':<25} {'Count':>8} {'Time (us)':>12} {'%':>8} {'Avg (us)':>10}")
    print("-" * 65)
    for cat, data in sorted_cats:
        pct = 100 * data['time_us'] / total_time if total_time > 0 else 0
        avg = data['time_us'] / data['count'] if data['count'] > 0 else 0
        print(f"{cat:<25} {data['count']:>8} {data['time_us']:>12.1f} {pct:>7.1f}% {avg:>9.2f}")

    print("-" * 65)
    print(f"{'TOTAL':<25} {total_count:>8} {total_time:>12.1f}")

    # Show example kernels for top categories
    print("\nTop kernel examples:")
    for cat, data in sorted_cats[:5]:
        if data['examples']:
            print(f"  {cat}:")
            for name, avg_time in data['examples'][:2]:
                print(f"    {name}: {avg_time:.2f} us")

    return total_time, total_count


def main():
    import sys
    sys.path.insert(0, '/local/scratch/muchane_681547/weightshared_torchtitan')

    from torchtitan.models.qwen3.model.weight_sharing import (
        CombinedSharingTransformerBlock,
    )
    from torchtitan.models.qwen3.model.args import Qwen3ModelArgs

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    # Model args matching 250M_shared config
    args = Qwen3ModelArgs(
        dim=1024,
        n_layers=20,
        n_heads=16,
        n_kv_heads=8,
        head_dim=64,
        hidden_dim=3072,
        vocab_size=151936,
        max_seq_len=2048,
        norm_eps=1e-6,
        rope_theta=500000.0,
        qk_norm=True,
        attn_type="sdpa",
    )

    # Attention config
    class AttentionConfig:
        head_sharing = True
        grouping = 1
        rank = 192
        two_step = False
        qkv_sharing = None

    attention_config = AttentionConfig()
    layer_lora_rank = 256

    batch, seq = 4, 2048

    print("=" * 70)
    print("FULL TRANSFORMER BLOCK ANALYSIS")
    print("=" * 70)
    print(f"Config: dim={args.dim}, hidden_dim={args.hidden_dim}")
    print(f"Heads: n_heads={args.n_heads}, n_kv_heads={args.n_kv_heads}")
    print(f"LoRA: layer_rank={layer_lora_rank}, head_rank={attention_config.rank}")
    print(f"Input: batch={batch}, seq={seq}")

    # Create shared weights
    shared_attention_weights = {
        "wq": nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False),
        "wk": nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False),
        "wv": nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False),
        "wo": nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False),
    }
    shared_ffn_weights = {
        "w1": nn.Linear(args.dim, args.hidden_dim, bias=False),
        "w2": nn.Linear(args.hidden_dim, args.dim, bias=False),
        "w3": nn.Linear(args.dim, args.hidden_dim, bias=False),
    }

    # Create block
    block = CombinedSharingTransformerBlock(
        layer_id=0,
        model_args=args,
        shared_attention_weights=shared_attention_weights,
        shared_ffn_weights=shared_ffn_weights,
        layer_lora_rank=layer_lora_rank,
        attention_config=attention_config,
    ).to(device).to(dtype)

    # Initialize weights
    for name, module in block.named_modules():
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()

    # Prepare inputs
    x = torch.randn(batch, seq, args.dim, device=device, dtype=dtype)
    # rope_cache shape is (seqlen, head_dim * 2) where the last dim is concat of cos and sin
    rope_cache = torch.randn(seq, args.head_dim * 2, device=device, dtype=dtype)

    # Import RoPE functions
    from torchtitan.models.qwen3.model.model import apply_rotary_emb, repeat_kv

    def forward_fn():
        return block(x, rope_cache, None, None, apply_rotary_emb, repeat_kv)

    print("\n--- Profiling Single Block ---")

    # Warmup
    for _ in range(10):
        _ = forward_fn()
    torch.cuda.synchronize()

    # Eager profile
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            _ = forward_fn()
        torch.cuda.synchronize()

    stats_eager = count_kernels_by_type(prof)
    t_eager, c_eager = print_detailed_stats(stats_eager, "CombinedSharingTransformerBlock (Eager)")

    # Compiled profile
    block_compiled = torch.compile(block)

    # Warmup compiled
    for _ in range(10):
        _ = block_compiled(x, rope_cache, None, None, apply_rotary_emb, repeat_kv)
    torch.cuda.synchronize()

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            _ = block_compiled(x, rope_cache, None, None, apply_rotary_emb, repeat_kv)
        torch.cuda.synchronize()

    stats_compiled = count_kernels_by_type(prof)
    t_compiled, c_compiled = print_detailed_stats(stats_compiled, "CombinedSharingTransformerBlock (torch.compile)")

    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\nTime: {t_eager:.1f} us (eager) -> {t_compiled:.1f} us (compiled)")
    print(f"Speedup: {t_eager/t_compiled:.2f}x")
    print(f"Kernels: {c_eager} (eager) -> {c_compiled} (compiled)")
    print(f"Reduction: {(1 - c_compiled/c_eager)*100:.1f}%")

    # Count GEMM-related operations
    gemm_eager = sum(v['count'] for k, v in stats_eager.items() if 'GEMM' in k or 'addmm' in k)
    gemm_compiled = sum(v['count'] for k, v in stats_compiled.items() if 'GEMM' in k or 'addmm' in k)
    add_eager = sum(v['count'] for k, v in stats_eager.items() if 'Add' in k)
    add_compiled = sum(v['count'] for k, v in stats_compiled.items() if 'Add' in k)

    print(f"\nGEMM-related kernels: {gemm_eager} (eager) -> {gemm_compiled} (compiled)")
    print(f"Unfused Add kernels: {add_eager} (eager) -> {add_compiled} (compiled)")

    # ============================================================
    # Analyze what Inductor is NOT fusing
    # ============================================================
    print("\n" + "=" * 70)
    print("ANALYSIS: What's NOT being fused?")
    print("=" * 70)

    # Look for patterns in compiled stats
    for cat, data in sorted(stats_compiled.items(), key=lambda x: -x[1]['time_us']):
        if 'unfused' in cat.lower() or data['count'] > 0:
            print(f"\n{cat}: {data['count']} kernels, {data['time_us']:.1f} us")
            for name, avg in data['examples']:
                print(f"  Example: {name}")


if __name__ == "__main__":
    main()
