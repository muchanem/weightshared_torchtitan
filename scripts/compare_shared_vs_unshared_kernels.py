#!/usr/bin/env python3
"""
Compare kernel counts between shared and unshared transformer blocks.

This identifies exactly where the overhead comes from.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


def count_kernels_by_type(prof):
    """Detailed kernel breakdown."""
    events = prof.key_averages()
    stats = defaultdict(lambda: {'count': 0, 'time_us': 0.0})

    for e in events:
        if e.self_device_time_total == 0:
            continue
        key_lower = e.key.lower()
        time_us = e.self_device_time_total

        if 'nvjet' in key_lower or 'gemm' in key_lower or 'cutlass' in key_lower:
            if '_badd_' in key_lower:
                cat = 'GEMM+add'
            elif '_bz_' in key_lower:
                cat = 'GEMM'
            else:
                cat = 'GEMM'
        elif 'addmm' in key_lower:
            cat = 'addmm'
        elif 'triton' in key_lower:
            cat = 'Triton'
        elif 'memcpy' in key_lower or 'copy' in key_lower:
            cat = 'Memcpy'
        elif 'add' in key_lower:
            cat = 'Add'
        elif 'mul' in key_lower:
            cat = 'Mul'
        elif 'rms' in key_lower or 'norm' in key_lower:
            cat = 'Norm'
        elif 'softmax' in key_lower or 'flash' in key_lower or 'attention' in key_lower:
            cat = 'Attn'
        else:
            cat = 'Other'

        stats[cat]['count'] += e.count
        stats[cat]['time_us'] += time_us

    return dict(stats)


def profile_model(model, x, rope_cache, name, is_shared=False, warmup=10, iters=5):
    """Profile a model and return kernel stats."""
    from torchtitan.models.qwen3.model.model import apply_rotary_emb, repeat_kv

    def forward_fn():
        if is_shared:
            # CombinedSharingTransformerBlock signature includes apply_rotary_emb, repeat_kv
            return model(x, rope_cache, None, None, apply_rotary_emb, repeat_kv)
        else:
            # TransformerBlock doesn't take those extra args
            return model(x, rope_cache, None, None)

    # Warmup
    for _ in range(warmup):
        _ = forward_fn()
    torch.cuda.synchronize()

    # Profile
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            _ = forward_fn()
        torch.cuda.synchronize()

    return count_kernels_by_type(prof)


def main():
    import sys
    sys.path.insert(0, '/local/scratch/muchane_681547/weightshared_torchtitan')

    from torchtitan.models.qwen3.model.weight_sharing import CombinedSharingTransformerBlock
    from torchtitan.models.qwen3.model.model import TransformerBlock
    from torchtitan.models.qwen3.model.args import Qwen3ModelArgs

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    # Model args - same for both
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

    batch, seq = 4, 2048
    x = torch.randn(batch, seq, args.dim, device=device, dtype=dtype)
    rope_cache = torch.randn(seq, args.head_dim * 2, device=device, dtype=dtype)

    print("=" * 70)
    print("SHARED vs UNSHARED KERNEL COMPARISON")
    print("=" * 70)

    # ============================================================
    # Create unshared block (baseline)
    # ============================================================
    unshared_block = TransformerBlock(
        layer_id=0,
        model_args=args,
    ).to(device).to(dtype)
    unshared_block.init_weights(device)

    # ============================================================
    # Create shared block
    # ============================================================
    class AttentionConfig:
        head_sharing = True
        grouping = 1
        rank = 192
        two_step = False
        qkv_sharing = None

    shared_attention_weights = {
        "wq": nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False).to(device).to(dtype),
        "wk": nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False).to(device).to(dtype),
        "wv": nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False).to(device).to(dtype),
        "wo": nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False).to(device).to(dtype),
    }
    shared_ffn_weights = {
        "w1": nn.Linear(args.dim, args.hidden_dim, bias=False).to(device).to(dtype),
        "w2": nn.Linear(args.hidden_dim, args.dim, bias=False).to(device).to(dtype),
        "w3": nn.Linear(args.dim, args.hidden_dim, bias=False).to(device).to(dtype),
    }

    shared_block = CombinedSharingTransformerBlock(
        layer_id=0,
        model_args=args,
        shared_attention_weights=shared_attention_weights,
        shared_ffn_weights=shared_ffn_weights,
        layer_lora_rank=256,
        attention_config=AttentionConfig(),
    ).to(device).to(dtype)
    shared_block.init_weights(device)

    # ============================================================
    # Profile both (eager mode)
    # ============================================================
    print("\n--- EAGER MODE ---")

    stats_unshared = profile_model(unshared_block, x, rope_cache, "Unshared", is_shared=False)
    stats_shared = profile_model(shared_block, x, rope_cache, "Shared", is_shared=True)

    print(f"\n{'Category':<12} {'Unshared':>12} {'Shared':>12} {'Delta':>10} {'Delta %':>10}")
    print("-" * 60)

    all_cats = set(stats_unshared.keys()) | set(stats_shared.keys())
    for cat in sorted(all_cats):
        u = stats_unshared.get(cat, {'count': 0, 'time_us': 0})
        s = stats_shared.get(cat, {'count': 0, 'time_us': 0})
        delta = s['count'] - u['count']
        delta_pct = (delta / u['count'] * 100) if u['count'] > 0 else float('inf')
        print(f"{cat:<12} {u['count']:>8} {s['count']:>12} {delta:>+10} {delta_pct:>+9.0f}%")

    t_unshared = sum(v['time_us'] for v in stats_unshared.values())
    t_shared = sum(v['time_us'] for v in stats_shared.values())
    c_unshared = sum(v['count'] for v in stats_unshared.values())
    c_shared = sum(v['count'] for v in stats_shared.values())

    print("-" * 60)
    print(f"{'TOTAL':<12} {c_unshared:>8} {c_shared:>12} {c_shared-c_unshared:>+10}")
    print(f"{'Time (us)':<12} {t_unshared:>8.0f} {t_shared:>12.0f} {t_shared-t_unshared:>+10.0f}")
    print(f"\nOverhead: {t_shared/t_unshared:.2f}x ({(t_shared/t_unshared-1)*100:.0f}%)")

    # ============================================================
    # Profile both (compiled)
    # ============================================================
    print("\n\n--- COMPILED MODE ---")

    unshared_compiled = torch.compile(unshared_block)
    shared_compiled = torch.compile(shared_block)

    stats_unshared_c = profile_model(unshared_compiled, x, rope_cache, "Unshared Compiled", is_shared=False)
    stats_shared_c = profile_model(shared_compiled, x, rope_cache, "Shared Compiled", is_shared=True)

    print(f"\n{'Category':<12} {'Unshared':>12} {'Shared':>12} {'Delta':>10} {'Delta %':>10}")
    print("-" * 60)

    all_cats = set(stats_unshared_c.keys()) | set(stats_shared_c.keys())
    for cat in sorted(all_cats):
        u = stats_unshared_c.get(cat, {'count': 0, 'time_us': 0})
        s = stats_shared_c.get(cat, {'count': 0, 'time_us': 0})
        delta = s['count'] - u['count']
        delta_pct = (delta / u['count'] * 100) if u['count'] > 0 else float('inf')
        print(f"{cat:<12} {u['count']:>8} {s['count']:>12} {delta:>+10} {delta_pct:>+9.0f}%")

    t_unshared_c = sum(v['time_us'] for v in stats_unshared_c.values())
    t_shared_c = sum(v['time_us'] for v in stats_shared_c.values())
    c_unshared_c = sum(v['count'] for v in stats_unshared_c.values())
    c_shared_c = sum(v['count'] for v in stats_shared_c.values())

    print("-" * 60)
    print(f"{'TOTAL':<12} {c_unshared_c:>8} {c_shared_c:>12} {c_shared_c-c_unshared_c:>+10}")
    print(f"{'Time (us)':<12} {t_unshared_c:>8.0f} {t_shared_c:>12.0f} {t_shared_c-t_unshared_c:>+10.0f}")
    print(f"\nOverhead: {t_shared_c/t_unshared_c:.2f}x ({(t_shared_c/t_unshared_c-1)*100:.0f}%)")

    # ============================================================
    # Detailed analysis
    # ============================================================
    print("\n\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Calculate expected GEMM count difference
    # Unshared: 4 attention projections + 3 FFN = 7 base GEMMs per layer
    # Shared:
    #   - 4 attention projections (base + layer LoRA) = 4 addmm + 8 LoRA matmuls
    #   - 3 head offset LoRAs = 6 matmuls
    #   - 3 FFN (base + layer LoRA) = 3 addmm + 6 LoRA matmuls
    # Total additional: 8 + 6 + 6 = 20 extra matmuls from LoRA

    print("""
Expected LoRA overhead per transformer block:
  Layer LoRA (attention): 4 × 2 matmuls = 8 matmuls
  Head offset LoRA:       3 × 2 matmuls = 6 matmuls
  Layer LoRA (FFN):       3 × 2 matmuls = 6 matmuls
  -------------------------------------------
  Total extra matmuls:    20 matmuls per block per forward pass

For 5 forward passes: 100 extra GEMM kernels expected
""")

    gemm_delta = (
        stats_shared.get('GEMM', {'count': 0})['count'] +
        stats_shared.get('GEMM+add', {'count': 0})['count'] -
        stats_unshared.get('GEMM', {'count': 0})['count'] -
        stats_unshared.get('GEMM+add', {'count': 0})['count']
    )
    print(f"Actual GEMM kernel delta (eager): {gemm_delta}")

    gemm_delta_c = (
        stats_shared_c.get('GEMM', {'count': 0})['count'] +
        stats_shared_c.get('GEMM+add', {'count': 0})['count'] +
        stats_shared_c.get('addmm', {'count': 0})['count'] -
        stats_unshared_c.get('GEMM', {'count': 0})['count'] -
        stats_unshared_c.get('GEMM+add', {'count': 0})['count'] -
        stats_unshared_c.get('addmm', {'count': 0})['count']
    )
    print(f"Actual GEMM kernel delta (compiled): {gemm_delta_c}")

    print("""
CONCLUSION:
-----------
The weight-shared model has significantly more GEMM kernels due to LoRA.
TorchInductor already fuses the elementwise adds, so manual torch.addmm
optimization provides minimal benefit.

The remaining optimization opportunity is to BATCH the LoRA matmuls:
- Instead of N separate small matmuls, combine them into fewer larger matmuls
- This requires restructuring how LoRA is applied
""")


if __name__ == "__main__":
    main()
