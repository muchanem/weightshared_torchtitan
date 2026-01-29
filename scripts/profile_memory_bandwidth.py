#!/usr/bin/env python3
"""
Profile memory bandwidth and arithmetic intensity for LoRA operations.

This script investigates:
1. Memory bandwidth utilization during LoRA ops
2. Arithmetic intensity (FLOPs per byte)
3. Which operations are memory-bound vs compute-bound
4. Intermediate tensor memory traffic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import math


def calculate_arithmetic_intensity(M, N, K, dtype=torch.bfloat16):
    """
    Calculate arithmetic intensity for a matmul: (M, K) @ (K, N) -> (M, N)

    FLOPs = 2 * M * N * K (multiply-add)
    Bytes = (M*K + K*N + M*N) * bytes_per_element

    For memory-bound ops, AI < GPU's ridge point (~100 FLOPs/byte for H200)
    """
    bytes_per_elem = 2 if dtype == torch.bfloat16 else 4

    flops = 2 * M * N * K
    bytes_read = (M * K + K * N) * bytes_per_elem  # Input matrices
    bytes_write = M * N * bytes_per_elem  # Output matrix
    total_bytes = bytes_read + bytes_write

    ai = flops / total_bytes
    return ai, flops, total_bytes


def profile_with_memory_metrics(fn, name, warmup=10, iters=5):
    """Profile a function and extract memory-related metrics."""

    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Profile with detailed metrics
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()

    return prof


def analyze_kernel_details(prof):
    """Extract detailed kernel information."""
    events = prof.key_averages()

    kernels = []
    for e in events:
        if e.self_device_time_total == 0:
            continue

        kernel_info = {
            'name': e.key,
            'count': e.count,
            'time_us': e.self_device_time_total,
            'avg_us': e.self_device_time_total / e.count if e.count > 0 else 0,
            'flops': getattr(e, 'flops', 0) or 0,
            'self_cuda_memory_usage': getattr(e, 'self_cuda_memory_usage', 0) or 0,
        }

        # Classify kernel type
        key_lower = e.key.lower()
        if '_badd_' in key_lower:
            kernel_info['fusion_type'] = 'GEMM+epilogue_add'
        elif '_bz_' in key_lower:
            kernel_info['fusion_type'] = 'GEMM_only'
        elif 'triton_poi_fused' in key_lower and 'add' in key_lower:
            kernel_info['fusion_type'] = 'Triton_elementwise_add'
        elif 'triton' in key_lower:
            kernel_info['fusion_type'] = 'Triton_other'
        elif 'addmm' in key_lower:
            kernel_info['fusion_type'] = 'cuBLAS_addmm'
        elif 'nvjet' in key_lower or 'gemm' in key_lower:
            kernel_info['fusion_type'] = 'GEMM_other'
        else:
            kernel_info['fusion_type'] = 'other'

        kernels.append(kernel_info)

    return kernels


def main():
    import sys
    sys.path.insert(0, '/local/scratch/muchane_681547/weightshared_torchtitan')

    from torchtitan.models.qwen3.model.weight_sharing import (
        ScaledLoRAModule,
        SharedLinearWithLoRA,
    )

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    # Config
    batch, seq = 4, 2048
    M = batch * seq  # 8192
    dim = 1024
    hidden_dim = 2816
    layer_lora_rank = 256
    head_lora_rank = 192
    n_heads = 16
    head_dim = 64

    print("=" * 70)
    print("MEMORY BANDWIDTH AND ARITHMETIC INTENSITY ANALYSIS")
    print("=" * 70)

    # ============================================================
    # Part 1: Theoretical Arithmetic Intensity Analysis
    # ============================================================
    print("\n" + "=" * 70)
    print("PART 1: Theoretical Arithmetic Intensity")
    print("=" * 70)
    print("\nH200 NVL specs:")
    print("  - Memory bandwidth: ~4.8 TB/s")
    print("  - Peak FP16 TFLOPS: ~1979 TFLOPS")
    print("  - Ridge point (AI threshold): ~412 FLOPs/byte")
    print("  - Operations with AI < 412 are memory-bound")

    print("\n--- Base Linear Operations ---")

    ops = [
        ("Base wq", M, n_heads * head_dim, dim),  # (8192, 1024) @ (1024, 1024)
        ("Base w1", M, hidden_dim, dim),  # (8192, 1024) @ (1024, 2816)
        ("Base w2", M, dim, hidden_dim),  # (8192, 2816) @ (2816, 1024)
    ]

    print(f"\n{'Operation':<20} {'Shape':<30} {'AI (FLOPs/B)':>12} {'Status':<15}")
    print("-" * 80)
    for name, m, n, k in ops:
        ai, flops, bytes_total = calculate_arithmetic_intensity(m, n, k, dtype)
        status = "COMPUTE-BOUND" if ai > 100 else "MEMORY-BOUND"
        print(f"{name:<20} ({m}, {k}) @ ({k}, {n}){'':<5} {ai:>12.1f} {status:<15}")

    print("\n--- LoRA Operations ---")

    lora_ops = [
        ("Layer LoRA A", M, layer_lora_rank, dim),  # (8192, 1024) @ (1024, 256)
        ("Layer LoRA B", M, hidden_dim, layer_lora_rank),  # (8192, 256) @ (256, 2816)
        ("Head LoRA A", M, head_lora_rank, dim),  # (8192, 1024) @ (1024, 192)
        ("Head LoRA B", M, n_heads * head_dim, head_lora_rank),  # (8192, 192) @ (192, 1024)
    ]

    print(f"\n{'Operation':<20} {'Shape':<30} {'AI (FLOPs/B)':>12} {'Status':<15}")
    print("-" * 80)
    for name, m, n, k in lora_ops:
        ai, flops, bytes_total = calculate_arithmetic_intensity(m, n, k, dtype)
        status = "COMPUTE-BOUND" if ai > 100 else "MEMORY-BOUND"
        print(f"{name:<20} ({m}, {k}) @ ({k}, {n}){'':<5} {ai:>12.1f} {status:<15}")

    # ============================================================
    # Part 2: Memory Traffic Analysis
    # ============================================================
    print("\n\n" + "=" * 70)
    print("PART 2: Memory Traffic per Forward Pass")
    print("=" * 70)

    # Calculate memory traffic for one SharedLinearWithLoRA
    def calc_shared_linear_traffic(M, in_dim, out_dim, rank, dtype=torch.bfloat16):
        """Calculate memory traffic for SharedLinearWithLoRA."""
        bpe = 2  # bytes per element for bf16

        # Input: x (M, in_dim) - read once, reused
        x_bytes = M * in_dim * bpe

        # Base linear: x @ W.T
        # Read: x (already counted), W (out_dim, in_dim)
        # Write: base_out (M, out_dim)
        w_bytes = out_dim * in_dim * bpe
        base_out_bytes = M * out_dim * bpe

        # LoRA A: x @ A.T
        # Read: x (reused), A (rank, in_dim)
        # Write: lora_h (M, rank)
        a_bytes = rank * in_dim * bpe
        lora_h_bytes = M * rank * bpe

        # LoRA B: lora_h @ B.T
        # Read: lora_h, B (out_dim, rank)
        # Write: lora_out (M, out_dim)
        b_bytes = out_dim * rank * bpe
        lora_out_bytes = M * out_dim * bpe

        # Add: base_out + lora_out (if not fused)
        # Read: base_out, lora_out
        # Write: final (M, out_dim) - might overwrite base_out

        total_read = x_bytes + w_bytes + a_bytes + lora_h_bytes + b_bytes
        total_write = base_out_bytes + lora_h_bytes + lora_out_bytes

        # If add is fused into GEMM epilogue, we save lora_out read/write
        fused_savings = lora_out_bytes * 2  # Don't write then read lora_out

        return {
            'total_unfused': total_read + total_write,
            'total_fused': total_read + total_write - fused_savings,
            'breakdown': {
                'x': x_bytes,
                'W': w_bytes,
                'A': a_bytes,
                'B': b_bytes,
                'lora_h': lora_h_bytes,
                'base_out': base_out_bytes,
                'lora_out': lora_out_bytes,
            }
        }

    traffic = calc_shared_linear_traffic(M, dim, hidden_dim, layer_lora_rank)

    print(f"\nMemory traffic for one SharedLinearWithLoRA (dim={dim} -> {hidden_dim}, rank={layer_lora_rank}):")
    print(f"  Input x:        {traffic['breakdown']['x'] / 1e6:>8.2f} MB")
    print(f"  Weight W:       {traffic['breakdown']['W'] / 1e6:>8.2f} MB")
    print(f"  LoRA A:         {traffic['breakdown']['A'] / 1e6:>8.2f} MB")
    print(f"  LoRA B:         {traffic['breakdown']['B'] / 1e6:>8.2f} MB")
    print(f"  Intermediate h: {traffic['breakdown']['lora_h'] / 1e6:>8.2f} MB")
    print(f"  Outputs:        {(traffic['breakdown']['base_out'] + traffic['breakdown']['lora_out']) / 1e6:>8.2f} MB")
    print(f"  ---")
    print(f"  Total (unfused): {traffic['total_unfused'] / 1e6:>7.2f} MB")
    print(f"  Total (fused):   {traffic['total_fused'] / 1e6:>7.2f} MB")
    print(f"  Savings from fusion: {(traffic['total_unfused'] - traffic['total_fused']) / 1e6:.2f} MB ({(1 - traffic['total_fused']/traffic['total_unfused'])*100:.1f}%)")

    # Per transformer block
    print("\n--- Memory Traffic per Transformer Block ---")

    # Attention: wq, wk, wv (dim -> dim), wo (dim -> dim)
    # Plus head offsets: 3 LoRAs (dim -> n_heads*head_dim)
    # FFN: w1, w3 (dim -> hidden), w2 (hidden -> dim)

    attn_wq = calc_shared_linear_traffic(M, dim, n_heads * head_dim, layer_lora_rank)
    attn_wk = calc_shared_linear_traffic(M, dim, n_heads * head_dim // 2, layer_lora_rank)  # GQA
    ffn_w1 = calc_shared_linear_traffic(M, dim, hidden_dim, layer_lora_rank)
    ffn_w2 = calc_shared_linear_traffic(M, hidden_dim, dim, layer_lora_rank)

    # Head offset LoRAs (no base weight, just LoRA)
    head_lora_traffic = M * dim * 2 + head_lora_rank * dim * 2 + M * head_lora_rank * 2 + \
                        head_lora_rank * n_heads * head_dim * 2 + M * n_heads * head_dim * 2

    block_traffic_unfused = (
        attn_wq['total_unfused'] * 2 +  # wq, wo
        attn_wk['total_unfused'] * 2 +  # wk, wv
        head_lora_traffic * 3 +  # 3 head offsets
        ffn_w1['total_unfused'] * 2 +  # w1, w3
        ffn_w2['total_unfused']  # w2
    )

    block_traffic_fused = (
        attn_wq['total_fused'] * 2 +
        attn_wk['total_fused'] * 2 +
        head_lora_traffic * 3 +  # Head offsets can't be epilogue-fused
        ffn_w1['total_fused'] * 2 +
        ffn_w2['total_fused']
    )

    print(f"\nTotal memory traffic per block:")
    print(f"  Unfused: {block_traffic_unfused / 1e9:.2f} GB")
    print(f"  Fused:   {block_traffic_fused / 1e9:.2f} GB")
    print(f"  At 4.8 TB/s bandwidth:")
    print(f"    Unfused: {block_traffic_unfused / 4.8e12 * 1e6:.2f} us (memory-bound floor)")
    print(f"    Fused:   {block_traffic_fused / 4.8e12 * 1e6:.2f} us (memory-bound floor)")

    # ============================================================
    # Part 3: Actual Kernel Profiling
    # ============================================================
    print("\n\n" + "=" * 70)
    print("PART 3: Actual Kernel Profiling")
    print("=" * 70)

    x = torch.randn(batch, seq, dim, device=device, dtype=dtype)

    # Test ScaledLoRAModule (head offset pattern)
    print("\n--- ScaledLoRAModule (Head Offset LoRA) ---")
    head_lora = ScaledLoRAModule(dim, n_heads * head_dim, rank=head_lora_rank).to(device).to(dtype)

    prof = profile_with_memory_metrics(lambda: head_lora(x), "HeadLoRA")
    kernels = analyze_kernel_details(prof)

    print(f"\n{'Kernel':<50} {'Type':<25} {'Count':>6} {'Time(us)':>10}")
    print("-" * 95)
    for k in sorted(kernels, key=lambda x: -x['time_us'])[:10]:
        print(f"{k['name'][:50]:<50} {k['fusion_type']:<25} {k['count']:>6} {k['time_us']:>10.1f}")

    # Test SharedLinearWithLoRA
    print("\n--- SharedLinearWithLoRA (Layer LoRA) ---")
    shared_linear = nn.Linear(dim, hidden_dim, bias=False).to(device).to(dtype)
    layer_lora = SharedLinearWithLoRA(shared_linear, dim, hidden_dim, layer_lora_rank).to(device).to(dtype)

    prof = profile_with_memory_metrics(lambda: layer_lora(x), "LayerLoRA")
    kernels = analyze_kernel_details(prof)

    print(f"\n{'Kernel':<50} {'Type':<25} {'Count':>6} {'Time(us)':>10}")
    print("-" * 95)
    for k in sorted(kernels, key=lambda x: -x['time_us'])[:10]:
        print(f"{k['name'][:50]:<50} {k['fusion_type']:<25} {k['count']:>6} {k['time_us']:>10.1f}")

    # Compiled versions
    print("\n--- SharedLinearWithLoRA (Compiled) ---")
    layer_lora_compiled = torch.compile(layer_lora)

    # Warmup compile
    for _ in range(5):
        _ = layer_lora_compiled(x)
    torch.cuda.synchronize()

    prof = profile_with_memory_metrics(lambda: layer_lora_compiled(x), "LayerLoRA_compiled")
    kernels = analyze_kernel_details(prof)

    print(f"\n{'Kernel':<50} {'Type':<25} {'Count':>6} {'Time(us)':>10}")
    print("-" * 95)
    for k in sorted(kernels, key=lambda x: -x['time_us'])[:10]:
        print(f"{k['name'][:50]:<50} {k['fusion_type']:<25} {k['count']:>6} {k['time_us']:>10.1f}")

    # Count fusion types
    fusion_summary = defaultdict(lambda: {'count': 0, 'time_us': 0})
    for k in kernels:
        fusion_summary[k['fusion_type']]['count'] += k['count']
        fusion_summary[k['fusion_type']]['time_us'] += k['time_us']

    print("\n--- Fusion Type Summary ---")
    print(f"{'Type':<30} {'Count':>8} {'Time(us)':>12} {'%':>8}")
    print("-" * 60)
    total_time = sum(v['time_us'] for v in fusion_summary.values())
    for ftype, data in sorted(fusion_summary.items(), key=lambda x: -x[1]['time_us']):
        pct = 100 * data['time_us'] / total_time if total_time > 0 else 0
        print(f"{ftype:<30} {data['count']:>8} {data['time_us']:>12.1f} {pct:>7.1f}%")

    # ============================================================
    # Part 4: Compare with theoretical minimum
    # ============================================================
    print("\n\n" + "=" * 70)
    print("PART 4: Theoretical vs Actual Performance")
    print("=" * 70)

    # Measure actual time
    import time

    def benchmark(fn, iters=100, warmup=20):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1e6 / iters  # us

    actual_time = benchmark(lambda: layer_lora(x))
    actual_time_compiled = benchmark(lambda: layer_lora_compiled(x))

    # Theoretical memory-bound time
    mem_bound_time = traffic['total_unfused'] / 4.8e12 * 1e6  # us

    print(f"\nSharedLinearWithLoRA (dim={dim} -> {hidden_dim}, rank={layer_lora_rank}):")
    print(f"  Theoretical memory-bound minimum: {mem_bound_time:.2f} us")
    print(f"  Actual (eager):                   {actual_time:.2f} us")
    print(f"  Actual (compiled):                {actual_time_compiled:.2f} us")
    print(f"  Efficiency (eager):               {mem_bound_time/actual_time*100:.1f}%")
    print(f"  Efficiency (compiled):            {mem_bound_time/actual_time_compiled*100:.1f}%")

    if actual_time > mem_bound_time * 2:
        print(f"\n  ⚠️  Actual time is {actual_time/mem_bound_time:.1f}x the memory-bound minimum.")
        print(f"      This suggests overhead beyond memory bandwidth (kernel launch, etc.)")
    else:
        print(f"\n  ✓  Performance is within 2x of memory-bound limit.")
        print(f"      Memory bandwidth is likely the primary bottleneck.")


if __name__ == "__main__":
    main()
