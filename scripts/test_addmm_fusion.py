#!/usr/bin/env python3
"""
Test whether torch.addmm actually fuses the add into the GEMM epilogue.

Expected: torch.addmm should use cuBLAS's beta parameter to fuse the add,
resulting in fewer kernel launches than separate matmul + add.
"""

import torch
import torch.nn.functional as F


def count_cuda_kernels(prof):
    """Count CUDA kernels by type from profiler output."""
    events = prof.key_averages()

    gemm_count = 0
    gemm_time = 0.0
    add_count = 0
    add_time = 0.0
    other_count = 0
    other_time = 0.0

    kernel_details = []

    for e in events:
        if e.self_device_time_total == 0:
            continue

        key_lower = e.key.lower()
        time_ms = e.self_device_time_total / 1000

        if 'nvjet' in key_lower or 'gemm' in key_lower or 'cutlass' in key_lower:
            gemm_count += e.count
            gemm_time += time_ms
            kernel_details.append(('GEMM', e.key, e.count, time_ms))
        elif 'add' in key_lower or 'elementwise' in key_lower:
            add_count += e.count
            add_time += time_ms
            kernel_details.append(('ADD', e.key, e.count, time_ms))
        elif e.self_device_time_total > 0:
            other_count += e.count
            other_time += time_ms
            kernel_details.append(('OTHER', e.key, e.count, time_ms))

    return {
        'gemm_count': gemm_count,
        'gemm_time': gemm_time,
        'add_count': add_count,
        'add_time': add_time,
        'other_count': other_count,
        'other_time': other_time,
        'total_count': gemm_count + add_count + other_count,
        'total_time': gemm_time + add_time + other_time,
        'details': kernel_details,
    }


def profile_operation(name, fn, iterations=10, warmup=5):
    """Profile an operation and return kernel statistics."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Profile
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(iterations):
            fn()
        torch.cuda.synchronize()

    stats = count_cuda_kernels(prof)

    print(f"\n{'='*60}")
    print(f"{name}")
    print('='*60)
    print(f"GEMM kernels: {stats['gemm_count']:3d} launches, {stats['gemm_time']:.3f} ms")
    print(f"ADD kernels:  {stats['add_count']:3d} launches, {stats['add_time']:.3f} ms")
    print(f"Other:        {stats['other_count']:3d} launches, {stats['other_time']:.3f} ms")
    print(f"TOTAL:        {stats['total_count']:3d} launches, {stats['total_time']:.3f} ms")

    print("\nKernel details:")
    for ktype, kname, count, time in sorted(stats['details'], key=lambda x: -x[3]):
        print(f"  [{ktype:5}] {kname[:60]}: count={count}, time={time:.3f}ms")

    return stats


def test_addmm_fusion():
    """Test if torch.addmm fuses add into GEMM epilogue."""
    print("\n" + "#"*60)
    print("# Testing torch.addmm Fusion")
    print("#"*60)

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    # Typical sizes from weight sharing
    batch, seq, dim = 4, 2048, 1024
    hidden_dim = 2816
    M = batch * seq  # 8192

    # Create tensors
    x = torch.randn(M, dim, device=device, dtype=dtype)
    W = torch.randn(hidden_dim, dim, device=device, dtype=dtype)
    lora_out = torch.randn(M, hidden_dim, device=device, dtype=dtype)

    # Test 1: Unfused (matmul + add)
    def unfused():
        base = x @ W.t()
        return base + lora_out

    stats_unfused = profile_operation("UNFUSED: x @ W.t() + lora_out", unfused)

    # Test 2: torch.addmm (should fuse)
    def fused_addmm():
        return torch.addmm(lora_out, x, W.t())

    stats_fused = profile_operation("FUSED: torch.addmm(lora_out, x, W.t())", fused_addmm)

    # Test 3: Full Linear+LoRA pattern (current implementation)
    rank = 256
    A = torch.randn(rank, dim, device=device, dtype=dtype)
    B = torch.randn(hidden_dim, rank, device=device, dtype=dtype)
    scale = 2.0

    def current_lora():
        base = x @ W.t()
        lora_h = x @ A.t()
        lora = lora_h @ B.t() * scale
        return base + lora

    stats_current = profile_operation("CURRENT: base + (x @ A.t() @ B.t() * scale)", current_lora)

    # Test 4: Optimized with addmm
    def optimized_lora():
        lora_h = x @ A.t()
        lora = lora_h @ B.t() * scale
        return torch.addmm(lora, x, W.t())

    stats_optimized = profile_operation("OPTIMIZED: torch.addmm(lora, x, W.t())", optimized_lora)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nUnfused vs Fused (simple case):")
    print(f"  Unfused: {stats_unfused['total_count']} kernels, {stats_unfused['total_time']:.3f} ms")
    print(f"  Fused:   {stats_fused['total_count']} kernels, {stats_fused['total_time']:.3f} ms")
    print(f"  Kernel reduction: {stats_unfused['total_count'] - stats_fused['total_count']}")
    print(f"  ADD kernels eliminated: {stats_unfused['add_count'] - stats_fused['add_count']}")

    print(f"\nCurrent vs Optimized (full LoRA pattern):")
    print(f"  Current:   {stats_current['total_count']} kernels, {stats_current['total_time']:.3f} ms")
    print(f"  Optimized: {stats_optimized['total_count']} kernels, {stats_optimized['total_time']:.3f} ms")
    print(f"  Kernel reduction: {stats_current['total_count'] - stats_optimized['total_count']}")
    print(f"  ADD kernels eliminated: {stats_current['add_count'] - stats_optimized['add_count']}")

    # Verify correctness
    print("\n" + "="*60)
    print("CORRECTNESS CHECK")
    print("="*60)

    out_unfused = unfused()
    out_fused = fused_addmm()
    diff = (out_unfused - out_fused).abs().max().item()
    print(f"Unfused vs Fused max diff: {diff:.6e} (should be ~0)")

    out_current = current_lora()
    out_optimized = optimized_lora()
    diff = (out_current - out_optimized).abs().max().item()
    print(f"Current vs Optimized max diff: {diff:.6e} (should be ~0)")

    # Timing benchmark
    print("\n" + "="*60)
    print("TIMING BENCHMARK (1000 iterations)")
    print("="*60)

    import time

    def benchmark(name, fn, iterations=1000):
        for _ in range(100):
            fn()
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        ms_per_iter = elapsed * 1000 / iterations
        print(f"{name}: {ms_per_iter:.4f} ms/iter")
        return ms_per_iter

    t_unfused = benchmark("Unfused (matmul + add)", unfused)
    t_fused = benchmark("Fused (addmm)", fused_addmm)
    t_current = benchmark("Current LoRA pattern", current_lora)
    t_optimized = benchmark("Optimized LoRA pattern", optimized_lora)

    print(f"\nSpeedup from addmm fusion:")
    print(f"  Simple case: {t_unfused/t_fused:.2f}x")
    print(f"  Full LoRA:   {t_current/t_optimized:.2f}x")

    return {
        'fusion_works': stats_fused['add_count'] < stats_unfused['add_count'],
        'speedup_simple': t_unfused / t_fused,
        'speedup_lora': t_current / t_optimized,
    }


def test_addmm_with_alpha_beta():
    """Test torch.addmm with explicit alpha/beta parameters."""
    print("\n" + "#"*60)
    print("# Testing torch.addmm with alpha/beta")
    print("#"*60)

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    M, K, N = 8192, 1024, 2816

    x = torch.randn(M, K, device=device, dtype=dtype)
    W = torch.randn(N, K, device=device, dtype=dtype)
    bias = torch.randn(M, N, device=device, dtype=dtype)

    # Test with beta=1.0 (default, should fuse add)
    def addmm_beta1():
        return torch.addmm(bias, x, W.t(), beta=1.0, alpha=1.0)

    # Test with beta=0.0 (bias ignored, just matmul)
    def addmm_beta0():
        return torch.addmm(bias, x, W.t(), beta=0.0, alpha=1.0)

    # Test with custom scale
    scale = 0.5
    def addmm_scaled():
        return torch.addmm(bias, x, W.t(), beta=scale, alpha=1.0)

    stats_beta1 = profile_operation("addmm with beta=1.0", addmm_beta1, iterations=5)
    stats_beta0 = profile_operation("addmm with beta=0.0", addmm_beta0, iterations=5)
    stats_scaled = profile_operation("addmm with beta=0.5", addmm_scaled, iterations=5)

    print("\nBeta parameter comparison:")
    print(f"  beta=1.0: {stats_beta1['add_count']} add kernels")
    print(f"  beta=0.0: {stats_beta0['add_count']} add kernels")
    print(f"  beta=0.5: {stats_scaled['add_count']} add kernels")


if __name__ == "__main__":
    results = test_addmm_fusion()
    test_addmm_with_alpha_beta()

    print("\n" + "#"*60)
    print("# CONCLUSION")
    print("#"*60)
    if results['fusion_works']:
        print("✓ torch.addmm DOES fuse the add operation into GEMM epilogue!")
        print(f"  Simple speedup: {results['speedup_simple']:.2f}x")
        print(f"  Full LoRA speedup: {results['speedup_lora']:.2f}x")
    else:
        print("✗ torch.addmm does NOT appear to fuse - investigate further")
