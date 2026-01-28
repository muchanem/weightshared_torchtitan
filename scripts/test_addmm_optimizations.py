#!/usr/bin/env python3
"""
Test additional optimizations to avoid the Memcpy overhead from torch.addmm.

Key insight: The Memcpy happens because addmm needs to copy the bias to output.
Let's try:
1. Pre-allocated output buffer with torch.addmm(..., out=buffer)
2. Folding scale into the B weights
3. Using baddbmm for batched operations
4. In-place operations
"""

import torch
import torch.nn.functional as F
import time


def count_cuda_kernels(prof):
    """Count CUDA kernels by type from profiler output."""
    events = prof.key_averages()

    stats = {'gemm': 0, 'add': 0, 'copy': 0, 'other': 0}
    times = {'gemm': 0.0, 'add': 0.0, 'copy': 0.0, 'other': 0.0}
    details = []

    for e in events:
        if e.self_device_time_total == 0:
            continue

        key_lower = e.key.lower()
        time_ms = e.self_device_time_total / 1000

        if 'nvjet' in key_lower or 'gemm' in key_lower or 'cutlass' in key_lower:
            stats['gemm'] += e.count
            times['gemm'] += time_ms
            details.append(('GEMM', e.key[:50], e.count, time_ms))
        elif 'memcpy' in key_lower or 'copy' in key_lower:
            stats['copy'] += e.count
            times['copy'] += time_ms
            details.append(('COPY', e.key[:50], e.count, time_ms))
        elif 'add' in key_lower or 'elementwise' in key_lower or 'mul' in key_lower:
            stats['add'] += e.count
            times['add'] += time_ms
            details.append(('ELEM', e.key[:50], e.count, time_ms))
        else:
            stats['other'] += e.count
            times['other'] += time_ms
            details.append(('OTHER', e.key[:50], e.count, time_ms))

    return stats, times, details


def profile_and_print(name, fn, iterations=10, warmup=5):
    """Profile and print results."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        for _ in range(iterations):
            fn()
        torch.cuda.synchronize()

    stats, times, details = count_cuda_kernels(prof)
    total = sum(stats.values())
    total_time = sum(times.values())

    print(f"\n{name}")
    print("-" * 50)
    print(f"  GEMM: {stats['gemm']:2d} ({times['gemm']:.3f}ms) | "
          f"ELEM: {stats['add']:2d} ({times['add']:.3f}ms) | "
          f"COPY: {stats['copy']:2d} ({times['copy']:.3f}ms) | "
          f"Total: {total:2d} ({total_time:.3f}ms)")

    return stats, times


def benchmark(name, fn, iterations=1000, warmup=100):
    """Time an operation."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    ms = elapsed * 1000 / iterations
    return ms


def main():
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    # Typical sizes
    M, K, N = 8192, 1024, 2816  # batch*seq, dim, hidden_dim
    rank = 256
    scale = 2.0

    # Create tensors
    x = torch.randn(M, K, device=device, dtype=dtype)
    W = torch.randn(N, K, device=device, dtype=dtype)
    A = torch.randn(rank, K, device=device, dtype=dtype)
    B = torch.randn(N, rank, device=device, dtype=dtype)

    # Pre-compute scaled B (for optimization 2)
    B_scaled = B * scale

    print("=" * 60)
    print("Testing LoRA Fusion Approaches")
    print("=" * 60)
    print(f"Shapes: x={tuple(x.shape)}, W={tuple(W.shape)}, A={tuple(A.shape)}, B={tuple(B.shape)}")
    print(f"Scale: {scale}")

    # =========================================
    # Baseline: Current implementation
    # =========================================
    def current():
        base = x @ W.t()
        h = x @ A.t()
        lora = h @ B.t() * scale
        return base + lora

    profile_and_print("1. CURRENT: base + (x @ A.t() @ B.t()) * scale", current)

    # =========================================
    # Approach 2: Use addmm (what we tested before)
    # =========================================
    def addmm_basic():
        h = x @ A.t()
        lora = h @ B.t() * scale
        return torch.addmm(lora, x, W.t())

    profile_and_print("2. ADDMM basic: addmm(lora, x, W.t())", addmm_basic)

    # =========================================
    # Approach 3: Fold scale into B weights
    # =========================================
    def addmm_prescaled():
        h = x @ A.t()
        lora = h @ B_scaled.t()  # scale baked into B
        return torch.addmm(lora, x, W.t())

    profile_and_print("3. ADDMM pre-scaled B: addmm(x @ A.t() @ B_scaled.t(), x, W.t())", addmm_prescaled)

    # =========================================
    # Approach 4: Use addmm's alpha parameter for scale
    # NOT correct semantically - just testing
    # =========================================
    # torch.addmm(bias, mat1, mat2, beta=1, alpha=1) = beta*bias + alpha*(mat1@mat2)
    # We want: (x @ A.t() @ B.t()) * scale + x @ W.t()
    # Can't directly express this with addmm since scale is on bias, not matmul

    # =========================================
    # Approach 5: Reorder operations
    # Compute base first, then add lora in-place
    # =========================================
    def inplace_add():
        out = x @ W.t()  # base
        h = x @ A.t()
        lora = h @ B.t()
        out.add_(lora, alpha=scale)  # in-place: out += lora * scale
        return out

    profile_and_print("4. IN-PLACE: out = x @ W.t(); out.add_(lora, alpha=scale)", inplace_add)

    # =========================================
    # Approach 6: torch.addmv for smaller M (not applicable here)
    # =========================================

    # =========================================
    # Approach 7: Use mm + addcmul pattern
    # =========================================
    def mm_addcmul():
        base = x @ W.t()
        h = x @ A.t()
        lora = h @ B.t()
        # addcmul: out = base + scale * lora (fused multiply-add)
        return torch.addcmul(base, lora, torch.ones_like(lora), value=scale)

    # Actually addcmul needs tensor * tensor, not scalar * tensor
    # Let's use add_ with alpha instead

    # =========================================
    # Approach 8: Manual fusion with pre-allocated output
    # =========================================
    output_buffer = torch.empty(M, N, device=device, dtype=dtype)

    def preallocated():
        # Compute base into buffer
        torch.mm(x, W.t(), out=output_buffer)
        # Compute lora
        h = x @ A.t()
        lora = h @ B.t()
        # Add in-place
        output_buffer.add_(lora, alpha=scale)
        return output_buffer

    profile_and_print("5. PRE-ALLOCATED: mm(out=buf); buf.add_(lora, alpha=scale)", preallocated)

    # =========================================
    # Approach 9: addmm with out parameter
    # =========================================
    def addmm_with_out():
        h = x @ A.t()
        lora = h @ B_scaled.t()
        torch.addmm(lora, x, W.t(), out=output_buffer)
        return output_buffer

    profile_and_print("6. ADDMM with out=: addmm(lora, x, W.t(), out=buf)", addmm_with_out)

    # =========================================
    # Timing comparison
    # =========================================
    print("\n" + "=" * 60)
    print("TIMING BENCHMARK (1000 iterations)")
    print("=" * 60)

    approaches = [
        ("Current (baseline)", current),
        ("addmm basic", addmm_basic),
        ("addmm pre-scaled B", addmm_prescaled),
        ("in-place add", inplace_add),
        ("pre-allocated buffer", preallocated),
        ("addmm with out=", addmm_with_out),
    ]

    times = []
    for name, fn in approaches:
        t = benchmark(name, fn)
        times.append((name, t))
        print(f"{name:25s}: {t:.4f} ms/iter")

    baseline = times[0][1]
    print("\nSpeedup vs baseline:")
    for name, t in times[1:]:
        print(f"  {name:25s}: {baseline/t:.2f}x")

    # Verify correctness
    print("\n" + "=" * 60)
    print("CORRECTNESS CHECK")
    print("=" * 60)

    ref = current()
    for name, fn in approaches[1:]:
        out = fn()
        diff = (ref - out).abs().max().item()
        status = "✓" if diff < 1e-3 else "✗"
        print(f"{status} {name}: max diff = {diff:.2e}")


def test_separate_scale_fusion():
    """Test if we can fuse the scale multiplication."""
    print("\n" + "=" * 60)
    print("Testing Scale Multiplication Fusion")
    print("=" * 60)

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    M, K, N = 8192, 1024, 2816
    rank = 256
    scale = 2.0

    x = torch.randn(M, K, device=device, dtype=dtype)
    A = torch.randn(rank, K, device=device, dtype=dtype)
    B = torch.randn(N, rank, device=device, dtype=dtype)

    # Test 1: Two matmuls + scale (current)
    def two_mm_scale():
        h = x @ A.t()
        return h @ B.t() * scale

    # Test 2: Pre-scaled B
    B_scaled = B * scale
    def prescaled_B():
        h = x @ A.t()
        return h @ B_scaled.t()

    # Test 3: Use addmm with alpha for the second matmul
    # addmm(zeros, h, B.t(), alpha=scale) = scale * (h @ B.t())
    zeros = torch.zeros(M, N, device=device, dtype=dtype)
    def addmm_alpha():
        h = x @ A.t()
        return torch.addmm(zeros, h, B.t(), alpha=scale)

    profile_and_print("Two mm + scale", two_mm_scale)
    profile_and_print("Pre-scaled B", prescaled_B)
    profile_and_print("addmm with alpha", addmm_alpha)

    t1 = benchmark("Two mm + scale", two_mm_scale)
    t2 = benchmark("Pre-scaled B", prescaled_B)
    t3 = benchmark("addmm alpha", addmm_alpha)

    print(f"\nTiming:")
    print(f"  Two mm + scale: {t1:.4f} ms")
    print(f"  Pre-scaled B:   {t2:.4f} ms ({t1/t2:.2f}x)")
    print(f"  addmm alpha:    {t3:.4f} ms ({t1/t3:.2f}x)")


if __name__ == "__main__":
    main()
    test_separate_scale_fusion()
