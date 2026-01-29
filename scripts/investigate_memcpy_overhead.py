#!/usr/bin/env python3
"""
Investigate why torch.addmm introduces Memcpy DtoD overhead.

Compare:
1. Original implementation (base + lora)
2. torch.addmm implementation
3. Pure operations without view/reshape
"""

import torch
import torch.nn as nn
import time
from collections import defaultdict


def profile_kernels(fn, name, warmup=10, iters=5):
    """Profile and return kernel breakdown."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()

    # Aggregate by kernel type
    stats = defaultdict(lambda: {'count': 0, 'time_us': 0.0})
    for e in prof.key_averages():
        if e.self_device_time_total == 0:
            continue
        key = e.key[:60]
        stats[key]['count'] += e.count
        stats[key]['time_us'] += e.self_device_time_total

    return dict(stats)


def benchmark(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / iters


def main():
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    batch, seq = 4, 2048
    dim = 1024
    hidden_dim = 2816
    rank = 256

    x = torch.randn(batch, seq, dim, device=device, dtype=dtype)
    W = torch.randn(hidden_dim, dim, device=device, dtype=dtype)
    A = torch.randn(rank, dim, device=device, dtype=dtype)
    B = torch.randn(hidden_dim, rank, device=device, dtype=dtype)

    print("=" * 70)
    print("MEMCPY OVERHEAD INVESTIGATION")
    print("=" * 70)
    print(f"Shape: x ({batch}, {seq}, {dim}), W ({hidden_dim}, {dim})")
    print(f"LoRA: A ({rank}, {dim}), B ({hidden_dim}, {rank})")

    # ============================================================
    # Implementation 1: Original (separate add)
    # ============================================================
    def impl_original():
        base_out = x @ W.t()
        lora_h = x @ A.t()
        lora_out = lora_h @ B.t()
        return base_out + lora_out

    print("\n--- Implementation 1: Original (base + lora) ---")
    stats1 = profile_kernels(impl_original, "original")
    t1 = benchmark(impl_original)

    print(f"\n{'Kernel':<55} {'Count':>6} {'Time(us)':>10}")
    print("-" * 75)
    for k, v in sorted(stats1.items(), key=lambda x: -x[1]['time_us']):
        print(f"{k:<55} {v['count']:>6} {v['time_us']:>10.1f}")
    print(f"\nBenchmark: {t1:.2f} us")

    # ============================================================
    # Implementation 2: torch.addmm with view/reshape
    # ============================================================
    def impl_addmm_view():
        lora_h = x @ A.t()
        lora_out = lora_h @ B.t()
        x_2d = x.view(-1, x.shape[-1])
        lora_2d = lora_out.view(-1, lora_out.shape[-1])
        y_2d = torch.addmm(lora_2d, x_2d, W.t())
        return y_2d.view(*x.shape[:-1], -1)

    print("\n--- Implementation 2: torch.addmm with view/reshape ---")
    stats2 = profile_kernels(impl_addmm_view, "addmm_view")
    t2 = benchmark(impl_addmm_view)

    print(f"\n{'Kernel':<55} {'Count':>6} {'Time(us)':>10}")
    print("-" * 75)
    for k, v in sorted(stats2.items(), key=lambda x: -x[1]['time_us']):
        print(f"{k:<55} {v['count']:>6} {v['time_us']:>10.1f}")
    print(f"\nBenchmark: {t2:.2f} us")

    # ============================================================
    # Implementation 3: torch.addmm with pre-flattened input
    # ============================================================
    x_flat = x.view(-1, dim).contiguous()

    def impl_addmm_flat():
        lora_h = x_flat @ A.t()
        lora_out = lora_h @ B.t()
        return torch.addmm(lora_out, x_flat, W.t())

    print("\n--- Implementation 3: torch.addmm with pre-flattened input ---")
    stats3 = profile_kernels(impl_addmm_flat, "addmm_flat")
    t3 = benchmark(impl_addmm_flat)

    print(f"\n{'Kernel':<55} {'Count':>6} {'Time(us)':>10}")
    print("-" * 75)
    for k, v in sorted(stats3.items(), key=lambda x: -x[1]['time_us']):
        print(f"{k:<55} {v['count']:>6} {v['time_us']:>10.1f}")
    print(f"\nBenchmark: {t3:.2f} us")

    # ============================================================
    # Implementation 4: In-place add (no new tensor)
    # ============================================================
    def impl_inplace():
        base_out = x @ W.t()
        lora_h = x @ A.t()
        lora_out = lora_h @ B.t()
        base_out.add_(lora_out)
        return base_out

    print("\n--- Implementation 4: In-place add ---")
    stats4 = profile_kernels(impl_inplace, "inplace")
    t4 = benchmark(impl_inplace)

    print(f"\n{'Kernel':<55} {'Count':>6} {'Time(us)':>10}")
    print("-" * 75)
    for k, v in sorted(stats4.items(), key=lambda x: -x[1]['time_us']):
        print(f"{k:<55} {v['count']:>6} {v['time_us']:>10.1f}")
    print(f"\nBenchmark: {t4:.2f} us")

    # ============================================================
    # Implementation 5: addmm with out= parameter
    # ============================================================
    def impl_addmm_out():
        lora_h = x_flat @ A.t()
        lora_out = lora_h @ B.t()
        out = torch.empty(x_flat.shape[0], hidden_dim, device=device, dtype=dtype)
        torch.addmm(lora_out, x_flat, W.t(), out=out)
        return out

    print("\n--- Implementation 5: torch.addmm with out= parameter ---")
    stats5 = profile_kernels(impl_addmm_out, "addmm_out")
    t5 = benchmark(impl_addmm_out)

    print(f"\n{'Kernel':<55} {'Count':>6} {'Time(us)':>10}")
    print("-" * 75)
    for k, v in sorted(stats5.items(), key=lambda x: -x[1]['time_us']):
        print(f"{k:<55} {v['count']:>6} {v['time_us']:>10.1f}")
    print(f"\nBenchmark: {t5:.2f} us")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Implementation':<40} {'Time (us)':>12} {'vs Original':>12}")
    print("-" * 65)
    print(f"{'1. Original (base + lora)':<40} {t1:>12.2f} {'1.00x':>12}")
    print(f"{'2. addmm with view/reshape':<40} {t2:>12.2f} {t1/t2:>11.2f}x")
    print(f"{'3. addmm with pre-flattened':<40} {t3:>12.2f} {t1/t3:>11.2f}x")
    print(f"{'4. In-place add':<40} {t4:>12.2f} {t1/t4:>11.2f}x")
    print(f"{'5. addmm with out=':<40} {t5:>12.2f} {t1/t5:>11.2f}x")

    # Count memcpy per implementation
    def count_memcpy(stats):
        return sum(v['count'] for k, v in stats.items() if 'memcpy' in k.lower() or 'copy' in k.lower())

    print(f"\n{'Implementation':<40} {'Memcpy count':>12}")
    print("-" * 55)
    print(f"{'1. Original':<40} {count_memcpy(stats1):>12}")
    print(f"{'2. addmm with view/reshape':<40} {count_memcpy(stats2):>12}")
    print(f"{'3. addmm with pre-flattened':<40} {count_memcpy(stats3):>12}")
    print(f"{'4. In-place add':<40} {count_memcpy(stats4):>12}")
    print(f"{'5. addmm with out=':<40} {count_memcpy(stats5):>12}")

    # ============================================================
    # Check if torch.compile helps
    # ============================================================
    print("\n\n--- torch.compile comparison ---")

    impl_original_compiled = torch.compile(impl_original)
    impl_addmm_compiled = torch.compile(impl_addmm_view)

    # Warmup
    for _ in range(10):
        _ = impl_original_compiled()
        _ = impl_addmm_compiled()
    torch.cuda.synchronize()

    t1_compiled = benchmark(impl_original_compiled)
    t2_compiled = benchmark(impl_addmm_compiled)

    stats1_c = profile_kernels(impl_original_compiled, "original_compiled")
    stats2_c = profile_kernels(impl_addmm_compiled, "addmm_compiled")

    print(f"\n{'Implementation':<40} {'Eager (us)':>12} {'Compiled (us)':>14} {'Speedup':>10}")
    print("-" * 80)
    print(f"{'Original (base + lora)':<40} {t1:>12.2f} {t1_compiled:>14.2f} {t1/t1_compiled:>9.2f}x")
    print(f"{'addmm with view/reshape':<40} {t2:>12.2f} {t2_compiled:>14.2f} {t2/t2_compiled:>9.2f}x")

    print(f"\nCompiled original memcpy count: {count_memcpy(stats1_c)}")
    print(f"Compiled addmm memcpy count: {count_memcpy(stats2_c)}")

    print("\n--- Compiled Original Kernels ---")
    for k, v in sorted(stats1_c.items(), key=lambda x: -x[1]['time_us'])[:8]:
        print(f"  {k:<55} {v['count']:>6} {v['time_us']:>10.1f}")

    print("\n--- Compiled addmm Kernels ---")
    for k, v in sorted(stats2_c.items(), key=lambda x: -x[1]['time_us'])[:8]:
        print(f"  {k:<55} {v['count']:>6} {v['time_us']:>10.1f}")


if __name__ == "__main__":
    main()
