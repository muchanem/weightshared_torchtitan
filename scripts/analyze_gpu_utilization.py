#!/usr/bin/env python3
"""
Analyze GPU utilization for LoRA operations.

Use CUPTI metrics to understand:
1. SM occupancy
2. Memory throughput utilization
3. Compute throughput utilization
4. Where time is actually being spent
"""

import torch
import torch.nn as nn
import subprocess
import tempfile
import os


def run_ncu_profile(script_content, metrics):
    """Run a script with ncu profiling and return metrics."""
    # Write script to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name

    try:
        # Run ncu
        cmd = [
            'ncu',
            '--metrics', ','.join(metrics),
            '--target-processes', 'all',
            '--print-summary', 'per-kernel',
            'python', script_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return result.stdout, result.stderr
    except FileNotFoundError:
        return None, "ncu not found"
    except subprocess.TimeoutExpired:
        return None, "timeout"
    finally:
        os.unlink(script_path)


def main():
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    batch, seq = 4, 2048
    M = batch * seq
    dim = 1024
    hidden_dim = 2816
    rank = 256

    print("=" * 70)
    print("GPU UTILIZATION ANALYSIS")
    print("=" * 70)

    # ============================================================
    # Part 1: Profile with torch.profiler FLOPS
    # ============================================================
    print("\n--- Part 1: FLOPS and Memory Analysis ---")

    x = torch.randn(batch, seq, dim, device=device, dtype=dtype)
    W = torch.randn(hidden_dim, dim, device=device, dtype=dtype)
    A = torch.randn(rank, dim, device=device, dtype=dtype)
    B = torch.randn(hidden_dim, rank, device=device, dtype=dtype)

    def compute_lora():
        base = x @ W.t()
        lora_h = x @ A.t()
        lora_out = lora_h @ B.t()
        return base + lora_out

    # Warmup
    for _ in range(20):
        _ = compute_lora()
    torch.cuda.synchronize()

    # Profile with FLOPS counting
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        with_flops=True,
        profile_memory=True,
    ) as prof:
        for _ in range(10):
            _ = compute_lora()
        torch.cuda.synchronize()

    print("\nKernel Analysis:")
    print(f"{'Kernel':<50} {'Count':>6} {'Time(us)':>10} {'FLOPS':>15}")
    print("-" * 85)

    total_flops = 0
    total_time = 0
    for e in prof.key_averages():
        if e.self_device_time_total == 0:
            continue
        flops = getattr(e, 'flops', 0) or 0
        total_flops += flops
        total_time += e.self_device_time_total
        if flops > 0 or e.self_device_time_total > 50:
            print(f"{e.key[:50]:<50} {e.count:>6} {e.self_device_time_total:>10.1f} {flops:>15,}")

    print(f"\nTotal FLOPS: {total_flops:,}")
    print(f"Total time: {total_time:.1f} us")
    print(f"Achieved TFLOPS: {total_flops / total_time / 1e6:.2f}")
    print(f"H200 peak BF16 TFLOPS: ~1979")
    print(f"Compute utilization: {total_flops / total_time / 1e6 / 1979 * 100:.2f}%")

    # ============================================================
    # Part 2: Theoretical analysis
    # ============================================================
    print("\n\n--- Part 2: Theoretical Analysis ---")

    # Calculate theoretical FLOPS for each operation
    ops = [
        ("Base GEMM", 2 * M * hidden_dim * dim),
        ("LoRA A GEMM", 2 * M * rank * dim),
        ("LoRA B GEMM", 2 * M * hidden_dim * rank),
        ("Add", M * hidden_dim),  # Not really FLOPs but...
    ]

    print(f"\n{'Operation':<20} {'FLOPs':>15} {'Per iter (us)':>15}")
    print("-" * 55)
    total_theoretical = 0
    for name, flops in ops:
        total_theoretical += flops
        # Theoretical time at peak
        time_at_peak = flops / (1979e12) * 1e6
        print(f"{name:<20} {flops:>15,} {time_at_peak:>15.2f}")

    print(f"\nTotal theoretical FLOPs per iteration: {total_theoretical:,}")
    print(f"At H200 peak (1979 TFLOPS): {total_theoretical / 1979e12 * 1e6:.2f} us")

    # ============================================================
    # Part 3: Memory bandwidth analysis
    # ============================================================
    print("\n\n--- Part 3: Memory Bandwidth Analysis ---")

    bpe = 2  # bytes per element for bf16

    mem_ops = [
        ("Read x (reused 3x)", M * dim * bpe),
        ("Read W", hidden_dim * dim * bpe),
        ("Write base_out", M * hidden_dim * bpe),
        ("Read A", rank * dim * bpe),
        ("Write lora_h", M * rank * bpe),
        ("Read lora_h", M * rank * bpe),
        ("Read B", hidden_dim * rank * bpe),
        ("Write lora_out", M * hidden_dim * bpe),
        ("Read base_out (for add)", M * hidden_dim * bpe),
        ("Read lora_out (for add)", M * hidden_dim * bpe),
        ("Write final", M * hidden_dim * bpe),
    ]

    print(f"\n{'Operation':<30} {'Bytes':>15}")
    print("-" * 50)
    total_bytes = 0
    for name, bytes_count in mem_ops:
        total_bytes += bytes_count
        print(f"{name:<30} {bytes_count / 1e6:>12.2f} MB")

    print(f"\nTotal memory traffic: {total_bytes / 1e9:.3f} GB")
    print(f"At H200 bandwidth (4.8 TB/s): {total_bytes / 4.8e12 * 1e6:.2f} us")

    # ============================================================
    # Part 4: Roofline analysis
    # ============================================================
    print("\n\n--- Part 4: Roofline Analysis ---")

    # H200 NVL specs
    peak_tflops = 1979  # BF16
    peak_bw_tbs = 4.8  # TB/s
    ridge_point = peak_tflops * 1e12 / (peak_bw_tbs * 1e12)  # FLOPS/byte

    print(f"H200 NVL Ridge Point: {ridge_point:.1f} FLOPS/byte")

    # Arithmetic intensity for each op
    ai_ops = [
        ("Base GEMM (M,K)@(K,N)", 2*M*hidden_dim*dim, (M*dim + hidden_dim*dim + M*hidden_dim)*bpe),
        ("LoRA A GEMM", 2*M*rank*dim, (M*dim + rank*dim + M*rank)*bpe),
        ("LoRA B GEMM", 2*M*hidden_dim*rank, (M*rank + hidden_dim*rank + M*hidden_dim)*bpe),
    ]

    print(f"\n{'Operation':<30} {'AI (FLOP/B)':>12} {'Bound':>15}")
    print("-" * 60)
    for name, flops, bytes_count in ai_ops:
        ai = flops / bytes_count
        bound = "COMPUTE" if ai > ridge_point else "MEMORY"
        print(f"{name:<30} {ai:>12.1f} {bound:>15}")

    # ============================================================
    # Part 5: Why 17% efficiency?
    # ============================================================
    print("\n\n--- Part 5: Efficiency Analysis ---")

    # Actual measured time (per iteration)
    import time

    def benchmark(fn, iters=100, warmup=20):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1e6 / iters

    actual_time = benchmark(compute_lora)

    print(f"\nActual time per iteration: {actual_time:.2f} us")
    print(f"Theoretical compute-bound: {total_theoretical / 1979e12 * 1e6:.2f} us")
    print(f"Theoretical memory-bound: {total_bytes / 4.8e12 * 1e6:.2f} us")

    # The actual time should be bounded by the maximum of compute and memory bound
    theoretical_bound = max(total_theoretical / 1979e12 * 1e6, total_bytes / 4.8e12 * 1e6)
    print(f"\nTheoretical roofline bound: {theoretical_bound:.2f} us")
    print(f"Actual efficiency: {theoretical_bound / actual_time * 100:.1f}%")

    # The gap comes from:
    overhead = actual_time - theoretical_bound
    print(f"\nOverhead beyond roofline: {overhead:.2f} us ({overhead/actual_time*100:.1f}%)")

    print("""
Possible sources of overhead:
1. Kernel launch overhead (~5-10 us per kernel × 4 kernels)
2. Memory not fully coalesced
3. GPU not fully occupied (wave quantization)
4. Cache effects (L2 misses)
5. Tensor stride/layout issues
""")

    # ============================================================
    # Part 6: Kernel launch overhead estimation
    # ============================================================
    print("\n--- Part 6: Kernel Launch Overhead ---")

    # Profile an empty kernel to estimate launch overhead
    def empty_op():
        torch.cuda.synchronize()

    # Profile single operations
    x_flat = x.view(-1, dim).contiguous()

    single_gemm_time = benchmark(lambda: x_flat @ W.t())
    double_gemm_time = benchmark(lambda: (x_flat @ W.t(), x_flat @ A.t()))

    # The difference should be roughly the incremental cost
    incremental_cost = double_gemm_time - single_gemm_time
    theoretical_lora_a_time = 2*M*rank*dim / 1979e12 * 1e6

    print(f"Single large GEMM time: {single_gemm_time:.2f} us")
    print(f"Two GEMMs time: {double_gemm_time:.2f} us")
    print(f"Incremental cost of LoRA A: {incremental_cost:.2f} us")
    print(f"Theoretical LoRA A at peak: {theoretical_lora_a_time:.2f} us")
    print(f"LoRA A overhead: {incremental_cost - theoretical_lora_a_time:.2f} us")


if __name__ == "__main__":
    main()
