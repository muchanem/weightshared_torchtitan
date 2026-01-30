#!/usr/bin/env python3
"""Compare TK LoRAFusion vs cuBLAS unfused baseline."""
import sys
import os

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

# Import TK kernel
import lorafusion_tk


def benchmark_kernel(fn, *args, warmup=10, iters=100, **kwargs):
    """Benchmark a kernel with CUDA event timing."""
    # Warmup
    for _ in range(warmup):
        _ = fn(*args, **kwargs)
    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        _ = fn(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters * 1000  # us


def unfused_xw_sb(x, W, S, B, scale):
    """Unfused: cuBLAS X@W.T + S@B.T * scale"""
    return F.linear(x, W) + F.linear(S, B) * scale


def main():
    print("=" * 70)
    print("TK LoRAFusion vs cuBLAS Unfused Benchmark")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name()}")
    print()

    # Test configs - must satisfy TK constraints: M%128=0, K%64=0, N%128=0, R=64
    configs = [
        (1024, 1024, 1024, 64),
        (2048, 1024, 1024, 64),
        (4096, 1024, 1024, 64),
        (8192, 1024, 1024, 64),
    ]

    print(f"{'Config':<25} {'cuBLAS':>12} {'TK':>12} {'Speedup':>10}")
    print("-" * 60)

    for M, K, N, R in configs:
        scale = 16.0 / R

        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        S = torch.randn(M, R, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(N, R, device="cuda", dtype=torch.bfloat16)

        # cuBLAS unfused
        cublas_time = benchmark_kernel(unfused_xw_sb, x, W, S, B, scale)

        # TK LoRAFusion: fused_forward(x, W, S, B, scale)
        tk_time = benchmark_kernel(lorafusion_tk.fused_forward, x, W, S, B, scale)

        speedup = cublas_time / tk_time
        config_str = f"{M}x{K}x{N}x{R}"
        print(f"{config_str:<25} {cublas_time:>10.1f}us {tk_time:>10.1f}us {speedup:>9.2f}x")

    print()

    # Also test correctness
    print("Correctness check (8192x1024x1024x64):")
    M, K, N, R = 8192, 1024, 1024, 64
    scale = 0.25
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    S = torch.randn(M, R, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N, R, device="cuda", dtype=torch.bfloat16)

    y_cublas = unfused_xw_sb(x, W, S, B, scale)
    y_tk = lorafusion_tk.fused_forward(x, W, S, B, scale)

    tk_diff = (y_tk - y_cublas).abs().max().item()
    print(f"  TK vs cuBLAS max diff: {tk_diff:.4f}")


if __name__ == "__main__":
    main()
