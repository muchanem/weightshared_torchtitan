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


def benchmark_kernel(fn, *args, warmup=3, iters=10, **kwargs):
    """Benchmark a kernel with CUDA event timing."""
    # Warmup with sync between each call
    for _ in range(warmup):
        _ = fn(*args, **kwargs)
        torch.cuda.synchronize()

    # Benchmark with sync between each call
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = fn(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # us

    return sum(times) / len(times)


def unfused_xw_sb(x, W, S, B, scale):
    """Unfused: cuBLAS X@W.T + S@B.T * scale"""
    return F.linear(x, W) + F.linear(S, B) * scale


def main():
    import sys
    print("=" * 70); sys.stdout.flush()
    print("TK LoRAFusion vs cuBLAS Unfused Benchmark"); sys.stdout.flush()
    print("=" * 70); sys.stdout.flush()
    print(f"Device: {torch.cuda.get_device_name()}"); sys.stdout.flush()
    print()

    # Test configs - must satisfy TK constraints: M%128=0, K%64=0, N%128=0, R=64
    configs = [
        (512, 256, 512, 64),     # Small
        (1024, 512, 1024, 64),   # Medium (verified working)
    ]

    print(f"{'Config':<25} {'cuBLAS':>12} {'TK':>12} {'Speedup':>10}"); sys.stdout.flush()
    print("-" * 60); sys.stdout.flush()

    for M, K, N, R in configs:
        print(f"Testing {M}x{K}x{N}x{R}..."); sys.stdout.flush()
        scale = 16.0 / R

        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        S = torch.randn(M, R, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(N, R, device="cuda", dtype=torch.bfloat16)

        import sys
        print("  Running cuBLAS..."); sys.stdout.flush()
        # cuBLAS unfused
        cublas_time = benchmark_kernel(unfused_xw_sb, x, W, S, B, scale)
        print(f"  cuBLAS done: {cublas_time:.1f}us"); sys.stdout.flush()

        print("  Running TK..."); sys.stdout.flush()
        # TK LoRAFusion: fused_forward(x, W, S, B, scale)
        tk_time = benchmark_kernel(lorafusion_tk.fused_forward, x, W, S, B, scale)
        print(f"  TK done: {tk_time:.1f}us"); sys.stdout.flush()

        speedup = cublas_time / tk_time
        config_str = f"{M}x{K}x{N}x{R}"
        print(f"{config_str:<25} {cublas_time:>10.1f}us {tk_time:>10.1f}us {speedup:>9.2f}x"); sys.stdout.flush()

    print()

    # Also test correctness
    print("Correctness check (1024x512x1024x64):")
    M, K, N, R = 1024, 512, 1024, 64
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
