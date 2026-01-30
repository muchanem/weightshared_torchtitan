#!/usr/bin/env python3
"""Benchmark full LoRA forward: Y = X@W.T + (X@A.T)@B.T * scale"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import lorafusion_tk

print("=" * 70)
print("Full LoRA Forward Benchmark: Y = X@W.T + (X@A.T)@B.T * scale")
print("=" * 70)
print(f"Device: {torch.cuda.get_device_name()}")
print()

# Must satisfy: M % 128 == 0, K % 64 == 0, N % 128 == 0, R == 64
configs = [
    (1024, 512, 1024, 64),
    (2048, 512, 1024, 64),
    (4096, 1024, 1024, 64),
    (8192, 1024, 1024, 64),
]

def benchmark(fn, iters=20):
    """Benchmark with CUDA events."""
    # Warmup
    for _ in range(5):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters * 1000  # us

print(f"{'Config':<25} {'Unfused':>10} {'Compiled':>10} {'LoRAFusion':>10} {'vs Unfused':>10} {'vs Compiled':>11}")
print("-" * 80)

for M, K, N, R in configs:
    scale = 16.0 / R

    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    A = torch.randn(R, K, device="cuda", dtype=torch.bfloat16)  # A is (R, K)
    B = torch.randn(N, R, device="cuda", dtype=torch.bfloat16)  # B is (N, R)

    # Unfused: 4 kernels
    # 1. X @ W.T
    # 2. X @ A.T -> S
    # 3. S @ B.T
    # 4. add + scale
    def unfused():
        return F.linear(x, W) + F.linear(F.linear(x, A), B) * scale

    # torch.compile'd unfused
    compiled = torch.compile(unfused, mode="max-autotune")

    # LoRAFusion: 2 kernels
    # 1. X @ A.T -> S (cuBLAS)
    # 2. X @ W.T + S @ B.T * scale (TK fused)
    def lorafusion():
        S = F.linear(x, A)  # cuBLAS
        return lorafusion_tk.fused_forward(x, W, S, B, scale)

    unfused_us = benchmark(unfused)
    compiled_us = benchmark(compiled)
    lorafusion_us = benchmark(lorafusion)

    speedup_unfused = unfused_us / lorafusion_us
    speedup_compiled = compiled_us / lorafusion_us

    config_str = f"{M}x{K}x{N}x{R}"
    print(f"{config_str:<25} {unfused_us:>8.1f}us {compiled_us:>8.1f}us {lorafusion_us:>8.1f}us {speedup_unfused:>9.2f}x {speedup_compiled:>10.2f}x")

    # Verify correctness
    y_ref = unfused()
    y_tk = lorafusion()
    torch.cuda.synchronize()
    max_diff = (y_ref - y_tk).abs().max().item()
    if max_diff > 2.0:
        print(f"  WARNING: max diff = {max_diff:.2f}")

print()
print("Unfused = 4 kernels: X@W.T, X@A.T, S@B.T, add+scale")
print("Compiled = torch.compile(unfused, mode='max-autotune')")
print("LoRAFusion = 2 kernels: X@A.T (cuBLAS), then TK fused (X@W.T + S@B.T*scale)")
