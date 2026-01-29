#!/usr/bin/env python3
"""
Test script for LoRAFusion ThunderKittens kernel.
"""

import sys
import torch
import torch.nn.functional as F

# Add scripts to path for reference implementations
sys.path.insert(0, "../../scripts")
from lora_reference import reference_lora_forward


def test_correctness():
    """Test that TK kernel matches reference."""
    try:
        import lorafusion_tk
    except ImportError:
        print("lorafusion_tk not compiled. Run 'make' first.")
        return False

    print("Testing LoRAFusion TK kernel correctness...")

    # Test configurations
    configs = [
        (256, 256, 256, 64),    # Small
        (1024, 512, 512, 128),  # Medium
        (8192, 1024, 1024, 256),  # Full size
    ]

    all_passed = True

    for M, K, N, R in configs:
        print(f"  Config: M={M}, K={K}, N={N}, R={R}")

        # Create inputs
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        A = torch.randn(R, K, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(N, R, device="cuda", dtype=torch.bfloat16)
        scale = 16.0 / R

        # Reference: full forward
        y_ref, s_ref = reference_lora_forward(x, W, A, B, scale)

        # LoRAFusion style: cuBLAS for S, TK for fused
        S = F.linear(x, A)  # cuBLAS
        y_tk = lorafusion_tk.fused_forward(x, W, S, B, scale)

        # Compare
        max_diff = (y_tk - y_ref).abs().max().item()
        passed = max_diff < 0.1  # bf16 tolerance

        status = "PASS" if passed else "FAIL"
        print(f"    Max diff: {max_diff:.6f} [{status}]")

        if not passed:
            all_passed = False

    return all_passed


def test_performance():
    """Benchmark TK kernel vs unfused."""
    try:
        import lorafusion_tk
    except ImportError:
        print("lorafusion_tk not compiled. Run 'make' first.")
        return

    print("\nBenchmarking LoRAFusion TK kernel...")

    M, K, N, R = 8192, 1024, 1024, 256
    scale = 16.0 / R

    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    A = torch.randn(R, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N, R, device="cuda", dtype=torch.bfloat16)

    # Pre-compute S
    S = F.linear(x, A)

    # Warmup
    for _ in range(10):
        _ = lorafusion_tk.fused_forward(x, W, S, B, scale)
    torch.cuda.synchronize()

    # Benchmark TK fused
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    iters = 100
    start.record()
    for _ in range(iters):
        _ = lorafusion_tk.fused_forward(x, W, S, B, scale)
    end.record()
    torch.cuda.synchronize()

    tk_time = start.elapsed_time(end) / iters * 1000  # us

    # Benchmark unfused (just the fused part: X@W + S@B*scale)
    for _ in range(10):
        _ = F.linear(x, W) + F.linear(S, B) * scale
    torch.cuda.synchronize()

    start.record()
    for _ in range(iters):
        _ = F.linear(x, W) + F.linear(S, B) * scale
    end.record()
    torch.cuda.synchronize()

    unfused_time = start.elapsed_time(end) / iters * 1000  # us

    print(f"  Config: M={M}, K={K}, N={N}, R={R}")
    print(f"  TK fused:  {tk_time:.1f} us")
    print(f"  Unfused:   {unfused_time:.1f} us")
    print(f"  Speedup:   {unfused_time/tk_time:.2f}x")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-correctness", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    if args.check_correctness or not (args.check_correctness or args.benchmark):
        success = test_correctness()
        if not success:
            sys.exit(1)

    if args.benchmark or not (args.check_correctness or args.benchmark):
        test_performance()

    print("\nAll tests completed!")
