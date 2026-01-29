#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark script for LoRA fusion kernels.

This script measures performance of various LoRA implementations:
1. Unfused (4 separate kernels)
2. LoRAFusion Triton (original)
3. LoRAFusion TK (ThunderKittens port)
4. Megakernel TK (fully fused)

Usage:
    python scripts/benchmark_lora_baselines.py --kernel=unfused
    python scripts/benchmark_lora_baselines.py --kernel=lorafusion_triton
    python scripts/benchmark_lora_baselines.py --kernel=lorafusion_tk
    python scripts/benchmark_lora_baselines.py --kernel=megakernel_tk
    python scripts/benchmark_lora_baselines.py --all
"""

import argparse
import time
from typing import Callable, Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F

# Import reference implementations
from lora_reference import (
    reference_lora_forward,
    reference_lora_backward,
    unfused_lora_forward,
)


# Configuration tuples: (M, K, N, R)
# M = batch * seq_len (typically 4 * 2048 = 8192)
BENCHMARK_CONFIGS = [
    (8192, 1024, 1024, 256),  # Layer LoRA (batch=4, seq=2048)
    (8192, 1024, 1024, 192),  # Attention LoRA (smaller rank)
    (8192, 1024, 3072, 256),  # FFN w1/w3 (dim -> hidden_dim)
    (8192, 3072, 1024, 256),  # FFN w2 (hidden_dim -> dim)
]


def benchmark_kernel(
    fn: Callable,
    x: torch.Tensor,
    W: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    scale: float,
    warmup_iters: int = 10,
    benchmark_iters: int = 100,
) -> float:
    """
    Benchmark a kernel with CUDA event timing.

    Returns:
        Average time in microseconds.
    """
    # Warmup
    for _ in range(warmup_iters):
        _ = fn(x, W, A, B, scale)

    torch.cuda.synchronize()

    # Benchmark with CUDA events
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(benchmark_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(benchmark_iters)]

    for i in range(benchmark_iters):
        start_events[i].record()
        _ = fn(x, W, A, B, scale)
        end_events[i].record()

    torch.cuda.synchronize()

    # Compute average time (ms -> us)
    times = [start_events[i].elapsed_time(end_events[i]) for i in range(benchmark_iters)]
    avg_time_us = sum(times) / len(times) * 1000  # ms to us

    return avg_time_us


def benchmark_forward_backward(
    fwd_fn: Callable,
    bwd_fn: Callable,
    x: torch.Tensor,
    W: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    scale: float,
    warmup_iters: int = 10,
    benchmark_iters: int = 100,
) -> Tuple[float, float]:
    """
    Benchmark forward and backward passes separately.

    Returns:
        (forward_time_us, backward_time_us)
    """
    # Warmup
    for _ in range(warmup_iters):
        y, s = fwd_fn(x, W, A, B, scale)
        grad_y = torch.randn_like(y)
        _ = bwd_fn(grad_y, x, W, A, B, s, scale)

    torch.cuda.synchronize()

    # Benchmark forward
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(benchmark_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(benchmark_iters)]

    for i in range(benchmark_iters):
        start_events[i].record()
        y, s = fwd_fn(x, W, A, B, scale)
        end_events[i].record()

    torch.cuda.synchronize()
    fwd_times = [start_events[i].elapsed_time(end_events[i]) for i in range(benchmark_iters)]
    fwd_avg_us = sum(fwd_times) / len(fwd_times) * 1000

    # Benchmark backward
    y, s = fwd_fn(x, W, A, B, scale)
    grad_y = torch.randn_like(y)

    for i in range(benchmark_iters):
        start_events[i].record()
        _ = bwd_fn(grad_y, x, W, A, B, s, scale)
        end_events[i].record()

    torch.cuda.synchronize()
    bwd_times = [start_events[i].elapsed_time(end_events[i]) for i in range(benchmark_iters)]
    bwd_avg_us = sum(bwd_times) / len(bwd_times) * 1000

    return fwd_avg_us, bwd_avg_us


def unfused_forward_wrapper(
    x: torch.Tensor,
    W: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Wrapper that returns only Y for unfused benchmark."""
    y, _ = unfused_lora_forward(x, W, A, B, scale)
    return y


def run_unfused_benchmark(configs: List[Tuple[int, int, int, int]]) -> Dict:
    """Benchmark unfused (4 separate kernels) implementation."""
    results = {}

    for M, K, N, R in configs:
        config_key = f"{M}x{K}x{N}x{R}"
        print(f"  Config: {config_key}")

        # Create tensors
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        A = torch.randn(R, K, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(N, R, device="cuda", dtype=torch.bfloat16)
        scale = 16.0 / R

        # Benchmark forward only
        fwd_time = benchmark_kernel(unfused_forward_wrapper, x, W, A, B, scale)

        # Benchmark with backward
        fwd_time_2, bwd_time = benchmark_forward_backward(
            unfused_lora_forward, reference_lora_backward, x, W, A, B, scale
        )

        results[config_key] = {
            "forward_us": fwd_time,
            "backward_us": bwd_time,
            "total_us": fwd_time_2 + bwd_time,
        }

        print(f"    Forward: {fwd_time:.1f} us")
        print(f"    Backward: {bwd_time:.1f} us")
        print(f"    Total: {fwd_time_2 + bwd_time:.1f} us")

    return results


def run_triton_lorafusion_benchmark(configs: List[Tuple[int, int, int, int]]) -> Optional[Dict]:
    """Benchmark original Triton LoRAFusion kernels."""
    try:
        from torchtitan.models.qwen3.model.fused_lora_kernels import (
            fused_linear_lora_forward,
            fused_linear_lora_backward,
        )
    except ImportError:
        print("  Triton kernels not available, skipping...")
        return None

    results = {}

    for M, K, N, R in configs:
        config_key = f"{M}x{K}x{N}x{R}"
        print(f"  Config: {config_key}")

        # Create tensors
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        A = torch.randn(R, K, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(N, R, device="cuda", dtype=torch.bfloat16)
        scale = 16.0 / R

        def triton_fwd_wrapper(x, W, A, B, scale):
            return fused_linear_lora_forward(x, W, A, B, scale)

        # Benchmark
        fwd_time, bwd_time = benchmark_forward_backward(
            triton_fwd_wrapper, fused_linear_lora_backward, x, W, A, B, scale
        )

        results[config_key] = {
            "forward_us": fwd_time,
            "backward_us": bwd_time,
            "total_us": fwd_time + bwd_time,
        }

        print(f"    Forward: {fwd_time:.1f} us")
        print(f"    Backward: {bwd_time:.1f} us")
        print(f"    Total: {fwd_time + bwd_time:.1f} us")

    return results


def run_tk_lorafusion_benchmark(configs: List[Tuple[int, int, int, int]]) -> Optional[Dict]:
    """Benchmark ThunderKittens LoRAFusion port."""
    try:
        import lorafusion_tk
    except ImportError:
        print("  TK LoRAFusion not available (not compiled), skipping...")
        return None

    results = {}

    for M, K, N, R in configs:
        config_key = f"{M}x{K}x{N}x{R}"
        print(f"  Config: {config_key}")

        # Create tensors
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        A = torch.randn(R, K, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(N, R, device="cuda", dtype=torch.bfloat16)
        scale = 16.0 / R

        def tk_fwd(x, W, A, B, scale):
            # cuBLAS for S, TK for fused part
            S = F.linear(x, A)
            Y = lorafusion_tk.fused_forward(x, W, S, B, scale)
            return Y, S

        def tk_bwd(grad_y, x, W, A, B, S, scale):
            # Placeholder - need to implement
            return reference_lora_backward(grad_y, x, W, A, B, S, scale)

        fwd_time, bwd_time = benchmark_forward_backward(tk_fwd, tk_bwd, x, W, A, B, scale)

        results[config_key] = {
            "forward_us": fwd_time,
            "backward_us": bwd_time,
            "total_us": fwd_time + bwd_time,
        }

        print(f"    Forward: {fwd_time:.1f} us")
        print(f"    Backward: {bwd_time:.1f} us")
        print(f"    Total: {fwd_time + bwd_time:.1f} us")

    return results


def run_tk_megakernel_benchmark(configs: List[Tuple[int, int, int, int]]) -> Optional[Dict]:
    """Benchmark ThunderKittens megakernel."""
    try:
        import lora_megakernel_tk
    except ImportError:
        print("  TK Megakernel not available (not compiled), skipping...")
        return None

    results = {}

    for M, K, N, R in configs:
        config_key = f"{M}x{K}x{N}x{R}"
        print(f"  Config: {config_key}")

        # Create tensors
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        A = torch.randn(R, K, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(N, R, device="cuda", dtype=torch.bfloat16)
        scale = 16.0 / R

        def mega_fwd(x, W, A, B, scale):
            return lora_megakernel_tk.forward(x, W, A, B, scale, True)

        def mega_bwd(grad_y, x, W, A, B, S, scale):
            return lora_megakernel_tk.backward(grad_y, x, W, A, B, S, scale)

        fwd_time, bwd_time = benchmark_forward_backward(mega_fwd, mega_bwd, x, W, A, B, scale)

        results[config_key] = {
            "forward_us": fwd_time,
            "backward_us": bwd_time,
            "total_us": fwd_time + bwd_time,
        }

        print(f"    Forward: {fwd_time:.1f} us")
        print(f"    Backward: {bwd_time:.1f} us")
        print(f"    Total: {fwd_time + bwd_time:.1f} us")

    return results


def print_summary(all_results: Dict[str, Dict]) -> None:
    """Print a summary comparison table."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Get all configs
    configs = list(next(iter(all_results.values())).keys()) if all_results else []

    if not configs:
        print("No results to summarize.")
        return

    # Print header
    print(f"\n{'Config':<25}", end="")
    for kernel_name in all_results.keys():
        print(f"{kernel_name:>20}", end="")
    print()
    print("-" * (25 + 20 * len(all_results)))

    # Print each config
    baseline_key = "unfused"
    for config in configs:
        print(f"{config:<25}", end="")
        baseline = all_results.get(baseline_key, {}).get(config, {}).get("forward_us", 0)

        for kernel_name, results in all_results.items():
            if config in results:
                fwd = results[config]["forward_us"]
                if baseline > 0 and kernel_name != baseline_key:
                    speedup = baseline / fwd
                    print(f"{fwd:>12.1f}us ({speedup:>4.2f}x)", end="")
                else:
                    print(f"{fwd:>12.1f}us (base)", end="")
            else:
                print(f"{'N/A':>20}", end="")
        print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark LoRA fusion kernels")
    parser.add_argument(
        "--kernel",
        type=str,
        choices=["unfused", "lorafusion_triton", "lorafusion_tk", "megakernel_tk", "all"],
        default="all",
        help="Which kernel to benchmark",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Specific config to benchmark (e.g., '8192x1024x1024x256')",
    )
    parser.add_argument(
        "--check-correctness",
        action="store_true",
        help="Also check correctness against reference",
    )
    args = parser.parse_args()

    # Filter configs if specified
    if args.config:
        parts = list(map(int, args.config.split("x")))
        configs = [tuple(parts)]
    else:
        configs = BENCHMARK_CONFIGS

    print("=" * 80)
    print("LoRA Fusion Kernel Benchmark")
    print("=" * 80)
    print(f"Configurations: {len(configs)}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print()

    all_results = {}

    # Run benchmarks
    if args.kernel in ("unfused", "all"):
        print("\n--- Unfused (4 kernels) ---")
        all_results["unfused"] = run_unfused_benchmark(configs)

    if args.kernel in ("lorafusion_triton", "all"):
        print("\n--- LoRAFusion Triton ---")
        results = run_triton_lorafusion_benchmark(configs)
        if results:
            all_results["lorafusion_triton"] = results

    if args.kernel in ("lorafusion_tk", "all"):
        print("\n--- LoRAFusion TK ---")
        results = run_tk_lorafusion_benchmark(configs)
        if results:
            all_results["lorafusion_tk"] = results

    if args.kernel in ("megakernel_tk", "all"):
        print("\n--- Megakernel TK ---")
        results = run_tk_megakernel_benchmark(configs)
        if results:
            all_results["megakernel_tk"] = results

    # Print summary
    print_summary(all_results)


if __name__ == "__main__":
    main()
