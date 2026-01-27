#!/usr/bin/env python3
"""
Microbenchmark for LoRA modules in weight sharing.

Compares performance of:
1. Standard nn.Linear
2. SharedLinearWithLoRA (base + LoRA)
3. Fused linear+LoRA (if implemented)

Usage:
    python scripts/benchmark_lora.py
"""

import torch
import torch.utils.benchmark as benchmark
from typing import Dict, List, Tuple
import pandas as pd

from torchtitan.models.qwen3.model.weight_sharing import (
    SharedLinearWithLoRA,
    ScaledLoRAModule,
    SharedFeedForward,
)


def benchmark_linear_vs_lora(
    batch_size: int = 4,
    seq_len: int = 2048,
    dim: int = 1024,
    lora_rank: int = 256,
    num_warmup: int = 10,
    num_runs: int = 100,
) -> Dict[str, float]:
    """Benchmark nn.Linear vs SharedLinearWithLoRA."""

    # Create modules
    linear = torch.nn.Linear(dim, dim, bias=False).cuda().bfloat16()
    shared_base = torch.nn.Linear(dim, dim, bias=False).cuda().bfloat16()
    shared_lora = SharedLinearWithLoRA(shared_base, dim, dim, lora_rank=lora_rank).cuda().bfloat16()

    x = torch.randn(batch_size, seq_len, dim, device="cuda", dtype=torch.bfloat16)

    # Warmup
    for _ in range(num_warmup):
        _ = linear(x)
        _ = shared_lora(x)
    torch.cuda.synchronize()

    # Benchmark
    t_linear = benchmark.Timer(
        stmt="linear(x)",
        globals={"linear": linear, "x": x},
        label="Forward pass",
        sub_label=f"batch={batch_size}, seq={seq_len}, dim={dim}",
        description="nn.Linear",
    )

    t_shared = benchmark.Timer(
        stmt="shared_lora(x)",
        globals={"shared_lora": shared_lora, "x": x},
        label="Forward pass",
        sub_label=f"batch={batch_size}, seq={seq_len}, dim={dim}",
        description=f"SharedLinearWithLoRA (r={lora_rank})",
    )

    linear_result = t_linear.timeit(num_runs)
    shared_result = t_shared.timeit(num_runs)

    return {
        "linear_ms": linear_result.mean * 1000,
        "shared_ms": shared_result.mean * 1000,
        "overhead_pct": (shared_result.mean / linear_result.mean - 1) * 100,
        "overhead_ratio": shared_result.mean / linear_result.mean,
    }


def benchmark_forward_backward(
    batch_size: int = 4,
    seq_len: int = 2048,
    dim: int = 1024,
    lora_rank: int = 256,
    num_warmup: int = 10,
    num_runs: int = 50,
) -> Dict[str, float]:
    """Benchmark forward+backward pass."""

    # Create modules
    linear = torch.nn.Linear(dim, dim, bias=False).cuda().bfloat16()
    shared_base = torch.nn.Linear(dim, dim, bias=False).cuda().bfloat16()
    shared_lora = SharedLinearWithLoRA(shared_base, dim, dim, lora_rank=lora_rank).cuda().bfloat16()

    x_linear = torch.randn(batch_size, seq_len, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_shared = torch.randn(batch_size, seq_len, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def fwd_bwd_linear():
        out = linear(x_linear)
        loss = out.sum()
        loss.backward()
        x_linear.grad = None
        linear.weight.grad = None

    def fwd_bwd_shared():
        out = shared_lora(x_shared)
        loss = out.sum()
        loss.backward()
        x_shared.grad = None
        for p in shared_lora.parameters():
            if p.grad is not None:
                p.grad = None

    # Warmup
    for _ in range(num_warmup):
        fwd_bwd_linear()
        fwd_bwd_shared()
    torch.cuda.synchronize()

    t_linear = benchmark.Timer(
        stmt="fn()",
        globals={"fn": fwd_bwd_linear},
        label="Forward+Backward",
        sub_label=f"batch={batch_size}, seq={seq_len}, dim={dim}",
        description="nn.Linear",
    )

    t_shared = benchmark.Timer(
        stmt="fn()",
        globals={"fn": fwd_bwd_shared},
        label="Forward+Backward",
        sub_label=f"batch={batch_size}, seq={seq_len}, dim={dim}",
        description=f"SharedLinearWithLoRA (r={lora_rank})",
    )

    linear_result = t_linear.timeit(num_runs)
    shared_result = t_shared.timeit(num_runs)

    return {
        "linear_ms": linear_result.mean * 1000,
        "shared_ms": shared_result.mean * 1000,
        "overhead_pct": (shared_result.mean / linear_result.mean - 1) * 100,
        "overhead_ratio": shared_result.mean / linear_result.mean,
    }


def benchmark_swiglu_ffn(
    batch_size: int = 4,
    seq_len: int = 2048,
    dim: int = 1024,
    hidden_dim: int = 3072,
    lora_rank: int = 256,
    num_warmup: int = 10,
    num_runs: int = 50,
) -> Dict[str, float]:
    """Benchmark standard FFN vs SharedFeedForward."""

    # Standard SwiGLU FFN
    class StandardFFN(torch.nn.Module):
        def __init__(self, dim, hidden_dim):
            super().__init__()
            self.w1 = torch.nn.Linear(dim, hidden_dim, bias=False)
            self.w2 = torch.nn.Linear(hidden_dim, dim, bias=False)
            self.w3 = torch.nn.Linear(dim, hidden_dim, bias=False)

        def forward(self, x):
            return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))

    standard_ffn = StandardFFN(dim, hidden_dim).cuda().bfloat16()

    shared_weights = {
        "w1": torch.nn.Linear(dim, hidden_dim, bias=False).cuda().bfloat16(),
        "w2": torch.nn.Linear(hidden_dim, dim, bias=False).cuda().bfloat16(),
        "w3": torch.nn.Linear(dim, hidden_dim, bias=False).cuda().bfloat16(),
    }
    shared_ffn = SharedFeedForward(shared_weights, dim=dim, hidden_dim=hidden_dim, lora_rank=lora_rank).cuda().bfloat16()

    x = torch.randn(batch_size, seq_len, dim, device="cuda", dtype=torch.bfloat16)

    # Warmup
    for _ in range(num_warmup):
        _ = standard_ffn(x)
        _ = shared_ffn(x)
    torch.cuda.synchronize()

    t_standard = benchmark.Timer(
        stmt="ffn(x)",
        globals={"ffn": standard_ffn, "x": x},
        label="FFN Forward",
        sub_label=f"batch={batch_size}, seq={seq_len}, dim={dim}, hidden={hidden_dim}",
        description="StandardFFN",
    )

    t_shared = benchmark.Timer(
        stmt="ffn(x)",
        globals={"ffn": shared_ffn, "x": x},
        label="FFN Forward",
        sub_label=f"batch={batch_size}, seq={seq_len}, dim={dim}, hidden={hidden_dim}",
        description=f"SharedFeedForward (r={lora_rank})",
    )

    standard_result = t_standard.timeit(num_runs)
    shared_result = t_shared.timeit(num_runs)

    return {
        "standard_ms": standard_result.mean * 1000,
        "shared_ms": shared_result.mean * 1000,
        "overhead_pct": (shared_result.mean / standard_result.mean - 1) * 100,
        "overhead_ratio": shared_result.mean / standard_result.mean,
    }


def sweep_lora_ranks():
    """Sweep different LoRA ranks to find optimal tradeoff."""
    print("\n" + "=" * 70)
    print("LoRA Rank Sweep (Forward Only)")
    print("=" * 70)

    ranks = [8, 16, 32, 64, 128, 256, 512]
    results = []

    for rank in ranks:
        result = benchmark_linear_vs_lora(lora_rank=rank)
        results.append({
            "rank": rank,
            "linear_ms": result["linear_ms"],
            "shared_ms": result["shared_ms"],
            "overhead_pct": result["overhead_pct"],
        })
        print(f"rank={rank:4d}: Linear={result['linear_ms']:.3f}ms, SharedLoRA={result['shared_ms']:.3f}ms, Overhead={result['overhead_pct']:+.1f}%")

    return pd.DataFrame(results)


def sweep_batch_sizes():
    """Sweep different batch sizes."""
    print("\n" + "=" * 70)
    print("Batch Size Sweep (Forward Only, rank=256)")
    print("=" * 70)

    batch_sizes = [1, 2, 4, 8, 16]
    results = []

    for batch in batch_sizes:
        result = benchmark_linear_vs_lora(batch_size=batch, lora_rank=256)
        results.append({
            "batch_size": batch,
            "linear_ms": result["linear_ms"],
            "shared_ms": result["shared_ms"],
            "overhead_pct": result["overhead_pct"],
        })
        print(f"batch={batch:2d}: Linear={result['linear_ms']:.3f}ms, SharedLoRA={result['shared_ms']:.3f}ms, Overhead={result['overhead_pct']:+.1f}%")

    return pd.DataFrame(results)


def sweep_dimensions():
    """Sweep different hidden dimensions."""
    print("\n" + "=" * 70)
    print("Dimension Sweep (Forward Only, rank=256, batch=4)")
    print("=" * 70)

    dims = [512, 768, 1024, 2048, 4096]
    results = []

    for dim in dims:
        result = benchmark_linear_vs_lora(dim=dim, lora_rank=256)
        results.append({
            "dim": dim,
            "linear_ms": result["linear_ms"],
            "shared_ms": result["shared_ms"],
            "overhead_pct": result["overhead_pct"],
        })
        print(f"dim={dim:5d}: Linear={result['linear_ms']:.3f}ms, SharedLoRA={result['shared_ms']:.3f}ms, Overhead={result['overhead_pct']:+.1f}%")

    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("LoRA Microbenchmark Suite")
    print("=" * 70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA device: {torch.cuda.get_device_name()}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Basic benchmark
    print("\n" + "=" * 70)
    print("Basic Benchmark (batch=4, seq=2048, dim=1024, rank=256)")
    print("=" * 70)

    fwd_result = benchmark_linear_vs_lora()
    print(f"\nForward Pass:")
    print(f"  nn.Linear:           {fwd_result['linear_ms']:.3f} ms")
    print(f"  SharedLinearWithLoRA: {fwd_result['shared_ms']:.3f} ms")
    print(f"  Overhead:            {fwd_result['overhead_pct']:+.1f}%")

    fwd_bwd_result = benchmark_forward_backward()
    print(f"\nForward + Backward:")
    print(f"  nn.Linear:           {fwd_bwd_result['linear_ms']:.3f} ms")
    print(f"  SharedLinearWithLoRA: {fwd_bwd_result['shared_ms']:.3f} ms")
    print(f"  Overhead:            {fwd_bwd_result['overhead_pct']:+.1f}%")

    ffn_result = benchmark_swiglu_ffn()
    print(f"\nSwiGLU FFN Forward:")
    print(f"  StandardFFN:      {ffn_result['standard_ms']:.3f} ms")
    print(f"  SharedFeedForward: {ffn_result['shared_ms']:.3f} ms")
    print(f"  Overhead:         {ffn_result['overhead_pct']:+.1f}%")

    # Sweeps
    sweep_lora_ranks()
    sweep_batch_sizes()
    sweep_dimensions()

    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
