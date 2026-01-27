#!/usr/bin/env python3
"""
Test suite for fused Linear+LoRA Triton kernels.

This script verifies correctness and benchmarks performance of the fused
Linear+LoRA operations compared to the unfused reference implementation.

Tests:
1. Forward pass correctness
2. Backward pass correctness (all gradients)
3. torch.compile compatibility
4. Performance benchmarks

Usage:
    uv run python scripts/test_fused_lora.py
"""

import argparse
import sys
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.benchmark as benchmark


def unfused_linear_lora(
    x: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Reference unfused implementation."""
    base_out = F.linear(x, w)
    lora_out = F.linear(F.linear(x, a), b) * scale
    return base_out + lora_out


def test_forward_correctness(
    batch_size: int = 4,
    seq_len: int = 2048,
    in_dim: int = 1024,
    out_dim: int = 1024,
    lora_rank: int = 256,
    rtol: float = 1e-2,
    atol: float = 1e-2,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[bool, float]:
    """Test forward pass correctness."""
    print(f"\n{'='*60}")
    print("Forward Pass Correctness Test")
    print(f"{'='*60}")
    print(f"Shape: batch={batch_size}, seq={seq_len}, in={in_dim}, out={out_dim}, rank={lora_rank}")

    # Import fused implementation
    try:
        from torchtitan.models.qwen3.model.fused_lora_module import fused_linear_lora
    except ImportError as e:
        print(f"ERROR: Could not import fused_linear_lora: {e}")
        return False, float('inf')

    # Create random tensors
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, in_dim, device=device, dtype=dtype)
    w = torch.randn(out_dim, in_dim, device=device, dtype=dtype) * 0.02
    a = torch.randn(lora_rank, in_dim, device=device, dtype=dtype) * 0.02
    b = torch.randn(out_dim, lora_rank, device=device, dtype=dtype) * 0.02
    scale = 2.0 / lora_rank

    # Compute both versions
    y_ref = unfused_linear_lora(x, w, a, b, scale)
    y_fused = fused_linear_lora(x, w, a, b, scale)

    # Compare
    max_diff = (y_ref - y_fused).abs().max().item()
    mean_diff = (y_ref - y_fused).abs().mean().item()

    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")

    passed = torch.allclose(y_ref, y_fused, rtol=rtol, atol=atol)
    if passed:
        print("  PASSED")
    else:
        print("  FAILED")

    return passed, max_diff


def test_backward_correctness(
    batch_size: int = 4,
    seq_len: int = 512,
    in_dim: int = 1024,
    out_dim: int = 1024,
    lora_rank: int = 128,
    rtol: float = 2e-2,
    # atol based on bf16 error analysis: eps * sqrt(K) * |dY| * |W| * 6_sigma
    # With K=1024, eps=2^-7, and typical randn values: ~1.0
    atol: float = 1.0,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[bool, dict]:
    """Test backward pass correctness for all gradients."""
    print(f"\n{'='*60}")
    print("Backward Pass Correctness Test")
    print(f"{'='*60}")
    print(f"Shape: batch={batch_size}, seq={seq_len}, in={in_dim}, out={out_dim}, rank={lora_rank}")

    try:
        from torchtitan.models.qwen3.model.fused_lora_module import fused_linear_lora
    except ImportError as e:
        print(f"ERROR: Could not import fused_linear_lora: {e}")
        return False, {}

    torch.manual_seed(42)

    # Create tensors with gradients
    x_ref = torch.randn(batch_size, seq_len, in_dim, device=device, dtype=dtype, requires_grad=True)
    w_ref = torch.randn(out_dim, in_dim, device=device, dtype=dtype, requires_grad=True)
    a_ref = torch.randn(lora_rank, in_dim, device=device, dtype=dtype, requires_grad=True)
    b_ref = torch.randn(out_dim, lora_rank, device=device, dtype=dtype, requires_grad=True)

    # Clone for fused version
    x_fused = x_ref.detach().clone().requires_grad_(True)
    w_fused = w_ref.detach().clone().requires_grad_(True)
    a_fused = a_ref.detach().clone().requires_grad_(True)
    b_fused = b_ref.detach().clone().requires_grad_(True)

    scale = 2.0 / lora_rank

    # Forward
    y_ref = unfused_linear_lora(x_ref, w_ref, a_ref, b_ref, scale)
    y_fused = fused_linear_lora(x_fused, w_fused, a_fused, b_fused, scale)

    # Backward
    grad_output = torch.randn_like(y_ref)
    y_ref.backward(grad_output)
    y_fused.backward(grad_output)

    # Compare gradients
    diffs = {}
    all_passed = True

    grad_names = [
        ("grad_x", x_ref.grad, x_fused.grad),
        ("grad_w", w_ref.grad, w_fused.grad),
        ("grad_a", a_ref.grad, a_fused.grad),
        ("grad_b", b_ref.grad, b_fused.grad),
    ]

    for name, ref_grad, fused_grad in grad_names:
        max_diff = (ref_grad - fused_grad).abs().max().item()
        mean_diff = (ref_grad - fused_grad).abs().mean().item()
        passed = torch.allclose(ref_grad, fused_grad, rtol=rtol, atol=atol)

        diffs[name] = max_diff
        all_passed = all_passed and passed

        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e} - {status}")

    if all_passed:
        print("  All gradients PASSED")
    else:
        print("  Some gradients FAILED")

    return all_passed, diffs


def test_module_integration(
    batch_size: int = 4,
    seq_len: int = 512,
    dim: int = 1024,
    lora_rank: int = 128,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> bool:
    """Test FusedSharedLinearWithLoRA module integration."""
    print(f"\n{'='*60}")
    print("Module Integration Test")
    print(f"{'='*60}")

    try:
        from torchtitan.models.qwen3.model.fused_lora_module import FusedSharedLinearWithLoRA
        from torchtitan.models.qwen3.model.weight_sharing import SharedLinearWithLoRA
    except ImportError as e:
        print(f"ERROR: Could not import modules: {e}")
        return False

    torch.manual_seed(42)

    # Create shared base linear
    shared_linear = nn.Linear(dim, dim, bias=False).to(device=device, dtype=dtype)

    # Create both module versions
    unfused = SharedLinearWithLoRA(shared_linear, dim, dim, lora_rank).to(device=device, dtype=dtype)
    fused = FusedSharedLinearWithLoRA(shared_linear, dim, dim, lora_rank).to(device=device, dtype=dtype)

    # Copy LoRA weights
    with torch.no_grad():
        fused.lora.w_a.weight.copy_(unfused.lora.w_a.weight)
        fused.lora.w_b.weight.copy_(unfused.lora.w_b.weight)

    # Test forward
    x = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype)
    y_unfused = unfused(x)
    y_fused = fused(x)

    max_diff = (y_unfused - y_fused).abs().max().item()
    passed = torch.allclose(y_unfused, y_fused, rtol=1e-2, atol=1e-2)

    print(f"  Forward max diff: {max_diff:.2e}")
    print(f"  Module integration: {'PASSED' if passed else 'FAILED'}")

    return passed


def test_torch_compile_compatibility(
    batch_size: int = 4,
    seq_len: int = 512,
    dim: int = 1024,
    lora_rank: int = 128,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> bool:
    """Test torch.compile compatibility."""
    print(f"\n{'='*60}")
    print("torch.compile Compatibility Test")
    print(f"{'='*60}")

    try:
        from torchtitan.models.qwen3.model.fused_lora_module import FusedSharedLinearWithLoRA
    except ImportError as e:
        print(f"ERROR: Could not import modules: {e}")
        return False

    torch.manual_seed(42)

    shared_linear = nn.Linear(dim, dim, bias=False).to(device=device, dtype=dtype)
    module = FusedSharedLinearWithLoRA(shared_linear, dim, dim, lora_rank, use_fused=True).to(device=device, dtype=dtype)

    x = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype, requires_grad=True)

    # Eager forward/backward
    y_eager = module(x)
    y_eager.sum().backward()
    grad_eager = x.grad.clone()
    x.grad = None

    # Try to compile
    try:
        compiled_module = torch.compile(module)
        y_compiled = compiled_module(x)
        y_compiled.sum().backward()
        grad_compiled = x.grad.clone()

        max_diff_fwd = (y_eager - y_compiled).abs().max().item()
        max_diff_grad = (grad_eager - grad_compiled).abs().max().item()

        print(f"  Forward max diff (eager vs compiled): {max_diff_fwd:.2e}")
        print(f"  Grad max diff (eager vs compiled): {max_diff_grad:.2e}")
        print("  torch.compile: PASSED")
        return True

    except Exception as e:
        print(f"  torch.compile failed: {e}")
        print("  torch.compile: FAILED (but module still works in eager mode)")
        return False


def benchmark_performance(
    batch_size: int = 4,
    seq_len: int = 2048,
    dim: int = 1024,
    lora_rank: int = 256,
    num_warmup: int = 10,
    num_runs: int = 100,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """Benchmark fused vs unfused performance."""
    print(f"\n{'='*60}")
    print("Performance Benchmark")
    print(f"{'='*60}")
    print(f"Shape: batch={batch_size}, seq={seq_len}, dim={dim}, rank={lora_rank}")

    try:
        from torchtitan.models.qwen3.model.fused_lora_module import fused_linear_lora
    except ImportError as e:
        print(f"ERROR: Could not import fused_linear_lora: {e}")
        return {}

    torch.manual_seed(42)

    x = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype)
    w = torch.randn(dim, dim, device=device, dtype=dtype) * 0.02
    a = torch.randn(lora_rank, dim, device=device, dtype=dtype) * 0.02
    b = torch.randn(dim, lora_rank, device=device, dtype=dtype) * 0.02
    scale = 2.0 / lora_rank

    # Warmup
    for _ in range(num_warmup):
        _ = unfused_linear_lora(x, w, a, b, scale)
        _ = fused_linear_lora(x, w, a, b, scale)
    torch.cuda.synchronize()

    # Benchmark forward
    t_unfused = benchmark.Timer(
        stmt="unfused_linear_lora(x, w, a, b, scale)",
        globals={"unfused_linear_lora": unfused_linear_lora, "x": x, "w": w, "a": a, "b": b, "scale": scale},
    )
    t_fused = benchmark.Timer(
        stmt="fused_linear_lora(x, w, a, b, scale)",
        globals={"fused_linear_lora": fused_linear_lora, "x": x, "w": w, "a": a, "b": b, "scale": scale},
    )

    unfused_result = t_unfused.timeit(num_runs)
    fused_result = t_fused.timeit(num_runs)

    fwd_speedup = unfused_result.mean / fused_result.mean

    print(f"\nForward Pass:")
    print(f"  Unfused: {unfused_result.mean * 1000:.3f} ms")
    print(f"  Fused:   {fused_result.mean * 1000:.3f} ms")
    print(f"  Speedup: {fwd_speedup:.2f}x")

    # Benchmark forward + backward
    x_unfused = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype, requires_grad=True)
    x_fused = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype, requires_grad=True)
    w_param = nn.Parameter(w.clone())
    a_param = nn.Parameter(a.clone())
    b_param = nn.Parameter(b.clone())

    def fwd_bwd_unfused():
        out = unfused_linear_lora(x_unfused, w_param, a_param, b_param, scale)
        out.sum().backward()
        x_unfused.grad = None
        w_param.grad = None
        a_param.grad = None
        b_param.grad = None

    def fwd_bwd_fused():
        out = fused_linear_lora(x_fused, w_param, a_param, b_param, scale)
        out.sum().backward()
        x_fused.grad = None
        w_param.grad = None
        a_param.grad = None
        b_param.grad = None

    # Warmup
    for _ in range(num_warmup):
        fwd_bwd_unfused()
        fwd_bwd_fused()
    torch.cuda.synchronize()

    t_unfused_bwd = benchmark.Timer(stmt="fn()", globals={"fn": fwd_bwd_unfused})
    t_fused_bwd = benchmark.Timer(stmt="fn()", globals={"fn": fwd_bwd_fused})

    unfused_bwd_result = t_unfused_bwd.timeit(num_runs // 2)
    fused_bwd_result = t_fused_bwd.timeit(num_runs // 2)

    bwd_speedup = unfused_bwd_result.mean / fused_bwd_result.mean

    print(f"\nForward + Backward:")
    print(f"  Unfused: {unfused_bwd_result.mean * 1000:.3f} ms")
    print(f"  Fused:   {fused_bwd_result.mean * 1000:.3f} ms")
    print(f"  Speedup: {bwd_speedup:.2f}x")

    return {
        "fwd_unfused_ms": unfused_result.mean * 1000,
        "fwd_fused_ms": fused_result.mean * 1000,
        "fwd_speedup": fwd_speedup,
        "bwd_unfused_ms": unfused_bwd_result.mean * 1000,
        "bwd_fused_ms": fused_bwd_result.mean * 1000,
        "bwd_speedup": bwd_speedup,
    }


def main():
    parser = argparse.ArgumentParser(description="Test fused Linear+LoRA kernels")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--seq", type=int, default=512, help="Sequence length")
    parser.add_argument("--dim", type=int, default=1024, help="Model dimension")
    parser.add_argument("--rank", type=int, default=128, help="LoRA rank")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip benchmarks")
    parser.add_argument("--skip-compile", action="store_true", help="Skip torch.compile test")
    args = parser.parse_args()

    print("=" * 60)
    print("Fused Linear+LoRA Kernel Test Suite")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    print(f"CUDA device: {torch.cuda.get_device_name()}")

    results = {
        "forward": False,
        "backward": False,
        "module": False,
        "compile": False,
    }

    # Test forward correctness
    results["forward"], _ = test_forward_correctness(
        batch_size=args.batch, seq_len=args.seq, in_dim=args.dim, out_dim=args.dim, lora_rank=args.rank
    )

    # Test backward correctness
    results["backward"], _ = test_backward_correctness(
        batch_size=args.batch, seq_len=args.seq, in_dim=args.dim, out_dim=args.dim, lora_rank=args.rank
    )

    # Test module integration
    results["module"] = test_module_integration(
        batch_size=args.batch, seq_len=args.seq, dim=args.dim, lora_rank=args.rank
    )

    # Test torch.compile compatibility
    if not args.skip_compile:
        results["compile"] = test_torch_compile_compatibility(
            batch_size=args.batch, seq_len=args.seq, dim=args.dim, lora_rank=args.rank
        )

    # Performance benchmarks
    if not args.skip_benchmark:
        benchmark_performance(
            batch_size=args.batch, seq_len=args.seq * 4, dim=args.dim, lora_rank=args.rank
        )

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print("=" * 60)
    all_passed = all(results.values())
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {test_name}: {status}")

    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
