#!/usr/bin/env python3
"""
Prototype fused Linear+LoRA operation.

This module implements a fused forward pass that computes:
    output = x @ W.T + (x @ A.T) @ B.T * scale

Instead of:
    base_output = x @ W.T          # write to memory
    lora_output = (x @ A.T) @ B.T  # write to memory
    output = base_output + lora_output * scale  # read both, write result

The fused version aims to reduce memory traffic by keeping intermediates in registers.

Usage:
    python scripts/fused_lora_prototype.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.benchmark as benchmark


class FusedLinearLoRA(torch.autograd.Function):
    """
    Fused Linear + LoRA operation.

    Forward: output = x @ W.T + (x @ A.T) @ B.T * scale
    Backward: Computes gradients for x, W, A, B
    """

    @staticmethod
    def forward(ctx, x, W, A, B, scale):
        # Compute base output
        base_out = F.linear(x, W)  # x @ W.T

        # Compute LoRA output
        # lora_out = (x @ A.T) @ B.T * scale
        xA = F.linear(x, A)  # x @ A.T
        lora_out = F.linear(xA, B) * scale  # (x @ A.T) @ B.T * scale

        # Combined output
        output = base_out + lora_out

        # Save for backward
        ctx.save_for_backward(x, W, A, B, xA)
        ctx.scale = scale

        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, W, A, B, xA = ctx.saved_tensors
        scale = ctx.scale

        # Gradient for x: grad_output @ W + grad_output @ B @ A * scale
        grad_x = grad_output @ W
        grad_x = grad_x + (grad_output @ B @ A) * scale

        # Gradient for W: grad_output.T @ x
        grad_W = grad_output.reshape(-1, grad_output.shape[-1]).T @ x.reshape(-1, x.shape[-1])

        # Gradient for A: (grad_output @ B).T @ x * scale
        grad_lora = grad_output * scale
        grad_xA = grad_lora @ B  # gradient w.r.t. xA
        grad_A = grad_xA.reshape(-1, grad_xA.shape[-1]).T @ x.reshape(-1, x.shape[-1])

        # Gradient for B: grad_output.T @ xA * scale
        grad_B = grad_lora.reshape(-1, grad_lora.shape[-1]).T @ xA.reshape(-1, xA.shape[-1])

        return grad_x, grad_W, grad_A, grad_B, None


def fused_linear_lora(x, W, A, B, scale):
    """Convenience function for FusedLinearLoRA."""
    return FusedLinearLoRA.apply(x, W, A, B, scale)


class FusedLinearLoRAModule(nn.Module):
    """
    Module wrapper for fused Linear+LoRA.

    This is a drop-in replacement for SharedLinearWithLoRA that uses
    the fused autograd function.
    """

    def __init__(self, shared_linear, in_dim, out_dim, lora_rank=256, alpha=None):
        super().__init__()
        self.shared_linear = shared_linear
        self.lora_rank = lora_rank
        self.alpha = alpha if alpha is not None else lora_rank
        self.scale = self.alpha / lora_rank

        # LoRA matrices (same initialization as original)
        self.lora_A = nn.Parameter(torch.randn(lora_rank, in_dim) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_dim, lora_rank))

    def forward(self, x):
        return fused_linear_lora(
            x,
            self.shared_linear.weight,
            self.lora_A,
            self.lora_B,
            self.scale
        )


class UnfusedLinearLoRAModule(nn.Module):
    """
    Unfused version for comparison (matches original SharedLinearWithLoRA).
    """

    def __init__(self, shared_linear, in_dim, out_dim, lora_rank=256, alpha=None):
        super().__init__()
        self.shared_linear = shared_linear
        self.lora_rank = lora_rank
        self.alpha = alpha if alpha is not None else lora_rank
        self.scale = self.alpha / lora_rank

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(lora_rank, in_dim) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_dim, lora_rank))

    def forward(self, x):
        base_out = self.shared_linear(x)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
        return base_out + lora_out


def verify_correctness():
    """Verify that fused and unfused produce same results."""
    print("\n" + "=" * 60)
    print("Correctness Verification")
    print("=" * 60)

    torch.manual_seed(42)

    dim, rank = 1024, 256
    shared_linear = nn.Linear(dim, dim, bias=False).cuda().bfloat16()

    fused = FusedLinearLoRAModule(shared_linear, dim, dim, lora_rank=rank).cuda().bfloat16()
    unfused = UnfusedLinearLoRAModule(shared_linear, dim, dim, lora_rank=rank).cuda().bfloat16()

    # Copy LoRA weights
    with torch.no_grad():
        unfused.lora_A.copy_(fused.lora_A)
        unfused.lora_B.copy_(fused.lora_B)

    x = torch.randn(4, 2048, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_fused = x.clone().detach().requires_grad_(True)
    x_unfused = x.clone().detach().requires_grad_(True)

    # Forward
    out_fused = fused(x_fused)
    out_unfused = unfused(x_unfused)

    fwd_diff = (out_fused - out_unfused).abs().max().item()
    print(f"Forward max diff: {fwd_diff:.2e}")

    # Backward
    grad_out = torch.randn_like(out_fused)
    out_fused.backward(grad_out)
    out_unfused.backward(grad_out)

    grad_x_diff = (x_fused.grad - x_unfused.grad).abs().max().item()
    grad_A_diff = (fused.lora_A.grad - unfused.lora_A.grad).abs().max().item()
    grad_B_diff = (fused.lora_B.grad - unfused.lora_B.grad).abs().max().item()

    print(f"Backward grad_x max diff: {grad_x_diff:.2e}")
    print(f"Backward grad_A max diff: {grad_A_diff:.2e}")
    print(f"Backward grad_B max diff: {grad_B_diff:.2e}")

    # Check if diffs are within tolerance (bfloat16 has ~3 digits of precision)
    tol = 1e-2
    all_pass = all([
        fwd_diff < tol,
        grad_x_diff < tol,
        grad_A_diff < tol,
        grad_B_diff < tol,
    ])

    if all_pass:
        print("\n✓ All correctness checks PASSED")
    else:
        print("\n✗ Some correctness checks FAILED")

    return all_pass


def benchmark_fused_vs_unfused(
    batch_size=4,
    seq_len=2048,
    dim=1024,
    lora_rank=256,
    num_warmup=10,
    num_runs=100,
):
    """Benchmark fused vs unfused implementations."""
    print("\n" + "=" * 60)
    print(f"Performance Benchmark (batch={batch_size}, seq={seq_len}, dim={dim}, rank={lora_rank})")
    print("=" * 60)

    shared_linear = nn.Linear(dim, dim, bias=False).cuda().bfloat16()

    fused = FusedLinearLoRAModule(shared_linear, dim, dim, lora_rank=lora_rank).cuda().bfloat16()
    unfused = UnfusedLinearLoRAModule(shared_linear, dim, dim, lora_rank=lora_rank).cuda().bfloat16()

    # Copy weights for fair comparison
    with torch.no_grad():
        unfused.lora_A.copy_(fused.lora_A)
        unfused.lora_B.copy_(fused.lora_B)

    x = torch.randn(batch_size, seq_len, dim, device="cuda", dtype=torch.bfloat16)

    # Warmup
    for _ in range(num_warmup):
        _ = fused(x)
        _ = unfused(x)
    torch.cuda.synchronize()

    # Forward benchmark
    t_fused = benchmark.Timer(
        stmt="module(x)",
        globals={"module": fused, "x": x},
    )
    t_unfused = benchmark.Timer(
        stmt="module(x)",
        globals={"module": unfused, "x": x},
    )

    fused_fwd = t_fused.timeit(num_runs)
    unfused_fwd = t_unfused.timeit(num_runs)

    print(f"\nForward Only:")
    print(f"  Unfused: {unfused_fwd.mean * 1000:.3f} ms")
    print(f"  Fused:   {fused_fwd.mean * 1000:.3f} ms")
    print(f"  Speedup: {unfused_fwd.mean / fused_fwd.mean:.2f}x")

    # Forward + Backward benchmark
    x_fused = torch.randn(batch_size, seq_len, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_unfused = torch.randn(batch_size, seq_len, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def fwd_bwd_fused():
        out = fused(x_fused)
        out.sum().backward()
        x_fused.grad = None
        for p in fused.parameters():
            if p.grad is not None:
                p.grad = None

    def fwd_bwd_unfused():
        out = unfused(x_unfused)
        out.sum().backward()
        x_unfused.grad = None
        for p in unfused.parameters():
            if p.grad is not None:
                p.grad = None

    # Warmup
    for _ in range(num_warmup):
        fwd_bwd_fused()
        fwd_bwd_unfused()
    torch.cuda.synchronize()

    t_fused_bwd = benchmark.Timer(stmt="fn()", globals={"fn": fwd_bwd_fused})
    t_unfused_bwd = benchmark.Timer(stmt="fn()", globals={"fn": fwd_bwd_unfused})

    fused_bwd = t_fused_bwd.timeit(num_runs // 2)
    unfused_bwd = t_unfused_bwd.timeit(num_runs // 2)

    print(f"\nForward + Backward:")
    print(f"  Unfused: {unfused_bwd.mean * 1000:.3f} ms")
    print(f"  Fused:   {fused_bwd.mean * 1000:.3f} ms")
    print(f"  Speedup: {unfused_bwd.mean / fused_bwd.mean:.2f}x")

    return {
        "fwd_unfused_ms": unfused_fwd.mean * 1000,
        "fwd_fused_ms": fused_fwd.mean * 1000,
        "fwd_speedup": unfused_fwd.mean / fused_fwd.mean,
        "bwd_unfused_ms": unfused_bwd.mean * 1000,
        "bwd_fused_ms": fused_bwd.mean * 1000,
        "bwd_speedup": unfused_bwd.mean / fused_bwd.mean,
    }


def main():
    print("=" * 60)
    print("Fused Linear+LoRA Prototype")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA device: {torch.cuda.get_device_name()}")

    # Verify correctness
    correct = verify_correctness()
    if not correct:
        print("\nAborting benchmarks due to correctness failure")
        return

    # Benchmark
    results = benchmark_fused_vs_unfused()

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if results["fwd_speedup"] > 1.0:
        print(f"✓ Forward speedup: {results['fwd_speedup']:.2f}x")
    else:
        print(f"✗ Forward slower: {results['fwd_speedup']:.2f}x")

    if results["bwd_speedup"] > 1.0:
        print(f"✓ Fwd+Bwd speedup: {results['bwd_speedup']:.2f}x")
    else:
        print(f"✗ Fwd+Bwd slower: {results['bwd_speedup']:.2f}x")

    print("\nNote: This prototype uses the same PyTorch ops but with")
    print("a custom autograd function. For actual speedup, we need")
    print("custom Triton kernels that fuse the operations at the")
    print("kernel level to avoid intermediate memory writes.")


if __name__ == "__main__":
    main()
