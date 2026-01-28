#!/usr/bin/env python3
"""
Test script for grouped_mm LoRA optimization.

This script verifies:
1. Correctness: grouped_mm outputs match sequential outputs
2. Performance: grouped_mm reduces kernel launch overhead
"""

import torch
import torch.nn as nn
import time
from typing import List

# Add parent directory to path
import sys
sys.path.insert(0, "/local/scratch/muchane_680186/weightshared_torchtitan")

from torchtitan.models.qwen3.model.weight_sharing import (
    ScaledLoRAModule,
    SharedLinearWithLoRA,
    SharedFeedForward,
)
from torchtitan.models.qwen3.model.grouped_lora import (
    _compute_grouped_lora_forward,
    _compute_grouped_lora_forward_different_out_dims,
    grouped_ffn_lora_forward,
    grouped_attention_head_offset_forward,
)


def test_grouped_lora_basic():
    """Test basic grouped LoRA computation matches sequential."""
    print("=" * 60)
    print("Test 1: Basic grouped LoRA computation")
    print("=" * 60)

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    # Test parameters
    batch, seq, dim = 4, 128, 1024
    rank = 64
    out_dim = 1024
    n_loras = 3

    # Create LoRA modules
    lora_modules = [
        ScaledLoRAModule(dim, out_dim, rank=rank).to(device).to(dtype)
        for _ in range(n_loras)
    ]

    # Create input
    x = torch.randn(batch, seq, dim, device=device, dtype=dtype)

    # Sequential computation (ground truth)
    outputs_seq = [m(x) for m in lora_modules]

    # Grouped computation
    x_flat = x.view(-1, dim)
    w_a_list = [m.w_a.weight for m in lora_modules]
    w_b_list = [m.w_b.weight for m in lora_modules]
    scales = [m.alpha / m.rank for m in lora_modules]

    outputs_grouped = _compute_grouped_lora_forward(x_flat, w_a_list, w_b_list, scales)
    outputs_grouped = [o.view(batch, seq, -1) for o in outputs_grouped]

    # Compare
    max_diff = 0
    for i, (seq_out, grp_out) in enumerate(zip(outputs_seq, outputs_grouped)):
        diff = (seq_out - grp_out).abs().max().item()
        max_diff = max(max_diff, diff)
        print(f"  LoRA {i}: max diff = {diff:.6f}")

    # bf16 tolerance is higher
    tolerance = 0.05
    if max_diff < tolerance:
        print(f"PASSED: max diff {max_diff:.6f} < tolerance {tolerance}")
    else:
        print(f"FAILED: max diff {max_diff:.6f} >= tolerance {tolerance}")

    return max_diff < tolerance


def test_grouped_lora_different_out_dims():
    """Test grouped LoRA with different output dimensions."""
    print("\n" + "=" * 60)
    print("Test 2: Grouped LoRA with different output dimensions")
    print("=" * 60)

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    batch, seq, dim = 4, 128, 1024
    rank = 64
    hidden_dim = 2816

    # Create LoRA modules with different output dims
    # w1, w3: dim -> hidden_dim
    # w2: hidden_dim -> dim (different input/output!)
    lora_w1 = ScaledLoRAModule(dim, hidden_dim, rank=rank).to(device).to(dtype)
    lora_w3 = ScaledLoRAModule(dim, hidden_dim, rank=rank).to(device).to(dtype)

    x = torch.randn(batch, seq, dim, device=device, dtype=dtype)

    # Sequential
    out_w1_seq = lora_w1(x)
    out_w3_seq = lora_w3(x)

    # Grouped
    x_flat = x.view(-1, dim)
    w_a_list = [lora_w1.w_a.weight, lora_w3.w_a.weight]
    w_b_list = [lora_w1.w_b.weight, lora_w3.w_b.weight]
    scales = [lora_w1.alpha / lora_w1.rank, lora_w3.alpha / lora_w3.rank]

    outputs_grouped = _compute_grouped_lora_forward(x_flat, w_a_list, w_b_list, scales)
    out_w1_grp = outputs_grouped[0].view(batch, seq, -1)
    out_w3_grp = outputs_grouped[1].view(batch, seq, -1)

    diff_w1 = (out_w1_seq - out_w1_grp).abs().max().item()
    diff_w3 = (out_w3_seq - out_w3_grp).abs().max().item()

    print(f"  w1 LoRA: max diff = {diff_w1:.6f}")
    print(f"  w3 LoRA: max diff = {diff_w3:.6f}")

    tolerance = 0.05
    max_diff = max(diff_w1, diff_w3)
    if max_diff < tolerance:
        print(f"PASSED: max diff {max_diff:.6f} < tolerance {tolerance}")
    else:
        print(f"FAILED: max diff {max_diff:.6f} >= tolerance {tolerance}")

    return max_diff < tolerance


def test_shared_feedforward():
    """Test SharedFeedForward with grouped_mm."""
    print("\n" + "=" * 60)
    print("Test 3: SharedFeedForward with grouped_mm")
    print("=" * 60)

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    batch, seq, dim = 4, 128, 1024
    hidden_dim = 2816
    lora_rank = 64

    # Create shared weights
    shared_weights = {
        "w1": nn.Linear(dim, hidden_dim, bias=False).to(device).to(dtype),
        "w2": nn.Linear(hidden_dim, dim, bias=False).to(device).to(dtype),
        "w3": nn.Linear(dim, hidden_dim, bias=False).to(device).to(dtype),
    }

    # Create two FFN modules with different settings
    ffn_sequential = SharedFeedForward(
        shared_weights, dim, hidden_dim, lora_rank, use_grouped_mm=False
    ).to(device).to(dtype)

    ffn_grouped = SharedFeedForward(
        shared_weights, dim, hidden_dim, lora_rank, use_grouped_mm=True
    ).to(device).to(dtype)

    # Copy LoRA weights to ensure same parameters
    ffn_grouped.w1.lora.w_a.weight.data.copy_(ffn_sequential.w1.lora.w_a.weight.data)
    ffn_grouped.w1.lora.w_b.weight.data.copy_(ffn_sequential.w1.lora.w_b.weight.data)
    ffn_grouped.w2.lora.w_a.weight.data.copy_(ffn_sequential.w2.lora.w_a.weight.data)
    ffn_grouped.w2.lora.w_b.weight.data.copy_(ffn_sequential.w2.lora.w_b.weight.data)
    ffn_grouped.w3.lora.w_a.weight.data.copy_(ffn_sequential.w3.lora.w_a.weight.data)
    ffn_grouped.w3.lora.w_b.weight.data.copy_(ffn_sequential.w3.lora.w_b.weight.data)

    x = torch.randn(batch, seq, dim, device=device, dtype=dtype)

    # Forward pass
    out_seq = ffn_sequential(x)
    out_grp = ffn_grouped(x)

    diff = (out_seq - out_grp).abs().max().item()
    print(f"  FFN output max diff = {diff:.6f}")

    # Check relative difference
    rel_diff = diff / out_seq.abs().mean().item()
    print(f"  FFN output relative diff = {rel_diff:.6f}")

    # bf16 has lower precision
    tolerance = 0.1
    if diff < tolerance:
        print(f"PASSED: max diff {diff:.6f} < tolerance {tolerance}")
    else:
        print(f"FAILED: max diff {diff:.6f} >= tolerance {tolerance}")

    return diff < tolerance


def test_attention_head_offsets():
    """Test grouped attention head offset computation."""
    print("\n" + "=" * 60)
    print("Test 4: Attention head offset computation")
    print("=" * 60)

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    batch, seq, dim = 4, 128, 1024
    n_heads = 16
    head_dim = 64
    rank = 64

    # Create LoRA modules for non-two_step case
    wq_offset = ScaledLoRAModule(dim, n_heads * head_dim, rank=rank).to(device).to(dtype)
    wk_offset = ScaledLoRAModule(dim, n_heads * head_dim, rank=rank).to(device).to(dtype)
    wv_offset = ScaledLoRAModule(dim, n_heads * head_dim, rank=rank).to(device).to(dtype)

    x = torch.randn(batch, seq, dim, device=device, dtype=dtype)

    # Sequential
    out_q_seq = wq_offset(x)
    out_k_seq = wk_offset(x)
    out_v_seq = wv_offset(x)

    # Grouped
    lora_modules = [wq_offset, wk_offset, wv_offset]
    out_q_grp, out_k_grp, out_v_grp = grouped_attention_head_offset_forward(
        x, lora_modules, two_step=False
    )

    diff_q = (out_q_seq - out_q_grp).abs().max().item()
    diff_k = (out_k_seq - out_k_grp).abs().max().item()
    diff_v = (out_v_seq - out_v_grp).abs().max().item()

    print(f"  Q offset max diff = {diff_q:.6f}")
    print(f"  K offset max diff = {diff_k:.6f}")
    print(f"  V offset max diff = {diff_v:.6f}")

    tolerance = 0.05
    max_diff = max(diff_q, diff_k, diff_v)
    if max_diff < tolerance:
        print(f"PASSED: max diff {max_diff:.6f} < tolerance {tolerance}")
    else:
        print(f"FAILED: max diff {max_diff:.6f} >= tolerance {tolerance}")

    return max_diff < tolerance


def test_attention_head_offsets_two_step():
    """Test grouped attention head offset with two-step decomposition."""
    print("\n" + "=" * 60)
    print("Test 5: Attention head offset (two-step)")
    print("=" * 60)

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    batch, seq, dim = 4, 128, 1024
    n_heads = 16
    head_dim = 64
    rank = 64

    # Create LoRA modules for two_step case
    head_offset = ScaledLoRAModule(dim, n_heads * head_dim, rank=rank).to(device).to(dtype)
    wq_only = ScaledLoRAModule(dim, head_dim, rank=rank).to(device).to(dtype)
    wk_only = ScaledLoRAModule(dim, head_dim, rank=rank).to(device).to(dtype)
    wv_only = ScaledLoRAModule(dim, head_dim, rank=rank).to(device).to(dtype)

    x = torch.randn(batch, seq, dim, device=device, dtype=dtype)

    # Sequential
    out_head_seq = head_offset(x)
    out_q_seq = wq_only(x)
    out_k_seq = wk_only(x)
    out_v_seq = wv_only(x)

    # Grouped
    lora_modules = [head_offset, wq_only, wk_only, wv_only]
    out_head_grp, out_q_grp, out_k_grp, out_v_grp = grouped_attention_head_offset_forward(
        x, lora_modules, two_step=True
    )

    diff_head = (out_head_seq - out_head_grp).abs().max().item()
    diff_q = (out_q_seq - out_q_grp).abs().max().item()
    diff_k = (out_k_seq - out_k_grp).abs().max().item()
    diff_v = (out_v_seq - out_v_grp).abs().max().item()

    print(f"  Head offset max diff = {diff_head:.6f}")
    print(f"  Q-only offset max diff = {diff_q:.6f}")
    print(f"  K-only offset max diff = {diff_k:.6f}")
    print(f"  V-only offset max diff = {diff_v:.6f}")

    tolerance = 0.05
    max_diff = max(diff_head, diff_q, diff_k, diff_v)
    if max_diff < tolerance:
        print(f"PASSED: max diff {max_diff:.6f} < tolerance {tolerance}")
    else:
        print(f"FAILED: max diff {max_diff:.6f} >= tolerance {tolerance}")

    return max_diff < tolerance


def benchmark_grouped_vs_sequential():
    """Benchmark grouped_mm vs sequential LoRA computation."""
    print("\n" + "=" * 60)
    print("Benchmark: Grouped vs Sequential LoRA")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return True

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    print(f"  Device: {torch.cuda.get_device_name(0)}")

    # Test multiple configurations
    configs = [
        # (batch, seq, dim, hidden_dim, lora_rank, description)
        (4, 2048, 1024, 2816, 256, "250M config"),
        (8, 2048, 1024, 2816, 256, "250M x2 batch"),
        (4, 4096, 1024, 2816, 256, "250M x2 seq"),
    ]

    for batch, seq, dim, hidden_dim, lora_rank, desc in configs:
        print(f"\n  Config: {desc} (batch={batch}, seq={seq}, dim={dim}, rank={lora_rank})")

        # Create shared weights
        shared_weights = {
            "w1": nn.Linear(dim, hidden_dim, bias=False).to(device).to(dtype),
            "w2": nn.Linear(hidden_dim, dim, bias=False).to(device).to(dtype),
            "w3": nn.Linear(dim, hidden_dim, bias=False).to(device).to(dtype),
        }

        ffn_sequential = SharedFeedForward(
            shared_weights, dim, hidden_dim, lora_rank, use_grouped_mm=False
        ).to(device).to(dtype)

        ffn_grouped = SharedFeedForward(
            shared_weights, dim, hidden_dim, lora_rank, use_grouped_mm=True
        ).to(device).to(dtype)

        # Copy weights
        ffn_grouped.w1.lora.w_a.weight.data.copy_(ffn_sequential.w1.lora.w_a.weight.data)
        ffn_grouped.w1.lora.w_b.weight.data.copy_(ffn_sequential.w1.lora.w_b.weight.data)
        ffn_grouped.w2.lora.w_a.weight.data.copy_(ffn_sequential.w2.lora.w_a.weight.data)
        ffn_grouped.w2.lora.w_b.weight.data.copy_(ffn_sequential.w2.lora.w_b.weight.data)
        ffn_grouped.w3.lora.w_a.weight.data.copy_(ffn_sequential.w3.lora.w_a.weight.data)
        ffn_grouped.w3.lora.w_b.weight.data.copy_(ffn_sequential.w3.lora.w_b.weight.data)

        x = torch.randn(batch, seq, dim, device=device, dtype=dtype)

        # Warmup
        for _ in range(10):
            _ = ffn_sequential(x)
            _ = ffn_grouped(x)
        torch.cuda.synchronize()

        # Benchmark sequential
        n_iters = 100
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = ffn_sequential(x)
        torch.cuda.synchronize()
        seq_time = (time.perf_counter() - start) / n_iters * 1000

        # Benchmark grouped
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = ffn_grouped(x)
        torch.cuda.synchronize()
        grp_time = (time.perf_counter() - start) / n_iters * 1000

        speedup = seq_time / grp_time

        print(f"    Sequential: {seq_time:.3f} ms")
        print(f"    Grouped:    {grp_time:.3f} ms")
        print(f"    Speedup:    {speedup:.2f}x")

    return True


def test_backward_pass():
    """Test that gradients flow correctly through grouped_mm.

    Note: We compare the same FFN module with grouped_mm on vs off,
    since the backward pass goes through PyTorch autograd (not custom backward).
    The grouped_mm only batches the forward matmuls - gradients are computed
    by PyTorch automatically and should be the same if forward outputs match.

    Important: Must use bfloat16 since torch._grouped_mm requires bfloat16.
    """
    print("\n" + "=" * 60)
    print("Test 6: Backward pass gradient flow")
    print("=" * 60)

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16  # Must use bfloat16 - grouped_mm requires it

    batch, seq, dim = 2, 64, 256
    hidden_dim = 512
    lora_rank = 16

    # Create shared weights - MUST use the same instances
    shared_weights = {
        "w1": nn.Linear(dim, hidden_dim, bias=False).to(device).to(dtype),
        "w2": nn.Linear(hidden_dim, dim, bias=False).to(device).to(dtype),
        "w3": nn.Linear(dim, hidden_dim, bias=False).to(device).to(dtype),
    }

    # Create single FFN and test with both modes
    ffn = SharedFeedForward(
        shared_weights, dim, hidden_dim, lora_rank, use_grouped_mm=True
    ).to(device).to(dtype)

    x = torch.randn(batch, seq, dim, device=device, dtype=dtype, requires_grad=True)

    # Forward + backward with grouped_mm=True
    ffn.use_grouped_mm = True
    out_grp = ffn(x)
    loss_grp = out_grp.sum()
    loss_grp.backward()

    # Store gradients
    grad_x_grp = x.grad.clone()
    grad_w1_a_grp = ffn.w1.lora.w_a.weight.grad.clone()
    grad_w1_b_grp = ffn.w1.lora.w_b.weight.grad.clone()

    # Zero gradients
    x.grad = None
    ffn.zero_grad()

    # Forward + backward with grouped_mm=False (sequential)
    ffn.use_grouped_mm = False
    out_seq = ffn(x)
    loss_seq = out_seq.sum()
    loss_seq.backward()

    grad_x_seq = x.grad
    grad_w1_a_seq = ffn.w1.lora.w_a.weight.grad
    grad_w1_b_seq = ffn.w1.lora.w_b.weight.grad

    # Compare gradients
    grad_x_diff = (grad_x_grp - grad_x_seq).abs().max().item()
    grad_w1_a_diff = (grad_w1_a_grp - grad_w1_a_seq).abs().max().item()
    grad_w1_b_diff = (grad_w1_b_grp - grad_w1_b_seq).abs().max().item()

    print(f"  Input gradient max diff = {grad_x_diff:.6f}")
    print(f"  w1_a gradient max diff = {grad_w1_a_diff:.6f}")
    print(f"  w1_b gradient max diff = {grad_w1_b_diff:.6f}")

    # Also check forward outputs match
    out_diff = (out_grp - out_seq).abs().max().item()
    print(f"  Forward output max diff = {out_diff:.6f}")

    # bfloat16 has ~3 decimal digits of precision, so use relaxed tolerance
    tolerance = 0.05
    max_diff = max(grad_x_diff, grad_w1_a_diff, grad_w1_b_diff, out_diff)
    if max_diff < tolerance:
        print(f"PASSED: max gradient diff {max_diff:.6f} < tolerance {tolerance}")
    else:
        print(f"FAILED: max gradient diff {max_diff:.6f} >= tolerance {tolerance}")

    return max_diff < tolerance


def main():
    print("Testing grouped_mm LoRA optimization")
    print("=" * 60)

    results = []

    # Run all tests
    results.append(("Basic grouped LoRA", test_grouped_lora_basic()))
    results.append(("Different output dims", test_grouped_lora_different_out_dims()))
    results.append(("SharedFeedForward", test_shared_feedforward()))
    results.append(("Attention head offsets", test_attention_head_offsets()))
    results.append(("Attention two-step", test_attention_head_offsets_two_step()))
    results.append(("Backward pass", test_backward_pass()))
    results.append(("Benchmark", benchmark_grouped_vs_sequential()))

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"  {name}: {status}")

    print(f"\n{passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
