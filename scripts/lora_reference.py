# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Reference implementations for LoRA fusion operations.

These provide numerically correct baselines for testing fused kernels.

Target operations:
    Forward:  Y = X @ W.T + (X @ A.T) @ B.T * scale
    Backward: grad_X = grad_Y @ W + (grad_Y @ B * scale) @ A
              grad_W = grad_Y.T @ X
              grad_B = (grad_Y.T @ S) * scale
              grad_A = ((grad_Y @ B * scale).T) @ X
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def reference_lora_forward(
    x: torch.Tensor,
    W: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reference forward pass for Linear+LoRA.

    Args:
        x: Input tensor of shape (M, K) where M = batch*seq
        W: Base weight matrix of shape (N, K)
        A: LoRA A matrix of shape (R, K)
        B: LoRA B matrix of shape (N, R)
        scale: LoRA scaling factor (alpha/rank)

    Returns:
        y: Output tensor of shape (M, N)
        s: Intermediate S = X @ A.T of shape (M, R) - saved for backward
    """
    # Compute intermediate S = X @ A.T
    s = F.linear(x, A)  # (M, R)

    # Compute LoRA output
    lora_out = F.linear(s, B) * scale  # (M, N)

    # Compute base output
    base_out = F.linear(x, W)  # (M, N)

    # Combine
    y = base_out + lora_out

    return y, s


def reference_lora_backward(
    grad_y: torch.Tensor,
    x: torch.Tensor,
    W: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    S: torch.Tensor,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reference backward pass for Linear+LoRA.

    Args:
        grad_y: Gradient of output, shape (M, N)
        x: Input tensor, shape (M, K)
        W: Base weight matrix, shape (N, K)
        A: LoRA A matrix, shape (R, K)
        B: LoRA B matrix, shape (N, R)
        S: Saved intermediate S = X @ A.T, shape (M, R)
        scale: LoRA scaling factor

    Returns:
        grad_x: Gradient of input, shape (M, K)
        grad_W: Gradient of base weight, shape (N, K)
        grad_A: Gradient of LoRA A, shape (R, K)
        grad_B: Gradient of LoRA B, shape (N, R)
    """
    # grad_W = grad_Y.T @ X
    grad_W = grad_y.t() @ x  # (N, K)

    # grad_X = grad_Y @ W + (grad_Y @ B * scale) @ A
    # Part 1: grad_Y @ W
    grad_x_base = grad_y @ W  # (M, K)

    # Part 2: (grad_Y @ B * scale) @ A
    # Compute grad_Y @ B
    grad_y_B = grad_y @ B  # (M, R)
    grad_y_B_scaled = grad_y_B * scale  # (M, R)
    grad_x_lora = grad_y_B_scaled @ A  # (M, K)

    grad_x = grad_x_base + grad_x_lora

    # grad_B = (grad_Y.T @ S) * scale
    grad_B = (grad_y.t() @ S) * scale  # (N, R)

    # grad_A = ((grad_Y @ B * scale).T) @ X
    # = grad_y_B_scaled.T @ X
    grad_A = grad_y_B_scaled.t() @ x  # (R, K)

    return grad_x, grad_W, grad_A, grad_B


def unfused_lora_forward(
    x: torch.Tensor,
    W: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unfused forward pass using separate operations (4 matmuls).

    This is the baseline we're trying to beat with fusion.
    """
    # 4 separate operations (4 kernel launches without fusion)
    base_out = F.linear(x, W)  # Kernel 1: X @ W.T
    s = F.linear(x, A)  # Kernel 2: X @ A.T
    lora_intermediate = F.linear(s, B)  # Kernel 3: S @ B.T
    lora_out = lora_intermediate * scale  # Kernel 4: elementwise

    y = base_out + lora_out
    return y, s


def lorafusion_style_forward(
    x: torch.Tensor,
    W: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    LoRAFusion-style 2-kernel approach.

    Kernel 1 (cuBLAS): S = X @ A.T
    Kernel 2 (fused): Y = X @ W.T + S @ B.T * scale
    """
    # Kernel 1: cuBLAS (optimal)
    s = F.linear(x, A)  # (M, R)

    # Kernel 2: Would be fused TK kernel in actual implementation
    # Here we simulate with separate ops
    base_out = F.linear(x, W)
    lora_out = F.linear(s, B) * scale
    y = base_out + lora_out

    return y, s


class LinearLoRAReference(torch.autograd.Function):
    """
    Autograd function for reference Linear+LoRA with backward.

    This provides a differentiable reference implementation.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        W: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        y, s = reference_lora_forward(x, W, A, B, scale)
        ctx.save_for_backward(x, W, A, B, s)
        ctx.scale = scale
        return y

    @staticmethod
    def backward(
        ctx, grad_y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None]:
        x, W, A, B, s = ctx.saved_tensors
        grad_x, grad_W, grad_A, grad_B = reference_lora_backward(
            grad_y, x, W, A, B, s, ctx.scale
        )
        return grad_x, grad_W, grad_A, grad_B, None


def linear_lora_reference(
    x: torch.Tensor,
    W: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Convenience wrapper for LinearLoRAReference.apply"""
    return LinearLoRAReference.apply(x, W, A, B, scale)


# =============================================================================
# Verification utilities
# =============================================================================


def check_forward_correctness(
    fused_fn,
    M: int = 8192,
    K: int = 1024,
    N: int = 1024,
    R: int = 256,
    scale: float = 0.0625,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> bool:
    """
    Check if a fused forward function matches reference.

    Args:
        fused_fn: Function with signature (x, W, A, B, scale) -> (y, s)
        M, K, N, R: Tensor dimensions
        scale: LoRA scaling factor
        atol, rtol: Tolerance for torch.allclose
        device: Device to run on
        dtype: Data type

    Returns:
        True if results match within tolerance
    """
    # Create random inputs
    x = torch.randn(M, K, device=device, dtype=dtype)
    W = torch.randn(N, K, device=device, dtype=dtype)
    A = torch.randn(R, K, device=device, dtype=dtype)
    B = torch.randn(N, R, device=device, dtype=dtype)

    # Reference
    y_ref, s_ref = reference_lora_forward(x, W, A, B, scale)

    # Fused
    y_fused, s_fused = fused_fn(x, W, A, B, scale)

    # Check
    y_match = torch.allclose(y_fused, y_ref, atol=atol, rtol=rtol)
    s_match = torch.allclose(s_fused, s_ref, atol=atol, rtol=rtol)

    if not y_match:
        max_diff = (y_fused - y_ref).abs().max().item()
        print(f"Y mismatch: max diff = {max_diff}")
    if not s_match:
        max_diff = (s_fused - s_ref).abs().max().item()
        print(f"S mismatch: max diff = {max_diff}")

    return y_match and s_match


def check_backward_correctness(
    fused_fn,
    M: int = 8192,
    K: int = 1024,
    N: int = 1024,
    R: int = 256,
    scale: float = 0.0625,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> bool:
    """
    Check if a fused function's gradients match reference.

    Args:
        fused_fn: Autograd-enabled function
        Other args: Same as check_forward_correctness

    Returns:
        True if all gradients match within tolerance
    """
    # Create random inputs with gradients
    x = torch.randn(M, K, device=device, dtype=dtype, requires_grad=True)
    W = torch.randn(N, K, device=device, dtype=dtype, requires_grad=True)
    A = torch.randn(R, K, device=device, dtype=dtype, requires_grad=True)
    B = torch.randn(N, R, device=device, dtype=dtype, requires_grad=True)

    # Clone for reference
    x_ref = x.detach().clone().requires_grad_(True)
    W_ref = W.detach().clone().requires_grad_(True)
    A_ref = A.detach().clone().requires_grad_(True)
    B_ref = B.detach().clone().requires_grad_(True)

    # Forward + backward for fused
    y_fused = fused_fn(x, W, A, B, scale)
    grad_y = torch.randn_like(y_fused)
    y_fused.backward(grad_y)

    # Forward + backward for reference
    y_ref = linear_lora_reference(x_ref, W_ref, A_ref, B_ref, scale)
    y_ref.backward(grad_y)

    # Compare gradients
    all_match = True

    for name, fused_grad, ref_grad in [
        ("grad_x", x.grad, x_ref.grad),
        ("grad_W", W.grad, W_ref.grad),
        ("grad_A", A.grad, A_ref.grad),
        ("grad_B", B.grad, B_ref.grad),
    ]:
        if not torch.allclose(fused_grad, ref_grad, atol=atol, rtol=rtol):
            max_diff = (fused_grad - ref_grad).abs().max().item()
            print(f"{name} mismatch: max diff = {max_diff}")
            all_match = False

    return all_match


if __name__ == "__main__":
    # Quick sanity check
    print("Testing reference implementations...")

    M, K, N, R = 8192, 1024, 1024, 256
    scale = 16.0 / R

    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    A = torch.randn(R, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N, R, device="cuda", dtype=torch.bfloat16)

    # Test forward
    y1, s1 = reference_lora_forward(x, W, A, B, scale)
    y2, s2 = unfused_lora_forward(x, W, A, B, scale)
    y3, s3 = lorafusion_style_forward(x, W, A, B, scale)

    print(f"reference vs unfused Y: {torch.allclose(y1, y2, atol=1e-2)}")
    print(f"reference vs lorafusion Y: {torch.allclose(y1, y3, atol=1e-2)}")

    # Test backward
    print(f"\nBackward correctness: {check_backward_correctness(linear_lora_reference)}")

    print("\nReference implementations OK!")
