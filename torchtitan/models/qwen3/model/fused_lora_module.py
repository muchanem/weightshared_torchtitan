# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
PyTorch module wrapper for fused Linear+LoRA operations.

This module provides torch.compile-friendly wrappers for the fused Triton kernels
using torch.library.triton_op for composability with AOTAutograd.

The key classes are:
- FusedSharedLinearWithLoRA: Drop-in replacement for SharedLinearWithLoRA
- fused_linear_lora: Functional API for fused Linear+LoRA
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import triton

# Import Triton kernels
from .fused_lora_kernels import (
    fused_linear_lora_fwd_kernel,
    fused_linear_lora_bwd_dx_kernel,
    fused_linear_lora_bwd_dw_kernel,
    fused_linear_lora_bwd_dlora_kernel,
)


__all__ = [
    "fused_linear_lora",
    "FusedSharedLinearWithLoRA",
    "FusedScaledLoRAModule",
]


# =============================================================================
# Autograd Function
# =============================================================================

class FusedLinearLoRAFunction(torch.autograd.Function):
    """
    Autograd function for fused Linear+LoRA.

    This implementation uses standard PyTorch autograd.Function which works
    with torch.compile but may have some overhead. For maximum performance,
    consider using the triton_op API when PyTorch version supports it well.

    Forward: y = x @ W.T + (x @ A.T) @ B.T * scale
    Backward: Computes gradients for x, W, A, B
    """

    @staticmethod
    def forward(ctx, x, w, a, b, scale):
        """
        Forward pass using Triton kernels.

        Args:
            x: Input tensor (*, K)
            w: Base weight (N, K)
            a: LoRA down-projection (R, K)
            b: LoRA up-projection (N, R)
            scale: LoRA scale factor (alpha/rank)

        Returns:
            Output tensor (*, N)
        """
        # Flatten batch dimensions
        orig_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1]).contiguous()  # (M, K)

        M, K = x_flat.shape
        N, _ = w.shape
        R, _ = a.shape

        # Allocate outputs
        y = torch.empty((M, N), device=x.device, dtype=x.dtype)
        s = torch.empty((M, R), device=x.device, dtype=x.dtype)

        # Compute grid
        BLOCK_M = 64
        BLOCK_N = 64
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # Launch forward kernel
        fused_linear_lora_fwd_kernel[grid](
            x_flat, w, a, b, y, s,
            M, N, K, R,
            scale,
            x_flat.stride(0), x_flat.stride(1),
            w.stride(0), w.stride(1),
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            y.stride(0), y.stride(1),
            s.stride(0), s.stride(1),
        )

        # Save for backward
        ctx.save_for_backward(x_flat, w, a, b, s)
        ctx.scale = scale
        ctx.orig_shape = orig_shape
        ctx.M = M
        ctx.N = N
        ctx.K = K
        ctx.R = R

        # Reshape output
        y = y.reshape(*orig_shape, N)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass using Triton kernels.

        Args:
            grad_output: Gradient of output (*, N)

        Returns:
            Gradients for x, w, a, b, None (for scale)
        """
        x_flat, w, a, b, s = ctx.saved_tensors
        scale = ctx.scale
        M, N, K, R = ctx.M, ctx.N, ctx.K, ctx.R

        # Flatten grad_output
        dy_flat = grad_output.reshape(-1, N).contiguous()

        # Allocate gradients
        dx = torch.empty((M, K), device=grad_output.device, dtype=grad_output.dtype)
        dw = torch.empty_like(w)
        da = torch.empty_like(a)
        db = torch.empty_like(b)

        # Launch dX kernel
        BLOCK_M = 64
        BLOCK_K = 64
        BLOCK_N = 64
        grid_dx = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))

        fused_linear_lora_bwd_dx_kernel[grid_dx](
            dy_flat, w, a, b, dx,
            M, N, K, R,
            scale,
            dy_flat.stride(0), dy_flat.stride(1),
            w.stride(0), w.stride(1),
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            dx.stride(0), dx.stride(1),
        )

        # Launch dW kernel
        grid_dw = (triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))

        fused_linear_lora_bwd_dw_kernel[grid_dw](
            dy_flat, x_flat, dw,
            M, N, K,
            dy_flat.stride(0), dy_flat.stride(1),
            x_flat.stride(0), x_flat.stride(1),
            dw.stride(0), dw.stride(1),
        )

        # Launch dA/dB kernel
        BLOCK_R = 32
        num_dB_blocks = triton.cdiv(N, BLOCK_N) * triton.cdiv(R, BLOCK_R)
        num_dA_blocks = triton.cdiv(R, BLOCK_R) * triton.cdiv(K, BLOCK_K)
        grid_dlora = (num_dB_blocks + num_dA_blocks,)

        fused_linear_lora_bwd_dlora_kernel[grid_dlora](
            dy_flat, x_flat, s, b, da, db,
            M, N, K, R,
            scale,
            dy_flat.stride(0), dy_flat.stride(1),
            x_flat.stride(0), x_flat.stride(1),
            s.stride(0), s.stride(1),
            b.stride(0), b.stride(1),
            da.stride(0), da.stride(1),
            db.stride(0), db.stride(1),
        )

        # Reshape dx
        dx = dx.reshape(*ctx.orig_shape, K)

        return dx, dw, da, db, None


def fused_linear_lora(
    x: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """
    Functional API for fused Linear+LoRA.

    Computes: y = x @ W.T + (x @ A.T) @ B.T * scale

    Args:
        x: Input tensor of shape (*, K)
        w: Base weight matrix of shape (N, K)
        a: LoRA A matrix of shape (R, K)
        b: LoRA B matrix of shape (N, R)
        scale: LoRA scaling factor (alpha/rank)

    Returns:
        Output tensor of shape (*, N)
    """
    return FusedLinearLoRAFunction.apply(x, w, a, b, scale)


# =============================================================================
# Reference (Unfused) Implementation for CPU and Fallback
# =============================================================================

def _unfused_linear_lora(
    x: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """
    Reference implementation without fusion.

    Used for CPU fallback and correctness testing.
    """
    base_out = torch.nn.functional.linear(x, w)
    lora_out = torch.nn.functional.linear(
        torch.nn.functional.linear(x, a), b
    ) * scale
    return base_out + lora_out


# =============================================================================
# PyTorch Modules
# =============================================================================

class FusedScaledLoRAModule(nn.Module):
    """
    Low-rank adaptation module with scaling (fused version).

    This is equivalent to ScaledLoRAModule but can be used with fused operations.

    Args:
        in_dim: Input dimension.
        out_dim: Output dimension.
        rank: LoRA rank (bottleneck dimension).
        alpha: Scaling factor. Default: 2 * rank.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int = 8,
        alpha: Optional[int] = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank * 2

        # LoRA matrices: A is down-projection, B is up-projection
        # A: (R, K) - projects from in_dim to rank
        # B: (N, R) - projects from rank to out_dim
        self.w_a = nn.Linear(in_dim, rank, bias=False)
        self.w_b = nn.Linear(rank, out_dim, bias=False)

    def reset_parameters(self) -> None:
        """Initialize LoRA weights. A is kaiming, B is zeros."""
        nn.init.kaiming_uniform_(self.w_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.w_b.weight)

    @property
    def scale(self) -> float:
        """LoRA scaling factor."""
        return self.alpha / self.rank

    @property
    def A(self) -> torch.Tensor:
        """LoRA A matrix (R, K)."""
        return self.w_a.weight

    @property
    def B(self) -> torch.Tensor:
        """LoRA B matrix (N, R)."""
        return self.w_b.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute low-rank delta (unfused for standalone use)."""
        return self.w_b(self.w_a(x)) * self.scale


class FusedSharedLinearWithLoRA(nn.Module):
    """
    Linear layer with shared base weights and per-instance LoRA adapter.

    This is a drop-in replacement for SharedLinearWithLoRA that uses fused
    Triton kernels for improved performance.

    Args:
        shared_linear: The shared base linear layer.
        in_features: Input dimension.
        out_features: Output dimension.
        lora_rank: LoRA rank. Set to 0 to disable LoRA.
        lora_alpha: LoRA scaling alpha. Default: 2 * rank.
        use_fused: Whether to use fused Triton kernels. Default: True.
    """

    def __init__(
        self,
        shared_linear: nn.Linear,
        in_features: int,
        out_features: int,
        lora_rank: int,
        lora_alpha: Optional[int] = None,
        use_fused: bool = True,
    ):
        super().__init__()
        self.shared_linear = shared_linear
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.use_fused = use_fused and lora_rank > 0

        if lora_rank > 0:
            self.lora = FusedScaledLoRAModule(
                in_features, out_features, rank=lora_rank, alpha=lora_alpha
            )

    def reset_parameters(self) -> None:
        """Reset LoRA parameters."""
        if self.lora_rank > 0:
            self.lora.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        If use_fused and LoRA is enabled, uses the fused Triton kernel.
        Otherwise, falls back to the standard implementation.
        """
        if not self.lora_rank > 0:
            return self.shared_linear(x)

        if self.use_fused and x.is_cuda:
            # Use fused kernel
            return fused_linear_lora(
                x,
                self.shared_linear.weight,
                self.lora.A,
                self.lora.B,
                self.lora.scale,
            )
        else:
            # Fallback to unfused
            base_output = self.shared_linear(x)
            return base_output + self.lora(x)


class FusedSharedFeedForward(nn.Module):
    """
    FeedForward module with shared weights and per-layer LoRA (fused version).

    This is a drop-in replacement for SharedFeedForward that uses fused
    Triton kernels for the Linear+LoRA operations.

    Args:
        shared_weights: Dict with "w1", "w2", "w3" shared linear layers.
        dim: Model dimension.
        hidden_dim: FFN hidden dimension.
        lora_rank: LoRA rank for per-layer adapters.
        use_fused: Whether to use fused Triton kernels. Default: True.
    """

    def __init__(
        self,
        shared_weights: dict,
        dim: int,
        hidden_dim: int,
        lora_rank: int,
        use_fused: bool = True,
    ):
        super().__init__()
        self.w1 = FusedSharedLinearWithLoRA(
            shared_weights["w1"], dim, hidden_dim, lora_rank, use_fused=use_fused
        )
        self.w2 = FusedSharedLinearWithLoRA(
            shared_weights["w2"], hidden_dim, dim, lora_rank, use_fused=use_fused
        )
        self.w3 = FusedSharedLinearWithLoRA(
            shared_weights["w3"], dim, hidden_dim, lora_rank, use_fused=use_fused
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU forward: w2(silu(w1(x)) * w3(x))"""
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float) -> None:
        """Initialize weights."""
        self.w1.reset_parameters()
        self.w2.reset_parameters()
        self.w3.reset_parameters()
