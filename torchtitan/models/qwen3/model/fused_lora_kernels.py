# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Triton kernels for fused Linear+LoRA operations.

This module provides high-performance Triton kernels that fuse the base linear
projection with LoRA adapters, reducing memory traffic and kernel launches.

The target operation is:
    y = x @ W.T + (x @ A.T) @ B.T * scale

Where:
    x: input tensor (M, K) where M = batch*seq
    W: base weight matrix (N, K)
    A: LoRA down-projection (R, K)
    B: LoRA up-projection (N, R)
    scale: LoRA scaling factor (alpha/rank)
    y: output tensor (M, N)

For backward pass, we also save S = x @ A.T as an intermediate.
"""

import torch
import triton
import triton.language as tl

__all__ = [
    "fused_linear_lora_fwd_kernel",
    "fused_linear_lora_bwd_dx_kernel",
    "fused_linear_lora_bwd_dw_kernel",
    "fused_linear_lora_bwd_dlora_kernel",
]


# =============================================================================
# Forward Kernel
# =============================================================================

@triton.autotune(
    configs=[
        # BLOCK_M=64, BLOCK_N=64 must match grid computation in Python wrapper
        # Only BLOCK_K can vary since it's an inner loop dimension
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def fused_linear_lora_fwd_kernel(
    # Pointers to tensors
    X_ptr,      # Input: (M, K)
    W_ptr,      # Base weight: (N, K)
    A_ptr,      # LoRA A: (R, K)
    B_ptr,      # LoRA B: (N, R)
    Y_ptr,      # Output: (M, N)
    S_ptr,      # Intermediate S = X @ A.T: (M, R), saved for backward
    # Tensor dimensions
    M,          # batch * seq_len
    N,          # out_features
    K,          # in_features
    R,          # lora_rank
    # LoRA scale
    scale,
    # Strides for X (row-major)
    stride_xm, stride_xk,
    # Strides for W (row-major: N x K)
    stride_wn, stride_wk,
    # Strides for A (row-major: R x K)
    stride_ar, stride_ak,
    # Strides for B (row-major: N x R)
    stride_bn, stride_br,
    # Strides for Y (row-major: M x N)
    stride_ym, stride_yn,
    # Strides for S (row-major: M x R)
    stride_sm, stride_sr,
    # Block sizes (constexpr for compile-time)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused forward kernel: Y = X @ W.T + (X @ A.T) @ B.T * scale

    Also saves S = X @ A.T for backward pass.

    This kernel tiles over M (batch) and N (output features), streaming through
    K (input features) to compute both the base matmul and LoRA intermediate.
    """
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute starting indices for this block
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    # Initialize block indices
    m_idx = m_start + tl.arange(0, BLOCK_M)  # (BLOCK_M,)
    n_idx = n_start + tl.arange(0, BLOCK_N)  # (BLOCK_N,)

    # Initialize accumulator for base matmul: X @ W.T
    acc_base = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Initialize accumulator for LoRA intermediate: X @ A.T
    # We need to compute this for all R ranks, but we'll do it in chunks
    # For simplicity, we'll compute the full S here if R fits in registers,
    # otherwise we need a second pass

    # First pass: compute X @ W.T (streaming through K)
    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + tl.arange(0, BLOCK_K)  # (BLOCK_K,)

        # Load X block: (BLOCK_M, BLOCK_K)
        x_ptrs = X_ptr + (m_idx[:, None] * stride_xm + k_idx[None, :] * stride_xk)
        x_mask = (m_idx[:, None] < M) & (k_idx[None, :] < K)
        x_block = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load W block: (BLOCK_N, BLOCK_K) -> we want W.T which is (K, N)
        # But W is stored as (N, K), so we load (BLOCK_N, BLOCK_K) and transpose
        w_ptrs = W_ptr + (n_idx[:, None] * stride_wn + k_idx[None, :] * stride_wk)
        w_mask = (n_idx[:, None] < N) & (k_idx[None, :] < K)
        w_block = tl.load(w_ptrs, mask=w_mask, other=0.0)  # (BLOCK_N, BLOCK_K)

        # Accumulate: X @ W.T = X @ W^T
        # x_block: (BLOCK_M, BLOCK_K), w_block: (BLOCK_N, BLOCK_K)
        # Want: (BLOCK_M, BLOCK_N) = (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N)
        acc_base += tl.dot(x_block, tl.trans(w_block))

    # Second: compute LoRA path
    # We need S = X @ A.T for all rows in this block, then Y_lora = S @ B.T
    # This is more complex because S has R columns which may be large

    # For now, we use a simpler approach: compute LoRA in a separate loop
    # This can be fused better with a more sophisticated implementation

    # Initialize LoRA accumulator
    acc_lora = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Compute S = X @ A.T in tiles and immediately multiply by B.T
    # We tile over R (lora rank)
    BLOCK_R: tl.constexpr = 32  # Tile size for lora rank

    for r_start in range(0, R, BLOCK_R):
        r_idx = r_start + tl.arange(0, BLOCK_R)  # (BLOCK_R,)

        # Compute S_block = X[m_idx, :] @ A[r_idx, :].T
        # = sum over k of X[:, k] * A[r, k]
        s_block = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            k_idx = k_start + tl.arange(0, BLOCK_K)

            # Load X block
            x_ptrs = X_ptr + (m_idx[:, None] * stride_xm + k_idx[None, :] * stride_xk)
            x_mask = (m_idx[:, None] < M) & (k_idx[None, :] < K)
            x_block = tl.load(x_ptrs, mask=x_mask, other=0.0)  # (BLOCK_M, BLOCK_K)

            # Load A block: (BLOCK_R, BLOCK_K)
            a_ptrs = A_ptr + (r_idx[:, None] * stride_ar + k_idx[None, :] * stride_ak)
            a_mask = (r_idx[:, None] < R) & (k_idx[None, :] < K)
            a_block = tl.load(a_ptrs, mask=a_mask, other=0.0)  # (BLOCK_R, BLOCK_K)

            # Accumulate: S_block += X @ A.T
            # x_block: (BLOCK_M, BLOCK_K), a_block: (BLOCK_R, BLOCK_K)
            # s_block: (BLOCK_M, BLOCK_R)
            s_block += tl.dot(x_block, tl.trans(a_block))

        # Convert s_block to bf16 to match unfused F.linear behavior
        # (unfused: intermediate = F.linear(x, a) produces bf16)
        s_block_bf16 = s_block.to(tl.bfloat16)

        # Save S block (for backward pass) - only for the first N block
        if pid_n == 0:
            s_ptrs = S_ptr + (m_idx[:, None] * stride_sm + r_idx[None, :] * stride_sr)
            s_mask = (m_idx[:, None] < M) & (r_idx[None, :] < R)
            tl.store(s_ptrs, s_block_bf16, mask=s_mask)

        # Load B block: (BLOCK_N, BLOCK_R)
        b_ptrs = B_ptr + (n_idx[:, None] * stride_bn + r_idx[None, :] * stride_br)
        b_mask = (n_idx[:, None] < N) & (r_idx[None, :] < R)
        b_block = tl.load(b_ptrs, mask=b_mask, other=0.0)  # (BLOCK_N, BLOCK_R)

        # Accumulate LoRA output: S_block @ B.T
        # Both s_block_bf16 and b_block are bf16, matching unfused behavior
        # Result: (BLOCK_M, BLOCK_N), accumulated in fp32
        acc_lora += tl.dot(s_block_bf16, tl.trans(b_block))

    # Combine base and LoRA outputs (cast scale to fp32 to avoid fp64 promotion)
    y_block = acc_base + acc_lora * scale.to(tl.float32)

    # Store output
    y_ptrs = Y_ptr + (m_idx[:, None] * stride_ym + n_idx[None, :] * stride_yn)
    y_mask = (m_idx[:, None] < M) & (n_idx[None, :] < N)
    tl.store(y_ptrs, y_block.to(Y_ptr.dtype.element_ty), mask=y_mask)


# =============================================================================
# Backward Kernels
# =============================================================================

@triton.autotune(
    configs=[
        # BLOCK_M=64, BLOCK_K=64 must match grid computation in Python wrapper
        # Only BLOCK_N can vary since it's an inner loop dimension
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def fused_linear_lora_bwd_dx_kernel(
    # Pointers to tensors
    dY_ptr,     # Gradient of output: (M, N)
    W_ptr,      # Base weight: (N, K)
    A_ptr,      # LoRA A: (R, K)
    B_ptr,      # LoRA B: (N, R)
    dX_ptr,     # Gradient of input: (M, K)
    # Tensor dimensions
    M, N, K, R,
    # LoRA scale
    scale,
    # Strides
    stride_dym, stride_dyn,
    stride_wn, stride_wk,
    stride_ar, stride_ak,
    stride_bn, stride_br,
    stride_dxm, stride_dxk,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Backward kernel for dX: dX = dY @ W + (dY @ B @ A) * scale

    dY: (M, N)
    W: (N, K) -> dY @ W: (M, N) @ (N, K) = (M, K)
    B: (N, R), A: (R, K) -> dY @ B @ A: (M, N) @ (N, R) @ (R, K) = (M, K)
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    k_start = pid_k * BLOCK_K

    m_idx = m_start + tl.arange(0, BLOCK_M)
    k_idx = k_start + tl.arange(0, BLOCK_K)

    # Accumulator for dX
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    # Compute dY @ W (streaming through N)
    for n_start in range(0, N, BLOCK_N):
        n_idx = n_start + tl.arange(0, BLOCK_N)

        # Load dY block: (BLOCK_M, BLOCK_N)
        dy_ptrs = dY_ptr + (m_idx[:, None] * stride_dym + n_idx[None, :] * stride_dyn)
        dy_mask = (m_idx[:, None] < M) & (n_idx[None, :] < N)
        dy_block = tl.load(dy_ptrs, mask=dy_mask, other=0.0)

        # Load W block: (BLOCK_N, BLOCK_K)
        w_ptrs = W_ptr + (n_idx[:, None] * stride_wn + k_idx[None, :] * stride_wk)
        w_mask = (n_idx[:, None] < N) & (k_idx[None, :] < K)
        w_block = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # dY @ W
        acc += tl.dot(dy_block, w_block)

    # Compute dY @ B @ A (LoRA part)
    # First: dY @ B -> (M, R), then multiply by A -> (M, K)
    BLOCK_R: tl.constexpr = 32

    # We need intermediate dY @ B
    for r_start in range(0, R, BLOCK_R):
        r_idx = r_start + tl.arange(0, BLOCK_R)

        # Compute dY @ B for this R block
        dy_b_block = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)

        for n_start in range(0, N, BLOCK_N):
            n_idx = n_start + tl.arange(0, BLOCK_N)

            # Load dY
            dy_ptrs = dY_ptr + (m_idx[:, None] * stride_dym + n_idx[None, :] * stride_dyn)
            dy_mask = (m_idx[:, None] < M) & (n_idx[None, :] < N)
            dy_block = tl.load(dy_ptrs, mask=dy_mask, other=0.0)

            # Load B: (BLOCK_N, BLOCK_R)
            b_ptrs = B_ptr + (n_idx[:, None] * stride_bn + r_idx[None, :] * stride_br)
            b_mask = (n_idx[:, None] < N) & (r_idx[None, :] < R)
            b_block = tl.load(b_ptrs, mask=b_mask, other=0.0)

            dy_b_block += tl.dot(dy_block, b_block)

        # Now multiply by A: dy_b_block @ A
        # Convert dy_b_block to bf16 to match unfused behavior (F.linear produces bf16)
        dy_b_block_bf16 = dy_b_block.to(tl.bfloat16)

        # dy_b_block: (BLOCK_M, BLOCK_R), A[r_idx, :]: (BLOCK_R, K)
        # But we only want the K columns in k_idx
        # Load A: (BLOCK_R, BLOCK_K)
        a_ptrs = A_ptr + (r_idx[:, None] * stride_ar + k_idx[None, :] * stride_ak)
        a_mask = (r_idx[:, None] < R) & (k_idx[None, :] < K)
        a_block = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Both bf16, matching unfused behavior (cast scale to fp32 to avoid fp64)
        acc += tl.dot(dy_b_block_bf16, a_block) * scale.to(tl.float32)

    # Store dX
    dx_ptrs = dX_ptr + (m_idx[:, None] * stride_dxm + k_idx[None, :] * stride_dxk)
    dx_mask = (m_idx[:, None] < M) & (k_idx[None, :] < K)
    tl.store(dx_ptrs, acc.to(dX_ptr.dtype.element_ty), mask=dx_mask)


@triton.autotune(
    configs=[
        # BLOCK_N and BLOCK_K must be 64 to match grid computation in Python wrapper
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 64, "BLOCK_M": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 64, "BLOCK_M": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 64, "BLOCK_M": 128}, num_stages=2, num_warps=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def fused_linear_lora_bwd_dw_kernel(
    # Pointers to tensors
    dY_ptr,     # Gradient of output: (M, N)
    X_ptr,      # Input: (M, K)
    dW_ptr,     # Gradient of base weight: (N, K)
    # Tensor dimensions
    M, N, K,
    # Strides
    stride_dym, stride_dyn,
    stride_xm, stride_xk,
    stride_dwn, stride_dwk,
    # Block sizes
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """
    Backward kernel for dW: dW = dY.T @ X

    dY: (M, N) -> dY.T: (N, M)
    X: (M, K)
    dW: (N, K) = (N, M) @ (M, K)
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    n_start = pid_n * BLOCK_N
    k_start = pid_k * BLOCK_K

    n_idx = n_start + tl.arange(0, BLOCK_N)
    k_idx = k_start + tl.arange(0, BLOCK_K)

    # Accumulator for dW
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

    # Stream through M
    for m_start in range(0, M, BLOCK_M):
        m_idx = m_start + tl.arange(0, BLOCK_M)

        # Load dY.T block: we want (BLOCK_N, BLOCK_M) from dY (M, N)
        # dY[m, n] -> dY.T[n, m]
        dy_ptrs = dY_ptr + (m_idx[None, :] * stride_dym + n_idx[:, None] * stride_dyn)
        dy_mask = (m_idx[None, :] < M) & (n_idx[:, None] < N)
        dy_t_block = tl.load(dy_ptrs, mask=dy_mask, other=0.0)  # (BLOCK_N, BLOCK_M)

        # Load X block: (BLOCK_M, BLOCK_K)
        x_ptrs = X_ptr + (m_idx[:, None] * stride_xm + k_idx[None, :] * stride_xk)
        x_mask = (m_idx[:, None] < M) & (k_idx[None, :] < K)
        x_block = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # dW += dY.T @ X
        acc += tl.dot(dy_t_block, x_block)

    # Store dW
    dw_ptrs = dW_ptr + (n_idx[:, None] * stride_dwn + k_idx[None, :] * stride_dwk)
    dw_mask = (n_idx[:, None] < N) & (k_idx[None, :] < K)
    tl.store(dw_ptrs, acc.to(dW_ptr.dtype.element_ty), mask=dw_mask)


@triton.autotune(
    configs=[
        # BLOCK_R=32, BLOCK_K=64, BLOCK_N=64 must match grid computation in Python wrapper
        triton.Config({"BLOCK_R": 32, "BLOCK_K": 64, "BLOCK_M": 32, "BLOCK_N": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_R": 32, "BLOCK_K": 64, "BLOCK_M": 64, "BLOCK_N": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_R": 32, "BLOCK_K": 64, "BLOCK_M": 128, "BLOCK_N": 64}, num_stages=2, num_warps=4),
    ],
    key=["M", "N", "K", "R"],
)
@triton.jit
def fused_linear_lora_bwd_dlora_kernel(
    # Pointers to tensors
    dY_ptr,     # Gradient of output: (M, N)
    X_ptr,      # Input: (M, K)
    S_ptr,      # Saved intermediate S = X @ A.T: (M, R)
    B_ptr,      # LoRA B: (N, R)
    dA_ptr,     # Gradient of A: (R, K)
    dB_ptr,     # Gradient of B: (N, R)
    # Tensor dimensions
    M, N, K, R,
    # LoRA scale
    scale,
    # Strides
    stride_dym, stride_dyn,
    stride_xm, stride_xk,
    stride_sm, stride_sr,
    stride_bn, stride_br,
    stride_dar, stride_dak,
    stride_dbn, stride_dbr,
    # Block sizes
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Backward kernel for dA and dB.

    dB = dY.T @ S * scale
       = (N, M) @ (M, R) = (N, R)

    dA = (dY @ B).T @ X * scale
       = (R, M) @ (M, K) = (R, K)

    where (dY @ B) has shape (M, R)
    """
    pid = tl.program_id(0)

    # Determine if this block computes dA or dB based on pid
    # We use a 2D grid: (num_dB_blocks + num_dA_blocks, 1)
    # First compute all dB blocks, then all dA blocks

    num_dB_n_blocks = tl.cdiv(N, BLOCK_N)
    num_dB_r_blocks = tl.cdiv(R, BLOCK_R)
    num_dB_blocks = num_dB_n_blocks * num_dB_r_blocks

    if pid < num_dB_blocks:
        # Compute dB block
        pid_n = pid // num_dB_r_blocks
        pid_r = pid % num_dB_r_blocks

        n_start = pid_n * BLOCK_N
        r_start = pid_r * BLOCK_R

        n_idx = n_start + tl.arange(0, BLOCK_N)
        r_idx = r_start + tl.arange(0, BLOCK_R)

        # dB = dY.T @ S * scale
        acc_db = tl.zeros((BLOCK_N, BLOCK_R), dtype=tl.float32)

        for m_start in range(0, M, BLOCK_M):
            m_idx = m_start + tl.arange(0, BLOCK_M)

            # Load dY.T: (BLOCK_N, BLOCK_M)
            dy_ptrs = dY_ptr + (m_idx[None, :] * stride_dym + n_idx[:, None] * stride_dyn)
            dy_mask = (m_idx[None, :] < M) & (n_idx[:, None] < N)
            dy_t_block = tl.load(dy_ptrs, mask=dy_mask, other=0.0)

            # Load S: (BLOCK_M, BLOCK_R)
            s_ptrs = S_ptr + (m_idx[:, None] * stride_sm + r_idx[None, :] * stride_sr)
            s_mask = (m_idx[:, None] < M) & (r_idx[None, :] < R)
            s_block = tl.load(s_ptrs, mask=s_mask, other=0.0)

            acc_db += tl.dot(dy_t_block, s_block)

        # Store dB with scale
        db_ptrs = dB_ptr + (n_idx[:, None] * stride_dbn + r_idx[None, :] * stride_dbr)
        db_mask = (n_idx[:, None] < N) & (r_idx[None, :] < R)
        tl.store(db_ptrs, (acc_db * scale.to(tl.float32)).to(dB_ptr.dtype.element_ty), mask=db_mask)

    else:
        # Compute dA block
        pid_a = pid - num_dB_blocks
        num_dA_r_blocks = tl.cdiv(R, BLOCK_R)
        num_dA_k_blocks = tl.cdiv(K, BLOCK_K)

        pid_r = pid_a // num_dA_k_blocks
        pid_k = pid_a % num_dA_k_blocks

        r_start = pid_r * BLOCK_R
        k_start = pid_k * BLOCK_K

        r_idx = r_start + tl.arange(0, BLOCK_R)
        k_idx = k_start + tl.arange(0, BLOCK_K)

        # dA = (dY @ B).T @ X * scale
        # First compute dY @ B for each M block, then multiply by X
        acc_da = tl.zeros((BLOCK_R, BLOCK_K), dtype=tl.float32)

        for m_start in range(0, M, BLOCK_M):
            m_idx = m_start + tl.arange(0, BLOCK_M)

            # Compute dY @ B for this M block: (BLOCK_M, R)
            # We only need the columns r_idx
            dy_b_block = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)

            for n_start in range(0, N, BLOCK_N):
                n_idx_inner = n_start + tl.arange(0, BLOCK_N)

                # Load dY: (BLOCK_M, BLOCK_N)
                dy_ptrs = dY_ptr + (m_idx[:, None] * stride_dym + n_idx_inner[None, :] * stride_dyn)
                dy_mask = (m_idx[:, None] < M) & (n_idx_inner[None, :] < N)
                dy_block = tl.load(dy_ptrs, mask=dy_mask, other=0.0)

                # Load B: (BLOCK_N, BLOCK_R)
                b_ptrs = B_ptr + (n_idx_inner[:, None] * stride_bn + r_idx[None, :] * stride_br)
                b_mask = (n_idx_inner[:, None] < N) & (r_idx[None, :] < R)
                b_block = tl.load(b_ptrs, mask=b_mask, other=0.0)

                dy_b_block += tl.dot(dy_block, b_block)

            # Convert dy_b_block to bf16 to match unfused behavior
            dy_b_block_bf16 = dy_b_block.to(tl.bfloat16)

            # Load X: (BLOCK_M, BLOCK_K)
            x_ptrs = X_ptr + (m_idx[:, None] * stride_xm + k_idx[None, :] * stride_xk)
            x_mask = (m_idx[:, None] < M) & (k_idx[None, :] < K)
            x_block = tl.load(x_ptrs, mask=x_mask, other=0.0)

            # (dY @ B).T @ X: (BLOCK_R, BLOCK_M) @ (BLOCK_M, BLOCK_K) = (BLOCK_R, BLOCK_K)
            # Both bf16, matching unfused behavior
            acc_da += tl.dot(tl.trans(dy_b_block_bf16), x_block)

        # Store dA with scale
        da_ptrs = dA_ptr + (r_idx[:, None] * stride_dar + k_idx[None, :] * stride_dak)
        da_mask = (r_idx[:, None] < R) & (k_idx[None, :] < K)
        tl.store(da_ptrs, (acc_da * scale.to(tl.float32)).to(dA_ptr.dtype.element_ty), mask=da_mask)


# =============================================================================
# Python Wrappers
# =============================================================================

def fused_linear_lora_forward(
    x: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused forward pass: y = x @ W.T + (x @ A.T) @ B.T * scale

    Args:
        x: Input tensor of shape (*, K) where * can be any number of dims
        w: Base weight matrix of shape (N, K)
        a: LoRA A matrix of shape (R, K)
        b: LoRA B matrix of shape (N, R)
        scale: LoRA scaling factor (alpha/rank)

    Returns:
        y: Output tensor of shape (*, N)
        s: Saved intermediate S = x @ A.T of shape (*, R) for backward
    """
    # Flatten batch dimensions
    orig_shape = x.shape[:-1]
    x_flat = x.reshape(-1, x.shape[-1])  # (M, K)

    M, K = x_flat.shape
    N, _ = w.shape
    R, _ = a.shape

    # Allocate outputs
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    s = torch.empty((M, R), device=x.device, dtype=x.dtype)

    # Compute grid
    BLOCK_M = 64  # Will be autotuned
    BLOCK_N = 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Launch kernel
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

    # Reshape outputs
    y = y.reshape(*orig_shape, N)
    s = s.reshape(*orig_shape, R)

    return y, s


def fused_linear_lora_backward(
    dy: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    s: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused backward pass.

    Args:
        dy: Gradient of output, shape (*, N)
        x: Input tensor, shape (*, K)
        w: Base weight matrix, shape (N, K)
        a: LoRA A matrix, shape (R, K)
        b: LoRA B matrix, shape (N, R)
        s: Saved intermediate S = x @ A.T, shape (*, R)
        scale: LoRA scaling factor

    Returns:
        dx: Gradient of input, shape (*, K)
        dw: Gradient of base weight, shape (N, K)
        da: Gradient of LoRA A, shape (R, K)
        db: Gradient of LoRA B, shape (N, R)
    """
    # Flatten batch dimensions
    orig_shape = dy.shape[:-1]
    dy_flat = dy.reshape(-1, dy.shape[-1])
    x_flat = x.reshape(-1, x.shape[-1])
    s_flat = s.reshape(-1, s.shape[-1])

    M, N = dy_flat.shape
    _, K = x_flat.shape
    R, _ = a.shape

    # Allocate gradients
    dx = torch.empty((M, K), device=dy.device, dtype=dy.dtype)
    dw = torch.empty_like(w)
    da = torch.empty_like(a)
    db = torch.empty_like(b)

    # Launch dX kernel
    BLOCK_M = 64
    BLOCK_K = 64
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
    BLOCK_N = 64
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
    num_dB_blocks = triton.cdiv(N, 64) * triton.cdiv(R, BLOCK_R)
    num_dA_blocks = triton.cdiv(R, BLOCK_R) * triton.cdiv(K, BLOCK_K)
    grid_dlora = (num_dB_blocks + num_dA_blocks,)

    fused_linear_lora_bwd_dlora_kernel[grid_dlora](
        dy_flat, x_flat, s_flat, b, da, db,
        M, N, K, R,
        scale,
        dy_flat.stride(0), dy_flat.stride(1),
        x_flat.stride(0), x_flat.stride(1),
        s_flat.stride(0), s_flat.stride(1),
        b.stride(0), b.stride(1),
        da.stride(0), da.stride(1),
        db.stride(0), db.stride(1),
    )

    # Reshape dx
    dx = dx.reshape(*orig_shape, K)

    return dx, dw, da, db
