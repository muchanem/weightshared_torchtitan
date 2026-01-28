# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Grouped LoRA operations using torch._grouped_mm for reduced kernel launches.

This module provides optimized LoRA computations by batching multiple independent
LoRA projections into single grouped matrix multiply calls, reducing kernel
launch overhead from O(n) to O(1) per forward/backward pass.

Key optimizations:
1. SharedFeedForward: 6 matmuls -> 2 grouped_mm calls (67% reduction)
2. Attention head offsets: 6-8 matmuls -> 2 grouped_mm calls (67-75% reduction)
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn


def _compute_grouped_lora_forward_grouped_mm(
    x: torch.Tensor,
    w_a_list: List[torch.Tensor],
    w_b_list: List[torch.Tensor],
    scales: List[float],
) -> List[torch.Tensor]:
    """Compute multiple LoRA projections using grouped_mm.

    Note: This approach has overhead from x.repeat() which may outweigh benefits.
    Prefer _compute_grouped_lora_forward_bmm for better performance.
    """
    n_loras = len(w_a_list)
    if n_loras == 0:
        return []

    M = x.shape[0]
    device = x.device
    dtype = x.dtype

    ranks = [w_a.shape[0] for w_a in w_a_list]
    out_dims = [w_b.shape[0] for w_b in w_b_list]
    all_same_rank = len(set(ranks)) == 1
    all_same_out_dim = len(set(out_dims)) == 1

    if all_same_rank and all_same_out_dim:
        rank = ranks[0]
        w_a_stack = torch.stack(w_a_list, dim=0)
        w_b_stack = torch.stack(w_b_list, dim=0)
        group_sizes = torch.full((n_loras,), M, dtype=torch.int32, device=device)
        offsets = torch.cumsum(group_sizes, dim=0, dtype=torch.int32)
        x_repeated = x.repeat(n_loras, 1)

        h = torch._grouped_mm(
            x_repeated.bfloat16(),
            w_a_stack.bfloat16().transpose(-2, -1),
            offs=offsets
        )
        out = torch._grouped_mm(
            h,
            w_b_stack.bfloat16().transpose(-2, -1),
            offs=offsets
        ).to(dtype)

        outputs = []
        for i in range(n_loras):
            outputs.append(out[i*M:(i+1)*M] * scales[i])
        return outputs
    else:
        outputs = []
        for w_a, w_b, scale in zip(w_a_list, w_b_list, scales):
            h = x @ w_a.t()
            out = h @ w_b.t() * scale
            outputs.append(out)
        return outputs


def _compute_grouped_lora_forward(
    x: torch.Tensor,
    w_a_list: List[torch.Tensor],
    w_b_list: List[torch.Tensor],
    scales: List[float],
) -> List[torch.Tensor]:
    """Compute multiple LoRA projections using batched matrix multiplication.

    Uses torch.bmm to batch N independent LoRA computations into 2 batched
    matmul calls, avoiding the input duplication overhead of grouped_mm.

    Args:
        x: Input tensor of shape (batch*seq, dim)
        w_a_list: List of N weight_A tensors, each of shape (rank, in_dim)
        w_b_list: List of N weight_B tensors, each of shape (out_dim, rank)
        scales: List of N scaling factors (alpha/rank for each LoRA)

    Returns:
        List of N output tensors, each of shape (batch*seq, out_dim_i)
    """
    n_loras = len(w_a_list)
    if n_loras == 0:
        return []

    M = x.shape[0]  # batch * seq_len
    dtype = x.dtype

    # Get dimensions for each LoRA
    ranks = [w_a.shape[0] for w_a in w_a_list]
    out_dims = [w_b.shape[0] for w_b in w_b_list]

    all_same_rank = len(set(ranks)) == 1
    all_same_out_dim = len(set(out_dims)) == 1

    if all_same_rank and all_same_out_dim:
        rank = ranks[0]
        out_dim = out_dims[0]

        # Stack weights: (n_loras, rank, in_dim) and (n_loras, out_dim, rank)
        w_a_stack = torch.stack(w_a_list, dim=0)  # (n_loras, rank, in_dim)
        w_b_stack = torch.stack(w_b_list, dim=0)  # (n_loras, out_dim, rank)

        # Expand x to match batch dimension: (n_loras, M, in_dim)
        x_expanded = x.unsqueeze(0).expand(n_loras, -1, -1)

        # Batched matmul: x @ A.T -> (n_loras, M, rank)
        # x_expanded: (n_loras, M, in_dim)
        # w_a_stack.transpose(-2, -1): (n_loras, in_dim, rank)
        h = torch.bmm(x_expanded.bfloat16(), w_a_stack.bfloat16().transpose(-2, -1))

        # Batched matmul: h @ B.T -> (n_loras, M, out_dim)
        out = torch.bmm(h, w_b_stack.bfloat16().transpose(-2, -1)).to(dtype)

        # Split and apply per-LoRA scaling
        outputs = []
        for i in range(n_loras):
            outputs.append(out[i] * scales[i])

        return outputs

    else:
        # Fallback for different dimensions
        outputs = []
        for w_a, w_b, scale in zip(w_a_list, w_b_list, scales):
            h = x @ w_a.t()
            out = h @ w_b.t() * scale
            outputs.append(out)
        return outputs


def _compute_grouped_lora_forward_different_out_dims(
    x: torch.Tensor,
    w_a_list: List[torch.Tensor],
    w_b_list: List[torch.Tensor],
    scales: List[float],
) -> List[torch.Tensor]:
    """Compute grouped LoRA when output dimensions differ.

    When LoRA modules have different output dimensions (e.g., w1/w3 output hidden_dim,
    w2 outputs dim), we can still batch the w_a projections (which have same input dim)
    but need separate handling for w_b projections.

    Strategy: Group LoRAs by output dimension, run grouped_mm for each group.
    """
    n_loras = len(w_a_list)
    if n_loras == 0:
        return []

    M = x.shape[0]
    device = x.device
    dtype = x.dtype

    ranks = [w_a.shape[0] for w_a in w_a_list]
    out_dims = [w_b.shape[0] for w_b in w_b_list]

    # Check if all ranks are the same (required for first grouped_mm)
    all_same_rank = len(set(ranks)) == 1

    if not all_same_rank:
        # Cannot batch w_a projections efficiently, fall back
        outputs = []
        for w_a, w_b, scale in zip(w_a_list, w_b_list, scales):
            h = x @ w_a.t()
            out = h @ w_b.t() * scale
            outputs.append(out)
        return outputs

    rank = ranks[0]

    # First grouped_mm: batch all w_a projections (same rank)
    w_a_stack = torch.stack(w_a_list, dim=0)  # (n_loras, rank, in_dim)
    group_sizes = torch.full((n_loras,), M, dtype=torch.int32, device=device)
    offsets = torch.cumsum(group_sizes, dim=0, dtype=torch.int32)
    x_repeated = x.repeat(n_loras, 1)  # (n_loras * M, in_dim)

    h = torch._grouped_mm(
        x_repeated.bfloat16(),
        w_a_stack.bfloat16().transpose(-2, -1),
        offs=offsets
    )  # (n_loras * M, rank)

    # Split h for each LoRA
    h_split = [h[i*M:(i+1)*M] for i in range(n_loras)]

    # Group indices by output dimension
    out_dim_to_indices = {}
    for i, out_dim in enumerate(out_dims):
        if out_dim not in out_dim_to_indices:
            out_dim_to_indices[out_dim] = []
        out_dim_to_indices[out_dim].append(i)

    # Process each output dimension group
    outputs = [None] * n_loras

    for out_dim, indices in out_dim_to_indices.items():
        n_group = len(indices)
        if n_group == 1:
            # Single LoRA with this out_dim, no batching benefit
            i = indices[0]
            out = (h_split[i] @ w_b_list[i].bfloat16().t()).to(dtype) * scales[i]
            outputs[i] = out
        else:
            # Batch w_b projections for this out_dim group
            w_b_group = torch.stack([w_b_list[i] for i in indices], dim=0)  # (n_group, out_dim, rank)
            h_group = torch.cat([h_split[i] for i in indices], dim=0)  # (n_group * M, rank)
            group_sizes_b = torch.full((n_group,), M, dtype=torch.int32, device=device)
            offsets_b = torch.cumsum(group_sizes_b, dim=0, dtype=torch.int32)

            out_group = torch._grouped_mm(
                h_group,
                w_b_group.bfloat16().transpose(-2, -1),
                offs=offsets_b
            ).to(dtype)  # (n_group * M, out_dim)

            # Split and scale
            for j, i in enumerate(indices):
                outputs[i] = out_group[j*M:(j+1)*M] * scales[i]

    return outputs


class GroupedLoRAComputation(nn.Module):
    """Module that batches multiple LoRA computations using grouped_mm.

    This is used internally by SharedFeedForward and attention modules to
    reduce kernel launch overhead when multiple independent LoRA operations
    need to be computed on the same input.
    """

    def __init__(self, lora_modules: List[nn.Module]):
        """Initialize with a list of ScaledLoRAModule instances.

        Args:
            lora_modules: List of ScaledLoRAModule instances to batch together.
        """
        super().__init__()
        self.lora_modules = nn.ModuleList(lora_modules)
        self.n_loras = len(lora_modules)

        # Check if all modules have same rank (enables more efficient batching)
        if self.n_loras > 0:
            ranks = [m.rank for m in lora_modules]
            out_dims = [m.out_dim for m in lora_modules]
            self.all_same_rank = len(set(ranks)) == 1
            self.all_same_out_dim = len(set(out_dims)) == 1

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Compute all LoRA outputs using grouped_mm.

        Args:
            x: Input tensor of shape (batch, seq, dim) or (batch*seq, dim)

        Returns:
            List of output tensors, one per LoRA module
        """
        if self.n_loras == 0:
            return []

        # Flatten input if needed
        input_shape = x.shape
        if x.dim() == 3:
            x = x.view(-1, x.shape[-1])

        # Collect weights and scales
        w_a_list = [m.w_a.weight for m in self.lora_modules]
        w_b_list = [m.w_b.weight for m in self.lora_modules]
        scales = [m.alpha / m.rank for m in self.lora_modules]

        # Compute using grouped_mm
        if self.all_same_rank and self.all_same_out_dim:
            outputs = _compute_grouped_lora_forward(x, w_a_list, w_b_list, scales)
        else:
            outputs = _compute_grouped_lora_forward_different_out_dims(
                x, w_a_list, w_b_list, scales
            )

        # Reshape outputs if input was 3D
        if len(input_shape) == 3:
            batch, seq = input_shape[0], input_shape[1]
            outputs = [out.view(batch, seq, -1) for out in outputs]

        return outputs


def grouped_ffn_lora_forward(
    x: torch.Tensor,
    w1_shared: nn.Linear,
    w2_shared: nn.Linear,
    w3_shared: nn.Linear,
    w1_lora_a: torch.Tensor,
    w1_lora_b: torch.Tensor,
    w1_scale: float,
    w2_lora_a: torch.Tensor,
    w2_lora_b: torch.Tensor,
    w2_scale: float,
    w3_lora_a: torch.Tensor,
    w3_lora_b: torch.Tensor,
    w3_scale: float,
) -> torch.Tensor:
    """Compute SharedFeedForward using grouped_mm for LoRA operations.

    Original computation:
        y = w2(silu(w1(x)) * w3(x))
    where each wi(x) = shared_wi(x) + lora_i(x)

    This function batches the 6 LoRA matmuls (w1_a, w1_b, w2_a, w2_b, w3_a, w3_b)
    into 2-3 grouped_mm calls.

    Args:
        x: Input tensor (batch, seq, dim)
        w1_shared, w2_shared, w3_shared: Shared linear layers
        w{1,2,3}_lora_{a,b}: LoRA weight tensors
        w{1,2,3}_scale: LoRA scaling factors (alpha/rank)

    Returns:
        Output tensor (batch, seq, dim)
    """
    batch, seq, dim = x.shape
    M = batch * seq
    device = x.device
    dtype = x.dtype

    # Flatten input
    x_flat = x.view(M, dim)

    # === Stage 1: Compute all LoRA A projections (3 matmuls -> 1 grouped_mm) ===
    # w1, w3 have same input dim (dim) and same rank (assumed)
    # w2 has different input dim (hidden_dim), computed separately later

    # Check if w1 and w3 have same rank (they should in typical configs)
    r1, r3 = w1_lora_a.shape[0], w3_lora_a.shape[0]

    if r1 == r3:
        # Batch w1_a and w3_a projections
        w_a_13 = torch.stack([w1_lora_a, w3_lora_a], dim=0)  # (2, rank, dim)
        offsets_13 = torch.tensor([M, 2*M], dtype=torch.int32, device=device)
        x_repeated_13 = x_flat.repeat(2, 1)  # (2*M, dim)

        h_13 = torch._grouped_mm(
            x_repeated_13.bfloat16(),
            w_a_13.bfloat16().transpose(-2, -1),
            offs=offsets_13
        )  # (2*M, rank)

        h1_a = h_13[:M]  # (M, rank)
        h3_a = h_13[M:]  # (M, rank)
    else:
        # Different ranks, compute separately
        h1_a = (x_flat.bfloat16() @ w1_lora_a.bfloat16().t())
        h3_a = (x_flat.bfloat16() @ w3_lora_a.bfloat16().t())

    # === Stage 2: Compute w1_b and w3_b projections ===
    # These have same input (rank) but potentially different output (hidden_dim)
    # In standard configs, w1 and w3 both output hidden_dim

    hidden_dim_1 = w1_lora_b.shape[0]
    hidden_dim_3 = w3_lora_b.shape[0]

    if hidden_dim_1 == hidden_dim_3 and r1 == r3:
        # Batch w1_b and w3_b projections
        w_b_13 = torch.stack([w1_lora_b, w3_lora_b], dim=0)  # (2, hidden_dim, rank)
        h_a_13 = torch.cat([h1_a, h3_a], dim=0)  # (2*M, rank)

        out_13 = torch._grouped_mm(
            h_a_13,
            w_b_13.bfloat16().transpose(-2, -1),
            offs=offsets_13
        ).to(dtype)  # (2*M, hidden_dim)

        w1_lora_out = out_13[:M] * w1_scale  # (M, hidden_dim)
        w3_lora_out = out_13[M:] * w3_scale  # (M, hidden_dim)
    else:
        w1_lora_out = (h1_a @ w1_lora_b.bfloat16().t()).to(dtype) * w1_scale
        w3_lora_out = (h3_a @ w3_lora_b.bfloat16().t()).to(dtype) * w3_scale

    # === Stage 3: Compute h = silu(w1(x)) * w3(x) ===
    w1_base = w1_shared(x_flat)  # (M, hidden_dim)
    w3_base = w3_shared(x_flat)  # (M, hidden_dim)

    h1 = w1_base + w1_lora_out
    h3 = w3_base + w3_lora_out
    h = torch.nn.functional.silu(h1) * h3  # (M, hidden_dim)

    # === Stage 4: Compute w2 LoRA on h ===
    # w2 LoRA: h @ w2_a.T @ w2_b.T * scale
    h2_a = h.bfloat16() @ w2_lora_a.bfloat16().t()  # (M, rank)
    w2_lora_out = (h2_a @ w2_lora_b.bfloat16().t()).to(dtype) * w2_scale  # (M, dim)

    # === Stage 5: Final output ===
    w2_base = w2_shared(h)  # (M, dim)
    output = w2_base + w2_lora_out

    return output.view(batch, seq, dim)


def grouped_attention_head_offset_forward(
    x: torch.Tensor,
    lora_modules: List[nn.Module],
    two_step: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """Compute attention head LoRA offsets using grouped_mm.

    For two_step=True (4 modules: head_offset, wq_only, wk_only, wv_only):
        8 matmuls -> 2 grouped_mm calls

    For two_step=False (3 modules: wq_head_offset, wk_head_offset, wv_head_offset):
        6 matmuls -> 2 grouped_mm calls

    Args:
        x: Input tensor (batch, seq, dim)
        lora_modules: List of ScaledLoRAModule instances
        two_step: Whether using two-step decomposition

    Returns:
        Tuple of offset tensors
    """
    n_loras = len(lora_modules)
    if n_loras == 0:
        return ()

    batch, seq, dim = x.shape
    M = batch * seq
    device = x.device
    dtype = x.dtype

    x_flat = x.view(M, dim)

    # Collect weights and scales
    w_a_list = [m.w_a.weight for m in lora_modules]
    w_b_list = [m.w_b.weight for m in lora_modules]
    scales = [m.alpha / m.rank for m in lora_modules]

    # Check if all have same rank
    ranks = [w.shape[0] for w in w_a_list]
    all_same_rank = len(set(ranks)) == 1

    if all_same_rank:
        rank = ranks[0]

        # Batch all w_a projections
        w_a_stack = torch.stack(w_a_list, dim=0)  # (n_loras, rank, dim)
        group_sizes = torch.full((n_loras,), M, dtype=torch.int32, device=device)
        offsets = torch.cumsum(group_sizes, dim=0, dtype=torch.int32)
        x_repeated = x_flat.repeat(n_loras, 1)  # (n_loras * M, dim)

        h = torch._grouped_mm(
            x_repeated.bfloat16(),
            w_a_stack.bfloat16().transpose(-2, -1),
            offs=offsets
        )  # (n_loras * M, rank)

        # Split h for each LoRA
        h_split = [h[i*M:(i+1)*M] for i in range(n_loras)]

        # Group by output dimension for w_b projections
        out_dims = [w.shape[0] for w in w_b_list]
        out_dim_to_indices = {}
        for i, out_dim in enumerate(out_dims):
            if out_dim not in out_dim_to_indices:
                out_dim_to_indices[out_dim] = []
            out_dim_to_indices[out_dim].append(i)

        outputs = [None] * n_loras

        for out_dim, indices in out_dim_to_indices.items():
            n_group = len(indices)
            if n_group == 1:
                i = indices[0]
                out = (h_split[i] @ w_b_list[i].bfloat16().t()).to(dtype) * scales[i]
                outputs[i] = out.view(batch, seq, -1)
            else:
                w_b_group = torch.stack([w_b_list[i] for i in indices], dim=0)
                h_group = torch.cat([h_split[i] for i in indices], dim=0)
                group_sizes_b = torch.full((n_group,), M, dtype=torch.int32, device=device)
                offsets_b = torch.cumsum(group_sizes_b, dim=0, dtype=torch.int32)

                out_group = torch._grouped_mm(
                    h_group,
                    w_b_group.bfloat16().transpose(-2, -1),
                    offs=offsets_b
                ).to(dtype)

                for j, i in enumerate(indices):
                    outputs[i] = (out_group[j*M:(j+1)*M] * scales[i]).view(batch, seq, -1)

        return tuple(outputs)

    else:
        # Different ranks, fall back to sequential
        outputs = []
        for m in lora_modules:
            out = m(x)
            outputs.append(out)
        return tuple(outputs)
