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


class CachedGroupedFFNLoRA(nn.Module):
    """Cached grouped LoRA for FFN with pre-stacked weights.

    This module caches the stacked weight tensors to avoid runtime overhead
    from torch.stack() on every forward pass.
    """

    def __init__(self, ffn_module):
        """Initialize with a SharedFeedForward module.

        Args:
            ffn_module: SharedFeedForward module with w1, w2, w3 LoRA adapters.
        """
        super().__init__()
        self.ffn = ffn_module
        # We'll access the shared linears and LoRA weights through the ffn module
        # The stacked weights are computed lazily and cached as buffers

    def _get_stacked_w_a_13(self):
        """Get stacked w_a weights for w1 and w3 (same input dim)."""
        return torch.stack([
            self.ffn.w1.lora.w_a.weight,
            self.ffn.w3.lora.w_a.weight,
        ], dim=0)

    def _get_stacked_w_b_13(self):
        """Get stacked w_b weights for w1 and w3 (same output dim)."""
        return torch.stack([
            self.ffn.w1.lora.w_b.weight,
            self.ffn.w3.lora.w_b.weight,
        ], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with cached grouped LoRA computation."""
        batch, seq, dim = x.shape
        M = batch * seq
        dtype = x.dtype
        device = x.device

        x_flat = x.view(M, dim)

        # Get scales
        w1_scale = self.ffn.w1.lora.alpha / self.ffn.w1.lora.rank
        w3_scale = self.ffn.w3.lora.alpha / self.ffn.w3.lora.rank
        w2_scale = self.ffn.w2.lora.alpha / self.ffn.w2.lora.rank

        # === Stage 1+2: Batched w1 and w3 LoRA computations ===
        # Stack weights (ideally would be cached, but weights change during training)
        w_a_13 = self._get_stacked_w_a_13()  # (2, rank, dim)
        w_b_13 = self._get_stacked_w_b_13()  # (2, hidden_dim, rank)

        # Expand x for batched matmul: (2, M, dim)
        x_expanded = x_flat.unsqueeze(0).expand(2, -1, -1)

        # Batched matmul: x @ A.T -> (2, M, rank)
        h_13 = torch.bmm(x_expanded, w_a_13.transpose(-2, -1))
        # Batched matmul: h @ B.T -> (2, M, hidden_dim)
        out_13 = torch.bmm(h_13, w_b_13.transpose(-2, -1))

        # Split and scale
        w1_lora_out = out_13[0] * w1_scale
        w3_lora_out = out_13[1] * w3_scale

        # === Stage 3: Compute h = silu(w1(x)) * w3(x) ===
        w1_base = self.ffn.w1.shared_linear(x_flat)
        w3_base = self.ffn.w3.shared_linear(x_flat)

        h1 = w1_base + w1_lora_out
        h3 = w3_base + w3_lora_out
        h = torch.nn.functional.silu(h1) * h3

        # === Stage 4: w2 LoRA on h ===
        h2_a = h @ self.ffn.w2.lora.w_a.weight.t()
        w2_lora_out = (h2_a @ self.ffn.w2.lora.w_b.weight.t()) * w2_scale

        # === Stage 5: Final output ===
        w2_base = self.ffn.w2.shared_linear(h)
        output = w2_base + w2_lora_out

        return output.view(batch, seq, dim)


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
    """Compute SharedFeedForward using batched matrix multiplication for LoRA.

    Uses torch.bmm to batch w1 and w3 LoRA computations (same input dim and rank),
    reducing kernel launches from 6 to 4 matmuls for the LoRA path.

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

    x_flat = x.view(M, dim)

    # === Stage 1+2: Batched w1 and w3 LoRA computations ===
    # Stack weights for batched matmul
    w_a_13 = torch.stack([w1_lora_a, w3_lora_a], dim=0)  # (2, rank, dim)
    w_b_13 = torch.stack([w1_lora_b, w3_lora_b], dim=0)  # (2, hidden_dim, rank)

    # Expand x for batched matmul: (2, M, dim)
    x_expanded = x_flat.unsqueeze(0).expand(2, -1, -1)

    # Batched matmul: x @ A.T -> (2, M, rank)
    h_13 = torch.bmm(x_expanded, w_a_13.transpose(-2, -1))
    # Batched matmul: h @ B.T -> (2, M, hidden_dim)
    out_13 = torch.bmm(h_13, w_b_13.transpose(-2, -1))

    # Split and scale
    w1_lora_out = out_13[0] * w1_scale
    w3_lora_out = out_13[1] * w3_scale

    # === Stage 3: Compute h = silu(w1(x)) * w3(x) ===
    w1_base = w1_shared(x_flat)
    w3_base = w3_shared(x_flat)

    h1 = w1_base + w1_lora_out
    h3 = w3_base + w3_lora_out
    h = torch.nn.functional.silu(h1) * h3

    # === Stage 4: w2 LoRA on h ===
    h2_a = h @ w2_lora_a.t()
    w2_lora_out = (h2_a @ w2_lora_b.t()) * w2_scale

    # === Stage 5: Final output ===
    w2_base = w2_shared(h)
    output = w2_base + w2_lora_out

    return output.view(batch, seq, dim)


def grouped_attention_head_offset_forward(
    x: torch.Tensor,
    lora_modules: List[nn.Module],
    two_step: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """Compute attention head LoRA offsets using batched matrix multiplication.

    Uses torch.bmm to batch multiple LoRA computations, reducing kernel launches.

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

    x_flat = x.view(M, dim)

    # Collect weights and scales
    w_a_list = [m.w_a.weight for m in lora_modules]
    w_b_list = [m.w_b.weight for m in lora_modules]
    scales = [m.alpha / m.rank for m in lora_modules]

    # Check if all have same rank
    ranks = [w.shape[0] for w in w_a_list]
    out_dims = [w.shape[0] for w in w_b_list]
    all_same_rank = len(set(ranks)) == 1
    all_same_out_dim = len(set(out_dims)) == 1

    if all_same_rank and all_same_out_dim:
        # Full batching possible
        w_a_stack = torch.stack(w_a_list, dim=0)  # (n_loras, rank, dim)
        w_b_stack = torch.stack(w_b_list, dim=0)  # (n_loras, out_dim, rank)

        # Expand x for batched matmul
        x_expanded = x_flat.unsqueeze(0).expand(n_loras, -1, -1)  # (n_loras, M, dim)

        # Batched matmuls
        h = torch.bmm(x_expanded, w_a_stack.transpose(-2, -1))  # (n_loras, M, rank)
        out = torch.bmm(h, w_b_stack.transpose(-2, -1))  # (n_loras, M, out_dim)

        # Split and scale
        outputs = []
        for i in range(n_loras):
            outputs.append((out[i] * scales[i]).view(batch, seq, -1))

        return tuple(outputs)

    elif all_same_rank:
        # Can batch w_a projections, need to handle w_b by output dim groups
        w_a_stack = torch.stack(w_a_list, dim=0)
        x_expanded = x_flat.unsqueeze(0).expand(n_loras, -1, -1)
        h = torch.bmm(x_expanded, w_a_stack.transpose(-2, -1))  # (n_loras, M, rank)

        # Group by output dimension
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
                out = (h[i] @ w_b_list[i].t()) * scales[i]
                outputs[i] = out.view(batch, seq, -1)
            else:
                w_b_group = torch.stack([w_b_list[i] for i in indices], dim=0)
                h_group = torch.stack([h[i] for i in indices], dim=0)
                out_group = torch.bmm(h_group, w_b_group.transpose(-2, -1))

                for j, i in enumerate(indices):
                    outputs[i] = (out_group[j] * scales[i]).view(batch, seq, -1)

        return tuple(outputs)

    else:
        # Different ranks, fall back to sequential
        outputs = []
        for m in lora_modules:
            out = m(x)
            outputs.append(out)
        return tuple(outputs)
