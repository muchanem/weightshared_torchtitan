# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Weight sharing modules for parameter-efficient transformer models.

This module implements various weight sharing techniques:
1. Attention sharing - Share Q/K/V projection weights with per-head LoRA offsets
2. Layer sharing - Share transformer block weights with per-layer LoRA adapters

The key components are:
- broadcast_add: Efficiently broadcast shared weights to full dimension
- ScaledLoRAModule: LoRA adapter with alpha/rank scaling
- TiedLinear: Wrapper that ties linear weights to another module
- SharedAttention: Attention with shared base weights and LoRA offsets
- SharedLinearWithLoRA: Linear layer with shared weights and per-instance LoRA
- SharedFeedForward: FFN with shared weights and LoRA adapters
- SharedTransformerBlock: Complete transformer block with weight sharing
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.models.common import (
    FlexAttentionWrapper,
    ScaledDotProductAttentionWrapper,
    VarlenAttentionWrapper,
    VarlenMetadata,
    RMSNorm,
)
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.protocols.module import Module
from .batched_lora import BatchedLoRAModule


def broadcast_add(
    base: torch.Tensor, delta: torch.Tensor, *, dim: int = -1, g: int = 1
) -> torch.Tensor:
    """Broadcast a smaller base tensor and add to a larger delta tensor.

    This function efficiently broadcasts shared projection weights (which may have
    fewer heads) to match the full head count, then adds per-head LoRA deltas.

    Args:
        base: Shared projection tensor of shape (..., g*D, ...) where g is the
            number of template heads and D is head_dim.
        delta: Per-head LoRA delta tensor of shape (..., H*D, ...) where H is
            the total number of heads.
        dim: The dimension containing the head features. Default: -1.
        g: Number of template heads in the shared projection. g=1 means full
            sharing across all heads.

    Returns:
        Tensor shaped like delta, equal to broadcast(base) + delta.

    Example:
        >>> # 4 heads sharing 1 template, head_dim=64
        >>> base = torch.randn(2, 128, 64)    # (batch, seq, 1*64)
        >>> delta = torch.randn(2, 128, 256)  # (batch, seq, 4*64)
        >>> result = broadcast_add(base, delta, dim=-1, g=1)
        >>> result.shape
        torch.Size([2, 128, 256])
    """
    if dim < 0:
        dim += base.dim()

    big = delta.shape[dim]  # H*D
    small = base.shape[dim]  # g*D
    assert big % small == 0, "delta and base feature dims incompatible"
    heads_per_group = big // small  # H / g
    D = small // g  # single-head feature size

    # Reshape base to (..., g, D, ...)
    new_shape = list(base.shape)
    new_shape[dim : dim + 1] = [g, D]
    base = base.reshape(*new_shape)

    # Insert broadcast axis after g: (..., g, 1, D, ...)
    base = base.unsqueeze(dim + 1)
    # Expand over heads_per_group: (..., g, H/g, D, ...)
    base = base.expand(
        *base.shape[: dim + 1], heads_per_group, D, *base.shape[dim + 3 :]
    )

    # Flatten back to (..., H*D, ...)
    base = base.reshape(*delta.shape)

    return base + delta


class TiedLinear(Module):
    """Linear layer that shares weights with another linear module.

    This wrapper enables weight sharing between linear layers without duplicating
    parameters. The tied linear uses the weight from the source module.

    Args:
        source: The source linear module whose weights to share.

    Example:
        >>> wq_base = nn.Linear(256, 256, bias=False)
        >>> wk_tied = TiedLinear(wq_base)  # Shares weights with wq_base
    """

    def __init__(self, source: nn.Linear):
        super().__init__()
        self.source = source

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.source.weight)


class ScaledLoRAModule(Module):
    """Low-rank adaptation module.

    Implements LoRA: output = x @ A @ B

    Note: Scaling factor (alpha/rank) is NOT used. The original LoRA paper uses
    scaling to maintain stable training dynamics when changing rank, but for
    weight sharing where we train from scratch, the learning rate can be
    adjusted instead. Removing scaling eliminates a kernel launch per operation.

    Args:
        in_dim: Input dimension.
        out_dim: Output dimension.
        rank: LoRA rank (bottleneck dimension).
        alpha: Unused, kept for API compatibility.

    Example:
        >>> lora = ScaledLoRAModule(256, 512, rank=8)
        >>> x = torch.randn(2, 128, 256)
        >>> delta = lora(x)  # shape: (2, 128, 512)
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
        # Keep alpha for API compatibility but don't use it
        self.alpha = alpha if alpha is not None else rank * 2

        self.w_a = nn.Linear(in_dim, rank, bias=False)
        self.w_b = nn.Linear(rank, out_dim, bias=False)

    def reset_parameters(self) -> None:
        """Initialize LoRA weights. A is kaiming, B is zeros."""
        nn.init.kaiming_uniform_(self.w_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.w_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute low-rank delta.

        Args:
            x: Input tensor of shape (..., in_dim).

        Returns:
            Delta tensor of shape (..., out_dim).
        """
        # Simple: x @ A.T @ B.T (no scaling)
        return self.w_b(self.w_a(x))


class SharedLinearWithLoRA(Module):
    """Linear layer with shared base weights and per-instance LoRA adapter.

    This module enables weight sharing across layers while allowing each layer
    to learn unique adjustments via LoRA.

    Note on performance: Various optimizations were investigated (torch.addmm
    epilogue fusion, batched matmuls, custom Triton kernels) but none provided
    meaningful E2E improvement because:
    1. TorchInductor already fuses elementwise adds via Triton kernels
    2. torch.addmm introduces Memcpy DtoD overhead that negates fusion benefits
    3. LoRA matmuls are memory-bound (AI ~200 vs ridge point 412)
    4. B matmuls cannot be batched (different inputs per LoRA)

    A true speedup would require a fused LoRA GEMM kernel that keeps
    intermediates in registers, plus corresponding backward kernels.

    Args:
        shared_linear: The shared base linear layer.
        in_features: Input dimension.
        out_features: Output dimension.
        lora_rank: LoRA rank. Set to 0 to disable LoRA.

    Example:
        >>> shared_w1 = nn.Linear(256, 1024, bias=False)
        >>> layer1_w1 = SharedLinearWithLoRA(shared_w1, 256, 1024, lora_rank=8)
        >>> layer2_w1 = SharedLinearWithLoRA(shared_w1, 256, 1024, lora_rank=8)
    """

    def __init__(
        self,
        shared_linear: nn.Linear,
        in_features: int,
        out_features: int,
        lora_rank: int,
    ):
        super().__init__()
        self.shared_linear = shared_linear
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank

        if lora_rank > 0:
            self.lora = ScaledLoRAModule(in_features, out_features, rank=lora_rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.shared_linear(x)
        if self.lora_rank > 0:
            return base_output + self.lora(x)
        return base_output

    def reset_parameters(self) -> None:
        if self.lora_rank > 0:
            self.lora.reset_parameters()


def _qkv_str_to_attr(s: str) -> str:
    """Convert q/k/v string to attribute name."""
    mapping = {"q": "wq_base", "k": "wk_base", "v": "wv_base"}
    if s not in mapping:
        raise ValueError(f"Invalid weight name: {s}. Expected 'q', 'k', or 'v'.")
    return mapping[s]


class SharedAttention(Module):
    """Attention module with shared weights and per-head LoRA offsets.

    Supports various sharing modes:
    - QKV sharing: Tie Q, K, V projection weights together
    - Head sharing: Share weights across attention heads with grouped templates
    - LoRA offsets: Per-head learned offsets via low-rank adapters

    Args:
        dim: Model dimension.
        n_heads: Number of attention heads.
        n_kv_heads: Number of key/value heads.
        head_dim: Dimension per attention head.
        qk_norm: Whether to apply QK normalization.
        norm_eps: Epsilon for normalization layers.
        attn_type: Attention implementation type ("sdpa", "flex", "varlen").
        qkv_sharing: Groups of tied projections, e.g., [["q","k","v"]].
        head_sharing: Whether to share weights across heads.
        grouping: Number of template heads per group.
        rank: LoRA rank for offset adapters.
        two_step: Use two-step decomposition (head offset + projection offset).
        use_grouped_mm: Whether to use grouped_mm for LoRA operations.
    """

    def __init__(
        self,
        *,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        qk_norm: bool,
        norm_eps: float,
        attn_type: str = "sdpa",
        qkv_sharing: Optional[Tuple[Tuple[str, ...], ...]] = None,
        head_sharing: bool = False,
        grouping: int = 1,
        rank: int = 8,
        two_step: bool = False,
        use_grouped_mm: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_kv_heads = (
            n_heads
            if n_kv_heads is None
            else n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.scaling = self.head_dim**-0.5
        self.attn_type = attn_type

        self.qkv_sharing = qkv_sharing
        self.head_sharing = head_sharing
        self.grouping = grouping if grouping else 1
        self.two_step = two_step
        self.rank = rank
        self.use_grouped_mm = use_grouped_mm and rank > 0

        # QK norms (Qwen3-specific)
        if qk_norm:
            self.q_norm = RMSNorm(
                self.head_dim, eps=norm_eps, elementwise_affine=True
            )
            self.k_norm = RMSNorm(
                self.head_dim, eps=norm_eps, elementwise_affine=True
            )
        else:
            self.q_norm = None
            self.k_norm = None

        # Compute base projection dimensions
        if qkv_sharing:
            # With QKV sharing, disable GQA (all heads have same count)
            self._n_kv_heads_actual = self.n_heads
            base_dim = (
                self.head_dim * self.grouping
                if head_sharing
                else self.head_dim * self.n_heads
            )

            # Create tied projections based on sharing groups
            for weight_group in qkv_sharing:
                for i, weight_name in enumerate(weight_group):
                    if i == 0:
                        w_base = nn.Linear(self.dim, base_dim, bias=False)
                        setattr(self, _qkv_str_to_attr(weight_name), w_base)
                    else:
                        w = TiedLinear(w_base)
                        setattr(self, _qkv_str_to_attr(weight_name), w)
        else:
            self._n_kv_heads_actual = self.n_kv_heads
            q_base_dim = (
                self.head_dim * self.grouping
                if head_sharing
                else self.head_dim * self.n_heads
            )
            kv_base_dim = (
                self.head_dim * self.grouping
                if head_sharing
                else self.head_dim * self.n_kv_heads
            )

            self.wq_base = nn.Linear(self.dim, q_base_dim, bias=False)
            self.wk_base = nn.Linear(self.dim, kv_base_dim, bias=False)
            self.wv_base = nn.Linear(self.dim, kv_base_dim, bias=False)

        # Create LoRA offset modules
        if two_step:
            self.head_offset = ScaledLoRAModule(
                self.dim, self.n_heads * self.head_dim, rank=rank
            )
            self.wq_only_offset = ScaledLoRAModule(self.dim, self.head_dim, rank=rank)
            self.wk_only_offset = ScaledLoRAModule(self.dim, self.head_dim, rank=rank)
            self.wv_only_offset = ScaledLoRAModule(self.dim, self.head_dim, rank=rank)
        else:
            self.wq_offset = ScaledLoRAModule(
                self.dim, self.n_heads * self.head_dim, rank=rank
            )
            self.wk_offset = ScaledLoRAModule(
                self.dim, self._n_kv_heads_actual * self.head_dim, rank=rank
            )
            self.wv_offset = ScaledLoRAModule(
                self.dim, self._n_kv_heads_actual * self.head_dim, rank=rank
            )

        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

        # Inner attention implementation
        match self.attn_type:
            case "flex":
                self.inner_attention = FlexAttentionWrapper()
            case "varlen":
                self.inner_attention = VarlenAttentionWrapper()
            case "sdpa":
                self.inner_attention = ScaledDotProductAttentionWrapper()
            case _:
                raise ValueError(f"Unknown attention type: {self.attn_type}")

    def init_weights(self, init_std: float) -> None:
        """Initialize weights."""
        for attr in ["wq_base", "wk_base", "wv_base"]:
            if hasattr(self, attr):
                w = getattr(self, attr)
                if isinstance(w, nn.Linear):
                    nn.init.trunc_normal_(w.weight, mean=0.0, std=0.02)

        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

        if self.q_norm is not None:
            self.q_norm.reset_parameters()
        if self.k_norm is not None:
            self.k_norm.reset_parameters()

        # Initialize LoRA modules
        if self.two_step:
            self.head_offset.reset_parameters()
            self.wq_only_offset.reset_parameters()
            self.wk_only_offset.reset_parameters()
            self.wv_only_offset.reset_parameters()
        else:
            self.wq_offset.reset_parameters()
            self.wk_offset.reset_parameters()
            self.wv_offset.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
        apply_rotary_emb_fn=None,
        repeat_kv_fn=None,
    ) -> torch.Tensor:
        """Forward pass with shared weights and LoRA offsets.

        Args:
            x: Input tensor of shape (batch, seq, dim).
            rope_cache: Precomputed RoPE cache.
            attention_masks: Attention masks.
            positions: Position indices for RoPE.
            apply_rotary_emb_fn: Function to apply rotary embeddings.
            repeat_kv_fn: Function to repeat K/V for GQA.

        Returns:
            Output tensor of shape (batch, seq, dim).
        """
        bsz, seq_len, _ = x.shape

        # Compute base projections
        q_base = self.wq_base(x)
        k_base = self.wk_base(x)
        v_base = self.wv_base(x)

        # Compute LoRA offsets
        if self.two_step:
            if self.use_grouped_mm:
                # Use grouped_mm: 4 LoRA modules -> 2 grouped_mm calls
                lora_modules = [
                    self.head_offset,
                    self.wq_only_offset,
                    self.wk_only_offset,
                    self.wv_only_offset,
                ]
                head_offset, wq_only, wk_only, wv_only = grouped_attention_head_offset_forward(
                    x, lora_modules, two_step=True
                )
            else:
                head_offset = self.head_offset(x)
                wq_only = self.wq_only_offset(x)
                wk_only = self.wk_only_offset(x)
                wv_only = self.wv_only_offset(x)

            wq_offset = broadcast_add(wq_only, head_offset).contiguous()
            wk_offset = broadcast_add(wk_only, head_offset).contiguous()
            wv_offset = broadcast_add(wv_only, head_offset).contiguous()
        else:
            if self.use_grouped_mm:
                # Use grouped_mm: 3 LoRA modules -> 2 grouped_mm calls
                lora_modules = [self.wq_offset, self.wk_offset, self.wv_offset]
                wq_offset, wk_offset, wv_offset = grouped_attention_head_offset_forward(
                    x, lora_modules, two_step=False
                )
            else:
                wq_offset = self.wq_offset(x)
                wk_offset = self.wk_offset(x)
                wv_offset = self.wv_offset(x)

        # Combine base and offset with broadcasting
        xq = broadcast_add(q_base, wq_offset, g=self.grouping)
        xk = broadcast_add(k_base, wk_offset, g=self.grouping)
        xv = broadcast_add(v_base, wv_offset, g=self.grouping)

        # Reshape for attention
        xq = xq.view(bsz, seq_len, -1, self.head_dim)
        if self.qkv_sharing:
            xk = xk.view(bsz, seq_len, self.n_heads, self.head_dim)
            xv = xv.view(bsz, seq_len, self.n_heads, self.head_dim)
        else:
            xk = xk.view(bsz, seq_len, -1, self.head_dim)
            xv = xv.view(bsz, seq_len, -1, self.head_dim)

        # Apply QK norms
        if self.q_norm is not None:
            xq = self.q_norm(xq)
        if self.k_norm is not None:
            xk = self.k_norm(xk)

        # Apply rotary embeddings
        if apply_rotary_emb_fn is not None:
            xq, xk = apply_rotary_emb_fn(xq, xk, rope_cache, positions)

        # Repeat KV for GQA (skip if QKV sharing)
        if not self.qkv_sharing and repeat_kv_fn is not None:
            keys = repeat_kv_fn(xk, self.n_rep)
            values = repeat_kv_fn(xv, self.n_rep)
        else:
            keys = xk
            values = xv

        # Transpose for attention: (bs, n_heads, seqlen, head_dim)
        xq = xq.transpose(1, 2)
        xk = keys.transpose(1, 2)
        xv = values.transpose(1, 2)

        # Compute attention
        match self.attn_type:
            case "flex":
                assert isinstance(attention_masks, BlockMask)
                output = self.inner_attention(
                    xq, xk, xv, block_mask=attention_masks, scale=self.scaling
                )
                output = output.transpose(1, 2).contiguous()
            case "varlen":
                assert isinstance(attention_masks, VarlenMetadata)
                output = self.inner_attention(
                    xq, xk, xv, self.head_dim, attention_masks, scale=self.scaling
                )
            case "sdpa":
                output = self.inner_attention(xq, xk, xv, scale=self.scaling)
                output = output.transpose(1, 2).contiguous()
            case _:
                raise ValueError(f"Unknown attention type: {self.attn_type}")

        output = output.view(bsz, seq_len, -1)
        return self.wo(output)


class SharedFeedForward(Module):
    """FeedForward module with shared weights and per-layer LoRA.

    Args:
        shared_weights: Dict with "w1", "w2", "w3" shared linear layers.
        dim: Model dimension.
        hidden_dim: FFN hidden dimension.
        lora_rank: LoRA rank for per-layer adapters.
        use_grouped_mm: Whether to use grouped_mm for LoRA operations.
    """

    def __init__(
        self,
        shared_weights: dict,
        dim: int,
        hidden_dim: int,
        lora_rank: int,
        use_batched_lora: bool = False,  # Sequential is faster at this scale
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.lora_rank = lora_rank
        self.use_batched_lora = use_batched_lora and lora_rank > 0

        # Shared base weights
        self.w1_shared = shared_weights["w1"]
        self.w2_shared = shared_weights["w2"]
        self.w3_shared = shared_weights["w3"]

        if self.use_batched_lora:
            # Pre-stacked LoRA weights for w1+w3 (same input dim, same output dim)
            # This avoids runtime torch.stack() overhead
            self.w13_lora = BatchedLoRAModule(
                n_loras=2,
                in_dim=dim,
                out_dim=hidden_dim,
                rank=lora_rank,
            )
            # w2 LoRA is separate (different input dim: hidden_dim -> dim)
            self.w2_lora = ScaledLoRAModule(hidden_dim, dim, rank=lora_rank)
        else:
            # Standard approach: separate LoRA modules
            self.w1 = SharedLinearWithLoRA(shared_weights["w1"], dim, hidden_dim, lora_rank)
            self.w2 = SharedLinearWithLoRA(shared_weights["w2"], hidden_dim, dim, lora_rank)
            self.w3 = SharedLinearWithLoRA(shared_weights["w3"], dim, hidden_dim, lora_rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_batched_lora:
            return self._forward_batched(x)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def _forward_batched(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with pre-stacked batched LoRA for w1+w3."""
        batch, seq, dim = x.shape
        x_flat = x.view(-1, dim)

        # Batched w1+w3 LoRA: 2 matmuls -> 2 batched matmuls (1 kernel each)
        lora_13 = self.w13_lora(x_flat)  # (2, M, hidden_dim)
        w1_lora = lora_13[0]  # (M, hidden_dim)
        w3_lora = lora_13[1]  # (M, hidden_dim)

        # Base projections + LoRA
        h1 = self.w1_shared(x_flat) + w1_lora
        h3 = self.w3_shared(x_flat) + w3_lora

        # SwiGLU activation
        h = F.silu(h1) * h3

        # w2 projection + LoRA
        output = self.w2_shared(h) + self.w2_lora(h)

        return output.view(batch, seq, self.dim)

    def init_weights(self, init_std: float) -> None:
        if self.use_batched_lora:
            # BatchedLoRAModule initializes itself; ScaledLoRAModule needs reset
            self.w2_lora.reset_parameters()
        else:
            self.w1.reset_parameters()
            self.w2.reset_parameters()
            self.w3.reset_parameters()


class SharedAttentionWithLoRA(Module):
    """Attention with shared base weights and per-layer LoRA adapters.

    This is used for layer sharing where different layers share the same
    attention weights but have unique LoRA adapters.

    Args:
        dim: Model dimension.
        n_heads: Number of attention heads.
        n_kv_heads: Number of key/value heads.
        head_dim: Dimension per attention head.
        qk_norm: Whether to apply QK normalization.
        norm_eps: Epsilon for normalization layers.
        shared_weights: Dict with "wq", "wk", "wv", "wo" shared linear layers.
        lora_rank: LoRA rank for per-layer adapters.
        attn_type: Attention implementation type.
    """

    def __init__(
        self,
        *,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        qk_norm: bool,
        norm_eps: float,
        shared_weights: dict,
        lora_rank: int,
        attn_type: str = "sdpa",
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = (
            n_heads
            if n_kv_heads is None
            else n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = head_dim
        self.scaling = self.head_dim**-0.5
        self.attn_type = attn_type

        # Shared weights with per-layer LoRA
        self.wq = SharedLinearWithLoRA(
            shared_weights["wq"],
            dim,
            n_heads * head_dim,
            lora_rank,
        )
        self.wk = SharedLinearWithLoRA(
            shared_weights["wk"],
            dim,
            self.n_kv_heads * head_dim,
            lora_rank,
        )
        self.wv = SharedLinearWithLoRA(
            shared_weights["wv"],
            dim,
            self.n_kv_heads * head_dim,
            lora_rank,
        )
        self.wo = SharedLinearWithLoRA(
            shared_weights["wo"],
            n_heads * head_dim,
            dim,
            lora_rank,
        )

        # QK norms
        if qk_norm:
            self.q_norm = RMSNorm(
                self.head_dim, eps=norm_eps, elementwise_affine=True
            )
            self.k_norm = RMSNorm(
                self.head_dim, eps=norm_eps, elementwise_affine=True
            )
        else:
            self.q_norm = None
            self.k_norm = None

        # Inner attention
        match self.attn_type:
            case "flex":
                self.inner_attention = FlexAttentionWrapper()
            case "varlen":
                self.inner_attention = VarlenAttentionWrapper()
            case "sdpa":
                self.inner_attention = ScaledDotProductAttentionWrapper()
            case _:
                raise ValueError(f"Unknown attention type: {self.attn_type}")

    def init_weights(self, init_std: float) -> None:
        self.wq.reset_parameters()
        self.wk.reset_parameters()
        self.wv.reset_parameters()
        self.wo.reset_parameters()
        if self.q_norm is not None:
            self.q_norm.reset_parameters()
        if self.k_norm is not None:
            self.k_norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
        apply_rotary_emb_fn=None,
        repeat_kv_fn=None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        # Reshape for attention
        xq = xq.view(bsz, seq_len, -1, self.head_dim)
        xk = xk.view(bsz, seq_len, -1, self.head_dim)
        xv = xv.view(bsz, seq_len, -1, self.head_dim)

        # QK norms
        if self.q_norm is not None:
            xq = self.q_norm(xq)
        if self.k_norm is not None:
            xk = self.k_norm(xk)

        # Apply rotary embeddings
        if apply_rotary_emb_fn is not None:
            xq, xk = apply_rotary_emb_fn(xq, xk, rope_cache, positions)

        # Repeat KV for GQA
        if repeat_kv_fn is not None:
            keys = repeat_kv_fn(xk, self.n_rep)
            values = repeat_kv_fn(xv, self.n_rep)
        else:
            keys = xk
            values = xv

        xq = xq.transpose(1, 2)
        xk = keys.transpose(1, 2)
        xv = values.transpose(1, 2)

        match self.attn_type:
            case "flex":
                assert isinstance(attention_masks, BlockMask)
                output = self.inner_attention(
                    xq, xk, xv, block_mask=attention_masks, scale=self.scaling
                )
                output = output.transpose(1, 2).contiguous()
            case "varlen":
                assert isinstance(attention_masks, VarlenMetadata)
                output = self.inner_attention(
                    xq, xk, xv, self.head_dim, attention_masks, scale=self.scaling
                )
            case "sdpa":
                output = self.inner_attention(xq, xk, xv, scale=self.scaling)
                output = output.transpose(1, 2).contiguous()
            case _:
                raise ValueError(f"Unknown attention type: {self.attn_type}")

        output = output.view(bsz, seq_len, -1)
        return self.wo(output)


class SharedTransformerBlock(Module):
    """Transformer block with shared weights and per-layer LoRA adapters.

    Args:
        layer_id: Layer identifier for init scaling.
        dim: Model dimension.
        n_heads: Number of attention heads.
        n_kv_heads: Number of key/value heads.
        head_dim: Dimension per attention head.
        hidden_dim: FFN hidden dimension.
        norm_eps: Epsilon for normalization layers.
        qk_norm: Whether to apply QK normalization.
        n_layers: Total number of layers.
        depth_init: Whether to use depth-dependent initialization.
        shared_attention_weights: Dict with shared attention linear layers.
        shared_ffn_weights: Dict with shared FFN linear layers.
        lora_rank: LoRA rank for per-layer adapters.
    """

    # Flag to identify weight-sharing blocks (used instead of isinstance for torch.compile compatibility)
    _is_weight_sharing_block = True

    def __init__(
        self,
        *,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        hidden_dim: int,
        norm_eps: float,
        qk_norm: bool,
        n_layers: int,
        depth_init: bool = False,
        shared_attention_weights: dict,
        shared_ffn_weights: dict,
        lora_rank: int,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.moe_enabled = False  # MoE not supported with weight sharing

        self.attention = SharedAttentionWithLoRA(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            qk_norm=qk_norm,
            norm_eps=norm_eps,
            shared_weights=shared_attention_weights,
            lora_rank=lora_rank,
        )
        self.feed_forward = SharedFeedForward(
            shared_ffn_weights, dim, hidden_dim, lora_rank
        )

        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

        if depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * n_layers) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
        apply_rotary_emb_fn=None,
        repeat_kv_fn=None,
    ) -> torch.Tensor:
        h = x + self.attention(
            self.attention_norm(x),
            rope_cache,
            attention_masks,
            positions,
            apply_rotary_emb_fn,
            repeat_kv_fn,
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def init_weights(self, buffer_device: torch.device) -> None:
        self.attention_norm.reset_parameters()
        self.ffn_norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)


class AttentionSharingTransformerBlock(Module):
    """Transformer block with attention head sharing (no layer sharing).

    This block uses SharedAttention for head-level weight sharing while keeping
    the FFN as standard (no layer sharing). Use this when you want attention
    sharing without layer sharing.

    Args:
        layer_id: Layer identifier for depth-dependent initialization.
        dim: Model dimension.
        n_heads: Number of attention heads.
        n_kv_heads: Number of key/value heads.
        head_dim: Dimension per attention head.
        hidden_dim: FFN hidden dimension.
        norm_eps: Epsilon for normalization layers.
        n_layers: Total number of layers.
        qk_norm: Whether to apply QK normalization.
        moe_enabled: Whether MoE is enabled.
        depth_init: Whether to use depth-dependent initialization.
        attention_config: AttentionSharingConfig with head_sharing, grouping, rank, etc.
    """

    # Flag to identify weight-sharing blocks (used instead of isinstance for torch.compile compatibility)
    _is_weight_sharing_block = True

    def __init__(
        self,
        *,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        hidden_dim: int,
        norm_eps: float,
        n_layers: int,
        qk_norm: bool,
        moe_enabled: bool = False,
        moe_args=None,
        moe_inter_dim: int = 0,
        depth_init: bool = False,
        attention_config,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.moe_enabled = moe_enabled

        # Use SharedAttention for head-level sharing
        self.attention = SharedAttention(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            qk_norm=qk_norm,
            norm_eps=norm_eps,
            qkv_sharing=tuple(tuple(g) for g in attention_config.qkv_sharing)
            if attention_config.qkv_sharing
            else None,
            head_sharing=attention_config.head_sharing,
            grouping=attention_config.grouping,
            rank=attention_config.rank,
            two_step=attention_config.two_step,
        )

        # Standard FFN (no layer sharing)
        if self.moe_enabled:
            from torchtitan.models.moe import MoE

            self.moe = MoE(
                moe_args,
                dim=dim,
                hidden_dim=moe_inter_dim,
            )
        else:
            self.feed_forward = FeedForward(
                dim=dim, hidden_dim=hidden_dim
            )

        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

        if depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * n_layers) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
        apply_rotary_emb_fn=None,
        repeat_kv_fn=None,
    ) -> torch.Tensor:
        h = x + self.attention(
            self.attention_norm(x),
            rope_cache,
            attention_masks,
            positions,
            apply_rotary_emb_fn,
            repeat_kv_fn,
        )

        if self.moe_enabled:
            out = h + self.moe(self.ffn_norm(h))
        else:
            out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def init_weights(self, buffer_device: torch.device) -> None:
        self.attention_norm.reset_parameters()
        self.ffn_norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        if self.moe_enabled:
            self.moe.init_weights(self.weight_init_std, buffer_device)
        else:
            self.feed_forward.init_weights(self.weight_init_std)


class FeedForward(Module):
    """Standard FeedForward module (copy for use in AttentionSharingTransformerBlock).

    This is a local copy to avoid circular imports with model.py.
    """

    def __init__(self, *, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float) -> None:
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class CombinedSharingTransformerBlock(Module):
    """Transformer block with both attention sharing AND layer sharing.

    This block combines:
    - Head-level sharing in attention (SharedAttention-style)
    - Layer-level sharing with external shared weights + LoRA

    Args:
        layer_id: Layer identifier.
        dim: Model dimension.
        n_heads: Number of attention heads.
        n_kv_heads: Number of key/value heads.
        head_dim: Dimension per attention head.
        hidden_dim: FFN hidden dimension.
        norm_eps: Epsilon for normalization layers.
        n_layers: Total number of layers.
        qk_norm: Whether to apply QK normalization.
        depth_init: Whether to use depth-dependent initialization.
        shared_attention_weights: Dict with shared "wq", "wk", "wv", "wo" linear layers.
        shared_ffn_weights: Dict with shared "w1", "w2", "w3" linear layers.
        layer_lora_rank: LoRA rank for per-layer adapters.
        attention_config: AttentionSharingConfig for head sharing settings.
    """

    # Flag to identify weight-sharing blocks (used instead of isinstance for torch.compile compatibility)
    _is_weight_sharing_block = True

    def __init__(
        self,
        *,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        hidden_dim: int,
        norm_eps: float,
        n_layers: int,
        qk_norm: bool,
        depth_init: bool = False,
        shared_attention_weights: dict,
        shared_ffn_weights: dict,
        layer_lora_rank: int,
        attention_config,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.moe_enabled = False  # MoE not supported with weight sharing

        # Combined attention: layer sharing + head sharing
        self.attention = CombinedSharingAttention(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            qk_norm=qk_norm,
            norm_eps=norm_eps,
            shared_weights=shared_attention_weights,
            layer_lora_rank=layer_lora_rank,
            attention_config=attention_config,
        )

        self.feed_forward = SharedFeedForward(
            shared_ffn_weights, dim, hidden_dim, layer_lora_rank
        )

        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

        if depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * n_layers) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
        apply_rotary_emb_fn=None,
        repeat_kv_fn=None,
    ) -> torch.Tensor:
        h = x + self.attention(
            self.attention_norm(x),
            rope_cache,
            attention_masks,
            positions,
            apply_rotary_emb_fn,
            repeat_kv_fn,
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def init_weights(self, buffer_device: torch.device) -> None:
        self.attention_norm.reset_parameters()
        self.ffn_norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)


class CombinedSharingAttention(Module):
    """Attention with both head sharing AND layer sharing.

    Combines:
    - Shared base weights across layers (from layer sharing)
    - Per-layer LoRA adapters (from layer sharing)
    - Head grouping/sharing (from attention sharing)
    - Per-head LoRA offsets (from attention sharing)

    Args:
        dim: Model dimension.
        n_heads: Number of attention heads.
        n_kv_heads: Number of key/value heads.
        head_dim: Dimension per attention head.
        qk_norm: Whether to apply QK normalization.
        norm_eps: Epsilon for normalization layers.
        shared_weights: Dict with "wq", "wk", "wv", "wo" shared linear layers.
        layer_lora_rank: LoRA rank for per-layer adapters.
        attention_config: AttentionSharingConfig for head sharing settings.
        attn_type: Attention implementation type.
        use_grouped_mm: Whether to use grouped_mm for LoRA operations.
    """

    def __init__(
        self,
        *,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        qk_norm: bool,
        norm_eps: float,
        shared_weights: dict,
        layer_lora_rank: int,
        attention_config,
        attn_type: str = "sdpa",
        use_grouped_mm: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_kv_heads = (
            n_heads
            if n_kv_heads is None
            else n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.scaling = self.head_dim**-0.5
        self.attn_type = attn_type

        # Attention sharing config
        self.head_sharing = attention_config.head_sharing
        self.grouping = attention_config.grouping if attention_config.grouping else 1
        self.head_rank = attention_config.rank
        self.two_step = attention_config.two_step
        self.qkv_sharing = attention_config.qkv_sharing
        self.use_grouped_mm = use_grouped_mm and (self.head_rank > 0 or layer_lora_rank > 0)

        # Shared weights with per-layer LoRA
        self.wq = SharedLinearWithLoRA(
            shared_weights["wq"],
            dim,
            n_heads * head_dim,
            layer_lora_rank,
        )
        self.wk = SharedLinearWithLoRA(
            shared_weights["wk"],
            dim,
            self.n_kv_heads * head_dim,
            layer_lora_rank,
        )
        self.wv = SharedLinearWithLoRA(
            shared_weights["wv"],
            dim,
            self.n_kv_heads * head_dim,
            layer_lora_rank,
        )
        self.wo = SharedLinearWithLoRA(
            shared_weights["wo"],
            n_heads * head_dim,
            dim,
            layer_lora_rank,
        )

        # Per-head LoRA offsets (for attention sharing)
        if self.head_sharing:
            if self.two_step:
                self.head_offset = ScaledLoRAModule(
                    self.dim, self.n_heads * self.head_dim, rank=self.head_rank
                )
                self.wq_only_offset = ScaledLoRAModule(
                    self.dim, self.head_dim, rank=self.head_rank
                )
                self.wk_only_offset = ScaledLoRAModule(
                    self.dim, self.head_dim, rank=self.head_rank
                )
                self.wv_only_offset = ScaledLoRAModule(
                    self.dim, self.head_dim, rank=self.head_rank
                )
            else:
                self.wq_head_offset = ScaledLoRAModule(
                    self.dim, self.n_heads * self.head_dim, rank=self.head_rank
                )
                self.wk_head_offset = ScaledLoRAModule(
                    self.dim, self.n_kv_heads * self.head_dim, rank=self.head_rank
                )
                self.wv_head_offset = ScaledLoRAModule(
                    self.dim, self.n_kv_heads * self.head_dim, rank=self.head_rank
                )

        # QK norms
        if qk_norm:
            self.q_norm = RMSNorm(
                self.head_dim, eps=norm_eps, elementwise_affine=True
            )
            self.k_norm = RMSNorm(
                self.head_dim, eps=norm_eps, elementwise_affine=True
            )
        else:
            self.q_norm = None
            self.k_norm = None

        # Inner attention
        match self.attn_type:
            case "flex":
                self.inner_attention = FlexAttentionWrapper()
            case "varlen":
                self.inner_attention = VarlenAttentionWrapper()
            case "sdpa":
                self.inner_attention = ScaledDotProductAttentionWrapper()
            case _:
                raise ValueError(f"Unknown attention type: {self.attn_type}")

    def init_weights(self, init_std: float) -> None:
        self.wq.reset_parameters()
        self.wk.reset_parameters()
        self.wv.reset_parameters()
        self.wo.reset_parameters()

        if self.head_sharing:
            if self.two_step:
                self.head_offset.reset_parameters()
                self.wq_only_offset.reset_parameters()
                self.wk_only_offset.reset_parameters()
                self.wv_only_offset.reset_parameters()
            else:
                self.wq_head_offset.reset_parameters()
                self.wk_head_offset.reset_parameters()
                self.wv_head_offset.reset_parameters()

        if self.q_norm is not None:
            self.q_norm.reset_parameters()
        if self.k_norm is not None:
            self.k_norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
        apply_rotary_emb_fn=None,
        repeat_kv_fn=None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        # Get base projections (with layer LoRA)
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        # Apply head-level offsets if head sharing enabled
        if self.head_sharing:
            if self.two_step:
                if self.use_grouped_mm:
                    # Use grouped_mm: 4 LoRA modules -> 2 grouped_mm calls
                    lora_modules = [
                        self.head_offset,
                        self.wq_only_offset,
                        self.wk_only_offset,
                        self.wv_only_offset,
                    ]
                    head_offset, wq_only, wk_only, wv_only = grouped_attention_head_offset_forward(
                        x, lora_modules, two_step=True
                    )
                else:
                    head_offset = self.head_offset(x)
                    wq_only = self.wq_only_offset(x)
                    wk_only = self.wk_only_offset(x)
                    wv_only = self.wv_only_offset(x)

                wq_offset = broadcast_add(wq_only, head_offset).contiguous()
                wk_offset = broadcast_add(wk_only, head_offset).contiguous()
                wv_offset = broadcast_add(wv_only, head_offset).contiguous()
            else:
                if self.use_grouped_mm:
                    # Use grouped_mm: 3 LoRA modules -> 2 grouped_mm calls
                    lora_modules = [self.wq_head_offset, self.wk_head_offset, self.wv_head_offset]
                    wq_offset, wk_offset, wv_offset = grouped_attention_head_offset_forward(
                        x, lora_modules, two_step=False
                    )
                else:
                    wq_offset = self.wq_head_offset(x)
                    wk_offset = self.wk_head_offset(x)
                    wv_offset = self.wv_head_offset(x)

            # Add head offsets
            xq = xq + wq_offset
            xk = xk + wk_offset
            xv = xv + wv_offset

        # Reshape for attention
        xq = xq.view(bsz, seq_len, -1, self.head_dim)
        xk = xk.view(bsz, seq_len, -1, self.head_dim)
        xv = xv.view(bsz, seq_len, -1, self.head_dim)

        # QK norms
        if self.q_norm is not None:
            xq = self.q_norm(xq)
        if self.k_norm is not None:
            xk = self.k_norm(xk)

        # Apply rotary embeddings
        if apply_rotary_emb_fn is not None:
            xq, xk = apply_rotary_emb_fn(xq, xk, rope_cache, positions)

        # Repeat KV for GQA
        if repeat_kv_fn is not None:
            keys = repeat_kv_fn(xk, self.n_rep)
            values = repeat_kv_fn(xv, self.n_rep)
        else:
            keys = xk
            values = xv

        xq = xq.transpose(1, 2)
        xk = keys.transpose(1, 2)
        xv = values.transpose(1, 2)

        match self.attn_type:
            case "flex":
                assert isinstance(attention_masks, BlockMask)
                output = self.inner_attention(
                    xq, xk, xv, block_mask=attention_masks, scale=self.scaling
                )
                output = output.transpose(1, 2).contiguous()
            case "varlen":
                assert isinstance(attention_masks, VarlenMetadata)
                output = self.inner_attention(
                    xq, xk, xv, self.head_dim, attention_masks, scale=self.scaling
                )
            case "sdpa":
                output = self.inner_attention(xq, xk, xv, scale=self.scaling)
                output = output.transpose(1, 2).contiguous()
            case _:
                raise ValueError(f"Unknown attention type: {self.attn_type}")

        output = output.view(bsz, seq_len, -1)
        return self.wo(output)
