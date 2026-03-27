# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Batched LoRA operations with pre-stacked weights.

Key insight: torch.bmm with pre-stacked weights reduces kernel launches from 4 to 2
for two independent LoRA computations. But runtime torch.stack() negates this benefit
by adding copy kernels. Solution: store weights in stacked format from initialization.

Performance (measured on H200):
- Sequential (4 matmuls): 0.071 ms, 15 kernel events
- Pre-stacked bmm (2 batched matmuls): 0.069 ms, 9 kernel events
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.protocols.module import Module


class BatchedLoRAModule(Module):
    """LoRA module with weights pre-stacked for batched matmul.

    Stores weights as (n_loras, rank, in_dim) and (n_loras, out_dim, rank) tensors
    so that batched operations use torch.bmm without runtime stacking overhead.

    Args:
        n_loras: Number of LoRA modules to batch together
        in_dim: Input dimension
        out_dim: Output dimension
        rank: LoRA rank
        alpha: LoRA scaling factor (applied as alpha/rank). Default: 2*rank (same as ScaledLoRAModule)
    """

    def __init__(
        self,
        n_loras: int,
        in_dim: int,
        out_dim: int,
        rank: int,
        alpha: float = None,
    ):
        super().__init__()
        self.n_loras = n_loras
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank * 2  # Match ScaledLoRAModule default
        self.scale = self.alpha / rank

        # Pre-stacked weights: (n_loras, rank, in_dim) and (n_loras, out_dim, rank)
        # This avoids runtime torch.stack() overhead
        self.w_a = nn.Parameter(torch.empty(n_loras, rank, in_dim))
        self.w_b = nn.Parameter(torch.empty(n_loras, out_dim, rank))

        self._init_weights()

    def _init_weights(self):
        """Initialize with Kaiming for w_a, zeros for w_b (standard LoRA init)."""
        for i in range(self.n_loras):
            nn.init.kaiming_uniform_(self.w_a[i])
        nn.init.zeros_(self.w_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute all LoRA outputs using batched matmul.

        Args:
            x: Input tensor of shape (batch*seq, dim)

        Returns:
            Tensor of shape (n_loras, batch*seq, out_dim)
        """
        M = x.shape[0]

        # Expand x for batched matmul: (n_loras, M, in_dim)
        # Note: expand() creates a view with strided access, no data copy
        x_expanded = x.unsqueeze(0).expand(self.n_loras, -1, -1)

        # Batched matmul: x @ A.T -> (n_loras, M, rank)
        h = torch.bmm(x_expanded, self.w_a.transpose(-2, -1))

        # Batched matmul: h @ B.T -> (n_loras, M, out_dim)
        out = torch.bmm(h, self.w_b.transpose(-2, -1))

        return out * self.scale

    def get_individual_weights(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get views into the stacked weights for a specific LoRA index.

        Useful for compatibility with code that expects individual weight tensors.
        Returns views (not copies) so modifications affect the stacked tensor.
        """
        return self.w_a[idx], self.w_b[idx]


class BatchedFFNLoRA(Module):
    """FFN with batched LoRA for w1 and w3 (which share same input).

    w1 and w3 LoRA computations are batched together since they:
    - Take the same input x
    - Have the same input dimension
    - Have the same rank (typically)
    - Have the same output dimension (hidden_dim)

    w2 is computed separately since its input is different (the gated output).

    This reduces kernel launches from 6 to 4 for the LoRA path:
    - w1+w3 LoRA: 4 matmuls -> 2 batched matmuls
    - w2 LoRA: 2 matmuls (unchanged)
    """

    def __init__(
        self,
        shared_w1: nn.Linear,
        shared_w2: nn.Linear,
        shared_w3: nn.Linear,
        dim: int,
        hidden_dim: int,
        rank: int,
        alpha: float = None,
    ):
        super().__init__()
        self.shared_w1 = shared_w1
        self.shared_w2 = shared_w2
        self.shared_w3 = shared_w3
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank * 2  # Match ScaledLoRAModule default
        self.scale = self.alpha / rank

        # Batched w1+w3 LoRA (same input dim, same output dim)
        self.w13_lora = BatchedLoRAModule(
            n_loras=2,
            in_dim=dim,
            out_dim=hidden_dim,
            rank=rank,
            alpha=self.alpha,
        )

        # Separate w2 LoRA (different input: hidden_dim -> dim)
        self.w2_lora_a = nn.Linear(hidden_dim, rank, bias=False)
        self.w2_lora_b = nn.Linear(rank, dim, bias=False)

        # Initialize w2 LoRA
        nn.init.kaiming_uniform_(self.w2_lora_a.weight)
        nn.init.zeros_(self.w2_lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with batched LoRA.

        Args:
            x: Input tensor (batch, seq, dim)

        Returns:
            Output tensor (batch, seq, dim)
        """
        batch, seq, dim = x.shape
        x_flat = x.view(-1, dim)

        # === Stage 1: Batched w1+w3 LoRA ===
        # Returns (2, M, hidden_dim)
        lora_13 = self.w13_lora(x_flat)
        w1_lora = lora_13[0]  # (M, hidden_dim)
        w3_lora = lora_13[1]  # (M, hidden_dim)

        # === Stage 2: Base projections + LoRA ===
        h1 = self.shared_w1(x_flat) + w1_lora
        h3 = self.shared_w3(x_flat) + w3_lora

        # === Stage 3: SwiGLU activation ===
        h = F.silu(h1) * h3

        # === Stage 4: w2 projection + LoRA ===
        w2_base = self.shared_w2(h)
        w2_lora = self.w2_lora_b(self.w2_lora_a(h)) * self.scale

        output = w2_base + w2_lora
        return output.view(batch, seq, dim)


def create_batched_ffn_from_shared(
    shared_weights: dict,
    dim: int,
    hidden_dim: int,
    lora_rank: int,
) -> BatchedFFNLoRA:
    """Create a BatchedFFNLoRA from shared weight dict.

    Args:
        shared_weights: Dict with "w1", "w2", "w3" nn.Linear modules
        dim: Model dimension
        hidden_dim: FFN hidden dimension
        lora_rank: LoRA rank

    Returns:
        BatchedFFNLoRA module
    """
    return BatchedFFNLoRA(
        shared_w1=shared_weights["w1"],
        shared_w2=shared_weights["w2"],
        shared_w3=shared_weights["w3"],
        dim=dim,
        hidden_dim=hidden_dim,
        rank=lora_rank,
    )
