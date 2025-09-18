"""
Module for definition of Low-Rank Adaptation (LoRA) Triton kernels.

See "LoRA: Low-Rank Adaptation of Large Language Models"
(https://arxiv.org/abs/2106.09685).

Credit to `unsloth` (https://unsloth.ai/) for inspiration for this implementation.
"""

from typing import Callable

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from .swiglu import (
    swiglu_act_backward as swiglu_backward,
    swiglu_act as swiglu_forward
)

def matmul_lora(
    X: torch.Tensor,
    W: torch.Tensor,
    A: torch.Tensor | None,
    B: torch.Tensor | None,
    s: float | None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Efficient fused matmul + LoRA computation.

    Args:
        X: Input tensor [*, in_features]
        W: Base weight matrix [out_features, in_features]
        A: LoRA A matrix [rank, in_features]
        B: LoRA B matrix [out_features, rank]
        s: LoRA scaling factor
        out: Optional output tensor for inplace operations

    Returns:
        Result of X @ W + X @ A @ B
    """
    dtype = X.dtype
    reshape = False
    if X.dim() == 3:
        batch, seq_len, _ = X.shape
        X = X.view(-1, X.shape[-1])
        reshape = True

    out = torch.matmul(X, W.t(), out=out)

    if A is not None:
        A, B = A.t().to(dtype), B.t().to(dtype)  # type: ignore[union-attr]
        out += s * X @ A @ B

    return out.view(batch, seq_len, -1) if reshape else out


class LoRA_MLP(torch.autograd.Function):
    """Optimized LoRA MLP implementation."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        X: torch.Tensor,
        gate_weight: torch.Tensor,
        gate_A: torch.Tensor | None,
        gate_B: torch.Tensor | None,
        gate_scale: float,
        up_weight: torch.Tensor,
        up_A: torch.Tensor | None,
        up_B: torch.Tensor | None,
        up_scale: float,
        down_weight: torch.Tensor,
        down_A: torch.Tensor | None,
        down_B: torch.Tensor | None,
        down_scale: float,
        activation_fn: Callable,
        activation_fn_backward: Callable,
        inplace: bool | None = True,
    ) -> torch.Tensor:
        """
        Forward pass for LoRA MLP.

        Args:
            ctx: Autograd context
            X: Input features
            gate_weight: Gate projection weight
            gate_A: Gate LoRA A matrix
            gate_B: Gate LoRA B matrix
            gate_scale: Gate LoRA scale
            up_weight: Up projection weight
            up_A: Up projection LoRA A matrix
            up_B: Up projection LoRA B matrix
            up_scale: Up projection LoRA scale
            down_weight: Down projection weight
            down_A: Down projection LoRA A matrix
            down_B: Down projection LoRA B matrix
            down_scale: Down projection LoRA scale
            activation_fn: Forward activation function
            activation_fn_backward: Backward activation function
            inplace: Whether to perform operations in-place

        Returns:
            Output transformed by multi-layer perceptron and activation function
        """
        # Compute projections
        gate = matmul_lora(
            X, gate_weight, gate_A, gate_B, gate_scale
        )
        up = matmul_lora(X, up_weight, up_A, up_B, up_scale)

        # Activation
        hidden = activation_fn(gate, up)

        # Down projection
        output = matmul_lora(
            hidden, down_weight, down_A, down_B, down_scale
        )

        # Save for backward
        ctx.save_for_backward(X, gate, up, gate_A, gate_B, up_A, up_B, down_A, down_B)
        ctx.scales = (gate_scale, up_scale, down_scale)
        ctx.weights = (gate_weight, up_weight, down_weight)
        ctx.activation_fn = activation_fn
        ctx.activation_fn_backward = activation_fn_backward
        ctx.inplace = inplace

        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output: torch.Tensor):
        (
            X,
            gate,
            up,
            gate_A,
            gate_B,
            up_A,
            up_B,
            down_A,
            down_B,
        ) = ctx.saved_tensors
        gate_scale, up_scale, down_scale = ctx.scales
        gate_weight, up_weight, down_weight = ctx.weights

        # Transpose all LoRA matrices
        gate_A, gate_B = (
            gate_A.t() if gate_A is not None else None,
            gate_B.t() if gate_B is not None else None,
        )
        up_A, up_B = (
            up_A.t() if up_A is not None else None,
            up_B.t() if up_B is not None else None,
        )
        down_A, down_B = (
            down_A.t() if down_A is not None else None,
            down_B.t() if down_B is not None else None,
        )

        # Flatten [B, S, *] -> [B*S, *]
        batch, seq_len, hd = X.shape
        grad_output = grad_output.view(-1, grad_output.shape[-1])
        X           = X.view(-1, X.shape[-1])
        gate        = gate.view(-1, gate.shape[-1])
        up          = up.view(-1, up.shape[-1])
        dtype = X.dtype

        # 1) Backprop through down-proj (LoRA-aware) to get grad wrt hidden
        grad_down = matmul_lora(
            grad_output,           # [N, out]
            down_weight.t(),       # note: matmul_lora expects [in, out]-shaped weight
            down_B,                # swap A/B for backward
            down_A,
            down_scale,
        )

        # 2) Activation backward -> hidden (a.k.a. h), and grads into gate/up paths
        h, grad_gate, grad_up = ctx.activation_fn_backward(grad_down, gate, up)
        # h has shape [N, hidden] and equals the forward "hidden" (post-activation) tensor

        # 3) LoRA grads (unchanged)
        d_down_A = d_down_B = d_up_A = d_up_B = d_gate_A = d_gate_B = None
        if down_A is not None and down_B is not None:
            d_down_A = h.t() @ (grad_output @ down_B.t())
            d_down_B = (down_A.t() @ h.t()) @ grad_output
            d_down_A *= down_scale
            d_down_B *= down_scale

        if up_A is not None and up_B is not None:
            d_up_A = X.t() @ (grad_up @ up_B.t())
            d_up_B = (up_A.t() @ X.t()) @ grad_up
            d_up_A *= up_scale
            d_up_B *= up_scale

        if gate_A is not None and gate_B is not None:
            d_gate_A = X.t() @ (grad_gate @ gate_B.t())
            d_gate_B = (gate_A.t() @ X.t()) @ grad_gate
            d_gate_A *= gate_scale
            d_gate_B *= gate_scale

        # 4) Input grad
        dX = torch.zeros_like(X) if ctx.needs_input_grad[0] else None
        if dX is not None:
            dX = torch.empty_like(X)                 # fresh buffer, correct dtype
            torch.matmul(grad_up, up_weight, out=dX)
            #dX = torch.matmul(grad_up, up_weight, out=X) if ctx.inplace else torch.matmul(grad_up, up_weight)
            del up_weight

            if up_A is not None and up_B is not None:
                dX += grad_up @ up_B.to(dtype).t() @ (up_scale * up_A.to(dtype).t())

            dX += grad_gate @ gate_weight
            del gate_weight

            if gate_A is not None and gate_B is not None:
                dX += grad_gate @ gate_B.to(dtype).t() @ (gate_scale * gate_A.to(dtype).t())

            dX = dX.view(batch, seq_len, hd)

        # 5) NEW: base weight/bias grads
        d_gate_W = d_up_W = d_down_W = None

        # forward used Y = X @ gate_weight.t(); so dW = grad_gate^T @ X
        d_gate_W = grad_gate.t() @ X  # [out, in]
        d_up_W   = grad_up.t()   @ X  # [out, in]
        d_down_W = grad_output.t() @ h  # [out, in], note: h is post-activation input to down


        # 6) Return in forward-arg order
        return (
            dX,            # X
            d_gate_W,      # gate_weight
            d_gate_A.t() if d_gate_A is not None else None,
            d_gate_B.t() if d_gate_B is not None else None,
            None,          # gate_scale
            d_up_W,        # up_weight
            d_up_A.t() if d_up_A is not None else None,
            d_up_B.t() if d_up_B is not None else None,
            None,          # up_scale
            d_down_W,      # down_weight
            d_down_A.t() if d_down_A is not None else None,
            d_down_B.t() if d_down_B is not None else None,
            None,          # down_scale
            None,          # activation_fn
            None,          # activation_fn_backward
            None,          # inplace
        )



def apply_lora_mlp_swiglu(self, X: torch.Tensor, inplace: bool = True) -> torch.Tensor:
    """
    Applies LoRA to MLP layer with SwiGLU activation.

    Args:
        X: Input tensor for the MLP layer
        inplace: Whether to perform operations in-place to save memory

    Returns:
        Output tensor after applying LoRA-adapted MLP with SwiGLU activation
    """
    gateW, gateb, gateW_quant, gateA, gateB, gateS = get_lora_parameters(self.gate_proj)
    upW, upb, upW_quant, upA, upB, upS = get_lora_parameters(self.up_proj)
    downW, downb, downW_quant, downA, downB, downS = get_lora_parameters(self.down_proj)

    out = LoRA_MLP.apply(
        X,
        gateW,
        gateb,
        gateW_quant,
        gateA,
        gateB,
        gateS,
        upW,
        upb,
        upW_quant,
        upA,
        upB,
        upS,
        downW,
        downb,
        downW_quant,
        downA,
        downB,
        downS,
        swiglu_forward,
        swiglu_backward,
        inplace,
    )

    return out
