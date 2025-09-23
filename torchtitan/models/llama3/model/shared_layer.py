import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.attention import build_attention, init_attention_mask
from torchtitan.protocols.train_spec import ModelProtocol

from .args import TransformerModelArgs
from torchtitan.tools.logging import logger
import math
from .kernels.lora import LoRA_MLP
from .kernels.swiglu import (
    swiglu_act_backward as swiglu_backward,
    swiglu_act as swiglu_forward
)
def broadcast_add(base: torch.Tensor, delta: torch.Tensor,
                  *, dim: int = -1, g: int = 1) -> torch.Tensor:
    """
    base  : (..., g*D, ...)          – shared projection (smaller)
    delta : (..., H*D, ...)          – per-head LoRA delta (bigger)
    dim   : dimension that holds g*D or H*D
    g     : #templates shared inside each group  (g == 1 ⇒ full sharing)
    returns a tensor shaped like `delta`, equal to broadcast(base) + delta
    """
    if dim < 0:
        dim += base.dim()            # canonicalise

    # ----- shapes & sanity -------------------------------------------------
    big  = delta.shape[dim]          # H*D
    small = base.shape[dim]          # g*D
    assert big % small == 0,  "delta and base feature dims incompatible"
    heads_per_group = big // small   # H / g
    D = small // g                   # single-head feature size

    # ----- reshape base to (..., g, D, ...) -------------------------------
    new_shape = list(base.shape)
    new_shape[dim:dim+1] = [g, D]    # split the feature dim
    base = base.reshape(*new_shape)

    # ----- expand over heads_per_group ------------------------------------
    # insert a broadcast axis right after 'g'
    base = base.unsqueeze(dim + 1)                              # (..., g, 1, D, ...)
    base = base.expand(*base.shape[:dim+1], heads_per_group, D,
                       *base.shape[dim+3:])                     # (..., g, H/g, D, ...)

    # ----- flatten back to (..., H*D, ...) -------------------------------
    flat_shape = list(delta.shape)
    base = base.reshape(*flat_shape)   # now exactly the same shape as delta

    return base + delta


class ScaledLoRAModule(nn.Module):
    def __init__(self,
        in_dim,
        out_dim,
        rank=8,
        bias=False,
        w_a_override=None,
        w_b_override=None):

        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.alpha = rank*2
        self.bias = bias

        # LoRA params
        if w_a_override is not None:
            self.w_a = w_a_override
        else:
            self.w_a = nn.Linear(
                in_dim,
            rank,
            bias=False,
        )  # A matrix

        if w_b_override is not None:
            self.w_b = w_b_override
        else:
            self.w_b = nn.Linear(
                rank,
                out_dim,
                bias=False,
        )  # B matrix

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.w_b.weight)


    def forward(self, x):
        #  low-rank update
        lora = self.w_b(self.w_a(x)) * (self.alpha / self.rank)
        return lora

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float | None): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    The input freqs_cis tensor is assumed to be of shape (max_seqlen, dim),
    and the first seqlen elements will be sliced, but dim must match x.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert ndim > 1
    seqlen = x.shape[1]
    freqs_cis = freqs_cis[0:seqlen]
    assert freqs_cis.shape == (seqlen, x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )




class LoRAModule(nn.Module):
    def __init__(self,
        in_dim,
        out_dim,
        rank=8,
        bias=False,
        w_a_override=None,
        w_b_override=None,
        dtype=None):

        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.bias = bias

        # LoRA params
        if w_a_override is not None:
            self.w_a = w_a_override
        else:
            self.w_a = nn.Linear(
                in_dim,
                rank,
                bias=False,
                dtype=dtype
        )  # A matrix

        if w_b_override is not None:
            self.w_b = w_b_override
        else:
            self.w_b = nn.Linear(
                rank,
                out_dim,
                bias=False,
                dtype=dtype
        )  # B matrix

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w_a.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w_b.weight, a=math.sqrt(5))
    def forward(self, x):
        #  low-rank update
        lora = self.w_b(self.w_a(x))
        return lora

def n_loras(
    n,
    in_dim,
    out_dim,
    rank=8,
    bias=False,
    w_a_override=None,
    w_b_override=None,
    dtype=None):
        return nn.ModuleList([
            LoRAModule(
                in_dim=in_dim,
                out_dim=out_dim,
                rank=rank,
                bias=bias,
                w_a_override=w_a_override,
                w_b_override=w_b_override,
                dtype=dtype
            )
            for _ in range(n)
        ])

def reset_n_loras(loras: nn.ModuleList):
    for lora in loras:
        lora.reset_parameters()
def str_to_attr(s: str):
    if s == 'q':
        return 'wq_base'
    if s == 'k':
        return 'wk_base'
    if s == 'v':
        return 'wv_base'

class Attention(nn.Module):
    """
    Multi-head attention module.

    Args:
        model_args (TransformerModelArgs): Model configuration arguments.

    Attributes:
        n_kv_heads (int): Number of key and value heads.
        n_heads (int): Number of query heads.
        n_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension size of each attention head.
        wq (Linear): Linear transformation for queries.
        wk (Linear): Linear transformation for keys.
        wv (Linear): Linear transformation for values.
        wo (Linear): Linear transformation for output.

    """

    def __init__(self, model_args: TransformerModelArgs, n: int):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = (
            model_args.n_heads
            if model_args.n_kv_heads is None
            else model_args.n_kv_heads
        )
        self.heads_per_group = self.n_heads // self.n_kv_heads
        self.qkv_sharing = model_args.shared_attn.qkv_sharing
        self.head_sharing = model_args.shared_attn.head_sharing
        self.grouping = model_args.shared_attn.grouping if model_args.shared_attn.grouping else 1
        self.two_step = model_args.shared_attn.two_step
        self.rank = model_args.shared_attn.rank
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.dim // model_args.n_heads
        rank = model_args.layer_sharing.rank

        if model_args.shared_attn.qkv_sharing:
            # disable GQA
            self.n_kv_heads = model_args.n_heads
            self.n_rep = 1
            base_dim = self.head_dim*model_args.shared_attn.grouping \
                if model_args.shared_attn.head_sharing else \
                self.head_dim * model_args.n_heads
            for weight_group in model_args.shared_attn.qkv_sharing:
                w_base = nn.Linear(
                    model_args.dim,
                    base_dim,
                    bias=False
                )

                w_base_offsets = n_loras(n, in_dim=model_args.dim, out_dim=base_dim, bias=False, rank=rank)
                name = str(weight_group)
                setattr(self, name, w_base)
                setattr(self, name+"_offsets", w_base)
        else:
            dim_mult = self.grouping if model_args.shared_attn.head_sharing else model_args.n_heads
            kv_dim_mult = max(1, math.ceil(dim_mult / self.n_rep))
            self.wq_base = nn.Linear(
                model_args.dim,
                self.head_dim * dim_mult,
                bias=False,
            )
            self.wq_base_offsets = n_loras(n, in_dim=model_args.dim, out_dim=int(self.head_dim * dim_mult), bias=False, rank=rank)

            self.wk_base = nn.Linear(
                model_args.dim,
                int(self.head_dim *  kv_dim_mult),
                bias=False,
            )
            self.wk_base_offsets = n_loras(n, in_dim=model_args.dim, out_dim=int(self.head_dim*kv_dim_mult), bias=False, rank=rank)

            self.wv_base = nn.Linear(
                model_args.dim,
                int(self.head_dim * kv_dim_mult),
                bias=False,
            )
            self.wv_base_offsets = n_loras(n, in_dim=model_args.dim, out_dim=int(self.head_dim * kv_dim_mult), bias=False, rank=rank)

        if model_args.shared_attn.two_step:
            self.head_offset = ScaledLoRAModule(
                model_args.dim,
                model_args.n_heads * self.head_dim,
                rank=model_args.shared_attn.rank,
                bias=False
            )
            self.head_offset_offsets = n_loras(n, in_dim=model_args.dim, out_dim=model_args.n_heads * self.head_dim, bias=False, rank=rank)

            self.wq_only_offset = ScaledLoRAModule(
                model_args.dim,
                self.head_dim,
                rank=model_args.shared_attn.rank,
                bias=False
            )
            self.wq_only_offset_offsets = n_loras(n, in_dim=model_args.dim, out_dim=self.head_dim, bias=False, rank=rank)

            self.wk_only_offset = ScaledLoRAModule(
                model_args.dim,
                self.head_dim,
                rank=model_args.shared_attn.rank,
                bias=False
            )
            self.wk_only_offset_offsets = n_loras(n, in_dim=model_args.dim, out_dim=self.head_dim, bias=False, rank=rank)

            self.wv_only_offset = ScaledLoRAModule(
                model_args.dim,
                self.head_dim,
                rank=model_args.shared_attn.rank,
            )
            self.wv_only_offset_offsets = n_loras(n, in_dim=model_args.dim, out_dim=self.head_dim, bias=False, rank=rank)

        else:
            self.wq_offset = ScaledLoRAModule(
                model_args.dim,
                model_args.n_heads * self.head_dim,
                rank=model_args.shared_attn.rank,
                bias=False
            )
            self.wq_offset_offsets = n_loras(n, in_dim=model_args.dim, out_dim=model_args.n_heads * self.head_dim, bias=False, rank=rank)


            self.wk_offset = ScaledLoRAModule(
                model_args.dim,
                self.n_kv_heads * self.head_dim,
                rank=model_args.shared_attn.rank,
                bias=False,
            )
            self.wk_offset_offsets = n_loras(n, in_dim=model_args.dim, out_dim=self.n_kv_heads*self.head_dim,bias=False, rank=rank)


            self.wv_offset = ScaledLoRAModule(
                model_args.dim,
                self.n_kv_heads * self.head_dim,
                rank=model_args.shared_attn.rank,
                bias=False,
            )
            self.wv_offset_offsets = n_loras(n, in_dim=model_args.dim, out_dim=self.n_kv_heads * self.head_dim, bias=False, rank=rank)

        self.wo = nn.Linear(
            model_args.n_heads * self.head_dim, model_args.dim, bias=False
        )
        self.wo_offsets = n_loras(n, in_dim=model_args.n_heads * self.head_dim, out_dim=model_args.dim, bias=False, rank=rank)
        self.sdpa = build_attention(model_args.use_flex_attn, model_args.attn_mask_type)

    def init_weights(self, init_std: float):
        if self.qkv_sharing:
            for weight_group in self.qkv_sharing:
                name = str(weight_group)
                nn.init.trunc_normal_(getattr(self, name).weight, mean=0.0, std=0.02)
                reset_n_loras(getattr(self, name+"_offsets"))

        else:
            for linear in (self.wq_base, self.wk_base, self.wv_base):
                nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
            reset_n_loras(self.wq_base_offsets)
            reset_n_loras(self.wk_base_offsets)
            reset_n_loras(self.wv_base_offsets)
        if self.two_step:
            for lora in (self.head_offset, self.wq_only_offset, self.wk_only_offset, self.wv_only_offset):
                lora.reset_parameters()
            reset_n_loras(self.head_offset_offsets)
            reset_n_loras(self.wq_only_offset_offsets)
            reset_n_loras(self.wk_only_offset_offsets)
            reset_n_loras(self.wv_only_offset_offsets)
        else:
            for lora in (self.wq_offset, self.wk_offset, self.wv_offset):
                lora.reset_parameters()
            reset_n_loras(self.wq_offset_offsets)
            reset_n_loras(self.wk_offset_offsets)
            reset_n_loras(self.wv_offset_offsets)

        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)
        reset_n_loras(self.wo_offsets)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        n: int
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        return self.offset_forward(x, freqs_cis, n)

    def std_forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bs, seqlen, _ = x.shape
        if self.two_step:
            head_offset = self.head_offset(x)
            wq_only_offset = self.wq_only_offset(x)
            wk_only_offset = self.wk_only_offset(x)
            wv_only_offset = self.wv_only_offset(x)

            wq_offset = broadcast_add(wq_only_offset, head_offset).contiguous()
            wk_offset = broadcast_add(wk_only_offset, head_offset).contiguous()
            wv_offset = broadcast_add(wv_only_offset, head_offset).contiguous()
        else:
            wq_offset = self.wq_offset(x)
            wk_offset = self.wk_offset(x)
            wv_offset = self.wv_offset(x)

        if self.qkv_sharing:
            for weight_group in self.qkv_sharing:
                val = getattr(self, str(weight_group))(x)
                for weight in weight_group:
                    if weight == 'q':
                        q_base = val
                    elif weight == 'k':
                        k_base = val
                    elif weight == 'v':
                        v_base = val
        else:
            q_base = self.wq_base(x)
            k_base = self.wk_base(x)
            v_base = self.wv_base(x)

        xq = broadcast_add(q_base, wq_offset, g=self.grouping)
        xk = broadcast_add(k_base, wk_offset, g=self.grouping)
        xv = broadcast_add(v_base, wv_offset, g=self.grouping)

        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xv = values.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        output = self.sdpa(xq, xk, xv)

        output = output.transpose(
            1, 2
        ).contiguous()  # (bs, seqlen, n_local_heads, head_dim)
        output = output.view(bs, seqlen, -1)
        return self.wo(output)

    def offset_forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        n: int
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bs, seqlen, _ = x.shape
        if self.two_step:
            head_offset = self.head_offset(x) + self.head_offset_offsets[n](x)
            wq_only_offset = self.wq_only_offset(x) + self.wq_only_offset_offsets[n](x)
            wk_only_offset = self.wk_only_offset(x) + self.wk_only_offset_offsets[n](x)
            wv_only_offset = self.wv_only_offset(x) + self.wv_only_offset_offsets[n](x)

            wq_offset = broadcast_add(wq_only_offset, head_offset).contiguous()
            wk_offset = broadcast_add(wk_only_offset, head_offset).contiguous()
            wv_offset = broadcast_add(wv_only_offset, head_offset).contiguous()
        else:
            wq_offset = self.wq_offset(x) + self.wq_offset_offsets[n](x)
            wk_offset = self.wk_offset(x) + self.wk_offset_offsets[n](x)
            wv_offset = self.wv_offset(x) + self.wv_offset_offsets[n](x)

        if self.qkv_sharing:
            for weight_group in self.qkv_sharing:
                val = getattr(self, str(weight_group))(x)
                for weight in weight_group:
                    if weight == 'q':
                        q_base = val
                    elif weight == 'k':
                        k_base = val
                    elif weight == 'v':
                        v_base = val
        else:
            q_base = self.wq_base(x) + self.wq_base_offsets[n](x)
            k_base = self.wk_base(x) + self.wk_base_offsets[n](x)
            v_base = self.wv_base(x) + self.wv_base_offsets[n](x)

        xq = broadcast_add(q_base, wq_offset, g=self.grouping)
        xk = broadcast_add(k_base, wk_offset, g=self.grouping)
        xv = broadcast_add(v_base, wv_offset, g=self.grouping)

        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xv = values.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        output = self.sdpa(xq, xk, xv)

        output = output.transpose(
            1, 2
        ).contiguous()  # (bs, seqlen, n_local_heads, head_dim)
        output = output.view(bs, seqlen, -1)
        return self.wo(output) + self.wo_offsets[n](output)

class OldAttention(nn.Module):
    """
    Multi-head attention module.

    Args:
        model_args (TransformerModelArgs): Model configuration arguments.

    Attributes:
        n_kv_heads (int): Number of key and value heads.
        n_heads (int): Number of query heads.
        n_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension size of each attention head.
        wq (Linear): Linear transformation for queries.
        wk (Linear): Linear transformation for keys.
        wv (Linear): Linear transformation for values.
        wo (Linear): Linear transformation for output.

    """

    def __init__(self, model_args: TransformerModelArgs, n: int):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = (
            model_args.n_heads
            if model_args.n_kv_heads is None
            else model_args.n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.dim // model_args.n_heads
        rank = model_args.layer_sharing.rank

        self.wq = nn.Linear(
            model_args.dim, model_args.n_heads * self.head_dim, bias=False
        )
        self.wq_offsets = n_loras(n, in_dim=model_args.dim, out_dim=model_args.n_heads * self.head_dim, bias=False, rank=rank)

        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wk_offsets = n_loras(n, in_dim=model_args.dim, out_dim=self.n_kv_heads * self.head_dim, bias=False, rank=rank)

        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv_offsets = n_loras(n, in_dim=model_args.dim, out_dim=self.n_kv_heads * self.head_dim, bias=False, rank=rank)

        self.wo = nn.Linear(
            model_args.n_heads * self.head_dim, model_args.dim, bias=False
        )
        self.wo_offsets = n_loras(n, in_dim=model_args.n_heads * self.head_dim, out_dim=model_args.dim, bias=False, rank=rank)

        self.sdpa = build_attention(model_args.use_flex_attn, model_args.attn_mask_type)

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            linear.reset_parameters()
        reset_n_loras(self.wq_offsets)
        reset_n_loras(self.wk_offsets)
        reset_n_loras(self.wv_offsets)

        self.wo.reset_parameters()
        reset_n_loras(self.wo_offsets)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        n: int
    ):
        return self.offset_forward(x, freqs_cis, n)

    def std_forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """

        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xv = values.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        output = self.sdpa(xq, xk, xv)

        output = output.transpose(
            1, 2
        ).contiguous()  # (bs, seqlen, n_local_heads, head_dim)
        output = output.view(bs, seqlen, -1)
        return self.wo(output)

    def offset_forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        n: int
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """

        bs, seqlen, _ = x.shape
        xq = self.wq(x) + self.wq_offsets[n](x)
        xk = self.wk(x) + self.wk_offsets[n](x)
        xv = self.wv(x) + self.wv_offsets[n](x)

        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xv = values.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        output = self.sdpa(xq, xk, xv)

        output = output.transpose(
            1, 2
        ).contiguous()  # (bs, seqlen, n_local_heads, head_dim)
        output = output.view(bs, seqlen, -1)
        return self.wo(output) + self.wo_offsets[n](x)

class FeedForward(nn.Module):
    """
    FeedForward module

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.
        multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        ffn_dim_multiplier (float | None): Custom multiplier for hidden dimension. Defaults to None.

    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.

    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
        model_args: TransformerModelArgs,
        n: int
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        rank = model_args.layer_sharing.rank

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w1_offsets = n_loras(n, in_dim=dim, out_dim=hidden_dim, rank=rank,
            )

        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w2_offsets = n_loras(n, in_dim=hidden_dim, out_dim=dim, rank=rank,
            )

        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3_offsets = n_loras(n, in_dim=dim, out_dim=hidden_dim, rank=rank,
            )

    def forward(self, x, n):
        return self.offset_fwd(x, n)

    def std_fwd(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def offset_fwd(self, x, n):
        """
        X = x.to(torch.bfloat16)
        upW = self.w1.weight
        upA = self.w1_offsets[n].w_a.weight
        upB = self.w1_offsets[n].w_b.weight

        downW = self.w2.weight
        downA = self.w2_offsets[n].w_a.weight
        downB = self.w2_offsets[n].w_b.weight

        gateW = self.w3.weight
        gateA = self.w3_offsets[n].w_a.weight
        gateB = self.w3_offsets[n].w_b.weight
        out = LoRA_MLP.apply(
            X,
            gateW,
            gateA,
            gateB,
            1.0,
            upW,
            upA,
            upB,
            1.0,
            downW,
            downA,
            downB,
            1.0,
            swiglu_forward,
            swiglu_backward,
            True,
        )
        return out
        """
        w1 = self.w1(x) + self.w1_offsets[n](x)
        w3 = self.w3(x) + self.w3_offsets[n](x)
        w_out = F.silu(w1) * w3
        return self.w2(w_out) + self.w2_offsets[n](w_out)

    def init_weights(self, init_std: float):
        self.w1.reset_parameters()
        reset_n_loras(self.w1_offsets)

        for linear in (self.w2, self.w3):
            linear.reset_parameters()

        reset_n_loras(self.w2_offsets)
        reset_n_loras(self.w3_offsets)


class TransformerBlock(nn.Module):
    """
    TransformerBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        model_args (TransformerModelArgs): Model configuration arguments.

    Attributes:
        n_heads (int): Number of attention heads.
        dim (int): Dimension size of the model.
        head_dim (int): Dimension size of each attention head.
        attention (Attention): Attention module.
        feed_forward (FeedForward): FeedForward module.
        layer_id (int): Identifier for the layer.
        attention_norm (RMSNorm): Layer normalization for attention output.
        ffn_norm (RMSNorm): Layer normalization for feedforward output.

    """

    def __init__(self, layer_id: int, model_args: TransformerModelArgs, n: int):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim
        self.n = n
        self.attention = Attention(model_args, n=n) if model_args.shared_attn.enabled else OldAttention(model_args, n=n)
        self.feed_forward = FeedForward(
            dim=model_args.dim,
            hidden_dim=4 * model_args.dim,
            multiple_of=model_args.multiple_of,
            ffn_dim_multiplier=model_args.ffn_dim_multiplier,
            model_args=model_args,
            n=n
        )
        self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)

        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * model_args.n_layers) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        for i in range(self.n):
            h = x + self.attention(self.attention_norm(x), freqs_cis, n=i)
            out = h + self.feed_forward(self.ffn_norm(h), n=i)
            x = out

        return out

    def init_weights(self):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)
