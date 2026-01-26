# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import and_masks, BlockMask

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.models.attention import (
    create_attention_mask,
    create_varlen_metadata_for_document,
    FlexAttentionWrapper,
    get_causal_mask_mod,
    get_document_mask_mod,
    ScaledDotProductAttentionWrapper,
    VarlenAttentionWrapper,
    VarlenMetadata,
)
from torchtitan.models.moe import MoE
from torchtitan.protocols.model import AttentionMasksType
from torchtitan.protocols.train_spec import ModelProtocol

from .args import Qwen3ModelArgs
from .factorized_embedding import FactorizedEmbedding, FactorizedOutput, StandardOutput
from .weight_sharing import (
    AttentionSharingTransformerBlock,
    CombinedSharingTransformerBlock,
    SharedAttention,
    SharedTransformerBlock,
)


# Adapted from https://github.com/pytorch/torchtune/blob/main/torchtune/models/qwen2/_positional_embeddings.py
def precompute_rope_cache(
    dim: int, max_seq_len: int, base: float = 1_000_000.0
) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # Create position indexes `[0, 1, ..., max_seq_len - 1]`
    t = torch.arange(max_seq_len, dtype=freqs.dtype, device=freqs.device)

    # Outer product of theta and position index; output tensor has
    # a shape of [max_seq_len, dim // 2]
    idx_theta = torch.outer(t, freqs).float()

    # We cache the cos and sin embeddings instead of the IDs. This helps
    # ensure we have correct behavior when training with bf16
    # Size: [max_seq_len, (dim * 2)]
    freqs = torch.cat([idx_theta, idx_theta], dim=-1)
    rope_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
    return rope_cache


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def reshape_for_broadcast(
    rope_cache: torch.Tensor, x: torch.Tensor, positions: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Reshape frequency tensor (represented by cos, sin) for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    The input freqs_cis tensor is assumed to be of shape (max_seqlen, head_dim * 2),
    and the first seqlen elements will be sliced, but dim must match x.

    Args:
        rope_cache (torch.Tensor): RoPE tensor (cos and sin) to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.
        positions (torch.Tensor | None): Position indices used to access/shuffle RoPE cache.
            Shape is (1, seqlen) or (bz, seqlen). Defaults to None.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert ndim > 1
    bz, seqlen, _, head_dim = x.shape
    if positions is None:
        rope_cache = rope_cache[0:seqlen]
        # The shape of rope_cache is (seqlen, head_dim * 2) because we concate cos and sin
        assert rope_cache.shape == (seqlen, head_dim * 2)
        shape = [-1, seqlen, 1, head_dim * 2]
        return rope_cache.view(*shape)
    elif positions.size(0) == 1:
        assert positions.shape == (1, seqlen)
        rope_cache = rope_cache[positions.squeeze(0)]
        # The shape of rope_cache is (seqlen, head_dim * 2)
        assert rope_cache.shape == (seqlen, head_dim * 2)
        shape = [-1, seqlen, 1, head_dim * 2]
        return rope_cache.view(*shape)
    else:
        assert positions.shape == (bz, seqlen)
        rope_cache_expanded = rope_cache[None, :, None, :].expand(bz, -1, -1, -1)
        rope_cache = torch.gather(
            rope_cache_expanded,
            dim=1,
            index=positions.view(bz, seqlen, 1, 1).expand(bz, seqlen, 1, head_dim * 2),
        )
        # The shape of rope_cache is (bz, seqlen, 1, head_dim * 2)
        assert rope_cache.shape == (bz, seqlen, 1, head_dim * 2)
        return rope_cache


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # input tensor x has shape [bsz, seq_len, num_heads, head_dim]
    head_dim = xq.shape[-1]

    rope_cache = reshape_for_broadcast(rope_cache, xq, positions)

    # [bsz, seq_len, 1, head_dim]
    cos = rope_cache[..., :head_dim].to(dtype=xq.dtype, device=xq.device)
    sin = rope_cache[..., head_dim:].to(dtype=xq.dtype, device=xq.device)

    # xq:  [bsz, seq_len, num_heads, head_dim]
    # xk:  [bsz, seq_len, num_kv_heads, head_dim]
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
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

    q_norm: nn.RMSNorm | None
    k_norm: nn.RMSNorm | None

    def __init__(self, model_args: Qwen3ModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = (
            model_args.n_heads
            if model_args.n_kv_heads is None
            else model_args.n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.head_dim
        self.scaling = self.head_dim**-0.5
        self.attn_type = getattr(model_args, "attn_type", "sdpa")

        # RMSNorm added here to the here to include the q-k norm
        # This is one of the main differences between Llama3 and Qwen3
        if model_args.qk_norm:
            self.q_norm = nn.RMSNorm(
                self.head_dim, eps=model_args.norm_eps, elementwise_affine=True
            )
            self.k_norm = nn.RMSNorm(
                self.head_dim, eps=model_args.norm_eps, elementwise_affine=True
            )
        else:
            self.q_norm = None
            self.k_norm = None

        self.wq = nn.Linear(
            model_args.dim, model_args.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(
            model_args.n_heads * self.head_dim, model_args.dim, bias=False
        )

        match self.attn_type:
            case "flex":
                self.inner_attention = FlexAttentionWrapper()
            case "varlen":
                # pyrefly: ignore [bad-assignment]
                self.inner_attention = VarlenAttentionWrapper()
            case "sdpa":
                # pyrefly: ignore [bad-assignment]
                self.inner_attention = ScaledDotProductAttentionWrapper()
            case _:
                raise ValueError(f"Unknown attention type: {self.attn_type}")

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)
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
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            rope_cache (torch.Tensor): Precomputed cosine and sine frequencies.
            attention_masks (AttentionMasksType | None): Masks used when calculating attention scores.
            positions (torch.Tensor | None): Position indices used to access/shuffle RoPE cache. Defaults to None.

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

        # Adding the q_norm and k_norm here
        # Last layer of adding q-k norm
        if self.q_norm:  # pyrefly: ignore[invalid-argument]
            xq = self.q_norm(xq)
        if self.k_norm:  # pyrefly: ignore[invalid-argument]
            xk = self.k_norm(xk)

        # Apply rotary embedding
        xq, xk = apply_rotary_emb(xq, xk, rope_cache, positions)

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xv = values.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        match self.attn_type:
            case "flex":
                assert isinstance(attention_masks, BlockMask), attention_masks
                output = self.inner_attention(
                    xq, xk, xv, block_mask=attention_masks, scale=self.scaling
                )
                output = output.transpose(
                    1, 2
                ).contiguous()  # (bs, seqlen, n_local_heads, head_dim)
            case "varlen":
                # TODO: pass self.scaling into varlen attention
                assert isinstance(attention_masks, VarlenMetadata), attention_masks
                output = self.inner_attention(
                    xq,
                    xk,
                    xv,
                    self.head_dim,
                    attention_masks,
                    scale=self.scaling,
                )
            case "sdpa":
                assert attention_masks is None
                output = self.inner_attention(xq, xk, xv, scale=self.scaling)
                output = output.transpose(
                    1, 2
                ).contiguous()  # (bs, seqlen, n_local_heads, head_dim)
            case _:
                raise ValueError(f"Unknown attention type: {self.attn_type}")

        output = output.view(bs, seqlen, -1)
        return self.wo(output)


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
    ):
        super().__init__()

        # Hidden dimension is directly added from the model argsS
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


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

    def __init__(self, layer_id: int, model_args: Qwen3ModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim

        self.attention = Attention(model_args)

        self.moe_enabled = model_args.moe_enabled
        if self.moe_enabled:
            self.moe = MoE(
                model_args.moe_args,
                dim=model_args.dim,
                hidden_dim=model_args.moe_inter_dim,
            )
        else:
            self.feed_forward = FeedForward(
                dim=model_args.dim, hidden_dim=model_args.hidden_dim
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
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            rope_cache (torch.Tensor): Precomputed cosine and sine frequencies.
            attention_masks (AttentionMasksType | None): Masks used when calculating attention scores.
            positions (torch.Tensor | None): Position indices used to access/shuffle RoPE cache. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        x = x + self.attention(
            self.attention_norm(x), rope_cache, attention_masks, positions
        )

        if self.moe_enabled:
            x = x + self.moe(self.ffn_norm(x))
        else:
            x = x + self.feed_forward(self.ffn_norm(x))
        return x

    def init_weights(self, buffer_device: torch.device):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        if self.moe_enabled:
            self.moe.init_weights(self.weight_init_std, buffer_device)
        else:
            self.feed_forward.init_weights(self.weight_init_std)


class Qwen3Model(ModelProtocol):
    """
    Qwen3Model Module

    Args:
        model_args (TransformerModelArgs): Model configuration arguments.

    Attributes:
        model_args (TransformerModelArgs): Model configuration arguments.
        vocab_size (int): Vocabulary size.
        n_layers (int): Number of layers in the model.
        tok_embeddings (ParallelEmbedding): Token embeddings.
        layers (torch.nn.ModuleList): List of Transformer blocks.
        norm (RMSNorm): Layer normalization for the model output.
        output (ColumnParallelLinear): Linear layer for final output.
        freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

    """

    def __init__(self, model_args: Qwen3ModelArgs):
        super().__init__(model_args)
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.eos_id = model_args.eos_id
        self.head_dim = model_args.head_dim

        # Check for weight sharing configuration
        ws_config = model_args.weight_sharing
        use_factorized_emb = (
            ws_config is not None
            and hasattr(ws_config, "embedding")
            and ws_config.embedding.enabled
        )
        use_layer_sharing = (
            ws_config is not None
            and hasattr(ws_config, "layer")
            and ws_config.layer.enabled
        )
        use_attention_sharing = (
            ws_config is not None
            and hasattr(ws_config, "attention")
            and ws_config.attention.enabled
        )

        # Embeddings: standard or factorized
        if use_factorized_emb:
            d_emb = ws_config.embedding.d_emb
            self.tok_embeddings = FactorizedEmbedding(
                model_args.vocab_size, d_emb, model_args.dim
            )
            self._use_factorized_embedding = True
        else:
            self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
            self._use_factorized_embedding = False

        self.register_buffer(
            "rope_cache", self._precompute_rope_cache(), persistent=False
        )

        # Build layers based on sharing configuration
        # Four cases: no sharing, attention only, layer only, both
        self.layers = torch.nn.ModuleDict()
        if use_layer_sharing and use_attention_sharing:
            # Combined: layer sharing + attention sharing
            self._build_combined_sharing_layers(
                model_args, ws_config.layer, ws_config.attention
            )
        elif use_layer_sharing:
            # Layer sharing only
            self._build_shared_layers(model_args, ws_config.layer)
        elif use_attention_sharing:
            # Attention sharing only
            self._build_attention_sharing_layers(model_args, ws_config.attention)
        else:
            # Standard layers (no sharing)
            for layer_id in range(model_args.n_layers):
                self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)

        self.norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)

        # Output layer: standard, tied, or factorized tied
        if use_factorized_emb and ws_config.embedding.tie_output:
            self.output = FactorizedOutput(self.tok_embeddings)
            self._output_tied = True
        elif model_args.enable_weight_tying and not use_factorized_emb:
            # Use StandardOutput for weight tying (avoids meta device bug)
            self.output = StandardOutput(self.tok_embeddings)
            self._output_tied = True
        else:
            self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)
            self._output_tied = False

    def _build_shared_layers(self, model_args: Qwen3ModelArgs, layer_config) -> None:
        """Build transformer layers with weight sharing.

        Creates shared weight sets for each layer group, then instantiates
        SharedTransformerBlock for each layer using the appropriate shared weights.

        Args:
            model_args: Model configuration arguments.
            layer_config: LayerSharingConfig with layer_groups and lora_rank.
        """
        layer_groups = layer_config.layer_groups
        lora_rank = layer_config.lora_rank

        if layer_groups is None:
            raise ValueError(
                "layer_groups must be specified when layer sharing is enabled"
            )

        # Validate layer_groups covers all layers exactly once
        all_layers = set(range(model_args.n_layers))
        grouped_layers = set()
        for group in layer_groups:
            for layer in group:
                if layer in grouped_layers:
                    raise ValueError(f"Layer {layer} assigned to multiple groups")
                grouped_layers.add(layer)
        if grouped_layers != all_layers:
            raise ValueError(
                f"layer_groups must cover all layers exactly once. "
                f"Expected {all_layers}, got {grouped_layers}"
            )

        # Create shared weight sets for each group
        init_std = model_args.dim ** (-0.5)
        weight_sets = {}

        for group in layer_groups:
            group_key = tuple(group)
            attention_weights = {
                "wq": nn.Linear(
                    model_args.dim, model_args.n_heads * model_args.head_dim, bias=False
                ),
                "wk": nn.Linear(
                    model_args.dim,
                    (model_args.n_kv_heads or model_args.n_heads) * model_args.head_dim,
                    bias=False,
                ),
                "wv": nn.Linear(
                    model_args.dim,
                    (model_args.n_kv_heads or model_args.n_heads) * model_args.head_dim,
                    bias=False,
                ),
                "wo": nn.Linear(
                    model_args.n_heads * model_args.head_dim, model_args.dim, bias=False
                ),
            }
            ffn_weights = {
                "w1": nn.Linear(model_args.dim, model_args.hidden_dim, bias=False),
                "w2": nn.Linear(model_args.hidden_dim, model_args.dim, bias=False),
                "w3": nn.Linear(model_args.dim, model_args.hidden_dim, bias=False),
            }

            # Initialize weights
            for w in attention_weights.values():
                nn.init.trunc_normal_(
                    w.weight, mean=0.0, std=init_std, a=-3 * init_std, b=3 * init_std
                )
            for w in ffn_weights.values():
                nn.init.trunc_normal_(
                    w.weight, mean=0.0, std=init_std, a=-3 * init_std, b=3 * init_std
                )

            weight_sets[group_key] = {
                "attention": attention_weights,
                "ffn": ffn_weights,
            }

        # Store shared weights as a module for proper parameter tracking
        self._shared_weight_sets = nn.ModuleDict()
        for group_key, weight_set in weight_sets.items():
            group_name = "_".join(map(str, group_key))
            self._shared_weight_sets[f"attn_{group_name}"] = nn.ModuleDict(
                weight_set["attention"]
            )
            self._shared_weight_sets[f"ffn_{group_name}"] = nn.ModuleDict(
                weight_set["ffn"]
            )

        # Create layers with appropriate weight sets
        for layer_id in range(model_args.n_layers):
            for group in layer_groups:
                if layer_id in group:
                    group_key = tuple(group)
                    weight_set = weight_sets[group_key]
                    # Use LoRA only if layer shares with others
                    layer_lora_rank = 0 if len(group) == 1 else lora_rank
                    self.layers[str(layer_id)] = SharedTransformerBlock(
                        layer_id,
                        model_args,
                        weight_set["attention"],
                        weight_set["ffn"],
                        layer_lora_rank,
                    )
                    break

    def _build_attention_sharing_layers(
        self, model_args: Qwen3ModelArgs, attention_config
    ) -> None:
        """Build transformer layers with attention head sharing only.

        Each layer has its own weights (no layer sharing), but uses SharedAttention
        which shares base weights across attention heads with per-head LoRA offsets.

        Args:
            model_args: Model configuration arguments.
            attention_config: AttentionSharingConfig with head_sharing, grouping, rank, etc.
        """
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = AttentionSharingTransformerBlock(
                layer_id, model_args, attention_config
            )

    def _build_combined_sharing_layers(
        self, model_args: Qwen3ModelArgs, layer_config, attention_config
    ) -> None:
        """Build transformer layers with both layer sharing AND attention sharing.

        Combines:
        - Layer sharing: multiple layers share base weights with per-layer LoRA
        - Attention sharing: attention heads share bases with per-head LoRA offsets

        Args:
            model_args: Model configuration arguments.
            layer_config: LayerSharingConfig with layer_groups and lora_rank.
            attention_config: AttentionSharingConfig for head sharing settings.
        """
        layer_groups = layer_config.layer_groups
        lora_rank = layer_config.lora_rank

        if layer_groups is None:
            raise ValueError(
                "layer_groups must be specified when layer sharing is enabled"
            )

        # Validate layer_groups covers all layers exactly once
        all_layers = set(range(model_args.n_layers))
        grouped_layers = set()
        for group in layer_groups:
            for layer in group:
                if layer in grouped_layers:
                    raise ValueError(f"Layer {layer} assigned to multiple groups")
                grouped_layers.add(layer)
        if grouped_layers != all_layers:
            raise ValueError(
                f"layer_groups must cover all layers exactly once. "
                f"Expected {all_layers}, got {grouped_layers}"
            )

        # Create shared weight sets for each group
        init_std = model_args.dim ** (-0.5)
        weight_sets = {}

        for group in layer_groups:
            group_key = tuple(group)
            attention_weights = {
                "wq": nn.Linear(
                    model_args.dim, model_args.n_heads * model_args.head_dim, bias=False
                ),
                "wk": nn.Linear(
                    model_args.dim,
                    (model_args.n_kv_heads or model_args.n_heads) * model_args.head_dim,
                    bias=False,
                ),
                "wv": nn.Linear(
                    model_args.dim,
                    (model_args.n_kv_heads or model_args.n_heads) * model_args.head_dim,
                    bias=False,
                ),
                "wo": nn.Linear(
                    model_args.n_heads * model_args.head_dim, model_args.dim, bias=False
                ),
            }
            ffn_weights = {
                "w1": nn.Linear(model_args.dim, model_args.hidden_dim, bias=False),
                "w2": nn.Linear(model_args.hidden_dim, model_args.dim, bias=False),
                "w3": nn.Linear(model_args.dim, model_args.hidden_dim, bias=False),
            }

            # Initialize weights
            for w in attention_weights.values():
                nn.init.trunc_normal_(
                    w.weight, mean=0.0, std=init_std, a=-3 * init_std, b=3 * init_std
                )
            for w in ffn_weights.values():
                nn.init.trunc_normal_(
                    w.weight, mean=0.0, std=init_std, a=-3 * init_std, b=3 * init_std
                )

            weight_sets[group_key] = {
                "attention": attention_weights,
                "ffn": ffn_weights,
            }

        # Store shared weights as a module for proper parameter tracking
        self._shared_weight_sets = nn.ModuleDict()
        for group_key, weight_set in weight_sets.items():
            group_name = "_".join(map(str, group_key))
            self._shared_weight_sets[f"attn_{group_name}"] = nn.ModuleDict(
                weight_set["attention"]
            )
            self._shared_weight_sets[f"ffn_{group_name}"] = nn.ModuleDict(
                weight_set["ffn"]
            )

        # Create layers with combined sharing
        for layer_id in range(model_args.n_layers):
            for group in layer_groups:
                if layer_id in group:
                    group_key = tuple(group)
                    weight_set = weight_sets[group_key]
                    # Use LoRA only if layer shares with others
                    layer_lora_rank = 0 if len(group) == 1 else lora_rank
                    self.layers[str(layer_id)] = CombinedSharingTransformerBlock(
                        layer_id,
                        model_args,
                        weight_set["attention"],
                        weight_set["ffn"],
                        layer_lora_rank,
                        attention_config,
                    )
                    break

    def init_weights(
        self,
        buffer_device: torch.device | None = None,
    ):
        """
        [Note: On ``init_weights`` vs. ``reset_parameters``]
        Modules may define ``reset_parameters`` to initialize parameter values.
        ``reset_parameters`` is meant to only initialize directly owned
        parameters/buffers, not those of their child modules, and it can be
        used to give the initial values for these tensors.
        Separately, users may want custom initialization for their modules,
        different from that in ``reset_parameters``. For this, we define
        ``init_weights``. We only call it in the constructor of this
        ``Transformer`` root module to avoid reinitializing tensors.
        """
        buffer_device = buffer_device or self.rope_cache.device
        with torch.device(buffer_device):
            self.rope_cache = self._precompute_rope_cache()

        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3

        # Initialize embeddings
        if self.tok_embeddings is not None:
            if self._use_factorized_embedding:
                # FactorizedEmbedding has its own init_weights method
                self.tok_embeddings.init_weights(final_out_std)
            else:
                nn.init.normal_(self.tok_embeddings.weight)

        # Initialize layers
        for layer in self.layers.values():
            if layer is not None:
                # pyrefly: ignore [not-callable]
                layer.init_weights(buffer_device)

        if self.norm is not None:
            self.norm.reset_parameters()

        # Initialize output layer (skip if tied to embeddings)
        if self.output is not None and not self._output_tied:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

    def _precompute_rope_cache(self) -> torch.Tensor:
        return precompute_rope_cache(
            self.model_args.head_dim,
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )

    def _get_flex_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        mask_mods = [get_causal_mask_mod()]
        match self.model_args.attn_mask_type:
            case "causal":
                B = 1
            case "block_causal":
                B = input_batch.shape[0]
                mask_mods.append(get_document_mask_mod(input_batch, tokenizer.eos_id))
            case _:
                raise ValueError(
                    f"Unknown attention mask type: {self.model_args.attn_mask_type}"
                )
        return create_attention_mask(
            and_masks(*mask_mods), B, None, input_batch.shape[1], input_batch.shape[1]
        )

    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        match self.model_args.attn_type:
            case "flex":
                return self._get_flex_attention_masks(
                    input_batch, tokenizer, extra_inputs
                )
            case "varlen":
                if self.model_args.attn_mask_type != "block_causal":
                    raise ValueError(
                        f"varlen attention is only supported with block_causal \
                        attention mask type, got {self.model_args.attn_mask_type}"
                    )
                return create_varlen_metadata_for_document(
                    input_batch, tokenizer.eos_id
                )
            case _:
                raise TypeError("Only varlen and flex attn masks are supported")

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices if pipeline parallelism is not enabled.
                If pipeline parallelism is enabled, this will be the input token indices
                for the ranks on the first pipeline stage. This will be the activation of the
                previous pipeline stage if the current rank is not on the first stage.
            attention_masks (AttentionMasksType | None): Masks used when calculating attention scores.
            positions (torch.Tensor | None): Position indices used to access/shuffle RoPE cache. Defaults to None.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        # pyrefly: ignore[not-callable, invalid-argument]
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        for layer in self.layers.values():
            # Check if this is a weight-sharing block that needs extra args
            # Use attribute check instead of isinstance for torch.compile compatibility
            if getattr(layer, "_is_weight_sharing_block", False):
                h = layer(
                    h,
                    self.rope_cache,
                    attention_masks,
                    positions,
                    apply_rotary_emb_fn=apply_rotary_emb,
                    repeat_kv_fn=repeat_kv,
                )
            else:
                h = layer(h, self.rope_cache, attention_masks, positions)

        # pyrefly: ignore[not-callable, invalid-argument]
        h = self.norm(h) if self.norm else h
        # pyrefly: ignore[not-callable, invalid-argument]
        output = self.output(h) if self.output else h
        return output
