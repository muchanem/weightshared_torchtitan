# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from dataclasses import dataclass, field

import torch
from torch import nn

from torchtitan.models.common import Linear, RMSNorm
from torchtitan.models.common.attention import AttentionMasksType, GQAttention
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.protocols.module import Module, ModuleDict
from torchtitan.tools.logging import logger

from .model import (
    AttentionSharingTransformerBlock,
    CombinedSharingTransformerBlock,
    FactorizedEmbedding,
    FactorizedOutput,
    SharedTransformerBlock,
)


@dataclass(kw_only=True, slots=True)
class AttentionSharingConfig:
    """Config for attention head sharing with LoRA offsets."""
    enabled: bool = False
    qkv_sharing: list[list[str]] | None = None
    head_sharing: bool = False
    grouping: int = 1
    rank: int = 8
    two_step: bool = False


@dataclass(kw_only=True, slots=True)
class LayerSharingConfig:
    """Config for layer-level weight sharing with LoRA adapters."""
    enabled: bool = False
    layer_groups: list[list[int]] | None = None
    lora_rank: int = 8


@dataclass(kw_only=True, slots=True)
class FactorizedEmbeddingConfig:
    """Config for factorized embedding decomposition."""
    enabled: bool = False
    d_emb: int = 256
    tie_output: bool = True


class Qwen3TransformerBlock(TransformerBlock):
    """
    Qwen3 TransformerBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        dim (int): Model dimension.
        n_layers (int): Total number of layers.
        config (Qwen3TransformerBlock.Config): Block configuration.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        depth_init: bool = True
        moe_enabled: bool = False

    def __init__(self, config: Config, *, layer_id: int, dim: int, n_layers: int):
        super().__init__()

        self.attention = config.attention.build(dim=dim)

        self.moe_enabled = config.moe_enabled
        if self.moe_enabled:
            assert config.moe is not None
            self.moe = config.moe.build(dim=dim)
        else:
            assert config.feed_forward is not None
            self.feed_forward = config.feed_forward.build(dim=dim)

        self.attention_norm = config.attention_norm.build(normalized_shape=dim)
        self.ffn_norm = config.ffn_norm.build(normalized_shape=dim)

        if config.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * n_layers) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        x = x + self.attention(
            self.attention_norm(x), freqs_cis, attention_masks, positions
        )

        if self.moe_enabled:
            x = x + self.moe(self.ffn_norm(x))
        else:
            x = x + self.feed_forward(self.ffn_norm(x))
        return x

    def init_weights(self, **kwargs):
        buffer_device: torch.device | None = kwargs.get("buffer_device")
        for norm in (self.attention_norm, self.ffn_norm):
            norm.init_weights()
        self.attention.init_weights(self.weight_init_std)
        if self.moe_enabled:
            self.moe.init_weights(
                init_std=self.weight_init_std, buffer_device=buffer_device
            )
        else:
            self.feed_forward.init_weights(self.weight_init_std)


class Qwen3Model(Decoder):
    """
    Qwen3Model Module

    Args:
        config (Qwen3Model.Config): Model configuration.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 1024
        n_layers: int = 28
        vocab_size: int = 151936
        enable_weight_tying: bool = False
        layer: TransformerBlock.Config
        attention_sharing: AttentionSharingConfig = field(
            default_factory=AttentionSharingConfig
        )
        layer_sharing: LayerSharingConfig = field(
            default_factory=LayerSharingConfig
        )
        factorized_embedding: FactorizedEmbeddingConfig = field(
            default_factory=FactorizedEmbeddingConfig
        )

        def update_from_config(
            self,
            *,
            trainer_config,
            **kwargs,
        ) -> None:
            training = trainer_config.training
            parallelism = trainer_config.parallelism
            debug = trainer_config.debug
            seq_len = training.seq_len
            if seq_len > self.rope.max_seq_len:
                logger.warning(
                    f"Sequence length {seq_len} exceeds original maximum {self.rope.max_seq_len}."
                )
            # Sync rope max_seq_len
            import dataclasses as _dc

            self.rope = _dc.replace(self.rope, max_seq_len=seq_len)

            if self.layer.moe is not None:
                self.layer.moe.router._debug_force_load_balance = (
                    debug.moe_force_load_balance
                )

            if (
                parallelism.context_parallel_degree > 1
                and self.layer.attention.attn_backend == "varlen"
            ):
                raise NotImplementedError(
                    f"Context Parallel only supports SDPA and FlexAttention."
                    f"Got attn_backend='{self.layer.attention.attn_backend}'. "
                    f"Varlen attention is not supported with CP."
                )

            if self.enable_weight_tying and parallelism.pipeline_parallel_degree > 1:
                raise NotImplementedError(
                    "Weight tying is not supported with Pipeline Parallel."
                )

            # Weight sharing is also incompatible with PP
            if self.layer_sharing.enabled or self.attention_sharing.enabled:
                if parallelism.pipeline_parallel_degree > 1:
                    raise ValueError(
                        "Weight sharing is not supported with Pipeline Parallel."
                    )

            tp = parallelism.tensor_parallel_degree
            if tp > 1:
                n_heads = self.layer.attention.n_heads
                # pyrefly: ignore [missing-attribute]
                n_kv_heads = self.layer.attention.n_kv_heads or n_heads
                if n_heads % tp != 0:
                    raise ValueError(
                        f"tensor_parallel_degree ({tp}) must divide n_heads ({n_heads})."
                    )
                if n_kv_heads % tp != 0:
                    raise ValueError(
                        f"tensor_parallel_degree ({tp}) must divide n_kv_heads ({n_kv_heads})."
                    )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            assert isinstance(self.layer.attention, GQAttention.Config)
            assert self.layer.attention.head_dim is not None
            return get_moe_model_nparams_and_flops(
                self,
                model,
                self.layer.attention.n_heads,
                2 * self.layer.attention.head_dim,
                seq_len,
            )

    def __init__(self, config: Config):
        # Don't call Decoder.__init__ — we need custom layer creation
        Module.__init__(self)
        self.config = config
        self.enable_weight_tying = config.enable_weight_tying

        use_factorized = config.factorized_embedding.enabled
        use_layer = config.layer_sharing.enabled
        use_attn = config.attention_sharing.enabled

        # Embeddings
        if use_factorized:
            self.tok_embeddings = FactorizedEmbedding(
                config.vocab_size, config.factorized_embedding.d_emb, config.dim
            )
        else:
            self.tok_embeddings = config.tok_embeddings.build(
                num_embeddings=config.vocab_size, embedding_dim=config.dim
            )

        # RoPE
        self.rope = config.rope.build()
        self.register_buffer("freqs_cis", self.rope.cache, persistent=False)

        # Layers — dispatch based on sharing config
        self.layers = ModuleDict()
        if use_layer and use_attn:
            self._build_combined_sharing_layers(config)
        elif use_layer:
            self._build_shared_layers(config)
        elif use_attn:
            self._build_attention_sharing_layers(config)
        else:
            for layer_id in range(config.n_layers):
                self.layers[str(layer_id)] = config.layer.build(
                    layer_id=layer_id, dim=config.dim, n_layers=config.n_layers
                )

        self.norm = config.norm.build(normalized_shape=config.dim)

        # Output
        if use_factorized and config.factorized_embedding.tie_output:
            self.output = FactorizedOutput(self.tok_embeddings)
        else:
            self.output = config.output.build(
                in_features=config.dim, out_features=config.vocab_size
            )
            # Weight tying using upstream's debugged pattern
            if self.enable_weight_tying and not use_factorized:
                self.tok_embeddings.weight = self.output.weight

    def _build_shared_layers(self, config: Config) -> None:
        layer_groups = config.layer_sharing.layer_groups
        lora_rank = config.layer_sharing.lora_rank

        if layer_groups is None:
            raise ValueError(
                "layer_groups must be specified when layer sharing is enabled"
            )

        # Validate
        all_layers = set(range(config.n_layers))
        grouped_layers = set()
        for group in layer_groups:
            for layer in group:
                if layer in grouped_layers:
                    raise ValueError(f"Layer {layer} assigned to multiple groups")
                grouped_layers.add(layer)
        if grouped_layers != all_layers:
            raise ValueError(
                f"layer_groups must cover all layers. "
                f"Expected {all_layers}, got {grouped_layers}"
            )

        n_heads = config.layer.attention.n_heads
        n_kv_heads = config.layer.attention.n_kv_heads or n_heads
        head_dim = config.layer.attention.head_dim
        hidden_dim = config.layer.feed_forward.hidden_dim
        qk_norm = config.layer.attention.q_norm is not None

        weight_sets = {}
        for group in layer_groups:
            group_key = tuple(group)
            attention_weights = {
                "wq": Linear.Config().build(
                    in_features=config.dim, out_features=n_heads * head_dim
                ),
                "wk": Linear.Config().build(
                    in_features=config.dim, out_features=n_kv_heads * head_dim
                ),
                "wv": Linear.Config().build(
                    in_features=config.dim, out_features=n_kv_heads * head_dim
                ),
                "wo": Linear.Config().build(
                    in_features=n_heads * head_dim, out_features=config.dim
                ),
            }
            ffn_weights = {
                "w1": Linear.Config().build(
                    in_features=config.dim, out_features=hidden_dim
                ),
                "w2": Linear.Config().build(
                    in_features=hidden_dim, out_features=config.dim
                ),
                "w3": Linear.Config().build(
                    in_features=config.dim, out_features=hidden_dim
                ),
            }
            weight_sets[group_key] = {
                "attention": attention_weights,
                "ffn": ffn_weights,
            }

        # Store shared weights as modules for parameter tracking
        self._shared_weight_sets = ModuleDict()
        for group_key, weight_set in weight_sets.items():
            group_name = "_".join(map(str, group_key))
            self._shared_weight_sets[f"attn_{group_name}"] = ModuleDict(
                weight_set["attention"]
            )
            self._shared_weight_sets[f"ffn_{group_name}"] = ModuleDict(
                weight_set["ffn"]
            )

        # Create layers
        for layer_id in range(config.n_layers):
            for group in layer_groups:
                if layer_id in group:
                    group_key = tuple(group)
                    weight_set = weight_sets[group_key]
                    layer_lora_rank = 0 if len(group) == 1 else lora_rank
                    self.layers[str(layer_id)] = SharedTransformerBlock(
                        layer_id=layer_id,
                        dim=config.dim,
                        n_heads=n_heads,
                        n_kv_heads=n_kv_heads,
                        head_dim=head_dim,
                        hidden_dim=hidden_dim,
                        norm_eps=config.norm.eps,
                        qk_norm=qk_norm,
                        n_layers=config.n_layers,
                        shared_attention_weights=weight_set["attention"],
                        shared_ffn_weights=weight_set["ffn"],
                        lora_rank=layer_lora_rank,
                    )
                    break

    def _build_attention_sharing_layers(self, config: Config) -> None:
        attn_cfg = config.attention_sharing
        n_heads = config.layer.attention.n_heads
        n_kv_heads = config.layer.attention.n_kv_heads or n_heads
        head_dim = config.layer.attention.head_dim
        hidden_dim = config.layer.feed_forward.hidden_dim
        qk_norm = config.layer.attention.q_norm is not None

        for layer_id in range(config.n_layers):
            self.layers[str(layer_id)] = AttentionSharingTransformerBlock(
                layer_id=layer_id,
                dim=config.dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                hidden_dim=hidden_dim,
                norm_eps=config.norm.eps,
                n_layers=config.n_layers,
                qk_norm=qk_norm,
                attention_config=attn_cfg,
            )

    def _build_combined_sharing_layers(self, config: Config) -> None:
        layer_groups = config.layer_sharing.layer_groups
        lora_rank = config.layer_sharing.lora_rank

        if layer_groups is None:
            raise ValueError(
                "layer_groups must be specified when layer sharing is enabled"
            )

        # Validate
        all_layers = set(range(config.n_layers))
        grouped_layers = set()
        for group in layer_groups:
            for layer in group:
                if layer in grouped_layers:
                    raise ValueError(f"Layer {layer} assigned to multiple groups")
                grouped_layers.add(layer)
        if grouped_layers != all_layers:
            raise ValueError(
                f"layer_groups must cover all layers. "
                f"Expected {all_layers}, got {grouped_layers}"
            )

        n_heads = config.layer.attention.n_heads
        n_kv_heads = config.layer.attention.n_kv_heads or n_heads
        head_dim = config.layer.attention.head_dim
        hidden_dim = config.layer.feed_forward.hidden_dim
        qk_norm = config.layer.attention.q_norm is not None

        weight_sets = {}
        for group in layer_groups:
            group_key = tuple(group)
            attention_weights = {
                "wq": Linear.Config().build(
                    in_features=config.dim, out_features=n_heads * head_dim
                ),
                "wk": Linear.Config().build(
                    in_features=config.dim, out_features=n_kv_heads * head_dim
                ),
                "wv": Linear.Config().build(
                    in_features=config.dim, out_features=n_kv_heads * head_dim
                ),
                "wo": Linear.Config().build(
                    in_features=n_heads * head_dim, out_features=config.dim
                ),
            }
            ffn_weights = {
                "w1": Linear.Config().build(
                    in_features=config.dim, out_features=hidden_dim
                ),
                "w2": Linear.Config().build(
                    in_features=hidden_dim, out_features=config.dim
                ),
                "w3": Linear.Config().build(
                    in_features=config.dim, out_features=hidden_dim
                ),
            }
            weight_sets[group_key] = {
                "attention": attention_weights,
                "ffn": ffn_weights,
            }

        # Store shared weights as modules for parameter tracking
        self._shared_weight_sets = ModuleDict()
        for group_key, weight_set in weight_sets.items():
            group_name = "_".join(map(str, group_key))
            self._shared_weight_sets[f"attn_{group_name}"] = ModuleDict(
                weight_set["attention"]
            )
            self._shared_weight_sets[f"ffn_{group_name}"] = ModuleDict(
                weight_set["ffn"]
            )

        # Create layers
        for layer_id in range(config.n_layers):
            for group in layer_groups:
                if layer_id in group:
                    group_key = tuple(group)
                    weight_set = weight_sets[group_key]
                    layer_lora_rank = 0 if len(group) == 1 else lora_rank
                    self.layers[str(layer_id)] = CombinedSharingTransformerBlock(
                        layer_id=layer_id,
                        dim=config.dim,
                        n_heads=n_heads,
                        n_kv_heads=n_kv_heads,
                        head_dim=head_dim,
                        hidden_dim=hidden_dim,
                        norm_eps=config.norm.eps,
                        n_layers=config.n_layers,
                        qk_norm=qk_norm,
                        shared_attention_weights=weight_set["attention"],
                        shared_ffn_weights=weight_set["ffn"],
                        layer_lora_rank=layer_lora_rank,
                        attention_config=config.attention_sharing,
                    )
                    break

    def init_weights(
        self,
        *,
        buffer_device: torch.device | None = None,
        **kwargs,
    ):
        use_factorized = self.config.factorized_embedding.enabled

        # Handle weight tying (upstream pattern)
        if self.enable_weight_tying and not use_factorized:
            assert self.tok_embeddings is not None and self.output is not None
            self.tok_embeddings.weight = self.output.weight

        # For factorized embeddings, init specially; otherwise delegate to Decoder
        if use_factorized:
            # Init factorized embedding
            if self.tok_embeddings is not None:
                self.tok_embeddings.init_weights()
            # Init RoPE
            buffer_device = buffer_device or self.freqs_cis.device
            if self.rope is not None:
                self.rope.init_weights(buffer_device=buffer_device)
                self.freqs_cis = self.rope.cache
            # Init layers
            for layer in self.layers.values():
                layer.init_weights(buffer_device=buffer_device)
            # Init norm
            if self.norm is not None:
                self.norm.init_weights()
            # Init output (FactorizedOutput has no weights of its own)
        else:
            super().init_weights(buffer_device=buffer_device, **kwargs)

        # Init shared weight sets if they exist
        if hasattr(self, '_shared_weight_sets'):
            for ws in self._shared_weight_sets.values():
                for linear in ws.values():
                    linear.init_weights()
