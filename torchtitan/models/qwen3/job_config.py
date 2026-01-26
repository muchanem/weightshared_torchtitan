# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Custom JobConfig extension for Qwen3 weight sharing.

This module provides dataclass-based configuration for weight sharing features:
- Attention sharing (shared Q/K/V projections with LoRA offsets)
- Layer sharing (shared transformer blocks with per-layer LoRA adapters)
- Factorized embeddings (lower-rank embedding factorization)
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AttentionSharingConfig:
    """Configuration for attention weight sharing.

    Attributes:
        enabled: Whether attention sharing is enabled.
        qkv_sharing: Groups of tied projections, e.g., [["q","k","v"]] to tie all.
        head_sharing: Whether to share weights across attention heads.
        grouping: Number of template heads per group (1 = full sharing).
        rank: LoRA rank for per-head offset adapters.
        two_step: Whether to use two-step offset decomposition
            (shared head offset + per-projection offset).
    """

    enabled: bool = False
    qkv_sharing: Optional[List[List[str]]] = None
    head_sharing: bool = False
    grouping: int = 1
    rank: int = 8
    two_step: bool = False


@dataclass
class LayerSharingConfig:
    """Configuration for layer weight sharing.

    Attributes:
        enabled: Whether layer sharing is enabled.
        layer_groups: Groups of layers that share weights, e.g., [[0], [1,2,3], [4]].
            Layers in the same group share base weights with per-layer LoRA adapters.
        lora_rank: LoRA rank for per-layer adapters. Set to 0 for pure weight sharing.
    """

    enabled: bool = False
    layer_groups: Optional[List[List[int]]] = None
    lora_rank: int = 8


@dataclass
class FactorizedEmbeddingConfig:
    """Configuration for factorized embeddings.

    Attributes:
        enabled: Whether factorized embeddings are enabled.
        d_emb: Intermediate embedding dimension (vocab -> d_emb -> dim).
        tie_output: Whether to tie output projection to embedding weights.
    """

    enabled: bool = False
    d_emb: int = 256
    tie_output: bool = True


@dataclass
class WeightSharingConfig:
    """Top-level configuration for all weight sharing features.

    Attributes:
        attention: Attention sharing configuration.
        layer: Layer sharing configuration.
        embedding: Factorized embedding configuration.
    """

    attention: AttentionSharingConfig = field(default_factory=AttentionSharingConfig)
    layer: LayerSharingConfig = field(default_factory=LayerSharingConfig)
    embedding: FactorizedEmbeddingConfig = field(default_factory=FactorizedEmbeddingConfig)


@dataclass
class JobConfig:
    """Extended JobConfig with weight sharing support.

    This config is loaded via the custom_config_module mechanism in torchtitan.
    When specified in a TOML file as:

        [job]
        custom_config_module = "torchtitan.models.qwen3.job_config"

    The weight_sharing section will be parsed and made available as
    job_config.weight_sharing.
    """

    weight_sharing: WeightSharingConfig = field(default_factory=WeightSharingConfig)
