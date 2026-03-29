# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.trainer import Trainer

from . import model_registry
from .model import (
    AttentionSharingConfig,
    FactorizedEmbeddingConfig,
    LayerSharingConfig,
    Qwen3Model,
    Qwen3TransformerBlock,
)
from .parallelize import parallelize_qwen3
from .state_dict_adapter import Qwen3StateDictAdapter
from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import Embedding, FeedForward, GQAttention, Linear, RoPE
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.protocols.model_spec import ModelSpec


def qwen3_debugmodel() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("debugmodel"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
    )


def qwen3_debugmodel_flex() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("debugmodel_flex"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
    )


def qwen3_0_6b() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./assets/hf/Qwen3-0.6B",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("0.6B"),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=OptimizersContainer.Config(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=10,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
    )


def qwen3_1_7b() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./assets/hf/Qwen3-1.7B",
        model_spec=model_registry("1.7B"),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=20),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=100,
        ),
        checkpoint=CheckpointManager.Config(
            interval=50,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
    )


def qwen3_14b() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./assets/hf/Qwen3-14B",
        model_spec=model_registry("14B"),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=600),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=3000,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            context_parallel_degree=1,
            pipeline_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="full",
        ),
    )


def qwen3_32b() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./assets/hf/Qwen3-32B",
        model_spec=model_registry("32B"),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=600),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=4096,
            steps=3000,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            context_parallel_degree=1,
            pipeline_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="full",
        ),
    )


def qwen3_moe_debug() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("debugmodel_moe"),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4_test",
        ),
        optimizer=OptimizersContainer.Config(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
            expert_tensor_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
    )


# ============================================================================
# Weight sharing experiment configs
# ============================================================================

_LAYER_GROUPS_20L = [
    [0],
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12],
    [13, 14, 15],
    [16, 17, 18],
    [19],
]


def _ws_training_base(model_spec) -> Trainer.Config:
    """Shared training config for all 250M weight sharing experiments."""
    return Trainer.Config(
        hf_assets_path="./assets/hf/Qwen3-0.6B-Base",
        metrics=MetricsProcessor.Config(log_freq=10),
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(dataset="hq_data_20bt"),
        optimizer=OptimizersContainer.Config(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=1500,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=22,
            seq_len=2048,
            steps=55500,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=8,
            data_parallel_shard_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=True,
        ),
        compile=CompileConfig(enable=True),
    )


def qwen3_250m_combined() -> Trainer.Config:
    """250M with all three sharing modes: attention + layer + embedding."""
    spec = model_registry("250M_shared")
    spec.model.attention_sharing = AttentionSharingConfig(
        enabled=True, head_sharing=True, grouping=1, rank=192, two_step=False,
    )
    spec.model.layer_sharing = LayerSharingConfig(
        enabled=True, layer_groups=_LAYER_GROUPS_20L, lora_rank=256,
    )
    spec.model.factorized_embedding = FactorizedEmbeddingConfig(
        enabled=True, d_emb=608, tie_output=True,
    )
    return _ws_training_base(spec)


def qwen3_250m_unshared() -> Trainer.Config:
    """250M baseline without weight sharing."""
    return _ws_training_base(model_registry("250M_unshared"))


def qwen3_250m_attention_only() -> Trainer.Config:
    """250M with attention head sharing only."""
    spec = model_registry("250M_shared")
    spec.model.attention_sharing = AttentionSharingConfig(
        enabled=True, head_sharing=True, grouping=1, rank=192,
    )
    return _ws_training_base(spec)


def qwen3_250m_layer_only() -> Trainer.Config:
    """250M with layer sharing only."""
    spec = model_registry("250M_shared")
    spec.model.layer_sharing = LayerSharingConfig(
        enabled=True, layer_groups=_LAYER_GROUPS_20L, lora_rank=256,
    )
    return _ws_training_base(spec)


def qwen3_250m_embedding_only() -> Trainer.Config:
    """250M with factorized embeddings only."""
    spec = model_registry("250M_shared")
    spec.model.factorized_embedding = FactorizedEmbeddingConfig(
        enabled=True, d_emb=608, tie_output=True,
    )
    return _ws_training_base(spec)


def qwen3_40m_layer() -> Trainer.Config:
    """40M with layer sharing."""
    spec = model_registry("40M")
    spec.model.layer_sharing = LayerSharingConfig(
        enabled=True,
        layer_groups=[[0], [1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11]],
        lora_rank=128,
    )
    return _ws_training_base(spec)


def qwen3_40m_grouped_head_share() -> Trainer.Config:
    """40M with grouped attention head sharing."""
    spec = model_registry("40M")
    spec.model.attention_sharing = AttentionSharingConfig(
        enabled=True, head_sharing=True, grouping=2, rank=64,
    )
    return _ws_training_base(spec)


def qwen3_debug_weightshared() -> Trainer.Config:
    """Debug model with weight sharing (for testing)."""
    spec = model_registry("debugmodel")
    spec.model.layer_sharing = LayerSharingConfig(
        enabled=True,
        layer_groups=[[0], [1, 2, 3], [4, 5, 6], [7]],
        lora_rank=32,
    )
    spec.model.factorized_embedding = FactorizedEmbeddingConfig(
        enabled=True, d_emb=128, tie_output=True,
    )
    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=spec,
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2, decay_ratio=0.8, decay_type="linear", min_lr_factor=0.0,
        ),
        training=TrainingConfig(local_batch_size=8, seq_len=2048, steps=10),
        checkpoint=CheckpointManager.Config(interval=10, last_save_model_only=False),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
    )


def qwen3_debug_unshared() -> Trainer.Config:
    """Debug model without weight sharing (for testing)."""
    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("debugmodel"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2, decay_ratio=0.8, decay_type="linear", min_lr_factor=0.0,
        ),
        training=TrainingConfig(local_batch_size=8, seq_len=2048, steps=10),
        checkpoint=CheckpointManager.Config(interval=10, last_save_model_only=False),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
    )


def _make_spec(dim, n_layers, n_heads, n_kv, ffn, flavor_name):
    """Build a ModelSpec from raw architecture parameters (head_dim=64, Qwen3 style)."""
    model = Qwen3Model.Config(
        vocab_size=151936, dim=dim, n_layers=n_layers,
        norm=RMSNorm.Config(eps=1e-6), enable_weight_tying=True,
        tok_embeddings=Embedding.Config(), output=Linear.Config(),
        layer=Qwen3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(eps=1e-6),
            ffn_norm=RMSNorm.Config(eps=1e-6),
            feed_forward=FeedForward.Config(hidden_dim=ffn),
            attention=GQAttention.Config(
                n_heads=n_heads, n_kv_heads=n_kv, head_dim=64,
                q_norm=RMSNorm.Config(eps=1e-6),
                k_norm=RMSNorm.Config(eps=1e-6),
                attn_backend="sdpa", rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=64, max_seq_len=2048, theta=1000000.0, backend="cos_sin",
        ),
    )
    return ModelSpec(
        name="qwen3", flavor=flavor_name, model=model,
        parallelize_fn=parallelize_qwen3, pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss, post_optimizer_build_fn=None,
        state_dict_adapter=Qwen3StateDictAdapter,
    )

# ============================================================================
# CoLM combined experiment configs (48 total: 12 per size × 4 sizes)
# Generated with verified param counts — see colm_configs_verified.json
# ============================================================================


def _seq_pairs(n):
    return [[i, i + 1] for i in range(0, n, 2)]


def _seq_triples(n):
    return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(n // 3)]


def _cycle(n):
    return [[i, i + n // 2] for i in range(n // 2)]


def _cyclerev(n):
    return [[i, n - 1 - i] for i in range(n // 2)]


def _colm_training(model_spec, size):
    """Training config for CoLM combined experiments, scaled per model size."""
    if size == "1b":
        batch, steps, ckpt_interval, eval_interval = 8, 152600, 15260, 3072
    else:
        batch, steps, ckpt_interval, eval_interval = 16, 76300, 15260, 1536

    dump = f"/checkpoint/optim/sanaelotfi/weight_shared_llms/checkpoints/colm_runs/combined/{model_spec.flavor}"

    return Trainer.Config(
        hf_assets_path="./assets/hf/Qwen3-0.6B-Base",
        dump_folder=dump,
        metrics=MetricsProcessor.Config(log_freq=10, enable_wandb=True),
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(dataset="hq_data_20bt"),
        optimizer=OptimizersContainer.Config(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2000,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=1e-6,
        ),
        training=TrainingConfig(
            local_batch_size=batch,
            seq_len=2048,
            steps=steps,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=8,
            data_parallel_shard_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            enable=True,
            interval=ckpt_interval,
            last_save_model_only=True,
        ),
        compile=CompileConfig(enable=True),
    )


def colm_100m_1_standard() -> Trainer.Config:
    spec = _make_spec(448, 12, 8, 4, 1344, "colm_100m_1_standard")
    return _colm_training(spec, "100m")


def colm_100m_2_wide() -> Trainer.Config:
    spec = _make_spec(512, 7, 8, 2, 1472, "colm_100m_2_wide")
    return _colm_training(spec, "100m")


def colm_100m_3_deep() -> Trainer.Config:
    spec = _make_spec(384, 22, 6, 2, 1216, "colm_100m_3_deep")
    return _colm_training(spec, "100m")


def colm_100m_4_factonly() -> Trainer.Config:
    spec = _make_spec(640, 12, 10, 5, 1920, "colm_100m_4_factonly")
    spec.model.factorized_embedding = FactorizedEmbeddingConfig(
        enabled=True, d_emb=256, tie_output=True,
    )
    return _colm_training(spec, "100m")


def colm_100m_5_conservative() -> Trainer.Config:
    spec = _make_spec(896, 12, 14, 7, 2688, "colm_100m_5_conservative")
    spec.model.attention_sharing = AttentionSharingConfig(
        enabled=True, head_sharing=True, grouping=1, rank=64,
    )
    spec.model.layer_sharing = LayerSharingConfig(
        enabled=True, layer_groups=_seq_pairs(12), lora_rank=48,
    )
    spec.model.factorized_embedding = FactorizedEmbeddingConfig(
        enabled=True, d_emb=176, tie_output=True,
    )
    return _colm_training(spec, "100m")


def colm_100m_6_deep_combined() -> Trainer.Config:
    spec = _make_spec(768, 18, 12, 6, 2304, "colm_100m_6_deep_combined")
    spec.model.attention_sharing = AttentionSharingConfig(
        enabled=True, head_sharing=True, grouping=1, rank=72,
    )
    spec.model.layer_sharing = LayerSharingConfig(
        enabled=True, layer_groups=_seq_triples(18), lora_rank=128,
    )
    spec.model.factorized_embedding = FactorizedEmbeddingConfig(
        enabled=True, d_emb=112, tie_output=True,
    )
    return _colm_training(spec, "100m")


def colm_100m_7_cycle() -> Trainer.Config:
    spec = _make_spec(896, 12, 14, 7, 2688, "colm_100m_7_cycle")
    spec.model.attention_sharing = AttentionSharingConfig(
        enabled=True, head_sharing=True, grouping=1, rank=64,
    )
    spec.model.layer_sharing = LayerSharingConfig(
        enabled=True, layer_groups=_cycle(12), lora_rank=48,
    )
    spec.model.factorized_embedding = FactorizedEmbeddingConfig(
        enabled=True, d_emb=176, tie_output=True,
    )
    return _colm_training(spec, "100m")


def colm_100m_8_cyclerev() -> Trainer.Config:
    spec = _make_spec(896, 12, 14, 7, 2688, "colm_100m_8_cyclerev")
    spec.model.attention_sharing = AttentionSharingConfig(
        enabled=True, head_sharing=True, grouping=1, rank=64,
    )
    spec.model.layer_sharing = LayerSharingConfig(
        enabled=True, layer_groups=_cyclerev(12), lora_rank=48,
    )
    spec.model.factorized_embedding = FactorizedEmbeddingConfig(
        enabled=True, d_emb=176, tie_output=True,
    )
    return _colm_training(spec, "100m")


def colm_100m_9_low_emb() -> Trainer.Config:
    spec = _make_spec(896, 12, 14, 7, 2688, "colm_100m_9_low_emb")
    spec.model.attention_sharing = AttentionSharingConfig(
        enabled=True, head_sharing=True, grouping=1, rank=72,
    )
    spec.model.layer_sharing = LayerSharingConfig(
        enabled=True, layer_groups=_seq_pairs(12), lora_rank=112,
    )
    spec.model.factorized_embedding = FactorizedEmbeddingConfig(
        enabled=True, d_emb=88, tie_output=True,
    )
    return _colm_training(spec, "100m")


def colm_100m_10_high_rank() -> Trainer.Config:
    spec = _make_spec(768, 12, 12, 6, 2304, "colm_100m_10_high_rank")
    spec.model.attention_sharing = AttentionSharingConfig(
        enabled=True, head_sharing=True, grouping=1, rank=128,
    )
    spec.model.layer_sharing = LayerSharingConfig(
        enabled=True, layer_groups=_seq_pairs(12), lora_rank=152,
    )
    spec.model.factorized_embedding = FactorizedEmbeddingConfig(
        enabled=True, d_emb=152, tie_output=True,
    )
    return _colm_training(spec, "100m")


def colm_100m_11_untied() -> Trainer.Config:
    spec = _make_spec(448, 12, 7, 7, 1344, "colm_100m_11_untied")
    spec.model.attention_sharing = AttentionSharingConfig(
        enabled=True, head_sharing=True, grouping=1, rank=96,
    )
    spec.model.layer_sharing = LayerSharingConfig(
        enabled=True, layer_groups=_seq_pairs(12), lora_rank=48,
    )
    spec.model.factorized_embedding = FactorizedEmbeddingConfig(
        enabled=True, d_emb=40, tie_output=False,
    )
    return _colm_training(spec, "100m")


def colm_100m_12_wide_combined() -> Trainer.Config:
    spec = _make_spec(1024, 12, 16, 8, 3072, "colm_100m_12_wide_combined")
    spec.model.attention_sharing = AttentionSharingConfig(
        enabled=True, head_sharing=True, grouping=1, rank=20,
    )
    spec.model.layer_sharing = LayerSharingConfig(
        enabled=True, layer_groups=_seq_pairs(12), lora_rank=12,
    )
    spec.model.factorized_embedding = FactorizedEmbeddingConfig(
        enabled=True, d_emb=120, tie_output=True,
    )
    return _colm_training(spec, "100m")


def colm_300m_1_standard() -> Trainer.Config:
    spec = _make_spec(896, 18, 14, 7, 2688, "colm_300m_1_standard")
    return _colm_training(spec, "300m")


def colm_300m_2_wide() -> Trainer.Config:
    spec = _make_spec(1152, 9, 18, 6, 3296, "colm_300m_2_wide")
    return _colm_training(spec, "300m")


def colm_300m_3_deep() -> Trainer.Config:
    spec = _make_spec(768, 28, 12, 4, 2304, "colm_300m_3_deep")
    return _colm_training(spec, "300m")


def colm_300m_4_factonly() -> Trainer.Config:
    spec = _make_spec(1152, 18, 18, 9, 3456, "colm_300m_4_factonly")
    spec.model.factorized_embedding = FactorizedEmbeddingConfig(
        enabled=True, d_emb=150, tie_output=True,
    )
    return _colm_training(spec, "300m")


def colm_300m_5_conservative() -> Trainer.Config:
    spec = _make_spec(1280, 18, 20, 10, 3840, "colm_300m_5_conservative")
    spec.model.attention_sharing = AttentionSharingConfig(
        enabled=True, head_sharing=True, grouping=1, rank=112,
    )
    spec.model.layer_sharing = LayerSharingConfig(
        enabled=True, layer_groups=_seq_pairs(18), lora_rank=184,
    )
    spec.model.factorized_embedding = FactorizedEmbeddingConfig(
        enabled=True, d_emb=256, tie_output=True,
    )
    return _colm_training(spec, "300m")


def colm_300m_6_deep_combined() -> Trainer.Config:
    spec = _make_spec(1280, 27, 20, 10, 3840, "colm_300m_6_deep_combined")
    spec.model.attention_sharing = AttentionSharingConfig(
        enabled=True, head_sharing=True, grouping=1, rank=80,
    )
    spec.model.layer_sharing = LayerSharingConfig(
        enabled=True, layer_groups=_seq_triples(27), lora_rank=136,
    )
    spec.model.factorized_embedding = FactorizedEmbeddingConfig(
        enabled=True, d_emb=192, tie_output=True,
    )
    return _colm_training(spec, "300m")


def colm_300m_7_cycle() -> Trainer.Config:
    spec = _make_spec(1280, 18, 20, 10, 3840, "colm_300m_7_cycle")
    spec.model.attention_sharing = AttentionSharingConfig(
        enabled=True, head_sharing=True, grouping=1, rank=112,
    )
    spec.model.layer_sharing = LayerSharingConfig(
        enabled=True, layer_groups=_cycle(18), lora_rank=184,
    )
    spec.model.factorized_embedding = FactorizedEmbeddingConfig(
        enabled=True, d_emb=256, tie_output=True,
    )
    return _colm_training(spec, "300m")


def colm_300m_8_cyclerev() -> Trainer.Config:
    spec = _make_spec(1280, 18, 20, 10, 3840, "colm_300m_8_cyclerev")
    spec.model.attention_sharing = AttentionSharingConfig(
        enabled=True, head_sharing=True, grouping=1, rank=112,
    )
    spec.model.layer_sharing = LayerSharingConfig(
        enabled=True, layer_groups=_cyclerev(18), lora_rank=184,
    )
    spec.model.factorized_embedding = FactorizedEmbeddingConfig(
        enabled=True, d_emb=256, tie_output=True,
    )
    return _colm_training(spec, "300m")


def colm_300m_9_low_emb() -> Trainer.Config:
    spec = _make_spec(1280, 18, 20, 10, 3840, "colm_300m_9_low_emb")
    spec.model.attention_sharing = AttentionSharingConfig(
        enabled=True, head_sharing=True, grouping=1, rank=160,
    )
    spec.model.layer_sharing = LayerSharingConfig(
        enabled=True, layer_groups=_seq_pairs(18), lora_rank=216,
    )
    spec.model.factorized_embedding = FactorizedEmbeddingConfig(
        enabled=True, d_emb=128, tie_output=True,
    )
    return _colm_training(spec, "300m")


def colm_300m_10_high_rank() -> Trainer.Config:
    spec = _make_spec(1152, 18, 18, 9, 3456, "colm_300m_10_high_rank")
    spec.model.attention_sharing = AttentionSharingConfig(
        enabled=True, head_sharing=True, grouping=1, rank=200,
    )
    spec.model.layer_sharing = LayerSharingConfig(
        enabled=True, layer_groups=_seq_pairs(18), lora_rank=280,
    )
    spec.model.factorized_embedding = FactorizedEmbeddingConfig(
        enabled=True, d_emb=230, tie_output=True,
    )
    return _colm_training(spec, "300m")


def colm_300m_11_untied() -> Trainer.Config:
    spec = _make_spec(960, 18, 15, 3, 2880, "colm_300m_11_untied")
    spec.model.attention_sharing = AttentionSharingConfig(
        enabled=True, head_sharing=True, grouping=1, rank=184,
    )
    spec.model.layer_sharing = LayerSharingConfig(
        enabled=True, layer_groups=_seq_pairs(18), lora_rank=128,
    )
    spec.model.factorized_embedding = FactorizedEmbeddingConfig(
        enabled=True, d_emb=96, tie_output=False,
    )
    return _colm_training(spec, "300m")


def colm_300m_12_wide_combined() -> Trainer.Config:
    spec = _make_spec(1408, 18, 22, 11, 4224, "colm_300m_12_wide_combined")
    spec.model.attention_sharing = AttentionSharingConfig(
        enabled=True, head_sharing=True, grouping=1, rank=72,
    )
    spec.model.layer_sharing = LayerSharingConfig(
        enabled=True, layer_groups=_seq_pairs(18), lora_rank=112,
    )
    spec.model.factorized_embedding = FactorizedEmbeddingConfig(
        enabled=True, d_emb=211, tie_output=True,
    )
    return _colm_training(spec, "300m")


def colm_500m_1_standard() -> Trainer.Config:
    spec = _make_spec(1024, 28, 16, 8, 3072, "colm_500m_1_standard")
    return _colm_training(spec, "500m")


def colm_500m_2_wide() -> Trainer.Config:
    spec = _make_spec(1280, 16, 20, 10, 3840, "colm_500m_2_wide")
    return _colm_training(spec, "500m")


def colm_500m_3_deep() -> Trainer.Config:
    spec = _make_spec(896, 39, 14, 7, 2688, "colm_500m_3_deep")
    return _colm_training(spec, "500m")


def colm_500m_4_factonly() -> Trainer.Config:
    spec = _make_spec(1408, 20, 22, 11, 4224, "colm_500m_4_factonly")
    spec.model.factorized_embedding = FactorizedEmbeddingConfig(
        enabled=True, d_emb=211, tie_output=True,
    )
    return _colm_training(spec, "500m")


def colm_500m_5_conservative() -> Trainer.Config:
    spec = _make_spec(1408, 28, 22, 11, 4224, "colm_500m_5_conservative")
    spec.model.attention_sharing = AttentionSharingConfig(
        enabled=True, head_sharing=True, grouping=1, rank=60,
    )
    spec.model.layer_sharing = LayerSharingConfig(
        enabled=True, layer_groups=_seq_pairs(28), lora_rank=160,
    )
    spec.model.factorized_embedding = FactorizedEmbeddingConfig(
        enabled=True, d_emb=281, tie_output=True,
    )
    return _colm_training(spec, "500m")


def colm_500m_6_deep_combined() -> Trainer.Config:
    spec = _make_spec(1280, 42, 20, 10, 3840, "colm_500m_6_deep_combined")
    spec.model.attention_sharing = AttentionSharingConfig(
        enabled=True, head_sharing=True, grouping=1, rank=28,
    )
    spec.model.layer_sharing = LayerSharingConfig(
        enabled=True, layer_groups=_seq_triples(42), lora_rank=192,
    )
    spec.model.factorized_embedding = FactorizedEmbeddingConfig(
        enabled=True, d_emb=192, tie_output=True,
    )
    return _colm_training(spec, "500m")


def colm_500m_7_cycle() -> Trainer.Config:
    spec = _make_spec(1408, 28, 22, 11, 4224, "colm_500m_7_cycle")
    spec.model.attention_sharing = AttentionSharingConfig(
        enabled=True, head_sharing=True, grouping=1, rank=60,
    )
    spec.model.layer_sharing = LayerSharingConfig(
        enabled=True, layer_groups=_cycle(28), lora_rank=160,
    )
    spec.model.factorized_embedding = FactorizedEmbeddingConfig(
        enabled=True, d_emb=281, tie_output=True,
    )
    return _colm_training(spec, "500m")


def colm_500m_8_cyclerev() -> Trainer.Config:
    spec = _make_spec(1408, 28, 22, 11, 4224, "colm_500m_8_cyclerev")
    spec.model.attention_sharing = AttentionSharingConfig(
        enabled=True, head_sharing=True, grouping=1, rank=60,
    )
    spec.model.layer_sharing = LayerSharingConfig(
        enabled=True, layer_groups=_cyclerev(28), lora_rank=160,
    )
    spec.model.factorized_embedding = FactorizedEmbeddingConfig(
        enabled=True, d_emb=281, tie_output=True,
    )
    return _colm_training(spec, "500m")


def colm_500m_9_low_emb() -> Trainer.Config:
    spec = _make_spec(1408, 28, 22, 11, 4224, "colm_500m_9_low_emb")
    spec.model.attention_sharing = AttentionSharingConfig(
        enabled=True, head_sharing=True, grouping=1, rank=48,
    )
    spec.model.layer_sharing = LayerSharingConfig(
        enabled=True, layer_groups=_seq_pairs(28), lora_rank=192,
    )
    spec.model.factorized_embedding = FactorizedEmbeddingConfig(
        enabled=True, d_emb=140, tie_output=True,
    )
    return _colm_training(spec, "500m")


def colm_500m_10_high_rank() -> Trainer.Config:
    spec = _make_spec(1280, 28, 20, 10, 2304, "colm_500m_10_high_rank")
    spec.model.attention_sharing = AttentionSharingConfig(
        enabled=True, head_sharing=True, grouping=1, rank=256,
    )
    spec.model.layer_sharing = LayerSharingConfig(
        enabled=True, layer_groups=_seq_pairs(28), lora_rank=416,
    )
    spec.model.factorized_embedding = FactorizedEmbeddingConfig(
        enabled=True, d_emb=256, tie_output=True,
    )
    return _colm_training(spec, "500m")


def colm_500m_11_untied() -> Trainer.Config:
    spec = _make_spec(1024, 28, 16, 8, 3072, "colm_500m_11_untied")
    spec.model.attention_sharing = AttentionSharingConfig(
        enabled=True, head_sharing=True, grouping=1, rank=144,
    )
    spec.model.layer_sharing = LayerSharingConfig(
        enabled=True, layer_groups=_seq_pairs(28), lora_rank=256,
    )
    spec.model.factorized_embedding = FactorizedEmbeddingConfig(
        enabled=True, d_emb=104, tie_output=False,
    )
    return _colm_training(spec, "500m")


def colm_500m_12_wide_combined() -> Trainer.Config:
    spec = _make_spec(1536, 28, 24, 12, 4608, "colm_500m_12_wide_combined")
    spec.model.attention_sharing = AttentionSharingConfig(
        enabled=True, head_sharing=True, grouping=1, rank=96,
    )
    spec.model.layer_sharing = LayerSharingConfig(
        enabled=True, layer_groups=_seq_pairs(28), lora_rank=68,
    )
    spec.model.factorized_embedding = FactorizedEmbeddingConfig(
        enabled=True, d_emb=230, tie_output=True,
    )
    return _colm_training(spec, "500m")


def colm_1b_1_standard() -> Trainer.Config:
    spec = _make_spec(1280, 32, 20, 10, 5120, "colm_1b_1_standard")
    return _colm_training(spec, "1b")


def colm_1b_2_wide() -> Trainer.Config:
    spec = _make_spec(1792, 17, 28, 7, 6272, "colm_1b_2_wide")
    return _colm_training(spec, "1b")


def colm_1b_3_deep() -> Trainer.Config:
    spec = _make_spec(1024, 50, 16, 8, 4352, "colm_1b_3_deep")
    return _colm_training(spec, "1b")


def colm_1b_4_factonly() -> Trainer.Config:
    spec = _make_spec(1664, 28, 26, 13, 4992, "colm_1b_4_factonly")
    spec.model.factorized_embedding = FactorizedEmbeddingConfig(
        enabled=True, d_emb=332, tie_output=True,
    )
    return _colm_training(spec, "1b")


def colm_1b_5_conservative() -> Trainer.Config:
    spec = _make_spec(1664, 32, 26, 13, 4992, "colm_1b_5_conservative")
    spec.model.attention_sharing = AttentionSharingConfig(
        enabled=True, head_sharing=True, grouping=1, rank=160,
    )
    spec.model.layer_sharing = LayerSharingConfig(
        enabled=True, layer_groups=_seq_pairs(32), lora_rank=352,
    )
    spec.model.factorized_embedding = FactorizedEmbeddingConfig(
        enabled=True, d_emb=332, tie_output=True,
    )
    return _colm_training(spec, "1b")


def colm_1b_6_deep_combined() -> Trainer.Config:
    spec = _make_spec(1536, 48, 24, 12, 5376, "colm_1b_6_deep_combined")
    spec.model.attention_sharing = AttentionSharingConfig(
        enabled=True, head_sharing=True, grouping=1, rank=128,
    )
    spec.model.layer_sharing = LayerSharingConfig(
        enabled=True, layer_groups=_seq_triples(48), lora_rank=256,
    )
    spec.model.factorized_embedding = FactorizedEmbeddingConfig(
        enabled=True, d_emb=230, tie_output=True,
    )
    return _colm_training(spec, "1b")


def colm_1b_7_cycle() -> Trainer.Config:
    spec = _make_spec(1664, 32, 26, 13, 4992, "colm_1b_7_cycle")
    spec.model.attention_sharing = AttentionSharingConfig(
        enabled=True, head_sharing=True, grouping=1, rank=160,
    )
    spec.model.layer_sharing = LayerSharingConfig(
        enabled=True, layer_groups=_cycle(32), lora_rank=352,
    )
    spec.model.factorized_embedding = FactorizedEmbeddingConfig(
        enabled=True, d_emb=332, tie_output=True,
    )
    return _colm_training(spec, "1b")


def colm_1b_8_cyclerev() -> Trainer.Config:
    spec = _make_spec(1664, 32, 26, 13, 4992, "colm_1b_8_cyclerev")
    spec.model.attention_sharing = AttentionSharingConfig(
        enabled=True, head_sharing=True, grouping=1, rank=160,
    )
    spec.model.layer_sharing = LayerSharingConfig(
        enabled=True, layer_groups=_cyclerev(32), lora_rank=352,
    )
    spec.model.factorized_embedding = FactorizedEmbeddingConfig(
        enabled=True, d_emb=332, tie_output=True,
    )
    return _colm_training(spec, "1b")


def colm_1b_9_low_emb() -> Trainer.Config:
    spec = _make_spec(1664, 32, 26, 13, 4992, "colm_1b_9_low_emb")
    spec.model.attention_sharing = AttentionSharingConfig(
        enabled=True, head_sharing=True, grouping=1, rank=128,
    )
    spec.model.layer_sharing = LayerSharingConfig(
        enabled=True, layer_groups=_seq_pairs(32), lora_rank=384,
    )
    spec.model.factorized_embedding = FactorizedEmbeddingConfig(
        enabled=True, d_emb=166, tie_output=True,
    )
    return _colm_training(spec, "1b")


def colm_1b_10_high_rank() -> Trainer.Config:
    spec = _make_spec(1664, 32, 26, 13, 4500, "colm_1b_10_high_rank")
    spec.model.attention_sharing = AttentionSharingConfig(
        enabled=True, head_sharing=True, grouping=1, rank=256,
    )
    spec.model.layer_sharing = LayerSharingConfig(
        enabled=True, layer_groups=_seq_pairs(32), lora_rank=384,
    )
    spec.model.factorized_embedding = FactorizedEmbeddingConfig(
        enabled=True, d_emb=332, tie_output=True,
    )
    return _colm_training(spec, "1b")


def colm_1b_11_untied() -> Trainer.Config:
    spec = _make_spec(1536, 32, 24, 12, 4608, "colm_1b_11_untied")
    spec.model.attention_sharing = AttentionSharingConfig(
        enabled=True, head_sharing=True, grouping=1, rank=128,
    )
    spec.model.layer_sharing = LayerSharingConfig(
        enabled=True, layer_groups=_seq_pairs(32), lora_rank=256,
    )
    spec.model.factorized_embedding = FactorizedEmbeddingConfig(
        enabled=True, d_emb=153, tie_output=False,
    )
    return _colm_training(spec, "1b")


def colm_1b_12_wide_combined() -> Trainer.Config:
    spec = _make_spec(1920, 32, 30, 15, 5120, "colm_1b_12_wide_combined")
    spec.model.attention_sharing = AttentionSharingConfig(
        enabled=True, head_sharing=True, grouping=1, rank=112,
    )
    spec.model.layer_sharing = LayerSharingConfig(
        enabled=True, layer_groups=_seq_pairs(32), lora_rank=224,
    )
    spec.model.factorized_embedding = FactorizedEmbeddingConfig(
        enabled=True, d_emb=288, tie_output=True,
    )
    return _colm_training(spec, "1b")
