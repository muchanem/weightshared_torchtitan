#!/usr/bin/env python3
"""
Microbenchmark for all weight sharing components.

Compares:
1. Standard nn.Embedding vs FactorizedEmbedding
2. Standard Attention vs CombinedSharingAttention (with head sharing)
3. Standard FFN vs SharedFeedForward (with LoRA)
4. Standard nn.Linear vs SharedLinearWithLoRA

Usage:
    python scripts/benchmark_components.py
"""

import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark
from dataclasses import dataclass
from typing import Dict

# Import weight sharing components
from torchtitan.models.qwen3.model.factorized_embedding import (
    FactorizedEmbedding,
    FactorizedOutput,
)
from torchtitan.models.qwen3.model.weight_sharing import (
    SharedLinearWithLoRA,
    SharedFeedForward,
    ScaledLoRAModule,
)


@dataclass
class BenchmarkConfig:
    batch_size: int = 4
    seq_len: int = 2048
    vocab_size: int = 151936  # Qwen3 vocab
    dim: int = 1024
    d_emb: int = 608  # Factorized embedding dim
    hidden_dim: int = 3072
    n_heads: int = 16
    n_kv_heads: int = 8
    head_dim: int = 64
    lora_rank: int = 256
    head_lora_rank: int = 192
    num_warmup: int = 10
    num_runs: int = 100


def benchmark_embedding(cfg: BenchmarkConfig) -> Dict[str, float]:
    """Benchmark standard vs factorized embedding."""
    print("\n" + "=" * 70)
    print("Embedding Benchmark")
    print("=" * 70)

    # Standard embedding
    standard_emb = nn.Embedding(cfg.vocab_size, cfg.dim).cuda().bfloat16()

    # Factorized embedding
    factorized_emb = FactorizedEmbedding(cfg.vocab_size, cfg.d_emb, cfg.dim).cuda().bfloat16()

    tokens = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len), device="cuda")

    # Warmup
    for _ in range(cfg.num_warmup):
        _ = standard_emb(tokens)
        _ = factorized_emb(tokens)
    torch.cuda.synchronize()

    # Benchmark forward
    t_standard = benchmark.Timer(
        stmt="emb(tokens)",
        globals={"emb": standard_emb, "tokens": tokens},
    )
    t_factorized = benchmark.Timer(
        stmt="emb(tokens)",
        globals={"emb": factorized_emb, "tokens": tokens},
    )

    standard_result = t_standard.timeit(cfg.num_runs)
    factorized_result = t_factorized.timeit(cfg.num_runs)

    standard_params = cfg.vocab_size * cfg.dim
    factorized_params = cfg.vocab_size * cfg.d_emb + cfg.d_emb * cfg.dim

    print(f"\nEmbedding Forward (batch={cfg.batch_size}, seq={cfg.seq_len}, vocab={cfg.vocab_size})")
    print(f"  Standard (dim={cfg.dim}):      {standard_result.mean * 1000:.3f} ms  ({standard_params/1e6:.1f}M params)")
    print(f"  Factorized (d_emb={cfg.d_emb}): {factorized_result.mean * 1000:.3f} ms  ({factorized_params/1e6:.1f}M params)")
    print(f"  Overhead:                      {(factorized_result.mean / standard_result.mean - 1) * 100:+.1f}%")
    print(f"  Param reduction:               {(1 - factorized_params / standard_params) * 100:.1f}%")

    return {
        "standard_ms": standard_result.mean * 1000,
        "factorized_ms": factorized_result.mean * 1000,
        "overhead_pct": (factorized_result.mean / standard_result.mean - 1) * 100,
        "param_reduction_pct": (1 - factorized_params / standard_params) * 100,
    }


def benchmark_output_projection(cfg: BenchmarkConfig) -> Dict[str, float]:
    """Benchmark standard vs factorized output projection."""
    print("\n" + "=" * 70)
    print("Output Projection Benchmark")
    print("=" * 70)

    # Standard: just a linear layer (or tied embedding weight)
    standard_output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False).cuda().bfloat16()

    # Factorized output (tied to factorized embedding)
    factorized_emb = FactorizedEmbedding(cfg.vocab_size, cfg.d_emb, cfg.dim).cuda().bfloat16()
    factorized_output = FactorizedOutput(factorized_emb).cuda().bfloat16()

    h = torch.randn(cfg.batch_size, cfg.seq_len, cfg.dim, device="cuda", dtype=torch.bfloat16)

    # Warmup
    for _ in range(cfg.num_warmup):
        _ = standard_output(h)
        _ = factorized_output(h)
    torch.cuda.synchronize()

    t_standard = benchmark.Timer(
        stmt="proj(h)",
        globals={"proj": standard_output, "h": h},
    )
    t_factorized = benchmark.Timer(
        stmt="proj(h)",
        globals={"proj": factorized_output, "h": h},
    )

    standard_result = t_standard.timeit(cfg.num_runs)
    factorized_result = t_factorized.timeit(cfg.num_runs)

    print(f"\nOutput Projection Forward (batch={cfg.batch_size}, seq={cfg.seq_len}, vocab={cfg.vocab_size})")
    print(f"  Standard Linear:   {standard_result.mean * 1000:.3f} ms")
    print(f"  Factorized Output: {factorized_result.mean * 1000:.3f} ms")
    print(f"  Overhead:          {(factorized_result.mean / standard_result.mean - 1) * 100:+.1f}%")

    return {
        "standard_ms": standard_result.mean * 1000,
        "factorized_ms": factorized_result.mean * 1000,
        "overhead_pct": (factorized_result.mean / standard_result.mean - 1) * 100,
    }


def benchmark_linear_lora(cfg: BenchmarkConfig) -> Dict[str, float]:
    """Benchmark standard linear vs SharedLinearWithLoRA."""
    print("\n" + "=" * 70)
    print("Linear + LoRA Benchmark")
    print("=" * 70)

    # Standard linear
    linear = nn.Linear(cfg.dim, cfg.dim, bias=False).cuda().bfloat16()

    # SharedLinearWithLoRA
    shared_base = nn.Linear(cfg.dim, cfg.dim, bias=False).cuda().bfloat16()
    shared_lora = SharedLinearWithLoRA(shared_base, cfg.dim, cfg.dim, lora_rank=cfg.lora_rank).cuda().bfloat16()

    x = torch.randn(cfg.batch_size, cfg.seq_len, cfg.dim, device="cuda", dtype=torch.bfloat16)

    # Warmup
    for _ in range(cfg.num_warmup):
        _ = linear(x)
        _ = shared_lora(x)
    torch.cuda.synchronize()

    t_linear = benchmark.Timer(
        stmt="layer(x)",
        globals={"layer": linear, "x": x},
    )
    t_lora = benchmark.Timer(
        stmt="layer(x)",
        globals={"layer": shared_lora, "x": x},
    )

    linear_result = t_linear.timeit(cfg.num_runs)
    lora_result = t_lora.timeit(cfg.num_runs)

    print(f"\nLinear Forward (batch={cfg.batch_size}, seq={cfg.seq_len}, dim={cfg.dim})")
    print(f"  nn.Linear:             {linear_result.mean * 1000:.3f} ms")
    print(f"  SharedLinearWithLoRA:  {lora_result.mean * 1000:.3f} ms  (rank={cfg.lora_rank})")
    print(f"  Overhead:              {(lora_result.mean / linear_result.mean - 1) * 100:+.1f}%")

    return {
        "linear_ms": linear_result.mean * 1000,
        "lora_ms": lora_result.mean * 1000,
        "overhead_pct": (lora_result.mean / linear_result.mean - 1) * 100,
    }


def benchmark_ffn(cfg: BenchmarkConfig) -> Dict[str, float]:
    """Benchmark standard SwiGLU FFN vs SharedFeedForward."""
    print("\n" + "=" * 70)
    print("FFN (SwiGLU) Benchmark")
    print("=" * 70)

    # Standard SwiGLU FFN
    class StandardFFN(nn.Module):
        def __init__(self, dim, hidden_dim):
            super().__init__()
            self.w1 = nn.Linear(dim, hidden_dim, bias=False)
            self.w2 = nn.Linear(hidden_dim, dim, bias=False)
            self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        def forward(self, x):
            return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))

    standard_ffn = StandardFFN(cfg.dim, cfg.hidden_dim).cuda().bfloat16()

    # SharedFeedForward
    shared_weights = {
        "w1": nn.Linear(cfg.dim, cfg.hidden_dim, bias=False).cuda().bfloat16(),
        "w2": nn.Linear(cfg.hidden_dim, cfg.dim, bias=False).cuda().bfloat16(),
        "w3": nn.Linear(cfg.dim, cfg.hidden_dim, bias=False).cuda().bfloat16(),
    }
    shared_ffn = SharedFeedForward(shared_weights, dim=cfg.dim, hidden_dim=cfg.hidden_dim, lora_rank=cfg.lora_rank).cuda().bfloat16()

    x = torch.randn(cfg.batch_size, cfg.seq_len, cfg.dim, device="cuda", dtype=torch.bfloat16)

    # Warmup
    for _ in range(cfg.num_warmup):
        _ = standard_ffn(x)
        _ = shared_ffn(x)
    torch.cuda.synchronize()

    t_standard = benchmark.Timer(
        stmt="ffn(x)",
        globals={"ffn": standard_ffn, "x": x},
    )
    t_shared = benchmark.Timer(
        stmt="ffn(x)",
        globals={"ffn": shared_ffn, "x": x},
    )

    standard_result = t_standard.timeit(cfg.num_runs)
    shared_result = t_shared.timeit(cfg.num_runs)

    print(f"\nFFN Forward (batch={cfg.batch_size}, seq={cfg.seq_len}, dim={cfg.dim}, hidden={cfg.hidden_dim})")
    print(f"  StandardFFN:        {standard_result.mean * 1000:.3f} ms")
    print(f"  SharedFeedForward:  {shared_result.mean * 1000:.3f} ms  (rank={cfg.lora_rank})")
    print(f"  Overhead:           {(shared_result.mean / standard_result.mean - 1) * 100:+.1f}%")

    return {
        "standard_ms": standard_result.mean * 1000,
        "shared_ms": shared_result.mean * 1000,
        "overhead_pct": (shared_result.mean / standard_result.mean - 1) * 100,
    }


def benchmark_attention_projections(cfg: BenchmarkConfig) -> Dict[str, float]:
    """Benchmark attention projections: standard vs shared+LoRA+head_offset."""
    print("\n" + "=" * 70)
    print("Attention Projections Benchmark (Q/K/V/O)")
    print("=" * 70)

    q_dim = cfg.n_heads * cfg.head_dim
    kv_dim = cfg.n_kv_heads * cfg.head_dim

    # Standard attention projections
    class StandardAttnProj(nn.Module):
        def __init__(self):
            super().__init__()
            self.wq = nn.Linear(cfg.dim, q_dim, bias=False)
            self.wk = nn.Linear(cfg.dim, kv_dim, bias=False)
            self.wv = nn.Linear(cfg.dim, kv_dim, bias=False)
            self.wo = nn.Linear(q_dim, cfg.dim, bias=False)

        def forward(self, x):
            q = self.wq(x)
            k = self.wk(x)
            v = self.wv(x)
            # Simulate output projection (normally after attention)
            return self.wo(q), k, v

    # Shared projections with layer LoRA + head LoRA offsets
    class SharedAttnProjWithHeadOffset(nn.Module):
        def __init__(self):
            super().__init__()
            # Shared base weights
            base_wq = nn.Linear(cfg.dim, q_dim, bias=False)
            base_wk = nn.Linear(cfg.dim, kv_dim, bias=False)
            base_wv = nn.Linear(cfg.dim, kv_dim, bias=False)
            base_wo = nn.Linear(q_dim, cfg.dim, bias=False)

            # Layer LoRA
            self.wq = SharedLinearWithLoRA(base_wq, cfg.dim, q_dim, cfg.lora_rank)
            self.wk = SharedLinearWithLoRA(base_wk, cfg.dim, kv_dim, cfg.lora_rank)
            self.wv = SharedLinearWithLoRA(base_wv, cfg.dim, kv_dim, cfg.lora_rank)
            self.wo = SharedLinearWithLoRA(base_wo, q_dim, cfg.dim, cfg.lora_rank)

            # Head LoRA offsets
            self.wq_head_offset = ScaledLoRAModule(cfg.dim, q_dim, rank=cfg.head_lora_rank)
            self.wk_head_offset = ScaledLoRAModule(cfg.dim, kv_dim, rank=cfg.head_lora_rank)
            self.wv_head_offset = ScaledLoRAModule(cfg.dim, kv_dim, rank=cfg.head_lora_rank)

        def forward(self, x):
            q = self.wq(x) + self.wq_head_offset(x)
            k = self.wk(x) + self.wk_head_offset(x)
            v = self.wv(x) + self.wv_head_offset(x)
            return self.wo(q), k, v

    standard_attn = StandardAttnProj().cuda().bfloat16()
    shared_attn = SharedAttnProjWithHeadOffset().cuda().bfloat16()

    x = torch.randn(cfg.batch_size, cfg.seq_len, cfg.dim, device="cuda", dtype=torch.bfloat16)

    # Warmup
    for _ in range(cfg.num_warmup):
        _ = standard_attn(x)
        _ = shared_attn(x)
    torch.cuda.synchronize()

    t_standard = benchmark.Timer(
        stmt="attn(x)",
        globals={"attn": standard_attn, "x": x},
    )
    t_shared = benchmark.Timer(
        stmt="attn(x)",
        globals={"attn": shared_attn, "x": x},
    )

    standard_result = t_standard.timeit(cfg.num_runs)
    shared_result = t_shared.timeit(cfg.num_runs)

    print(f"\nAttention Projections Forward (batch={cfg.batch_size}, seq={cfg.seq_len})")
    print(f"  n_heads={cfg.n_heads}, n_kv_heads={cfg.n_kv_heads}, head_dim={cfg.head_dim}")
    print(f"  Standard (4 linears):     {standard_result.mean * 1000:.3f} ms")
    print(f"  Shared+LoRA+HeadOffset:   {shared_result.mean * 1000:.3f} ms")
    print(f"  Overhead:                 {(shared_result.mean / standard_result.mean - 1) * 100:+.1f}%")

    # Count operations
    std_matmuls = 4  # wq, wk, wv, wo
    shared_matmuls = 4 * 3 + 3 * 2  # 4 SharedLinearWithLoRA (3 each) + 3 head offsets (2 each)
    print(f"  Matmul count: {std_matmuls} vs {shared_matmuls} ({shared_matmuls/std_matmuls:.1f}x)")

    return {
        "standard_ms": standard_result.mean * 1000,
        "shared_ms": shared_result.mean * 1000,
        "overhead_pct": (shared_result.mean / standard_result.mean - 1) * 100,
    }


def main():
    print("=" * 70)
    print("Weight Sharing Component Benchmark Suite")
    print("=" * 70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA device: {torch.cuda.get_device_name()}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    cfg = BenchmarkConfig()
    print(f"\nConfiguration:")
    print(f"  batch_size={cfg.batch_size}, seq_len={cfg.seq_len}")
    print(f"  dim={cfg.dim}, hidden_dim={cfg.hidden_dim}")
    print(f"  vocab_size={cfg.vocab_size}, d_emb={cfg.d_emb}")
    print(f"  lora_rank={cfg.lora_rank}, head_lora_rank={cfg.head_lora_rank}")

    results = {}

    # Run all benchmarks
    results["embedding"] = benchmark_embedding(cfg)
    results["output"] = benchmark_output_projection(cfg)
    results["linear_lora"] = benchmark_linear_lora(cfg)
    results["ffn"] = benchmark_ffn(cfg)
    results["attention"] = benchmark_attention_projections(cfg)

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Component':<30} {'Standard':>12} {'Shared':>12} {'Overhead':>12}")
    print("-" * 70)
    print(f"{'Embedding (forward)':<30} {results['embedding']['standard_ms']:>10.3f}ms {results['embedding']['factorized_ms']:>10.3f}ms {results['embedding']['overhead_pct']:>+10.1f}%")
    print(f"{'Output Projection (forward)':<30} {results['output']['standard_ms']:>10.3f}ms {results['output']['factorized_ms']:>10.3f}ms {results['output']['overhead_pct']:>+10.1f}%")
    print(f"{'Linear + LoRA (forward)':<30} {results['linear_lora']['linear_ms']:>10.3f}ms {results['linear_lora']['lora_ms']:>10.3f}ms {results['linear_lora']['overhead_pct']:>+10.1f}%")
    print(f"{'FFN SwiGLU (forward)':<30} {results['ffn']['standard_ms']:>10.3f}ms {results['ffn']['shared_ms']:>10.3f}ms {results['ffn']['overhead_pct']:>+10.1f}%")
    print(f"{'Attention Projections (Q/K/V/O)':<30} {results['attention']['standard_ms']:>10.3f}ms {results['attention']['shared_ms']:>10.3f}ms {results['attention']['overhead_pct']:>+10.1f}%")

    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
