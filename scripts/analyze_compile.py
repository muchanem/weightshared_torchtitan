#!/usr/bin/env python3
"""
Analyze torch.compile behavior on weight sharing modules.

This script checks for graph breaks and fusion opportunities in the
weight sharing implementation.

Usage:
    python scripts/analyze_compile.py
"""

import logging
import torch
import torch._dynamo as dynamo

# Enable verbose dynamo logging
dynamo.config.verbose = True
dynamo.config.suppress_errors = False

# Set up logging
logging.basicConfig(level=logging.DEBUG)
torch._logging.set_logs(dynamo=logging.DEBUG)

# Import after setting up logging
from torchtitan.models.qwen3.model.weight_sharing import (
    SharedLinearWithLoRA,
    ScaledLoRAModule,
    SharedFeedForward,
)


def test_scaled_lora_compile():
    """Test compilation of ScaledLoRAModule."""
    print("\n" + "=" * 60)
    print("Testing ScaledLoRAModule compilation")
    print("=" * 60)

    lora = ScaledLoRAModule(in_dim=1024, out_dim=1024, rank=256).cuda().bfloat16()
    x = torch.randn(4, 2048, 1024, device="cuda", dtype=torch.bfloat16)

    # Compile
    compiled_lora = torch.compile(lora, fullgraph=True)

    # Warm up
    for _ in range(3):
        out = compiled_lora(x)

    # Check for graph breaks
    print(f"\nScaledLoRAModule forward graph captured successfully")
    return compiled_lora


def test_shared_linear_compile():
    """Test compilation of SharedLinearWithLoRA."""
    print("\n" + "=" * 60)
    print("Testing SharedLinearWithLoRA compilation")
    print("=" * 60)

    shared_linear = torch.nn.Linear(1024, 1024, bias=False).cuda().bfloat16()
    layer = SharedLinearWithLoRA(shared_linear, 1024, 1024, lora_rank=256).cuda().bfloat16()
    x = torch.randn(4, 2048, 1024, device="cuda", dtype=torch.bfloat16)

    # Compile
    try:
        compiled_layer = torch.compile(layer, fullgraph=True)
        for _ in range(3):
            out = compiled_layer(x)
        print(f"\nSharedLinearWithLoRA forward graph captured successfully (fullgraph=True)")
    except Exception as e:
        print(f"\nfullgraph=True failed: {e}")
        print("Trying without fullgraph...")
        compiled_layer = torch.compile(layer)
        for _ in range(3):
            out = compiled_layer(x)
        print(f"SharedLinearWithLoRA compiled with graph breaks")

    return compiled_layer


def test_shared_ffn_compile():
    """Test compilation of SharedFeedForward."""
    print("\n" + "=" * 60)
    print("Testing SharedFeedForward compilation")
    print("=" * 60)

    shared_weights = {
        "w1": torch.nn.Linear(1024, 3072, bias=False).cuda().bfloat16(),
        "w2": torch.nn.Linear(3072, 1024, bias=False).cuda().bfloat16(),
        "w3": torch.nn.Linear(1024, 3072, bias=False).cuda().bfloat16(),
    }
    ffn = SharedFeedForward(shared_weights, dim=1024, hidden_dim=3072, lora_rank=256).cuda().bfloat16()
    x = torch.randn(4, 2048, 1024, device="cuda", dtype=torch.bfloat16)

    # Compile
    try:
        compiled_ffn = torch.compile(ffn, fullgraph=True)
        for _ in range(3):
            out = compiled_ffn(x)
        print(f"\nSharedFeedForward forward graph captured successfully (fullgraph=True)")
    except Exception as e:
        print(f"\nfullgraph=True failed: {e}")
        print("Trying without fullgraph...")
        compiled_ffn = torch.compile(ffn)
        for _ in range(3):
            out = compiled_ffn(x)
        print(f"SharedFeedForward compiled with graph breaks")

    return compiled_ffn


def count_graph_breaks():
    """Get statistics about graph breaks."""
    print("\n" + "=" * 60)
    print("Graph Break Statistics")
    print("=" * 60)

    # Reset counters
    dynamo.reset()

    shared_linear = torch.nn.Linear(1024, 1024, bias=False).cuda().bfloat16()
    layer = SharedLinearWithLoRA(shared_linear, 1024, 1024, lora_rank=256).cuda().bfloat16()
    x = torch.randn(4, 2048, 1024, device="cuda", dtype=torch.bfloat16)

    # Use explain mode to get detailed info
    explanation = dynamo.explain(layer)(x)
    print(f"\nExplanation for SharedLinearWithLoRA:")
    print(f"  Graph count: {explanation.graph_count}")
    print(f"  Graph break count: {explanation.graph_break_count}")
    print(f"  Break reasons: {explanation.break_reasons}")

    return explanation


def benchmark_compiled_vs_eager():
    """Compare compiled vs eager performance."""
    print("\n" + "=" * 60)
    print("Compiled vs Eager Benchmark")
    print("=" * 60)

    import torch.utils.benchmark as benchmark

    shared_linear = torch.nn.Linear(1024, 1024, bias=False).cuda().bfloat16()
    layer = SharedLinearWithLoRA(shared_linear, 1024, 1024, lora_rank=256).cuda().bfloat16()
    layer_compiled = torch.compile(layer, mode="reduce-overhead")

    x = torch.randn(4, 2048, 1024, device="cuda", dtype=torch.bfloat16)

    # Warmup compiled
    for _ in range(10):
        _ = layer_compiled(x)
    torch.cuda.synchronize()

    # Benchmark
    t_eager = benchmark.Timer(
        stmt="layer(x)",
        globals={"layer": layer, "x": x},
    )

    t_compiled = benchmark.Timer(
        stmt="layer(x)",
        globals={"layer": layer_compiled, "x": x},
    )

    eager_result = t_eager.timeit(100)
    compiled_result = t_compiled.timeit(100)

    print(f"\nEager:    {eager_result.mean * 1000:.3f} ms")
    print(f"Compiled: {compiled_result.mean * 1000:.3f} ms")
    print(f"Speedup:  {eager_result.mean / compiled_result.mean:.2f}x")


if __name__ == "__main__":
    print("=" * 60)
    print("torch.compile Analysis for Weight Sharing Modules")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")

    # Run tests
    test_scaled_lora_compile()
    test_shared_linear_compile()
    test_shared_ffn_compile()
    count_graph_breaks()
    benchmark_compiled_vs_eager()

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
