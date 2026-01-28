#!/usr/bin/env python3
"""
Profile 20 SharedFeedForward layers to simulate full model and verify kernel reduction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def count_kernels(prof):
    """Count CUDA kernels by type."""
    events = prof.key_averages()
    stats = {'gemm': 0, 'add': 0, 'copy': 0, 'other': 0}
    times = {'gemm': 0.0, 'add': 0.0, 'copy': 0.0, 'other': 0.0}

    for e in events:
        if e.self_device_time_total == 0:
            continue
        key_lower = e.key.lower()
        time_ms = e.self_device_time_total / 1000

        if 'nvjet' in key_lower or 'gemm' in key_lower or 'cutlass' in key_lower:
            stats['gemm'] += e.count
            times['gemm'] += time_ms
        elif 'memcpy' in key_lower or 'copy' in key_lower:
            stats['copy'] += e.count
            times['copy'] += time_ms
        elif 'add' in key_lower or 'elementwise' in key_lower or 'mul' in key_lower:
            stats['add'] += e.count
            times['add'] += time_ms
        else:
            stats['other'] += e.count
            times['other'] += time_ms

    return stats, times


class OldScaledLoRAModule(nn.Module):
    """Original implementation with scaling."""
    def __init__(self, in_dim, out_dim, rank=8, alpha=None):
        super().__init__()
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank * 2
        self.w_a = nn.Linear(in_dim, rank, bias=False)
        self.w_b = nn.Linear(rank, out_dim, bias=False)

    def forward(self, x):
        return self.w_b(self.w_a(x)) * (self.alpha / self.rank)


class OldSharedLinearWithLoRA(nn.Module):
    def __init__(self, shared_linear, in_features, out_features, lora_rank):
        super().__init__()
        self.shared_linear = shared_linear
        self.lora_rank = lora_rank
        if lora_rank > 0:
            self.lora = OldScaledLoRAModule(in_features, out_features, rank=lora_rank)

    def forward(self, x):
        base_output = self.shared_linear(x)
        if self.lora_rank > 0:
            return base_output + self.lora(x)
        return base_output


class OldSharedFeedForward(nn.Module):
    def __init__(self, shared_weights, dim, hidden_dim, lora_rank):
        super().__init__()
        self.w1 = OldSharedLinearWithLoRA(shared_weights["w1"], dim, hidden_dim, lora_rank)
        self.w2 = OldSharedLinearWithLoRA(shared_weights["w2"], hidden_dim, dim, lora_rank)
        self.w3 = OldSharedLinearWithLoRA(shared_weights["w3"], dim, hidden_dim, lora_rank)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def main():
    from torchtitan.models.qwen3.model.weight_sharing import SharedFeedForward

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    # Model config
    n_layers = 20
    dim = 1024
    hidden_dim = 2816
    lora_rank = 256
    batch, seq = 4, 2048

    print(f"Simulating {n_layers} FFN layers with LoRA rank {lora_rank}")
    print(f"Input: batch={batch}, seq={seq}, dim={dim}")

    # Create shared weights (8 unique groups for 20 layers)
    shared_weights = {
        "w1": nn.Linear(dim, hidden_dim, bias=False).to(device).to(dtype),
        "w2": nn.Linear(hidden_dim, dim, bias=False).to(device).to(dtype),
        "w3": nn.Linear(dim, hidden_dim, bias=False).to(device).to(dtype),
    }

    # Create old and new FFN stacks
    old_ffns = nn.ModuleList([
        OldSharedFeedForward(shared_weights, dim, hidden_dim, lora_rank)
        for _ in range(n_layers)
    ]).to(device).to(dtype)

    new_ffns = nn.ModuleList([
        SharedFeedForward(shared_weights, dim, hidden_dim, lora_rank)
        for _ in range(n_layers)
    ]).to(device).to(dtype)

    x = torch.randn(batch, seq, dim, device=device, dtype=dtype)

    def run_old():
        out = x
        for ffn in old_ffns:
            out = ffn(out)
        return out

    def run_new():
        out = x
        for ffn in new_ffns:
            out = ffn(out)
        return out

    # Warmup
    print("Warming up...")
    for _ in range(3):
        run_old()
        run_new()
    torch.cuda.synchronize()

    # Profile old
    print("Profiling OLD implementation...")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            run_old()
        torch.cuda.synchronize()
    old_stats, old_times = count_kernels(prof)

    # Profile new
    print("Profiling NEW implementation...")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            run_new()
        torch.cuda.synchronize()
    new_stats, new_times = count_kernels(prof)

    print("\n" + "=" * 60)
    print(f"FFN STACK PROFILE ({n_layers} layers, 3 forward passes)")
    print("=" * 60)

    print("\nOLD Implementation:")
    print(f"  GEMM: {old_stats['gemm']:4d} ({old_times['gemm']:.2f} ms)")
    print(f"  ELEM: {old_stats['add']:4d} ({old_times['add']:.2f} ms)")
    print(f"  COPY: {old_stats['copy']:4d} ({old_times['copy']:.2f} ms)")
    print(f"  Total time: {sum(old_times.values()):.2f} ms")

    print("\nNEW Implementation:")
    print(f"  GEMM: {new_stats['gemm']:4d} ({new_times['gemm']:.2f} ms)")
    print(f"  ELEM: {new_stats['add']:4d} ({new_times['add']:.2f} ms)")
    print(f"  COPY: {new_stats['copy']:4d} ({new_times['copy']:.2f} ms)")
    print(f"  Total time: {sum(new_times.values()):.2f} ms")

    print("\n" + "-" * 60)
    print("Changes:")
    print(f"  ELEM kernels: {old_stats['add']} -> {new_stats['add']} ({new_stats['add'] - old_stats['add']:+d})")
    print(f"  COPY kernels: {old_stats['copy']} -> {new_stats['copy']} ({new_stats['copy'] - old_stats['copy']:+d})")
    print(f"  Time: {sum(old_times.values()):.2f} ms -> {sum(new_times.values()):.2f} ms")
    print(f"  Speedup: {sum(old_times.values())/sum(new_times.values()):.2f}x")


if __name__ == "__main__":
    main()
