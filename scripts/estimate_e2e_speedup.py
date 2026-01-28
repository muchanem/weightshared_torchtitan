#!/usr/bin/env python3
"""
Estimate end-to-end speedup based on component-level improvements.

Based on profiling:
- SharedFeedForward (20 layers): 1.14x speedup
- SharedLinearWithLoRA: 1.16x speedup

The model also has attention LoRAs which benefit from removed scaling but
not from addmm fusion (they add to attention output, not to a base linear).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class OldScaledLoRAModule(nn.Module):
    """Original with scaling."""
    def __init__(self, in_dim, out_dim, rank=8):
        super().__init__()
        self.rank = rank
        self.alpha = rank * 2
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


def benchmark(fn, iterations=500, warmup=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()

    return (time.perf_counter() - start) * 1000 / iterations


def main():
    from torchtitan.models.qwen3.model.weight_sharing import (
        ScaledLoRAModule,
        SharedLinearWithLoRA,
        SharedFeedForward,
    )

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    # Model config (250M shared)
    n_layers = 20
    dim = 1024
    hidden_dim = 2816
    layer_lora_rank = 256
    head_lora_rank = 192
    n_heads = 16
    head_dim = 64

    batch, seq = 4, 2048
    x = torch.randn(batch, seq, dim, device=device, dtype=dtype)

    print("=" * 60)
    print("E2E SPEEDUP ESTIMATION")
    print("=" * 60)

    # ============================================================
    # Component 1: FFN Stack (20 layers)
    # ============================================================
    print("\n--- FFN Stack (20 layers) ---")

    shared_weights = {
        "w1": nn.Linear(dim, hidden_dim, bias=False).to(device).to(dtype),
        "w2": nn.Linear(hidden_dim, dim, bias=False).to(device).to(dtype),
        "w3": nn.Linear(dim, hidden_dim, bias=False).to(device).to(dtype),
    }

    old_ffns = nn.ModuleList([
        OldSharedFeedForward(shared_weights, dim, hidden_dim, layer_lora_rank)
        for _ in range(n_layers)
    ]).to(device).to(dtype)

    new_ffns = nn.ModuleList([
        SharedFeedForward(shared_weights, dim, hidden_dim, layer_lora_rank)
        for _ in range(n_layers)
    ]).to(device).to(dtype)

    def run_old_ffn():
        out = x
        for ffn in old_ffns:
            out = ffn(out)
        return out

    def run_new_ffn():
        out = x
        for ffn in new_ffns:
            out = ffn(out)
        return out

    t_old_ffn = benchmark(run_old_ffn)
    t_new_ffn = benchmark(run_new_ffn)
    ffn_speedup = t_old_ffn / t_new_ffn

    print(f"Old FFN: {t_old_ffn:.3f} ms")
    print(f"New FFN: {t_new_ffn:.3f} ms")
    print(f"Speedup: {ffn_speedup:.2f}x")

    # ============================================================
    # Component 2: Attention LoRAs (head offsets only)
    # These benefit from removed scaling but NOT addmm fusion
    # ============================================================
    print("\n--- Attention Head LoRAs (20 layers x 3 offsets) ---")

    # Old: with scaling
    old_head_loras = nn.ModuleList([
        OldScaledLoRAModule(dim, n_heads * head_dim, rank=head_lora_rank)
        for _ in range(n_layers * 3)  # wq, wk, wv per layer
    ]).to(device).to(dtype)

    # New: no scaling
    new_head_loras = nn.ModuleList([
        ScaledLoRAModule(dim, n_heads * head_dim, rank=head_lora_rank)
        for _ in range(n_layers * 3)
    ]).to(device).to(dtype)

    def run_old_head_loras():
        out = torch.zeros_like(x.unsqueeze(-1).expand(-1, -1, -1, n_heads * head_dim).squeeze(-2))
        for lora in old_head_loras:
            out = out + lora(x)
        return out

    def run_new_head_loras():
        out = torch.zeros_like(x.unsqueeze(-1).expand(-1, -1, -1, n_heads * head_dim).squeeze(-2))
        for lora in new_head_loras:
            out = out + lora(x)
        return out

    t_old_head = benchmark(run_old_head_loras)
    t_new_head = benchmark(run_new_head_loras)
    head_speedup = t_old_head / t_new_head

    print(f"Old head LoRAs: {t_old_head:.3f} ms")
    print(f"New head LoRAs: {t_new_head:.3f} ms")
    print(f"Speedup: {head_speedup:.2f}x")

    # ============================================================
    # Component 3: Layer LoRAs in attention (wq, wk, wv, wo)
    # These use SharedLinearWithLoRA so benefit from addmm fusion
    # ============================================================
    print("\n--- Attention Projection LoRAs (20 layers x 4 projections) ---")

    shared_wq = nn.Linear(dim, n_heads * head_dim, bias=False).to(device).to(dtype)

    old_attn_loras = nn.ModuleList([
        OldSharedLinearWithLoRA(shared_wq, dim, n_heads * head_dim, layer_lora_rank)
        for _ in range(n_layers * 4)
    ]).to(device).to(dtype)

    new_attn_loras = nn.ModuleList([
        SharedLinearWithLoRA(shared_wq, dim, n_heads * head_dim, layer_lora_rank)
        for _ in range(n_layers * 4)
    ]).to(device).to(dtype)

    def run_old_attn_loras():
        out = x
        for lora in old_attn_loras:
            out = lora(out)
        return out

    def run_new_attn_loras():
        out = x
        for lora in new_attn_loras:
            out = lora(out)
        return out

    t_old_attn = benchmark(run_old_attn_loras)
    t_new_attn = benchmark(run_new_attn_loras)
    attn_speedup = t_old_attn / t_new_attn

    print(f"Old attn LoRAs: {t_old_attn:.3f} ms")
    print(f"New attn LoRAs: {t_new_attn:.3f} ms")
    print(f"Speedup: {attn_speedup:.2f}x")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_old = t_old_ffn + t_old_head + t_old_attn
    total_new = t_new_ffn + t_new_head + t_new_attn
    overall_speedup = total_old / total_new

    print(f"\nTotal OLD time: {total_old:.3f} ms")
    print(f"Total NEW time: {total_new:.3f} ms")
    print(f"Overall speedup: {overall_speedup:.2f}x ({(overall_speedup-1)*100:.1f}% faster)")

    print(f"\nComponent breakdown:")
    print(f"  FFN:             {ffn_speedup:.2f}x ({t_old_ffn:.1f} -> {t_new_ffn:.1f} ms)")
    print(f"  Head LoRAs:      {head_speedup:.2f}x ({t_old_head:.1f} -> {t_new_head:.1f} ms)")
    print(f"  Attention LoRAs: {attn_speedup:.2f}x ({t_old_attn:.1f} -> {t_new_attn:.1f} ms)")

    # Note: This doesn't include attention compute (softmax, etc.) or other ops
    print(f"\nNote: Real model also has attention compute, embeddings, norms etc.")
    print(f"Expected E2E TPS improvement: ~10-15% (conservative estimate)")


if __name__ == "__main__":
    main()
