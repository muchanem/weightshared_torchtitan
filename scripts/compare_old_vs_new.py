#!/usr/bin/env python3
"""
Quick comparison of old vs new SharedLinearWithLoRA implementation.
"""

import torch
import torch.nn as nn
import time


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
    """Original implementation without addmm fusion."""
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


def benchmark(name, fn, iterations=1000, warmup=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()

    return (time.perf_counter() - start) * 1000 / iterations


def main():
    from torchtitan.models.qwen3.model.weight_sharing import SharedLinearWithLoRA

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    batch, seq, dim = 4, 2048, 1024
    hidden_dim = 2816
    lora_rank = 256

    shared_linear = nn.Linear(dim, hidden_dim, bias=False).to(device).to(dtype)

    old = OldSharedLinearWithLoRA(shared_linear, dim, hidden_dim, lora_rank).to(device).to(dtype)
    new = SharedLinearWithLoRA(shared_linear, dim, hidden_dim, lora_rank).to(device).to(dtype)

    # Copy weights
    new.lora.w_a.weight.data.copy_(old.lora.w_a.weight.data)
    new.lora.w_b.weight.data.copy_(old.lora.w_b.weight.data)

    x = torch.randn(batch, seq, dim, device=device, dtype=dtype)

    t_old = benchmark("Old", lambda: old(x))
    t_new = benchmark("New", lambda: new(x))

    print(f"Old implementation: {t_old:.4f} ms/iter")
    print(f"New implementation: {t_new:.4f} ms/iter")
    print(f"Speedup: {t_old/t_new:.2f}x ({(t_old/t_new - 1)*100:.1f}% faster)")


if __name__ == "__main__":
    main()
