#!/usr/bin/env python3
"""Simple benchmark for TK LoRAFusion kernel."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import lorafusion_tk

# Must satisfy: M % 128 == 0, K % 64 == 0, N % 128 == 0, R == 64
M, K, N, R = 8192, 1024, 1024, 64
scale = 0.25

print(f"Config: M={M}, K={K}, N={N}, R={R}")

x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
S = torch.randn(M, R, device="cuda", dtype=torch.bfloat16)
B = torch.randn(N, R, device="cuda", dtype=torch.bfloat16)

def unfused():
    return F.linear(x, W) + F.linear(S, B) * scale

def tk_fused():
    return lorafusion_tk.fused_forward(x, W, S, B, scale)

# Warmup
print("Warming up...")
for _ in range(5):
    _ = unfused()
    _ = tk_fused()
torch.cuda.synchronize()

# Benchmark
iters = 20
print(f"Benchmarking ({iters} iterations)...")

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for _ in range(iters):
    _ = unfused()
end.record()
torch.cuda.synchronize()
unfused_us = start.elapsed_time(end) / iters * 1000

start.record()
for _ in range(iters):
    _ = tk_fused()
end.record()
torch.cuda.synchronize()
tk_us = start.elapsed_time(end) / iters * 1000

print(f"cuBLAS unfused: {unfused_us:.1f} us")
print(f"TK fused:       {tk_us:.1f} us")
print(f"Speedup:        {unfused_us/tk_us:.2f}x")
