#!/usr/bin/env python3
"""Check the source of the difference."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import lorafusion_tk

torch.manual_seed(42)

M, K, N, R = 8192, 1024, 1024, 64
scale = 0.25

x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
A = torch.randn(R, K, device="cuda", dtype=torch.bfloat16)
B = torch.randn(N, R, device="cuda", dtype=torch.bfloat16)

# Compute S both ways
S_cublas = F.linear(x, A)

# Full computations
y_unfused = F.linear(x, W) + F.linear(S_cublas, B) * scale
y_lorafusion = lorafusion_tk.fused_forward(x, W, S_cublas, B, scale)
torch.cuda.synchronize()

diff = (y_unfused - y_lorafusion).abs()
print(f"Full result comparison:")
print(f"  Max diff: {diff.max().item():.4f}")
print(f"  Mean diff: {diff.mean().item():.6f}")
print(f"  Output range: [{y_unfused.min().item():.2f}, {y_unfused.max().item():.2f}]")
print(f"  Relative max diff: {diff.max().item() / y_unfused.abs().max().item() * 100:.4f}%")

# Check how many elements have large diff
large_diff = (diff > 1.0).sum().item()
print(f"  Elements with diff > 1.0: {large_diff} / {diff.numel()} ({large_diff/diff.numel()*100:.4f}%)")
