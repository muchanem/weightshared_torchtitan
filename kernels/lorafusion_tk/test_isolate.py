#!/usr/bin/env python3
"""Isolate which part of the kernel is wrong."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import lorafusion_tk

torch.manual_seed(42)

M, K, N, R = 128, 64, 128, 64

x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
S = torch.randn(M, R, device="cuda", dtype=torch.bfloat16)
B = torch.randn(N, R, device="cuda", dtype=torch.bfloat16)

print("Test 1: scale=0 (only X@W.T)")
print("-" * 40)
y_ref = F.linear(x, W)
y_tk = lorafusion_tk.fused_forward(x, W, S, B, 0.0)
torch.cuda.synchronize()

diff = (y_tk - y_ref).abs()
print(f"Max diff: {diff.max().item():.4f}")
print(f"ref[0,:8]: {y_ref[0,:8].tolist()}")
print(f"tk[0,:8]:  {y_tk[0,:8].tolist()}")

# Check if columns are permuted
print("\nChecking if TK columns match different ref columns...")
for i in range(min(5, N)):
    tk_col = y_tk[:, i]
    for j in range(min(10, N)):
        ref_col = y_ref[:, j]
        if (tk_col - ref_col).abs().max().item() < 0.5:
            print(f"  TK col {i} matches ref col {j}")
            break

print("\n\nTest 2: X=0 (only S@B.T, but kernel still does X@W.T)")
print("-" * 40)
x_zero = torch.zeros_like(x)
y_ref2 = F.linear(S, B) * 0.25
y_tk2 = lorafusion_tk.fused_forward(x_zero, W, S, B, 0.25)
torch.cuda.synchronize()

diff2 = (y_tk2 - y_ref2).abs()
print(f"Max diff: {diff2.max().item():.4f}")
print(f"ref[0,:8]: {y_ref2[0,:8].tolist()}")
print(f"tk[0,:8]:  {y_tk2[0,:8].tolist()}")
