#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import lorafusion_tk

M, K, N, R = 8192, 1024, 1024, 64
print(f"Testing single call: M={M}, K={K}, N={N}, R={R}")
sys.stdout.flush()

x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
S = torch.randn(M, R, device="cuda", dtype=torch.bfloat16)
B = torch.randn(N, R, device="cuda", dtype=torch.bfloat16)

print("Calling kernel...")
sys.stdout.flush()
y = lorafusion_tk.fused_forward(x, W, S, B, 0.25)
torch.cuda.synchronize()
print(f"Done! Shape: {y.shape}")
