#!/usr/bin/env python3
"""Test repeated kernel calls."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import lorafusion_tk

print("Testing repeated kernel calls...")
sys.stdout.flush()

M, K, N, R = 128, 64, 128, 64
scale = 0.25

x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
S = torch.randn(M, R, device="cuda", dtype=torch.bfloat16)
B = torch.randn(N, R, device="cuda", dtype=torch.bfloat16)

print("Call 1...")
sys.stdout.flush()
try:
    y1 = lorafusion_tk.fused_forward(x, W, S, B, scale)
    print("  Kernel launched, synchronizing...")
    sys.stdout.flush()
    torch.cuda.synchronize()
    err = torch.cuda.get_last_error()
    if err:
        print(f"  CUDA error: {err}")
    print(f"  Done, shape: {y1.shape}")
except Exception as e:
    print(f"  Exception: {e}")
    import traceback
    traceback.print_exc()
sys.stdout.flush()

print("Call 2...")
sys.stdout.flush()
y2 = lorafusion_tk.fused_forward(x, W, S, B, scale)
torch.cuda.synchronize()
print(f"  Done, shape: {y2.shape}")
sys.stdout.flush()

print("Call 3...")
sys.stdout.flush()
y3 = lorafusion_tk.fused_forward(x, W, S, B, scale)
torch.cuda.synchronize()
print(f"  Done, shape: {y3.shape}")
sys.stdout.flush()

# Verify outputs are the same
diff_12 = (y1 - y2).abs().max().item()
diff_23 = (y2 - y3).abs().max().item()
print(f"Diff 1-2: {diff_12}, Diff 2-3: {diff_23}")
print("SUCCESS!")
