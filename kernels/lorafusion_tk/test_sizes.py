#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import lorafusion_tk

configs = [
    (128, 64, 128, 64),    # 1 block each - known working
    (256, 64, 128, 64),    # 2 M blocks
    (128, 128, 128, 64),   # 2 K iterations
    (128, 64, 256, 64),    # 2 N blocks
    (256, 128, 256, 64),   # Multiple
    (512, 256, 512, 64),   # Larger
]

for M, K, N, R in configs:
    print(f"Testing M={M}, K={K}, N={N}, R={R}...", end=" ", flush=True)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    S = torch.randn(M, R, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N, R, device="cuda", dtype=torch.bfloat16)

    try:
        y = lorafusion_tk.fused_forward(x, W, S, B, 0.25)
        torch.cuda.synchronize()
        print(f"OK (shape={y.shape})")
    except Exception as e:
        print(f"ERROR: {e}")
