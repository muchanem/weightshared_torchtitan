#!/usr/bin/env python3
"""Detailed correctness test for TK LoRAFusion kernel."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import lorafusion_tk

torch.manual_seed(42)

configs = [
    (128, 64, 128, 64),    # Minimal
    (256, 128, 256, 64),   # Small
    (512, 256, 512, 64),   # Medium
    (1024, 512, 1024, 64), # Larger
]

print("Correctness Test for TK LoRAFusion")
print("=" * 60)

for M, K, N, R in configs:
    print(f"\nConfig: M={M}, K={K}, N={N}, R={R}")

    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    S = torch.randn(M, R, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N, R, device="cuda", dtype=torch.bfloat16)
    scale = 0.25

    # Reference: Y = X @ W.T + S @ B.T * scale
    y_ref = F.linear(x, W) + F.linear(S, B) * scale

    # TK kernel
    y_tk = lorafusion_tk.fused_forward(x, W, S, B, scale)
    torch.cuda.synchronize()

    # Compare
    diff = (y_tk - y_ref).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Check individual components
    xw_ref = F.linear(x, W)
    sb_ref = F.linear(S, B) * scale

    print(f"  Reference X@W.T range: [{xw_ref.min().item():.2f}, {xw_ref.max().item():.2f}]")
    print(f"  Reference S@B.T*s range: [{sb_ref.min().item():.2f}, {sb_ref.max().item():.2f}]")
    print(f"  Reference Y range: [{y_ref.min().item():.2f}, {y_ref.max().item():.2f}]")
    print(f"  TK Y range: [{y_tk.min().item():.2f}, {y_tk.max().item():.2f}]")
    print(f"  Max diff: {max_diff:.4f}, Mean diff: {mean_diff:.6f}")

    # Sample values
    print(f"  Sample ref[0,:5]: {y_ref[0,:5].tolist()}")
    print(f"  Sample tk[0,:5]:  {y_tk[0,:5].tolist()}")

    if max_diff < 1.0:
        print("  STATUS: PASS")
    else:
        print("  STATUS: FAIL")
