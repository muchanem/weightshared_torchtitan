#!/usr/bin/env python3
"""
Validate the theoretical ceiling for stacking optimizations and measure stacking overhead.
"""

import torch
import time


def benchmark(fn, iters=200, warmup=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / iters


def main():
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    batch, seq = 4, 2048
    M = batch * seq  # 8192
    dim = 1024
    rank = 192  # head offset rank
    n_offsets = 3  # wq, wk, wv

    x = torch.randn(M, dim, device=device, dtype=dtype)

    # Individual A matrices
    A_wq = torch.randn(rank, dim, device=device, dtype=dtype)
    A_wk = torch.randn(rank, dim, device=device, dtype=dtype)
    A_wv = torch.randn(rank, dim, device=device, dtype=dtype)

    # Pre-stacked A matrix
    A_stacked = torch.cat([A_wq, A_wk, A_wv], dim=0).contiguous()  # (576, 1024)

    print("=" * 70)
    print("STACKING CEILING VALIDATION")
    print("=" * 70)
    print(f"Shape: x ({M}, {dim})")
    print(f"LoRA A: ({rank}, {dim}) × {n_offsets}")
    print(f"Stacked A: ({rank * n_offsets}, {dim})")

    # ============================================================
    # Test 1: Baseline - 3 separate matmuls
    # ============================================================
    def baseline():
        h_wq = x @ A_wq.t()
        h_wk = x @ A_wk.t()
        h_wv = x @ A_wv.t()
        return h_wq, h_wk, h_wv

    t_baseline = benchmark(baseline)
    print(f"\n1. Baseline (3 separate matmuls): {t_baseline:.2f} us")

    # ============================================================
    # Test 2: Stacked (1 larger matmul + split)
    # ============================================================
    def stacked_split():
        h_all = x @ A_stacked.t()  # (M, 576)
        h_wq, h_wk, h_wv = h_all.split(rank, dim=-1)
        return h_wq, h_wk, h_wv

    t_stacked = benchmark(stacked_split)
    print(f"2. Stacked + split: {t_stacked:.2f} us")
    print(f"   Speedup: {t_baseline/t_stacked:.2f}x")

    # ============================================================
    # Test 3: Stacking overhead (if we had to stack at runtime)
    # ============================================================
    def stacked_with_runtime_cat():
        A_cat = torch.cat([A_wq, A_wk, A_wv], dim=0)
        h_all = x @ A_cat.t()
        h_wq, h_wk, h_wv = h_all.split(rank, dim=-1)
        return h_wq, h_wk, h_wv

    t_runtime_cat = benchmark(stacked_with_runtime_cat)
    print(f"3. Runtime cat + stacked: {t_runtime_cat:.2f} us")
    print(f"   Overhead from cat: {t_runtime_cat - t_stacked:.2f} us")

    # ============================================================
    # Test 4: Just the cat operation
    # ============================================================
    def just_cat():
        return torch.cat([A_wq, A_wk, A_wv], dim=0)

    t_cat = benchmark(just_cat)
    print(f"4. Just torch.cat: {t_cat:.2f} us")

    # ============================================================
    # Test 5: Full LoRA (A then B) comparison
    # ============================================================
    print("\n--- Full LoRA (A then B) ---")

    B_wq = torch.randn(dim, rank, device=device, dtype=dtype)
    B_wk = torch.randn(dim, rank, device=device, dtype=dtype)
    B_wv = torch.randn(dim, rank, device=device, dtype=dtype)

    def full_lora_separate():
        h_wq = x @ A_wq.t()
        h_wk = x @ A_wk.t()
        h_wv = x @ A_wv.t()
        out_wq = h_wq @ B_wq.t()
        out_wk = h_wk @ B_wk.t()
        out_wv = h_wv @ B_wv.t()
        return out_wq, out_wk, out_wv

    t_full_separate = benchmark(full_lora_separate)
    print(f"5. Full LoRA separate (6 matmuls): {t_full_separate:.2f} us")

    # Can we batch B as well? No - different inputs (h_wq, h_wk, h_wv)
    # But we CAN do A stacked, then B separate

    def full_lora_a_stacked():
        h_all = x @ A_stacked.t()
        h_wq, h_wk, h_wv = h_all.split(rank, dim=-1)
        out_wq = h_wq @ B_wq.t()
        out_wk = h_wk @ B_wk.t()
        out_wv = h_wv @ B_wv.t()
        return out_wq, out_wk, out_wv

    t_full_a_stacked = benchmark(full_lora_a_stacked)
    print(f"6. Full LoRA (A stacked, B separate): {t_full_a_stacked:.2f} us")
    print(f"   Speedup: {t_full_separate/t_full_a_stacked:.2f}x")

    # ============================================================
    # Test 6: What's the theoretical ceiling?
    # ============================================================
    print("\n--- Theoretical Ceiling Analysis ---")

    # Time for just the stacked A matmul (best case for A)
    def just_stacked_a():
        return x @ A_stacked.t()

    t_stacked_a_only = benchmark(just_stacked_a)

    # Time for 3 separate B matmuls (can't be stacked - different inputs)
    h_all = x @ A_stacked.t()
    h_wq, h_wk, h_wv = h_all.split(rank, dim=-1)

    def just_b_matmuls():
        out_wq = h_wq @ B_wq.t()
        out_wk = h_wk @ B_wk.t()
        out_wv = h_wv @ B_wv.t()
        return out_wq, out_wk, out_wv

    t_b_only = benchmark(just_b_matmuls)

    print(f"Stacked A matmul only: {t_stacked_a_only:.2f} us")
    print(f"3 separate B matmuls: {t_b_only:.2f} us")
    print(f"Theoretical minimum (A stacked + B separate): {t_stacked_a_only + t_b_only:.2f} us")
    print(f"Actual with stacking: {t_full_a_stacked:.2f} us")
    print(f"Overhead from split: {t_full_a_stacked - t_stacked_a_only - t_b_only:.2f} us")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    savings_from_a_stacking = t_full_separate - t_full_a_stacked
    pct_savings = savings_from_a_stacking / t_full_separate * 100

    print(f"\nFull LoRA (3 Q/K/V offsets):")
    print(f"  Separate: {t_full_separate:.2f} us")
    print(f"  A-stacked: {t_full_a_stacked:.2f} us")
    print(f"  Savings: {savings_from_a_stacking:.2f} us ({pct_savings:.1f}%)")

    # In context of full SharedLinearWithLoRA
    # From earlier: full operation is ~150 us
    full_op_time = 150  # us (from earlier benchmarks)
    pct_of_total = savings_from_a_stacking / full_op_time * 100
    print(f"\n  In context of full operation (~{full_op_time} us):")
    print(f"  Maximum possible savings: {pct_of_total:.1f}%")

    print("""
CONCLUSION:
-----------
1. Stacking A matrices gives modest speedup on LoRA-only path
2. B matrices CANNOT be stacked (different inputs h_wq, h_wk, h_wv)
3. Maximum achievable improvement is limited by B matmuls
4. A true fused kernel would need to:
   - Stream through stacked A and B matrices
   - Compute x @ A_i.T @ B_i.T without writing intermediate h_i
   - This requires custom CUDA/Triton kernel that we already tried (8x slower)
""")


if __name__ == "__main__":
    main()
