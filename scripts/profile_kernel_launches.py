#!/usr/bin/env python3
"""
Profile exact kernel launches for sequential vs grouped LoRA.
Goal: Understand WHY grouped approach isn't reducing kernel count.
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, "/local/scratch/muchane_680186/weightshared_torchtitan")


def profile_operation(name, fn, warmup=3, iterations=1):
    """Profile a single operation and show all kernel launches."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    print(f"\n{'='*60}")
    print(f"Profiling: {name}")
    print('='*60)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(iterations):
            fn()
        torch.cuda.synchronize()

    events = prof.key_averages()

    # Count kernel types
    gemm_count = 0
    copy_count = 0
    other_count = 0

    print("\nAll CUDA kernels:")
    for e in sorted(events, key=lambda x: -x.self_device_time_total):
        key_lower = e.key.lower()
        if 'gemm' in key_lower or 'mm' in key_lower or 'matmul' in key_lower:
            kernel_type = "GEMM"
            gemm_count += e.count
        elif 'copy' in key_lower or 'memcpy' in key_lower:
            kernel_type = "COPY"
            copy_count += e.count
        else:
            kernel_type = "OTHER"
            other_count += e.count

        print(f"  [{kernel_type:5}] {e.key}: count={e.count}, cuda_time={e.self_device_time_total/1000:.3f}ms")

    print(f"\nSummary: GEMM={gemm_count}, COPY={copy_count}, OTHER={other_count}, TOTAL={gemm_count+copy_count+other_count}")
    return events


def main():
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    # Typical sizes
    batch, seq, dim = 4, 2048, 1024
    hidden_dim = 2816
    rank = 256
    M = batch * seq  # 8192

    # Create tensors
    x = torch.randn(M, dim, device=device, dtype=dtype)

    # LoRA weights for w1 and w3
    w1_a = torch.randn(rank, dim, device=device, dtype=dtype)
    w1_b = torch.randn(hidden_dim, rank, device=device, dtype=dtype)
    w3_a = torch.randn(rank, dim, device=device, dtype=dtype)
    w3_b = torch.randn(hidden_dim, rank, device=device, dtype=dtype)

    # =========================================
    # Test 1: Sequential matmuls (baseline)
    # =========================================
    def sequential_lora():
        h1 = x @ w1_a.t()
        out1 = h1 @ w1_b.t()
        h3 = x @ w3_a.t()
        out3 = h3 @ w3_b.t()
        return out1, out3

    profile_operation("Sequential LoRA (4 matmuls)", sequential_lora)

    # =========================================
    # Test 2: Current grouped approach (with runtime stack)
    # =========================================
    def grouped_with_stack():
        # This is what our current code does
        w_a_stack = torch.stack([w1_a, w3_a], dim=0)  # COPY!
        w_b_stack = torch.stack([w1_b, w3_b], dim=0)  # COPY!
        x_exp = x.unsqueeze(0).expand(2, -1, -1)

        h = torch.bmm(x_exp, w_a_stack.transpose(-2, -1))
        out = torch.bmm(h, w_b_stack.transpose(-2, -1))
        return out[0], out[1]

    profile_operation("Grouped with runtime torch.stack()", grouped_with_stack)

    # =========================================
    # Test 3: Pre-stacked weights (no runtime stack)
    # =========================================
    # Pre-stack weights ONCE (simulates init-time stacking)
    w_a_prestacked = torch.stack([w1_a, w3_a], dim=0).contiguous()
    w_b_prestacked = torch.stack([w1_b, w3_b], dim=0).contiguous()

    def grouped_prestacked():
        # No runtime stack - weights already batched
        x_exp = x.unsqueeze(0).expand(2, -1, -1)
        h = torch.bmm(x_exp, w_a_prestacked.transpose(-2, -1))
        out = torch.bmm(h, w_b_prestacked.transpose(-2, -1))
        return out[0], out[1]

    profile_operation("Grouped with PRE-stacked weights", grouped_prestacked)

    # =========================================
    # Test 4: Contiguous expanded x (avoid strided access)
    # =========================================
    def grouped_contiguous():
        x_exp = x.unsqueeze(0).expand(2, -1, -1).contiguous()  # Make contiguous
        h = torch.bmm(x_exp, w_a_prestacked.transpose(-2, -1))
        out = torch.bmm(h, w_b_prestacked.transpose(-2, -1))
        return out[0], out[1]

    profile_operation("Grouped with contiguous x (pre-stacked weights)", grouped_contiguous)

    # =========================================
    # Test 5: torch._grouped_mm (MoE-style)
    # =========================================
    def grouped_mm_style():
        # Repeat x for grouped_mm format
        x_repeated = x.repeat(2, 1)  # (2*M, dim)
        offsets = torch.tensor([M, 2*M], dtype=torch.int32, device=device)

        h = torch._grouped_mm(x_repeated, w_a_prestacked.transpose(-2, -1), offs=offsets)
        out = torch._grouped_mm(h, w_b_prestacked.transpose(-2, -1), offs=offsets)

        return out[:M], out[M:]

    profile_operation("torch._grouped_mm with pre-stacked weights", grouped_mm_style)

    # =========================================
    # Timing comparison
    # =========================================
    print("\n" + "="*60)
    print("TIMING COMPARISON (100 iterations)")
    print("="*60)

    import time

    def benchmark(name, fn, iterations=100):
        # Warmup
        for _ in range(10):
            fn()
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        print(f"{name}: {elapsed*1000/iterations:.3f} ms/iter")
        return elapsed / iterations

    t_seq = benchmark("Sequential (4 matmuls)", sequential_lora)
    t_stack = benchmark("Grouped + runtime stack", grouped_with_stack)
    t_pre = benchmark("Grouped + pre-stacked", grouped_prestacked)
    t_contig = benchmark("Grouped + contiguous x", grouped_contiguous)
    t_gmm = benchmark("torch._grouped_mm", grouped_mm_style)

    print(f"\nSpeedup vs sequential:")
    print(f"  Runtime stack: {t_seq/t_stack:.2f}x")
    print(f"  Pre-stacked:   {t_seq/t_pre:.2f}x")
    print(f"  Contiguous x:  {t_seq/t_contig:.2f}x")
    print(f"  grouped_mm:    {t_seq/t_gmm:.2f}x")


if __name__ == "__main__":
    main()
