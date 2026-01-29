#!/usr/bin/env python3
"""
Comprehensive analysis of Hopper kernel options for LoRA fusion.

This script analyzes:
1. Whether CUTLASS two-op fusion (example 13) is viable for LoRA dimensions
2. Memory bandwidth vs compute characteristics
3. Arithmetic intensity for different fusion strategies
4. Comparison with LoRAFusion approach
"""

import torch
import torch.utils.benchmark as benchmark
from dataclasses import dataclass
from typing import List, Tuple
import math

# H200 NVL specifications
H200_MEMORY_BW = 4.8e12  # 4.8 TB/s HBM3e bandwidth
H200_PEAK_FLOPS_BF16 = 989.4e12  # 989.4 TFLOPS BF16 tensor cores
H200_RIDGE_POINT = H200_PEAK_FLOPS_BF16 / H200_MEMORY_BW  # FLOP/byte

@dataclass
class LoRADimensions:
    """Dimensions for a LoRA linear layer."""
    batch_seq: int  # M dimension (batch * seq_len)
    in_dim: int     # K dimension (input features)
    out_dim: int    # N dimension (output features)
    rank: int       # R dimension (LoRA rank)

    def __str__(self):
        return f"M={self.batch_seq}, K={self.in_dim}, N={self.out_dim}, R={self.rank}"

def calculate_arithmetic_intensity(dims: LoRADimensions) -> dict:
    """
    Calculate arithmetic intensity for different operations and fusion strategies.

    Arithmetic Intensity = FLOPs / Bytes moved
    Higher AI means more compute-bound (good for GPU utilization).
    """
    M, K, N, R = dims.batch_seq, dims.in_dim, dims.out_dim, dims.rank
    bytes_per_elem = 2  # bfloat16

    results = {}

    # Base GEMM: X @ W.T -> (M, K) @ (K, N) = (M, N)
    # FLOPs: 2*M*K*N
    # Bytes: read X (M*K) + read W (K*N) + write Y (M*N)
    base_flops = 2 * M * K * N
    base_bytes = (M * K + K * N + M * N) * bytes_per_elem
    results['base_gemm'] = {
        'flops': base_flops,
        'bytes': base_bytes,
        'ai': base_flops / base_bytes,
        'bound': 'compute' if base_flops / base_bytes > H200_RIDGE_POINT else 'memory'
    }

    # LoRA A: X @ A.T -> (M, K) @ (K, R) = (M, R)
    lora_a_flops = 2 * M * K * R
    lora_a_bytes = (M * K + K * R + M * R) * bytes_per_elem
    results['lora_a'] = {
        'flops': lora_a_flops,
        'bytes': lora_a_bytes,
        'ai': lora_a_flops / lora_a_bytes,
        'bound': 'compute' if lora_a_flops / lora_a_bytes > H200_RIDGE_POINT else 'memory'
    }

    # LoRA B: S @ B.T -> (M, R) @ (R, N) = (M, N)
    lora_b_flops = 2 * M * R * N
    lora_b_bytes = (M * R + R * N + M * N) * bytes_per_elem
    results['lora_b'] = {
        'flops': lora_b_flops,
        'bytes': lora_b_bytes,
        'ai': lora_b_flops / lora_b_bytes,
        'bound': 'compute' if lora_b_flops / lora_b_bytes > H200_RIDGE_POINT else 'memory'
    }

    # Add operation: base + lora_out -> (M, N) + (M, N) = (M, N)
    add_flops = M * N  # One add per element
    add_bytes = (M * N + M * N + M * N) * bytes_per_elem  # Read two, write one
    results['add'] = {
        'flops': add_flops,
        'bytes': add_bytes,
        'ai': add_flops / add_bytes,
        'bound': 'memory'  # Always memory bound
    }

    # Strategy 1: Unfused (current) - 4 kernels
    # Read: X twice (base + lora_a), W, A, B, lora_out (for add)
    # Write: base_out, S (intermediate), lora_out, final_out
    unfused_flops = base_flops + lora_a_flops + lora_b_flops + add_flops
    unfused_bytes = (
        M * K * 2 +  # X read twice (base + lora_a)
        K * N +      # W read
        K * R +      # A read
        M * R +      # S write
        M * R +      # S read (for lora_b)
        R * N +      # B read
        M * N +      # base_out write
        M * N +      # lora_out write
        M * N * 2 +  # add reads
        M * N        # final write
    ) * bytes_per_elem
    results['unfused'] = {
        'flops': unfused_flops,
        'bytes': unfused_bytes,
        'ai': unfused_flops / unfused_bytes,
        'bound': 'compute' if unfused_flops / unfused_bytes > H200_RIDGE_POINT else 'memory'
    }

    # Strategy 2: LoRAFusion style - fuse (base GEMM + lora_b @ S) into one kernel
    # The S intermediate stays in registers, saving M*R read/write
    # Read: X, W, S (pre-computed), B
    # Write: final_out
    lorafusion_flops = base_flops + lora_b_flops + add_flops  # A is done separately
    lorafusion_bytes = (
        M * K +      # X read
        K * N +      # W read
        M * R +      # S read (pre-computed in separate kernel)
        R * N +      # B read
        M * N        # final write
    ) * bytes_per_elem
    results['lorafusion_fwd'] = {
        'flops': lorafusion_flops,
        'bytes': lorafusion_bytes,
        'ai': lorafusion_flops / lorafusion_bytes,
        'bound': 'compute' if lorafusion_flops / lorafusion_bytes > H200_RIDGE_POINT else 'memory'
    }

    # Strategy 3: Two-op fusion (A@B in registers)
    # Fuse lora_a and lora_b, keep S in registers
    # Read: X, A, B
    # Write: lora_out
    two_op_lora_flops = lora_a_flops + lora_b_flops
    two_op_lora_bytes = (
        M * K +      # X read
        K * R +      # A read
        R * N +      # B read
        M * N        # lora_out write
    ) * bytes_per_elem
    results['two_op_lora'] = {
        'flops': two_op_lora_flops,
        'bytes': two_op_lora_bytes,
        'ai': two_op_lora_flops / two_op_lora_bytes,
        'bound': 'compute' if two_op_lora_flops / two_op_lora_bytes > H200_RIDGE_POINT else 'memory'
    }

    # Strategy 4: Full fusion (base + two-op lora) - theoretical maximum
    # Everything in one kernel, no intermediates to memory
    # Read: X, W, A, B
    # Write: final_out
    full_fusion_flops = unfused_flops
    full_fusion_bytes = (
        M * K +      # X read (once!)
        K * N +      # W read
        K * R +      # A read
        R * N +      # B read
        M * N        # final write
    ) * bytes_per_elem
    results['full_fusion'] = {
        'flops': full_fusion_flops,
        'bytes': full_fusion_bytes,
        'ai': full_fusion_flops / full_fusion_bytes,
        'bound': 'compute' if full_fusion_flops / full_fusion_bytes > H200_RIDGE_POINT else 'memory'
    }

    return results


def check_two_op_fusion_viability(dims: LoRADimensions) -> dict:
    """
    Check if CUTLASS two-op fusion (example 13) constraints are satisfied.

    Constraints from CUTLASS example 13:
    1. thread_block_tile_N = problem_N (for RF-resident) OR
    2. Use shared memory staging (relaxed constraint)
    3. Same number of CTAs for both GEMMs
    """
    M, K, N, R = dims.batch_seq, dims.in_dim, dims.out_dim, dims.rank

    # GEMM1: X @ A.T -> (M, K) @ (K, R) = (M, R)
    # GEMM2: S @ B.T -> (M, R) @ (R, N) = (M, N)

    results = {
        'gemm1_dims': f"({M}, {K}) @ ({K}, {R}) = ({M}, {R})",
        'gemm2_dims': f"({M}, {R}) @ ({R}, {N}) = ({M}, {N})",
    }

    # Check intermediate dimension compatibility
    # GEMM1 output N = R, GEMM2 input K = R (they match!)
    results['intermediate_match'] = R == R  # Always true, sanity check

    # Check if R (rank) can fit in a tile for RF-resident fusion
    typical_tile_sizes = [64, 128, 256]
    for tile_n in typical_tile_sizes:
        results[f'rf_resident_tile_{tile_n}'] = R <= tile_n

    # For shared memory staging, check register pressure
    # Accumulator for (tile_M, tile_N) needs tile_M * tile_N * 4 bytes (fp32)
    for tile_m, tile_n in [(64, 64), (128, 128), (128, 256)]:
        accum_regs = tile_m * tile_n  # Each fp32 accumulator = 1 register
        results[f'accum_regs_{tile_m}x{tile_n}'] = accum_regs
        # Hopper has 65536 registers per SM, ~256 registers per thread for good occupancy
        results[f'rf_viable_{tile_m}x{tile_n}'] = accum_regs <= 256 * 32  # 32 warps

    # Check if problem sizes allow efficient tiling
    results['m_alignment_128'] = M % 128 == 0
    results['k_alignment_64'] = K % 64 == 0
    results['n_alignment_128'] = N % 128 == 0
    results['r_alignment_64'] = R % 64 == 0

    return results


def check_batched_grouped_gemm_viability(dims: LoRADimensions, num_loras: int = 3) -> dict:
    """
    Check if batched/grouped GEMM (examples 56/57) can help.

    For attention with 3 LoRA modules (wq, wk, wv):
    - Stack A matrices: (3, R, K)
    - But B matrices have DIFFERENT inputs (h_wq, h_wk, h_wv)!
    """
    M, K, N, R = dims.batch_seq, dims.in_dim, dims.out_dim, dims.rank

    results = {
        'num_loras': num_loras,
    }

    # Can we batch the A matmuls? Yes - same input X
    # X @ [A_wq; A_wk; A_wv].T = X @ (3*R, K).T = (M, 3*R)
    results['a_batchable'] = True
    results['batched_a_dims'] = f"({M}, {K}) @ ({K}, {num_loras * R}) = ({M}, {num_loras * R})"

    # Can we batch the B matmuls? NO - different inputs!
    # h_wq @ B_wq.T, h_wk @ B_wk.T, h_wv @ B_wv.T
    # Each h_* is different (the intermediate from A matmul)
    results['b_batchable'] = False
    results['b_reason'] = "B matrices have different inputs (h_wq, h_wk, h_wv from A matmuls)"

    # Grouped GEMM could work but...
    # - Each group has same M but different K (=R) and N (=output_dim)
    # - Groups don't share data, so no memory savings
    results['grouped_gemm_applicable'] = True
    results['grouped_gemm_benefit'] = "Reduces kernel launch overhead only, no memory savings"

    # Memory traffic comparison
    # Separate: 3 * (M*K + K*R + M*R + M*R + R*N + M*N)
    # A-batched: 1 * (M*K + K*3R + M*3R) + 3 * (M*R + R*N + M*N)
    separate_bytes = 3 * (M*K + K*R + M*R + M*R + R*N + M*N) * 2
    batched_a_bytes = (M*K + K*num_loras*R + M*num_loras*R +
                       3 * (M*R + R*N + M*N)) * 2

    results['separate_traffic_mb'] = separate_bytes / 1e6
    results['batched_a_traffic_mb'] = batched_a_bytes / 1e6
    results['a_batching_savings_pct'] = (1 - batched_a_bytes / separate_bytes) * 100

    return results


def benchmark_operations(dims: LoRADimensions, warmup: int = 10, iters: int = 100):
    """Benchmark actual kernel performance."""
    M, K, N, R = dims.batch_seq, dims.in_dim, dims.out_dim, dims.rank
    device = 'cuda'
    dtype = torch.bfloat16

    # Allocate tensors
    X = torch.randn(M, K, device=device, dtype=dtype)
    W = torch.randn(N, K, device=device, dtype=dtype)  # Transposed storage
    A = torch.randn(R, K, device=device, dtype=dtype)
    B = torch.randn(N, R, device=device, dtype=dtype)
    S = torch.randn(M, R, device=device, dtype=dtype)

    results = {}

    # Warmup
    for _ in range(warmup):
        _ = X @ W.T
    torch.cuda.synchronize()

    # Base GEMM
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        base_out = X @ W.T
    end.record()
    torch.cuda.synchronize()
    results['base_gemm_us'] = start.elapsed_time(end) * 1000 / iters

    # LoRA A
    start.record()
    for _ in range(iters):
        s = X @ A.T
    end.record()
    torch.cuda.synchronize()
    results['lora_a_us'] = start.elapsed_time(end) * 1000 / iters

    # LoRA B
    start.record()
    for _ in range(iters):
        lora_out = S @ B.T
    end.record()
    torch.cuda.synchronize()
    results['lora_b_us'] = start.elapsed_time(end) * 1000 / iters

    # Add
    lora_out = S @ B.T
    start.record()
    for _ in range(iters):
        final = base_out + lora_out
    end.record()
    torch.cuda.synchronize()
    results['add_us'] = start.elapsed_time(end) * 1000 / iters

    # Full unfused
    start.record()
    for _ in range(iters):
        base_out = X @ W.T
        s = X @ A.T
        lora_out = s @ B.T
        final = base_out + lora_out
    end.record()
    torch.cuda.synchronize()
    results['unfused_total_us'] = start.elapsed_time(end) * 1000 / iters

    # LoRAFusion style (S pre-computed, fuse base + S@B + add)
    # We can't actually fuse without a custom kernel, but we can measure
    # the theoretical speedup from eliminating intermediate writes
    start.record()
    for _ in range(iters):
        # This is still unfused, but gives us baseline
        final = X @ W.T + S @ B.T
    end.record()
    torch.cuda.synchronize()
    results['lorafusion_style_us'] = start.elapsed_time(end) * 1000 / iters

    return results


def print_analysis(dims: LoRADimensions):
    """Print comprehensive analysis for given dimensions."""
    print("=" * 80)
    print(f"ANALYSIS FOR: {dims}")
    print("=" * 80)

    # Arithmetic intensity analysis
    print("\n1. ARITHMETIC INTENSITY ANALYSIS")
    print("-" * 60)
    ai_results = calculate_arithmetic_intensity(dims)
    print(f"H200 Ridge Point: {H200_RIDGE_POINT:.1f} FLOP/byte")
    print(f"Operations below ridge point are memory-bound\n")

    for op, data in ai_results.items():
        bound_str = "COMPUTE" if data['bound'] == 'compute' else "MEMORY"
        print(f"{op:20s}: AI={data['ai']:8.1f} FLOP/byte, "
              f"FLOPs={data['flops']/1e9:8.2f}G, "
              f"Bytes={data['bytes']/1e6:8.2f}MB, "
              f"[{bound_str}]")

    # Memory savings from fusion
    print(f"\nMemory Traffic Reduction:")
    print(f"  Unfused:       {ai_results['unfused']['bytes']/1e6:.2f} MB")
    print(f"  LoRAFusion:    {ai_results['lorafusion_fwd']['bytes']/1e6:.2f} MB "
          f"({(1 - ai_results['lorafusion_fwd']['bytes']/ai_results['unfused']['bytes'])*100:.1f}% reduction)")
    print(f"  Two-op LoRA:   {ai_results['two_op_lora']['bytes']/1e6:.2f} MB")
    print(f"  Full Fusion:   {ai_results['full_fusion']['bytes']/1e6:.2f} MB "
          f"({(1 - ai_results['full_fusion']['bytes']/ai_results['unfused']['bytes'])*100:.1f}% reduction)")

    # Two-op fusion viability
    print("\n2. CUTLASS TWO-OP FUSION VIABILITY (Example 13)")
    print("-" * 60)
    two_op = check_two_op_fusion_viability(dims)
    print(f"GEMM1 (LoRA A): {two_op['gemm1_dims']}")
    print(f"GEMM2 (LoRA B): {two_op['gemm2_dims']}")
    print(f"\nConstraint checks:")
    print(f"  Intermediate dimension match: {'YES' if two_op['intermediate_match'] else 'NO'}")
    print(f"  RF-resident with tile_N=64:   {'YES' if two_op['rf_resident_tile_64'] else 'NO'} (R={dims.rank})")
    print(f"  RF-resident with tile_N=128:  {'YES' if two_op['rf_resident_tile_128'] else 'NO'}")
    print(f"  RF-resident with tile_N=256:  {'YES' if two_op['rf_resident_tile_256'] else 'NO'}")
    print(f"  M alignment (128): {'YES' if two_op['m_alignment_128'] else 'NO'}")
    print(f"  K alignment (64):  {'YES' if two_op['k_alignment_64'] else 'NO'}")
    print(f"  R alignment (64):  {'YES' if two_op['r_alignment_64'] else 'NO'}")

    # Batched/Grouped GEMM
    print("\n3. BATCHED/GROUPED GEMM ANALYSIS (Examples 56/57)")
    print("-" * 60)
    batched = check_batched_grouped_gemm_viability(dims, num_loras=3)
    print(f"For {batched['num_loras']} LoRA modules (wq, wk, wv):")
    print(f"  A matrices batchable: {'YES' if batched['a_batchable'] else 'NO'}")
    print(f"    Batched A dims: {batched['batched_a_dims']}")
    print(f"  B matrices batchable: {'YES' if batched['b_batchable'] else 'NO'}")
    print(f"    Reason: {batched['b_reason']}")
    print(f"\nMemory traffic:")
    print(f"  Separate:   {batched['separate_traffic_mb']:.2f} MB")
    print(f"  A-batched:  {batched['batched_a_traffic_mb']:.2f} MB ({batched['a_batching_savings_pct']:.1f}% savings)")

    # Benchmark
    print("\n4. ACTUAL KERNEL TIMINGS")
    print("-" * 60)
    bench = benchmark_operations(dims)
    for op, time_us in bench.items():
        print(f"  {op:25s}: {time_us:8.2f} us")

    print(f"\nKernel overhead:")
    individual_sum = bench['base_gemm_us'] + bench['lora_a_us'] + bench['lora_b_us'] + bench['add_us']
    print(f"  Sum of individual:  {individual_sum:.2f} us")
    print(f"  Full unfused:       {bench['unfused_total_us']:.2f} us")
    print(f"  Difference:         {individual_sum - bench['unfused_total_us']:.2f} us "
          f"({(individual_sum/bench['unfused_total_us'] - 1)*100:.1f}% overhead from measurement)")


def main():
    print("=" * 80)
    print("HOPPER KERNEL OPTIONS FOR LORA FUSION - COMPREHENSIVE ANALYSIS")
    print("=" * 80)

    # Our actual dimensions from the weight-shared model
    # batch=4, seq=2048, dim=1024, hidden=3072, layer_lora_rank=256, head_lora_rank=192

    configs = [
        # FFN projection (dim -> hidden)
        LoRADimensions(batch_seq=4*2048, in_dim=1024, out_dim=3072, rank=256),
        # FFN down projection (hidden -> dim)
        LoRADimensions(batch_seq=4*2048, in_dim=3072, out_dim=1024, rank=256),
        # Attention projections (dim -> dim)
        LoRADimensions(batch_seq=4*2048, in_dim=1024, out_dim=1024, rank=256),
        # Head offset LoRA (smaller rank)
        LoRADimensions(batch_seq=4*2048, in_dim=1024, out_dim=1024, rank=192),
    ]

    for config in configs:
        print_analysis(config)
        print("\n")

    # Key conclusions
    print("=" * 80)
    print("KEY CONCLUSIONS")
    print("=" * 80)
    print("""
1. TWO-OP FUSION (CUTLASS Example 13) IS VIABLE FOR LORA:
   - The LoRA rank (192-256) fits within typical tile sizes (64-256)
   - RF-resident fusion is possible with tile_N >= rank
   - The intermediate S tensor (M, R) stays in registers

2. THE CTA COUNT MISMATCH PROBLEM FROM THE FORUM DOES NOT APPLY:
   - Forum issue: GEMM1 produces (M, 256), GEMM2 needs (M, 256) @ (256, COL)
   - Our case: GEMM1 produces (M, R), GEMM2 needs (M, R) @ (R, N)
   - The intermediate dimension R matches, and R is small enough for RF-resident

3. B MATRICES CANNOT BE BATCHED (critical limitation):
   - The B matmuls have different inputs (h_wq, h_wk, h_wv)
   - Grouped GEMM only reduces launch overhead, not memory traffic
   - A-matrix batching provides ~6-8% memory savings max

4. RECOMMENDED APPROACH:
   Phase 1: Two-op fusion for LoRA (A @ B in one kernel)
     - Uses CUTLASS example 13 as template
     - Port to Hopper with TMA/warp specialization from examples 48/49
     - Expected benefit: ~2x reduction in LoRA memory traffic

   Phase 2: Fuse with base GEMM (if Phase 1 succeeds)
     - Add the base GEMM output to the fused LoRA result in epilogue
     - Uses EVT (Epilogue Visitor Tree) for the add
     - Expected benefit: Eliminate separate add kernel

   Phase 3: Consider stream parallelism (advanced)
     - Run base GEMM and two-op LoRA in parallel (independent)
     - Merge results at the end
     - Requires warp specialization or TBC clusters

5. WHAT WE CAN LEARN FROM LORAFUSION:
   - They fuse (base + S@B) AFTER computing S separately
   - We want to also fuse (A@B) to eliminate S write entirely
   - Their TMA usage for X/W is directly applicable
   - Key difference: They have frozen W, we need gradients for all weights
""")


if __name__ == "__main__":
    main()
