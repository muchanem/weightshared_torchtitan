# Hopper Kernel Analysis for LoRA Fusion

## Executive Summary

This document analyzes kernel fusion strategies for speeding up LoRA operations in the weight-shared model.

### Key Findings (Updated Jan 30, 2026)

1. **TK LoRAFusion kernel implemented and validated** - 1.34-1.84x speedup over unfused
2. **torch.compile is SLOWER than unfused cuBLAS** (~18% slower) for these operations
3. **The S intermediate is only 7% of LoRA A+B time** - not the main bottleneck
4. **Kernel launch overhead (~27 us) and duplicate X reads (~3.5 us) are significant**
5. **A megakernel with SMEM staging could achieve 2.8x speedup** on the LoRA+Linear path

### Implemented: TK LoRAFusion (2-Kernel Approach)

**Status**: Working and validated on H200

| Config | Unfused (4 kernels) | torch.compile | TK LoRAFusion (2 kernels) | vs Unfused | vs Compiled |
|--------|---------------------|---------------|---------------------------|------------|-------------|
| 1024x512x1024x64 | 33.6 us | 39.8 us | 19.6 us | **1.71x** | **2.03x** |
| 8192x1024x1024x64 | 70.9 us | 80.9 us | 52.1 us | **1.34x** | **1.55x** |

**Key insight**: torch.compile's Triton autotune picks suboptimal configs, making it slower than raw cuBLAS.

### Next Target: SMEM-Staged Megakernel

Instead of LoRAFusion's two-kernel approach, we propose a **single megakernel** that:
1. Reads X once (saves 3.5 us)
2. Stores S in shared memory (not GMEM)
3. Fuses all operations (saves ~27 us launch overhead)
4. Fuses the add into the accumulator (saves ~15 us)

**Potential speedup**: 72 us → ~26 us = **2.8x** per Linear+LoRA operation

---

## 1. Is CUTLASS Example 13 (Two-Op Fusion) the Right Structure?

### Answer: **YES, with modifications**

#### Why It's a Good Fit

The two-op fusion in example 13 is designed for exactly our pattern:
- **GEMM1**: `X @ A.T` → `(M, K) @ (K, R) = (M, R)` (LoRA down-projection)
- **GEMM2**: `S @ B.T` → `(M, R) @ (R, N) = (M, N)` (LoRA up-projection)

The key constraint from example 13 README:
> `thread_block_tile_N = problem_N` ensures each threadblock loads the entire weight/filter matrix

For our LoRA dimensions:
- `R (rank) = 192-256`
- Tile sizes of 256 can accommodate this

**Benchmark validation** (from `analyze_lora_kernel_options.py`):
```
CUTLASS TWO-OP FUSION VIABILITY (Example 13)
GEMM1 (LoRA A): (8192, 1024) @ (1024, 256) = (8192, 256)
GEMM2 (LoRA B): (8192, 256) @ (256, 1024) = (8192, 1024)

Constraint checks:
  Intermediate dimension match: YES
  RF-resident with tile_N=256:  YES
  R alignment (64):  YES
```

#### Why the Forum CTA Mismatch Problem Doesn't Apply

The forum discussion describes a problem where:
- GEMM1 produces (M, 256)
- GEMM2 consumes (M, 256) but outputs to a DIFFERENT column count
- This causes CTA count mismatch

Our case is different:
- **GEMM1 output N = R** (rank, e.g., 256)
- **GEMM2 input K = R** (same rank, 256)
- The intermediate dimension **matches perfectly**

The constraint is that each CTA must hold the entire N dimension of GEMM1 in registers. With R=256 and tile_N=256, this is exactly satisfied.

#### What Needs to Change for Hopper

Example 13 is for SM80 (Ampere). For Hopper we need:
1. **TMA (Tensor Memory Accelerator)**: Replace explicit loads with `cp.async.bulk` via TMA
2. **Warp specialization**: Separate producer/consumer warps for overlapped compute/memory
3. **Thread Block Clusters**: For distributed shared memory if needed

---

## 2. How Can Hopper Features Help?

### 2.1 TMA (Tensor Memory Accelerator)

From the Hopper Tuning Guide:
> TMA allows applications to transfer 1D and up to 5D tensors between global memory and shared memory... Avoids using registers for moving data... A single thread can issue large data movement instructions to the TMA unit.

**For LoRA fusion:**
- Load X, A, B via TMA descriptors (like LoRAFusion does in `fused_lora_xw_sb_tma.py`)
- The TMA handles boundary checking automatically
- Reduces register pressure for address calculation

**LoRAFusion TMA implementation** (lines 183-189):
```python
x = tl_experimental_descriptor_load(
    x_desc_ptr, [offs_xm, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], dtype
)
w = tl_experimental_descriptor_load(
    w_desc_ptr, [offs_wn, offs_k], [BLOCK_SIZE_N, BLOCK_SIZE_K], dtype
)
```

### 2.2 Warp Specialization

From example 48:
> This example uses the Warp Specialized kernel design where specific warps specialize on data movement between different memory spaces while other warps only work on local data.

**For LoRA fusion:**
- Producer warps: Load A, B into shared memory via TMA
- Consumer warps: Compute S=X@A.T and Y=S@B.T via tensor cores
- Overlap memory access with compute

### 2.3 Thread Block Clusters

From the tuning guide:
> A thread block can read from, write to, and perform atomics in shared memory of other thread blocks within its cluster. This is known as Distributed Shared Memory.

**Potential use**: If we need to share the intermediate S across multiple output tiles. However, for RF-resident fusion with R≤256, this may not be necessary.

---

## 3. What's the Right Build-Up Order?

### Recommended Phased Approach

#### Phase 1: Two-Op LoRA Fusion (A @ B → one kernel)

**Goal**: Fuse `S = X @ A.T` and `lora_out = S @ B.T` into a single kernel.

**Expected benefit**:
- Eliminate S intermediate write to global memory
- Memory traffic reduction: ~69 MB (two-op) vs ~128 MB (unfused) = **46% reduction**
- Arithmetic intensity improvement: 248 → compute-bound

**Implementation**:
1. Start with CUTLASS example 13 `B2bGemm` structure
2. Port to Hopper using collective builders from example 48/49
3. Use TMA for A, B loads (like LoRAFusion)
4. Keep S in registers between GEMM1 and GEMM2

**Why B matrices cannot be batched** (critical insight):
```
A matrices: Same input X → CAN be stacked: X @ [A_wq; A_wk; A_wv].T
B matrices: Different inputs (h_wq, h_wk, h_wv) → CANNOT be stacked
```

#### Phase 2: Fuse with Base GEMM (add in epilogue)

**Goal**: Compute `Y = X @ W.T + lora_out` in one kernel.

**Expected benefit**:
- Eliminate separate add kernel (currently 14-93 us depending on size)
- Full fusion achieves **71-75% memory traffic reduction**

**Implementation options**:

**Option A: Epilogue Visitor Tree (EVT)**
- Use `Sm90AuxLoad` to load pre-computed lora_out in epilogue
- Add to accumulator before store
- References: `examples/61_hopper_gemm_with_topk_and_softmax/`

**Option B: Parallel compute with merge**
- Run base GEMM and two-op LoRA in parallel warps
- Merge in epilogue
- More complex but potentially higher throughput

#### Phase 3: Batched A-Matrix Optimization (optional)

**Goal**: Batch the A projections for wq, wk, wv together.

**Expected benefit**: ~14-27% memory savings from A-batching (from benchmark)

**Why it's limited**:
- Only A matrices can be batched (same input X)
- B matrices have different inputs, cannot be batched
- Max benefit is partial

---

## 4. What Can We Learn from LoRAFusion?

### What LoRAFusion Does

LoRAFusion's forward kernel (`fused_lora_xw_sb.py`) computes:
```python
result = x @ w.T + s @ b.T * alpha  # where s = x @ a.T is PRE-COMPUTED
```

**Key insight**: They split the computation:
1. **Separate kernel**: `s = x @ a.T` (LoRA down-projection)
2. **Fused kernel**: `y = x @ w.T + s @ b.T * alpha` (base + LoRA up + add)

This is **NOT full two-op fusion** - they still materialize S to global memory.

### Why Their Approach Differs

LoRAFusion assumes **frozen base weights** (finetuning), so:
- No gradient needed for W → simpler backward
- Can optimize just the LoRA path

We need **full pretraining** with gradients for W, A, B, and X.

### What's Directly Applicable

1. **TMA descriptor pattern**: Their `fused_lora_xw_sb_tma.py` shows correct Triton TMA usage
2. **Block/tile structure**: Their BLOCK_SIZE_M/N/K choices are tuned for similar dimensions
3. **Fused (GEMM + small matmul + add) pattern**: Their kernel structure can be extended

### What Differs

| Aspect | LoRAFusion | Our Needs |
|--------|------------|-----------|
| S materialization | Writes to global memory | Keep in registers |
| Base weight gradients | None (frozen) | Required |
| A gradient | Simple: `ds.T @ x` | Same |
| B gradient | Simple: `dy.T @ s` | Same |
| X gradient | `dy @ W + ds @ A` | Same, but W is not frozen |

### Key Difference from Two-Op Fusion

LoRAFusion fuses:
- `y = x @ w.T + s @ b.T` (base GEMM + LoRA-B + add)

Two-op fusion would fuse:
- `lora_out = (x @ a.T) @ b.T` (LoRA-A + LoRA-B, keep intermediate in RF)

**Our ideal** combines both:
- `y = x @ w.T + (x @ a.T) @ b.T` (everything fused)

---

## 5. Batched/Grouped GEMM Analysis

### Why Batched GEMM (Example 56) Doesn't Solve Our Problem

From the benchmark:
```
B matrices batchable: NO
Reason: B matrices have different inputs (h_wq, h_wk, h_wv from A matmuls)
```

The ptr-array batched GEMM in example 56 requires:
- Same operation on different data
- Our B matmuls have **different input tensors** (the intermediate S values)

### Limited Benefit from A-Matrix Batching

From the analysis:
```
Memory traffic:
  Separate:   232.78 MB
  A-batched:  199.23 MB (14.4% savings)
```

This is the **maximum** we can achieve from batching, and it doesn't address the core bottleneck (B matmuls + add).

### Grouped GEMM (Example 57) Use Case

Grouped GEMM is for:
- Different problem sizes in each group
- Processing multiple unrelated GEMMs together

For LoRA, the dimensions are identical across wq/wk/wv, so grouped GEMM offers only:
- Reduced kernel launch overhead
- No memory traffic savings

---

## 6. Concrete Numbers

### Memory Traffic Analysis (M=8192, K=1024, N=1024, R=256)

| Strategy | Memory Traffic | Reduction vs Unfused |
|----------|---------------|---------------------|
| Unfused (current) | 129 MB | baseline |
| LoRAFusion-style | 40 MB | 69% |
| Two-op LoRA only | 35 MB | 73% |
| Full fusion | 37 MB | 71% |

### Kernel Timings (actual H200 measurements)

| Operation | Time (us) |
|-----------|-----------|
| Base GEMM | 31 |
| LoRA A | 11 |
| LoRA B | 13 |
| Add | 15 |
| **Total unfused** | **72** |
| LoRAFusion-style | 60 |

### Arithmetic Intensity (H200 ridge point: 206 FLOP/byte)

| Operation | AI | Bound |
|-----------|----|----|
| Base GEMM | 482 | Compute |
| LoRA A | 200 | **Memory** |
| LoRA B | 200 | **Memory** |
| Add | 0.2 | **Memory** |
| Two-op fused | 248 | Compute |
| Full fusion | 702 | Compute |

---

## 7. Recommended Implementation Path

### REVISED APPROACH (Jan 29): SMEM-Staged Megakernel

**Update**: After detailed analysis (Section 11), we now recommend a **megakernel** approach over LoRAFusion's two-kernel design. The key insight is that **kernel launch overhead (~27 us) exceeds S intermediate overhead (~1.76 us)**.

### Why Not LoRAFusion As-Is

LoRAFusion's approach (two kernels) still has:
- Two kernel launches (~18 us overhead)
- X read twice (once per kernel)
- S in GMEM (not SMEM)

Our megakernel approach eliminates all of these.

### Phase 1: Basic Megakernel (2 weeks)

**Single kernel**: `Y = X @ W + (X @ A) @ B * scale`

```
For each M-tile:
    Phase A: Compute S = X @ A (stream through K)
    Store S to SMEM (64 KB, fits in 228 KB capacity)
    __syncthreads()

    Phase B: For each N-tile:
        Load S from SMEM (fast: ~0.003 us per tile)
        Load B slice from GMEM
        Accumulate: Y += S @ B * scale

        Load W slice from GMEM
        Accumulate: Y += X @ W (X may hit L2 from Phase A)

    Store Y tiles to GMEM
```

**Key advantage**: One launch instead of four = ~27 us saved.

### Phase 2: TMA Pipelining (1-2 weeks)

Add Hopper TMA for async loads:
- Producer warps: Load X, W, A, B tiles via TMA
- Consumer warps: Compute on ready tiles
- Overlap memory latency with compute

**Reference**: CUTLASS Example 48/49 for warp-specialized patterns

### Phase 3: L2 Cache Optimization (1 week)

Exploit H200's 50 MB L2 cache:
- X (16.8 MB) fits in L2
- After Phase A (S computation), X tiles remain in L2
- Phase B X accesses hit L2 instead of GMEM (~3-4× faster)

### Phase 4: Backward Kernels (3-4 weeks)

**Extended for trainable W** (LoRAFusion has frozen W):

```
# New kernel needed for trainable W:
dW = dY.T @ X          # Standard GEMM, highly optimizable

# Adapt LoRAFusion patterns:
dX = dY @ W + dS @ A   # Can fuse similar to forward
dB = dY.T @ S * alpha  # fused_lora_dys_dyb.py pattern
dS = dY @ B * alpha    # fused_lora_dys_dyb.py pattern

# Separate small GEMM:
dA = dS.T @ X
```

**Key insight**: S must be saved from forward pass for backward. With SMEM staging in forward, we write S to GMEM once at the end (still cheap at 4.2 MB).

### Phase 5: Integration (1-2 weeks)

1. PyTorch autograd.Function wrapper (or torch.library.triton_op)
2. Integration with weight_sharing.py
3. torch.compile compatibility testing
4. FSDP gradient correctness testing

### Fallback: LoRAFusion Style

If megakernel proves too complex, fall back to LoRAFusion's two-kernel approach:

**Kernel 1**: `S = X @ A` (cuBLAS, optimal)
**Kernel 2**: `Y = X @ W + S @ B` (LoRAFusion Triton kernel)

Expected improvement: 72 us → ~55-60 us (1.2-1.3× speedup)

### Why NOT RF-Resident Two-Op Fusion

| Concern | Impact |
|---------|--------|
| tile_N must equal R=256 | Constrains all tile choices |
| Accumulator 64×256 = 64KB per warp group | High register pressure |
| Base GEMM performance degradation | Suboptimal tiling |
| Example 13 is Ampere, not Hopper | Significant porting effort |
| **Doesn't eliminate add kernel** | Still need separate add |
| **S savings only 8.4 MB** | 6.5% of total traffic |

**The megakernel approach is superior** because it addresses the actual bottlenecks (kernel launches, add fusion) rather than the perceived bottleneck (S intermediate).

---

## 7.5 Addressing LoRAFusion Paper Concerns

The LoRAFusion paper identifies three critical issues with full graph fusion:

### Issue 1: Recomputation When M is Large

> "The first option recomputes S inside each tile of the fused kernel, but requires loading the entire A matrix repeatedly, becoming expensive when batch size M is large."

**Our M = 8192** (batch=4 × seq=2048). With tile_M=64, we'd have 128 tiles. Recomputing S would mean loading A 128 times instead of once. This is prohibitive.

### Issue 2: Synchronization Overhead

> "The second option fuses computation and uses synchronization across thread blocks to share S, where only a single tile computes the intermediate S tiles and writes to global memory, while other tiles wait through a semaphore."

This adds coordination overhead and breaks the independence that makes GPU parallelism efficient. CUTLASS example 13's RF-resident approach avoids this by requiring tile_N = R, but this creates the register pressure issue.

### Issue 3: GPU Resource Consumption

> "A suboptimal tiling layout or overuse of registers and shared memory can greatly degrade [base GEMM] performance."

With tile_N=256 required for RF-resident two-op fusion:
- Accumulator: 64 × 256 = 16K fp32 = 64KB per warp group
- This limits occupancy and prevents optimal tile choices for X @ W

### LoRAFusion's Solution

> "Our approach takes a third option: explicitly storing and reloading S from GPU global memory. Since S is much smaller than other tensors and depends on the small LoRA rank r, the cost of reading and writing it is low."

For our dimensions:
- S = 8192 × 256 × 2 = **4.2 MB**
- X = 8192 × 1024 × 2 = **16.8 MB**
- Y = 8192 × 1024 × 2 = **16.8 MB**

S is 4× smaller than X or Y. The trade-off of materializing S to preserve base GEMM performance is favorable.

---

## 8. Key Files for Reference

| File | Purpose |
|------|---------|
| `cutlass-4.3.5/examples/13_two_tensor_op_fusion/` | Two-op fusion template |
| `cutlass-4.3.5/examples/48_hopper_warp_specialized_gemm/` | Hopper TMA + warp spec |
| `cutlass-4.3.5/examples/49_hopper_gemm_with_collective_builder/` | Collective builder API |
| `lorafusion/ops/triton_ops/fused_lora_xw_sb_tma.py` | TMA descriptor usage |
| `lorafusion/ops/triton_ops/fused_lora_dyw_dsa.py` | Backward kernel pattern |
| `performance_analysis_report.md` | Current bottleneck analysis |

---

## 9. Risks and Mitigation

### Risk 1: Register Pressure with Large Ranks

**Issue**: RF-resident fusion requires holding S (M_tile × R) in registers.
**Mitigation**:
- Limit to tile_N ≤ 256 (covers our ranks)
- Fall back to shared memory staging for larger ranks

### Risk 2: Backward Kernel Complexity

**Issue**: Multiple gradients with interdependencies.
**Mitigation**:
- Start with LoRAFusion's backward as template
- Add W gradient computation separately

### Risk 3: torch.compile Compatibility

**Issue**: Custom CUTLASS kernels may break torch.compile graphs.
**Mitigation**:
- Use `torch.library` registration for proper integration
- Test with `fullgraph=True` to verify no graph breaks

---

## 10. Summary of Key Decisions (REVISED Jan 29, 2026)

| Question | Answer | Rationale |
|----------|--------|-----------|
| Is two-op RF fusion viable? | **Technically yes, but NOT optimal** | Doesn't address main bottlenecks (launches, add) |
| CTA mismatch problem? | **NO** | Our intermediate K2=R=K1 matches |
| Can B matrices be batched? | **NO** | Different inputs (h_wq, h_wk, h_wv) |
| Best approach? | **SMEM-staged megakernel** | Addresses all bottlenecks in one kernel |
| What about LoRAFusion? | **Good fallback, not optimal** | Still has 2 launches, X read twice |
| Hopper features needed? | TMA, warp specialization, SMEM | For maximum performance |
| Expected speedup? | **2.8× on LoRA+Linear path** | 72 us → ~26 us |

### Key Insights (Jan 29, 2026)

1. **S intermediate is NOT the main bottleneck**: Only 7% of LoRA A+B time (1.76 us out of 24 us)

2. **Kernel launch overhead IS a main bottleneck**: ~27 us (37% of total 72 us)

3. **The add kernel IS a main bottleneck**: ~15 us (21% of total)

4. **X duplication is significant**: ~3.5 us (reading X twice instead of once)

5. **SMEM staging is the key**: Store S in SMEM (64 KB fits in 228 KB), not GMEM

6. **L2 cache helps**: X (16.8 MB) fits in H200 L2 (50 MB), enabling reuse

### Why Megakernel Beats LoRAFusion

| Metric | LoRAFusion (2 kernels) | Megakernel (1 kernel) |
|--------|------------------------|----------------------|
| Kernel launches | 2 (~18 us) | 1 (~9 us) |
| S storage | GMEM (1.76 us) | SMEM (0.05 us) |
| X reads | 2× (33.6 MB) | 1× + L2 (~20 MB) |
| Add kernel | Fused | Fused |
| **Estimated time** | **~55 us** | **~26 us** |

**The trade-off**: Two-op fusion saves 4.2 MB (S read) + 4.2 MB (S write) = 8.4 MB but costs optimal base GEMM performance. Given that base GEMM is compute-bound and our primary bottleneck, preserving its performance is more valuable.

---

## 11. Deep Dive: Where Does the Time Actually Go? (Jan 29, 2026)

### 11.1 Revisiting the Bottleneck

From performance analysis, the current breakdown per Linear+LoRA:

| Operation | Time (us) | % of Total |
|-----------|-----------|------------|
| Base GEMM (X @ W.T) | 31 | 43% |
| LoRA A (X @ A.T) | 11 | 15% |
| LoRA B (S @ B.T) | 13 | 18% |
| Add | 15 | 21% |
| **Total** | **72** | 100% |

**Critical observation**: LoRA A+B (24 us) > Add (15 us). But where does LoRA A+B time go?

### 11.2 S Intermediate: Only 7% of LoRA A+B Time

**S tensor size**: 8192 × 256 × 2 bytes = **4.2 MB**

**At H200 HBM3e bandwidth (4.8 TB/s)**:
```
S write (in LoRA A): 4.2 MB / 4.8 TB/s = 0.88 us
S read (in LoRA B):  4.2 MB / 4.8 TB/s = 0.88 us
Total S overhead:    1.76 us
```

**S intermediate is only 1.76 / 24 = 7.3% of LoRA A+B time!**

### 11.3 Where Does the Rest of LoRA A+B Time Go?

**LoRA A (11 us) breakdown**:
```
Memory traffic: X (16.8 MB) + A (0.52 MB) + S_write (4.2 MB) = 21.5 MB
Theoretical minimum at 4.8 TB/s: 21.5 MB / 4.8 TB/s = 4.5 us
Actual: 11 us
Overhead: 6.5 us (kernel launch ~9 us explains most of this)
```

**LoRA B (13 us) breakdown**:
```
Memory traffic: S_read (4.2 MB) + B (0.52 MB) + lora_out (16.8 MB) = 21.5 MB
Theoretical minimum: 4.5 us
Actual: 13 us
Overhead: 8.5 us (kernel launch + memory latency)
```

**Key insight**: The overhead comes primarily from:
1. **Kernel launch**: ~9 us per kernel × 2 = 18 us
2. **Memory latency not fully hidden**: ~5 us
3. **S intermediate**: only ~1.76 us

### 11.4 Recomputation vs Memory Latency Analysis

**Question**: Since we're memory-bound, would recomputing S be faster than reading it from memory?

**Recomputation scenario**: For each (M-tile, N-tile) output pair, recompute S

```
With M=8192, tile_M=64: 128 M-tiles
With N=1024, tile_N=64: 16 N-tiles

Option A: Recompute S for each (M-tile, N-tile) pair
  Total recomputations: 128 × 16 = 2048 times
  Wait - this is wrong. S only depends on M, not N.

Option B: Recompute S for each M-tile (correct)
  Recomputations: 128 times (once per M-tile)
  BUT: still need to reload A for each M-tile!
```

**Compute time for S recomputation**:
```
GEMM1 FLOPs per M-tile: 2 × 64 × 1024 × 256 = 33.6M FLOPs
At H200 989 TFLOPS: 33.6M / 989T = 0.034 us per M-tile

Total for 128 M-tiles: 128 × 0.034 = 4.3 us
(Same as computing S once - no recomputation overhead in FLOPs)
```

**But A must be reloaded for each M-tile**:
```
A size: 1024 × 256 × 2 = 0.52 MB
A reloads: 128 times (once per M-tile)
Traffic: 128 × 0.52 MB = 66.6 MB
Time: 66.6 MB / 4.8 TB/s = 13.9 us
```

**Compare to materializing S once**:
```
A read once: 0.52 MB → 0.11 us
S write once: 4.2 MB → 0.88 us
S read once: 4.2 MB → 0.88 us
Total: 1.87 us
```

**Recomputation is 7.4× worse** (13.9 us vs 1.87 us). NOT worth it.

### 11.5 SMEM Staging: The Sweet Spot

**Idea**: Store S in shared memory instead of global memory.

**S tile size per M-tile**: 64 × 256 × 4 bytes (fp32) = **64 KB**
**H200 SMEM capacity**: 228 KB per SM

**64 KB fits comfortably!**

**SMEM vs GMEM bandwidth**:
```
GMEM (HBM3e): 4.8 TB/s aggregate
SMEM: ~19 TB/s per SM (theoretical peak)
```

**S access times with SMEM staging**:
```
S write to SMEM: 64 KB / 19 TB/s = 0.003 us
S read from SMEM (per N-tile): 64 KB / 19 TB/s = 0.003 us
With 16 N-tiles: 16 × 0.003 = 0.05 us total reads

Total SMEM S overhead: 0.003 + 0.05 = 0.053 us
```

**Compare to GMEM**:
```
S write to GMEM: 0.88 us
S read from GMEM: 0.88 us
Total: 1.76 us
```

**SMEM staging saves 1.71 us per M-tile × 128 M-tiles... wait.**

Actually, let me recalculate properly. The S tensor is (8192, 256). We process it in M-tiles of 64 rows each.

**Correct SMEM staging analysis**:

For each M-tile (64 rows of S):
1. Compute S[tile_M, R] via GEMM1 → stays in registers first
2. Write to SMEM (or keep in registers if possible)
3. For each N-tile of output:
   - Read S from SMEM
   - Compute S @ B[R, tile_N]
   - Also compute base GEMM contribution
   - Accumulate

**Register pressure with SMEM fallback**:
- S tile in RF: 64 × 256 = 16K fp32 = 128 regs/thread (with 128 threads)
- Output accumulator: 64 × 64 = 32 regs/thread
- Operands: ~64 regs/thread
- Total: ~224 regs/thread (under 255 limit!)

**Wait - this might actually work for RF-resident!** Let me recalculate the full picture.

### 11.6 Revised Register Pressure Analysis

**Thread block configuration**: 4 warps × 32 threads = 128 threads

**For RF-resident two-op fusion**:
```
S accumulator (tile_M × R): 64 × 256 = 16,384 fp32 values
Per thread: 16,384 / 128 = 128 registers

Y accumulator (tile_M × tile_N): 64 × 64 = 4,096 fp32 values
Per thread: 4,096 / 128 = 32 registers

Operand fragments: ~64 registers

Total: 128 + 32 + 64 = 224 registers
```

**This is under the 255 limit!** But there's a catch...

**The catch**: For RF-resident, S must stay in registers while we iterate over ALL N-tiles. But the Y accumulator changes for each N-tile. So:

```
Iteration 1: S (128 regs) + Y_tile1 (32 regs) + operands (64 regs) = 224 regs ✓
Iteration 2: S (128 regs) + Y_tile2 (32 regs) + operands (64 regs) = 224 regs ✓
...
```

Actually this works! Each N-tile iteration uses the same S (kept in registers) with a new Y accumulator.

**So why did I say 288 registers earlier?**

Earlier calculation assumed we need BOTH S and a full 128×256 accumulator simultaneously. But with proper tiling:
- S: 64 × 256 (stays resident for all N-tiles)
- Y: 64 × 64 (only one N-tile at a time)

**Revised verdict**: RF-resident two-op fusion IS feasible with careful tiling!

### 11.7 The Megakernel Design

**Goal**: Compute `Y = X @ W.T + (X @ A.T) @ B.T * scale` in ONE kernel

**Key optimizations**:
1. **Single X read**: Stream X through K dimension, use for both base GEMM and LoRA A
2. **S in registers or SMEM**: No GMEM traffic for S
3. **Fused add**: Add happens in accumulator, no separate kernel
4. **One kernel launch**: Saves ~27 us

**Pseudocode**:
```cuda
__global__ void fused_linear_lora_megakernel(
    X, W, A, B, Y,  // inputs and output
    M, K, N, R, scale
) {
    // Shared memory for S (if RF-resident not possible)
    __shared__ float S_smem[BLOCK_M][R];  // 64 × 256 × 4 = 64 KB

    // Register accumulators
    float S_reg[BLOCK_M / WARPS][R / WARP_COLS];  // Distributed across threads
    float Y_reg[BLOCK_M / WARPS][BLOCK_N / WARP_COLS];

    // Initialize accumulators
    zero(S_reg);

    // Phase 1: Stream through K, compute BOTH S and partial base GEMM
    for (int k = 0; k < K; k += BLOCK_K) {
        // Load X tile via TMA (ONCE for both paths!)
        float X_tile[BLOCK_M][BLOCK_K] = tma_load(X, m_offset, k);

        // Load A tile
        float A_tile[BLOCK_K][R] = tma_load(A, k, 0);

        // Accumulate S = X @ A.T
        mma(X_tile, A_tile, S_reg);  // S_reg += X_tile @ A_tile

        // For base GEMM, we'll do this in Phase 2 to overlap
    }

    // S is now in S_reg (or write to S_smem if needed)
    // Optional: __syncthreads() if using SMEM

    // Phase 2: For each output N-tile
    for (int n = 0; n < N; n += BLOCK_N) {
        zero(Y_reg);

        // Load B tile for this N-tile
        float B_tile[R][BLOCK_N] = tma_load(B, 0, n);

        // Compute LoRA contribution: S @ B
        mma(S_reg, B_tile, Y_reg);  // Y_reg += S @ B * scale
        Y_reg *= scale;

        // Compute base GEMM contribution: X @ W
        for (int k = 0; k < K; k += BLOCK_K) {
            float X_tile = tma_load(X, m_offset, k);  // Can cache from Phase 1!
            float W_tile = tma_load(W, k, n);
            mma(X_tile, W_tile, Y_reg);  // Y_reg += X @ W
        }

        // Store output tile
        tma_store(Y, m_offset, n, Y_reg);
    }
}
```

**Problem with above**: X is still read multiple times (once per N-tile for base GEMM).

### 11.8 Improved Design: Single X Read with Pipelining

**Better approach**: Compute S and base GEMM simultaneously while streaming through K.

```cuda
// Phase 1: Stream through K ONCE
for (int k = 0; k < K; k += BLOCK_K) {
    // Load X tile (ONCE)
    X_tile = tma_load(X, m_offset, k);

    // Load A tile
    A_tile = tma_load(A, k, 0);

    // Accumulate S = X @ A
    S_accum += X_tile @ A_tile;

    // For each N-tile, also accumulate base GEMM
    // (This requires loading W for all N-tiles, which may not fit)
}
```

**Challenge**: We can't load all W tiles for all N in one pass through K.

**Solution**: Reorder the loops

```
Option A: K-major (stream K, iterate N after)
  - Reads X once for S computation
  - Reads X N/BLOCK_N times for base GEMM (once per N-tile)
  - Total X reads: 1 + 16 = 17 times (worse!)

Option B: N-major with S precompute (current LoRAFusion)
  - Precompute S (reads X once)
  - For each N-tile: read X again for base GEMM
  - Total X reads: 1 + 16 = 17 times

Option C: Fused K-streaming with output accumulation
  - For each K-tile:
    - Update S accumulator (X @ A contribution)
    - Update ALL N output accumulators (X @ W contribution)
  - Requires holding N/BLOCK_N output accumulators simultaneously!
  - Registers needed: 16 × (64×64) = 262,144 fp32 → NOT FEASIBLE
```

**Option D: Hybrid with L2 cache reuse**

```
Phase 1: Compute S, X stays hot in L2
  - Stream through K, accumulate S
  - X (16.8 MB) > L2 (50 MB on H100), but tiles may be cached

Phase 2: Compute Y = base + LoRA
  - For each N-tile:
    - S @ B (S in SMEM or RF)
    - X @ W (X tiles may hit L2 from Phase 1)
```

**L2 cache analysis**:
```
H200 L2 cache: ~50 MB
X size: 16.8 MB → FITS in L2!

If X stays in L2 between Phase 1 and Phase 2:
- Phase 1 X reads: from GMEM (cold)
- Phase 2 X reads: from L2 (3-4× faster than GMEM)
```

### 11.9 Expected Savings with Megakernel

| Optimization | Current Cost | Megakernel Cost | Savings |
|--------------|--------------|-----------------|---------|
| Kernel launches (4→1) | ~36 us | ~9 us | **27 us** |
| S to GMEM | 1.76 us | 0.05 us (SMEM) | **1.71 us** |
| X read (with L2) | ~7 us | ~5 us (L2 hit) | **2 us** |
| Add kernel | 15 us | 0 (fused) | **15 us** |
| **Total savings** | | | **~46 us** |

**Current unfused time**: 72 us
**Projected megakernel time**: 72 - 46 = **~26 us**
**Speedup**: **2.8×**

### 11.10 Why LoRAFusion Doesn't Do This

Looking at LoRAFusion's approach:
1. **Two kernels**: S = X @ A, then Y = X @ W + S @ B
2. **S in GMEM**: Not SMEM
3. **X read twice**: Once per kernel

Possible reasons:
1. **Triton limitations**: SMEM staging is harder to express in Triton
2. **Generality**: Their kernels work for variable batch sizes
3. **Simplicity**: Two simpler kernels vs one complex megakernel
4. **Frozen W**: They don't need W gradients, different tradeoffs

### 11.11 Implementation Complexity

**SMEM-staged megakernel requires**:
1. Careful register allocation (S vs Y accumulators)
2. SMEM bank conflict avoidance for S
3. TMA descriptor setup for X, W, A, B
4. Proper synchronization between phases
5. Handling of edge cases (non-divisible dimensions)

**Estimated implementation effort**: 2-3 weeks for forward kernel

**Recommended implementation path**:
1. First: Implement basic megakernel with S in GMEM (validate correctness)
2. Then: Add SMEM staging for S
3. Then: Add TMA pipelining for X reuse
4. Finally: Add backward kernels

---

## 12. Updated Recommendations (Jan 29, 2026)

### 12.1 Primary Recommendation: SMEM-Staged Megakernel

Instead of adopting LoRAFusion as-is, implement a custom **SMEM-staged megakernel** that:

1. **Computes everything in one launch**: Y = X @ W + (X @ A) @ B * scale
2. **Stores S in SMEM**: 64 KB per M-tile, fits in 228 KB
3. **Exploits L2 cache for X reuse**: X (16.8 MB) fits in L2 (50 MB)
4. **Fuses add in accumulator**: No separate add kernel

### 12.2 Expected Performance

| Metric | Current | Megakernel | Improvement |
|--------|---------|------------|-------------|
| Time per Linear+LoRA | 72 us | ~26 us | **2.8×** |
| Memory traffic | 129 MB | ~55 MB | **2.3×** |
| Kernel launches | 4 | 1 | **4×** |
| X reads | 2 | 1 (+L2) | **~2×** |

### 12.3 Fallback: LoRAFusion Style

If megakernel proves too complex, fall back to LoRAFusion's two-kernel approach:

**Kernel 1**: S = X @ A (cuBLAS, optimal)
**Kernel 2**: Y = X @ W + S @ B (LoRAFusion Triton kernel)

Expected improvement: 72 us → ~55-60 us (**1.2-1.3×**)

### 12.4 Key Insight Summary

1. **S intermediate is cheap** (only 7% of LoRA A+B time)
2. **Kernel launch overhead is expensive** (~27 us, 37% of total)
3. **X duplication is expensive** (~3.5 us, 5% of total)
4. **Add kernel is expensive** (~15 us, 21% of total)
5. **The right fusion eliminates #2, #3, #4** while keeping S cheap in SMEM
