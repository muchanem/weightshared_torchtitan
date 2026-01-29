# Hopper Kernel Analysis for LoRA Fusion

## Executive Summary

This document analyzes kernel fusion strategies for speeding up LoRA operations in the weight-shared model. The key finding is that **CUTLASS two-op fusion is the right starting point**, but it requires Hopper-specific modifications for optimal performance. The forum concerns about CTA count mismatch **do not apply** to our dimensions.

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

### REVISED APPROACH: Adopt LoRAFusion's Graph-Splitting Strategy

**Why we're revising**: The LoRAFusion paper identifies critical trade-offs that make two-op fusion (A @ B in registers) suboptimal:

1. **Register/shared memory consumption**: RF-resident fusion requires tile_N = R (256), consuming resources needed for optimal base GEMM tiling
2. **Recomputation problem**: Full fusion either recomputes S per tile (expensive when M=8192) or requires cross-threadblock sync
3. **S is cheap to materialize**: S has shape (M, R) = 4.2 MB, much smaller than X at 16.8 MB

**The LoRAFusion insight**: Split the computation graph at S (the small intermediate), not at full-sized tensors.

### Phase 1: Implement LoRAFusion-Style Forward (2 weeks)

**Kernel 1**: `S = dropout(X) @ A`
- Standard small GEMM (output is M × R)
- Can use cuBLAS or simple Triton kernel
- Low optimization priority (memory-bound anyway)

**Kernel 2**: `Y = X @ W + S @ B * alpha` (horizontally fused)
- Base GEMM (X @ W) uses optimal tiling without R constraint
- LoRA contribution (S @ B) fused in accumulator or epilogue
- Add happens in registers, no separate add kernel

**Reference**: `lorafusion/ops/triton_ops/fused_lora_xw_sb.py`

### Phase 2: Port to CUTLASS/Hopper (2-3 weeks)

Options:
- **Option A**: Use LoRAFusion's Triton kernels directly (faster to implement)
- **Option B**: Port to CUTLASS with Hopper features for maximum performance

For CUTLASS Option B:
1. Use Collective Builder (Example 49) for base GEMM
2. Add EVT epilogue to load S and compute S @ B + base_accum
3. TMA for X, W, S, B loads

### Phase 3: Backward Kernels (3-4 weeks)

**Extended for trainable W** (LoRAFusion has frozen W):

```
# New kernel needed for trainable W:
dW = dY.T @ X          # Standard GEMM, highly optimizable

# LoRAFusion kernels (can adopt directly):
dX = dY @ W + dS @ A   # fused_lora_dyw_dsa.py
dB = dY.T @ S * alpha  # fused_lora_dys_dyb.py (fused with dS computation)
dS = dY @ B * alpha    # fused_lora_dys_dyb.py

# Separate small GEMM:
dA = dS.T @ X
```

**Key insight**: S must be saved from forward pass for backward. This is acceptable because S is small (4.2 MB for our dimensions).

### Phase 4: Integration (1-2 weeks)

1. PyTorch autograd.Function wrapper
2. Integration with weight_sharing.py
3. FSDP compatibility testing

### Why NOT Two-Op Fusion

| Concern | Impact |
|---------|--------|
| tile_N must equal R=256 | Constrains all tile choices |
| Accumulator 64×256 = 64KB per warp group | High register pressure |
| Base GEMM performance degradation | Suboptimal tiling |
| Example 13 is Ampere, not Hopper | Significant porting effort |

**LoRAFusion's approach avoids all these issues** by keeping the base GEMM unconstrained and only materializing the small S tensor.

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

## 10. Summary of Key Decisions (REVISED)

| Question | Answer | Rationale |
|----------|--------|-----------|
| Is two-op fusion viable? | **Technically yes, but NOT recommended** | Register pressure degrades base GEMM tiling |
| CTA mismatch problem? | **NO** | Our intermediate K2=R=K1 matches |
| Can B matrices be batched? | **NO** | Different inputs (h_wq, h_wk, h_wv) |
| Best starting point? | **LoRAFusion's fused kernels** | Graph-splitting at S is more practical |
| Hopper features needed? | TMA, warp specialization | For competitive performance |
| What from LoRAFusion? | **Their entire approach** | Split at S, fuse X@W + S@B |
| Expected speedup? | ~60-70% memory reduction | Same as LoRAFusion achieves |

### Key Insight from LoRAFusion Paper

**S is cheap to materialize**: With shape (M, R) = (8192, 256) = 4.2 MB, S is only 1/4 the size of X. The cost of reading/writing S is acceptable when it enables:
1. Optimal tiling for the compute-bound base GEMM
2. No register pressure from large tile_N constraint
3. Simpler kernel design without recomputation or synchronization

**The trade-off**: Two-op fusion saves 4.2 MB (S read) + 4.2 MB (S write) = 8.4 MB but costs optimal base GEMM performance. Given that base GEMM is compute-bound and our primary bottleneck, preserving its performance is more valuable.
