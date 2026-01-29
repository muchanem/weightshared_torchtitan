# Weight-Shared LLM Performance Analysis - Complete Data Extraction

**Generated from:** 35 JSONL chat transcript files (~18MB)
**Total lines processed:** 9,842
**Last updated:** 2026-01-28

---

## 1. Executive Summary

The weight-shared LLM architecture introduces a **33% performance slowdown** compared to the unshared baseline:
- **Unshared TPS:** 73,139 tokens/sec
- **Shared TPS:** 48,737 tokens/sec
- **Ratio:** 0.67x

**Root cause:** Memory-bound operations, not compute. ~160k us (23% of compute time) is spent in elementwise ops (add, copy, mul) for LoRA additions.

**Key findings:**
1. torch.compile makes LoRA modules **3x slower** than eager mode
2. Custom Triton kernels are **8x slower** than cuBLAS
3. torch.addmm fusion provides **1.16x micro-benchmark speedup** but only **1% E2E improvement**
4. Embedding factorization alone provides **2.08x speedup** (152,265 TPS)
5. The inductor backend already auto-fuses similar patterns, limiting manual optimization gains

---

## 2. Performance Metrics Tables

### 2.1 Primary Comparison (250M Models, H200 NVL GPUs)

| Metric | Unshared | Shared (Combined) | Ratio |
|--------|----------|-------------------|-------|
| TPS (tokens/sec) | 73,139 | 48,737 | **0.67x** |
| TFLOPS | 105.27 | 85.95 | 0.82x |
| MFU | 10.64% | 8.69% | 0.82x |
| Memory | 60.06 GiB | 67.60 GiB | 1.13x |
| Parameters | 294M | 303M | 1.03x |

### 2.2 Ablation Study Results

| Configuration | TPS (step 15-20) | vs Unshared | Parameters |
|---------------|------------------|-------------|------------|
| Unshared (dim=768) | 73,139 | 1.00x | 294M |
| Combined (all sharing) | 48,737 | 0.67x | 303M |
| Layer sharing only | 44,748 | 0.61x | 502M |
| Attention sharing only | 40,472 | 0.55x | 545M |
| **Embedding only** (dim=1024) | **152,265** | **2.08x** | 345M |

### 2.3 HTA Trace Temporal Breakdown (rank 0)

| Metric | Unshared (us) | Shared (us) | Change |
|--------|---------------|-------------|--------|
| Compute time | 256,039 | 696,624 | **+172%** |
| Communication | 86,771 | 121,607 | +40% |
| Memory ops | 1,237 | 4,281 | +246% |
| Idle time | 147,976 | 42,450 | -71% |
| **Total kernel time** | 473,602 | 799,735 | **+69%** |

### 2.4 Microbenchmark Results (batch=4, seq=2048, dim=1024, rank=256)

| Operation | nn.Linear | SharedLinearWithLoRA | Overhead |
|-----------|-----------|----------------------|----------|
| Forward | 0.029 ms | 0.078 ms | **+172%** |
| Forward+Backward | 0.129 ms | 0.263 ms | **+104%** |
| SwiGLU FFN | 0.305 ms | 0.597 ms | **+96%** |

### 2.5 LoRA Rank Sweep (Forward Only)

| Rank | Overhead |
|------|----------|
| 8 | +144% |
| 32 | +142% |
| 128 | +150% |
| 256 | +171% |
| 512 | +212% |

### 2.6 Batch Size Sweep (Forward Only, rank=256)

| Batch | Overhead |
|-------|----------|
| 1 | +317% |
| 4 | +170% |
| 16 | +157% |

### 2.7 Dimension Sweep (Forward Only, rank=256, batch=4)

| Dimension | Overhead |
|-----------|----------|
| 512 | +271% |
| 1024 | +167% |
| 2048 | +90% |
| 4096 | **+54%** |

---

## 3. Kernel Analysis

### 3.1 Top Time-Consuming Kernels (Shared Model)

| Kernel | Time (us) | Type |
|--------|-----------|------|
| direct_copy_kernel (float) | 68,880 | Memory-bound |
| nvjet_tst_256x128_64x4 (matmul) | 45,334 | Compute |
| CUDAFunctor_add (float) | 39,861 | Memory-bound |
| bfloat16_copy_kernel | 33,641 | Memory-bound |
| nvjet_tst_192x192_64x4 (matmul) | 20,661 | Compute |
| CUDAFunctor_add (bf16) | 20,191 | Memory-bound |
| SoftMaxBackward | 17,252 | Compute |
| SoftMaxForward | 16,325 | Compute |
| nvjet_tst_128x192 | 15,998 | Compute |
| nvjet_tst_320x128 | 12,415 | Compute |
| BinaryFunctor Mul (bf16) | 11,560 | Memory-bound |

### 3.2 Kernel Launch Comparison

| Metric | Unshared | Shared | Change |
|--------|----------|--------|--------|
| aten::mm calls | 423 | 1,545 | **+265%** |
| cuLaunchKernelEx calls | 503 | 1,625 | +223% |
| cudaMemsetAsync calls | 393 | 1,042 | +165% |

### 3.3 Kernel Fusion Indicators

| Mode | Kernel Name Pattern |
|------|---------------------|
| Unfused (beta=0) | `nvjet_sm90_tst_*_bz_*` |
| Fused (beta add) | `nvjet_sm90_tst_*_badd_*` |

### 3.4 LoRA Overhead Breakdown (per Linear+LoRA)

| Operation | Time | Comment |
|-----------|------|---------|
| Base matmul (x @ W.T) | 0.076 ms | |
| LoRA A matmul (x @ A.T) | 0.010 ms | |
| LoRA B matmul (h @ B.T) | 0.024 ms | |
| Add (base + lora) | 0.036 ms | **47% of LoRA overhead** |

### 3.5 Bottleneck Analysis (SharedFeedForward)

| Category | Time | Percentage | Kernel Launches |
|----------|------|------------|-----------------|
| GEMM kernels | 3.19 ms | 60.8% | 90 |
| Elementwise (adds) | 2.02 ms | **38.6%** | 80 |

### 3.6 Trace Files

- Unshared: `/local/scratch/muchane_673547/weightshared_torchtitan/fixed_traces/unshared/`
- Shared: `/local/scratch/muchane_673547/weightshared_torchtitan/fixed_traces/shared/`
- HTA location: `/local/scratch/muchane_673547/HolisticTraceAnalysis`

---

## 4. Model Configurations Tested

### 4.1 Qwen3 Model Variants

| Variant | vocab_size | dim | n_layers | n_heads | n_kv_heads | hidden_dim |
|---------|------------|-----|----------|---------|------------|------------|
| debugmodel | 2,048 | 256 | 8 | 16 | 8 | 3,072 |
| debugmodel_weightshared | 2,048 | 256 | 8 | 8 | 4 | 512 |
| 40M | 151,936 | 512 | 12 | 16 | 8 | 2,048 |
| **250M_shared** | 151,936 | 1,024 | 20 | 16 | 8 | 3,072 |
| **250M_unshared** | 151,936 | 768 | 20 | 16 | 8 | 3,072 |
| 0.6B | 151,936 | 1,024 | 28 | 16 | 8 | 3,072 |
| 1.7B | 151,936 | 2,048 | 28 | 16 | 8 | 6,144 |
| 4B | 151,936 | 2,560 | 36 | 32 | 8 | 9,728 |
| 8B | 151,936 | 4,096 | 36 | 32 | 8 | 12,288 |
| 14B | 151,936 | 5,120 | 40 | 40 | 8 | 17,408 |
| 32B | 151,936 | 5,120 | 64 | 64 | 8 | 25,600 |

### 4.2 250M Combined Configuration

```toml
[model]
name = "qwen3"
flavor = "250M_shared"
dim = 1024
n_layers = 20
n_heads = 16
n_kv_heads = 8
head_dim = 64
hidden_dim = 3072
vocab_size = 151936
max_seq_len = 2048

[weight_sharing.attention]
enabled = true
head_sharing = true
grouping = 1
rank = 192
two_step = false

[weight_sharing.layer]
enabled = true
layer_groups = [[0], [1,2,3], [4,5,6], [7,8,9], [10,11,12], [13,14,15], [16,17,18], [19]]
lora_rank = 256

[weight_sharing.embedding]
enabled = true
d_emb = 608
tie_output = true

[training]
local_batch_size = 22
seq_len = 2048
max_norm = 1.0
steps = 55500
mixed_precision_param = "bfloat16"
mixed_precision_reduce = "float32"

[compile]
enable = true
```

### 4.3 Training Configuration

| Parameter | Value |
|-----------|-------|
| local_batch_size | 22 |
| seq_len | 2048 |
| mixed_precision_param | bfloat16 |
| mixed_precision_reduce | float32 |
| compile | enabled (inductor) |
| parallelism | dp_replicate=4 |

### 4.4 Hardware Configuration

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA H200 NVL |
| Memory | 139.80 GiB per GPU |
| Peak FLOPS | 9.890e+14 |
| Cluster | 4 GPUs |

---

## 5. Weight Sharing Strategies

### 5.1 Layer Sharing

**Configuration:**
```python
@dataclass
class LayerSharingConfig:
    enabled: bool = False
    layer_groups: Optional[List[List[int]]] = None
    lora_rank: int = 8
```

**Layer Group Examples:**
```yaml
# Maximum sharing (all layers share)
layer_groups: [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
lora_rank: 8  # 80.6% parameter reduction

# Pairwise sharing
layer_groups: [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]]
lora_rank: 8  # 6 unique weight sets

# Production config (20 layers → 8 groups)
layer_groups: [[0], [1,2,3], [4,5,6], [7,8,9], [10,11,12], [13,14,15], [16,17,18], [19]]
lora_rank: 256
```

**Performance Impact:** Layer sharing alone = 0.61x baseline (39% slower)

### 5.2 Head Sharing (Attention)

**Configuration:**
```python
@dataclass
class AttentionSharingConfig:
    enabled: bool = False
    qkv_sharing: Optional[List[List[str]]] = None
    head_sharing: bool = False
    grouping: int = 1
    rank: int = 8
    two_step: bool = False
```

**Performance Impact:** Attention sharing alone = 0.55x baseline (45% slower)

### 5.3 Factorized Embeddings

**Configuration:**
```python
@dataclass
class FactorizedEmbeddingConfig:
    enabled: bool = False
    d_emb: int = 256
    tie_output: bool = True
```

**Parameter Savings:**
| Component | Original | Factorized | Reduction |
|-----------|----------|------------|-----------|
| Qwen3 Embeddings (vocab=151,936, dim=1024, d_emb=256) | 155.6M | 39.2M | **75%** |

**Performance Impact:** Embedding only = **2.08x baseline** (fastest configuration)

---

## 6. Implementation Approaches

### 6.1 Compilation Strategies

| Strategy | Result |
|----------|--------|
| torch.compile (inductor) on full model | Works, ~25% MFU |
| torch.compile on isolated LoRA modules | **3x slower** than eager |
| torch.compile with fullgraph=True | No graph breaks |
| Custom Triton fused kernel | **8x slower** than cuBLAS |

### 6.2 Optimization Attempts

| Experiment | Micro-benchmark | E2E Result |
|------------|-----------------|------------|
| torch._grouped_mm batching | No improvement | - |
| Custom Triton fused kernel | 0.12x (8x slower) | - |
| torch.compile on LoRA modules | 0.32x (3x slower) | - |
| Fused autograd.Function | 0.88x (slower) | - |
| **torch.addmm fusion** | **1.16x** | **1.01x (1%)** |
| Remove LoRA scaling | - | Eliminates 1 kernel |
| **Pre-scale B weights** | **1.72x on LoRA path** | - |

### 6.3 torch.addmm Optimization Results

| Approach | GEMM | ELEM | COPY | Total | Time | Speedup |
|----------|------|------|------|-------|------|---------|
| Current (baseline) | 30 | 20 | 0 | 70 | 0.1790 ms | 1.00x |
| addmm basic | 30 | 10 | 10 | 70 | 0.1763 ms | 1.02x |
| **addmm pre-scaled B** | 30 | 0 | 10 | 60 | 0.1533 ms | **1.17x** |
| in-place add | 30 | 10 | 0 | 60 | 0.1593 ms | 1.12x |
| pre-allocated buffer | 30 | 10 | 0 | 60 | 0.1581 ms | 1.13x |
| addmm with out= | 30 | 0 | 10 | 60 | 0.1549 ms | 1.16x |

### 6.4 Scale Multiplication Fusion

| Approach | GEMM | ELEM | COPY | Total | Time | Speedup |
|----------|------|------|------|-------|------|---------|
| Two mm + scale | 20 | 10 | 0 | 40 | 0.0626 ms | 1.00x |
| **Pre-scaled B** | 20 | 0 | 0 | 30 | 0.0363 ms | **1.72x** |
| addmm with alpha | 20 | 0 | 10 | 40 | 0.0802 ms | 0.78x |

### 6.5 Optimized Implementation Results

**SharedLinearWithLoRA:**
| Metric | Old | New | Change |
|--------|-----|-----|--------|
| GEMM kernels | 30 | 30 | +0 |
| ELEM kernels | 20 | 0 | -20 |
| COPY kernels | 0 | 10 | +10 |
| Total time | 1.70 ms | 1.41 ms | **-17%** |
| Per-iteration | 0.178 ms | 0.153 ms | **1.16x** |

**SharedFeedForward:**
| Metric | Old | New | Change |
|--------|-----|-----|--------|
| GEMM kernels | 90 | 90 | +0 |
| ELEM kernels | 80 | 20 | -60 |
| COPY kernels | 0 | 30 | +30 |
| Total time | 5.28 ms | 4.59 ms | **-13%** |
| Per-iteration | 0.557 ms | 0.498 ms | **1.12x** |

**20-Layer FFN Stack (3 forward passes):**
| Metric | Old | New | Change |
|--------|-----|-----|--------|
| GEMM | 540 (18.55 ms) | 540 (19.69 ms) | +6% |
| ELEM | 480 (13.08 ms) | 120 (4.75 ms) | **-75%** |
| COPY | 0 (0.00 ms) | 180 (3.33 ms) | +180 |
| Total time | 31.84 ms | 27.97 ms | **1.14x** |

### 6.6 End-to-End Training Results

| Model | TPS (step 10) | TPS (step 20) | MFU | Memory |
|-------|---------------|---------------|-----|--------|
| Unshared | 157,700 | 191,866 | 27.92% | 69.54 GiB |
| Optimized Shared | 104,049-106,158 | 119,270-120,342 | 21.27-21.28% | 77.38 GiB |
| Old Shared | 103,264 | 118,119 | 21.06% | 77.33 GiB |

**E2E Speedup from optimization:**
- Old shared: 118,119 TPS
- New shared: 119,366 TPS
- **Speedup: 1.01x (1%)**

### 6.7 What Worked

1. **Pre-scaling LoRA B weights** - Eliminates scale multiply kernel (1.72x on LoRA path)
2. **torch.addmm fusion** - Fuses add into GEMM epilogue (1.16x micro-benchmark)
3. **Embedding factorization** - Reduces parameters 75%, improves throughput 2x
4. **torch.compile on full model** - No graph breaks, ~25% MFU

### 6.8 What Didn't Work

1. **Custom Triton kernels** - 8x slower than cuBLAS due to memory streaming
2. **torch.compile on isolated LoRA** - 3x slower than eager
3. **Fused autograd.Function** - No benefit without kernel fusion
4. **torch._grouped_mm** - Designed for MoE variable token counts, not same-input batching
5. **GEMM batching** - Reduces GEMM kernels 22% but elementwise kernels unchanged

### 6.9 Why E2E Improvement is Minimal

The inductor backend **automatically fuses** elementwise operations with matmuls in the compiled code path, achieving similar optimization to manual torch.addmm. This explains why:
- Micro-benchmark: 14-16% improvement
- E2E: Only 1% improvement

---

## 7. Critical Findings

### 7.1 Root Cause Analysis

**~160k us (23% of compute)** is spent in memory-bound elementwise ops:
- `CUDAFunctor_add (float)`: 39,861 us
- `CUDAFunctor_add (bf16)`: 20,191 us
- `BinaryFunctor Mul (bf16)`: 11,560 us
- `direct_copy_kernel`: 68,880 us
- `bfloat16_copy_kernel`: 33,641 us

These are the LoRA additions and intermediate tensor copies.

### 7.2 Theoretical vs Actual Overhead

**Theoretical FLOP overhead:**
- LoRA rank 256, dim 1024: LoRA matmul = 2 × 1024 × 256 = 524K FLOPs
- Base matmul = 1024² = 1M FLOPs
- Expected overhead: ~50%

**Actual overhead: +172%**

The discrepancy is due to:
1. Small matrix inefficiency (LoRA matmuls don't saturate GPU)
2. Excessive kernel launches (4 kernels per Linear+LoRA)
3. Memory traffic explosion (intermediate tensors)
4. No fusion (each operation is a separate kernel)

### 7.3 Scaling Observations

Larger dimensions **reduce** overhead:
- dim=512: +271% overhead
- dim=4096: +54% overhead

Larger batches **amortize** launch overhead:
- batch=1: +317% overhead
- batch=16: +157% overhead

### 7.4 CUTLASS Epilogue Fusion (Recommended Next Step)

To achieve significant speedup, implement CUTLASS epilogue fusion:
- Fuse `add` into GEMM epilogue at ~zero cost
- Kernel pattern: `nvjet_*_badd_*` instead of `nvjet_*_bz_*`

Expected improvements:
| Metric | Current | Expected | Improvement |
|--------|---------|----------|-------------|
| Forward kernel launches | 5 | 1 | 5x |
| Forward memory traffic | ~5MN | ~2MN | 2-2.5x |
| Forward overhead | +172% | +40-60% | 2-3x better |
| E2E step time | 1.33x baseline | ~1.1x baseline | ~20% faster |

---

## 8. Files and Paths Reference

### 8.1 Key Source Files

| File | Purpose |
|------|---------|
| `torchtitan/models/qwen3/model/weight_sharing.py` | SharedLinear, SharedFFN, SharedAttention |
| `torchtitan/models/qwen3/model/factorized_embedding.py` | FactorizedEmbedding, FactorizedOutput |
| `torchtitan/models/qwen3/job_config.py` | WeightSharingConfig |
| `torchtitan/models/qwen3/__init__.py` | Model variants |

### 8.2 Config Files

| Config | Description |
|--------|-------------|
| `250m_combined.toml` | Layer + attention + factorized embeddings |
| `250m_unshared.toml` | Baseline without sharing |
| `250m_layer_only.toml` | Layer sharing only |
| `250m_attention_only.toml` | Attention sharing only |
| `250m_embedding_only.toml` | Embedding factorization only |
| `qwen3_debug_weightshared.toml` | Debug config |

### 8.3 Benchmark Scripts

| Script | Purpose |
|--------|---------|
| `scripts/benchmark_lora.py` | Microbenchmark suite |
| `scripts/analyze_compile.py` | torch.compile debug |
| `scripts/test_addmm_fusion.py` | addmm fusion test |
| `scripts/test_addmm_optimizations.py` | Compare 6 fusion approaches |
| `scripts/test_optimized_lora.py` | Old vs new comparison |
| `scripts/profile_full_model.py` | 20-layer FFN profile |
| `scripts/profile_lora.py` | LoRA profiling |
| `scripts/profile_kernel_launches.py` | Kernel launch counts |

### 8.4 Dataset

| Dataset | Path |
|---------|------|
| fw_edu (train) | `/net/projects2/interp/Efficient-LLMs/data_8gpu/fineweb_edu_10bt_shuffled` |
| fw_edu (val) | `/net/projects2/interp/Efficient-LLMs/data_8gpu/fineweb_edu_10bt_shuffled` |
| c4_test | Used for debug runs |

---

## 9. Critical Analysis: Why torch.addmm Optimization Failed

### 9.1 TorchInductor Already Fuses Add Operations

**Key Discovery (Jan 28, 2026):** TorchInductor automatically fuses elementwise add operations with adjacent operations. This explains why manual torch.addmm optimization provided only 1% E2E improvement despite 14-16% micro-benchmark speedup.

#### Kernel Comparison: Single Transformer Block

| Metric | Unshared | Shared | Delta |
|--------|----------|--------|-------|
| **Eager Mode** | | | |
| Total kernels | 155 | 315 | +160 (+103%) |
| GEMM kernels | 35 | 135 | +100 (+286%) |
| Add kernels (unfused) | 20 | 35 | +15 |
| Time per block | 4,668 us | 6,615 us | **+42%** |
| **Compiled Mode** | | | |
| Total kernels | 95 | 230 | +135 (+142%) |
| GEMM kernels | 35 | 165 | +130 (+371%) |
| **Add kernels (unfused)** | **0** | **0** | **0** |
| Time per block | 2,774 us | 4,610 us | **+66%** |

**Critical Observation:** In compiled mode, TorchInductor fuses ALL add operations to zero unfused kernels. The torch.addmm optimization was redundant.

### 9.2 The Real Bottleneck: GEMM Kernel Proliferation

The shared model has **~100 additional GEMM kernels per transformer block** due to LoRA:

| Component | Matmuls per Block |
|-----------|-------------------|
| Layer LoRA (wq, wk, wv, wo) | 4 × 2 = 8 |
| Head offset LoRA (q, k, v) | 3 × 2 = 6 |
| Layer LoRA (w1, w2, w3) | 3 × 2 = 6 |
| **Total additional** | **20 matmuls** |

For 20 layers: 400 extra matmuls per forward pass.

### 9.3 Why Compiled Mode Has Higher Overhead

Counter-intuitively, the compiled model shows **higher relative overhead** (66%) compared to eager mode (42%):

| Mode | Unshared | Shared | Overhead |
|------|----------|--------|----------|
| Eager | 4,668 us | 6,615 us | 1.42x |
| Compiled | 2,774 us | 4,610 us | 1.66x |

This is because:
1. Inductor generates efficient fused Triton kernels for the simple unshared model (40 Triton kernels)
2. The shared model's LoRA pattern results in more, smaller kernels that don't benefit as much from fusion
3. The addmm operations don't batch as well as pure Triton-generated code

### 9.4 CombinedSharingAttention: The Unfused Head Offsets (Corrected Analysis)

The initial hypothesis that head offset adds weren't being fused was **incorrect**. Looking at the code:

```python
# In CombinedSharingAttention.forward()
xq = self.wq(x)  # SharedLinearWithLoRA (layer LoRA)
xk = self.wk(x)
xv = self.wv(x)

# Head offsets (THESE ARE FUSED BY INDUCTOR)
xq = xq + self.wq_head_offset(x)  # ← Inductor fuses this add
xk = xk + self.wk_head_offset(x)  # ← Inductor fuses this add
xv = xv + self.wv_head_offset(x)  # ← Inductor fuses this add
```

TorchInductor successfully fuses all elementwise adds into Triton kernels. The overhead comes purely from the additional GEMM operations.

---

## 10. Final Conclusions (Jan 28, 2026)

### 10.1 torch.addmm Optimization Reverted

The torch.addmm optimization has been **reverted** because:
- Inductor already fuses elementwise adds via Triton kernels
- torch.addmm introduces Memcpy DtoD overhead (~24 us) that exactly offsets the epilogue fusion benefit (~24 us saved)
- Net E2E improvement: **0%**

### 10.2 Root Cause: Memory-Bound LoRA Matmuls

The fundamental issue is that **LoRA matmuls are memory-bound**:

| Operation | Arithmetic Intensity | Bound | Time |
|-----------|---------------------|-------|------|
| Base GEMM | 687.9 FLOP/byte | COMPUTE | 75 us |
| LoRA A GEMM | 199.8 FLOP/byte | **MEMORY** | 10 us |
| LoRA B GEMM | 228.1 FLOP/byte | **MEMORY** | 22 us |
| Elementwise Add | - | MEMORY | 36 us |

H200 ridge point: 412.3 FLOP/byte. Operations below this are memory-bandwidth limited.

### 10.3 Why Batching/Stacking Doesn't Help

| Approach | Result | Reason |
|----------|--------|--------|
| torch._grouped_mm | No improvement | Runs ops separately, same memory traffic |
| Stack A matrices | 6.5% max savings | B matmuls cannot be stacked (different inputs) |
| torch.addmm | No improvement | Memcpy DtoD offsets epilogue fusion |
| Custom Triton kernel | 8x slower | Cannot match cuBLAS GEMM performance |

**Stacking validation results:**
- Full LoRA separate (6 matmuls): 65.41 us
- A-stacked + B separate: 55.62 us
- Savings: 9.79 us (15% of LoRA path, **6.5% of full operation**)
- B matmuls alone: 37.13 us (57% of LoRA time) - **cannot be optimized**

### 10.4 What Would Actually Help

A true solution requires a **fused LoRA GEMM kernel** that:

1. **Forward pass:**
   - Streams through stacked A and B matrices
   - Computes `x @ A_i.T @ B_i.T` without writing intermediate `h_i` to global memory
   - Keeps all intermediates in registers/shared memory
   - Achieves higher arithmetic intensity by processing multiple LoRAs per kernel

2. **Backward pass:**
   - Fused gradient computation for A, B, and x
   - Efficient gradient accumulation across LoRA modules

3. **Communication:**
   - Optimized gradient reduction for distributed training
   - Potentially fused with backward computation

**Estimated complexity:** This would require ~2-4 weeks of expert CUDA kernel development and is not feasible with Triton alone (our attempt was 8x slower than cuBLAS).

### 10.5 Practical Recommendation

**Accept the ~33-66% overhead** as the cost of weight sharing parameter efficiency, unless:
- A high-performance fused LoRA kernel becomes available (e.g., from NVIDIA or the community)
- The model is scaled up significantly (larger dimensions improve arithmetic intensity)

For the 250M model with current LoRA ranks (192-256), the overhead is largely irreducible without custom CUDA kernels.

---

## 11. Technical Deep Dive: Why Each Optimization Failed

### 11.1 torch.addmm Failure Analysis

```
Original:  GEMM (378 us) + elementwise add (182 us) = 560 us
addmm:     GEMM+add (414 us) + Memcpy DtoD (123 us) = 537 us
Net:       ~4% improvement (within noise)
```

The Memcpy DtoD is required because cuBLAS needs the bias tensor (`lora_out`) to be contiguous for epilogue fusion. The copy overhead negates the fusion benefit.

### 11.2 Triton Kernel Failure Analysis

Our custom Triton kernel achieved correct results but was 8x slower because:
- Triton's code generation cannot match cuBLAS's hand-tuned assembly
- cuBLAS uses architecture-specific optimizations (tensor cores, memory prefetching)
- The kernel read input `x` multiple times (once for base, once per LoRA)

### 11.3 Stacking Failure Analysis

```
3 separate A matmuls: 25.76 us (AI ~200, memory-bound)
1 stacked A matmul:   17.68 us (AI ~600, compute-bound)
Savings: 8 us

But B matmuls have different inputs (h_wq, h_wk, h_wv):
3 separate B matmuls: 37.13 us - CANNOT be stacked
```

The B matmuls dominate (57% of LoRA time) and cannot be optimized without a fused kernel.

### 11.4 Inductor Fusion Patterns (H200 NVL)

| Pattern | Meaning |
|---------|---------|
| `nvjet_sm90_tst_*_bz_*` | GEMM with beta=0 (no add fusion) |
| `nvjet_sm90_tst_*_badd_*` | GEMM with beta add (fused add) |
| `triton_poi_fused_*` | Inductor-generated fused pointwise kernel |
| `Memcpy DtoD` | Device-to-device copy (contiguity requirement) |

---

## 12. Efficiency Analysis

### 12.1 GPU Utilization

| Metric | Value |
|--------|-------|
| Actual time per SharedLinearWithLoRA | 150 us |
| Theoretical memory-bound minimum | 55 us |
| **Efficiency** | **37%** |
| Overhead beyond roofline | 95 us (63%) |

### 12.2 Overhead Breakdown

| Source | Estimated Impact |
|--------|------------------|
| Kernel launch overhead (~8 us × 4 kernels) | ~32 us |
| Memory not fully saturated (wave quantization) | ~30 us |
| Cache effects and synchronization | ~33 us |

### 12.3 Per-Kernel Overhead

| Metric | Value |
|--------|-------|
| Theoretical LoRA A time at peak | 2.17 us |
| Actual incremental cost | 11.52 us |
| **Per-kernel overhead** | **~9 us** |

---

*Report finalized 2026-01-28. torch.addmm optimization reverted. Weight sharing overhead (~33-66%) is largely irreducible without custom fused CUDA kernels for LoRA operations.*
