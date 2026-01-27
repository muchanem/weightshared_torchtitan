# Weight-Shared Model Performance Analysis

**Date**: January 27, 2026
**Hardware**: NVIDIA H200 NVL (139.80 GiB)
**PyTorch**: 2.11.0.dev20260125+cu130

## Executive Summary

The weight-shared Qwen3 250M model is **29% slower** than the unshared baseline (TPS: 122,145 vs 171,330). The primary bottleneck is **excessive kernel launches** from LoRA adapters, with:
- 3.65x more matmul operations (1545 vs 423)
- 3.23x more kernel launches (1625 vs 503 cuLaunchKernelEx)
- 2.27x more Command Buffer Full overhead (207ms vs 91ms)

---

## 1. End-to-End Training Performance

### 1.1 Configuration Comparison

| Configuration | TPS (step 10) | Params | Memory | vs Unshared |
|--------------|---------------|--------|--------|-------------|
| Unshared | 171,330 | 294M | 58.98 GiB | 1.00x |
| **Combined (all sharing)** | **122,145** | 303M | 66.46 GiB | **0.71x** |
| Layer only | 119,479 | 502M | 69.06 GiB | 0.70x |
| Attention only | 133,520 | 545M | 67.82 GiB | 0.78x |
| Embedding only | 150,910 | 345M | 62.87 GiB | 0.88x |

### 1.2 Key Observations

1. **Layer sharing** causes the most overhead (~30% slowdown alone)
2. **Attention sharing** adds ~22% overhead
3. **Embedding factorization** adds ~12% overhead
4. **Combined** overhead (29%) is similar to layer-only, suggesting layer sharing dominates

The non-additive nature of overheads suggests some components share common bottlenecks (kernel launch overhead, memory bandwidth).

---

## 2. Trace Analysis

### 2.1 Temporal Breakdown (from PyTorch Profiler)

| Metric | Unshared | Shared | Change |
|--------|----------|--------|--------|
| Total tracked time | 2,479 ms | 3,440 ms | +39% |
| ProfilerStep#9 (GPU) | 224 ms | 326 ms | +45% |
| aten::mm calls | 423 | 1,545 | +265% |
| cuLaunchKernelEx calls | 503 | 1,625 | +223% |
| cudaMemsetAsync calls | 393 | 1,042 | +165% |

### 2.2 Top Time Differences (Shared - Unshared)

| Operation | Unshared (ms) | Shared (ms) | Diff (ms) | Change |
|-----------|---------------|-------------|-----------|--------|
| aten::mm | 32.01 | 148.45 | +116.45 | **+364%** |
| CompiledFunctionBackward | 105.43 | 221.31 | +115.88 | +110% |
| Command Buffer Full | 90.98 | 206.65 | +115.66 | +127% |
| cuLaunchKernelEx | 17.03 | 97.76 | +80.73 | +474% |

### 2.3 Root Cause Analysis

The shared model has **3.65x more matmul operations** due to:
- Each `SharedLinearWithLoRA` layer: 3 matmuls (base + LoRA A + LoRA B) vs 1 matmul
- Per transformer block: 4 attention projections + 3 FFN projections = 7 SharedLinear layers
- 20 layers total = 140 LoRA modules = **280 extra matmuls per forward**
- Backward pass adds another ~560 matmuls for gradients

The **474% increase in kernel launches** creates significant overhead from:
- CUDA driver API calls (~98ms vs ~17ms)
- Command buffer management (~207ms vs ~91ms)

---

## 3. Microbenchmark Results

### 3.1 Basic Benchmark (batch=4, seq=2048, dim=1024, rank=256)

| Operation | nn.Linear | SharedLinearWithLoRA | Overhead |
|-----------|-----------|---------------------|----------|
| Forward | 0.029 ms | 0.077 ms | **+169%** |
| Forward+Backward | 0.129 ms | 0.262 ms | **+103%** |
| SwiGLU FFN | 0.308 ms | 0.599 ms | **+94%** |

### 3.2 LoRA Rank Sweep (Forward Only)

| Rank | Overhead |
|------|----------|
| 8 | +142% |
| 16 | +143% |
| 32 | +142% |
| 64 | +142% |
| 128 | +151% |
| 256 | +170% |
| 512 | +212% |

**Observation**: Overhead is dominated by kernel launch cost, not compute. Even rank=8 has +142% overhead.

### 3.3 Batch Size Sweep (rank=256)

| Batch | Overhead |
|-------|----------|
| 1 | +320% |
| 2 | +185% |
| 4 | +169% |
| 8 | +164% |
| 16 | +154% |

**Observation**: Larger batches amortize launch overhead better.

### 3.4 Dimension Sweep (rank=256, batch=4)

| Dim | Overhead |
|-----|----------|
| 512 | +257% |
| 768 | +212% |
| 1024 | +165% |
| 2048 | +87% |
| 4096 | +48% |

**Observation**: Larger dimensions hide launch overhead with more compute. At dim=4096, overhead drops to 48%.

### 3.5 Component-Level Microbenchmarks

Isolated benchmarks for each weight sharing component (batch=4, seq=2048, vocab=151936):

| Component | Standard | Shared | Overhead | Notes |
|-----------|----------|--------|----------|-------|
| **Embedding (lookup)** | 0.008 ms | 0.030 ms | **+264%** | 40% param reduction |
| **Output Projection** | 4.296 ms | 2.935 ms | **-32%** | FASTER (smaller matmul) |
| **Linear + LoRA** | 0.034 ms | 0.088 ms | **+158%** | 3 matmuls vs 1 |
| **FFN (SwiGLU)** | 0.310 ms | 0.614 ms | **+98%** | 9 matmuls vs 3 |
| **Attention Projections** | 0.093 ms | 0.365 ms | **+293%** | 18 matmuls vs 4 |

**Key Insights:**

1. **Attention projections with head offsets are the worst** (+293%)
   - 4 base projections (Q/K/V/O) become 18 matmuls with layer LoRA + head LoRA
   - This explains why attention-only ablation shows 22% end-to-end slowdown

2. **Factorized output projection is actually FASTER** (-32%)
   - Large vocab (151936) makes standard output very expensive
   - Factorization: (batch, seq, dim) @ (dim, d_emb) @ (d_emb, vocab) is faster than (batch, seq, dim) @ (dim, vocab)
   - This partially offsets embedding overhead

3. **Embedding lookup overhead is high** (+264%)
   - Small operation (0.008ms baseline) means kernel launch dominates
   - Factorization adds a linear projection after lookup

4. **FFN overhead is moderate** (+98%)
   - 3 SharedLinearWithLoRA = 9 matmuls vs 3 standard matmuls
   - Larger matrices (hidden_dim=3072) help amortize launch overhead

---

## 4. Bottleneck Summary

### 4.1 Primary Bottlenecks (in order of impact)

1. **Kernel Launch Overhead** (~116ms additional)
   - 3.65x more matmul operations
   - Each SharedLinearWithLoRA = 4 kernels (base mm, LoRA A mm, LoRA B mm, add)
   - CUDA driver overhead scales with launch count

2. **Memory Traffic** (~45ms additional)
   - Each intermediate tensor written to DRAM
   - LoRA outputs stored, then added to base output
   - 2-3x memory traffic per linear layer

3. **Command Buffer Saturation** (~116ms additional)
   - GPU command buffer fills faster with more launches
   - Causes CPU-GPU synchronization stalls

### 4.2 Why torch.compile Doesn't Help

- torch.compile captures the full graph (no breaks)
- Compiled code is actually **slower** for isolated LoRA modules (0.32x speedup)
- Inductor generates multiple small kernels, not fused operations
- cuBLAS matmuls are already highly optimized

---

## 5. Optimization Opportunities

### 5.1 High Impact (Estimated 20-40% improvement)

1. **Batch LoRA Operations**
   - Group multiple LoRA A projections into single batched matmul
   - Group multiple LoRA B projections into single batched matmul
   - Reduces kernel count by ~50%

2. **CUDA Graphs**
   - Capture entire forward/backward as CUDA graph
   - Eliminates CPU-side kernel launch overhead
   - Requires static shapes (compatible with training)

3. **Use Larger Model Dimensions**
   - At dim=4096, overhead drops to 48%
   - Trade-off: more memory, but better efficiency

### 5.2 Medium Impact (Estimated 10-20% improvement)

1. **Elementwise Fusion**
   - Fuse `base_output + lora_output * scale` into single kernel
   - Triton kernel for base+LoRA addition
   - Saves 1 kernel per SharedLinear

2. **Reduce LoRA Rank Where Possible**
   - Lower ranks don't significantly reduce overhead
   - But smaller rank = less memory traffic

### 5.3 Not Recommended

1. **Custom Triton GEMM Kernels**
   - Attempted: 8x slower than cuBLAS
   - cuBLAS is too optimized to beat for pure matmuls
   - Triton better suited for memory-bound ops

2. **torch.compile mode="max-autotune"**
   - Tested: No significant improvement
   - Problem is operation count, not individual kernel speed

---

## 6. Trace Files

Profiler traces saved to:
- `outputs/traces/250m_unshared/iteration_10/rank0_trace.json`
- `outputs/traces/250m_shared/iteration_10/rank0_trace.json`
- `outputs/traces/250m_layer_only/iteration_10/rank0_trace.json`
- `outputs/traces/250m_attention_only/iteration_10/rank0_trace.json`
- `outputs/traces/250m_embedding_only/iteration_10/rank0_trace.json`

View in Chrome: `chrome://tracing`

---

## 7. Recommendations

### Short-term (Quick Wins)
1. Implement CUDA graphs for the training loop
2. Increase batch size where memory permits
3. Consider larger model dimensions if parameter budget allows

### Medium-term (Requires Development)
1. Batch multiple LoRA projections together
2. Implement Triton kernel for elementwise fusion only
3. Profile with NVIDIA Nsight for deeper analysis

### Long-term (Research)
1. Explore sparse LoRA structures
2. Investigate weight quantization to reduce memory traffic
3. Consider alternative decomposition methods (e.g., Kronecker products)

---

## Appendix: Analysis Scripts

| Script | Purpose |
|--------|---------|
| `scripts/analyze_traces.py` | Parse and compare profiler traces |
| `scripts/analyze_traces_hta.py` | HTA-based trace analysis |
| `scripts/benchmark_lora.py` | LoRA overhead microbenchmarks |
| `scripts/benchmark_components.py` | Component-level microbenchmarks (embedding, attention, FFN) |
| `scripts/analyze_compile.py` | torch.compile behavior analysis |

---

## Appendix: Reproducing Results

```bash
# Generate traces
WANDB_MODE=disabled uv run torchrun --nproc_per_node=1 -m torchtitan.train \
    --job.config-file torchtitan/models/qwen3/train_configs/250m_combined.toml \
    --profiling.enable-profiling --profiling.save-traces-folder traces/250m_shared \
    --profiling.profile-freq 10 --training.steps 15 --checkpoint.no-enable \
    --parallelism.data-parallel-replicate-degree 1

# Run microbenchmarks
uv run scripts/benchmark_lora.py
uv run scripts/benchmark_components.py

# Analyze traces
uv run scripts/analyze_traces.py outputs/traces/250m_unshared/iteration_10/rank0_trace.json \
    outputs/traces/250m_shared/iteration_10/rank0_trace.json
```
