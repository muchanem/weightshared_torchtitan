# Combined Sharing Training Investigation

## Date: 2026-03-29

## Problem
Combined sharing runs show worse results than expected. The user suspects a bug
based on gradient norm patterns and loss values.

---

## Key Findings

### 🔴 BUG 1 (HIGH): `grouping` parameter IGNORED in CombinedSharingAttention

**File:** `torchtitan/models/qwen3/ws_modules/weight_sharing.py`
**Lines:** 1169, 1176-1199, 1288-1326

In `CombinedSharingAttention`, the `grouping` parameter from `AttentionSharingConfig`
is stored (`self.grouping`) but **never used** in `forward()`. Compare:

- `SharedAttention` (standalone): Uses `broadcast_add(q_base, wq_offset, g=self.grouping)`
  which groups attention heads and broadcasts from grouped to full dimension.
- `CombinedSharingAttention`: Does `xq = xq + wq_offset` — simple addition, no grouping!

**Impact:** The attention head grouping feature has NO EFFECT in combined mode.
All head offsets are independent (no parameter savings from grouping). This means
the model has more parameters than intended in the attention sharing component,
but those parameters are NOT leveraging the grouped structure that made standalone
attention sharing effective.

### 🔴 BUG 2 (HIGH): `qkv_sharing` NOT implemented in CombinedSharingAttention

**File:** `torchtitan/models/qwen3/ws_modules/weight_sharing.py`
**Line:** 1172

```python
self.qkv_sharing = attention_config.qkv_sharing
```

This is stored but **never used** in `forward()`. In standalone `SharedAttention`,
QKV sharing ties Q/K/V projections via `TiedLinear`, saving parameters.

### 🟡 BUG 3 (MEDIUM): Inconsistent LoRA scaling

**Files:**
- `weight_sharing.py` lines 175-176: `ScaledLoRAModule.forward()` does NO scaling
- `batched_lora.py` line 91: `BatchedLoRAModule.forward()` applies `self.scale = alpha/rank`

When `use_batched_lora=True` (for FFN), outputs are scaled by alpha/rank.
When using default sequential mode (attention), NO scaling is applied.
This inconsistency may affect the balance between attention and FFN LoRA contributions.

### 🟡 BUG 4 (MEDIUM): TP plan may not handle weight sharing modules

**File:** `torchtitan/models/qwen3/parallelize.py` lines 264-305

The tensor parallelism plan assumes standard layer structure (`attention.wq` → `Linear`),
but weight sharing blocks use `SharedLinearWithLoRA` wrappers. There's NO safety check
for TP + weight sharing (unlike PP which has an explicit rejection).

### 🔴 BUG 5 (DORMANT): Missing import for grouped_attention_head_offset_forward

**File:** `weight_sharing.py` lines 449, 465, 1299, 1315

Called but never imported. Crashes if `use_grouped_mm=True` (default is False).

---

## Training Dynamics Comparison

### Combined Sharing 100M (torchtitan, colm_100m_1_standard):
| Step | Loss | Grad Norm |
|------|------|-----------|
| 1 | 12.35 | 7.71 |
| 100 | 10.29 | 1.78 |
| 500 | ~5.5 | ~1.5 |
| 1000 | ~4.2 | ~1.0 |
| 3000 | ~3.2 | ~0.5 |
| 6300 | ~2.7 | ~0.38 |

### Meta Lingua 100M emb_rank_100pct (baseline, no sharing):
| Step | Loss | Grad Norm |
|------|------|-----------|
| 100 | 10.00 | 1.58 |
| 500 | 5.67 | 2.77 |
| 1000 | 4.55 | 1.43 |
| 3000 | 2.83 | 0.42 |
| 76300 | 2.02 | 0.52 |

### Meta Lingua 100M LAYERWISE_v3/100M_base (layer sharing only):
| Step | Loss | Grad Norm |
|------|------|-----------|
| 100 | 9.97 | 1.22 |
| 500 | 5.65 | 1.58 |
| 1000 | 4.53 | 1.00 |
| 3000 | 2.86 | 0.33 |

**Note:** Direct comparison is limited because combined sharing uses:
- Qwen3 tokenizer (vocab=151,936) vs LLaMA tokenizer (vocab=128,256)
- `hq_data_20bt` dataset vs the Meta Lingua dataset
- Different framework (torchtitan vs Meta Lingua)

---

## Recommendations

### Priority 1: Fix `grouping` in CombinedSharingAttention
Implement proper `broadcast_add` for head offset computation, matching
`SharedAttention`'s behavior. This is likely the biggest source of
performance degradation — the attention sharing is effectively disabled.

### Priority 2: Implement `qkv_sharing` in CombinedSharingAttention
Add `TiedLinear` support for Q/K/V tying when `qkv_sharing=True`.

### Priority 3: Unify LoRA scaling
Either apply scaling consistently in both sequential and batched modes,
or document the design decision clearly.

### Priority 4: Apples-to-apples comparison
Run a non-sharing baseline ON THE SAME framework (torchtitan + Qwen3 + hq_data_20bt)
to isolate whether the performance gap is from sharing or from framework differences.
