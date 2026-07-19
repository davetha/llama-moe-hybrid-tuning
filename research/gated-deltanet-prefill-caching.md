# Prefill / state caching for Gated DeltaNet hybrid MoE models

*Research notes on "can you avoid re-prefilling the whole context every turn?" for models like
Qwen3-Next / Qwen3.5 / Qwen3.6 (Gated DeltaNet linear-attention + minority full-attention + MoE).
Compiled 2026-07 from primary sources — issue/PR numbers are the reliable anchors; verify against
live threads before quoting exact benchmark numbers.*

## TL;DR

- **The hard limit is real.** Gated DeltaNet's "memory" is a single fixed-size matrix updated *in
  place* over the whole sequence — a lossy running summary. You **cannot** drop/edit a *middle* span
  of tokens like you can with a KV cache. **Any prefix change forces recompute from the divergence
  point to the end.** No engine beats this lower bound.
- **The realistic optimum is a solved problem:** *checkpoint the recurrent state every N tokens;
  on a prefix change, resume from the nearest checkpoint before the divergence and recompute only the
  tail.*
- **It's already implemented** — in llama.cpp (context checkpoints, with recent hybrid fixes) and in
  SGLang (`MambaRadixCache`). vLLM's version is new/experimental and only helps long prefixes.
- **The biggest lever is on your side:** keep a **stable prefix and only mutate the tail** during
  context compaction, so the divergence point stays *late*. If compaction rewrites early history, no
  cache helps.

## What Gated DeltaNet is (why it's different from attention)

Lineage: linear attention → DeltaNet (delta rule = online error-correction write) → Mamba2/GLA
(gating = data-dependent forgetting) → **Gated DeltaNet** (fuses both):

```
S_t = α_t · S_{t-1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ
```
- `α_t ∈ (0,1)`: gate/decay (global forgetting, from Mamba2)
- `β_t ∈ (0,1)`: delta write strength (targeted key overwrite, from DeltaNet)
- State `S`: a **fixed-size matrix per head** (`d_k × d_v`, independent of sequence length) — a
  "fast-weight" associative memory, updated in place, depends on *all* tokens 0..t.

Paper: Yang, Kautz, Hatamizadeh, "Gated Delta Networks: Improving Mamba2 with Delta Rule",
[arXiv:2412.06464](https://arxiv.org/abs/2412.06464) (ICLR 2025).

**Qwen3-Next family layout:** hybrid 3:1 — 3 Gated DeltaNet layers per 1 full-attention layer, on an
ultra-sparse MoE. The GDN layers give near-linear, KV-free long-context; the minority full-attention
layers preserve exact recall. Qwen3.6-35B-A3B is this family.
([vLLM Qwen3-Next blog](https://vllm.ai/blog/2025-09-11-qwen3-next))

**Why a normal KV cache trick doesn't apply:** a KV cache stores one vector *per token* — you can keep
token 0..p and q..end and drop the middle. The GDN state is *one tensor for the whole prefix*, with no
per-token addressability. There is no exact way to "subtract" middle tokens from it.

## llama.cpp

**How state is stored** (`src/llama-memory-*`):
- `llama_kv_cache` — per-token `cache_k`/`cache_v` for full-attention layers (droppable/shiftable)
- `llama_memory_recurrent` — fixed-size `cache_r` (conv/shift) + `cache_s` (recurrent state) for the
  GDN/Mamba layers (one slot per sequence, **not** per-token)
- `llama_memory_hybrid` — wraps both; routes layers via `hparams.is_recr(il)`

**Why `--cache-reuse` / context-shift is disabled:** `llama_memory_recurrent::seq_rm` only supports
removing a *suffix* (rollback) or a whole sequence — it rejects interior-range erasure because the
state is a non-linear recurrence with no "subtract middle tokens" op. Hence the log line
`cache_reuse is not supported by this context, it will be disabled`. **Fundamental, not lazy.**

**What llama.cpp *does* offer — context checkpoints** (snapshot full hybrid state at positions, resume
from nearest). Recent, on-point work:
- Discussion [#19264](https://github.com/ggml-org/llama.cpp/discussions/19264) — the "partial reuse via
  checkpointing" request; maintainer: checkpointing *is* the intended mechanism.
- [#22384](https://github.com/ggml-org/llama.cpp/issues/22384) — **fixes checkpoint-restore for
  hybrid/recurrent** (the `pos_min` vs `pos_max` search bug meant checkpoints *never* restored for
  recurrent models; also lowers the min-token threshold). Reported: Qwen3.6-27B turn-2 prefill
  **~11 s → 115 ms**.
- PR [#22929](https://github.com/ggml-org/llama.cpp/pull/22929) — more checkpoint-creation fixes
  (checkpoints at chat-message boundaries — relevant to compaction).
- PR [#17428](https://github.com/ggml-org/llama.cpp/pull/17428) / `--checkpoint-every-nb` (PR #20087) —
  multiple checkpoints during prefill instead of one at the end.
- Regression to avoid: [#24055](https://github.com/ggml-org/llama.cpp/issues/24055) (checkpoints erased,
  build ~b9354). Pain reports: [#20225](https://github.com/ggml-org/llama.cpp/issues/20225),
  [#21831](https://github.com/ggml-org/llama.cpp/issues/21831)
  (`forcing full prompt re-processing … hybrid/recurrent memory`).
- Arch support shipped via PRs #19125 (`ggml_gated_delta_net`), #19408 (Qwen3-Next), #19435 (Qwen3.5).
- **SYCL gap:** the fused Gated DeltaNet kernel is **not implemented on SYCL** → slow unfused path
  (`fused Gated Delta Net (chunked) not supported, set to disabled`). Reports say **Vulkan works where
  SYCL is slow/broken on Arc** ([#20338](https://github.com/ggml-org/llama.cpp/issues/20338),
  [#20423](https://github.com/ggml-org/llama.cpp/issues/20423)).
- Manual escape hatch: `POST /slots/{id}/save|restore` persists full KV+recurrent state to disk.

**Bottom line:** get on a build **after** #22384/#22929, raise checkpoint count / lower interval, align
checkpoints to compaction boundaries. That gives the exact-prefix + resume win. Interior edits still
force tail-recompute (fundamental).

## vLLM

- **Arch is shipped:** `Qwen3_5MoeForConditionalGeneration` (module `qwen3_5`) — this is what
  Qwen3.6-35B-A3B loads under. Landed ~v0.10.2. Uses FLA Triton kernels + a hybrid KV-cache manager
  ("full-attention + one efficient type") + CUDA-graph decode + MTP.
- **Prefix caching for GDN is new/experimental** — tracker
  [#26201](https://github.com/vllm-project/vllm/issues/26201); GDN PRs
  [#26807](https://github.com/vllm-project/vllm/pull/26807) (align mode),
  [#30877](https://github.com/vllm-project/vllm/pull/30877) (all mode). Even when on, a coarse
  **528-token block** means prompts **< ~528 tokens get 0% hit rate**
  ([#40696](https://github.com/vllm-project/vllm/issues/40696)) — helps long shared prefixes only.
- **Feature gaps:** no batch-invariant GDN path ([#42960](https://github.com/vllm-project/vllm/issues/42960)).
- **MI210 (CDNA2 / gfx90a) is the weak link.** AMD's day-0 support for this arch is **CDNA3 only**
  (MI300X/MI325X/MI355X). There's an **open, orphaned** MI210 crash for this exact arch —
  [#25030](https://github.com/vllm-project/vllm/issues/25030) ("arange's range must be a power of 2" in
  the Triton unified-attention kernel), same *class* as the consumer-Arc hang. No AITER kernels for
  gfx90a (baseline Triton only). RCCL/TP is fine on MI210; the *kernels* are the risk.

## SGLang

- **`MambaRadixCache`** — the most mature prefix-caching + state-checkpointing for Mamba/linear state
  today. Radix tree stores/forks a checkpoint copy of the recurrent state at prefix nodes; separate LRU
  eviction for state vs KV; atomic state transfer. ([PyTorch blog](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/))
- Interval prefill checkpointing proposed in [#22326](https://github.com/sgl-project/sglang/issues/22326).

## Research frontier (for context, not for you to build)

- **Marconi** ([arXiv:2411.19379](https://arxiv.org/html/2411.19379v1)) — prefix cache for hybrid LLMs:
  judicious admission (only checkpoint states seen ≥ twice) + FLOP-aware eviction. 4.5–34× hit rate,
  up to 71% lower P95 TTFT. Research prototype.
- **Sparse Prefix Caching** ([arXiv:2605.05219](https://arxiv.org/pdf/2605.05219)) — budget-optimal
  checkpoint placement. Algorithmic.
- Enabling kernels with `initial_state`→`final_state` (the resume primitive):
  [flash-linear-attention](https://github.com/fla-org/flash-linear-attention),
  [FlashInfer `gdn_prefill`](https://docs.flashinfer.ai/generated/flashinfer.gdn_prefill.chunk_gated_delta_rule.html),
  [QwenLM FlashQLA](https://github.com/QwenLM/FlashQLA).
- **Editing the middle of a recurrent state** (approximate state patching) = genuine open research; no
  production method. Out of scope.

## Engine comparison

| Engine | Arch support | Prefix caching for GDN | MI210 (gfx90a) |
|---|---|---|---|
| **llama.cpp ROCm** | ✅ shipped | ✅ checkpoints (post-#22384/#22929) | ✅ safe, arch-agnostic |
| **vLLM** | ✅ `Qwen3_5Moe…` | ⚠️ new/experimental, ≥528-tok prefixes only | ❌ orphaned crash #25030, CDNA3-only |
| **SGLang** | ✅ | ✅ most mature (`MambaRadixCache`) | ⚠️ unverified on MI210 |

## Recommendations

1. **Free, do first:** compaction that preserves a **stable prefix** and mutates only the **tail**.
   Keeps the divergence point late so any cache actually has something to reuse. Helps every engine.
2. **B580 / llama.cpp now:** upgrade to a **post-#22384/#22929** build for checkpoint reuse (the
   ~11 s→115 ms win) — validate output correctness first (recent master has had a Qwen3.6-Q4_K
   correctness regression on Battlemage; pairs with an oneDNN-flash-attention rebuild). Consider Vulkan
   vs SYCL on Arc (fused GDN kernel missing on SYCL).
3. **MI210 arrival:** start with **llama.cpp ROCm** (safe, whole model on 64 GB GPU, checkpoints work,
   cold prefill is fast anyway so re-prefill pain mostly evaporates). Treat **vLLM / SGLang** as the
   throughput/prefix-caching upgrade *once gfx90a is confirmed working* — and note MI210+vLLM (#25030)
   is the orphaned gap where contribution would matter.

**Net:** not a research project, not a from-scratch patch. "Newer llama.cpp build + stable-prefix
compaction" now; "llama.cpp-ROCm first, vLLM/SGLang later" on the MI210s.
