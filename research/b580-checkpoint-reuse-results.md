# B580 checkpoint reuse: WORKING and DEPLOYED — the earlier gibberish was oneDNN, not the quant kernel

*Empirical validation on the reference box (Arc B580 / Battlemage-G21, `8086:e20b`) of the theory in
[gated-deltanet-prefill-caching.md](gated-deltanet-prefill-caching.md). Updated 2026-07-19. Model:
Qwen3.6-35B-A3B Q4_K, `-ngl 999 -ot exps=CPU -c 262144 --no-mmap -b 4096 -ub 4096`.*

**Bottom line:** hybrid checkpoint reuse now runs correctly in production on the B580. An earlier build
gibberished and was rolled back — but the cause was **not** the checkpoint code and **not** an
unfixable quant bug. It was **oneDNN**. Rebuilding with oneDNN *disabled* gives correct output **and**
the full checkpoint-reuse win.

## The two-cause story (why the first attempt failed)

The gibberish ([#25708](https://github.com/ggml-org/llama.cpp/issues/25708)) had two independent causes,
and we initially only knew about the first:

1. **A real quant-kernel bug** in the SYCL reorder path (introduced by
   [#25063](https://github.com/ggml-org/llama.cpp/pull/25063): an incorrect row calculation made the
   `dmmv` kernel overwrite the same row). **This was fixed upstream in
   [#25690](https://github.com/ggml-org/llama.cpp/pull/25690)** (merged 2026-07-17), and confirmed
   clean on Battlemage (Arc B70) by another user with this exact model.

2. **oneDNN (`-DGGML_SYCL_DNN=ON`).** Our first rebuild *already contained* the #25690 fix (it was 11
   non-SYCL commits past the merge) and **still gibberished** — because we had compiled in oneDNN, which
   routes matmul through a **separate** `DnnlGemmWrapper` GEMM path that #25690 never touched, and which
   is **broken on the B580 (G21)**. It's the same oneDNN that corrupts the `-fa` flash-attention path on
   this die. The confirmed-clean Battlemage build did not use oneDNN.

**Fix:** rebuild **without** oneDNN — `-DGGML_SYCL_F16=ON`, drop `-DGGML_SYCL_DNN=ON` and the
`dnnl-devel` package. Native SYCL kernels are correct on this card. Recipe:
[../docker/Dockerfile.sycl-checkpoint](../docker/Dockerfile.sycl-checkpoint).

## Result 1 — correctness (validated with REAL prompts, not toy ones)

Every real prompt is coherent on the oneDNN-off build — including the exact conditions that gibberished
before: multi-turn (the specific #25690 failure mode), long code, step-by-step reasoning, and large
(6K / 15K-token) context. Zero CJK/garbage runs. The live production endpoint was re-verified after
deploy (coherent multi-turn answer, `cjk=0`).

## Result 2 — checkpoint reuse is live, and the recompute is CONSTANT

Cold turn re-prefills the whole prompt; a same-prefix follow-up resumes from the nearest checkpoint and
recomputes only a **fixed ~4,097-token segment** — *independent of total context size*:

| Context (cold prefill) | Turn-2 reprocess (same prefix) | Wall-clock saving |
|---|---|---|
| 6,054 tokens | **4,097 tokens** | 1.5× |
| 14,894 tokens | **4,097 tokens** | 3.4× (22 s → 6.5 s) |
| 9,258 tokens (live prod) | **4,098 tokens** | verified on :8084 |

Identical ~4,097 at 6K and 15K → the win scales: **~12× at 50K, ~24× at 100K**. This is the re-prefill
win for agentic / long-doc workloads, now real on the B580.

## Result 3 — prefill speed

~650–680 t/s on multi-thousand-token prompts (matches/beats the prior `:f16` build). Generation
unchanged at ~38 t/s (memory-bandwidth-bound on the CPU experts).

## Lessons (still the important part)

1. **Validate correctness with REAL, long, agentic prompts — not toy prompts.** Short prompts happened
   to dodge the bug; the actual workload triggered it. A "healthy" server that emits nonsense is the
   worst failure mode. This is what caught the bad build before, and what confirmed the good one.
2. **A merged upstream fix in your build is not proof your build is correct** — a *different* enabled
   code path (here, oneDNN) can still be broken on your specific hardware. Isolate by build option.
3. **On Battlemage-G21, build SYCL without oneDNN.** Both its GEMM and its flash-attention paths corrupt
   output on this die; the native SYCL kernels are correct and fast.

## Status

**Deployed.** Reference box runs the oneDNN-off build in production (GPU hybrid), with checkpoint reuse
active. #25708-class gibberish resolved for this card via oneDNN-off + the #25690 fix.
