# B580 rebuild results: checkpoint reuse works, but the build gibberishes — NOT deployable (yet)

*Empirical validation on the reference box (Arc B580 / Battlemage-G21, `8086:e20b`) of the theory in
[gated-deltanet-prefill-caching.md](gated-deltanet-prefill-caching.md). 2026-07. Model:
Qwen3.6-35B-A3B Q4_K, `-ngl 999 -ot exps=CPU -c 262144 --no-mmap -b 4096 -ub 4096`.*

**Bottom line:** a recent llama.cpp build (commit `571d0d5`) *does* give working hybrid checkpoint
reuse and ~50% faster prefill — but on this card it **also produces gibberish on real/longer requests**
(the Battlemage-Q4_K correctness regression, upstream
[#25708](https://github.com/ggml-org/llama.cpp/issues/25708)), so it was **rolled back**. The
checkpoint-reuse win is real but **gated behind the upstream correctness fix**. Documented here so the
next person doesn't repeat the deploy.

Build recipe: [../docker/Dockerfile.sycl-checkpoint](../docker/Dockerfile.sycl-checkpoint). We A/B'd
against the older build as a **separate image**.

## Result 1 — checkpoint reuse WORKS (mechanically)

Old build re-prefilled the **whole context every turn** (the #22384 restore bug). New build resumes
from the nearest checkpoint. Log: `selected slot by LCP similarity, sim_best=0.998`.

| Request | Prefilled tokens |
|---|---|
| req 1 (cold, ~9.3K prompt) | 9,360 |
| req 2 (same prefix, new question) | **4,100** |

Recompute is a **constant ~4,100 tokens** (≈ the `-ub 4096` checkpoint segment) *regardless of context
size* — so on a 50K context a new turn would recompute ~4K (~4 s) instead of all 50K. That is the
~10–20× re-prefill win we were after **mechanically**. The problem is correctness (below), so we can't
use it yet.

## Result 2 — base build is faster (mechanically)

| Build | prefill (t/s) |
|---|---|
| older (`e8f19cc`) | ~637 |
| new (`571d0d5`, no `-fa`) | ~950 |

## Result 3 — CORRECTNESS FAILURE — the whole build gibberishes on real requests

This is the blocker, and the trap we fell into:

- **Short/simple prompts passed** correctness (`17×23=391`, one-sentence answers were coherent). This
  gave a **false green light**.
- **Real, longer requests (agentic / code / thinking) produced gibberish** in production, e.g.
  `"/settingsブログ村 相信在 Spe...xnxxaf小额 bipartisan..."`. The server health check still says "ok"
  (the process is up) — it just emits nonsense.
- The `-fa on` (oneDNN flash-attention, PR #25222) path was *even worse* (gibberish immediately;
  `GGML_SYCL_DISABLE_OPT=1` fixed short prompts but still errored at depth), but **the base path is not
  safe either.**
- Matches upstream [#25708](https://github.com/ggml-org/llama.cpp/issues/25708) (Qwen3.6-Q4_K on
  Battlemage, still open).

**We rolled back to the older correct build.** Trade-off accepted: lose the checkpoint-reuse + faster
prefill, keep correct output. A gibberish-generating model is worthless regardless of speed.

## Lessons (the important part)

1. **Validate correctness with REAL, long, agentic prompts — not toy prompts.** Short prompts happened
   to dodge the reorder/quant bug; the actual workload triggered it. A "healthy" server that emits
   nonsense is the worst failure mode (it looks fine).
2. The checkpoint-reuse machinery (#22384/#22929) is genuinely there and genuinely fast — this is a
   *correctness* blocker on recent master for Battlemage-Q4_K (#25708), **not** a caching problem. When
   that regression is fixed upstream (or on a card/quant it doesn't affect), the win is ready to claim.
3. Keep the old image as a one-command fallback and roll back the instant real output looks wrong.

## Status

Reference box is back on the older correct build. Re-try a fresh master build (and re-validate with
real prompts) once #25708 is resolved, or try a different quant / the `bf16` weights that may dodge the
reorder path.
