# llama.cpp MoE Hybrid Tuning (experts-on-CPU, small-VRAM GPU)

Field notes + scripts for running **large MoE models on a small-VRAM GPU** with llama.cpp,
where the attention/KV live on the GPU and the **MoE experts run on the CPU** (`-ot exps=CPU`).
This is the practical way to run a 20–130 GB MoE on a 12–16 GB card.

The headline finding: **for this hybrid layout, prefill was silently ~10× slower than it
needed to be** — and the fixes are almost all **backend-agnostic** (they help AMD ROCm/Vulkan
and NVIDIA setups too, not just Intel).

Reference box for the numbers below:
- GPU: Intel Arc **B580** (Battlemage, 12 GB), llama.cpp SYCL backend
- Host: AMD **EPYC 74F3** (Zen 3, 24c), 512 GB DDR4-2933
- Model: **Qwen3.6-35B-A3B** (MoE, 3B active) Q4_K, ~20 GB, 256K context
- Layout: attention + KV cache on GPU, experts on CPU (`-ot exps=CPU`)

## TL;DR results (same GPU, same model, same 256K context)

| Change | Prefill (pp) | Note |
|---|---|---|
| baseline (mmap, default batch) | **~83 t/s** | |
| `--no-mmap` | ~197 t/s | 2.4× — the big one |
| `--no-mmap` + `-ub 2048 -b 2048` | ~527 t/s | bench; 462 t/s live |
| `--no-mmap` + `-ub 4096 -b 4096` + THP + boost + uncapped clock | **~637 t/s** | **~8–14× vs baseline** |

Generation stayed ~38 t/s throughout — it's memory-bandwidth-bound on the CPU experts and was
already near its ceiling. **The whole win is prefill**, which is what dominates large-context
(agentic / long-doc) workloads.

Full numbers (llama-bench sweep + live-server validation) in [BENCHMARKS.md](BENCHMARKS.md).

## What generalizes to which hardware

| Finding | AMD (ROCm/Vulkan) | NVIDIA (CUDA) | Intel (SYCL) | Why |
|---|:-:|:-:|:-:|---|
| **`--no-mmap`** with `-ot exps=CPU` | ✅ | ✅ | ✅ | mmap page-faults on every CPU-resident expert access during prefill; a contiguous RAM buffer avoids it. Pure CPU-side, backend-agnostic. |
| **`-ub` / `-b` batch bump** | ✅ | ✅ | ✅ | bigger prefill matmuls = better hw utilization, less per-chunk overhead |
| **THP = always** | ✅ | ✅ | ✅ | fewer TLB misses on the CPU expert-weight reads (pairs with `--no-mmap`) |
| **CPU boost / `performance` governor** | ✅ | ✅ | ✅ | CPU-resident expert compute is CPU-bound |
| **Real-prefill VRAM fit test** | ✅ | ✅ | ✅ | the fit *estimator* lies; a real large prefill is the only trustworthy check |
| **`amd_iommu=off`** for host DMA timeouts | ✅ (AMD host) | ✅ (AMD host) | ✅ (AMD host) | it's a *host IOMMU* fix (EPYC/Ryzen), independent of GPU vendor |
| **dmesg fault-watchdog → revert to CPU** | ✅ (amdgpu ring reset) | ⚠️ (Xid) | ✅ (xe reset) | adapt the dmesg regex per driver |
| GuC firmware / SYCL cache / oneDNN FA | ❌ | ❌ | Intel only | Battlemage/SYCL-specific |

**If you take one thing:** on an experts-on-CPU hybrid, add **`--no-mmap`** and bump **`-ub`**.
That alone was ~6× prefill here, on any backend.

## Quick start

```bash
# 1) The core llama-server flags for an experts-on-CPU hybrid:
llama-server -m MODEL.gguf \
  -ngl 999 -ot "exps=CPU" \       # attention+KV on GPU, experts on CPU
  --no-mmap \                     # << the big prefill win
  -b 4096 -ub 4096 \              # << big prefill batches (validate it fits your VRAM+ctx)
  -c 262144 --kv-unified -np 3 --jinja

# 2) Host tweaks (see scripts/perf-tweaks.sh):
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
echo 1      | sudo tee /sys/devices/system/cpu/cpufreq/boost
```

Then **measure, don't assume** — see [TESTING.md](TESTING.md) and `scripts/bench-knobs.sh`.

## Scripts

- `scripts/bench-knobs.sh` — llama-bench A/B sweep of mmap / ubatch (find your best config)
- `scripts/fit-test.sh` — validate a config actually fits your target context (real prefill + dmesg fault check)
- `scripts/gpu-fault-watchdog.sh` — watch dmesg for GPU resets/faults; auto-revert to a CPU-only config
- `scripts/perf-tweaks.sh` — THP + CPU boost + (optional) GPU clock, as a persistent systemd unit

## The gotchas that cost us the most time

1. **The fit estimator is not a fit test.** llama.cpp's `common_fit_params` can report "0 warnings"
   and then page-fault the GPU on the first real prefill. Always validate with an actual large
   prefill (`fit-test.sh`), and keep ~1 GB of real VRAM headroom.
2. **VRAM overcommit ≠ a clean failure.** On some drivers it wedges the GPU (needs a reboot),
   not a tidy OOM. Size conservatively.
3. **On an AMD host, a GPU DMA stall can time out the IOMMU and hang the *whole box*.** If you see
   `AMD-Vi: Completion-Wait loop timed out` / `IOTLB_INV_TIMEOUT`, try `amd_iommu=off`.
4. **Measure prefill and generation separately, warmup-corrected.** A change can help one and
   wreck the other (e.g. flash-attention on CPU tanked our prefill 14×).

See [FINDINGS.md](FINDINGS.md) for the full lever-by-lever detail and the Intel-specific notes,
and [research/gated-deltanet-prefill-caching.md](research/gated-deltanet-prefill-caching.md) for a
deep dive on *why you re-prefill every turn* on Gated DeltaNet (Qwen3-Next) hybrids and what actually
fixes it (checkpointing, stable-prefix compaction, engine choice). A newer llama.cpp build *does* ship
the checkpoint-reuse fix — but on Battlemage-Q4_K it currently gibberishes (upstream #25708), so it's
not deployable yet: see [research/b580-checkpoint-reuse-results.md](research/b580-checkpoint-reuse-results.md)
and [docker/Dockerfile.sycl-checkpoint](docker/Dockerfile.sycl-checkpoint).

---
*These are field notes from one box, shared in case they save someone else the week we spent.
Numbers are from the reference config above; your mileage will vary — the point is the method
and the direction, not the exact t/s.*
