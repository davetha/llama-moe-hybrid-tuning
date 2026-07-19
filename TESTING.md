# Testing methodology (how to not fool yourself)

We burned a lot of time on measurement mistakes. These are the rules that gave trustworthy numbers.

## 1. Measure prefill and generation separately
They have different bottlenecks and a change can help one while wrecking the other. In our case:
- **Prefill (pp)** = processing the prompt. Bound by expert compute (CPU) + attention (GPU). This is
  what large-context / agentic workloads actually wait on.
- **Generation (tg)** = producing tokens. Bound by memory bandwidth of the active experts. Near its
  ceiling on CPU; hard to move.

`llama-bench -p <prefill_len> -n <gen_len>` reports both cleanly. Use it for knob A/Bs.

## 2. Always warm up first
The first request after any (re)start pays one-time costs (kernel JIT on SYCL/Vulkan, cache cold,
graph build). Discard a warmup request, or your "regression" is just cold start. `llama-bench`'s
`-r 2+` repeats handle this; ad-hoc curl timing does not.

## 3. The fit *estimator* is not a fit *test*
llama.cpp's `common_fit_params` can print **0 warnings** and then page-fault the GPU on the first
real prefill. Trust only a **real large prefill with a dmesg fault check** (`scripts/fit-test.sh`).
Keep ~1 GB of real VRAM headroom — a bigger `-ub` grows the compute buffer, which competes with the
KV cache for VRAM.

## 4. Bench small, but validate at your real context
`llama-bench` runs at a small context, so a config that benches great (big `-ub`) may not fit
alongside a 128K–256K KV cache. Bench to find the *fast* config, then `fit-test.sh` it at your
actual `-c`. Only deploy configs that pass both.

## 5. Isolate one variable at a time
Change one knob per run. When we combined `--no-mmap` + bigger `-ub` we still ran them as a ladder
(baseline → +no-mmap → +ub2048 → +ub4096) so we could attribute the gain.

## 6. Watch the right signals under load
- **GPU busy?** clock/energy under load (Intel: `.../gt0/freq0/act_freq` + hwmon energy delta;
  AMD: `amdgpu` sysfs / `radeontop`). A pegged clock = compute-bound; an idle clock during work =
  you're bottlenecked elsewhere (PCIe, CPU, memory).
- **PCIe link:** read the *right* device. A card behind an on-board switch reports the internal
  (idle, downtrained) link on the GPU function; the real host link is on the **switch upstream port**.
  Cross-check `lspci -vv` `LnkSta` on the upstream bridge, and the kernel's boot-time
  "available PCIe bandwidth, limited by …" line (it stays silent if the link is fine).
- **Faults:** `dmesg` for driver resets. Distinguish harmless load-time resets (recover) from a real
  wedge (model stops responding) — confirm with a generation probe, don't revert on the log line alone.

## 7. Correct for concurrency
`llama-bench` is single-sequence. A live server with `-np N` slots shares CPU/GPU across concurrent
prefills, so per-request numbers are lower than the isolated bench. Validate the final config on the
live server too (we saw 527 t/s bench → 462–637 t/s live depending on config).
