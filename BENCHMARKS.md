# Benchmarks

Raw numbers behind the README. One box, one model — the point is the **deltas and method**, not
that your t/s will match.

## Setup

| | |
|---|---|
| GPU | Intel Arc B580 (Battlemage, 12 GB), llama.cpp SYCL (`GGML_SYCL_F16`) |
| Host | AMD EPYC 74F3 (Zen 3, 24c), 512 GB DDR4-2933 (single NUMA node) |
| Model | Qwen3.6-35B-A3B (MoE, 3B active) Q4_K, ~20 GB |
| Layout | `-ngl 999 -ot exps=CPU` (attention + KV on GPU, experts on CPU), `-c 262144` (256K) |

## 1. Knob sweep — `llama-bench` (isolated, single-sequence, pp4096 / tg64)

CPU boost ON, GPU at 2400 MHz cap during this sweep.

| Config | prefill pp4096 (t/s) | gen tg64 (t/s) |
|---|---:|---:|
| A — baseline (mmap on, default batch) | **82.96** | 37.10 |
| B — `--mmap 0` (`--no-mmap`) | **196.62** | 36.15 |
| C — `--no-mmap -ub 2048 -b 2048` | **526.60** | 38.79 |

- `--no-mmap` alone: **2.37×** prefill, generation flat.
- `--no-mmap` + `-ub 2048`: **6.35×** prefill vs baseline.
- Generation is memory-bandwidth-bound on the CPU experts and barely moves — the win is all prefill.

## 2. Live server validation (`llama-server`, `-np 3`, real 12,657-token prefill)

GPU **uncapped to 2850 MHz**, CPU boost ON. These are end-to-end on the running server (lower than the
isolated bench because of concurrent slots — see TESTING.md #7).

| Config | prefill (12,657 tok) | notes |
|---|---:|---|
| `--no-mmap -ub 2048 -b 2048` | **462 t/s** | fits 256K, 0 faults |
| `--no-mmap -ub 4096 -b 4096` + THP + uncapped | **636.67 t/s** | fits 256K, 0 faults |

That 12.6K-token prefill went from ~290 s (baseline ~44-83 t/s) to **~20 s**.

## 3. Cumulative

| Stage | prefill |
|---|---:|
| original baseline | ~44-83 t/s |
| `--no-mmap` | ~197 t/s |
| + `-ub 2048` | ~463-527 t/s |
| + `-ub 4096` + THP + uncapped clock + boost | **~637 t/s** |

**≈ 8-14× prefill speedup, 256K context intact, zero faults, generation unchanged (~38 t/s).**

## Method notes

- `llama-bench -p 4096 -n 64 -r 2` for the isolated sweep; `--mmap 0`, `-ub`, `-b` are the swept flags.
- Live numbers via a real ~12.6K-token chat/completions request, prefill t/s read from the server's
  `prompt eval time` log line.
- VRAM fit for each `-ub` validated with a real large prefill + dmesg fault check (the estimator was
  not trustworthy). See `scripts/fit-test.sh`.
- All runs warmup-corrected (first request after (re)start pays SYCL JIT).
