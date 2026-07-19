# Findings — lever by lever

Detail behind the README table. Each lever notes **what it does**, **measured impact**, and
**what generalizes**. Reference box: Arc B580 12GB + EPYC 74F3 512GB, Qwen3.6-35B-A3B Q4_K, 256K,
`-ot exps=CPU`.

## Cross-vendor wins (the important ones)

### `--no-mmap` — the big one (~2.4× prefill)
With `-ot exps=CPU`, the expert tensors live in CPU RAM. By default llama.cpp **mmaps** the GGUF, so
those experts are file-backed pages that page-fault on access — during prefill that's a fault on
essentially every expert, every token. `--no-mmap` loads them into a contiguous `malloc` buffer
instead. Prefill **83 → 197 t/s**. llama.cpp even prints a hint:
`tensor overrides to CPU are used with mmap enabled — consider using --no-mmap`. We ignored it for
too long. **Generalizes to every backend** (it's CPU-side). Cost: full model loaded into RAM +
slower first load. Free on a big-RAM host.

### Prefill batch: `-ub` / `-b` (~2.7× more on top)
Prefill is processed in `-ub`-sized chunks. Default 512 → **2048** → **4096** made the matmuls
bigger and kept the hardware fed. Prefill **197 → 527 → 637 t/s** (bench). **Generalizes.** Caveat:
`-ub` grows the GPU compute buffer, so validate it fits your KV cache at your real context
(`fit-test.sh`). 4096 fit our 256K with headroom; yours may not.

### Transparent Huge Pages = always
The 20 GB model as 4 KB pages is ~5M pages — far more than the TLB holds, so CPU expert reads miss
the TLB constantly. 2 MB huge pages cut that ~500×. Pairs with `--no-mmap` (contiguous buffer is
promotable). Modest but free. **Generalizes** to any CPU-offloaded model.

### CPU boost / governor
Expert prefill is CPU-bound; boost (or the `performance` governor) keeps cores at max clock.
~single-digit %. **Generalizes.**

### `amd_iommu=off` — for AMD *hosts* (not AMD GPUs)
On our EPYC host, a GPU DMA stall made the **AMD IOMMU** time out (`Completion-Wait loop timed out`,
`IOTLB_INV_TIMEOUT device=<gpu>`) and hung the **entire machine** (needed a power cycle). Disabling
the IOMMU (`amd_iommu=off` on the kernel cmdline) turned host-lockups into recoverable GPU resets.
**Applies to any GPU on an AMD EPYC/Ryzen host.** Trade-off: no VFIO/PCI-passthrough, tiny SWIOTLB
overhead. Only if you're not doing GPU passthrough to VMs.

### Fault watchdog → auto-revert
Small daemon: watch `dmesg` for driver resets, and on a *confirmed* wedge (dmesg signal **plus** a
failed generation probe, past the load-grace window) auto-swap the model to a CPU-only config.
Adapt the regex: Intel `xe … Engine reset` / AMD `amdgpu … ring … timeout` / NVIDIA `Xid`.

## Ruled out (so you don't waste time)

- **More experts on GPU at large context** — VRAM overcommit → page-fault. At 256K the KV cache eats
  the budget; even 4 expert layers overcommitted 12 GB. It's a context-vs-experts trade, not a free win.
- **NUMA tuning** — only helps if the host is NPS>1 (multiple NUMA nodes). Single-node = nothing to do.
- **Speculative decoding** on MoE — draft-mtp was −19% here (MoE non-amortization + hybrid PCIe toll);
  ngram was neutral. Weak for this arch.
- **Persistent JIT cache (SYCL)** — noise for us; the cold-start cost was already small and the real
  "slow after restart" was the KV prompt-cache wipe, which no kernel cache fixes.

## Intel Battlemage / SYCL-specific (won't transfer)

- **GuC firmware** matters: a `-ENOENT` / `ccs`+`bcs` engine-reset wedge (Intel compute-runtime
  #946/#948) under sustained inference on kernel 7.0. Updating `bmg_guc_70.bin` 70.58.0 → **70.65.0**
  eliminated the previously-instant faults (firmware ships in `linux-firmware`, independent of kernel).
- **oneDNN XMX flash-attention** (llama.cpp PR #25222, merged b10016): up to ~4× prefill *at depth* on
  Battlemage with f16 KV — but needs a rebuild, and recent master had a Qwen3.6-Q4_K **gibberish** bug
  on BMG (#25708) — validate output correctness before trusting a rebuild.
- **PCIe "Gen1 x1" reads** are a red herring: the B580 has an on-card Intel switch; the real host link
  is Gen4 x8 on the switch's upstream port. Read the upstream bridge, not the GPU function.
- `-DGGML_SYCL_F16=ON` (keep it), `GGML_SYCL_DNN=ON` (oneDNN); AOT `-DGGML_SYCL_DEVICE_ARCH=bmg_g21`
  has an open Xe2 correctness bug (#21893); IPEX-LLM was not faster than mainline SYCL.
