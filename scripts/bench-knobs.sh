#!/bin/bash
# bench-knobs.sh — A/B sweep the high-impact llama.cpp knobs for an experts-on-CPU MoE hybrid.
# Uses llama-bench for clean, isolated prefill (pp) / generation (tg) numbers.
#
# Usage:  MODEL=/path/model.gguf LLAMA_BENCH=/path/to/llama-bench ./bench-knobs.sh
# Env:
#   MODEL        (required) path to the GGUF
#   LLAMA_BENCH  path to llama-bench binary (default: llama-bench on PATH)
#   OT           tensor-offload regex for experts-on-CPU (default: "exps=CPU")
#   NGL          gpu layers (default: 999 = all non-overridden tensors on GPU)
#   PP           prefill length to bench (default: 4096)
#   TG           gen length to bench (default: 64)
#
# This is backend-agnostic: works for SYCL (Intel), ROCm/Vulkan (AMD), CUDA (NVIDIA).
set -u
: "${MODEL:?set MODEL=/path/to/model.gguf}"
LB="${LLAMA_BENCH:-llama-bench}"
OT="${OT:-exps=CPU}"; NGL="${NGL:-999}"; PP="${PP:-4096}"; TG="${TG:-64}"

run() { # $1=label  $2..=extra flags
  local label="$1"; shift
  echo "########## $label ##########"
  "$LB" -m "$MODEL" -ngl "$NGL" -ot "$OT" -p "$PP" -n "$TG" -r 2 "$@" 2>&1 \
    | grep -iE "pp${PP}|tg${TG}|error|fail|out of|cannot" || echo "  (no result rows — check for a load error)"
  echo
}

echo "== bench-knobs: MODEL=$MODEL OT=$OT PP=$PP TG=$TG =="
run "A_baseline (mmap on, default batch)"
run "B_no-mmap"                       --mmap 0
run "C_no-mmap +ub2048"               --mmap 0 -ub 2048 -b 2048
run "D_no-mmap +ub4096"               --mmap 0 -ub 4096 -b 4096
echo "Pick the fastest config whose ubatch still FITS your target context (validate with fit-test.sh)."
echo "Note: llama-bench runs at a small context; a bigger -ub may not fit alongside a large KV cache."
