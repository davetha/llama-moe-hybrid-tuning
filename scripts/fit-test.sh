#!/bin/bash
# fit-test.sh — Does a llama-server config ACTUALLY fit your target context?
# The fit ESTIMATOR lies: it can report "0 warnings" then page-fault the GPU on the first real
# prefill. The only trustworthy test is to launch the config, fire a large real prefill, and
# watch dmesg for a GPU fault. Keep ~1 GB of real VRAM headroom.
#
# Usage:  ENDPOINT=http://localhost:8080 ./fit-test.sh
#   assumes a llama-server is already running with the config you want to validate.
# Env:
#   ENDPOINT     (default http://localhost:8080) llama-server base URL
#   PREFILL_TOK  approx prompt tokens to force (default ~12000)
#   FAULT_RE     dmesg regex for a GPU fault (default covers Intel xe + AMD amdgpu)
set -u
EP="${ENDPOINT:-http://localhost:8080}"
FAULT_RE="${FAULT_RE:-Fault response|Engine reset|gpu hang|ring .* timeout|amdgpu.*reset|GPU reset|IOTLB_INV_TIMEOUT}"
N="${PREFILL_TOK:-750}"   # ~16 tokens per filler line

faults() { sudo dmesg 2>/dev/null | grep -ciE "$FAULT_RE"; }
BIG=$(for i in $(seq 1 "$N"); do printf 'Datapoint %s about caches, schedulers, and thermodynamics interacting under load. ' "$i"; done)

echo "== fit-test against $EP =="
curl -sf -m5 "$EP/health" >/dev/null 2>&1 && echo "server: healthy" || { echo "server not healthy at $EP"; exit 1; }

echo "warmup (JIT/first-run)…"
curl -s -m120 "$EP/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"max_tokens":4}' >/dev/null 2>&1

F0=$(faults)
echo "firing large prefill (~${N} lines)…"
curl -s -m180 "$EP/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"x\",\"messages\":[{\"role\":\"user\",\"content\":\"$BIG Summarize.\"}],\"max_tokens\":32}" >/dev/null 2>&1
RC=$?
F1=$(faults)

echo "curl exit: $RC   new GPU faults: $((F1-F0))"
if [ "$((F1-F0))" -gt 0 ]; then
  echo "RESULT: ❌ FAULTED — this config OVERCOMMITS VRAM at this context. Reduce -ub or context."
  sudo dmesg | grep -iE "$FAULT_RE" | tail -4
  exit 1
elif [ "$RC" -ne 0 ]; then
  echo "RESULT: ⚠️ request failed/timed out (not a GPU fault) — check server logs."
  exit 2
else
  echo "RESULT: ✅ fits — no GPU fault on a real large prefill."
fi
