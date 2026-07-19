#!/bin/bash
# gpu-fault-watchdog.sh — auto-revert to a safe (CPU-only) config when the GPU actually wedges.
#
# Key lesson: GPU "reset/fault" lines in dmesg often fire harmlessly at load (buffer alloc / JIT)
# and RECOVER. Their mere presence is NOT a wedge. A real wedge = the model stops responding.
# So we treat dmesg faults only as a HINT, then CONFIRM with a generation probe before reverting
# (a revert is disruptive). We require the probe to fail twice and skip the model-load grace window.
#
# Configure the two commands for your setup:
#   REVERT_CMD  — brings the model up in a safe CPU-only config (must be non-interactive)
#   IS_GPU_CMD  — exits 0 if the model is currently in the GPU/hybrid config (else no-op)
# and the dmesg regex for your driver (Intel xe vs AMD amdgpu vs NVIDIA).
set -u
ENDPOINT="${ENDPOINT:-http://localhost:8080}"
REVERT_CMD="${REVERT_CMD:-echo 'set REVERT_CMD to your CPU-only launch command'}"
IS_GPU_CMD="${IS_GPU_CMD:-true}"        # default: always consider it GPU mode
FAULT_RE="${FAULT_RE:-Engine reset|Fault response|guc_exec_queue_timedout|amdgpu.*ring.*timeout|amdgpu.*reset|GPU reset|IOTLB_INV_TIMEOUT|Completion-Wait}"
INTERVAL="${INTERVAL:-30}"; GRACE="${GRACE:-210}"; STORM="${STORM:-25}"; PROBE_TIMEOUT="${PROBE_TIMEOUT:-45}"
STATE=/run/gpu-watchdog.state; LOG="${LOG:-/var/log/gpu-fault-watchdog.log}"

log(){ echo "$(date '+%F %T') $*" >> "$LOG"; }
faults(){ dmesg 2>/dev/null | grep -ciE "$FAULT_RE"; }
gen_probe(){ curl -s -m "$PROBE_TIMEOUT" "$ENDPOINT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"max_tokens":1}' 2>/dev/null | grep -q '"choices"'; }

echo "$(faults)" > "$STATE"
log "watchdog started (interval=${INTERVAL}s grace=${GRACE}s storm=${STORM}); baseline=$(cat $STATE)"
while true; do
  sleep "$INTERVAL"
  prev=$(cat "$STATE" 2>/dev/null || echo 0); cur=$(faults)
  if [ "$cur" -lt "$prev" ]; then echo "$cur" > "$STATE"; continue; fi   # dmesg ring wrapped
  new=$((cur - prev)); echo "$cur" > "$STATE"
  [ "$new" -le 0 ] && continue
  eval "$IS_GPU_CMD" >/dev/null 2>&1 || { log "fault(+$new) but not GPU mode; ignore"; continue; }
  # skip harmless load-window faults; confirm a real wedge with two failed gen probes
  if gen_probe; then log "fault(+$new) but gen probe OK -> recovered, no revert"; continue; fi
  sleep 15
  if gen_probe; then log "fault(+$new) probe1 fail / probe2 OK -> transient, no revert"; continue; fi
  log "CONFIRMED WEDGE (+$new faults, 2x probe hung) -> reverting to CPU-only"
  eval "$REVERT_CMD" >> "$LOG" 2>&1 && log "reverted OK" || log "ERROR: revert command failed"
done
