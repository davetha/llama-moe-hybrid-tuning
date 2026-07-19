#!/bin/bash
# perf-tweaks.sh — persistent host tweaks that help CPU-resident MoE experts (any GPU vendor).
#   - Transparent Huge Pages = always  (fewer TLB misses on the CPU expert reads; pairs w/ --no-mmap)
#   - CPU core boost = on               (expert prefill is CPU-bound)
# Optional GPU clock un-cap is vendor-specific; see the commented block.
# Idempotent and safe if a path is absent. Install as a systemd oneshot (see bottom).
set -u

echo always > /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || true

# CPU boost: acpi-cpufreq / amd_pstate-passive use cpufreq/boost; intel_pstate uses no_turbo (inverted)
if [ -w /sys/devices/system/cpu/cpufreq/boost ]; then
  echo 1 > /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || true
elif [ -w /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
  echo 0 > /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || true
fi

# Optional: keep the CPU at max clock (removes ramp latency). Uncomment if you want it:
# for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$g" 2>/dev/null || true; done

# --- Intel Arc (xe) GPU clock un-cap: pin max_freq to hardware rp0 (do NOT downclock) ---
# GT=/sys/class/drm/card1/device/tile0/gt0/freq0
# [ -r "$GT/rp0_freq" ] && cat "$GT/rp0_freq" > "$GT/max_freq" 2>/dev/null || true
#
# --- AMD (amdgpu) GPU: set the power profile / performance level instead ---
# echo high > /sys/class/drm/card1/device/power_dpm_force_performance_level 2>/dev/null || true

echo "applied: THP=$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null), boost/turbo set"

# ---------------------------------------------------------------------------------------------
# Install as a persistent systemd oneshot:
#   sudo cp perf-tweaks.sh /usr/local/bin/llm-perf-tweaks && sudo chmod +x /usr/local/bin/llm-perf-tweaks
#   sudo tee /etc/systemd/system/llm-perf-tweaks.service >/dev/null <<'UNIT'
#   [Unit]
#   Description=LLM host perf tweaks (THP, CPU boost, GPU clock)
#   After=multi-user.target
#   [Service]
#   Type=oneshot
#   RemainAfterExit=yes
#   ExecStart=/usr/local/bin/llm-perf-tweaks
#   [Install]
#   WantedBy=multi-user.target
#   UNIT
#   sudo systemctl daemon-reload && sudo systemctl enable --now llm-perf-tweaks
# ---------------------------------------------------------------------------------------------
