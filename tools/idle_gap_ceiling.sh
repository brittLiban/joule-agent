#!/usr/bin/env bash
# Idle-gap ceiling: how much power does a static clock lock burn during a traffic
# gap that a controller could give back?
#
# This is the cheapest test that can kill a phase-aware DVFS thesis, and it needs
# no serving engine, no model and no load generator -- four clock locks and a
# power counter, about six minutes. Run it on new hardware BEFORE building a
# controller for that hardware.
#
#     ceiling = f_idle x dP / P_loaded
#
# where dP is the idle-power difference between the tuned static clock and the
# lowest lockable clock, f_idle is the fraction of wall time the workload spends
# with the GPU idle, and P_loaded is mean power under the real workload. If the
# ceiling lands under your target, no controller can reach the target by giving
# back idle-gap power, and none needed to be written to find that out.
#
# Four legs, interleaved A/B/A/B, so leg order and residual heat cannot
# masquerade as the effect -- on the reference card absolute level drifted 2.0 W
# across the session while the PAIRED differences agreed to 0.02 W.
#
# Only the clock write runs as root; the recorder stays unprivileged so it does
# not leave root-owned files that later unprivileged runs cannot read.
#
# Usage: tools/idle_gap_ceiling.sh <high_mhz> [low_mhz] [seconds_per_leg]
set -euo pipefail

HIGH_MHZ=${1:?usage: idle_gap_ceiling.sh <high_mhz> [low_mhz] [seconds]}
LOW_MHZ=${2:-}
SECS=${3:-60}

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
PY=.venv/bin/python
CONSENT=--i-understand-this-changes-gpu-settings
MANIFEST=runs/idle_gap_manifest.tsv

if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "REFUSING: NVML unusable. If the driver was upgraded while running, reboot."
  nvidia-smi -L 2>&1 | head -2
  exit 1
fi

# The lowest LOCKABLE clock is not the idle clock -- SetGpuLockedClocks cannot
# reach the deepest idle P-state. Ask the device rather than assuming.
if [ -z "$LOW_MHZ" ]; then
  LOW_MHZ=$($PY -c "
import pynvml as n
n.nvmlInit(); h=n.nvmlDeviceGetHandleByIndex(0)
print(sorted(n.nvmlDeviceGetSupportedGraphicsClocks(h, n.nvmlDeviceGetSupportedMemoryClocks(h)[0]))[0])")
  echo "lowest lockable clock on this device: ${LOW_MHZ} MHz"
fi

echo "== caching sudo credentials =="
sudo -v

echo "== clearing any stale guard record =="
sudo $PY -m joule_agent.gpu_guard --gpu 0 restore --quarantine-unreadable || true

mkdir -p runs
printf 'leg\trequested_mhz\trun_dir\n' > "$MANIFEST"

for LEG in 1:$HIGH_MHZ 2:$LOW_MHZ 3:$HIGH_MHZ 4:$LOW_MHZ; do
  IDX=${LEG%%:*}; MHZ=${LEG##*:}
  echo "== leg $IDX: locked ${MHZ} MHz, ${SECS}s, NO TRAFFIC =="

  sudo $PY -m joule_agent.gpu_guard lock --mhz "$MHZ" \
       --hold "$((SECS + 15))" $CONSENT &
  LOCK_PID=$!
  sleep 5   # let the guard arm before the recorder opens its window

  OUT=$($PY -m joule_agent.telemetry record --duration "$SECS" --out-dir runs)
  echo "$OUT"
  printf '%s\t%s\t%s\n' "$IDX" "$MHZ" \
    "$(echo "$OUT" | awk '/^run:/ {print $2}')" >> "$MANIFEST"

  wait $LOCK_PID
  sleep 10  # settle back to stock idle before the next leg
done

echo
echo "== guard status (must read stale:false) =="
$PY -m joule_agent.gpu_guard status || true
echo
$PY tools/idle_gap_ceiling.py "$MANIFEST"
