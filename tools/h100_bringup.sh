#!/usr/bin/env bash
# Bring-up and go/no-go gates for a rented datacenter GPU.
#
# Ordered so the cheapest thing that can end the session runs first. Every phase
# has an explicit go/no-go; a failure means stop and read, not push on.
#
# NOTHING HERE IS A MEASUREMENT YOU CAN TRUST YET. Phases 0-2 establish that the
# instrument works on this hardware. Only phase 3 produces a number, and only if
# phase 2 passed -- energy-counter agreement is per-device and a counter that
# has not been checked against an oscillating load is not evidence.
#
# Usage:
#   tools/h100_bringup.sh check     # phase 0-1, read-only + guard identity
#   tools/h100_bringup.sh energy    # phase 2, counter agreement gate
#   tools/h100_bringup.sh idlegap   # phase 3, the ceiling measurement
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
PY=.venv/bin/python
CONSENT=--i-understand-this-changes-gpu-settings
PHASE=${1:-check}

hr() { printf '=%.0s' {1..70}; echo; }
ok() { echo "  PASS: $*"; }
no() { echo "  FAIL: $*"; echo; echo "STOP. Do not spend GPU time past a failed gate."; exit 1; }

# ---------------------------------------------------------------------------
# Phase 0 -- can this instance write clocks at all?
# ---------------------------------------------------------------------------
phase0() {
  hr; echo "PHASE 0: capability"; hr
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,power.max_limit \
             --format=csv || no "nvidia-smi unavailable"

  NGPU=$(nvidia-smi --list-gpus | wc -l)
  echo "  GPUs visible: $NGPU"

  # THE blocker on rented hardware: many container runtimes grant NVML reads and
  # refuse clock writes. Find out in 5 seconds, not after an hour of setup.
  if sudo nvidia-smi -i 0 -lgc 1200 >/dev/null 2>&1; then
    sudo nvidia-smi -i 0 -rgc >/dev/null 2>&1 || true
    ok "clock writes permitted"
  else
    no "clock writes REFUSED on this instance. Everything past the calibration
      half is impossible here. You need bare metal or a privileged container --
      ask the provider before renting more of this SKU."
  fi

  $PY -c "
import pynvml as n
n.nvmlInit()
for i in range(n.nvmlDeviceGetCount()):
    h = n.nvmlDeviceGetHandleByIndex(i)
    mem = n.nvmlDeviceGetSupportedMemoryClocks(h)[0]
    c = sorted(n.nvmlDeviceGetSupportedGraphicsClocks(h, mem))
    name = n.nvmlDeviceGetName(h)
    print(f'  gpu{i} {name if isinstance(name,str) else name.decode()}: '
          f'{len(c)} clocks, {c[0]}-{c[-1]} MHz, step {c[1]-c[0] if len(c)>1 else \"?\"}')
    try:
        n.nvmlDeviceGetTotalEnergyConsumption(h)
        print(f'  gpu{i} energy counter: present')
    except Exception as e:
        print(f'  gpu{i} energy counter: ABSENT ({e}) -- joules cannot be trusted')
" || no "NVML enumeration failed"
}

# ---------------------------------------------------------------------------
# Phase 1 -- the guard's device-identity gate, on real multi-GPU hardware
# ---------------------------------------------------------------------------
phase1() {
  hr; echo "PHASE 1: guard device identity (multi-GPU)"; hr
  NGPU=$(nvidia-smi --list-gpus | wc -l)
  if [ "$NGPU" -lt 2 ]; then
    echo "  single GPU -- identity gate not exercisable here, skipping."
    echo "  (It has still never run against real multi-GPU hardware.)"
    return 0
  fi

  echo "  This gate has only ever run against a FAKE NVML layer. The failure it"
  echo "  guards against is applying one card's saved state to another card."
  echo "  Stranding a record on gpu1, then checking gpu0 refuses to act on it."

  # (a) strand a record for GPU 1 by crashing mid-lock
  LOW=$($PY -c "
import pynvml as n
n.nvmlInit(); h=n.nvmlDeviceGetHandleByIndex(1)
print(sorted(n.nvmlDeviceGetSupportedGraphicsClocks(h, n.nvmlDeviceGetSupportedMemoryClocks(h)[0]))[len(sorted(n.nvmlDeviceGetSupportedGraphicsClocks(h, n.nvmlDeviceGetSupportedMemoryClocks(h)[0])))//2])")
  echo "  (a) stranding a record on gpu1 at ${LOW} MHz ..."
  sudo $PY -m joule_agent.gpu_guard --gpu 1 lock --mhz "$LOW" --hold 30 \
       --crash-after 3 $CONSENT >/dev/null 2>&1 || true

  # (b) gpu0 status must report OTHER DEVICE and point at gpu1
  echo "  (b) gpu0 status:"
  OUT=$($PY -m joule_agent.gpu_guard --gpu 0 status 2>&1 || true)
  echo "$OUT" | grep -qi "other device\|--gpu 1\|record_is_for_this_device.*false" \
    && ok "gpu0 recognises the record belongs to another device" \
    || no "gpu0 did NOT flag the record as another device's. Recovery could apply
      gpu1's snapshot to gpu0. Do not sweep on this node."

  # (c) gpu0 restore must REFUSE
  echo "  (c) gpu0 restore must refuse:"
  if sudo $PY -m joule_agent.gpu_guard --gpu 0 restore >/dev/null 2>&1; then
    no "gpu0 restore SUCCEEDED against gpu1's record. That is the exact
      cross-device failure this gate exists to catch."
  else
    ok "gpu0 restore refused"
  fi

  # (d) gpu1 restore must succeed and clear the record
  echo "  (d) gpu1 restore:"
  sudo $PY -m joule_agent.gpu_guard --gpu 1 restore >/dev/null 2>&1 \
    && ok "gpu1 restore cleared its own record" \
    || no "gpu1 could not restore its own record"

  $PY -m joule_agent.gpu_guard --gpu 1 status | grep -E '"stale"'
}

# ---------------------------------------------------------------------------
# Phase 2 -- energy counter agreement. Per-device; never inherited.
# ---------------------------------------------------------------------------
phase2() {
  hr; echo "PHASE 2: energy counter agreement under oscillating load"; hr
  echo "  The idle row proves nothing -- every integration method agrees on a"
  echo "  constant signal. This drives a 0.5 Hz square wave and checks the"
  echo "  counter against a trapezoidal integration of the power buffer."
  $PY tools/synthetic_gpu_load.py --duration 32 --period 2.0 --duty 0.5 \
      > /tmp/joule_synload.log 2>&1 &
  LOAD=$!
  sleep 3
  $PY -m joule_agent.telemetry verify --duration 25 || true
  wait $LOAD 2>/dev/null || true
  tail -2 /tmp/joule_synload.log
  echo
  echo "  GO/NO-GO: agreement must be well under 2%, AND mean power must be"
  echo "  far above idle -- if the synthetic load did not actually load this"
  echo "  card, the gate tested nothing. Check the watts."
}

# ---------------------------------------------------------------------------
# Phase 3 -- the ceiling. The question the whole trip is for.
# ---------------------------------------------------------------------------
phase3() {
  hr; echo "PHASE 3: idle-gap ceiling"; hr
  HIGH=${2:-}
  if [ -z "$HIGH" ]; then
    # Highest supported clock: gives the LARGEST dP, hence the most generous
    # ceiling. If even the most generous number is small, the branch is closed.
    HIGH=$($PY -c "
import pynvml as n
n.nvmlInit(); h=n.nvmlDeviceGetHandleByIndex(0)
print(sorted(n.nvmlDeviceGetSupportedGraphicsClocks(h, n.nvmlDeviceGetSupportedMemoryClocks(h)[0]))[-1])")
  fi
  echo "  high leg: ${HIGH} MHz (max supported -- most generous dP)"
  echo "  NOTE: no traffic runs during this. Stop any serving engine first."
  bash tools/idle_gap_ceiling.sh "$HIGH" "" 60
  echo
  echo "  dP is above. The CEILING needs a denominator -- mean power under a"
  echo "  real workload at the tuned clock -- which you do not have until a"
  echo "  benchmark runs. Re-run tools/idle_gap_ceiling.py with --loaded-w then."
}

case "$PHASE" in
  check)   phase0; phase1 ;;
  energy)  phase2 ;;
  idlegap) phase3 "$@" ;;
  *) echo "usage: $0 {check|energy|idlegap}"; exit 2 ;;
esac
