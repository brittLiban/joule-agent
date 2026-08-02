"""Verify that a clock lock actually changes the device -- not just returns success.

Why this exists: this driver exposes no `nvmlDeviceGetGpuLockedClocks`, so
`gpu_guard` cannot confirm by readback that a lock took effect. NVML returning
success is not evidence. GeForce parts restrict some NVML write APIs, and a
silently ignored `nvmlDeviceSetGpuLockedClocks` would be the worst possible
failure for this project: every configuration in the static sweep would measure
identically, and the sweep would look like it ran while proving nothing.

The check is behavioural. Put the GPU under sustained load and watch the SM
clock in three phases:

    baseline   no lock          -> clock boosts to whatever the card wants
    locked     lock at target   -> clock should sit at the target
    recovered  after restore    -> clock should boost freely again

If `locked` is indistinguishable from `baseline`, the lock is a no-op on this
hardware and the sweep must not be trusted here.

All clock writes go through :mod:`joule_agent.gpu_guard`, per the project rule
that nothing else touches GPU state. Requires root for the locked phase.

Usage:
    sudo .venv/bin/python tools/verify_clock_lock.py --target-mhz 900
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from joule_agent.gpu_guard import (  # noqa: E402
    GpuGuard,
    GpuGuardError,
    NvmlGpuController,
)

LOAD_SCRIPT = REPO / "tools" / "synthetic_gpu_load.py"


def sample_sm_clock(seconds: float, interval: float = 0.1) -> list:
    import pynvml

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    out = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        try:
            out.append(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
        except Exception:
            pass
        time.sleep(interval)
    pynvml.nvmlShutdown()
    return out


def phase(name: str, seconds: float, python: str) -> dict:
    """Run sustained load and sample the SM clock throughout."""
    proc = subprocess.Popen(
        [python, str(LOAD_SCRIPT), "--duration", str(seconds + 3),
         "--period", str(seconds + 3), "--duty", "1.0"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(2.0)  # let the load ramp before sampling
        s = sample_sm_clock(seconds)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    if not s:
        return {"phase": name, "samples": 0}
    return {
        "phase": name,
        "samples": len(s),
        "min_mhz": min(s),
        "max_mhz": max(s),
        "median_mhz": statistics.median(s),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--target-mhz", type=int, default=900)
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--tolerance-mhz", type=int, default=120,
                   help="how close the locked median must be to the target")
    p.add_argument("--out", default="results/clock_lock_verification.json")
    args = p.parse_args(argv)

    python = sys.executable
    results = {"target_mhz": args.target_mhz, "phases": []}

    print("phase 1/3: baseline (no lock)", flush=True)
    baseline = phase("baseline", args.seconds, python)
    results["phases"].append(baseline)
    print(f"  {baseline}", flush=True)

    print(f"phase 2/3: locked at {args.target_mhz} MHz", flush=True)
    controller = NvmlGpuController(index=0)
    try:
        guard = GpuGuard(controller, consent=True)
        with guard:
            guard.lock_clocks(args.target_mhz)
            locked = phase("locked", args.seconds, python)
        # guard.__exit__ has now restored
    except GpuGuardError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    finally:
        controller.close()
    results["phases"].append(locked)
    print(f"  {locked}", flush=True)

    print("phase 3/3: recovered (after restore)", flush=True)
    recovered = phase("recovered", args.seconds, python)
    results["phases"].append(recovered)
    print(f"  {recovered}", flush=True)

    # ---- verdict -------------------------------------------------------
    verdict = {}
    if not (baseline.get("samples") and locked.get("samples") and recovered.get("samples")):
        verdict = {"result": "INCONCLUSIVE", "reason": "a phase collected no samples"}
    else:
        near_target = abs(locked["median_mhz"] - args.target_mhz) <= args.tolerance_mhz
        below_baseline = locked["median_mhz"] < baseline["median_mhz"] - args.tolerance_mhz
        boosts_again = recovered["median_mhz"] > locked["median_mhz"] + args.tolerance_mhz
        verdict = {
            "locked_near_target": near_target,
            "locked_below_baseline": below_baseline,
            "recovered_boosts_again": boosts_again,
        }
        if near_target and below_baseline and boosts_again:
            verdict["result"] = "LOCK WORKS"
            verdict["reason"] = (
                f"clock held at ~{locked['median_mhz']:.0f} MHz vs "
                f"{baseline['median_mhz']:.0f} baseline, and returned to "
                f"{recovered['median_mhz']:.0f} after restore"
            )
        elif not below_baseline:
            verdict["result"] = "LOCK IS A NO-OP"
            verdict["reason"] = (
                f"locked median {locked['median_mhz']:.0f} MHz is not below the "
                f"{baseline['median_mhz']:.0f} MHz baseline. NVML accepted the "
                "write but the device ignored it. A static sweep on this "
                "hardware would compare identical configurations."
            )
        elif not boosts_again:
            verdict["result"] = "RESTORE FAILED"
            verdict["reason"] = (
                f"clock did not recover after restore ({recovered['median_mhz']:.0f} "
                f"MHz vs {baseline['median_mhz']:.0f} baseline). The GPU may still "
                "be locked -- run `sudo python -m joule_agent.gpu_guard restore`."
            )
        else:
            verdict["result"] = "PARTIAL"
            verdict["reason"] = (
                f"lock changed the clock to {locked['median_mhz']:.0f} MHz but not "
                f"to the requested {args.target_mhz} MHz. The device may snap to "
                "supported clock bins; treat the achieved value as the real setting."
            )
    results["verdict"] = verdict

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 62)
    print(f"VERDICT: {verdict['result']}")
    print(f"  {verdict.get('reason', '')}")
    print("=" * 62)
    print(f"written to {out}")
    return 0 if verdict.get("result") in ("LOCK WORKS", "PARTIAL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
