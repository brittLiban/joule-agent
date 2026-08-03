"""Verify that a clock lock actually changes the device -- not just returns success.

Why this exists: this driver exposes no `nvmlDeviceGetGpuLockedClocks`, so
`gpu_guard` cannot confirm by readback that a lock took effect. NVML returning
success is not evidence. GeForce parts restrict some NVML write APIs, and a
silently ignored `nvmlDeviceSetGpuLockedClocks` would be the worst possible
failure for this project: every configuration in the static sweep would measure
identically, and the sweep would look like it ran while proving nothing.

The check is behavioural. Put the GPU under sustained load and watch the SM
clock:

    baseline      no lock            -> clock boosts to whatever the card wants
    locked@T      lock at each T     -> clock should sit at T
    recovered     after restore      -> clock should return to BASELINE

**Several targets, not one.** A single locked point shows that a lock differs
from stock. The sweep needs more than that: it needs different locks to differ
*from each other*, because every row of its output is labelled by a requested
clock. A device that honours 900 and snaps everything else to one bin passes a
single-point check and silently corrupts the sweep, so the targets are swept and
their achieved clocks checked for mutual separation and monotonicity.

**What the predicates must not do** -- both of these were real defects here:

* ``recovered`` is compared against **baseline**, never against the locked
  value. Comparing it to the locked value asks only "did the clock go up",
  which a card left pinned at 1400 MHz after a *failed* restore satisfies --
  and this tool then printed LOCK WORKS and exited 0. The restore check is the
  one that must not be loose; it is the guard's whole contract.
* A target close to the unlocked boost clock **cannot be discriminated** by
  this method, and is reported INCONCLUSIVE rather than as a no-op. The old
  fixed "must be below baseline minus tolerance" rule labelled a perfectly
  working 1800 MHz lock on a 1920 MHz card a NO-OP -- failing precisely in the
  upper half of the range the sweep wants to use.

**Each phase carries evidence the GPU was loaded.** The load runs in a
subprocess with its output discarded; if it dies at startup (no `libcuda`,
`cuInit` failing under sudo's environment, PTX rejected for the arch) an
unchecked phase samples an *idle* GPU. That is not hypothetical -- this repo
already shipped a tool that read an idle 210 MHz as a lock. So every phase
asserts the loader is still alive and records GPU utilization alongside the
clock; a phase that was not loaded is not a measurement.

All clock writes go through :mod:`joule_agent.gpu_guard`, per the project rule
that nothing else touches GPU state. Requires root for the locked phases.

Usage:
    sudo .venv/bin/python tools/verify_clock_lock.py --targets 600,1000,1400,1800

Known limits, deliberately not fixed here: the device index is fixed at 0, so
"re-run per GPU" is not executable as-is on a multi-GPU node; and this tool
gates the *clock* axis only -- the power-limit axis of component 4 has no
equivalent efficacy gate.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
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

#: A phase with utilization below this was not meaningfully loaded, so its
#: clock reading describes an idle card rather than a configuration.
MIN_LOADED_UTIL_PCT = 50.0


def load_argv(seconds: float, python: str) -> list:
    """The exact command each phase runs to drive the GPU.

    One definition, used by both :func:`phase` and :func:`provenance`, so the
    recorded load is the load that ran rather than a hand-copied description
    of it that can drift.
    """
    return [python, str(LOAD_SCRIPT),
            "--duration", str(seconds + 3),
            "--period", str(seconds + 3),
            "--duty", "1.0"]


def _git(*args) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                           text=True, timeout=10)
    except Exception:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def _dirty_flag():
    """``True``/``False``/``None`` -- unknown is not clean.

    ``bool(_git(...))`` collapses a failed git call into ``False``, which reads
    as a clean tree. A provenance field whose failure mode is
    indistinguishable from a real answer is the defect this whole block exists
    to prevent.
    """
    porcelain = _git("status", "--porcelain")
    return None if porcelain is None else bool(porcelain)


def _sha256(path: Path) -> str | None:
    """Content hash of the module under test.

    A git sha does not identify the code when the tree is dirty -- and the run
    that first carried provenance here was itself taken on a dirty tree. This
    answers "is this the code that was measured?" regardless of tree state.
    """
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def sample_device(seconds: float, interval: float = 0.1, proc=None) -> dict:
    """Sample SM clock and utilization, and watch that the load stays alive."""
    import pynvml

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    clocks, utils = [], []
    load_died = False
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < seconds:
            if proc is not None and proc.poll() is not None:
                load_died = True
                break
            try:
                clocks.append(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
            except Exception:
                pass
            try:
                utils.append(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
            except Exception:
                pass
            time.sleep(interval)
    finally:
        pynvml.nvmlShutdown()
    return {"clocks": clocks, "utils": utils, "load_died": load_died}


def phase(name: str, seconds: float, python: str) -> dict:
    """Run sustained load and sample the device throughout."""
    proc = subprocess.Popen(
        load_argv(seconds, python),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(2.0)  # let the load ramp before sampling
        if proc.poll() is not None:
            return {"phase": name, "samples": 0, "loaded": False,
                    "error": f"load process exited during ramp (rc={proc.returncode}); "
                             "this phase measured nothing"}
        s = sample_device(seconds, proc=proc)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    clocks, utils = s["clocks"], s["utils"]
    if not clocks:
        return {"phase": name, "samples": 0, "loaded": False,
                "error": "no clock samples collected"}
    util_median = statistics.median(utils) if utils else None
    loaded = (not s["load_died"]
              and util_median is not None
              and util_median >= MIN_LOADED_UTIL_PCT)
    out = {
        "phase": name,
        "samples": len(clocks),
        "min_mhz": min(clocks),
        "max_mhz": max(clocks),
        "median_mhz": statistics.median(clocks),
        "util_median_pct": util_median,
        "loaded": loaded,
    }
    if not loaded:
        out["error"] = (
            f"GPU was not under load (median utilization {util_median}%, "
            f"load process died: {s['load_died']}). An unloaded card sits near "
            "its idle clock whether or not a lock is applied, so this phase "
            "cannot distinguish them."
        )
    return out


def provenance(args, python: str) -> dict:
    """Stamp what produced this result: when, which code, which card."""
    prov = {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": _dirty_flag(),
        "gpu_guard_sha256": _sha256(REPO / "joule_agent" / "gpu_guard.py"),
        "python": python,
        "args": {"targets_mhz": args.targets, "seconds": args.seconds,
                 "tolerance_mhz": args.tolerance_mhz},
        "load_argv": load_argv(args.seconds, python),
        "driver_version": None,
        "nvml_version": None,
        "device": None,
    }
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            for key, fn in (("driver_version", pynvml.nvmlSystemGetDriverVersion),
                            ("nvml_version", pynvml.nvmlSystemGetNVMLVersion)):
                try:
                    v = fn()
                    prov[key] = v.decode() if isinstance(v, bytes) else str(v)
                except Exception:
                    pass
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            dev = {"index": 0}
            for key, fn in (("name", pynvml.nvmlDeviceGetName),
                            ("uuid", pynvml.nvmlDeviceGetUUID)):
                try:
                    v = fn(h)
                    dev[key] = v.decode() if isinstance(v, bytes) else str(v)
                except Exception:
                    dev[key] = None
            prov["device"] = dev
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    return prov


def judge(baseline: dict, locked: list, recovered: dict, targets: list,
          tolerance: int) -> dict:
    """Decide what the phases prove. Pure -- no hardware, so it is testable."""
    phases = [baseline, *locked, recovered]
    unloaded = [p["phase"] for p in phases if not p.get("loaded")]
    if unloaded:
        return {"result": "INCONCLUSIVE",
                "reason": f"phases not under load: {', '.join(unloaded)}. "
                          "An idle card's clock says nothing about a lock."}

    base = baseline["median_mhz"]
    rec = recovered["median_mhz"]

    # Restore is checked against BASELINE. Never against the locked value --
    # that only asks whether the clock rose, which a partial restore satisfies.
    restored_to_baseline = abs(rec - base) <= tolerance

    per_target, effective, inconclusive = [], [], []
    for tgt, ph in zip(targets, locked):
        achieved = ph["median_mhz"]
        near = abs(achieved - tgt) <= tolerance
        # A target within tolerance of the unlocked boost clock cannot be
        # distinguished from "no lock at all" by this method. That is a limit
        # of the probe, not a failure of the device.
        discriminable = abs(tgt - base) > tolerance
        row = {"target_mhz": tgt, "achieved_median_mhz": achieved,
               "near_target": near, "discriminable": discriminable,
               "changed_from_baseline": abs(achieved - base) > tolerance}
        per_target.append(row)
        if not discriminable:
            inconclusive.append(tgt)
        elif near and row["changed_from_baseline"]:
            effective.append(tgt)

    verdict = {
        "per_target": per_target,
        "recovered_to_baseline": restored_to_baseline,
        "baseline_median_mhz": base,
        "recovered_median_mhz": rec,
    }

    if not restored_to_baseline:
        verdict["result"] = "RESTORE FAILED"
        verdict["reason"] = (
            f"after restore the clock sat at {rec:.0f} MHz against a "
            f"{base:.0f} MHz baseline. The GPU may still be locked -- run "
            "`sudo python -m joule_agent.gpu_guard restore`."
        )
        return verdict

    testable = [r for r in per_target if r["discriminable"]]
    if not testable:
        verdict["result"] = "INCONCLUSIVE"
        verdict["reason"] = (
            f"every target was within {tolerance} MHz of the {base:.0f} MHz "
            "unlocked clock, so none could be distinguished from no lock. "
            "Choose targets well below the boost clock."
        )
        return verdict

    if not any(r["changed_from_baseline"] for r in testable):
        verdict["result"] = "LOCK IS A NO-OP"
        verdict["reason"] = (
            "no target changed the clock from its unlocked baseline. NVML "
            "accepted the writes but the device ignored them. A static sweep "
            "on this hardware would compare identical configurations."
        )
        return verdict

    off_target = [r for r in testable if not r["near_target"]]
    achieved = [r["achieved_median_mhz"] for r in testable]
    # Distinct requests must produce distinct clocks, or sweep rows labelled by
    # requested clock are not distinct configurations.
    separated = all(abs(a - b) > tolerance
                    for i, a in enumerate(achieved) for b in achieved[i + 1:])
    monotonic = all(a <= b for a, b in zip(achieved, achieved[1:]))
    verdict["targets_mutually_separated"] = separated
    verdict["achieved_monotonic_in_request"] = monotonic

    if off_target:
        verdict["result"] = "PARTIAL"
        verdict["reason"] = (
            "the device changed clock but snapped to bins other than the "
            "requested values: "
            + "; ".join(f"asked {r['target_mhz']} got "
                        f"{r['achieved_median_mhz']:.0f}" for r in off_target)
            + ". Treat achieved values as the real settings; a sweep row "
              "labelled by its request would be mislabelled."
        )
    elif not separated:
        verdict["result"] = "PARTIAL"
        verdict["reason"] = (
            "two or more targets achieved the same clock, so they are one "
            "configuration measured twice rather than distinct sweep points."
        )
    else:
        verdict["result"] = "LOCK WORKS"
        verdict["reason"] = (
            "each target held its requested clock, the targets were mutually "
            f"separated, and the clock returned to {rec:.0f} MHz against a "
            f"{base:.0f} MHz baseline after restore"
        )
    if inconclusive:
        verdict["reason"] += (
            f". Targets {inconclusive} were within {tolerance} MHz of the "
            "unlocked clock and could not be tested."
        )
    return verdict


def locked_phases_for(targets, run_phase, controller_factory=None,
                      guard_factory=None):
    """Lock each target in turn and yield the phase measured under it.

    Extracted from ``main()`` so a test can drive it with a fake controller.
    The first version of this loop lived entirely inside ``main()``, shared one
    controller across every target, and **failed on the second target the first
    time it was run for real** -- ``_ClaimMixin.claim`` refuses a second owner
    and nothing releases ``_claimed_by``. The identical defect had already been
    found in ``sweep.py`` by a review pass; writing it again here is what
    established that per-guard controller ownership is a rule rather than an
    optimisation, and that a loop only ``main()`` can reach is a loop nothing
    checks.
    """
    controller_factory = controller_factory or (lambda: NvmlGpuController(index=0))
    guard_factory = guard_factory or (lambda c: GpuGuard(c, consent=True))
    for tgt in targets:
        print(f"phase: locked at {tgt} MHz", flush=True)
        controller = controller_factory()
        guard = guard_factory(controller)
        try:
            with guard:
                guard.lock_clocks(tgt)
                ph = run_phase(tgt)
        finally:
            # Do NOT close while this guard still owes a restore. close() shuts
            # NVML down, and the atexit last-chance restore would then fail
            # against a dead handle -- turning a possible recovery into a
            # certain failure.
            if getattr(guard, "_restored", True):
                controller.close()
            else:
                print(
                    "WARNING [verify_clock_lock]: guard state unrestored; "
                    "leaving NVML open so the exit-time restore can run.",
                    file=sys.stderr,
                )
        yield ph


def _write(results: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"written to {out}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--targets", default="600,1000,1400,1800",
                   help="comma-separated MHz targets to sweep")
    p.add_argument("--target-mhz", type=int, default=None,
                   help="single target (compatibility alias for --targets)")
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--tolerance-mhz", type=int, default=120,
                   help="how close an achieved clock must be to its target, "
                        "and how close recovered must be to baseline")
    p.add_argument("--out", default=None,
                   help="output path; default is a per-run timestamped file so "
                        "a re-run cannot erase the evidence from the last one")
    args = p.parse_args(argv)

    if args.target_mhz is not None:
        args.targets = str(args.target_mhz)
    targets = sorted(int(t) for t in args.targets.split(",") if t.strip())
    if not targets:
        print("REFUSED: no targets", file=sys.stderr)
        return 2

    python = sys.executable
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = Path(args.out) if args.out else Path(
        f"results/clock_lock_verification-{stamp}.json")

    args.targets = targets
    results = {"provenance": provenance(args, python), "targets_mhz": targets,
               "phases": []}

    def bail(reason: str, code: int) -> int:
        # An artifact is written even on a refusal. Returning early without one
        # leaves the PREVIOUS run's verdict on disk as the newest evidence --
        # the exact "cited for a tree it never measured" failure this file's
        # provenance block exists to end.
        results["verdict"] = {"result": "NOT RUN", "reason": reason}
        _write(results, out)
        print(f"REFUSED: {reason}", file=sys.stderr)
        return code

    print("phase: baseline (no lock)", flush=True)
    baseline = phase("baseline", args.seconds, python)
    results["phases"].append(baseline)
    print(f"  {baseline}", flush=True)

    locked_phases = []
    try:
        for ph in locked_phases_for(targets,
                                    lambda t: phase(f"locked@{t}", args.seconds, python)):
            locked_phases.append(ph)
            results["phases"].append(ph)
            print(f"  {ph}", flush=True)
    except GpuGuardError as exc:
        return bail(str(exc), 2)

    print("phase: recovered (after restore)", flush=True)
    recovered = phase("recovered", args.seconds, python)
    results["phases"].append(recovered)
    print(f"  {recovered}", flush=True)

    results["verdict"] = judge(baseline, locked_phases, recovered, targets,
                               args.tolerance_mhz)
    verdict = results["verdict"]

    prov = results["provenance"]
    dev = prov.get("device") or {}
    dirty = prov.get("git_dirty")
    print("\n" + "=" * 62)
    print(f"VERDICT: {verdict['result']}")
    print(f"  {verdict.get('reason', '')}")
    print("=" * 62)
    print(f"  tree      {prov.get('git_sha')}"
          f"{' (DIRTY)' if dirty else ' (UNKNOWN dirty state)' if dirty is None else ''}")
    print(f"  gpu_guard sha256:{(prov.get('gpu_guard_sha256') or '?')[:16]}")
    print(f"  device    {dev.get('name')} {dev.get('uuid')}")
    print(f"  driver    {prov.get('driver_version')}")
    _write(results, out)
    return 0 if verdict.get("result") == "LOCK WORKS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
