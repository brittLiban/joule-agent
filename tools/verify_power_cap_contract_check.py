"""Exercise `tools/verify_power_cap.py` as a subprocess, against the CLI seam.

Why this exists: the first `hardware-safety-reviewer` pass on `verify_power_cap.py`
returned 5 BLOCKING findings, and **both of the argument-refusal ones lived in
`main()`**, where the 25 unit tests -- all of which call `judge_power`,
`capped_phases_for` or `baseline_phase_for` directly -- could not reach them:

  * no argument-only clock-floor pre-check, so `--clock-mhz 200` entered the
    context manager, **armed, published a guard claim**, and only then raised
    `ClockFloorError`. `gpu_guard.main()` pre-checks consent *and* the floor
    before `with guard:` for exactly this reason;
  * the cap-range pre-check was conditional on an NVML read that failed
    silently into `None`, so when it lapsed the guaranteed refusal was deferred
    to mid-run -- after a clock lock had already landed on the device, with the
    durable artifact then recording `NOT RUN`.

That is the third component in this repo where the CLI seam hid a defect the
mocked suite could not see (`gpu_guard`: 3 defects, `sweep.py`: 2). The rule
that followed is in CLAUDE.MD: any new CLI entry point gets a
`*_contract_check.py` subprocess exercise.

**Every check here is reachable without root, and none of them starts the GPU
load.** That is not a limitation, it is the point: the argument-only ordering
rule ("every refusal depending only on the arguments or the caller -- never on
the device -- must happen before the guard is entered") is what was broken, and
those paths all return before `baseline_phase_for` is called.

The load-bearing assertion in most checks is not the exit code but **what the
run left on disk**: a refused caller must leave no guard state file at all,
because `arm()` publishes a claim and a caller that was never allowed to write
must never have claimed the device.

Usage:
    .venv/bin/python tools/verify_power_cap_contract_check.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "verify_power_cap.py"
PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok and detail:
        print(f"        {detail[:400]}")


def run(*args, state=None, out=None):
    argv = [sys.executable, str(TOOL), *args]
    if state is not None:
        argv += ["--state-path", str(state)]
    if out is not None:
        argv += ["--out", str(out)]
    return subprocess.run(argv, cwd=REPO, capture_output=True, text=True,
                          timeout=180)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="vpc-contract-"))

    # ---------------------------------------------------------------- refusals
    print("\nargument-only refusals (must precede the guard entirely)")

    cases = [
        ("caps below the device minimum are refused",
         ["--caps-w", "55"], "outside this device's settable range"),
        ("caps above the device maximum are refused",
         ["--caps-w", "250"], "outside this device's settable range"),
        ("an all-at-or-above-default ladder is refused as a tautology",
         ["--caps-w", "200"], "tautology"),
        ("an empty cap list is refused",
         ["--caps-w", ""], "no caps"),
        # THE B5 regression. Below MIN_SM_CLOCK_MHZ = 300.
        ("a below-floor clock is refused BEFORE the guard is entered",
         ["--caps-w", "130", "--clock-mhz", "200"], "below the"),
        ("a zero clock is refused: the lock is the control variable",
         ["--caps-w", "130", "--clock-mhz", "0"], "control variable"),
    ]
    for i, (name, extra, needle) in enumerate(cases):
        state = tmp / f"state{i}.json"
        art = tmp / f"art{i}.json"
        r = run(*extra, state=state, out=art)
        msg = r.stdout + r.stderr
        check(f"{name}: exit 2", r.returncode == 2, msg)
        check(f"{name}: says why", needle in msg, msg)
        # The load-bearing one.
        check(f"{name}: leaves NO guard state file",
              not state.exists(),
              f"state file exists: {state.read_text()[:200] if state.exists() else ''}")

    # ------------------------------------------------------------- artifacts
    print("\nevidence on disk")

    state = tmp / "state-art.json"
    art = tmp / "refusal.json"
    r = run("--caps-w", "55", state=state, out=art)
    check("a refusal still writes an artifact", art.exists(),
          "no artifact: a refusal that writes nothing leaves the PREVIOUS run's "
          "verdict on disk as the newest evidence")
    if art.exists():
        doc = json.loads(art.read_text())
        check("the refusal artifact says NOT RUN",
              doc.get("verdict", {}).get("result") == "NOT RUN", str(doc)[:200])
        check("the refusal artifact carries provenance",
              bool(doc.get("provenance", {}).get("utc")), str(doc)[:200])
        check("provenance records the device power range it validated against",
              doc["provenance"].get("power_range_read_ok") is True
              and doc["provenance"].get("power_min_w") is not None,
              str(doc.get("provenance"))[:300])

    # ------------------------------------------------------------ empty caps
    # `--caps-w ""` returns before an artifact path is even chosen; assert the
    # message rather than a file, and that it still leaves no claim.
    print("\nrejudge (re-analysis must be traceable and must refuse partials)")

    part = tmp / "partial.json"
    part.write_text(json.dumps({
        "caps_w": [130, 100], "clock_mhz": 1800,
        "phases": [{"phase": "baseline"}, {"phase": "cap@130W"}]}))
    r = run("--rejudge", str(part), out=tmp / "rj1.json")
    check("rejudge refuses an incomplete run rather than mis-indexing",
          r.returncode != 0 and "not a complete run" in (r.stdout + r.stderr),
          r.stdout + r.stderr)

    def ph(name, mhz, peak, limit_mw):
        return {"phase": name, "samples": 80, "median_mhz": mhz,
                "min_mhz": mhz, "max_mhz": mhz, "util_median_pct": 99.0,
                "loaded": True, "power_samples": 400, "ring_samples": 400,
                "polled_samples": 0, "power_source": "ring",
                "power_mean_w": peak - 10, "power_p95_w": peak - 2,
                "power_peak_w": peak, "power_limit_readback_mw": limit_mw}

    full = tmp / "full.json"
    full.write_text(json.dumps({
        "caps_w": [130, 100], "clock_mhz": 1800,
        "verdict": {"result": "INCONCLUSIVE"},
        "phases": [ph("baseline", 1800, 150.0, 200_000),
                   ph("cap@130W", 1700, 128.0, 130_000),
                   ph("cap@100W", 1450, 99.0, 100_000),
                   ph("recovered", 1800, 150.0, 200_000)]}))
    rj = tmp / "rj2.json"
    r = run("--rejudge", str(full), out=rj)
    check("rejudge of a complete run succeeds", rj.exists(), r.stdout + r.stderr)
    if rj.exists():
        doc = json.loads(rj.read_text())
        check("rejudge names its source and the original verdict",
              doc["rejudged_from"]["original_verdict"] == "INCONCLUSIVE"
              and str(full) in doc["rejudged_from"]["artifact"], str(doc)[:200])
        check("rejudge records the predicates it actually used",
              "predicates" in doc["rejudged_from"], str(doc)[:300])
        check("rejudge says it is not a re-run",
              "NOT a re-run" in doc["rejudged_from"]["note"], str(doc)[:300])
        check("rejudge exit code tracks the verdict",
              r.returncode == 0 and doc["verdict"]["result"] == "CAP WORKS",
              f"rc={r.returncode} verdict={doc['verdict']['result']}")

    # -------------------------------------------------------------- privilege
    # A VALID argument set, unprivileged. This must reach the privilege refusal
    # WITHOUT starting the GPU load -- `lock_clocks` raises before `run_phase`
    # is called -- so it is safe to run here and on a busy machine.
    #
    # Unlike the argument refusals above, a state file MAY exist: gpu_guard
    # deliberately checks privilege after arming (CLAUDE.MD records this as
    # accepted). What must hold is that the record is SETTLED, never a live
    # claim on a device this process was not allowed to touch.
    print("\nprivilege refusal (valid arguments, no root, no load started)")
    state = tmp / "state-priv.json"
    art = tmp / "priv.json"
    r = run("--caps-w", "130", "--clock-mhz", "1800", "--seconds", "1",
            state=state, out=art)
    msg = r.stdout + r.stderr
    check("an unprivileged run does not exit 0", r.returncode != 0, msg)
    check("an unprivileged run never started a phase",
          "phase: capped" not in msg,
          "a capped phase began: the privilege refusal must precede any load")
    if state.exists():
        rec = json.loads(state.read_text()).get("record", {})
        check("any state left by a privilege refusal is SETTLED, not a live claim",
              rec.get("restored") is True, state.read_text()[:300])
        # Stronger than `restored`: the record must show the device was never
        # touched. `restored: true` alone would also describe a guard that
        # locked clocks and successfully put them back -- which an unprivileged
        # caller must never have been able to do.
        check("a privilege refusal recorded NO hardware write",
              rec.get("clocks_locked") is False
              and rec.get("power_limit_written_mw") is None
              and rec.get("restore_failed") is False,
              state.read_text()[:400])
    else:
        check("any state left by a privilege refusal is SETTLED, not a live claim",
              True, "")
        check("a privilege refusal recorded NO hardware write", True, "")

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED: " + "; ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
