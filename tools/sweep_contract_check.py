"""Exercise `python -m joule_agent.sweep` as a subprocess, against the CLI seam.

Why this exists: a `hardware-safety-reviewer` pass found two blocking defects
in `sweep.py` that **241 unit tests could not see**, because both lived in
`main()` and no test imported it. One called `check_stale_state` with the wrong
signature (uncaught `TypeError`, so every locked sweep died on a traceback and
the guard-contract pre-flight had never run); the other handed one NVML
controller to a fresh `GpuGuard` per point, which `_ClaimMixin` refuses on the
second point, silently degrading every locked sweep to "one point, then abort".

That is the same seam `tools/cli_contract_check.py` was built for after three
defects reached committed code through it in `gpu_guard`. Unit tests construct
`SweepRunner` with injected fakes and never cross into `main()`, so the
argument parsing, the refusal ordering, the controller lifecycle and the
on-disk residue are all unexercised by them.

**Every check here is reachable without root**, deliberately: the refusal paths
are where the argument-only ordering rule lives ("every refusal depending only
on the arguments or the caller must happen before the guard is entered"), and
they are the ones that were broken. A run that needs root would also need a
serving engine to be meaningful, which is the separate live gate.

The load-bearing assertion in most checks is not the exit code but **what the
run left on disk**: a refused sweep must leave no guard state file at all,
because arming publishes a claim and a caller that was never allowed to write
must never have claimed the device.

Usage:
    .venv/bin/python tools/sweep_contract_check.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok and detail:
        print(f"        {detail[:300]}")


def run(*args, state=None, out=None):
    argv = [sys.executable, "-m", "joule_agent.sweep", "--model", "m", *args]
    if state is not None:
        argv += ["--state-path", str(state)]
    if out is not None:
        argv += ["--out", str(out)]
    return subprocess.run(argv, cwd=REPO, capture_output=True, text=True,
                          timeout=120)


CONSENT = "--i-understand-this-changes-gpu-settings"


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="sweep-contract-"))
    print(__doc__.splitlines()[0])
    print()

    # -- refusals that depend only on the arguments --------------------------
    print("argument-only refusals (no device, no root):")

    s = tmp / "a.json"
    r = run("--clocks", "900", state=s, out=tmp / "a")
    check("a clock lock without consent is REFUSED", r.returncode == 2,
          f"rc={r.returncode} {r.stderr[:160]}")
    check("...and the refusal names the flag that would grant consent",
          CONSENT in r.stderr, r.stderr[:200])
    check("...and nothing armed: no guard state file", not s.exists())

    s = tmp / "b.json"
    r = run("--clocks", "10", CONSENT, state=s, out=tmp / "b")
    check("a clock below the floor is REFUSED", r.returncode == 2,
          f"rc={r.returncode} {r.stderr[:160]}")
    check("...and the refusal names the floor, not privilege",
          "floor" in r.stderr.lower() and "sudo" not in r.stderr.lower(),
          r.stderr[:200])
    check("...and nothing armed: no guard state file", not s.exists())

    # The floor must be refused BEFORE privilege: a bad argument should be
    # reported as such rather than sending an operator to sudo first. This is
    # the rule gpu_guard states for itself; the sweep inherits it.
    s = tmp / "c.json"
    r = run("--clocks", "900", CONSENT, state=s, out=tmp / "c")
    check("a lock without root is REFUSED (running unprivileged)",
          r.returncode == 2, f"rc={r.returncode} {r.stderr[:160]}")
    check("...and the refusal is about privilege", "root" in r.stderr.lower(),
          r.stderr[:200])
    check("...and nothing armed: no guard state file", not s.exists())

    r = run("--preset", "saturated", out=tmp / "d")
    check("a closed-loop preset is REFUSED", r.returncode == 2,
          f"rc={r.returncode}")
    check("...and the refusal explains the confound",
          "response time" in r.stderr, r.stderr[:200])

    r = run("--preset", "nope", out=tmp / "e")
    check("an unknown preset is REFUSED", r.returncode == 2, f"rc={r.returncode}")

    r = run("--clocks", "1300:900:100", CONSENT, out=tmp / "f")
    check("an inverted clock range is REFUSED", r.returncode == 2,
          f"rc={r.returncode}")

    r = run("--repeats", "0", out=tmp / "g")
    check("--repeats 0 is REFUSED", r.returncode == 2, f"rc={r.returncode}")

    # -- output destination, checked before any measurement ------------------
    print("\noutput destination:")
    blocked = tmp / "blocked"
    blocked.mkdir()
    blocked.chmod(0o500)
    try:
        r = run("--duration", "1", "--repeats", "1", out=blocked / "sub")
        check("an unwritable --out is REFUSED before measuring",
              r.returncode == 2 and "output directory" in r.stderr,
              f"rc={r.returncode} {r.stderr[:200]}")
        check("...and no sweep ran (no report printed)",
              "STATIC SWEEP" not in r.stdout, r.stdout[:160])
    finally:
        blocked.chmod(0o700)

    # -- the stock-only dry run needs neither root nor consent ---------------
    print("\nstock-only dry run (no privilege, no consent, no hardware write):")
    s = tmp / "h.json"
    out = tmp / "h"
    r = run("--duration", "1", "--repeats", "1", state=s, out=out)
    check("a stock-only sweep RUNS unprivileged", r.returncode == 0,
          f"rc={r.returncode} {r.stderr[:300]}")
    check("...and writes no guard state file (it armed nothing)",
          not s.exists())

    doc = {}
    if (out / "sweep.json").exists():
        doc = json.loads((out / "sweep.json").read_text())
    check("...and writes sweep.json", bool(doc))
    check("...with provenance naming the tree and the device",
          bool(doc.get("provenance", {}).get("gpu_guard_sha256"))
          and "device" in doc.get("provenance", {}),
          str(doc.get("provenance"))[:200])
    check("...and a tri-state dirty flag, never a bare False from a failed read",
          doc.get("provenance", {}).get("git_dirty") in (True, False, None))
    check("...and names no best point when it has no token counts",
          doc.get("best_point") is None, str(doc.get("best_point")))
    check("...and says so rather than reporting a silent zero",
          any("no best static point" in w or "SLO threshold" in w
              for w in doc.get("warnings", [])),
          str(doc.get("warnings"))[:250])
    check("...and did not claim the GPU may be modified",
          doc.get("hardware_may_be_modified") is False)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
