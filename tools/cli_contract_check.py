#!/usr/bin/env python3
"""Exercise the gpu_guard CLI against a REAL GPU, the way an operator does.

Why this exists
---------------
Two real defects reached committed code and were missed entirely by 112
fake-NVML tests: `main()`'s ``lock`` path entered the context manager (which
arms, and arming publishes a claim) *before* checking consent, and later before
checking the clock floor. Both were found by typing the command at a shell.

The unit suite constructs ``GpuGuard`` directly and never crosses that seam.
This does the opposite: every check here is a subprocess invocation of
``python -m joule_agent.gpu_guard``, against a real device, asserting on exit
code, message content, and what the run left on disk.

Running it
----------
    .venv/bin/python tools/cli_contract_check.py            # unprivileged half
    sudo .venv/bin/python tools/cli_contract_check.py --root

(Use the venv interpreter explicitly. ``sudo -E`` is ignored on this system and
plain ``sudo python`` picks up the system interpreter, which has no pynvml.)

Unprivileged, it verifies every refusal path and that no refusal leaves a claim
behind. With ``--root`` it additionally runs the kill tests -- clean exit,
SIGINT, SIGTERM, and SIGKILL-then-recover -- **verifying the lock under load**,
not by reading the idle clock. See ``clock_under_load``: an idle card reads
~210 MHz whether or not it is locked, so a bare clock read cannot tell
"restored" from "still locked" in either direction. The load-based checks make
this take a couple of minutes with ``--root``.

Nothing here is skipped silently -- a check that cannot run says why. The kill
tests also refuse to run at all if the card is already locked when they start,
because every later measurement would be against a baseline that is not stock.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable
CONSENT = "--i-understand-this-changes-gpu-settings"

PASS, FAIL, SKIP = [], [], []


def run(args, state, timeout=90, **kw):
    return subprocess.run(
        [PY, "-m", "joule_agent.gpu_guard", "--gpu", "0", "--state-path", str(state)]
        + args,
        cwd=REPO, capture_output=True, text=True, timeout=timeout, **kw)


def check(name, cond, detail=""):
    """Record a check. `detail` prints on PASS as well as FAIL, deliberately.

    A run whose numbers are only visible when something breaks cannot be
    audited: a reader cannot tell a capped 900 MHz from a silently-dead load
    reading 210 MHz, because both satisfy the same assertion.
    """
    (PASS if cond else FAIL).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}"
          + (f"\n          {detail}" if detail else ""))
    return cond


def skip(name, why, covers=1):
    """Record a skip. `covers` is how many checks did NOT run.

    One skip() guarding a branch of three checks reported "1 skipped" and hid
    two. A tally that undercounts what was not run is the same error as a
    check that cannot fail.
    """
    SKIP.extend([name] * covers)
    print(f"  skip  {name}" + (f" ({covers} checks)" if covers > 1 else "")
          + f"\n          {why}")


def read_record(state):
    return json.loads(Path(state).read_text())["record"]


#: A card locked at 900 MHz cannot exceed it under load; an unlocked 3060 Ti
#: reaches ~1900. Wide margins on both sides so thermal or power capping cannot
#: turn an unlocked card into a false "locked".
LOCKED_CEILING = 1000
UNLOCKED_FLOOR = 1200


class Measurement:
    """A clock measurement that knows whether it is trustworthy.

    `peak` alone cannot be believed. If the synthetic load never ran, sampling
    an idle card returns ~210 MHz -- which satisfies "the lock capped the
    clock" just as well as a real 900 MHz cap does, so the check that exists to
    stop a release-test passing against a never-locked card would itself pass
    vacuously. Carry the load's exit status and the valid-sample count, and
    treat a bad measurement as an error rather than as a number.
    """

    #: A window is ~60 samples. Ten was too lenient: ten idle readings satisfy
    #: `usable`, and an idle reading also satisfies `capped`, which is the v2
    #: bug in miniature. Require most of the window.
    MIN_SAMPLES = 30

    def __init__(self, samples, load_rc, utils=()):
        # 0 MHz is not a reading, it is a failed query. Count how many were
        # dropped rather than quietly shrinking the window -- dropping the
        # lowest values also biases the median upward, which makes `released`
        # easier to satisfy, so it must not happen silently.
        self.dropped = sum(1 for s in samples if not s)
        self.samples = sorted(s for s in samples if s)
        self.load_rc = load_rc
        self.n = len(self.samples)
        # GPU utilisation, sampled alongside the clock. This is the only
        # INDEPENDENT evidence that the load actually drove the device. The
        # load's exit code says the kernel did not error; it does not say the
        # GPU was busy while we were sampling. Without this, a fully idle
        # 60-sample window with rc=0 is "usable" and satisfies `capped` -- the
        # v2 bug, surviving in a narrower form.
        u = sorted(utils)
        self.util = u[len(u) // 2] if u else None
        self.peak = self.samples[-1] if self.samples else 0
        if not self.samples:
            self.median = 0
        elif self.n % 2:
            self.median = self.samples[self.n // 2]
        else:
            # True median. samples[n // 2] is the UPPER middle, which biases
            # high -- in the permissive direction for `released`.
            lo, hi = self.samples[self.n // 2 - 1], self.samples[self.n // 2]
            self.median = (lo + hi) / 2

    @property
    def usable(self):
        """Trustworthy only if the load really ran AND we really sampled."""
        return (self.load_rc == 0 and self.n >= self.MIN_SAMPLES
                and self.dropped <= self.n // 10
                and self.util is not None and self.util >= 50)

    def __str__(self):
        return (f"peak {self.peak} MHz, median {self.median:g} MHz, "
                f"{self.n} samples"
                + (f" (+{self.dropped} failed)" if self.dropped else "")
                + f", util {self.util if self.util is not None else '?'}%"
                + f", load rc={self.load_rc}"
                + ("" if self.usable else "  <-- UNUSABLE"))


def clock_under_load(seconds=6.0):
    """Sample the SM clock while the GPU is actually being driven.

    **The idle clock is not a probe for a clock lock, and using it as one is a
    claim without evidence.** An idle card sits at ~210 MHz whether or not a
    lock is applied, and a card locked at [900, 900] sits at 900 when idle too
    -- so a bare `nvmlDeviceGetClockInfo` read cannot distinguish "restored"
    from "still locked" in either direction. The first version of this file did
    exactly that and reported four false failures, including "GPU LEFT
    MODIFIED" for a card reading 210 MHz, which is precisely the unlocked-idle
    value.

    This driver exposes no `nvmlDeviceGetGpuLockedClocks`. The only way to
    observe a lock is to demand clocks and see whether they are capped.
    """
    try:
        proc = subprocess.Popen(
            [PY, str(REPO / "tools/synthetic_gpu_load.py"),
             "--duration", str(seconds + 3), "--period", str(seconds + 3),
             "--duty", "1.0"],
            cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as exc:
        # The load could not even start. Return an UNUSABLE measurement rather
        # than raising: every caller already treats unusable as a failed check,
        # and a raise here would abort the run half-finished with the card
        # possibly still locked.
        return Measurement([], f"could not start: {exc}")
    samples, utils = [], []
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        time.sleep(2.0)                       # let the clocks ramp
        end = time.time() + seconds
        while time.time() < end:
            try:
                samples.append(
                    pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
                utils.append(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
            except Exception:
                pass
            time.sleep(0.1)
        pynvml.nvmlShutdown()
    except Exception:
        pass
    finally:
        try:
            rc = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = -1
    return Measurement(samples, rc, utils)


def capped(m, name):
    """The lock is holding: NO sample may exceed the ceiling (strong form)."""
    return check(name, m.usable and m.peak <= LOCKED_CEILING, str(m))


def released(m, name):
    """The lock is gone: the MEDIAN must clear the floor.

    Not the peak. `peak >= floor` is satisfied by a single spurious sample,
    and these are the claim's central assertions -- they should not use the
    statistic that is easiest to satisfy by accident.
    """
    return check(name, m.usable and m.median >= UNLOCKED_FLOOR, str(m))


def sm_clock():
    """Live SM clock, read straight from NVML. None if unreadable."""
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        v = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
        pynvml.nvmlShutdown()
        return v
    except Exception:
        return None


def nominal_check_count(func_name):
    """How many checks the named function runs on its nominal path.

    Derived from this file's own AST rather than maintained by hand. The
    hand-written `covers=` constants drifted twice -- once hiding two checks
    behind a single skip, once off by one after a check was added -- and a
    tally that undercounts what did not run is the same error as a check that
    cannot fail: the transcript reports more coverage than it has.
    """
    import ast
    tree = ast.parse(Path(__file__).read_text())
    fn = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == func_name), None)
    if fn is None:
        return None

    def sites(node):
        return [n for n in ast.walk(node)
                if isinstance(n, ast.Call)
                and getattr(n.func, "id", "") in ("check", "capped", "released")]

    looped = {id(c) for lp in ast.walk(fn) if isinstance(lp, ast.For)
              for c in sites(lp)}
    all_sites = sites(fn)
    outside = sum(1 for c in all_sites if id(c) not in looped)
    inside = sum(1 for c in all_sites if id(c) in looped)
    # Two signals per loop; one site is the mutually-exclusive timeout branch.
    return outside + 2 * max(inside - 1, 0)


def device_provenance():
    """Name the hardware in the transcript rather than relying on the operator.

    A result recorded as "on the 3060 Ti, driver 595.71.05" is only as good as
    whoever typed that afterwards. Print it from the device.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        def txt(v):
            return v.decode() if isinstance(v, bytes) else str(v)
        parts = [
            txt(pynvml.nvmlDeviceGetName(h)),
            f"uuid {txt(pynvml.nvmlDeviceGetUUID(h))}",
            f"driver {txt(pynvml.nvmlSystemGetDriverVersion())}",
        ]
        try:
            mode = pynvml.nvmlDeviceGetPersistenceMode(h)
            parts.append(f"persistence {'on' if mode else 'off'}")
        except Exception:
            parts.append("persistence unreadable")
        pynvml.nvmlShutdown()
        return " | ".join(parts)
    except Exception as exc:
        return f"device provenance UNREADABLE ({type(exc).__name__}: {exc})"


# ---------------------------------------------------------------------------
# Unprivileged: refusals, and what they leave behind
# ---------------------------------------------------------------------------


def unprivileged(tmp):
    print("\n-- refusals (no root needed) --")

    s = tmp / "a.json"
    r = run(["status"], s)
    ok = r.returncode in (0, 1)
    try:
        rep = json.loads(r.stdout)
    except Exception:
        rep = {}
    check("status runs unprivileged and emits JSON", ok and bool(rep),
          f"rc={r.returncode}, {len(rep)} fields")
    check("status needs no consent flag and reads the device",
          rep.get("power_readable") is True,
          f"power_readable={rep.get('power_readable')}")
    check("status leaves no state file", not s.exists(),
          f"state file present: {s.exists()}")

    s = tmp / "b.json"
    r = run(["lock", "--mhz", "900", "--hold", "1"], s)
    # rc==2 alone proves nothing: argparse exits 2 for its OWN errors, so a
    # renamed flag would satisfy it. Require gpu_guard's refusal.
    refused = r.returncode == 2 and r.stderr.startswith("REFUSED:")
    check("lock without consent is REFUSED by the guard", refused,
          f"rc={r.returncode}, stderr starts {r.stderr[:40]!r}")
    check("...and the refusal is about consent, not something else",
          "without explicit consent" in r.stderr, r.stderr[:160])
    check("...and publishes NO claim", not s.exists(),
          f"state file present: {s.exists()}")

    s = tmp / "c.json"
    r = run(["lock", "--mhz", "100", "--hold", "1", CONSENT], s)
    refused = r.returncode == 2 and r.stderr.startswith("REFUSED:")
    check("lock below the floor is REFUSED by the guard", refused,
          f"rc={r.returncode}, stderr starts {r.stderr[:40]!r}")
    check("...and refuses rather than clamping",
          "below the" in r.stderr and "NOT clamped" in r.stderr, r.stderr[:160])
    check("...and publishes NO claim", not s.exists(),
          f"state file present: {s.exists()}")

    if os.geteuid() == 0:
        # Do NOT run it. As root this is a real lock-and-restore cycle whose
        # result is then discarded -- an unasserted GPU write inside the
        # section labelled "no root needed", moments before the kill tests
        # take their unlocked baseline.
        skip("lock without root fails loudly",
             "running as root; the privilege refusal cannot be exercised here",
             covers=3)
    else:
        s = tmp / "d.json"
        r = run(["lock", "--mhz", "900", "--hold", "1", CONSENT], s)
        refused = r.returncode == 2 and r.stderr.startswith("REFUSED:")
        check("lock without root is REFUSED by the guard", refused,
              f"rc={r.returncode}, stderr starts {r.stderr[:40]!r}")
        check("...and says so rather than no-oping",
              "require root" in r.stderr, r.stderr[:160])
        # NOT an or-chain. "no file" and "settled record" are different
        # outcomes and only one of them is what this path should produce; an
        # or-chain also passes when the code was never reached at all.
        check("...and the record it left is settled",
              s.exists() and read_record(s)["restored"] is True,
              f"exists={s.exists()}, restored="
              f"{read_record(s)['restored'] if s.exists() else 'n/a'}")

    # A stale record for ANOTHER device must not send the operator at this one.
    s = tmp / "e.json"
    s.write_text(json.dumps({"schema": 3, "record": {
        "pid": 999_999, "started_utc": "2026-01-01T00:00:00+00:00",
        "device_index": 1, "device_uuid": "GPU-not-a-real-card",
        "snapshot": {"index": 1, "uuid": "GPU-not-a-real-card"},
        "clocks_locked": True, "locked_min_mhz": 900, "locked_max_mhz": 900,
        "power_limit_written_mw": None, "restored": False, "restored_utc": None,
        "restore_failed": False, "restore_errors": []}}))
    r = run(["status"], s)
    rep = json.loads(r.stdout) if r.stdout.strip().startswith("{") else {}
    check("a record for another card is reported stale", r.returncode == 1,
          f"rc={r.returncode}")
    check("...and is not claimed to be about this device",
          rep.get("record_is_for_this_device") is False,
          f"record_is_for_this_device={rep.get('record_is_for_this_device')}")
    check("...and the recommendation names the recorded GPU",
          "--gpu 1" in (rep.get("recommendation") or ""),
          f"{rep.get('recommendation')!r}")

    s = tmp / "f.json"
    s.write_text("{ this is not json")
    r = run(["status"], s)
    rep = json.loads(r.stdout) if r.stdout.strip().startswith("{") else {}
    check("an uninterpretable record is stale, not clean", r.returncode == 1,
          f"rc={r.returncode}")
    check("...and is labelled UNREADABLE rather than guessed",
          any("UNREADABLE" in f for f in rep.get("findings", [])),
          "; ".join(f.split(":")[0] for f in rep.get("findings", [])) or "no findings")
    check("...and status did not overwrite it",
          s.read_text() == "{ this is not json",
          f"file on disk: {s.read_text()!r}")

    r = run(["--gpu", "99", "status"], tmp / "g.json")
    check("a nonexistent device fails cleanly, not with a traceback",
          r.returncode != 0 and "Traceback" not in r.stderr,
          f"rc={r.returncode}, traceback={'Traceback' in r.stderr}")


# ---------------------------------------------------------------------------
# Root: the kill tests
# ---------------------------------------------------------------------------


def kill_tests(tmp):
    print("\n-- kill tests (real clock writes, verified under load) --")
    idle = sm_clock()
    if idle is None:
        skip("all kill tests", "could not read the SM clock from NVML", covers=22)
        return
    base = clock_under_load()
    print(f"          idle {idle} MHz; unlocked baseline: {base}")
    if not check("the card is unlocked before we start (harness precondition, "
                 "not a gpu_guard property)",
                 base.usable and base.median >= UNLOCKED_FLOOR, str(base)):
        skip("remaining kill tests",
             "cannot measure against a baseline that is not stock; run "
             "`sudo .venv/bin/python -m joule_agent.gpu_guard restore` first",
             covers=21)
        return

    def lock_proc(state, hold, extra=()):
        return subprocess.Popen(
            [PY, "-m", "joule_agent.gpu_guard", "--gpu", "0",
             "--state-path", str(state), "lock", "--mhz", "900",
             "--hold", str(hold), CONSENT, *extra],
            cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # 1. clean exit
    s = tmp / "k1.json"
    p = lock_proc(s, 30)
    capped(clock_under_load(), "clean run: the lock really caps the clock")
    rc = p.wait(timeout=120)
    check("clean run exits 0", rc == 0, f"rc={rc} {p.stderr.read()[-300:]!r}")
    check("clean run: the record ends settled",
          s.exists() and read_record(s)["restored"] is True,
          f"restored={read_record(s)['restored'] if s.exists() else 'no file'}")
    released(clock_under_load(), "clean run: the clock is released")

    # 2/3. SIGINT and SIGTERM mid-lock
    for signame, signum in (("SIGINT", signal.SIGINT), ("SIGTERM", signal.SIGTERM)):
        s = tmp / f"k-{signame}.json"
        p = lock_proc(s, 150)
        # Measured BEFORE the release check, so a release cannot pass against a
        # card that was never locked -- provided the load really ran, which
        # Measurement.usable is what establishes.
        capped(clock_under_load(), f"{signame}: the lock was applied first")
        p.send_signal(signum)
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill()
            check(f"{signame} terminates the run", False, "still running after 60s")
            continue
        released(clock_under_load(), f"{signame}: the clock is restored")
        check(f"{signame}: the record ends settled",
              s.exists() and read_record(s)["restored"] is True,
              f"restored={read_record(s)['restored'] if s.exists() else 'no file'}")

    # 4. SIGKILL, then recover
    s = tmp / "k4.json"
    r = run(["lock", "--mhz", "900", "--hold", "150", "--crash-after", "6", CONSENT],
            s, timeout=180)
    check("SIGKILL: the process really died by signal", r.returncode == -signal.SIGKILL,
          f"rc={r.returncode}")
    capped(clock_under_load(), "SIGKILL: the GPU is left modified, as expected")
    check("SIGKILL: the record survives saying so",
          s.exists() and read_record(s)["restored"] is False
          and read_record(s)["clocks_locked"] is True,
          f"{read_record(s) if s.exists() else 'NO FILE -- no evidence exists'}")

    r = run(["status"], s)
    check("SIGKILL: status reports it stale", r.returncode == 1, f"rc={r.returncode}")

    # CONTROL. Everything else in this file measures the card AFTER the last
    # NVML client detached, and with persistence off the driver tears down GPU
    # state on last-client-detach. So "the clock is free afterwards" has two
    # possible causes -- gpu_guard restored it, or the driver did -- and the
    # measurements alone cannot tell them apart.
    #
    # This is the discriminator: the lock has now survived a SIGKILL, a full
    # CUDA context create/destroy cycle (the load run by the capped() check
    # above), and an NVML attach/detach (`status`). If it is STILL capped here,
    # then detach does not release it on this system, and `restore` is the only
    # thing that can explain the release measured next.
    control = clock_under_load()
    if not capped(control, "CONTROL: the lock survives client detach, so a "
                           "release can only be attributed to `restore`"):
        skip("recovery released the clock",
             "the lock vanished without `restore` running -- the driver "
             "released it on detach, so no release measurement in this file "
             "can be attributed to gpu_guard")

    r = run(["restore"], s, timeout=120)
    check("recovery exits 0", r.returncode == 0, f"rc={r.returncode}")
    if control.usable and control.peak <= LOCKED_CEILING:
        released(clock_under_load(), "recovery released the clock")
    check("recovery cleared the record",
          s.exists() and read_record(s)["restored"] is True,
          f"restored={read_record(s)['restored'] if s.exists() else 'no file'}")

    r = run(["status"], s)
    check("status is clean afterwards", r.returncode == 0, f"rc={r.returncode}")

    r = run(["restore"], s, timeout=120)
    check("recovery is idempotent (exit code)", r.returncode == 0, f"rc={r.returncode}")
    # ...and in hardware. Exiting 0 twice says nothing about what the second
    # run did to the device; a re-lock would still exit 0.
    released(clock_under_load(), "recovery is idempotent (the card is still free)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", action="store_true",
                    help="also run the kill tests (requires root; writes clocks)")
    args = ap.parse_args()

    print(f"gpu_guard CLI contract check -- uid {os.geteuid()}")
    print("  " + device_provenance())
    with tempfile.TemporaryDirectory(prefix="gpu-guard-cli-") as d:
        tmp = Path(d)
        unprivileged(tmp)
        if args.root:
            if os.geteuid() != 0:
                skip("kill tests", "--root given but not running as root",
                     covers=22)
            else:
                kill_tests(tmp)
        else:
            skip("kill tests",
                 "re-run with sudo and --root to exercise real writes",
                 covers=22)

    total = len(PASS) + len(FAIL) + len(SKIP)
    expected = (nominal_check_count("unprivileged") or 0) \
        + (nominal_check_count("kill_tests") or 0)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped "
          f"= {total} of {expected} checks accounted for")
    if total != expected:
        # Not cosmetic. If the tally does not add up, some check neither ran
        # nor was declared skipped, and the transcript overstates coverage.
        print(f"  ACCOUNTING ERROR: {expected - total} check(s) unaccounted "
              f"for -- the skip counts do not match the code.")
        FAIL.append("check accounting")
    for f in FAIL:
        print("  FAILED:", f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
