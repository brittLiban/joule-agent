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
    (PASS if cond else FAIL).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"\n          {detail}" if detail and not cond else ""))
    return cond


def skip(name, why):
    SKIP.append(name)
    print(f"  skip  {name}\n          {why}")


def read_record(state):
    return json.loads(Path(state).read_text())["record"]


#: A card locked at 900 MHz cannot exceed it under load; an unlocked 3060 Ti
#: reaches ~1900. Wide margins on both sides so thermal or power capping cannot
#: turn an unlocked card into a false "locked".
LOCKED_CEILING = 1000
UNLOCKED_FLOOR = 1200


def clock_under_load(seconds=5.0):
    """Peak SM clock while the GPU is actually being driven.

    **The idle clock is not a probe for a clock lock, and using it as one is a
    claim without evidence.** An idle card sits at ~210 MHz whether or not a
    lock is applied, and a card locked at [900, 900] sits at 900 when idle too
    -- so a bare `nvmlDeviceGetClockInfo` read cannot distinguish "restored"
    from "still locked" in either direction. The first version of this file did
    exactly that and reported four false failures, including "GPU LEFT
    MODIFIED" for a card reading 210 MHz, which is precisely the unlocked-idle
    value.

    This driver exposes no `nvmlDeviceGetGpuLockedClocks`. The only way to
    observe a lock is to demand clocks and see whether they are capped, which
    is what `tools/verify_clock_lock.py` does and what this now does too.
    """
    proc = subprocess.Popen(
        [PY, str(REPO / "tools/synthetic_gpu_load.py"),
         "--duration", str(seconds + 3), "--period", str(seconds + 3),
         "--duty", "1.0"],
        cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    peak = 0
    try:
        time.sleep(2.0)                       # let the clocks ramp
        end = time.time() + seconds
        while time.time() < end:
            v = sm_clock()
            if v:
                peak = max(peak, v)
            time.sleep(0.25)
    finally:
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    return peak


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
          f"rc={r.returncode} {r.stdout[:200]!r}")
    check("status needs no consent flag and reads the device",
          rep.get("power_readable") is True, f"{rep!r}")
    check("status leaves no state file", not s.exists())

    s = tmp / "b.json"
    r = run(["lock", "--mhz", "900", "--hold", "1"], s)
    check("lock without consent exits 2", r.returncode == 2, f"rc={r.returncode}")
    check("...and names the consent flag", CONSENT in r.stderr, r.stderr[:200])
    check("...and publishes NO claim", not s.exists(),
          "a caller that was never allowed to write left a record")

    s = tmp / "c.json"
    r = run(["lock", "--mhz", "100", "--hold", "1", CONSENT], s)
    check("lock below the floor exits 2", r.returncode == 2, f"rc={r.returncode}")
    check("...and refuses rather than clamping",
          "NOT clamped" in r.stderr, r.stderr[:200])
    check("...and publishes NO claim", not s.exists(),
          "a refused clock value left a record")

    s = tmp / "d.json"
    r = run(["lock", "--mhz", "900", "--hold", "1", CONSENT], s)
    if os.geteuid() == 0:
        skip("lock without root fails loudly", "running as root")
    else:
        check("lock without root exits 2", r.returncode == 2, f"rc={r.returncode}")
        check("...and says so rather than no-oping",
              "require root" in r.stderr, r.stderr[:200])
        check("...and any record it left is settled",
              (not s.exists()) or read_record(s)["restored"] is True,
              "a privilege refusal stranded an unrestored record")

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
          rep.get("record_is_for_this_device") is False, f"{rep!r}")
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
          any("UNREADABLE" in f for f in rep.get("findings", [])), f"{rep!r}")
    check("...and status did not overwrite it",
          s.read_text() == "{ this is not json", "status destroyed evidence")

    r = run(["--gpu", "99", "status"], tmp / "g.json")
    check("a nonexistent device fails cleanly, not with a traceback",
          r.returncode != 0 and "Traceback" not in r.stderr,
          f"rc={r.returncode} {r.stderr[-300:]!r}")


# ---------------------------------------------------------------------------
# Root: the kill tests
# ---------------------------------------------------------------------------


def kill_tests(tmp):
    print("\n-- kill tests (real clock writes, verified under load) --")
    idle = sm_clock()
    if idle is None:
        skip("all kill tests", "could not read the SM clock from NVML")
        return
    base = clock_under_load()
    print(f"          idle {idle} MHz; peak under load {base} MHz")
    if not check("the card is unlocked before we start", base >= UNLOCKED_FLOOR,
                 f"peak {base} MHz -- something already holds a lock; run "
                 f"`sudo python -m joule_agent.gpu_guard restore` first"):
        skip("remaining kill tests", "cannot measure against a locked baseline")
        return

    def lock_proc(state, hold, extra=()):
        return subprocess.Popen(
            [PY, "-m", "joule_agent.gpu_guard", "--gpu", "0",
             "--state-path", str(state), "lock", "--mhz", "900",
             "--hold", str(hold), CONSENT, *extra],
            cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # 1. clean exit
    s = tmp / "k1.json"
    p = lock_proc(s, 25)
    held = clock_under_load()
    rc = p.wait(timeout=120)
    check("clean run: the lock really caps the clock", held <= LOCKED_CEILING,
          f"peak {held} MHz under load with a 900 MHz lock applied")
    check("clean run exits 0", rc == 0, f"rc={rc} {p.stderr.read()[-300:]!r}")
    check("clean run: the record ends settled",
          s.exists() and read_record(s)["restored"] is True)
    after = clock_under_load()
    check("clean run: the clock is released", after >= UNLOCKED_FLOOR,
          f"peak {after} MHz -- GPU LEFT MODIFIED")

    # 2/3. SIGINT and SIGTERM mid-lock
    for signame, signum in (("SIGINT", signal.SIGINT), ("SIGTERM", signal.SIGTERM)):
        s = tmp / f"k-{signame}.json"
        p = lock_proc(s, 120)
        held = clock_under_load()
        check(f"{signame}: the lock was applied first", held <= LOCKED_CEILING,
              f"peak {held} MHz -- the lock never took effect, so the rest "
              f"proves nothing")
        p.send_signal(signum)
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill()
            check(f"{signame} terminates the run", False, "still running after 60s")
            continue
        after = clock_under_load()
        check(f"{signame}: the clock is restored", after >= UNLOCKED_FLOOR,
              f"peak {after} MHz -- GPU LEFT MODIFIED")
        check(f"{signame}: the record ends settled",
              s.exists() and read_record(s)["restored"] is True,
              f"{read_record(s) if s.exists() else 'no file'}")

    # 4. SIGKILL, then recover
    s = tmp / "k4.json"
    r = run(["lock", "--mhz", "900", "--hold", "120", "--crash-after", "6", CONSENT],
            s, timeout=180)
    check("SIGKILL: the process really died by signal", r.returncode == -signal.SIGKILL,
          f"rc={r.returncode}")
    killed = clock_under_load()
    check("SIGKILL: the GPU is left modified, as expected",
          killed <= LOCKED_CEILING,
          f"peak {killed} MHz -- nothing was left to recover, so the recovery "
          f"checks below prove nothing")
    check("SIGKILL: the record survives saying so",
          s.exists() and read_record(s)["restored"] is False
          and read_record(s)["clocks_locked"] is True,
          f"{read_record(s) if s.exists() else 'NO FILE -- no evidence exists'}")

    r = run(["status"], s)
    check("SIGKILL: status reports it stale", r.returncode == 1, f"rc={r.returncode}")

    r = run(["restore"], s, timeout=120)
    check("recovery exits 0", r.returncode == 0, f"{r.stdout!r} {r.stderr!r}")
    recovered = clock_under_load()
    check("recovery released the clock", recovered >= UNLOCKED_FLOOR,
          f"peak {recovered} MHz -- RECOVERY FAILED, GPU STILL LOCKED")
    check("recovery cleared the record",
          read_record(s)["restored"] is True, f"{read_record(s)}")

    r = run(["status"], s)
    check("status is clean afterwards", r.returncode == 0, f"rc={r.returncode}")

    r = run(["restore"], s, timeout=120)
    check("recovery is idempotent", r.returncode == 0, f"rc={r.returncode}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", action="store_true",
                    help="also run the kill tests (requires root; writes clocks)")
    args = ap.parse_args()

    print(f"gpu_guard CLI contract check -- uid {os.geteuid()}")
    with tempfile.TemporaryDirectory(prefix="gpu-guard-cli-") as d:
        tmp = Path(d)
        unprivileged(tmp)
        if args.root:
            if os.geteuid() != 0:
                skip("kill tests", "--root given but not running as root")
            else:
                kill_tests(tmp)
        else:
            skip("kill tests", "re-run with sudo and --root to exercise real writes")

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped")
    for f in FAIL:
        print("  FAILED:", f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
