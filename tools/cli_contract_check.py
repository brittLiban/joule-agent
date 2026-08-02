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
    python tools/cli_contract_check.py           # unprivileged half
    sudo -E .venv/bin/python tools/cli_contract_check.py --root

Unprivileged, it verifies every refusal path and that no refusal leaves a claim
behind. With ``--root`` it additionally runs the kill tests: clean exit, SIGINT,
SIGTERM, and SIGKILL-then-recover, checking the real clock actually moved and
came back. Nothing here is skipped silently -- a check that cannot run says so.
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
    print("\n-- kill tests (real clock writes) --")
    base = sm_clock()
    if base is None:
        skip("all kill tests", "could not read the SM clock from NVML")
        return
    print(f"          baseline SM clock: {base} MHz")

    # 1. clean exit
    s = tmp / "k1.json"
    r = run(["lock", "--mhz", "900", "--hold", "3", CONSENT], s, timeout=120)
    check("clean lock run exits 0", r.returncode == 0, f"{r.stdout!r} {r.stderr!r}")
    check("...and the record ends settled",
          s.exists() and read_record(s)["restored"] is True)
    check("...and the clock came back", (sm_clock() or 0) > 900,
          f"still at {sm_clock()} MHz")

    # 2/3. SIGINT and SIGTERM mid-lock
    for signame, signum in (("SIGINT", signal.SIGINT), ("SIGTERM", signal.SIGTERM)):
        s = tmp / f"k-{signame}.json"
        p = subprocess.Popen(
            [PY, "-m", "joule_agent.gpu_guard", "--gpu", "0",
             "--state-path", str(s), "lock", "--mhz", "900", "--hold", "30",
             CONSENT],
            cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        time.sleep(6)
        locked = sm_clock()
        p.send_signal(signum)
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill()
            check(f"{signame} terminates the run", False, "still running after 60s")
            continue
        time.sleep(2)
        check(f"{signame}: the lock was actually applied first",
              locked is not None and locked <= 950, f"clock was {locked} MHz")
        check(f"{signame}: the clock is restored", (sm_clock() or 0) > 950,
              f"still at {sm_clock()} MHz -- GPU LEFT MODIFIED")
        check(f"{signame}: the record ends settled",
              s.exists() and read_record(s)["restored"] is True,
              f"{read_record(s) if s.exists() else 'no file'}")

    # 4. SIGKILL, then recover
    s = tmp / "k4.json"
    r = run(["lock", "--mhz", "900", "--hold", "30", "--crash-after", "5", CONSENT],
            s, timeout=120)
    check("SIGKILL: the process really died by signal", r.returncode == -signal.SIGKILL,
          f"rc={r.returncode}")
    killed_at = sm_clock()
    check("SIGKILL: the GPU is left modified, as expected",
          killed_at is not None and killed_at <= 950, f"clock is {killed_at} MHz")
    check("SIGKILL: the record survives saying so",
          s.exists() and read_record(s)["restored"] is False
          and read_record(s)["clocks_locked"] is True,
          f"{read_record(s) if s.exists() else 'NO FILE -- no evidence exists'}")

    r = run(["status"], s)
    check("SIGKILL: status reports it stale", r.returncode == 1, f"rc={r.returncode}")

    r = run(["restore"], s, timeout=120)
    check("recovery exits 0", r.returncode == 0, f"{r.stdout!r} {r.stderr!r}")
    check("recovery reset the clock", (sm_clock() or 0) > 950,
          f"still at {sm_clock()} MHz -- RECOVERY FAILED")
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
