"""Tests for the GPU write guard.

The guard's whole value is a promise about crash paths, so these tests exercise
crash paths rather than the happy one: exceptions inside the context manager,
SIGINT and SIGTERM delivered to a real child process, and SIGKILL with recovery
from the state file afterwards.

The module must be provable without being trusted. Where a test asserts the
guard restored something, it asserts against the FakeGpuController's recorded
state, and `FakeGpuController(fail_restore=True)` exists so the tests can be
shown to fail when restore is broken.

No GPU, no root, no NVML.
"""

import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from joule_agent.gpu_guard import (
    CONSENT_FLAG,
    MIN_SM_CLOCK_MHZ,
    ClockFloorError,
    ConsentError,
    FakeGpuController,
    GpuGuard,
    GpuGuardError,
    GuardBusyError,
    PrivilegeError,
    ThreadAffinityError,
    check_stale_state,
    restore_from_state_file,
)


def record(state_path, pid=None) -> dict:
    """Read the guard record. Schema 3 stores exactly one."""
    doc = json.loads(Path(state_path).read_text())
    rec = doc["record"]
    if pid is not None:
        assert rec["pid"] == pid, f"record is for pid {rec['pid']}, wanted {pid}"
    return rec


def guard(tmp_path, **kw) -> GpuGuard:
    kw.setdefault("consent", True)
    kw.setdefault("require_root", False)  # privilege is tested separately
    controller = kw.pop("controller", None) or FakeGpuController()
    return GpuGuard(controller, state_path=tmp_path / "state.json", **kw)


# ---------------------------------------------------------------------------
# refusals: consent, privilege, clock floor
# ---------------------------------------------------------------------------


def test_write_without_consent_is_refused(tmp_path):
    c = FakeGpuController()
    g = GpuGuard(c, consent=False, require_root=False, state_path=tmp_path / "s.json")
    with pytest.raises(ConsentError) as e:
        g.lock_clocks(1400)
    assert CONSENT_FLAG in str(e.value)
    assert c.set_clock_calls == 0, "hardware was touched despite refusal"


def test_reads_never_require_consent(tmp_path):
    c = FakeGpuController()
    g = GpuGuard(c, consent=False, require_root=False, state_path=tmp_path / "s.json")
    g.arm()
    assert g.snapshot.name == "FakeGPU"
    assert c.set_clock_calls == 0


def test_missing_privilege_fails_loudly_not_silently(tmp_path):
    c = FakeGpuController()
    g = GpuGuard(c, consent=True, require_root=True, state_path=tmp_path / "s.json")
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("running as root")
    with pytest.raises(PrivilegeError):
        g.lock_clocks(1400)
    assert c.set_clock_calls == 0, "a no-op would be worse than an error"


@pytest.mark.parametrize("mhz", [0, 1, 100, MIN_SM_CLOCK_MHZ - 1])
def test_clock_below_floor_is_refused_not_clamped(tmp_path, mhz):
    c = FakeGpuController()
    g = guard(tmp_path, controller=c)
    with pytest.raises(ClockFloorError):
        g.lock_clocks(mhz)
    assert c.locked is None, "a refused lock must not reach the device"
    assert c.set_clock_calls == 0


def test_clock_floor_is_checked_before_privilege(tmp_path):
    """An unprivileged caller with a bad clock hears about the clock.

    Ordering matters for the operator: reporting "need root" first would send
    them to sudo and only then reveal the real problem, implying sudo would
    have made 100 MHz acceptable.
    """
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("running as root")
    c = FakeGpuController()
    g = GpuGuard(c, consent=True, require_root=True, state_path=tmp_path / "s.json")
    with pytest.raises(ClockFloorError):
        g.lock_clocks(100)
    assert c.set_clock_calls == 0


def test_clock_at_floor_is_allowed(tmp_path):
    c = FakeGpuController()
    with guard(tmp_path, controller=c) as g:
        g.lock_clocks(MIN_SM_CLOCK_MHZ)
        assert c.locked == (MIN_SM_CLOCK_MHZ, MIN_SM_CLOCK_MHZ)


def test_inverted_clock_range_is_refused(tmp_path):
    g = guard(tmp_path)
    with pytest.raises(ClockFloorError):
        g.lock_clocks(1800, 1400)


def test_power_limit_outside_device_constraints_is_refused(tmp_path):
    c = FakeGpuController()
    g = guard(tmp_path, controller=c)
    with pytest.raises(GpuGuardError):
        g.set_power_limit_mw(50_000)   # below device min
    with pytest.raises(GpuGuardError):
        g.set_power_limit_mw(500_000)  # above device max
    assert c.set_power_calls == 0


# ---------------------------------------------------------------------------
# restore on ordinary paths
# ---------------------------------------------------------------------------


def test_context_manager_restores_on_clean_exit(tmp_path):
    c = FakeGpuController()
    with guard(tmp_path, controller=c) as g:
        g.lock_clocks(1400)
        assert c.locked == (1400, 1400)
    assert c.locked is None
    assert c.reset_calls >= 1


def test_context_manager_restores_on_exception(tmp_path):
    c = FakeGpuController()
    with pytest.raises(ValueError):
        with guard(tmp_path, controller=c) as g:
            g.lock_clocks(1400)
            raise ValueError("boom")
    assert c.locked is None, "an exception must not leave the GPU locked"


def test_exception_is_not_swallowed(tmp_path):
    with pytest.raises(RuntimeError, match="propagate"):
        with guard(tmp_path) as g:
            g.lock_clocks(1400)
            raise RuntimeError("propagate")


def test_restore_is_idempotent(tmp_path):
    c = FakeGpuController()
    g = guard(tmp_path, controller=c)
    with g:
        g.lock_clocks(1400)
    first = c.reset_calls
    g.restore()
    g.restore()
    assert c.reset_calls == first, "repeat restore must be a no-op"


def test_power_limit_restored_to_device_default(tmp_path):
    c = FakeGpuController(power_limit_mw=200_000, power_default_limit_mw=200_000)
    with guard(tmp_path, controller=c) as g:
        g.set_power_limit_mw(120_000)
        assert c.get_power_limit_mw() == 120_000
    assert c.get_power_limit_mw() == 200_000


def test_nested_guard_is_reentrant(tmp_path):
    c = FakeGpuController()
    g = guard(tmp_path, controller=c)
    with g:
        g.lock_clocks(1400)
        with g:
            g.lock_clocks(1500)
        # Inner exit must NOT restore while the outer block is still open.
        assert c.locked == (1500, 1500), "inner exit restored too early"
    assert c.locked is None


def test_restore_without_any_write_is_harmless(tmp_path):
    c = FakeGpuController()
    with guard(tmp_path, controller=c):
        pass
    assert c.reset_calls == 0
    assert c.locked is None


# ---------------------------------------------------------------------------
# state file: written before hardware, survives a hard kill
# ---------------------------------------------------------------------------


def test_state_file_written_before_hardware_touch(tmp_path):
    """The crash window between locking and recording must not exist."""
    sp = tmp_path / "state.json"
    order = []
    c = FakeGpuController()
    real_set = c.set_locked_clocks

    def spy(mn, mx):
        order.append(("hardware", sp.exists()))
        real_set(mn, mx)

    c.set_locked_clocks = spy
    with guard(tmp_path, controller=c) as g:
        g.lock_clocks(1400)
    assert order == [("hardware", True)], "state file did not exist before the write"


def test_state_file_marks_unrestored_while_held(tmp_path):
    sp = tmp_path / "state.json"
    c = FakeGpuController()
    g = guard(tmp_path, controller=c)
    with g:
        g.lock_clocks(1400)
        mid = record(sp)
        assert mid["restored"] is False
        assert mid["clocks_locked"] is True
        assert mid["locked_min_mhz"] == 1400
    after = record(sp)
    assert after["restored"] is True
    assert after["clocks_locked"] is False


def test_state_file_documents_that_clock_lock_is_asserted(tmp_path):
    """The device cannot be queried; the file must not imply otherwise."""
    sp = tmp_path / "state.json"
    with guard(tmp_path) as g:
        g.lock_clocks(1400)
    note = record(sp)["_note"]
    assert "ASSERTED" in note or "asserted" in note.lower()


# ---------------------------------------------------------------------------
# stale-state detection: two tiers of confidence
# ---------------------------------------------------------------------------


def test_clean_state_is_not_stale(tmp_path):
    c = FakeGpuController()
    rep = check_stale_state(c, tmp_path / "none.json")
    assert rep["stale"] is False
    assert rep["findings"] == []


def test_power_modification_is_detected_as_verified(tmp_path):
    c = FakeGpuController(power_limit_mw=120_000, power_default_limit_mw=200_000)
    rep = check_stale_state(c, tmp_path / "none.json")
    assert rep["stale"] is True
    assert rep["verified_power_modified"] is True
    assert any("VERIFIED" in f for f in rep["findings"])


def test_orphaned_clock_lock_is_detected_as_asserted_only(tmp_path):
    sp = tmp_path / "state.json"
    sp.write_text(json.dumps({
        "pid": 999999, "restored": False, "clocks_locked": True,
        "locked_min_mhz": 1400, "locked_max_mhz": 1400,
        "snapshot": {"power_default_limit_mw": 200_000},
    }))
    rep = check_stale_state(FakeGpuController(), sp)
    assert rep["stale"] is True
    assert rep["asserted_clock_lock"] is True
    assert rep["verified_power_modified"] is False
    finding = " ".join(rep["findings"])
    assert "ASSERTED" in finding and "NOT verifiable" in finding
    assert rep["recommendation"]


def test_live_writer_pid_is_distinguished_from_a_dead_one(tmp_path):
    sp = tmp_path / "state.json"
    sp.write_text(json.dumps({
        "pid": os.getpid(), "restored": False, "clocks_locked": True,
        "snapshot": {},
    }))
    rep = check_stale_state(FakeGpuController(), sp)
    assert rep["writer_pid_alive"] is True
    assert "still running" in " ".join(rep["findings"])


def test_restore_from_state_file_is_unconditional(tmp_path):
    """Clock reset always fires; power follows only a record of OUR write."""
    sp = tmp_path / "state.json"
    sp.write_text(json.dumps({
        "schema": 2,
        "guards": {"999999": {
            "pid": 999999, "restored": False, "clocks_locked": True,
            "power_limit_written_mw": 120_000,
            "snapshot": {"power_limit_mw": 200_000,
                         "power_default_limit_mw": 200_000},
        }},
    }))
    c = FakeGpuController(power_limit_mw=120_000, power_default_limit_mw=200_000)
    res = restore_from_state_file(c, sp)
    assert res["reset_clocks"] is True
    assert res["power_restored_to_mw"] == 200_000
    assert c.get_power_limit_mw() == 200_000
    assert record(sp, 999999)["restored"] is True


def test_schema1_state_file_still_recoverable(tmp_path):
    """A flat record from an older build must not become unrecoverable."""
    sp = tmp_path / "state.json"
    sp.write_text(json.dumps({
        "pid": 999999, "restored": False, "clocks_locked": True,
        "power_limit_written_mw": 120_000,
        "snapshot": {"power_limit_mw": 200_000},
    }))
    rep = check_stale_state(FakeGpuController(), sp)
    assert rep["stale"] is True
    assert rep["asserted_clock_lock"] is True


def test_power_untouched_when_no_record_claims_it(tmp_path):
    """An operator's deliberate cap must not be wiped by a recovery run."""
    c = FakeGpuController(power_limit_mw=120_000, power_default_limit_mw=200_000)
    res = restore_from_state_file(c, tmp_path / "missing.json")
    assert res["reset_clocks"] is True
    assert res["power_restored_to_mw"] is None
    assert c.get_power_limit_mw() == 120_000, "recovery undid an operator cap"
    assert "power_untouched_reason" in res


def test_force_power_default_is_the_explicit_override(tmp_path):
    c = FakeGpuController(power_limit_mw=120_000, power_default_limit_mw=200_000)
    res = restore_from_state_file(c, tmp_path / "missing.json", force_power_default=True)
    assert res["power_restored_to_mw"] == 200_000
    assert c.get_power_limit_mw() == 200_000


def test_recovery_resets_clocks_even_with_no_state_file(tmp_path):
    c = FakeGpuController(power_limit_mw=200_000, power_default_limit_mw=200_000)
    res = restore_from_state_file(c, tmp_path / "missing.json")
    assert res["reset_clocks"] is True


# ---------------------------------------------------------------------------
# real subprocess crash paths
# ---------------------------------------------------------------------------


CHILD = textwrap.dedent(
    """
    import json, os, signal, sys, time
    sys.path.insert(0, {repo!r})
    from joule_agent.gpu_guard import GpuGuard, FakeGpuController

    state_path, marker, mode = sys.argv[1], sys.argv[2], sys.argv[3]

    class Recording(FakeGpuController):
        def reset_locked_clocks(self):
            super().reset_locked_clocks()
            with open(marker, "a") as fh:
                fh.write("restored\\n")
                fh.flush()
                os.fsync(fh.fileno())

    c = Recording()
    g = GpuGuard(c, consent=True, require_root=False, state_path=state_path)
    with g:
        g.lock_clocks(1400)
        print("LOCKED", flush=True)
        if mode == "sigkill":
            os.kill(os.getpid(), signal.SIGKILL)
        time.sleep(30)
    """
)


def _spawn(tmp_path, mode):
    repo = str(Path(__file__).resolve().parent.parent)
    script = tmp_path / "child.py"
    script.write_text(CHILD.format(repo=repo))
    state = tmp_path / "state.json"
    marker = tmp_path / "marker.txt"
    proc = subprocess.Popen(
        [sys.executable, str(script), str(state), str(marker), mode],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    # Wait until the child confirms it holds the lock.
    deadline = time.time() + 30
    while time.time() < deadline:
        line = proc.stdout.readline()
        if "LOCKED" in line:
            return proc, state, marker
        if proc.poll() is not None:
            break
    proc.kill()
    raise AssertionError(f"child never locked: {proc.stderr.read()[:500]}")


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_child_process_restores_on_signal(tmp_path, sig):
    """A real signal to a real process -- the Ctrl+C case.

    Verified by mutation: with signal-handler installation removed, the
    SIGTERM case FAILS and the SIGINT case still PASSES. That asymmetry is
    real and worth knowing -- SIGINT is protected twice over, because
    Python's default handler raises KeyboardInterrupt which unwinds through
    ``__exit__`` and restores on the ordinary exception path. SIGTERM has no
    such fallback: the default action terminates the process without
    unwinding, so only the installed handler saves it.

    So: SIGTERM is the case that actually isolates the handler. SIGINT is
    kept because Ctrl+C is the realistic operator action, but a green SIGINT
    result alone would not prove the handler is installed.
    """
    proc, state, marker = _spawn(tmp_path, "signal")
    proc.send_signal(sig)
    proc.wait(timeout=30)
    assert marker.exists(), f"{sig!r} did not trigger restore"
    assert "restored" in marker.read_text()
    assert record(state, proc.pid)["restored"] is True


def test_sigkill_leaves_recoverable_state_file(tmp_path):
    """SIGKILL cannot be caught. The state file is the whole contingency."""
    proc, state, marker = _spawn(tmp_path, "sigkill")
    proc.wait(timeout=30)
    assert proc.returncode == -signal.SIGKILL

    # No handler could have run.
    assert not marker.exists(), "restore ran despite SIGKILL?"
    # But the file records the un-restored lock, written before the hardware write.
    recorded = record(state, proc.pid)
    assert recorded["restored"] is False
    assert recorded["clocks_locked"] is True

    # And the recovery path acts on it.
    rep = check_stale_state(FakeGpuController(), state)
    assert rep["stale"] is True
    assert rep["writer_pid_alive"] is False
    c = FakeGpuController()
    res = restore_from_state_file(c, state)
    assert res["reset_clocks"] is True
    assert record(state, proc.pid)["restored"] is True


# ---------------------------------------------------------------------------
# the tests must discriminate
# ---------------------------------------------------------------------------


def test_broken_restore_is_caught_by_these_tests(tmp_path):
    """Proves the suite fails when the guard is broken, rather than passing.

    A test that cannot fail buys false confidence -- worse than no test at all.
    """
    c = FakeGpuController(fail_restore=True)
    with guard(tmp_path, controller=c) as g:
        g.lock_clocks(1400)
    # With restore deliberately broken, the assertion used by the real tests
    # above must be the thing that catches it.
    assert c.locked is not None, "fail_restore did not actually break restore"
    with pytest.raises(AssertionError):
        assert c.locked is None  # the exact assertion in the clean-exit test


def test_broken_power_restore_is_caught(tmp_path):
    class NoPowerRestore(FakeGpuController):
        def set_power_limit_mw(self, mw):
            if mw == 200_000:
                return  # silently refuse to restore
            super().set_power_limit_mw(mw)

    c = NoPowerRestore(power_limit_mw=200_000, power_default_limit_mw=200_000)
    with guard(tmp_path, controller=c) as g:
        g.set_power_limit_mw(120_000)
    assert c.get_power_limit_mw() == 120_000
    with pytest.raises(AssertionError):
        assert c.get_power_limit_mw() == 200_000


# ---------------------------------------------------------------------------
# regressions from the hardware-safety-reviewer pass
# ---------------------------------------------------------------------------


class RaisingRestore(FakeGpuController):
    """Hardware refuses the reset -- the case that used to be recorded as success."""

    def reset_locked_clocks(self):
        self._require_claim()
        self.reset_calls += 1
        raise RuntimeError("NVML reset failed")


def test_failed_restore_does_not_record_success(tmp_path):
    """F1, the severe one.

    A failed reset previously wrote restored:true, leaving the device locked
    with the only evidence erased and every retry path disarmed. Because there
    is no readback, that state file is all there is.
    """
    sp = tmp_path / "state.json"
    c = RaisingRestore()
    g = GpuGuard(c, consent=True, require_root=False, state_path=sp)
    with pytest.raises(GpuGuardError, match="MAY STILL BE MODIFIED"):
        with g:
            g.lock_clocks(1400)

    assert c.locked == (1400, 1400), "precondition: device still locked"
    assert g._restored is False, "guard must stay armed after a failed restore"
    rec = record(sp)
    assert rec["restored"] is False, "failed restore recorded as success"
    assert rec["restore_failed"] is True
    assert rec["restore_errors"]


def test_failed_restore_leaves_retry_armed(tmp_path):
    c = RaisingRestore()
    g = GpuGuard(c, consent=True, require_root=False, state_path=tmp_path / "s.json")
    with pytest.raises(GpuGuardError):
        with g:
            g.lock_clocks(1400)
    before = c.reset_calls
    with pytest.raises(GpuGuardError):
        g.restore()
    assert c.reset_calls > before, "retry was suppressed after a failed restore"


def test_failed_restore_is_visible_to_stale_check(tmp_path):
    sp = tmp_path / "state.json"
    c = RaisingRestore()
    g = GpuGuard(c, consent=True, require_root=False, state_path=sp)
    with pytest.raises(GpuGuardError):
        with g:
            g.lock_clocks(1400)
    rep = check_stale_state(FakeGpuController(), sp)
    assert rep["stale"] is True
    assert rep["restore_failed"] is True
    assert any("RESTORE FAILED" in f for f in rep["findings"])


def test_unclaimed_controller_refuses_direct_writes(tmp_path):
    """F5: constructing a controller must not grant a bypass around the guard."""
    c = FakeGpuController()
    with pytest.raises(GpuGuardError, match="not held by a GpuGuard"):
        c.set_locked_clocks(100, 100)
    with pytest.raises(GpuGuardError):
        c.set_power_limit_mw(50_000)
    assert c.set_clock_calls == 0


def test_guard_claims_its_controller(tmp_path):
    c = FakeGpuController()
    with guard(tmp_path, controller=c) as g:
        g.lock_clocks(1400)
        assert c.locked == (1400, 1400)


def test_power_limit_of_zero_is_still_restored(tmp_path):
    """F7: 0 mW is falsy; a truthiness gate skipped the restore entirely."""
    c = FakeGpuController(power_limit_mw=200_000, power_default_limit_mw=200_000)
    g = guard(tmp_path, controller=c)
    g.arm()
    g.snapshot.power_min_mw = None   # constraints unreadable, as on GeForce
    g.snapshot.power_max_mw = None
    with g:
        g.set_power_limit_mw(0)
        assert c.get_power_limit_mw() == 0
    assert c.get_power_limit_mw() == 200_000, "0 mW write was never restored"


def test_restore_targets_observed_limit_not_factory_default(tmp_path):
    """F9: an operator's pre-existing cap must survive our restore."""
    c = FakeGpuController(power_limit_mw=150_000, power_default_limit_mw=200_000)
    with guard(tmp_path, controller=c) as g:
        g.set_power_limit_mw(120_000)
    assert c.get_power_limit_mw() == 150_000, "restore overshot to factory default"


def test_write_through_closed_controller_fails_loudly(tmp_path):
    """F8, tested against the REAL controller's _write, not a hand-written stub.

    The previous version of this test asserted a message raised by its own
    fixture, so deleting the production check left it green.
    """
    from joule_agent.gpu_guard import NvmlGpuController

    c = object.__new__(NvmlGpuController)   # no NVML init
    c._closed = True
    c._claimed_by = "test"
    called = []
    with pytest.raises(GpuGuardError, match="closed"):
        c._write(lambda: called.append(1))
    assert called == [], "a closed controller still issued the write"


def test_write_through_unclaimed_real_controller_is_refused():
    from joule_agent.gpu_guard import NvmlGpuController

    c = object.__new__(NvmlGpuController)
    c._closed = False
    c._claimed_by = None
    with pytest.raises(GpuGuardError, match="not held by a GpuGuard"):
        c._write(lambda: None)


def test_sig_ign_is_not_converted_into_process_death(tmp_path, monkeypatch):
    """F6, asserted on os.kill rather than on process survival.

    Survival alone does not discriminate: `prev` (SIG_IGN) is reinstalled
    before dispatch, so a fallthrough os.kill would be swallowed by the OS and
    the process would live either way.
    """
    import joule_agent.gpu_guard as gg

    killed = []
    monkeypatch.setattr(gg.os, "kill", lambda pid, sig: killed.append(sig))

    c = FakeGpuController()
    g = GpuGuard(c, consent=True, require_root=False, state_path=tmp_path / "s.json")
    prev = signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        g.arm()
        g.lock_clocks(1400)
        g._signal_restore(signal.SIGTERM, None)
        assert c.locked is None, "restore did not run"
        assert killed == [], f"guard killed a process that ignores SIGTERM: {killed}"
    finally:
        signal.signal(signal.SIGTERM, prev)


def test_atexit_restore_reports_failure_on_stderr(tmp_path, capfd):
    """F3 had no coverage at all; a bare `except: pass` would pass silently."""
    c = RaisingRestore()
    g = GpuGuard(c, consent=True, require_root=False, state_path=tmp_path / "s.json")
    g.arm()
    g.lock_clocks(1400)
    g._atexit_restore()          # must not raise, must not be silent
    err = capfd.readouterr().err
    assert "ERROR [gpu_guard]" in err
    assert "restore failed during exit" in err


# ---------------------------------------------------------------------------
# second-pass regressions
# ---------------------------------------------------------------------------


def test_unknown_power_restore_target_is_an_error_not_a_skip(tmp_path):
    """If both snapshot reads failed, we cannot restore -- and must say so."""
    c = FakeGpuController(power_limit_mw=200_000, power_default_limit_mw=200_000)
    g = guard(tmp_path, controller=c)
    g.arm()
    g.snapshot.power_limit_mw = None        # both NVML reads failed at arm()
    g.snapshot.power_default_limit_mw = None
    g.snapshot.power_min_mw = None
    g.snapshot.power_max_mw = None
    g.set_power_limit_mw(120_000)
    with pytest.raises(GpuGuardError, match="no restore target is known"):
        g.restore()
    assert g._restored is False


def test_claim_is_exclusive(tmp_path):
    """N5: two guards must not silently share one controller."""
    c = FakeGpuController()
    c.claim("owner-a")
    with pytest.raises(GpuGuardError, match="already claimed"):
        c.claim("owner-b")
    c.claim("owner-a")  # same owner is fine


def test_status_flags_unreadable_power_rather_than_implying_clean(tmp_path):
    """N11: a clean report must not be produced by a check that never ran."""
    class Unreadable(FakeGpuController):
        def get_power_limit_mw(self):
            return None

        def get_power_default_limit_mw(self):
            return None

    rep = check_stale_state(Unreadable(), tmp_path / "none.json")
    assert rep["power_readable"] is False
    assert any("UNKNOWN" in f for f in rep["findings"])


# ---------------------------------------------------------------------------
# pass-3 regressions
# ---------------------------------------------------------------------------


def test_recovery_writes_state_file_atomically(tmp_path):
    """N7: recovery used plain write_text -- a crash mid-write truncates the
    only evidence, which _read_doc then reports as 'no guard ever ran'."""
    import joule_agent.gpu_guard as gg

    sp = tmp_path / "state.json"
    sp.write_text(json.dumps({
        "schema": 2,
        "guards": {"999999": {"pid": 999999, "restored": False,
                              "clocks_locked": True, "snapshot": {}}},
    }))
    seen = []
    real = gg.os.replace
    gg.os.replace = lambda a, b: (seen.append((str(a), str(b))), real(a, b))[1]
    try:
        restore_from_state_file(FakeGpuController(), sp)
    finally:
        gg.os.replace = real
    assert seen, "recovery published the state file without an atomic rename"
    assert seen[-1][1] == str(sp)


def test_main_does_not_close_nvml_while_state_is_unrestored(tmp_path, monkeypatch):
    """N6, behavioural. The previous version asserted literal source
    indentation and stayed green when the condition was inverted."""
    import joule_agent.gpu_guard as gg

    class Unrestorable(FakeGpuController):
        def reset_locked_clocks(self):
            self._require_claim()
            self.reset_calls += 1
            raise RuntimeError("reset refused")

    c = Unrestorable()
    monkeypatch.setattr(gg, "_controller", lambda args: c)
    monkeypatch.setattr(gg.os, "geteuid", lambda: 0)   # bypass privilege gate

    rc = gg.main([
        "--state-path", str(tmp_path / "s.json"),
        "lock", "--mhz", "1400", "--hold", "0",
        CONSENT_FLAG,
    ])
    assert c.closed is False, (
        "NVML was closed while the guard held unrestored state; the atexit "
        "last-chance restore can no longer run"
    )
    assert rc != 0


def test_nvml_controller_claim_is_exclusive():
    """Uses a RUNTIME-CONSTRUCTED string so constant folding cannot make
    `is` succeed -- the interning that made the previous version blind to a
    reverted `_same_owner`."""
    from joule_agent.gpu_guard import NvmlGpuController

    owner = "".join(["own", "er-a"])          # equal to "owner-a", not identical
    assert owner == "owner-a" and owner is not "owner-a"  # noqa: F632

    c = object.__new__(NvmlGpuController)
    c._claimed_by = None
    c._closed = False
    c.claim("owner-a")
    c.claim(owner)                             # equality must be enough
    with pytest.raises(GpuGuardError, match="already claimed"):
        c.claim("owner-b")



# ---------------------------------------------------------------------------
# the two structural refusals -- load-bearing safety behaviour
# ---------------------------------------------------------------------------


def test_arming_off_the_main_thread_is_refused_cleanly(tmp_path):
    """Restore depends on signal handlers, which only the main thread can
    install. Arming elsewhere would silently degrade the guarantee to the state
    file alone, so it is refused rather than allowed-with-a-warning."""
    c = FakeGpuController()
    g = GpuGuard(c, consent=True, require_root=False, state_path=tmp_path / "s.json")

    caught = []

    def worker():
        try:
            g.arm()
        except BaseException as exc:      # must be a clean refusal, not a crash
            caught.append(exc)

    th = threading.Thread(target=worker)
    th.start()
    th.join(10)

    assert caught, "arming off the main thread was allowed"
    exc = caught[0]
    assert isinstance(exc, ThreadAffinityError), f"got {type(exc).__name__}"
    assert "main thread" in str(exc)
    assert c.set_clock_calls == 0
    assert not (tmp_path / "s.json").exists(), "a refused arm wrote state"


def test_writing_from_a_non_owner_thread_is_refused(tmp_path):
    """Armed on the main thread, then written from a worker."""
    c = FakeGpuController()
    g = GpuGuard(c, consent=True, require_root=False, state_path=tmp_path / "s.json")
    g.arm()

    caught = []

    def worker():
        try:
            g.lock_clocks(1400)
        except BaseException as exc:
            caught.append(exc)

    th = threading.Thread(target=worker)
    th.start()
    th.join(10)

    assert caught, "a worker thread was allowed to write"
    assert isinstance(caught[0], ThreadAffinityError)
    assert c.set_clock_calls == 0, "hardware was touched from a non-owner thread"
    assert c.locked is None


def test_arming_while_a_live_pid_holds_a_record_is_refused(tmp_path):
    """One guard per machine. A live holder means another guard is running."""
    sp = tmp_path / "state.json"
    sp.write_text(json.dumps({
        "schema": 3,
        "record": {"pid": os.getppid(), "restored": False, "clocks_locked": True,
                   "locked_min_mhz": 900, "locked_max_mhz": 900, "snapshot": {}},
    }))
    c = FakeGpuController()
    g = GpuGuard(c, consent=True, require_root=False, state_path=sp)
    with pytest.raises(GuardBusyError, match="is running"):
        g.arm()
    assert c.set_clock_calls == 0
    # the other guard's record must survive untouched
    assert record(sp)["restored"] is False


def test_arming_after_a_dead_pid_left_a_record_is_refused(tmp_path):
    """A dead holder means recovery is owed. Arming would overwrite the only
    evidence that the GPU may still be modified."""
    sp = tmp_path / "state.json"
    sp.write_text(json.dumps({
        "schema": 3,
        "record": {"pid": 999999, "restored": False, "clocks_locked": True,
                   "locked_min_mhz": 900, "locked_max_mhz": 900, "snapshot": {}},
    }))
    c = FakeGpuController()
    g = GpuGuard(c, consent=True, require_root=False, state_path=sp)
    with pytest.raises(GuardBusyError, match="died without restoring"):
        g.arm()
    assert "gpu_guard restore" in str(
        pytest.raises(GuardBusyError, g.arm).value
    ), "the refusal must name the recovery command"
    assert record(sp)["restored"] is False, "the evidence was overwritten"


def test_arming_proceeds_when_the_prior_record_is_restored(tmp_path):
    """The refusal must not wedge normal operation."""
    sp = tmp_path / "state.json"
    sp.write_text(json.dumps({
        "schema": 3,
        "record": {"pid": 999999, "restored": True, "clocks_locked": False,
                   "snapshot": {}},
    }))
    c = FakeGpuController()
    with GpuGuard(c, consent=True, require_root=False, state_path=sp) as g:
        g.lock_clocks(1400)
        assert c.locked == (1400, 1400)
    assert c.locked is None


def test_signals_are_blocked_for_the_duration_of_a_hardware_write(tmp_path):
    """Replaces the in-flight bookkeeping that four review passes kept breaking.

    A handler running between "record intent" and "issue the write" could reset
    the device and mark clean, after which the interrupted frame applies the
    setting. Blocking the signals makes that interleaving unrepresentable
    rather than merely detectable.
    """
    observed = {}

    class CheckMask(FakeGpuController):
        def set_locked_clocks(self, mn, mx):
            self._require_claim()
            observed["blocked"] = signal.pthread_sigmask(signal.SIG_BLOCK, [])
            self.set_clock_calls += 1
            self.locked = (mn, mx)

    c = CheckMask()
    before = signal.pthread_sigmask(signal.SIG_BLOCK, [])
    with GpuGuard(c, consent=True, require_root=False,
                  state_path=tmp_path / "s.json") as g:
        g.lock_clocks(1400)
    after = signal.pthread_sigmask(signal.SIG_BLOCK, [])

    assert signal.SIGINT in observed["blocked"], "SIGINT was deliverable mid-write"
    assert signal.SIGTERM in observed["blocked"], "SIGTERM was deliverable mid-write"
    assert after == before, "the signal mask was not restored after the write"


def test_a_signal_raised_during_a_write_is_delivered_after_it(tmp_path):
    """Blocked, not lost: the pending signal must arrive once the write ends."""
    delivered = []
    original = signal.getsignal(signal.SIGTERM)

    class RaiseMidWrite(FakeGpuController):
        def set_locked_clocks(self, mn, mx):
            self._require_claim()
            os.kill(os.getpid(), signal.SIGTERM)   # blocked -> stays pending
            assert not delivered, "handler ran mid-write despite the block"
            self.set_clock_calls += 1
            self.locked = (mn, mx)

    try:
        signal.signal(signal.SIGTERM, lambda *a: delivered.append(True))
        c = RaiseMidWrite()
        g = GpuGuard(c, consent=True, require_root=False,
                     state_path=tmp_path / "s.json")
        g.arm()
        signal.signal(signal.SIGTERM, lambda *a: delivered.append(True))
        g.lock_clocks(1400)
        assert c.locked == (1400, 1400)
        assert delivered, "the pending signal was swallowed, not deferred"
    finally:
        signal.signal(signal.SIGTERM, original)


def test_a_second_guard_in_one_process_is_refused(tmp_path):
    """The own-pid exemption was unreachable for a single guard (_armed is
    never reset, so arm() short-circuits) and therefore only ever admitted a
    SECOND guard object -- which would snapshot the first's modified settings
    as if they were stock, and 'restore' the device to them.
    """
    sp = tmp_path / "state.json"
    c1 = FakeGpuController(power_limit_mw=200_000, power_default_limit_mw=200_000)
    c2 = FakeGpuController(power_limit_mw=200_000, power_default_limit_mw=200_000)
    g1 = GpuGuard(c1, consent=True, require_root=False, state_path=sp)
    g2 = GpuGuard(c2, consent=True, require_root=False, state_path=sp)

    g1.arm()
    g1.set_power_limit_mw(120_000)

    with pytest.raises(GuardBusyError, match="another GpuGuard in this process"):
        g2.arm()
    assert g2.snapshot is None, "the refused guard captured a snapshot anyway"
    assert c2.set_power_calls == 0

    g1.restore()
    assert c1.get_power_limit_mw() == 200_000


def test_re_arming_the_same_guard_is_idempotent(tmp_path):
    """The refusal must not break a guard re-entering arm() via lock_clocks."""
    c = FakeGpuController()
    g = GpuGuard(c, consent=True, require_root=False, state_path=tmp_path / "s.json")
    g.arm()
    g.lock_clocks(1400)      # leaves restored=false for this pid
    g.arm()                  # short-circuits on _armed, must not raise
    g.lock_clocks(1500)      # re-enters arm() internally
    assert c.locked == (1500, 1500)
    g.restore()


def test_a_fresh_guard_after_a_clean_restore_may_arm(tmp_path):
    """Single-instance must not wedge sequential use."""
    sp = tmp_path / "state.json"
    c1 = FakeGpuController()
    with GpuGuard(c1, consent=True, require_root=False, state_path=sp) as g1:
        g1.lock_clocks(1400)
    c2 = FakeGpuController()
    with GpuGuard(c2, consent=True, require_root=False, state_path=sp) as g2:
        g2.lock_clocks(1500)
        assert c2.locked == (1500, 1500)
    assert c2.locked is None


def test_recovery_refuses_to_clear_a_live_guards_record(tmp_path):
    """Recovery resets the clock but must not destroy a running guard's
    evidence -- arming refuses over a live record for exactly this reason."""
    sp = tmp_path / "state.json"
    sp.write_text(json.dumps({
        "schema": 3,
        "record": {"pid": os.getppid(), "restored": False, "clocks_locked": True,
                   "locked_min_mhz": 900, "locked_max_mhz": 900, "snapshot": {}},
    }))
    c = FakeGpuController()
    res = restore_from_state_file(c, sp)
    assert res["reset_clocks"] is True, "the clock reset must still happen"
    assert res["record_cleared"] is False
    assert "still running" in res["record_not_cleared_reason"]
    assert record(sp)["restored"] is False, "a live guard's evidence was erased"


def test_recovery_refuses_to_publish_a_stale_read(tmp_path):
    """If a guard writes while recovery is mid-flight, recovery must not
    publish the record it read before those hardware calls."""
    sp = tmp_path / "state.json"
    sp.write_text(json.dumps({
        "schema": 3,
        "record": {"pid": 999999, "restored": False, "clocks_locked": True,
                   "locked_min_mhz": 900, "locked_max_mhz": 900, "snapshot": {}},
    }))

    class MutateDuringReset(FakeGpuController):
        def reset_locked_clocks(self):
            self._require_claim()
            self.reset_calls += 1
            self.locked = None
            # A guard records new intent while we are resetting.
            sp.write_text(json.dumps({
                "schema": 3,
                "record": {"pid": 999999, "restored": False,
                           "clocks_locked": True, "locked_min_mhz": 1400,
                           "locked_max_mhz": 1400, "snapshot": {}},
            }))

    res = restore_from_state_file(MutateDuringReset(), sp)
    assert res["reset_clocks"] is True
    assert res["record_cleared"] is False
    assert "changed while recovery was running" in res["record_not_cleared_reason"]
    assert record(sp)["locked_min_mhz"] == 1400, "the newer record was clobbered"


def test_unreadable_state_file_is_reported_stale_not_clean(tmp_path):
    """'The file is the only evidence' -- an unparseable one must not read as
    'nothing ever ran'."""
    sp = tmp_path / "state.json"
    sp.write_text("{ this is not json")
    rep = check_stale_state(FakeGpuController(), sp)
    assert rep["state_file_present"] is True
    assert rep["state_file_unreadable"] is True
    assert rep["stale"] is True
    assert any("UNREADABLE" in f for f in rep["findings"])


# ---------------------------------------------------------------------------
# pass-6: state-file validation and device identity
# ---------------------------------------------------------------------------


def _write_record(path, **over):
    rec = {"pid": 999999, "restored": False, "clocks_locked": True,
           "device_index": 0, "locked_min_mhz": 900, "locked_max_mhz": 900,
           "snapshot": {"power_limit_mw": 150_000,
                        "power_default_limit_mw": 200_000}}
    rec.update(over)
    Path(path).write_text(json.dumps({"schema": 3, "record": rec}))
    return rec


def test_recovery_refuses_a_record_for_another_device(tmp_path):
    """The severe one. A record for GPU 1 must not be applied to GPU 0.

    On a homogeneous node the other device's power value is inside this one's
    constraints, so NVML accepts it silently: GPU 0 gets mis-capped, GPU 1's
    record is cleared, and GPU 1 stays locked with its evidence destroyed.
    """
    sp = tmp_path / "state.json"
    _write_record(sp, device_index=1, power_limit_written_mw=120_000)
    c0 = FakeGpuController(power_limit_mw=200_000, power_default_limit_mw=200_000)

    res = restore_from_state_file(c0, sp)          # controller is GPU 0

    assert res["reset_clocks"] is True, "the local clock reset must still apply"
    assert c0.get_power_limit_mw() == 200_000, "another GPU's snapshot was applied"
    assert res["record_cleared"] is False
    assert res["incomplete"] is True
    assert any("names GPU 1" in e for e in res["errors"])
    assert record(sp)["restored"] is False, "another device's evidence was erased"


def test_stale_check_names_the_device_and_the_gpu_flag(tmp_path):
    """The recommendation used to omit --gpu, walking the operator into the
    wrong-device write."""
    sp = tmp_path / "state.json"
    _write_record(sp, device_index=1)
    rep = check_stale_state(FakeGpuController(), sp)   # controller is GPU 0
    assert rep["record_is_for_this_device"] is False
    assert any("OTHER DEVICE" in f for f in rep["findings"])
    assert "--gpu 1" in rep["recommendation"]


def test_stale_recommendation_carries_gpu_for_this_device_too(tmp_path):
    sp = tmp_path / "state.json"
    _write_record(sp, device_index=0)
    rep = check_stale_state(FakeGpuController(), sp)
    assert rep["record_is_for_this_device"] is True
    assert "--gpu 0" in rep["recommendation"]


@pytest.mark.parametrize("body", [
    "{ not json",
    '["a", "list"]',
    '{"schema": 4, "records": []}',
    '{"guards": [1, 2]}',
    '{"guards": {"1": "not-a-record"}}',
    '{"record": "not-a-record"}',
])
def test_malformed_state_files_are_unreadable_not_crashes(tmp_path, body):
    """A document that parses but has no shape we recognise is equally
    unusable -- and must not raise AttributeError out of the read-only
    `status` command."""
    sp = tmp_path / "state.json"
    sp.write_text(body)
    rep = check_stale_state(FakeGpuController(), sp)   # must not raise
    assert rep["state_file_unreadable"] is True
    assert rep["stale"] is True
    assert any("UNREADABLE" in f for f in rep["findings"])


def test_arming_over_an_unreadable_state_file_is_refused(tmp_path):
    """_read_record's own docstring says callers that care must consult the
    unreadable state; this is the caller that cares most."""
    sp = tmp_path / "state.json"
    sp.write_text('{"schema": 4, "records": []}')
    c = FakeGpuController()
    g = GpuGuard(c, consent=True, require_root=False, state_path=sp)
    with pytest.raises(GuardBusyError, match="could not be interpreted"):
        g.arm()
    assert c.set_clock_calls == 0
    assert sp.read_text() == '{"schema": 4, "records": []}', "the file was overwritten"


def test_unreadable_file_can_be_resolved_by_quarantine(tmp_path):
    """Previously a permanent wedge: status said stale + 'run restore', and
    restore exited 0 having changed nothing."""
    sp = tmp_path / "state.json"
    sp.write_text("{ corrupt")
    c = FakeGpuController()

    res = restore_from_state_file(c, sp)
    assert res["incomplete"] is True and res["errors"], "wedge not reported"

    res2 = restore_from_state_file(c, sp, quarantine_unreadable=True)
    assert res2["quarantined_to"]
    assert Path(res2["quarantined_to"]).exists(), "the original was not preserved"
    assert not sp.exists()
    assert check_stale_state(c, sp)["stale"] is False, "status still wedged"


def test_legacy_multi_record_file_is_not_silently_collapsed(tmp_path):
    """A schema-2 file with two unrestored records would lose one to the
    schema-3 rewrite -- possibly one naming a different GPU."""
    sp = tmp_path / "state.json"
    sp.write_text(json.dumps({"schema": 2, "guards": {
        "111": {"pid": 111, "restored": False, "clocks_locked": True,
                "device_index": 0, "snapshot": {}},
        "222": {"pid": 222, "restored": False, "clocks_locked": True,
                "device_index": 1, "snapshot": {}},
    }}))
    rep = check_stale_state(FakeGpuController(), sp)
    assert rep["state_file_unreadable"] is True
    assert any("destroy evidence" in f for f in rep["findings"])
    res = restore_from_state_file(FakeGpuController(), sp)
    assert res["record_cleared"] is False
    assert json.loads(sp.read_text())["schema"] == 2, "the legacy file was rewritten"


def test_failed_restore_then_new_guard_names_the_real_cause(tmp_path):
    """Refusing is right; the message must not blame a second guard object."""
    sp = tmp_path / "state.json"
    _write_record(sp, pid=os.getpid(), restore_failed=True,
                  restore_errors=["reset_locked_clocks: boom"])
    g = GpuGuard(FakeGpuController(), consent=True, require_root=False,
                 state_path=sp)
    with pytest.raises(GuardBusyError, match="FAILED its restore"):
        g.arm()


# ---------------------------------------------------------------------------
# multi-GPU identity -- protected surface
# ---------------------------------------------------------------------------


def test_recovery_refuses_a_record_from_a_different_uuid(tmp_path):
    """Two physical cards, simulated with the fake NVML layer.

    A record written by GPU 1 must never be applied to GPU 0: the snapshot's
    power value belongs to the other card, and on a homogeneous node it is
    inside this card's constraints so NVML accepts it silently.
    """
    sp = tmp_path / "state.json"
    gpu1 = FakeGpuController(index=1, uuid="GPU-1111",
                             power_limit_mw=150_000, power_default_limit_mw=200_000)
    gpu0 = FakeGpuController(index=0, uuid="GPU-0000",
                             power_limit_mw=200_000, power_default_limit_mw=200_000)

    # GPU 1 runs, caps power, and dies without restoring.
    g1 = GpuGuard(gpu1, consent=True, require_root=False, state_path=sp)
    g1.arm()
    g1.set_power_limit_mw(120_000)
    assert record(sp)["device_uuid"] == "GPU-1111"

    res = restore_from_state_file(gpu0, sp)      # operator points at GPU 0

    assert res["reset_clocks"] is True, "the local clock reset must still apply"
    assert gpu0.get_power_limit_mw() == 200_000, "GPU 1's snapshot hit GPU 0"
    assert res["incomplete"] is True
    assert res["record_cleared"] is False
    assert any("GPU-1111" in e for e in res["errors"])
    assert record(sp)["restored"] is False, "GPU 1's evidence was destroyed"

    # And pointed at the right card it works. A fresh controller, because
    # recovery normally runs in a separate process after the writer died --
    # the exclusive claim correctly refuses a controller a live guard holds.
    gpu1_after = FakeGpuController(index=1, uuid="GPU-1111",
                                   power_limit_mw=120_000,
                                   power_default_limit_mw=200_000)
    res2 = restore_from_state_file(gpu1_after, sp)
    assert res2["errors"] == [], res2["errors"]
    assert res2["record_cleared"] is True
    assert gpu1_after.get_power_limit_mw() == 150_000, "restored to the wrong value"


def test_uuid_mismatch_overrides_a_matching_index(tmp_path):
    """Index is an enumeration order, not an identity.

    After a reboot or a CUDA_VISIBLE_DEVICES change, index 0 can be a different
    physical card. A record whose index matches but whose UUID does not must be
    refused -- this is the case an index-only check cannot see.
    """
    sp = tmp_path / "state.json"
    before = FakeGpuController(index=0, uuid="GPU-AAAA")
    g = GpuGuard(before, consent=True, require_root=False, state_path=sp)
    g.arm()
    g.lock_clocks(1400)

    after_reboot = FakeGpuController(index=0, uuid="GPU-BBBB")   # same index!
    res = restore_from_state_file(after_reboot, sp)

    assert res["incomplete"] is True
    assert res["record_cleared"] is False
    assert any("enumeration changed" in e for e in res["errors"])
    assert record(sp)["restored"] is False


def test_stale_check_reports_uuid_mismatch_and_the_right_gpu_flag(tmp_path):
    sp = tmp_path / "state.json"
    gpu1 = FakeGpuController(index=1, uuid="GPU-1111")
    g = GpuGuard(gpu1, consent=True, require_root=False, state_path=sp)
    g.arm()
    g.lock_clocks(1400)

    rep = check_stale_state(FakeGpuController(index=0, uuid="GPU-0000"), sp)
    assert rep["record_is_for_this_device"] is False
    assert rep["record_device_uuid"] == "GPU-1111"
    assert any("OTHER DEVICE" in f for f in rep["findings"])
    assert "--gpu 1" in rep["recommendation"]


def test_matching_uuid_is_accepted_even_if_the_index_moved(tmp_path):
    """The converse: the same card re-enumerated must still be recoverable."""
    sp = tmp_path / "state.json"
    was_1 = FakeGpuController(index=1, uuid="GPU-SAME")
    g = GpuGuard(was_1, consent=True, require_root=False, state_path=sp)
    g.arm()
    g.lock_clocks(1400)

    now_0 = FakeGpuController(index=0, uuid="GPU-SAME")   # same card, new index
    res = restore_from_state_file(now_0, sp)
    assert res["errors"] == []
    assert res["record_cleared"] is True


# ---------------------------------------------------------------------------
# Pass 7: three defects re-derived after reverting an over-large rewrite.
# Each is a hole in the *kept* core, not in machinery added by a review pass.
# ---------------------------------------------------------------------------


def test_arming_publishes_the_record_before_the_first_hardware_write(tmp_path):
    """Two guards must not both get past arm().

    `_check_not_busy` reads the state file, but the record was only written at
    the first hardware touch. Between one guard's arm() and its first write, a
    second guard saw a clean file and armed too. Both then held snapshots and
    both wrote the file wholesale, so the second one's record ERASED the first
    one's -- and a record that says `power_limit_written_mw: null` tells
    recovery to leave a cap the first guard actually applied.

    The window closes only if arming itself publishes the claim.
    """
    sp = tmp_path / "state.json"
    first = GpuGuard(FakeGpuController(), consent=True, require_root=False,
                     state_path=sp)
    first.arm()

    # No hardware touched yet -- this is the whole window.
    assert first.state.power_limit_written_mw is None
    assert first.state.clocks_locked is False

    assert sp.exists(), "arm() left no evidence, so the claim is invisible"
    assert record(sp)["restored"] is False, (
        "the published claim must read as unrestored, or a second guard's "
        "_check_not_busy will step straight over it"
    )

    second = GpuGuard(FakeGpuController(), consent=True, require_root=False,
                      state_path=sp)
    with pytest.raises(GuardBusyError):
        second.arm()


def test_a_second_process_cannot_arm_during_the_pre_touch_window(tmp_path):
    """The same race across processes, where the in-process claim cannot help.

    Two guards in one process are refused by the controller claim. Two
    processes hold separate controller objects, so only the state file stands
    between them.
    """
    sp = tmp_path / "state.json"
    holder = GpuGuard(FakeGpuController(), consent=True, require_root=False,
                      state_path=sp)
    holder.arm()                       # armed, nothing written to hardware

    probe = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(f"""
            import sys
            sys.path.insert(0, {str(Path.cwd())!r})
            from joule_agent.gpu_guard import (
                FakeGpuController, GpuGuard, GuardBusyError)
            g = GpuGuard(FakeGpuController(), consent=True, require_root=False,
                         state_path={str(sp)!r})
            try:
                g.arm()
            except GuardBusyError:
                print("REFUSED")
            else:
                print("ARMED")
        """)],
        capture_output=True, text=True, timeout=60,
    )
    assert "REFUSED" in probe.stdout, (
        f"a second process armed over a live claim: {probe.stdout!r} "
        f"{probe.stderr!r}"
    )


def test_a_cleared_record_does_not_replay_power_over_a_later_operator_cap(tmp_path):
    """`restored: true` means the debt is settled, not that it is still owed.

    Nothing clears `power_limit_written_mw` when restore succeeds, so a
    finished run leaves a clean record that still names a power write. Recovery
    read that field alone and rewrote the snapshot's historical limit --
    silently undoing a cap the operator set AFTER the run finished, and
    reporting success.
    """
    sp = tmp_path / "state.json"
    c = FakeGpuController(power_limit_mw=200_000, power_default_limit_mw=200_000)
    with GpuGuard(c, consent=True, require_root=False, state_path=sp) as g:
        g.set_power_limit_mw(150_000)

    settled = record(sp)
    assert settled["restored"] is True
    assert settled["power_limit_written_mw"] == 150_000, (
        "precondition: the cleared record still names the historical write"
    )
    assert c.get_power_limit_mw() == 200_000

    # The operator deliberately caps the card after the run. Recovery runs in
    # its own process, so it gets its own controller for the same card.
    later = FakeGpuController(power_limit_mw=180_000,
                              power_default_limit_mw=200_000)

    res = restore_from_state_file(later, sp)

    assert later.get_power_limit_mw() == 180_000, (
        "recovery replayed a settled record's power write over a later "
        "operator cap"
    )
    assert later.set_power_calls == 0
    assert res["power_restored_to_mw"] is None
    assert res["reset_clocks"] is True      # belt-and-braces still unconditional


def test_an_unreadable_live_uuid_is_not_covered_by_a_matching_index(tmp_path):
    """An index match is not evidence when the identity cannot be read.

    A record that carries a UUID was written by a guard that could read one.
    If the live UUID read fails, identity is UNVERIFIABLE -- but the comparison
    fell through to the index, and an index match returned "no mismatch". After
    a re-enumeration that is precisely the case where the index lies, so the
    fallback accepted the one situation it exists to catch.
    """
    sp = tmp_path / "state.json"
    written_by = FakeGpuController(index=0, uuid="GPU-AAAA")
    g = GpuGuard(written_by, consent=True, require_root=False, state_path=sp)
    g.arm()
    g.lock_clocks(1400)
    g.set_power_limit_mw(150_000)

    class UuidUnreadable(FakeGpuController):
        def snapshot(self):
            snap = super().snapshot()
            return replace(snap, uuid=None)

    # Same index, unknowable identity: may or may not be the recorded card.
    unknown = UuidUnreadable(index=0, uuid="GPU-AAAA",
                             power_limit_mw=180_000,
                             power_default_limit_mw=200_000)
    res = restore_from_state_file(unknown, sp)

    assert res["incomplete"] is True
    assert any("identity" in e for e in res["errors"]), res["errors"]
    assert res["power_restored_to_mw"] is None, (
        "applied a snapshot from a record whose device could not be confirmed"
    )
    assert res["record_cleared"] is False, (
        "cleared the only evidence for a card that may still be modified"
    )
    assert unknown.get_power_limit_mw() == 180_000
    assert res["reset_clocks"] is True


def test_force_power_default_still_escapes_an_unverifiable_identity(tmp_path):
    """Fail-closed must not mean fail-stuck.

    Refusing on an unreadable UUID is only acceptable if the operator retains a
    way to put the card back. --force-power-default needs no snapshot, so it
    must survive the refusal.
    """
    sp = tmp_path / "state.json"
    g = GpuGuard(FakeGpuController(index=0, uuid="GPU-AAAA"), consent=True,
                 require_root=False, state_path=sp)
    g.arm()
    g.set_power_limit_mw(150_000)

    class UuidUnreadable(FakeGpuController):
        def snapshot(self):
            return replace(super().snapshot(), uuid=None)

    unknown = UuidUnreadable(index=0, power_limit_mw=150_000,
                             power_default_limit_mw=200_000)
    res = restore_from_state_file(unknown, sp, force_power_default=True)

    assert unknown.get_power_limit_mw() == 200_000
    assert res["power_restored_to_mw"] == 200_000
    assert res["incomplete"] is True, (
        "the identity refusal must still be reported; only the cap was undone"
    )


def test_a_finished_guard_that_never_wrote_releases_its_claim(tmp_path):
    """Publishing a claim at arm() is only safe if exiting releases it.

    A guard that arms, writes nothing, and exits cleanly must not leave a
    record saying the device is held. If it does, the next run reads a live
    claim from a dead pid and refuses to arm -- and `status` reports a stale
    record that names no hardware change at all.

    The release must not reset clocks: a guard that never wrote must not touch
    the device on its way out.
    """
    sp = tmp_path / "state.json"
    c = FakeGpuController()
    with GpuGuard(c, consent=True, require_root=False, state_path=sp):
        assert record(sp)["restored"] is False, "precondition: claim published"
    assert record(sp)["restored"] is True, "the claim outlived the guard"
    assert c.reset_calls == 0, "releasing an unwritten claim touched hardware"

    # The next run must be able to arm over it.
    again = GpuGuard(FakeGpuController(), consent=True, require_root=False,
                     state_path=sp)
    again.arm()
    again.restore()


# ---------------------------------------------------------------------------
# Pass 8: findings from the pass-7 protected-surface review.
# ---------------------------------------------------------------------------


def test_state_writes_run_with_the_handled_signals_blocked(tmp_path):
    """_write_state must not be re-enterable from a signal handler.

    Four call sites (arm, the claim release, and both of restore's write
    branches) run outside any caller-level signal block. A handler delivered
    mid-write re-enters here via _signal_restore -> restore: the inner call
    publishes the newer truth and the interrupted outer frame then resumes and
    replaces it with its own stale content. The worst instance is restore's
    failure branch, where the record being overwritten is the only evidence
    that a still-modified GPU exists.
    """
    import joule_agent.gpu_guard as gg

    seen = {}
    real_mkstemp = gg.tempfile.mkstemp

    def spy(*a, **kw):
        seen["blocked"] = signal.pthread_sigmask(signal.SIG_BLOCK, [])
        return real_mkstemp(*a, **kw)

    gg.tempfile.mkstemp = spy
    try:
        before = signal.pthread_sigmask(signal.SIG_BLOCK, [])
        gg._write_state(tmp_path / "s.json", gg.GuardState(pid=os.getpid()))
    finally:
        gg.tempfile.mkstemp = real_mkstemp

    assert signal.SIGINT in seen["blocked"], "SIGINT was deliverable mid-write"
    assert signal.SIGTERM in seen["blocked"], "SIGTERM was deliverable mid-write"
    assert signal.pthread_sigmask(signal.SIG_BLOCK, []) == before, (
        "the mask was not restored"
    )


def test_a_failed_state_write_leaves_no_debris_and_no_lost_record(tmp_path):
    """A shared temp name let two calls collide on one inode. Prove uniqueness
    is real by proving the temp file is neither reused nor left behind."""
    import joule_agent.gpu_guard as gg

    sp = tmp_path / "s.json"
    gg._write_state(sp, gg.GuardState(pid=1234))
    good = sp.read_text()

    real_replace = os.replace

    def boom(src, dst):
        raise OSError("no space left on device")

    os.replace = boom
    try:
        with pytest.raises(OSError):
            gg._write_state(sp, gg.GuardState(pid=5678))
    finally:
        os.replace = real_replace

    assert sp.read_text() == good, "a failed write destroyed the live record"
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "s.json"]
    assert leftovers == [], f"temp files left behind: {leftovers}"


def test_installing_handlers_twice_does_not_chain_the_guard_to_itself(tmp_path):
    """The second install must not record the first install's own handler.

    signal.getsignal() returns whatever is currently installed. On a second
    pass that is self._signal_restore, so _prev_handlers[sig] becomes the
    guard's own handler -- and _signal_restore's chain-to-previous call at the
    end then calls itself, unbounded, re-issuing NVML restores all the way
    down. The process survives a SIGTERM it was meant to die from, and the
    application's real handler is lost for good.
    """
    original = signal.getsignal(signal.SIGINT)
    g = guard(tmp_path)
    try:
        g._install_handlers()
        g._install_handlers()
        for sig in (signal.SIGINT, signal.SIGTERM):
            assert g._prev_handlers[sig] is not g._signal_restore, (
                f"{sig!r}: the guard chained itself to its own handler"
            )
        assert g._prev_handlers[signal.SIGINT] is original
    finally:
        for sig, prev in g._prev_handlers.items():
            signal.signal(sig, prev)


def test_the_previous_handler_runs_exactly_once_after_a_reused_guard(tmp_path):
    """The same defect, end to end, in a real process taking a real signal."""
    script = textwrap.dedent(f"""
        import signal, sys
        sys.path.insert(0, {str(Path.cwd())!r})
        from joule_agent.gpu_guard import FakeGpuController, GpuGuard

        calls = []
        signal.signal(signal.SIGTERM, lambda s, f: calls.append(s))

        g = GpuGuard(FakeGpuController(), consent=True, require_root=False,
                     state_path={str(tmp_path / "s.json")!r})
        with g:
            pass                      # arms, then surrenders and disarms
        with g:
            g.lock_clocks(1400)       # re-arms: handlers installed a second time
            import os
            os.kill(os.getpid(), signal.SIGTERM)
        print("PREV_CALLS", len(calls))
    """)
    p = subprocess.run([sys.executable, "-c", script],
                       capture_output=True, text=True, timeout=60)
    assert "PREV_CALLS 1" in p.stdout, (
        f"the application's handler did not run exactly once: "
        f"{p.stdout!r} {p.stderr[-400:]!r}"
    )
    assert "RecursionError" not in p.stderr, "the guard chained into itself"


def test_a_surrendered_claim_disarms_so_reuse_rechecks_the_device(tmp_path):
    """Releasing the claim without disarming leaves a guard that can write
    hardware while holding nothing on disk.

    `_armed` short-circuits arm(), so a reused guard object would reach the
    device having never re-run _check_not_busy -- and in the meantime another
    process is free to arm over the released claim.
    """
    sp = tmp_path / "state.json"
    g = GpuGuard(FakeGpuController(), consent=True, require_root=False,
                 state_path=sp)
    with g:
        pass                                  # claim published, then surrendered
    assert g._armed is False, "the guard stayed armed with no claim on disk"

    # Another owner takes the device in the gap.
    other = GpuGuard(FakeGpuController(), consent=True, require_root=False,
                     state_path=sp)
    other.arm()

    with pytest.raises(GuardBusyError):
        g.lock_clocks(1400)


def test_status_tells_cannot_tell_apart_from_other_card(tmp_path):
    """Prescribing --gpu N for an unverifiable identity is a non-terminating loop.

    The index is exactly what cannot be trusted in this case, so re-running
    against it re-enters the same refusal. The report must say the identity is
    unknowable and name the one action that still works.
    """
    sp = tmp_path / "state.json"
    g = GpuGuard(FakeGpuController(index=0, uuid="GPU-AAAA"), consent=True,
                 require_root=False, state_path=sp)
    g.arm()
    g.lock_clocks(1400)

    class UuidUnreadable(FakeGpuController):
        def snapshot(self):
            return replace(super().snapshot(), uuid=None)

    rep = check_stale_state(UuidUnreadable(index=0), sp)
    assert rep["identity_unverifiable"] is True
    assert rep["record_is_for_this_device"] is False
    assert not any("OTHER DEVICE" in f for f in rep["findings"]), (
        "reported 'cannot tell' as 'other card'"
    )
    assert any("UNVERIFIABLE" in f for f in rep["findings"])
    assert "--force-power-default" in rep["recommendation"]

    # The contrast case must still get the plain --gpu advice.
    other = check_stale_state(FakeGpuController(index=1, uuid="GPU-BBBB"), sp)
    assert other["identity_unverifiable"] is False
    assert any("OTHER DEVICE" in f for f in other["findings"])
    assert "--gpu 0" in other["recommendation"]


def test_force_power_default_does_not_claim_a_write_it_never_made(tmp_path):
    """An operator who asked for a hardware change must not be told it happened.

    get_power_default_limit_mw is _try-wrapped and swallows every NVML error,
    so "no default" arrives as None. The flag was set before that was checked,
    producing power_forced_to_default: true with an empty error list and exit 0.
    """
    class NoDefault(FakeGpuController):
        def get_power_default_limit_mw(self):
            return None

    c = NoDefault(power_limit_mw=150_000)
    res = restore_from_state_file(c, tmp_path / "missing.json",
                                  force_power_default=True)

    assert res.get("power_forced_to_default") is not True, (
        "claimed a forced write that never happened"
    )
    assert res["power_restored_to_mw"] is None
    assert res["incomplete"] is True
    assert any("could not be read" in e for e in res["errors"]), res["errors"]
    assert c.get_power_limit_mw() == 150_000


def test_the_busy_message_does_not_overstate_an_arm_only_record(tmp_path):
    """arm() publishes before the first write, so a record can be unrestored and
    name no hardware change at all. Saying "the GPU may still be modified" about
    a record that states the opposite is a false statement in the one place an
    operator decides what to trust."""
    sp = tmp_path / "state.json"
    doc = {
        "schema": 3,
        "record": {
            "pid": 999_999, "started_utc": "2026-01-01T00:00:00+00:00",
            "device_index": 0, "device_uuid": "GPU-FAKE",
            "snapshot": {"index": 0, "uuid": "GPU-FAKE"},
            "clocks_locked": False, "locked_min_mhz": None,
            "locked_max_mhz": None, "power_limit_written_mw": None,
            "restored": False, "restored_utc": None,
            "restore_failed": False, "restore_errors": [],
        },
    }
    sp.write_text(json.dumps(doc))

    with pytest.raises(GuardBusyError) as exc:
        guard(tmp_path).arm()
    msg = str(exc.value)
    assert "may still be modified" not in msg, msg
    assert "no clock lock and no power write" in msg
    assert "restore --gpu 0" in msg, "the resolving command is not named"

    # A record that DOES name a write must still say so.
    doc["record"]["clocks_locked"] = True
    doc["record"]["locked_min_mhz"] = 1400
    doc["record"]["locked_max_mhz"] = 1400
    sp.write_text(json.dumps(doc))
    with pytest.raises(GuardBusyError) as exc2:
        guard(tmp_path).arm()
    assert "may still be modified" in str(exc2.value)


def test_a_caller_without_consent_leaves_no_claim_behind(tmp_path):
    """arm() publishes a claim, so consent must be checked before it runs.

    Otherwise a caller that was never allowed to touch the device leaves an
    operator-visible record on disk, and a kill in that window strands it.
    """
    sp = tmp_path / "state.json"
    for call in (lambda g: g.lock_clocks(1400),
                 lambda g: g.set_power_limit_mw(150_000)):
        g = GpuGuard(FakeGpuController(), consent=False, require_root=False,
                     state_path=sp)
        with pytest.raises(ConsentError):
            call(g)
        assert not sp.exists(), "a consent-refused caller published a claim"
        assert g._armed is False


def test_the_cli_lock_refuses_consent_before_it_arms(tmp_path):
    """Found by running the real CLI, not by a unit test.

    lock_clocks checks consent before its own auto-arm, but `main`'s lock path
    enters the context manager first -- and __enter__ arms, which publishes a
    claim. A caller who was never going to be allowed to write left a record on
    disk; a kill in that window would strand it.
    """
    from joule_agent.gpu_guard import main

    sp = tmp_path / "state.json"
    rc = main(["--gpu", "0", "--state-path", str(sp), "lock", "--mhz", "900",
               "--hold", "0"])
    assert rc == 2, "a consentless lock did not fail loudly"
    assert not sp.exists(), "a consent-refused CLI run published a claim"


def test_reads_still_need_no_consent_after_that_change(tmp_path):
    """Invariant 5 in the direction the fix could have broken.

    The refusal belongs in the CLI's write path, NOT in arm(): snapshotting is
    a read, and a guard armed without consent must still be able to report the
    device.
    """
    c = FakeGpuController()
    g = GpuGuard(c, consent=False, require_root=False,
                 state_path=tmp_path / "s.json")
    g.arm()
    assert g.snapshot.name == "FakeGPU"
    assert c.set_clock_calls == 0
    assert check_stale_state(FakeGpuController(), tmp_path / "s.json") is not None


# ---------------------------------------------------------------------------
# Pass 9: findings from the pass-8 protected-surface review.
# ---------------------------------------------------------------------------


def test_a_failed_final_write_leaves_the_retry_alive(tmp_path):
    """The in-memory flag must never claim more than the disk does.

    restore() flipped state.restored to True and THEN wrote. If the write
    failed it returned quietly -- and because that same flag gates
    _surrender_claim's write block, every later restore path skipped the retry.
    Hardware back to stock, disk still saying it is modified, and nothing on
    any path ever correcting it.
    """
    import joule_agent.gpu_guard as gg

    sp = tmp_path / "state.json"
    c = FakeGpuController()
    g = GpuGuard(c, consent=True, require_root=False, state_path=sp)
    g.arm()
    g.lock_clocks(1400)

    real = gg._write_state
    gg._write_state = lambda *a, **kw: (_ for _ in ()).throw(OSError("ENOSPC"))
    try:
        g.restore()                       # hardware restored; record write fails
    finally:
        gg._write_state = real

    assert c.locked is None, "precondition: the hardware really was restored"
    assert record(sp)["restored"] is False, "precondition: the disk is stale"
    assert g.state.restored is False, (
        "the in-memory flag outran the disk, which kills every retry"
    )
    assert g._armed is True, "disarmed while the claim was still published"

    g.restore()                           # the retry must now actually write
    assert record(sp)["restored"] is True, "no restore path ever retried"
    assert g._armed is False


def test_a_handler_installed_between_uses_is_not_lost(tmp_path):
    """Recording the previous handler ONCE is wrong now that the guard disarms.

    An application can install its own handler between one `with` block and the
    next. A once-only record overwrites it with _signal_restore and chains to
    the stale handler from the first arm, so the application's current handler
    never runs again -- and is unrecoverable.
    """
    script = textwrap.dedent(f"""
        import os, signal, sys
        sys.path.insert(0, {str(Path.cwd())!r})
        from joule_agent.gpu_guard import FakeGpuController, GpuGuard

        first, second = [], []
        signal.signal(signal.SIGTERM, lambda s, f: first.append(s))

        g = GpuGuard(FakeGpuController(), consent=True, require_root=False,
                     state_path={str(tmp_path / "s.json")!r})
        with g:
            pass                          # arms, surrenders, disarms

        # The application changes its mind between uses.
        signal.signal(signal.SIGTERM, lambda s, f: second.append(s))

        with g:
            g.lock_clocks(1400)
            os.kill(os.getpid(), signal.SIGTERM)
        print("FIRST", len(first), "SECOND", len(second))
    """)
    p = subprocess.run([sys.executable, "-c", script],
                       capture_output=True, text=True, timeout=60)
    assert "FIRST 0 SECOND 1" in p.stdout, (
        f"the handler installed between uses was lost: "
        f"{p.stdout!r} {p.stderr[-400:]!r}"
    )


def test_the_stored_handler_is_one_stable_object(tmp_path):
    """The mechanism the fix depends on, pinned directly.

    `self._signal_restore` builds a new bound method on every access, so an
    `is` comparison against it is always False -- which would make the "is the
    handler already mine?" question unanswerable and silently reinstate the
    once-only bug.
    """
    g = guard(tmp_path)
    assert g._signal_restore is not g._signal_restore, (
        "bound methods stopped being per-access; the guard below is moot"
    )
    assert g._handler is g._handler
    original = signal.getsignal(signal.SIGINT)
    try:
        g._install_handlers()
        assert signal.getsignal(signal.SIGINT) is g._handler
    finally:
        for sig, prev in g._prev_handlers.items():
            signal.signal(sig, prev)
    assert original is signal.getsignal(signal.SIGINT)


def test_a_failed_claim_release_also_leaves_the_retry_alive(tmp_path):
    """The same rollback, on the arm-only path.

    A guard that never wrote hardware still has a claim to release, and
    _surrender_claim's own write can fail. Its flag gates its own write block,
    so leaving it set makes every later call a no-op over a claim that is still
    published -- which then refuses the next arm forever.
    """
    import joule_agent.gpu_guard as gg

    sp = tmp_path / "state.json"
    c = FakeGpuController()
    g = GpuGuard(c, consent=True, require_root=False, state_path=sp)
    g.arm()                                     # claim published, nothing written
    assert record(sp)["restored"] is False

    real = gg._write_state
    gg._write_state = lambda *a, **kw: (_ for _ in ()).throw(OSError("EIO"))
    try:
        g.restore()                             # goes straight to _surrender_claim
    finally:
        gg._write_state = real

    assert c.reset_calls == 0, "an arm-only guard touched hardware"
    assert record(sp)["restored"] is False, "precondition: the claim is still up"
    assert g.state.restored is False, (
        "the flag outran the disk, so _surrender_claim's own retry is dead"
    )
    assert g._armed is True

    g.restore()
    assert record(sp)["restored"] is True, "the claim was never released"
    assert g._armed is False
    assert c.reset_calls == 0


def test_a_moved_enumeration_is_not_told_to_re_run_the_command_that_failed(tmp_path):
    """The recorded index still points here, but at a different card.

    Whatever sent the operator to `--gpu 0` sends them here again, so that
    advice cannot terminate. The sibling case (unverifiable identity) was fixed
    in pass 8 with exactly this reasoning; this one was left with the loop. The
    only usable handle left is the recorded UUID.
    """
    sp = tmp_path / "state.json"
    was_here = FakeGpuController(index=0, uuid="GPU-AAAA")
    g = GpuGuard(was_here, consent=True, require_root=False, state_path=sp)
    g.arm()
    g.lock_clocks(1400)
    g.set_power_limit_mw(150_000)

    # A reboot re-enumerates: index 0 is now a different physical card.
    now_here = FakeGpuController(index=0, uuid="GPU-BBBB",
                                 power_limit_mw=180_000)

    rep = check_stale_state(now_here, sp)
    assert rep["record_index_no_longer_locates"] is True
    assert rep["identity_unverifiable"] is False
    assert any("RE-ENUMERATED" in f for f in rep["findings"])
    assert "--gpu 0" not in rep["recommendation"], (
        "prescribed the index that just failed -- a loop with no exit"
    )
    assert "GPU-AAAA" in rep["recommendation"], "the recorded UUID is the handle"
    assert "nvidia-smi" in rep["recommendation"]

    res = restore_from_state_file(now_here, sp)
    assert res["incomplete"] is True
    assert res["record_cleared"] is False
    assert now_here.get_power_limit_mw() == 180_000, "mis-capped the wrong card"
    joined = " ".join(res["errors"])
    assert "--gpu 0" not in joined, joined
    assert "GPU-AAAA" in joined and "nvidia-smi" in joined


def test_a_moved_card_still_gets_the_plain_gpu_flag(tmp_path):
    """The contrast case the new branch must not capture.

    When the indexes differ, --gpu <recorded index> is exactly right and the
    operator should not be sent on a UUID hunt.
    """
    sp = tmp_path / "state.json"
    g = GpuGuard(FakeGpuController(index=1, uuid="GPU-AAAA"), consent=True,
                 require_root=False, state_path=sp)
    g.arm()
    g.lock_clocks(1400)

    elsewhere = FakeGpuController(index=0, uuid="GPU-BBBB")
    rep = check_stale_state(elsewhere, sp)
    assert rep["record_index_no_longer_locates"] is False
    assert any("OTHER DEVICE" in f for f in rep["findings"])
    assert "--gpu 1" in rep["recommendation"]
    assert "nvidia-smi" not in rep["recommendation"]

    res = restore_from_state_file(elsewhere, sp)
    assert "--gpu 1" in " ".join(res["errors"])
