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
    PrivilegeError,
    check_stale_state,
    restore_from_state_file,
)


def record(state_path, pid=None) -> dict:
    """Read one guard record from the per-pid state document."""
    doc = json.loads(Path(state_path).read_text())
    guards = doc["guards"]
    return guards[str(pid)] if pid is not None else next(iter(guards.values()))


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


def test_concurrent_guards_do_not_clobber_each_others_records(tmp_path):
    """F4: a clean restore by one guard must not erase another's live record."""
    sp = tmp_path / "state.json"
    a = FakeGpuController()
    b = FakeGpuController()
    ga = GpuGuard(a, consent=True, require_root=False, state_path=sp)
    gb = GpuGuard(b, consent=True, require_root=False, state_path=sp)
    ga.arm(); gb.arm()
    ga.state.pid = 1111
    gb.state.pid = 2222
    ga.lock_clocks(1400)
    gb.lock_clocks(900)
    ga.restore()

    doc = json.loads(sp.read_text())
    assert doc["guards"]["1111"]["restored"] is True
    assert doc["guards"]["2222"]["restored"] is False, "B's record was clobbered"
    rep = check_stale_state(FakeGpuController(), sp)
    assert rep["stale"] is True, "B still holds a lock but status reported clean"


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


def test_handler_install_failure_is_recorded_not_silent(tmp_path, capfd):
    """F2: the load-bearing half is the stderr warning, not an in-memory list."""
    import threading as _t

    c = FakeGpuController()
    g = GpuGuard(c, consent=True, require_root=False, state_path=tmp_path / "s.json")
    th = _t.Thread(target=g.arm)
    th.start(); th.join()

    err = capfd.readouterr().err
    assert g.handler_install_failures, "off-main-thread arm reported no degradation"
    assert "WARNING [gpu_guard]" in err, "degradation was silent on stderr"
    assert "not armed" in err.lower() or "NOT armed" in err


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


def test_signal_during_a_write_does_not_record_a_false_clean(tmp_path):
    """A signal handler can land BETWEEN the state write and the hardware write.

    It re-enters restore() through the RLock, marks everything clean, returns,
    and the interrupted frame then applies the lock -- GPU modified, disk clean.
    """
    sp = tmp_path / "state.json"

    class SignalMidWrite(FakeGpuController):
        guard = None
        _fired = False

        def set_locked_clocks(self, mn, mx):
            self._require_claim()
            if self.guard is not None and not self._fired:
                self._fired = True
                self.guard._signal_restore(signal.SIGTERM, None)
            self.set_clock_calls += 1
            self.locked = (mn, mx)

    c = SignalMidWrite()
    prev = signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        g = GpuGuard(c, consent=True, require_root=False, state_path=sp)
        c.guard = g
        g.arm()
        g.lock_clocks(1400)
        assert c.locked == (1400, 1400), "precondition: the lock was applied"
        assert g._restored is False, "guard disarmed while the GPU is modified"
        assert record(sp)["restored"] is False, "disk claims clean, GPU is locked"
        assert check_stale_state(FakeGpuController(), sp)["stale"] is True
    finally:
        signal.signal(signal.SIGTERM, prev)


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


CONCURRENT_CHILD = textwrap.dedent(
    """
    import sys, time
    sys.path.insert(0, {repo!r})
    from joule_agent.gpu_guard import GpuGuard, FakeGpuController

    state_path, pid_tag, start_at = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    c = FakeGpuController()
    g = GpuGuard(c, consent=True, require_root=False, state_path=state_path)
    g.arm()
    g.state.pid = pid_tag
    # Busy-wait to a common wall-clock instant so every process performs its
    # read-modify-write in the same window. ONE write each: repeated writes
    # would self-heal a lost update and make this test unable to fail.
    while time.time() < start_at:
        pass
    g._mark_touched()
    """
)


def test_concurrent_processes_do_not_delete_each_others_records(tmp_path):
    """V3/N3: the state document is read, merged and rewritten wholesale.

    Without an inter-process lock, two writers each read the same document and
    the later one silently drops the earlier one's record. A dropped record
    with restored:false is a GPU whose power cap never gets restored, since
    recovery only rewrites power when a record says this repo changed it.

    Every process writes exactly once, released together, so a lost update
    cannot be repaired by a subsequent write.
    """
    import time as _time

    repo = str(Path(__file__).resolve().parent.parent)
    script = tmp_path / "conc.py"
    script.write_text(CONCURRENT_CHILD.format(repo=repo))
    sp = tmp_path / "state.json"

    n = 8
    start_at = _time.time() + 3.0     # give every child time to reach the spin
    procs = [
        subprocess.Popen(
            [sys.executable, str(script), str(sp), str(4000 + i), str(start_at)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        for i in range(n)
    ]
    errs = []
    for pr in procs:
        _, e = pr.communicate(timeout=120)
        if pr.returncode != 0:
            errs.append(e[-300:])
    assert not errs, f"concurrent writers crashed: {errs[:2]}"

    doc = json.loads(sp.read_text())
    assert doc.get("schema") == 2
    kept = sorted(int(k) for k in doc["guards"])
    assert kept == [4000 + i for i in range(n)], (
        f"records lost to concurrent writers: kept {kept}"
    )


def test_recovery_does_not_clear_a_live_writers_record(tmp_path):
    """N9: clearing a running process's record erases live evidence."""
    sp = tmp_path / "state.json"
    sp.write_text(json.dumps({
        "schema": 2,
        "guards": {
            str(os.getppid()): {"pid": os.getppid(), "restored": False,
                                "clocks_locked": True, "snapshot": {}},
            "999999": {"pid": 999999, "restored": False,
                       "clocks_locked": True, "snapshot": {}},
        },
    }))
    c = FakeGpuController()
    res = restore_from_state_file(c, sp)
    doc = json.loads(sp.read_text())
    assert doc["guards"]["999999"]["restored"] is True, "dead record not cleared"
    assert doc["guards"][str(os.getppid())]["restored"] is False, \
        "cleared a record belonging to a live process"
    assert os.getppid() in res.get("skipped_live_pids", [])


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


def test_restore_racing_a_write_on_another_thread_does_not_report_clean(tmp_path):
    """V1: the unlocked signal path can overlap a write on a worker thread.

    The write applies its setting AFTER the restore's reset and BEFORE the
    restore's marking, so an exit-side check alone sees depth back at 0 and an
    unchanged generation. The guard must capture in-flight state at entry.
    """
    sp = tmp_path / "state.json"
    hw_entered = threading.Event()
    may_finish = threading.Event()
    finally_ran = threading.Event()

    class Interleave(FakeGpuController):
        def set_locked_clocks(self, mn, mx):
            self._require_claim()
            hw_entered.set()
            may_finish.wait(10)
            self.set_clock_calls += 1
            self.locked = (mn, mx)          # applied AFTER the reset below

        def reset_locked_clocks(self):
            self._require_claim()
            self.reset_calls += 1
            self.locked = None
            may_finish.set()                # let the writer land now
            finally_ran.wait(10)            # ...before we mark anything

    c = Interleave()
    g = GpuGuard(c, consent=True, require_root=False, state_path=sp)
    g.arm()

    def worker():
        g.lock_clocks(1400)
        finally_ran.set()

    w = threading.Thread(target=worker)
    w.start()
    assert hw_entered.wait(10)
    g._restore_impl()                        # signal path, unlocked
    w.join(15)

    assert c.locked == (1400, 1400), "precondition: the write landed"
    assert g._restored is False, "guard disarmed while the GPU is modified"
    assert record(sp)["restored"] is False, "disk reports clean, GPU is locked"
    assert check_stale_state(FakeGpuController(), sp)["stale"] is True


def test_write_depth_is_actually_read(tmp_path):
    """V2: the counter was incremented and decremented but never consulted."""
    import inspect
    from joule_agent import gpu_guard as gg

    src = inspect.getsource(gg.GpuGuard._restore_impl)
    assert "_write_depth" in src, "_restore_impl ignores the in-flight counter"
    assert "_write_gen" in src, "_restore_impl ignores the write generation"


def test_signal_path_does_not_block_behind_a_held_lock(tmp_path, monkeypatch):
    """N4: restore() re-acquiring the same RLock made the bounded acquire moot.

    os.kill is stubbed because _signal_restore re-raises the signal on the way
    out, which would terminate the test runner itself.
    """
    import joule_agent.gpu_guard as gg

    monkeypatch.setattr(gg.os, "kill", lambda pid, sig: None)
    c = FakeGpuController()
    g = GpuGuard(c, consent=True, require_root=False, state_path=tmp_path / "s.json")
    g.arm()
    g.lock_clocks(1400)

    holder_in = threading.Event()
    release = threading.Event()

    def holder():
        with g._lock:
            holder_in.set()
            release.wait(30)

    th = threading.Thread(target=holder)
    th.start()
    assert holder_in.wait(10)
    try:
        t0 = time.monotonic()
        g._signal_restore(signal.SIGTERM, None)
        elapsed = time.monotonic() - t0
    finally:
        release.set()
        th.join(10)

    # Bounded acquire is 2s; blocking on restore() would run to the 30s holder.
    assert elapsed < 10, f"signal handler blocked {elapsed:.1f}s behind the lock"
    assert c.locked is None, "restore did not run"


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


def test_state_tmp_name_is_unique_per_thread(tmp_path):
    """V4: keying the tmp name on pid alone lets two threads collide."""
    import joule_agent.gpu_guard as gg

    names = []
    real_open = open

    def spy_open(path, *a, **kw):
        if str(path).endswith(".tmp"):
            names.append(str(path))
        return real_open(path, *a, **kw)

    sp = tmp_path / "state.json"
    monkey = gg.__dict__
    monkey["open"] = spy_open
    try:
        def w(tag):
            c = FakeGpuController()
            g = GpuGuard(c, consent=True, require_root=False, state_path=sp)
            g.arm()
            g.state.pid = tag
            g._mark_touched()

        ths = [threading.Thread(target=w, args=(3000 + i,)) for i in range(4)]
        for th in ths:
            th.start()
        for th in ths:
            th.join(20)
    finally:
        monkey.pop("open", None)

    assert len(names) == len(set(names)), f"tmp name collision across threads: {names}"


def test_main_does_not_close_nvml_while_state_is_unrestored(tmp_path):
    """N6: closing in `finally` before atexit turns a possible last-chance
    restore into a guaranteed failure against a dead NVML handle."""
    import joule_agent.gpu_guard as gg
    import inspect

    src = inspect.getsource(gg.main)
    assert "_active_guard" in src
    assert "c.close()" in src
    # the close must be conditional, not unconditional in `finally`
    assert "else:\n            c.close()" in src, "close is unconditional again"


def test_nvml_controller_claim_is_exclusive():
    """The exclusive-claim test previously covered only the fake."""
    from joule_agent.gpu_guard import NvmlGpuController

    c = object.__new__(NvmlGpuController)
    c._claimed_by = None
    c._closed = False
    c.claim("owner-a")
    c.claim("owner-a")                      # same owner is fine
    with pytest.raises(GpuGuardError, match="already claimed"):
        c.claim("owner-b")
