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
