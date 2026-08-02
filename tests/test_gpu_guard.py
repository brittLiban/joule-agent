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
        mid = json.loads(sp.read_text())
        assert mid["restored"] is False
        assert mid["clocks_locked"] is True
        assert mid["locked_min_mhz"] == 1400
    after = json.loads(sp.read_text())
    assert after["restored"] is True
    assert after["clocks_locked"] is False


def test_state_file_documents_that_clock_lock_is_asserted(tmp_path):
    """The device cannot be queried; the file must not imply otherwise."""
    sp = tmp_path / "state.json"
    with guard(tmp_path) as g:
        g.lock_clocks(1400)
    note = json.loads(sp.read_text())["_note"]
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
    sp = tmp_path / "state.json"
    sp.write_text(json.dumps({
        "pid": 999999, "restored": False, "clocks_locked": True,
        "snapshot": {"power_default_limit_mw": 200_000},
    }))
    c = FakeGpuController(power_limit_mw=120_000, power_default_limit_mw=200_000)
    res = restore_from_state_file(c, sp)
    assert res["reset_clocks"] is True
    assert res["power_restored_to_mw"] == 200_000
    assert c.get_power_limit_mw() == 200_000
    assert json.loads(sp.read_text())["restored"] is True


def test_recovery_works_with_no_state_file_at_all(tmp_path):
    c = FakeGpuController(power_limit_mw=120_000, power_default_limit_mw=200_000)
    res = restore_from_state_file(c, tmp_path / "missing.json")
    assert res["reset_clocks"] is True
    assert c.get_power_limit_mw() == 200_000


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
    assert json.loads(state.read_text())["restored"] is True


def test_sigkill_leaves_recoverable_state_file(tmp_path):
    """SIGKILL cannot be caught. The state file is the whole contingency."""
    proc, state, marker = _spawn(tmp_path, "sigkill")
    proc.wait(timeout=30)
    assert proc.returncode == -signal.SIGKILL

    # No handler could have run.
    assert not marker.exists(), "restore ran despite SIGKILL?"
    # But the file records the un-restored lock, written before the hardware write.
    recorded = json.loads(state.read_text())
    assert recorded["restored"] is False
    assert recorded["clocks_locked"] is True

    # And the recovery path acts on it.
    rep = check_stale_state(FakeGpuController(), state)
    assert rep["stale"] is True
    assert rep["writer_pid_alive"] is False
    c = FakeGpuController()
    res = restore_from_state_file(c, state)
    assert res["reset_clocks"] is True
    assert json.loads(state.read_text())["restored"] is True


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
