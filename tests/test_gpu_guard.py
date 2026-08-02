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

import errno
import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import joule_agent.gpu_guard as gpu_guard_module

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


@pytest.fixture(autouse=True)
def isolated_machine_lease(tmp_path, monkeypatch):
    """Keep the production lease machine-wide while isolating pytest cases."""
    monkeypatch.setattr(
        gpu_guard_module, "MACHINE_LEASE_PATH", tmp_path
    )
    monkeypatch.setattr(
        gpu_guard_module, "MACHINE_STATE_PATH", tmp_path / "machine-state.json"
    )


def schema3_record(**over) -> dict:
    """A complete record exactly matching the current writer's authority."""
    rec = {
        "pid": 999999,
        "started_utc": "2026-01-01T00:00:00+00:00",
        "device_index": 0,
        "device_uuid": "GPU-FAKE",
        "snapshot": {
            "index": 0,
            "name": "FakeGPU",
            "uuid": "GPU-FAKE",
            "power_limit_mw": 150_000,
            "power_default_limit_mw": 200_000,
            "power_min_mw": 100_000,
            "power_max_mw": 200_000,
            "max_sm_clock_mhz": 2100,
            "sm_clock_at_snapshot_mhz": 900,
            "default_applications_sm_clock_mhz": None,
        },
        "clocks_locked": True,
        "locked_min_mhz": 900,
        "locked_max_mhz": 900,
        "power_limit_written_mw": None,
        "restored": False,
        "restored_utc": None,
        "restore_failed": False,
        "restore_errors": [],
    }
    rec.update(over)
    snapshot = {
        "index": rec.get("device_index"),
        "name": "FakeGPU",
        "uuid": rec.get("device_uuid"),
        "power_limit_mw": 150_000,
        "power_default_limit_mw": 200_000,
        "power_min_mw": 100_000,
        "power_max_mw": 200_000,
        "max_sm_clock_mhz": 2100,
        "sm_clock_at_snapshot_mhz": 900,
        "default_applications_sm_clock_mhz": None,
    }
    if "snapshot" in over:
        snapshot.update(over["snapshot"] or {})
    rec["snapshot"] = snapshot
    if rec["restored"]:
        if "clocks_locked" not in over:
            rec["clocks_locked"] = False
        if "restore_failed" not in over:
            rec["restore_failed"] = False
        if "restore_errors" not in over:
            rec["restore_errors"] = []
        if "restored_utc" not in over:
            rec["restored_utc"] = "2026-01-01T00:01:00+00:00"
    return rec


def simulate_guard_crash(guard_object) -> None:
    """Drop only process ownership while preserving modified fake hardware."""
    guard_object._release_lifetime_lease()
    guard_object._forked_from_pid = os.getpid()  # suppress inherited atexit work
    for path in (guard_object._state_path, guard_object._machine_state_path):
        doc = json.loads(Path(path).read_text())
        doc["record"]["pid"] = 999999
        Path(path).write_text(json.dumps(doc))


def record(state_path, pid=None) -> dict:
    """Read the guard record. Schema 3 stores exactly one."""
    doc = json.loads(Path(state_path).read_text())
    rec = doc["record"]
    if pid is not None:
        assert rec["pid"] == pid, f"record is for pid {rec['pid']}, wanted {pid}"
    return rec


def preserved_blocker_record(state_path) -> dict:
    """Decode the exact active schema-3 record wrapped by a safety blocker."""
    blocker = json.loads(Path(state_path).read_text())
    assert blocker["_joule_machine_blocker"] == 1
    payload = bytes.fromhex(blocker["preserved_evidence_hex"])
    return json.loads(payload)["record"]


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


@pytest.mark.parametrize("truthy_non_boolean", [1, "true", "false", object()])
def test_truthy_non_boolean_is_not_explicit_consent(
    tmp_path, truthy_non_boolean
):
    c = FakeGpuController()
    g = GpuGuard(
        c,
        consent=truthy_non_boolean,
        require_root=False,
        state_path=tmp_path / "s.json",
    )
    try:
        with pytest.raises(ConsentError):
            g.lock_clocks(1400)
        assert c.set_clock_calls == 0
    finally:
        g.restore()


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


@pytest.mark.parametrize("mhz", [1400.5, float("nan"), True, "1400"])
def test_clock_bounds_must_be_integer_evidence_values(tmp_path, mhz):
    c = FakeGpuController()
    g = guard(tmp_path, controller=c)
    try:
        with pytest.raises(ClockFloorError, match="integer MHz"):
            g.lock_clocks(mhz)
        assert c.set_clock_calls == 0
        assert not (tmp_path / "machine-state.json").exists()
    finally:
        g.restore()


def test_constructor_cannot_lower_the_hard_clock_floor(tmp_path):
    with pytest.raises(ClockFloorError, match="may not lower"):
        GpuGuard(
            FakeGpuController(),
            consent=True,
            require_root=False,
            state_path=tmp_path / "s.json",
            min_clock_mhz=0,
        )


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
    machine = tmp_path / "machine-state.json"
    order = []
    c = FakeGpuController()
    real_set = c.set_locked_clocks

    def spy(mn, mx):
        order.append(
            (
                "hardware",
                sp.exists(),
                machine.exists(),
                preserved_blocker_record(machine)["restored"]
                if machine.exists()
                else None,
            )
        )
        real_set(mn, mx)

    c.set_locked_clocks = spy
    with guard(tmp_path, controller=c) as g:
        g.lock_clocks(1400)
    assert order == [("hardware", True, True, False)], (
        "both project and machine evidence must be unrestored before the write"
    )


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


def test_clean_publication_commits_machine_record_last(tmp_path, monkeypatch):
    sp = tmp_path / "state.json"
    machine = tmp_path / "machine-state.json"
    calls = []
    real_write = gpu_guard_module._write_state

    def observe(path, state):
        calls.append((Path(path), state.restored))
        return real_write(path, state)

    monkeypatch.setattr(gpu_guard_module, "_write_state", observe)
    c = FakeGpuController()
    g = GpuGuard(c, consent=True, require_root=False, state_path=sp)
    g.lock_clocks(1400)
    calls.clear()

    g.restore()

    assert calls == [(sp.resolve(), True), (machine, True)]
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


def test_permission_denied_while_probing_a_pid_means_it_is_alive(monkeypatch):
    """An unprivileged status process cannot signal a root-owned guard."""
    import joule_agent.gpu_guard as gg

    def denied(pid, sig):
        raise PermissionError(errno.EPERM, "operation not permitted")

    monkeypatch.setattr(gg.os, "kill", denied)
    assert gg._pid_alive(4242) is True


def test_restore_from_state_file_is_unconditional(tmp_path):
    """Clock reset always fires; power follows only a record of OUR write."""
    sp = tmp_path / "state.json"
    legacy = schema3_record(
        power_limit_written_mw=120_000,
        snapshot={
            "index": 0,
            "uuid": "GPU-FAKE",
            "power_limit_mw": 200_000,
            "power_default_limit_mw": 200_000,
        },
    )
    sp.write_text(json.dumps({
        "schema": 2,
        "guards": {"999999": legacy},
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


def test_clean_record_cannot_overwrite_a_later_operator_power_cap(tmp_path):
    """A completed run is historical evidence, not recovery authority.

    The clean record deliberately retains what the guard wrote and the stock
    snapshot for auditability.  A later ``restore`` command must not replay
    that old snapshot over a power cap an operator applied after the run.
    """
    sp = tmp_path / "state.json"
    original = FakeGpuController(
        power_limit_mw=175_000,
        power_default_limit_mw=200_000,
        uuid="GPU-SAME",
    )
    with GpuGuard(original, consent=True, require_root=False, state_path=sp) as g:
        g.set_power_limit_mw(120_000)

    assert original.get_power_limit_mw() == 175_000
    assert record(sp)["restored"] is True
    assert record(sp)["power_limit_written_mw"] == 120_000

    later = FakeGpuController(
        power_limit_mw=150_000,
        power_default_limit_mw=200_000,
        uuid="GPU-SAME",
    )
    res = restore_from_state_file(later, sp)

    assert res["reset_clocks"] is True
    assert res["power_restored_to_mw"] is None
    assert later.set_power_calls == 0
    assert later.get_power_limit_mw() == 150_000, (
        "historical state erased an operator cap"
    )


def test_clean_record_is_not_republished_by_recovery(tmp_path, monkeypatch):
    """Historical state must not race a new guard's first intent write.

    A read/compare followed by ``os.replace`` is not a compare-and-swap: a new
    guard can publish an unrestored record between those operations. Recovery
    has nothing to clear in a clean record, so it must leave the file untouched.
    """
    import joule_agent.gpu_guard as gg

    sp = tmp_path / "state.json"
    original = FakeGpuController(uuid="GPU-HISTORY")
    with GpuGuard(original, consent=True, require_root=False, state_path=sp) as g:
        g.set_power_limit_mw(120_000)

    before = sp.read_bytes()
    replacements = []
    real_replace = gg.os.replace

    def observe_replace(source, destination):
        replacements.append((Path(source), Path(destination)))
        return real_replace(source, destination)

    monkeypatch.setattr(gg.os, "replace", observe_replace)
    res = restore_from_state_file(FakeGpuController(uuid="GPU-HISTORY"), sp)

    assert res["errors"] == []
    assert replacements == [], "recovery republished a record already marked clean"
    assert sp.read_bytes() == before


def test_clean_record_cannot_redirect_recovery_for_a_later_cap(tmp_path):
    """A historical GPU-1 record is not authority over a GPU-0 cap."""
    sp = tmp_path / "state.json"
    _write_record(
        sp,
        restored=True,
        device_index=1,
        device_uuid="GPU-OLD",
        clocks_locked=False,
    )
    current = FakeGpuController(
        index=0,
        uuid="GPU-CURRENT",
        power_limit_mw=150_000,
        power_default_limit_mw=200_000,
    )

    report = check_stale_state(current, sp)

    assert report["verified_power_modified"] is True
    assert "--gpu 0" in report["recommendation"]
    assert "--force-power-default" in report["recommendation"]
    assert "--gpu 1" not in report["recommendation"]


def test_force_power_default_is_the_explicit_override(tmp_path):
    c = FakeGpuController(power_limit_mw=120_000, power_default_limit_mw=200_000)
    res = restore_from_state_file(c, tmp_path / "missing.json", force_power_default=True)
    assert res["power_restored_to_mw"] == 200_000
    assert res["power_forced_to_default"] is True
    assert c.get_power_limit_mw() == 200_000


def test_unreadable_default_makes_force_power_default_incomplete(tmp_path):
    c = FakeGpuController(power_limit_mw=120_000, power_default_limit_mw=None)

    res = restore_from_state_file(
        c, tmp_path / "missing.json", force_power_default=True
    )

    assert res["incomplete"] is True
    assert res["errors"]
    assert "power_forced_to_default" not in res
    assert res["power_restored_to_mw"] is None
    assert c.set_power_calls == 0
    assert c.get_power_limit_mw() == 120_000


def test_recovery_does_not_guess_factory_default_for_unknown_prior_cap(tmp_path):
    sp = tmp_path / "state.json"
    _write_record(
        sp,
        device_uuid="GPU-FAKE",
        power_limit_written_mw=110_000,
        snapshot={
            "power_limit_mw": None,
            "power_default_limit_mw": 200_000,
        },
    )
    c = FakeGpuController(power_limit_mw=110_000)

    result = restore_from_state_file(c, sp)

    assert result["incomplete"] is True
    assert result["record_cleared"] is False
    assert result["power_restored_to_mw"] is None
    assert c.set_power_calls == 0
    assert c.get_power_limit_mw() == 110_000
    assert any("--force-power-default" in error for error in result["errors"])

    forced = restore_from_state_file(c, sp, force_power_default=True)
    assert forced["record_cleared"] is True
    assert forced["power_forced_to_default"] is True
    assert c.get_power_limit_mw() == 200_000


def test_clock_only_record_recommends_explicit_reset_for_an_unclaimed_cap(tmp_path):
    sp = tmp_path / "state.json"
    _write_record(
        sp,
        device_uuid="GPU-SAME",
        power_limit_written_mw=None,
    )
    current = FakeGpuController(
        index=0,
        uuid="GPU-SAME",
        power_limit_mw=150_000,
        power_default_limit_mw=200_000,
    )

    report = check_stale_state(current, sp)

    assert report["asserted_clock_lock"] is True
    assert report["verified_power_modified"] is True
    assert "--gpu 0" in report["recommendation"]
    assert "--force-power-default" in report["recommendation"]


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
    from pathlib import Path
    sys.path.insert(0, {repo!r})
    import joule_agent.gpu_guard as gg

    state_path, marker, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    gg.MACHINE_LEASE_PATH = Path(state_path).parent
    gg.MACHINE_STATE_PATH = Path(state_path).parent / "machine-state.json"
    GpuGuard, FakeGpuController = gg.GpuGuard, gg.FakeGpuController

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
    with pytest.raises(GpuGuardError, match="verification failed"):
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
    with pytest.raises(GpuGuardError, match="active GpuGuard"):
        c.set_locked_clocks(100, 100)
    with pytest.raises(GpuGuardError):
        c.set_power_limit_mw(50_000)
    assert c.set_clock_calls == 0


def test_manually_claiming_controller_still_cannot_bypass_guard_policy():
    c = FakeGpuController()
    c.claim(object())

    with pytest.raises(GpuGuardError, match="active GpuGuard"):
        c.set_locked_clocks(1, 1)
    with pytest.raises(GpuGuardError, match="active GpuGuard"):
        c.set_power_limit_mw(1)

    assert c.set_clock_calls == 0
    assert c.set_power_calls == 0


def test_guard_claims_its_controller(tmp_path):
    c = FakeGpuController()
    with guard(tmp_path, controller=c) as g:
        g.lock_clocks(1400)
        assert c.locked == (1400, 1400)


def test_power_limit_of_zero_is_still_restored(tmp_path):
    """F7: 0 mW is falsy; a truthiness gate skipped the restore entirely."""
    c = FakeGpuController(
        power_limit_mw=200_000,
        power_default_limit_mw=200_000,
        power_min_mw=None,
        power_max_mw=None,
    )
    g = guard(tmp_path, controller=c)
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
    with gpu_guard_module._controller_write_access(
        "test", "reset_locked_clocks", ()
    ):
        with pytest.raises(GpuGuardError, match="closed"):
            c._write("reset_locked_clocks", (), lambda: called.append(1))
    assert called == [], "a closed controller still issued the write"


def test_write_through_unclaimed_real_controller_is_refused():
    from joule_agent.gpu_guard import NvmlGpuController

    c = object.__new__(NvmlGpuController)
    c._closed = False
    c._claimed_by = None
    with pytest.raises(GpuGuardError, match="active GpuGuard"):
        c._write("reset_locked_clocks", (), lambda: None)


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
        with pytest.raises(GpuGuardError, match="interrupted by signal"):
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


def test_power_write_is_refused_when_exact_restore_target_is_unknown(tmp_path):
    """A factory default cannot stand in for an unreadable operator cap."""
    c = FakeGpuController(
        power_limit_mw=None,
        power_default_limit_mw=200_000,
    )
    g = guard(tmp_path, controller=c)

    with pytest.raises(GpuGuardError, match="pre-write current limit was unreadable"):
        g.set_power_limit_mw(120_000)

    assert c.set_power_calls == 0
    assert c.get_power_limit_mw() is None
    assert not (tmp_path / "state.json").exists()
    g.restore()


def test_claim_is_exclusive(tmp_path):
    """N5: two guards must not silently share one controller."""
    c = FakeGpuController()
    owner_a = object()
    owner_b = object()
    c.claim(owner_a)
    with pytest.raises(GpuGuardError, match="already claimed"):
        c.claim(owner_b)
    c.claim(owner_a)  # the identical owner token is fine


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
    legacy = schema3_record()
    sp.write_text(json.dumps({
        "schema": 2,
        "guards": {"999999": legacy},
    }))
    seen = []
    real = gg.os.replace
    gg.os.replace = lambda a, b: (seen.append((str(a), str(b))), real(a, b))[1]
    try:
        restore_from_state_file(FakeGpuController(), sp)
    finally:
        gg.os.replace = real
    assert seen, "recovery published the state file without an atomic rename"
    destinations = {destination for _, destination in seen}
    assert str(sp) in destinations
    assert str(gg.MACHINE_STATE_PATH) in destinations


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


def test_nvml_controller_claim_requires_the_identical_owner_token():
    from joule_agent.gpu_guard import NvmlGpuController

    claimed_owner = "owner-a"
    owner = "".join(["own", "er-a"])  # equal value, different object
    assert owner == claimed_owner and owner is not claimed_owner

    c = object.__new__(NvmlGpuController)
    c._claimed_by = None
    c._closed = False
    c.claim(claimed_owner)
    with pytest.raises(GpuGuardError, match="already claimed"):
        c.claim(owner)
    with pytest.raises(GpuGuardError, match="already claimed"):
        c.claim("owner-b")


def test_adversarial_equality_cannot_impersonate_controller_owner():
    class EqualToEverything:
        def __eq__(self, other):
            return True

    c = FakeGpuController()
    c.claim(EqualToEverything())

    with pytest.raises(GpuGuardError, match="direct hardware write"):
        c.set_locked_clocks(1, 1)

    assert c.set_clock_calls == 0



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


def test_worker_restore_cannot_race_the_owner_hardware_write(tmp_path):
    """A worker must not publish clean state while the owner is mid-write."""
    entered_write = threading.Event()
    continue_write = threading.Event()
    caught = []

    class PausedWrite(FakeGpuController):
        def set_locked_clocks(self, min_mhz, max_mhz):
            entered_write.set()
            assert continue_write.wait(10), "restore worker never released write"
            super().set_locked_clocks(min_mhz, max_mhz)

    c = PausedWrite()
    sp = tmp_path / "s.json"
    g = GpuGuard(c, consent=True, require_root=False, state_path=sp)
    g.arm()

    def worker():
        assert entered_write.wait(10), "owner never reached hardware write"
        try:
            g.restore()
        except BaseException as exc:
            caught.append(exc)
        finally:
            continue_write.set()

    thread = threading.Thread(target=worker, name="racing-restore")
    thread.start()
    try:
        g.lock_clocks(1400)
        thread.join(10)
        assert not thread.is_alive()
        assert caught and isinstance(caught[0], ThreadAffinityError)
        assert c.locked == (1400, 1400)
        assert record(sp)["restored"] is False
        assert g._lease_fd is not None
    finally:
        continue_write.set()
        thread.join(10)
        g.restore()

    assert c.locked is None
    assert record(sp)["restored"] is True


@pytest.mark.parametrize("write_kind", ["clock", "power"])
def test_hardware_write_requires_a_live_device_uuid(tmp_path, write_kind):
    c = FakeGpuController(uuid=None)
    sp = tmp_path / "s.json"
    g = GpuGuard(c, consent=True, require_root=False, state_path=sp)

    with pytest.raises(GpuGuardError, match="UUID was unavailable"):
        if write_kind == "clock":
            g.lock_clocks(1400)
        else:
            g.set_power_limit_mw(150_000)

    assert c.set_clock_calls == 0
    assert c.set_power_calls == 0
    assert not sp.exists()
    g.restore()


@pytest.mark.parametrize(
    "controller_kwargs",
    [
        {"index": True},
        {"power_limit_mw": 150_000.5},
        {"power_default_limit_mw": 200_000.5},
    ],
)
def test_prewrite_snapshot_must_be_readable_recovery_evidence(
    tmp_path, controller_kwargs
):
    c = FakeGpuController(**controller_kwargs)
    sp = tmp_path / "s.json"
    g = GpuGuard(c, consent=True, require_root=False, state_path=sp)
    try:
        with pytest.raises(GpuGuardError, match="recovery evidence"):
            g.lock_clocks(900)
        assert c.set_clock_calls == 0
        assert c.locked is None
        assert not sp.exists()
        assert not (tmp_path / "machine-state.json").exists()
    finally:
        g.restore()


@pytest.mark.parametrize("target", [150_000.5, True, "150000"])
def test_power_target_must_be_integer_milliwatts(tmp_path, target):
    c = FakeGpuController()
    g = guard(tmp_path, controller=c)
    try:
        with pytest.raises(GpuGuardError, match="integer number of milliwatts"):
            g.set_power_limit_mw(target)
        assert c.set_power_calls == 0
        assert not (tmp_path / "machine-state.json").exists()
    finally:
        g.restore()


def test_arming_while_a_live_pid_holds_a_record_is_refused(tmp_path):
    """One guard per machine. A live holder means another guard is running."""
    sp = tmp_path / "state.json"
    sp.write_text(json.dumps({
        "schema": 3,
        "record": schema3_record(pid=os.getppid()),
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
        "record": schema3_record(),
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
        "record": schema3_record(restored=True, clocks_locked=False),
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


@pytest.mark.skipif(
    not hasattr(signal, "pthread_kill") or not hasattr(signal, "pthread_sigmask"),
    reason="requires POSIX per-thread signal delivery",
)
def test_worker_delivered_signal_cannot_clean_before_owner_write(tmp_path):
    """Regression: per-thread masking alone left locked hardware + clean state."""
    intent_published = threading.Event()
    worker_ready = threading.Event()
    signal_sent = threading.Event()
    handler_seen = threading.Event()
    worker_errors = []
    sp = tmp_path / "s.json"

    class PauseAfterIntent(GpuGuard):
        def _mark_touched(self):
            super()._mark_touched()
            intent_published.set()
            assert signal_sent.wait(10), "worker did not send SIGTERM"
            deadline = time.time() + 10
            while not handler_seen.is_set() and time.time() < deadline:
                time.sleep(0.001)
            assert handler_seen.is_set(), "Python signal handler never ran"

        def _signal_restore(self, signum, frame):
            handler_seen.set()
            return super()._signal_restore(signum, frame)

    def signal_worker():
        try:
            signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM})
            worker_ready.set()
            assert intent_published.wait(10), "owner never published intent"
            signal.pthread_kill(threading.get_ident(), signal.SIGTERM)
            signal_sent.set()
        except BaseException as exc:
            worker_errors.append(exc)
            signal_sent.set()

    previous = signal.signal(signal.SIGTERM, signal.SIG_IGN)
    c = FakeGpuController()
    g = PauseAfterIntent(c, consent=True, require_root=False, state_path=sp)
    worker = threading.Thread(target=signal_worker, name="unblocked-signal-worker")
    try:
        g.arm()
        worker.start()
        assert worker_ready.wait(10)

        with pytest.raises(GpuGuardError, match="interrupted by signal"):
            g.lock_clocks(1400)

        worker.join(10)
        assert not worker.is_alive()
        assert worker_errors == []
        assert c.locked is None
        assert record(sp)["restored"] is True
        assert g._lease_fd is None

        fresh = GpuGuard(
            FakeGpuController(),
            consent=True,
            require_root=False,
            state_path=tmp_path / "fresh.json",
        )
        fresh.arm()
        fresh.restore()
    finally:
        signal_sent.set()
        worker.join(10)
        g.restore()
        signal.signal(signal.SIGTERM, previous)


def test_latched_signal_reports_failed_restore_as_hardware_maybe_modified(tmp_path):
    sp = tmp_path / "s.json"
    guard_ref = {}
    original = signal.getsignal(signal.SIGTERM)

    class FailingSignalRestore(FakeGpuController):
        fail_reset = True

        def set_locked_clocks(self, min_mhz, max_mhz):
            super().set_locked_clocks(min_mhz, max_mhz)
            guard_ref["guard"]._signal_restore(signal.SIGTERM, None)

        def reset_locked_clocks(self):
            if self.fail_reset:
                self.reset_calls += 1
                raise RuntimeError("signal reset failed")
            super().reset_locked_clocks()

    c = FailingSignalRestore()
    g = GpuGuard(c, consent=True, require_root=False, state_path=sp)
    guard_ref["guard"] = g
    g.arm()
    g._prev_handlers[signal.SIGTERM] = signal.SIG_IGN
    try:
        with pytest.raises(GpuGuardError, match="GPU MAY STILL BE MODIFIED"):
            g.lock_clocks(1400)

        assert c.locked == (1400, 1400)
        assert record(sp)["restored"] is False
        assert g._lease_fd is not None
    finally:
        c.fail_reset = False
        g.restore()
        signal.signal(signal.SIGTERM, original)


def test_all_latched_returning_signals_are_drained_before_interruption(tmp_path):
    sp = tmp_path / "s.json"
    guard_ref = {}
    handled = []
    original = {
        signal.SIGINT: signal.getsignal(signal.SIGINT),
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
    }

    class TwoSignals(FakeGpuController):
        def set_locked_clocks(self, min_mhz, max_mhz):
            super().set_locked_clocks(min_mhz, max_mhz)
            guard_ref["guard"]._signal_restore(signal.SIGTERM, None)
            guard_ref["guard"]._signal_restore(signal.SIGINT, None)

    c = TwoSignals()
    g = GpuGuard(c, consent=True, require_root=False, state_path=sp)
    guard_ref["guard"] = g
    try:
        g.arm()
        g._prev_handlers[signal.SIGTERM] = lambda *args: handled.append("TERM")
        g._prev_handlers[signal.SIGINT] = lambda *args: handled.append("INT")

        with pytest.raises(GpuGuardError, match="multiple returning signals"):
            g.lock_clocks(1400)

        assert handled == ["TERM", "INT"]
        assert g._pending_signals == []
        assert c.locked is None
        assert record(sp)["restored"] is True
    finally:
        g.restore()
        for sig, handler in original.items():
            signal.signal(sig, handler)


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


def test_a_second_guard_is_refused_before_the_first_hardware_write(tmp_path):
    """Arming itself must reserve the single-instance slot atomically.

    Previously the state file appeared only at first hardware touch, so two
    guards could both pass the busy check, snapshot, and install restore paths
    before either one left evidence on disk.
    """
    sp = tmp_path / "state.json"
    g1 = GpuGuard(
        FakeGpuController(), consent=True, require_root=False, state_path=sp
    )
    g2 = GpuGuard(
        FakeGpuController(), consent=True, require_root=False, state_path=sp
    )

    g1.arm()
    try:
        with pytest.raises(GuardBusyError, match="another GpuGuard"):
            g2.arm()
        assert g2.snapshot is None
    finally:
        g1.restore()

    # A no-touch restore owes no hardware recovery and must release ownership.
    g2.arm()
    try:
        # The completed first object must not bypass the released lease by
        # reusing its old snapshot while a fresh guard owns the slot.
        with pytest.raises(GpuGuardError, match="one-shot"):
            g1.lock_clocks(1400)
    finally:
        g2.restore()


def test_machine_lease_cannot_be_bypassed_with_a_different_state_path(tmp_path):
    """Caller-selected evidence paths must not create hardware namespaces."""
    first = GpuGuard(
        FakeGpuController(),
        consent=True,
        require_root=False,
        state_path=tmp_path / "first.json",
    )
    second = GpuGuard(
        FakeGpuController(),
        consent=True,
        require_root=False,
        state_path=tmp_path / "second.json",
    )

    first.arm()
    try:
        with pytest.raises(GuardBusyError, match="machine.lease|another GpuGuard"):
            second.arm()
        assert second.snapshot is None
    finally:
        first.restore()


def test_machine_evidence_blocks_another_path_after_lease_owner_dies(tmp_path):
    """A released flock cannot hide persisted recovery debt behind a new path."""
    machine = tmp_path / "machine-state.json"
    machine.write_text(
        json.dumps(
            {
                "schema": 3,
                "record": schema3_record(
                    evidence_path=str(tmp_path / "first.json"),
                    power_limit_written_mw=120_000,
                ),
            }
        )
    )
    second = GpuGuard(
        FakeGpuController(),
        consent=True,
        require_root=False,
        state_path=tmp_path / "second.json",
    )

    with pytest.raises(GuardBusyError, match="died without restoring"):
        second.arm()

    assert second.snapshot is None


def test_recovery_finds_machine_debt_through_a_different_requested_path(tmp_path):
    original = tmp_path / "first.json"
    unrelated = tmp_path / "second.json"
    machine = tmp_path / "machine-state.json"
    debt = schema3_record(
        evidence_path=str(original),
        device_uuid="GPU-FAKE",
        power_limit_written_mw=120_000,
    )
    document = json.dumps({"schema": 3, "record": debt})
    original.write_text(document)
    machine.write_text(document)
    c = FakeGpuController(power_limit_mw=120_000)

    result = restore_from_state_file(c, unrelated)

    assert result["errors"] == []
    assert result["record_cleared"] is True
    assert result["power_restored_to_mw"] == 150_000
    assert record(original)["restored"] is True
    assert record(machine)["restored"] is True
    assert not unrelated.exists(), "recovery fabricated an unrelated mirror"


def test_guard_canonicalizes_its_state_path_before_cwd_can_change(
    tmp_path, monkeypatch
):
    """The lease and state record must never drift into different directories."""
    original_cwd = tmp_path / "original"
    later_cwd = tmp_path / "later"
    original_cwd.mkdir()
    later_cwd.mkdir()
    monkeypatch.chdir(original_cwd)
    g = GpuGuard(
        FakeGpuController(),
        consent=True,
        require_root=False,
        state_path="state.json",
    )

    monkeypatch.chdir(later_cwd)
    with g:
        g.lock_clocks(1400)

    assert (original_cwd / "state.json").exists()
    assert not (later_cwd / "state.json").exists()


def test_signal_during_arm_cannot_return_success_without_the_lease(tmp_path):
    """A handled/ignored signal may complete a no-touch restore during arm."""
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    class SignalDuringInstall(GpuGuard):
        def _install_handlers(self):
            super()._install_handlers()
            self._signal_restore(signal.SIGTERM, None)

    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    sp = tmp_path / "state.json"
    interrupted = SignalDuringInstall(
        FakeGpuController(), consent=True, require_root=False, state_path=sp
    )
    try:
        with pytest.raises(GpuGuardError, match="arming was interrupted"):
            interrupted.arm()
        assert interrupted._lease_fd is None

        fresh = GpuGuard(
            FakeGpuController(), consent=True, require_root=False, state_path=sp
        )
        fresh.arm()
        fresh.restore()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


@pytest.mark.parametrize("write_kind", ["clock", "power"])
def test_returning_signal_before_write_cannot_bypass_a_released_lease(
    tmp_path, write_kind
):
    """Ownership is rechecked only after signals become blocked for the write."""
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    sp = tmp_path / "state.json"
    second = None

    class ReleaseDuringConsent(GpuGuard):
        def _check_consent(self):
            nonlocal second
            super()._check_consent()
            self._signal_restore(signal.SIGTERM, None)
            second = GpuGuard(
                FakeGpuController(),
                consent=True,
                require_root=False,
                state_path=sp,
            )
            second.arm()

    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    controller = FakeGpuController()
    first = ReleaseDuringConsent(
        controller, consent=True, require_root=False, state_path=sp
    )
    first.arm()
    try:
        with pytest.raises(GpuGuardError, match="one-shot"):
            if write_kind == "clock":
                first.lock_clocks(1400)
            else:
                first.set_power_limit_mw(150_000)
        assert controller.set_clock_calls == 0
        assert controller.set_power_calls == 0
        assert not sp.exists(), "a lease-free guard published hardware intent"
    finally:
        if second is not None:
            second.restore()
        first.restore()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_fork_child_cannot_use_or_retain_the_parent_guards_lease(tmp_path):
    """A copied guard is not ownership, and an idle child must not wedge it."""
    sp = tmp_path / "state.json"
    result = tmp_path / "child-result"
    release = tmp_path / "release-child"
    parent_guard = GpuGuard(
        FakeGpuController(), consent=True, require_root=False, state_path=sp
    )
    parent_guard.arm()

    child_pid = os.fork()
    if child_pid == 0:  # pragma: no cover - assertions execute in the parent
        try:
            try:
                parent_guard.arm()
            except GpuGuardError as exc:
                outcome = "REFUSED" if "fork" in str(exc).lower() else f"ERROR:{exc}"
            else:
                outcome = "ALLOWED"
            result.write_text(outcome)
            deadline = time.time() + 10
            while not release.exists() and time.time() < deadline:
                time.sleep(0.005)
        finally:
            os._exit(0)

    fresh = None
    try:
        deadline = time.time() + 10
        while not result.exists() and time.time() < deadline:
            time.sleep(0.01)
        assert result.exists(), "fork child did not report"

        parent_guard.restore()
        fresh = GpuGuard(
            FakeGpuController(), consent=True, require_root=False, state_path=sp
        )
        fresh.arm()

        assert result.read_text() == "REFUSED"
    finally:
        if fresh is not None:
            fresh.restore()
        parent_guard.restore()
        release.write_text("release")
        os.waitpid(child_pid, 0)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
@pytest.mark.filterwarnings(
    r"ignore:This process .* is multi-threaded, use of fork\(\) may lead to "
    r"deadlocks in the child.:DeprecationWarning"
)
def test_fork_during_lease_open_cannot_inherit_an_untracked_fd(
    tmp_path, monkeypatch
):
    """The open/flock/publish handoff is atomic with respect to another thread."""
    import joule_agent.gpu_guard as gg

    sp = tmp_path / "state.json"
    child_ready = tmp_path / "fork-child-ready"
    release_child = tmp_path / "release-fork-child"
    opened = threading.Event()
    fork_attempted = threading.Event()
    fork_finished = threading.Event()
    child_pids = []
    worker_errors = []
    real_open = gg.os.open
    pause_once = True

    def paused_open(*args, **kwargs):
        nonlocal pause_once
        fd = real_open(*args, **kwargs)
        if pause_once:
            pause_once = False
            opened.set()
            assert fork_attempted.wait(10), "fork worker never attempted fork"
            # Without the at-fork gate, this is ample time for the child to be
            # created while the fd exists only as this function's local value.
            time.sleep(0.1)
        return fd

    def fork_worker():
        try:
            assert opened.wait(10), "lease fd was never opened"
            fork_attempted.set()
            pid = os.fork()
            if pid == 0:  # pragma: no cover - assertions execute in parent
                try:
                    child_ready.write_text("ready")
                    deadline = time.time() + 10
                    while not release_child.exists() and time.time() < deadline:
                        time.sleep(0.005)
                finally:
                    os._exit(0)
            child_pids.append(pid)
        except BaseException as exc:
            worker_errors.append(exc)
        finally:
            fork_finished.set()

    guard = GpuGuard(
        FakeGpuController(), consent=True, require_root=False, state_path=sp
    )
    worker = threading.Thread(target=fork_worker, name="fork-during-arm")
    monkeypatch.setattr(gg.os, "open", paused_open)
    worker.start()
    fresh = None
    try:
        guard.arm()
        monkeypatch.setattr(gg.os, "open", real_open)
        assert fork_finished.wait(10), "fork did not finish after lease publication"
        worker.join(timeout=10)
        assert not worker.is_alive()
        assert worker_errors == []
        assert child_pids

        deadline = time.time() + 10
        while not child_ready.exists() and time.time() < deadline:
            time.sleep(0.01)
        assert child_ready.exists()

        guard.restore()
        fresh = GpuGuard(
            FakeGpuController(), consent=True, require_root=False, state_path=sp
        )
        fresh.arm()
    finally:
        monkeypatch.setattr(gg.os, "open", real_open)
        if fresh is not None:
            fresh.restore()
        guard.restore()
        release_child.write_text("release")
        worker.join(timeout=10)
        for pid in child_pids:
            os.waitpid(pid, 0)


RACING_ARM_CHILD = textwrap.dedent(
    """
    import os, sys, time
    from pathlib import Path
    sys.path.insert(0, {repo!r})
    import joule_agent.gpu_guard as gg

    state, barrier, result, release = map(Path, sys.argv[1:])
    gg.MACHINE_LEASE_PATH = state.parent
    gg.MACHINE_STATE_PATH = state.parent / "machine-state.json"
    FakeGpuController = gg.FakeGpuController
    GpuGuard = gg.GpuGuard
    GuardBusyError = gg.GuardBusyError

    class BarrierGuard(GpuGuard):
        def _check_not_busy(self):
            # Complete the old check before synchronising. Without an atomic
            # reservation ahead of this method, both processes observe a clean
            # state file and both proceed to snapshot.
            super()._check_not_busy()
            (barrier / f"ready-{{os.getpid()}}").write_text("ready")
            deadline = time.time() + 0.5
            while time.time() < deadline:
                if len(list(barrier.glob("ready-*"))) >= 2:
                    break
                time.sleep(0.005)

    guard = BarrierGuard(
        FakeGpuController(), consent=True, require_root=False, state_path=state
    )
    armed = False
    try:
        guard.arm()
        armed = True
        result.write_text("ARMED")
        deadline = time.time() + 10
        while not release.exists() and time.time() < deadline:
            time.sleep(0.005)
    except GuardBusyError:
        result.write_text("BUSY")
    except BaseException as exc:
        result.write_text(f"ERROR:{{type(exc).__name__}}: {{exc}}")
    finally:
        if armed:
            guard.restore()
    """
)


def test_arm_reservation_is_atomic_across_processes(tmp_path):
    """Exactly one process may cross the clean-state check and arm."""
    repo = str(Path(__file__).resolve().parent.parent)
    script = tmp_path / "racing_arm.py"
    script.write_text(RACING_ARM_CHILD.format(repo=repo))
    state = tmp_path / "state.json"
    barrier = tmp_path / "barrier"
    barrier.mkdir()
    release = tmp_path / "release"
    results = [tmp_path / f"result-{i}" for i in range(2)]
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                str(script),
                str(state),
                str(barrier),
                str(result),
                str(release),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for result in results
    ]

    try:
        deadline = time.time() + 10
        outcomes = []
        while time.time() < deadline:
            outcomes = [path.read_text() if path.exists() else "" for path in results]
            if all(outcomes):
                break
            time.sleep(0.01)
        assert all(outcomes), "arm racers did not report"
        assert sorted(outcomes) == ["ARMED", "BUSY"]
    finally:
        release.write_text("release")
        for proc in processes:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
    for proc in processes:
        assert proc.returncode == 0, proc.stderr.read()


def test_sigkill_before_hardware_touch_releases_the_arm_lease(tmp_path):
    """Kernel fd cleanup must not create recovery debt where none exists."""
    repo = str(Path(__file__).resolve().parent.parent)
    state = tmp_path / "state.json"
    child = textwrap.dedent(
        f"""
        import sys, time
        from pathlib import Path
        sys.path.insert(0, {repo!r})
        import joule_agent.gpu_guard as gg
        state = Path(sys.argv[1])
        gg.MACHINE_LEASE_PATH = state.parent
        gg.MACHINE_STATE_PATH = state.parent / "machine-state.json"
        FakeGpuController, GpuGuard = gg.FakeGpuController, gg.GpuGuard
        guard = GpuGuard(
            FakeGpuController(), consent=True, require_root=False,
            state_path=state
        )
        guard.arm()
        print("ARMED", flush=True)
        time.sleep(30)
        """
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", child, str(state)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout.readline().strip() == "ARMED"
        proc.kill()
        proc.wait(timeout=10)
        assert proc.returncode == -signal.SIGKILL

        fresh = GpuGuard(
            FakeGpuController(), consent=True, require_root=False, state_path=state
        )
        fresh.arm()
        fresh.restore()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)


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
        "record": schema3_record(pid=os.getppid()),
    }))
    c = FakeGpuController()
    res = restore_from_state_file(c, sp)
    assert res["reset_clocks"] is False, "a live holder must remain untouched"
    assert res["record_cleared"] is False
    assert "still running" in res["record_not_cleared_reason"]
    assert record(sp)["restored"] is False, "a live guard's evidence was erased"


def test_recovery_cannot_race_a_live_guard_holding_the_machine_lease(tmp_path):
    sp = tmp_path / "state.json"
    live = GpuGuard(
        FakeGpuController(), consent=True, require_root=False, state_path=sp
    )
    live.arm()
    recovery_controller = FakeGpuController()
    try:
        result = restore_from_state_file(recovery_controller, tmp_path / "other.json")
        assert result["incomplete"] is True
        assert any("machine lease" in error for error in result["errors"])
        assert result["reset_clocks"] is False
        assert recovery_controller.reset_calls == 0
    finally:
        live.restore()


def test_recovery_refuses_to_clear_a_live_guard_in_the_same_process(tmp_path):
    """A second controller is not allowed to exempt its own process ID."""
    sp = tmp_path / "state.json"
    _write_record(sp, pid=os.getpid())

    res = restore_from_state_file(FakeGpuController(), sp)

    assert res["reset_clocks"] is False
    assert res["record_cleared"] is False
    assert res["incomplete"] is True
    assert "still running" in res["record_not_cleared_reason"]
    assert record(sp)["restored"] is False


def test_recovery_refuses_to_publish_a_stale_read(tmp_path):
    """If a guard writes while recovery is mid-flight, recovery must not
    publish the record it read before those hardware calls."""
    sp = tmp_path / "state.json"
    sp.write_text(json.dumps({
        "schema": 3,
        "record": schema3_record(),
    }))

    class MutateDuringReset(FakeGpuController):
        def reset_locked_clocks(self):
            self._require_claim()
            self.reset_calls += 1
            self.locked = None
            # A guard records new intent while we are resetting.
            sp.write_text(json.dumps({
                "schema": 3,
                "record": schema3_record(
                    locked_min_mhz=1400, locked_max_mhz=1400
                ),
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
    rec = schema3_record(**over)
    Path(path).write_text(json.dumps({"schema": 3, "record": rec}))
    return rec


def test_recovery_refuses_a_record_for_another_device(tmp_path):
    """The severe one. A record for GPU 1 must not be applied to GPU 0.

    On a homogeneous node the other device's power value is inside this one's
    constraints, so NVML accepts it silently: GPU 0 gets mis-capped, GPU 1's
    record is cleared, and GPU 1 stays locked with its evidence destroyed.
    """
    sp = tmp_path / "state.json"
    _write_record(
        sp,
        device_index=1,
        device_uuid="GPU-OTHER",
        power_limit_written_mw=120_000,
    )
    c0 = FakeGpuController(power_limit_mw=200_000, power_default_limit_mw=200_000)

    res = restore_from_state_file(c0, sp)          # controller is GPU 0

    assert res["reset_clocks"] is False, "the wrong GPU must remain untouched"
    assert c0.get_power_limit_mw() == 200_000, "another GPU's snapshot was applied"
    assert res["record_cleared"] is False
    assert res["incomplete"] is True
    assert any("GPU-OTHER" in e for e in res["errors"])
    assert record(sp)["restored"] is False, "another device's evidence was erased"


def test_stale_check_names_the_device_and_the_gpu_flag(tmp_path):
    """The recommendation used to omit --gpu, walking the operator into the
    wrong-device write."""
    sp = tmp_path / "state.json"
    _write_record(sp, device_index=1, device_uuid="GPU-OTHER")
    rep = check_stale_state(FakeGpuController(), sp)   # controller is GPU 0
    assert rep["record_is_for_this_device"] is False
    assert any("OTHER DEVICE" in f for f in rep["findings"])
    assert "--gpu 1" in rep["recommendation"]


def test_requested_evidence_is_promoted_before_live_pid_refusal(tmp_path):
    """A path-B guard must see debt even when recovery stops at live PID."""
    sp = tmp_path / "requested.json"
    machine = tmp_path / "machine-state.json"
    _write_record(
        sp,
        pid=os.getpid(),
        device_index=1,
        device_uuid="GPU-RECORDED",
    )

    res = restore_from_state_file(
        FakeGpuController(index=1, uuid="GPU-RECORDED"), sp
    )

    assert res["incomplete"] is True
    assert res["reset_clocks"] is False
    assert "still running" in res["record_not_cleared_reason"]
    promoted = record(machine)
    assert promoted["restored"] is False
    assert promoted["pid"] == os.getpid()
    assert promoted["device_index"] == 1
    assert promoted["device_uuid"] == "GPU-RECORDED"
    assert promoted["evidence_path"] == str(sp)

    path_b = GpuGuard(
        FakeGpuController(),
        consent=True,
        require_root=False,
        state_path=tmp_path / "unrelated.json",
    )
    with pytest.raises(GuardBusyError):
        path_b.arm()


def test_promotion_precedes_identity_refusal_without_inventing_uuid(tmp_path):
    """The selected controller cannot fill a UUID missing from old evidence."""
    sp = tmp_path / "requested.json"
    machine = tmp_path / "machine-state.json"
    legacy = {
        "pid": 999999,
        "restored": False,
        "device_index": 1,
        "device_uuid": None,
        "snapshot": {
            "index": 1,
            "uuid": None,
            "power_limit_mw": 150_000,
        },
        "clocks_locked": True,
        "power_limit_written_mw": 120_000,
    }
    sp.write_text(json.dumps({"schema": 2, "guards": {"999999": legacy}}))
    live = FakeGpuController(
        index=0,
        uuid="GPU-LIVE",
        power_limit_mw=120_000,
        power_default_limit_mw=200_000,
    )

    res = restore_from_state_file(live, sp)

    assert res["incomplete"] is True
    assert res["reset_clocks"] is False
    assert res["record_cleared"] is False
    assert live.set_power_calls == 0
    marker = json.loads(machine.read_text())
    assert marker["_joule_machine_blocker"] == 1
    preserved = bytes.fromhex(marker["preserved_evidence_hex"])
    assert json.loads(preserved)["guards"]["999999"]["device_uuid"] is None
    assert any("could not be verified" in e for e in res["errors"])

    path_b = GpuGuard(
        FakeGpuController(),
        consent=True,
        require_root=False,
        state_path=tmp_path / "unrelated.json",
    )
    with pytest.raises(GuardBusyError):
        path_b.arm()


def test_force_power_default_cannot_override_active_identity_mismatch(tmp_path):
    sp = tmp_path / "requested.json"
    _write_record(
        sp,
        device_index=1,
        device_uuid="GPU-OTHER",
        power_limit_written_mw=120_000,
    )
    selected = FakeGpuController(
        index=0,
        uuid="GPU-SELECTED",
        power_limit_mw=130_000,
        power_default_limit_mw=200_000,
    )

    res = restore_from_state_file(
        selected, sp, force_power_default=True
    )

    assert res["incomplete"] is True
    assert res["reset_clocks"] is False
    assert res["record_cleared"] is False
    assert selected.set_power_calls == 0
    assert selected.get_power_limit_mw() == 130_000


def test_recovery_rechecks_identity_after_controller_claim(tmp_path):
    sp = tmp_path / "state.json"
    machine = tmp_path / "machine-state.json"
    _write_record(
        sp,
        device_uuid="GPU-A",
        power_limit_written_mw=120_000,
    )

    class RetargetOnClaim(FakeGpuController):
        def claim(self, owner):
            super().claim(owner)
            self._uuid = "GPU-B"

    controller = RetargetOnClaim(
        uuid="GPU-A",
        power_limit_mw=120_000,
        power_default_limit_mw=200_000,
    )
    res = restore_from_state_file(controller, sp)

    assert res["incomplete"] is True
    assert res["record_cleared"] is False
    assert controller.reset_calls == 0
    assert controller.set_power_calls == 0
    assert json.loads(machine.read_text())["authority_conflict"] is True


def test_recovery_rechecks_identity_after_clock_reset(tmp_path):
    sp = tmp_path / "state.json"
    machine = tmp_path / "machine-state.json"
    _write_record(
        sp,
        device_uuid="GPU-A",
        power_limit_written_mw=120_000,
    )

    class RetargetOnReset(FakeGpuController):
        def reset_locked_clocks(self):
            super().reset_locked_clocks()
            self._uuid = "GPU-B"

    controller = RetargetOnReset(
        uuid="GPU-A",
        power_limit_mw=120_000,
        power_default_limit_mw=200_000,
    )
    res = restore_from_state_file(controller, sp)

    assert res["incomplete"] is True
    assert res["record_cleared"] is False
    assert controller.reset_calls == 1
    assert controller.set_power_calls == 0
    assert controller.get_power_limit_mw() == 120_000
    assert json.loads(machine.read_text())["authority_conflict"] is True


def test_recovery_rechecks_identity_after_power_callback(tmp_path):
    sp = tmp_path / "state.json"
    machine = tmp_path / "machine-state.json"
    _write_record(
        sp,
        device_uuid="GPU-A",
        power_limit_written_mw=120_000,
    )

    class RetargetOnPower(FakeGpuController):
        def set_power_limit_mw(self, milliwatts):
            super().set_power_limit_mw(milliwatts)
            self._uuid = "GPU-B"

    controller = RetargetOnPower(
        uuid="GPU-A",
        power_limit_mw=120_000,
        power_default_limit_mw=200_000,
    )
    res = restore_from_state_file(controller, sp)

    assert res["incomplete"] is True
    assert res["record_cleared"] is False
    assert controller.set_power_calls == 1
    assert json.loads(machine.read_text())["authority_conflict"] is True

    retry = FakeGpuController(
        uuid="GPU-A",
        power_limit_mw=120_000,
        power_default_limit_mw=200_000,
    )
    retry_res = restore_from_state_file(retry, sp)
    assert retry_res["incomplete"] is True
    assert retry_res["record_cleared"] is False
    assert retry.reset_calls == 0
    assert json.loads(machine.read_text())["authority_conflict"] is True


def test_active_evidence_is_promoted_before_first_recovery_write(tmp_path):
    """An interruption inside NVML must leave canonical active authority."""
    sp = tmp_path / "requested.json"
    machine = tmp_path / "machine-state.json"
    _write_record(sp)

    class InterruptReset(FakeGpuController):
        def reset_locked_clocks(self):
            self._require_claim()
            blocker = json.loads(machine.read_text())
            assert blocker["authority_conflict"] is True
            assert preserved_blocker_record(machine)["restored"] is False
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        restore_from_state_file(InterruptReset(), sp)

    assert record(machine)["restored"] is False
    path_b = GpuGuard(
        FakeGpuController(),
        consent=True,
        require_root=False,
        state_path=tmp_path / "unrelated.json",
    )
    with pytest.raises(GuardBusyError):
        path_b.arm()


def test_refused_arm_promotes_path_specific_active_debt_before_path_b(tmp_path):
    path_a = tmp_path / "path-a.json"
    machine = tmp_path / "machine-state.json"
    _write_record(path_a)

    first = GpuGuard(
        FakeGpuController(),
        consent=True,
        require_root=False,
        state_path=path_a,
    )
    with pytest.raises(GuardBusyError):
        first.arm()

    assert record(machine)["restored"] is False
    second = GpuGuard(
        FakeGpuController(),
        consent=True,
        require_root=False,
        state_path=tmp_path / "path-b.json",
    )
    with pytest.raises(GuardBusyError):
        second.arm()

    recovered = restore_from_state_file(FakeGpuController(), path_a)
    assert recovered["errors"] == []
    assert recovered["record_cleared"] is True
    assert record(path_a)["restored"] is True
    assert record(machine)["restored"] is True


def test_refused_arm_globalizes_unreadable_debt_before_path_b(tmp_path):
    path_a = tmp_path / "path-a.json"
    machine = tmp_path / "machine-state.json"
    path_a.write_text("{ corrupt")

    first = GpuGuard(
        FakeGpuController(),
        consent=True,
        require_root=False,
        state_path=path_a,
    )
    with pytest.raises(GuardBusyError):
        first.arm()

    marker = json.loads(machine.read_text())
    assert marker["_joule_machine_blocker"] == 1
    second = GpuGuard(
        FakeGpuController(),
        consent=True,
        require_root=False,
        state_path=tmp_path / "path-b.json",
    )
    with pytest.raises(GuardBusyError):
        second.arm()


def test_active_machine_plus_unreadable_requested_becomes_durable_conflict(
    tmp_path,
):
    requested = tmp_path / "requested.json"
    machine = tmp_path / "machine-state.json"
    requested.write_text("{ corrupt")
    _write_record(machine)
    machine_before = machine.read_bytes()
    controller = FakeGpuController()

    res = restore_from_state_file(controller, requested)

    assert res["incomplete"] is True
    assert controller.reset_calls == 0
    marker = json.loads(machine.read_text())
    assert marker["authority_conflict"] is True
    assert bytes.fromhex(marker["preserved_evidence_hex"]) == machine_before

    later = FakeGpuController()
    later_res = restore_from_state_file(later, tmp_path / "path-c.json")
    assert later_res["incomplete"] is True
    assert later.reset_calls == 0


@pytest.mark.parametrize("force_power_default", [False, True])
def test_quarantine_never_selects_between_valid_active_records(
    tmp_path, force_power_default
):
    requested = tmp_path / "requested.json"
    machine = tmp_path / "machine-state.json"
    _write_record(
        requested,
        pid=999998,
        device_uuid="GPU-REQUESTED",
        power_limit_written_mw=120_000,
    )
    _write_record(
        machine,
        pid=999997,
        device_uuid="GPU-MACHINE",
        power_limit_written_mw=110_000,
    )
    requested_before = requested.read_bytes()
    machine_before = machine.read_bytes()
    controller = FakeGpuController(
        uuid="GPU-REQUESTED",
        power_limit_mw=130_000,
        power_default_limit_mw=200_000,
    )

    res = restore_from_state_file(
        controller,
        requested,
        quarantine_unreadable=True,
        force_power_default=force_power_default,
    )

    assert res["incomplete"] is True
    assert res["record_cleared"] is False
    assert "quarantined_to" not in res
    assert any("authority conflict" in e for e in res["errors"])
    assert controller.reset_calls == 0
    assert controller.set_power_calls == 0
    assert requested.read_bytes() == requested_before
    marker = json.loads(machine.read_text())
    assert marker["_joule_machine_blocker"] == 1
    assert marker["authority_conflict"] is True
    assert bytes.fromhex(marker["preserved_evidence_hex"]) == machine_before
    assert not list(tmp_path.glob("machine-state.corrupt-*"))

    later = FakeGpuController(uuid="GPU-MACHINE")
    later_res = restore_from_state_file(later, tmp_path / "path-c.json")
    assert later_res["incomplete"] is True
    assert later.reset_calls == 0
    assert json.loads(machine.read_text())["authority_conflict"] is True


def test_valid_mirror_cannot_discharge_unknown_corrupt_machine_state(
    tmp_path,
):
    requested = tmp_path / "requested.json"
    machine = tmp_path / "machine-state.json"
    _write_record(
        requested,
        power_limit_written_mw=120_000,
        snapshot={
            "power_limit_mw": 150_000,
            "power_default_limit_mw": 200_000,
        },
    )
    machine.write_text("{ corrupt")
    controller = FakeGpuController(
        power_limit_mw=120_000,
        power_default_limit_mw=200_000,
    )

    res = restore_from_state_file(
        controller, requested, quarantine_unreadable=True
    )

    assert res["incomplete"] is True
    assert res["record_cleared"] is False
    assert controller.reset_calls == 0
    assert controller.set_power_calls == 0
    assert controller.get_power_limit_mw() == 120_000
    assert Path(res["quarantined_to"]).read_text() == "{ corrupt"
    assert record(requested)["restored"] is False
    assert json.loads(machine.read_text())["_joule_machine_blocker"] == 1


def test_identical_active_machine_and_requested_mirrors_both_finish_clean(
    tmp_path,
):
    requested = tmp_path / "requested.json"
    machine = tmp_path / "machine-state.json"
    active_doc = json.dumps({"schema": 3, "record": schema3_record()})
    requested.write_text(active_doc)
    machine.write_text(active_doc)

    res = restore_from_state_file(FakeGpuController(), requested)

    assert res["errors"] == []
    assert res["record_cleared"] is True
    assert record(requested)["restored"] is True
    assert record(machine)["restored"] is True
    assert record(requested)["evidence_path"] == str(requested)


def test_force_quarantine_cannot_guess_device_for_unknown_evidence(tmp_path):
    requested = tmp_path / "requested.json"
    machine = tmp_path / "machine-state.json"
    requested.write_text("{ corrupt")

    controller = FakeGpuController(
        power_limit_mw=120_000,
        power_default_limit_mw=200_000,
    )
    res = restore_from_state_file(
        controller,
        requested,
        quarantine_unreadable=True,
        force_power_default=True,
    )

    assert res["incomplete"] is True
    assert controller.reset_calls == 0
    assert controller.set_power_calls == 0
    assert controller.get_power_limit_mw() == 120_000
    assert json.loads(machine.read_text())["_joule_machine_blocker"] == 1
    path_b = GpuGuard(
        FakeGpuController(),
        consent=True,
        require_root=False,
        state_path=tmp_path / "unrelated.json",
    )
    with pytest.raises(GuardBusyError):
        path_b.arm()


def test_nested_same_thread_restore_joins_outer_restore(tmp_path):
    requested = tmp_path / "requested.json"

    class NestedRestoreController(FakeGpuController):
        guard = None
        nested_calls = 0

        def reset_locked_clocks(self):
            self._require_claim()
            self.nested_calls += 1
            self.guard.restore()
            super().reset_locked_clocks()

    controller = NestedRestoreController()
    guarded = GpuGuard(
        controller,
        consent=True,
        require_root=False,
        state_path=requested,
    )
    controller.guard = guarded
    guarded.lock_clocks(1400)

    guarded.restore()

    assert controller.nested_calls == 1
    assert record(requested)["restored"] is True


def test_forward_write_is_refused_while_restore_is_active(tmp_path):
    requested = tmp_path / "requested.json"
    machine = tmp_path / "machine-state.json"

    class WriteDuringReset(FakeGpuController):
        guard = None
        nested_error = None

        def reset_locked_clocks(self):
            self._require_claim()
            try:
                self.guard.lock_clocks(1000)
            except GpuGuardError as exc:
                self.nested_error = exc
            super().reset_locked_clocks()

    controller = WriteDuringReset()
    guarded = GpuGuard(
        controller,
        consent=True,
        require_root=False,
        state_path=requested,
    )
    controller.guard = guarded
    guarded.lock_clocks(1400)

    guarded.restore()

    assert controller.nested_error is not None
    assert "restore transaction is active" in str(controller.nested_error)
    assert controller.locked is None
    assert record(requested)["restored"] is True
    assert record(machine)["restored"] is True
    assert guarded._lease_fd is None


def test_discovered_unreadable_debt_blocks_every_other_state_path(tmp_path):
    requested = tmp_path / "requested.json"
    machine = tmp_path / "machine-state.json"
    requested.write_text("{ corrupt")

    res = restore_from_state_file(FakeGpuController(), requested)

    assert res["incomplete"] is True
    _prior, status, _extra = gpu_guard_module._load_state(machine)
    assert status == gpu_guard_module.STATE_UNREADABLE
    path_b = GpuGuard(
        FakeGpuController(),
        consent=True,
        require_root=False,
        state_path=tmp_path / "unrelated.json",
    )
    with pytest.raises(GuardBusyError):
        path_b.arm()


def test_clean_machine_record_cannot_discard_matching_active_mirror(tmp_path):
    requested = tmp_path / "requested.json"
    machine = tmp_path / "machine-state.json"
    active = schema3_record(
        evidence_path=str(requested),
        power_limit_written_mw=120_000,
    )
    clean = dict(active)
    clean.update(
        restored=True,
        clocks_locked=False,
        restore_failed=False,
        restore_errors=[],
        restored_utc="2026-01-01T00:01:00+00:00",
    )
    requested.write_text(json.dumps({"schema": 3, "record": active}))
    machine.write_text(json.dumps({"schema": 3, "record": clean}))
    requested_before = requested.read_bytes()
    later_operator_cap = FakeGpuController(
        power_limit_mw=150_000,
        power_default_limit_mw=200_000,
    )

    res = restore_from_state_file(later_operator_cap, requested)

    assert res["incomplete"] is True
    assert res["errors"]
    assert res["record_cleared"] is False
    assert later_operator_cap.reset_calls == 0
    assert res["power_restored_to_mw"] is None
    assert later_operator_cap.set_power_calls == 0
    assert later_operator_cap.get_power_limit_mw() == 150_000
    assert requested.read_bytes() == requested_before
    blocker = json.loads(machine.read_text())
    assert blocker["authority_conflict"] is True
    path_b = GpuGuard(
        FakeGpuController(),
        consent=True,
        require_root=False,
        state_path=tmp_path / "path-b.json",
    )
    with pytest.raises(GuardBusyError):
        path_b.arm()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
@pytest.mark.filterwarnings(
    r"ignore:This process .* is multi-threaded, use of fork\(\) may lead to "
    r"deadlocks in the child:DeprecationWarning"
)
def test_recovery_fork_child_cannot_continue_after_losing_lease(tmp_path):
    requested = tmp_path / "requested.json"
    child_result = tmp_path / "recovery-child.txt"
    _write_record(requested)

    class ForkDuringReset(FakeGpuController):
        in_child = False

        def reset_locked_clocks(self):
            self._require_claim()
            child_pid = os.fork()
            if child_pid == 0:
                self.in_child = True
                return
            waited, status = os.waitpid(child_pid, 0)
            assert waited == child_pid and os.waitstatus_to_exitcode(status) == 0
            super().reset_locked_clocks()

    controller = ForkDuringReset()
    try:
        res = restore_from_state_file(controller, requested)
    except GpuGuardError as exc:
        if controller.in_child:
            child_result.write_text(f"REFUSED:{exc}")
            os._exit(0)
        raise
    if controller.in_child:
        child_result.write_text("CONTINUED")
        os._exit(1)

    assert res["errors"] == []
    assert child_result.read_text().startswith("REFUSED:")
    assert "fork child" in child_result.read_text()


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
    '{"record": {"pid": 999999, "restored": "false", "snapshot": {}}}',
    '{"record": {"pid": 999999, "restored": false, "snapshot": [1]}}',
    '{"record": {"pid": 999999, "device_index": 0, "device_uuid": false}}',
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


@pytest.mark.parametrize(
    "snapshot_identity",
    [
        {"index": 0, "uuid": "GPU-OTHER"},
        {"index": 1, "uuid": "GPU-FAKE"},
    ],
)
def test_snapshot_identity_must_match_record_before_recovery(
    tmp_path, snapshot_identity
):
    sp = tmp_path / "state.json"
    rec = schema3_record(power_limit_written_mw=120_000)
    rec["snapshot"].update(snapshot_identity)
    sp.write_text(json.dumps({"schema": 3, "record": rec}))
    before = sp.read_bytes()
    controller = FakeGpuController(
        power_limit_mw=120_000,
        power_default_limit_mw=200_000,
    )

    _prior, status, _extra = gpu_guard_module._load_state(sp)
    res = restore_from_state_file(controller, sp)

    assert status == gpu_guard_module.STATE_UNREADABLE
    assert res["incomplete"] is True
    assert res["record_cleared"] is False
    assert controller.reset_calls == 0
    assert controller.set_power_calls == 0
    assert controller.get_power_limit_mw() == 120_000
    assert sp.read_bytes() == before


@pytest.mark.parametrize(
    "contradiction",
    [
        {"clocks_locked": True},
        {"restore_failed": True, "restore_errors": ["failed"]},
        {"restore_failed": False, "restore_errors": ["orphan error"]},
        {"restored_utc": None},
    ],
)
def test_contradictory_clean_schema3_record_is_unreadable_and_blocks_arm(
    tmp_path, contradiction
):
    sp = tmp_path / "state.json"
    rec = schema3_record(restored=True)
    rec.update(contradiction)
    sp.write_text(json.dumps({"schema": 3, "record": rec}))

    _prior, status, _extra = gpu_guard_module._load_state(sp)
    guarded = GpuGuard(
        FakeGpuController(),
        consent=True,
        require_root=False,
        state_path=sp,
    )

    assert status == gpu_guard_module.STATE_UNREADABLE
    with pytest.raises(GuardBusyError):
        guarded.arm()
    assert json.loads((tmp_path / "machine-state.json").read_text())[
        "_joule_machine_blocker"
    ] == 1


def test_markerless_active_schema3_cannot_be_cleared_as_clock_only_debt(tmp_path):
    sp = tmp_path / "state.json"
    rec = schema3_record(
        clocks_locked=False,
        locked_min_mhz=None,
        locked_max_mhz=None,
        power_limit_written_mw=None,
        restore_failed=False,
        restore_errors=[],
    )
    sp.write_text(json.dumps({"schema": 3, "record": rec}))
    controller = FakeGpuController(power_limit_mw=120_000)

    _prior, status, _extra = gpu_guard_module._load_state(sp)
    res = restore_from_state_file(controller, sp)

    assert status == gpu_guard_module.STATE_UNREADABLE
    assert res["incomplete"] is True
    assert res["record_cleared"] is False
    assert controller.reset_calls == 0
    assert controller.set_power_calls == 0
    assert controller.get_power_limit_mw() == 120_000


@pytest.mark.parametrize(
    "missing_field", sorted(gpu_guard_module._SCHEMA3_REQUIRED_FIELDS)
)
def test_schema3_missing_required_field_has_no_recovery_authority(
    tmp_path, missing_field
):
    sp = tmp_path / "state.json"
    rec = schema3_record(power_limit_written_mw=120_000)
    rec.pop(missing_field)
    sp.write_text(json.dumps({"schema": 3, "record": rec}))
    before = sp.read_bytes()
    c = FakeGpuController(power_limit_mw=120_000)

    report = check_stale_state(c, sp)
    result = restore_from_state_file(c, sp)

    assert report["state_file_unreadable"] is True
    assert result["incomplete"] is True
    assert result["record_cleared"] is False
    assert c.set_power_calls == 0
    assert c.get_power_limit_mw() == 120_000
    assert sp.read_bytes() == before


@pytest.mark.parametrize(
    "envelope",
    [
        "future-record",
        "schema3-guards",
        "schema2-record",
        "schema3-both",
        "record-without-schema",
        "boolean-schema",
        "string-schema",
    ],
)
def test_unknown_or_contradictory_envelope_cannot_drive_recovery(
    tmp_path, envelope
):
    rec = schema3_record(power_limit_written_mw=120_000)
    guards = {str(rec["pid"]): rec}
    documents = {
        "future-record": {"schema": 999, "record": rec},
        "schema3-guards": {"schema": 3, "guards": guards},
        "schema2-record": {"schema": 2, "record": rec},
        "schema3-both": {"schema": 3, "record": rec, "guards": guards},
        "record-without-schema": {"record": rec},
        "boolean-schema": {"schema": True, "record": rec},
        "string-schema": {"schema": "3", "record": rec},
    }
    sp = tmp_path / "state.json"
    sp.write_text(json.dumps(documents[envelope]))
    before = sp.read_bytes()
    c = FakeGpuController(power_limit_mw=120_000)

    result = restore_from_state_file(c, sp)

    assert result["incomplete"] is True
    assert result["record_cleared"] is False
    assert c.set_power_calls == 0
    assert c.get_power_limit_mw() == 120_000
    assert sp.read_bytes() == before


@pytest.mark.parametrize(
    "invalid_field",
    [
        {"restored": "false", "snapshot": {"power_limit_mw": 150_000}},
        {"restored": False, "snapshot": [1]},
        {
            "restored": False,
            "snapshot": {"power_limit_mw": 150_000},
            "device_index": 0,
            "device_uuid": False,
        },
        {
            "restored": False,
            "snapshot": {"power_limit_mw": 150_000},
            "device_index": False,
        },
        {
            "restored": False,
            "snapshot": {"power_limit_mw": 175_000},
            "power_limit_written_mw": False,
        },
        {
            "restored": False,
            "snapshot": {"power_limit_mw": False},
            "power_limit_written_mw": 120_000,
        },
        {"pid": 0},
        {"pid": -1},
        {"pid": False},
    ],
)
def test_malformed_record_types_cannot_drive_or_crash_recovery(
    tmp_path, invalid_field
):
    sp = tmp_path / "state.json"
    rec = schema3_record(power_limit_written_mw=120_000)
    rec.update(invalid_field)
    sp.write_text(json.dumps({"schema": 3, "record": rec}))
    before = sp.read_bytes()
    controller = FakeGpuController(
        uuid="GPU-DIFFERENT",
        power_limit_mw=120_000,
        power_default_limit_mw=200_000,
    )

    result = restore_from_state_file(controller, sp)

    assert result["reset_clocks"] is False
    assert result["incomplete"] is True
    assert result["errors"]
    assert result["record_cleared"] is False
    assert controller.set_power_calls == 0
    assert controller.get_power_limit_mw() == 120_000
    assert sp.read_bytes() == before


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


def test_unreadable_file_can_be_preserved_but_not_guessed_by_quarantine(tmp_path):
    """Quarantine preserves evidence but cannot silently declare hardware safe."""
    sp = tmp_path / "state.json"
    sp.write_text("{ corrupt")
    c = FakeGpuController()

    res = restore_from_state_file(c, sp)
    assert res["incomplete"] is True and res["errors"], "wedge not reported"

    res2 = restore_from_state_file(c, sp, quarantine_unreadable=True)
    assert res2["quarantined_to"]
    assert Path(res2["quarantined_to"]).exists(), "the original was not preserved"
    assert not sp.exists()
    assert res2["incomplete"] is True
    assert check_stale_state(c, sp)["stale"] is True

    unresolved = restore_from_state_file(
        c,
        sp,
        quarantine_unreadable=True,
        force_power_default=True,
    )
    assert unresolved["incomplete"] is True
    assert unresolved["record_cleared"] is False
    assert c.reset_calls == 0
    assert c.set_power_calls == 0
    assert check_stale_state(c, sp)["stale"] is True


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
    simulate_guard_crash(g1)

    res = restore_from_state_file(gpu0, sp)      # operator points at GPU 0

    assert res["reset_clocks"] is False, "the wrong GPU must remain untouched"
    assert gpu0.get_power_limit_mw() == 200_000, "GPU 1's snapshot hit GPU 0"
    assert res["incomplete"] is True
    assert res["record_cleared"] is False
    assert any("GPU-1111" in e for e in res["errors"])
    assert record(sp)["restored"] is False, "GPU 1's evidence was destroyed"

    # And pointed at the right card it works. A fresh controller, because
    # Recovery normally runs in a separate process after the writer died.
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
    simulate_guard_crash(g)

    after_reboot = FakeGpuController(index=0, uuid="GPU-BBBB")   # same index!
    res = restore_from_state_file(after_reboot, sp)

    assert res["incomplete"] is True
    assert res["record_cleared"] is False
    assert any("enumeration changed" in e for e in res["errors"])
    assert record(sp)["restored"] is False


def test_recovery_fails_closed_when_record_uuid_cannot_be_live_verified(tmp_path):
    """A matching index is not identity when the record carries a UUID.

    If live UUID telemetry fails, recovery must preserve both the device state
    and the unrestored record rather than applying a foreign snapshot on the
    strength of an enumeration index.
    """
    sp = tmp_path / "state.json"
    _write_record(
        sp,
        device_index=0,
        device_uuid="GPU-AAAA",
        power_limit_written_mw=120_000,
    )
    unverifiable = FakeGpuController(
        index=0,
        uuid=None,
        power_limit_mw=130_000,
        power_default_limit_mw=200_000,
    )

    stale = check_stale_state(unverifiable, sp)
    assert stale["record_is_for_this_device"] is None
    assert any("IDENTITY UNVERIFIABLE" in finding for finding in stale["findings"])
    assert "Do not recover by GPU index alone" in stale["recommendation"]

    res = restore_from_state_file(unverifiable, sp)

    assert res["reset_clocks"] is False
    assert res["incomplete"] is True
    assert res["record_cleared"] is False
    assert unverifiable.set_power_calls == 0
    assert unverifiable.get_power_limit_mw() == 130_000
    assert any(
        "UUID" in error and "could not be read" in error for error in res["errors"]
    )
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
    simulate_guard_crash(g)

    now_0 = FakeGpuController(index=0, uuid="GPU-SAME")   # same card, new index
    res = restore_from_state_file(now_0, sp)
    assert res["errors"] == []
    assert res["record_cleared"] is True
