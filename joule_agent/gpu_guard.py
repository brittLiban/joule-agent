"""GPU write guard: the only thing in this codebase that changes hardware state.

Contract, and it is the whole point of the module: **whatever happens -- clean
exit, unhandled exception, SIGINT, SIGTERM, or a hard kill -- the GPU returns to
stock clock and power settings.**

Nothing else may call ``nvmlDeviceSetGpuLockedClocks``,
``nvmlDeviceSetPowerManagementLimit``, ``nvidia-smi -lgc`` or ``nvidia-smi -pl``.
The sweep runner and the dynamic controller go through here. That is not a style
preference: a safety net that lives inside the code it protects is not a net,
because the first buggy draft of that code runs on real hardware with nothing
underneath it.

What can and cannot be verified
-------------------------------
This driver exposes no ``nvmlDeviceGetGpuLockedClocks``. There is no way to ask
the device whether a clock lock is currently applied. Two consequences, both
deliberate:

* **Restore is unconditional.** :meth:`GpuGuard.restore` always issues
  ``nvmlDeviceResetGpuLockedClocks`` rather than checking first and skipping.
  Resetting an unlocked GPU is a no-op, so this is idempotent for free, and it
  removes any dependence on a readback that does not exist.
* **Stale-state detection has two tiers of confidence**, and
  :func:`check_stale_state` labels which is which:

  ``power limit``
      Verifiable. ``GetPowerManagementLimit() != GetPowerManagementDefaultLimit()``
      is direct evidence from the device that a previous run left a cap behind.
  ``clock lock``
      Not verifiable. Detection relies entirely on the state file. A file with
      ``restored: false`` means a process died holding a lock. The device itself
      cannot confirm or deny it.

Clock-lock state in the state file is therefore *asserted by us*, never
*observed from the device*.

Crash ordering
--------------
The state file is written and fsynced **before** the first hardware write, with
``restored: false``. The window this closes is a crash between locking the clock
and recording that we did -- precisely the case the file exists for. The flag
flips to ``restored: true`` only after the reset returns.

Privilege
---------
NVML clock and power writes require root on Linux. Absent that, writes fail
loudly and immediately -- never a silent no-op. Reads never require privilege or
consent.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

__all__ = [
    "MIN_SM_CLOCK_MHZ",
    "CONSENT_FLAG",
    "GpuGuardError",
    "ConsentError",
    "PrivilegeError",
    "ClockFloorError",
    "DeviceSnapshot",
    "GuardState",
    "GpuController",
    "NvmlGpuController",
    "FakeGpuController",
    "GpuGuard",
    "gpu_guard",
    "check_stale_state",
    "restore_from_state_file",
    "DEFAULT_STATE_PATH",
]

#: Hard floor. A lock below this is REFUSED, never silently clamped -- a caller
#: that asked for 100 MHz has a bug or a bad config, and quietly giving them
#: something else hides it. Deliberately conservative; raise it per-device only
#: with measurements showing the device is stable lower.
MIN_SM_CLOCK_MHZ = 300

#: Required before the first hardware write in a process.
CONSENT_FLAG = "--i-understand-this-changes-gpu-settings"

DEFAULT_STATE_PATH = Path("results/gpu_guard_state.json")


class GpuGuardError(RuntimeError):
    """Base for every guard refusal."""


class ConsentError(GpuGuardError):
    pass


class PrivilegeError(GpuGuardError):
    pass


class ClockFloorError(GpuGuardError):
    pass


# ---------------------------------------------------------------------------
# Device abstraction
# ---------------------------------------------------------------------------


@dataclass
class DeviceSnapshot:
    """Stock state, captured before anything is touched."""

    index: int = 0
    name: str | None = None
    uuid: str | None = None
    power_limit_mw: int | None = None
    power_default_limit_mw: int | None = None
    power_min_mw: int | None = None
    power_max_mw: int | None = None
    max_sm_clock_mhz: int | None = None
    sm_clock_at_snapshot_mhz: int | None = None
    #: Unreadable on GeForce parts -- recorded as None, never guessed.
    default_applications_sm_clock_mhz: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class GpuController(Protocol):
    """Narrow write surface. Kept tiny so the guard is testable without a GPU."""

    def snapshot(self) -> DeviceSnapshot: ...
    def set_locked_clocks(self, min_mhz: int, max_mhz: int) -> None: ...
    def reset_locked_clocks(self) -> None: ...
    def set_power_limit_mw(self, milliwatts: int) -> None: ...
    def get_power_limit_mw(self) -> int | None: ...
    def get_power_default_limit_mw(self) -> int | None: ...
    def close(self) -> None: ...


class NvmlGpuController:
    """Real NVML. The only place in the codebase that writes GPU state."""

    def __init__(self, index: int = 0) -> None:
        import pynvml

        self._n = pynvml.nvmlInit() or pynvml
        self._nvml = pynvml
        self._h = pynvml.nvmlDeviceGetHandleByIndex(index)
        self._index = index
        self._closed = False

    def _try(self, fn, default=None):
        try:
            return fn()
        except Exception:
            return default

    def snapshot(self) -> DeviceSnapshot:
        n, h = self._nvml, self._h
        constraints = self._try(
            lambda: n.nvmlDeviceGetPowerManagementLimitConstraints(h), (None, None)
        )
        return DeviceSnapshot(
            index=self._index,
            name=self._try(lambda: _text(n.nvmlDeviceGetName(h))),
            uuid=self._try(lambda: _text(n.nvmlDeviceGetUUID(h))),
            power_limit_mw=self._try(lambda: n.nvmlDeviceGetPowerManagementLimit(h)),
            power_default_limit_mw=self._try(
                lambda: n.nvmlDeviceGetPowerManagementDefaultLimit(h)
            ),
            power_min_mw=constraints[0] if constraints else None,
            power_max_mw=constraints[1] if constraints else None,
            max_sm_clock_mhz=self._try(
                lambda: n.nvmlDeviceGetMaxClockInfo(h, n.NVML_CLOCK_SM)
            ),
            sm_clock_at_snapshot_mhz=self._try(
                lambda: n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM)
            ),
            default_applications_sm_clock_mhz=self._try(
                lambda: n.nvmlDeviceGetDefaultApplicationsClock(h, n.NVML_CLOCK_SM)
            ),
        )

    def set_locked_clocks(self, min_mhz: int, max_mhz: int) -> None:
        self._write(lambda: self._nvml.nvmlDeviceSetGpuLockedClocks(self._h, min_mhz, max_mhz))

    def reset_locked_clocks(self) -> None:
        self._write(lambda: self._nvml.nvmlDeviceResetGpuLockedClocks(self._h))

    def set_power_limit_mw(self, milliwatts: int) -> None:
        self._write(
            lambda: self._nvml.nvmlDeviceSetPowerManagementLimit(self._h, int(milliwatts))
        )

    def get_power_limit_mw(self) -> int | None:
        return self._try(lambda: self._nvml.nvmlDeviceGetPowerManagementLimit(self._h))

    def get_power_default_limit_mw(self) -> int | None:
        return self._try(
            lambda: self._nvml.nvmlDeviceGetPowerManagementDefaultLimit(self._h)
        )

    def _write(self, fn) -> None:
        try:
            fn()
        except Exception as exc:
            if type(exc).__name__ == "NVMLError_NoPermission":
                raise PrivilegeError(
                    "NVML refused the write (NVML_ERROR_NO_PERMISSION). Clock and "
                    "power writes require root on Linux. Re-run under sudo. This "
                    "is a hard failure, not a no-op -- nothing was changed."
                ) from exc
            raise

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass


def _text(v) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)


class FakeGpuController:
    """In-memory controller. Lets every crash path be tested without a GPU."""

    def __init__(
        self,
        *,
        power_limit_mw: int = 200_000,
        power_default_limit_mw: int = 200_000,
        fail_restore: bool = False,
    ) -> None:
        self._power = power_limit_mw
        self._default = power_default_limit_mw
        self.locked: tuple | None = None
        self.reset_calls = 0
        self.set_clock_calls = 0
        self.set_power_calls = 0
        self.closed = False
        #: Simulates a broken restore, so tests can prove they discriminate.
        self.fail_restore = fail_restore

    def snapshot(self) -> DeviceSnapshot:
        return DeviceSnapshot(
            index=0,
            name="FakeGPU",
            uuid="GPU-FAKE",
            power_limit_mw=self._power,
            power_default_limit_mw=self._default,
            power_min_mw=100_000,
            power_max_mw=200_000,
            max_sm_clock_mhz=2100,
            sm_clock_at_snapshot_mhz=210,
            default_applications_sm_clock_mhz=None,
        )

    def set_locked_clocks(self, min_mhz: int, max_mhz: int) -> None:
        self.set_clock_calls += 1
        self.locked = (min_mhz, max_mhz)

    def reset_locked_clocks(self) -> None:
        self.reset_calls += 1
        if self.fail_restore:
            return  # deliberately does not clear `locked`
        self.locked = None

    def set_power_limit_mw(self, milliwatts: int) -> None:
        self.set_power_calls += 1
        self._power = int(milliwatts)

    def get_power_limit_mw(self) -> int | None:
        return self._power

    def get_power_default_limit_mw(self) -> int | None:
        return self._default

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# Persisted state
# ---------------------------------------------------------------------------


@dataclass
class GuardState:
    pid: int = 0
    started_utc: str = ""
    device_index: int = 0
    device_uuid: str | None = None
    snapshot: dict = field(default_factory=dict)
    #: Asserted by us -- the device cannot be queried for this.
    clocks_locked: bool = False
    locked_min_mhz: int | None = None
    locked_max_mhz: int | None = None
    power_limit_written_mw: int | None = None
    restored: bool = True
    restored_utc: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["_note"] = (
            "clocks_locked is ASSERTED by the writing process, not observed from "
            "the device -- this driver exposes no way to read back a clock lock. "
            "restored:false means a process died holding GPU state; run "
            "`python -m joule_agent.gpu_guard restore` (as root)."
        )
        return d


def _write_state_atomic(path: Path, state: GuardState) -> None:
    """Write + fsync before any hardware write. Survives a hard kill."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(state.to_dict(), fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    # fsync the directory so the rename itself is durable.
    try:
        dfd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:
        pass


def _read_state(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


class GpuGuard:
    """Owns every GPU write and guarantees restoration on all exit paths."""

    def __init__(
        self,
        controller: GpuController,
        *,
        consent: bool = False,
        state_path: Path | str = DEFAULT_STATE_PATH,
        require_root: bool = True,
        min_clock_mhz: int = MIN_SM_CLOCK_MHZ,
    ) -> None:
        self._c = controller
        self._consent = consent
        self._state_path = Path(state_path)
        self._require_root = require_root
        self._min_clock = min_clock_mhz

        self._lock = threading.RLock()
        self._depth = 0
        self._armed = False
        self._restored = True
        self._touched = False
        self._prev_handlers: dict = {}
        self._atexit_registered = False

        self.snapshot: DeviceSnapshot | None = None
        self.state: GuardState | None = None

    # -- guards ----------------------------------------------------------

    def _check_consent(self) -> None:
        if not self._consent:
            raise ConsentError(
                "Refusing to change GPU settings without explicit consent. Pass "
                f"consent=True (CLI: {CONSENT_FLAG}). This flag exists because "
                "clock and power writes affect hardware shared with everything "
                "else running on this machine. Reads never require it."
            )

    def _check_privilege(self) -> None:
        if self._require_root and hasattr(os, "geteuid") and os.geteuid() != 0:
            raise PrivilegeError(
                "NVML clock/power writes require root on Linux and this process "
                f"is uid {os.geteuid()}. Failing now rather than silently doing "
                "nothing. Re-run under sudo."
            )

    def _check_clock_floor(self, min_mhz: int, max_mhz: int) -> None:
        floor = self._min_clock
        if min_mhz < floor or max_mhz < floor:
            raise ClockFloorError(
                f"Refusing clock lock [{min_mhz}, {max_mhz}] MHz: below the "
                f"{floor} MHz floor. NOT clamped -- a caller asking for this has "
                "a bug or a bad config, and quietly substituting a different "
                "value would hide it."
            )
        if min_mhz > max_mhz:
            raise ClockFloorError(
                f"Refusing clock lock: min {min_mhz} > max {max_mhz} MHz."
            )

    # -- lifecycle -------------------------------------------------------

    def arm(self) -> "GpuGuard":
        """Snapshot stock state and install every restore path."""
        with self._lock:
            if self._armed:
                return self
            self.snapshot = self._c.snapshot()
            self.state = GuardState(
                pid=os.getpid(),
                started_utc=datetime.now(timezone.utc).isoformat(),
                device_index=self.snapshot.index,
                device_uuid=self.snapshot.uuid,
                snapshot=self.snapshot.to_dict(),
                restored=True,
            )
            self._install_handlers()
            self._armed = True
            return self

    def _install_handlers(self) -> None:
        if not self._atexit_registered:
            atexit.register(self._atexit_restore)
            self._atexit_registered = True
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._prev_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self._signal_restore)
            except (ValueError, OSError):
                # Not the main thread, or a platform without this signal.
                pass

    def _atexit_restore(self) -> None:
        try:
            self.restore()
        except Exception:
            pass

    def _signal_restore(self, signum, frame) -> None:
        try:
            self.restore()
        finally:
            prev = self._prev_handlers.get(signum)
            # Restore default behaviour and re-raise, so the process still dies
            # the way the sender intended.
            try:
                signal.signal(signum, prev if callable(prev) else signal.SIG_DFL)
            except Exception:
                pass
            if callable(prev) and prev not in (signal.SIG_DFL, signal.SIG_IGN):
                prev(signum, frame)
            else:
                os.kill(os.getpid(), signum)

    # -- writes ----------------------------------------------------------

    def _mark_touched(self) -> None:
        """Record intent to disk BEFORE touching hardware."""
        assert self.state is not None
        self.state.restored = False
        self._restored = False
        self._touched = True
        _write_state_atomic(self._state_path, self.state)

    def lock_clocks(self, min_mhz: int, max_mhz: int | None = None) -> None:
        max_mhz = min_mhz if max_mhz is None else max_mhz
        with self._lock:
            if not self._armed:
                self.arm()
            self._check_consent()
            # Argument validation BEFORE the environment check: a bad clock
            # value is the caller's bug and is free to detect, so report it
            # even when unprivileged. Checking root first would tell an
            # operator to sudo and only then reveal the real problem --
            # implying sudo would have made 100 MHz acceptable.
            self._check_clock_floor(min_mhz, max_mhz)
            self._check_privilege()
            assert self.state is not None
            self.state.clocks_locked = True
            self.state.locked_min_mhz = min_mhz
            self.state.locked_max_mhz = max_mhz
            self._mark_touched()
            self._c.set_locked_clocks(min_mhz, max_mhz)

    def set_power_limit_mw(self, milliwatts: int) -> None:
        with self._lock:
            if not self._armed:
                self.arm()
            self._check_consent()
            self._check_privilege()
            snap = self.snapshot
            if snap and snap.power_min_mw is not None and milliwatts < snap.power_min_mw:
                raise GpuGuardError(
                    f"Refusing power limit {milliwatts} mW: below the device "
                    f"minimum {snap.power_min_mw} mW. Not clamped."
                )
            if snap and snap.power_max_mw is not None and milliwatts > snap.power_max_mw:
                raise GpuGuardError(
                    f"Refusing power limit {milliwatts} mW: above the device "
                    f"maximum {snap.power_max_mw} mW. Not clamped."
                )
            assert self.state is not None
            self.state.power_limit_written_mw = int(milliwatts)
            self._mark_touched()
            self._c.set_power_limit_mw(int(milliwatts))

    # -- restore ---------------------------------------------------------

    def restore(self) -> None:
        """Return the GPU to stock. Idempotent; safe to call any number of times.

        The clock reset is issued unconditionally rather than gated on a
        readback, because no readback exists.
        """
        with self._lock:
            if self._restored:
                return
            errors = []
            try:
                self._c.reset_locked_clocks()
            except Exception as exc:
                errors.append(f"reset_locked_clocks: {exc}")

            snap = self.snapshot
            target = snap.power_default_limit_mw if snap else None
            if target is None and snap is not None:
                target = snap.power_limit_mw
            if target is not None and self.state and self.state.power_limit_written_mw:
                try:
                    self._c.set_power_limit_mw(int(target))
                except Exception as exc:
                    errors.append(f"set_power_limit_mw: {exc}")

            self._restored = True
            if self.state is not None:
                self.state.restored = True
                self.state.clocks_locked = False
                self.state.restored_utc = datetime.now(timezone.utc).isoformat()
                try:
                    _write_state_atomic(self._state_path, self.state)
                except Exception:
                    pass
            if errors:
                raise GpuGuardError("restore incomplete: " + "; ".join(errors))

    # -- context manager -------------------------------------------------

    def __enter__(self) -> "GpuGuard":
        with self._lock:
            self.arm()
            self._depth += 1
            return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        with self._lock:
            self._depth -= 1
            if self._depth <= 0:
                self.restore()
        return False  # never swallow the exception


def gpu_guard(
    controller: GpuController | None = None,
    *,
    consent: bool = False,
    state_path: Path | str = DEFAULT_STATE_PATH,
    require_root: bool = True,
    index: int = 0,
) -> GpuGuard:
    """Primary API: ``with gpu_guard(consent=True) as g: g.lock_clocks(1400)``."""
    if controller is None:
        controller = NvmlGpuController(index=index)
    return GpuGuard(
        controller, consent=consent, state_path=state_path, require_root=require_root
    )


# ---------------------------------------------------------------------------
# Stale state
# ---------------------------------------------------------------------------


def check_stale_state(
    controller: GpuController, state_path: Path | str = DEFAULT_STATE_PATH
) -> dict:
    """Detect GPU state left behind by a process that died without restoring.

    Returns a report separating what the device confirms from what only the
    state file asserts. The caller should surface both, labelled.
    """
    path = Path(state_path)
    prior = _read_state(path)
    report = {
        "state_file": str(path),
        "state_file_present": prior is not None,
        "verified_power_modified": False,
        "asserted_clock_lock": False,
        "stale": False,
        "findings": [],
        "recommendation": None,
    }

    # Tier 1: verifiable directly from the device.
    cur = controller.get_power_limit_mw()
    dfl = controller.get_power_default_limit_mw()
    if cur is not None and dfl is not None and cur != dfl:
        report["verified_power_modified"] = True
        report["stale"] = True
        report["findings"].append(
            f"VERIFIED (device): power limit is {cur} mW but the device default "
            f"is {dfl} mW. Something left a cap behind."
        )

    # Tier 2: assertion only -- the device cannot be asked.
    if prior and not prior.get("restored", True):
        report["stale"] = True
        pid = prior.get("pid")
        alive = _pid_alive(pid) if pid else False
        report["asserted_clock_lock"] = bool(prior.get("clocks_locked"))
        report["findings"].append(
            f"ASSERTED (state file, NOT verifiable): pid {pid} recorded "
            f"restored=false"
            + (
                f" and a clock lock at [{prior.get('locked_min_mhz')}, "
                f"{prior.get('locked_max_mhz')}] MHz."
                if prior.get("clocks_locked")
                else "."
            )
            + (
                f" That pid is still running -- it may simply be mid-run."
                if alive
                else " That pid is gone; it died without restoring."
            )
        )
        report["writer_pid_alive"] = alive

    if report["stale"]:
        report["recommendation"] = (
            "Run `sudo python -m joule_agent.gpu_guard restore` to return the "
            "device to stock. Safe to run even if nothing is actually locked -- "
            "the clock reset is a no-op on an unlocked GPU."
        )
    return report


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def restore_from_state_file(
    controller: GpuController, state_path: Path | str = DEFAULT_STATE_PATH
) -> dict:
    """Recovery path for after a hard kill. Unconditional and idempotent."""
    prior = _read_state(Path(state_path))
    result = {"reset_clocks": False, "power_restored_to_mw": None, "errors": []}

    try:
        controller.reset_locked_clocks()
        result["reset_clocks"] = True
    except Exception as exc:
        result["errors"].append(f"reset_locked_clocks: {exc}")

    target = None
    if prior:
        target = (prior.get("snapshot") or {}).get("power_default_limit_mw")
    if target is None:
        target = controller.get_power_default_limit_mw()
    if target is not None:
        try:
            controller.set_power_limit_mw(int(target))
            result["power_restored_to_mw"] = int(target)
        except Exception as exc:
            result["errors"].append(f"set_power_limit_mw: {exc}")

    if prior is not None:
        prior["restored"] = True
        prior["clocks_locked"] = False
        prior["restored_utc"] = datetime.now(timezone.utc).isoformat()
        prior["_recovered_by"] = "restore_from_state_file"
        try:
            p = Path(state_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(prior, indent=2))
        except Exception as exc:
            result["errors"].append(f"state_file: {exc}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _controller(args) -> GpuController:
    return NvmlGpuController(index=args.gpu)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="joule_agent.gpu_guard",
        description="Guarded GPU clock/power writes with guaranteed restore.",
    )
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--state-path", default=str(DEFAULT_STATE_PATH))
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="report stale state (read-only, no privilege)")
    sub.add_parser("restore", help="return the device to stock (requires root)")

    lk = sub.add_parser("lock", help="lock the SM clock, hold, then restore")
    lk.add_argument("--mhz", type=int, required=True)
    lk.add_argument("--hold", type=float, default=5.0, help="seconds")
    lk.add_argument(CONSENT_FLAG, dest="consent", action="store_true")
    lk.add_argument(
        "--crash-after",
        type=float,
        default=None,
        help="TEST ONLY: hard-kill this process mid-lock to exercise recovery",
    )

    args = p.parse_args(argv)
    c = _controller(args)
    try:
        if args.cmd == "status":
            rep = check_stale_state(c, args.state_path)
            print(json.dumps(rep, indent=2))
            return 1 if rep["stale"] else 0

        if args.cmd == "restore":
            res = restore_from_state_file(c, args.state_path)
            print(json.dumps(res, indent=2))
            return 1 if res["errors"] else 0

        if args.cmd == "lock":
            g = GpuGuard(c, consent=args.consent, state_path=args.state_path)
            with g:
                g.lock_clocks(args.mhz)
                print(f"locked SM clock at {args.mhz} MHz; holding {args.hold}s")
                if args.crash_after is not None:
                    time.sleep(args.crash_after)
                    print("simulating hard kill (SIGKILL to self)", flush=True)
                    os.kill(os.getpid(), signal.SIGKILL)
                time.sleep(args.hold)
            print("restored to stock")
            return 0
    except GpuGuardError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    finally:
        c.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
