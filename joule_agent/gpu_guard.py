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

#: Relative by default, so `gpu_guard restore` must run from the project root.
#: Override with JOULE_GUARD_STATE to make recovery cwd-independent.
DEFAULT_STATE_PATH = Path(
    os.environ.get("JOULE_GUARD_STATE", "results/gpu_guard_state.json")
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


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

    def claim(self, owner) -> None: ...
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
        self._claimed_by = None

    def claim(self, owner) -> None:
        """Mark this controller as owned by a guard (or the recovery path).

        Exclusive: a second claim by a different owner is refused, so two
        guards cannot silently share one controller. This is a guard-rail
        against accident, NOT a security boundary -- `claim()` is public and
        Python has no way to make it otherwise. It exists so that the common
        mistake (constructing a controller and calling it directly) fails
        loudly instead of bypassing the floor, consent and privilege checks.
        """
        if self._claimed_by is not None and self._claimed_by is not owner:
            raise GpuGuardError(
                f"controller already claimed by {self._claimed_by!r}; refusing "
                "to hand the same device to a second owner"
            )
        self._claimed_by = owner

    def _require_claim(self) -> None:
        if self._claimed_by is None:
            raise GpuGuardError(
                "Refusing a direct hardware write: this controller is not held "
                "by a GpuGuard. Constructing NvmlGpuController does not grant "
                "write access -- it would bypass the clock floor, the consent "
                "flag and the privilege check. Use `with gpu_guard(...)`."
            )

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
        self._require_claim()
        if self._closed:
            # An atexit retry can fire after close(); a swallowed
            # NVMLError_Uninitialized here would look like a successful restore.
            raise GpuGuardError(
                "controller is closed; NVML was shut down before this write. "
                "The GPU may still be modified -- run "
                "`sudo python -m joule_agent.gpu_guard restore`."
            )
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
        self._claimed_by = None
        #: Simulates a broken restore, so tests can prove they discriminate.
        self.fail_restore = fail_restore

    def claim(self, owner) -> None:
        if self._claimed_by is not None and self._claimed_by is not owner:
            raise GpuGuardError(
                f"controller already claimed by {self._claimed_by!r}"
            )
        self._claimed_by = owner

    def _require_claim(self) -> None:
        if self._claimed_by is None:
            raise GpuGuardError(
                "Refusing a direct hardware write: controller not held by a "
                "GpuGuard."
            )

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
        self._require_claim()
        self.set_clock_calls += 1
        self.locked = (min_mhz, max_mhz)

    def reset_locked_clocks(self) -> None:
        self._require_claim()
        self.reset_calls += 1
        if self.fail_restore:
            return  # deliberately does not clear `locked`
        self.locked = None

    def set_power_limit_mw(self, milliwatts: int) -> None:
        self._require_claim()
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
    #: Set when a restore attempt FAILED. The device may still be modified.
    restore_failed: bool = False
    restore_errors: list = field(default_factory=list)
    last_restore_attempt_utc: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["_note"] = (
            "clocks_locked is ASSERTED by the writing process, not observed from "
            "the device -- this driver exposes no way to read back a clock lock. "
            "restored:false means a process died holding GPU state; run "
            "`python -m joule_agent.gpu_guard restore` (as root)."
        )
        return d


STATE_SCHEMA = 2


def _write_doc_atomic(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    try:
        dfd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:
        pass



def _write_state_atomic(path: Path, state: GuardState) -> None:
    """Write + fsync before any hardware write. Survives a hard kill.

    Records are keyed by pid. A single shared document with one record would
    let a second guard's clean restore overwrite a first guard's un-restored
    record, destroying the only evidence that a GPU is still modified.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = _read_doc(path)
    doc["schema"] = STATE_SCHEMA
    doc.setdefault("guards", {})[str(state.pid)] = state.to_dict()
    # Per-process tmp name. A shared ".tmp" lets one process rename another's
    # in-flight file away, which deletes records outright and raises
    # FileNotFoundError in os.replace -- strictly worse than the single-record
    # clobbering this schema replaced.
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=2)
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


def _read_doc(path: Path) -> dict:
    try:
        doc = json.loads(Path(path).read_text())
    except Exception:
        return {}
    if not isinstance(doc, dict):
        return {}
    if "guards" not in doc:
        # Schema 1: a single bare record. Migrate it in memory so old state
        # files from a previous run remain recoverable.
        if "pid" in doc:
            return {"schema": 1, "guards": {str(doc.get("pid")): doc}}
        return {}
    return doc


def _all_records(path: Path) -> list:
    return list(_read_doc(path).get("guards", {}).values())


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
        self._write_depth = 0
        self._armed = False
        self._restored = True
        self._prev_handlers: dict = {}
        self._atexit_registered = False
        self.handler_install_failures: list = []

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
            claim = getattr(self._c, "claim", None)
            if callable(claim):
                claim(self)
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
            except (ValueError, OSError) as exc:
                # Not the main thread, or a platform without this signal.
                # This DEGRADES the guarantee -- SIGTERM's default action
                # terminates without running atexit -- so it must never be
                # silent. Recovery then rests entirely on the state file.
                self.handler_install_failures.append(str(sig))
                print(
                    f"WARNING [gpu_guard]: could not install a {sig!r} handler "
                    f"({type(exc).__name__}: {exc}). In-process restore on that "
                    "signal is NOT armed; recovery depends on the state file at "
                    f"{self._state_path}. Arm the guard from the main thread to "
                    "restore the guarantee.",
                    file=sys.stderr,
                )

    def _atexit_restore(self) -> None:
        try:
            self.restore()
        except Exception as exc:
            # Cannot re-raise usefully at interpreter shutdown, but silence here
            # would hide a GPU left in a modified state.
            print(
                f"ERROR [gpu_guard]: restore failed during exit: {exc}",
                file=sys.stderr,
            )

    def _signal_restore(self, signum, frame) -> None:
        # A worker thread may hold self._lock. Signal handlers run on the main
        # thread, so blocking here would make SIGINT/SIGTERM unresponsive and
        # skip restore entirely. Bounded wait, then proceed regardless: restore
        # is idempotent and the clock reset is a no-op on an unlocked GPU, so a
        # racy restore is strictly safer than no restore.
        acquired = self._lock.acquire(timeout=2.0)
        try:
            # _restore_impl, not restore(): restore() would re-acquire this same
            # RLock and block for as long as the worker holds it, making the
            # bounded acquire above do nothing. Proceeding unlocked risks a race;
            # restore is idempotent and the clock reset is a no-op on an
            # unlocked GPU, so a racy restore beats no restore.
            if not acquired:
                print(
                    "WARNING [gpu_guard]: lock held elsewhere; restoring without "
                    "it to avoid hanging the signal handler.",
                    file=sys.stderr,
                )
            self._restore_impl()
        except Exception as exc:
            print(
                f"ERROR [gpu_guard]: restore failed handling {signum}: {exc}",
                file=sys.stderr,
            )
        finally:
            if acquired:
                try:
                    self._lock.release()
                except RuntimeError:
                    pass

        # Chaining happens after the try/except, never inside `finally` -- a
        # `return` in a finally block silently discards any in-flight exception.
        prev = self._prev_handlers.get(signum)
        try:
            signal.signal(signum, prev if prev is not None else signal.SIG_DFL)
        except Exception:
            pass
        if prev is signal.SIG_IGN:
            # The process deliberately ignored this signal before we armed.
            # Restoring the GPU is ours to do; killing the process is not.
            return
        if callable(prev) and prev not in (signal.SIG_DFL, signal.SIG_IGN):
            prev(signum, frame)
        else:
            # Re-raise so the process still dies the way the sender intended.
            os.kill(os.getpid(), signum)

    # -- writes ----------------------------------------------------------

    def _mark_touched(self) -> None:
        """Record intent to disk BEFORE touching hardware."""
        assert self.state is not None
        self.state.restored = False
        self._restored = False
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
            self._write_depth += 1
            try:
                self._c.set_locked_clocks(min_mhz, max_mhz)
            finally:
                self._write_depth -= 1
                # A signal handler can run BETWEEN _mark_touched() and the line
                # above, re-enter restore() through the RLock, mark everything
                # clean, and then let this frame resume and apply the lock. Any
                # restore that overlapped this window is void.
                if self._restored:
                    self._restored = False
                    if self.state is not None:
                        self.state.restored = False
                        try:
                            _write_state_atomic(self._state_path, self.state)
                        except Exception:
                            pass

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
            self._write_depth += 1
            try:
                self._c.set_power_limit_mw(int(milliwatts))
            finally:
                self._write_depth -= 1
                if self._restored:
                    self._restored = False
                    if self.state is not None:
                        self.state.restored = False
                        try:
                            _write_state_atomic(self._state_path, self.state)
                        except Exception:
                            pass

    # -- restore ---------------------------------------------------------

    def restore(self) -> None:
        """Return the GPU to stock. Idempotent; safe to call any number of times.

        The clock reset is issued unconditionally rather than gated on a
        readback, because no readback exists.
        """
        with self._lock:
            self._restore_impl()

    def _restore_impl(self) -> None:
        """Restore body WITHOUT taking the lock.

        Split out so the signal path can proceed when the lock is held by
        another thread. ``restore()`` re-entering the same RLock would make a
        bounded acquire pointless -- the handler would simply block on the next
        line instead.
        """
        if True:
            if self._restored:
                return
            errors = []
            try:
                self._c.reset_locked_clocks()
            except Exception as exc:
                errors.append(f"reset_locked_clocks: {exc}")

            snap = self.snapshot
            # Prefer what the limit ACTUALLY was when we armed, not the factory
            # default: an operator may have set a deliberate cap before we ran,
            # and restoring past it would silently undo their configuration.
            target = snap.power_limit_mw if snap else None
            if target is None and snap is not None:
                target = snap.power_default_limit_mw
            if self.state and self.state.power_limit_written_mw is not None:
                if target is None:
                    # We changed the limit but never learned what it was (both
                    # NVML reads failed at arm() time). Silently skipping would
                    # record success while leaving the cap in place.
                    errors.append(
                        "power limit was modified but no restore target is "
                        "known (snapshot reads failed at arm time); the cap is "
                        "still in effect"
                    )
                else:
                    try:
                        self._c.set_power_limit_mw(int(target))
                    except Exception as exc:
                        errors.append(f"set_power_limit_mw: {exc}")

            if errors:
                # DO NOT mark restored. The device may still be locked, and with
                # no readback available the state file is the only evidence that
                # exists -- recording success here would erase it precisely when
                # it matters, and would disarm every remaining restore path
                # (atexit, signals, a later explicit retry) for this process.
                if self.state is not None:
                    self.state.restored = False
                    self.state.restore_failed = True
                    self.state.restore_errors = list(errors)
                    self.state.last_restore_attempt_utc = _utcnow()
                    try:
                        _write_state_atomic(self._state_path, self.state)
                    except Exception:
                        pass
                raise GpuGuardError(
                    "restore incomplete, GPU MAY STILL BE MODIFIED: "
                    + "; ".join(errors)
                    + f". State file {self._state_path} retains restored=false; "
                    "run `sudo python -m joule_agent.gpu_guard restore`."
                )

            self._restored = True
            if self.state is not None:
                self.state.restored = True
                self.state.restore_failed = False
                self.state.restore_errors = []
                self.state.clocks_locked = False
                self.state.restored_utc = _utcnow()
                try:
                    _write_state_atomic(self._state_path, self.state)
                except Exception:
                    pass

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
    records = _all_records(path)
    report = {
        "state_file": str(path),
        "state_file_present": path.exists(),
        "verified_power_modified": False,
        "asserted_clock_lock": False,
        "restore_failed": False,
        "stale": False,
        "findings": [],
        "recommendation": None,
    }

    # Tier 1: verifiable directly from the device.
    cur = controller.get_power_limit_mw()
    dfl = controller.get_power_default_limit_mw()
    report["power_readable"] = cur is not None and dfl is not None
    if not report["power_readable"]:
        report["findings"].append(
            "UNKNOWN (device): the power limit could not be read, so the "
            "verifiable tier did not run. A clean result here does NOT mean the "
            "power limit is at default."
        )
    if cur is not None and dfl is not None and cur != dfl:
        report["verified_power_modified"] = True
        report["stale"] = True
        # Deliberately does NOT claim this repo set the cap -- an operator may
        # have configured it before we ever ran. Restoring blindly would undo
        # their configuration, so the finding states the fact, not a cause.
        report["findings"].append(
            f"VERIFIED (device): power limit is {cur} mW, device default is "
            f"{dfl} mW. A cap is in effect. This may or may not have been set "
            "by this repo -- check the state file records below before restoring."
        )

    # Tier 2: assertion only -- the device cannot be asked about clock locks.
    unrestored = [r for r in records if not r.get("restored", True)]
    for rec in unrestored:
        report["stale"] = True
        pid = rec.get("pid")
        alive = _pid_alive(pid) if pid else False
        if rec.get("clocks_locked"):
            report["asserted_clock_lock"] = True
        if rec.get("restore_failed"):
            report["restore_failed"] = True
            report["findings"].append(
                f"RESTORE FAILED (state file): pid {pid} attempted a restore and "
                f"it errored: {rec.get('restore_errors')}. The device is very "
                "likely still modified."
            )
        report["findings"].append(
            f"ASSERTED (state file, NOT verifiable): pid {pid} recorded "
            f"restored=false"
            + (
                f" and a clock lock at [{rec.get('locked_min_mhz')}, "
                f"{rec.get('locked_max_mhz')}] MHz."
                if rec.get("clocks_locked")
                else "."
            )
            + (
                " That pid is still running -- it may simply be mid-run."
                if alive
                else " That pid is gone; it died without restoring."
            )
        )
        report["writer_pid_alive"] = alive
    report["unrestored_records"] = len(unrestored)

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
    controller: GpuController,
    state_path: Path | str = DEFAULT_STATE_PATH,
    *,
    force_power_default: bool = False,
) -> dict:
    """Recovery path for after a hard kill. Unconditional and idempotent."""
    doc = _read_doc(Path(state_path))
    records = list(doc.get("guards", {}).values())
    result = {"reset_clocks": False, "power_restored_to_mw": None,
              "records_cleared": 0, "errors": []}

    claim = getattr(controller, "claim", None)
    if callable(claim):
        claim("restore_from_state_file")

    try:
        controller.reset_locked_clocks()
        result["reset_clocks"] = True
    except Exception as exc:
        result["errors"].append(f"reset_locked_clocks: {exc}")

    # Restore power only if a record says WE changed it. Rewriting the limit
    # unconditionally would wipe an operator's deliberate pre-existing cap.
    wrote_power = [r for r in records if r.get("power_limit_written_mw") is not None]
    target = None
    for r in wrote_power:
        snap = r.get("snapshot") or {}
        target = snap.get("power_limit_mw")
        if target is None:
            target = snap.get("power_default_limit_mw")
        if target is not None:
            break
    if target is None and force_power_default:
        # Explicit operator override for when the state file was lost. Off by
        # default because it cannot distinguish our cap from a deliberate one.
        target = controller.get_power_default_limit_mw()
        result["power_forced_to_default"] = True
    if target is None and not wrote_power:
        result["power_untouched_reason"] = (
            "no state record shows this repo changed the power limit; leaving it "
            "alone so a deliberate operator cap is not silently undone. Pass "
            "--force-power-default to override."
        )
    if target is not None:
        try:
            controller.set_power_limit_mw(int(target))
            result["power_restored_to_mw"] = int(target)
        except Exception as exc:
            result["errors"].append(f"set_power_limit_mw: {exc}")

    if records and not result["errors"]:
        for rec in records:
            pid = rec.get("pid")
            if pid and pid != os.getpid() and _pid_alive(pid):
                # That process is still running and may still hold a lock.
                # Clearing its record would erase live evidence.
                result.setdefault("skipped_live_pids", []).append(pid)
                continue
            rec["restored"] = True
            rec["clocks_locked"] = False
            rec["restore_failed"] = False
            rec["restored_utc"] = _utcnow()
            rec["_recovered_by"] = "restore_from_state_file"
            result["records_cleared"] += 1
        try:
            _write_doc_atomic(Path(state_path), doc)
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
    rs = sub.add_parser("restore", help="return the device to stock (requires root)")
    rs.add_argument(
        "--force-power-default",
        action="store_true",
        help="also reset the power limit to the device default even when no "
             "state record shows this repo changed it (use if the state file "
             "was lost; may undo a deliberate operator cap)",
    )

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
            res = restore_from_state_file(
                c,
                args.state_path,
                force_power_default=getattr(args, "force_power_default", False),
            )
            print(json.dumps(res, indent=2))
            return 1 if res["errors"] else 0

        if args.cmd == "lock":
            g = GpuGuard(c, consent=args.consent, state_path=args.state_path)
            main._active_guard = g
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
        # Do NOT close while a guard still holds unrestored state: closing shuts
        # NVML down, and the atexit last-chance restore would then fail against a
        # dead handle -- converting a possible recovery into a certain failure.
        held = getattr(main, "_active_guard", None)
        if held is not None and not held._restored:
            print(
                "WARNING [gpu_guard]: leaving NVML open so the exit-time restore "
                "can still run.",
                file=sys.stderr,
            )
        else:
            c.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
