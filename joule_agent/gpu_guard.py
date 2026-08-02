"""GPU write guard: the only thing in this codebase that changes hardware state.

Contract, and it is the whole point of the module: **whatever happens -- clean
exit, unhandled exception, SIGINT, SIGTERM, or a hard kill -- the GPU returns to
stock clock and power settings.**

Nothing else may call ``nvmlDeviceSetGpuLockedClocks``,
``nvmlDeviceSetPowerManagementLimit``, ``nvidia-smi -lgc`` or ``nvidia-smi -pl``.
A safety net that lives inside the code it protects is not a net, because the
first buggy draft of that code runs on real hardware with nothing underneath it.

Deliberately single-thread, single-instance
-------------------------------------------
Two constraints are enforced rather than defended against:

* **One thread.** The guard must be armed on the main thread, and only that
  thread may write. A worker thread calling :meth:`GpuGuard.lock_clocks` is
  refused.
* **One instance per machine.** Arming is refused while any unrestored record
  exists in the state file -- whether the process that wrote it is alive
  (another guard is running) or dead (a previous run needs recovery first).

This is a narrowing, and it was chosen on evidence. Four review passes of an
earlier design found 27 defects; the large majority lived in machinery added to
make concurrent use safe -- an unlocked signal-path restore, write-depth and
generation counters, per-pid state merging, an inter-process ``flock``. That
apparatus generated defects faster than it retired hazards, and one of its own
fixes introduced a hang on the Ctrl+C path, which is the exact failure this
module exists to prevent. None of it was reachable by any real caller. A future
dynamic controller owns its guard on one thread.

The residual same-thread hazard is closed by construction, not by detection:
SIGINT and SIGTERM are **blocked for the duration of each hardware write** (see
:func:`_deferred_signals`). A signal arriving mid-write stays pending and is
delivered the instant the write completes, so a handler can never observe or act
on a half-applied write. That removes the need for in-flight bookkeeping.

What can and cannot be verified
-------------------------------
This driver exposes no ``nvmlDeviceGetGpuLockedClocks``. There is no way to ask
the device whether a clock lock is applied. Two consequences:

* **Restore is unconditional.** :meth:`GpuGuard.restore` always issues
  ``nvmlDeviceResetGpuLockedClocks`` rather than checking first. Resetting an
  unlocked GPU is a no-op, so this is idempotent for free and depends on no
  readback.
* **Stale detection has two tiers**, and :func:`check_stale_state` labels which
  is which. The power limit is *verifiable from the device*; a clock lock is
  *asserted by the state file only*.

Crash ordering
--------------
The state file is written and fsynced **before** the first hardware write, with
``restored: false``. That closes the window between locking a clock and
recording that we did. The flag flips to ``restored: true`` only after a restore
completes without error -- a failed restore leaves it false, because with no
readback the file is the only evidence that exists.

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
from contextlib import contextmanager
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
    "ThreadAffinityError",
    "GuardBusyError",
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
#: something else hides it.
MIN_SM_CLOCK_MHZ = 300

#: Required before the first hardware write in a process.
CONSENT_FLAG = "--i-understand-this-changes-gpu-settings"

#: Relative by default, so `gpu_guard restore` must run from the project root.
#: Override with JOULE_GUARD_STATE to make recovery cwd-independent.
DEFAULT_STATE_PATH = Path(
    os.environ.get("JOULE_GUARD_STATE", "results/gpu_guard_state.json")
)

STATE_SCHEMA = 3

#: Signals deferred across a hardware write.
_DEFERRED = (signal.SIGINT, signal.SIGTERM)


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


class ThreadAffinityError(GpuGuardError):
    """Armed or written from a thread that is not allowed to."""


class GuardBusyError(GpuGuardError):
    """Another guard holds the device, or a previous run never restored."""


@contextmanager
def _deferred_signals():
    """Block SIGINT/SIGTERM for the duration of a hardware write.

    A signal handler running between "record intent" and "issue the write"
    could reset the device and mark everything clean, after which the
    interrupted frame applies the setting -- GPU modified, state clean. Rather
    than detect that interleaving with in-flight bookkeeping, make it
    impossible: the signal stays pending and is delivered the moment the write
    completes, so the handler always sees a settled state.

    No-op where pthread_sigmask is unavailable.
    """
    if not hasattr(signal, "pthread_sigmask"):  # pragma: no cover - non-POSIX
        yield
        return
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, set(_DEFERRED))
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


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


def _same_owner(current, owner) -> bool:
    """True if `owner` may claim a controller `current` already holds.

    Identity first (guards are objects), then equality, because the recovery
    path claims with a string literal and `is` on strings only works via
    CPython interning.
    """
    if current is None or current is owner:
        return True
    try:
        return bool(current == owner)
    except Exception:
        return False


class _ClaimMixin:
    """Refuses hardware writes from a controller no guard is holding.

    A guard-rail against the common mistake of constructing a controller and
    calling it directly, which would bypass the clock floor, the consent flag
    and the privilege check. ``claim`` is public and Python offers no way to
    make it otherwise, so this is not a security boundary.
    """

    _claimed_by = None

    def claim(self, owner) -> None:
        if not _same_owner(self._claimed_by, owner):
            raise GpuGuardError(
                f"controller already claimed by {self._claimed_by!r}; refusing "
                "to hand the same device to a second owner"
            )
        self._claimed_by = owner

    def _require_claim(self) -> None:
        if self._claimed_by is None:
            raise GpuGuardError(
                "Refusing a direct hardware write: this controller is not held "
                "by a GpuGuard. Constructing a controller does not grant write "
                "access. Use `with gpu_guard(...)`."
            )


class NvmlGpuController(_ClaimMixin):
    """Real NVML. The only place in the codebase that writes GPU state."""

    def __init__(self, index: int = 0) -> None:
        import pynvml

        pynvml.nvmlInit()
        self._nvml = pynvml
        self._h = pynvml.nvmlDeviceGetHandleByIndex(index)
        self._index = index
        self._closed = False
        self._claimed_by = None

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
        self._write(
            lambda: self._nvml.nvmlDeviceSetGpuLockedClocks(self._h, min_mhz, max_mhz)
        )

    def reset_locked_clocks(self) -> None:
        self._write(lambda: self._nvml.nvmlDeviceResetGpuLockedClocks(self._h))

    def set_power_limit_mw(self, milliwatts: int) -> None:
        self._write(
            lambda: self._nvml.nvmlDeviceSetPowerManagementLimit(
                self._h, int(milliwatts)
            )
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
            # An exit-time retry can fire after close(); a swallowed
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
                    "NVML refused the write (NVML_ERROR_NO_PERMISSION). Clock "
                    "and power writes require root on Linux. Re-run under sudo. "
                    "This is a hard failure, not a no-op -- nothing changed."
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


class FakeGpuController(_ClaimMixin):
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
# Persisted state -- ONE record, because only one guard may arm at a time
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

    def to_dict(self) -> dict:
        d = asdict(self)
        d["_note"] = (
            "clocks_locked is ASSERTED by the writing process, not observed "
            "from the device -- this driver exposes no way to read back a clock "
            "lock. restored:false means a process left GPU state behind; run "
            "`python -m joule_agent.gpu_guard restore` (as root)."
        )
        return d


def _write_state(path: Path, state: GuardState) -> None:
    """Publish the record atomically, fsynced, before any hardware write.

    No inter-process locking and no merging: arming is refused while any
    unrestored record exists, so two guards never write concurrently by
    construction. Operator-initiated recovery is the only other writer and is a
    deliberate act.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"schema": STATE_SCHEMA, "record": state.to_dict()}
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


#: Result of reading the state file.
STATE_ABSENT = "absent"
STATE_OK = "ok"
STATE_UNREADABLE = "unreadable"


def _load_state(path: Path | str) -> tuple:
    """Return ``(record_or_None, status, extra)``. Never raises.

    Three states, kept distinct because "no file" and "a file I cannot make
    sense of" must not collapse into the same answer -- this module's doctrine
    is that the file is the only evidence a clock lock ever existed, so an
    unusable one is a reason to stop, not a reason to proceed.

    "Unreadable" covers more than invalid JSON: a document that parses but has
    no shape we recognise (wrong keys, a list where a mapping belongs, a future
    schema) is equally unusable. Validating here rather than at the call sites
    is what stops an ``AttributeError`` escaping the read-only ``status``
    command.
    """
    path = Path(path)
    if not path.exists():
        return None, STATE_ABSENT, {}
    try:
        doc = json.loads(path.read_text())
    except Exception as exc:
        return None, STATE_UNREADABLE, {"reason": f"invalid JSON: {exc}"}
    if not isinstance(doc, dict):
        return None, STATE_UNREADABLE, {"reason": "top level is not an object"}

    def _valid(rec):
        return isinstance(rec, dict) and "pid" in rec

    if "record" in doc:                                   # schema 3
        rec = doc["record"]
        if not _valid(rec):
            return None, STATE_UNREADABLE, {"reason": "'record' is not a record"}
        return rec, STATE_OK, {}

    if "guards" in doc:                                   # schema 2 (legacy)
        guards = doc["guards"]
        if not isinstance(guards, dict):
            return None, STATE_UNREADABLE, {"reason": "'guards' is not a mapping"}
        recs = [r for r in guards.values() if _valid(r)]
        if len(recs) != len(guards):
            return None, STATE_UNREADABLE, {"reason": "'guards' holds a non-record"}
        unrestored = [r for r in recs if not r.get("restored", True)]
        if len(unrestored) > 1:
            # The deleted per-pid design could produce this. Collapsing it to
            # one schema-3 record would destroy the others' evidence, and they
            # may name different devices.
            return None, STATE_UNREADABLE, {
                "reason": (
                    f"legacy schema-2 file holds {len(unrestored)} unrestored "
                    "records; collapsing it to a single record would destroy "
                    "evidence. Resolve by hand."
                )
            }
        if unrestored:
            return unrestored[0], STATE_OK, {"legacy_schema": 2}
        return (recs[0] if recs else None), (STATE_OK if recs else STATE_UNREADABLE), (
            {"legacy_schema": 2} if recs else {"reason": "'guards' is empty"}
        )

    if _valid(doc):                                       # schema 1 (legacy)
        return doc, STATE_OK, {"legacy_schema": 1}
    return None, STATE_UNREADABLE, {"reason": "no recognised schema"}


def _read_record(path: Path) -> dict | None:
    """Convenience wrapper. Callers that must distinguish "absent" from
    "unreadable" use :func:`_load_state` instead."""
    rec, _status, _extra = _load_state(path)
    return rec


def _controller_index(controller) -> int | None:
    idx = getattr(controller, "_index", None)
    if idx is not None:
        return idx
    try:
        return controller.snapshot().index
    except Exception:
        return None


def _pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


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

        self._depth = 0
        self._armed = False
        self._restored = True
        self._owner_thread: int | None = None
        self._prev_handlers: dict = {}
        self._atexit_registered = False
        self.handler_install_failures: list = []

        self.snapshot: DeviceSnapshot | None = None
        self.state: GuardState | None = None

    # -- refusals --------------------------------------------------------

    def _check_consent(self) -> None:
        if not self._consent:
            raise ConsentError(
                "Refusing to change GPU settings without explicit consent. Pass "
                f"consent=True (CLI: {CONSENT_FLAG}). Clock and power writes "
                "affect hardware shared with everything else on this machine. "
                "Reads never require it."
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

    def _check_owner_thread(self, action: str) -> None:
        if self._owner_thread is None:
            return
        if threading.get_ident() != self._owner_thread:
            raise ThreadAffinityError(
                f"Refusing to {action} from thread "
                f"{threading.current_thread().name!r}: this guard is owned by "
                "the thread that armed it. The guard is deliberately "
                "single-threaded -- a component that needs concurrency owns its "
                "guard on one thread. See the module docstring."
            )

    def _check_not_busy(self) -> None:
        prior, status, extra = _load_state(self._state_path)
        if status == STATE_UNREADABLE:
            # Arming would overwrite a file we could not interpret. If it held
            # an unrestored record, that is the only evidence the GPU is still
            # modified, and this module's whole doctrine is that the file is
            # the only evidence.
            raise GuardBusyError(
                f"Refusing to arm: {self._state_path} exists but could not be "
                f"interpreted ({extra.get('reason', 'unknown')}). Arming would "
                "overwrite it. Inspect it, then delete or repair it by hand, or "
                "run `sudo python -m joule_agent.gpu_guard restore "
                "--quarantine-unreadable`."
            )
        if not prior or prior.get("restored", True):
            return
        pid = prior.get("pid")
        if pid == os.getpid():
            # A guard cannot reach here about itself -- _armed is never reset,
            # so arm() short-circuits after the first call. So this is either a
            # SECOND GpuGuard object in this process, or this process's earlier
            # guard whose restore FAILED. Both must be refused; they need
            # different diagnoses.
            if prior.get("restore_failed"):
                raise GuardBusyError(
                    f"Refusing to arm: an earlier guard in this process (pid "
                    f"{pid}) FAILED its restore ({prior.get('restore_errors')}) "
                    "and the device may still be modified. Run `sudo python -m "
                    f"joule_agent.gpu_guard restore --gpu {prior.get('device_index', 0)}` "
                    "before arming again."
                )
            raise GuardBusyError(
                f"Refusing to arm: another GpuGuard in this process (pid {pid}) "
                f"holds an unrestored record in {self._state_path}. One guard "
                "per device -- a second would capture the first's modified "
                "settings as if they were stock, and restore to them."
            )
        if _pid_alive(pid):
            raise GuardBusyError(
                f"Refusing to arm: pid {pid} is running and holds an unrestored "
                f"guard record in {self._state_path}. Only one guard may hold "
                "the device at a time."
            )
        raise GuardBusyError(
            f"Refusing to arm: pid {pid} died without restoring, and its record "
            f"in {self._state_path} says the GPU may still be modified"
            + (
                f" (clock lock at [{prior.get('locked_min_mhz')}, "
                f"{prior.get('locked_max_mhz')}] MHz)."
                if prior.get("clocks_locked")
                else "."
            )
            + " Run `sudo python -m joule_agent.gpu_guard restore` first. "
            "Arming would overwrite the only evidence that exists."
        )

    # -- lifecycle -------------------------------------------------------

    def arm(self) -> "GpuGuard":
        """Snapshot stock state and install every restore path."""
        if self._armed:
            return self
        if threading.current_thread() is not threading.main_thread():
            raise ThreadAffinityError(
                "Refusing to arm off the main thread. Restore depends on signal "
                "handlers, which only the main thread can install, so arming "
                "elsewhere would silently degrade the guarantee to the state "
                "file alone. Arm on the main thread, or give the worker its own "
                "process."
            )
        self._check_not_busy()

        claim = getattr(self._c, "claim", None)
        if callable(claim):
            claim(self)
        self.snapshot = self._c.snapshot()
        self.state = GuardState(
            pid=os.getpid(),
            started_utc=_utcnow(),
            device_index=self.snapshot.index,
            device_uuid=self.snapshot.uuid,
            snapshot=self.snapshot.to_dict(),
            restored=True,
        )
        self._owner_thread = threading.get_ident()
        self._install_handlers()
        self._armed = True
        return self

    def _install_handlers(self) -> None:
        if not self._atexit_registered:
            atexit.register(self._atexit_restore)
            self._atexit_registered = True
        for sig in _DEFERRED:
            try:
                self._prev_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self._signal_restore)
            except (ValueError, OSError) as exc:  # pragma: no cover
                self.handler_install_failures.append(str(sig))
                print(
                    f"WARNING [gpu_guard]: could not install a {sig!r} handler "
                    f"({type(exc).__name__}: {exc}). In-process restore on that "
                    "signal is NOT armed; recovery depends on the state file at "
                    f"{self._state_path}.",
                    file=sys.stderr,
                )

    def _atexit_restore(self) -> None:
        try:
            self.restore()
        except Exception as exc:
            # Cannot re-raise usefully at interpreter shutdown, but silence
            # would hide a GPU left in a modified state.
            print(
                f"ERROR [gpu_guard]: restore failed during exit: {exc}",
                file=sys.stderr,
            )

    def _signal_restore(self, signum, frame) -> None:
        # Hardware writes run with these signals blocked, so this handler can
        # never observe a half-applied write. No in-flight bookkeeping needed.
        try:
            self.restore()
        except Exception as exc:
            print(
                f"ERROR [gpu_guard]: restore failed handling {signum}: {exc}",
                file=sys.stderr,
            )

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
        _write_state(self._state_path, self.state)

    def lock_clocks(self, min_mhz: int, max_mhz: int | None = None) -> None:
        max_mhz = min_mhz if max_mhz is None else max_mhz
        if not self._armed:
            self.arm()
        self._check_owner_thread("lock clocks")
        self._check_consent()
        # Argument validation BEFORE the environment check: a bad clock value is
        # the caller's bug and is free to detect, so report it even when
        # unprivileged. Checking root first would send an operator to sudo and
        # only then reveal the real problem.
        self._check_clock_floor(min_mhz, max_mhz)
        self._check_privilege()

        assert self.state is not None
        self.state.clocks_locked = True
        self.state.locked_min_mhz = min_mhz
        self.state.locked_max_mhz = max_mhz
        with _deferred_signals():
            self._mark_touched()
            self._c.set_locked_clocks(min_mhz, max_mhz)

    def set_power_limit_mw(self, milliwatts: int) -> None:
        if not self._armed:
            self.arm()
        self._check_owner_thread("set the power limit")
        self._check_consent()
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
        self._check_privilege()

        assert self.state is not None
        self.state.power_limit_written_mw = int(milliwatts)
        with _deferred_signals():
            self._mark_touched()
            self._c.set_power_limit_mw(int(milliwatts))

    # -- restore ---------------------------------------------------------

    def restore(self) -> None:
        """Return the GPU to stock. Idempotent; safe to call any number of times.

        The clock reset is issued unconditionally rather than gated on a
        readback, because no readback exists.

        **Deliberately has no thread-affinity check.** Every in-repo caller
        (``__exit__``, ``atexit``, the signal handler) runs on the arming
        thread, and adding a check would let a restore be REFUSED, which would
        break the module's whole contract. The premise is therefore documented
        rather than enforced: do not call this from a worker thread while the
        owning thread is mid-write -- signal blocking cannot protect against
        another thread.
        """
        if self._restored:
            return
        errors = []
        try:
            self._c.reset_locked_clocks()
        except Exception as exc:
            errors.append(f"reset_locked_clocks: {exc}")

        snap = self.snapshot
        # Prefer what the limit ACTUALLY was when we armed, not the factory
        # default: an operator may have set a deliberate cap before we ran.
        target = snap.power_limit_mw if snap else None
        if target is None and snap is not None:
            target = snap.power_default_limit_mw
        if self.state and self.state.power_limit_written_mw is not None:
            if target is None:
                errors.append(
                    "power limit was modified but no restore target is known "
                    "(snapshot reads failed at arm time); the cap is still in "
                    "effect"
                )
            else:
                try:
                    self._c.set_power_limit_mw(int(target))
                except Exception as exc:
                    errors.append(f"set_power_limit_mw: {exc}")

        if errors:
            # DO NOT mark restored. With no readback, the state file is the only
            # evidence that exists -- recording success would erase it exactly
            # when it matters, and disarm every remaining restore path.
            if self.state is not None:
                self.state.restored = False
                self.state.restore_failed = True
                self.state.restore_errors = list(errors)
                try:
                    _write_state(self._state_path, self.state)
                except Exception:
                    pass
            raise GpuGuardError(
                "restore incomplete, GPU MAY STILL BE MODIFIED: "
                + "; ".join(errors)
                + f". State file {self._state_path} retains restored=false; run "
                "`sudo python -m joule_agent.gpu_guard restore`."
            )

        self._restored = True
        if self.state is not None:
            self.state.restored = True
            self.state.restore_failed = False
            self.state.restore_errors = []
            self.state.clocks_locked = False
            self.state.restored_utc = _utcnow()
            try:
                _write_state(self._state_path, self.state)
            except Exception:
                pass

    # -- context manager -------------------------------------------------

    def __enter__(self) -> "GpuGuard":
        self.arm()
        self._depth += 1
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
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
# Stale state and recovery
# ---------------------------------------------------------------------------


def check_stale_state(
    controller: GpuController, state_path: Path | str = DEFAULT_STATE_PATH
) -> dict:
    """Detect GPU state left behind by a process that died without restoring.

    Separates what the device confirms from what only the state file asserts.
    """
    path = Path(state_path)
    prior, status, extra = _load_state(path)
    dev = _controller_index(controller)
    report = {
        "state_file": str(path),
        "state_file_present": path.exists(),
        "state_file_unreadable": status == STATE_UNREADABLE,
        "device_index": dev,
        "record_device_index": (prior or {}).get("device_index"),
        "record_is_for_this_device": None,
        "power_readable": False,
        "verified_power_modified": False,
        "asserted_clock_lock": False,
        "restore_failed": False,
        "stale": False,
        "findings": [],
        "recommendation": None,
    }

    if report["state_file_unreadable"]:
        report["stale"] = True
        report["findings"].append(
            "UNREADABLE (state file): the file exists but could not be "
            f"interpreted ({extra.get('reason', 'unknown')}), so the asserted "
            "tier could not run. Treat as possibly-modified rather than clean."
        )

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
    elif cur != dfl:
        report["verified_power_modified"] = True
        report["stale"] = True
        # Deliberately does NOT claim this repo set the cap -- an operator may
        # have configured it before we ever ran.
        report["findings"].append(
            f"VERIFIED (device): power limit is {cur} mW, device default is "
            f"{dfl} mW. A cap is in effect. This may or may not have been set by "
            "this repo -- check the record below before restoring."
        )

    # Tier 2: assertion only -- the device cannot be asked about clock locks.
    if prior is not None:
        rec_dev = prior.get("device_index")
        report["record_is_for_this_device"] = (
            None if (dev is None or rec_dev is None) else rec_dev == dev
        )
    if prior and not prior.get("restored", True) and report["record_is_for_this_device"] is False:
        # Tier 1 above describes the device this controller holds; this record
        # describes a different one. Saying "stale" without saying which device
        # would send an operator to reset the wrong GPU.
        report["stale"] = True
        report["findings"].append(
            f"OTHER DEVICE (state file): the unrestored record names GPU "
            f"{prior.get('device_index')}, but this report was produced against "
            f"GPU {dev}. Nothing here is evidence about GPU {dev}'s clock lock. "
            f"Re-run with --gpu {prior.get('device_index')}."
        )
        report["recommendation"] = (
            "Run `sudo python -m joule_agent.gpu_guard restore --gpu "
            f"{prior.get('device_index')}` -- recovery must target the device "
            "the record names."
        )
        return report
    if prior and not prior.get("restored", True):
        report["stale"] = True
        pid = prior.get("pid")
        alive = _pid_alive(pid) if pid else False
        report["writer_pid_alive"] = alive
        report["asserted_clock_lock"] = bool(prior.get("clocks_locked"))
        if prior.get("restore_failed"):
            report["restore_failed"] = True
            report["findings"].append(
                f"RESTORE FAILED (state file): pid {pid} attempted a restore and "
                f"it errored: {prior.get('restore_errors')}. The device is very "
                "likely still modified."
            )
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
                " That pid is still running -- it may simply be mid-run."
                if alive
                else " That pid is gone; it died without restoring."
            )
        )

    if report["stale"]:
        if report.get("writer_pid_alive"):
            report["recommendation"] = (
                "A guard process is STILL RUNNING and owns this record. Prefer "
                "stopping it -- its own restore will run on exit. Recovery will "
                "reset the clock but will refuse to clear a live guard's record."
            )
        else:
            target = report.get("record_device_index")
            if target is None:
                target = dev if dev is not None else 0
            report["recommendation"] = (
                f"Run `sudo python -m joule_agent.gpu_guard restore --gpu "
                f"{target}` to return the device to stock. The --gpu flag "
                "matters: recovery applies the record's snapshot to whatever "
                "device it is pointed at. Safe even if nothing is actually "
                "locked -- the clock reset is a no-op on an unlocked GPU."
            )
    return report


def restore_from_state_file(
    controller: GpuController,
    state_path: Path | str = DEFAULT_STATE_PATH,
    *,
    force_power_default: bool = False,
    quarantine_unreadable: bool = False,
) -> dict:
    """Recovery path for after a hard kill. Unconditional and idempotent."""
    prior, status, extra = _load_state(state_path)
    dev = _controller_index(controller)
    result = {
        "reset_clocks": False,
        "power_restored_to_mw": None,
        "record_cleared": False,
        "incomplete": False,
        "errors": [],
    }

    if status == STATE_UNREADABLE:
        if quarantine_unreadable:
            stamp = _utcnow().replace(":", "").replace("-", "")
            dest = Path(state_path).with_suffix(f".corrupt-{stamp}")
            try:
                os.replace(Path(state_path), dest)
                result["quarantined_to"] = str(dest)
            except Exception as exc:
                result["errors"].append(f"quarantine: {exc}")
        else:
            # Without this, `status` reports stale forever and `restore` exits 0
            # having changed nothing -- a recommendation that cannot resolve the
            # condition it recommends for.
            result["incomplete"] = True
            result["errors"].append(
                f"{state_path} exists but could not be interpreted "
                f"({extra.get('reason', 'unknown')}). The clock reset below "
                "still applies, but the record cannot be cleared. Inspect the "
                "file, then re-run with --quarantine-unreadable to move it "
                "aside so `status` can come clean."
            )

    # A record for another GPU must not be applied to this one: the snapshot's
    # power value belongs to that device, and on a homogeneous node NVML will
    # accept it silently.
    rec_dev = (prior or {}).get("device_index")
    wrong_device = prior is not None and dev is not None and rec_dev is not None and rec_dev != dev
    if wrong_device:
        result["incomplete"] = True
        result["record_device_index"] = rec_dev
        result["errors"].append(
            f"the record names GPU {rec_dev} but recovery was pointed at GPU "
            f"{dev}. Refusing to write that device's snapshot here or to clear "
            f"its record. Re-run with --gpu {rec_dev}."
        )
        prior = None      # do not use it for power, do not clear it

    claim = getattr(controller, "claim", None)
    if callable(claim):
        claim("restore_from_state_file")

    try:
        controller.reset_locked_clocks()
        result["reset_clocks"] = True
    except Exception as exc:
        result["errors"].append(f"reset_locked_clocks: {exc}")

    # Restore power only if the record says WE changed it. Rewriting the limit
    # unconditionally would wipe an operator's deliberate pre-existing cap.
    wrote_power = bool(prior and prior.get("power_limit_written_mw") is not None)
    target = None
    if wrote_power:
        snap = (prior or {}).get("snapshot") or {}
        target = snap.get("power_limit_mw")
        if target is None:
            target = snap.get("power_default_limit_mw")
    if target is None and force_power_default:
        target = controller.get_power_default_limit_mw()
        result["power_forced_to_default"] = True
    if target is None and wrote_power:
        result["errors"].append(
            "the record shows this repo changed the power limit, but no restore "
            "target is recoverable from its snapshot; the cap is still in "
            "effect. Re-run with --force-power-default to reset to the device "
            "default."
        )
    elif target is None:
        result["power_untouched_reason"] = (
            "no state record shows this repo changed the power limit; leaving it "
            "alone so a deliberate operator cap is not silently undone. Pass "
            "--force-power-default to override."
        )
    else:
        try:
            controller.set_power_limit_mw(int(target))
            result["power_restored_to_mw"] = int(target)
        except Exception as exc:
            result["errors"].append(f"set_power_limit_mw: {exc}")

    # Do not clear a record a live guard still owns, and do not publish a
    # read that went stale while the hardware calls above were running. Arming
    # is refused over a live record precisely because that destroys evidence;
    # recovery must not do the same destruction silently.
    if prior is not None:
        holder = prior.get("pid")
        if holder and holder != os.getpid() and _pid_alive(holder):
            result["record_not_cleared_reason"] = (
                f"pid {holder} is still running and owns this record; the clock "
                "reset above still applied, but the record is left alone so a "
                "live guard's evidence is not destroyed. Stop that process (its "
                "own restore will run) or re-run recovery afterwards."
            )
            return result
        current = _read_record(Path(state_path))
        if current != prior:
            result["record_not_cleared_reason"] = (
                "the state record changed while recovery was running (a guard "
                "wrote to it); refusing to publish a stale read over it. The "
                "clock reset above still applied -- re-run recovery."
            )
            # Says "re-run recovery", so it must not report success: a script
            # doing `restore && echo ok` would otherwise call this a clean run.
            result["incomplete"] = True
            return result

    if prior is not None and not result["errors"]:
        prior = dict(prior)
        prior["restored"] = True
        prior["clocks_locked"] = False
        prior["restore_failed"] = False
        prior["restored_utc"] = _utcnow()
        prior["_recovered_by"] = "restore_from_state_file"
        try:
            path = Path(state_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
            with open(tmp, "w") as fh:
                json.dump({"schema": STATE_SCHEMA, "record": prior}, fh, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
            result["record_cleared"] = True
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
        "--quarantine-unreadable",
        action="store_true",
        help="if the state file cannot be interpreted, move it aside to a "
             ".corrupt-<timestamp> file so `status` can come clean; the "
             "original is preserved for inspection",
    )
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
    guard = None
    try:
        if args.cmd == "status":
            rep = check_stale_state(c, args.state_path)
            print(json.dumps(rep, indent=2))
            return 1 if rep["stale"] else 0

        if args.cmd == "restore":
            res = restore_from_state_file(
                c,
                args.state_path,
                force_power_default=args.force_power_default,
                quarantine_unreadable=args.quarantine_unreadable,
            )
            print(json.dumps(res, indent=2))
            return 1 if (res["errors"] or res.get("incomplete")) else 0

        if args.cmd == "lock":
            guard = GpuGuard(c, consent=args.consent, state_path=args.state_path)
            with guard:
                guard.lock_clocks(args.mhz)
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
        # Do NOT close while the guard still holds unrestored state: close()
        # shuts NVML down, and the exit-time restore would then fail against a
        # dead handle -- converting a possible recovery into a certain failure.
        if guard is not None and not guard._restored:
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
