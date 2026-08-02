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
* **One instance per machine.** Arming first takes a nonblocking lifetime lease
  at one fixed, process-independent path, then refuses while any unrestored
  record exists -- whether the process that wrote it is alive (another guard is
  running) or dead (a previous run needs recovery first). The lease is not
  derived from the caller-selected state path, so two paths cannot create two
  owners of the same hardware.

This is a narrowing, and it was chosen on evidence. Four review passes of an
earlier design found 27 defects; the large majority lived in machinery added to
make concurrent use safe -- an unlocked signal-path restore, write-depth and
generation counters, per-pid state merging, and a blocking ``flock`` acquired
again inside state writes. That apparatus generated defects faster than it
retired hazards, and one of its own fixes introduced a hang on the Ctrl+C path,
which is the exact failure this module exists to prevent. The narrow lifetime
guard lifetime lease used now is acquired only by :meth:`GpuGuard.arm`, never
reacquired by a write, in-process restore, or signal handler, so those paths
cannot self-deadlock on it. Operator crash recovery takes the same fixed lease
once around its complete read -> hardware -> publish transaction. A future
dynamic controller owns its guard on one thread.

The residual same-thread hazard is closed by a small write transaction. SIGINT
and SIGTERM are blocked on the owner thread during each hardware write, and an
in-progress flag also catches Python handlers scheduled from a signal delivered
to another unblocked worker thread. Such handlers are latched, then drained only
after the hardware call and signal-mask restoration settle. A handler therefore
cannot mark the record clean and release ownership before an interrupted frame
applies its hardware write.

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
import copy
import errno
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

#: A fixed machine-wide lease closes the otherwise caller-controlled namespace
#: created by ``state_path``. Lock an existing, root-owned runtime directory
#: rather than a replaceable file in /tmp: unlinking and recreating a lock file
#: creates a second inode with an independent flock. ``flock`` is advisory, so
#: this does not impede ordinary users of /run; only other Joule processes ask
#: for the same lock.
MACHINE_LEASE_PATH = Path("/run")

#: Persistent, machine-scoped recovery evidence. ``/run`` is reset by reboot
#: (which also resets GPU state) and is writable only by the privileged process
#: that is allowed to perform hardware writes. Tests replace this constant with
#: a per-case path; production callers cannot select it through ``state_path``.
MACHINE_STATE_PATH = Path("/run/joule-agent/gpu_guard_state.json")

_SCHEMA3_REQUIRED_FIELDS = frozenset(
    {
        "pid",
        "started_utc",
        "device_index",
        "device_uuid",
        "snapshot",
        "clocks_locked",
        "locked_min_mhz",
        "locked_max_mhz",
        "power_limit_written_mw",
        "restored",
        "restored_utc",
        "restore_failed",
        "restore_errors",
    }
)

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
    interrupted frame applies the setting -- GPU modified, state clean. POSIX
    masks are per-thread, so this is only one half of the construction:
    :class:`GpuGuard` also latches Python handlers that become pending through
    another unblocked worker. The latch remains active while this context
    restores the old mask, because unmasking itself may run a pending handler.

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


_RECOVERY_FORK_GATE = threading.RLock()
_RECOVERY_LEASE_FD: int | None = None
_RECOVERY_LEASE_PID: int | None = None
_RECOVERY_ATFORK_REGISTERED = False


def _recovery_before_fork() -> None:
    _RECOVERY_FORK_GATE.acquire()


def _recovery_after_fork_parent() -> None:
    _RECOVERY_FORK_GATE.release()


def _recovery_after_fork_child() -> None:
    """Discard recovery authority and raw-write access copied by ``fork``."""
    global _RECOVERY_LEASE_FD, _RECOVERY_LEASE_PID
    try:
        # The thread-local is defined later in the module but always exists by
        # the time a public recovery call can register this callback.
        _CONTROLLER_ACCESS.owner = None
        fd = _RECOVERY_LEASE_FD
        _RECOVERY_LEASE_FD = None
        _RECOVERY_LEASE_PID = None
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
    finally:
        _RECOVERY_FORK_GATE.release()


def _register_recovery_atfork_cleanup() -> None:
    global _RECOVERY_ATFORK_REGISTERED
    with _RECOVERY_FORK_GATE:
        if _RECOVERY_ATFORK_REGISTERED:
            return
        register = getattr(os, "register_at_fork", None)
        if not callable(register):  # pragma: no cover - NVIDIA target is Linux
            raise GpuGuardError(
                "Recovery requires fork-child lease cleanup on this platform."
            )
        register(
            before=_recovery_before_fork,
            after_in_parent=_recovery_after_fork_parent,
            after_in_child=_recovery_after_fork_child,
        )
        _RECOVERY_ATFORK_REGISTERED = True


def _require_recovery_process_owner() -> None:
    """Reject a recovery frame that continued in a fork child."""
    if (
        _RECOVERY_LEASE_FD is None
        or _RECOVERY_LEASE_PID is None
        or _RECOVERY_LEASE_PID != os.getpid()
    ):
        raise GpuGuardError(
            "Refusing recovery work in a fork child or without the live "
            "machine recovery lease."
        )


@contextmanager
def _standalone_machine_lease():
    """Reserve the machine during operator-initiated crash recovery.

    The published descriptor is also covered by an at-fork hook. A library
    callback that forks during recovery therefore cannot leave a child holding
    the parent's flock or inheriting its thread-local raw-write capability.
    """
    global _RECOVERY_LEASE_FD, _RECOVERY_LEASE_PID
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - NVIDIA target is POSIX
        raise GpuGuardError(
            "Recovery requires the same POSIX machine lease as guarded writes."
        ) from exc

    _register_recovery_atfork_cleanup()
    path = Path(MACHINE_LEASE_PATH)
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = None
    owner_pid = os.getpid()
    try:
        with _RECOVERY_FORK_GATE:
            fd = os.open(path, flags)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                os.close(fd)
                fd = None
                if exc.errno in (errno.EACCES, errno.EAGAIN):
                    raise GuardBusyError(
                        "Refusing crash recovery: a live GpuGuard or another "
                        f"recovery owns the machine lease {path}."
                    ) from exc
                raise
            _RECOVERY_LEASE_FD = fd
            _RECOVERY_LEASE_PID = owner_pid
        yield
    finally:
        # The child hook already closed its copied descriptor. Do not let the
        # unwinding child frame close an unrelated descriptor that later reused
        # the same integer.
        if fd is not None and os.getpid() == owner_pid:
            with _RECOVERY_FORK_GATE:
                if (
                    _RECOVERY_LEASE_FD == fd
                    and _RECOVERY_LEASE_PID == owner_pid
                ):
                    _RECOVERY_LEASE_FD = None
                    _RECOVERY_LEASE_PID = None
                os.close(fd)


# ---------------------------------------------------------------------------
# Device abstraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
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


_DEVICE_SNAPSHOT_FIELDS = frozenset(DeviceSnapshot.__dataclass_fields__)
_SNAPSHOT_NUMERIC_FIELDS = frozenset(
    {
        "power_limit_mw",
        "power_default_limit_mw",
        "power_min_mw",
        "power_max_mw",
        "max_sm_clock_mhz",
        "sm_clock_at_snapshot_mhz",
        "default_applications_sm_clock_mhz",
    }
)


def _snapshot_shape_problem(snapshot, *, require_complete: bool) -> str | None:
    """Return why persisted snapshot data cannot be durable recovery evidence."""
    if not isinstance(snapshot, dict):
        return "snapshot is not an object"
    if require_complete:
        missing = sorted(_DEVICE_SNAPSHOT_FIELDS.difference(snapshot))
        if missing:
            return "snapshot is missing fields: " + ", ".join(missing)
    index = snapshot.get("index")
    if index is not None and (
        not isinstance(index, int) or isinstance(index, bool) or index < 0
    ):
        return f"snapshot index {index!r} is not a nonnegative integer"
    uuid = snapshot.get("uuid")
    if uuid is not None and (not isinstance(uuid, str) or not uuid):
        return f"snapshot UUID {uuid!r} is not a nonempty string"
    name = snapshot.get("name")
    if name is not None and not isinstance(name, str):
        return f"snapshot name {name!r} is not text"
    for field_name in _SNAPSHOT_NUMERIC_FIELDS:
        value = snapshot.get(field_name)
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            return (
                f"snapshot {field_name}={value!r} is not a nonnegative integer"
            )
    return None


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


_CONTROLLER_ACCESS = threading.local()
_RECOVERY_OWNER = object()


_CONTROLLER_WRITE_OPERATIONS = frozenset(
    {"set_locked_clocks", "reset_locked_clocks", "set_power_limit_mw"}
)


@contextmanager
def _controller_write_access(
    owner,
    operation: str,
    arguments: tuple = (),
):
    """Grant one method-scoped, one-shot raw controller capability.

    Controller methods are arbitrary Python callbacks.  An owner-only token
    would let a clock-reset callback issue an unrecorded power write (or repeat
    its own write) before returning.  Bind the capability to exactly one raw
    operation and consume it in the concrete controller method instead.
    """
    if operation not in _CONTROLLER_WRITE_OPERATIONS:
        raise GpuGuardError(f"unknown controller write operation {operation!r}")
    if owner is _RECOVERY_OWNER:
        _require_recovery_process_owner()
    else:
        process_check = getattr(owner, "_check_process_owner", None)
        if callable(process_check):
            process_check(f"enter raw {operation} callback")
    previous = (
        getattr(_CONTROLLER_ACCESS, "owner", None),
        getattr(_CONTROLLER_ACCESS, "operation", None),
        getattr(_CONTROLLER_ACCESS, "arguments", ()),
        getattr(_CONTROLLER_ACCESS, "consumed", False),
    )
    _CONTROLLER_ACCESS.owner = owner
    _CONTROLLER_ACCESS.operation = operation
    _CONTROLLER_ACCESS.arguments = tuple(arguments)
    _CONTROLLER_ACCESS.consumed = False
    try:
        yield
        if not getattr(_CONTROLLER_ACCESS, "consumed", False):
            raise GpuGuardError(
                f"Refusing to accept {operation} callback success: the exact "
                "one-shot raw hardware capability was never consumed."
            )
    finally:
        (
            _CONTROLLER_ACCESS.owner,
            _CONTROLLER_ACCESS.operation,
            _CONTROLLER_ACCESS.arguments,
            _CONTROLLER_ACCESS.consumed,
        ) = previous


def _same_owner(current, owner) -> bool:
    """True only for an unclaimed controller or the identical owner token.

    Equality is attacker-controlled Python code. Accepting it lets an object
    whose ``__eq__`` always returns true impersonate the active guard token and
    grant raw writes outside the guarded transaction.
    """
    return current is None or current is owner


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

    def _require_claim(
        self,
        operation: str | None = None,
        arguments: tuple = (),
        *,
        consume: bool = False,
    ) -> None:
        active_owner = getattr(_CONTROLLER_ACCESS, "owner", None)
        if self._claimed_by is None or not _same_owner(
            self._claimed_by, active_owner
        ):
            raise GpuGuardError(
                "Refusing a direct hardware write: raw controller methods are "
                "enabled only for the active GpuGuard/recovery transaction. "
                "Constructing or manually claiming a controller does not grant "
                "write access. Use `with gpu_guard(...)`."
            )
        if operation is not None:
            permitted = getattr(_CONTROLLER_ACCESS, "operation", None)
            permitted_arguments = getattr(_CONTROLLER_ACCESS, "arguments", ())
            consumed = getattr(_CONTROLLER_ACCESS, "consumed", False)
            actual_arguments = tuple(arguments)
            arguments_match = (
                len(actual_arguments) == len(permitted_arguments)
                and all(
                    type(actual) is int
                    and type(expected) is int
                    and actual == expected
                    for actual, expected in zip(
                        actual_arguments, permitted_arguments
                    )
                )
            )
            if permitted != operation or not arguments_match:
                raise GpuGuardError(
                    f"Refusing raw {operation}{actual_arguments!r}: this "
                    "callback capability authorizes only "
                    f"{permitted or 'no operation'}{permitted_arguments!r}."
                )
            if consumed:
                raise GpuGuardError(
                    f"Refusing a repeated raw {operation}: each guarded "
                    "callback capability authorizes exactly one hardware call."
                )
            if consume:
                _CONTROLLER_ACCESS.consumed = True


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
            "set_locked_clocks",
            (min_mhz, max_mhz),
            lambda: self._nvml.nvmlDeviceSetGpuLockedClocks(self._h, min_mhz, max_mhz)
        )

    def reset_locked_clocks(self) -> None:
        self._write(
            "reset_locked_clocks",
            (),
            lambda: self._nvml.nvmlDeviceResetGpuLockedClocks(self._h),
        )

    def set_power_limit_mw(self, milliwatts: int) -> None:
        if type(milliwatts) is not int:
            raise GpuGuardError(
                "raw power-limit hardware calls require an exact built-in int"
            )
        self._write(
            "set_power_limit_mw",
            (milliwatts,),
            lambda: self._nvml.nvmlDeviceSetPowerManagementLimit(
                self._h, milliwatts
            )
        )

    def get_power_limit_mw(self) -> int | None:
        return self._try(lambda: self._nvml.nvmlDeviceGetPowerManagementLimit(self._h))

    def get_power_default_limit_mw(self) -> int | None:
        return self._try(
            lambda: self._nvml.nvmlDeviceGetPowerManagementDefaultLimit(self._h)
        )

    def _write(self, operation: str, arguments: tuple, fn) -> None:
        self._require_claim(operation, arguments, consume=True)
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
        index: int = 0,
        uuid: str = "GPU-FAKE",
        power_min_mw: int | None = 100_000,
        power_max_mw: int | None = 200_000,
    ) -> None:
        self._index = index
        self._uuid = uuid
        self._power = power_limit_mw
        self._default = power_default_limit_mw
        self._power_min = power_min_mw
        self._power_max = power_max_mw
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
            index=self._index,
            name="FakeGPU",
            uuid=self._uuid,
            power_limit_mw=self._power,
            power_default_limit_mw=self._default,
            power_min_mw=self._power_min,
            power_max_mw=self._power_max,
            max_sm_clock_mhz=2100,
            sm_clock_at_snapshot_mhz=210,
            default_applications_sm_clock_mhz=None,
        )

    def set_locked_clocks(self, min_mhz: int, max_mhz: int) -> None:
        self._require_claim(
            "set_locked_clocks", (min_mhz, max_mhz), consume=True
        )
        self.set_clock_calls += 1
        self.locked = (min_mhz, max_mhz)

    def reset_locked_clocks(self) -> None:
        self._require_claim("reset_locked_clocks", (), consume=True)
        self.reset_calls += 1
        if self.fail_restore:
            return  # deliberately does not clear `locked`
        self.locked = None

    def set_power_limit_mw(self, milliwatts: int) -> None:
        if type(milliwatts) is not int:
            raise GpuGuardError(
                "raw power-limit hardware calls require an exact built-in int"
            )
        self._require_claim(
            "set_power_limit_mw", (milliwatts,), consume=True
        )
        self.set_power_calls += 1
        self._power = milliwatts

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
    #: Operator/project mirror of the fixed machine-wide recovery record.
    evidence_path: str | None = None
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


def _write_record_document(path: Path, record: dict) -> None:
    """Atomically publish one complete schema-3 record and fsync it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"schema": STATE_SCHEMA, "record": record}
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


def _record_document_bytes(record: dict) -> bytes:
    """Serialize a recoverable record for preservation inside a blocker."""
    return json.dumps(
        {"schema": STATE_SCHEMA, "record": record}, indent=2
    ).encode()


def _write_unreadable_machine_marker(
    payload: bytes | None,
    *,
    alternate_payload: bytes | None = None,
    reason: str | None = None,
    authority_conflict: bool = False,
    evidence_path: Path | str | None = None,
    machine_path: Path | str | None = None,
) -> None:
    """Keep unknown debt globally blocking until explicitly resolved.

    The marker wraps and preserves the source bytes instead of trying to parse
    or choose them. The exact original is also retained at the quarantine
    destination when quarantine was requested.
    """
    path = Path(MACHINE_STATE_PATH if machine_path is None else machine_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        {
            "_joule_machine_blocker": 1,
            "authority_conflict": authority_conflict,
            "reason": reason
            or (
                "conflicting recovery authorities"
                if authority_conflict
                else "unreadable recovery authority"
            ),
            "preserved_evidence_hex": (
                payload.hex() if payload is not None else None
            ),
            "alternate_evidence_hex": (
                alternate_payload.hex()
                if alternate_payload is not None
                else None
            ),
            "evidence_path": (
                str(evidence_path) if evidence_path is not None else None
            ),
        },
        indent=2,
    ).encode()
    tmp = path.with_suffix(
        f"{path.suffix}.{os.getpid()}.{time.time_ns()}.corrupt.tmp"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(tmp, flags, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        dfd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _write_quarantine_copy(path: Path, payload: bytes) -> None:
    """Durably preserve exact evidence without first removing its authority."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        dfd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    finally:
        os.close(fd)


def _write_state(path: Path, state: GuardState) -> None:
    """Publish the record atomically, fsynced, before any hardware write.

    No state-file transaction lock and no merging: ``GpuGuard.arm`` holds the
    machine lease for the guard lifetime, while operator crash recovery holds
    that same lease once around its whole transaction. Neither path reacquires
    it inside a state write or signal handler.
    """
    _write_record_document(path, state.to_dict())


#: Result of reading the state file.
STATE_ABSENT = "absent"
STATE_OK = "ok"
STATE_UNREADABLE = "unreadable"


def _record_state_semantic_problem(rec: dict) -> str | None:
    """Return a cross-field contradiction in an otherwise typed record."""
    restored = rec.get("restored")
    restore_failed = rec.get("restore_failed", False)
    restore_errors = rec.get("restore_errors", [])
    if bool(restore_errors) != restore_failed:
        return "restore_failed does not agree with restore_errors"
    if restored:
        if rec.get("clocks_locked", False):
            return "a clean record still asserts a clock lock"
        if restore_failed or restore_errors:
            return "a clean record still asserts a failed restore"
        restored_utc = rec.get("restored_utc")
        if not isinstance(restored_utc, str) or not restored_utc:
            return "a clean record lacks its restore completion timestamp"
    else:
        if rec.get("restored_utc") is not None:
            return "an active record already carries a restore timestamp"
        if (
            not rec.get("clocks_locked", False)
            and rec.get("power_limit_written_mw") is None
            and not restore_failed
        ):
            return "an active record has no hardware-debt marker"
    return None


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
    if doc.get("_joule_machine_blocker") == 1:
        return None, STATE_UNREADABLE, {
            "reason": str(doc.get("reason") or "machine-wide safety blocker"),
            "authority_conflict": bool(doc.get("authority_conflict", False)),
            "blocked_evidence_path": doc.get("evidence_path"),
        }

    def _valid(rec, *, current_schema=False):
        if not isinstance(rec, dict) or "pid" not in rec or "restored" not in rec:
            return False
        # Do not let malformed JSON exploit Python truthiness. In particular,
        # "restored": "false" is truthy and would erase recovery authority,
        # while a truthy list snapshot reaches `.get()` and crashes recovery.
        pid = rec.get("pid")
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            return False
        if not isinstance(rec["restored"], bool):
            return False
        if current_schema and not _SCHEMA3_REQUIRED_FIELDS.issubset(rec):
            return False
        if "snapshot" in rec and not isinstance(rec["snapshot"], dict):
            return False
        if "clocks_locked" in rec and not isinstance(rec["clocks_locked"], bool):
            return False
        for field_name in ("locked_min_mhz", "locked_max_mhz"):
            value = rec.get(field_name)
            if value is not None and (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                return False
        if "restore_failed" in rec and not isinstance(rec["restore_failed"], bool):
            return False
        if "restore_errors" in rec:
            errors = rec["restore_errors"]
            if not isinstance(errors, list) or not all(
                isinstance(item, str) for item in errors
            ):
                return False
        if "started_utc" in rec and not isinstance(rec["started_utc"], str):
            return False
        if "evidence_path" in rec:
            evidence_path = rec["evidence_path"]
            if evidence_path is not None and (
                not isinstance(evidence_path, str) or not evidence_path
            ):
                return False
        if "restored_utc" in rec:
            restored_utc = rec["restored_utc"]
            if restored_utc is not None and not isinstance(restored_utc, str):
                return False
        marker = rec.get("power_limit_written_mw")
        if marker is not None and (
            not isinstance(marker, int) or isinstance(marker, bool) or marker < 0
        ):
            return False
        snapshot = rec.get("snapshot") or {}
        if _snapshot_shape_problem(
            snapshot, require_complete=current_schema
        ) is not None:
            return False
        for field_name in ("power_limit_mw", "power_default_limit_mw"):
            value = snapshot.get(field_name)
            if value is not None and (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                return False
        snapshot_index = snapshot.get("index")
        if snapshot_index is not None and (
            not isinstance(snapshot_index, int)
            or isinstance(snapshot_index, bool)
            or snapshot_index < 0
        ):
            return False
        snapshot_uuid = snapshot.get("uuid")
        if snapshot_uuid is not None and (
            not isinstance(snapshot_uuid, str) or not snapshot_uuid
        ):
            return False
        index = None
        if "device_index" in rec:
            index = rec["device_index"]
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
            ):
                return False
        uuid = None
        if "device_uuid" in rec:
            uuid = rec["device_uuid"]
            if uuid is not None and (not isinstance(uuid, str) or not uuid):
                return False
        if uuid is not None and snapshot_uuid is not None and uuid != snapshot_uuid:
            return False
        if (
            index is not None
            and snapshot_index is not None
            and index != snapshot_index
        ):
            return False
        if current_schema and (
            not isinstance(uuid, str)
            or not uuid
            or snapshot_uuid != uuid
            or snapshot_index != index
        ):
            return False
        if current_schema and _record_state_semantic_problem(rec):
            return False
        if not current_schema and not rec["restored"]:
            # Legacy records may omit completion metadata, but an active one
            # still needs a hardware marker unless it explicitly records a
            # failed clean publication. Complete legacy candidates face the
            # full current semantic gate before promotion below.
            if (
                not rec.get("clocks_locked", False)
                and rec.get("power_limit_written_mw") is None
                and not rec.get("restore_failed", False)
            ):
                return False
        return True

    missing_schema = object()
    schema = doc.get("schema", missing_schema)
    if schema is missing_schema:
        if "record" in doc or "guards" in doc:
            return None, STATE_UNREADABLE, {
                "reason": "wrapped state document has no declared schema"
            }
        if _valid(doc):                                   # schema 1 (legacy)
            return doc, STATE_OK, {"legacy_schema": 1}
        return None, STATE_UNREADABLE, {"reason": "no recognised schema"}

    if not isinstance(schema, int) or isinstance(schema, bool):
        return None, STATE_UNREADABLE, {"reason": "schema is not an integer"}

    if schema == STATE_SCHEMA:                            # schema 3
        if "record" not in doc or "guards" in doc:
            return None, STATE_UNREADABLE, {
                "reason": "schema 3 requires only the 'record' envelope"
            }
        rec = doc["record"]
        if not _valid(rec, current_schema=True):
            return None, STATE_UNREADABLE, {
                "reason": "schema-3 'record' is incomplete or malformed"
            }
        return rec, STATE_OK, {}

    if schema == 2:                                       # schema 2 (legacy)
        if "guards" not in doc or "record" in doc:
            return None, STATE_UNREADABLE, {
                "reason": "schema 2 requires only the 'guards' envelope"
            }
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

    if schema == 1:                                       # schema 1 (legacy)
        if "record" in doc or "guards" in doc or not _valid(doc):
            return None, STATE_UNREADABLE, {
                "reason": "schema-1 flat record is malformed"
            }
        return doc, STATE_OK, {"legacy_schema": 1}

    return None, STATE_UNREADABLE, {
        "reason": f"unsupported schema {schema}"
    }


def _same_guard_lineage(left: dict, right: dict) -> bool:
    """Compare fields that do not change when one run becomes clean."""
    fields = (
        "pid",
        "started_utc",
        "device_index",
        "device_uuid",
        "snapshot",
        "locked_min_mhz",
        "locked_max_mhz",
        "power_limit_written_mw",
    )
    # ``evidence_path`` is intentionally excluded because promotion may add it
    # between machine-first and mirror-second publication. Every actual lineage
    # field remains strict: wildcarding a missing legacy UUID/snapshot/started
    # value could collapse an unrelated record solely because its PID matches.
    return all(left.get(field) == right.get(field) for field in fields)


def _load_authoritative_state(
    path: Path | str,
    *,
    machine_path: Path | str | None = None,
) -> tuple:
    """Load caller evidence plus the fixed machine-wide recovery mirror.

    An active machine record always wins over a different caller-selected path.
    Without that rule, SIGKILL would release the flock and a new caller could
    select another empty state file, snapshot modified hardware as stock, and
    erase the original recovery debt.
    """
    requested = Path(path)
    prior, status, extra = _load_state(requested)
    machine = Path(MACHINE_STATE_PATH if machine_path is None else machine_path)
    try:
        same_path = requested.resolve() == machine.resolve()
    except OSError:
        same_path = requested.absolute() == machine.absolute()
    if same_path:
        return prior, status, {**extra, "source_path": str(requested)}

    machine_prior, machine_status, machine_extra = _load_state(machine)
    if machine_status == STATE_UNREADABLE:
        return None, STATE_UNREADABLE, {
            "reason": (
                f"machine-wide evidence {machine} is unreadable: "
                f"{machine_extra.get('reason', 'unknown')}"
            ),
            "source_path": str(machine),
            "authority_conflict": bool(
                machine_extra.get("authority_conflict", False)
            ),
            "blocked_evidence_path": machine_extra.get(
                "blocked_evidence_path"
            ),
        }

    machine_active = bool(
        machine_prior and not machine_prior.get("restored", True)
    )
    requested_active = bool(prior and not prior.get("restored", True))
    if machine_active:
        if status == STATE_UNREADABLE:
            return None, STATE_UNREADABLE, {
                "reason": (
                    "machine-wide evidence is active while the caller-selected "
                    "file is unreadable; refusing to assume the unknown file is "
                    "only a damaged copy of the canonical debt"
                ),
                "source_path": str(machine),
                "requested_path": str(requested),
                "authority_conflict": True,
            }
        if requested_active and prior != machine_prior:
            if _same_guard_lineage(machine_prior, prior):
                return machine_prior, STATE_OK, {
                    "source_path": str(machine),
                    "machine_wide": True,
                    "torn_active_mirror": str(requested),
                    "requested_path": str(requested),
                }
            return None, STATE_UNREADABLE, {
                "reason": (
                    "caller-selected and machine-wide files contain different "
                    "unrestored records; refusing to choose one and destroy the "
                    "other's recovery authority"
                ),
                "source_path": str(machine),
                "authority_conflict": True,
            }
        return machine_prior, STATE_OK, {
            "source_path": str(machine),
            "machine_wide": True,
            "requested_path": str(requested),
        }

    if machine_prior is not None and requested_active:
        return None, STATE_UNREADABLE, {
            "reason": (
                "machine-wide evidence is clean but the caller-selected path "
                "contains active recovery authority"
                + (
                    " from the same apparent lineage"
                    if _same_guard_lineage(machine_prior, prior)
                    else " from a different lineage"
                )
                + "; no crash ordering in this writer publishes clean machine "
                "evidence before an active project mirror, so refusing to "
                "guess which timeline is authoritative"
            ),
            "source_path": str(machine),
            "authority_conflict": True,
        }

    return prior, status, {**extra, "source_path": str(requested)}


def _read_record(path: Path) -> dict | None:
    """Convenience wrapper. Callers that must distinguish "absent" from
    "unreadable" use :func:`_load_state` instead."""
    rec, _status, _extra = _load_state(path)
    return rec


def _controller_identity(controller) -> tuple:
    """Return ``(index, uuid)`` for the device a controller holds.

    UUID is the authoritative half. Device *index* is an enumeration order, not
    an identity: it can shift across a reboot, a driver reload, or a change to
    CUDA_VISIBLE_DEVICES. A record written before such a change can name index 0
    and mean a different physical card, so an index match is not proof and a
    UUID mismatch overrides it.
    """
    idx = getattr(controller, "_index", None)
    uuid = None
    try:
        snap = controller.snapshot()
        uuid = snap.uuid
        if idx is None:
            idx = snap.index
    except Exception:
        pass
    return idx, uuid


def _canonicalize_record(
    record: dict,
    controller,
    *,
    evidence_path: Path | str,
    preserve_recorded_identity: bool = False,
) -> dict:
    """Normalize legacy/additive records before machine-wide publication."""
    if preserve_recorded_identity:
        # Recovery selection has not passed the physical-identity gate yet.
        # Filling a missing UUID from the selected controller here would turn
        # an unverifiable record into apparent proof that it belongs to that
        # controller.
        live_index, live_uuid = None, None
    else:
        live_index, live_uuid = _controller_identity(controller)
    canonical = GuardState(started_utc=_utcnow()).to_dict()
    canonical.update(record)
    canonical["evidence_path"] = str(evidence_path)
    if record.get("device_index") is None and live_index is not None:
        canonical["device_index"] = live_index
    if not record.get("device_uuid") and live_uuid:
        canonical["device_uuid"] = live_uuid
    if not preserve_recorded_identity:
        snapshot = dict(canonical.get("snapshot") or {})
        if (
            snapshot.get("index") is None
            and canonical.get("device_index") is not None
        ):
            snapshot["index"] = canonical["device_index"]
        if snapshot.get("uuid") is None and canonical.get("device_uuid"):
            snapshot["uuid"] = canonical["device_uuid"]
        canonical["snapshot"] = snapshot
    return canonical


def _recorded_identity_problem(record: dict) -> str | None:
    """Validate that the snapshot itself names the record's physical device."""
    snapshot = record.get("snapshot") or {}
    rec_uuid = record.get("device_uuid")
    snap_uuid = snapshot.get("uuid")
    rec_idx = record.get("device_index")
    snap_idx = snapshot.get("index")
    if not rec_uuid or not snap_uuid:
        return (
            "the record and its pre-write snapshot do not both carry a device "
            "UUID, so the snapshot's physical identity is unverifiable"
        )
    if rec_uuid != snap_uuid:
        return (
            f"record device UUID {rec_uuid} disagrees with snapshot UUID "
            f"{snap_uuid}"
        )
    if rec_idx is None or snap_idx is None:
        return (
            "the record and its pre-write snapshot do not both carry the "
            "captured device index"
        )
    if rec_idx != snap_idx:
        return (
            f"record device index {rec_idx} disagrees with snapshot index "
            f"{snap_idx}"
        )
    return None


def _recorded_lineage_problem(record: dict) -> str | None:
    """Require enough recorded structure for crash-safe mirror promotion."""
    identity_problem = _recorded_identity_problem(record)
    if identity_problem:
        return identity_problem
    snapshot_problem = _snapshot_shape_problem(
        record.get("snapshot"), require_complete=True
    )
    if snapshot_problem:
        return snapshot_problem
    missing = sorted(_SCHEMA3_REQUIRED_FIELDS.difference(record))
    if missing:
        return (
            "the legacy record lacks complete stable lineage fields: "
            + ", ".join(missing)
        )
    if record.get("restored") is not False:
        return "the candidate is not active recovery evidence"
    semantic_problem = _record_state_semantic_problem(record)
    if semantic_problem:
        return f"the legacy record is semantically contradictory: {semantic_problem}"
    return None


def _device_mismatch(record: dict, controller) -> str | None:
    """Describe why `record` is not about `controller`'s device, or None.

    Protected surface: every restore that applies a snapshot must pass this
    first. Applying one device's stock values to another silently mis-caps the
    wrong card and clears the right card's evidence.
    """
    recorded_problem = _recorded_identity_problem(record)
    if recorded_problem:
        return recorded_problem
    idx, uuid = _controller_identity(controller)
    rec_uuid = record.get("device_uuid")
    rec_idx = record.get("device_index")
    if not rec_uuid:
        return (
            f"the record has no device UUID, so physical identity cannot be "
            f"verified; recorded GPU index {rec_idx} and live index {idx} are "
            "only enumeration positions"
        )
    if rec_uuid and not uuid:
        # A UUID-bearing record must never fall back to index.  Index is only an
        # enumeration order, so accepting it when the authoritative identity
        # read failed recreates the wrong-device recovery hazard this gate
        # exists to prevent.
        return (
            f"the record names device UUID {rec_uuid}, but this controller's "
            f"UUID could not be read; GPU index {idx} is not proof of identity"
        )
    if uuid and rec_uuid and uuid != rec_uuid:
        return (
            f"the record is for device {rec_uuid} but this controller holds "
            f"{uuid}"
            + (
                f" (both report index {idx}; the enumeration changed, so the "
                "index is not proof of identity)"
                if rec_idx == idx
                else f" (record index {rec_idx}, controller index {idx})"
            )
        )
    if uuid and rec_uuid and uuid == rec_uuid:
        return None                     # UUID match is authoritative
    return None


def _identity_unverifiable(mismatch: str | None) -> bool:
    return bool(
        mismatch
        and (
            "UUID could not be read" in mismatch
            or "no device UUID" in mismatch
            or "unverifiable" in mismatch
            or "do not both carry" in mismatch
        )
    )


def _pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # kill(pid, 0) returning EPERM proves the process exists; this caller
        # simply lacks permission to signal it (common for a root-owned guard).
        return True
    except OSError as exc:
        # Fail closed for uncommon platform errors. Only ESRCH proves absence.
        return exc.errno != errno.ESRCH
    except (ValueError, TypeError):
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
        # Resolve once. If cwd changes after construction, the state record and
        # its lease must not silently drift into different directories.
        self._state_path = Path(state_path).resolve()
        self._require_root = require_root
        if (
            not isinstance(min_clock_mhz, int)
            or isinstance(min_clock_mhz, bool)
            or min_clock_mhz < MIN_SM_CLOCK_MHZ
        ):
            raise ClockFloorError(
                f"Configured clock floor {min_clock_mhz!r} may not lower the "
                f"hard safety floor of {MIN_SM_CLOCK_MHZ} MHz."
            )
        self._min_clock = min_clock_mhz

        self._depth = 0
        self._armed = False
        self._restored = True
        self._lease_fd: int | None = None
        # Do not resolve this path: following a pre-created symlink before the
        # O_NOFOLLOW open would defeat the lease-file fail-closed check.
        self._lease_path = Path(MACHINE_LEASE_PATH)
        # Freeze the authoritative evidence path for this guard's lifetime too;
        # module configuration must not drift between intent and restore.
        self._machine_state_path = Path(MACHINE_STATE_PATH)
        self._armed_pid: int | None = None
        self._forked_from_pid: int | None = None
        self._atfork_registered = False
        self._fork_gate = threading.Lock()
        self._write_depth = 0
        self._restore_depth = 0
        self._pending_signals: list[tuple[int, object, bool]] = []
        self._draining_signals = False
        self._owner_thread: int | None = None
        self._prev_handlers: dict = {}
        self._atexit_registered = False
        self.handler_install_failures: list = []

        # The objects that authorize restore are private. ``snapshot`` is a
        # frozen public value and ``state`` returns a deep copy, so caller code
        # cannot erase a power marker or substitute a different restore target
        # after we have persisted intent.
        self._snapshot: DeviceSnapshot | None = None
        self._state: GuardState | None = None
        self._identity_conflict = False

    @property
    def snapshot(self) -> DeviceSnapshot | None:
        return self._snapshot

    @property
    def state(self) -> GuardState | None:
        return copy.deepcopy(self._state)

    # -- refusals --------------------------------------------------------

    def _check_consent(self) -> None:
        if self._consent is not True:
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
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in (min_mhz, max_mhz)
        ):
            raise ClockFloorError(
                f"Refusing clock lock [{min_mhz!r}, {max_mhz!r}] MHz: both "
                "bounds must be integer MHz values so the fsynced evidence and "
                "the hardware request cannot disagree."
            )
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

    def _acquire_lifetime_lease(self) -> None:
        """Atomically reserve this machine's GPU write surface until restore.

        This is intentionally not a state-file transaction lock. It is acquired
        once, nonblocking, by ``arm`` and is never reacquired from a hardware
        write, restore, or signal handler. That distinction prevents the
        signal-path self-deadlock caused by the deleted per-write lock design.
        """
        self._register_atfork_cleanup()

        try:
            import fcntl
        except ImportError as exc:  # pragma: no cover - NVIDIA target is POSIX
            raise GpuGuardError(
                "Refusing to arm: this platform cannot provide the required "
                "single-instance file lease. Running without it would allow "
                "two guards to snapshot and modify the same GPU."
            ) from exc

        fd = None
        # A second thread may call fork even though the guard itself is
        # single-threaded. Its at-fork `before` callback takes this same gate,
        # so no child can be created between os.open() and publishing the fd on
        # self where the child cleanup callback can see it.
        with self._fork_gate:
            try:
                # Block the handled signals across the tiny acquire/assignment
                # handoff. If SIGINT becomes pending, arm() cleanup sees the
                # assigned fd and releases it; SIGKILL closes it in-kernel.
                with _deferred_signals():
                    flags = os.O_RDONLY
                    if hasattr(os, "O_DIRECTORY"):
                        flags |= os.O_DIRECTORY
                    if hasattr(os, "O_NOFOLLOW"):
                        flags |= os.O_NOFOLLOW
                    fd = os.open(self._lease_path, flags)
                    self._lease_fd = fd
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except OSError as exc:
                        if exc.errno in (errno.EACCES, errno.EAGAIN):
                            raise GuardBusyError(
                                "Refusing to arm: another GpuGuard in this "
                                "process or another process already holds the "
                                f"lifetime lease {self._lease_path}."
                            ) from exc
                        raise GpuGuardError(
                            "Refusing to arm: could not establish the required "
                            f"single-instance lease {self._lease_path}: {exc}"
                        ) from exc
            except BaseException:
                # Includes KeyboardInterrupt between flock() and return. Never
                # let an unsuccessful arm retain ownership in a survivor.
                if self._lease_fd == fd:
                    self._lease_fd = None
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                raise

    def _register_atfork_cleanup(self) -> None:
        """Ensure a fork child cannot inherit ownership of the parent GPU."""
        if self._atfork_registered:
            return
        register = getattr(os, "register_at_fork", None)
        if not callable(register):  # pragma: no cover - NVIDIA target is Linux
            raise GpuGuardError(
                "Refusing to arm: this platform cannot discard the lifetime "
                "lease in a fork child, so a child could inherit GPU ownership."
            )
        try:
            register(
                before=self._before_fork,
                after_in_parent=self._after_fork_in_parent,
                after_in_child=self._after_fork_in_child,
            )
        except Exception as exc:  # pragma: no cover - platform/runtime failure
            raise GpuGuardError(
                "Refusing to arm: could not install the required fork-safety "
                f"hook: {exc}"
            ) from exc
        self._atfork_registered = True

    def _before_fork(self) -> None:
        self._fork_gate.acquire()

    def _after_fork_in_parent(self) -> None:
        self._fork_gate.release()

    def _after_fork_in_child(self) -> None:
        """Drop the copied fd without restoring hardware owned by the parent."""
        try:
            _CONTROLLER_ACCESS.owner = None
            fd = self._lease_fd
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
                self._lease_fd = None
                self._forked_from_pid = self._armed_pid or os.getppid()
        finally:
            self._fork_gate.release()

    def _check_process_owner(self, action: str) -> None:
        owner = self._armed_pid
        if self._forked_from_pid is not None or (
            owner is not None and owner != os.getpid()
        ):
            inherited_from = self._forked_from_pid or owner
            raise GpuGuardError(
                f"Refusing to {action}: this GpuGuard was inherited across "
                f"fork from pid {inherited_from}. A child must construct and "
                "arm its own guard after the parent releases ownership."
            )

    def _release_lifetime_lease(self) -> None:
        """Release ownership after a clean/no-touch restore; never unlink it."""
        fd = self._lease_fd
        if fd is None:
            return
        with _deferred_signals():
            try:
                os.close(fd)  # closing releases flock, including on SIGKILL
            finally:
                self._lease_fd = None

    def _check_not_busy(self) -> None:
        prior, status, extra = _load_authoritative_state(
            self._state_path,
            machine_path=self._machine_state_path,
        )
        source_path = Path(extra.get("source_path", self._state_path))
        machine_path = self._machine_state_path
        if status == STATE_UNREADABLE:
            # Discovery under path A must block a later path B even though this
            # arm is about to refuse and release its flock. Conflicts replace M
            # with a tagged wrapper preserving M's exact bytes; ordinary corrupt
            # path evidence is preserved in the same global blocker.
            _machine_prior, machine_status, machine_extra = _load_state(
                machine_path
            )
            if machine_status != STATE_UNREADABLE:
                payload_path = (
                    machine_path
                    if extra.get("authority_conflict")
                    else source_path
                )
                try:
                    try:
                        payload = payload_path.read_bytes()
                    except OSError:
                        payload = None
                    _write_unreadable_machine_marker(
                        payload,
                        reason=extra.get("reason", "unknown recovery authority"),
                        authority_conflict=bool(extra.get("authority_conflict")),
                        evidence_path=self._state_path,
                        machine_path=machine_path,
                    )
                except Exception as exc:
                    raise GuardBusyError(
                        "Refusing to arm, and could not make the discovered "
                        f"recovery debt machine-wide: {exc}"
                    ) from exc
            # Arming would overwrite a file we could not interpret. If it held
            # an unrestored record, that is the only evidence the GPU is still
            # modified, and this module's whole doctrine is that the file is
            # the only evidence.
            raise GuardBusyError(
                f"Refusing to arm: recovery evidence at {source_path} could not be "
                f"interpreted ({extra.get('reason', 'unknown')}). Arming would "
                "overwrite it. Inspect and repair it by hand. "
                "`restore --quarantine-unreadable` can preserve a forensic copy, "
                "but intentionally will not clear unknown machine-wide debt."
            )
        if not prior or prior.get("restored", True):
            return

        if os.path.abspath(source_path) != os.path.abspath(machine_path):
            identity_problem = _recorded_lineage_problem(prior)
            try:
                if identity_problem:
                    try:
                        payload = source_path.read_bytes()
                    except OSError:
                        payload = None
                    _write_unreadable_machine_marker(
                        payload,
                        reason=identity_problem,
                        evidence_path=source_path,
                        machine_path=machine_path,
                    )
                else:
                    promoted = _canonicalize_record(
                        prior,
                        self._c,
                        evidence_path=prior.get("evidence_path") or source_path,
                        preserve_recorded_identity=True,
                    )
                    promoted["restored"] = False
                    _write_record_document(machine_path, promoted)
                    _write_record_document(source_path, promoted)
                    prior = promoted
            except Exception as exc:
                raise GuardBusyError(
                    "Refusing to arm, and could not promote the discovered "
                    f"active recovery debt to {machine_path}: {exc}"
                ) from exc
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
            f"in {source_path} says the GPU may still be modified"
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
        self._check_process_owner("arm")
        if self._armed:
            if self._lease_fd is not None:
                return self
            raise GpuGuardError(
                "Refusing to re-arm a completed GpuGuard object. Guard objects "
                "are one-shot so an old snapshot cannot be reused after its "
                "single-instance lease is released; construct a fresh guard."
            )
        if threading.current_thread() is not threading.main_thread():
            raise ThreadAffinityError(
                "Refusing to arm off the main thread. Restore depends on signal "
                "handlers, which only the main thread can install, so arming "
                "elsewhere would silently degrade the guarantee to the state "
                "file alone. Arm on the main thread, or give the worker its own "
                "process."
            )
        try:
            # Keep handled signals pending until ownership, snapshot, handlers,
            # and the public armed flag are one committed transition. A pending
            # ignored/custom signal can run on unmask and perform a no-touch
            # restore, so verify below that it did not surrender the lease.
            with _deferred_signals():
                self._acquire_lifetime_lease()
                self._check_not_busy()

                claim = getattr(self._c, "claim", None)
                if callable(claim):
                    claim(self)
                observed = self._c.snapshot()
                try:
                    observed_data = observed.to_dict()
                except Exception as exc:
                    raise GpuGuardError(
                        "Refusing to arm because the pre-write snapshot could "
                        f"not become durable recovery evidence: {exc}"
                    ) from exc
                snapshot_problem = _snapshot_shape_problem(
                    observed_data, require_complete=True
                )
                if snapshot_problem:
                    raise GpuGuardError(
                        "Refusing to arm because the pre-write snapshot cannot "
                        "become readable recovery evidence: "
                        f"{snapshot_problem}."
                    )
                self._snapshot = DeviceSnapshot(**observed_data)
                self._state = GuardState(
                    pid=os.getpid(),
                    started_utc=_utcnow(),
                    evidence_path=str(self._state_path),
                    device_index=self._snapshot.index,
                    device_uuid=self._snapshot.uuid,
                    snapshot=self._snapshot.to_dict(),
                    restored=True,
                )
                self._owner_thread = threading.get_ident()
                self._armed_pid = os.getpid()
                self._install_handlers()
                self._armed = True
            if self._lease_fd is None:
                raise GpuGuardError(
                    "Refusing to report arm success: arming was interrupted by "
                    "a signal or restore that already released ownership. This "
                    "guard is completed; construct a fresh guard."
                )
        except BaseException:
            self._release_lifetime_lease()
            raise
        return self

    def _install_handlers(self) -> None:
        installed = []
        for sig in _DEFERRED:
            try:
                self._prev_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self._signal_restore)
                installed.append(sig)
            except (ValueError, OSError) as exc:  # pragma: no cover
                self.handler_install_failures.append(str(sig))
                for installed_sig in reversed(installed):
                    try:
                        signal.signal(
                            installed_sig,
                            self._prev_handlers.get(installed_sig, signal.SIG_DFL),
                        )
                    except Exception:
                        pass
                raise GpuGuardError(
                    f"Refusing to arm: could not install the required {sig!r} "
                    f"restore handler ({type(exc).__name__}: {exc}). Running "
                    "with only state-file recovery would violate the guard's "
                    "SIGINT/SIGTERM restore contract."
                ) from exc
        if not self._atexit_registered:
            atexit.register(self._atexit_restore)
            self._atexit_registered = True

    def _atexit_restore(self) -> None:
        # A fork child inherited this callback but not ownership. Its at-fork
        # hook already closed the copied lease; touching hardware here would
        # reset settings while the parent guard is still actively using them.
        if self._forked_from_pid is not None or (
            self._armed_pid is not None and self._armed_pid != os.getpid()
        ):
            return
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
        # A process-directed signal can be delivered to an unblocked worker and
        # still schedule this Python handler on the main thread while that
        # thread's POSIX mask is blocked. Always latch first. The drain runs
        # immediately when settled, or from the transaction's finally block
        # after a write/restore and mask-unwind have both completed.
        # Recovery debt remains the authoritative "operation is live" bit even
        # in the bytecode-sized window after a transaction clears its depth and
        # before the public method returns. A returning signal there must still
        # abort rather than restore hardware and let the write report success.
        interrupted_operation = bool(
            self._write_depth or self._restore_depth or not self._restored
        )
        for index, (sig, old_frame, was_interrupted) in enumerate(
            self._pending_signals
        ):
            if sig == signum:
                self._pending_signals[index] = (
                    sig,
                    old_frame,
                    was_interrupted or interrupted_operation,
                )
                break
        else:
            self._pending_signals.append(
                (signum, frame, interrupted_operation)
            )
        if self._write_depth or self._restore_depth:
            return
        self._drain_pending_signals()

    def _drain_pending_signals(self) -> None:
        if self._draining_signals or self._write_depth or self._restore_depth:
            return
        self._draining_signals = True
        interruption_errors = []
        try:
            while self._pending_signals:
                signum, frame, interrupted_operation = self._pending_signals.pop(0)
                interruption = self._finish_signal(
                    signum, frame, interrupted_operation
                )
                if interruption is not None:
                    interruption_errors.append(interruption)
        finally:
            self._draining_signals = False
        if interruption_errors:
            if len(interruption_errors) == 1:
                raise interruption_errors[0]
            raise GpuGuardError(
                "GPU operation was interrupted by multiple returning signals: "
                + "; ".join(str(error) for error in interruption_errors)
            )

    def _finish_signal(
        self, signum, frame, interrupted_operation=False
    ) -> GpuGuardError | None:
        """Restore settled hardware, then preserve the prior disposition."""
        restore_error = None
        try:
            self.restore()
        except Exception as exc:
            restore_error = exc
            print(
                f"ERROR [gpu_guard]: restore failed handling {signum}: {exc}",
                file=sys.stderr,
            )

        prev = self._prev_handlers.get(signum)
        try:
            signal.signal(signum, prev if prev is not None else signal.SIG_DFL)
        except Exception:
            pass
        try:
            if prev is signal.SIG_IGN:
                # The process deliberately ignored this signal before we armed.
                # Restoring the GPU is ours to do; killing the process is not.
                if interrupted_operation:
                    return self._signal_interruption_error(
                        signum, "the prior disposition ignored it", restore_error
                    )
                return None
            if callable(prev) and prev not in (signal.SIG_DFL, signal.SIG_IGN):
                prev(signum, frame)
                if interrupted_operation:
                    return self._signal_interruption_error(
                        signum, "the prior handler returned", restore_error
                    )
            else:
                # Re-raise so the process still dies the way the sender intended.
                os.kill(os.getpid(), signum)
            return None
        finally:
            # A returning/ignored prior disposition leaves the process alive.
            # If restore failed, keep the guard handler installed so a later
            # signal gets another in-process attempt instead of bypassing
            # straight to the old disposition while hardware remains modified.
            if restore_error is not None and not self._restored:
                try:
                    signal.signal(signum, self._signal_restore)
                except Exception as exc:
                    print(
                        "ERROR [gpu_guard]: could not re-arm failed signal "
                        f"restore handler for {signum}: {exc}",
                        file=sys.stderr,
                    )

    @staticmethod
    def _signal_interruption_error(signum, disposition, restore_error):
        if restore_error is None:
            return GpuGuardError(
                f"GPU operation interrupted by signal {signum}; {disposition}, "
                "so the guard restored hardware and aborted the operation "
                "rather than report false success."
            )
        error = GpuGuardError(
            f"GPU operation interrupted by signal {signum}; {disposition}, "
            "but the required restore FAILED and the GPU MAY STILL BE MODIFIED: "
            f"{restore_error}"
        )
        error.__cause__ = restore_error
        return error

    # -- writes ----------------------------------------------------------

    def _publish_owned_state(self) -> None:
        """Publish active machine-first; publish clean machine-last.

        Machine-first closes the pre-write crash window. Machine-last closes
        the opposite clean-publication window: a cut may leave the canonical
        record conservatively active, but can never leave it clean while a
        stale active project mirror still has recovery authority.
        """
        assert self._state is not None
        machine_path = self._machine_state_path
        paths_differ = os.path.abspath(machine_path) != os.path.abspath(
            self._state_path
        )
        if self._state.restored and paths_differ:
            _write_state(self._state_path, self._state)
            _write_state(machine_path, self._state)
        else:
            _write_state(machine_path, self._state)
            if paths_differ:
                _write_state(self._state_path, self._state)

    @contextmanager
    def _hardware_write_transaction(self):
        """Keep signal-driven restore behind one complete hardware write."""
        if self._write_depth or self._restore_depth:
            raise GpuGuardError(
                "Refusing a nested GPU hardware write while another write or "
                "restore transaction is active. The guard write surface is "
                "deliberately non-reentrant."
            )
        self._write_depth = 1
        try:
            # Keep the latch set through __exit__: restoring the old POSIX mask
            # may itself run a handler made pending through this or another
            # thread.
            with _deferred_signals():
                yield
        finally:
            self._write_depth = 0
            self._drain_pending_signals()

    def _check_write_identity(self) -> None:
        self._check_process_owner("verify live GPU identity")
        snap = self._snapshot
        state = self._state
        if self._identity_conflict:
            raise GpuGuardError(
                "Refusing a GPU hardware write because this guard already "
                "observed uncertain or changed controller identity. The "
                "machine-wide safety blocker must be resolved manually."
            )
        if snap is None or not isinstance(snap.uuid, str) or not snap.uuid:
            raise GpuGuardError(
                "Refusing the first GPU hardware write because the live device "
                "UUID was unavailable at arm time. An index is only an "
                "enumeration order, so a UUID-less record cannot authorize "
                "safe crash recovery. Restore UUID telemetry and retry."
            )
        snapshot_problem = _snapshot_shape_problem(
            snap.to_dict(), require_complete=True
        )
        if snapshot_problem:
            raise GpuGuardError(
                "Refusing the first GPU hardware write because its pre-write "
                f"snapshot cannot be serialized as readable recovery evidence: "
                f"{snapshot_problem}."
            )
        if (
            state is None
            or state.snapshot != snap.to_dict()
            or state.device_uuid != snap.uuid
            or state.device_index != snap.index
        ):
            raise GpuGuardError(
                "Refusing a GPU hardware write because the guard's private "
                "snapshot no longer exactly matches its durable recovery "
                "record. No hardware was touched."
            )
        mismatch = _device_mismatch(state.to_dict(), self._c)
        if mismatch:
            if not self._restored:
                self._publish_live_identity_conflict(
                    "before hardware write", mismatch
                )
            raise GpuGuardError(
                "Refusing a GPU hardware write because controller identity "
                f"changed after arm: {mismatch}."
            )

    def _publish_live_identity_conflict(
        self,
        phase: str,
        mismatch: str,
        *,
        preserved_payload: bytes | None = None,
    ) -> None:
        """Make observed in-process controller drift globally irreversible."""
        self._identity_conflict = True
        if self._restored or self._state is None:
            return
        try:
            current_payload = self._machine_state_path.read_bytes()
        except OSError:
            current_payload = None
        alternate_payload = preserved_payload
        if preserved_payload is None:
            preserved_payload = current_payload
            alternate_payload = None
        else:
            preserved_payload = current_payload
        _write_unreadable_machine_marker(
            preserved_payload,
            alternate_payload=alternate_payload,
            reason=(
                f"live controller identity drift {phase}: {mismatch}; a "
                "callback may have touched more than one GPU"
            ),
            authority_conflict=True,
            evidence_path=self._state.evidence_path or self._state_path,
            machine_path=self._machine_state_path,
        )

    def _begin_live_hardware_callback(
        self,
        phase: str,
    ) -> tuple[bytes, bytes]:
        """Replace active authority with a crash-durable uncertainty marker."""
        self._check_process_owner(phase)
        self._check_write_identity()
        self._check_process_owner(phase)
        assert self._state is not None
        expected = self._state.to_dict()
        current, status, _extra = _load_state(self._machine_state_path)
        if status != STATE_OK or current != expected or current.get("restored", True):
            self._identity_conflict = True
            try:
                changed_payload = self._machine_state_path.read_bytes()
            except OSError:
                changed_payload = None
            _write_unreadable_machine_marker(
                changed_payload,
                alternate_payload=_record_document_bytes(expected),
                reason=(
                    "machine-wide recovery authority changed before a live "
                    f"hardware callback ({phase}); refusing to choose between "
                    "it and the guard's active project evidence"
                ),
                authority_conflict=True,
                evidence_path=self._state.evidence_path or self._state_path,
                machine_path=self._machine_state_path,
            )
            raise GpuGuardError(
                "Refusing to enter a hardware callback because machine-wide "
                "recovery authority no longer exactly matches this guard's "
                "active record."
            )
        try:
            active_payload = self._machine_state_path.read_bytes()
            _write_unreadable_machine_marker(
                active_payload,
                reason=(
                    f"hardware callback in progress ({phase}); the process "
                    "may die before physical-device identity is revalidated"
                ),
                authority_conflict=True,
                evidence_path=self._state.evidence_path or self._state_path,
                machine_path=self._machine_state_path,
            )
            marker_payload = self._machine_state_path.read_bytes()
        except BaseException:
            # No callback has run. Keep the ordinary active evidence in place
            # if marker publication failed, but do not attempt another callback
            # in this guard transaction.
            self._identity_conflict = True
            raise
        return active_payload, marker_payload

    def _finish_live_hardware_callback(
        self,
        phase: str,
        active_payload: bytes,
        marker_payload: bytes,
    ) -> None:
        """Return BLOCKED -> ACTIVE only after callback return and UUID proof."""
        self._check_process_owner(f"settle {phase}")
        assert self._state is not None
        mismatch = _device_mismatch(self._state.to_dict(), self._c)
        self._check_process_owner(f"settle {phase}")
        if mismatch:
            self._publish_live_identity_conflict(
                f"after {phase}",
                mismatch,
                preserved_payload=active_payload,
            )
            raise GpuGuardError(
                f"GPU identity became unsafe after {phase}: {mismatch}. The "
                "machine-wide conflict blocker remains."
            )
        try:
            if self._machine_state_path.read_bytes() != marker_payload:
                self._identity_conflict = True
                try:
                    changed_payload = self._machine_state_path.read_bytes()
                except OSError:
                    changed_payload = None
                _write_unreadable_machine_marker(
                    changed_payload,
                    alternate_payload=active_payload,
                    reason=(
                        "machine-wide callback blocker changed after live "
                        f"hardware callback ({phase}); refusing to overwrite "
                        "unknown recovery authority"
                    ),
                    authority_conflict=True,
                    evidence_path=(
                        self._state.evidence_path or self._state_path
                    ),
                    machine_path=self._machine_state_path,
                )
                raise GpuGuardError(
                    "machine-wide callback blocker changed before identity "
                    "settlement; refusing to overwrite unknown authority"
                )
            self._check_process_owner(f"publish settled {phase} evidence")
            _write_record_document(self._machine_state_path, self._state.to_dict())
        except BaseException:
            self._identity_conflict = True
            raise

    def _run_live_hardware_callback(
        self,
        phase: str,
        operation: str,
        arguments: tuple,
        callback,
    ) -> None:
        """Run one callback with a durable barrier across every BaseException."""
        active_payload, marker_payload = self._begin_live_hardware_callback(phase)
        callback_error = None
        try:
            with _controller_write_access(self, operation, arguments):
                callback()
        except BaseException as exc:
            callback_error = exc
        try:
            self._finish_live_hardware_callback(
                phase, active_payload, marker_payload
            )
        except BaseException as settle_error:
            if callback_error is not None:
                raise settle_error from callback_error
            raise
        if callback_error is not None:
            raise callback_error

    def _verify_live_power_limit(self, target: int, phase: str) -> None:
        """Power has readback; never treat callback return as proof of success."""
        self._check_process_owner(phase)
        try:
            observed = self._c.get_power_limit_mw()
        except Exception as exc:
            raise GpuGuardError(
                f"could not read back the power limit {phase}: {exc}"
            ) from exc
        assert self._state is not None
        self._check_process_owner(phase)
        mismatch = _device_mismatch(self._state.to_dict(), self._c)
        self._check_process_owner(phase)
        if mismatch:
            self._publish_live_identity_conflict(phase, mismatch)
            raise GpuGuardError(
                f"GPU identity became unsafe {phase}: {mismatch}."
            )
        if (
            not isinstance(observed, int)
            or isinstance(observed, bool)
            or observed != target
        ):
            raise GpuGuardError(
                f"power-limit verification failed {phase}: requested exactly "
                f"{target} mW but read back {observed!r}; recovery evidence "
                "remains active"
            )

    def _mark_touched(self) -> None:
        """Record intent to disk BEFORE touching hardware."""
        assert self._state is not None
        self._state.restored = False
        self._restored = False
        self._publish_owned_state()

    def lock_clocks(self, min_mhz: int, max_mhz: int | None = None) -> None:
        max_mhz = min_mhz if max_mhz is None else max_mhz
        self.arm()
        self._check_owner_thread("lock clocks")
        self._check_consent()
        # Argument validation BEFORE the environment check: a bad clock value is
        # the caller's bug and is free to detect, so report it even when
        # unprivileged. Checking root first would send an operator to sudo and
        # only then reveal the real problem.
        self._check_clock_floor(min_mhz, max_mhz)
        self._check_write_identity()
        self._check_privilege()

        with self._hardware_write_transaction():
            # Consent/validation above deliberately happens before blocking and
            # can run arbitrary Python. A returning signal handler may have
            # restored the no-touch guard and released its lease in that gap.
            # Re-enter arm while signals are blocked so ownership cannot vanish
            # between this check, intent publication, and the hardware write.
            self.arm()
            assert self._state is not None
            self._state.clocks_locked = True
            self._state.locked_min_mhz = min_mhz
            self._state.locked_max_mhz = max_mhz
            self._mark_touched()
            self._run_live_hardware_callback(
                "set locked clocks",
                "set_locked_clocks",
                (min_mhz, max_mhz),
                lambda: self._c.set_locked_clocks(min_mhz, max_mhz),
            )

    def set_power_limit_mw(self, milliwatts: int) -> None:
        self.arm()
        self._check_owner_thread("set the power limit")
        self._check_consent()
        if not isinstance(milliwatts, int) or isinstance(milliwatts, bool):
            raise GpuGuardError(
                f"Refusing power limit {milliwatts!r}: the target must be an "
                "integer number of milliwatts so hardware and recovery evidence "
                "cannot disagree."
            )
        snap = self._snapshot
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
        self._check_write_identity()
        if snap is None or snap.power_limit_mw is None:
            raise GpuGuardError(
                "Refusing to change the power limit because the pre-write "
                "current limit was unreadable at arm time. The factory default "
                "is not an exact substitute for a deliberate operator cap, so "
                "this write could not be restored safely."
            )
        self._check_privilege()

        with self._hardware_write_transaction():
            self.arm()
            assert self._state is not None
            self._state.power_limit_written_mw = int(milliwatts)
            self._mark_touched()
            target = int(milliwatts)
            self._run_live_hardware_callback(
                "set power limit",
                "set_power_limit_mw",
                (target,),
                lambda: self._c.set_power_limit_mw(target),
            )
            self._verify_live_power_limit(target, "after the forward write")

    # -- restore ---------------------------------------------------------

    def restore(self) -> None:
        """Return the GPU to stock. Idempotent; safe to call any number of times.

        The clock reset is issued unconditionally rather than gated on a
        readback, because no readback exists.

        Restore is owner-thread-only for the same reason writes are. A worker
        restore racing the owner's intent publication could otherwise mark the
        record clean and release the lease immediately before the owner applies
        a setting. Signal-driven restores are latched by the transaction logic,
        not refused.
        """
        self._check_process_owner("restore hardware from a fork child")
        self._check_owner_thread("restore GPU state")
        if self._write_depth:
            raise GpuGuardError(
                "Refusing a re-entrant restore during a GPU hardware write. "
                "Signal restores are deferred until that write settles."
            )
        if self._restore_depth:
            # Same-thread nested restore (for example, a controller callback)
            # joins the outer restore. The outer operation owns signal latching,
            # hardware calls, evidence publication, and lease release.
            return
        self._restore_depth = 1
        try:
            # The signal latch stays set through mask restoration, just as it
            # does for a forward write, so a second signal cannot recursively
            # publish clean state midway through recovery.
            with _deferred_signals():
                self._restore_settled()
        finally:
            self._restore_depth = 0
            self._drain_pending_signals()

    def _restore_settled(self) -> None:
        """Restore implementation entered only with signal recursion latched."""
        if self._restored:
            # Arming without a write still owns the single-instance slot. There
            # is no recovery debt, so surrender it immediately and permanently.
            self._release_lifetime_lease()
            return
        if self._identity_conflict:
            raise GpuGuardError(
                "restore refused because a prior callback left controller "
                "identity or recovery authority uncertain. The machine-wide "
                "conflict blocker remains; use read-only inspection and "
                "manual recovery."
            )
        errors = []
        try:
            self._run_live_hardware_callback(
                "reset locked clocks",
                "reset_locked_clocks",
                (),
                self._c.reset_locked_clocks,
            )
        except Exception as exc:
            errors.append(f"reset_locked_clocks: {exc}")

        snap = self._snapshot
        # Prefer what the limit ACTUALLY was when we armed, not the factory
        # default: an operator may have set a deliberate cap before we ran.
        target = snap.power_limit_mw if snap else None
        if (
            not self._identity_conflict
            and self._state
            and self._state.power_limit_written_mw is not None
        ):
            if target is None:
                errors.append(
                    "power limit was modified but no restore target is known "
                    "(snapshot reads failed at arm time); the cap is still in "
                    "effect"
                )
            else:
                try:
                    exact_target = int(target)
                    self._run_live_hardware_callback(
                        "restore power limit",
                        "set_power_limit_mw",
                        (exact_target,),
                        lambda: self._c.set_power_limit_mw(exact_target),
                    )
                    self._verify_live_power_limit(
                        exact_target, "after in-process power restore"
                    )
                except Exception as exc:
                    errors.append(f"set_power_limit_mw: {exc}")

        # A markerless restore-failed record can be legitimate only after a
        # clean-publication failure.  Since power is readable, require proof it
        # still equals the captured value before clearing that ambiguous debt.
        if (
            not self._identity_conflict
            and self._state is not None
            and self._state.restore_failed
            and not self._state.clocks_locked
            and self._state.power_limit_written_mw is None
        ):
            if target is None:
                errors.append(
                    "markerless failed recovery has no readable captured power "
                    "limit, so it cannot be proven safe to clear"
                )
            else:
                try:
                    self._verify_live_power_limit(
                        int(target),
                        "while validating markerless failed recovery debt",
                    )
                except Exception as exc:
                    errors.append(f"markerless power verification: {exc}")

        if not self._identity_conflict:
            try:
                self._check_write_identity()
            except Exception as exc:
                errors.append(f"final device identity: {exc}")

        if errors:
            # DO NOT mark restored. With no readback, the state file is the only
            # evidence that exists -- recording success would erase it exactly
            # when it matters, and disarm every remaining restore path.
            if self._state is not None:
                self._state.restored = False
                self._state.restored_utc = None
                self._state.restore_failed = True
                self._state.restore_errors = list(errors)
                if not self._identity_conflict:
                    try:
                        self._publish_owned_state()
                    except Exception:
                        pass
            raise GpuGuardError(
                "restore incomplete, GPU MAY STILL BE MODIFIED: "
                + "; ".join(errors)
                + f". State file {self._state_path} retains restored=false; run "
                "`sudo python -m joule_agent.gpu_guard restore`."
            )

        if self._state is not None:
            self._state.restored = True
            self._state.restore_failed = False
            self._state.restore_errors = []
            self._state.clocks_locked = False
            self._state.restored_utc = _utcnow()
            try:
                self._publish_owned_state()
            except Exception as exc:
                # Hardware is stock, but releasing the machine lease while its
                # canonical evidence still says active would let later callers
                # split on state-path namespace. Keep ownership and republish a
                # conservative unrestored record wherever possible.
                message = f"state publication after restore: {exc}"
                self._state.restored = False
                self._state.restored_utc = None
                self._state.restore_failed = True
                self._state.restore_errors = [message]
                try:
                    self._publish_owned_state()
                except Exception:
                    pass
                raise GpuGuardError(
                    "GPU hardware was restored, but recovery evidence could "
                    f"not be marked clean ({exc}); lifetime ownership is "
                    "retained and restored=false remains authoritative."
                ) from exc
        self._restored = True
        self._release_lifetime_lease()

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
    requested_path = Path(state_path)
    prior, status, extra = _load_authoritative_state(requested_path)
    path = Path(extra.get("source_path", requested_path))
    dev, _ = _controller_identity(controller)
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
    record_needs_recovery = bool(prior and not prior.get("restored", True))
    record_claims_power = bool(
        record_needs_recovery
        and prior.get("power_limit_written_mw") is not None
    )
    mismatch = _device_mismatch(prior, controller) if prior else None
    identity_unverifiable = _identity_unverifiable(mismatch)
    if prior is not None:
        report["record_device_uuid"] = prior.get("device_uuid")
        report["record_is_for_this_device"] = (
            None if identity_unverifiable else mismatch is None
        )
    if record_needs_recovery and mismatch:
        # Tier 1 above describes the device this controller holds; this record
        # describes a different one. Saying "stale" without saying which device
        # would send an operator to reset the wrong GPU.
        report["stale"] = True
        report["asserted_clock_lock"] = bool(prior.get("clocks_locked"))
        report["restore_failed"] = bool(prior.get("restore_failed"))
        holder = prior.get("pid")
        report["writer_pid_alive"] = _pid_alive(holder) if holder else False
        if identity_unverifiable:
            report["findings"].append(
                f"IDENTITY UNVERIFIABLE (state file): {mismatch}. Recovery "
                "must not apply the snapshot until the live UUID can be read."
            )
            if report["asserted_clock_lock"]:
                report["findings"].append(
                    "ASSERTED (state file, NOT verifiable): this unidentified "
                    f"record claims a clock lock at [{prior.get('locked_min_mhz')}, "
                    f"{prior.get('locked_max_mhz')}] MHz. Preserve the claim, "
                    "but do not apply it to a device until identity is proven."
                )
            if report["writer_pid_alive"]:
                report["findings"].append(
                    f"WRITER LIVE: pid {holder} is still running; do not race "
                    "its recovery ownership."
                )
            report["recommendation"] = (
                "Restore UUID telemetry (for example, verify the NVIDIA driver "
                "and NVML access), then retry. Do not recover by GPU index "
                "alone; an index is not a physical-device identity."
            )
        else:
            report["findings"].append(
                f"OTHER DEVICE (state file): {mismatch}. Nothing here is "
                f"evidence about GPU {dev}'s clock lock, and recovery pointed "
                "at this device would apply the other card's snapshot. Re-run "
                f"with --gpu {prior.get('device_index')}."
            )
            report["recommendation"] = (
                "Run `sudo python -m joule_agent.gpu_guard restore --gpu "
                f"{prior.get('device_index')}` -- recovery must target the "
                "device the record names."
            )
        return report
    if record_needs_recovery:
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
        if report["state_file_unreadable"]:
            report["recommendation"] = (
                "Inspect and repair the unreadable recovery evidence. "
                "`restore --quarantine-unreadable` preserves a forensic copy "
                "but intentionally keeps the machine-wide blocker: neither "
                "--gpu nor --force-power-default can prove which physical "
                "device an unreadable record named."
            )
        elif report.get("writer_pid_alive"):
            report["recommendation"] = (
                "A guard process is STILL RUNNING and owns this record. Prefer "
                "stopping it -- its own restore will run on exit. Recovery will "
                "reset the clock but will refuse to clear a live guard's record."
            )
        elif report["verified_power_modified"] and not record_claims_power:
            target = dev if dev is not None else 0
            report["recommendation"] = (
                f"A power cap is verified on GPU {target}, but no unrestored "
                "guard record claims this repo wrote that cap. If it is "
                "deliberate, leave it alone. Otherwise run `sudo python -m "
                f"joule_agent.gpu_guard restore --gpu {target} "
                "--force-power-default`; that explicit flag may undo an "
                "operator cap."
            )
        else:
            # This report already verified that the selected controller is the
            # record's device. Its live index wins over a historical index that
            # may have changed across a reboot or driver reload.
            target = dev
            if target is None:
                target = report.get("record_device_index")
            if target is None:
                target = 0
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
    """Recovery path after a hard kill, serialized with every live guard."""
    lease = _standalone_machine_lease()
    try:
        lease.__enter__()
    except (GpuGuardError, OSError) as exc:
        return {
            "reset_clocks": False,
            "power_restored_to_mw": None,
            "record_cleared": False,
            "incomplete": True,
            "errors": [f"machine lease: {exc}"],
            "recovery_state_path": str(state_path),
        }
    try:
        return _restore_from_state_file_locked(
            controller,
            state_path,
            force_power_default=force_power_default,
            quarantine_unreadable=quarantine_unreadable,
        )
    finally:
        lease.__exit__(None, None, None)


def _restore_from_state_file_locked(
    controller: GpuController,
    state_path: Path | str = DEFAULT_STATE_PATH,
    *,
    force_power_default: bool = False,
    quarantine_unreadable: bool = False,
) -> dict:
    """Implementation entered only while the machine recovery lease is held."""
    _require_recovery_process_owner()
    # Freeze configuration for the whole transaction. Controller callbacks are
    # arbitrary Python and must not be able to redirect later evidence writes
    # by mutating the module constant between begin and clean publication.
    machine_state_path = Path(MACHINE_STATE_PATH)
    requested_path = Path(state_path)
    prior, status, extra = _load_authoritative_state(
        requested_path,
        machine_path=machine_state_path,
    )
    _require_recovery_process_owner()
    state_path = Path(extra.get("source_path", requested_path))
    blocked_evidence_path = extra.get("blocked_evidence_path")
    if (
        status == STATE_UNREADABLE
        and not extra.get("authority_conflict")
        and blocked_evidence_path
    ):
        blocked_path = Path(blocked_evidence_path)
        _blocked_prior, blocked_status, _blocked_extra = _load_state(blocked_path)
        if blocked_status == STATE_UNREADABLE:
            state_path = blocked_path
    result = {
        "reset_clocks": False,
        "power_restored_to_mw": None,
        "record_cleared": False,
        "incomplete": False,
        "errors": [],
        "recovery_state_path": str(state_path),
    }
    if status == STATE_UNREADABLE:
        reason = extra.get("reason", "unknown")
        source_path = Path(state_path)
        machine_path = machine_state_path
        source_prior, source_status, _ = _load_state(source_path)

        # STATE_UNREADABLE can mean two individually valid files disagree, not
        # that either one is corrupt. Quarantine is not permission to select a
        # preferred history. Replace M with a tagged blocker that preserves its
        # exact bytes and names R. Leaving an active M in place is insufficient:
        # a later recovery through path C could clean M and strand R's different
        # debt outside global authority.
        if extra.get("authority_conflict") or source_status != STATE_UNREADABLE:
            _machine_prior, machine_status, machine_extra = _load_state(
                machine_path
            )
            already_tagged_conflict = bool(
                machine_status == STATE_UNREADABLE
                and machine_extra.get("authority_conflict")
            )
            if not already_tagged_conflict:
                try:
                    _require_recovery_process_owner()
                    try:
                        machine_payload = machine_path.read_bytes()
                    except OSError:
                        machine_payload = None
                    _write_unreadable_machine_marker(
                        machine_payload,
                        reason=reason,
                        authority_conflict=True,
                        evidence_path=requested_path,
                        machine_path=machine_path,
                    )
                except Exception as exc:
                    result["errors"].append(
                        f"machine-wide conflict blocker: {exc}"
                    )
            result["incomplete"] = True
            result["errors"].append(
                f"recovery authority conflict: {reason}. Refusing to "
                "quarantine, choose, replay, or clear either valid history; "
                "resolve the records manually."
            )
            return result

        try:
            unreadable_payload = source_path.read_bytes()
        except OSError:
            unreadable_payload = None

        # Discovery of path-specific unreadable evidence must become globally
        # visible before this lease can be released. Otherwise another guard
        # selecting path B could arm against an absent/clean machine record.
        if os.path.abspath(source_path) != os.path.abspath(machine_path):
            try:
                _require_recovery_process_owner()
                _write_unreadable_machine_marker(
                    unreadable_payload,
                    reason=reason,
                    evidence_path=source_path,
                    machine_path=machine_path,
                )
            except Exception as exc:
                result["incomplete"] = True
                result["errors"].append(
                    f"could not publish machine-wide unreadable blocker: {exc}"
                )
                return result

        if quarantine_unreadable:
            stamp = _utcnow().replace(":", "").replace("-", "")
            dest = source_path.with_suffix(
                f".corrupt-{stamp}-{time.time_ns()}"
            )
            try:
                _require_recovery_process_owner()
                if unreadable_payload is None:
                    raise GpuGuardError(
                        "the unreadable evidence bytes could not be preserved"
                    )
                if os.path.abspath(source_path) == os.path.abspath(machine_path):
                    # Copy first, then atomically replace M with a blocking
                    # marker. There is never a post-rename window in which M is
                    # absent and another path can arm after SIGKILL.
                    _write_quarantine_copy(dest, unreadable_payload)
                    _write_unreadable_machine_marker(
                        unreadable_payload,
                        reason=reason,
                        evidence_path=source_path,
                        machine_path=machine_path,
                    )
                else:
                    # M already holds the blocker published above.
                    os.replace(source_path, dest)
                result["quarantined_to"] = str(dest)
                result["incomplete"] = True
                result["errors"].append(
                    "the unreadable recovery authority was preserved in "
                    "quarantine, but its device UUID and prior settings remain "
                    "unknown. Neither a parseable caller mirror nor "
                    "--force-power-default proves which physical GPU the "
                    "canonical debt named, so no hardware was touched and the "
                    "machine-wide blocker remains. Inspect and repair the "
                    "quarantined evidence manually."
                )
                return result
            except Exception as exc:
                result["incomplete"] = True
                result["errors"].append(f"quarantine: {exc}")
                try:
                    _require_recovery_process_owner()
                    _write_unreadable_machine_marker(
                        unreadable_payload,
                        reason=reason,
                        evidence_path=source_path,
                        machine_path=machine_path,
                    )
                except Exception as marker_exc:
                    result["errors"].append(
                        f"machine quarantine marker: {marker_exc}"
                    )
                return result
        result["incomplete"] = True
        result["errors"].append(
            f"{state_path} exists but could not be interpreted ({reason}). A "
            "machine-wide blocker now preserves that unknown debt. Refusing "
            "all hardware writes because the record cannot prove which device "
            "or prior settings are authoritative; inspect or repair it, or use "
            "--quarantine-unreadable to preserve a forensic copy while the "
            "blocker remains."
        )
        return result

    # A clean record is historical evidence, not authority to replay an old
    # snapshot.  Only an unrestored record can drive identity-gated recovery or
    # a power rewrite; --force-power-default is an explicit override only when
    # no active record has failed its physical-identity gate.
    record_needs_recovery = bool(prior and not prior.get("restored", True))

    if not record_needs_recovery:
        result["clock_untouched_reason"] = (
            "no active UUID-bearing recovery record authorizes a hardware "
            "write; clean/absent evidence is historical, not permission to "
            "reset an operator-selected GPU"
        )
        result["power_untouched_reason"] = (
            "no active UUID-bearing recovery record authorizes a power write"
        )
        if force_power_default:
            result["incomplete"] = True
            result["errors"].append(
                "--force-power-default is refused without active, "
                "UUID-bearing recovery authority; a lost state file cannot "
                "prove which physical GPU or prior operator cap is safe"
            )
        return result

    if record_needs_recovery:
        # A legacy/path-specific active record must become machine-wide before
        # identity or live-PID diagnosis can return and before the first
        # recovery hardware call. Identity is deliberately preserved exactly:
        # the selected controller is not evidence for a missing recorded UUID.
        original_prior = prior
        recorded_identity_problem = _recorded_lineage_problem(original_prior)
        if recorded_identity_problem:
            try:
                _require_recovery_process_owner()
                try:
                    original_payload = Path(state_path).read_bytes()
                except OSError:
                    original_payload = None
                _write_unreadable_machine_marker(
                    original_payload,
                    reason=recorded_identity_problem,
                    evidence_path=state_path,
                    machine_path=machine_state_path,
                )
            except Exception as exc:
                result["errors"].append(
                    f"could not publish unverifiable identity blocker: {exc}"
                )
            result["incomplete"] = True
            result["errors"].append(
                "recovery lineage could not be verified: "
                f"{recorded_identity_problem}. Refusing all hardware writes "
                "and record clearing."
            )
            return result
        _requested_prior, requested_status, _requested_extra = _load_state(
            requested_path
        )
        _require_recovery_process_owner()
        requested_matches = bool(
            requested_status == STATE_OK
            and _requested_prior
            and not _requested_prior.get("restored", True)
            and (
                _requested_prior == original_prior
                or _same_guard_lineage(_requested_prior, original_prior)
            )
        )
        evidence_path = prior.get("evidence_path") or (
            requested_path if requested_matches else state_path
        )
        promoted = _canonicalize_record(
            prior,
            controller,
            evidence_path=evidence_path,
            preserve_recorded_identity=True,
        )
        promoted["restored"] = False
        try:
            _require_recovery_process_owner()
            machine_path = machine_state_path
            _write_record_document(machine_path, promoted)
            mirror_paths = [Path(state_path)]
            if requested_matches:
                mirror_paths.append(requested_path)
            published = {os.path.abspath(machine_path)}
            for mirror_path in mirror_paths:
                mirror_key = os.path.abspath(mirror_path)
                if mirror_key in published:
                    continue
                _require_recovery_process_owner()
                _write_record_document(mirror_path, promoted)
                published.add(mirror_key)
        except Exception as exc:
            result["incomplete"] = True
            result["errors"].append(
                f"could not promote active recovery evidence before hardware: {exc}"
            )
            return result
        prior = promoted

    # A record for another GPU must not be applied to this one: the snapshot's
    # power value belongs to that device, and on a homogeneous node NVML will
    # accept it silently.
    mismatch = (
        _device_mismatch(prior, controller) if record_needs_recovery else None
    )
    _require_recovery_process_owner()
    identity_mismatch = False

    def refuse_identity(
        mismatch_reason: str,
        phase: str,
        *,
        durable_conflict: bool = False,
    ) -> None:
        nonlocal identity_mismatch, prior
        rec_dev = prior.get("device_index") if prior else None
        rec_uuid = prior.get("device_uuid") if prior else None
        evidence_path = prior.get("evidence_path") if prior else state_path
        if durable_conflict:
            try:
                _require_recovery_process_owner()
                machine_path = machine_state_path
                try:
                    payload = machine_path.read_bytes()
                except OSError:
                    payload = None
                _write_unreadable_machine_marker(
                    payload,
                    reason=(
                        f"controller identity drift {phase}: {mismatch_reason}; "
                        "a callback may have touched more than one GPU"
                    ),
                    authority_conflict=True,
                    evidence_path=evidence_path,
                    machine_path=machine_path,
                )
            except Exception as exc:
                result["errors"].append(
                    f"could not publish identity-drift conflict blocker: {exc}"
                )
        identity_mismatch = True
        result["incomplete"] = True
        result["record_device_index"] = rec_dev
        result["record_device_uuid"] = rec_uuid
        if _identity_unverifiable(mismatch_reason):
            result["errors"].append(
                f"device identity could not be verified {phase}: "
                f"{mismatch_reason}. Refusing "
                "to write the snapshot or clear its record. Restore live UUID "
                "telemetry, then retry; selecting the same GPU index again is "
                "not identity proof."
            )
        else:
            result["errors"].append(
                f"device identity mismatch {phase}: {mismatch_reason}. "
                "Refusing to write that "
                "device's snapshot here or to clear its record -- doing so "
                "would mis-cap this card and destroy the other card's evidence. "
                f"Re-run with --gpu {rec_dev}."
            )
        prior = None      # do not use it for power, do not clear it

    def recheck_identity(phase: str) -> None:
        if not record_needs_recovery or prior is None:
            return
        drift = _device_mismatch(prior, controller)
        _require_recovery_process_owner()
        if drift:
            refuse_identity(drift, phase, durable_conflict=True)

    callback_barrier_failed = False

    def run_recovery_hardware_callback(
        phase: str,
        operation: str,
        arguments: tuple,
        callback,
    ) -> bool:
        """Run one recovery callback behind a durable BLOCKED -> ACTIVE gate."""
        nonlocal callback_barrier_failed, identity_mismatch, prior
        active_payload = None
        marker_payload = None
        barrier_record = prior if record_needs_recovery else None

        if barrier_record is not None:
            machine_path = machine_state_path
            current, current_status, _ = _load_state(machine_path)
            if (
                current_status != STATE_OK
                or current != barrier_record
                or current.get("restored", True)
            ):
                callback_barrier_failed = True
                identity_mismatch = True
                try:
                    changed_payload = machine_path.read_bytes()
                except OSError:
                    changed_payload = None
                _write_unreadable_machine_marker(
                    changed_payload,
                    alternate_payload=_record_document_bytes(barrier_record),
                    reason=(
                        "machine-wide recovery authority changed before the "
                        f"{phase} callback; refusing to choose it over the "
                        "selected active recovery evidence"
                    ),
                    authority_conflict=True,
                    evidence_path=(
                        barrier_record.get("evidence_path") or state_path
                    ),
                    machine_path=machine_path,
                )
                raise GpuGuardError(
                    "machine-wide active recovery authority changed before "
                    f"the {phase} callback; no hardware call was made"
                )
            try:
                _require_recovery_process_owner()
                active_payload = machine_path.read_bytes()
                _write_unreadable_machine_marker(
                    active_payload,
                    reason=(
                        f"hardware callback in progress ({phase}); the process "
                        "may die before physical-device identity is revalidated"
                    ),
                    authority_conflict=True,
                    evidence_path=(
                        barrier_record.get("evidence_path") or state_path
                    ),
                    machine_path=machine_path,
                )
                marker_payload = machine_path.read_bytes()
            except BaseException:
                # The callback has not run. Ordinary active evidence remains if
                # begin-publication failed, but this transaction must stop.
                callback_barrier_failed = True
                identity_mismatch = True
                raise

        callback_error = None
        try:
            with _controller_write_access(
                _RECOVERY_OWNER, operation, arguments
            ):
                callback()
        except BaseException as exc:
            callback_error = exc

        settle_error = None
        if barrier_record is not None:
            try:
                _require_recovery_process_owner()
                drift = _device_mismatch(barrier_record, controller)
                _require_recovery_process_owner()
                if drift:
                    _write_unreadable_machine_marker(
                        active_payload,
                        reason=(
                            f"controller identity drift after {phase}: {drift}; "
                            "a callback may have touched more than one GPU"
                        ),
                        authority_conflict=True,
                        evidence_path=(
                            barrier_record.get("evidence_path") or state_path
                        ),
                        machine_path=machine_state_path,
                    )
                    refuse_identity(drift, f"after {phase}")
                else:
                    machine_path = machine_state_path
                    if machine_path.read_bytes() != marker_payload:
                        try:
                            changed_payload = machine_path.read_bytes()
                        except OSError:
                            changed_payload = None
                        _write_unreadable_machine_marker(
                            changed_payload,
                            alternate_payload=active_payload,
                            reason=(
                                "machine-wide callback blocker changed after "
                                f"the {phase} callback; refusing to overwrite "
                                "unknown recovery authority"
                            ),
                            authority_conflict=True,
                            evidence_path=(
                                barrier_record.get("evidence_path") or state_path
                            ),
                            machine_path=machine_path,
                        )
                        raise GpuGuardError(
                            "machine-wide callback blocker changed before "
                            "identity settlement; refusing to overwrite it"
                        )
                    _write_record_document(machine_path, barrier_record)
            except BaseException as exc:
                settle_error = exc
                callback_barrier_failed = True
                identity_mismatch = True
                result["incomplete"] = True

        if settle_error is not None:
            if callback_error is not None:
                raise settle_error from callback_error
            raise settle_error
        if callback_error is not None:
            raise callback_error
        return not identity_mismatch and not callback_barrier_failed

    if mismatch:
        refuse_identity(mismatch, "before recovery")

    if record_needs_recovery and prior is not None:
        holder = prior.get("pid")
        holder_alive = _pid_alive(holder)
        _require_recovery_process_owner()
        if holder_alive:
            result["record_not_cleared_reason"] = (
                f"pid {holder} is still running and owns this record; "
                "refusing all recovery writes while its lifetime is live. "
                "Stop that process (its own restore will run) or retry after "
                "it exits."
            )
            result["incomplete"] = True
            return result

    claim = getattr(controller, "claim", None)
    if callable(claim):
        _require_recovery_process_owner()
        claim(_RECOVERY_OWNER)
        _require_recovery_process_owner()
    recheck_identity("after controller claim")

    if identity_mismatch or callback_barrier_failed:
        result["clock_untouched_reason"] = (
            "the unrestored record's physical-device identity did not match "
            "this controller; refusing to reset clocks on the wrong or "
            "unverifiable GPU"
        )
    else:
        try:
            if run_recovery_hardware_callback(
                "reset locked clocks",
                "reset_locked_clocks",
                (),
                controller.reset_locked_clocks,
            ):
                _require_recovery_process_owner()
                result["reset_clocks"] = True
        except Exception as exc:
            result["incomplete"] = True
            result["errors"].append(f"reset_locked_clocks: {exc}")

    # The controller callback is arbitrary Python in tests and a driver call in
    # production. It may retarget its handle; never let a successful clock reset
    # authorize a snapshot power write on a newly selected device.
    recheck_identity("after clock reset")

    # Restore power only if the record says WE changed it. Rewriting the limit
    # unconditionally would wipe an operator's deliberate pre-existing cap.
    wrote_power = bool(
        record_needs_recovery
        and prior
        and prior.get("power_limit_written_mw") is not None
    )
    markerless_failed_debt = bool(
        record_needs_recovery
        and prior
        and prior.get("restore_failed")
        and not prior.get("clocks_locked", False)
        and prior.get("power_limit_written_mw") is None
    )
    target = None
    if wrote_power:
        snap = (prior or {}).get("snapshot") or {}
        target = snap.get("power_limit_mw")
    forcing_power_default = False
    if target is None and force_power_default and not identity_mismatch:
        _require_recovery_process_owner()
        target = controller.get_power_default_limit_mw()
        _require_recovery_process_owner()
        if target is None:
            result["incomplete"] = True
            result["errors"].append(
                "--force-power-default was requested, but the device default "
                "power limit could not be read; no power write was attempted."
            )
        else:
            forcing_power_default = True
    recheck_identity("immediately before power restore")
    if identity_mismatch or callback_barrier_failed:
        target = None
        wrote_power = False
        forcing_power_default = False
    if target is None and wrote_power and not force_power_default:
        result["incomplete"] = True
        result["errors"].append(
            "the record shows this repo changed the power limit, but no restore "
            "target is recoverable from its snapshot; the cap is still in "
            "effect. Re-run with --force-power-default to reset to the device "
            "default."
        )
    elif target is None:
        if identity_mismatch:
            result["power_untouched_reason"] = (
                "--force-power-default cannot override an active record's "
                "physical-device identity mismatch; no power write was made."
            )
        elif prior and not record_needs_recovery:
            result["power_untouched_reason"] = (
                "the state record is already restored; its power-write fields "
                "are historical evidence and will not be replayed over a later "
                "operator cap. Pass --force-power-default to override."
            )
        else:
            result["power_untouched_reason"] = (
                "no state record shows this repo changed the power limit; leaving "
                "it alone so a deliberate operator cap is not silently undone. "
                "Pass --force-power-default to override."
            )
    else:
        try:
            exact_target = int(target)
            if run_recovery_hardware_callback(
                "restore power limit",
                "set_power_limit_mw",
                (exact_target,),
                lambda: controller.set_power_limit_mw(exact_target),
            ):
                _require_recovery_process_owner()
                observed = controller.get_power_limit_mw()
                _require_recovery_process_owner()
                recheck_identity("after power-limit readback")
                if identity_mismatch:
                    raise GpuGuardError(
                        "controller identity changed during power readback"
                    )
                if (
                    not isinstance(observed, int)
                    or isinstance(observed, bool)
                    or observed != exact_target
                ):
                    raise GpuGuardError(
                        "power-limit verification failed: requested exactly "
                        f"{exact_target} mW but read back {observed!r}"
                    )
                result["power_restored_to_mw"] = exact_target
                if forcing_power_default:
                    result["power_forced_to_default"] = True
        except Exception as exc:
            result["incomplete"] = True
            result["errors"].append(f"set_power_limit_mw: {exc}")
    recheck_identity("after power restore")

    if (
        markerless_failed_debt
        and prior is not None
        and not identity_mismatch
        and not result["errors"]
    ):
        captured_power = (prior.get("snapshot") or {}).get("power_limit_mw")
        try:
            _require_recovery_process_owner()
            observed_power = controller.get_power_limit_mw()
            _require_recovery_process_owner()
            recheck_identity("after markerless-debt power readback")
            if identity_mismatch:
                raise GpuGuardError(
                    "controller identity changed during markerless-debt "
                    "power verification"
                )
            if (
                not isinstance(captured_power, int)
                or isinstance(captured_power, bool)
                or not isinstance(observed_power, int)
                or isinstance(observed_power, bool)
                or observed_power != captured_power
            ):
                raise GpuGuardError(
                    "markerless failed recovery cannot be proven clean: "
                    f"captured power was {captured_power!r} mW and live "
                    f"readback is {observed_power!r} mW"
                )
        except Exception as exc:
            result["incomplete"] = True
            result["errors"].append(f"markerless power verification: {exc}")

    # Do not clear a record a live guard still owns, and do not publish a
    # read that went stale while the hardware calls above were running. Arming
    # is refused over a live record precisely because that destroys evidence;
    # recovery must not do the same destruction silently.
    if prior is not None and not record_needs_recovery:
        result["record_not_cleared_reason"] = (
            "the state record is already restored historical evidence; leaving "
            "it byte-for-byte unchanged avoids racing a new guard's intent "
            "publication."
        )

    if record_needs_recovery and prior is not None:
        _require_recovery_process_owner()
        current = _read_record(Path(state_path))
        _require_recovery_process_owner()
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

    recheck_identity("before clean evidence publication")
    if record_needs_recovery and prior is not None and not result["errors"]:
        # Legacy schema-1/2 records were intentionally permissive. Normalize
        # every successful recovery through today's complete schema before
        # publishing, otherwise the next read would correctly reject the
        # promoted-but-incomplete record as malformed.
        _require_recovery_process_owner()
        canonical = _canonicalize_record(
            prior,
            controller,
            evidence_path=prior.get("evidence_path") or state_path,
            preserve_recorded_identity=True,
        )
        _require_recovery_process_owner()
        canonical["restored"] = True
        canonical["clocks_locked"] = False
        canonical["restore_failed"] = False
        canonical["restore_errors"] = []
        canonical["restored_utc"] = _utcnow()
        canonical["_recovered_by"] = "restore_from_state_file"
        try:
            source_path = Path(state_path)
            machine_path = machine_state_path
            mirror_paths = []
            evidence_path = prior.get("evidence_path")
            candidates = [Path(evidence_path)] if evidence_path else []
            requested_record, requested_status, _ = _load_state(requested_path)
            if requested_status == STATE_OK and requested_record == prior:
                candidates.append(requested_path)
            for candidate in candidates:
                if candidate is None:
                    continue
                if os.path.abspath(candidate) not in {
                    os.path.abspath(source_path),
                    os.path.abspath(machine_path),
                }:
                    mirror_paths.append(Path(candidate))

            # Write project mirrors first and the authoritative machine record
            # last. Until that last atomic rename, a new guard still sees
            # restored=false and cannot race these publications.
            for path in mirror_paths:
                _require_recovery_process_owner()
                existing, mirror_status, _ = _load_state(path)
                if mirror_status == STATE_UNREADABLE:
                    raise GpuGuardError(f"mirror {path} became unreadable")
                if (
                    existing is not None
                    and existing != prior
                    and not _same_guard_lineage(existing, prior)
                ):
                    # A clean unrelated historical mirror is not this
                    # recovery's authority and must remain byte-identical.
                    if existing.get("restored", True):
                        continue
                    raise GpuGuardError(
                        f"mirror {path} changed to a different active record"
                    )
                _write_record_document(path, canonical)

            if os.path.abspath(source_path) != os.path.abspath(machine_path):
                _require_recovery_process_owner()
                _write_record_document(source_path, canonical)
            _require_recovery_process_owner()
            _write_record_document(machine_path, canonical)
            result["record_cleared"] = True
        except Exception as exc:
            result["errors"].append(f"state_file: {exc}")
            result["incomplete"] = True
    if result["errors"]:
        result["incomplete"] = True
    _require_recovery_process_owner()
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
        help="if recovery evidence cannot be interpreted, preserve its exact "
             "bytes in a .corrupt-<timestamp> forensic copy; the machine-wide "
             "blocker remains because an unknown record cannot identify which "
             "GPU or prior settings are safe",
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
