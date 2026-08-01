"""GPU telemetry recorder (NVML, read-only).

Records GPU power, utilization, clocks and the cumulative energy counter for a
single NVIDIA device. Read-only: this module never writes clocks or power limits
and requires no elevated privileges.

Design notes
------------
The driver exposes power and utilization through two different mechanisms, and
they do not have the same fidelity:

* ``nvmlDeviceGetPowerUsage()`` returns the *latest* value the driver has
  published. Polling it faster than the driver refreshes yields repeated values
  -- an apparent sample rate that the underlying sensor does not support.
* ``nvmlDeviceGetSamples()`` returns the driver's own internal ring buffer, with
  a driver-supplied timestamp per sample, at the sensor's native cadence.

We therefore do not resample at collection time. The recorder drains the driver
buffers and stores every sample with the timestamp the driver assigned it.
Resampling to a uniform grid, if wanted, is an analysis-time decision made on
data that still carries its original timing.

The ring buffer is small. Its capacity is discovered at startup rather than
assumed; the drain interval must stay well below the span it covers or samples
are silently dropped. Drains that come back with a full ring are recorded as
possible-loss events instead of being passed off as complete.

Sensor cadence is a per-device, per-driver property, not a constant. Nothing
here hard-codes an expected rate: :func:`characterize_sensors` measures what the
hardware actually does and the result is written into the run metadata, so each
recording documents the fidelity of the device that produced it.

Energy
------
``nvmlDeviceGetTotalEnergyConsumption()`` is a monotonic millijoule counter. It
reads the same sensor chain as everything else -- it is not independent evidence
about sensor fidelity -- but the driver integrates it without gaps, so it does
not inherit the error of integrating a coarse signal ourselves. It is the
intended source of truth for run-level joules. :func:`verify_energy_agreement`
cross-checks it against an integration of the power buffer; run that before
depending on either number.

Scope: measurement only. Clock/power control and restore-on-exit handling are a
separate component and are deliberately absent here.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

__all__ = [
    "StreamSpec",
    "STREAMS",
    "GpuReader",
    "NvmlGpuReader",
    "FakeGpuReader",
    "SensorProfile",
    "characterize_sensors",
    "TelemetryRecorder",
    "RunPaths",
    "integrate_power_samples",
    "verify_energy_agreement",
]


# --------------------------------------------------------------------------
# Stream definitions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamSpec:
    """A driver-buffered sample stream."""

    name: str
    nvml_const: str
    unit: str


STREAMS: tuple[StreamSpec, ...] = (
    StreamSpec("power", "NVML_TOTAL_POWER_SAMPLES", "mW"),
    StreamSpec("gpu_util", "NVML_GPU_UTILIZATION_SAMPLES", "percent"),
    StreamSpec("mem_util", "NVML_MEMORY_UTILIZATION_SAMPLES", "percent"),
)

# nvmlValueType_t -> field name on the c_nvmlValue_t union. Reading the wrong
# member returns garbage rather than raising, so dispatch on the type the driver
# reports instead of assuming one.
_VALUE_TYPE_FIELD: dict[int, str] = {
    0: "dVal",  # NVML_VALUE_TYPE_DOUBLE
    1: "uiVal",  # NVML_VALUE_TYPE_UNSIGNED_INT
    2: "ulVal",  # NVML_VALUE_TYPE_UNSIGNED_LONG
    3: "ullVal",  # NVML_VALUE_TYPE_UNSIGNED_LONG_LONG
    4: "sllVal",  # NVML_VALUE_TYPE_SIGNED_LONG_LONG
    5: "siVal",  # NVML_VALUE_TYPE_SIGNED_INT
    6: "usVal",  # NVML_VALUE_TYPE_UNSIGNED_SHORT
}


# --------------------------------------------------------------------------
# Reader interface
# --------------------------------------------------------------------------


class GpuReader(Protocol):
    """Narrow read-only view of one GPU.

    Kept deliberately small so the recorder, the characterizer and the
    verification helper can all be exercised without NVML or a GPU present.
    """

    def device_info(self) -> dict: ...

    def supported_streams(self) -> tuple[str, ...]: ...

    def drain(self, stream: str, cursor_us: int) -> list[tuple[int, float]]:
        """Return ``(timestamp_us, value)`` newer than ``cursor_us``.

        Timestamps are the driver's, in microseconds since the Unix epoch.
        An empty list means no new samples -- not an error.
        """
        ...

    def poll(self) -> dict:
        """Read the values that have no driver-side buffer."""
        ...

    def close(self) -> None: ...


class NvmlGpuReader:
    """:class:`GpuReader` backed by NVML. Read-only."""

    def __init__(self, index: int = 0) -> None:
        import pynvml

        self._nvml = pynvml
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        self._index = index
        self._closed = False
        self._supported = self._probe_supported_streams()

    # -- setup -----------------------------------------------------------

    def _probe_supported_streams(self) -> tuple[str, ...]:
        ok = []
        for spec in STREAMS:
            const = getattr(self._nvml, spec.nvml_const, None)
            if const is None:
                continue
            try:
                self._nvml.nvmlDeviceGetSamples(self._handle, const, 0)
            except self._nvml.NVMLError_NotFound:
                # No samples buffered yet; the stream itself exists.
                ok.append(spec.name)
            except self._nvml.NVMLError:
                continue
            else:
                ok.append(spec.name)
        return tuple(ok)

    def supported_streams(self) -> tuple[str, ...]:
        return self._supported

    def device_info(self) -> dict:
        n = self._nvml
        info: dict = {"index": self._index}

        def attempt(key, fn):
            try:
                info[key] = fn()
            except Exception as exc:  # noqa: BLE001 - informational only
                info[key] = None
                info.setdefault("_unavailable", []).append(
                    f"{key}: {type(exc).__name__}"
                )

        attempt("name", lambda: _as_text(n.nvmlDeviceGetName(self._handle)))
        attempt("uuid", lambda: _as_text(n.nvmlDeviceGetUUID(self._handle)))
        attempt("driver_version", lambda: _as_text(n.nvmlSystemGetDriverVersion()))
        attempt("nvml_version", lambda: _as_text(n.nvmlSystemGetNVMLVersion()))
        attempt(
            "power_limit_mw", lambda: n.nvmlDeviceGetPowerManagementLimit(self._handle)
        )
        attempt(
            "max_sm_clock_mhz",
            lambda: n.nvmlDeviceGetMaxClockInfo(self._handle, n.NVML_CLOCK_SM),
        )
        attempt(
            "max_mem_clock_mhz",
            lambda: n.nvmlDeviceGetMaxClockInfo(self._handle, n.NVML_CLOCK_MEM),
        )
        return info

    # -- reads -----------------------------------------------------------

    def drain(self, stream: str, cursor_us: int) -> list[tuple[int, float]]:
        spec = _stream_spec(stream)
        const = getattr(self._nvml, spec.nvml_const)
        try:
            value_type, samples = self._nvml.nvmlDeviceGetSamples(
                self._handle, const, cursor_us
            )
        except self._nvml.NVMLError_NotFound:
            # NVML signals "nothing newer than the cursor" as NotFound. This is
            # the steady state between drains, not a failure.
            return []
        field_name = _VALUE_TYPE_FIELD.get(value_type)
        if field_name is None:
            raise RuntimeError(
                f"NVML reported unknown sample value type {value_type} for {stream!r}"
            )
        out = []
        for s in samples:
            if s.timeStamp > cursor_us:
                out.append((int(s.timeStamp), getattr(s.sampleValue, field_name)))
        out.sort(key=lambda pair: pair[0])
        return out

    def poll(self) -> dict:
        n = self._nvml
        row: dict = {
            "wall_us": int(time.time() * 1e6),
            "sm_clock_mhz": None,
            "mem_clock_mhz": None,
            "energy_mj": None,
            "power_mw": None,
            "errors": 0,
        }

        def attempt(key, fn):
            try:
                row[key] = fn()
            except Exception:  # noqa: BLE001 - per-field failures must not stop a run
                row[key] = None
                row["errors"] += 1

        attempt(
            "sm_clock_mhz",
            lambda: n.nvmlDeviceGetClockInfo(self._handle, n.NVML_CLOCK_SM),
        )
        attempt(
            "mem_clock_mhz",
            lambda: n.nvmlDeviceGetClockInfo(self._handle, n.NVML_CLOCK_MEM),
        )
        attempt(
            "energy_mj", lambda: n.nvmlDeviceGetTotalEnergyConsumption(self._handle)
        )
        attempt("power_mw", lambda: n.nvmlDeviceGetPowerUsage(self._handle))
        return row

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self._nvml.nvmlShutdown()
            except Exception:  # noqa: BLE001 - nothing useful to do at teardown
                pass


def _as_text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _stream_spec(name: str) -> StreamSpec:
    for spec in STREAMS:
        if spec.name == name:
            return spec
    raise KeyError(f"unknown stream {name!r}")


# --------------------------------------------------------------------------
# Fake reader (tests, and development without a GPU)
# --------------------------------------------------------------------------


class FakeGpuReader:
    """Deterministic in-memory :class:`GpuReader`.

    Generates samples on a fixed cadence per stream and enforces a ring
    capacity, so drain-interval and sample-loss behaviour can be tested with no
    NVML present.
    """

    def __init__(
        self,
        *,
        cadence_us: dict[str, int] | None = None,
        ring_capacity: int = 119,
        start_us: int | None = None,
        power_mw: float = 22_000.0,
        streams: tuple[str, ...] = ("power", "gpu_util", "mem_util"),
    ) -> None:
        self.cadence_us = cadence_us or {
            "power": 20_000,
            "gpu_util": 200_000,
            "mem_util": 200_000,
        }
        self.ring_capacity = ring_capacity
        self.start_us = start_us if start_us is not None else int(time.time() * 1e6)
        self.power_mw = power_mw
        self._streams = streams
        self.now_us = self.start_us
        self.closed = False
        self._poll_count = 0

    # -- clock control for tests ----------------------------------------

    def advance(self, seconds: float) -> None:
        self.now_us += int(seconds * 1e6)

    # -- GpuReader ------------------------------------------------------

    def device_info(self) -> dict:
        return {"index": 0, "name": "FakeGPU", "uuid": "GPU-FAKE", "driver_version": "0"}

    def supported_streams(self) -> tuple[str, ...]:
        return self._streams

    def drain(self, stream: str, cursor_us: int) -> list[tuple[int, float]]:
        if stream not in self._streams:
            raise KeyError(stream)
        cadence = self.cadence_us[stream]
        # Every sample instant on the grid, up to now.
        first = max(cursor_us + 1, self.start_us)
        k0 = -(-(first - self.start_us) // cadence)  # ceil division
        k1 = (self.now_us - self.start_us) // cadence
        stamps = [self.start_us + k * cadence for k in range(k0, k1 + 1)]
        # The ring only holds the most recent `ring_capacity` samples; anything
        # older than that window is gone by the time we ask for it.
        if len(stamps) > self.ring_capacity:
            stamps = stamps[-self.ring_capacity :]
        return [(ts, self._value(stream, ts)) for ts in stamps]

    def _value(self, stream: str, ts: int) -> float:
        if stream == "power":
            return self.power_mw
        return 50.0

    def poll(self) -> dict:
        self._poll_count += 1
        elapsed_s = (self.now_us - self.start_us) / 1e6
        return {
            "wall_us": self.now_us,
            "sm_clock_mhz": 1000,
            "mem_clock_mhz": 500,
            "energy_mj": int(self.power_mw * elapsed_s),
            "power_mw": int(self.power_mw),
            "errors": 0,
        }

    def close(self) -> None:
        self.closed = True


# --------------------------------------------------------------------------
# Sensor characterization
# --------------------------------------------------------------------------


@dataclass
class SensorProfile:
    """Measured -- not assumed -- behaviour of this device's sensors."""

    streams: dict = field(default_factory=dict)
    polled: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"streams": self.streams, "polled": self.polled, "notes": self.notes}

    def min_stream_span_s(self) -> float | None:
        spans = [
            s["ring_span_s"]
            for s in self.streams.values()
            if s.get("ring_span_s") is not None
        ]
        return min(spans) if spans else None


def characterize_sensors(
    reader: GpuReader, *, probe_seconds: float = 2.0, poll_interval_s: float = 0.01
) -> SensorProfile:
    """Measure each sensor's real update cadence and buffer depth.

    Update rates are a property of the device and driver, so they are measured
    here rather than assumed. The result is recorded alongside every run: a
    recording is only as trustworthy as the sensors that produced it, and this
    is how a later reader can tell what that fidelity was.
    """
    profile = SensorProfile()

    # Buffered streams: the driver's own timestamps give the cadence directly.
    for name in reader.supported_streams():
        samples = reader.drain(name, 0)
        entry: dict = {
            "unit": _stream_spec(name).unit,
            "ring_capacity_samples": len(samples) or None,
            "ring_span_s": None,
            "median_period_ms": None,
        }
        if len(samples) >= 2:
            stamps = [ts for ts, _ in samples]
            deltas = [b - a for a, b in zip(stamps, stamps[1:])]
            entry["ring_span_s"] = round((stamps[-1] - stamps[0]) / 1e6, 4)
            entry["median_period_ms"] = round(statistics.median(deltas) / 1000.0, 3)
        profile.streams[name] = entry

    # Polled values have no driver timestamp; time the value changes instead.
    changes: dict[str, list[float]] = {"sm_clock_mhz": [], "power_mw": [], "energy_mj": []}
    last: dict[str, object] = {}
    t0 = time.monotonic()
    while time.monotonic() - t0 < probe_seconds:
        row = reader.poll()
        for key in changes:
            v = row.get(key)
            if v is not None and last.get(key) != v:
                if key in last:
                    changes[key].append(time.monotonic() - t0)
                last[key] = v
        time.sleep(poll_interval_s)

    for key, times in changes.items():
        deltas = [round((b - a) * 1000, 2) for a, b in zip(times, times[1:])]
        profile.polled[key] = {
            "observed_changes": len(times),
            "median_period_ms": round(statistics.median(deltas), 2) if deltas else None,
        }

    profile.notes.append(
        f"Characterized over {probe_seconds:.1f}s at {poll_interval_s * 1000:.0f}ms poll "
        "interval. Cadence measured under whatever load was present at the time; "
        "it may differ under sustained GPU load."
    )
    return profile


# --------------------------------------------------------------------------
# Recorder
# --------------------------------------------------------------------------


@dataclass
class RunPaths:
    root: Path
    meta: Path
    polled: Path
    streams: dict

    @classmethod
    def create(cls, out_dir: Path, run_id: str, stream_names) -> "RunPaths":
        root = out_dir / run_id
        root.mkdir(parents=True, exist_ok=True)
        return cls(
            root=root,
            meta=root / "meta.json",
            polled=root / "polled.csv",
            streams={n: root / f"{n}_samples.csv" for n in stream_names},
        )


class TelemetryRecorder:
    """Drains driver sample buffers on a background thread and writes them raw.

    Samples keep the driver's timestamps; nothing is resampled, interpolated or
    smoothed on the way to disk.
    """

    def __init__(
        self,
        reader: GpuReader,
        out_dir: Path | str,
        *,
        drain_interval_s: float = 1.0,
        run_id: str | None = None,
        profile: SensorProfile | None = None,
        clock: "callable" = time.monotonic,
    ) -> None:
        self._reader = reader
        self._drain_interval_s = drain_interval_s
        # Waiting happens on the stop Event, not time.sleep, so shutdown is
        # immediate rather than deferred to the end of the current interval.
        self._clock = clock
        self._profile = profile
        self._stream_names = tuple(reader.supported_streams())
        self.run_id = run_id or datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        self.paths = RunPaths.create(Path(out_dir), self.run_id, self._stream_names)

        self._cursors: dict[str, int] = {n: 0 for n in self._stream_names}
        self._files: dict[str, object] = {}
        self._writers: dict[str, object] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self.sample_counts: dict[str, int] = {n: 0 for n in self._stream_names}
        self.loss_events: list = []
        self.poll_rows = 0
        self.poll_errors = 0
        self._started_wall_us: int | None = None
        self._started_monotonic: float | None = None
        self._energy_first: int | None = None
        self._energy_last: int | None = None

        # Guard rail: the ring only spans a couple of seconds. Draining slower
        # than it fills loses samples with no error from NVML.
        span = profile.min_stream_span_s() if profile else None
        if span is not None and drain_interval_s > span * 0.5:
            self.loss_events.append(
                {
                    "kind": "config_warning",
                    "detail": (
                        f"drain_interval_s={drain_interval_s} exceeds half the "
                        f"shortest ring span ({span:.2f}s); samples will be dropped"
                    ),
                }
            )

    # -- lifecycle -------------------------------------------------------

    def __enter__(self) -> "TelemetryRecorder":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("recorder already started")
        self._open_files()
        # Seed cursors to "now" so the first drain does not dump the seconds of
        # history the ring already held before the run began.
        for name in self._stream_names:
            existing = self._reader.drain(name, 0)
            if existing:
                self._cursors[name] = existing[-1][0]
        self._started_wall_us = int(time.time() * 1e6)
        self._started_monotonic = self._clock()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="joule-telemetry", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        self._drain_once()  # capture whatever landed after the last tick
        self._write_meta()
        self._close_files()

    # -- worker ----------------------------------------------------------

    def _run(self) -> None:
        tick = 0
        start = self._clock()
        while not self._stop.is_set():
            self._drain_once()
            tick += 1
            # Absolute deadlines: a slow drain must not make the interval drift.
            deadline = start + tick * self._drain_interval_s
            remaining = deadline - self._clock()
            if remaining > 0:
                if self._stop.wait(remaining):
                    break
            else:
                self.loss_events.append(
                    {
                        "kind": "drain_overrun",
                        "wall_us": int(time.time() * 1e6),
                        "behind_s": round(-remaining, 4),
                    }
                )
                # Re-anchor instead of trying to catch up. Chasing a deadline we
                # are already past would drain back-to-back with no wait, which
                # spins the CPU and delays observing the stop event -- and the
                # backlog can never be paid off if drains are simply slower than
                # the interval. Always yield, even when late.
                start = self._clock()
                tick = 0
                if self._stop.wait(min(self._drain_interval_s, 0.05)):
                    break

    def _drain_once(self) -> None:
        for name in self._stream_names:
            try:
                samples = self._reader.drain(name, self._cursors[name])
            except Exception as exc:  # noqa: BLE001 - keep other streams alive
                self.loss_events.append(
                    {
                        "kind": "drain_error",
                        "stream": name,
                        "error": f"{type(exc).__name__}: {exc}",
                        "wall_us": int(time.time() * 1e6),
                    }
                )
                continue
            if not samples:
                continue

            capacity = None
            if self._profile:
                capacity = self._profile.streams.get(name, {}).get(
                    "ring_capacity_samples"
                )
            if capacity and len(samples) >= capacity:
                # A full ring means older samples were very likely overwritten
                # before we got to them. Record it rather than presenting the
                # file as a complete record.
                self.loss_events.append(
                    {
                        "kind": "possible_sample_loss",
                        "stream": name,
                        "returned": len(samples),
                        "ring_capacity_samples": capacity,
                        "wall_us": int(time.time() * 1e6),
                    }
                )

            writer = self._writers[name]
            for ts, value in samples:
                writer.writerow((ts, value))
            self._cursors[name] = samples[-1][0]
            self.sample_counts[name] += len(samples)
            self._files[name].flush()

        row = self._reader.poll()
        self.poll_errors += row.get("errors", 0)
        energy = row.get("energy_mj")
        if energy is not None:
            if self._energy_first is None:
                self._energy_first = energy
            self._energy_last = energy
        self._writers["_polled"].writerow(
            (
                row.get("wall_us"),
                row.get("sm_clock_mhz"),
                row.get("mem_clock_mhz"),
                row.get("energy_mj"),
                row.get("power_mw"),
                row.get("errors", 0),
            )
        )
        self.poll_rows += 1
        self._files["_polled"].flush()

    # -- files -----------------------------------------------------------

    def _open_files(self) -> None:
        for name in self._stream_names:
            fh = open(self.paths.streams[name], "w", newline="")
            w = csv.writer(fh)
            w.writerow(("driver_timestamp_us", f"value_{_stream_spec(name).unit}"))
            self._files[name] = fh
            self._writers[name] = w
        fh = open(self.paths.polled, "w", newline="")
        w = csv.writer(fh)
        w.writerow(
            (
                "wall_us",
                "sm_clock_mhz",
                "mem_clock_mhz",
                "energy_mj",
                "power_mw_polled",
                "read_errors",
            )
        )
        self._files["_polled"] = fh
        self._writers["_polled"] = w

    def _close_files(self) -> None:
        for fh in self._files.values():
            try:
                fh.close()
            except Exception:  # noqa: BLE001
                pass
        self._files.clear()
        self._writers.clear()

    def _write_meta(self) -> None:
        energy_delta = None
        if self._energy_first is not None and self._energy_last is not None:
            energy_delta = self._energy_last - self._energy_first
        meta = {
            "run_id": self.run_id,
            "schema_version": 1,
            "device": self._reader.device_info(),
            "host": {
                "platform": platform.platform(),
                "python": sys.version.split()[0],
            },
            "recording": {
                "drain_interval_s": self._drain_interval_s,
                "started_wall_us": self._started_wall_us,
                "duration_s": (
                    round(self._clock() - self._started_monotonic, 4)
                    if self._started_monotonic is not None
                    else None
                ),
                "streams": self._stream_names,
                "sample_counts": self.sample_counts,
                "polled_rows": self.poll_rows,
                "polled_read_errors": self.poll_errors,
            },
            "energy_counter": {
                "first_mj": self._energy_first,
                "last_mj": self._energy_last,
                "delta_mj": energy_delta,
                "note": (
                    "Driver-integrated counter over the same sensor chain as the "
                    "power stream. Complete (no missed intervals), not independent "
                    "evidence of sensor fidelity. Cross-check with "
                    "verify_energy_agreement before relying on it."
                ),
            },
            "sensor_profile": self._profile.to_dict() if self._profile else None,
            "integrity": {
                "events": self.loss_events,
                "clean": not any(
                    e["kind"] in ("possible_sample_loss", "drain_error")
                    for e in self.loss_events
                ),
            },
            "columns": {
                "*_samples.csv": (
                    "driver_timestamp_us: driver-assigned timestamp, microseconds "
                    "since Unix epoch. Raw, never resampled."
                ),
                "polled.csv": (
                    "Host-timestamped reads with no driver buffer. power_mw_polled "
                    "updates only as fast as the driver publishes it -- consecutive "
                    "identical values are one measurement, not several. Use the "
                    "power stream or the energy counter for analysis."
                ),
            },
        }
        self.paths.meta.write_text(json.dumps(meta, indent=2, default=str))


# --------------------------------------------------------------------------
# Energy cross-check
# --------------------------------------------------------------------------


def integrate_power_samples(samples) -> float:
    """Trapezoidal integral of ``(timestamp_us, milliwatts)`` -> joules."""
    ordered = sorted(samples, key=lambda p: p[0])
    total = 0.0
    for (t0, p0), (t1, p1) in zip(ordered, ordered[1:]):
        dt_s = (t1 - t0) / 1e6
        total += dt_s * ((p0 + p1) / 2.0) / 1000.0
    return total


def verify_energy_agreement(
    reader: GpuReader, *, seconds: float = 20.0, drain_interval_s: float = 1.0
) -> dict:
    """Compare the energy counter against an integration of the power buffer.

    The two share a sensor chain, so this does not validate the sensor. What it
    checks is that the driver's integration and ours describe the same run --
    if they disagree by more than a percent or two, one of them is being read
    wrong and neither should be trusted until that is resolved.
    """
    start = reader.poll()
    collected: list = []
    cursor = 0
    seed = reader.drain("power", 0)
    if seed:
        cursor = seed[-1][0]

    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        time.sleep(min(drain_interval_s, max(0.0, seconds - (time.monotonic() - t0))))
        new = reader.drain("power", cursor)
        if new:
            collected.extend(new)
            cursor = new[-1][0]
    end = reader.poll()

    integrated_j = integrate_power_samples(collected)
    counter_j = None
    if start.get("energy_mj") is not None and end.get("energy_mj") is not None:
        counter_j = (end["energy_mj"] - start["energy_mj"]) / 1000.0

    # The integral only covers the span between the first and last sample, which
    # is shorter than the wall window; compare over that span.
    span_s = (
        (collected[-1][0] - collected[0][0]) / 1e6 if len(collected) >= 2 else 0.0
    )
    wall_s = (end["wall_us"] - start["wall_us"]) / 1e6

    result = {
        "samples": len(collected),
        "integrated_span_s": round(span_s, 4),
        "wall_span_s": round(wall_s, 4),
        "integrated_j": round(integrated_j, 4),
        "counter_j": round(counter_j, 4) if counter_j is not None else None,
        "integrated_mean_w": round(integrated_j / span_s, 3) if span_s else None,
        "counter_mean_w": (
            round(counter_j / wall_s, 3) if counter_j is not None and wall_s else None
        ),
    }
    if result["integrated_mean_w"] and result["counter_mean_w"]:
        a, b = result["integrated_mean_w"], result["counter_mean_w"]
        result["mean_power_diff_pct"] = round(abs(a - b) / ((a + b) / 2) * 100, 3)
    else:
        result["mean_power_diff_pct"] = None
    return result


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_reader(args) -> GpuReader:
    return NvmlGpuReader(index=args.gpu)


def _cmd_characterize(args) -> int:
    reader = _build_reader(args)
    try:
        profile = characterize_sensors(reader, probe_seconds=args.probe_seconds)
    finally:
        reader.close()
    print(json.dumps(profile.to_dict(), indent=2))
    return 0


def _cmd_record(args) -> int:
    reader = _build_reader(args)
    try:
        profile = characterize_sensors(reader, probe_seconds=args.probe_seconds)
        rec = TelemetryRecorder(
            reader,
            args.out_dir,
            drain_interval_s=args.drain_interval,
            profile=profile,
        )
        with rec:
            deadline = time.monotonic() + args.duration
            try:
                while time.monotonic() < deadline:
                    time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
            except KeyboardInterrupt:
                print("\ninterrupted; finalizing run", file=sys.stderr)
    finally:
        reader.close()

    print(f"run:     {rec.paths.root}")
    for name, count in rec.sample_counts.items():
        print(f"  {name:<9} {count:>7} samples -> {rec.paths.streams[name].name}")
    print(f"  polled    {rec.poll_rows:>7} rows    -> {rec.paths.polled.name}")
    if rec._energy_first is not None and rec._energy_last is not None:
        print(f"  energy    {(rec._energy_last - rec._energy_first) / 1000.0:.2f} J (counter delta)")
    problems = [e for e in rec.loss_events if e["kind"] != "drain_overrun"]
    if problems:
        print(f"  INTEGRITY {len(problems)} event(s) -- see meta.json")
    return 0


def _cmd_verify(args) -> int:
    reader = _build_reader(args)
    try:
        result = verify_energy_agreement(
            reader, seconds=args.duration, drain_interval_s=args.drain_interval
        )
    finally:
        reader.close()
    print(json.dumps(result, indent=2))
    pct = result["mean_power_diff_pct"]
    if pct is None:
        print("\ninconclusive: counter or samples unavailable", file=sys.stderr)
        return 1
    verdict = "AGREE" if pct <= 2.0 else "DISAGREE"
    print(f"\n{verdict}: mean power differs by {pct:.2f}% (threshold 2%)")
    return 0 if verdict == "AGREE" else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="joule_agent.telemetry",
        description="Read-only NVML telemetry recorder.",
    )
    p.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("characterize", help="measure this device's sensor cadences")
    c.add_argument("--probe-seconds", type=float, default=2.0)
    c.set_defaults(func=_cmd_characterize)

    r = sub.add_parser("record", help="record a run")
    r.add_argument("--duration", type=float, required=True, help="seconds")
    r.add_argument("--out-dir", default="runs")
    r.add_argument("--drain-interval", type=float, default=1.0)
    r.add_argument("--probe-seconds", type=float, default=2.0)
    r.set_defaults(func=_cmd_record)

    v = sub.add_parser("verify", help="cross-check energy counter vs power buffer")
    v.add_argument("--duration", type=float, default=20.0)
    v.add_argument("--drain-interval", type=float, default=1.0)
    v.set_defaults(func=_cmd_verify)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
