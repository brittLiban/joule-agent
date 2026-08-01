"""Tests for the telemetry recorder.

All tests run against FakeGpuReader -- no NVML, no GPU. The properties under
test are the ones that make a recording trustworthy: driver timestamps survive
to disk unmodified, pre-run history is not included, and sample loss is
detected rather than hidden.
"""

import csv
import json
import math
from pathlib import Path

import pytest

from joule_agent.telemetry import (
    FakeGpuReader,
    SensorProfile,
    TelemetryRecorder,
    characterize_sensors,
    integrate_power_samples,
)


def read_csv(path: Path):
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    return rows[0], rows[1:]


class ManualClock:
    """Monotonic clock the test drives, so runs are deterministic and instant."""

    def __init__(self, reader: FakeGpuReader) -> None:
        self.reader = reader
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds
        self.reader.advance(seconds)


# ---------------------------------------------------------------------------
# integration helper
# ---------------------------------------------------------------------------


def test_integrate_constant_power():
    # 10 W held for 1 s == 10 J.
    samples = [(i * 100_000, 10_000.0) for i in range(11)]
    assert integrate_power_samples(samples) == pytest.approx(10.0)


def test_integrate_handles_unordered_input():
    ordered = [(0, 1000.0), (500_000, 2000.0), (1_000_000, 3000.0)]
    assert integrate_power_samples(list(reversed(ordered))) == pytest.approx(
        integrate_power_samples(ordered)
    )


def test_integrate_ramp_matches_analytic():
    # Linear ramp 0 -> 10 W over 1 s: trapezoid is exact, area = 5 J.
    samples = [(i * 100_000, i * 1000.0) for i in range(11)]
    assert integrate_power_samples(samples) == pytest.approx(5.0)


def test_integrate_degenerate_inputs():
    assert integrate_power_samples([]) == 0.0
    assert integrate_power_samples([(0, 5000.0)]) == 0.0


# ---------------------------------------------------------------------------
# fake reader semantics
# ---------------------------------------------------------------------------


def test_drain_cursor_yields_no_duplicates():
    r = FakeGpuReader(start_us=1_000_000_000)
    r.advance(1.0)
    first = r.drain("power", 0)
    r.advance(1.0)
    second = r.drain("power", first[-1][0])
    assert first and second
    assert set(t for t, _ in first).isdisjoint(t for t, _ in second)
    assert second[0][0] > first[-1][0]


def test_drain_respects_ring_capacity():
    r = FakeGpuReader(start_us=0, ring_capacity=10)
    r.advance(5.0)  # 250 power samples at 20ms, ring holds 10
    assert len(r.drain("power", 0)) == 10


def test_empty_drain_is_not_an_error():
    r = FakeGpuReader(start_us=0)
    r.advance(1.0)
    cursor = r.drain("power", 0)[-1][0]
    assert r.drain("power", cursor) == []


# ---------------------------------------------------------------------------
# characterization
# ---------------------------------------------------------------------------


def test_characterize_measures_cadence_not_assumes_it():
    r = FakeGpuReader(
        start_us=0, cadence_us={"power": 20_000, "gpu_util": 200_000, "mem_util": 200_000}
    )
    r.advance(3.0)
    profile = characterize_sensors(r, probe_seconds=0.05, poll_interval_s=0.005)
    assert profile.streams["power"]["median_period_ms"] == pytest.approx(20.0)
    assert profile.streams["gpu_util"]["median_period_ms"] == pytest.approx(200.0)


def test_characterize_reports_different_hardware_differently():
    """Cadence must come from the device, not a constant in the source."""
    slow = FakeGpuReader(
        start_us=0, cadence_us={"power": 100_000, "gpu_util": 500_000, "mem_util": 500_000}
    )
    slow.advance(10.0)
    profile = characterize_sensors(slow, probe_seconds=0.05, poll_interval_s=0.005)
    assert profile.streams["power"]["median_period_ms"] == pytest.approx(100.0)


def test_characterize_records_ring_depth():
    r = FakeGpuReader(start_us=0, ring_capacity=42)
    r.advance(10.0)
    profile = characterize_sensors(r, probe_seconds=0.05, poll_interval_s=0.005)
    assert profile.streams["power"]["ring_capacity_samples"] == 42
    assert profile.min_stream_span_s() is not None


# ---------------------------------------------------------------------------
# recorder
# ---------------------------------------------------------------------------


def _record(tmp_path, reader, *, ticks, drain_interval=1.0, profile=None):
    """Drive the recorder synchronously via _drain_once for determinism."""
    clock = ManualClock(reader)
    rec = TelemetryRecorder(
        reader,
        tmp_path,
        drain_interval_s=drain_interval,
        profile=profile,
        clock=clock,
    )
    rec._open_files()
    for name in rec._stream_names:
        existing = reader.drain(name, 0)
        if existing:
            rec._cursors[name] = existing[-1][0]
    rec._started_wall_us = reader.now_us
    rec._started_monotonic = clock()
    for _ in range(ticks):
        clock.sleep(drain_interval)
        rec._drain_once()
    rec._write_meta()
    rec._close_files()
    return rec


def test_driver_timestamps_reach_disk_unmodified(tmp_path):
    r = FakeGpuReader(start_us=1_700_000_000_000_000)
    rec = _record(tmp_path, r, ticks=3)
    _, rows = read_csv(rec.paths.streams["power"])
    stamps = [int(row[0]) for row in rows]
    assert stamps == sorted(stamps)
    # Exactly on the generator's 20ms grid -- no rounding, no resampling.
    assert all((t - r.start_us) % 20_000 == 0 for t in stamps)
    # Consecutive deltas are the native cadence, not the 1s drain interval.
    deltas = {b - a for a, b in zip(stamps, stamps[1:])}
    assert deltas == {20_000}


def test_startup_does_not_capture_pre_run_history(tmp_path):
    r = FakeGpuReader(start_us=0)
    r.advance(2.0)  # ring already full of history before recording starts
    rec = _record(tmp_path, r, ticks=2)
    _, rows = read_csv(rec.paths.streams["power"])
    assert rows, "expected samples from the recording window"
    earliest = min(int(row[0]) for row in rows)
    assert earliest > 2_000_000, "pre-run samples leaked into the run"


def test_sample_loss_is_detected_not_hidden(tmp_path):
    r = FakeGpuReader(start_us=0, ring_capacity=20)
    profile = SensorProfile(streams={"power": {"ring_capacity_samples": 20, "ring_span_s": 0.4}})
    # 2s between drains against a 0.4s ring -> guaranteed loss.
    rec = _record(tmp_path, r, ticks=2, drain_interval=2.0, profile=profile)
    kinds = [e["kind"] for e in rec.loss_events]
    assert "possible_sample_loss" in kinds
    meta = json.loads(rec.paths.meta.read_text())
    assert meta["integrity"]["clean"] is False


def test_clean_run_is_marked_clean(tmp_path):
    r = FakeGpuReader(start_us=0, ring_capacity=119)
    profile = SensorProfile(
        streams={"power": {"ring_capacity_samples": 119, "ring_span_s": 2.38}}
    )
    rec = _record(tmp_path, r, ticks=3, drain_interval=0.5, profile=profile)
    meta = json.loads(rec.paths.meta.read_text())
    assert meta["integrity"]["clean"] is True


def test_slow_drain_interval_is_flagged_at_construction(tmp_path):
    r = FakeGpuReader(start_us=0)
    profile = SensorProfile(streams={"power": {"ring_span_s": 2.4}})
    rec = TelemetryRecorder(r, tmp_path, drain_interval_s=5.0, profile=profile)
    assert any(e["kind"] == "config_warning" for e in rec.loss_events)


def test_energy_counter_delta_recorded(tmp_path):
    r = FakeGpuReader(start_us=0, power_mw=100_000.0)  # 100 W
    rec = _record(tmp_path, r, ticks=4, drain_interval=1.0)
    meta = json.loads(rec.paths.meta.read_text())
    delta_j = meta["energy_counter"]["delta_mj"] / 1000.0
    # 100 W across the 3 intervals between the 4 polls == 300 J.
    assert delta_j == pytest.approx(300.0, rel=0.01)


def test_meta_records_measured_profile(tmp_path):
    r = FakeGpuReader(start_us=0)
    r.advance(3.0)
    profile = characterize_sensors(r, probe_seconds=0.05, poll_interval_s=0.005)
    rec = _record(tmp_path, r, ticks=2, profile=profile)
    meta = json.loads(rec.paths.meta.read_text())
    assert meta["sensor_profile"]["streams"]["power"]["median_period_ms"] == 20.0
    assert meta["schema_version"] == 1
    assert meta["device"]["name"] == "FakeGPU"


def test_polled_csv_keeps_clocks(tmp_path):
    r = FakeGpuReader(start_us=0)
    rec = _record(tmp_path, r, ticks=3)
    header, rows = read_csv(rec.paths.polled)
    assert header[:4] == ["wall_us", "sm_clock_mhz", "mem_clock_mhz", "energy_mj"]
    assert len(rows) == 3
    assert all(row[1] == "1000" for row in rows)


def test_run_loop_flags_drain_overrun(tmp_path):
    """A drain slower than the interval must be recorded, not silently absorbed."""

    class SlowReader(FakeGpuReader):
        def drain(self, stream, cursor_us):
            clock.t += 0.5  # every drain overruns the 0.1s interval
            return super().drain(stream, cursor_us)

    clock = ManualClock(None)
    r = SlowReader(start_us=0)
    clock.reader = r
    rec = TelemetryRecorder(r, tmp_path, drain_interval_s=0.1, clock=clock)
    rec._open_files()
    rec._started_monotonic = clock()

    # Drive the loop body directly: stop after a few ticks.
    ticks = {"n": 0}
    real_wait = rec._stop.wait

    def counting_wait(timeout=None):
        ticks["n"] += 1
        if ticks["n"] >= 3:
            rec._stop.set()
        return real_wait(0)

    rec._stop.wait = counting_wait
    rec._run()
    rec._close_files()

    assert any(e["kind"] == "drain_overrun" for e in rec.loss_events)
    assert all(e["behind_s"] > 0 for e in rec.loss_events if e["kind"] == "drain_overrun")


def test_recorder_thread_lifecycle(tmp_path):
    """The real threaded path starts, records and shuts down cleanly."""
    r = FakeGpuReader(start_us=int(1e15))
    rec = TelemetryRecorder(r, tmp_path, drain_interval_s=0.01)
    with rec:
        for _ in range(20):
            r.advance(0.05)
            __import__("time").sleep(0.005)
    assert rec._thread is None
    assert rec.paths.meta.exists()
    assert rec.sample_counts["power"] > 0
