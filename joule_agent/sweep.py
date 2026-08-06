"""Static clock/power sweep -- the baseline the dynamic controller must beat.

This is component 4. Its output is the *tuned static* baseline, which is the
real comparison target for this project: beating stock default proves nothing,
because stock default is not what a competent operator would ship.

What it does: for each configuration in a grid (an SM clock lock, optionally a
power cap, plus an unlocked "stock" reference), replay one identical traffic
schedule, and record joules per token, tokens per second, latency percentiles
and the SLO-violation rate. Then name the best static point.

Four decisions in here are load-bearing; changing them changes what the numbers
mean.

**One Schedule object, built once, replayed everywhere.** Component 3 made
reproducibility a type-system property: an open-loop ``Schedule`` fixes every
arrival time, prompt and output length before the run starts. The sweep does
not rebuild it per point -- it constructs one and replays that same object, so
digest equality across points is true *by construction* rather than by
convention. It is still asserted per run, because a construction argument is
not a measurement. Closed-loop presets (``reproducible=False``) are refused
outright: their offered load is a function of server response time, so a faster
clock receives more traffic and the comparison confounds the config change with
the traffic change it induces.

**Arm the guard per point, not once for the sweep.** ``gpu_guard`` exposes no
unlock -- only ``restore`` -- so a "stock" point cannot be expressed inside a
single armed session. Rather than add an unlock to the module that exists to
have as few hardware-touching paths as possible, each locked point arms, writes,
runs and restores; the stock point runs with no guard armed at all, which the
guard's own contract covers ("a guard that never writes releases the claim on
exit without resetting clocks"). The cost is one NVML snapshot round trip per
point, which is nothing beside a 60s load run. The benefit is that every point
starts from verified stock, and a crash mid-sweep strands one point's lock
instead of the whole sweep's.

**A failed restore aborts the sweep.** ``_check_not_busy`` would refuse the next
arm anyway; making that implicit refusal an explicit abort means the partial
results say why they are partial. A sweep that continues past a failed restore
is measuring a card in an unknown state.

**Repeats are round-robin, and stock is the drift detector.** Component 3
measured ~2% run-to-run variance under bit-identical traffic, so a single run
per configuration makes the "best" point partly a noise draw. Points are
therefore repeated, and the repeats are interleaved (all points, then all points
again) so thermal drift spreads across configurations instead of loading onto
whichever ran last. Interleaving hides drift rather than detecting it, so the
stock point -- which recurs every round -- is compared against itself across
rounds: if stock's first and last round differ by more than the noise floor, the
sweep drifted and every between-config comparison is contaminated. That is
reported as ``stock_drift_pct``, not left for the reader to reconstruct.

Requires root for any locked point, and explicit consent via
``--i-understand-this-changes-gpu-settings``. The stock-only sweep needs
neither, and is a useful dry run of the whole harness.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path

from joule_agent.gpu_guard import (
    MIN_SM_CLOCK_MHZ,
    ClockFloorError,
    ConsentError,
    GpuGuard,
    GpuGuardError,
    GuardBusyError,
    NvmlGpuController,
    PrivilegeError,
    check_stale_state,
)
from joule_agent.loadgen import PRESETS, LoadGenerator, RunReport, Schedule, build_schedule
from joule_agent.workload import compose, read_snapshot

#: Fraction difference in stock joules/token between the first and last round
#: above which the sweep is considered thermally contaminated. Component 3
#: measured ~2% run-to-run variance under bit-identical traffic; anything at or
#: under that is indistinguishable from that floor.
NOISE_FLOOR_FRACTION = 0.02

#: Default multiplier on the stock p99 that defines the latency SLO. The
#: project question is "at equal p99 latency", and 10% is the defensible
#: reading of "equal" given a 2% measurement noise floor.
DEFAULT_SLO_SLACK = 1.10


class SweepError(RuntimeError):
    """The sweep cannot produce a comparable result."""


#: Guard errors that are refusals: the guard declined *before* touching
#: hardware, so nothing was written. Distinguished from a failed restore,
#: which is the one case where the GPU may genuinely still be modified.
REFUSALS = (ConsentError, PrivilegeError, ClockFloorError, GuardBusyError)


# ---------------------------------------------------------------------------
# Configuration grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepPoint:
    """One configuration to measure.

    ``clock_mhz is None`` means no clock lock -- the stock reference. It is a
    point in the grid like any other, because the report must name stock as a
    baseline and because it doubles as the drift detector.
    """

    clock_mhz: int | None = None
    power_limit_mw: int | None = None

    @property
    def name(self) -> str:
        if self.clock_mhz is None and self.power_limit_mw is None:
            return "stock"
        bits = []
        if self.clock_mhz is not None:
            bits.append(f"clk{self.clock_mhz}")
        if self.power_limit_mw is not None:
            bits.append(f"pl{self.power_limit_mw // 1000}W")
        return "-".join(bits)

    @property
    def is_stock(self) -> bool:
        return self.clock_mhz is None and self.power_limit_mw is None

    def to_dict(self) -> dict:
        return {"name": self.name, "clock_mhz": self.clock_mhz,
                "power_limit_mw": self.power_limit_mw}


def build_grid(clocks, power_limits_mw=None, *, include_stock: bool = True) -> list:
    """Cross the requested clocks and power limits into a point list.

    Stock leads the grid so the very first measurement of a sweep is the
    unlocked reference, and so a sweep that dies early still carries the
    baseline every other number is relative to.
    """
    points = [SweepPoint()] if include_stock else []
    # `None` first, ALWAYS -- so every clock is also measured with no power cap.
    # Crossing clocks only against the requested caps meant that asking for any
    # power limit left the clock axis with no uncapped row at all: every
    # non-stock point carried a cap, every one was `axis_verified: false`, and
    # `best_point` could only ever come back "stock" because stock was the sole
    # verified row. Hours of GPU time for a vacuous answer. The trusted clock
    # curve has to exist independently of the axis that has no gate.
    powers = [None, *power_limits_mw] if power_limits_mw else [None]
    # `clocks or [None]`, not `clocks`: crossing only inside `for clk in
    # clocks` meant that requesting power limits with NO clocks produced a
    # stock-only grid -- so the operator was pushed through consent, root and
    # the stale pre-flight, waited out the full run, and had not one power
    # limit applied. A silent no-op on an explicitly requested hardware change.
    for clk in (clocks or [None]):
        for pl in powers:
            if clk is None and pl is None:
                continue  # that is the stock point, already added
            points.append(SweepPoint(clock_mhz=clk, power_limit_mw=pl))
    if not points:
        raise SweepError("empty grid: nothing to measure")
    return points


# ---------------------------------------------------------------------------
# Per-run measurement
# ---------------------------------------------------------------------------


@dataclass
class PointRun:
    """One execution of one point. Intended config and achieved config both."""

    point: str = ""
    requested_clock_mhz: int | None = None
    requested_power_limit_mw: int | None = None
    round_index: int = 0
    execution_index: int = 0
    started_utc: str = ""

    #: What the device actually did. A card that snaps to supported clock bins
    #: can return the same achieved clock for two different requests -- those
    #: are not two data points, and nothing else here would catch it.
    achieved_clock_median_mhz: float | None = None
    achieved_clock_min_mhz: float | None = None
    achieved_clock_max_mhz: float | None = None
    clock_samples: int = 0
    temperature_start_c: float | None = None
    temperature_end_c: float | None = None

    energy_j: float | None = None
    elapsed_s: float | None = None
    generated_tokens: float | None = None
    prompt_tokens: float | None = None
    joules_per_generated_token: float | None = None
    joules_per_total_token: float | None = None
    generated_tokens_per_s: float | None = None
    mean_power_w: float | None = None

    latency_p50_s: float | None = None
    latency_p95_s: float | None = None
    latency_p99_s: float | None = None
    slo_violation_rate: float | None = None
    completed: int = 0
    failed: int = 0

    schedule_digest: str = ""
    #: Gap between the token bracket and the energy bracket. The two /metrics
    #: round trips can differ by tens of ms while the NVML counter reads are
    #: microseconds, so the skew does not cancel exactly. At 60s runs this is
    #: negligible -- recorded so that is a measurement, not an assumption.
    bracket_skew_ms: float | None = None
    counter_reset: bool = False
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class MeasuredRun:
    """Holds a :class:`PointRun` plus the raw latencies behind it."""

    def __init__(self, run: PointRun, latencies_s: list) -> None:
        self.run = run
        self.latencies_s = latencies_s


def _clock_stats(samples: list) -> tuple:
    vals = [s for s in samples if s is not None]
    if not vals:
        return None, None, None, 0
    return statistics.median(vals), min(vals), max(vals), len(vals)


def slo_violation_rate(latencies_s: list, threshold_s: float) -> float | None:
    """Fraction of completed requests slower than the SLO.

    ``None`` when nothing completed -- a rate over zero requests is not zero
    violations, it is no measurement, and reporting 0.0 there would read as a
    configuration that met the SLO perfectly.
    """
    if not latencies_s:
        return None
    return sum(1 for v in latencies_s if v > threshold_s) / len(latencies_s)


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


class SweepRunner:
    """Executes a grid against one endpoint.

    Every collaborator is injected so the whole control flow -- ordering,
    digest enforcement, abort-on-failed-restore, drift detection -- is testable
    without NVML, without root and without a serving engine.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        schedule: Schedule,
        points: list,
        repeats: int = 3,
        slo_slack: float = DEFAULT_SLO_SLACK,
        slo_ms: float | None = None,
        settle_s: float = 5.0,
        warmup_runs: int = 1,
        guard_factory=None,
        load_runner=None,
        gpu_sampler=None,
        metrics_reader=None,
        clock=time.monotonic,
        sleep=time.sleep,
    ) -> None:
        if schedule.reproducible is False:
            raise SweepError(
                f"preset {schedule.preset!r} is not reproducible (closed loop): "
                "offered load depends on server response time, so a faster clock "
                "receives MORE traffic and the comparison confounds the config "
                "change with the traffic change it causes. Use an open-loop "
                "preset: " + ", ".join(
                    n for n, p in PRESETS.items() if p.reproducible
                )
            )
        if repeats < 1:
            raise SweepError("repeats must be at least 1")
        if not points:
            raise SweepError("empty grid: nothing to measure")

        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.schedule = schedule
        self.expected_digest = schedule.digest()
        self.points = points
        self.repeats = repeats
        self.slo_slack = slo_slack
        self.slo_ms = slo_ms
        self.settle_s = settle_s
        self.warmup_runs = warmup_runs

        self._guard_factory = guard_factory
        self._load_runner = load_runner or self._default_load_runner
        self._gpu_sampler = gpu_sampler
        self._metrics_reader = metrics_reader or self._default_metrics_reader
        self._clock = clock
        self._sleep = sleep

        self.runs: list = []
        self.aborted = False
        self.abort_reason: str | None = None
        #: True once any guard has exited still owing a restore. Nothing may
        #: shut NVML down while this holds.
        self.hardware_may_be_modified = False
        #: What the current point actually got onto the device, so an abort can
        #: report writes rather than infer them from the exception class.
        self._point_writes: list = []

    # -- default collaborators (real I/O) --------------------------------

    def _default_load_runner(self, schedule: Schedule) -> RunReport:
        gen = LoadGenerator(self.endpoint, self.model, schedule)
        return gen.run()

    def _default_metrics_reader(self):
        with urllib.request.urlopen(f"{self.endpoint}/metrics", timeout=10.0) as resp:
            return read_snapshot(resp.read())

    # -- one execution ----------------------------------------------------

    def _measure(self, point: SweepPoint, round_index: int, execution_index: int):
        """Run the schedule once at an already-applied configuration."""
        run = PointRun(
            point=point.name,
            requested_clock_mhz=point.clock_mhz,
            requested_power_limit_mw=point.power_limit_mw,
            round_index=round_index,
            execution_index=execution_index,
            started_utc=_utcnow(),
        )

        # Bracket order is identical at both ends so the skew is systematic
        # rather than random, and both ends are timestamped so it is visible.
        t_metrics_before = self._clock()
        snap_before = self._safe_metrics(run, "before")
        t_energy_before = self._clock()
        gpu_before = self._sample_gpu()

        # try/finally, because a raise from the load generator would otherwise
        # leave the polling thread alive through `guard.__exit__` -> restore().
        # `pthread_sigmask` is PER-THREAD, so a live sampler thread with
        # SIGINT/SIGTERM unblocked can take a delivery that gpu_guard's
        # signal-blocked sections exist to defer -- and the handler then runs
        # on the main thread inside the very window it was protecting.
        if self._gpu_sampler is not None:
            self._gpu_sampler.start()
        t0 = self._clock()
        try:
            report = self._load_runner(self.schedule)
        finally:
            elapsed = self._clock() - t0
            if self._gpu_sampler is not None:
                self._gpu_sampler.stop()

        gpu_after = self._sample_gpu()
        t_energy_after = self._clock()
        snap_after = self._safe_metrics(run, "after")
        t_metrics_after = self._clock()

        run.elapsed_s = round(elapsed, 3)
        run.bracket_skew_ms = round(
            abs((t_energy_before - t_metrics_before)
                + (t_metrics_after - t_energy_after)) * 1000.0, 2)

        self._apply_schedule_identity(run, report)
        self._apply_energy(run, gpu_before, gpu_after, elapsed)
        self._apply_tokens(run, snap_before, snap_after)
        self._apply_clocks(run)

        run.latency_p50_s = report.latency_p50_s
        run.latency_p95_s = report.latency_p95_s
        run.latency_p99_s = report.latency_p99_s
        run.completed = report.completed
        run.failed = report.failed
        run.warnings.extend(report.warnings)

        return MeasuredRun(run, list(getattr(report, "latencies_s", []) or []))

    def _apply_schedule_identity(self, run: PointRun, report: RunReport) -> None:
        """The traffic must be the same traffic, or nothing else compares."""
        run.schedule_digest = report.schedule_digest
        if report.schedule_digest != self.expected_digest:
            raise SweepError(
                f"schedule digest changed at point {run.point!r}: expected "
                f"{self.expected_digest} got {report.schedule_digest}. The "
                "configurations were not compared under the same traffic; "
                "every number in this sweep is void."
            )

    def _apply_energy(self, run, before, after, elapsed) -> None:
        e0 = (before or {}).get("energy_mj")
        e1 = (after or {}).get("energy_mj")
        run.temperature_start_c = (before or {}).get("temperature_c")
        run.temperature_end_c = (after or {}).get("temperature_c")
        if e0 is None or e1 is None:
            run.warnings.append("energy counter unreadable; joules/token unavailable")
            return
        if e1 < e0:
            run.counter_reset = True
            run.warnings.append(
                "energy counter went backwards (driver reload or wrap); "
                "this run's energy is not usable"
            )
            return
        run.energy_j = round((e1 - e0) / 1000.0, 3)
        if elapsed > 0:
            run.mean_power_w = round(run.energy_j / elapsed, 2)

    def _apply_tokens(self, run, before, after) -> None:
        if before is None or after is None:
            run.warnings.append("metrics scrape failed; token counts unavailable")
            return
        comp = compose(before, after)
        if comp.counter_reset:
            run.counter_reset = True
            run.warnings.append(
                "a vLLM token counter went backwards: the server restarted "
                "mid-run and this run's tokens are not usable"
            )
            return
        run.generated_tokens = comp.generation_tokens
        run.prompt_tokens = comp.prompt_tokens
        gen, pro = comp.generation_tokens, comp.prompt_tokens
        if run.energy_j is not None and gen:
            run.joules_per_generated_token = round(run.energy_j / gen, 6)
            if pro is not None:
                total = gen + pro
                if total:
                    run.joules_per_total_token = round(run.energy_j / total, 6)
        if gen and run.elapsed_s:
            run.generated_tokens_per_s = round(gen / run.elapsed_s, 2)

    def _apply_clocks(self, run) -> None:
        if self._gpu_sampler is None:
            return
        med, lo, hi, n = _clock_stats(self._gpu_sampler.samples())
        run.achieved_clock_median_mhz = med
        run.achieved_clock_min_mhz = lo
        run.achieved_clock_max_mhz = hi
        run.clock_samples = n

    def _safe_metrics(self, run: PointRun, when: str):
        try:
            return self._metrics_reader()
        except Exception as exc:  # noqa: BLE001 - a failed scrape is data, not a crash
            run.warnings.append(f"/metrics scrape {when} the run failed: {exc}")
            return None

    def _sample_gpu(self):
        if self._gpu_sampler is None:
            return None
        try:
            return self._gpu_sampler.poll()
        except Exception:  # noqa: BLE001
            return None

    # -- configuration application ---------------------------------------

    def _run_point(self, point: SweepPoint, round_index: int, execution_index: int):
        """Apply the configuration, measure under it, and return to stock.

        The stock point deliberately never constructs a guard. Arming publishes
        a claim, and a claim for a run that will not write is an operator-
        visible record of a modification that never happened.
        """
        if point.is_stock:
            # Stock settles too. It leads each round, so from round 1 on it
            # starts while the card is still ramping back from the previous
            # point's restore -- and by this module's own reasoning those
            # seconds belong to neither configuration. Charging them to the
            # drift detector and the SLO baseline would bias both.
            self._sleep(self.settle_s)
            return self._measure(point, round_index, execution_index)

        if self._guard_factory is None:
            raise SweepError(
                f"point {point.name!r} needs a hardware write but no guard was "
                "supplied. This is a bug, not a configuration error."
            )
        # Refuse before arming. Arming publishes a claim, and a claim for a
        # caller that was never allowed to write is an operator-visible record
        # of a modification that never happened. Ordering matters within the
        # block too: a bad clock must not be discovered after the power cap has
        # already landed on the device.
        if point.clock_mhz is not None and point.clock_mhz < MIN_SM_CLOCK_MHZ:
            raise SweepError(
                f"point {point.name!r} requests {point.clock_mhz} MHz, below "
                f"the {MIN_SM_CLOCK_MHZ} MHz floor; nothing was armed or written"
            )
        guard = self._guard_factory()
        # What actually reached the device for THIS point. The exception class
        # cannot answer that: `NvmlGpuController._write` translates an
        # NVML NoPermission into `PrivilegeError`, which is a *refusal* class,
        # so a device that accepts the clock lock and then refuses the power
        # write would be reported as "nothing was written" -- false, and false
        # in a durable artifact. Ordering the two writes differently only moves
        # which one is the lie.
        self._point_writes = []
        try:
            with guard:
                if point.clock_mhz is not None:
                    guard.lock_clocks(point.clock_mhz)
                    self._point_writes.append(f"clock {point.clock_mhz} MHz")
                if point.power_limit_mw is not None:
                    guard.set_power_limit_mw(point.power_limit_mw)
                    self._point_writes.append(
                        f"power {point.power_limit_mw // 1000} W")
                # Let the device settle at the new configuration before
                # measuring; a clock change is not instantaneous and the first
                # seconds after it belong to neither configuration.
                self._sleep(self.settle_s)
                measured = self._measure(point, round_index, execution_index)
        finally:
            # `_restored` False means the guard exited still owing a restore.
            # Nothing downstream may shut NVML down while that is true.
            if not getattr(guard, "_restored", True):
                self.hardware_may_be_modified = True
        return measured

    # -- the sweep --------------------------------------------------------

    def run(self) -> "SweepResult":
        execution_index = 0
        try:
            # Warm the card to thermal steady state BEFORE the first measured
            # point, and discard what the warm-up produces.
            #
            # Measured on real traffic: across three identical stock rounds,
            # generated tokens and tokens/sec were flat (18683/18603/18609 and
            # 686/683/683) while mean power rose 123.9 -> 128.3 -> 129.5 W as
            # the die went 40 -> 70 C. Same work, more joules: leakage current
            # rising with temperature. That is +4.95% joules/token of pure
            # thermal ramp, larger than the effect a sweep is trying to resolve.
            #
            # It is not merely noise, it is BIASED noise: stock leads the grid
            # and leads every round, so stock's first run is the coldest
            # measurement in the sweep -- and stock is the reference every other
            # point is compared against. Uncorrected, a cold stock flatters
            # itself and understates every clock lock.
            for _ in range(self.warmup_runs):
                self._load_runner(self.schedule)
            for round_index in range(self.repeats):
                for point in self.points:
                    measured = self._run_point(point, round_index, execution_index)
                    self.runs.append(measured)
                    execution_index += 1
        except REFUSALS as exc:
            # A refusal usually proves nothing was written, and saying "may
            # still be modified" there would spend the credibility of the one
            # message that has to be believed when it IS said. But a device
            # that refuses *mid-sequence* raises a refusal class after a write
            # already landed, so report what actually reached the device rather
            # than inferring it from the exception type.
            self.aborted = True
            self.abort_reason = f"guard refused at execution {execution_index}: {exc}. "
            if self._point_writes:
                self.abort_reason += (
                    "The device accepted "
                    + " and ".join(self._point_writes)
                    + " before refusing; those writes were restored on exit."
                )
            else:
                self.abort_reason += "Nothing was written to the GPU."
        except GpuGuardError as exc:
            self.aborted = True
            self.abort_reason = (
                f"guard failed at execution {execution_index}: {exc}."
            )
            if self.hardware_may_be_modified:
                self.abort_reason += (
                    " THE GPU MAY STILL BE MODIFIED -- run `sudo python -m "
                    "joule_agent.gpu_guard status` before trusting this "
                    "machine's next measurement."
                )
        except SweepError as exc:
            self.aborted = True
            self.abort_reason = str(exc)
        except Exception as exc:  # noqa: BLE001
            # Deliberately broad -- but NOT BaseException. `_write` translates
            # only NVML NoPermission into a GpuGuardError, so a NotSupported
            # power write -- ordinary on consumer parts -- escapes as a raw
            # NVMLError and would otherwise surface as "an unexpected error"
            # with no statement about hardware. A loop that writes GPU state
            # must say what it left behind whatever the exception class.
            # KeyboardInterrupt deliberately propagates: main()'s finally still
            # writes the partial results, and swallowing an operator's Ctrl+C
            # to keep measuring is not this tool's call to make.
            self.aborted = True
            self.abort_reason = (
                f"{type(exc).__name__} at execution {execution_index}: {exc}."
            )
            if self.hardware_may_be_modified:
                self.abort_reason += (
                    " THE GPU MAY STILL BE MODIFIED -- run `sudo python -m "
                    "joule_agent.gpu_guard status`."
                )
            elif self._point_writes:
                self.abort_reason += (
                    " The device accepted " + " and ".join(self._point_writes)
                    + " before failing; those writes were restored on exit."
                )
        return self._summarise()

    def partial_result(self, reason: str) -> "SweepResult":
        """Summarise what was collected before an unanticipated failure."""
        self.aborted = True
        self.abort_reason = reason
        return self._summarise()

    def _summarise(self) -> "SweepResult":
        result = SweepResult(
            preset=self.schedule.preset,
            seed=self.schedule.seed,
            schedule_digest=self.expected_digest,
            duration_s=self.schedule.duration_s,
            repeats=self.repeats,
            points=[p.to_dict() for p in self.points],
            runs=[m.run for m in self.runs],
            aborted=self.aborted,
            abort_reason=self.abort_reason,
            hardware_may_be_modified=self.hardware_may_be_modified,
        )
        result.finalise(self.runs, slo_ms=self.slo_ms, slo_slack=self.slo_slack)
        return result


# ---------------------------------------------------------------------------
# Result and analysis
# ---------------------------------------------------------------------------


@dataclass
class PointSummary:
    name: str = ""
    runs: int = 0
    joules_per_token_mean: float | None = None
    joules_per_token_stdev: float | None = None
    tokens_per_s_mean: float | None = None
    latency_p99_mean_s: float | None = None
    slo_violation_rate_mean: float | None = None
    achieved_clock_median_mhz: float | None = None
    meets_slo: bool = False
    #: False for any point that caps power. The clock axis has a behavioural
    #: efficacy gate (`tools/verify_clock_lock.py`); the power axis has none,
    #: so a power row's numbers are collected but not trusted.
    axis_verified: bool = True
    #: True when this point's mean is within one standard deviation of the
    #: best point's mean -- i.e. the sweep cannot tell them apart.
    tied_with_best: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SweepResult:
    preset: str = ""
    seed: int = 0
    schedule_digest: str = ""
    duration_s: float = 0.0
    repeats: int = 0
    points: list = field(default_factory=list)
    runs: list = field(default_factory=list)
    aborted: bool = False
    abort_reason: str | None = None
    hardware_may_be_modified: bool = False

    summaries: list = field(default_factory=list)
    slo_threshold_s: float | None = None
    slo_threshold_source: str = ""
    best_point: str | None = None
    tied_points: list = field(default_factory=list)
    #: Set only when an UNVERIFIED (power-capping) point beat every verified
    #: one. Kept separate from `best_point` so the headline answer never rests
    #: on an axis with no efficacy gate.
    best_unverified_point: str | None = None
    stock_drift_pct: float | None = None
    warnings: list = field(default_factory=list)

    # -- analysis ---------------------------------------------------------

    def finalise(self, measured: list, *, slo_ms=None, slo_slack=DEFAULT_SLO_SLACK) -> None:
        by_point: dict = {}
        for m in measured:
            by_point.setdefault(m.run.point, []).append(m)

        self._derive_slo(by_point, slo_ms, slo_slack)
        self._detect_drift(by_point)
        self._summarise_points(by_point)
        if self.aborted:
            # An aborted sweep must not name a best static point. The grid was
            # not measured, the rounds are unequal, and in the failed-restore
            # case the card was in an unknown state -- a durable JSON naming a
            # winner from that is the "artifact cited for a run that never
            # happened" failure this project has already had once.
            self.warnings.append(
                "sweep aborted, so no best point was chosen: the grid is "
                "incomplete and its points were not measured under comparable "
                "conditions. Every figure below is partial."
            )
            return
        uneven = {s.name: s.runs for s in self.summaries
                  if s.runs and s.runs != self.repeats}
        if uneven:
            self.warnings.append(
                f"unequal repeat counts {uneven} against --repeats "
                f"{self.repeats}: means over different sample counts are not "
                "comparable, and a single draw can out-rank a three-run mean."
            )
        self._pick_best()

    def _derive_slo(self, by_point, slo_ms, slo_slack) -> None:
        if slo_ms is not None:
            self.slo_threshold_s = slo_ms / 1000.0
            self.slo_threshold_source = f"explicit --slo-ms {slo_ms}"
            return
        stock = [m.run.latency_p99_s for m in by_point.get("stock", [])
                 if m.run.latency_p99_s is not None]
        if not stock:
            self.warnings.append(
                "no stock p99 available, so no SLO threshold could be derived; "
                "SLO-violation rates are absent. Pass --slo-ms to set one."
            )
            return
        base = statistics.median(stock)
        self.slo_threshold_s = round(base * slo_slack, 4)
        self.slo_threshold_source = (
            f"derived: stock median p99 {base:.4f}s x {slo_slack}"
        )

    def _detect_drift(self, by_point) -> None:
        """Compare stock against itself across rounds.

        Interleaving repeats spreads thermal drift across configurations, which
        hides it rather than measuring it. Stock recurs every round and is by
        definition the same configuration each time, so any change in it is
        drift -- and drift above the noise floor contaminates every
        between-configuration comparison in the sweep.
        """
        runs = sorted(
            (m.run for m in by_point.get("stock", [])
             if m.run.joules_per_generated_token),
            key=lambda r: r.round_index,
        )
        if len(runs) < 2:
            return
        first, last = runs[0].joules_per_generated_token, runs[-1].joules_per_generated_token
        if not first:
            return
        self.stock_drift_pct = round((last - first) / first * 100.0, 2)
        if abs(self.stock_drift_pct) > NOISE_FLOOR_FRACTION * 100.0:
            self.warnings.append(
                f"stock joules/token drifted {self.stock_drift_pct:+.2f}% between "
                f"the first and last round, above the {NOISE_FLOOR_FRACTION:.0%} "
                "run-to-run noise floor. The machine did not hold still, so "
                "between-configuration differences of this size are not "
                "attributable to the configuration."
            )

    def _summarise_points(self, by_point) -> None:
        for spec in self.points:
            name = spec["name"]
            group = by_point.get(name, [])
            s = PointSummary(name=name, runs=len(group),
                             axis_verified=spec.get("power_limit_mw") is None)
            jpt = [m.run.joules_per_generated_token for m in group
                   if m.run.joules_per_generated_token is not None]
            tps = [m.run.generated_tokens_per_s for m in group
                   if m.run.generated_tokens_per_s is not None]
            p99 = [m.run.latency_p99_s for m in group if m.run.latency_p99_s is not None]
            clk = [m.run.achieved_clock_median_mhz for m in group
                   if m.run.achieved_clock_median_mhz is not None]
            if jpt:
                s.joules_per_token_mean = round(statistics.mean(jpt), 6)
                s.joules_per_token_stdev = (
                    round(statistics.stdev(jpt), 6) if len(jpt) > 1 else 0.0
                )
            if tps:
                s.tokens_per_s_mean = round(statistics.mean(tps), 2)
            if p99:
                s.latency_p99_mean_s = round(statistics.mean(p99), 4)
            if clk:
                s.achieved_clock_median_mhz = round(statistics.mean(clk), 1)

            if self.slo_threshold_s is not None:
                rates = [
                    r for r in (slo_violation_rate(m.latencies_s, self.slo_threshold_s)
                                for m in group)
                    if r is not None
                ]
                if rates:
                    s.slo_violation_rate_mean = round(statistics.mean(rates), 4)
                s.meets_slo = (
                    s.latency_p99_mean_s is not None
                    and s.latency_p99_mean_s <= self.slo_threshold_s
                )
            self.summaries.append(s)

        self._flag_indistinguishable_clocks()
        self._flag_requested_vs_achieved()

    def _flag_requested_vs_achieved(self) -> None:
        """A row named for its request is mislabelled if the device snapped.

        The 3060 Ti snaps SM clocks to a 15 MHz grid: ask 1400, get 1410; ask
        1000, get 1005. `SweepPoint.name` and therefore `best_point` are built
        from the REQUEST, so "best = clk1400" would name a configuration the
        card never ran. The achieved value is the real setting and is what any
        downstream citation must use.
        """
        by_name = {s["name"]: s for s in self.points}
        drifted = []
        for s in self.summaries:
            spec = by_name.get(s.name, {})
            req = spec.get("clock_mhz")
            got = s.achieved_clock_median_mhz
            if req is None or got is None:
                continue
            if abs(got - req) >= 1:
                drifted.append(f"{s.name} ran at {got:.0f} MHz")
        if drifted:
            self.warnings.append(
                "requested clocks are not the achieved clocks -- "
                + "; ".join(drifted)
                + ". Point names are built from the REQUEST, so cite "
                  "achieved_clock_median_mhz, not the label."
            )

    def _flag_indistinguishable_clocks(self) -> None:
        """Two requests that achieved the same clock are not two data points."""
        seen: dict = {}
        for s in self.summaries:
            if s.achieved_clock_median_mhz is None:
                continue
            seen.setdefault(round(s.achieved_clock_median_mhz), []).append(s.name)
        for achieved, names in seen.items():
            if len(names) > 1:
                self.warnings.append(
                    f"points {', '.join(names)} all achieved ~{achieved} MHz: the "
                    "device snapped different requests to the same clock bin, so "
                    "these are one configuration measured repeatedly, not several."
                )

    def _pick_best(self) -> None:
        """Lowest joules/token among points that meet the SLO -- with ties.

        Not a bare argmax. At a ~2% noise floor the difference between the top
        two points is routinely smaller than the sweep can resolve, and naming
        one of them "best" asserts a distinction the measurement does not
        support.
        """
        scored = [s for s in self.summaries
                  if s.joules_per_token_mean is not None
                  and (self.slo_threshold_s is None or s.meets_slo)]
        # The headline best point is chosen among VERIFIED rows only. Power
        # rows are collected -- a null result there cheaply corroborates the
        # published "illusion of power capping" finding, and "nothing happened"
        # is a weak claim that weak evidence can carry -- but a power row
        # WINNING is a surprising result, and that needs the efficacy gate this
        # axis does not have before it becomes the answer.
        eligible = [s for s in scored if s.axis_verified]
        unverified = [s for s in scored if not s.axis_verified]
        if unverified:
            self.warnings.append(
                f"{len(unverified)} power-capping point(s) were measured on an "
                "axis with NO efficacy gate: nothing has confirmed on this "
                "hardware that a power cap changes behaviour rather than being "
                "silently ignored. Their numbers are recorded, not trusted, and "
                "they are excluded from the best-point choice."
            )
        if not eligible:
            self.warnings.append(
                "no configuration both produced a joules/token figure and met "
                "the SLO; there is no best static point in this sweep."
            )
            return
        best = min(eligible, key=lambda s: s.joules_per_token_mean)
        self.best_point = best.name
        band = max(best.joules_per_token_stdev or 0.0,
                   best.joules_per_token_mean * NOISE_FLOOR_FRACTION)
        for s in eligible:
            if s.joules_per_token_mean <= best.joules_per_token_mean + band:
                s.tied_with_best = True
                if s.name != best.name:
                    self.tied_points.append(s.name)
        if self.tied_points:
            self.warnings.append(
                f"{best.name} is best by mean, but {', '.join(self.tied_points)} "
                "are within the resolution of this sweep. Treat them as tied "
                "rather than ranked."
            )
        beat_it = [s for s in unverified
                   if s.joules_per_token_mean < best.joules_per_token_mean - band]
        if beat_it:
            winner = min(beat_it, key=lambda s: s.joules_per_token_mean)
            self.best_unverified_point = winner.name
            self.warnings.append(
                f"{winner.name} beat the best verified point "
                f"({winner.joules_per_token_mean:.6f} vs "
                f"{best.joules_per_token_mean:.6f} J/token) -- but it caps "
                "power, an axis with no efficacy gate on this hardware. That is "
                "a SURPRISING result on an unverified axis, which is exactly "
                "the case that needs the gate built before it is believed. "
                "Build the power-limit equivalent of tools/verify_clock_lock.py "
                "before citing this."
            )

    def to_dict(self) -> dict:
        return {
            "preset": self.preset,
            "seed": self.seed,
            "schedule_digest": self.schedule_digest,
            "duration_s": self.duration_s,
            "repeats": self.repeats,
            "points": self.points,
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
            "hardware_may_be_modified": self.hardware_may_be_modified,
            "slo_threshold_s": self.slo_threshold_s,
            "slo_threshold_source": self.slo_threshold_source,
            "best_point": self.best_point,
            "tied_points": self.tied_points,
            "best_unverified_point": self.best_unverified_point,
            "stock_drift_pct": self.stock_drift_pct,
            "summaries": [s.to_dict() for s in self.summaries],
            "runs": [r.to_dict() for r in self.runs],
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Live GPU sampling
# ---------------------------------------------------------------------------


class NvmlSweepSampler:
    """Samples SM clock and temperature in a background thread during a run.

    Read-only: it never writes GPU state, so it holds no guard and needs no
    consent. Uses its own NVML session rather than borrowing the guard's, so a
    stock point -- which arms no guard at all -- samples identically to a
    locked one.
    """

    def __init__(self, index: int = 0, interval_s: float = 0.25) -> None:
        import pynvml

        self._nvml = pynvml
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        self.interval_s = interval_s
        self._samples: list = []
        self._thread = None
        self._stop = None

    def poll(self) -> dict:
        n = self._nvml
        out = {"energy_mj": None, "temperature_c": None, "sm_clock_mhz": None}
        for key, fn in (
            ("energy_mj", lambda: n.nvmlDeviceGetTotalEnergyConsumption(self._handle)),
            ("temperature_c", lambda: n.nvmlDeviceGetTemperature(
                self._handle, n.NVML_TEMPERATURE_GPU)),
            ("sm_clock_mhz", lambda: n.nvmlDeviceGetClockInfo(
                self._handle, n.NVML_CLOCK_SM)),
        ):
            try:
                out[key] = fn()
            except Exception:  # noqa: BLE001 - a missing field is not a failed run
                pass
        return out

    def start(self) -> None:
        import threading

        self._samples = []
        self._stop = threading.Event()

        def loop():
            while not self._stop.is_set():
                self._samples.append(self.poll().get("sm_clock_mhz"))
                self._stop.wait(self.interval_s)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def samples(self) -> list:
        return list(self._samples)

    def close(self) -> None:
        self.stop()
        try:
            self._nvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _git(*args) -> str | None:
    try:
        r = subprocess.run(["git", *args],
                           cwd=Path(__file__).resolve().parent.parent,
                           capture_output=True, text=True, timeout=10)
    except Exception:  # noqa: BLE001
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def _dirty_flag():
    """``True``/``False``/``None`` -- unknown is not clean.

    ``bool(_git(...))`` turns a failed git call into ``False``, which reads as
    a clean tree: a field whose failure mode is indistinguishable from a real
    answer. And a sha does not identify the code when the tree IS dirty, which
    is why the module hash below is recorded beside it.
    """
    porcelain = _git("status", "--porcelain")
    return None if porcelain is None else bool(porcelain)


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def provenance(args) -> dict:
    """Stamp what produced this sweep.

    Same reasoning as `tools/verify_clock_lock.py`: a results file with no note
    of which tree, driver and card produced it can be cited for a code path it
    never measured. That has already happened once in this repo.
    """
    here = Path(__file__).resolve().parent
    prov = {
        "utc": _utcnow(),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": _dirty_flag(),
        "sweep_sha256": _sha256(here / "sweep.py"),
        "gpu_guard_sha256": _sha256(here / "gpu_guard.py"),
        "python": sys.executable,
        "argv": list(sys.argv),
        "endpoint": args.endpoint,
        "model": args.model,
        "driver_version": None,
        "device": None,
    }
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            try:
                v = pynvml.nvmlSystemGetDriverVersion()
                prov["driver_version"] = v.decode() if isinstance(v, bytes) else str(v)
            except Exception:  # noqa: BLE001
                pass
            h = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)
            dev = {"index": args.gpu}
            for key, fn in (("name", pynvml.nvmlDeviceGetName),
                            ("uuid", pynvml.nvmlDeviceGetUUID)):
                try:
                    v = fn(h)
                    dev[key] = v.decode() if isinstance(v, bytes) else str(v)
                except Exception:  # noqa: BLE001
                    dev[key] = None
            prov["device"] = dev
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001
        pass
    return prov


def _parse_clocks(spec: str) -> list:
    """``900,1200,1500`` or ``900:1900:200`` (min:max:step, inclusive)."""
    spec = spec.strip()
    if not spec:
        return []
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) != 3:
            raise SweepError(f"bad clock range {spec!r}: want MIN:MAX:STEP")
        lo, hi, step = (int(p) for p in parts)
        if step <= 0:
            raise SweepError("clock step must be positive")
        if hi < lo:
            raise SweepError(f"bad clock range {spec!r}: max is below min")
        return list(range(lo, hi + 1, step))
    return [int(p) for p in spec.split(",") if p.strip()]


def preflight_stale(controller, state_path) -> str | None:
    """``None`` if the card is clean, else the operator-facing refusal.

    Lives outside ``main()`` on purpose. The first version of this call passed
    ``(state_path, index=...)`` against a signature of ``(controller,
    state_path)`` -- an uncaught ``TypeError`` that made every locked sweep die
    on a traceback, and that 241 tests could not see because none of them
    imported ``main``. Logic that only ``main()`` can reach is logic nothing
    checks.
    """
    stale = check_stale_state(controller, state_path)
    if not stale.get("stale"):
        return None
    lines = ["the guard state file is not clean, so this card's baseline "
             "cannot be assumed to be stock:"]
    lines += [f"  - {f}" for f in stale.get("findings", [])]
    rec = stale.get("recommendation")
    if rec:
        lines.append(f"  {rec}")
    return "\n".join(lines)


def _chown_to_invoking_user(written) -> None:
    """Hand root-created output back to the operator who ran sudo.

    A sweep needs root, so without this every artifact it produces is
    root-owned and the analysis that follows -- which needs no privilege --
    cannot read it. That is already true of `results/gpu_guard_state.json`.
    """
    uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
    if uid is None or gid is None:
        return
    # Only what this run wrote. `rglob("*")` over a caller-supplied --out
    # chowns pre-existing files it did not create, as root -- `--out /etc`
    # being the obvious disaster. follow_symlinks=False because a symlink
    # planted in the output directory would otherwise be chowned through to
    # its target.
    for p in written:
        try:
            os.chown(p, int(uid), int(gid), follow_symlinks=False)
        except OSError:
            pass


def write_outputs(result: SweepResult, measured: list, out_dir: Path,
                  prov: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [out_dir]
    doc = result.to_dict()
    doc["provenance"] = prov
    (out_dir / "sweep.json").write_text(json.dumps(doc, indent=2))
    written.append(out_dir / "sweep.json")

    if result.runs:
        written.append(out_dir / "runs.csv")
        with (out_dir / "runs.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(result.runs[0].to_dict().keys()))
            w.writeheader()
            for r in result.runs:
                row = r.to_dict()
                row["warnings"] = "; ".join(row.get("warnings") or [])
                w.writerow(row)

    lat_dir = out_dir / "latencies"
    lat_dir.mkdir(exist_ok=True)
    written.append(lat_dir)
    for m in measured:
        if not m.latencies_s:
            continue
        name = f"{m.run.point}_r{m.run.round_index}.csv"
        with (lat_dir / name).open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["latency_s"])
            for v in m.latencies_s:
                w.writerow([v])
        written.append(lat_dir / name)
    _chown_to_invoking_user(written)


def format_report(result: SweepResult) -> str:
    lines = ["", "=" * 78, "STATIC SWEEP", "=" * 78,
             f"preset {result.preset}  seed {result.seed}  "
             f"digest {result.schedule_digest[:16]}  repeats {result.repeats}"]
    if result.slo_threshold_s is not None:
        lines.append(f"SLO    p99 <= {result.slo_threshold_s:.3f}s "
                     f"({result.slo_threshold_source})")
    lines.append("")
    lines.append(f"{'point':<14}{'n':>4}{'J/tok':>12}{'+/-':>10}{'tok/s':>9}"
                 f"{'p99 s':>9}{'SLO viol':>10}{'clk MHz':>9}")
    lines.append("-" * 78)
    for s in result.summaries:
        def f(v, spec=".4f"):
            return "n/a" if v is None else format(v, spec)
        flag = ""
        if s.name == result.best_point:
            flag = "  <- best"
        elif s.tied_with_best:
            flag = "  (tied)"
        if not s.axis_verified:
            flag += "  *UNVERIFIED AXIS"
        lines.append(
            f"{s.name:<14}{s.runs:>4}{f(s.joules_per_token_mean, '.6f'):>12}"
            f"{f(s.joules_per_token_stdev, '.6f'):>10}"
            f"{f(s.tokens_per_s_mean, '.1f'):>9}"
            f"{f(s.latency_p99_mean_s, '.3f'):>9}"
            f"{f(s.slo_violation_rate_mean, '.3f'):>10}"
            f"{f(s.achieved_clock_median_mhz, '.0f'):>9}{flag}"
        )
    lines.append("-" * 78)
    if any(not s.axis_verified for s in result.summaries):
        lines.append("* power-capping rows: measured, NOT trusted -- the power "
                     "axis has no efficacy gate on this hardware.")
    if result.stock_drift_pct is not None:
        lines.append(f"stock drift across rounds: {result.stock_drift_pct:+.2f}%")
    if result.aborted:
        lines.append("")
        lines.append(f"ABORTED: {result.abort_reason}")
    if result.hardware_may_be_modified:
        lines.append("THE GPU MAY STILL BE MODIFIED -- "
                     "`sudo python -m joule_agent.gpu_guard status`")
    for w in result.warnings:
        lines.append(f"WARNING: {w}")
    lines.append("=" * 78)
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m joule_agent.sweep",
        description="Static clock/power sweep -- finds the tuned-static baseline.",
    )
    p.add_argument("--endpoint", default="http://localhost:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--preset", default="poisson",
                   help="open-loop preset; closed-loop presets are refused")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--clocks", default="",
                   help="MHz list '900,1200,1500' or range 'MIN:MAX:STEP'. "
                        "Empty means stock only -- a valid dry run needing no root.")
    p.add_argument("--power-limits-w", default="",
                   help="optional power caps in watts, e.g. '150,175,200'")
    p.add_argument("--repeats", type=int, default=3,
                   help="rounds over the whole grid (default 3). Below 3 the "
                        "result cannot be separated from the ~2%% noise floor.")
    p.add_argument("--slo-ms", type=float, default=None,
                   help="explicit p99 latency SLO. Default: derived from the "
                        "stock point's median p99 x --slo-slack.")
    p.add_argument("--slo-slack", type=float, default=DEFAULT_SLO_SLACK,
                   help=f"multiplier on stock p99 defining the SLO "
                        f"(default {DEFAULT_SLO_SLACK}).")
    p.add_argument("--warmup-runs", type=int, default=1,
                   help="discarded full-load passes before the first measured "
                        "point, to reach thermal steady state (default 1). "
                        "Measured on this card: joules/token rose ~5%% across "
                        "three identical stock rounds purely from the die "
                        "warming 40->70 C, and stock runs first, so an "
                        "uncorrected sweep flatters its own reference.")
    p.add_argument("--settle-s", type=float, default=5.0,
                   help="pause after applying a config before measuring")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", default=None, help="output directory")
    p.add_argument("--state-path", default="results/gpu_guard_state.json")
    p.add_argument("--i-understand-this-changes-gpu-settings",
                   dest="consent", action="store_true",
                   help="required for any point that locks clocks or caps power")
    args = p.parse_args(argv)

    # --- refusals that depend only on the arguments, before any device or
    # --- guard is touched. This is the rule the CLI seam section records:
    # --- every refusal that does not depend on the device happens first.
    try:
        clocks = _parse_clocks(args.clocks)
        powers = [int(w) * 1000 for w in args.power_limits_w.split(",") if w.strip()]
    except (SweepError, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    if args.preset not in PRESETS:
        print(f"REFUSED: unknown preset {args.preset!r}. Available: "
              f"{', '.join(sorted(PRESETS))}", file=sys.stderr)
        return 2
    if not PRESETS[args.preset].reproducible:
        print(f"REFUSED: preset {args.preset!r} is closed-loop, so offered load "
              "depends on server response time and configurations would not be "
              "compared under the same traffic. Open-loop presets: "
              + ", ".join(n for n, q in PRESETS.items() if q.reproducible),
              file=sys.stderr)
        return 2
    if args.repeats < 1:
        print("REFUSED: --repeats must be at least 1", file=sys.stderr)
        return 2

    # The grid is built here, before any refusal that depends on it, because
    # `build_grid` touches no device -- so this keeps the argument-only
    # ordering rule while letting the refusals describe the run that will
    # actually happen. Deriving `writes_hardware` from the arguments instead
    # let the two diverge: power limits with no clocks produced a stock-only
    # grid that still demanded consent, root and the stale pre-flight, then
    # applied nothing.
    points = build_grid(clocks, powers)
    writes_hardware = any(not p.is_stock for p in points)

    if writes_hardware and not args.consent:
        print("REFUSED: this sweep locks clocks and/or caps power. Pass "
              "--i-understand-this-changes-gpu-settings to proceed. "
              "Run with no --clocks/--power-limits-w for a stock-only dry run.",
              file=sys.stderr)
        return 2

    # The clock floor depends only on the argument, so it is refused here --
    # not inside the guard, and never after a power write has already landed.
    # gpu_guard learned this in pass 12; reintroducing it one layer out would
    # arm, publish a claim, cap power, and only then refuse.
    below_floor = [c for c in clocks if c < MIN_SM_CLOCK_MHZ]
    if below_floor:
        print(f"REFUSED: clock(s) {below_floor} are below the "
              f"{MIN_SM_CLOCK_MHZ} MHz floor. The floor refuses rather than "
              "clamping, so no part of this grid will be run.", file=sys.stderr)
        return 2

    # Privilege depends only on the caller, never the device, so it belongs
    # beside consent. Without it an unprivileged sweep runs a full stock point
    # -- minutes of load -- then arms, publishes a claim, and only then fails.
    if writes_hardware and os.geteuid() != 0:
        print("REFUSED: locking clocks or capping power requires root. "
              "Re-run under sudo, or drop --clocks/--power-limits-w for a "
              "stock-only dry run that needs no privilege.", file=sys.stderr)
        return 2

    if args.repeats < 3:
        print(f"WARNING: --repeats {args.repeats} is below 3. Component 3 "
              "measured ~2% run-to-run variance under bit-identical traffic, so "
              "the best point will not be separable from noise.", file=sys.stderr)

    # Create the output directory BEFORE measuring anything. A sweep is minutes
    # of GPU time; discovering at write time that the destination is not
    # writable throws all of it away. `results/` ends up root-owned as soon as
    # any privileged run touches it, which makes this the likely failure, not
    # an exotic one.
    out_dir = Path(args.out) if args.out else Path("results/sweeps") / time.strftime(
        "%Y%m%dT%H%M%SZ", time.gmtime())
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"REFUSED: cannot create output directory {out_dir}: {exc}",
              file=sys.stderr)
        print("  A previous privileged run may own it. Either pass --out to a "
              "writable path, or hand it back with "
              f"`sudo chown -R $USER {out_dir.parents[-2]}`.", file=sys.stderr)
        return 2

    schedule = build_schedule(args.preset, seed=args.seed, duration_s=args.duration)

    guard_factory = None
    sampler = None
    # One controller per point. `_ClaimMixin.claim` refuses a second owner and
    # nothing ever releases `_claimed_by`, so handing one controller to a fresh
    # GpuGuard per point gets the SECOND point refused -- which `run()` would
    # then report as an abort saying the GPU may still be modified. Releasing
    # the claim would mean editing the protected surface; a fresh controller
    # per point does not, at the cost of an NVML init per point.
    controllers: list = []
    if writes_hardware:
        # A sweep must not start on a card another process is holding, or on
        # one owing a restore -- both mean the baseline is not stock.
        # check_stale_state reads the device, so it needs a controller.
        # Wrapped: NvmlGpuController.__init__ does nvmlInit() then
        # nvmlDeviceGetHandleByIndex, so a mistyped --gpu raises a raw
        # NVMLError traceback rather than an answer. gpu_guard._controller()
        # exists because that exact bug reached committed code there; the
        # sampler eight lines below was already wrapped and this was not.
        try:
            probe = NvmlGpuController(index=args.gpu)
        except Exception as exc:  # noqa: BLE001
            print(f"REFUSED: cannot open GPU {args.gpu}: {exc}", file=sys.stderr)
            print("  Check `nvidia-smi --query-gpu=index,uuid --format=csv` "
                  "for the indexes that exist on this machine.", file=sys.stderr)
            return 2
        try:
            refusal = preflight_stale(probe, args.state_path)
        finally:
            probe.close()
        if refusal is not None:
            print(f"REFUSED: {refusal}", file=sys.stderr)
            return 2

        def guard_factory():
            c = NvmlGpuController(index=args.gpu)
            controllers.append(c)
            return GpuGuard(c, consent=True, state_path=args.state_path)

    try:
        sampler = NvmlSweepSampler(index=args.gpu)
    except Exception as exc:  # noqa: BLE001
        sampler = None
        if writes_hardware:
            # Locking clocks for minutes and producing no energy, no
            # joules/token and no achieved clocks is a hardware-writing run
            # that cannot answer the question it writes hardware to answer.
            print(f"REFUSED: GPU sampling is unavailable ({exc}), so this "
                  "sweep could not measure energy or achieved clocks. Refusing "
                  "to write GPU state for a run that cannot produce a result.",
                  file=sys.stderr)
            return 2
        print(f"WARNING: no GPU sampling ({exc}); achieved clocks and energy "
              "will be absent from this sweep.", file=sys.stderr)

    runner = SweepRunner(
        endpoint=args.endpoint, model=args.model, schedule=schedule,
        points=points, repeats=args.repeats, slo_ms=args.slo_ms,
        slo_slack=args.slo_slack, settle_s=args.settle_s,
        warmup_runs=args.warmup_runs,
        guard_factory=guard_factory, gpu_sampler=sampler,
    )
    prov = provenance(args)
    result = None
    try:
        result = runner.run()
    finally:
        if sampler is not None:
            sampler.close()
        # close() shuts NVML down. If any guard is still holding unrestored
        # state, its atexit last-chance restore would then run against a dead
        # handle -- turning a possible recovery into a certain failure. Same
        # hazard gpu_guard.main() and tools/verify_clock_lock.py guard against.
        if runner.hardware_may_be_modified:
            print("WARNING [sweep]: a restore did not complete; leaving NVML "
                  "open so the exit-time restore can still run. Check `sudo "
                  "python -m joule_agent.gpu_guard status`.", file=sys.stderr)
        else:
            for c in controllers:
                c.close()
        # Results are written even if the sweep died in a way it did not
        # anticipate -- a KeyboardInterrupt, or a raise out of the load
        # generator. Minutes of GPU time should not be discarded because the
        # failure was of an unexpected shape.
        if result is None:
            # Also when nothing completed: an empty root-owned output
            # directory with no sweep.json is worse than a document saying the
            # run died, because only one of the two can be read afterwards.
            try:
                result = runner.partial_result(
                    "the sweep ended on an unexpected error; these results are "
                    "partial and no best point was chosen")
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING [sweep]: could not summarise partial results: "
                      f"{exc}", file=sys.stderr)
        if result is not None:
            # Guarded: a raise here (ENOSPC, EPERM) would replace whatever
            # exception is already unwinding and swallow the report with it.
            # The hardware warning above is printed first for that reason.
            try:
                write_outputs(result, runner.runs, out_dir, prov)
                print(format_report(result))
                print(f"\nwritten to {out_dir}")
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING [sweep]: results could not be written to "
                      f"{out_dir}: {exc}", file=sys.stderr)

    return 1 if (result is None or result.aborted) else 0


if __name__ == "__main__":
    raise SystemExit(main())
