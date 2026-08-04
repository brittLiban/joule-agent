"""Tests for the static sweep runner.

Component 4 is where tests start guarding *results* rather than hardware state,
so the bar here is that each test can fail for the reason it names. Where a
test pins a property that a plausible refactor would silently break, the
mutation that breaks it is named in the docstring.
"""

import pytest

from joule_agent.loadgen import RunReport, build_schedule
from joule_agent.gpu_guard import GpuGuardError
from joule_agent.sweep import (
    DEFAULT_SLO_SLACK,
    NOISE_FLOOR_FRACTION,
    SweepError,
    SweepPoint,
    SweepRunner,
    build_grid,
    slo_violation_rate,
    _parse_clocks,
)
from joule_agent.workload import WorkloadSnapshot


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeGuard:
    """Records what was applied, and can be told to fail like the real one.

    Models `_restored` because the runner keys its "GPU may still be modified"
    decision on it. A fake without that attribute would let every test pass
    through `getattr(guard, "_restored", True)` and never exercise the branch.
    """

    def __init__(self, log, fail_on_enter=None, fail_on_exit=None):
        self.log = log
        self.fail_on_enter = fail_on_enter
        self.fail_on_exit = fail_on_exit
        self._restored = True

    def __enter__(self):
        if self.fail_on_enter:
            raise self.fail_on_enter
        self.log.append("arm")
        return self

    def __exit__(self, *exc):
        if self.fail_on_exit:
            self._restored = False
            raise self.fail_on_exit
        self.log.append("restore")
        return False

    def lock_clocks(self, mhz, max_mhz=None):
        self.log.append(("lock", mhz))

    def set_power_limit_mw(self, mw):
        self.log.append(("power", mw))


class FakeSampler:
    """Energy counter and clock samples that respond to the applied config."""

    def __init__(self, clock_by_call=None, energy_step_mj=1000.0):
        self.energy_mj = 0.0
        self.energy_step_mj = energy_step_mj
        self.clock_by_call = clock_by_call or []
        self._call = 0
        self._samples = []
        self.temperature_c = 40.0

    def poll(self):
        out = {"energy_mj": self.energy_mj,
               "temperature_c": self.temperature_c,
               "sm_clock_mhz": 1000}
        self.energy_mj += self.energy_step_mj
        return out

    def start(self):
        idx = min(self._call, len(self.clock_by_call) - 1) if self.clock_by_call else 0
        self._samples = ([self.clock_by_call[idx]] * 5) if self.clock_by_call else []
        self._call += 1

    def stop(self):
        pass

    def samples(self):
        return list(self._samples)


def snapshot(gen, prompt=0.0):
    return WorkloadSnapshot(generation_tokens_total=gen, prompt_tokens_total=prompt)


class FakeWorld:
    """A serving engine and a GPU whose behaviour the test dictates."""

    def __init__(self, digest, *, tokens_per_run=1000.0, latencies=None,
                 energy_per_run=None):
        self.digest = digest
        self.tokens_per_run = tokens_per_run
        self.latencies = latencies if latencies is not None else [0.1] * 10
        self.energy_per_run = energy_per_run or []
        self.gen_total = 0.0
        self.runs = 0
        self._pending = None

    def metrics(self):
        return snapshot(self.gen_total)

    def load_runner(self, schedule):
        self.gen_total += self.tokens_per_run
        rep = RunReport(preset=schedule.preset, seed=schedule.seed,
                        schedule_digest=self.digest, reproducible=True,
                        completed=len(self.latencies))
        lat = sorted(self.latencies)
        rep.latencies_s = lat
        rep.latency_p50_s = lat[len(lat) // 2]
        rep.latency_p99_s = lat[-1]
        self.runs += 1
        return rep


# ---------------------------------------------------------------------------
# Traffic identity -- the contract inherited from component 3
# ---------------------------------------------------------------------------


def test_a_closed_loop_preset_is_refused_before_anything_runs():
    """Mutation: dropping the reproducible check lets the sweep construct."""
    sched = build_schedule("saturated", seed=1, duration_s=5.0)
    with pytest.raises(SweepError) as exc:
        SweepRunner(endpoint="e", model="m", schedule=sched,
                    points=[SweepPoint()])
    assert "not reproducible" in str(exc.value)
    assert "confound" in str(exc.value)


def test_every_point_replays_the_same_schedule_object():
    """The digest is identical by construction, not by convention.

    Deliberately NOT `id()`. CPython reuses addresses for same-sized objects --
    200 build-and-drop cycles here yielded 18 distinct ids with a run of 178
    identical ones -- so an id-based assertion's truth condition is "the
    allocator did not hand back the same address", which is incidental to
    whatever else `_measure` allocates. A marker attribute is carried by the
    object itself and survives being sliced out of the free list.

    Mutation: rebuild the schedule per point and the marker is absent.
    """
    sched = build_schedule("poisson", seed=7, duration_s=5.0)
    sched._replay_marker = object()
    seen = []

    def runner(schedule):
        seen.append(getattr(schedule, "_replay_marker", None))
        rep = RunReport(schedule_digest=sched.digest())
        rep.latencies_s = [0.1]
        return rep

    SweepRunner(endpoint="e", model="m", schedule=sched,
                points=[SweepPoint(), SweepPoint(clock_mhz=900)], repeats=2,
                guard_factory=lambda: FakeGuard([]), load_runner=runner,
                sleep=lambda s: None).run()
    assert len(seen) == 4
    assert all(m is sched._replay_marker for m in seen)


def test_a_changed_digest_voids_the_sweep():
    """A run reporting different traffic must abort, not be averaged in.

    Mutation: removing the digest comparison lets the sweep complete.
    """
    sched = build_schedule("poisson", seed=7, duration_s=5.0)

    def runner(schedule):
        rep = RunReport(schedule_digest="not-the-same-traffic")
        rep.latencies_s = [0.1]
        return rep

    result = SweepRunner(endpoint="e", model="m", schedule=sched,
                         points=[SweepPoint()], load_runner=runner,
                         sleep=lambda s: None).run()
    assert result.aborted
    assert "digest changed" in result.abort_reason
    assert "void" in result.abort_reason


# ---------------------------------------------------------------------------
# Hardware handling
# ---------------------------------------------------------------------------


def test_the_stock_point_never_constructs_a_guard():
    """Arming publishes a claim; a run that will not write must not claim.

    Mutation: guarding the stock point makes the factory run and the log fill.
    """
    calls = []

    def factory():
        calls.append("built")
        return FakeGuard([])

    sched = build_schedule("poisson", seed=7, duration_s=5.0)
    world = FakeWorld(sched.digest())
    runner = SweepRunner(endpoint="e", model="m", schedule=sched,
                         points=[SweepPoint()], repeats=2,
                         guard_factory=factory, load_runner=world.load_runner,
                         metrics_reader=world.metrics, sleep=lambda s: None)
    result = runner.run()
    assert calls == []
    # `calls == []` alone is satisfied by the point never running at all, so
    # pin that it DID run -- otherwise refusing every stock point would pass.
    assert world.runs == 2
    assert len(runner.runs) == 2
    assert not result.aborted


def test_a_locked_point_arms_writes_and_restores_in_that_order():
    """Clock first, then power.

    The clock is the write that can be refused (the floor); the power cap
    cannot. Writing power first means a refused clock leaves a cap already on
    the device. gpu_guard fixed this shape inside itself in pass 12.
    """
    log = []
    sched = build_schedule("poisson", seed=7, duration_s=5.0)
    world = FakeWorld(sched.digest())
    SweepRunner(endpoint="e", model="m", schedule=sched,
                points=[SweepPoint(clock_mhz=900, power_limit_mw=150000)],
                repeats=1, guard_factory=lambda: FakeGuard(log),
                load_runner=world.load_runner, metrics_reader=world.metrics,
                sleep=lambda s: None).run()
    assert log == ["arm", ("lock", 900), ("power", 150000), "restore"]


def test_a_below_floor_clock_is_refused_before_the_guard_is_ever_constructed():
    """Arming publishes a claim; a caller that cannot write must not claim.

    Mutation: move the floor check inside the `with` and the factory runs,
    leaving an operator-visible record for a point that was never allowed.
    """
    built = []
    sched = build_schedule("poisson", seed=7, duration_s=5.0)
    world = FakeWorld(sched.digest())

    def factory():
        built.append(1)
        return FakeGuard([])

    result = SweepRunner(
        endpoint="e", model="m", schedule=sched,
        points=[SweepPoint(clock_mhz=10)], guard_factory=factory,
        load_runner=world.load_runner, metrics_reader=world.metrics,
        sleep=lambda s: None).run()
    assert result.aborted
    assert "below the" in result.abort_reason
    assert "nothing was armed or written" in result.abort_reason
    assert built == [], "the guard must not have been constructed"
    assert world.runs == 0


def test_a_failed_restore_aborts_the_whole_sweep():
    """Continuing past a failed restore measures a card in an unknown state.

    Mutation: catching the guard error per point and continuing lets the second
    point run and leaves `aborted` False.
    """
    from joule_agent.gpu_guard import GpuGuardError

    sched = build_schedule("poisson", seed=7, duration_s=5.0)
    world = FakeWorld(sched.digest())
    boom = GpuGuardError("restore incomplete, GPU MAY STILL BE MODIFIED")
    points = [SweepPoint(clock_mhz=900), SweepPoint(clock_mhz=1200)]
    runner = SweepRunner(
        endpoint="e", model="m", schedule=sched, points=points,
        guard_factory=lambda: FakeGuard([], fail_on_exit=boom),
        load_runner=world.load_runner, metrics_reader=world.metrics,
        sleep=lambda s: None)
    result = runner.run()
    assert result.aborted
    assert result.hardware_may_be_modified
    assert "MAY STILL BE MODIFIED" in result.abort_reason
    assert "gpu_guard status" in result.abort_reason
    assert world.runs == 1, "the second point must not have run"
    assert result.best_point is None, "an aborted sweep must not name a winner"


def test_a_refusal_does_not_claim_the_gpu_may_be_modified():
    """The complement, and the one that keeps the warning believable.

    A refused arm wrote nothing by definition. gpu_guard applies exactly this
    reasoning to its own dead-pid refusal: saying "may still be modified" about
    a case that proves otherwise spends the credibility of the one message that
    has to be believed when it IS said.

    Mutation: catch GpuGuardError alone (dropping the REFUSALS branch) and this
    test sees the modification warning.
    """
    from joule_agent.gpu_guard import GuardBusyError

    sched = build_schedule("poisson", seed=7, duration_s=5.0)
    world = FakeWorld(sched.digest())
    refusal = GuardBusyError("another guard holds this device")
    result = SweepRunner(
        endpoint="e", model="m", schedule=sched,
        points=[SweepPoint(clock_mhz=900)],
        guard_factory=lambda: FakeGuard([], fail_on_enter=refusal),
        load_runner=world.load_runner, metrics_reader=world.metrics,
        sleep=lambda s: None).run()
    assert result.aborted
    assert result.hardware_may_be_modified is False
    assert "Nothing was written" in result.abort_reason
    assert "MAY STILL BE MODIFIED" not in result.abort_reason


def test_an_aborted_sweep_names_no_best_point_even_with_usable_numbers():
    """Mutation: call _pick_best() unconditionally and best_point is set."""
    sched = build_schedule("poisson", seed=7, duration_s=5.0)
    world = FakeWorld(sched.digest())
    boom = GpuGuardError("restore incomplete")
    result = SweepRunner(
        endpoint="e", model="m", schedule=sched,
        points=[SweepPoint(), SweepPoint(clock_mhz=900)], repeats=2,
        guard_factory=lambda: FakeGuard([], fail_on_exit=boom),
        load_runner=world.load_runner, gpu_sampler=FakeSampler(),
        metrics_reader=world.metrics, sleep=lambda s: None).run()
    assert result.aborted
    # The stock point DID produce a usable joules/token figure...
    stock = next(s for s in result.summaries if s.name == "stock")
    assert stock.joules_per_token_mean is not None
    # ...and it still must not be crowned.
    assert result.best_point is None
    assert result.tied_points == []
    assert any("no best point was chosen" in w for w in result.warnings)


def test_a_point_needing_hardware_without_a_guard_is_a_refusal_not_a_silent_stock_run():
    """Otherwise a locked point would quietly measure stock and be labelled 900."""
    sched = build_schedule("poisson", seed=7, duration_s=5.0)
    world = FakeWorld(sched.digest())
    result = SweepRunner(endpoint="e", model="m", schedule=sched,
                         points=[SweepPoint(clock_mhz=900)],
                         guard_factory=None, load_runner=world.load_runner,
                         metrics_reader=world.metrics, sleep=lambda s: None).run()
    assert result.aborted
    assert "no guard was supplied" in result.abort_reason
    assert world.runs == 0


# ---------------------------------------------------------------------------
# Ordering and drift
# ---------------------------------------------------------------------------


def test_repeats_are_round_robin_not_grouped_by_point():
    """Grouping loads thermal drift onto whichever config ran last.

    Mutation: swapping the loop nesting produces stock,stock,clk,clk.
    """
    sched = build_schedule("poisson", seed=7, duration_s=5.0)
    world = FakeWorld(sched.digest())
    runner = SweepRunner(
        endpoint="e", model="m", schedule=sched,
        points=[SweepPoint(), SweepPoint(clock_mhz=900)], repeats=2,
        guard_factory=lambda: FakeGuard([]), load_runner=world.load_runner,
        metrics_reader=world.metrics, sleep=lambda s: None)
    runner.run()
    assert [m.run.point for m in runner.runs] == [
        "stock", "clk900", "stock", "clk900"]


def test_stock_drifting_between_rounds_is_reported_and_warned():
    """Interleaving hides drift; stock-vs-itself is what detects it.

    Energy per run rises 10% between round 0 and round 1 at identical tokens.
    """
    sched = build_schedule("poisson", seed=7, duration_s=5.0)
    world = FakeWorld(sched.digest(), tokens_per_run=1000.0)

    class DriftingSampler(FakeSampler):
        def __init__(self):
            super().__init__()
            self.step = 0

        def poll(self):
            out = {"energy_mj": self.energy_mj, "temperature_c": 40.0,
                   "sm_clock_mhz": 1000}
            # first stock run burns 1000 mJ, second burns 1100
            self.energy_mj += 1000.0 if self.step < 2 else 1100.0
            self.step += 1
            return out

    runner = SweepRunner(
        endpoint="e", model="m", schedule=sched, points=[SweepPoint()],
        repeats=2, load_runner=world.load_runner, gpu_sampler=DriftingSampler(),
        metrics_reader=world.metrics, sleep=lambda s: None)
    result = runner.run()
    assert result.stock_drift_pct == pytest.approx(10.0, abs=0.5)
    assert any("drifted" in w for w in result.warnings)


def test_stock_drifting_downward_is_warned_too():
    """A card getting CHEAPER mid-sweep contaminates it just as much.

    Both drift fixtures were non-negative, so `abs()` in the threshold was
    unpinned: dropping it left the suite green while a -10% drift went
    unwarned.
    """
    sched = build_schedule("poisson", seed=7, duration_s=5.0)
    world = FakeWorld(sched.digest(), tokens_per_run=1000.0)

    class CoolingSampler(FakeSampler):
        def __init__(self):
            super().__init__()
            self.step = 0

        def poll(self):
            out = {"energy_mj": self.energy_mj, "temperature_c": 40.0,
                   "sm_clock_mhz": 1000}
            self.energy_mj += 1000.0 if self.step < 2 else 900.0
            self.step += 1
            return out

    result = SweepRunner(
        endpoint="e", model="m", schedule=sched, points=[SweepPoint()],
        repeats=2, load_runner=world.load_runner, gpu_sampler=CoolingSampler(),
        metrics_reader=world.metrics, sleep=lambda s: None).run()
    assert result.stock_drift_pct == pytest.approx(-10.0, abs=0.5)
    assert any("drifted" in w for w in result.warnings)


def test_the_tie_band_widens_to_the_observed_spread_not_just_the_noise_floor():
    """`band = max(stdev, mean * NOISE_FLOOR)` -- the stdev half was untested.

    Every other test gives identical repeats, so stdev is 0.0 suite-wide and
    `band = mean * NOISE_FLOOR` alone passed everything. The band is taken from
    the BEST point's spread, so the winner is the one that has to be noisy:
    clk900 wins on the mean but swings +/-40%, and stock sits inside that
    spread. A sweep that cannot resolve them must not rank them.

    Mutation: `band = mean * NOISE_FLOOR_FRACTION` and stock drops out of the
    tied set.
    """
    sched = build_schedule("poisson", seed=7, duration_s=5.0)
    world = FakeWorld(sched.digest(), tokens_per_run=1000.0)
    # clk900 burns 400/690/980 mJ across rounds; stock a steady 950.
    energies = {"stock": [950.0] * 3, "clk900": [400.0, 690.0, 980.0]}
    order = ["stock", "clk900"]

    class NoisySampler(FakeSampler):
        def __init__(self):
            super().__init__()
            self.i = 0

        def poll(self):
            run_idx = self.i // 2
            name = order[run_idx % len(order)]
            if self.i % 2 == 1:
                self.energy_mj += energies[name][run_idx // len(order)]
            out = {"energy_mj": self.energy_mj, "temperature_c": 40.0,
                   "sm_clock_mhz": 1000}
            self.i += 1
            return out

    result = SweepRunner(
        endpoint="e", model="m", schedule=sched,
        points=[SweepPoint(), SweepPoint(clock_mhz=900)], repeats=3,
        guard_factory=lambda: FakeGuard([]), load_runner=world.load_runner,
        gpu_sampler=NoisySampler(), metrics_reader=world.metrics,
        sleep=lambda s: None).run()
    best = next(s for s in result.summaries if s.name == "clk900")
    assert best.joules_per_token_stdev > best.joules_per_token_mean * 0.02
    assert result.best_point == "clk900"
    assert "stock" in result.tied_points


def test_stock_drift_within_the_noise_floor_does_not_warn():
    """The complement: the warning must be capable of NOT firing."""
    sched = build_schedule("poisson", seed=7, duration_s=5.0)
    world = FakeWorld(sched.digest())
    runner = SweepRunner(
        endpoint="e", model="m", schedule=sched, points=[SweepPoint()],
        repeats=2, load_runner=world.load_runner, gpu_sampler=FakeSampler(),
        metrics_reader=world.metrics, sleep=lambda s: None)
    result = runner.run()
    assert result.stock_drift_pct == pytest.approx(0.0)
    assert not any("drifted" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# SLO
# ---------------------------------------------------------------------------


def test_slo_threshold_is_derived_from_stock_p99_and_says_so():
    sched = build_schedule("poisson", seed=7, duration_s=5.0)
    world = FakeWorld(sched.digest(), latencies=[0.1] * 9 + [1.0])
    runner = SweepRunner(
        endpoint="e", model="m", schedule=sched, points=[SweepPoint()],
        load_runner=world.load_runner, gpu_sampler=FakeSampler(),
        metrics_reader=world.metrics, sleep=lambda s: None)
    result = runner.run()
    assert result.slo_threshold_s == pytest.approx(1.0 * DEFAULT_SLO_SLACK)
    assert "derived" in result.slo_threshold_source
    assert "stock median p99" in result.slo_threshold_source


def test_an_explicit_slo_overrides_the_derived_one():
    sched = build_schedule("poisson", seed=7, duration_s=5.0)
    world = FakeWorld(sched.digest(), latencies=[0.1] * 10)
    runner = SweepRunner(
        endpoint="e", model="m", schedule=sched, points=[SweepPoint()],
        slo_ms=500.0, load_runner=world.load_runner, gpu_sampler=FakeSampler(),
        metrics_reader=world.metrics, sleep=lambda s: None)
    result = runner.run()
    assert result.slo_threshold_s == pytest.approx(0.5)
    assert "explicit" in result.slo_threshold_source


def test_no_completed_requests_is_not_a_zero_violation_rate():
    """0.0 would read as 'met the SLO perfectly'. It is no measurement."""
    assert slo_violation_rate([], 1.0) is None
    assert slo_violation_rate([0.5, 2.0], 1.0) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Choosing the best point
# ---------------------------------------------------------------------------


def test_points_within_the_noise_floor_are_reported_tied_not_ranked():
    """A 1% gap at a 2% noise floor is not a ranking the sweep can support.

    Mutation: picking a bare argmax leaves tied_points empty.
    """
    sched = build_schedule("poisson", seed=7, duration_s=5.0)

    energies = {"stock": 1000.0, "clk900": 990.0, "clk1200": 2000.0}
    order = ["stock", "clk900", "clk1200"]

    class PerPointSampler(FakeSampler):
        def __init__(self):
            super().__init__()
            self.i = 0

        def poll(self):
            name = order[(self.i // 2) % len(order)]
            # Charge the run's energy on its CLOSING read, so the delta the
            # runner computes (after - before) is that point's consumption.
            if self.i % 2 == 1:
                self.energy_mj += energies[name]
            out = {"energy_mj": self.energy_mj, "temperature_c": 40.0,
                   "sm_clock_mhz": 1000}
            self.i += 1
            return out

    world = FakeWorld(sched.digest(), tokens_per_run=1000.0)
    runner = SweepRunner(
        endpoint="e", model="m", schedule=sched,
        points=[SweepPoint(), SweepPoint(clock_mhz=900), SweepPoint(clock_mhz=1200)],
        guard_factory=lambda: FakeGuard([]), load_runner=world.load_runner,
        gpu_sampler=PerPointSampler(), metrics_reader=world.metrics,
        sleep=lambda s: None)
    result = runner.run()
    assert result.best_point == "clk900"
    assert "stock" in result.tied_points
    assert "clk1200" not in result.tied_points
    assert any("tied" in w for w in result.warnings)


def test_two_points_that_achieved_the_same_clock_are_flagged_as_one():
    """Bin-snapping makes distinct requests one configuration measured twice."""
    sched = build_schedule("poisson", seed=7, duration_s=5.0)
    world = FakeWorld(sched.digest())
    sampler = FakeSampler(clock_by_call=[1000, 1000])
    runner = SweepRunner(
        endpoint="e", model="m", schedule=sched,
        points=[SweepPoint(clock_mhz=900), SweepPoint(clock_mhz=1000)],
        guard_factory=lambda: FakeGuard([]), load_runner=world.load_runner,
        gpu_sampler=sampler, metrics_reader=world.metrics, sleep=lambda s: None)
    result = runner.run()
    assert any("same clock bin" in w for w in result.warnings)


def test_distinct_achieved_clocks_do_not_trigger_the_bin_warning():
    """The complement. `len(names) > 1` -> `>= 1` fires on every sweep, and
    nothing caught it because no test asserted the warning stays silent."""
    sched = build_schedule("poisson", seed=7, duration_s=5.0)
    world = FakeWorld(sched.digest())
    sampler = FakeSampler(clock_by_call=[900, 1200])
    result = SweepRunner(
        endpoint="e", model="m", schedule=sched,
        points=[SweepPoint(clock_mhz=900), SweepPoint(clock_mhz=1200)],
        repeats=1, guard_factory=lambda: FakeGuard([]),
        load_runner=world.load_runner, gpu_sampler=sampler,
        metrics_reader=world.metrics, sleep=lambda s: None).run()
    assert not any("same clock bin" in w for w in result.warnings)


def test_counter_reset_is_false_on_a_healthy_run():
    """`counter_reset` was asserted True-only, so hard-wiring it True passed."""
    sched = build_schedule("poisson", seed=7, duration_s=5.0)
    world = FakeWorld(sched.digest())
    runner = SweepRunner(
        endpoint="e", model="m", schedule=sched, points=[SweepPoint()],
        repeats=1, load_runner=world.load_runner, gpu_sampler=FakeSampler(),
        metrics_reader=world.metrics, sleep=lambda s: None)
    runner.run()
    run = runner.runs[0].run
    assert run.counter_reset is False
    assert run.energy_j is not None
    assert run.generated_tokens == 1000.0


def test_a_clearly_worse_point_is_not_marked_tied():
    """`tied_with_best` was asserted True-only; hard-wiring it True passed."""
    sched = build_schedule("poisson", seed=7, duration_s=5.0)
    world = FakeWorld(sched.digest(), tokens_per_run=1000.0)
    energies = {"stock": 1000.0, "clk900": 3000.0}
    order = ["stock", "clk900"]

    class Spread(FakeSampler):
        def __init__(self):
            super().__init__()
            self.i = 0

        def poll(self):
            name = order[(self.i // 2) % len(order)]
            if self.i % 2 == 1:
                self.energy_mj += energies[name]
            out = {"energy_mj": self.energy_mj, "temperature_c": 40.0,
                   "sm_clock_mhz": 1000}
            self.i += 1
            return out

    result = SweepRunner(
        endpoint="e", model="m", schedule=sched,
        points=[SweepPoint(), SweepPoint(clock_mhz=900)], repeats=2,
        guard_factory=lambda: FakeGuard([]), load_runner=world.load_runner,
        gpu_sampler=Spread(), metrics_reader=world.metrics,
        sleep=lambda s: None).run()
    assert result.best_point == "stock"
    assert result.tied_points == []
    worse = next(s for s in result.summaries if s.name == "clk900")
    assert worse.tied_with_best is False


def test_the_noise_floor_constant_is_what_the_docs_claim():
    """CLAUDE.MD and the module both cite ~2%; the constant decides which
    benchmark differences are called real. It survived being set to 0.08."""
    assert NOISE_FLOOR_FRACTION == 0.02


# ---------------------------------------------------------------------------
# Measurement integrity
# ---------------------------------------------------------------------------


def test_a_backwards_energy_counter_voids_that_run_rather_than_reporting_negatives():
    sched = build_schedule("poisson", seed=7, duration_s=5.0)
    world = FakeWorld(sched.digest())

    class Rewinding(FakeSampler):
        def poll(self):
            out = {"energy_mj": self.energy_mj, "temperature_c": 40.0,
                   "sm_clock_mhz": 1000}
            self.energy_mj -= 500.0
            return out

    runner = SweepRunner(
        endpoint="e", model="m", schedule=sched, points=[SweepPoint()],
        load_runner=world.load_runner, gpu_sampler=Rewinding(),
        metrics_reader=world.metrics, sleep=lambda s: None)
    runner.run()
    run = runner.runs[0].run
    assert run.energy_j is None
    assert run.counter_reset
    assert any("backwards" in w for w in run.warnings)


def test_a_restarted_server_voids_the_token_count():
    """vLLM counters resetting mid-run means the tokens are not this run's."""
    sched = build_schedule("poisson", seed=7, duration_s=5.0)
    world = FakeWorld(sched.digest())
    seq = [snapshot(5000.0), snapshot(10.0)]

    # repeats=1: with the default 3, round 1 pops an empty list and the
    # IndexError is swallowed by _safe_metrics -- the assertions below would
    # still hold, but for a reason the test does not name.
    runner = SweepRunner(
        endpoint="e", model="m", schedule=sched, points=[SweepPoint()],
        repeats=1, load_runner=world.load_runner, gpu_sampler=FakeSampler(),
        metrics_reader=lambda: seq.pop(0), sleep=lambda s: None)
    runner.run()
    run = runner.runs[0].run
    assert run.counter_reset
    assert run.generated_tokens is None
    assert run.joules_per_generated_token is None


def test_a_failed_metrics_scrape_is_recorded_not_swallowed():
    sched = build_schedule("poisson", seed=7, duration_s=5.0)
    world = FakeWorld(sched.digest())

    def boom():
        raise OSError("connection refused")

    runner = SweepRunner(
        endpoint="e", model="m", schedule=sched, points=[SweepPoint()],
        load_runner=world.load_runner, gpu_sampler=FakeSampler(),
        metrics_reader=boom, sleep=lambda s: None)
    runner.run()
    run = runner.runs[0].run
    assert run.generated_tokens is None
    # Not just "some warning mentions scrape" -- `_apply_tokens` emits its own
    # such warning independently, so that assertion passed with the recording
    # path in `_safe_metrics` deleted entirely. Pin the exception text and both
    # bracket positions, which only `_safe_metrics` can produce.
    assert any("connection refused" in w for w in run.warnings)
    assert any("scrape before the run failed" in w for w in run.warnings)
    assert any("scrape after the run failed" in w for w in run.warnings)


def test_the_bracket_skew_is_measured_rather_than_assumed_to_cancel():
    """The two /metrics round trips do not cancel exactly; record the gap."""
    sched = build_schedule("poisson", seed=7, duration_s=5.0)
    world = FakeWorld(sched.digest())
    # metrics_before, energy_before, t0, t_end, energy_after, metrics_after.
    # Leading gap 0.5s, trailing gap 0.9s -> 1.4s of bracket skew.
    ticks = iter([0.0, 0.5, 0.5, 10.0, 10.0, 10.9])

    runner = SweepRunner(
        endpoint="e", model="m", schedule=sched, points=[SweepPoint()],
        repeats=1, load_runner=world.load_runner, gpu_sampler=FakeSampler(),
        metrics_reader=world.metrics, clock=lambda: next(ticks),
        sleep=lambda s: None)
    runner.run()
    assert runner.runs[0].run.bracket_skew_ms == pytest.approx(1400.0)


def test_the_achieved_clock_is_recorded_alongside_the_requested_one():
    sched = build_schedule("poisson", seed=7, duration_s=5.0)
    world = FakeWorld(sched.digest())
    runner = SweepRunner(
        endpoint="e", model="m", schedule=sched,
        points=[SweepPoint(clock_mhz=900)], guard_factory=lambda: FakeGuard([]),
        load_runner=world.load_runner, gpu_sampler=FakeSampler(clock_by_call=[880]),
        metrics_reader=world.metrics, sleep=lambda s: None)
    runner.run()
    run = runner.runs[0].run
    assert run.requested_clock_mhz == 900
    assert run.achieved_clock_median_mhz == 880


# ---------------------------------------------------------------------------
# Grid parsing
# ---------------------------------------------------------------------------


def test_grid_leads_with_stock_so_an_early_death_still_has_a_baseline():
    grid = build_grid([900, 1200])
    assert grid[0].is_stock
    assert [p.name for p in grid] == ["stock", "clk900", "clk1200"]


def _energy_by_point(order, energies):
    """A sampler that charges each point its own energy on the closing read."""
    class S(FakeSampler):
        def __init__(self):
            super().__init__()
            self.i = 0

        def poll(self):
            name = order[(self.i // 2) % len(order)]
            if self.i % 2 == 1:
                self.energy_mj += energies[name]
            out = {"energy_mj": self.energy_mj, "temperature_c": 40.0,
                   "sm_clock_mhz": 1000}
            self.i += 1
            return out
    return S()


def test_a_power_capping_point_is_measured_but_not_trusted():
    """The power axis has no efficacy gate, so its rows are data, not answers.

    Collecting them is deliberate: a null result there cheaply corroborates the
    published "illusion of power capping" finding, and "nothing happened" is a
    weak claim that weak evidence can carry.

    Mutation: drop `axis_verified` from the eligibility filter and the power
    point becomes best_point.
    """
    sched = build_schedule("poisson", seed=7, duration_s=5.0)
    world = FakeWorld(sched.digest(), tokens_per_run=1000.0)
    order = ["stock", "clk900", "pl150W"]
    # the power point is the cheapest by a wide margin
    energies = {"stock": 1000.0, "clk900": 900.0, "pl150W": 400.0}

    result = SweepRunner(
        endpoint="e", model="m", schedule=sched,
        points=[SweepPoint(), SweepPoint(clock_mhz=900),
                SweepPoint(power_limit_mw=150000)],
        repeats=2, guard_factory=lambda: FakeGuard([]),
        load_runner=world.load_runner,
        gpu_sampler=_energy_by_point(order, energies),
        metrics_reader=world.metrics, sleep=lambda s: None).run()

    pl = next(s for s in result.summaries if s.name == "pl150W")
    assert pl.axis_verified is False
    assert pl.joules_per_token_mean is not None, "it must still be MEASURED"
    # ...but it must not be the answer.
    assert result.best_point == "clk900"
    assert result.best_unverified_point == "pl150W"
    assert any("no efficacy gate" in w for w in result.warnings)
    assert any("SURPRISING result on an unverified axis" in w
               for w in result.warnings)


def test_clock_only_points_are_trusted():
    """The complement: `axis_verified` must not be False for everything."""
    sched = build_schedule("poisson", seed=7, duration_s=5.0)
    world = FakeWorld(sched.digest())
    result = SweepRunner(
        endpoint="e", model="m", schedule=sched,
        points=[SweepPoint(), SweepPoint(clock_mhz=900)], repeats=2,
        guard_factory=lambda: FakeGuard([]), load_runner=world.load_runner,
        gpu_sampler=FakeSampler(), metrics_reader=world.metrics,
        sleep=lambda s: None).run()
    assert all(s.axis_verified for s in result.summaries)
    assert result.best_unverified_point is None
    assert not any("no efficacy gate" in w for w in result.warnings)


def test_a_snapped_clock_is_reported_because_the_point_name_is_the_request():
    """This card snaps SM clocks to a 15 MHz grid: ask 1400, get 1410.

    `SweepPoint.name` -- and so `best_point` -- is built from the REQUEST, so
    "best = clk1400" would name a configuration the card never ran.
    """
    sched = build_schedule("poisson", seed=7, duration_s=5.0)
    world = FakeWorld(sched.digest())
    result = SweepRunner(
        endpoint="e", model="m", schedule=sched,
        points=[SweepPoint(clock_mhz=1400)], repeats=1,
        guard_factory=lambda: FakeGuard([]), load_runner=world.load_runner,
        gpu_sampler=FakeSampler(clock_by_call=[1410]),
        metrics_reader=world.metrics, sleep=lambda s: None).run()
    assert any("clk1400 ran at 1410 MHz" in w for w in result.warnings)
    assert any("cite achieved_clock_median_mhz" in w for w in result.warnings)


def test_an_exactly_honoured_clock_raises_no_mislabel_warning():
    """The complement -- otherwise the warning fires on every sweep."""
    sched = build_schedule("poisson", seed=7, duration_s=5.0)
    world = FakeWorld(sched.digest())
    result = SweepRunner(
        endpoint="e", model="m", schedule=sched,
        points=[SweepPoint(clock_mhz=1410)], repeats=1,
        guard_factory=lambda: FakeGuard([]), load_runner=world.load_runner,
        gpu_sampler=FakeSampler(clock_by_call=[1410]),
        metrics_reader=world.metrics, sleep=lambda s: None).run()
    assert not any("ran at" in w for w in result.warnings)


def test_power_limits_without_clocks_still_produce_points():
    """Regression: crossing only inside `for clk in clocks` dropped them all.

    `--power-limits-w 150,175` with no --clocks built a stock-only grid, so the
    operator was pushed through consent, root and the stale pre-flight, waited
    out the whole run, and had not one power limit applied. A silent no-op on
    an explicitly requested hardware change.

    Mutation: `for clk in clocks` and this returns ["stock"] alone.
    """
    grid = build_grid([], [150000, 175000])
    assert [p.name for p in grid] == ["stock", "pl150W", "pl175W"]
    assert sum(1 for p in grid if not p.is_stock) == 2
    # and the stock point is not duplicated by the None x None cell
    assert sum(1 for p in grid if p.is_stock) == 1


def test_a_mid_sequence_device_refusal_reports_the_writes_that_landed():
    """A refusal class does not prove nothing was written.

    `_write` translates NVML NoPermission into `PrivilegeError`, which is a
    REFUSAL -- so a device that accepts the clock lock and then refuses the
    power write would be reported as "Nothing was written to the GPU". False,
    and false in a durable artifact.

    Mutation: infer from the exception class alone and the claim comes back.
    """
    from joule_agent.gpu_guard import PrivilegeError

    sched = build_schedule("poisson", seed=7, duration_s=5.0)
    world = FakeWorld(sched.digest())

    class RefusesPower(FakeGuard):
        def set_power_limit_mw(self, mw):
            raise PrivilegeError("NVML refused the power write")

    result = SweepRunner(
        endpoint="e", model="m", schedule=sched,
        points=[SweepPoint(clock_mhz=900, power_limit_mw=150000)],
        guard_factory=lambda: RefusesPower([]), load_runner=world.load_runner,
        metrics_reader=world.metrics, sleep=lambda s: None).run()
    assert result.aborted
    assert "Nothing was written" not in result.abort_reason
    assert "accepted clock 900 MHz" in result.abort_reason
    assert "restored on exit" in result.abort_reason


def test_a_raw_nvml_error_still_states_what_happened_to_the_hardware():
    """Only NoPermission becomes a GpuGuardError; NotSupported stays raw.

    Without the broad handler that surfaced as "an unexpected error" with no
    statement about the device -- from the one loop that writes GPU state.
    """
    sched = build_schedule("poisson", seed=7, duration_s=5.0)
    world = FakeWorld(sched.digest())

    class Exotic(FakeGuard):
        def set_power_limit_mw(self, mw):
            raise RuntimeError("NVMLError_NotSupported")

    result = SweepRunner(
        endpoint="e", model="m", schedule=sched,
        points=[SweepPoint(clock_mhz=900, power_limit_mw=150000)],
        guard_factory=lambda: Exotic([]), load_runner=world.load_runner,
        metrics_reader=world.metrics, sleep=lambda s: None).run()
    assert result.aborted
    assert "RuntimeError" in result.abort_reason
    assert "accepted clock 900 MHz" in result.abort_reason


def test_the_sampler_is_stopped_even_when_the_load_generator_raises():
    """pthread_sigmask is PER-THREAD.

    A sampler thread left alive through `guard.__exit__` -> restore() can take
    a signal delivery that gpu_guard's signal-blocked sections exist to defer,
    and the handler then runs on the main thread inside the window it was
    protecting.

    Mutation: drop the try/finally and `stopped` stays False.
    """
    sched = build_schedule("poisson", seed=7, duration_s=5.0)

    class Tracking(FakeSampler):
        def __init__(self):
            super().__init__()
            self.stopped = False

        def stop(self):
            self.stopped = True

    sampler = Tracking()

    def exploding_load(schedule):
        raise OSError("endpoint went away mid-run")

    result = SweepRunner(
        endpoint="e", model="m", schedule=sched, points=[SweepPoint()],
        load_runner=exploding_load, gpu_sampler=sampler,
        sleep=lambda s: None).run()
    assert sampler.stopped, "the polling thread outlived the measurement"
    assert result.aborted
    assert "OSError" in result.abort_reason


def test_clock_range_syntax():
    assert _parse_clocks("900:1300:200") == [900, 1100, 1300]
    assert _parse_clocks("900,1200") == [900, 1200]
    assert _parse_clocks("") == []


@pytest.mark.parametrize("bad", ["900:1300", "1300:900:100", "900:1300:0"])
def test_bad_clock_ranges_are_refused(bad):
    with pytest.raises(SweepError):
        _parse_clocks(bad)


def test_the_stale_preflight_calls_check_stale_state_correctly(tmp_path):
    """The regression test for a defect no unit test could see.

    `check_stale_state(state_path, index=gpu)` against a signature of
    `(controller, state_path)` raised an uncaught TypeError, so every locked
    sweep died on a traceback and the guard-contract pre-flight had never once
    run. It was invisible because it lived in `main()`, which nothing imported.

    Mutation: swap the argument order back and this raises TypeError.
    """
    from joule_agent.gpu_guard import FakeGpuController
    from joule_agent.sweep import preflight_stale

    state = tmp_path / "s.json"
    # A clean card with no state file at all must not be refused.
    assert preflight_stale(FakeGpuController(), state) is None


def test_the_stale_preflight_refuses_a_card_owing_a_restore(tmp_path):
    """The complement: `None` must not be the only reachable answer."""
    import json

    from joule_agent.gpu_guard import FakeGpuController
    from joule_agent.sweep import preflight_stale

    c = FakeGpuController()
    snap = c.snapshot()
    state = tmp_path / "s.json"
    state.write_text(json.dumps({"schema": 3, "record": {
        "pid": 999999, "device_index": 0, "device_uuid": snap.uuid,
        "clocks_locked": True, "restored": False,
        "snapshot": snap.to_dict(), "power_limit_written_mw": None,
    }}))
    refusal = preflight_stale(c, state)
    assert refusal is not None
    assert "cannot be assumed to be stock" in refusal


def test_repeats_below_one_is_refused():
    sched = build_schedule("poisson", seed=7, duration_s=5.0)
    with pytest.raises(SweepError):
        SweepRunner(endpoint="e", model="m", schedule=sched,
                    points=[SweepPoint()], repeats=0)
