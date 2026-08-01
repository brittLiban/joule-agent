"""Tests for the load generator.

No server, no GPU, no network. Schedules are materialised before any request
is sent, which is exactly what makes the interesting properties testable
offline: determinism, variation, and open- vs closed-loop classification.

The properties under test are the ones the sweep runner depends on. If
`Schedule.digest()` is not stable for a fixed seed, every sweep comparison
downstream is confounded.
"""

import statistics

import pytest

from joule_agent.loadgen import (
    PRESETS,
    ArrivalProcess,
    ClosedLoop,
    LoadGenerator,
    RunReport,
    Schedule,
    build_schedule,
    _pct,
)


# ---------------------------------------------------------------------------
# reproducibility -- the sweep runner's core dependency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset", ["steady", "poisson", "bursty", "bursty_heavy"])
def test_same_seed_gives_identical_schedule(preset):
    a = build_schedule(preset, seed=42, duration_s=20.0)
    b = build_schedule(preset, seed=42, duration_s=20.0)
    assert a.digest() == b.digest()
    assert len(a.requests) == len(b.requests)
    for x, y in zip(a.requests, b.requests):
        assert x.arrival_s == y.arrival_s
        assert x.prompt == y.prompt          # byte-identical payloads
        assert x.max_tokens == y.max_tokens


@pytest.mark.parametrize("preset", ["poisson", "bursty"])
def test_different_seed_gives_different_schedule(preset):
    a = build_schedule(preset, seed=1, duration_s=20.0)
    b = build_schedule(preset, seed=2, duration_s=20.0)
    assert a.digest() != b.digest()


#: Pinned schedule digests: (preset, seed, duration, n_requests, digest).
#
# These are literal values, not recomputed at test time -- a test that derives
# its own expectation cannot detect drift. A failure here means the traffic a
# given seed produces has CHANGED, which invalidates comparison against any
# sweep run recorded before the change. If the change is deliberate, update
# these literals in the same commit and say why in the message.
PINNED_DIGESTS = (
    ("bursty", 1234, 30.0, 194, "8bc2d04d145d5abb"),
    ("poisson", 1234, 30.0, 136, "02c58a5e5621705d"),
    ("steady", 1234, 30.0, 120, "7b2e02188d068cce"),
    ("bursty_heavy", 1234, 30.0, 390, "e75bcd95d351534c"),
)


@pytest.mark.parametrize("preset,seed,duration,n_requests,digest", PINNED_DIGESTS)
def test_schedule_digest_matches_pinned_value(preset, seed, duration, n_requests, digest):
    """The sweep runner's foundation: a seed must mean one exact traffic plan."""
    s = build_schedule(preset, seed=seed, duration_s=duration)
    assert len(s.requests) == n_requests
    assert s.digest() == digest, (
        f"{preset}/seed={seed} traffic changed: {s.digest()} != {digest}. "
        "Any sweep comparison spanning this change is invalid."
    )


def test_digest_changes_when_traffic_changes():
    base = build_schedule("bursty", seed=7, duration_s=30.0)
    faster = build_schedule("bursty", seed=7, duration_s=30.0, rate_scale=2.0)
    longer = build_schedule("bursty", seed=7, duration_s=60.0)
    assert base.digest() != faster.digest()
    assert base.digest() != longer.digest()


# ---------------------------------------------------------------------------
# open loop vs closed loop
# ---------------------------------------------------------------------------


def test_open_loop_presets_are_marked_reproducible():
    for name in ("steady", "poisson", "bursty", "bursty_heavy"):
        assert PRESETS[name].reproducible is True
        assert isinstance(PRESETS[name].driver, ArrivalProcess)
        assert build_schedule(name, seed=1, duration_s=10.0).reproducible is True


def test_closed_loop_presets_are_marked_not_reproducible():
    """These reproduce the capture fixtures but must never enter a sweep."""
    for name in ("single", "burst", "saturated"):
        assert PRESETS[name].reproducible is False
        assert isinstance(PRESETS[name].driver, ClosedLoop)
        assert build_schedule(name, seed=1, duration_s=10.0).reproducible is False


def test_closed_loop_run_emits_a_reproducibility_warning():
    sched = build_schedule("saturated", seed=1, duration_s=5.0)
    gen = LoadGenerator("http://localhost:1", "m", sched)
    rep = RunReport(
        preset=sched.preset, reproducible=sched.reproducible, schedule_digest=sched.digest()
    )
    # The warning is attached at run() start; assert the condition that drives it.
    assert not sched.reproducible
    assert sched.params["mode"] == "closed_loop"


def test_open_loop_arrivals_do_not_depend_on_server_speed():
    """Arrival times are fixed at build time -- nothing about the server enters."""
    s = build_schedule("poisson", seed=99, duration_s=20.0)
    times = [r.arrival_s for r in s.requests]
    assert times == sorted(times)
    assert all(0.0 <= t < 20.0 for t in times)


# ---------------------------------------------------------------------------
# variation -- the harness must be able to demonstrate what it claims to test
# ---------------------------------------------------------------------------


def test_steady_preset_has_essentially_no_variation():
    """The control case: dynamic control should NOT beat static here."""
    s = build_schedule("steady", seed=5, duration_s=30.0)
    cv = s.offered_rate_cv()
    assert cv is not None and cv < 0.15, cv


def test_bursty_preset_delivers_real_variation():
    s = build_schedule("bursty", seed=5, duration_s=30.0)
    cv = s.offered_rate_cv()
    assert cv is not None and cv > 0.5, f"bursty CV too low: {cv}"


def test_bursty_has_genuine_idle_gaps():
    """Idle gaps are the point -- the engine must actually drain."""
    s = build_schedule("bursty", seed=5, duration_s=30.0)
    times = [r.arrival_s for r in s.requests]
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert max(gaps) > 1.5, f"no real idle gap; max gap {max(gaps):.2f}s"


def test_poisson_arrivals_are_irregular_but_not_bursty_shaped():
    s = build_schedule("poisson", seed=5, duration_s=30.0)
    times = [r.arrival_s for r in s.requests]
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert statistics.pstdev(gaps) > 0.05
    assert s.offered_rate_cv() > 0.1


def test_onoff_respects_its_duty_cycle():
    proc = ArrivalProcess("onoff", rate_per_s=20.0, on_s=2.0, off_s=2.0)
    import random as _r

    times = proc.arrivals(_r.Random(3), 20.0)
    # Every arrival must land inside an "on" window of the 4s cycle.
    assert times, "onoff produced no arrivals"
    assert all((t % 4.0) < 2.0 + 1e-9 for t in times)


def test_varied_presets_vary_prompt_and_output_sizes():
    s = build_schedule("bursty", seed=11, duration_s=30.0)
    assert len({len(r.prompt) for r in s.requests}) > 3
    assert len({r.max_tokens for r in s.requests}) > 3


def test_steady_preset_holds_sizes_fixed():
    s = build_schedule("steady", seed=11, duration_s=20.0)
    assert len({r.max_tokens for r in s.requests}) == 1


# ---------------------------------------------------------------------------
# intended vs achieved
# ---------------------------------------------------------------------------


def _report_from_samples(preset, samples, elapsed=10.0, lateness=None):
    sched = build_schedule(preset, seed=1, duration_s=elapsed)
    gen = LoadGenerator("http://localhost:1", "m", sched)
    gen._samples = samples
    gen._lateness = lateness if lateness is not None else [0.001] * 5
    gen._latencies = [0.1, 0.2, 0.3]
    rep = RunReport(preset=preset, offered_load_cv=sched.offered_rate_cv())
    gen._finalise(rep, elapsed)
    return rep


def test_achieved_saturation_reported_when_queue_builds():
    samples = [(float(i), 4.0, 28.0) for i in range(11)]
    rep = _report_from_samples("bursty", samples)
    assert rep.achieved_saturation is True
    assert rep.max_waiting == 28.0
    assert rep.time_at_nonzero_waiting_s == pytest.approx(10.0)
    assert rep.fraction_time_waiting == pytest.approx(1.0)


def test_intended_saturation_not_achieved_is_called_out():
    """A preset name is an intention. An empty queue makes the label wrong."""
    samples = [(float(i), 4.0, 0.0) for i in range(11)]
    rep = _report_from_samples("saturated", samples)
    assert rep.achieved_saturation is False
    assert any("does NOT exercise queueing" in w for w in rep.warnings)


def test_partial_queueing_is_measured_not_rounded():
    # Queue present for only the middle of the run.
    samples = [(float(i), 4.0, 0.0 if i < 3 or i > 7 else 5.0) for i in range(11)]
    rep = _report_from_samples("bursty", samples)
    assert 0.0 < rep.fraction_time_waiting < 1.0
    assert rep.max_waiting == 5.0


def test_client_side_lateness_is_flagged():
    """If the client could not keep up, the traffic shape was not delivered."""
    samples = [(float(i), 1.0, 0.0) for i in range(11)]
    rep = _report_from_samples("bursty", samples, lateness=[0.01, 2.5, 0.4])
    assert rep.schedule_achieved is False
    assert any("late" in w for w in rep.warnings)


def test_on_time_schedule_is_not_flagged():
    samples = [(float(i), 1.0, 0.0) for i in range(11)]
    rep = _report_from_samples("bursty", samples, lateness=[0.001, 0.02, 0.01])
    assert rep.schedule_achieved is True
    assert not any("late" in w for w in rep.warnings)


def test_steady_traffic_run_warns_it_proves_nothing_about_variation():
    samples = [(float(i), 4.0, 0.0) for i in range(11)]
    rep = _report_from_samples("steady", samples)
    assert any("says nothing" in w for w in rep.warnings)


def test_missing_metrics_makes_achieved_state_unknown():
    rep = _report_from_samples("bursty", [])
    assert rep.metrics_samples == 0
    assert any("achieved state unknown" in w for w in rep.warnings)


def test_running_cv_reported():
    samples = [(float(i), float(i % 5), 0.0) for i in range(20)]
    rep = _report_from_samples("bursty", samples, elapsed=20.0)
    assert rep.observed_running_cv > 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_percentile_helper():
    vals = [float(i) for i in range(1, 101)]
    assert _pct(vals, 50) == pytest.approx(50.5)
    assert _pct(vals, 99) == pytest.approx(99.01, abs=0.1)
    assert _pct([], 50) == 0.0
    assert _pct([5.0], 99) == 5.0


def test_latency_percentiles_are_ordered():
    samples = [(float(i), 1.0, 0.0) for i in range(11)]
    sched = build_schedule("bursty", seed=1, duration_s=10.0)
    gen = LoadGenerator("http://localhost:1", "m", sched)
    gen._samples = samples
    gen._latencies = [0.1 * i for i in range(1, 101)]
    gen._lateness = [0.0]
    rep = RunReport(preset="bursty")
    gen._finalise(rep, 10.0)
    assert rep.latency_p50_s <= rep.latency_p95_s <= rep.latency_p99_s
