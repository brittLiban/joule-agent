"""Tests for the `joule benchmark` entry point.

Component 6 is the surface a user actually reads, so the bar here is that the
report cannot state more than was measured. Two properties matter more than the
formatting: a joules-per-token figure must never appear without the workload
that produced it, and a refusal must fire before any work is done rather than
after ten minutes of traffic.

No GPU, no root, no serving engine.
"""

import json

import pytest

from joule_agent.benchmark import (
    build_arg_parser,
    calibrate_slo,
    format_benchmark_report,
    main,
)
from joule_agent.sweep import SweepError, SweepResult


class FakeSummary:
    def __init__(self, name, j, tok_s, p99, mhz, runs=3, sd=0.001,
                 viol=0.0, tied=False):
        self.name = name
        self.runs = runs
        self.joules_per_token_mean = j
        self.joules_per_token_stdev = sd
        self.tokens_per_s_mean = tok_s
        self.latency_p99_mean_s = p99
        self.slo_violation_rate_mean = viol
        self.achieved_clock_median_mhz = mhz
        self.meets_slo = True
        self.axis_verified = True
        self.tied_with_best = tied


def result(summaries, best, **kw):
    r = SweepResult()
    r.preset = kw.get("preset", "bursty")
    r.seed = 1234
    r.schedule_digest = "cdc07a267dc0015c"
    r.repeats = 3
    r.summaries = summaries
    r.best_point = best
    r.slo_threshold_s = kw.get("slo_threshold_s", 0.6644)
    r.slo_threshold_source = kw.get("source", "explicit --slo-ms 664.4")
    r.stock_drift_pct = kw.get("drift")
    r.warnings = kw.get("warnings", [])
    r.tied_points = kw.get("tied_points", [])
    return r


REAL = [FakeSummary("stock", 0.214631, 595.3, 0.6043, 1920),
        FakeSummary("clk1590", 0.151030, 592.7, 0.6593, 1590)]


# ---------------------------------------------------------------------------
# What the report may claim
# ---------------------------------------------------------------------------


def test_the_report_states_what_this_run_measured_not_what_the_tool_does():
    """The honesty rule, enforced on the surface a user screenshots.

    Mutation: reword to "Joule cuts energy by X%" and this fails. The repo's
    non-negotiable rule is that no efficiency percentage appears that was not
    measured in that run.
    """
    out = format_benchmark_report(result(REAL, "clk1590"))
    assert "THIS RUN MEASURED" in out
    assert "29.6% lower energy per token" in out
    low = out.lower()
    for banned in ("joule cuts", "joule saves", "joule reduces", "up to "):
        assert banned not in low, f"product claim in report: {banned!r}"


def test_every_energy_figure_carries_its_workload():
    """A J/token number without its workload invites a comparison it cannot support.

    On the reference hardware load shape moved J/total-token ~5x, far more than
    the clock knob. Mutation: drop the workload line and this fails.
    """
    out = format_benchmark_report(result(REAL, "clk1590"))
    assert "workload" in out
    assert "bursty" in out
    assert "cdc07a267dc0" in out, "schedule digest is not shown"
    assert "do not transfer" in out


def test_the_report_names_the_driver_because_it_is_a_per_driver_property():
    out = format_benchmark_report(result(REAL, "clk1590"), driver="595.71.05")
    assert "595.71.05" in out


def test_a_grid_where_stock_wins_says_so_instead_of_inventing_a_saving():
    """Mutation: compute a reduction unconditionally and this reports -0.0%.

    A tool that always prints a percentage will print a meaningless one.
    """
    only_stock = [FakeSummary("stock", 0.214631, 595.3, 0.6043, 1920)]
    out = format_benchmark_report(result(only_stock, "stock"))
    assert "no clock in this grid beat stock" in out
    assert "THIS RUN MEASURED" not in out


def test_warnings_and_abort_reach_the_report():
    r = result(REAL, "clk1590", warnings=["the SLO threshold was derived from 3 "
                                          "stock run(s)"])
    r.aborted = True
    r.abort_reason = "a restore did not complete"
    r.hardware_may_be_modified = True
    out = format_benchmark_report(r)
    assert "derived from 3 stock run" in out
    assert "ABORTED" in out
    assert "MAY STILL BE MODIFIED" in out


def test_tied_points_are_disclosed_rather_than_silently_ranked():
    """Picking one of several statistically indistinguishable points and
    presenting it alone overstates the resolution of the measurement."""
    r = result(REAL, "clk1590", tied_points=["clk1605", "clk1620"])
    out = format_benchmark_report(r)
    assert "tied within noise" in out
    assert "clk1605" in out


# ---------------------------------------------------------------------------
# Refusals -- before any work, not after
# ---------------------------------------------------------------------------


def test_a_closed_loop_preset_is_refused(capsys):
    rc = main(["--model", "m", "--preset", "saturated"])
    assert rc == 2
    assert "closed-loop" in capsys.readouterr().err


def test_an_unknown_preset_is_refused(capsys):
    rc = main(["--model", "m", "--preset", "nope"])
    assert rc == 2
    assert "unknown preset" in capsys.readouterr().err


def test_missing_consent_is_refused_before_calibration_runs(capsys, monkeypatch):
    """Calibration is ten minutes of real traffic.

    Mutation: move the consent check back below calibration and this fails,
    because the boom sentinel runs first. This is gpu_guard's rule -- refusals
    that depend only on the arguments happen before any work.
    """
    def boom(*a, **k):
        raise AssertionError("calibration ran before the consent refusal")

    monkeypatch.setattr("joule_agent.benchmark.calibrate_slo", boom)
    rc = main(["--model", "m", "--preset", "bursty"])
    assert rc == 2
    assert "REFUSED" in capsys.readouterr().err


def test_missing_root_is_refused_before_calibration_runs(capsys, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("calibration ran before the privilege refusal")

    monkeypatch.setattr("joule_agent.benchmark.calibrate_slo", boom)
    monkeypatch.setattr("joule_agent.benchmark._has_root", lambda: False)
    rc = main(["--model", "m", "--preset", "bursty",
               "--i-understand-this-changes-gpu-settings"])
    assert rc == 2
    assert "requires root" in capsys.readouterr().err


def test_calibrate_only_needs_neither_consent_nor_root(tmp_path, monkeypatch):
    """The unprivileged half must stay unprivileged.

    An operator has to be able to measure the baseline and inspect the budget
    before deciding to let anything touch the GPU.
    """
    monkeypatch.setattr("joule_agent.benchmark._has_root", lambda: False)
    monkeypatch.setattr(
        "joule_agent.benchmark.calibrate_slo",
        lambda **k: (662.9, {"stock_runs": 10, "stock_p99_median_s": 0.6027,
                             "stock_p99_stdev_s": 0.0022, "slo_slack": 1.1,
                             "result": None}))
    rc = main(["--model", "m", "--preset", "bursty", "--calibrate-only",
               "--out", str(tmp_path)])
    assert rc == 0
    doc = json.loads((tmp_path / "calibration.json").read_text())
    assert doc["slo_ms"] == pytest.approx(662.9)
    assert doc["calibration"]["stock_runs"] == 10
    assert "result" not in doc["calibration"], "unserialisable object leaked"


# ---------------------------------------------------------------------------
# Calibrate-and-freeze
# ---------------------------------------------------------------------------


def test_calibration_derives_from_the_median_and_reports_its_own_spread():
    class FakeRun:
        def __init__(self, p99):
            self.point = "stock"
            self.latency_p99_s = p99

    class FakeResult:
        aborted = False
        abort_reason = None
        runs = [FakeRun(p) for p in (0.60, 0.61, 0.62)]

    class FakeRunner:
        def __init__(self, **kw):
            self.kw = kw

        def run(self):
            return FakeResult()

    ms, prov = calibrate_slo(
        endpoint="e", model="m", schedule=None, repeats=3, slo_slack=1.1,
        gpu=0, settle_s=0, warmup_runs=0, runner_factory=FakeRunner)
    assert ms == pytest.approx(0.61 * 1.1 * 1000)
    assert prov["stock_runs"] == 3
    assert prov["stock_p99_median_s"] == pytest.approx(0.61)
    assert prov["stock_p99_stdev_s"] is not None


def test_an_aborted_calibration_does_not_yield_a_threshold():
    """Mutation: return a threshold anyway and every later verdict rests on a
    budget derived from a run that failed."""
    class FakeResult:
        aborted = True
        abort_reason = "a restore did not complete"
        runs = []

    class FakeRunner:
        def __init__(self, **kw):
            pass

        def run(self):
            return FakeResult()

    with pytest.raises(SweepError) as exc:
        calibrate_slo(endpoint="e", model="m", schedule=None, repeats=3,
                      slo_slack=1.1, gpu=0, settle_s=0, warmup_runs=0,
                      runner_factory=FakeRunner)
    assert "aborted" in str(exc.value)


def test_the_default_calibration_depth_is_the_calibrated_minimum():
    """Mutation: default to 3 and the shipped tool hands users the exact
    under-sampling defect that flipped clk1575/clk1590 between sweeps."""
    from joule_agent.sweep import MIN_STOCK_RUNS_FOR_SLO
    args = build_arg_parser().parse_args(["--model", "m"])
    assert args.calibrate_repeats == MIN_STOCK_RUNS_FOR_SLO


def test_the_power_axis_is_not_exposed():
    """sweep.py keeps --power-limits-w as a research flag. The benchmark must
    not offer it: every power-axis claim this project made was retracted and
    the efficacy gate that would justify a new one was never built."""
    flags = build_arg_parser().format_help()
    assert "--power-limits-w" not in flags
    assert "power-limit" not in flags
