"""Tests for the oracle_idle analysis.

This file exists because a use-before-assignment of `ladder_mode` shipped in a
version that had only ever been exercised on a NON-ladder run. Nothing could
have caught it without a GPU, because the analysis lived inline in `main()`.
Analysis that decides a published claim has to be reachable from a unit test.

No GPU, no root, no serving engine.
"""

import importlib.util
import math
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "oracle_idle", Path(__file__).resolve().parents[1] / "tools" / "oracle_idle.py")
oracle_idle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oracle_idle)

analyse = oracle_idle.analyse
burst_windows = oracle_idle.burst_windows
plan_transitions = oracle_idle.plan_transitions


def leg(label, p99_ms, j):
    return {"label": label, "p99_s": p99_ms / 1000.0, "j_per_gen_token": j,
            "mean_w": 85.0, "transitions": 0}


# The measured run this file was written against, at the artifact's OWN
# precision. Rounding the p99 values to whole milliseconds (as the report prints
# them) moves the interpolated comparator enough to change the answer -- 678.5 ms
# against 678 ms shifts one leg's delta from +0.16% to +0.26%. A fixture built
# from a report rather than from the artifact would encode that error.
LADDER = [
    leg("exact1560", 666.1, 0.150289), leg("oracle", 679.0, 0.145140),
    leg("exact1530", 673.1, 0.147657), leg("oracle", 691.7, 0.146207),
    leg("exact1500", 683.0, 0.144416), leg("oracle", 678.5, 0.146276),
]


# ---------------------------------------------------------------------------
# The defect this file exists for
# ---------------------------------------------------------------------------


def test_ladder_mode_analysis_runs_at_all():
    """Mutation: reference ladder_mode before assigning it (the shipped bug) and
    this raises UnboundLocalError instead of returning."""
    a = analyse(LADDER)
    assert a["ladder_mode"] is True
    assert a["iso"] is not None


def test_non_ladder_mode_still_uses_paired_deltas():
    rows = [leg("exact1560", 664, 0.15002), leg("oracle", 682, 0.14575),
            leg("exact1560", 663, 0.15037), leg("oracle", 678, 0.14663),
            leg("exact1560", 662, 0.15082), leg("oracle", 679, 0.14647)]
    a = analyse(rows)
    assert a["ladder_mode"] is False
    assert a["iso"] is None, "iso-p99 needs a curve; one clock is not a curve"
    assert a["dj"] == pytest.approx(-2.74, abs=0.05)


# ---------------------------------------------------------------------------
# Iso-p99: the comparison the project's question actually asks for
# ---------------------------------------------------------------------------


def test_each_oracle_leg_is_compared_at_its_own_p99():
    """Mutation: average oracle p99 first and one long leg drags the comparison
    to a latency no leg ran at -- which is how a 692 ms leg pushed the mean past
    the curve's edge and produced a spurious 'extrapolation impossible'."""
    iso = analyse(LADDER)["iso"]
    got = {round(x["p99_s"] * 1000): x for x in iso["per_leg"]}
    assert set(got) == {679, 692, 678}
    assert got[679]["delta_pct"] == pytest.approx(-0.40, abs=0.05)
    assert got[678]["delta_pct"] == pytest.approx(+0.26, abs=0.05)
    assert got[692]["out_of_range"] is True, "692 ms is past the 683 ms curve"
    assert iso["n"] == 2, "the out-of-range leg must not be counted"
    assert iso["delta_pct"] == pytest.approx(-0.07, abs=0.02)


def test_a_leg_outside_the_curve_is_never_extrapolated():
    """Extrapolating invents the comparator. Mutation: clamp instead of
    excluding, and an unbracketed leg silently acquires a static reference that
    was never measured."""
    rows = [leg("exact1560", 666, 0.15029), leg("oracle", 900, 0.14000),
            leg("exact1500", 683, 0.14442), leg("oracle", 400, 0.14000)]
    iso = analyse(rows)["iso"]
    assert all(x.get("out_of_range") for x in iso["per_leg"])
    assert iso["n"] == 0
    assert iso["all_out_of_range"] is True
    assert iso["delta_pct"] is None, "a mean over zero comparable legs is not 0"


def test_interpolation_is_linear_between_bracketing_points():
    rows = [leg("exactA", 600, 0.10), leg("oracle", 650, 0.11),
            leg("exactB", 700, 0.20)]
    iso = analyse(rows)["iso"]
    # halfway in p99 -> halfway in energy: 0.15
    assert iso["per_leg"][0]["static_j"] == pytest.approx(0.15)
    assert iso["per_leg"][0]["delta_pct"] == pytest.approx(100 * (0.11 - 0.15) / 0.15)


def test_a_leg_exactly_on_the_curve_edge_is_comparable():
    """683 ms against a curve ending at 683 ms is inside it. Mutation: use a
    strict inequality and the boundary leg is silently discarded."""
    rows = [leg("exact1560", 666, 0.15029), leg("oracle", 683, 0.14500),
            leg("exact1500", 683, 0.14442)]
    iso = analyse(rows)["iso"]
    assert iso["n"] == 1
    assert iso["per_leg"][0]["static_j"] == pytest.approx(0.14442)


# ---------------------------------------------------------------------------
# Guards that must not fire on the ladder itself
# ---------------------------------------------------------------------------


def test_the_cold_pair_guard_does_not_fire_in_ladder_mode():
    """It uses repeated exact legs as their own control, so it is meaningless
    when they are different configurations on purpose. Mutation: drop the
    ladder_mode condition and it excludes pair 1 for being a 1560 leg among
    1530 and 1500 legs -- which is the ladder working as designed."""
    a = analyse(LADDER)
    assert a["excluded"] is None
    assert len(a["iso"]["per_leg"]) == 3


def test_paired_deltas_are_suppressed_in_ladder_mode():
    """Each pair would compare the oracle against a DIFFERENT static clock, so
    their mean is not a quantity. Mutation: return a number and a run reports
    '+0.15%' as though it meant something."""
    assert math.isnan(analyse(LADDER)["dj"])


def test_the_cold_pair_guard_still_fires_when_legs_match():
    rows = [leg("exact1560", 664, 0.14885), leg("oracle", 680, 0.14509),
            leg("exact1560", 665, 0.14433), leg("oracle", 679, 0.14413),
            leg("exact1560", 664, 0.14435), leg("oracle", 678, 0.14354)]
    a = analyse(rows)
    assert a["excluded"] == pytest.approx(3.1, abs=0.2)
    assert len(a["deltas"]) == 2, "the cold pair must be dropped, not averaged"


# ---------------------------------------------------------------------------
# The oracle plan is derived from the schedule, which is what makes it foresight
# ---------------------------------------------------------------------------


class FakeReq:
    def __init__(self, t):
        self.arrival_s = t


class FakeSched:
    def __init__(self, ts):
        self.requests = [FakeReq(t) for t in ts]


def test_bursts_split_on_schedule_quiet_not_on_engine_state():
    w = burst_windows(FakeSched([0.1, 0.5, 0.9, 4.0, 4.3, 8.0]), quiet_s=1.0)
    assert w == [(0.1, 0.9), (4.0, 4.3), (8.0, 8.0)]


def test_the_plan_upclocks_before_the_burst_which_is_the_foresight():
    """Mutation: drop the lead and the oracle loses the one advantage no causal
    controller can have -- acting on traffic that has not arrived."""
    plan = plan_transitions([(5.0, 8.0)], low=405, high=1560,
                            lead=0.3, drain=0.5, horizon=60)
    assert plan[0] == (4.7, 1560), "did not upclock before the burst"
    assert plan[1] == (8.5, 405), "did not wait out the drain"


def test_the_plan_never_emits_a_redundant_write():
    plan = plan_transitions([(1.0, 2.0), (2.2, 3.0)], low=405, high=1560,
                            lead=0.5, drain=0.5, horizon=60)
    clocks = [c for _, c in plan]
    assert all(a != b for a, b in zip(clocks, clocks[1:])), plan


def test_the_plan_is_bounded_by_the_horizon():
    plan = plan_transitions([(1.0, 2.0), (100.0, 101.0)], low=405, high=1560,
                            lead=0.3, drain=0.5, horizon=60)
    assert all(t <= 60 for t, _ in plan)
