"""Tests for the power-cap efficacy verdict and its hardware-writing loop.

`judge_power()` is extracted from `main()` for the same reason `judge()` was:
the verdict logic is the part that can certify the absence of a problem. A
defect here is worse than a defect in the thing being verified.

The property this suite exists to pin above all others is `can_bind`. The
retracted power claim in CLAUDE.MD was not produced by a coding error -- every
number in it was correctly measured. It was produced by counting caps that were
*physically incapable of acting* as null results. A suite that pins only "a
working cap is reported as working" reproduces the exact blindness that made the
retraction necessary.

The second thing pinned here is the set of defects a `hardware-safety-reviewer`
pass found in this file's FIRST draft, before it ever ran on hardware: a
`CAP WORKS` verdict whose reason string asserted "each cap" while the code
checked "any cap"; one margin constant serving two predicates that pull in
opposite directions; a restore check that silently skipped when an NVML read
returned `None`; and a per-guard-controller loop with no test holding it in
place. Each has a test below named for it.
"""

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "vpc", REPO / "tools" / "verify_power_cap.py")
vpc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vpc)

CLOCK = 1800
DEFAULT_MW = 200_000


def ph(name, median_mhz, *, peak_w, mean_w=None, limit_mw=DEFAULT_MW,
       loaded=True, util=99.0, source="ring"):
    """A measured phase. Defaults describe a healthy, loaded, uncapped card."""
    return {
        "phase": name,
        "samples": 80,
        "median_mhz": median_mhz,
        "min_mhz": median_mhz,
        "max_mhz": median_mhz,
        "util_median_pct": util,
        "loaded": loaded,
        "power_samples": 400,
        "ring_samples": 400 if source == "ring" else 0,
        "polled_samples": 0 if source == "ring" else 400,
        "power_source": source,
        "power_mean_w": mean_w if mean_w is not None else peak_w - 10.0,
        "power_p95_w": peak_w - 2.0,
        "power_peak_w": peak_w,
        "power_limit_readback_mw": limit_mw,
    }


def judge(baseline, capped, recovered, caps_w, **kw):
    kw.setdefault("requested_clock_mhz", CLOCK)
    return vpc.judge_power(baseline, capped, recovered, caps_w, **kw)


# --------------------------------------------------------------------------
# The predicate the retraction was made of.
# --------------------------------------------------------------------------

def test_caps_above_the_uncapped_peak_are_inconclusive_not_null():
    """THE regression test for the retracted claim.

    Reproduces the sweep's real situation: the card drew a 107 W peak and the
    caps swept were 120 W and 160 W. Neither could act. The conclusion drawn was
    "capping does nothing" -- which the data cannot support, because a cap that
    cannot bind changing nothing is a tautology.

    INCONCLUSIVE, never CAP IS A NO-OP. Those two strings are the whole
    difference between "we could not test this" and "we tested this and it does
    not work".

    Mutation: make `can_bind` unconditionally True -> CAP IS A NO-OP, i.e. it
    asserts the retracted claim.
    """
    base = ph("baseline", CLOCK, peak_w=107.0)
    capped = [ph("cap@160W", CLOCK, peak_w=107.0, limit_mw=160_000),
              ph("cap@120W", CLOCK, peak_w=107.0, limit_mw=120_000)]
    v = judge(base, capped, ph("recovered", CLOCK, peak_w=107.0), [160, 120])

    assert v["result"] == "INCONCLUSIVE"
    assert "tautology" in v["reason"]
    assert all(r["can_bind"] is False for r in v["per_cap"])


def test_a_cap_that_can_bind_and_does_nothing_is_a_no_op():
    """The other side of the same coin -- and proof the test above is not
    passing merely because the function never says NO-OP.
    """
    base = ph("baseline", CLOCK, peak_w=150.0)
    capped = [ph("cap@100W", CLOCK, peak_w=150.0, limit_mw=100_000)]
    v = judge(base, capped, ph("recovered", CLOCK, peak_w=150.0), [100])

    assert v["result"] == "CAP IS A NO-OP"
    assert v["per_cap"][0]["can_bind"] is True


def test_binding_is_judged_against_peak_not_mean():
    """The correction to the retraction's own reasoning.

    A 120 W cap dragged clk1800 from 1800 to 1605 MHz against a 106.5 W *mean*.
    Caps act on instantaneous draw, so a cap between the mean and the peak is
    capable of binding.

    Mutation: compare `can_bind` against `power_mean_w` -> the cap is excluded
    and this returns INCONCLUSIVE.
    """
    base = ph("baseline", CLOCK, peak_w=140.0, mean_w=106.5)
    capped = [ph("cap@120W", 1605, peak_w=118.0, mean_w=95.0, limit_mw=120_000)]
    v = judge(base, capped, ph("recovered", CLOCK, peak_w=140.0, mean_w=106.5),
              [120])

    assert v["per_cap"][0]["can_bind"] is True
    assert v["result"] == "PARTIAL"
    assert "n=1" in v["reason"]


# --------------------------------------------------------------------------
# Findings from the first draft's safety review.
# --------------------------------------------------------------------------

def test_a_mixed_ladder_is_partial_and_the_reason_never_says_each():
    """Review finding B1. The first draft gated on ANY bindable cap responding,
    then printed a reason asserting every cap "each ... reduced the achieved
    clock and peak draw" -- a durable artifact stating something unmeasured.

    Here 130 W and 115 W do nothing while 100 W acts.

    Mutation: restore `if not responded` (any-semantics) -> CAP WORKS.
    """
    base = ph("baseline", CLOCK, peak_w=150.0)
    capped = [ph("cap@130W", CLOCK, peak_w=150.0, limit_mw=130_000),
              ph("cap@115W", CLOCK, peak_w=150.0, limit_mw=115_000),
              ph("cap@100W", 1400, peak_w=99.0, limit_mw=100_000)]
    v = judge(base, capped, ph("recovered", CLOCK, peak_w=150.0),
              [130, 115, 100])

    assert v["result"] == "PARTIAL"
    assert "130" in v["reason"] and "115" in v["reason"]
    assert "each" not in v["reason"]


def test_the_enforce_margin_does_not_widen_when_the_bind_margin_does():
    """Review finding B2: one constant cannot serve two predicates that pull in
    opposite directions. `can_bind` wants a LARGE margin (be conservative about
    what is testable); `peak_below_cap` wants a SMALL one (or a device drawing
    above its cap reads as obedient).

    The card here draws 112 W peak under a 100 W cap -- it did NOT honour it --
    and drops no clock. With the margins separate, raising bind_margin_w to 20
    cannot turn that into enforcement.

    Mutation: pass `enforce_margin_w=bind_margin_w` (the first draft's shared
    constant) -> `peak_below_cap` becomes True and the verdict flips.
    """
    base = ph("baseline", CLOCK, peak_w=150.0)
    capped = [ph("cap@100W", CLOCK, peak_w=112.0, limit_mw=100_000)]
    rec = ph("recovered", CLOCK, peak_w=150.0)

    v = judge(base, capped, rec, [100], bind_margin_w=20.0)
    assert v["per_cap"][0]["can_bind"] is True
    assert v["per_cap"][0]["peak_below_cap"] is False
    assert v["result"] == "CAP IS A NO-OP"

    # The shared-constant behaviour, shown directly to be different.
    v2 = judge(base, capped, rec, [100], bind_margin_w=20.0, enforce_margin_w=20.0)
    assert v2["per_cap"][0]["peak_below_cap"] is True
    assert v2["result"] != "CAP IS A NO-OP"


def test_an_unreadable_power_limit_is_unverified_not_passed():
    """Review finding B3. The first draft's restore check was

        if default_limit_w is not None and rec_limit_w is not None and ...

    so either read returning None silently disabled the strongest check in the
    tool, and a still-capped card could reach CAP WORKS with a reason string
    asserting "the device returned to its default limit".

    Unverifiable is not verified -- gpu_guard's own doctrine for an unreadable
    UUID.

    Mutation: restore the `is not None` guards -> CAP WORKS.
    """
    base = ph("baseline", CLOCK, peak_w=150.0)
    capped = [ph("cap@130W", 1700, peak_w=128.0, limit_mw=130_000),
              ph("cap@100W", 1450, peak_w=99.0, limit_mw=100_000)]
    rec = ph("recovered", CLOCK, peak_w=150.0, limit_mw=None)

    v = judge(base, capped, rec, [130, 100])

    assert v["result"] == "RESTORE UNVERIFIED"
    assert "UNKNOWN" in v["reason"]


def test_restore_is_measured_against_the_baseline_limit_not_the_device_default():
    """Review finding H2. `gpu_guard.restore()` restores the limit in force at
    ARM time, not the device default. An operator running with their own
    deliberate 150 W cap would be told RESTORE FAILED -- and pointed at a
    command that would change their setting -- by a guard that behaved
    perfectly.
    """
    base = ph("baseline", CLOCK, peak_w=140.0, limit_mw=150_000)
    capped = [ph("cap@130W", 1700, peak_w=128.0, limit_mw=130_000),
              ph("cap@100W", 1450, peak_w=99.0, limit_mw=100_000)]
    rec = ph("recovered", CLOCK, peak_w=140.0, limit_mw=150_000)

    v = judge(base, capped, rec, [130, 100])

    assert v["result"] != "RESTORE FAILED"
    assert v["baseline_power_limit_w"] == 150.0


def test_a_still_capped_card_is_never_reported_as_working():
    """The clock gate once printed LOCK WORKS with the card still locked.

    Every cap behaved perfectly here, so an efficacy-first ordering returns
    CAP WORKS and exit 0 while the GPU is still capped.
    """
    base = ph("baseline", CLOCK, peak_w=150.0)
    capped = [ph("cap@130W", 1700, peak_w=128.0, limit_mw=130_000),
              ph("cap@100W", 1450, peak_w=99.0, limit_mw=100_000)]
    rec = ph("recovered", CLOCK, peak_w=150.0, limit_mw=100_000)

    v = judge(base, capped, rec, [130, 100])

    assert v["result"] == "RESTORE FAILED"
    assert "still capped" in v["reason"]
    assert "gpu_guard restore" in v["reason"]


def test_the_power_restore_check_outranks_the_liveness_gate():
    """Review finding H1. The readback is a device fact that does not need
    load. Behind the liveness gate, a loader dying during the recovered phase
    reported "not under load" about a card still holding a cap -- with no
    mention of restore.

    Mutation: move the liveness gate above the power-restore check -> this
    returns INCONCLUSIVE and never mentions the cap left on the card.
    """
    base = ph("baseline", CLOCK, peak_w=150.0)
    capped = [ph("cap@100W", 1450, peak_w=99.0, limit_mw=100_000)]
    rec = ph("recovered", 210, peak_w=25.0, limit_mw=100_000,
             loaded=False, util=2.0)

    v = judge(base, capped, rec, [100])

    assert v["result"] == "RESTORE FAILED"
    assert "gpu_guard restore" in v["reason"]


def test_a_baseline_that_missed_its_clock_lock_cannot_attribute_anything():
    """Review finding H3. Every clock comparison below assumes the lock landed.
    An unlocked card drifts ~60 MHz run to run -- more than the 45 MHz margin --
    so `clock_dropped` would fire on drift alone.
    """
    base = ph("baseline", 1920, peak_w=150.0)   # boost clock: lock never landed
    capped = [ph("cap@100W", 1450, peak_w=99.0, limit_mw=100_000)]
    rec = ph("recovered", 1920, peak_w=150.0)

    v = judge(base, capped, rec, [100], requested_clock_mhz=1800)

    assert v["result"] == "INCONCLUSIVE"
    assert "control variable" in v["reason"]


def test_a_polled_power_peak_cannot_carry_an_enforcement_claim():
    """Review finding H4. The first draft's fallback comment claimed the polled
    source "is recorded"; nothing recorded it. A peak from a ~500 ms poll is
    smoothed low, which inflates `peak_below_cap` -- manufacturing evidence that
    a device obeyed its cap.
    """
    base = ph("baseline", CLOCK, peak_w=150.0)
    capped = [ph("cap@100W", 1450, peak_w=99.0, limit_mw=100_000, source="polled")]
    rec = ph("recovered", CLOCK, peak_w=150.0)

    v = judge(base, capped, rec, [100])

    assert v["result"] == "INCONCLUSIVE"
    assert "polled" in v["reason"]


# --------------------------------------------------------------------------
# Stored vs enforced, and the ladder requirement.
# --------------------------------------------------------------------------

def test_a_cap_the_driver_never_stored_is_refused_not_measured():
    """Readback is the one check the power axis has and the clock axis cannot.
    A cap that does not read back was never applied.
    """
    base = ph("baseline", CLOCK, peak_w=150.0)
    capped = [ph("cap@100W", 1450, peak_w=99.0, limit_mw=DEFAULT_MW)]
    v = judge(base, capped, ph("recovered", CLOCK, peak_w=150.0), [100])

    assert v["result"] == "REFUSED BY DEVICE"
    assert v["per_cap"][0]["stored"] is False


def test_a_working_ladder_closes_the_axis():
    """The passing case: several binding caps, each storing, each acting,
    monotone in the cap, clean restore.
    """
    base = ph("baseline", CLOCK, peak_w=150.0)
    capped = [ph("cap@130W", 1700, peak_w=128.0, limit_mw=130_000),
              ph("cap@115W", 1600, peak_w=113.0, limit_mw=115_000),
              ph("cap@100W", 1450, peak_w=99.0, limit_mw=100_000)]
    v = judge(base, capped, ph("recovered", CLOCK, peak_w=150.0),
              [130, 115, 100])

    assert v["result"] == "CAP WORKS"
    assert v["response_monotonic_in_cap"] is True
    assert v["bindable_caps_w"] == [130, 115, 100]


def test_a_non_monotone_response_is_partial_not_a_win():
    """A tighter cap producing a HIGHER clock means the rows are not ordered by
    the thing they are labelled with.
    """
    base = ph("baseline", CLOCK, peak_w=150.0)
    capped = [ph("cap@130W", 1450, peak_w=99.0, limit_mw=130_000),
              ph("cap@100W", 1700, peak_w=128.0, limit_mw=100_000)]
    v = judge(base, capped, ph("recovered", CLOCK, peak_w=150.0), [130, 100])

    assert v["result"] == "PARTIAL"
    assert v["response_monotonic_in_cap"] is False


def test_non_binding_caps_are_named_alongside_a_win_not_silently_dropped():
    """Silently dropping excluded caps is how a report comes to imply four caps
    were tested when two were.
    """
    base = ph("baseline", CLOCK, peak_w=150.0)
    capped = [ph("cap@160W", CLOCK, peak_w=150.0, limit_mw=160_000),
              ph("cap@130W", 1700, peak_w=128.0, limit_mw=130_000),
              ph("cap@100W", 1450, peak_w=99.0, limit_mw=100_000)]
    v = judge(base, capped, ph("recovered", CLOCK, peak_w=150.0),
              [160, 130, 100])

    assert v["result"] == "CAP WORKS"
    assert "[160]" in v["reason"]
    assert v["bindable_caps_w"] == [130, 100]


# --------------------------------------------------------------------------
# Liveness and scope.
# --------------------------------------------------------------------------

def test_an_unloaded_phase_cannot_judge_anything():
    """An idle card draws idle power under any cap. Reading that as a working
    cap is the 210 MHz idle-clock defect, one axis over.
    """
    base = ph("baseline", CLOCK, peak_w=150.0)
    capped = [ph("cap@100W", 210, peak_w=25.0, limit_mw=100_000,
                 loaded=False, util=2.0)]
    v = judge(base, capped, ph("recovered", CLOCK, peak_w=150.0), [100])

    assert v["result"] == "INCONCLUSIVE"
    assert "not under load" in v["reason"]


def test_phase_marks_a_sample_less_run_unloaded(monkeypatch):
    """Review finding O5: the first draft's version of this test hand-built a
    dict with `loaded=False` and never called `phase()` -- so it duplicated the
    test above and certified nothing about `phase()`.

    This one drives the real function with a live loader and an empty sample
    return, and pins that `phase()` -- not the test -- sets `loaded=False` and a
    `None` peak. That matters: a `loaded=True` phase with a `None` peak reaches
    the `f"{base_peak:.1f}"` format in `judge_power` and raises TypeError.

    Mutation: delete `phase()`'s empty-`pmw` branch -> TypeError here.
    """
    monkeypatch.setattr(vpc, "sample_device", lambda *a, **k: {
        "clocks": [1800] * 10, "utils": [99] * 10, "power_mw": [],
        "ring_samples": 0, "polled_samples": 0, "load_died": False})
    monkeypatch.setattr(vpc, "read_power_limit_mw", lambda: DEFAULT_MW)

    class LiveProc:
        returncode = None

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    monkeypatch.setattr(vpc.subprocess, "Popen", lambda *a, **k: LiveProc())
    monkeypatch.setattr(vpc.time, "sleep", lambda *_: None)

    out = vpc.phase("cap@100W", 0.01, "python")

    assert out["loaded"] is False
    assert out["power_peak_w"] is None
    # And the verdict survives it rather than crashing on a None format.
    v = judge(ph("baseline", CLOCK, peak_w=150.0), [out],
              ph("recovered", CLOCK, peak_w=150.0), [100])
    assert v["result"] == "INCONCLUSIVE"


def test_a_win_carries_the_synthetic_load_caveat():
    """Efficacy under an FMA kernel is not efficacy under decode."""
    base = ph("baseline", CLOCK, peak_w=150.0)
    capped = [ph("cap@130W", 1700, peak_w=128.0, limit_mw=130_000),
              ph("cap@100W", 1450, peak_w=99.0, limit_mw=100_000)]
    v = judge(base, capped, ph("recovered", CLOCK, peak_w=150.0), [130, 100])

    assert v["result"] == "CAP WORKS"
    assert "does NOT license" in v["scope_note"]
    assert "inference" in v["scope_note"]


# --------------------------------------------------------------------------
# The hardware-writing loop. Review finding H6.
# --------------------------------------------------------------------------

class FakeController:
    """Minimal stand-in; only identity and close() matter here.

    Identity is a monotonic serial, NOT ``id(self)``. CPython recycles object
    ids once a controller is garbage collected, so an id-keyed claim set made
    controller 3 look like controller 1 and reported a sharing violation that
    had not happened -- a false alarm in the test guarding the defect that
    killed `verify_clock_lock` on hardware. A test that cries wolf about the
    one property it exists to pin is worse than no test.
    """

    _next = 0

    def __init__(self, log):
        FakeController._next += 1
        self.serial = FakeController._next
        self.log = log
        self.closed = False
        log.append(("construct", self.serial))

    def close(self):
        self.closed = True
        self.log.append(("close", self.serial))


class FakeGuard:
    """Mimics the one property that killed verify_clock_lock on hardware:
    a controller may be claimed by exactly one guard, and nothing releases it.
    """

    claimed: set = set()

    def __init__(self, controller, log):
        self.c = controller
        self.log = log
        self._restored = True

    def __enter__(self):
        if self.c.serial in FakeGuard.claimed:
            raise vpc.GpuGuardError(
                f"controller already claimed by <GpuGuard {self.c.serial}>")
        FakeGuard.claimed.add(self.c.serial)
        return self

    def __exit__(self, *exc):
        return False

    def lock_clocks(self, mhz):
        self.log.append(("lock", mhz))

    def set_power_limit_mw(self, mw):
        self.log.append(("cap", mw))


@pytest.fixture(autouse=True)
def _clear_claims():
    FakeGuard.claimed = set()
    yield
    FakeGuard.claimed = set()


def test_every_cap_phase_gets_its_own_controller():
    """Review finding H6. `_ClaimMixin.claim` refuses a second owner and nothing
    releases `_claimed_by`, so a controller hoisted out of the loop fails at
    phase 2 -- which is exactly how `verify_clock_lock.py` died on real
    hardware, after a review pass had already found the identical defect in
    `sweep.py`. Three occurrences in two files made it a rule; this tool ships a
    4-cap default, so the fourth would have manifested on hardware too.

    Mutation: hoist `controller_factory()` out of the loop in
    `capped_phases_for` -> GpuGuardError("already claimed") at cap 2.
    """
    log = []
    phases = list(vpc.capped_phases_for(
        1800, [130_000, 115_000, 100_000],
        lambda mw: {"phase": f"cap@{mw // 1000}W"},
        controller_factory=lambda: FakeController(log),
        guard_factory=lambda c: FakeGuard(c, log)))

    assert [p["phase"] for p in phases] == ["cap@130W", "cap@115W", "cap@100W"]
    constructed = [e for e in log if e[0] == "construct"]
    assert len({e[1] for e in constructed}) == 3, "controllers must not be shared"


def test_the_clock_lock_is_applied_before_the_cap():
    """`sweep.py` orders the writes this way, and the order is load-bearing:
    `set_power_limit_mw` refuses an out-of-range value AFTER arm(), so the two
    writes must not be able to interleave differently between tools.
    """
    log = []
    list(vpc.capped_phases_for(
        1800, [100_000], lambda mw: {},
        controller_factory=lambda: FakeController(log),
        guard_factory=lambda c: FakeGuard(c, log)))

    ops = [e[0] for e in log]
    assert ops.index("lock") < ops.index("cap")


def test_a_controller_owed_a_restore_is_not_closed():
    """close() shuts NVML down; the atexit last-chance restore would then run
    against a dead handle, turning a possible recovery into a certain failure.
    """
    log = []
    controllers = []

    def factory():
        c = FakeController(log)
        controllers.append(c)
        return c

    def guard_factory(c):
        g = FakeGuard(c, log)
        g._restored = False           # wrote, and the restore did not complete
        return g

    list(vpc.capped_phases_for(1800, [100_000], lambda mw: {},
                               controller_factory=factory,
                               guard_factory=guard_factory))

    assert controllers[0].closed is False


def test_the_baseline_phase_shares_the_capped_phase_write_path():
    """Review finding O1. The baseline was a hand-copied hardware-writing block
    inside `main()`, where no test could reach it -- the rule broken by two
    blocking `sweep.py` defects and the `verify_clock_lock` hardware failure.

    It is now `baseline_phase_for`, and it must apply the clock lock and NO cap.
    """
    log = []
    out = vpc.baseline_phase_for(
        1800, lambda: {"phase": "baseline"},
        controller_factory=lambda: FakeController(log),
        guard_factory=lambda c: FakeGuard(c, log))

    assert out["phase"] == "baseline"
    assert ("lock", 1800) in log
    assert not any(e[0] == "cap" for e in log), "the baseline must be uncapped"


# --------------------------------------------------------------------------
# rejudge. Review finding H5.
# --------------------------------------------------------------------------

def test_rejudge_matches_phases_by_name_and_refuses_an_incomplete_run(tmp_path):
    """Review finding H5. Positional indexing (`phases[-1]` as `recovered`) read
    the last CAPPED phase of a bailed run as the recovered phase, fabricating a
    RESTORE FAILED diagnosis about a run that never had one.
    """
    doc = {"caps_w": [130, 100], "clock_mhz": CLOCK,
           "phases": [ph("baseline", CLOCK, peak_w=150.0),
                      ph("cap@130W", 1700, peak_w=128.0, limit_mw=130_000)]}
    p = tmp_path / "bailed.json"
    p.write_text(json.dumps(doc))

    with pytest.raises(SystemExit) as e:
        vpc.rejudge(p, 45, 5.0, 2.0)
    assert "not a complete run" in str(e.value)


def test_rejudge_records_the_predicates_it_actually_used(tmp_path):
    """The copied provenance block still names the ORIGINAL run's margins, so a
    re-judged artifact that does not record the new ones misstates the
    predicates behind its own verdict.
    """
    doc = {"caps_w": [130, 100], "clock_mhz": CLOCK,
           "verdict": {"result": "INCONCLUSIVE"},
           "provenance": {"args": {"bind_margin_w": 5.0}},
           "phases": [ph("baseline", CLOCK, peak_w=150.0),
                      ph("cap@130W", 1700, peak_w=128.0, limit_mw=130_000),
                      ph("cap@100W", 1450, peak_w=99.0, limit_mw=100_000),
                      ph("recovered", CLOCK, peak_w=150.0)]}
    p = tmp_path / "run.json"
    p.write_text(json.dumps(doc))

    out = vpc.rejudge(p, 45, 7.5, 1.5)

    assert out["verdict"]["result"] == "CAP WORKS"
    assert out["rejudged_from"]["original_verdict"] == "INCONCLUSIVE"
    assert out["rejudged_from"]["predicates"] == {
        "clock_margin_mhz": 45, "bind_margin_w": 7.5, "enforce_margin_w": 1.5}


# --------------------------------------------------------------------------
# Exit codes. Review finding O3.
# --------------------------------------------------------------------------

def test_a_failed_restore_gets_its_own_exit_code():
    """The first draft collapsed "the GPU may still be capped" into the same
    exit 2 as "you typed a bad argument". The worst outcome must not be
    reported in the words -- or the code -- of the mildest.
    """
    assert vpc._exit_code("CAP WORKS") == 0
    assert vpc._exit_code("CAP IS A NO-OP") == 1
    assert vpc._exit_code("INCONCLUSIVE") == 1
    assert vpc._exit_code("NOT RUN") == 2
    assert vpc._exit_code("RESTORE FAILED") == 3
    assert vpc._exit_code("RESTORE UNVERIFIED") == 3


# --------------------------------------------------------------------------
# The fail-closed branch the contract check structurally cannot reach.
# --------------------------------------------------------------------------

def test_an_unreadable_power_range_refuses_before_arming(monkeypatch, tmp_path):
    """Review finding B4, and the one part of it no subprocess check can cover.

    The first draft gated the cap-range pre-check on
    `if lo is not None and hi is not None`, where both came from an NVML read
    wrapped in a bare `except`. When that read failed the ENTIRE argument
    refusal evaporated silently: the run proceeded, `lock_clocks` landed on the
    device, `set_power_limit_mw` then raised the range refusal `gpu_guard`
    performs after arm(), and the durable artifact recorded `NOT RUN` for a run
    that had already written hardware.

    `tools/verify_power_cap_contract_check.py` cannot test this: it drives a
    real GPU whose range read always succeeds, so the mutation is a no-op there
    (verified -- 32/32 still pass with the branch removed). The failure has to
    be injected, which is what this test does.

    Mutation: change the guard back to `if False:` -> this reaches the phase
    ladder instead of refusing, and the assertion on "could not read" fails.
    """
    real = vpc.provenance

    def blind(args, python):
        p = real(args, python)
        p["power_range_read_ok"] = False
        p["power_min_w"] = p["power_max_w"] = None
        return p

    monkeypatch.setattr(vpc, "provenance", blind)
    # If the refusal fails to fire, this makes the failure loud rather than
    # letting the test hang on a real 8-second GPU phase.
    monkeypatch.setattr(vpc, "baseline_phase_for", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("reached the hardware path with an unvalidated cap range")))

    art = tmp_path / "out.json"
    rc = vpc.main(["--caps-w", "130", "--clock-mhz", "1800",
                   "--out", str(art), "--state-path", str(tmp_path / "s.json")])

    assert rc == vpc.EXIT_REFUSED
    doc = json.loads(art.read_text())
    assert doc["verdict"]["result"] == "NOT RUN"
    assert "could not read" in doc["verdict"]["reason"]
    assert not (tmp_path / "s.json").exists(), "a refused caller must claim nothing"
