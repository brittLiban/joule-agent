"""Tests for the clock-lock efficacy verdict.

`judge()` was extracted from `main()` precisely so it could be tested without
root or a GPU. The verdict logic is the part that had a defect capable of
reporting a still-locked card as a success with exit 0, and a defect in a
verification instrument is worse than a defect in the thing it verifies -- it
certifies the absence of problems.
"""

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "vcl", REPO / "tools" / "verify_clock_lock.py")
vcl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vcl)

TOL = 120


def ph(name, median, *, loaded=True, util=99.0):
    return {"phase": name, "samples": 80, "median_mhz": median,
            "min_mhz": median, "max_mhz": median,
            "util_median_pct": util, "loaded": loaded}


def test_a_partial_restore_is_not_a_success():
    """THE regression test.

    The original predicate was `recovered > locked + tolerance`, which asks
    only whether the clock rose. A card left pinned at 1400 MHz after a failed
    restore satisfies that against a 900 MHz lock -- and the tool printed
    `LOCK WORKS`, exit 0, while the GPU was still modified. Restore must be
    judged against BASELINE.

    Mutation: restore the old predicate and this test reports LOCK WORKS.
    """
    v = vcl.judge(
        baseline=ph("baseline", 1920),
        locked=[ph("locked@900", 900)],
        recovered=ph("recovered", 1400),
        targets=[900], tolerance=TOL,
    )
    assert v["result"] == "RESTORE FAILED"
    assert v["recovered_to_baseline"] is False
    assert "still be locked" in v["reason"]


def test_a_clean_run_over_separated_targets_is_the_only_lock_works():
    v = vcl.judge(
        baseline=ph("baseline", 1920),
        locked=[ph("locked@600", 600), ph("locked@1000", 1000),
                ph("locked@1400", 1400)],
        recovered=ph("recovered", 1905),
        targets=[600, 1000, 1400], tolerance=TOL,
    )
    assert v["result"] == "LOCK WORKS"
    assert v["targets_mutually_separated"] is True
    assert v["achieved_monotonic_in_request"] is True


def test_an_unloaded_phase_is_inconclusive_not_a_lock():
    """An idle card sits near its floor whether or not a lock is applied.

    This repo already shipped a tool that read an idle 210 MHz as a lock; a
    load that dies during the LOCKED phase produces exactly that reading.
    """
    v = vcl.judge(
        baseline=ph("baseline", 1920),
        locked=[ph("locked@900", 210, loaded=False, util=2.0)],
        recovered=ph("recovered", 1920),
        targets=[900], tolerance=TOL,
    )
    assert v["result"] == "INCONCLUSIVE"
    assert "not under load" in v["reason"]


def test_a_target_within_tolerance_of_the_boost_clock_cannot_be_tested():
    """The old rule called a working high lock a NO-OP.

    `below_baseline` required achieved < baseline - tolerance, so a perfectly
    functioning 1850 MHz lock on a 1920 MHz card was reported as the device
    ignoring the write -- failing in exactly the upper half of the range the
    sweep wants to use. The honest answer is that the probe cannot discriminate.
    """
    v = vcl.judge(
        baseline=ph("baseline", 1920),
        locked=[ph("locked@1850", 1850)],
        recovered=ph("recovered", 1920),
        targets=[1850], tolerance=TOL,
    )
    assert v["result"] == "INCONCLUSIVE"
    assert "none could be distinguished from no lock" in v["reason"]
    assert v["per_target"][0]["discriminable"] is False


def test_a_high_but_discriminable_target_is_still_judged():
    """The complement: 'inconclusive' must not swallow every high target.

    Asserting only the overall verdict was too weak -- narrowing
    `discriminable` to `tgt < base - 400` silently drops 1600 while 900 alone
    still yields LOCK WORKS. The claim is about the 1600 row, so assert on it.
    """
    v = vcl.judge(
        baseline=ph("baseline", 1920),
        locked=[ph("locked@900", 900), ph("locked@1600", 1600)],
        recovered=ph("recovered", 1920),
        targets=[900, 1600], tolerance=TOL,
    )
    assert v["result"] == "LOCK WORKS"
    high = next(r for r in v["per_target"] if r["target_mhz"] == 1600)
    assert high["discriminable"] is True
    assert high["near_target"] is True
    assert high["changed_from_baseline"] is True
    assert "1600" not in v["reason"], "1600 must not be listed as untestable"


def test_a_non_monotonic_device_is_reported_as_such():
    """`achieved_monotonic_in_request` survived being hard-wired True.

    It is a reported field: a device whose achieved clocks invert against the
    request would make every sweep row's label meaningless in a way the verdict
    is supposed to surface.
    """
    v = vcl.judge(
        baseline=ph("baseline", 1920),
        locked=[ph("locked@600", 1000), ph("locked@1000", 600)],
        recovered=ph("recovered", 1920),
        targets=[600, 1000], tolerance=TOL,
    )
    assert v["achieved_monotonic_in_request"] is False


def test_recovered_to_baseline_is_true_on_a_clean_restore():
    """The field was only ever asserted False; hard-wiring it False passed."""
    v = vcl.judge(
        baseline=ph("baseline", 1920),
        locked=[ph("locked@900", 900)],
        recovered=ph("recovered", 1905),
        targets=[900], tolerance=TOL,
    )
    assert v["recovered_to_baseline"] is True
    assert v["result"] == "LOCK WORKS"


def test_bin_snapping_to_one_clock_is_partial_not_success():
    """Two requests that achieve one clock are one configuration, not two."""
    v = vcl.judge(
        baseline=ph("baseline", 1920),
        locked=[ph("locked@900", 950), ph("locked@1000", 950)],
        recovered=ph("recovered", 1920),
        targets=[900, 1000], tolerance=TOL,
    )
    # Not an or-chain across the two PARTIAL branches: "snapped to bins" is the
    # OFF-TARGET branch's wording, so accepting either let a mutation reroute
    # this scenario through off-target and destroy the separation check while
    # staying green. Pin the separation branch specifically.
    assert v["result"] == "PARTIAL"
    assert v["targets_mutually_separated"] is False
    assert "achieved the same clock" in v["reason"]
    assert "one configuration measured twice" in v["reason"]


def test_an_off_target_achieved_clock_is_partial_and_names_the_gap():
    v = vcl.judge(
        baseline=ph("baseline", 1920),
        locked=[ph("locked@900", 1300)],
        recovered=ph("recovered", 1920),
        targets=[900], tolerance=TOL,
    )
    assert v["result"] == "PARTIAL"
    assert "asked 900 got 1300" in v["reason"]
    assert "mislabelled" in v["reason"]


def test_a_device_that_ignores_every_write_is_a_no_op():
    v = vcl.judge(
        baseline=ph("baseline", 1920),
        locked=[ph("locked@600", 1920), ph("locked@900", 1915)],
        recovered=ph("recovered", 1920),
        targets=[600, 900], tolerance=TOL,
    )
    assert v["result"] == "LOCK IS A NO-OP"
    assert "compare identical configurations" in v["reason"]


def test_restore_is_checked_before_anything_else_is_reported_good():
    """A failed restore outranks a working lock: the card is still modified."""
    v = vcl.judge(
        baseline=ph("baseline", 1920),
        locked=[ph("locked@900", 900)],
        recovered=ph("recovered", 900),
        targets=[900], tolerance=TOL,
    )
    assert v["result"] == "RESTORE FAILED"


def test_every_target_gets_its_own_controller(tmp_path):
    """The regression test for a defect that only appeared on real hardware.

    The first multi-target run died at target 2 with `controller already
    claimed by <GpuGuard ...>`: one controller was shared across targets, and
    `_ClaimMixin.claim` refuses a second owner with nothing ever releasing
    `_claimed_by`. The identical defect had already been found in `sweep.py`
    by a review pass and was then written again here -- which is why the loop
    was extracted from `main()`, where no test could reach it.

    Mutation: hoist the controller out of the loop and target 2 raises.
    """
    from joule_agent.gpu_guard import FakeGpuController, GpuGuard

    built, phases = [], []

    def controller_factory():
        c = FakeGpuController()
        built.append(c)
        return c

    def guard_factory(c):
        return GpuGuard(c, consent=True, require_root=False,
                        state_path=tmp_path / "s.json")

    out = list(vcl.locked_phases_for(
        [600, 1000, 1400],
        lambda t: phases.append(t) or {"phase": f"locked@{t}"},
        controller_factory=controller_factory,
        guard_factory=guard_factory,
    ))
    assert phases == [600, 1000, 1400], "every target must be measured"
    assert len(out) == 3
    assert len(built) == 3, "one controller per target, never shared"
    assert len({id(c) for c in built}) == 3


def test_a_controller_owed_a_restore_is_not_closed(tmp_path):
    """close() shuts NVML down; the atexit restore would then hit a dead
    handle, turning a possible recovery into a certain failure."""
    from joule_agent.gpu_guard import FakeGpuController, GpuGuard

    closed = []

    class Watched(FakeGpuController):
        def close(self):
            closed.append(self)
            super().close()

    def guard_factory(c):
        g = GpuGuard(c, consent=True, require_root=False,
                     state_path=tmp_path / "s.json")
        return g

    list(vcl.locked_phases_for([600], lambda t: {"phase": "x"},
                               controller_factory=Watched,
                               guard_factory=guard_factory))
    assert len(closed) == 1, "a restored guard's controller must be closed"

    closed.clear()

    class NeverRestores:
        _restored = False

        def __init__(self, c):
            self.c = c

        def __enter__(self):
            return self

        def __exit__(self, *e):
            return False

        def lock_clocks(self, mhz, max_mhz=None):
            pass

    list(vcl.locked_phases_for([600], lambda t: {"phase": "x"},
                               controller_factory=Watched,
                               guard_factory=NeverRestores))
    assert closed == [], "a guard still owing a restore must keep NVML open"


def test_an_unknown_dirty_state_is_not_reported_as_clean(monkeypatch):
    """`bool(None)` is False, which reads as a clean tree.

    A provenance field whose failure mode is indistinguishable from a real
    answer is the defect the provenance block exists to prevent.
    """
    monkeypatch.setattr(vcl, "_git", lambda *a: None)
    assert vcl._dirty_flag() is None

    monkeypatch.setattr(vcl, "_git", lambda *a: "")
    assert vcl._dirty_flag() is False

    monkeypatch.setattr(vcl, "_git", lambda *a: " M CLAUDE.MD")
    assert vcl._dirty_flag() is True


def test_the_guard_module_is_hashed_so_a_dirty_tree_still_identifies_the_code(tmp_path):
    """`len(h) == 64` is true of ANY sha256, including one of b"".

    That assertion survived `path.read_bytes(); return sha256(b"").hexdigest()`
    -- the provenance field could hash *nothing* and still look right. This is
    the field whose entire purpose is answering "is this the code that was
    measured?", so it has to be pinned against a known digest and shown to
    distinguish two different files.
    """
    import hashlib

    target = REPO / "joule_agent" / "gpu_guard.py"
    expected = hashlib.sha256(target.read_bytes()).hexdigest()
    assert vcl._sha256(target) == expected
    assert vcl._sha256(target) != hashlib.sha256(b"").hexdigest()

    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"clocks locked")
    b.write_bytes(b"clocks unlocked")
    assert vcl._sha256(a) != vcl._sha256(b), "the digest must track content"

    assert vcl._sha256(REPO / "does-not-exist") is None
