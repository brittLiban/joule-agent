"""Behavioural efficacy gate for the POWER-LIMIT axis, the twin of verify_clock_lock.py.

The clock axis has a gate; the power axis has never had one. `sweep.py`
iterates power limits anyway, labels those rows `axis_verified: false`, and
excludes them from the best-point choice -- a holding pattern, not an answer.
This tool is the answer.

**What went wrong without it, stated first because it is the whole design
input.** The 84-run static sweep was read as independently reproducing the
published "illusion of power capping" finding: caps changed almost nothing, so
capping must be ineffective during decode. That claim was RETRACTED. Mean power
never exceeded 107 W at any locked clock, the card's minimum settable limit is
100 W, and the swept caps were 120 W and 160 W -- so seven of nine clock rows
tested a cap that was *physically incapable of acting*. A non-binding cap
changing nothing is a tautology, not evidence.

**So the load-bearing predicate here is `can_bind`, not `changed`.** A cap above
the observed power draw is excluded from the verdict rather than counted as a
null result. That is the same shape as `discriminable` in verify_clock_lock.py,
where a target within the baseline margin of the boost clock is INCONCLUSIVE --
a limit of the probe, not a verdict about the device.

**Bind against PEAK, never mean.** The retraction's own reasoning was still
half-wrong: it argued from mean power. The sweep data refute that directly --
`clk1800` drew a 106.5 W *mean* and a 120 W cap dragged it from 1800 MHz to
1605 MHz. A cap acts on instantaneous power inside the driver's averaging
window, so mean understates binding badly. Power is therefore sampled from the
driver's own ~20 ms ring (`nvmlDeviceGetSamples`) rather than by polling
`nvmlDeviceGetPowerUsage`, which refreshes only ~500 ms on this card and would
smooth away exactly the transients a cap responds to.

**THREE margins, not one -- the mistake this repo has already paid for once.**
A single `tolerance` in verify_clock_lock.py served four predicates whose
directional requirements conflicted, and the hole it opened let a card left
pinned at 1800 MHz print LOCK WORKS. The same trap is live here:
``bind_margin_w`` (is a cap capable of acting?) wants to be LARGE to be
conservative, while ``enforce_margin_w`` (did the device hold the cap?) wants to
be SMALL or a device exceeding its cap by 5 W reads as obedient. One constant
cannot be both, so there are two, plus ``clock_margin_mhz`` for the clock.

**Stored is not enforced, and here that distinction is measurable.** Unlike the
clock lock -- which this driver exposes no readback for -- a power limit *can*
be read back. That readback proves the driver STORED the cap; it does not prove
the device ENFORCES it. `stored` comes from the readback, `enforced` from the
clock and power response under load, and the two are never conflated.

**Unverifiable is not verified.** Every NVML read here can fail. Where a failed
read would disable a safety check, the verdict is RESTORE UNVERIFIED or
INCONCLUSIVE -- never a silent skip that lets `CAP WORKS` be printed over a
check that never ran. That rule is `gpu_guard`'s own (a UUID that cannot be read
refuses rather than falling through to the index), and the first draft of this
file broke it in three places.

The load is the synthetic FMA kernel, as in verify_clock_lock.py. That makes
this a gate on the DEVICE, not a measurement of inference: FMA draws harder than
decode, so it is the favourable condition for demonstrating that a cap can act.
Efficacy here does not license a power-axis claim about inference traffic.

Run:
    sudo .venv/bin/python tools/verify_power_cap.py \
        --clock-mhz 1800 --caps-w 160,130,115,100
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from joule_agent.gpu_guard import (  # noqa: E402
    MIN_SM_CLOCK_MHZ,
    GpuGuard,
    GpuGuardError,
    NvmlGpuController,
)

LOAD_SCRIPT = REPO / "tools" / "synthetic_gpu_load.py"

#: A phase with utilization below this was not meaningfully loaded. An idle
#: card draws idle power and sits at an idle clock whether or not a cap is
#: applied, so such a phase cannot distinguish them.
MIN_LOADED_UTIL_PCT = 50.0

#: How far the achieved clock must fall below the uncapped baseline before the
#: drop is attributed to the cap rather than to sampling spread. The baseline
#: here is CLOCK-LOCKED, which the clock gate measured at zero variance, so this
#: guards against sampling noise only -- NOT against the ~60 MHz run-to-run
#: drift of an *unlocked* card, which is why an unlocked baseline is refused.
DEFAULT_CLOCK_MARGIN_MHZ = 45

#: Watts of headroom a cap needs BELOW the uncapped peak before it counts as
#: capable of acting. Wants to be LARGE: being conservative here means refusing
#: to draw a conclusion from a cap that barely clips anything, which is the
#: error that had to be retracted.
DEFAULT_BIND_MARGIN_W = 5.0

#: Watts by which an observed peak may EXCEED its cap and still count as the
#: device having honoured it. Wants to be SMALL: this is evidence of
#: enforcement, and a generous value manufactures it. Deliberately a separate
#: constant from the one above -- they pull in opposite directions.
DEFAULT_ENFORCE_MARGIN_W = 2.0


def load_argv(seconds: float, python: str) -> list:
    """The exact command each phase runs to drive the GPU.

    One definition, used by both :func:`phase` and :func:`provenance`, so the
    recorded load is the load that ran rather than a hand-copied description of
    it that can drift.
    """
    return [python, str(LOAD_SCRIPT),
            "--duration", str(seconds + 3),
            "--period", str(seconds + 3),
            "--duty", "1.0"]


def _git(*args) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                           text=True, timeout=10)
    except Exception:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def _dirty_flag():
    """``True``/``False``/``None`` -- unknown is not clean."""
    porcelain = _git("status", "--porcelain")
    return None if porcelain is None else bool(porcelain)


def _sha256(path: Path) -> str | None:
    """Content hash of the module under test."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _pct(sorted_vals: list, q: float) -> float:
    """Nearest-rank percentile on an already-sorted list."""
    if not sorted_vals:
        raise ValueError("empty")
    k = max(0, min(len(sorted_vals) - 1,
                   int(round(q / 100.0 * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


def _controller(index: int = 0):
    """Construct the real controller with its failures translated.

    Unwrapped construction is how a mistyped ``--gpu`` reached committed code as
    a raw ``NVMLError`` traceback in ``gpu_guard``, and again in ``sweep.py``'s
    stale pre-flight one review pass later. A traceback here would also escape
    ``main()``'s handler and leave the PREVIOUS run's verdict on disk as the
    newest evidence.
    """
    try:
        return NvmlGpuController(index=index)
    except GpuGuardError:
        raise
    except Exception as exc:                      # noqa: BLE001
        raise GpuGuardError(
            f"cannot open GPU {index}: {exc.__class__.__name__}: {exc}") from exc


def sample_device(seconds: float, interval: float = 0.1, proc=None) -> dict:
    """Sample clock, utilization and the driver's own power ring.

    Power comes from ``nvmlDeviceGetSamples`` with a ``lastSeenTimeStamp``
    cursor -- the same mechanism ``telemetry.py`` uses and for the same reason.
    ``nvmlDeviceGetPowerUsage`` refreshes only ~500 ms on this card, so polling
    it faster repeats values and fakes a resolution the sensor does not have.

    The polled fallback is COUNTED, not silently mixed in. This gate turns on
    peak power, and a peak taken from a 500 ms poll is smoothed low -- which
    would inflate the evidence that a device obeyed its cap. A verdict must be
    able to refuse a peak it cannot attribute.
    """
    import pynvml

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    clocks, utils, power_mw = [], [], []
    ring_n = polled_n = 0
    load_died = False
    cursor = 0
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < seconds:
            if proc is not None and proc.poll() is not None:
                load_died = True
                break
            try:
                clocks.append(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
            except Exception:
                pass
            try:
                utils.append(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
            except Exception:
                pass
            try:
                _vt, samples = pynvml.nvmlDeviceGetSamples(
                    h, pynvml.NVML_TOTAL_POWER_SAMPLES, cursor)
                for s in samples:
                    if s.timeStamp > cursor:
                        cursor = s.timeStamp
                    power_mw.append(s.sampleValue.uiVal)
                    ring_n += 1
            except Exception:
                try:
                    power_mw.append(pynvml.nvmlDeviceGetPowerUsage(h))
                    polled_n += 1
                except Exception:
                    pass
            time.sleep(interval)
    finally:
        pynvml.nvmlShutdown()
    return {"clocks": clocks, "utils": utils, "power_mw": power_mw,
            "ring_samples": ring_n, "polled_samples": polled_n,
            "load_died": load_died}


def read_power_limit_mw() -> int | None:
    """Device-side readback of the enforced limit.

    This is the axis's one advantage over the clock axis: the driver will tell
    you what limit it holds. It answers STORED, never ENFORCED. ``None`` means
    *unknown*, and every caller must treat unknown as unverified rather than as
    agreement.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            return int(pynvml.nvmlDeviceGetPowerManagementLimit(h))
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None


def phase(name: str, seconds: float, python: str) -> dict:
    """Run sustained load and sample the device throughout."""
    try:
        proc = subprocess.Popen(
            load_argv(seconds, python),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        return {"phase": name, "samples": 0, "loaded": False,
                "power_peak_w": None, "power_mean_w": None,
                "power_limit_readback_mw": read_power_limit_mw(),
                "error": f"could not start the load ({exc}); this phase "
                         "measured nothing"}
    try:
        time.sleep(2.0)  # let the load ramp before sampling
        if proc.poll() is not None:
            return {"phase": name, "samples": 0, "loaded": False,
                    "power_peak_w": None, "power_mean_w": None,
                    "power_limit_readback_mw": read_power_limit_mw(),
                    "error": f"load process exited during ramp (rc={proc.returncode}); "
                             "this phase measured nothing"}
        s = sample_device(seconds, proc=proc)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    clocks, utils, pmw = s["clocks"], s["utils"], s["power_mw"]
    # The readback is taken for EVERY phase, including failed ones: it is a
    # device fact that does not depend on the load, and it is the only evidence
    # that can prove a cap outlived its guard.
    readback = read_power_limit_mw()
    if not clocks:
        return {"phase": name, "samples": 0, "loaded": False,
                "power_peak_w": None, "power_mean_w": None,
                "power_limit_readback_mw": readback,
                "error": "no clock samples collected"}
    util_median = statistics.median(utils) if utils else None
    loaded = (not s["load_died"]
              and util_median is not None
              and util_median >= MIN_LOADED_UTIL_PCT)
    out = {
        "phase": name,
        "samples": len(clocks),
        "min_mhz": min(clocks),
        "max_mhz": max(clocks),
        "median_mhz": statistics.median(clocks),
        "util_median_pct": util_median,
        "power_samples": len(pmw),
        "ring_samples": s["ring_samples"],
        "polled_samples": s["polled_samples"],
        "power_source": ("ring" if s["polled_samples"] == 0 and s["ring_samples"]
                         else "polled" if s["ring_samples"] == 0 and s["polled_samples"]
                         else "mixed" if pmw else "none"),
        "loaded": loaded,
        "power_limit_readback_mw": readback,
    }
    if pmw:
        sp = sorted(pmw)
        out["power_mean_w"] = round(statistics.mean(pmw) / 1000.0, 2)
        out["power_p95_w"] = round(_pct(sp, 95) / 1000.0, 2)
        out["power_peak_w"] = round(sp[-1] / 1000.0, 2)
    else:
        out["power_mean_w"] = out["power_p95_w"] = out["power_peak_w"] = None
        out["loaded"] = False
        out["error"] = ("no power samples collected; this gate turns on power "
                        "and cannot judge a phase without it")
        return out
    if not loaded:
        out["error"] = (
            f"GPU was not under load (median utilization {util_median}%, "
            f"load process died: {s['load_died']}). An unloaded card draws idle "
            "power whether or not a cap is applied, so this phase cannot "
            "distinguish them."
        )
    return out


def provenance(args, python: str) -> dict:
    """Stamp what produced this result: when, which code, which card."""
    prov = {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": _dirty_flag(),
        "gpu_guard_sha256": _sha256(REPO / "joule_agent" / "gpu_guard.py"),
        "python": python,
        "args": {"clock_mhz": args.clock_mhz, "caps_w": args.caps_w,
                 "seconds": args.seconds,
                 "clock_margin_mhz": args.clock_margin_mhz,
                 "bind_margin_w": args.bind_margin_w,
                 "enforce_margin_w": args.enforce_margin_w},
        "load_argv": load_argv(args.seconds, python),
        "driver_version": None,
        "nvml_version": None,
        "device": None,
        "power_min_w": None,
        "power_max_w": None,
        "power_default_w": None,
        "power_range_read_ok": False,
    }
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            for key, fn in (("driver_version", pynvml.nvmlSystemGetDriverVersion),
                            ("nvml_version", pynvml.nvmlSystemGetNVMLVersion)):
                try:
                    v = fn()
                    prov[key] = v.decode() if isinstance(v, bytes) else str(v)
                except Exception:
                    pass
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            try:
                u = pynvml.nvmlDeviceGetUUID(h)
                prov["device"] = u.decode() if isinstance(u, bytes) else str(u)
            except Exception:
                pass
            # The settable range is the whole reason this tool exists in this
            # shape: caps outside it are refusals, and caps above the observed
            # peak are tautologies. `power_range_read_ok` exists because a
            # silently-`None` range disabled the argument pre-check entirely in
            # the first draft, pushing a guaranteed refusal to mid-run -- after
            # a clock lock had already landed.
            try:
                lo, hi = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h)
                prov["power_min_w"] = lo / 1000.0
                prov["power_max_w"] = hi / 1000.0
                prov["power_range_read_ok"] = True
            except Exception:
                pass
            try:
                prov["power_default_w"] = (
                    pynvml.nvmlDeviceGetPowerManagementDefaultLimit(h) / 1000.0)
            except Exception:
                pass
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    return prov


def judge_power(baseline: dict, capped: list, recovered: dict, caps_w: list,
                clock_margin: int = DEFAULT_CLOCK_MARGIN_MHZ,
                bind_margin_w: float = DEFAULT_BIND_MARGIN_W,
                enforce_margin_w: float = DEFAULT_ENFORCE_MARGIN_W,
                requested_clock_mhz: int | None = None) -> dict:
    """Decide what the phases prove. Pure -- no hardware, so it is testable.

    **Check order is load-bearing** and was got wrong in the first draft:

    1. The power-limit restore check runs FIRST, before the load-liveness gate,
       because the readback is a device fact that does not need load. Behind the
       liveness gate, a loader that died during the recovered phase reported
       "not under load" about a card still holding a cap.
    2. Restore is measured against the limit observed at BASELINE -- the limit
       in force when the guard armed, which is what ``gpu_guard.restore()``
       actually restores. Comparing against the device *default* accuses a
       correct guard of failing whenever the operator had their own cap set.
    3. Only then efficacy.

    Anything unverifiable returns RESTORE UNVERIFIED or INCONCLUSIVE. A skipped
    check must never be silent, because the reason strings below assert what was
    verified and a durable artifact that overstates that is this project's
    signature defect.
    """
    verdict: dict = {}
    base_limit = baseline.get("power_limit_readback_mw")
    rec_limit = recovered.get("power_limit_readback_mw")
    verdict["baseline_power_limit_w"] = (
        None if base_limit is None else round(base_limit / 1000.0, 1))
    verdict["recovered_power_limit_w"] = (
        None if rec_limit is None else round(rec_limit / 1000.0, 1))

    # -- 1. restore, on the load-independent axis, fail-closed ---------------
    if base_limit is None or rec_limit is None:
        verdict["result"] = "RESTORE UNVERIFIED"
        verdict["reason"] = (
            "the power limit could not be read back "
            f"({'at baseline' if base_limit is None else 'after restore'}), so "
            "whether the cap was removed is UNKNOWN. Unverifiable is not "
            "verified -- check the card with `sudo python -m "
            "joule_agent.gpu_guard status` before trusting it."
        )
        return verdict
    if abs(rec_limit - base_limit) > 500:  # 0.5 W in mW
        verdict["result"] = "RESTORE FAILED"
        verdict["reason"] = (
            f"after restore the device holds a {rec_limit / 1000.0:.0f} W limit "
            f"against the {base_limit / 1000.0:.0f} W limit in force at baseline. "
            "The GPU is still capped -- run "
            "`sudo python -m joule_agent.gpu_guard restore`."
        )
        return verdict

    # -- 2. liveness ---------------------------------------------------------
    phases = [baseline, *capped, recovered]
    unloaded = [p["phase"] for p in phases if not p.get("loaded")]
    if unloaded:
        verdict["result"] = "INCONCLUSIVE"
        verdict["reason"] = (
            f"phases not under load: {', '.join(unloaded)}. An idle card draws "
            "idle power whether or not a cap is applied. (The power limit was "
            "read back and matches baseline, so the card is not left capped.)"
        )
        return verdict

    # A peak taken from a ~500 ms poll is smoothed low, and every enforcement
    # predicate below turns on the peak. Refuse rather than quietly weaken them.
    polled = [p["phase"] for p in phases if p.get("power_source") in ("polled", "mixed")]
    if polled:
        verdict["result"] = "INCONCLUSIVE"
        verdict["reason"] = (
            f"phases {polled} fell back to polled power readings, which refresh "
            "~500 ms and smooth the peak this gate turns on. A peak that cannot "
            "be attributed to the driver's sample ring cannot carry an "
            "enforcement claim."
        )
        return verdict

    base_clock = baseline["median_mhz"]
    base_peak = baseline["power_peak_w"]
    rec_clock = recovered["median_mhz"]
    verdict.update({
        "baseline_median_mhz": base_clock,
        "baseline_power_mean_w": baseline.get("power_mean_w"),
        "baseline_power_peak_w": base_peak,
        "recovered_median_mhz": rec_clock,
    })

    # -- 3. the control variable actually held -------------------------------
    # Everything below attributes clock movement to the cap. That premise fails
    # silently if the clock lock never landed, and an unlocked card drifts ~60
    # MHz run to run -- more than clock_margin -- so `clock_dropped` would fire
    # on drift alone.
    if requested_clock_mhz and abs(base_clock - requested_clock_mhz) > clock_margin:
        verdict["result"] = "INCONCLUSIVE"
        verdict["reason"] = (
            f"the baseline was asked to hold {requested_clock_mhz} MHz and sat at "
            f"{base_clock:.0f} MHz. The clock lock is the control variable here; "
            "without it, clock movement cannot be attributed to the cap."
        )
        return verdict
    if abs(rec_clock - base_clock) > clock_margin:
        verdict["result"] = "RESTORE FAILED"
        verdict["reason"] = (
            f"after restore the clock sat at {rec_clock:.0f} MHz against a "
            f"{base_clock:.0f} MHz baseline. The power limit came back but the "
            "clock did not -- run "
            "`sudo python -m joule_agent.gpu_guard restore`."
        )
        return verdict

    # -- 4. per-cap ----------------------------------------------------------
    per_cap, bindable = [], []
    for cap_w, ph in zip(caps_w, capped):
        rb = ph.get("power_limit_readback_mw")
        readback_w = None if rb is None else round(rb / 1000.0, 1)
        stored = (rb is not None and abs(rb - cap_w * 1000) <= 500)
        # THE predicate. A cap at or above the uncapped peak cannot act, so
        # "nothing changed" under it is a tautology and must never be counted as
        # evidence that capping is ineffective.
        can_bind = (base_peak is not None
                    and cap_w < base_peak - bind_margin_w)
        peak = ph.get("power_peak_w")
        row = {
            "cap_w": cap_w,
            "power_limit_readback_w": readback_w,
            "stored": stored,
            "can_bind": can_bind,
            "achieved_median_mhz": ph["median_mhz"],
            "power_mean_w": ph.get("power_mean_w"),
            "power_peak_w": peak,
            "clock_dropped": (base_clock - ph["median_mhz"]) > clock_margin,
            # Enforcement uses the SMALL margin. Sharing bind_margin_w here let
            # a device exceeding its cap by 5 W read as obedient.
            "peak_below_cap": (peak is not None
                               and peak <= cap_w + enforce_margin_w),
        }
        row["responded"] = bool(row["clock_dropped"] or row["peak_below_cap"])
        per_cap.append(row)
        if can_bind:
            bindable.append(row)
    verdict["per_cap"] = per_cap

    not_stored = [r["cap_w"] for r in per_cap if not r["stored"]]
    if not_stored:
        verdict["result"] = "REFUSED BY DEVICE"
        verdict["reason"] = (
            f"caps {not_stored} W did not read back from the device after being "
            "written. The driver did not store them, so nothing downstream was "
            "measuring what its label says."
        )
        return verdict

    if not bindable:
        # The branch the retraction needed and did not have.
        verdict["result"] = "INCONCLUSIVE"
        verdict["reason"] = (
            f"every cap tested sat at or above the uncapped peak draw of "
            f"{base_peak:.1f} W, so none of them could act. A cap that cannot "
            "bind changing nothing is a tautology, not evidence that capping is "
            "ineffective -- this is exactly the reasoning that had to be "
            f"retracted once. Choose caps below {base_peak:.1f} W (device "
            "minimum permitting) and re-run."
        )
        return verdict

    verdict["bindable_caps_w"] = [r["cap_w"] for r in bindable]
    unresponsive = [r["cap_w"] for r in bindable if not r["responded"]]
    skipped = [r["cap_w"] for r in per_cap if not r["can_bind"]]

    if len(unresponsive) == len(bindable):
        verdict["result"] = "CAP IS A NO-OP"
        verdict["reason"] = (
            f"caps {unresponsive} W were capable of binding (below the "
            f"{base_peak:.1f} W uncapped peak), stored on the device, and "
            "changed neither the clock nor the peak draw. The driver accepted "
            "the writes and the device ignored them: a static sweep's power rows "
            "on this hardware would compare identical configurations."
        )
        return verdict

    # Distinct caps must produce distinct behaviour, or sweep rows labelled by
    # requested watts are not distinct configurations. Same requirement the
    # clock axis is held to.
    order = sorted(range(len(bindable)), key=lambda i: bindable[i]["cap_w"])
    clocks = [bindable[i]["achieved_median_mhz"] for i in order]
    peaks = [bindable[i]["power_peak_w"] for i in order]
    monotonic_clock = all(a <= b + clock_margin for a, b in zip(clocks, clocks[1:]))
    monotonic_power = all(
        a is not None and b is not None and a <= b + enforce_margin_w
        for a, b in zip(peaks, peaks[1:]))
    verdict["response_monotonic_in_cap"] = bool(monotonic_clock and monotonic_power)

    if unresponsive:
        verdict["result"] = "PARTIAL"
        verdict["reason"] = (
            f"of {len(bindable)} caps capable of binding, {len(unresponsive)} "
            f"({unresponsive} W) changed neither the clock nor the peak draw "
            "while the rest did. The axis is not uniformly effective, so a sweep "
            "row's power label does not reliably name a distinct configuration."
        )
    elif len(bindable) == 1:
        verdict["result"] = "PARTIAL"
        verdict["reason"] = (
            f"exactly one cap ({bindable[0]['cap_w']} W) was capable of binding, "
            "and it did. That shows a cap can act, but not that two different "
            "caps are two different configurations -- which is what a swept "
            "power axis is labelled on. n=1 is the state the clock axis was in "
            "before its four-target run, and it was not enough there either."
        )
    elif not verdict["response_monotonic_in_cap"]:
        verdict["result"] = "PARTIAL"
        verdict["reason"] = (
            "every binding cap acted, but the response was not monotone in the "
            "cap: a tighter cap did not produce a lower clock and peak. Treat "
            "the achieved values as the real settings; a sweep row labelled by "
            "its requested watts would be mislabelled."
        )
    else:
        verdict["result"] = "CAP WORKS"
        verdict["reason"] = (
            f"all {len(bindable)} caps below the {base_peak:.1f} W uncapped peak "
            "stored on the device and each reduced the achieved clock or held "
            "the peak at or under its cap, monotonically in the cap, and the "
            "device returned to its baseline limit and clock afterwards."
        )

    if skipped:
        verdict["reason"] += (
            f". Caps {skipped} W were at or above the {base_peak:.1f} W uncapped "
            "peak and were excluded as incapable of acting, not counted as nulls."
        )
    verdict["scope_note"] = (
        "Efficacy was established under the synthetic FMA load, which draws "
        "harder than decode. This gate licenses the statement that this device "
        "honours a power cap -- it does NOT license any claim about what "
        "capping does to inference traffic. That question belongs to the sweep, "
        "and only for caps this gate shows can bind."
    )
    return verdict


def _run_guarded_phase(run_phase, label, clock_mhz, cap_mw,
                       controller_factory, guard_factory):
    """Arm one guard, apply at most one clock lock and one cap, measure, restore.

    ONE controller per guard, always. ``_ClaimMixin.claim`` refuses a second
    owner and nothing releases ``_claimed_by``, so a shared controller fails at
    phase 2 -- which is what killed ``verify_clock_lock.py`` on real hardware
    after a review pass had already found the identical defect in ``sweep.py``.
    Three occurrences in two files made it a rule.

    ``lock_clocks`` runs before ``set_power_limit_mw``, matching ``sweep.py``.
    """
    print(f"phase: {label}", flush=True)
    controller = controller_factory()
    guard = guard_factory(controller)
    try:
        with guard:
            if clock_mhz:
                guard.lock_clocks(clock_mhz)
            if cap_mw is not None:
                guard.set_power_limit_mw(cap_mw)
            return run_phase()
    finally:
        # Do NOT close while this guard still owes a restore. close() shuts NVML
        # down, and the atexit last-chance restore would then fail against a
        # dead handle -- turning a possible recovery into a certain failure.
        if getattr(guard, "_restored", True):
            controller.close()
        else:
            print("WARNING [verify_power_cap]: guard state unrestored; leaving "
                  "NVML open so the exit-time restore can run.", file=sys.stderr)


def baseline_phase_for(clock_mhz, run_phase, controller_factory=None,
                       guard_factory=None):
    """The uncapped, clock-locked reference.

    Module-level rather than inline in ``main()`` because a hardware-writing
    path only ``main()`` can reach is a path no test checks -- the rule written
    after two blocking ``sweep.py`` defects and the ``verify_clock_lock``
    hardware failure all hid there.
    """
    controller_factory = controller_factory or (lambda: _controller(0))
    guard_factory = guard_factory or (lambda c: GpuGuard(c, consent=True))
    return _run_guarded_phase(run_phase, "baseline (clock locked, no cap)",
                              clock_mhz, None, controller_factory, guard_factory)


def capped_phases_for(clock_mhz, caps_mw, run_phase, controller_factory=None,
                      guard_factory=None):
    """Apply each cap in turn and yield the phase measured under it."""
    controller_factory = controller_factory or (lambda: _controller(0))
    guard_factory = guard_factory or (lambda c: GpuGuard(c, consent=True))
    for cap_mw in caps_mw:
        yield _run_guarded_phase(
            lambda mw=cap_mw: run_phase(mw), f"capped at {cap_mw // 1000} W",
            clock_mhz, cap_mw, controller_factory, guard_factory)


def rejudge(artifact: Path, clock_margin: int, bind_margin_w: float,
            enforce_margin_w: float) -> dict:
    """Re-analyse a recorded run's raw phases under the current predicates.

    Legitimate because the phases are *measurements* and :func:`judge_power` is
    a pure function of them -- changing a threshold is re-analysis, not
    re-measurement. It CANNOT tell you whether the run reproduces.

    Phases are matched BY NAME, never by position. A ``bail()`` artifact records
    however many phases were collected, so positional indexing read the last
    capped phase as ``recovered`` and produced a fabricated RESTORE FAILED about
    a run that never had a recovered phase.
    """
    doc = json.loads(artifact.read_text())
    phases = doc.get("phases") or []
    caps_w = doc.get("caps_w") or []
    by_name = {p.get("phase"): p for p in phases}
    baseline = by_name.get("baseline")
    recovered = by_name.get("recovered")
    capped = [by_name.get(f"cap@{w}W") for w in caps_w]
    if baseline is None or recovered is None or any(c is None for c in capped):
        raise SystemExit(
            f"{artifact}: not a complete run (missing phases). Present: "
            f"{sorted(by_name)}. Re-judging needs baseline, every cap phase, "
            "and recovered.")
    out = dict(doc)
    out["verdict"] = judge_power(baseline, capped, recovered, caps_w,
                                 clock_margin, bind_margin_w, enforce_margin_w,
                                 doc.get("clock_mhz"))
    out["rejudged_from"] = {
        "artifact": str(artifact),
        "original_verdict": (doc.get("verdict") or {}).get("result"),
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        # The copied provenance block still names the ORIGINAL run's margins, so
        # the ones actually used are recorded here or the artifact misstates the
        # predicates that produced its own verdict.
        "predicates": {"clock_margin_mhz": clock_margin,
                       "bind_margin_w": bind_margin_w,
                       "enforce_margin_w": enforce_margin_w},
        "note": "re-analysis of stored phases under current predicates; NOT a "
                "re-run, and it cannot say whether the measurement reproduces",
    }
    return out


def _write(results: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


#: Exit codes. A failed restore gets its OWN code: collapsing "the GPU may still
#: be capped" into the same 2 as "you typed a bad argument" is how the worst
#: outcome becomes indistinguishable from the mildest.
EXIT_OK = 0
EXIT_NOT_PROVEN = 1
EXIT_REFUSED = 2
EXIT_HARDWARE_MAY_BE_MODIFIED = 3


def _exit_code(result: str) -> int:
    if result == "CAP WORKS":
        return EXIT_OK
    if result in ("RESTORE FAILED", "RESTORE UNVERIFIED"):
        return EXIT_HARDWARE_MAY_BE_MODIFIED
    if result == "NOT RUN":
        return EXIT_REFUSED
    return EXIT_NOT_PROVEN


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--clock-mhz", type=int, default=1800,
                   help="SM clock to lock for every phase, so the cap is the "
                        "only thing that varies. Required: an unlocked card "
                        "drifts ~60 MHz run to run, more than --clock-margin-mhz, "
                        "so clock movement could not be attributed to the cap.")
    p.add_argument("--caps-w", default="160,130,115,100",
                   help="comma-separated power caps in watts. Caps outside the "
                        "device's settable range are REFUSED by gpu_guard, not "
                        "clamped; caps above the measured uncapped peak are "
                        "excluded from the verdict as incapable of acting.")
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--clock-margin-mhz", type=int, default=DEFAULT_CLOCK_MARGIN_MHZ,
                   help="how far the clock must move before the move is "
                        f"attributed to the cap (default {DEFAULT_CLOCK_MARGIN_MHZ})")
    p.add_argument("--bind-margin-w", type=float, default=DEFAULT_BIND_MARGIN_W,
                   help="watts of headroom a cap needs BELOW the uncapped peak "
                        "to count as capable of binding. Larger is more "
                        f"conservative (default {DEFAULT_BIND_MARGIN_W}).")
    p.add_argument("--enforce-margin-w", type=float, default=DEFAULT_ENFORCE_MARGIN_W,
                   help="watts a peak may EXCEED its cap and still count as "
                        "honoured. Smaller is more conservative; deliberately "
                        "separate from --bind-margin-w, which pulls the other "
                        f"way (default {DEFAULT_ENFORCE_MARGIN_W}).")
    p.add_argument("--rejudge", default=None, metavar="ARTIFACT",
                   help="re-analyse a recorded run's raw phases under the "
                        "current predicates and write a new stamped artifact")
    p.add_argument("--state-path", default=None,
                   help="gpu_guard state file. Exists so tools/"
                        "verify_power_cap_contract_check.py can assert what each "
                        "refusal path leaves on disk without touching the real "
                        "one -- the CLI seam is where both of this tool's "
                        "BLOCKING defects lived.")
    p.add_argument("--out", default=None,
                   help="output path; default is a per-run timestamped file so "
                        "a re-run cannot erase the evidence from the last one")
    args = p.parse_args(argv)

    if args.rejudge:
        src = Path(args.rejudge)
        doc = rejudge(src, args.clock_margin_mhz, args.bind_margin_w,
                      args.enforce_margin_w)
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        out = Path(args.out) if args.out else Path(
            f"results/power_cap_rejudged-{stamp}.json")
        v = doc["verdict"]
        print(f"re-judged {src}")
        print(f"  was: {doc['rejudged_from']['original_verdict']}")
        print(f"  now: {v['result']}")
        print(f"  {v.get('reason', '')}")
        _write(doc, out)
        return _exit_code(v.get("result", ""))

    # ---- argument-only refusals, BEFORE any guard is entered ---------------
    # Entering the context manager arms, and arming publishes a claim. A caller
    # never allowed to write must never have claimed the device -- the rule
    # written after three defects reached committed code through this seam.
    # gpu_guard.main() pre-checks consent AND the clock floor for exactly this
    # reason; the first draft of this file pre-checked neither.
    caps_w = sorted({int(w) for w in args.caps_w.split(",") if w.strip()},
                    reverse=True)
    if not caps_w:
        print("REFUSED: no caps", file=sys.stderr)
        return EXIT_REFUSED
    args.caps_w = caps_w

    if not args.clock_mhz:
        print("REFUSED: --clock-mhz is required. The clock lock is this gate's "
              "control variable; an unlocked card drifts ~60 MHz run to run, "
              "which exceeds --clock-margin-mhz, so a clock drop could not be "
              "attributed to the cap.", file=sys.stderr)
        return EXIT_REFUSED
    if args.clock_mhz < MIN_SM_CLOCK_MHZ:
        print(f"REFUSED: --clock-mhz {args.clock_mhz} is below the "
              f"{MIN_SM_CLOCK_MHZ} MHz floor. gpu_guard refuses this rather "
              "than clamping it.", file=sys.stderr)
        return EXIT_REFUSED

    python = sys.executable
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = Path(args.out) if args.out else Path(
        f"results/power_cap_verification-{stamp}.json")

    prov = provenance(args, python)
    results = {"provenance": prov, "caps_w": caps_w,
               "clock_mhz": args.clock_mhz, "phases": []}

    def bail(reason: str, code: int, result: str = "NOT RUN") -> int:
        # An artifact is written even on a refusal. Returning early without one
        # leaves the PREVIOUS run's verdict on disk as the newest evidence.
        results["verdict"] = {"result": result, "reason": reason}
        _write(results, out)
        print(f"{'REFUSED' if result == 'NOT RUN' else result}: {reason}",
              file=sys.stderr)
        return code

    # The settable range is a DEVICE fact, read WITHOUT arming anything. If the
    # read failed we refuse rather than proceeding: a silently-`None` range
    # disabled this check entirely in the first draft, deferring a guaranteed
    # refusal to mid-run, after a clock lock had landed.
    if not prov.get("power_range_read_ok"):
        return bail("could not read this device's settable power range from "
                    "NVML, so the caps cannot be validated before arming. "
                    "Refusing rather than discovering the refusal mid-run with "
                    "a clock lock already applied.", EXIT_REFUSED)
    lo, hi = prov["power_min_w"], prov["power_max_w"]
    outside = [w for w in caps_w if w * 1000 < lo * 1000 - 0.5
               or w * 1000 > hi * 1000 + 0.5]
    if outside:
        return bail(
            f"caps {outside} W are outside this device's settable range "
            f"{lo:.1f}-{hi:.1f} W. gpu_guard refuses these rather than clamping "
            "them, so the run would abort partway through with a clock lock "
            "already applied.", EXIT_REFUSED)
    ref = prov.get("power_default_w") or hi
    if all(w >= ref for w in caps_w):
        return bail(
            f"every cap is at or above the {ref:.0f} W default limit, so none of "
            "them can change anything. This is the tautology the retracted power "
            "claim was built on.", EXIT_REFUSED)

    def _guard(c):
        kw = {"consent": True}
        if args.state_path:
            kw["state_path"] = Path(args.state_path)
        return GpuGuard(c, **kw)

    run = lambda: phase("baseline", args.seconds, python)  # noqa: E731
    try:
        baseline = baseline_phase_for(args.clock_mhz, run, guard_factory=_guard)
    except GpuGuardError as exc:
        return bail(str(exc), EXIT_REFUSED)
    except Exception as exc:                      # noqa: BLE001
        return bail(f"unexpected failure during the baseline phase: "
                    f"{exc.__class__.__name__}: {exc}", EXIT_REFUSED)
    results["phases"].append(baseline)
    print(f"  {baseline}", flush=True)

    # The baseline must be UNCAPPED for `can_bind` to mean anything. A card left
    # capped by a previous failed run makes every cap look non-bindable and the
    # verdict INCONCLUSIVE for the wrong reason. cli_contract_check.py sets the
    # precedent of refusing to start when the card is not stock.
    base_limit = baseline.get("power_limit_readback_mw")
    default_mw = (None if prov.get("power_default_w") is None
                  else prov["power_default_w"] * 1000)
    if base_limit is None:
        return bail("the power limit could not be read back at baseline, so the "
                    "reference this gate measures restores against is unknown.",
                    EXIT_REFUSED)
    if default_mw is not None and abs(base_limit - default_mw) > 500:
        return bail(
            f"the card already holds a {base_limit / 1000.0:.0f} W limit against a "
            f"{default_mw / 1000.0:.0f} W default, so the 'uncapped' baseline is "
            "not uncapped and every cap would be judged against it. Clear it "
            "with `sudo python -m joule_agent.gpu_guard restore` first.",
            EXIT_REFUSED)

    capped_phases = []
    try:
        for ph in capped_phases_for(
                args.clock_mhz, [w * 1000 for w in caps_w],
                lambda mw: phase(f"cap@{mw // 1000}W", args.seconds, python),
                guard_factory=_guard):
            capped_phases.append(ph)
            results["phases"].append(ph)
            print(f"  {ph}", flush=True)
    except GpuGuardError as exc:
        # A failed restore arrives here too, and it is NOT a refusal: something
        # reached the device. Distinguish it, or the worst outcome is reported
        # in the words of the mildest.
        msg = str(exc)
        if "MAY STILL BE MODIFIED" in msg or "restore incomplete" in msg:
            return bail(msg, EXIT_HARDWARE_MAY_BE_MODIFIED, "RESTORE FAILED")
        return bail(msg, EXIT_REFUSED)
    except Exception as exc:                      # noqa: BLE001
        return bail(f"unexpected failure during the capped phases: "
                    f"{exc.__class__.__name__}: {exc}",
                    EXIT_HARDWARE_MAY_BE_MODIFIED, "RESTORE FAILED")

    print("phase: recovered (after restore)", flush=True)
    recovered = phase("recovered", args.seconds, python)
    results["phases"].append(recovered)
    print(f"  {recovered}", flush=True)

    results["verdict"] = judge_power(
        baseline, capped_phases, recovered, caps_w,
        args.clock_margin_mhz, args.bind_margin_w, args.enforce_margin_w,
        args.clock_mhz)
    v = results["verdict"]
    print(f"\n{v['result']}: {v.get('reason', '')}")
    if v.get("scope_note"):
        print(f"\nscope: {v['scope_note']}")
    _write(results, out)
    return _exit_code(v["result"])


if __name__ == "__main__":
    raise SystemExit(main())
