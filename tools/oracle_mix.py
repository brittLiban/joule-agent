"""Per-request dynamic oracle over a completed static sweep.

The question: given perfect foresight and zero switching cost, can ANY assignment
of requests to clocks beat the best static clock at the same p99?

Why the construction is legitimate: every point in a sweep replayed the SAME
`Schedule` object (identical digest, asserted per run), so request i at one clock
and request i at another are the same request -- same prompt, same output length,
same arrival time. Its measured latency at each clock is a paired observation, so
an assignment of requests to clocks can be scored from measured data with no
simulator.

Why it is an UPPER bound on any real controller:
  - perfect foresight (knows every request's latency at every clock in advance)
  - per-request granularity (a real controller switches in time, not per request)
  - zero transition cost, zero actuation latency, zero detection delay

READ THIS BEFORE TRUSTING A RESULT -- the bound is only meaningful over a NARROW
clock range. One clock applies to the whole GPU. Under continuous batching many
requests are in flight at once, so a per-request assignment is realizable only
where concurrent requests would have received the same clock. Across a wide range
the oracle draws request i's latency from a run where the entire card was at one
clock and its concurrent neighbour's from a run where the entire card was at
another; those cannot both be true of the same instant. The bound is then taken
over a policy class containing physically impossible policies, and is useful ONLY
when it comes back BELOW the bar. A result above the bar proves nothing.

Measured on the reference hardware (RTX 3060 Ti, driver 595.71.05):
  refined grid 1530-1650 (7.8% spread) -> -5.79%, meaningful, below the 10% bar
  wide grid     600-1800 (3x spread)   -> -25.1%, METHOD BREAKDOWN, not evidence

Further caveat that cannot be removed from this data: serving request i at a
different clock changes the queue state seen by request i+1, and each clock was
measured in isolation. Direction is knowable even where size is not -- a promoted
request in a mostly-slow run would still queue behind slow requests, so it would
NOT achieve the latency it had in the all-fast run. The oracle overstates the
benefit of promotion.

Energy weighting is by request COUNT (per-request token counts are not in the
latency artifact). This also biases in the oracle's favour: promoted requests are
the high-latency ones, hence the long ones, so they carry more tokens than an
equal-count share.

Usage:
    python tools/oracle_mix.py results/sweeps/<dir> [more dirs ...]
"""

import csv
import json
import os
import statistics as st
import sys


def pct(sorted_vals, p):
    """Identical to loadgen._pct so percentiles match the sweep's own."""
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def load(sweep, exclude_capped=True):
    meta = json.load(open(os.path.join(sweep, "sweep.json")))
    energy, clock = {}, {}
    for s in meta["summaries"]:
        if exclude_capped and "-pl" in s["name"]:
            continue          # power axis is a different knob; claims retracted
        energy[s["name"]] = s["joules_per_token_mean"]
        clock[s["name"]] = s["achieved_clock_median_mhz"]
    lat = {}
    for name in energy:
        for r in range(20):
            p = os.path.join(sweep, "latencies", f"{name}_r{r}.csv")
            if os.path.exists(p):
                lat[(name, r)] = [float(row["latency_s"])
                                  for row in csv.DictReader(open(p))]
    return meta, energy, clock, lat


def optimise(points, energy, lat_r, T):
    """points: any order. Returns (p99, J/token, n_above_cheapest) or None.

    Energy is separable per request and p99 over n samples is set by the value at
    interpolated index (n-1)*0.99, so the constraint is essentially "at most k
    requests may exceed the threshold" and the optimum is reachable directly:
    give each request its cheapest clock that meets T, then greedily demote for
    as long as p99 still fits, taking the largest energy saving each step.
    """
    n = len(lat_r[points[0]])
    order = sorted(range(len(points)), key=lambda j: energy[points[j]])
    cheapest = order[0]

    assign = []
    for i in range(n):
        pick = next((j for j in order if lat_r[points[j]][i] <= T), None)
        assign.append(pick if pick is not None else cheapest)

    cur = [lat_r[points[assign[i]]][i] for i in range(n)]
    if pct(sorted(cur), 99) > T:
        return None

    while True:
        best = None
        for i in range(n):
            for j in order:
                if energy[points[j]] >= energy[points[assign[i]]]:
                    continue
                saving = energy[points[assign[i]]] - energy[points[j]]
                if best is not None and saving <= best[0]:
                    continue
                trial = list(cur)
                trial[i] = lat_r[points[j]][i]
                if pct(sorted(trial), 99) <= T:
                    best = (saving, i, j, trial)
        if best is None:
            break
        _, i, j, trial = best
        assign[i], cur = j, trial

    e = sum(energy[points[a]] for a in assign) / n
    used = {points[a] for a in assign}
    return (pct(sorted(cur), 99), e,
            sum(1 for a in assign if a != cheapest), used)


def run(sweep):
    meta, energy, clock, lat = load(sweep)
    T = meta["slo_threshold_s"]
    points = sorted(energy, key=lambda x: energy[x])
    sm = {s["name"]: s["latency_p99_mean_s"] for s in meta["summaries"]}
    reps = sorted({r for (_, r) in lat})
    clocks = sorted(c for c in (clock[p] for p in points) if c)
    spread = (clocks[-1] / clocks[0] - 1) * 100 if clocks else 0

    print(f"\n{'=' * 74}\n{sweep}\n{'=' * 74}")
    print(f"clocks: {clocks}   (spread {spread:.1f}%)")
    print(f"frozen SLO threshold: {T:.4f}s   driver "
          f"{meta.get('provenance', {}).get('driver_version', '?')}")

    bad = [n for n in points
           if abs(st.mean([pct(sorted(lat[(n, r)]), 99)
                           for r in reps if (n, r) in lat]) - sm[n]) > 1e-4]
    print(f"validity (all-at-one-clock reproduces sweep p99): "
          f"{'PASS' if not bad else 'FAIL ' + str(bad)}")
    if bad:
        return None

    best_j, best_n = min((energy[n], n) for n in points if sm[n] <= T)
    print(f"static baseline: {best_n} at {best_j:.5f} J/token (p99 {sm[best_n]:.4f})")

    res, used_all = [], set()
    for r in reps:
        lat_r = {n: lat[(n, r)] for n in points if (n, r) in lat}
        if len(lat_r) != len(points):
            continue
        out = optimise(points, energy, lat_r, T)
        if out is None:
            print(f"  r{r}: infeasible")
            continue
        p99, e, moved, used = out
        res.append(e)
        used_all |= used
        print(f"  r{r}: p99 {p99:.4f}  J/token {e:.5f}  "
              f"({moved}/{len(lat_r[points[0]])} above cheapest)  "
              f"{100 * (e - best_j) / best_j:+.2f}% vs static")

    if not res:
        return None
    m = st.mean(res)
    d = 100.0 * (m - best_j) / best_j

    # The diagnostic that decides whether this number means anything is the
    # spread of the clocks the oracle ACTUALLY ASSIGNS -- not the spread of the
    # grid it was offered. A grid containing stock is fine if the oracle never
    # reaches for it. Concurrent requests share one clock, so the pairing holds
    # only where the assigned clocks are close enough that a request's latency
    # is nearly the same whichever leg it is drawn from.
    uc = sorted(clock[p] for p in used_all if clock[p])
    used_spread = (uc[-1] / uc[0] - 1) * 100 if len(uc) > 1 else 0.0

    print(f"\n  ORACLE {m:.5f} vs STATIC {best_j:.5f}  ->  {d:+.2f}%")
    print(f"  clocks actually used: {uc}  (spread {used_spread:.1f}%)")
    if used_spread > 20:
        print("  *** NOT A VALID BOUND. One clock applies to the whole GPU, and")
        print("  *** these assignments put concurrent requests at clocks "
              f"{used_spread:.0f}% apart,")
        print("  *** which cannot be true of the same instant. Ignore the "
              "percentage.")
    elif d <= -10:
        print("  clears the 10% bar -- dynamic is alive on this workload.")
    else:
        print("  does NOT clear the 10% bar, and the bound is tight over this")
        print("  spread: no controller in this range can do better.")
    return d


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__.strip().splitlines()[-1])
        raise SystemExit(2)
    for d in sys.argv[1:]:
        run(d)
