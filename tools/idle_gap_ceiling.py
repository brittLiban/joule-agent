"""Analyse the interleaved legs written by ``tools/idle_gap_ceiling.sh``.

Reports the paired power difference between a high and a low locked clock at
idle, and turns it into a ceiling on what any idle-gap controller could give
back. Power comes from the GPU energy counter (delta / elapsed) rather than the
sampled power stream: the counter is gap-free, and on the reference card one leg
of four had a sample stream corrupted by ring-drain artifacts while its counter
value was consistent with its pair.

The PAIRED difference is the measurement. Absolute level drifts across a session
-- 2.0 W on the reference card -- while the interleaved pairs agreed to 0.02 W.
Reading leg 1 against leg 4 instead would report mostly drift.

Usage: python tools/idle_gap_ceiling.py runs/idle_gap_manifest.tsv [--loaded-w W]
"""

import argparse
import csv
import statistics as st
import sys
from pathlib import Path


def leg_power(run_dir):
    """Mean watts over the leg, from the energy counter."""
    rows = list(csv.DictReader(open(Path(run_dir) / "polled.csv")))
    e = [float(r["energy_mj"]) for r in rows if r["energy_mj"] not in ("", "None")]
    t = [float(r["wall_us"]) for r in rows]
    clk = [int(r["sm_clock_mhz"]) for r in rows
           if r["sm_clock_mhz"] not in ("", "None")]
    dur = (t[-1] - t[0]) / 1e6
    return (e[-1] - e[0]) / 1000.0 / dur, st.median(clk), dur


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("--loaded-w", type=float, default=None,
                   help="mean power under the real workload at the tuned clock; "
                        "without it the ceiling cannot be expressed as a "
                        "percentage and only dP is reported")
    p.add_argument("--f-idle", type=float, default=0.5,
                   help="fraction of wall time the workload leaves the GPU idle "
                        "(default 0.5 -- a generous upper bound for a 50%% duty "
                        "cycle; the achieved figure is usually lower)")
    args = p.parse_args(argv)

    legs = list(csv.DictReader(open(args.manifest), delimiter="\t"))
    if len(legs) < 4:
        print(f"need 4 interleaved legs, found {len(legs)}", file=sys.stderr)
        return 2

    print(f"{'leg':>4}{'req MHz':>9}{'W':>9}{'achieved':>10}{'secs':>7}")
    vals = []
    for leg in legs:
        w, clk, dur = leg_power(leg["run_dir"])
        vals.append((int(leg["requested_mhz"]), w))
        print(f"{leg['leg']:>4}{leg['requested_mhz']:>9}{w:>9.2f}"
              f"{clk:>10.0f}{dur:>7.1f}")

    # legs are HIGH, LOW, HIGH, LOW -- pair adjacent legs so session drift cancels
    d1 = vals[0][1] - vals[1][1]
    d2 = vals[2][1] - vals[3][1]
    dp = (d1 + d2) / 2
    drift = abs(vals[2][1] - vals[0][1])

    print(f"\npaired dP:  {d1:.2f} W and {d2:.2f} W  ->  mean {dp:.2f} W")
    print(f"absolute drift across the session: {drift:.2f} W")
    if drift > abs(d1 - d2) * 3:
        print("  (the pairs agree far better than the level drifts -- which is "
              "what the interleaving bought)")

    if args.loaded_w:
        ceiling = 100.0 * args.f_idle * dp / args.loaded_w
        print(f"\nCEILING = f_idle {args.f_idle} x {dp:.2f} W / "
              f"{args.loaded_w:.2f} W = {ceiling:.1f}%")
        print("\nThis is an UPPER bound and nothing approaches it: a controller "
              "gives back\npower only during a gap it has DETECTED, and "
              "detection is bounded by the\npower-sensor refresh (~500 ms on "
              "the reference card) plus the serving\nengine's metrics scrape "
              "cadence. It must also exit the gap early to avoid\npaying tail "
              "latency for a late upclock.")
    else:
        print("\nPass --loaded-w <mean watts under your real workload at the "
              "tuned clock>\nto convert dP into a ceiling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
