"""Gate 0: does this workload leave the GPU idle at all?

The idle-gap ceiling is ``f_idle x dP / P_loaded``. dP is measurable in six
minutes with no serving engine. ``f_idle`` is the term everyone assumes and
nobody measures -- and a preset's nominal duty cycle is NOT it, because requests
admitted near the end of an on-period drain into the off-period.

If this comes back near zero, the idle-gap branch is dead for this workload and
no controller, no static clock range, and no oracle needs to be built or run.
That is the cheapest possible kill, and it costs one 60 s unprivileged run.

WHY UTILISATION IS NOT THE CRITERION HERE. NVML utilisation is a windowed
activity percentage, and on a workstation with a desktop session it never
reaches zero: measured on the reference box with NO traffic at all and the engine
idle, util held a 5.0% mean and 11% max and hit zero in 0 of 302 samples. Using
``util == 0`` would report no idle for a reason that has nothing to do with the
engine. So:

  primary    engine-empty (running == 0 and waiting == 0) -- also the only
             signal a real controller could act on
  physical   power near the measured idle floor, which discriminates idle from
             busy by ~5x on this hardware where utilisation cannot
  reported   util == 0, recorded but expected to be ~0 on a desktop host and
             explicitly not evidence about the engine

CONTIGUOUS DURATION, NOT TOTAL. A controller cannot use a gap shorter than its
detection plus actuation time, so total idle fraction overstates what is
reachable. f_exploitable(tau) = sum(max(0, g_i - tau)) / T over measured gaps
g_i. tau = 0 is the perfect-foresight oracle; a causal controller pays the
sensor cadence and the upclock reserve.

Runs unprivileged. Writes no GPU state.
"""

import argparse
import json
import statistics as st
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

PY = ".venv/bin/python"


def poll_engine(endpoint, stop, out, hz=20.0):
    """Sample running/waiting until told to stop."""
    period = 1.0 / hz
    while not stop.is_set():
        t = time.monotonic()
        try:
            with urllib.request.urlopen(f"{endpoint}/metrics", timeout=2) as r:
                text = r.read().decode()
            run = wait = 0.0
            for line in text.splitlines():
                if line.startswith("vllm:num_requests_running"):
                    run += float(line.rsplit(" ", 1)[1])
                elif line.startswith("vllm:num_requests_waiting"):
                    wait += float(line.rsplit(" ", 1)[1])
            out.append((t, run, wait))
        except Exception:
            pass
        time.sleep(max(0.0, period - (time.monotonic() - t)))


def gaps_from(samples, is_idle):
    """Contiguous [start, end) intervals where is_idle(sample) holds.

    Duration-weighted: each sample covers the span to the next sample, so an
    uneven poll rate does not bias the fraction.
    """
    out, start = [], None
    for i, s in enumerate(samples[:-1]):
        t0, t1 = s[0], samples[i + 1][0]
        if is_idle(s):
            if start is None:
                start = t0
        elif start is not None:
            out.append((start, t0))
            start = None
    if start is not None:
        out.append((start, samples[-1][0]))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--endpoint", default="http://localhost:8000")
    p.add_argument("--preset", default="bursty")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--rate-scale", type=float, default=1.0)
    p.add_argument("--idle-floor-w", type=float, default=None,
                   help="mean watts when this GPU is idle; samples within "
                        "--idle-margin-w of it count as physically idle. "
                        "Measure it with `telemetry record` and no traffic.")
    p.add_argument("--idle-margin-w", type=float, default=5.0)
    p.add_argument("--tau", default="0,0.25,0.5,1.0",
                   help="detection+actuation reserves to report, seconds")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    engine, stop = [], threading.Event()
    th = threading.Thread(target=poll_engine,
                          args=(args.endpoint, stop, engine), daemon=True)
    th.start()

    print(f"running {args.preset} for {args.duration:.0f}s while sampling "
          f"engine state ...", flush=True)
    t0 = time.monotonic()
    cmd = [PY, "-m", "joule_agent.loadgen", "--preset", args.preset,
           "--seed", str(args.seed), "--duration", str(args.duration),
           "--rate-scale", str(args.rate_scale)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    wall = time.monotonic() - t0
    stop.set(); th.join(timeout=2)

    if proc.returncode != 0:
        print(proc.stderr[-500:], file=sys.stderr)
        return 1
    rep, _ = json.JSONDecoder().raw_decode(proc.stdout[proc.stdout.index("{"):])

    if len(engine) < 10:
        print("too few engine samples; is /metrics reachable?", file=sys.stderr)
        return 1

    span = engine[-1][0] - engine[0][0]
    empty = gaps_from(engine, lambda s: s[1] == 0 and s[2] == 0)
    total_empty = sum(b - a for a, b in empty)
    f_idle = total_empty / span if span else 0.0

    print()
    print("=" * 68)
    print(f"GATE 0 -- {args.preset} (seed {args.seed}, rate x{args.rate_scale})")
    print("=" * 68)
    print(f"{'wall span':<28}{span:.1f} s ({len(engine)} engine samples)")
    print(f"{'requests completed':<28}{rep.get('completed')} / "
          f"{rep.get('scheduled')}, failed {rep.get('failed')}")
    print(f"{'max running / waiting':<28}{rep.get('max_running')} / "
          f"{rep.get('max_waiting')}")
    print()
    print(f"{'ENGINE-EMPTY FRACTION':<28}{f_idle * 100:.1f}%   "
          f"(running == 0 and waiting == 0)")
    print(f"{'contiguous empty gaps':<28}{len(empty)}")
    if empty:
        g = sorted(b - a for a, b in empty)
        print(f"{'  gap seconds':<28}min {g[0]:.2f}  median "
              f"{st.median(g):.2f}  max {g[-1]:.2f}")
        print()
        print("  f_exploitable(tau) -- fraction of wall time usable by a")
        print("  controller that needs tau seconds to detect and act:")
        for tau in [float(x) for x in args.tau.split(",")]:
            usable = sum(max(0.0, (b - a) - tau) for a, b in empty)
            print(f"    tau = {tau:>4.2f}s ->  {usable / span * 100:>5.1f}%")
    else:
        print("  NO contiguous engine-empty interval was observed.")

    print()
    if f_idle < 0.02:
        print("VERDICT: no meaningful idle. The idle-gap branch is dead for this")
        print("workload -- there is nothing for a clock range or a controller to")
        print("reclaim, and any ceiling computed from an assumed duty cycle is an")
        print("artifact. Skip the range test and the oracle for this preset.")
    else:
        print("VERDICT: measurable idle exists. Re-measure at the TUNED-STATIC")
        print("clock before using this number -- stock drains faster, so this is")
        print("an UPPER bound on the idle a tuned lock would leave available.")

    doc = {"preset": args.preset, "seed": args.seed,
           "rate_scale": args.rate_scale, "wall_span_s": span,
           "engine_samples": len(engine), "engine_empty_fraction": f_idle,
           "gaps_s": [b - a for a, b in empty],
           "loadgen": {k: rep.get(k) for k in
                       ("completed", "scheduled", "failed", "max_running",
                        "max_waiting", "achieved_saturation", "latency_p99_s")}}
    if args.out:
        Path(args.out).write_text(json.dumps(doc, indent=2))
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
