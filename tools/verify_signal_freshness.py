"""How fresh is the signal a causal controller would actually act on?

The idle-gap ceiling has been discounted by a "signal age" term taken from the
POWER sensor's ~500 ms refresh. That is the wrong term. A controller's input is
the serving engine's queue and token state, not power -- power is how the
*experiment* measures energy, not how the *policy* sees the world. Charging the
controller for a sensor it does not read handicaps it with someone else's
latency.

So measure the real budget:

    signal age  =  how stale the engine's exported state is when read
                +  how long the read itself takes
                (+ decision time, negligible)
                (+ clock-write latency, measured separately at ~23 ms call /
                   <=200 ms to observe)

This polls /metrics as fast as it can and records when each field's VALUE
actually CHANGES. The observed change interval is the *upper* bound on the
export cadence: if the exporter updates faster than the poll loop, this reports
the poll period instead, so the loop's own rate is measured and printed first.
Same discipline as component 1's `characterize_sensors()`, which found the power
sensor at ~500 ms and refused to poll faster and pretend.

Run it under real traffic. An idle engine's counters do not move, and a cadence
measured on a signal that never changes is not a cadence.

No root, no GPU writes.
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

WATCH = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:generation_tokens_total",
    "vllm:prompt_tokens_total",
)


def scrape(endpoint):
    t0 = time.monotonic()
    with urllib.request.urlopen(f"{endpoint}/metrics", timeout=2) as r:
        text = r.read().decode()
    fetch_ms = (time.monotonic() - t0) * 1000
    vals = {}
    for line in text.splitlines():
        for k in WATCH:
            if line.startswith(k):
                vals[k] = vals.get(k, 0.0) + float(line.rsplit(" ", 1)[1])
    return time.monotonic(), fetch_ms, vals


def sampler(endpoint, stop, out):
    while not stop.is_set():
        try:
            out.append(scrape(endpoint))
        except Exception:
            pass


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--endpoint", default="http://localhost:8000")
    p.add_argument("--preset", default="bursty")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--duration", type=float, default=45.0)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    samples, stop = [], threading.Event()
    th = threading.Thread(target=sampler, args=(args.endpoint, stop, samples),
                          daemon=True)
    th.start()
    print(f"scraping /metrics as fast as possible while {args.preset} runs "
          f"for {args.duration:.0f}s ...", flush=True)
    proc = subprocess.run(
        [PY, "-m", "joule_agent.loadgen", "--preset", args.preset,
         "--seed", str(args.seed), "--duration", str(args.duration)],
        capture_output=True, text=True, timeout=1800)
    stop.set(); th.join(timeout=3)
    if proc.returncode != 0:
        print(proc.stderr[-400:], file=sys.stderr)
        return 1

    if len(samples) < 50:
        print("too few scrapes", file=sys.stderr)
        return 1

    span = samples[-1][0] - samples[0][0]
    periods = [samples[i + 1][0] - samples[i][0] for i in range(len(samples) - 1)]
    fetch = [s[1] for s in samples]
    poll_ms = st.median(periods) * 1000

    print()
    print("=" * 68)
    print("SIGNAL FRESHNESS")
    print("=" * 68)
    print(f"{'scrapes':<26}{len(samples)} in {span:.1f}s")
    print(f"{'poll period (median)':<26}{poll_ms:.1f} ms   "
          f"<- the observer's own floor")
    print(f"{'GET /metrics latency':<26}"
          f"median {st.median(fetch):.1f} ms, p95 "
          f"{sorted(fetch)[int(len(fetch) * .95)]:.1f} ms")
    print()
    print(f"{'field':<34}{'changes':>9}{'median gap':>13}{'p95':>9}")
    out = {"poll_period_ms": poll_ms, "fetch_ms_median": st.median(fetch),
           "span_s": span, "scrapes": len(samples), "fields": {}}
    for k in WATCH:
        ts = [(t, v[k]) for t, _, v in samples if k in v]
        if len(ts) < 10:
            print(f"{k:<34}{'absent':>9}")
            continue
        changes = [ts[i + 1][0] for i in range(len(ts) - 1)
                   if ts[i + 1][1] != ts[i][1]]
        if len(changes) < 2:
            print(f"{k:<34}{len(changes):>9}   never moved -- cadence unknown")
            out["fields"][k] = {"changes": len(changes)}
            continue
        gaps = sorted(changes[i + 1] - changes[i]
                      for i in range(len(changes) - 1))
        med = st.median(gaps) * 1000
        p95 = gaps[int(len(gaps) * 0.95)] * 1000
        print(f"{k:<34}{len(changes):>9}{med:>11.1f} ms{p95:>7.0f} ms")
        out["fields"][k] = {"changes": len(changes), "median_gap_ms": med,
                            "p95_gap_ms": p95}

    ctrl = [out["fields"].get(k, {}).get("median_gap_ms")
            for k in ("vllm:num_requests_running", "vllm:num_requests_waiting")]
    ctrl = [c for c in ctrl if c]
    print()
    if ctrl:
        age = max(ctrl) + st.median(fetch)
        print(f"CONTROLLER SIGNAL AGE  ~{age:.0f} ms")
        print(f"  = queue-state export cadence ({max(ctrl):.0f} ms)")
        print(f"  + scrape latency ({st.median(fetch):.1f} ms)")
        print()
        print("This is what a queue-driven policy pays, NOT the power sensor's")
        print("~500 ms -- the controller never reads power. Add clock-write")
        print("latency (~23 ms call, <=200 ms to observe) for the full budget.")
        if max(ctrl) <= poll_ms * 1.5:
            print()
            print("  CAVEAT: the observed cadence is at the poll floor, so the")
            print("  exporter may update faster than this can see. Treat it as")
            print("  an UPPER bound on staleness, not a measurement of it.")
        out["controller_signal_age_ms"] = age
    else:
        print("queue fields never changed -- rerun under traffic that queues.")

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
