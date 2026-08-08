"""How fast does an EXPLICIT clock change actually take effect?

Distinct from the question Gate 1 answered. Gate 1 showed the device's own
governor will not descend inside a 2.5 s idle gap when given a range. That is a
statement about the governor's policy, not about the actuator: a direct
``nvmlDeviceSetGpuLockedClocks`` is a command, not a hint, and its latency has
never been measured here.

This number sets three things:

  * the control epoch -- a period shorter than the settling time is fiction
  * tau in ``f_exploitable(tau) = sum(max(0, g_i - tau)) / T``, which decides how
    much of a measured idle gap a causal controller can actually reach
  * whether a perfect-foresight oracle is worth building at all. If a transition
    costs a large fraction of a 2.5 s gap, the oracle's ceiling collapses toward
    the static baseline before any policy exists.

MEASURED SEPARATELY, because they are different costs:

  call_ms       how long the guarded write itself blocks. Includes the guard's
                state-file fsync, which is correct for crash safety and is a
                real per-decision cost for a controller.
  settle_ms     wall time from issuing the command to first OBSERVING the target
                clock. This is the number that bounds control, and it is not the
                same as the call returning.

Polling has its own resolution: the sampler reads NVML in a tight loop, so
settle_ms is quantised by however fast that loop actually turns, which is
reported. A settle time at the polling floor means "faster than this instrument
can see", not "instant".

Run WITH GPU load. An idle card under an exact lock does sit at the locked
clock, but the transition being characterised is the one a controller performs
mid-inference, and a busy pipeline is not the same as a quiet one.

Requires root.
"""

import argparse
import json
import statistics as st
import subprocess
import time
from pathlib import Path

PY = ".venv/bin/python"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--low", type=int, default=None,
                   help="low clock (default: device minimum)")
    p.add_argument("--high", type=int, default=None,
                   help="high clock (default: a mid-high supported clock)")
    p.add_argument("--reps", type=int, default=20,
                   help="transitions per direction")
    p.add_argument("--timeout-s", type=float, default=5.0)
    p.add_argument("--load", action="store_true",
                   help="drive synthetic GPU load during the measurement")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    import pynvml as n
    from joule_agent.gpu_guard import GpuGuard, NvmlGpuController

    n.nvmlInit()
    h = n.nvmlDeviceGetHandleByIndex(args.gpu)
    supported = sorted(n.nvmlDeviceGetSupportedGraphicsClocks(
        h, n.nvmlDeviceGetSupportedMemoryClocks(h)[0]))
    low = args.low or supported[0]
    high = args.high or supported[int(len(supported) * 0.8)]
    print(f"transitions between {low} and {high} MHz, {args.reps} each way")

    # Calibrate the observer before trusting any settle time it reports.
    t0 = time.monotonic()
    for _ in range(200):
        n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM)
    poll_ms = (time.monotonic() - t0) / 200 * 1000
    print(f"polling resolution: {poll_ms:.3f} ms per NVML clock read\n")

    load = None
    if args.load:
        load = subprocess.Popen(
            [PY, "tools/synthetic_gpu_load.py", "--duration",
             str(int(args.reps * 2 * (args.timeout_s + 0.5) + 30)),
             "--period", "600", "--duty", "1.0"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(4)
        print("synthetic load running\n")

    rows = []
    try:
        guard = GpuGuard(NvmlGpuController(args.gpu), consent=True)
        with guard:
            guard.lock_clocks(high)
            time.sleep(1.0)
            for i in range(args.reps * 2):
                target = low if i % 2 == 0 else high
                before = n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM)

                t_call = time.monotonic()
                guard.lock_clocks(target)
                t_ret = time.monotonic()

                seen, settle = None, None
                deadline = t_ret + args.timeout_s
                while time.monotonic() < deadline:
                    seen = n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM)
                    if seen == target:
                        settle = (time.monotonic() - t_call) * 1000
                        break
                rows.append({
                    "i": i, "direction": "down" if target == low else "up",
                    "from_mhz": before, "to_mhz": target, "observed_mhz": seen,
                    "call_ms": (t_ret - t_call) * 1000, "settle_ms": settle,
                })
                time.sleep(0.3)
    finally:
        if load:
            load.terminate()

    def stats(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        vals.sort()
        return {"n": len(vals), "median": st.median(vals),
                "p95": vals[min(len(vals) - 1, int(len(vals) * 0.95))],
                "max": vals[-1]}

    print(f"{'direction':<12}{'n':>4}{'call ms':>26}{'settle ms':>26}")
    print(f"{'':12}{'':4}{'median / p95 / max':>26}{'median / p95 / max':>26}")
    out = {}
    for d in ("up", "down"):
        sub = [r for r in rows if r["direction"] == d]
        c, s = stats([r["call_ms"] for r in sub]), stats([r["settle_ms"] for r in sub])
        out[d] = {"call": c, "settle": s,
                  "unreached": sum(1 for r in sub if r["settle_ms"] is None)}
        cs = f"{c['median']:.1f} / {c['p95']:.1f} / {c['max']:.1f}" if c else "-"
        ss = f"{s['median']:.0f} / {s['p95']:.0f} / {s['max']:.0f}" if s else "NEVER REACHED"
        print(f"{d:<12}{len(sub):>4}{cs:>26}{ss:>26}")
        if out[d]["unreached"]:
            print(f"{'':16}{out[d]['unreached']} of {len(sub)} never reached "
                  f"target within {args.timeout_s}s")

    print()
    worst = max((out[d]["settle"]["p95"] for d in out if out[d]["settle"]),
                default=None)
    if worst is None:
        print("VERDICT: transitions did not complete. Per-epoch control is not")
        print("possible on this device by this mechanism.")
    else:
        # settle_ms is an UPPER bound on actuation, not a measurement of it.
        # On the reference card 29 of 30 transitions landed within 3 ms of
        # 200 ms in BOTH directions while the guarded write itself returned in
        # ~23 ms. A physical settling process has variance; that constancy is
        # the reporting path, not the hardware. NVML's clock reading refreshes
        # with ~200 ms latency, the same class of limitation component 1 found
        # in the power sensor (~500 ms). True actuation lies between the call
        # duration and the first observation, and this instrument cannot
        # separate them.
        spread = worst - min(
            (out[d]["settle"]["median"] for d in out if out[d]["settle"]),
            default=worst)
        print(f"actuation is BOUNDED BY ~{worst / 1000:.2f}s, not measured at it:")
        print(f"  the guarded write returns in ~{out['up']['call']['median']:.0f} ms;"
              f" first observation is ~{worst:.0f} ms.")
        if spread < 0.05 * worst:
            print("  settle times are near-constant across directions, which is a")
            print("  REPORTING GRID -- treat the true actuation as unknown within")
            print("  that interval rather than equal to it.")
        print("Add the signal age a causal controller pays on top -- on this")
        print("hardware the power sensor refreshes ~500 ms and engine metrics")
        print("are scraped, so actuation is a LOWER bound on tau, not tau.")
        print()
        print("Against a 2.53 s median idle gap, an up+down pair costs")
        print(f"~{2 * worst / 1000:.2f}s of it before any energy is saved.")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"low_mhz": low, "high_mhz": high, "poll_ms": poll_ms,
             "loaded": bool(args.load), "summary": out, "rows": rows}, indent=2))
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
