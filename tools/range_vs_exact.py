"""Does a static clock RANGE beat an exact lock on the same traffic?

Every "tuned static" number in this project used an exact ``[f, f]`` lock, which
holds the clock up through idle gaps. NVML has always accepted ``[min, max]``,
which lets the device's own governor descend during idle while keeping the tuned
ceiling under load. If a range captures the idle savings, then the idle-gap
opportunity belongs to a better STATIC baseline and was never a dynamic
opportunity at all -- and the dynamic bar's denominator moves.

Measured separately on the reference card: the range mechanism DOES descend --
``[405, 1560]`` with no traffic sat at 405 MHz. That proves the mechanism, not
that it is useful: a 2.5 s idle gap is only exploitable if the governor descends
promptly and climbs back without costing tail latency. This measures that under
real traffic.

DESIGN. Legs are interleaved exact/range/exact/range so session drift cancels in
the PAIRED differences -- the same construction that made the idle-gap result
trustworthy, where absolute level drifted 2.0 W while the pairs agreed to 0.02 W.
Energy comes from the GPU counter (gap-free integration); tokens come from the
serving engine's own counters; latency comes from the load generator's report.

READ BOTH COMPARISONS. "Both configurations pass the frozen SLO" is a
feasibility test, not an equal-latency one -- a range that ramps late can look
efficient because it spent latency, not because it saved energy. So this reports
J/token AND p99 for each leg, and the interpretation rule is: a range only wins
if it is cheaper at NO WORSE p99. If it is cheaper with worse p99, the honest
comparison is against an exact lock at the range's achieved p99, which this
script does not measure -- run a sweep point there.

Requires root (clock writes). Stop nothing else on the GPU; the point is to
measure the configuration you would actually deploy.
"""

import argparse
import json
import statistics as st
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PY = ".venv/bin/python"
CONSENT = "--i-understand-this-changes-gpu-settings"


def counters(endpoint):
    with urllib.request.urlopen(f"{endpoint}/metrics", timeout=5) as r:
        text = r.read().decode()
    gen = pro = 0.0
    for line in text.splitlines():
        if line.startswith("vllm:generation_tokens_total"):
            gen += float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:prompt_tokens_total"):
            pro += float(line.rsplit(" ", 1)[1])
    return gen, pro


def energy_j():
    import pynvml as n
    n.nvmlInit()
    return n.nvmlDeviceGetTotalEnergyConsumption(
        n.nvmlDeviceGetHandleByIndex(0)) / 1000.0


def clock_now():
    import pynvml as n
    n.nvmlInit()
    return n.nvmlDeviceGetClockInfo(n.nvmlDeviceGetHandleByIndex(0),
                                    n.NVML_CLOCK_SM)


def leg(endpoint, preset, seed, duration, lo, hi, hold):
    """One measured leg under a lock. hi=None -> exact lock at lo."""
    cmd = [PY, "-m", "joule_agent.gpu_guard", "lock", "--mhz", str(lo),
           "--hold", str(hold), CONSENT]
    if hi is not None:
        cmd[6:6] = ["--max-mhz", str(hi)]
    lock = subprocess.Popen(["sudo", "-n", *cmd],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    time.sleep(5)
    if lock.poll() is not None:
        raise SystemExit(f"lock failed: {lock.stderr.read().decode()[:300]}")

    seen = [clock_now()]
    g0, p0, e0, t0 = *counters(endpoint), energy_j(), time.monotonic()
    proc = subprocess.run(
        [PY, "-m", "joule_agent.loadgen", "--preset", preset,
         "--seed", str(seed), "--duration", str(duration)],
        capture_output=True, text=True, timeout=1800)
    g1, p1, e1, t1 = *counters(endpoint), energy_j(), time.monotonic()
    seen.append(clock_now())
    lock.wait(timeout=hold + 60)

    rep, _ = json.JSONDecoder().raw_decode(proc.stdout[proc.stdout.index("{"):])
    gen, el = g1 - g0, t1 - t0
    return {
        "config": f"[{lo},{hi}]" if hi else f"[{lo},{lo}]",
        "j_per_gen_token": (e1 - e0) / gen if gen else None,
        "j_per_total_token": (e1 - e0) / (gen + p1 - p0) if gen else None,
        "mean_w": (e1 - e0) / el, "elapsed_s": el,
        "gen_tokens": gen, "tok_per_s": gen / el,
        "p99_s": rep.get("latency_p99_s"), "p50_s": rep.get("latency_p50_s"),
        "completed": rep.get("completed"), "failed": rep.get("failed"),
        "clock_samples": seen,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--endpoint", default="http://localhost:8000")
    p.add_argument("--preset", default="bursty")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--exact", type=int, required=True,
                   help="the tuned-static clock, e.g. 1560")
    p.add_argument("--range-min", type=int, default=None,
                   help="range lower bound; default = device minimum")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    lo = args.range_min
    if lo is None:
        import pynvml as n
        n.nvmlInit(); h = n.nvmlDeviceGetHandleByIndex(0)
        lo = sorted(n.nvmlDeviceGetSupportedGraphicsClocks(
            h, n.nvmlDeviceGetSupportedMemoryClocks(h)[0]))[0]
    hold = args.duration + 45

    print(f"exact [{args.exact},{args.exact}]  vs  range [{lo},{args.exact}]")
    print(f"interleaved x2 so session drift cancels in the paired difference\n")

    rows = []
    for i, (a, b) in enumerate([(args.exact, None), (lo, args.exact)] * 2, 1):
        r = leg(args.endpoint, args.preset, args.seed, args.duration,
                a, b, hold)
        rows.append(r)
        print(f"  leg {i}  {r['config']:<14}{r['j_per_gen_token']:.5f} J/tok  "
              f"{r['mean_w']:6.1f} W  p99 {r['p99_s'] * 1000:.0f} ms  "
              f"clk {r['clock_samples']}", flush=True)
        time.sleep(10)

    ex = [r for r in rows if r["config"] == f"[{args.exact},{args.exact}]"]
    rg = [r for r in rows if r["config"] == f"[{lo},{args.exact}]"]
    d1 = 100 * (rg[0]["j_per_gen_token"] - ex[0]["j_per_gen_token"]) / ex[0]["j_per_gen_token"]
    d2 = 100 * (rg[1]["j_per_gen_token"] - ex[1]["j_per_gen_token"]) / ex[1]["j_per_gen_token"]
    dj = (d1 + d2) / 2
    dp = st.mean(r["p99_s"] for r in rg) - st.mean(r["p99_s"] for r in ex)

    print(f"\n{'=' * 66}")
    print(f"paired energy delta: {d1:+.2f}%  and  {d2:+.2f}%  ->  {dj:+.2f}%")
    print(f"p99 delta:           {dp * 1000:+.0f} ms "
          f"({st.mean(r['p99_s'] for r in ex) * 1000:.0f} -> "
          f"{st.mean(r['p99_s'] for r in rg) * 1000:.0f})")
    print("=" * 66)
    if dj < -1.0 and dp <= 0.002:
        print("RANGE WINS, and not by spending latency. The tuned-static")
        print("baseline should become the range, which MOVES THE DENOMINATOR")
        print("for any dynamic claim. Re-derive the champion before Gate 2.")
    elif dj < -1.0:
        print("Range is cheaper BUT p99 is worse. Not a clean win -- compare")
        print("against an exact lock at the range's achieved p99 before")
        print("crediting it; energy bought with latency is not efficiency.")
    else:
        print("No material range advantage. The governor does not descend fast")
        print("enough to exploit these gaps, so exact-lock remains the champion")
        print("and the measured idle-gap ceiling stands as the dynamic bound.")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"exact_mhz": args.exact, "range_min_mhz": lo,
             "energy_delta_pct": dj, "p99_delta_s": dp, "legs": rows}, indent=2))
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
