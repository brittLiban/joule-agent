"""Capacity ladder: is this GPU actually the throughput bottleneck?

Answer this before drawing any conclusion from a sweep. A clock-energy curve
measured on an under-utilised card is a property of the under-utilisation, not
of the hardware -- on the reference box every sweep ran at roughly 11% of the
card's demonstrated throughput, which quietly conditions every number those
sweeps produced.

Method. Closed-loop presets (N workers firing back-to-back) are REFUSED for
sweeps, because offered load becomes a function of server speed and a faster
clock would receive more traffic -- confounding the config change with the
traffic change it induced. For a CAPACITY measurement that property is exactly
what is wanted: N workers saturate the engine and the achieved token rate is the
capacity. So this uses them deliberately, for the one job they are valid for.

Throughput is read from the serving engine's own counters rather than inferred.
Energy is read alongside when NVML is available and omitted when it is not --
the capacity question does not need it, and losing NVML must not lose the run.

Usage:
    python tools/capacity_ladder.py [--endpoint URL] [--duration S] [--out FILE]
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request

PY = ".venv/bin/python"

# closed loop first (a capacity probe), then the open-loop presets a sweep would
# actually use, so the two are directly comparable in one table
LADDER = [("single", None), ("burst", None), ("saturated", None),
          ("bursty", None), ("bursty_heavy", None),
          ("bursty_heavy", 2.0), ("bursty_heavy", 3.0)]


def counters(endpoint):
    with urllib.request.urlopen(f"{endpoint}/metrics", timeout=10) as r:
        text = r.read().decode()
    out = {}
    for line in text.splitlines():
        for key in ("vllm:generation_tokens_total", "vllm:prompt_tokens_total"):
            if line.startswith(key):
                out[key] = out.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
    return out


def energy_j():
    """None when NVML is unusable; the capacity question does not need it."""
    try:
        import pynvml as n
        n.nvmlInit()
        return n.nvmlDeviceGetTotalEnergyConsumption(
            n.nvmlDeviceGetHandleByIndex(0)) / 1000.0
    except Exception:
        return None


def run_one(endpoint, preset, duration, rate_scale):
    cmd = [PY, "-m", "joule_agent.loadgen", "--preset", preset,
           "--seed", "1234", "--duration", str(duration)]
    if rate_scale is not None:
        cmd += ["--rate-scale", str(rate_scale)]
    c0, e0, t0 = counters(endpoint), energy_j(), time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    c1, e1, t1 = counters(endpoint), energy_j(), time.monotonic()
    if proc.returncode != 0:
        return {"preset": preset, "rate_scale": rate_scale,
                "error": proc.stderr[-300:]}
    rep, _ = json.JSONDecoder().raw_decode(proc.stdout[proc.stdout.index("{"):])
    gen = c1["vllm:generation_tokens_total"] - c0["vllm:generation_tokens_total"]
    pro = c1["vllm:prompt_tokens_total"] - c0["vllm:prompt_tokens_total"]
    el = t1 - t0
    have_e = e0 is not None and e1 is not None
    return {
        "preset": preset, "rate_scale": rate_scale, "elapsed_s": round(el, 1),
        "gen_tok_s": round(gen / el, 1), "total_tok_s": round((gen + pro) / el, 1),
        "j_per_gen_tok": round((e1 - e0) / gen, 5) if have_e and gen else None,
        "mean_w": round((e1 - e0) / el, 1) if have_e else None,
        "completed": rep.get("completed"), "failed": rep.get("failed"),
        "max_running": rep.get("max_running"), "max_waiting": rep.get("max_waiting"),
        "saturated": rep.get("achieved_saturation"), "p99_s": rep.get("latency_p99_s"),
        "closed_loop": rep.get("reproducible") is False,
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", default="http://localhost:8000")
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument("--out", default="capacity_ladder.json")
    args = p.parse_args(argv)

    rows = []
    for preset, rs in LADDER:
        r = run_one(args.endpoint, preset, args.duration, rs)
        rows.append(r)
        label = preset + (f" x{rs}" if rs else "")
        if "error" in r:
            print(f"{label:<20} ERROR {r['error'][:60]}", flush=True)
        else:
            print(f"{label:<20} {r['gen_tok_s']:>8} gen tok/s  "
                  f"running={r['max_running']} waiting={r['max_waiting']}",
                  flush=True)

    ok = [r for r in rows if "error" not in r]
    print("\n" + "=" * 94)
    print(f"{'condition':<20}{'loop':>7}{'gen tok/s':>11}{'tot tok/s':>11}"
          f"{'W':>7}{'run':>6}{'wait':>6}{'sat':>7}{'p99 s':>8}")
    for r in ok:
        label = r["preset"] + (f" x{r['rate_scale']}" if r["rate_scale"] else "")
        na = lambda v: "-" if v is None else v          # noqa: E731
        print(f"{label:<20}{'closed' if r['closed_loop'] else 'open':>7}"
              f"{r['gen_tok_s']:>11}{r['total_tok_s']:>11}"
              f"{str(na(r['mean_w'])):>7}{r['max_running']:>6}"
              f"{r['max_waiting']:>6}{str(r['saturated']):>7}{r['p99_s']:>8}")

    if ok:
        peak = max(r["gen_tok_s"] for r in ok)
        print(f"\npeak observed: {peak:.0f} generated tok/s")
        for r in ok:
            if not r["closed_loop"]:
                label = r["preset"] + (f" x{r['rate_scale']}" if r["rate_scale"] else "")
                print(f"  {label:<18} runs the card at "
                      f"{100 * r['gen_tok_s'] / peak:5.1f}% of that")
        if all(not r["saturated"] for r in ok):
            print("\nNOTE: no condition reported a non-empty queue, so peak is a "
                  "FLOOR on\ncapacity, not a ceiling -- the card was never "
                  "pushed to its knee. Any\nclock-energy result measured below "
                  "this is conditioned on that occupancy.")
    json.dump(rows, open(args.out, "w"), indent=1)
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
