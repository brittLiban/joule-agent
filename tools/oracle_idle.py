"""oracle_idle -- a perfect-foresight, physically realizable idle-gap controller.

WHAT THIS IS AND IS NOT. This exercises exactly one mechanism: dropping the
clock during traffic gaps and raising it before traffic returns. It does NOT
exercise state-dependent control during active work. A result here may not be
added to the separately-estimated active-work figure and called a combined
oracle -- run ``oracle_combined`` for that, and only once an ``oracle_active``
policy exists that is realizable rather than per-request.

WHY IT IS AN ORACLE. The schedule is materialized before the run, so this knows
every arrival time in advance. It upclocks BEFORE a burst arrives, which no
causal controller can do -- a causal policy must first observe traffic that has
already queued. That head start is the whole advantage foresight buys, and it is
swept as ``--lead``.

WHY IT IS STILL PHYSICALLY REALIZABLE, unlike the per-request oracle in
``oracle_mix.py``. One clock applies to the whole GPU, so this sets ONE clock at
a time on a timeline. Transitions are real writes through the guard, the queue
coupling is real, the energy is real, and a request spanning a transition
experiences whatever actually happens to it.

READ THE VERDICT AS AN UPPER BOUND. Perfect foresight, hand-tuned timings, and
no prediction error. If this cannot clear the bar, no causal controller built on
this mechanism can.

Requires root. Interleaved against exact-lock legs so session drift cancels in
the paired difference, with the same cold-first-pair guard as
``range_vs_exact.py`` -- a 60 s warm-up was measured to be insufficient on this
hardware, so warm-up alone is not trusted.
"""

import argparse
import hashlib
import json
import platform
import statistics as st
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from joule_agent.loadgen import build_schedule

PY = ".venv/bin/python"
CONSENT = "--i-understand-this-changes-gpu-settings"


def burst_windows(schedule, quiet_s):
    """Contiguous arrival clusters, split wherever the schedule is quiet.

    Derived purely from arrival times -- this is the foresight. `quiet_s` is the
    gap in the SCHEDULE that counts as a burst boundary; it is not a measurement
    of engine drain, which the `drain` parameter covers separately.
    """
    ts = sorted(r.arrival_s for r in schedule.requests)
    if not ts:
        return []
    out, start, prev = [], ts[0], ts[0]
    for t in ts[1:]:
        if t - prev > quiet_s:
            out.append((start, prev))
            start = t
        prev = t
    out.append((start, prev))
    return out


def plan_transitions(windows, low, high, lead, drain, horizon):
    """[(t_seconds, clock)] -- the oracle's timeline, computed before the run."""
    events = []
    for a, b in windows:
        events.append((max(0.0, a - lead), high))
        events.append((b + drain, low))
    events.sort()
    merged = []
    for t, c in events:
        if merged and merged[-1][1] == c:
            continue
        if t > horizon:
            break
        merged.append((t, c))
    return merged


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


def energy_j(gpu=0):
    import pynvml as n
    n.nvmlInit()
    return n.nvmlDeviceGetTotalEnergyConsumption(
        n.nvmlDeviceGetHandleByIndex(gpu)) / 1000.0


def run_leg(args, plan, label):
    """One measured leg. plan=None -> exact lock at args.high for the whole run.

    The guard is thread-affine, so every clock write happens on THIS thread and
    the load generator runs as a subprocess. That is also why the transition
    loop cannot simply sleep: it must wake, write, and keep its own clock.
    """
    from joule_agent.gpu_guard import GpuGuard, NvmlGpuController

    guard = GpuGuard(NvmlGpuController(args.gpu), consent=True)
    applied = []
    with guard:
        guard.lock_clocks(args.high)
        time.sleep(2.0)

        g0, p0 = counters(args.endpoint)
        e0 = energy_j(args.gpu)
        t0 = time.monotonic()

        proc = subprocess.Popen(
            [PY, "-m", "joule_agent.loadgen", "--endpoint", args.endpoint,
             "--preset", args.preset, "--seed", str(args.seed),
             "--duration", str(args.duration)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if plan:
            for when, clock in plan:
                delay = (t0 + when) - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                if proc.poll() is not None:
                    break
                guard.lock_clocks(clock)
                applied.append((round(time.monotonic() - t0, 3), clock))
        out, err = proc.communicate(timeout=1800)

        e1 = energy_j(args.gpu)
        g1, p1 = counters(args.endpoint)
        t1 = time.monotonic()
        guard.lock_clocks(args.high)

    if proc.returncode != 0:
        raise SystemExit(f"loadgen failed: {err[-400:]}")
    rep, _ = json.JSONDecoder().raw_decode(out[out.index("{"):])
    gen, el = g1 - g0, t1 - t0
    return {
        "label": label, "j_per_gen_token": (e1 - e0) / gen if gen else None,
        "mean_w": (e1 - e0) / el, "elapsed_s": el, "gen_tokens": gen,
        "p99_s": rep.get("latency_p99_s"), "p50_s": rep.get("latency_p50_s"),
        "completed": rep.get("completed"), "failed": rep.get("failed"),
        "transitions": len(applied), "applied": applied[:40],
    }


def provenance(args, low, schedule):
    """Everything needed to know what produced a number, recorded with it.

    An artifact without this cannot be audited later: the analysis code changes,
    the driver changes, the session changes, and a bare JSON of results silently
    becomes unattributable. `sweep.py` has recorded this since component 4 and
    these tools did not.
    """
    def sha(path):
        try:
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
        except OSError:
            return None

    def sh(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=10).stdout.strip() or None
        except Exception:
            return None

    drv = uuid = None
    try:
        import pynvml as n
        n.nvmlInit()
        h = n.nvmlDeviceGetHandleByIndex(args.gpu)
        drv = n.nvmlSystemGetDriverVersion()
        drv = drv.decode() if isinstance(drv, bytes) else drv
        uuid = n.nvmlDeviceGetUUID(h)
        uuid = uuid.decode() if isinstance(uuid, bytes) else uuid
    except Exception:
        pass

    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "tool": "tools/oracle_idle.py",
        "tool_sha256": sha("tools/oracle_idle.py"),
        "gpu_guard_sha256": sha("joule_agent/gpu_guard.py"),
        "loadgen_sha256": sha("joule_agent/loadgen.py"),
        "git_sha": sh(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(sh(["git", "status", "--porcelain"])),
        "argv": sys.argv,
        "driver_version": drv,
        "device": {"index": args.gpu, "uuid": uuid},
        "endpoint": args.endpoint,
        "model_reported_by_server": sh(
            ["curl", "-s", "-m", "5", f"{args.endpoint}/v1/models"])[:400]
        if sh(["curl", "-s", "-m", "5", f"{args.endpoint}/v1/models"]) else None,
        "schedule_digest": schedule.digest(),
        "preset": args.preset, "seed": args.seed,
        "duration_s": args.duration, "high_mhz": args.high, "low_mhz": low,
        "lead_s": args.lead, "drain_s": args.drain,
        "exact_clocks": args.exact_clocks, "warmup_s": args.warmup,
        "python": platform.python_version(), "host": platform.node(),
    }


def analyse(rows):
    """Pure analysis of measured legs. No I/O, so it can be tested.

    Split out of ``main`` because it was not: a use-before-assignment of
    ``ladder_mode`` shipped in a version that had been exercised only on a
    non-ladder run, and nothing could have caught it without a GPU. Analysis
    that decides a published claim has to be reachable from a unit test.

    Returns a dict; ``iso`` is the load-bearing field in ladder mode.
    """
    ex = [r for r in rows if r["label"].startswith("exact")]
    orc = [r for r in rows if r["label"] == "oracle"]
    n = min(len(ex), len(orc))
    deltas = [100 * (orc[i]["j_per_gen_token"] - ex[i]["j_per_gen_token"])
              / ex[i]["j_per_gen_token"] for i in range(n)]

    # A ladder means the static legs are DIFFERENT configurations on purpose.
    ladder_mode = len({r["label"] for r in ex}) > 1

    # The cold-first-pair guard uses repeated exact legs as their own control,
    # so it only means anything when they are the same configuration. In ladder
    # mode it fires on the ladder itself.
    excluded = None
    if not ladder_mode and len(ex) >= 3:
        later = st.median([r["j_per_gen_token"] for r in ex[1:]])
        drift = 100 * (ex[0]["j_per_gen_token"] - later) / later
        if abs(drift) > 1.0:
            excluded, deltas = drift, deltas[1:]

    # Leg-adjacent deltas also stop meaning anything in ladder mode: each pair
    # would compare the oracle against a different static clock.
    dj = st.mean(deltas) if deltas and not ladder_mode else float("nan")
    dp = (st.mean(r["p99_s"] for r in orc) - st.mean(r["p99_s"] for r in ex)
          if ex and orc else float("nan"))

    # ISO-p99: what does STATIC cost at the latency the oracle actually
    # delivered? "Both pass the SLO" credits a policy for energy it bought with
    # latency. Compare EACH oracle leg at ITS OWN p99 -- averaging p99 first
    # lets one long leg drag the comparison to a latency no leg ran at.
    # Interpolate only; never extrapolate, which would invent the comparator.
    iso, per_leg = None, []
    if ladder_mode and ex and orc:
        pts = sorted((r["p99_s"], r["j_per_gen_token"]) for r in ex)
        for r in orc:
            po, jo = r["p99_s"], r["j_per_gen_token"]
            if po < pts[0][0] or po > pts[-1][0]:
                per_leg.append({"p99_s": po, "oracle_j": jo,
                                "out_of_range": True})
                continue
            for a, b in zip(pts, pts[1:]):
                if a[0] <= po <= b[0]:
                    f = (po - a[0]) / (b[0] - a[0]) if b[0] > a[0] else 0.0
                    js = a[1] + f * (b[1] - a[1])
                    per_leg.append({"p99_s": po, "oracle_j": jo, "static_j": js,
                                    "delta_pct": 100 * (jo - js) / js})
                    break
        ok = [x for x in per_leg if "delta_pct" in x]
        iso = {"per_leg": per_leg, "n": len(ok),
               "delta_pct": st.mean(x["delta_pct"] for x in ok) if ok else None,
               "curve_p99_s": [x[0] for x in pts],
               "all_out_of_range": not ok}
    return {"ex": ex, "orc": orc, "deltas": deltas, "excluded": excluded,
            "dj": dj, "dp": dp, "iso": iso, "ladder_mode": ladder_mode}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--endpoint", default="http://localhost:8000")
    p.add_argument("--preset", default="bursty")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--high", type=int, required=True,
                   help="the tuned-static clock; also the baseline")
    p.add_argument("--low", type=int, default=None,
                   help="gap clock (default: device minimum)")
    p.add_argument("--lead", type=float, default=0.3,
                   help="seconds to upclock BEFORE a burst. This is what "
                        "foresight buys and no causal policy has.")
    p.add_argument("--drain", type=float, default=0.5,
                   help="seconds after the last arrival of a burst before "
                        "dropping, to let the engine empty")
    p.add_argument("--quiet-s", type=float, default=1.0,
                   help="schedule gap that counts as a burst boundary")
    p.add_argument("--exact-clocks", default=None,
                   help="comma-separated clocks for the STATIC legs, one per "
                        "pair. Each oracle leg is paired with the static leg it "
                        "immediately followed, so drift cancels locally AND the "
                        "run yields an in-session static curve. Needed because "
                        "the oracle changes p99: 'both pass the SLO' is a "
                        "feasibility test, and the project's question is at "
                        "EQUAL p99. Cross-session curves cannot substitute -- "
                        "exact-1560 measured 6.5% apart between two sessions on "
                        "the same driver. Default: --high repeated.")
    p.add_argument("--pairs", type=int, default=3)
    p.add_argument("--warmup", type=float, default=90.0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    low = args.low
    if low is None:
        import pynvml as n
        n.nvmlInit(); h = n.nvmlDeviceGetHandleByIndex(args.gpu)
        low = sorted(n.nvmlDeviceGetSupportedGraphicsClocks(
            h, n.nvmlDeviceGetSupportedMemoryClocks(h)[0]))[0]

    sched = build_schedule(args.preset, seed=args.seed, duration_s=args.duration)
    windows = burst_windows(sched, args.quiet_s)
    plan = plan_transitions(windows, low, args.high, args.lead, args.drain,
                            args.duration + 5)

    print(f"oracle_idle -- exact [{args.high}] vs schedule-driven "
          f"[{low} in gaps, {args.high} in bursts]")
    print(f"  schedule digest {sched.digest()}  {len(sched.requests)} requests")
    print(f"  {len(windows)} bursts, {len(plan)} transitions, "
          f"lead {args.lead}s drain {args.drain}s")
    print(f"  IDLE-GAP MECHANISM ONLY -- not a combined oracle\n")

    if args.warmup > 0:
        print(f"  warming ({args.warmup:.0f}s, discarded) ...", flush=True)
        subprocess.run([PY, "-m", "joule_agent.loadgen",
                        "--endpoint", args.endpoint, "--preset", args.preset,
                        "--seed", str(args.seed), "--duration", str(args.warmup)],
                       capture_output=True, text=True, timeout=1800)
        time.sleep(10)

    if args.exact_clocks:
        ladder = [int(x) for x in args.exact_clocks.split(",")]
        args.pairs = len(ladder)
    else:
        ladder = [args.high] * args.pairs

    rows = []
    for i in range(args.pairs):
        for label, pl in ((f"exact{ladder[i]}", None), ("oracle", plan)):
            if pl is None:
                args.high, prev = ladder[i], args.high
            r = run_leg(args, pl, label)
            if pl is None:
                args.high = prev
            rows.append(r)
            print(f"  {label:<8}{r['j_per_gen_token']:.5f} J/tok  "
                  f"{r['mean_w']:5.1f} W  p99 {r['p99_s'] * 1000:.0f} ms  "
                  f"{r['transitions']} transitions", flush=True)
            time.sleep(10)

    a = analyse(rows)
    ex, orc = a["ex"], a["orc"]
    deltas, excluded, dj, dp, iso = (a["deltas"], a["excluded"], a["dj"],
                                     a["dp"], a["iso"])
    ladder_mode = a["ladder_mode"]

    print(f"\n{'=' * 68}")
    print("paired energy delta: " + "  ".join(f"{d:+.2f}%" for d in deltas)
          + ("  ->  suppressed (ladder mode: each pair uses a different static "
             "clock)" if dj != dj else f"  ->  mean {dj:+.2f}%"))
    if excluded is not None:
        print(f"  pair 1 excluded: exact leg ran {excluded:+.1f}% vs later ones")
    print(f"p99 delta: {dp * 1000:+.0f} ms")
    print("=" * 68)

    if iso and not iso["all_out_of_range"]:
        print("ISO-p99, each oracle leg against the in-session static curve:")
        for x in iso["per_leg"]:
            if "delta_pct" in x:
                print(f"  p99 {x['p99_s'] * 1000:>4.0f} ms  oracle "
                      f"{x['oracle_j']:.5f}  static {x['static_j']:.5f}  "
                      f"{x['delta_pct']:+.2f}%")
            else:
                print(f"  p99 {x['p99_s'] * 1000:>4.0f} ms  outside the curve "
                      f"{[round(v * 1000) for v in iso['curve_p99_s']]} "
                      f"-- not comparable, widen --exact-clocks")
        print(f"  mean over {iso['n']} comparable leg(s): "
              f"{iso['delta_pct']:+.2f}%")
        print("=" * 68)
        if iso["delta_pct"] >= 0:
            print("THE ORACLE LOSES TO A STATIC LOCK AT THE SAME LATENCY.")
            print("Its energy saving was bought with p99, and simply lowering the")
            print("clock buys more of it. Perfect foresight on the idle mechanism")
            print("is not merely short of the bar -- it is dominated by the")
            print("baseline it was meant to beat.")
        elif iso["delta_pct"] > -10.0:
            print(f"Beats static at equal p99 by {-iso['delta_pct']:.2f}%, short of "
                  "the 10% bar.")
        else:
            print("CLEARS the bar at equal p99. Build the causal controller.")
    elif iso and iso["all_out_of_range"]:
        print("NO COMPARABLE LEG. Every oracle leg fell outside the static")
        print("curve, so there is nothing to compare against without inventing")
        print("a comparator. Widen --exact-clocks.")
    elif len(deltas) > 1 and max(deltas) - min(deltas) > abs(dj):
        print("NO VERDICT -- surviving pairs disagree by more than the effect.")
    elif dp > 0.002:
        print("The oracle spent latency. Compare against a static point at the")
        print("oracle's achieved p99 before crediting any energy difference.")
    elif dj <= -10.0:
        print("CLEARS the 10% bar on the idle mechanism ALONE. Build the causal")
        print("controller -- the oracle-to-causal penalty here was measured at")
        print("0.3-1.0 points, so most of this should survive.")
    else:
        print(f"DOES NOT CLEAR. {-dj:.2f}% against a 10% bar, with perfect")
        print("foresight and zero prediction error. No causal controller built")
        print("on the idle-gap mechanism can do better than this.")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"provenance": provenance(args, low, sched),
             "high": args.high, "low": low, "lead": args.lead,
             "drain": args.drain, "digest": sched.digest(),
             "bursts": len(windows), "plan": plan, "deltas": deltas,
             "ladder_mode": ladder_mode, "excluded_pair_pct": excluded,
             "mean_delta_pct": None if dj != dj else dj,
             "p99_delta_s": dp, "iso_p99": iso, "legs": rows}, indent=2))
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
