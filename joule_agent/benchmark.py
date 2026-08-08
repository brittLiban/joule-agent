"""``joule benchmark`` -- the one-command entry point.

This is component 6, and it is deliberately thin. Everything it does is already
implemented and tested in :mod:`joule_agent.sweep`; this module chooses the
protocol, not the mechanics. Three decisions are load-bearing enough to state
here rather than bury:

**It calibrates, freezes, then compares.** The SLO threshold is derived ONCE
from a stock-only calibration pass and then frozen for every clock point in the
comparison. Re-deriving per configuration would move the goalposts between the
things being compared. If the operator supplies ``--slo-ms``, that absolute
threshold is used unchanged and no calibration runs -- a customer's service does
not get to become 1% slower because the machine is having a slow day.

**It reports what THIS run measured, never a product claim.** Every figure it
prints is accompanied by the workload that produced it, because on the reference
hardware load shape moved joules-per-token roughly five times harder than the
clock did. A joules-per-token number without its workload is not a result.

**It does not expose the power axis.** ``sweep.py`` keeps ``--power-limits-w``
as a research flag, but every power-axis claim this project made has been
retracted and the efficacy gate that would justify a new one has not been built.
A tool that prints a number the project has disowned is worse than a tool that
prints nothing.

The clock axis, the guard, the digest contract and the abort-on-failed-restore
behaviour all come from ``sweep.py`` unchanged. If a rule seems missing here,
look there before adding it.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

from joule_agent.loadgen import PRESETS, build_schedule
from joule_agent.sweep import (
    DEFAULT_SLO_SLACK,
    MIN_STOCK_RUNS_FOR_SLO,
    SweepError,
    SweepPoint,
    SweepRunner,
    _parse_clocks,
    build_grid,
)

#: Clocks to sweep when the operator names none. Deliberately coarse and wide:
#: the knee's location is a property of the hardware, the model and the
#: workload, so a default grid should locate it, not presume it. Refinement
#: around the knee is a second pass -- see ``--clocks``.
DEFAULT_COARSE_STEP_MHZ = 150


def _has_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0


def default_out_dir() -> Path:
    """Where results go when the operator names no directory.

    Under ``sudo``, ``Path.cwd()`` is still the invoking user's directory but
    every file written lands owned by root, which makes the NEXT unprivileged
    run unable to read its own history. Writes go to a user-owned location and
    ownership is repaired after the run (see :func:`_chown_back`).
    """
    return Path.cwd() / "joule-results" / time.strftime("%Y%m%dT%H%M%SZ",
                                                        time.gmtime())


def _chown_back(path: Path) -> None:
    """Hand results back to the invoking user after a privileged run.

    ``sudo`` sets SUDO_UID/SUDO_GID. Without this, a user who runs
    ``sudo joule benchmark`` once can never again read or write that directory
    without sudo -- which is exactly how this repo's own ``results/`` tree
    became unusable to its unprivileged runs.
    """
    uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
    if uid is None or gid is None:
        return
    try:
        for p in [path, *path.rglob("*")]:
            os.chown(p, int(uid), int(gid))
    except OSError:
        pass  # best effort: a results file we cannot chown is still readable data


def calibrate_slo(*, endpoint, model, schedule, repeats, slo_slack, gpu,
                  settle_s, warmup_runs, runner_factory=None):
    """Stock-only pass whose median p99 becomes the frozen threshold.

    Returns ``(threshold_ms, provenance)``. Runs no clock writes and therefore
    needs no root -- which is the point: an operator can calibrate without
    privilege, inspect the number, and only then decide to run the locked half.
    """
    factory = runner_factory or SweepRunner
    runner = factory(
        endpoint=endpoint, model=model, schedule=schedule,
        points=[SweepPoint()], repeats=repeats, slo_slack=slo_slack,
        settle_s=settle_s, warmup_runs=warmup_runs,
    )
    result = runner.run()
    if result.aborted:
        raise SweepError(f"calibration aborted: {result.abort_reason}")
    p99 = [r.latency_p99_s for r in result.runs
           if r.point == "stock" and r.latency_p99_s is not None]
    if not p99:
        raise SweepError("calibration produced no stock p99; cannot set an SLO")
    median = statistics.median(p99)
    return median * slo_slack * 1000.0, {
        "stock_runs": len(p99),
        "stock_p99_median_s": round(median, 5),
        "stock_p99_stdev_s": (round(statistics.stdev(p99), 5)
                              if len(p99) > 1 else None),
        "slo_slack": slo_slack,
        "result": result,
    }


def format_benchmark_report(result, *, calib=None, driver=None, model=None,
                            endpoint=None) -> str:
    """The operator-facing report.

    Every joules-per-token figure is printed adjacent to the workload that
    produced it. That is not decoration: on the reference hardware, moving from
    one open-loop preset to another changed joules per total token by ~5x, far
    more than the clock knob being tuned. A number without its workload invites
    exactly the cross-hardware comparison it cannot support.
    """
    L = []
    w = 74
    L += ["", "=" * w, "JOULE BENCHMARK", "=" * w]

    stock = next((s for s in result.summaries if s.name == "stock"), None)
    best = next((s for s in result.summaries if s.name == result.best_point), None)

    L.append(f"{'endpoint':<22}{endpoint or 'n/a'}")
    L.append(f"{'model':<22}{model or 'n/a'}")
    if driver:
        L.append(f"{'driver':<22}{driver}")
    L.append(f"{'workload':<22}{result.preset}  (seed {result.seed}, "
             f"digest {result.schedule_digest[:12]}, {result.repeats} repeats)")

    if result.slo_threshold_s is not None:
        L.append(f"{'latency budget':<22}p99 <= {result.slo_threshold_s * 1000:.1f} ms")
        L.append(f"{'  source':<22}{result.slo_threshold_source}")
    if calib:
        sd = calib.get("stock_p99_stdev_s")
        L.append(f"{'  calibration':<22}{calib['stock_runs']} stock runs, "
                 f"median p99 {calib['stock_p99_median_s'] * 1000:.1f} ms"
                 + (f", stdev {sd * 1000:.1f} ms" if sd else ""))

    L += ["", "-" * w]
    if stock is None or best is None or stock.joules_per_token_mean is None:
        L.append("No comparable result: the stock reference or the best point "
                 "is missing.")
        L += ["-" * w]
    elif best.name == "stock":
        L.append("RESULT: no clock in this grid beat stock within the latency "
                 "budget.")
        L.append(f"  stock measured {stock.joules_per_token_mean:.5f} J per "
                 f"generated token at p99 {stock.latency_p99_mean_s * 1000:.0f} ms.")
        L.append("  Either the budget is too tight for a lower clock, or the "
                 "grid does not")
        L.append("  reach one. Widen with --clocks, or relax --slo-ms if the "
                 "budget allows.")
        L += ["-" * w]
    else:
        red = 100.0 * (stock.joules_per_token_mean - best.joules_per_token_mean) \
            / stock.joules_per_token_mean
        dp99 = 100.0 * (best.latency_p99_mean_s - stock.latency_p99_mean_s) \
            / stock.latency_p99_mean_s
        dtp = 100.0 * (best.tokens_per_s_mean - stock.tokens_per_s_mean) \
            / stock.tokens_per_s_mean
        L.append(f"{'stock energy':<22}{stock.joules_per_token_mean:.5f} J "
                 f"per generated token")
        L.append(f"{'suggested SM clock':<22}{best.achieved_clock_median_mhz:.0f} MHz")
        L.append(f"{'measured energy':<22}{best.joules_per_token_mean:.5f} J "
                 f"per generated token")
        L.append("")
        L.append(f"{'THIS RUN MEASURED':<22}{red:.1f}% lower energy per token")
        L.append(f"{'  p99 latency':<22}{dp99:+.1f}%  "
                 f"({stock.latency_p99_mean_s * 1000:.0f} -> "
                 f"{best.latency_p99_mean_s * 1000:.0f} ms)")
        L.append(f"{'  throughput':<22}{dtp:+.1f}%  "
                 f"({stock.tokens_per_s_mean:.0f} -> "
                 f"{best.tokens_per_s_mean:.0f} tok/s)")
        if result.tied_points:
            L.append(f"{'  tied within noise':<22}"
                     f"{', '.join(result.tied_points)}")
        L += ["-" * w]

    L.append("")
    L.append(f"{'point':<12}{'n':>3}{'J/tok':>11}{'+/-':>10}{'tok/s':>9}"
             f"{'p99 ms':>9}{'SLO viol':>10}{'MHz':>7}")
    for s in result.summaries:
        def f(v, spec=".4f"):
            return "n/a" if v is None else format(v, spec)
        flag = "  <- best" if s.name == result.best_point else (
            "  (tied)" if s.tied_with_best else "")
        p99ms = ("n/a" if s.latency_p99_mean_s is None
                 else f"{s.latency_p99_mean_s * 1000:.0f}")
        L.append(f"{s.name:<12}{s.runs:>3}{f(s.joules_per_token_mean, '.5f'):>11}"
                 f"{f(s.joules_per_token_stdev, '.5f'):>10}"
                 f"{f(s.tokens_per_s_mean, '.0f'):>9}{p99ms:>9}"
                 f"{f(s.slo_violation_rate_mean, '.3f'):>10}"
                 f"{f(s.achieved_clock_median_mhz, '.0f'):>7}{flag}")

    L.append("")
    L.append("Every figure above was measured in THIS run, on THIS workload.")
    L.append("Energy per token depends on load shape at least as strongly as on")
    L.append("clock, so these numbers do not transfer to another workload, "
             "another")
    L.append("model, or another GPU. Re-run it there.")

    if result.stock_drift_pct is not None:
        L.append("")
        L.append(f"stock drift across rounds: {result.stock_drift_pct:+.2f}%")
    if result.aborted:
        L += ["", f"ABORTED: {result.abort_reason}"]
    if result.hardware_may_be_modified:
        L.append("THE GPU MAY STILL BE MODIFIED -- run "
                 "`sudo python -m joule_agent.gpu_guard --gpu 0 status`")
    for warn in result.warnings:
        L.append(f"WARNING: {warn}")
    L.append("=" * w)
    return "\n".join(L)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="joule benchmark",
        description="Measure tokens-per-joule on a running inference server "
                    "and find the lowest-energy GPU clock that still meets a "
                    "latency budget.",
        epilog="Locking clocks requires root. Without it, --calibrate-only "
               "still measures the stock baseline and the latency budget.",
    )
    p.add_argument("--endpoint", default="http://localhost:8000",
                   help="OpenAI-compatible endpoint (default: %(default)s)")
    p.add_argument("--model", required=True,
                   help="model name as the server reports it")
    p.add_argument("--preset", default="bursty",
                   help="open-loop workload preset; closed-loop presets are "
                        "refused because offered load would depend on server "
                        "speed (default: %(default)s)")
    p.add_argument("--list-presets", action="store_true",
                   help="print the available workload presets and exit")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--rate-scale", type=float, default=1.0,
                   help="multiply the preset's arrival rate, to benchmark at "
                        "an occupancy closer to production")
    p.add_argument("--duration", type=float, default=60.0,
                   help="seconds per run (default: %(default)s)")
    p.add_argument("--repeats", type=int, default=3,
                   help="measured runs per clock point (default: %(default)s)")
    p.add_argument("--clocks", default="",
                   help="MHz list '1500,1600' or range 'MIN:MAX:STEP'. "
                        "Default: a coarse grid across the device's supported "
                        "range, to locate the knee before refining around it.")
    p.add_argument("--slo-ms", type=float, default=None,
                   help="absolute p99 budget in ms. When given, it is used "
                        "unchanged and NO calibration runs.")
    p.add_argument("--slo-slack", type=float, default=DEFAULT_SLO_SLACK,
                   help="when deriving the budget, multiply the calibrated "
                        "stock p99 by this (default: %(default)s)")
    p.add_argument("--calibrate-repeats", type=int, default=MIN_STOCK_RUNS_FOR_SLO,
                   help="stock runs in the calibration pass (default: "
                        "%(default)s). Below this the threshold carries enough "
                        "sampling error to flip points near the knee.")
    p.add_argument("--calibrate-only", action="store_true",
                   help="measure the stock baseline and the latency budget, "
                        "then stop. Needs no root.")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", default=None,
                   help="output directory (default: ./joule-results/<utc>)")
    p.add_argument("--settle-s", type=float, default=5.0)
    p.add_argument("--warmup-runs", type=int, default=1)
    p.add_argument("--i-understand-this-changes-gpu-settings",
                   dest="consent", action="store_true",
                   help="required to lock clocks")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.list_presets:
        for name, preset in sorted(PRESETS.items()):
            mark = "" if preset.reproducible else "   [closed loop -- refused]"
            print(f"{name:<15}{preset.purpose}{mark}")
        return 0

    if args.preset not in PRESETS:
        print(f"REFUSED: unknown preset {args.preset!r}. Available: "
              f"{', '.join(sorted(PRESETS))}", file=sys.stderr)
        return 2
    if not PRESETS[args.preset].reproducible:
        print(f"REFUSED: preset {args.preset!r} is closed-loop. Offered load "
              "would depend on server response time, so a faster clock would "
              "receive more traffic and the comparison would confound the "
              "config change with the traffic change it caused. Open-loop "
              "presets: "
              + ", ".join(n for n, p in PRESETS.items() if p.reproducible),
              file=sys.stderr)
        return 2

    # Refuse on the ARGUMENTS before doing any work. Calibration is ten minutes
    # of real traffic; discovering afterwards that the caller was never going to
    # be allowed to write is the same defect gpu_guard fixed by validating
    # arguments before the privilege check.
    if not args.calibrate_only and not args.consent:
        print("REFUSED: sweeping clocks changes GPU settings. Pass "
              "--i-understand-this-changes-gpu-settings, or use "
              "--calibrate-only to stop after the baseline.", file=sys.stderr)
        return 2
    if not args.calibrate_only and not _has_root():
        print("REFUSED: locking clocks requires root. Re-run under sudo, or "
              "use --calibrate-only, which measures the stock baseline and the "
              "latency budget with no privilege.", file=sys.stderr)
        return 2

    out_dir = Path(args.out) if args.out else default_out_dir()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"REFUSED: cannot create {out_dir}: {exc}", file=sys.stderr)
        return 2

    schedule = build_schedule(args.preset, seed=args.seed,
                              duration_s=args.duration,
                              rate_scale=args.rate_scale)

    # --- calibrate ------------------------------------------------------
    slo_ms, calib = args.slo_ms, None
    if slo_ms is None:
        print(f"[1/2] calibrating latency budget: {args.calibrate_repeats} "
              f"stock runs, no clock changes, no root needed ...")
        try:
            slo_ms, calib = calibrate_slo(
                endpoint=args.endpoint, model=args.model, schedule=schedule,
                repeats=args.calibrate_repeats, slo_slack=args.slo_slack,
                gpu=args.gpu, settle_s=args.settle_s,
                warmup_runs=args.warmup_runs)
        except SweepError as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return 2
        print(f"      stock p99 median {calib['stock_p99_median_s'] * 1000:.1f} ms"
              f"  ->  budget {slo_ms:.1f} ms (x{args.slo_slack})")
    else:
        print(f"[1/2] using the supplied latency budget: {slo_ms:.1f} ms "
              "(no calibration)")

    if args.calibrate_only:
        payload = {"slo_ms": slo_ms,
                   "calibration": {k: v for k, v in (calib or {}).items()
                                   if k != "result"}}
        (out_dir / "calibration.json").write_text(json.dumps(payload, indent=2))
        _chown_back(out_dir)
        print(f"\nlatency budget: {slo_ms:.1f} ms")
        print(f"written to {out_dir}")
        print("\nRe-run without --calibrate-only (as root) to sweep clocks, "
              f"passing --slo-ms {slo_ms:.1f} to freeze this budget.")
        return 0

    # --- sweep ----------------------------------------------------------
    from joule_agent import sweep as _sweep

    try:
        clocks = (_parse_clocks(args.clocks) if args.clocks
                  else _default_clock_grid(args.gpu))
    except (SweepError, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    print(f"[2/2] sweeping {len(clocks)} clocks x {args.repeats} repeats "
          f"at a frozen budget of {slo_ms:.1f} ms ...")

    argv_rest = [
        "--endpoint", args.endpoint, "--model", args.model,
        "--preset", args.preset, "--seed", str(args.seed),
        "--rate-scale", str(args.rate_scale),
        "--duration", str(args.duration), "--repeats", str(args.repeats),
        "--clocks", ",".join(str(c) for c in clocks),
        "--slo-ms", f"{slo_ms:.4f}", "--gpu", str(args.gpu),
        "--out", str(out_dir), "--settle-s", str(args.settle_s),
        "--warmup-runs", str(args.warmup_runs),
        "--i-understand-this-changes-gpu-settings",
    ]
    rc = _sweep.main(argv_rest)
    _chown_back(out_dir)

    sweep_json = out_dir / "sweep.json"
    if rc == 0 and sweep_json.exists():
        print(f"\nfull report: {sweep_json}")
    return rc


def _default_clock_grid(gpu: int) -> list:
    """A coarse grid across the device's own supported range.

    Deliberately not a hard-coded list: supported clocks, and the granularity
    the driver rounds requests to, are per-device and per-driver properties.
    Asking the device is the only way that stays true on hardware this has
    never run on.
    """
    try:
        import pynvml as n
        n.nvmlInit()
        h = n.nvmlDeviceGetHandleByIndex(gpu)
        mem = n.nvmlDeviceGetSupportedMemoryClocks(h)[0]
        supported = sorted(n.nvmlDeviceGetSupportedGraphicsClocks(h, mem))
    except Exception as exc:
        raise SweepError(
            f"could not read supported clocks from GPU {gpu} ({exc}). Pass "
            "--clocks explicitly, e.g. --clocks 1200:1900:150."
        ) from exc
    lo, hi = supported[0], supported[-1]
    grid, c = [], lo
    while c <= hi:
        nearest = min(supported, key=lambda s: abs(s - c))
        if nearest not in grid:
            grid.append(nearest)
        c += DEFAULT_COARSE_STEP_MHZ
    return grid


if __name__ == "__main__":
    raise SystemExit(main())
