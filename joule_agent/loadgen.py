"""Load generator: drive reproducible traffic at a serving endpoint.

Component 3. Exists to feed the static sweep and the dynamic controller with
traffic that is *identical* across configurations, so a difference in measured
joules/token is attributable to the configuration and not to the traffic.

Open loop vs closed loop
------------------------
This is the load-shape decision the sweep depends on, so it is enforced in the
type system rather than left to discipline.

**Open loop** (:class:`ArrivalProcess`): every arrival time, prompt and output
length is drawn from a seeded RNG and materialised into a :class:`Schedule`
*before the run starts*. Requests fire on that schedule regardless of how the
server is behaving. Two runs with the same seed send byte-identical requests at
identical offsets. This is the only mode valid for sweeps.

**Closed loop** (:class:`ClosedLoop`): N workers, each firing as soon as its
previous request returns. Offered load is a *function of server response time*.
Speed the server up and it receives more traffic; slow it down and it receives
less. Comparing two clock configurations under closed-loop traffic confounds
the configuration change with the traffic change it induces. Closed loop is
kept because it is the cheapest way to pin an engine at a target concurrency
(it is what captured the fixtures in tests/fixtures/), but :attr:`Preset.
reproducible` is False for those presets and the sweep runner should refuse
them.

Intended vs achieved
--------------------
A preset named "saturated" is an intention. Whether the queue actually built is
an outcome, and the two come apart routinely -- if the client cannot keep up,
or the engine is faster than expected, "saturated" traffic produces an empty
queue. Every run therefore polls /metrics through
:mod:`joule_agent.workload` and reports what actually happened:
:attr:`RunReport.time_at_nonzero_waiting_s`, the observed queue maxima, and
schedule adherence (how late arrivals actually fired). A run that intended
saturation and did not achieve it is a run whose label is wrong, and the report
says so rather than letting the preset name stand as evidence.

Variation is mandatory, not optional
------------------------------------
The thesis under test is that phase-aware dynamic control beats a well-tuned
static clock. Dynamic control can only win where there is variation to exploit.
A harness that offers only steady-state traffic structurally favours static and
would bias the kill test toward a "no" verdict that is an artifact of the
harness rather than the physics. Presets therefore include Poisson arrivals and
an on/off burst cycle with genuine idle gaps, and :attr:`RunReport.
offered_load_cv` reports the coefficient of variation actually delivered, so a
"static wins" result can be checked against whether the traffic ever varied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import threading
import time
import urllib.request
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from joule_agent.workload import read_snapshot

__all__ = [
    "ScheduledRequest",
    "Schedule",
    "ArrivalProcess",
    "ClosedLoop",
    "Preset",
    "PRESETS",
    "build_schedule",
    "RunReport",
    "LoadGenerator",
]

# Deterministic prompt filler. Fixed vocabulary so a given seed produces
# byte-identical prompt text on any machine and any Python build.
_WORDS = (
    "gpu kernel tensor memory bandwidth latency throughput scheduler cache "
    "warp block thread register occupancy pipeline clock power energy joule "
    "inference decode prefill batch token attention matrix multiply reduce"
).split()


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScheduledRequest:
    index: int
    arrival_s: float
    prompt: str
    max_tokens: int


@dataclass
class Schedule:
    """A fully materialised, replayable traffic plan."""

    requests: tuple
    seed: int
    preset: str
    duration_s: float
    reproducible: bool
    params: dict = field(default_factory=dict)

    def digest(self) -> str:
        """Content hash. Two runs of a sweep must show the same digest."""
        h = hashlib.sha256()
        for r in self.requests:
            h.update(f"{r.index}|{r.arrival_s:.9f}|{r.max_tokens}|".encode())
            h.update(r.prompt.encode())
        return h.hexdigest()[:16]

    def offered_rate_cv(self, bucket_s: float = 1.0) -> float | None:
        """Coefficient of variation of arrivals per bucket.

        0.0 means perfectly steady traffic -- a harness offering only that
        cannot demonstrate anything about adapting to variation.
        """
        if len(self.requests) < 2:
            return None
        buckets: dict[int, int] = {}
        n = max(1, int(self.duration_s / bucket_s))
        for i in range(n):
            buckets[i] = 0
        for r in self.requests:
            b = min(n - 1, int(r.arrival_s / bucket_s))
            buckets[b] = buckets.get(b, 0) + 1
        counts = list(buckets.values())
        mean = statistics.mean(counts)
        if mean == 0:
            return None
        return statistics.pstdev(counts) / mean


# ---------------------------------------------------------------------------
# Arrival processes (open loop)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArrivalProcess:
    """Open-loop arrival generator. Deterministic given a seed."""

    kind: str
    rate_per_s: float = 1.0
    on_s: float = 0.0
    off_s: float = 0.0

    def arrivals(self, rng: random.Random, duration_s: float) -> list:
        if self.kind == "constant":
            step = 1.0 / self.rate_per_s
            out, t = [], 0.0
            while t < duration_s:
                out.append(t)
                t += step
            return out

        if self.kind == "poisson":
            out, t = [], 0.0
            while True:
                # Exponential inter-arrival: the memoryless process real
                # request traffic approximates.
                t += rng.expovariate(self.rate_per_s)
                if t >= duration_s:
                    return out
                out.append(t)

        if self.kind == "onoff":
            # Poisson bursts separated by genuine idle gaps. The engine drains
            # during off periods, which is where a dynamic controller has
            # something to do that a static clock cannot.
            out, t, cycle = [], 0.0, self.on_s + self.off_s
            while t < duration_s:
                phase = t % cycle if cycle > 0 else 0.0
                if phase < self.on_s:
                    t += rng.expovariate(self.rate_per_s)
                    if t < duration_s and (t % cycle) < self.on_s:
                        out.append(t)
                else:
                    t += (cycle - phase)
            return out

        raise ValueError(f"unknown arrival kind {self.kind!r}")


@dataclass(frozen=True)
class ClosedLoop:
    """N workers firing back-to-back. Offered load depends on the server."""

    concurrency: int


@dataclass(frozen=True)
class Preset:
    name: str
    driver: object  # ArrivalProcess | ClosedLoop
    prompt_words: tuple  # (min, max)
    max_tokens: tuple  # (min, max)
    reproducible: bool
    purpose: str


PRESETS: dict = {
    p.name: p
    for p in (
        # --- open loop: valid for sweeps ---
        Preset(
            "steady",
            ArrivalProcess("constant", rate_per_s=4.0),
            (20, 20),
            (64, 64),
            True,
            "fixed rate, fixed sizes. Baseline with zero variation; a dynamic "
            "controller should NOT beat static here.",
        ),
        Preset(
            "poisson",
            ArrivalProcess("poisson", rate_per_s=4.0),
            (10, 60),
            (32, 160),
            True,
            "memoryless arrivals with varied sizes. Realistic steady traffic.",
        ),
        Preset(
            "bursty",
            ArrivalProcess("onoff", rate_per_s=12.0, on_s=3.0, off_s=3.0),
            (10, 60),
            (32, 160),
            True,
            "Poisson bursts with real idle gaps. THE preset the dynamic-vs-"
            "static question is about -- if dynamic control cannot win here it "
            "cannot win anywhere.",
        ),
        Preset(
            "bursty_heavy",
            ArrivalProcess("onoff", rate_per_s=40.0, on_s=2.0, off_s=4.0),
            (40, 200),
            (64, 256),
            True,
            "bursts that overshoot engine capacity, so the queue builds and "
            "drains within each cycle.",
        ),
        # --- closed loop: reproduces fixture conditions, NOT sweep-valid ---
        Preset(
            "single",
            ClosedLoop(1),
            (20, 20),
            (128, 128),
            False,
            "one request at a time. Matches the `single` capture fixture.",
        ),
        Preset(
            "burst",
            ClosedLoop(16),
            (20, 20),
            (128, 128),
            False,
            "16 concurrent. Matches the `burst` capture fixture.",
        ),
        Preset(
            "saturated",
            ClosedLoop(32),
            (20, 20),
            (128, 128),
            False,
            "32 concurrent; queues only if the engine's batch size binds. "
            "Matches the `saturated` capture fixture (--max-num-seqs 4).",
        ),
    )
}


def build_schedule(
    preset_name: str, *, seed: int, duration_s: float, rate_scale: float = 1.0
) -> Schedule:
    """Materialise a preset into a concrete, replayable :class:`Schedule`.

    Everything random is drawn here, from one seeded RNG, before any request is
    sent. Same seed and preset -> same :meth:`Schedule.digest`.
    """
    preset = PRESETS[preset_name]
    rng = random.Random(seed)

    if isinstance(preset.driver, ClosedLoop):
        # Closed loop has no arrival schedule by definition; sizes are still
        # drawn deterministically so at least the payloads are identical.
        n = preset.driver.concurrency
        reqs = tuple(
            ScheduledRequest(i, 0.0, _prompt(rng, preset.prompt_words), _tok(rng, preset.max_tokens))
            for i in range(n)
        )
        return Schedule(
            reqs,
            seed,
            preset_name,
            duration_s,
            reproducible=False,
            params={"mode": "closed_loop", "concurrency": n},
        )

    proc = preset.driver
    if rate_scale != 1.0:
        proc = ArrivalProcess(proc.kind, proc.rate_per_s * rate_scale, proc.on_s, proc.off_s)

    times = proc.arrivals(rng, duration_s)
    reqs = tuple(
        ScheduledRequest(i, t, _prompt(rng, preset.prompt_words), _tok(rng, preset.max_tokens))
        for i, t in enumerate(times)
    )
    return Schedule(
        reqs,
        seed,
        preset_name,
        duration_s,
        reproducible=True,
        params={
            "mode": "open_loop",
            "arrival_kind": proc.kind,
            "rate_per_s": proc.rate_per_s,
            "on_s": proc.on_s,
            "off_s": proc.off_s,
        },
    )


def _prompt(rng: random.Random, words: tuple) -> str:
    n = rng.randint(*words)
    return " ".join(rng.choice(_WORDS) for _ in range(n))


def _tok(rng: random.Random, rng_range: tuple) -> int:
    return rng.randint(*rng_range)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


@dataclass
class RunReport:
    """What the run actually did -- never what the preset name claims."""

    preset: str = ""
    seed: int = 0
    schedule_digest: str = ""
    reproducible: bool = True
    duration_s: float = 0.0

    scheduled: int = 0
    sent: int = 0
    completed: int = 0
    failed: int = 0

    #: How late arrivals actually fired. Large values mean the CLIENT was the
    #: bottleneck and the intended traffic shape was not delivered.
    arrival_lateness_mean_s: float | None = None
    arrival_lateness_max_s: float | None = None
    schedule_achieved: bool = True

    #: Intended-vs-achieved saturation.
    time_at_nonzero_waiting_s: float = 0.0
    fraction_time_waiting: float = 0.0
    max_waiting: float = 0.0
    max_running: float = 0.0
    achieved_saturation: bool = False

    #: Was there variation to exploit at all?
    offered_load_cv: float | None = None
    observed_running_cv: float | None = None

    latency_p50_s: float | None = None
    latency_p95_s: float | None = None
    latency_p99_s: float | None = None

    metrics_samples: int = 0
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class LoadGenerator:
    """Replays a :class:`Schedule` and measures what actually happened."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        schedule: Schedule,
        *,
        poll_interval_s: float = 0.25,
        request_timeout_s: float = 300.0,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.schedule = schedule
        self.poll_interval_s = poll_interval_s
        self.request_timeout_s = request_timeout_s

        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latencies: list = []
        self._lateness: list = []
        self._completed = 0
        self._failed = 0
        self._sent = 0
        self._samples: list = []

    # -- request path ----------------------------------------------------

    def _fire(self, req: ScheduledRequest) -> None:
        body = json.dumps(
            {
                "model": self.model,
                "prompt": req.prompt,
                "max_tokens": req.max_tokens,
                "temperature": 0.0,
            }
        ).encode()
        r = urllib.request.Request(
            f"{self.endpoint}/v1/completions",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(r, timeout=self.request_timeout_s) as resp:
                resp.read()
            dt = time.monotonic() - t0
            with self._lock:
                self._latencies.append(dt)
                self._completed += 1
        except Exception:
            with self._lock:
                self._failed += 1

    # -- metrics sampling ------------------------------------------------

    def _poll_metrics(self) -> None:
        while not self._stop.is_set():
            t = time.monotonic()
            try:
                with urllib.request.urlopen(
                    f"{self.endpoint}/metrics", timeout=5.0
                ) as resp:
                    snap = read_snapshot(resp.read())
                with self._lock:
                    self._samples.append(
                        (
                            t,
                            snap.num_requests_running,
                            snap.num_requests_waiting,
                        )
                    )
            except Exception:
                pass
            self._stop.wait(self.poll_interval_s)

    # -- run -------------------------------------------------------------

    def run(self) -> RunReport:
        rep = RunReport(
            preset=self.schedule.preset,
            seed=self.schedule.seed,
            schedule_digest=self.schedule.digest(),
            reproducible=self.schedule.reproducible,
            duration_s=self.schedule.duration_s,
            scheduled=len(self.schedule.requests),
            offered_load_cv=self.schedule.offered_rate_cv(),
        )
        if not self.schedule.reproducible:
            rep.warnings.append(
                "closed-loop preset: offered load depends on server response "
                "time, so traffic is NOT identical across configurations. "
                "Not valid for sweep comparisons."
            )

        poller = threading.Thread(target=self._poll_metrics, daemon=True)
        poller.start()

        t_start = time.monotonic()
        threads: list = []
        if self.schedule.params.get("mode") == "closed_loop":
            deadline = t_start + self.schedule.duration_s
            for req in self.schedule.requests:
                th = threading.Thread(
                    target=self._closed_loop_worker, args=(req, deadline), daemon=True
                )
                th.start()
                threads.append(th)
        else:
            for req in self.schedule.requests:
                target = t_start + req.arrival_s
                delay = target - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                lateness = time.monotonic() - target
                with self._lock:
                    self._lateness.append(lateness)
                    self._sent += 1
                th = threading.Thread(target=self._fire, args=(req,), daemon=True)
                th.start()
                threads.append(th)

        for th in threads:
            th.join(timeout=self.request_timeout_s)
        self._stop.set()
        poller.join(timeout=5.0)

        self._finalise(rep, time.monotonic() - t_start)
        return rep

    def _closed_loop_worker(self, req: ScheduledRequest, deadline: float) -> None:
        while time.monotonic() < deadline and not self._stop.is_set():
            with self._lock:
                self._sent += 1
            self._fire(req)

    def _finalise(self, rep: RunReport, elapsed: float) -> None:
        with self._lock:
            lat = sorted(self._latencies)
            late = list(self._lateness)
            samples = list(self._samples)
            rep.sent = self._sent
            rep.completed = self._completed
            rep.failed = self._failed

        if late:
            rep.arrival_lateness_mean_s = round(statistics.mean(late), 4)
            rep.arrival_lateness_max_s = round(max(late), 4)
            # A schedule fired more than 250ms late was not the schedule asked for.
            rep.schedule_achieved = rep.arrival_lateness_max_s < 0.25
            if not rep.schedule_achieved:
                rep.warnings.append(
                    f"arrivals fired up to {rep.arrival_lateness_max_s:.2f}s late; "
                    "the client, not the server, shaped this traffic"
                )

        if lat:
            rep.latency_p50_s = round(_pct(lat, 50), 4)
            rep.latency_p95_s = round(_pct(lat, 95), 4)
            rep.latency_p99_s = round(_pct(lat, 99), 4)

        rep.metrics_samples = len(samples)
        if samples:
            waiting = [w for _, _, w in samples if w is not None]
            running = [r for _, r, _ in samples if r is not None]
            if waiting:
                rep.max_waiting = max(waiting)
                # Trapezoidal over sample timestamps: time spent with a queue.
                t_wait = 0.0
                for (t0, _, w0), (t1, _, w1) in zip(samples, samples[1:]):
                    if w0 and w1:
                        t_wait += t1 - t0
                rep.time_at_nonzero_waiting_s = round(t_wait, 3)
                rep.fraction_time_waiting = round(t_wait / elapsed, 4) if elapsed else 0.0
                rep.achieved_saturation = rep.max_waiting > 0
            if running:
                rep.max_running = max(running)
                m = statistics.mean(running)
                rep.observed_running_cv = (
                    round(statistics.pstdev(running) / m, 4) if m > 0 else None
                )
        else:
            rep.warnings.append("no /metrics samples collected; achieved state unknown")

        # Intended vs achieved, stated plainly.
        if "saturat" in rep.preset and not rep.achieved_saturation:
            rep.warnings.append(
                "preset intends saturation but the queue never left zero -- "
                "this run does NOT exercise queueing despite its name"
            )
        if rep.offered_load_cv is not None and rep.offered_load_cv < 0.05:
            rep.warnings.append(
                f"offered-load CV {rep.offered_load_cv:.3f}: traffic is nearly "
                "steady. A dynamic-vs-static result from this run says nothing "
                "about adapting to variation."
            )


def _pct(sorted_vals: list, p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Reproducible load generator for vLLM.")
    p.add_argument("--endpoint", default="http://localhost:8000")
    p.add_argument("--model", default=None)
    p.add_argument("--preset", default="bursty", choices=sorted(PRESETS))
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--rate-scale", type=float, default=1.0)
    p.add_argument("--out", default=None, help="write the run report as JSON")
    p.add_argument("--list-presets", action="store_true")
    args = p.parse_args(argv)

    if args.list_presets:
        for name in sorted(PRESETS):
            pr = PRESETS[name]
            tag = "open-loop " if pr.reproducible else "CLOSED-LOOP"
            print(f"  {name:14s} [{tag}] {pr.purpose}")
        return 0

    sched = build_schedule(
        args.preset, seed=args.seed, duration_s=args.duration, rate_scale=args.rate_scale
    )
    print(f"preset={sched.preset} seed={sched.seed} digest={sched.digest()} "
          f"reproducible={sched.reproducible} requests={len(sched.requests)}")

    model = args.model
    if model is None:
        with urllib.request.urlopen(f"{args.endpoint.rstrip('/')}/v1/models", timeout=10) as r:
            model = json.loads(r.read())["data"][0]["id"]
    print(f"model={model}")

    rep = LoadGenerator(args.endpoint, model, sched).run()
    out = rep.to_dict()
    out["captured_utc"] = datetime.now(timezone.utc).isoformat()
    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
    for w in rep.warnings:
        print(f"WARNING: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
