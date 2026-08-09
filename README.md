# joule-agent

**Find the lowest-energy GPU clock your LLM server can run at without missing your latency budget — by measuring it, on your hardware, under your traffic.**

```bash
joule benchmark --endpoint http://localhost:8000 --model your/model
```

```
THIS RUN MEASURED     33.2% lower energy per token
  p99 latency         +10.7%  (599 -> 664 ms)
  throughput          -0.3%   (596 -> 594 tok/s)
```

---

## Why this exists

GPUs serving inference run at a boost clock tuned for peak throughput. Most inference
deployments are latency-bounded, not throughput-bounded — so that clock is buying speed
you aren't using and charging you watts for it.

The catch is that the right clock isn't a number you can look up. We measured the same
card, same model, same traffic, across a **routine background driver update**: the
optimum moved from 1590 MHz to 1560 MHz and the saving went from 29.6% to 33.2%. A tool
that shipped "set your 3060 Ti to 1590" would have been wrong within one unattended
`apt upgrade`.

So this measures instead of advising. It measures carefully, it shows its work, and it
tells you when the answer is "no useful saving here."

---

## Quickstart

```bash
git clone https://github.com/brittLiban/joule-agent && cd joule-agent
python3 -m venv .venv && .venv/bin/pip install -e .
```

Python 3.10+, an NVIDIA GPU, and a running OpenAI-compatible inference server
(developed against vLLM). Measuring needs no privileges; locking clocks needs root.

### 1. Calibrate — no root

```bash
joule benchmark --endpoint http://localhost:8000 \
                --model your/model --calibrate-only
```

```
[1/2] calibrating latency budget: 10 stock runs, no clock changes, no root needed ...
      stock p99 median 604.3 ms  ->  budget 664.7 ms (x1.1)
```

It measures your server at its stock clock and derives a latency budget. **Nothing
touches the GPU.** Look at the number before letting anything change hardware state.

### 2. Sweep — needs root

```bash
sudo joule benchmark --endpoint http://localhost:8000 \
                     --model your/model \
                     --slo-ms 664.7 \
                     --i-understand-this-changes-gpu-settings
```

Passing `--slo-ms` freezes the budget you just inspected. Omit it and the benchmark
calibrates and freezes its own in one pass.

### 3. Apply it yourself

`joule-agent` recommends. It installs nothing, runs no daemon, and restores your GPU
before it exits.

```bash
sudo nvidia-smi -lgc <recommended>     # revert: sudo nvidia-smi -rgc
```

---

## How it measures

The measurement discipline *is* the product. Anyone can lock a clock and watch a
wattmeter; getting a number that survives scrutiny is the hard part.

**Identical traffic at every clock.** Arrival times, prompts and output lengths are drawn
from one seeded RNG *before* the run. Every clock point replays the same `Schedule`
object, and its digest is asserted on every run — a mismatch voids the sweep rather than
warning.

**Open loop only.** Closed-loop load (N workers firing back-to-back) is refused: offered
load would depend on server speed, so a faster clock receives more traffic and the
comparison confounds the config change with the traffic change it caused.

**Energy from the GPU's own counter,** cross-checked against a trapezoidal integration of
the power buffer under an oscillating load — agreement within 0.5% on the reference card.
That gate is re-run per device, because agreement there is not agreement here.

**Calibrate, freeze, compare.** The latency budget is derived once and frozen for every
configuration. Re-deriving per config would move the goalposts mid-comparison. An explicit
`--slo-ms` is used unchanged: your SLO is a requirement, not an estimate.

**Warm the card first.** A cold reference measures itself, not your configuration. We
learned this twice — 60 s of warm-up still wasn't enough — so the tools now use repeated
baseline legs as their own control to detect it.

**The GPU always goes back.** One module performs every clock write and restores stock on
clean exit, exception, SIGINT, SIGTERM, and — via a state file the next run reads — after
SIGKILL. A failed restore aborts the sweep rather than measuring a card in an unknown
state.

---

## What we measured

On one RTX 3060 Ti running Qwen2.5-0.5B under vLLM. **These are observations about one
setup, not product claims.** Full detail, caveats and reproduction commands in
[docs/findings.md](docs/findings.md).

**Energy per token is monotone in clock** across 600–1920 MHz — no interior optimum, so
the best static point is simply the lowest clock that meets your budget. That makes the
answer a property of *your* latency budget, which is why the tool measures it rather than
shipping a recommendation.

**Load shape moved energy per token ~5× harder than clock did.** Two open-loop presets at
the same stock clock differed by ~80% in joules per token, driven by batch occupancy.
**A joules-per-token figure without its workload is not a result** — which is why the
report prints them together, and why cross-hardware comparisons that don't match load
shape are meaningless.

**A recommended clock does not survive a driver upgrade.** 595.71.05 → 595.84 moved the
optimum from 1590 to 1560 MHz, and every clock point got 1.5–3.5% cheaper.

**Clock requests round up to a 15 MHz grid** on this card. Bin width is per-device and
per-driver, so the tool asks the device rather than assuming.

---

## Why there's no dynamic controller

This project set out to build one — a daemon adjusting clocks in real time to exploit
traffic gaps. The evidence didn't support it, and *how* it failed is more interesting
than the fact that it did.

**There is real idle to exploit.** Under bursty traffic the engine sits empty 42% of the
time, in ~2.5-second stretches. NVIDIA's own governor does not downclock into it.

**A static clock range doesn't capture it.** `[405, 1560]` descends to 405 MHz at
sustained idle — but under traffic it reaches only ~5% of the theoretical saving. The
governor descends at sustained idle, not inside 2.5-second gaps.

**Perfect foresight doesn't obviously capture it either.** We built a physically
realizable oracle that reads the schedule in advance and upclocks *before* each burst
arrives — something no real controller can do. Against the static baseline it looked like
a 2.74% win, **until you check its latency: p99 rose 663 → 680 ms and it failed the
664.7 ms budget it was measured under.** That part rests on three paired legs and does not
depend on any interpolation.

Compared at *equal* p99 against a static curve measured in the same session — four legs,
every one bracketed — the oracle came out **+1.04% worse than simply locking to a lower
clock**, with three of four legs agreeing on the sign.

The mechanism is visible in the spans. Over 665–699 ms the static curve trades latency for
energy across an **8.4%** range. The oracle's energy varies **1.0%** across a 12 ms latency
spread: it delivers one near-constant operating point, and that point is off the curve.

**Why: it refuses to slow down where the work is.** The oracle holds the tuned clock
through every burst and harvests only the gaps — but a 42%-idle trace keeps most of its
joules in the bursts. A static lock that runs the whole trace slightly slower captures
more. Perfect foresight about *when* traffic arrives doesn't help if the policy declines
to act *during* it.

So this isn't "short of the 10% bar." On this hardware the idle-gap mechanism lands on the
wrong side of zero against the baseline it was built to beat.

**One caveat, and it's real:** all of this ran at ~11% of the card's demonstrated
throughput. A loaded datacenter GPU may behave differently, and that generalization is
untested.

---

## What this does not do

- **No daemon, no service, no boot hook.** It measures, prints, and restores.
- **No power-limit tuning.** A power-cap axis exists in the raw sweep as a research flag,
  but every power-axis claim this project made was retracted after review and the efficacy
  gate that would justify a new one was never built. The benchmark doesn't expose it.
- **No dynamic control.** See above.
- **No percentage it didn't measure in that run.** If you find a hard-coded efficiency
  number anywhere in this codebase, that's a bug — please file it.

---

## Safety

Clock changes are real writes to shared hardware. The design assumes you're right to be
nervous.

- One module (`gpu_guard.py`) performs every clock and power write. Nothing else may call
  `nvmlDeviceSetGpuLockedClocks`, `nvmlDeviceSetPowerManagementLimit`, or shell out to
  `nvidia-smi`.
- Restore is guaranteed on every exit path, including hard kill, via a state file the next
  run reads before it will arm.
- A clock below the module's floor is **refused**, never silently clamped.
- Missing root fails loudly. It never silently no-ops.
- Writes require `--i-understand-this-changes-gpu-settings`. Reads never do.

Check for a stranded lock at any time:

```bash
python -m joule_agent.gpu_guard --gpu 0 status
sudo python -m joule_agent.gpu_guard --gpu 0 restore   # safe even if nothing is locked
```

---

## Research tools

The instruments behind the findings, all re-runnable:

| tool | question it answers |
|---|---|
| `tools/idle_fraction.py` | Does this workload leave the GPU idle at all, in usable stretches? |
| `tools/idle_gap_ceiling.sh` | What's the ceiling on reclaiming idle power? Six minutes, no serving engine. |
| `tools/range_vs_exact.py` | Does a static clock *range* beat an exact lock? |
| `tools/oracle_idle.py` | Can perfect foresight beat static at equal p99? |
| `tools/oracle_mix.py` | Per-request upper bound over a completed sweep. |
| `tools/capacity_ladder.py` | Is the GPU actually the bottleneck, or is your load too light? |
| `tools/verify_clock_transition.py` | How fast does an explicit clock change take effect? |
| `tools/verify_signal_freshness.py` | How stale is the engine state a controller would act on? |
| `tools/h100_bringup.sh` | Go/no-go gates for a rented datacenter GPU. |

---

## Development

```bash
.venv/bin/pytest          # no GPU, no root, no serving engine required
```

329 tests against injected fakes, verifying control flow — ordering, digest enforcement,
abort-on-failed-restore — without hardware. Tests that pin a property name, in their
docstring, the mutation that should break them.

Contributions welcome, especially **measurements on hardware we don't have**. If you run
this on a datacenter GPU, the results are interesting whichever way they come out —
including, especially, if they contradict what's above.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
