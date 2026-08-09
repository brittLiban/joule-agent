# Measured findings

Every number here was measured on one machine. Read the hardware line before the
percentage. The tool exists so you can find out what is true on yours.

**Reference hardware:** NVIDIA RTX 3060 Ti, vLLM 0.26.0, Qwen2.5-0.5B-Instruct,
`--gpu-memory-utilization 0.60 --max-model-len 2048 --dtype bfloat16`.
**Driver:** 595.71.05 unless a row says otherwise. See
[the driver boundary](#the-driver-boundary).

---

## 1. The static result

Energy per token is **monotone in clock** across the full tested range
(600–1920 MHz). There is no interior optimum, so the best static point is simply
*the lowest clock that still meets the latency budget*.

That makes the answer a property of your budget, not of the hardware — which is
why the tool measures rather than ships a recommended clock.

| driver | stock J/tok | tuned J/tok | best clock | saving | p99 |
|---|---|---|---|---|---|
| 595.71.05 | 0.21463 | 0.15103 | 1590 MHz | −29.6% | 659 ms |
| 595.84 | 0.21131 | 0.14125 | **1560 MHz** | **−33.2%** | 664 ms |

`bursty`, seed 1234, 3 repeats, 60 s runs, identical schedule digest, budgets
664.4 and 664.7 ms.

**The knee moved two bins across a driver upgrade, and the whole curve shifted
down.** Matched point by point, 595.84 is cheaper everywhere — 1.5% at stock,
1.8–3.5% at the locked clocks — and the improvement is *structured*, largest in
the middle of the range and smaller at both edges. Prior cross-session drift on
this box was stock +0.75% / clk1650 +1.39%, so this is 2–5x larger and has a
shape noise does not.

At n=1 on the new driver this cannot fully separate a driver effect from session
drift. It does not need to for the conclusion that matters: **a recommended clock
is not portable across driver versions**, which is the strongest available
argument that the product is the procedure and not the number. Caveat on the
comparison: the 595.84 session's stock was noisier (J/token stdev 1.7% vs 0.5%).

Clock requests **round up to a 15 MHz grid** on this card (1580 → 1590,
1595 → 1605). Verified under load at 100% utilisation; an idle probe would prove
nothing. Bin width is a per-device, per-driver property — the tool asks the
device.

---

## 2. Occupancy dominates the clock knob

This is the largest effect measured in the project, and it is not the one the
project set out to measure.

| condition | concurrency | gen tok/s | J / total token |
|---|---|---|---|
| `bursty` — the preset behind every sweep here | 9 | 711.6 | 0.15255 |
| `bursty_heavy` | 43 | 2417.8 | 0.02937 |

Same card, same stock clock, ~same power (127.4 W vs 120.2 W). **3.8× the
throughput and 5.2× better joules per total token, from load shape alone.** The
whole clock-tuning win is −29.6%; this is −80%.

Cause is batch occupancy — a decode forward pass emitting ~56 tokens instead of
~8. It is not a prefill-share artifact: the ratio holds on *total* tokens, which
normalises the mix (28.7% vs 44.5% prompt).

**Consequence: a joules-per-token figure without its workload is not a result.**
The benchmark prints them together for this reason, and cross-hardware
comparisons that don't match load shape are meaningless.

### The card was never the bottleneck

A closed-loop concurrency ladder at stock:

| condition | concurrency | gen tok/s |
|---|---|---|
| `single` | 1 | 299.5 |
| `burst` | 16 | 3898.8 |
| `saturated` | 32 | **6630.1** |

Throughput was **still scaling at 32 workers** and the queue never left zero, so
6630 tok/s is a *floor* on capacity, not a ceiling.

**`bursty` runs the card at ~11% of that.** So every result on this page — the
−29.6% win included — is a property of a card at roughly a tenth of its
throughput. That cuts both ways: a loaded card has less idle slack to reclaim,
so the positive result is as exposed as the negatives.

A sweep-valid high-occupancy regime exists and was verified:
`bursty_heavy --rate-scale 3.0` gives 6814 gen tok/s at 225 concurrent,
open-loop, 931/931 completed, draining cleanly.

---

## 3. Why there is no dynamic controller

The project's original question was whether phase-aware clock control beats a
well-tuned static lock by more than ~10% at equal p99. Two independent methods
put the available margin at roughly half that.

### Idle-gap ceiling: 4.7% with perfect foresight, 2.8% realistic

`f_idle` was measured, not assumed. Under `bursty` at the tuned-static clock the
engine is empty **42.1%** of wall time, in 12 contiguous gaps with a 2.53 s
median (stock: 43.4% / 43.0% across two runs, agreeing to 0.30%). The nominal
duty cycle of 0.5 was optimistic by ~13%.

Contiguous duration is what a controller can use, so the exploitable fraction is
`Σ max(0, gᵢ − τ) / T` for a detection-plus-actuation reserve τ:

| | f | ceiling |
|---|---|---|
| perfect foresight (τ = 0) | 0.421 | **4.7%** |
| τ = 0.5 s (sensor cadence) | 0.332 | **3.7%** |
| τ = 1.0 s (realistic causal) | 0.254 | **2.8%** |

A controller needing one second to detect and act loses 39% of the opportunity
before it writes a clock. Reproduce with `tools/idle_fraction.py`.

**τ, measured rather than guessed.** A guarded clock write returns in **~23 ms**
(including the state-file fsync). First *observation* of the new clock is
**~200 ms** — but 29 of 30 transitions landed within 3 ms of that figure in both
directions, and a physical settling process has variance. That constancy is the
reporting path: `nvmlDeviceGetClockInfo` refreshes with ~200 ms latency, the same
class of limit component 1 found in the power sensor. **True actuation lies
somewhere in [23, 200] ms and this instrument cannot separate them** — so 200 ms
is an upper bound, not a measurement. It also fills a hole in the sensor profile,
which previously recorded `sm_clock_mhz: observed_changes 0`.

Folding that in, with an up+down pair costing at most 400 ms of a 2.53 s gap:

| | f | ceiling |
|---|---|---|
| oracle, optimistic actuation (2×25 ms) | 0.410 | 4.5% |
| oracle, pessimistic actuation (2×200 ms) | 0.348 | **3.9%** |
| causal, 500 ms signal age + actuation | 0.269 | **3.0%** |
| causal, 1 s signal age + actuation | 0.191 | 2.1% |

**Per-epoch control is physically viable on this device** — actuation costs at
most 16% of a median gap. What it cannot do is reach 10% from idle gaps alone.
Reproduce with `tools/verify_clock_transition.py`.

**CORRECTION: the controller's signal is ~37 ms, not 500 ms.** Earlier versions
of this page discounted a causal controller by the power sensor's ~500 ms
refresh. That was the wrong term — a queue-driven policy never reads power;
power is how the *experiment* measures energy, not how the *policy* sees the
world. Measured by scraping `/metrics` 17,099 times in 45 s under load:

| field | changes | median gap |
|---|---|---|
| `num_requests_running` | 557 | **34.9 ms** |
| `generation_tokens_total` | 7070 | **2.9 ms** |
| `num_requests_waiting` | 0 | never moved |
| `GET /metrics` latency | — | 2.5 ms median |

So the controller's budget is **~37 ms of signal age + ~23 ms to issue the write
+ ≤200 ms for it to take effect** — not the ~900 ms previously assumed.

| | f | ceiling |
|---|---|---|
| oracle, zero switching cost | 0.421 | 4.7% |
| causal, optimistic (2×60 ms) | 0.396 | **4.4%** |
| causal, pessimistic (2×260 ms) | 0.329 | **3.6%** |
| *causal, old 500 ms assumption* | *0.269* | *3.0%* |

**This makes the negative result stronger, not weaker.** The usual defence of
dynamic control is that the oracle is high and only prediction is missing. Here
the oracle-to-causal penalty is **0.3–1.0 percentage points** — there is almost
no gap for better engineering to close. The ceiling itself is the problem.
`tools/verify_signal_freshness.py`.

**And a static clock range takes almost none of it.** Every "tuned static"
number here used an exact `[f, f]` lock, which holds the clock up through gaps;
NVML has always accepted `[min, max]`, but the CLI could not express it. The
mechanism works — `[405, 1560]` with no traffic descends to 405 MHz and 13.5 W.

Under `bursty` it barely helps. Across two runs, three settled exact/range pairs
on identical traffic measured **−0.07%, −0.14%, −0.57%** (mean −0.26%, stdev
0.27) at p99 within 1 ms. Mean power differs by **0.2 W**. Full idle capture
would show 3–4 W, so the range reaches roughly **5% of the theoretical idle
saving**. The direction is consistent across all three pairs, so a small real
effect is plausible; the magnitude does not move any downstream number.

**The governor descends at sustained idle but not within 2.5 s gaps**, so the
idle opportunity stays out of static's reach. Under the preregistered
tie-handling rule — unresolved uncertainty favours the stronger comparator — the
range's (marginally lower) energy is the demanding denominator for any dynamic
claim, and the difference is small enough that it does not change the Gate 2
target. `tools/range_vs_exact.py`.

**Method note, because the first two attempts both got this wrong.** A 60 s
warm-up was *not* enough: the first exact leg still read 3.1% above the two
later exact legs, which agreed with each other to 0.02%. Averaging that cold
pair in produced a −1.08% "range wins" from a run whose settled pairs read
−0.14% and −0.57%. The tool now uses its own repeated exact legs as a control,
excludes a first pair that deviates by more than 1%, and refuses to render any
verdict when the surviving pairs disagree by more than the effect.

### The superseded figure: 5.5%

The originally published 5.5% assumed `f_idle = 0.5` from the preset's duty
cycle. The ΔP measurement below stands; only the assumed fraction was wrong.

`ceiling = f_idle × ΔP / P_loaded`. Four 60 s legs with **no traffic**,
interleaved 1590/405/1590/405:

| leg | W (energy counter) | SM | mem |
|---|---|---|---|
| 1590-A | 29.41 | 1590 | 810 |
| 405-A | 19.48 | 405 | 405 |
| 1590-B | 31.41 | 1590 | 810 |
| 405-B | 21.50 | 405 | 405 |

Absolute level drifted **2.0 W** across the session; the **paired** differences
agreed to **0.02 W** (9.93 and 9.91). That is what the interleaving bought.

ΔP = 9.92 W, so with a generous `f_idle` = 0.5 and 89.52 W measured under load:
**5.5%** against a >10% bar. The threshold was pre-registered before the number
existed: reaching 10% needed `P@1590_idle ≥ 41.6 W`; measured 30.41 W.

Reproduce with `tools/idle_gap_ceiling.sh`.

**Caveats, both pointing the same way.** Locking SM to 1590 also lifts memory
405 → 810, so ΔP is SM and memory P-state *jointly*; if the memory half is not
recoverable the ceiling is *lower*. And 5.5% is a bound nothing approaches — a
controller gives back power only during a gap it has *detected*, and detection is
bounded by how fast it can see the engine's state -- **measured at ~37 ms**, not
the ~500 ms an earlier version of this file claimed. See the correction below.

### Per-request oracle: 5.79%

Every sweep point replays the same `Schedule` (digest asserted per run), so
request *i* at one clock and request *i* at another are the **same request** —
paired observations. That allows an oracle with perfect foresight, per-request
granularity and zero switching cost, scored entirely from measured data.

Over the clock range where the bound is tight it reaches **−5.79%** against
tuned static. **An upper bound falling short is the informative case:** no
controller in that range can do better.

**The bound is only valid over a narrow clock range, and the tool now checks
this.** One clock applies to the whole GPU; under continuous batching a
per-request assignment is realizable only where concurrent requests would have
received the same clock. The diagnostic is the spread of the clocks the oracle
*actually assigns*, not the grid it was offered:

| grid offered | clocks used | spread | result | standing |
|---|---|---|---|---|
| 1530–1650 + stock | 1530–1590 | 3.9% | −5.79% | **valid** |
| 600–1800 + stock | 600–1650 | 175% | −25.1% | **not a bound** |

The wide-grid −25% is a method breakdown, recorded so it is not re-derived and
believed. It puts concurrent requests at clocks 175% apart, which cannot be true
of the same instant.

Reproduce with `tools/oracle_mix.py <sweep-dir>`.

### The idle oracle, run on hardware: it does not beat turning the clock down

The analytical bounds above are superseded by a measurement. `oracle_idle`
is a perfect-foresight, physically realizable controller: it reads the
materialized schedule, so it upclocks **before** each burst arrives — something
no causal policy can do — and drops to 405 MHz in the gaps. One GPU-wide clock
on a timeline, real writes through the guard, real queue coupling, real energy.

Against `exact-1560` it looked like a win: **−2.74%** energy, three pairs
agreeing to 0.4 points. But its p99 rose from 663 to 680 ms, which **fails the
frozen 664.7 ms SLO** — it bought that energy with latency.

So the comparison has to be at equal p99, against a static curve measured in the
same session. (Not a borrowed one: `exact-1560` read 6.5% apart between two
sessions on the same driver, which is why cross-session curves cannot serve as
comparators here.)

| | p99 | J/token |
|---|---|---|
| exact 1560 | 666 ms | 0.15029 |
| exact 1530 | 673 ms | 0.14766 |
| exact 1500 | 683 ms | 0.14442 |

Each oracle leg against that curve at **its own** p99:

| oracle p99 | oracle | static | delta |
|---|---|---|---|
| 679 ms | 0.14514 | 0.14573 | **−0.40%** |
| 678 ms | 0.14628 | 0.14589 | **+0.26%** |
| 692 ms | 0.14621 | outside curve | not comparable |

**Mean −0.07% over n=2, against a −10% bar.**

**Read that as n=2, because it is.** The 691.7 ms leg fell outside a static curve
ending at 683 ms and was excluded rather than extrapolated — extrapolating would
have invented the comparator. A rerun with a ladder extending past every oracle
leg (`--exact-clocks 1560,1515,1470,1440`) is outstanding, and the wording here
will be revisited when it lands.

**The load-bearing finding does not depend on that interpolation at all:** the
oracle's p99 rose 663 → 680 ms and **failed the frozen 664.7 ms budget**. That
rests on three paired legs and is the robust half of this result.

What the evidence supports: the idle-gap oracle buys its energy with tail
latency, and at matched latency shows no advantage large enough to resolve at
this sample size. It does **not** yet support a confident "exactly zero", nor
the stronger phrasing an earlier version of this page used ("produces no
separation from the baseline it was built to beat"). Two independent reviews
(test-integrity, measurement-skeptic) remain outstanding on this claim.

Reproduce with `tools/oracle_idle.py --exact-clocks 1560,1530,1500`.

### What is left

The wide-grid run does establish one real thing: **the static optimum is forced
up by about ten requests out of 358.** At 1500 MHz, 96% of requests already fit
the budget and the point fails only on the tail, while being 11.4% cheaper than
the point that passes. Whether a controller can rescue those ten is unknown —
rescuing them means clocking up the whole GPU for the window containing them.

Untested: whether the safe knee moves between occupancy regimes, and whether any
of this holds on datacenter hardware. The idle-gap test is hardware-portable and
costs about six minutes and $3 on a rented GPU.

---

## 4. Measurement discipline

Things learned the expensive way, recorded so they are not relearned.

**A coarse grid silently eats the margin.** The knee sits between grid points. A
150 MHz grid put the tuned-static figure at 0.15499; refining to 15 MHz moved it
to 0.15103 — a third of a 10% margin, lost to grid spacing.

**A derived SLO threshold inherits the sampling error of the runs it came from.**
Deriving from a 3-sample median of the noisiest configuration left enough error
to flip points between sessions. Ten stock runs give stdev 0.37%, SEM 0.12%. The
sweep now records `slo_stock_samples` and warns below ten.

**Between-session drift is real and larger than within-session noise.** Stock p99
across three sessions: 0.5989, 0.6040, 0.5987 — ~0.9% spread against 0.37%
within-session. This is why the threshold is derived once per session and frozen,
rather than pinned to a literal from another day.

**Warm the card first.** A cold reference measures itself, not the config.

**Mean achieved clock does not determine J/token for a modulating clock.** Two
points with mean clocks 0.3% apart measured 1.70% apart in energy, because their
clock *distributions* differed. Matching on the mean of a distribution against a
fixed value is not a controlled comparison.

**Retracted:** every power-cap claim. Caps that never bind change nothing, which
is a tautology rather than evidence; and at equal p99 the apparent benefit
reversed. No power-axis claim belongs here until the axis has an efficacy gate.
`joule benchmark` does not expose it.

---

## The driver boundary

An unattended upgrade moved this box from **595.71.05 to 595.84** on 2026-08-07.
Sensor cadence, clock bin width and counter agreement are per-driver properties
that this project does not hard-code, so they are re-established rather than
inherited.

| property | 595.71.05 | 595.84 | status |
|---|---|---|---|
| power sensor refresh | ~500 ms | 499.75 ms | unchanged |
| energy counter cadence | — | 99.94 ms | recorded |
| energy agreement, oscillating load | 0.18% @ 110 W | 0.43% @ 113.8 W | passes (<2%) |
| clock bin width | 15 MHz, rounds up | pending | open |
| stock p99 median (`bursty`) | 598.7 ms | 604.3 ms | inside prior range |
| derived budget (`bursty`) | 658.6 ms | 664.7 ms | +0.93% |

**Stock p99 did not move outside the band session drift already explains.** Three
pre-upgrade sessions measured 598.9 / 604.0 / 598.7 ms — a 0.89% spread. The
595.84 session measured **604.3 ms**, which is +0.05% from the closest prior
session and inside that range. At n=1 on the new driver this cannot separate a
small driver effect from ordinary between-session drift, and it does not need to:
either way the derivation is per-session and re-derived, which is why the protocol
freezes a fresh threshold rather than importing a literal.

This says nothing about the clock-energy curve or bin width, which remain open.

The sensor cadence carrying over unchanged is the load-bearing one: it bounds any
control loop for ENERGY ATTRIBUTION, and the ~500 ms figure still bounds how
finely energy can be attributed to a decision. It does NOT bound the decision
itself -- see the signal-freshness correction in section 3.

The agreement figure moved 0.18% → 0.43%. Both are far inside the 2% gate, so the
gate closes either way. It is n=1 on each side under unmatched conditions (the
595.84 run was a freshly rebooted box, cooler card, no desktop processes
resident). That is one observation under different conditions — not a regression,
and not established noise either.

**Percentages on this page were measured on 595.71.05 and are pending
re-verification.** Run the tool rather than citing them.
