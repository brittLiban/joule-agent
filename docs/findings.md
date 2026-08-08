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

| | J / generated token | p99 | achieved MHz |
|---|---|---|---|
| stock | 0.21463 | 604 ms | 1920 |
| tuned static | 0.15103 | 659 ms | 1590 |

**−29.6% at +9.1% p99**, budget 664.4 ms, `bursty` preset, 3 repeats, 60 s runs.

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

### Idle-gap ceiling: 5.5%

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
bounded by the ~500 ms power-sensor refresh plus the metrics scrape cadence. On a
3 s gap that is most of the gap at the wrong clock.

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
control loop, and the ~500 ms figure that makes a 20 ms loop unevaluable still
holds.

The agreement figure moved 0.18% → 0.43%. Both are far inside the 2% gate, so the
gate closes either way. It is n=1 on each side under unmatched conditions (the
595.84 run was a freshly rebooted box, cooler card, no desktop processes
resident). That is one observation under different conditions — not a regression,
and not established noise either.

**Percentages on this page were measured on 595.71.05 and are pending
re-verification.** Run the tool rather than citing them.
