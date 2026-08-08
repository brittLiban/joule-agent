# joule-agent

Measure tokens-per-joule on a running LLM inference server, and find the lowest-energy
GPU clock that still meets your latency budget.

```
joule benchmark --endpoint http://localhost:8000 --model your/model
```

GPUs serving inference usually run at their default boost clock, which is tuned for
peak throughput, not for energy. If your deployment is latency-bounded rather than
throughput-bounded, a lower clock can cut energy per token at a latency cost you
choose. `joule-agent` measures whether that is true **on your hardware, your model
and your traffic** — and tells you when it isn't.

It measures. It doesn't guess, and it doesn't ship a number it didn't observe.

---

## Status

**Working:** GPU telemetry sampler, vLLM metrics reader, reproducible load generator,
static clock sweep, a GPU write guard with restore-on-any-exit, and the `joule
benchmark` command.

**Not built:** a dynamic/phase-aware controller. That was the original thesis; the
evidence so far does not support it on the hardware tested (see
[What we found](#what-we-found)).

**Tested on:** NVIDIA RTX 3060 Ti, drivers 595.71.05 and 595.84, vLLM 0.26.0,
Qwen2.5-0.5B-Instruct.
Datacenter GPU validation is pending. Treat everything here as verified on one
consumer card until that changes.

---

## Install

```bash
git clone https://github.com/brittLiban/joule-agent
cd joule-agent
python -m venv .venv && .venv/bin/pip install -e .
```

Requires Python 3.10+, an NVIDIA GPU, and `nvidia-ml-py` (installed automatically).
Locking clocks needs root; measuring does not.

---

## Use

### 1. Calibrate first — no root required

```bash
joule benchmark --endpoint http://localhost:8000 \
                --model your/model --calibrate-only
```

Runs your server at its stock clock and reports the latency budget it derives:

```
[1/2] calibrating latency budget: 10 stock runs, no clock changes, no root needed ...
      stock p99 median 602.6 ms  ->  budget 662.9 ms (x1.1)
```

Inspect that number before letting anything touch the GPU.

### 2. Sweep

```bash
sudo joule benchmark --endpoint http://localhost:8000 \
                     --model your/model \
                     --slo-ms 662.9 \
                     --i-understand-this-changes-gpu-settings
```

Passing `--slo-ms` freezes the budget you just calibrated. Omit it and the benchmark
calibrates and freezes its own, in one pass.

### 3. Read the report

```
THIS RUN MEASURED     33.2% lower energy per token
  p99 latency         +10.7%  (599 -> 664 ms)
  throughput          -0.3%  (596 -> 594 tok/s)
```

Then set the clock yourself — `joule-agent` recommends, it does not install anything
persistent:

```bash
sudo nvidia-smi -lgc <recommended>     # revert with: sudo nvidia-smi -rgc
```

---

## How it measures

**One workload, replayed identically at every clock.** Arrival times, prompts and
output lengths are drawn from one seeded RNG *before* the run, so every clock point
receives byte-identical traffic. The schedule's digest is asserted on every run; a
mismatch voids the sweep rather than warning.

**Open loop only.** Closed-loop load (N workers firing back-to-back) is refused,
because offered load would then depend on server speed — a faster clock receives more
traffic, and the comparison confounds the config change with the traffic change it
caused.

**Energy from the GPU's own counter.** `nvmlDeviceGetTotalEnergyConsumption`, verified
against a trapezoidal integration of the power sample buffer under an oscillating
load — agreement within 0.2% through 5 Hz on the reference card. Re-verify on new
hardware; agreement there is not agreement here.

**Calibrate, freeze, compare.** The latency budget is derived once from a stock
baseline and frozen for every point in the comparison. Re-deriving per configuration
would move the goalposts. An explicit `--slo-ms` is used unchanged — your SLO is a
requirement, not an estimate.

**The GPU always goes back.** Every clock write goes through one guard that restores
stock on clean exit, exception, SIGINT, SIGTERM, and — via a state file — after
SIGKILL. A failed restore aborts the whole sweep rather than measuring a card in an
unknown state.

---

## What we found

Measured on the reference card. **These are observations about one consumer GPU
running a 0.5B model, not product claims.** The tool exists so you can find out what
is true on yours.

- **Energy per token is monotone in clock over the tested range** — no interior
  optimum, so the best static point is simply the lowest clock that still meets the
  budget. This is a property of a card that was never the throughput bottleneck; a
  saturated GPU may behave differently.
- **Load shape moved energy per token about 5x harder than clock did.** Two open-loop
  presets at the same stock clock differed by ~80% in joules per token. **A
  joules-per-token figure without its workload is not a result**, which is why the
  report prints them together.
- **Idle-gap downclocking has a measured ceiling of ~5.5%** on this card, against the
  >10% bar the dynamic controller would have needed. An independent per-request oracle
  with perfect foresight and zero switching cost reached 5.79% over the same range.
  Two unrelated methods, roughly half the bar.
- **Clock requests round up to a 15 MHz grid** on this card. Bin width is a
  per-driver, per-device property — the tool asks the device rather than assuming.
- **A recommended clock does not survive a driver upgrade.** The same sweep, same
  workload, same seed, run on 595.71.05 and then 595.84, moved the best point from
  **1590 MHz to 1560 MHz** and the saving from **29.6% to 33.2%**. Every clock got
  1.5–3.5% cheaper, most in the middle of the range. A tool that shipped "set this
  card to 1590" would have been wrong within one unattended upgrade. This is the
  clearest argument for measuring rather than looking a number up.

Percentages here move with driver, workload and hardware. Run the tool rather than
citing them.

Full detail, with the caveats attached and the reproduction commands, is in
[docs/findings.md](docs/findings.md).

---

## What this does not do

- **It does not install anything persistent.** No daemon, no service, no boot-time
  hook. It measures, prints, and restores.
- **It does not tune the power limit.** A power-cap axis exists in the raw sweep as a
  research flag, but every power-axis claim this project made was retracted after
  review, and the efficacy gate that would justify a new one has not been built. The
  benchmark does not expose it.
- **It does not do dynamic/phase-aware control.** See above.
- **It does not claim a percentage it did not measure in that run.** If you find a
  hard-coded efficiency number anywhere in this codebase, that's a bug — please file
  it.

---

## Safety

Clock changes are real writes to shared hardware. The design assumes you are right to
be nervous:

- One module (`gpu_guard.py`) performs every clock/power write. Nothing else may call
  `nvmlDeviceSetGpuLockedClocks`, `nvmlDeviceSetPowerManagementLimit`, or shell out to
  `nvidia-smi`.
- Restore is guaranteed on every exit path, including hard kill, via a state file that
  the next run reads before it will arm.
- A clock below the module's floor is **refused**, never silently clamped.
- Missing root fails loudly. It never silently no-ops.
- Writes require `--i-understand-this-changes-gpu-settings`. Reads never do.

Check for a stranded lock at any time:

```bash
python -m joule_agent.gpu_guard --gpu 0 status
sudo python -m joule_agent.gpu_guard --gpu 0 restore   # safe even if nothing is locked
```

---

## Development

```bash
.venv/bin/pytest          # no GPU, no root, no serving engine required
```

The suite runs against injected fakes, so it verifies control flow — ordering, digest
enforcement, abort-on-failed-restore — without hardware. Tests that pin a property
name, in their docstring, the mutation that should break them.

Contributions welcome, especially measurements on hardware we don't have. If you run
this on a datacenter GPU, the results are interesting to us whichever way they come
out.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
