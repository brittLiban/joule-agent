# joule-agent

## What this is
A benchmark + optimization tool that measures and improves tokens-per-joule on LLM inference servers (starting with vLLM on NVIDIA GPUs). It reads live serving-engine metrics and GPU power/clock telemetry, and recommends (later: applies) clock/power settings that cut energy per token without hurting latency.

This repo is the **open-source single-node agent** — the free "doorway" product. It is NOT the fleet-level control plane (that's a future, closed, paid product — don't build fleet orchestration, RBAC, multi-tenant SLO enforcement, or a calibration database here).

## The one question this repo exists to answer
**Does dynamic, phase-aware clock control beat a well-tuned STATIC clock lock by more than ~10% tokens-per-joule, at equal p99 latency, on real inference traffic?**

- If the answer is no → this becomes a benchmark report and a portfolio piece. Fine outcome.
- If yes → this becomes the doorway into a company.

Everything we build should serve getting an honest, reproducible answer to that question — not impress anyone, not guess, not oversell.

## v1 target (the whole first milestone)
A single CLI command:

```
joule benchmark --endpoint http://localhost:8000
```

Producing a report shaped like:

```
Current energy:       4.8 joules/token
Suggested SM clock:   1410 MHz
Estimated reduction:  11.7%
Observed throughput:  unchanged
P99 latency change:   +1.8%
```

The comparison MUST include, as named baselines (not just "before/after"):
1. Stock/default GPU behavior (no clock changes)
2. Best static clock lock found via sweep (`nvidia-smi -lgc`) — THIS is the real baseline to beat, not stock default
3. Our dynamic/phase-aware controller
4. NVIDIA Workload Power Profiles, if available on the hardware (optional, nice-to-have)

## Core components to build, in order
1. **Telemetry sampler** — `pynvml`-based background thread sampling GPU power draw (W), SM clock (MHz), memory clock, utilization at ~50–100ms intervals. Writes timestamped samples.
2. **Workload signal reader** — poll vLLM's `/metrics` (Prometheus format) endpoint for queue depth, running/waiting request counts, and anything indicating prefill vs. decode composition of the current batch. Note: batches are MIXED (chunked prefill can put prefill+decode work in the same iteration) — don't assume a clean phase label exists. Reason about batch composition, not a binary phase flag.
3. **Load generator** — wrap vLLM's own `benchmark_serving.py` (or reimplement minimally) to drive realistic bursty traffic: varying prompt/output lengths, varying concurrency, not just a flat load.
4. **Static sweep runner** — iterate `nvidia-smi -pl <watts>` and `nvidia-smi -lgc <mhz>` settings, run the load generator at each, log tokens/sec, joules/token, p50/p95/p99 latency, SLO-violation rate. Find the best static point. THIS IS A REQUIRED BASELINE, not optional.
5. **Dynamic controller (simple first)** — a control loop: read workload signal → decide a clock/power target based on current batch composition and remaining latency headroom → apply via NVML → measure → repeat sub-second. Start with a dumb/simple rule. Don't over-engineer the policy before we have data.
6. **Benchmark/report engine** — runs all baselines back-to-back on the same workload, outputs the CLI report above plus a CSV/JSON of raw samples and a before/after plot (matplotlib is fine).
7. **Safety boundaries** — hard clock floor (never lock below a safe minimum), always restore default clocks/power on exit or crash (register cleanup handlers), refuse to run without explicit `--i-understand-this-changes-gpu-hardware-settings` type confirmation flag the first time.

## Non-negotiable safety/honesty rules
- Never claim a specific efficiency percentage in code comments, README, or CLI output that hasn't actually been measured in that run. No placeholder "20-30%" claims anywhere.
- Report savings as a curve against p99 degradation, never a single point number in isolation.
- Always compare against tuned-static, never just stock default — beating stock default proves nothing.
- Auto-restore GPU clocks to default on any exit path (including Ctrl+C, exceptions).
- NVML clock/power writes require root/admin — fail with a clear error message if not running with sufficient privileges, don't silently no-op.

## Stack / language
- Python only for v1. No Rust, no Go, no C++ yet — the interesting part right now is the control policy, not the runtime performance of the control loop itself (100ms–1s loop, Python is plenty fast).
- Key libs: `pynvml`, `requests` (for vLLM `/metrics`), `prometheus-client` parsing, `matplotlib` for plots, vLLM's own benchmark scripts as reference/dependency where possible.
- Target GPU for local dev: whatever's available (e.g. RTX 3060 Ti) for plumbing/wiring only — it's underpowered and power-locked, so don't expect a representative number locally. The real number comes from a rented H100/H200 (RunPod/Lambda/Vast).

## License plan (for when we get to publishing)
- This repo (the node-level agent + benchmark CLI): Apache 2.0.
- Do NOT build, in this repo: fleet orchestration, cross-server coordination, a calibration database/dataset product, canary/rollback automation for a fleet, dashboards, RBAC/SSO, financial/capacity reporting for sales purposes. Those are future closed/paid product territory — keep this repo scoped to single-server measurement and single-server optimization.

## Context on the person building this
Solo founder, nights/weekends, building this as both a possible company and as material for an MSECE thesis in AI hardware efficiency / inference optimization. Bootstrapped, no rush, correctness and honest measurement matter far more than speed or impressiveness. When in doubt, favor the option that produces a more trustworthy number over the option that produces a more exciting demo.