# Roadmap and adoption analysis

Written 2026-08-09, after the dynamic thesis closed and the benchmark shipped.
This is the "who can actually use this, and what stops them" document. It is
opinionated on purpose; disagree with it in a commit rather than in your head.

---

## Where it stands

`joule benchmark` works end to end. It calibrates a latency budget without root,
sweeps clocks under root, and reports what that run measured. It is installed as
a console script, it is on GitHub with a README and an Apache-2.0 licence, and
the numbers behind it are reproducible from committed tools.

**Who can use it today,** all conditions required:

- an NVIDIA GPU they control
- root on that machine
- vLLM as the serving engine
- a latency-bounded workload with slack to trade
- willingness to clone a repo and `pip install -e .`

That is a real audience — self-hosted inference, on-prem, cost-conscious teams —
and a narrow one.

---

## What gates the audience, ranked by how many people it excludes

### 1. It benchmarks synthetic traffic, not the user's

**This is the big one.** `--preset bursty` seeds an RNG and generates arrivals
that came from nowhere. The recommendation that comes out is correct *for that
pattern*.

That would be a footnote except for this project's own strongest finding:
**load shape moved energy per token ~5x harder than clock did.** So a clock
tuned under `bursty` may simply not be the right clock for someone serving
steady high-occupancy chat traffic. The tool is honest about what it measured;
it just measured something adjacent to what the user cares about.

**The fix is trace replay**, and the architecture is already shaped for it.

    joule benchmark --trace my_production_traffic.csv --model their/model

    arrival_s,prompt_tokens,max_tokens
    0.000,412,180
    0.284,89,64
    0.291,1203,512
    2.847,340,96

The load generator still *issues* requests — you cannot replay history against a
live server, something has to be sent. But the **shape** is theirs: their arrival
timing, burstiness, prompt-to-output ratio, concurrency.

Why it is cheap: `Schedule` is already a list of `(arrival_s, prompt,
max_tokens)` plus a digest. A CSV loader produces one beside `build_schedule()`,
and every downstream guarantee — identical replay at every clock, digest
assertion, void-the-sweep-on-mismatch — applies unchanged. Roughly an afternoon.

Why it matters commercially: **it needs no prompt text.** Energy depends on token
counts and arrival timing, not on what the words were, so no customer data leaves
their machine. That is the difference between a company being able to run this
and not.

Where the trace comes from: most serving stacks already log it. vLLM logs
per-request timing and token counts; a gateway or proxy access log has arrival
times and sizes. An export script, not new instrumentation.

### 2. vLLM only

`workload.py` has 26 hard references to `vllm:` metric names. SGLang, TGI,
TensorRT-LLM and Ollama users get nothing.

The fix is an adapter interface — one class per engine exposing running, waiting
and token counters. The parsing is already isolated, so this is a refactor rather
than a rewrite. The real cost is that **each engine needs its own frozen
fixtures**, because the project's rule is that metric names are never guessed.

Do it when someone asks for a second engine, not before.

### 3. NVIDIA only

Everything is NVML. AMD MI300X is a serious share of inference now and none of
this runs there. ROCm SMI has different semantics, different counters and its own
energy story — this is the largest lift on the page. **Hold until demand
exists.**

### 4. Not on PyPI

`pip install joule-agent` versus clone-and-editable-install. An hour of work, and
it is the difference between people trying it and bouncing.

### 5. One hardware datapoint

Every number is from a consumer card running a 0.5B model at ~11% of its
demonstrated throughput. This is a credibility gate, not a capability gate, and
no company will act on it. See "Open measurement items".

### 6. Inherent exclusions — do not try to fix

- **Managed or serverless inference**: no clock access. Nothing to build.
- **Saturated GPUs**: little latency slack to trade. The tool should say so
  clearly rather than manufacture a recommendation.

---

## Could a GPU cloud provider use this?

Short answer: **not as it stands, and the reason is structural, not technical.**

**They do not own the workload.** A neocloud rents capacity; the tenant picks the
model, the stack and the traffic. This tool works by generating load against a
server it can reach and changing clocks on a card it controls. A provider has the
second half and not the first — often no visibility inside the container at all.

**Locking a tenant's clock is a contract problem before it is a config one.** A
customer who benchmarked their throughput at boost clock has bought that
throughput. Changing it needs a pricing or SLA construct first.

**It is single-node and interactive.** Root, one GPU, one server, twenty minutes
at a terminal. A fleet needs unattended operation across thousands of cards with
per-tenant policy and rollback — which `CLAUDE.MD` deliberately puts out of scope
for this repo.

**What a provider WOULD care about:**

- **The occupancy finding.** Load shape beating clock by ~5x says packing density
  is worth far more than DVFS. For a power-constrained datacenter, joules per
  token *is* deployable capacity, and the lever is scheduling, not clocks. They
  can act on that without running any of this code.
- **The negative result.** If anyone there is scoping dynamic DVFS for the fleet,
  this says the idle-gap mechanism is dominated by a static lock. That is weeks
  of engineering not spent.
- **The saving — but only at their conditions.** Their cards run large models near
  saturation. A saturated card has far less slack, so the honest expectation is
  that ~30% shrinks, possibly a lot.

**What would make it usable by them**, in order:

1. **Datacenter-hardware evidence at realistic occupancy.** Without this nothing
   else matters.
2. **Non-invasive mode** — estimate the optimal clock from *observed* traffic
   without generating load or changing state. Much harder: a modelling problem
   rather than a sweep. This is what a provider actually needs.
3. **Fleet control plane** — policy, rollback, per-tenant opt-in, audit.
   Explicitly not this repo.
4. **Vendor breadth**, if their fleet is not all NVIDIA.

The strategic line is already drawn correctly: open-source single-node
measurement here, fleet orchestration as a separate product. A provider is a
customer for the second, and the first is what makes the second credible.

---

## The pushback: do not chase universality

"Amazing tool that every company can use" is not reachable, and chasing it would
damage the thing that makes this credible.

The differentiator is not coverage. It is that **the answers are trustworthy** —
the refusals (closed-loop presets, the power axis, unmeasured percentages), the
retractions, the published negative result. A tool that works excellently for
self-hosted GPU inference and says clearly *"there is no saving at your operating
point"* is worth more than one that half-works everywhere and hedges.

Breadth that dilutes the rigour trades away the only real moat.

---

## Build order

| # | item | effort | unlocks |
|---|---|---|---|
| 1 | **Trace replay** | ~afternoon | The recommendation applies to the user's actual workload |
| 2 | **PyPI publish** | ~hour | People try it instead of bouncing |
| 3 | **Datacenter evidence** | GPU rental | Anyone at a company taking it seriously |
| 4 | **Engine adapter interface** | days + fixtures per engine | Non-vLLM users |
| 5 | **AMD / ROCm** | large | Non-NVIDIA fleets |
| 6 | **Non-invasive estimation** | research | Providers who cannot generate load |

Items 1 and 2 before telling anyone about it. Item 3 before telling a company.

---

## Open measurement items

- **H100/H200 run.** `tools/h100_bringup.sh check` first — capability gate, then
  the multi-GPU guard identity test, which has never run on real hardware. Then
  the static benchmark at realistic occupancy with a model large enough to load
  the card (8B minimum, `--rate-scale` pushed up).
- **test-integrity-auditor** on `tests/test_oracle_idle.py`. Launched, terminated
  on a session limit, never produced a verdict.
- **measurement-skeptic** on the `+1.04%` oracle claim. Same — never ran. The
  claim changed sign since the review was scoped, so it deserves the pass more
  now than it did then.
- **`oracle_active`** has no realizable policy and is not scheduled. The 5.79%
  figure came from per-request assignment, which one-clock-per-GPU makes
  impossible. Do not resurrect without one.
