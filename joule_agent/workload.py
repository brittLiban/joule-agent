"""Workload signal reader: parse vLLM's Prometheus /metrics into a typed snapshot.

Component 2. Answers "what is the serving engine doing right now" so the
controller can reason about batch composition and queue pressure.

Two design rules, both learned the hard way elsewhere in this project.

**The expected metric set is explicit.** :data:`EXPECTED_METRICS` is a
module-level constant, not a scatter of ``.get(name, None)`` calls. vLLM's
metric surface moves between releases. When it moves, a missing name shows up
in :attr:`WorkloadSnapshot.missing` and in a failing test naming the metric,
instead of a silent ``None`` propagating into a benchmark number.

**Signal provenance is tracked, not assumed.** A metric that parses is not a
metric that has been validated. Every expected metric carries a
:class:`Exercise` state recording what the captured fixtures actually
demonstrated:

``EXERCISED``
    Observed non-zero across a meaningful range in fixtures.
``LOW_RANGE_ONLY``
    Observed non-zero, but only near the bottom of its range. Parses fine; its
    behaviour in the regime that matters is unvalidated.
``PRESENT_UNEXERCISED``
    Present in every scrape and always zero. The dangerous middle category --
    it parses, it never errors, and nothing has ever confirmed what a non-zero
    value looks like or means.
``ABSENT``
    Expected and not found. Reported loudly.

The states come from fixtures captured on one GPU with one model. When a
consumer starts making decisions off a ``LOW_RANGE_ONLY`` or
``PRESENT_UNEXERCISED`` signal, the code itself should be able to say that its
non-zero behaviour was never observed here. Same honesty pattern as the
telemetry sampler's ``possible_sample_loss``: report what is not known rather
than defaulting it away.

Batch composition is deliberately not a phase flag. Chunked prefill puts
prefill and decode work in the same engine iteration, so there is no clean
label to read. :func:`compose` diffs two snapshots and reports the prefill
token *share* of the interval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

__all__ = [
    "Exercise",
    "MetricSpec",
    "EXPECTED_METRICS",
    "PROVENANCE_SOURCE",
    "ParsedExposition",
    "WorkloadSnapshot",
    "Composition",
    "parse_exposition",
    "read_snapshot",
    "compose",
]


class Exercise(Enum):
    """What the captured fixtures actually demonstrated about a metric."""

    EXERCISED = "exercised"
    LOW_RANGE_ONLY = "low_range_only"
    PRESENT_UNEXERCISED = "present_unexercised"
    ABSENT = "absent"


#: Where the exercise states below were established. Any consumer reasoning
#: about signal trustworthiness should be able to name this.
PROVENANCE_SOURCE = (
    "tests/fixtures/vllm-0.26.0 -- RTX 3060 Ti (8GB, driver 595.71.05), "
    "Qwen2.5-0.5B-Instruct, 176 scrapes across idle/single/burst/saturated. "
    "Exercise states describe THAT hardware and workload only; re-establish "
    "them from a fresh capture on different hardware."
)


@dataclass(frozen=True)
class MetricSpec:
    name: str
    kind: str  # gauge | counter | histogram
    exercise: Exercise
    purpose: str
    note: str = ""


EXPECTED_METRICS: tuple[MetricSpec, ...] = (
    # --- queue and batch state: the controller's primary pressure signal ---
    MetricSpec(
        "vllm:num_requests_running",
        "gauge",
        Exercise.EXERCISED,
        "requests currently in the running batch",
        "fixtures: 0 / 1 / 16 / 4 across idle/single/burst/saturated",
    ),
    MetricSpec(
        "vllm:num_requests_waiting",
        "gauge",
        Exercise.EXERCISED,
        "requests queued and not yet scheduled",
        "fixtures: only non-zero under `saturated` (max 28); the default batch "
        "size never queued on this hardware",
    ),
    MetricSpec(
        "vllm:num_requests_waiting_by_reason",
        "gauge",
        Exercise.EXERCISED,
        "queue depth split by cause; labelled `reason`",
        "fixtures: reason=capacity reached 28; reason=deferred never left 0",
    ),
    MetricSpec(
        "vllm:kv_cache_usage_perc",
        "gauge",
        Exercise.LOW_RANGE_ONLY,
        "fraction of KV cache in use",
        "fixtures: max 0.00498 (~0.5%). Short prompts against 391,952 tokens "
        "of cache. Behaviour under memory pressure is UNVALIDATED here.",
    ),
    MetricSpec(
        "vllm:num_preemptions_total",
        "counter",
        Exercise.PRESENT_UNEXERCISED,
        "requests preempted due to insufficient cache",
        "fixtures: flat zero in all four conditions. Never observed non-zero.",
    ),
    # --- batch composition: prefill vs decode, as rates not as a flag ---
    MetricSpec(
        "vllm:prompt_tokens_total",
        "counter",
        Exercise.EXERCISED,
        "cumulative prefill tokens; numerator of prefill share",
    ),
    MetricSpec(
        "vllm:generation_tokens_total",
        "counter",
        Exercise.EXERCISED,
        "cumulative decode tokens; numerator of decode share",
    ),
    MetricSpec(
        "vllm:iteration_tokens_total",
        "histogram",
        Exercise.EXERCISED,
        "tokens per engine iteration; _sum/_count give mean batch token count",
    ),
    # --- cost attribution ---
    MetricSpec(
        "vllm:request_prefill_time_seconds",
        "histogram",
        Exercise.EXERCISED,
        "cumulative prefill wall time",
    ),
    MetricSpec(
        "vllm:request_decode_time_seconds",
        "histogram",
        Exercise.EXERCISED,
        "cumulative decode wall time",
    ),
    # --- cache effectiveness: confounds energy-per-token comparisons ---
    MetricSpec(
        "vllm:prefix_cache_queries_total",
        "counter",
        Exercise.EXERCISED,
        "prefix cache lookups",
    ),
    MetricSpec(
        "vllm:prefix_cache_hits_total",
        "counter",
        Exercise.EXERCISED,
        "prefix cache hits; a high hit rate makes tokens artificially cheap",
    ),
)

_SPECS_BY_NAME = {m.name: m for m in EXPECTED_METRICS}

# name{label="v",...} value    /    name value
_SAMPLE = re.compile(r"^(?P<name>[a-zA-Z_:][\w:]*)(?:\{(?P<labels>.*)\})?\s+(?P<value>.+)$")
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


@dataclass
class ParsedExposition:
    """Raw parse of a Prometheus text exposition. No interpretation."""

    #: (metric_name, frozenset of (label, value)) -> float
    samples: dict = field(default_factory=dict)
    types: dict = field(default_factory=dict)
    parse_errors: list = field(default_factory=list)

    def get(self, name: str, **labels) -> float | None:
        for (n, lbls), v in self.samples.items():
            if n != name:
                continue
            if all((k, str(val)) in lbls for k, val in labels.items()):
                return v
        return None

    def all_labelled(self, name: str, label: str) -> dict:
        out = {}
        for (n, lbls), v in self.samples.items():
            if n != name:
                continue
            for k, val in lbls:
                if k == label:
                    out[val] = v
        return out

    def has_family(self, name: str) -> bool:
        """True if the family appears, including via histogram child series."""
        if name in self.types:
            return True
        suffixes = ("", "_sum", "_count", "_bucket", "_total", "_created")
        for n, _ in self.samples:
            for suf in suffixes:
                if n == name + suf:
                    return True
        return False


def parse_exposition(text: str | bytes) -> ParsedExposition:
    """Parse Prometheus text exposition format.

    Tolerant by design: a malformed line is recorded in ``parse_errors`` and
    skipped rather than aborting the scrape. A partially readable scrape is
    more useful than none, provided the damage is visible.
    """
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")

    parsed = ParsedExposition()
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            parts = line.split(None, 3)
            if len(parts) >= 4 and parts[1] == "TYPE":
                parsed.types[parts[2]] = parts[3]
            continue

        m = _SAMPLE.match(line)
        if not m:
            parsed.parse_errors.append({"line": lineno, "text": raw[:200]})
            continue

        raw_value = m.group("value").split()[0]
        try:
            value = float(raw_value)
        except ValueError:
            # Prometheus permits Nan/+Inf; anything else is a real problem.
            low = raw_value.lower()
            if low in ("nan", "+inf", "-inf", "inf"):
                value = float(low.replace("+", ""))
            else:
                parsed.parse_errors.append({"line": lineno, "text": raw[:200]})
                continue

        labels = frozenset(
            (k, v) for k, v in _LABEL.findall(m.group("labels") or "")
        )
        parsed.samples[(m.group("name"), labels)] = value

    return parsed


@dataclass
class WorkloadSnapshot:
    """One scrape, interpreted. Missing values stay None -- never defaulted."""

    model_name: str | None = None
    engine: str | None = None

    num_requests_running: float | None = None
    num_requests_waiting: float | None = None
    waiting_by_reason: dict = field(default_factory=dict)
    kv_cache_usage_perc: float | None = None
    num_preemptions_total: float | None = None

    prompt_tokens_total: float | None = None
    generation_tokens_total: float | None = None
    iteration_tokens_sum: float | None = None
    iteration_tokens_count: float | None = None

    prefill_time_seconds_total: float | None = None
    decode_time_seconds_total: float | None = None

    prefix_cache_queries_total: float | None = None
    prefix_cache_hits_total: float | None = None

    #: Expected metric families not found in this scrape.
    missing: list = field(default_factory=list)
    #: Present-and-parsed but never observed non-zero in the fixtures that
    #: established provenance. Not an error -- a caveat that travels with the data.
    unvalidated: list = field(default_factory=list)
    parse_errors: list = field(default_factory=list)

    @property
    def total_requests(self) -> float | None:
        if self.num_requests_running is None or self.num_requests_waiting is None:
            return None
        return self.num_requests_running + self.num_requests_waiting

    @property
    def mean_iteration_tokens(self) -> float | None:
        if not self.iteration_tokens_count:
            return None
        if self.iteration_tokens_sum is None:
            return None
        return self.iteration_tokens_sum / self.iteration_tokens_count

    @property
    def prefix_cache_hit_rate(self) -> float | None:
        if not self.prefix_cache_queries_total:
            return None
        if self.prefix_cache_hits_total is None:
            return None
        return self.prefix_cache_hits_total / self.prefix_cache_queries_total

    def is_healthy(self) -> bool:
        """True when every expected family was found and the text parsed cleanly."""
        return not self.missing and not self.parse_errors


def read_snapshot(text: str | bytes) -> WorkloadSnapshot:
    """Parse a raw /metrics scrape into a :class:`WorkloadSnapshot`."""
    p = parse_exposition(text)
    snap = WorkloadSnapshot(parse_errors=list(p.parse_errors))

    for spec in EXPECTED_METRICS:
        if not p.has_family(spec.name):
            snap.missing.append(spec.name)
        elif spec.exercise in (Exercise.PRESENT_UNEXERCISED, Exercise.LOW_RANGE_ONLY):
            snap.unvalidated.append(
                {
                    "metric": spec.name,
                    "state": spec.exercise.value,
                    "note": spec.note,
                    "provenance": PROVENANCE_SOURCE,
                }
            )

    # Identity labels, taken from whichever family is present.
    for (name, lbls) in p.samples:
        if name.startswith("vllm:"):
            d = dict(lbls)
            snap.model_name = snap.model_name or d.get("model_name")
            snap.engine = snap.engine or d.get("engine")
            if snap.model_name and snap.engine:
                break

    snap.num_requests_running = p.get("vllm:num_requests_running")
    snap.num_requests_waiting = p.get("vllm:num_requests_waiting")
    snap.waiting_by_reason = p.all_labelled(
        "vllm:num_requests_waiting_by_reason", "reason"
    )
    snap.kv_cache_usage_perc = p.get("vllm:kv_cache_usage_perc")
    snap.num_preemptions_total = p.get("vllm:num_preemptions_total")

    snap.prompt_tokens_total = p.get("vllm:prompt_tokens_total")
    snap.generation_tokens_total = p.get("vllm:generation_tokens_total")
    snap.iteration_tokens_sum = p.get("vllm:iteration_tokens_total_sum")
    snap.iteration_tokens_count = p.get("vllm:iteration_tokens_total_count")

    snap.prefill_time_seconds_total = p.get("vllm:request_prefill_time_seconds_sum")
    snap.decode_time_seconds_total = p.get("vllm:request_decode_time_seconds_sum")

    snap.prefix_cache_queries_total = p.get("vllm:prefix_cache_queries_total")
    snap.prefix_cache_hits_total = p.get("vllm:prefix_cache_hits_total")

    return snap


@dataclass
class Composition:
    """Batch composition over the interval between two snapshots.

    Deliberately a share, not a label. Chunked prefill mixes prefill and decode
    work inside one engine iteration, so no scrape can report "the batch is
    prefilling" -- only how much of the interval's token work was each kind.
    """

    prompt_tokens: float | None = None
    generation_tokens: float | None = None
    prefill_seconds: float | None = None
    decode_seconds: float | None = None
    #: Fraction of tokens in the interval that were prefill. None if no work.
    prefill_token_share: float | None = None
    #: Fraction of engine time in the interval spent on prefill.
    prefill_time_share: float | None = None
    #: True if a counter went backwards -- the server restarted mid-run.
    counter_reset: bool = False
    #: Deltas cannot be computed when either side is missing.
    incomplete: list = field(default_factory=list)


def _delta(prev, cur, label: str, incomplete: list) -> tuple[float | None, bool]:
    if prev is None or cur is None:
        incomplete.append(label)
        return None, False
    if cur < prev:
        return None, True
    return cur - prev, False


def compose(prev: WorkloadSnapshot, cur: WorkloadSnapshot) -> Composition:
    """Diff two snapshots into an interval composition."""
    inc: list = []
    comp = Composition(incomplete=inc)
    reset = False

    comp.prompt_tokens, r1 = _delta(
        prev.prompt_tokens_total, cur.prompt_tokens_total, "prompt_tokens_total", inc
    )
    comp.generation_tokens, r2 = _delta(
        prev.generation_tokens_total,
        cur.generation_tokens_total,
        "generation_tokens_total",
        inc,
    )
    comp.prefill_seconds, r3 = _delta(
        prev.prefill_time_seconds_total,
        cur.prefill_time_seconds_total,
        "request_prefill_time_seconds_sum",
        inc,
    )
    comp.decode_seconds, r4 = _delta(
        prev.decode_time_seconds_total,
        cur.decode_time_seconds_total,
        "request_decode_time_seconds_sum",
        inc,
    )
    reset = any((r1, r2, r3, r4))
    comp.counter_reset = reset
    if reset:
        # A counter reset makes every delta in this interval meaningless.
        comp.prompt_tokens = comp.generation_tokens = None
        comp.prefill_seconds = comp.decode_seconds = None
        return comp

    if comp.prompt_tokens is not None and comp.generation_tokens is not None:
        tot = comp.prompt_tokens + comp.generation_tokens
        comp.prefill_token_share = (comp.prompt_tokens / tot) if tot > 0 else None

    if comp.prefill_seconds is not None and comp.decode_seconds is not None:
        tot = comp.prefill_seconds + comp.decode_seconds
        comp.prefill_time_share = (comp.prefill_seconds / tot) if tot > 0 else None

    return comp


def describe_provenance(specs: Iterable[MetricSpec] = EXPECTED_METRICS) -> str:
    """Human-readable provenance table. Belongs in any report citing these signals."""
    lines = [f"signal provenance: {PROVENANCE_SOURCE}", ""]
    for s in sorted(specs, key=lambda m: (m.exercise.value, m.name)):
        lines.append(f"  {s.exercise.value:<20s} {s.name}")
        if s.note:
            lines.append(f"  {'':<20s}   {s.note}")
    return "\n".join(lines)
