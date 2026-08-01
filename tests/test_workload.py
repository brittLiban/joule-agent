"""Tests for the workload signal reader.

Driven entirely by the frozen fixtures in tests/fixtures/vllm-0.26.0/ -- no
vLLM, no GPU, no network, matching the telemetry suite.

The fixtures are ground truth captured from a real server. Where a test
asserts a concrete number it comes from those scrapes, not from what vLLM's
documentation says the metric should be.
"""

import glob
from pathlib import Path

import pytest

from joule_agent.workload import (
    EXPECTED_METRICS,
    Exercise,
    compose,
    describe_provenance,
    parse_exposition,
    read_snapshot,
)

FIXTURES = Path(__file__).parent / "fixtures" / "vllm-0.26.0"


def scrapes(condition: str):
    files = sorted(glob.glob(str(FIXTURES / f"metrics_{condition}_*.txt")))
    assert files, f"no fixtures for {condition}"
    return files


def load(condition: str, idx: int = 0) -> str:
    return Path(scrapes(condition)[idx]).read_text()


# ---------------------------------------------------------------------------
# exposition parsing
# ---------------------------------------------------------------------------


def test_parses_real_scrape_without_errors():
    p = parse_exposition(load("saturated", 30))
    assert p.parse_errors == [], f"unexpected parse errors: {p.parse_errors[:3]}"
    assert len(p.samples) > 100
    assert p.types["vllm:num_requests_running"] == "gauge"


def test_every_fixture_parses_cleanly():
    """All 176 captured scrapes, not just a representative one."""
    bad = []
    for cond in ("idle", "single", "burst", "saturated"):
        for fp in scrapes(cond):
            p = parse_exposition(Path(fp).read_text())
            if p.parse_errors:
                bad.append((Path(fp).name, p.parse_errors[:1]))
    assert not bad, f"{len(bad)} fixtures failed to parse: {bad[:3]}"


def test_labels_are_parsed():
    p = parse_exposition(load("saturated", 30))
    assert p.get("vllm:num_requests_waiting_by_reason", reason="capacity") == 28.0
    assert p.get("vllm:num_requests_waiting_by_reason", reason="deferred") == 0.0


def test_malformed_line_is_recorded_not_fatal():
    text = "# TYPE x gauge\nx 1.0\nthis is not a metric line {{{\ny 2.0\n"
    p = parse_exposition(text)
    assert p.get("x") == 1.0
    assert p.get("y") == 2.0
    assert len(p.parse_errors) == 1


def test_special_float_values():
    p = parse_exposition("a 1.0\nb NaN\nc +Inf\n")
    assert p.get("a") == 1.0
    assert p.get("b") != p.get("b")  # NaN
    assert p.get("c") == float("inf")


def test_accepts_bytes():
    snap = read_snapshot(Path(scrapes("burst")[0]).read_bytes())
    assert snap.num_requests_running is not None


# ---------------------------------------------------------------------------
# snapshot: values match the captured ground truth
# ---------------------------------------------------------------------------


def test_snapshot_identity_labels():
    snap = read_snapshot(load("burst", 10))
    assert snap.model_name == "Qwen/Qwen2.5-0.5B-Instruct"
    assert snap.engine == "0"


@pytest.mark.parametrize(
    "condition,expected_running",
    [("idle", 0.0), ("single", 1.0), ("burst", 16.0), ("saturated", 4.0)],
)
def test_running_count_matches_condition(condition, expected_running):
    """Each captured condition produces a distinguishable running-batch size."""
    observed = {read_snapshot(Path(f).read_text()).num_requests_running for f in scrapes(condition)}
    assert expected_running in observed, f"{condition}: saw {sorted(observed)}"


def test_queue_depth_only_nonzero_under_saturation():
    """The fixture set's central finding, pinned as a test.

    A default batch size never queued on this hardware; only the engineered
    saturated condition exercises the waiting signal at all.
    """
    maxima = {}
    for cond in ("idle", "single", "burst", "saturated"):
        maxima[cond] = max(
            read_snapshot(Path(f).read_text()).num_requests_waiting for f in scrapes(cond)
        )
    assert maxima["idle"] == 0
    assert maxima["single"] == 0
    assert maxima["burst"] == 0
    assert maxima["saturated"] == 28


def test_waiting_by_reason_attributes_the_queue():
    snap = read_snapshot(load("saturated", 30))
    assert snap.waiting_by_reason["capacity"] == 28.0
    assert snap.waiting_by_reason["deferred"] == 0.0
    assert sum(snap.waiting_by_reason.values()) == snap.num_requests_waiting


def test_total_requests_combines_running_and_waiting():
    snap = read_snapshot(load("saturated", 30))
    assert snap.total_requests == 32.0  # 4 running + 28 queued == 32 clients


def test_healthy_scrape_reports_no_missing_metrics():
    snap = read_snapshot(load("burst", 20))
    assert snap.missing == [], f"missing: {snap.missing}"
    assert snap.is_healthy()


def test_all_expected_metrics_present_in_every_fixture():
    """Guards the EXPECTED_METRICS constant against drift from ground truth."""
    for cond in ("idle", "single", "burst", "saturated"):
        for fp in scrapes(cond):
            snap = read_snapshot(Path(fp).read_text())
            assert not snap.missing, f"{Path(fp).name} missing {snap.missing}"


# ---------------------------------------------------------------------------
# provenance: the three-state honesty contract
# ---------------------------------------------------------------------------


def test_missing_metric_is_reported_not_defaulted():
    text = load("burst", 0)
    stripped = "\n".join(
        l for l in text.splitlines() if "vllm:kv_cache_usage_perc" not in l
    )
    snap = read_snapshot(stripped)
    assert "vllm:kv_cache_usage_perc" in snap.missing
    assert snap.kv_cache_usage_perc is None  # None, never 0.0
    assert not snap.is_healthy()


def test_unvalidated_signals_travel_with_the_snapshot():
    snap = read_snapshot(load("burst", 0))
    names = {u["metric"] for u in snap.unvalidated}
    assert "vllm:num_preemptions_total" in names
    assert "vllm:kv_cache_usage_perc" in names
    for u in snap.unvalidated:
        assert u["provenance"], "an unvalidated signal must name its provenance"


def test_exercise_states_match_what_fixtures_actually_show():
    """The constant must not claim more validation than the fixtures provide."""
    maxima = {}
    for cond in ("idle", "single", "burst", "saturated"):
        for fp in scrapes(cond):
            snap = read_snapshot(Path(fp).read_text())
            for attr, metric in (
                ("num_preemptions_total", "vllm:num_preemptions_total"),
                ("kv_cache_usage_perc", "vllm:kv_cache_usage_perc"),
                ("num_requests_waiting", "vllm:num_requests_waiting"),
            ):
                v = getattr(snap, attr)
                if v is not None:
                    maxima[metric] = max(maxima.get(metric, 0.0), v)

    by_name = {m.name: m for m in EXPECTED_METRICS}
    # Flat zero everywhere -> must be declared PRESENT_UNEXERCISED.
    assert maxima["vllm:num_preemptions_total"] == 0.0
    assert by_name["vllm:num_preemptions_total"].exercise is Exercise.PRESENT_UNEXERCISED
    # Non-zero but tiny -> LOW_RANGE_ONLY, not EXERCISED.
    assert 0 < maxima["vllm:kv_cache_usage_perc"] < 0.01
    assert by_name["vllm:kv_cache_usage_perc"].exercise is Exercise.LOW_RANGE_ONLY
    # Genuinely spans a range -> EXERCISED.
    assert maxima["vllm:num_requests_waiting"] == 28.0
    assert by_name["vllm:num_requests_waiting"].exercise is Exercise.EXERCISED


def test_provenance_description_names_its_hardware():
    text = describe_provenance()
    assert "3060 Ti" in text
    assert "vllm-0.26.0" in text


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------


def test_composition_over_burst_is_decode_dominated():
    """Long generations against short prompts: decode should dominate tokens."""
    files = scrapes("burst")
    prev = read_snapshot(Path(files[10]).read_text())
    cur = read_snapshot(Path(files[40]).read_text())
    comp = compose(prev, cur)
    assert not comp.counter_reset
    assert comp.prompt_tokens > 0 and comp.generation_tokens > 0
    assert 0.0 < comp.prefill_token_share < 0.5, comp.prefill_token_share
    # Prefill processes its tokens in parallel; decode is sequential. So
    # prefill's share of engine TIME is far below its share of TOKENS. On these
    # fixtures: ~14% of tokens, ~1.8% of time. If these ever converge, either
    # the workload changed shape or the two metrics are no longer comparable.
    assert comp.prefill_time_share < comp.prefill_token_share, (
        f"token_share={comp.prefill_token_share:.4f} "
        f"time_share={comp.prefill_time_share:.4f}"
    )


def test_composition_is_a_share_not_a_flag():
    files = scrapes("saturated")
    comp = compose(
        read_snapshot(Path(files[5]).read_text()),
        read_snapshot(Path(files[50]).read_text()),
    )
    assert isinstance(comp.prefill_token_share, float)
    assert 0.0 <= comp.prefill_token_share <= 1.0


def test_idle_interval_has_no_token_work():
    files = scrapes("idle")
    comp = compose(
        read_snapshot(Path(files[0]).read_text()),
        read_snapshot(Path(files[-1]).read_text()),
    )
    assert comp.prompt_tokens == 0
    assert comp.generation_tokens == 0
    assert comp.prefill_token_share is None  # no work -> no share, not 0.0


def test_counter_reset_is_detected_and_poisons_the_interval():
    later = read_snapshot(load("burst", 50))
    earlier = read_snapshot(load("burst", 0))
    comp = compose(later, earlier)  # deliberately backwards
    assert comp.counter_reset
    assert comp.prompt_tokens is None
    assert comp.prefill_token_share is None


def test_missing_counter_makes_interval_incomplete():
    a = read_snapshot(load("burst", 10))
    b = read_snapshot(load("burst", 20))
    b.prompt_tokens_total = None
    comp = compose(a, b)
    assert "prompt_tokens_total" in comp.incomplete
    assert comp.prefill_token_share is None


# ---------------------------------------------------------------------------
# derived signals
# ---------------------------------------------------------------------------


def test_prefix_cache_hit_rate_is_reported():
    """Cache hits make tokens artificially cheap -- it confounds joules/token."""
    snap = read_snapshot(load("burst", 40))
    assert 0.0 <= snap.prefix_cache_hit_rate <= 1.0


def test_mean_iteration_tokens_available():
    snap = read_snapshot(load("burst", 40))
    assert snap.mean_iteration_tokens > 0


def test_derived_signals_are_none_when_inputs_missing():
    snap = read_snapshot(load("idle", 0))
    snap.prefix_cache_queries_total = 0.0
    assert snap.prefix_cache_hit_rate is None
    snap.iteration_tokens_count = 0.0
    assert snap.mean_iteration_tokens is None
