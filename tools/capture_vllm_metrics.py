"""Capture raw vLLM /metrics scrapes as test fixtures.

Purpose: produce ground truth. vLLM's metric names and semantics have changed
across releases, so the workload-signal reader (component 2) is written against
scrapes this server actually emitted, never against remembered or documented
metric names. The captured bytes are stored verbatim -- no parsing, no
filtering, no reformatting -- because a fixture's only value is being exactly
what the server produced.

Scrapes are taken *during* load, on a loop. Queue depth and waiting counts
collapse the instant a burst drains, so a single scrape taken after the
requests return describes an idle server and is worthless for the signal we
care about. Every condition produces a time series, not one sample.

Stdlib only: the fixture capture must not depend on the client libraries whose
behaviour it is trying to pin down.

Usage:
    python tools/capture_vllm_metrics.py --condition idle   --duration 6
    python tools/capture_vllm_metrics.py --condition single --duration 20
    python tools/capture_vllm_metrics.py --condition burst  --duration 30 --concurrency 16
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def http_get(url: str, timeout: float = 10.0) -> tuple[int, bytes]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read()


def http_post_json(url: str, payload: dict, timeout: float = 300.0) -> tuple[int, bytes]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, method="POST", headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read()


def wait_for_server(endpoint: str, timeout_s: float = 900.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            status, _ = http_get(f"{endpoint}/health", timeout=5.0)
            if status == 200:
                return True
        except Exception:
            pass
        time.sleep(2.0)
    return False


def discover_model(endpoint: str) -> str | None:
    try:
        _, body = http_get(f"{endpoint}/v1/models")
        data = json.loads(body)
        return data["data"][0]["id"]
    except Exception:
        return None


class LoadDriver:
    """Drives completion requests against the server on background threads."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        concurrency: int,
        max_tokens: int,
        prompt: str,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.concurrency = concurrency
        self.max_tokens = max_tokens
        self.prompt = prompt
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.completed = 0
        self.failed = 0
        self._lock = threading.Lock()

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                http_post_json(
                    f"{self.endpoint}/v1/completions",
                    {
                        "model": self.model,
                        "prompt": self.prompt,
                        "max_tokens": self.max_tokens,
                        "temperature": 0.0,
                    },
                )
                with self._lock:
                    self.completed += 1
            except Exception:
                with self._lock:
                    self.failed += 1
                if self._stop.is_set():
                    return
                time.sleep(0.2)

    def start(self) -> None:
        for _ in range(self.concurrency):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self, timeout: float = 30.0) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=timeout / max(1, len(self._threads)))


def capture(args) -> int:
    endpoint = args.endpoint.rstrip("/")

    print(f"waiting for {endpoint}/health ...", flush=True)
    if not wait_for_server(endpoint, args.startup_timeout):
        print("server never became healthy", flush=True)
        return 1

    model = args.model or discover_model(endpoint)
    if model is None:
        print("could not determine served model", flush=True)
        return 1
    print(f"model: {model}", flush=True)

    # Version comes from the server's own package, not from a flag, so the
    # fixture cannot be mislabelled.
    try:
        import vllm

        vllm_version = vllm.__version__
    except Exception:
        vllm_version = "unknown"

    out_dir = Path(args.out_dir) / f"vllm-{vllm_version}"
    out_dir.mkdir(parents=True, exist_ok=True)

    driver = None
    if args.concurrency > 0:
        driver = LoadDriver(
            endpoint,
            model,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            prompt=args.prompt,
        )
        driver.start()
        # Let requests reach the engine so the first scrape sees real state.
        time.sleep(args.warmup)

    scrapes = []
    t0 = time.monotonic()
    seq = 0
    try:
        while time.monotonic() - t0 < args.duration:
            wall = time.time()
            try:
                status, body = http_get(f"{endpoint}/metrics")
            except Exception as exc:
                print(f"  scrape {seq} failed: {type(exc).__name__}", flush=True)
                time.sleep(args.interval)
                continue
            if status != 200:
                print(f"  scrape {seq} status {status}", flush=True)
                time.sleep(args.interval)
                continue

            name = f"metrics_{args.condition}_{seq:03d}.txt"
            # Raw bytes, written verbatim. Anything else defeats the fixture.
            (out_dir / name).write_bytes(body)
            scrapes.append(
                {
                    "seq": seq,
                    "file": name,
                    "wall_us": int(wall * 1e6),
                    "monotonic_s": round(time.monotonic() - t0, 4),
                    "bytes": len(body),
                }
            )
            seq += 1
            time.sleep(args.interval)
    finally:
        if driver is not None:
            driver.stop()

    meta_path = out_dir / f"capture_{args.condition}.json"
    meta = {
        "condition": args.condition,
        "vllm_version": vllm_version,
        "model": model,
        "endpoint": endpoint,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "load": {
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "prompt_chars": len(args.prompt),
            "requests_completed": driver.completed if driver else 0,
            "requests_failed": driver.failed if driver else 0,
        },
        "scrape_interval_s": args.interval,
        "scrapes": scrapes,
        "note": (
            "Raw /metrics bytes, unmodified. Scrapes taken during load, not "
            "after it drained. Server flags are recorded in serve_flags.txt "
            "alongside these fixtures."
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\ncaptured {len(scrapes)} scrapes -> {out_dir}", flush=True)
    if driver:
        print(
            f"requests: {driver.completed} completed, {driver.failed} failed",
            flush=True,
        )
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--endpoint", default="http://localhost:8000")
    p.add_argument("--out-dir", default="tests/fixtures")
    p.add_argument("--condition", required=True, help="idle | single | burst")
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--interval", type=float, default=0.5)
    p.add_argument("--concurrency", type=int, default=0, help="0 = no load (idle)")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--warmup", type=float, default=1.5)
    p.add_argument("--startup-timeout", type=float, default=900.0)
    p.add_argument("--model", default=None)
    p.add_argument(
        "--prompt",
        default="Explain in detail how a GPU executes a matrix multiplication, "
        "step by step, including memory hierarchy considerations.",
    )
    return capture(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
