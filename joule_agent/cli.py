"""Console-script dispatch for ``joule``.

The documented invocation is ``joule benchmark --endpoint ...``. Rather than
build a subcommand tree around a tool that currently has one command, this
accepts the ``benchmark`` word and forwards everything after it. Adding a real
second command means turning this into an argparse subparser -- do that when a
second command exists, not before.
"""

from __future__ import annotations

import sys

COMMANDS = {"benchmark", "sweep", "guard", "telemetry"}

_USAGE = """usage: joule <command> [options]

commands:
  benchmark    measure tokens-per-joule and find the lowest-energy clock that
               still meets a latency budget  (the one you want)
  sweep        the raw static sweep, including research-only flags
  guard        inspect or restore GPU clock/power state
  telemetry    record GPU power/clock samples

`joule benchmark --help` for the full option list.
"""


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0] in ("-h", "--help", "help"):
        print(_USAGE)
        return 0
    if not argv:
        print(_USAGE, file=sys.stderr)
        return 2

    cmd = argv[0] if argv[0] in COMMANDS else "benchmark"
    rest = argv[1:] if argv[0] in COMMANDS else argv

    if cmd == "benchmark":
        from joule_agent.benchmark import main as run
    elif cmd == "sweep":
        from joule_agent.sweep import main as run
    elif cmd == "guard":
        from joule_agent.gpu_guard import main as run
    else:
        from joule_agent.telemetry import main as run
    return run(rest)


if __name__ == "__main__":
    raise SystemExit(main())
