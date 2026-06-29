"""Subprocess entry point for one Inspect video run."""

from __future__ import annotations

import signal
import sys
import threading


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m app.inspect.worker <run_id>", file=sys.stderr)
        return 2
    run_id = int(sys.argv[1])

    from app.logging_setup import configure_worker_logging

    configure_worker_logging()

    abort_event = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: abort_event.set())
    signal.signal(signal.SIGINT, lambda *_: abort_event.set())

    from app.inspect.engine import run_inspect

    run_inspect(run_id, abort_event)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
