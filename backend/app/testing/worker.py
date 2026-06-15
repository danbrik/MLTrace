"""Subprocess entry point for one testing (inference) run.

Launched by the scheduler as ``python -m app.testing.worker <id>``. SIGTERM
(sent on abort) sets an event the engine polls so the run ends as ``aborted``.
"""

from __future__ import annotations

import signal
import sys
import threading


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m app.testing.worker <run_id>", file=sys.stderr)
        return 2
    run_id = int(sys.argv[1])

    abort_event = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: abort_event.set())
    signal.signal(signal.SIGINT, lambda *_: abort_event.set())

    from app.testing.engine import run_testing

    run_testing(run_id, abort_event)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
