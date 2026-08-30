from __future__ import annotations

import signal
import sys
import threading


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m app.analysis.image_distribution_worker <run_id>", file=sys.stderr)
        return 2
    abort_event = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: abort_event.set())
    signal.signal(signal.SIGINT, lambda *_: abort_event.set())

    from app.logging_setup import configure_worker_logging
    from app.analysis.image_distribution import run_scheduled

    configure_worker_logging()
    run_scheduled(int(sys.argv[1]), abort_event)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
