"""Subprocess entry point for one training run.

Launched by the scheduler as ``python -m app.training.worker <run_id>`` with
``CUDA_VISIBLE_DEVICES`` pinned to a single GPU. A SIGTERM (sent on abort) sets
an event the engine polls so the run terminates cleanly as ``aborted``.
"""

from __future__ import annotations

import signal
import sys
import threading
import logging
import os


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m app.training.worker <run_id>", file=sys.stderr)
        return 2
    run_id = int(sys.argv[1])

    from app.logging_setup import configure_worker_logging

    configure_worker_logging()
    logger = logging.getLogger("mltrace.training.worker")
    logger.info("Training worker booted for run %s (pid=%s, cwd=%s)", run_id, os.getpid(), os.getcwd())

    abort_event = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: abort_event.set())
    signal.signal(signal.SIGINT, lambda *_: abort_event.set())

    from app.training.engine import run_training

    logger.info("Training worker entering engine for run %s", run_id)
    run_training(run_id, abort_event)
    logger.info("Training worker finished engine for run %s", run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
