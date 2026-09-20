from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time


def wait_for_dispatch(run_id: int, pid: int) -> bool:
    """Do not write worker status before the parent has committed its PID/device."""
    from app.database import SessionLocal
    from app.models import RepresentationRun
    deadline = time.monotonic() + 65
    while time.monotonic() < deadline:
        with SessionLocal() as db:
            run = db.get(RepresentationRun, run_id)
            if run is None or run.status not in {"queued", "running"}:
                return False
            if run.status == "running" and run.pid == pid:
                return True
        time.sleep(0.05)
    raise RuntimeError("Scheduler did not confirm worker dispatch.")


def main() -> int:
    if len(sys.argv) != 2:
        return 2
    from app.logging_setup import configure_worker_logging
    from app.analysis.dinov3_service import AbortedError, run_scheduled
    abort_event = threading.Event()

    def abort(*_):
        abort_event.set()
        raise AbortedError()

    signal.signal(signal.SIGTERM, abort)
    signal.signal(signal.SIGINT, abort)
    configure_worker_logging()
    try:
        if wait_for_dispatch(int(sys.argv[1]), os.getpid()):
            run_scheduled(int(sys.argv[1]), abort_event)
    except AbortedError:
        # A signal can arrive while waiting for the dispatch transaction.
        run_scheduled(int(sys.argv[1]), abort_event)
    except Exception:
        logging.getLogger(__name__).exception("Worker startup failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
