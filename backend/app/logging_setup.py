"""Logging setup for scheduler worker subprocesses.

Workers are launched by the scheduler with stdout/stderr redirected into
``worker.log``. Without a configured handler, ``logging`` writes nothing, so
those logs stayed empty even on failures. ``configure_worker_logging`` installs
a single stdout handler so ``logger.info``/``logger.exception`` become visible.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def configure_worker_logging(level: int = logging.INFO) -> None:
    """Idempotently route INFO+ logs to stdout (captured into worker.log)."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    _CONFIGURED = True


def log_device_diagnostics(logger: logging.Logger, gpu_index: int | None) -> None:
    """Log why a worker runs on CPU vs GPU (helps diagnose empty GPU fields)."""
    import os

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        import torch

        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count()) if available else 0
        logger.info(
            "torch %s · cuda_available=%s · device_count=%s · CUDA_VISIBLE_DEVICES=%r · gpu_index=%s",
            torch.__version__,
            available,
            count,
            visible,
            gpu_index,
        )
    except Exception as exc:  # noqa: BLE001 - torch optional; diagnostics must never crash a run
        logger.info("torch unavailable (%s) · CUDA_VISIBLE_DEVICES=%r · gpu_index=%s", exc, visible, gpu_index)
