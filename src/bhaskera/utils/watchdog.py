"""
bhaskera.utils.watchdog
=======================
Fail fast when a worker never completes its first training step.

Why: two 8-node Qwen3-14B runs sat for 3.5h and 1.4h with no training step
and no error -- the job only ever looked RUNNING, and the only visible signal
was Ray Data's split progress bar at 0 rows. Nothing in the stack bounds this
on its own: the dataset split waits on its coordinator indefinitely, NCCL
blocking-wait timeouts are an hour or more, and a large model load can
legitimately take a long time.

StartupWatchdog bounds time from worker start to the first completed
forward+backward window. While
waiting it logs which phase each rank is in; on expiry it dumps every
thread's Python stack (so the log shows *where* the hang is) and exits the
worker process. Ray Train then reports the worker failure and applies its own
max_failures policy.
"""
from __future__ import annotations

import faulthandler
import logging
import os
import sys
import threading
import time

logger = logging.getLogger(__name__)

STARTUP_TIMEOUT_EXIT_CODE = 86


class StartupWatchdog:
    def __init__(self, timeout_s: float, rank: int, heartbeat_s: float = 600.0) -> None:
        self.timeout_s = float(timeout_s or 0)
        self.rank = rank
        self.heartbeat_s = heartbeat_s
        self._done = threading.Event()
        self._start = self._phase_start = time.monotonic()
        self._phase = "worker start"
        self._thread: threading.Thread | None = None

    @property
    def enabled(self) -> bool:
        return self.timeout_s > 0

    def start(self) -> StartupWatchdog:
        if self.enabled and self._thread is None:
            self._thread = threading.Thread(
                target=self._run, name="bhaskera-startup-watchdog", daemon=True
            )
            self._thread.start()
        return self

    def phase(self, name: str) -> None:
        now = time.monotonic()
        if self.rank == 0:
            logger.info(
                f"[startup] '{self._phase}' took {now - self._phase_start:.0f}s; now '{name}'"
            )
        self._phase, self._phase_start = name, now

    def done(self) -> None:
        if self._done.is_set():
            return
        self._done.set()
        if self.rank == 0:
            logger.info(
                f"[startup] first training step (forward+backward) completed "
                f"{time.monotonic() - self._start:.0f}s after worker start"
            )

    def _run(self) -> None:
        deadline = self._start + self.timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._expire()
                return
            if self._done.wait(timeout=min(self.heartbeat_s, remaining)):
                return
            now = time.monotonic()
            if now < deadline:
                logger.warning(
                    f"[rank {self.rank}] startup: still in '{self._phase}' after "
                    f"{now - self._phase_start:.0f}s ({now - self._start:.0f}s since worker "
                    f"start; timeout {self.timeout_s:.0f}s)"
                )

    def _expire(self) -> None:
        logger.error(
            f"[rank {self.rank}] STARTUP TIMEOUT: no training step within "
            f"{self.timeout_s:.0f}s. Stuck in '{self._phase}' for "
            f"{time.monotonic() - self._phase_start:.0f}s. Stacks of every thread follow; "
            f"exiting with code {STARTUP_TIMEOUT_EXIT_CODE}. Raise "
            f"training.startup_timeout_s if this phase is legitimately slow."
        )
        for handler in logging.getLogger().handlers:
            try:
                handler.flush()
            except Exception:
                pass
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        sys.stderr.flush()
        os._exit(STARTUP_TIMEOUT_EXIT_CODE)
