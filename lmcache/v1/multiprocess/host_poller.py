# SPDX-License-Identifier: Apache-2.0
"""One thread that watches host-side completions without blocking a worker.

A handler that starts work completing outside the stream (a transfer-channel
read, a recorded event) hands a non-blocking ``check`` to :class:`HostPoller`
and returns. The poller calls each ``check`` in turn, backing off from 50 us
to 1 ms while nothing completes and sleeping while it watches nothing. When a
check passes, ``on_ready`` runs on the affinity worker that called
:meth:`HostPoller.watch`, so per-instance state is only touched on that
worker; outside a worker it runs on the poller thread.
"""

# Standard
from dataclasses import dataclass
from typing import Callable
import threading
import time

# First Party
from lmcache.v1.common.logging import init_logger
from lmcache.v1.multiprocess.affinity_pool import current_worker_poster

logger = init_logger(__name__)

POLL_MIN_S = 50e-6
POLL_MAX_S = 1e-3


@dataclass(eq=False)
class _Job:
    check: Callable[[], bool]
    on_ready: Callable[[], None]
    on_abandon: Callable[[], None]
    post: Callable[[Callable[[], None]], None] | None


def _deliver(job: _Job, callback: Callable[[], None]) -> None:
    if job.post is not None:
        job.post(callback)
    else:
        callback()


class HostPoller:
    """Watches host-side completions on one thread.

    Args:
        name: Name of the poller thread.
    """

    def __init__(self, name: str = "lmcache-host-poller") -> None:
        self._cond = threading.Condition()
        self._jobs: list[_Job] = []
        self._stopped = False
        self._thread = threading.Thread(target=self._poll, daemon=True, name=name)
        self._thread.start()

    def watch(
        self,
        check: Callable[[], bool],
        on_ready: Callable[[], None],
        on_abandon: Callable[[], None],
    ) -> None:
        """Run ``on_ready`` once ``check()`` returns True.

        Args:
            check: Non-blocking; True once the work completed (successfully
                or not). Called on the poller thread.
            on_ready: Runs once, after ``check`` returned True.
            on_abandon: Runs instead of ``on_ready`` if ``check`` raises or
                the poller stops first.

        Both callbacks run on the affinity worker that called ``watch``, or
        on the poller thread when called outside a worker; they must handle
        their own exceptions. After :meth:`stop`, ``on_abandon`` is posted at
        once.
        """
        job = _Job(check, on_ready, on_abandon, current_worker_poster())
        with self._cond:
            if not self._stopped:
                self._jobs.append(job)
                self._cond.notify()
                return
        _deliver(job, job.on_abandon)

    def stop(self) -> None:
        """Abandon every watched job and stop the thread; idempotent."""
        with self._cond:
            self._stopped = True
            jobs, self._jobs = self._jobs, []
            self._cond.notify()
        for job in jobs:
            _deliver(job, job.on_abandon)
        if self._thread is not threading.current_thread():
            self._thread.join()

    def _poll(self) -> None:
        delay = POLL_MIN_S
        while True:
            with self._cond:
                while not self._jobs and not self._stopped:
                    self._cond.wait()
                    delay = POLL_MIN_S
                if self._stopped:
                    return
                jobs = list(self._jobs)
            progressed = False
            for job in jobs:
                try:
                    ready = job.check()
                except Exception:
                    logger.exception("Completion check failed; abandoning the job")
                    callback = job.on_abandon
                else:
                    if not ready:
                        continue
                    callback = job.on_ready
                with self._cond:
                    if job not in self._jobs:
                        continue  # abandoned by stop()
                    self._jobs.remove(job)
                _deliver(job, callback)
                progressed = True
            if progressed:
                delay = POLL_MIN_S
            else:
                time.sleep(delay)
                delay = min(delay * 2, POLL_MAX_S)
