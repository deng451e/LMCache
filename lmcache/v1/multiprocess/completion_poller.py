# SPDX-License-Identifier: Apache-2.0
"""One thread that watches host-side completions (transfer-channel reads,
recorded events) so no worker blocks on them. It backs off from 50 us to 1 ms
while nothing completes, sleeps while it watches nothing, and runs each
callback on the affinity worker that asked (else on its own thread)."""

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

    def deliver(self, callback: Callable[[], None]) -> None:
        if self.post is not None:
            self.post(callback)
        else:
            callback()


class CompletionPoller:
    """Watches host-side completions on one thread named ``name``."""

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
        """Run ``on_ready`` once the non-blocking ``check()`` returns True, or
        ``on_abandon`` instead if it raises or the poller stops first. Both
        run on the calling affinity worker and handle their own exceptions."""
        job = _Job(check, on_ready, on_abandon, current_worker_poster())
        with self._cond:
            if not self._stopped:
                self._jobs.append(job)
                self._cond.notify()
                return
        job.deliver(on_abandon)

    def stop(self) -> None:
        """Abandon every watched job and stop the thread; idempotent."""
        with self._cond:
            self._stopped = True
            jobs, self._jobs = self._jobs, []
            self._cond.notify()
        for job in jobs:
            job.deliver(job.on_abandon)
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
                    if not job.check():
                        continue
                    callback = job.on_ready
                except Exception:
                    logger.exception("Completion check failed; abandoning the job")
                    callback = job.on_abandon
                with self._cond:
                    if job not in self._jobs:
                        continue  # abandoned by stop()
                    self._jobs.remove(job)
                job.deliver(callback)
                progressed = True
            if progressed:
                delay = POLL_MIN_S
            else:
                time.sleep(delay)
                delay = min(delay * 2, POLL_MAX_S)
