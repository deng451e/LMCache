# SPDX-License-Identifier: Apache-2.0
"""HostPoller: non-blocking completion checks on one thread, results handed
back to the affinity worker that asked."""

# Standard
from typing import Callable
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.affinity_pool import AffinityThreadPool
from lmcache.v1.multiprocess.host_poller import HostPoller


@pytest.fixture
def poller():
    p = HostPoller()
    yield p
    p.stop()


class _Outcome:
    """Records which callback ran, on which thread, and how often."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.done = threading.Event()

    def callback(self, name: str) -> Callable[[], None]:
        def run() -> None:
            self.calls.append((name, threading.current_thread().name))
            self.done.set()

        return run


def test_on_ready_runs_once_after_the_check_passes(poller: HostPoller) -> None:
    ready = threading.Event()
    outcome = _Outcome()
    poller.watch(ready.is_set, outcome.callback("ready"), outcome.callback("abandon"))
    time.sleep(0.01)
    assert outcome.calls == []
    ready.set()
    assert outcome.done.wait(5)
    time.sleep(0.01)
    assert [name for name, _ in outcome.calls] == ["ready"]


def test_callbacks_run_on_the_affinity_worker_that_watched(
    poller: HostPoller,
) -> None:
    pool = AffinityThreadPool(1, thread_name_prefix="peer-retrieve")
    outcome = _Outcome()
    ready = threading.Event()
    try:
        pool.submit(
            lambda: poller.watch(
                ready.is_set, outcome.callback("ready"), outcome.callback("abandon")
            ),
            affinity_key=1,
        ).result(timeout=5)
        ready.set()
        assert outcome.done.wait(5)
    finally:
        pool.shutdown()
    assert outcome.calls == [("ready", "peer-retrieve-0")]


def test_a_check_that_raises_abandons_the_job(poller: HostPoller) -> None:
    def broken() -> bool:
        raise RuntimeError("query failed")

    outcome = _Outcome()
    poller.watch(broken, outcome.callback("ready"), outcome.callback("abandon"))
    assert outcome.done.wait(5)
    assert [name for name, _ in outcome.calls] == ["abandon"]


def test_stop_abandons_pending_and_later_jobs() -> None:
    poller = HostPoller()
    pending, late = _Outcome(), _Outcome()
    poller.watch(lambda: False, pending.callback("ready"), pending.callback("abandon"))
    poller.stop()
    poller.watch(lambda: True, late.callback("ready"), late.callback("abandon"))
    assert pending.done.wait(5) and late.done.wait(5)
    assert [n for n, _ in pending.calls] == ["abandon"]
    assert [n for n, _ in late.calls] == ["abandon"]
    poller.stop()


def test_polling_backs_off_while_nothing_completes(poller: HostPoller) -> None:
    """At a fixed 50 us a 100 ms wait would take ~2000 checks; backing off to
    1 ms needs about 100."""
    checks = [0]

    def never() -> bool:
        checks[0] += 1
        return False

    outcome = _Outcome()
    poller.watch(never, outcome.callback("ready"), outcome.callback("abandon"))
    time.sleep(0.1)
    assert 1 <= checks[0] < 300


def test_completed_jobs_are_not_checked_again(poller: HostPoller) -> None:
    checks = [0]

    def once() -> bool:
        checks[0] += 1
        return True

    outcome = _Outcome()
    poller.watch(once, outcome.callback("ready"), outcome.callback("abandon"))
    assert outcome.done.wait(5)
    time.sleep(0.02)
    assert checks[0] == 1
