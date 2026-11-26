# SPDX-License-Identifier: Apache-2.0
"""HostPoller: non-blocking checks on one thread, callbacks on the affinity
worker that asked."""

# Standard
import threading
import time

# First Party
from lmcache.v1.multiprocess.affinity_pool import AffinityThreadPool
from lmcache.v1.multiprocess.host_poller import HostPoller


def _watch(poller: HostPoller, check) -> tuple[list, threading.Event]:
    calls: list = []
    done = threading.Event()

    def callback(name: str):
        return lambda: (
            calls.append((name, threading.current_thread().name)),
            done.set(),
        )

    poller.watch(check, callback("ready"), callback("abandon"))
    return calls, done


def test_ready_runs_once_on_the_worker_that_watched() -> None:
    poller = HostPoller()
    pool = AffinityThreadPool(1, thread_name_prefix="w")
    ready = threading.Event()

    try:
        calls, done = pool.submit(lambda: _watch(poller, ready.is_set)).result(
            timeout=5
        )
        time.sleep(0.05)
        assert calls == []
        ready.set()
        assert done.wait(5)
        time.sleep(0.02)
        assert calls == [("ready", "w-0")]
    finally:
        pool.shutdown()
        poller.stop()


def test_a_raising_check_and_stop_abandon_jobs() -> None:
    poller = HostPoller()

    def broken() -> bool:
        raise RuntimeError("query failed")

    raised, raised_done = _watch(poller, broken)
    pending, _ = _watch(poller, lambda: False)
    assert raised_done.wait(5)
    poller.stop()
    late, _ = _watch(poller, lambda: True)
    assert [c[0] for c in raised + pending + late] == ["abandon"] * 3
