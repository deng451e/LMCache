# SPDX-License-Identifier: Apache-2.0
"""NIC DMA engine: reads objects a peer owns into GPU memory without blocking.

Transfers return the runs of peer-owned objects (:class:`DeferredNicRun`).
:meth:`NicDmaEngine.read` reads them in waves into a NIC staging buffer that
nothing on the stream uses, queues each wave's kernels once its reads landed,
and frees the slots once those kernels finished. A :class:`HostPoller` checks
the reads and events and hands each step to the instance's affinity worker;
no stream waits on a host-written value. See
``docs/design/v1/distributed/l2_adapters/gdr_peer_l2_adapter.md``.
"""

# Standard
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Sequence
import threading

# First Party
from lmcache import torch_dev
from lmcache.v1.common.logging import init_logger
from lmcache.v1.distributed.transfer_channel import (
    TransferChannelStagingRegistration,
    get_transfer_channel_context,
)
from lmcache.v1.memory_management import (
    BorrowedMemoryObj,
    BorrowedReader,
    PendingRead,
)
from lmcache.v1.multiprocess.affinity_pool import DeferredResponse
from lmcache.v1.multiprocess.host_poller import HostPoller
from lmcache.v1.platform.base.cache_context import BaseCacheContext
from lmcache.v1.platform.base.object_group_transfer import (
    DeferredNicRun,
    StagingBuffer,
)
from lmcache.v1.platform.stream import record_completion_event

logger = init_logger(__name__)

# Slot bytes per context: 384-512 MiB in flight filled a 400 Gb/s link in the
# module benchmark (PR #6411) for 6 MiB and 80 MiB chunks alike.
NIC_STAGING_BYTES = 512 << 20
# Rows after the slots for an unfull chunk: one row plus at most one lead
# block per kernel group.
_SPARE_ROWS = 2


def _complete(
    cache_context: BaseCacheContext,
    response: DeferredResponse,
    finish: Callable[[bool], object],
    succeeded: bool,
) -> None:
    """Complete ``response`` with ``finish(succeeded)``, called under the
    context's device and stream."""
    try:
        with (
            torch_dev.device(cache_context.device),
            torch_dev.stream(cache_context.stream),
        ):
            response.set_result(finish(succeeded))
    except Exception as error:
        logger.exception("Cannot complete a retrieve of objects a peer owns")
        response.set_exception(error)


class _SlotPool:
    """Slots shared by a context's retrieves; waves start in submission order
    once enough slots are free, each taking at most half of them."""

    def __init__(self, slots: Sequence[int]) -> None:
        self.free = list(slots)
        self.wave_slots = max(1, len(slots) // 2)
        self.waiting: deque["_Wave"] = deque()
        self._starting = False

    def submit(self, wave: "_Wave") -> None:
        self.waiting.append(wave)
        self.start_waiting()

    def release(self, slots: Sequence[int]) -> None:
        self.free.extend(slots)
        self.start_waiting()

    def start_waiting(self) -> None:
        # A wave that frees its slots while starting re-enters; the outer loop
        # starts what they allow.
        if self._starting:
            return
        self._starting = True
        try:
            while self.waiting and len(self.free) >= self.waiting[0].num_slots:
                wave = self.waiting.popleft()
                wave.start([self.free.pop() for _ in range(wave.num_slots)])
        finally:
            self._starting = False


@dataclass(eq=False)
class _Wave:
    retrieve: "_Retrieve"
    pool: _SlotPool
    runs: list[DeferredNicRun] = field(default_factory=list)
    started: bool = False

    @property
    def num_slots(self) -> int:
        return sum(len(run.memory_objs) for run in self.runs)

    def start(self, slots: list[int]) -> None:
        self.started = True
        self.retrieve.start_wave(self, slots)


@dataclass
class _NicStaging:
    buffer: StagingBuffer
    registration: TransferChannelStagingRegistration
    pool: _SlotPool
    spare: _SlotPool  # the rows at the end, one unfull chunk at a time
    # Retrieves not completed yet; drain() waits on the condition for zero.
    active: int = 0
    idle: threading.Condition = field(default_factory=threading.Condition)


class _Retrieve:
    """One retrieve's waves; every method runs on the affinity worker."""

    def __init__(
        self,
        cache_context: BaseCacheContext,
        staging: _NicStaging,
        poller: HostPoller,
        runs: Sequence[DeferredNicRun],
        finish: Callable[[bool], object],
    ) -> None:
        self._cache_context = cache_context
        self._staging = staging
        self._poller = poller
        self._finish = finish
        self._failed = False
        self.response = DeferredResponse()
        self._waves: list[_Wave] = []
        for run in runs:
            pool = staging.spare if run.spans_slots else staging.pool
            wave = self._waves[-1] if self._waves else None
            if (
                wave is None
                or wave.pool is not pool
                or wave.num_slots + len(run.memory_objs) > pool.wave_slots
            ):
                wave = _Wave(self, pool)
                self._waves.append(wave)
            wave.runs.append(run)
        self._unsettled = len(self._waves)
        for wave in list(self._waves):
            if not self._failed:
                wave.pool.submit(wave)

    def start_wave(self, wave: _Wave, slots: list[int]) -> None:
        try:
            reads = self._submit_reads(wave, slots)
        except Exception:
            # Transport errors have many types; nothing was started that
            # could land in the slots.
            logger.exception("Cannot start reading objects a peer owns")
            self._fail()
            wave.pool.release(slots)
            self._settle()
            return
        self._poller.watch(
            lambda: all(read.is_finished() for read in reads),
            lambda: self._landed(wave, slots, reads),
            self._abandoned,
        )

    def _submit_reads(self, wave: _Wave, slots: list[int]) -> list[PendingRead]:
        """One read per peer of the wave's objects."""
        buffer = self._staging.buffer
        base = buffer.buffer.data_ptr()
        per_reader: dict[
            int, tuple[BorrowedReader, list[tuple[BorrowedMemoryObj, int]]]
        ] = {}
        free = iter(slots)
        for run in wave.runs:
            run_slots = [next(free) for _ in run.memory_objs]
            for obj, target in zip(
                run.memory_objs, run.targets(buffer, run_slots), strict=True
            ):
                if not isinstance(obj, BorrowedMemoryObj):
                    raise ValueError(f"Not an object a peer owns: {obj!r}")
                if obj.get_size() != target.nbytes:
                    raise ValueError(
                        f"Object of {obj.get_size()} bytes, slot of {target.nbytes}"
                    )
                entry = per_reader.setdefault(id(obj.reader), (obj.reader, []))
                entry[1].append((obj, target.data_ptr() - base))
        region_id = self._staging.registration.region_id
        return [reader.read_many(region_id, t) for reader, t in per_reader.values()]

    def _landed(self, wave: _Wave, slots: list[int], reads: list[PendingRead]) -> None:
        if not all(read.succeeded() for read in reads):
            logger.warning("A peer read failed; the retrieve's blocks are recomputed")
            self._fail()
        if self._failed:
            wave.pool.release(slots)
            self._settle()
            return
        cc = self._cache_context
        with torch_dev.device(cc.device), torch_dev.stream(cc.stream):
            try:
                free = iter(slots)
                for run in wave.runs:
                    run.launch(
                        self._staging.buffer, [next(free) for _ in run.memory_objs]
                    )
            except Exception:
                logger.exception("Cannot queue the kernels of a peer read")
                self._fail()
            # Also after a failed launch: kernels queued before it read slots.
            event = record_completion_event(cc.device, cc.stream)
        pool = wave.pool
        # On shutdown the kernels may still run, so the slots stay taken.
        self._poller.watch(event.is_complete, lambda: pool.release(slots), lambda: None)
        self._settle()

    def _abandoned(self) -> None:
        # The reads may still land, so their slots are not released.
        self._fail()
        self._settle()

    def _fail(self) -> None:
        if self._failed:
            return
        self._failed = True
        unstarted = [wave for wave in self._waves if not wave.started]
        for pool in (self._staging.pool, self._staging.spare):
            pool.waiting = deque(w for w in pool.waiting if w not in unstarted)
        self._unsettled -= len(unstarted)

    def _settle(self) -> None:
        self._unsettled -= 1
        if self._unsettled:
            return
        _complete(self._cache_context, self.response, self._finish, not self._failed)
        with self._staging.idle:
            self._staging.active -= 1
            self._staging.idle.notify_all()


class NicDmaEngine:
    """Reads runs of objects a peer owns without blocking.

    A context gets its NIC staging buffer on its first read:
    :data:`NIC_STAGING_BYTES` of slots (at least two kernel batches) plus two
    rows for an unfull chunk, registered with the transfer channel.

    Args:
        poller: Watches reads and kernel events; started on first use if None.
    """

    def __init__(self, poller: HostPoller | None = None) -> None:
        self._poller = poller
        self._closed = False
        self._lock = threading.Lock()
        # By id(context); drain() removes a context's entry before it closes.
        self._staging: dict[int, _NicStaging] = {}

    def read(
        self,
        cache_context: BaseCacheContext,
        runs: Sequence[DeferredNicRun],
        finish: Callable[[bool], object],
    ) -> DeferredResponse:
        """Read ``runs`` and queue their kernels; return at once. Call on the
        context's affinity worker after the retrieve queued its other work.

        Args:
            cache_context: The retrieve's context.
            runs: The runs its transfers returned.
            finish: Called once on the worker, under the context's device
                and stream: True after the last run's
                kernels are queued; False once a read failed or was abandoned
                and every started read stopped (later runs are not read), or
                at once if the NIC staging buffer cannot be created.

        Returns:
            A response completed with ``finish``'s result or exception.
        """
        with self._lock:
            if self._poller is None:
                self._poller = HostPoller("lmcache-nic-dma")
            poller = self._poller
        try:
            staging = self._staging.get(id(cache_context)) or self._create(
                cache_context
            )
        except Exception:
            # Allocation and transport registration errors have many types.
            logger.exception("Cannot create the NIC staging buffer")
            response = DeferredResponse()
            _complete(cache_context, response, finish, False)
            return response
        with staging.idle:
            staging.active += 1
        return _Retrieve(cache_context, staging, poller, runs, finish).response

    def drain(self, cache_context: BaseCacheContext) -> None:
        """Wait until every retrieve of ``cache_context`` completed. Call
        before closing the context, and not on its affinity worker: closing
        deregisters the slots reads land in and unmaps the KV cache their
        kernels fill. Returns at once after :meth:`close`.

        Args:
            cache_context: The context about to be closed.
        """
        staging = self._staging.get(id(cache_context))
        if staging is None:
            return
        with staging.idle:
            staging.idle.wait_for(lambda: staging.active == 0 or self._closed)
        del self._staging[id(cache_context)]

    def close(self) -> None:
        """Stop the poller; reads in flight are abandoned."""
        with self._lock:
            poller = self._poller
        if poller is not None:
            poller.stop()
        self._closed = True
        for staging in list(self._staging.values()):
            with staging.idle:
                staging.idle.notify_all()

    def _create(self, cache_context: BaseCacheContext) -> _NicStaging:
        groups = range(cache_context.kv_layer_groups_manager.num_object_groups)
        row = sum(
            cache_context.get_temp_object_group_buffer(0, g).nbytes for g in groups
        )
        num_slots = max(2 * cache_context.max_batch_size, NIC_STAGING_BYTES // row)
        registration = TransferChannelStagingRegistration()
        buffer = cache_context.create_staging_buffer(
            num_slots + _SPARE_ROWS,
            get_transfer_channel_context().align_bytes,
            registration,
        )
        logger.info(
            "NIC staging buffer: %d slots, %d MiB on %s",
            num_slots + _SPARE_ROWS,
            buffer.buffer.nbytes >> 20,
            cache_context.device,
        )
        staging = _NicStaging(
            buffer, registration, _SlotPool(range(num_slots)), _SlotPool([num_slots])
        )
        self._staging[id(cache_context)] = staging
        return staging
