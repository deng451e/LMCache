# SPDX-License-Identifier: Apache-2.0
"""NIC DMA engine: reads objects a peer owns into GPU memory without blocking.

A retrieve's transfers return the runs of objects a peer owns
(:class:`DeferredNicRun`) instead of staging them. :meth:`NicDmaEngine.read`
reads them in waves into a NIC staging buffer that only these reads use, so
a read never waits for queued copies:

1. A wave takes free staging slots (FIFO across the context's retrieves) and
   starts one transfer-channel read per peer.
2. When the reads landed, the wave's kernels are queued on the context's
   stream and an event is recorded after them.
3. When that event completed, the slots are free again.

One :class:`~lmcache.v1.multiprocess.host_poller.HostPoller` watches steps 2
and 3 and hands them to the instance's affinity worker. No stream ever waits
on a host-written value. See
``docs/design/v1/distributed/l2_adapters/gdr_peer_l2_adapter.md``.
"""

# Standard
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Collection, Protocol, Sequence
import threading
import weakref

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.v1.common.logging import init_logger
from lmcache.v1.distributed.transfer_channel import (
    TransferChannelStagingRegistration,
    find_staging_region,
    staging_alignment,
)
from lmcache.v1.memory_management import (
    BorrowedMemoryObj,
    BorrowedReader,
    MemoryObj,
    PendingRead,
)
from lmcache.v1.multiprocess.affinity_pool import DeferredResponse
from lmcache.v1.multiprocess.host_poller import HostPoller
from lmcache.v1.multiprocess.object_group_transfer.staged import (
    iter_staged_batches,
    launch_batch_kernels,
)
from lmcache.v1.platform.base.cache_context import BaseCacheContext
from lmcache.v1.platform.base.object_group_transfer import (
    DeferredNicRun,
    StagingBuffer,
)
from lmcache.v1.platform.stream import record_completion_event
import lmcache.lmcache_native as lmcache_native

logger = init_logger(__name__)

# Bytes of NIC staging slots per cache context. 384-512 MiB in flight filled
# a 400 Gb/s link in the module benchmark (PR #6411) for 6 MiB and 80 MiB
# chunks alike; fewer than two waves serialize the reads with the copies out.
NIC_STAGING_BYTES = 512 << 20

# Rows at the end of the NIC staging buffer for an unfull chunk: its object
# is one row plus at most one lead block per kernel group.
_SPARE_ROWS = 2

_H2D = lmcache_native.TransferDirection.H2D


def defer_peer_run(
    cache_context: BaseCacheContext,
    block_ids_gpu: list[torch.Tensor],
    memory_objs: Sequence[MemoryObj | None],
    object_group_id: int,
    batch_size: int,
    skip_first_n_tokens: int,
    chunk_range: tuple[int, int],
) -> list[DeferredNicRun]:
    """Turn a retrieve's run of objects a peer owns into deferred runs.

    One run per kernel batch, with the same batches, skips and sliding
    window as the staged loop. Each run copies the block ids it needs now: a
    later transfer restages the shared buffer they live in.

    Args:
        cache_context: Context of the transfer; its stream is current.
        block_ids_gpu: Staged block IDs per kernel group.
        memory_objs: The object group's objects, one per chunk.
        object_group_id: Object group index.
        batch_size: Chunks per kernel batch.
        skip_first_n_tokens: Leading tokens to leave untouched.
        chunk_range: ``[start, end)`` chunks of the run.

    Returns:
        The runs, in chunk order.

    Raises:
        ValueError: If an object is missing.
    """
    batches = list(
        iter_staged_batches(
            cache_context,
            memory_objs,
            object_group_id,
            batch_size,
            skip_first_n_tokens,
            _H2D,
            chunk_range,
        )
    )
    if not batches:
        return []
    # Stream-ordered copy, queued before a later transfer restages the
    # shared block-id buffer.
    block_ids = [ids.clone() for ids in block_ids_gpu]
    runs: list[DeferredNicRun] = []
    for batch in batches:
        first = batch.first_staged

        def targets(staging: StagingBuffer, slots: Sequence[int]) -> list[torch.Tensor]:
            return [
                staging.get_temp_object_group_buffer(slot, object_group_id)
                for slot in slots
            ]

        def launch(
            staging: StagingBuffer,
            slots: Sequence[int],
            batch=batch,
            first=first,
        ) -> None:
            # The kernel skips the chunks before the first staged one; their
            # pointers only fill the list.
            launch_batch_kernels(
                cache_context,
                block_ids,
                object_group_id,
                batch.start_object_idx,
                len(batch.memory_objs),
                batch.skip_tokens_in_chunk,
                _H2D,
                staging=staging,
                slots=[slots[0]] * first + list(slots),
            )

        runs.append(DeferredNicRun(batch.memory_objs[first:], targets, launch))
    return runs


class SlotWave(Protocol):
    """Chunks that land together in staging slots."""

    @property
    def num_slots(self) -> int:
        """Slots the wave needs."""
        ...

    def start(self, slots: list[int]) -> None:
        """Start the wave in ``slots``; it gives them back with
        :meth:`StagingSlotPool.release` once nothing uses them any more."""
        ...


class StagingSlotPool:
    """Staging slots shared by a context's retrieves.

    Waves start in submission order, each once enough slots are free. Only
    the context's affinity worker touches the pool.

    Args:
        slots: The slot indices.

    Raises:
        ValueError: If ``slots`` is empty.
    """

    def __init__(self, slots: Sequence[int]) -> None:
        if not slots:
            raise ValueError("A staging slot pool needs at least one slot")
        self._free = list(slots)
        self._wave_slots = max(1, len(slots) // 2)
        self._waiting: deque[SlotWave] = deque()
        self._starting = False

    @property
    def wave_slots(self) -> int:
        """Most slots a wave may take: half the pool (at least 1), so one
        wave lands while the previous one is copied out."""
        return self._wave_slots

    def submit(self, wave: SlotWave) -> None:
        """Queue ``wave``; start it, and any waves behind it, when slots are
        free.

        Raises:
            ValueError: If the wave needs more than :attr:`wave_slots`.
        """
        if wave.num_slots > self._wave_slots:
            raise ValueError(
                f"A wave of {wave.num_slots} slots exceeds the "
                f"{self._wave_slots}-slot limit"
            )
        self._waiting.append(wave)
        self._start_waiting()

    def withdraw(self, waves: Collection[SlotWave]) -> None:
        """Drop the queued waves among ``waves``; started ones are kept."""
        self._waiting = deque(
            w for w in self._waiting if not any(w is other for other in waves)
        )

    def release(self, slots: Sequence[int]) -> None:
        """Take back slots a wave no longer uses and start queued waves."""
        self._free.extend(slots)
        self._start_waiting()

    def _start_waiting(self) -> None:
        # A wave that gives its slots back while starting re-enters here;
        # the outer loop starts whatever those slots allow.
        if self._starting:
            return
        self._starting = True
        try:
            while self._waiting and len(self._free) >= self._waiting[0].num_slots:
                wave = self._waiting.popleft()
                slots = [self._free.pop() for _ in range(wave.num_slots)]
                wave.start(slots)
        finally:
            self._starting = False


@dataclass
class _NicStaging:
    """A context's NIC staging buffer and its slot pools."""

    buffer: StagingBuffer
    pool: StagingSlotPool
    # The slot at the end of the buffer, for runs whose object extends past
    # its slot.
    spare: StagingSlotPool


@dataclass(eq=False)
class _Wave:
    retrieve: "_Retrieve"
    pool: StagingSlotPool
    runs: list[DeferredNicRun] = field(default_factory=list)
    started: bool = False

    @property
    def num_slots(self) -> int:
        return sum(len(run.memory_objs) for run in self.runs)

    def start(self, slots: list[int]) -> None:
        self.started = True
        self.retrieve.start_wave(self, slots)


class _Retrieve:
    """One retrieve's waves; runs on the context's affinity worker."""

    def __init__(
        self,
        cache_context: BaseCacheContext,
        staging: _NicStaging,
        poller: HostPoller,
        finish: Callable[[bool], object],
    ) -> None:
        self._cache_context = cache_context
        self._staging = staging
        self._poller = poller
        self._finish = finish
        self.response = DeferredResponse()
        self._waves: list[_Wave] = []
        self._unsettled = 0
        self._failed = False

    def plan(self, runs: Sequence[DeferredNicRun]) -> None:
        """Pack consecutive runs into waves of at most a pool's wave size."""
        wave: _Wave | None = None
        for run in runs:
            pool = self._staging.spare if run.spans_slots else self._staging.pool
            if (
                wave is None
                or wave.pool is not pool
                or wave.num_slots + len(run.memory_objs) > pool.wave_slots
            ):
                wave = _Wave(self, pool)
                self._waves.append(wave)
            wave.runs.append(run)

    def submit(self) -> None:
        self._unsettled = len(self._waves)
        for wave in self._waves:
            if self._failed:
                break
            wave.pool.submit(wave)

    def start_wave(self, wave: _Wave, slots: list[int]) -> None:
        try:
            reads = self._submit_reads(wave, slots)
        except Exception:
            # Submitting raises for transport errors of any type; nothing was
            # started that could land in the slots after a failed submit.
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
        """Start one read per peer of the wave's objects."""
        buffer = self._staging.buffer
        region = find_staging_region(buffer.buffer.data_ptr())
        if region is None:
            raise RuntimeError("The NIC staging buffer is not registered")
        per_reader: dict[int, tuple[BorrowedReader, list]] = {}
        free = iter(slots)
        for run in wave.runs:
            run_slots = [next(free) for _ in run.memory_objs]
            for memory_obj, target in zip(
                run.memory_objs, run.targets(buffer, run_slots), strict=True
            ):
                if not isinstance(memory_obj, BorrowedMemoryObj):
                    raise ValueError(f"Not an object a peer owns: {memory_obj!r}")
                if memory_obj.get_size() != target.nbytes:
                    raise ValueError(
                        f"Size mismatch: object {memory_obj.get_size()} bytes, "
                        f"staging slot {target.nbytes} bytes"
                    )
                reader = memory_obj.reader
                targets = per_reader.setdefault(id(reader), (reader, []))[1]
                targets.append((memory_obj, target.data_ptr() - region.base_ptr))
        return [
            reader.read_many(region.region_id, targets)
            for reader, targets in per_reader.values()
        ]

    def _landed(self, wave: _Wave, slots: list[int], reads: list[PendingRead]) -> None:
        if not all(read.succeeded() for read in reads):
            logger.warning(
                "A read of objects a peer owns failed; the retrieve fails and "
                "its blocks are recomputed"
            )
            self._fail()
        if self._failed:
            wave.pool.release(slots)
            self._settle()
            return
        cache_context = self._cache_context
        with (
            torch_dev.device(cache_context.device),
            torch_dev.stream(cache_context.stream),
        ):
            try:
                free = iter(slots)
                for run in wave.runs:
                    run.launch(
                        self._staging.buffer, [next(free) for _ in run.memory_objs]
                    )
            except Exception:
                logger.exception("Cannot queue the kernels of a peer read")
                self._fail()
            # Also after a failed launch: kernels queued before it still
            # read the slots.
            event = record_completion_event(cache_context.device, cache_context.stream)
        pool = wave.pool
        self._poller.watch(
            event.is_complete,
            lambda: pool.release(slots),
            # Shutdown: the kernels may still run, so the slots stay taken.
            lambda: None,
        )
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
        for pool in {id(w.pool): w.pool for w in unstarted}.values():
            pool.withdraw(unstarted)
        self._unsettled -= len(unstarted)

    def _settle(self) -> None:
        self._unsettled -= 1
        if self._unsettled:
            return
        try:
            self.response.set_result(self._finish(not self._failed))
        except Exception as error:
            logger.exception("Cannot complete a retrieve of objects a peer owns")
            self.response.set_exception(error)


class NicDmaEngine:
    """Reads runs of objects a peer owns without blocking.

    Each cache context gets a NIC staging buffer on its first read: about
    :data:`NIC_STAGING_BYTES` of slots for full chunks (at least two kernel
    batches), plus two rows at the end for one unfull chunk. It is registered
    with the transfer channel through
    :class:`~lmcache.v1.distributed.transfer_channel.TransferChannelStagingRegistration`.

    Args:
        poller: Watches reads and kernel events; one is started on first
            use when None.
    """

    def __init__(self, poller: HostPoller | None = None) -> None:
        self._poller = poller
        self._lock = threading.Lock()
        self._staging: weakref.WeakKeyDictionary[BaseCacheContext, _NicStaging] = (
            weakref.WeakKeyDictionary()
        )

    def read(
        self,
        cache_context: BaseCacheContext,
        runs: Sequence[DeferredNicRun],
        finish: Callable[[bool], object],
    ) -> DeferredResponse:
        """Read ``runs`` and queue their kernels; return at once.

        Called on the context's affinity worker, after the retrieve queued
        all its other device work.

        Args:
            cache_context: The retrieve's context.
            runs: The runs its transfers returned.
            finish: Called once on the worker: with True after the last
                run's kernels are queued; with False once a read failed (or
                was abandoned) and every read already started has stopped.
                Later runs are then never read and no kernel is queued.

        Returns:
            A response completed with ``finish``'s result, or with its
            exception.
        """
        retrieve = _Retrieve(
            cache_context, self._staging_of(cache_context), self._get_poller(), finish
        )
        retrieve.plan(runs)
        retrieve.submit()
        return retrieve.response

    def close(self) -> None:
        """Stop the poller; reads in flight are abandoned."""
        with self._lock:
            poller = self._poller
        if poller is not None:
            poller.stop()

    def _get_poller(self) -> HostPoller:
        with self._lock:
            if self._poller is None:
                self._poller = HostPoller("lmcache-nic-dma")
            return self._poller

    def _staging_of(self, cache_context: BaseCacheContext) -> _NicStaging:
        """Return the context's NIC staging buffer, created on first use."""
        staging = self._staging.get(cache_context)
        if staging is not None:
            return staging
        groups = range(cache_context.kv_layer_groups_manager.num_object_groups)
        row_bytes = sum(
            cache_context.get_temp_object_group_buffer(0, group).nbytes
            for group in groups
        )
        num_slots = max(
            2 * cache_context.max_batch_size, NIC_STAGING_BYTES // row_bytes
        )
        buffer = cache_context.create_staging_buffer(
            num_slots + _SPARE_ROWS,
            staging_alignment(),
            TransferChannelStagingRegistration(),
        )
        logger.info(
            "Created a NIC staging buffer of %d slots (%d MiB) on %s",
            num_slots + _SPARE_ROWS,
            buffer.buffer.nbytes >> 20,
            cache_context.device,
        )
        staging = _NicStaging(
            buffer, StagingSlotPool(range(num_slots)), StagingSlotPool([num_slots])
        )
        self._staging[cache_context] = staging
        return staging
