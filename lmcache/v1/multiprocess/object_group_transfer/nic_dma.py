# SPDX-License-Identifier: Apache-2.0
"""NIC DMA engine: reads objects a peer owns into GPU memory without blocking.

A transfer hands the runs of peer-owned objects (:class:`DeferredNicRun`) to
the caller's :func:`collect_peer_runs` block: a cache context's transfer
callback returns nothing. :meth:`NicDmaEngine.read` reads them in waves into
a NIC staging buffer that nothing on the stream uses, queues each wave's
kernels once its reads landed, and frees the slots once those kernels
finished. A :class:`HostPoller` checks
the reads and events and hands each step to the instance's affinity worker;
no stream waits on a host-written value. See
``docs/design/v1/distributed/l2_adapters/gdr_peer_l2_adapter.md``.
"""

# Standard
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Callable, Protocol, Sequence, runtime_checkable
import threading
import weakref

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.v1.common.buffer_registration import BufferRegistration
from lmcache.v1.common.logging import init_logger
from lmcache.v1.distributed.transfer_channel import (
    TransferChannelStagingRegistration,
)
from lmcache.v1.memory_management import (
    BorrowedMemoryObj,
    BorrowedReader,
    MemoryObj,
    PendingRead,
)
from lmcache.v1.multiprocess.affinity_pool import DeferredResponse
from lmcache.v1.multiprocess.host_poller import HostPoller
from lmcache.v1.platform.base.cache_context import BaseCacheContext
from lmcache.v1.platform.stream import record_completion_event

logger = init_logger(__name__)

# Slot bytes per context: 384-512 MiB in flight filled a 400 Gb/s link in the
# module benchmark (PR #6411) for 6 MiB and 80 MiB chunks alike.
NIC_STAGING_BYTES = 512 << 20
# Rows after the slots for an unfull chunk: one row plus at most one lead
# block per kernel group.
_SPARE_ROWS = 2


class StagingBuffer(Protocol):
    """Staging slots with a cache context's layout.

    Slot ``i`` holds one chunk: object group ``g`` at
    ``get_temp_object_group_buffer(i, g)``, kernel group ``k`` (as the block
    kernel reads it) at ``get_temp_kernel_group_buffer(i, k)``.
    """

    @property
    def buffer(self) -> torch.Tensor:
        """The flat buffer every slot lies in."""
        ...

    def get_temp_object_group_buffer(
        self, batch_idx: int, object_group_idx: int
    ) -> torch.Tensor:
        """Flat uint8 region of one object group in slot ``batch_idx``."""
        ...

    def get_temp_kernel_group_buffer(
        self, batch_idx: int, kernel_group_idx: int
    ) -> torch.Tensor:
        """One kernel group in slot ``batch_idx``, shaped for the kernel."""
        ...


class Staging(Protocol):
    """Where one transfer step is staged, indexed by its chunks: chunk ``i``
    of the step is in slot ``i``. A cache context is one (its shared
    buffer); the NIC DMA engine passes a view of the slots a wave got."""

    def get_temp_object_group_buffer(
        self, batch_idx: int, object_group_idx: int
    ) -> torch.Tensor:
        """Flat uint8 region of one object group of chunk ``batch_idx``."""
        ...

    def get_temp_kernel_group_buffer(
        self, batch_idx: int, kernel_group_idx: int
    ) -> torch.Tensor:
        """One kernel group of chunk ``batch_idx``, shaped for the kernel."""
        ...

    def get_temp_staging_buffer(self, num_bytes: int) -> torch.Tensor | None:
        """The first ``num_bytes`` bytes from chunk 0 on, as one flat region;
        None if they do not fit."""
        ...


@runtime_checkable
class StagingBufferFactory(Protocol):
    """A cache context that can allocate staging buffers apart from the one
    its transfers share. The NIC DMA engine needs one; a context without it
    cannot retrieve objects a peer owns."""

    def create_staging_buffer(
        self,
        num_slots: int,
        registration: BufferRegistration,
    ) -> StagingBuffer:
        """Allocate a staging buffer of ``num_slots`` slots with the
        context's layout.

        Every slot starts on ``registration.staging_alignment()``. The buffer
        is registered with ``registration`` on the context's
        stream and deregistered when the context is closed.

        Args:
            num_slots: Slots in the buffer.
            registration: Registers the buffer, e.g. with a transfer channel.

        Returns:
            The buffer.

        Raises:
            RuntimeError: If the registration fails.
        """
        ...


@dataclass
class DeferredNicRun:
    """Device work that waits for objects a peer owns to be read.

    A transfer hands it over instead of reading inline; the NIC DMA engine
    reads ``memory_objs`` into staging slots it picks, then queues the
    kernels.

    Attributes:
        memory_objs: The objects to read, in chunk order.
        block_ids: The block ids the kernels use.
        targets: ``staging -> regions``: where each object lands.
        launch: ``(staging, block_ids) -> None``: queues the kernels that
            copy the landed bytes into the paged KV cache.
        skipped: Chunks of the step before the first object; the kernels
            skip them, so they get no slot.
        spans_slots: The object may extend past its slot into the following
            ones (an unfull chunk with lead blocks), so the run needs the
            slot at the end of the buffer that has room after it.
    """

    memory_objs: Sequence[MemoryObj]
    block_ids: list[torch.Tensor]
    targets: Callable[[Staging], list[torch.Tensor]]
    launch: Callable[[Staging, list[torch.Tensor]], None]
    skipped: int = 0
    spans_slots: bool = False


_peer_run_sink: ContextVar[list[DeferredNicRun] | None] = ContextVar(
    "lmcache_peer_run_sink", default=None
)


@contextmanager
def collect_peer_runs(runs: list[DeferredNicRun]) -> Iterator[None]:
    """Append to ``runs`` the runs that transfers called inside the block,
    on this thread, defer.

    Args:
        runs: Receives the runs, in transfer order.
    """
    token = _peer_run_sink.set(runs)
    try:
        yield
    finally:
        _peer_run_sink.reset(token)


def defer_peer_runs(runs: Sequence[DeferredNicRun]) -> None:
    """Hand ``runs`` to the caller's :func:`collect_peer_runs` block.

    Args:
        runs: The runs of objects a peer owns that a transfer did not read.

    Raises:
        NotImplementedError: If the caller collects none: it cannot read
            objects a peer owns.
    """
    sink = _peer_run_sink.get()
    if sink is None:
        raise NotImplementedError(
            "Objects a peer owns need the NIC DMA engine: transfer them "
            "inside collect_peer_runs()"
        )
    sink.extend(runs)


def defer_run(run: DeferredNicRun) -> None:
    """Hand one run to the caller's :func:`collect_peer_runs` block, with
    its own copy of the block ids: the next transfer restages the buffer
    they live in.

    Args:
        run: The run a transfer did not read.

    Raises:
        NotImplementedError: If the caller collects none.
    """
    defer_peer_runs([replace(run, block_ids=[ids.clone() for ids in run.block_ids])])


class _SlotView:
    """The slots a run got in a NIC staging buffer, as its :class:`Staging`."""

    def __init__(self, buffer: StagingBuffer, slots: Sequence[int], skipped: int):
        self._buffer = buffer
        self._slots = slots
        self._skipped = skipped

    def get_temp_object_group_buffer(
        self, batch_idx: int, object_group_idx: int
    ) -> torch.Tensor:
        return self._buffer.get_temp_object_group_buffer(
            self._slot(batch_idx), object_group_idx
        )

    def get_temp_kernel_group_buffer(
        self, batch_idx: int, kernel_group_idx: int
    ) -> torch.Tensor:
        return self._buffer.get_temp_kernel_group_buffer(
            self._slot(batch_idx), kernel_group_idx
        )

    def get_temp_staging_buffer(self, num_bytes: int) -> torch.Tensor | None:
        flat = self._buffer.buffer
        start = self.get_temp_object_group_buffer(0, 0).data_ptr() - flat.data_ptr()
        return flat[start : start + num_bytes]

    def _slot(self, batch_idx: int) -> int:
        # The kernels skip the chunks before the first object; their
        # pointers only fill the list.
        return self._slots[max(0, batch_idx - self._skipped)]


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


@dataclass(eq=False)
class _NicStaging:
    buffer: StagingBuffer
    registration: TransferChannelStagingRegistration
    pool: _SlotPool
    spare: _SlotPool  # the rows at the end, one unfull chunk at a time
    # Retrieves not completed yet; drain() waits on the condition for zero.
    active: int = 0
    idle: threading.Condition = field(default_factory=threading.Condition)
    # Set by drain(): waves not started yet fail, so the wait covers only the
    # reads in flight.
    released: bool = False


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
        reads = None
        if not self._staging.released:
            try:
                reads = self._submit_reads(wave, slots)
            except Exception:
                # Transport errors have many types; nothing was started
                # that could land in the slots.
                logger.exception("Cannot start reading objects a peer owns")
        if reads is None:
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
            view = _SlotView(buffer, [next(free) for _ in run.memory_objs], run.skipped)
            for obj, target in zip(run.memory_objs, run.targets(view), strict=True):
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
                    run_slots = [next(free) for _ in run.memory_objs]
                    run.launch(
                        _SlotView(self._staging.buffer, run_slots, run.skipped),
                        run.block_ids,
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
        # By id(context); drain() removes a context's entry before the
        # context can be freed and its id reused.
        self._staging: dict[int, _NicStaging] = {}
        # Contexts drain() was called for: a later read fails.
        self._released: weakref.WeakSet[BaseCacheContext] = weakref.WeakSet()
        # Buffers being drained, for close() to wake the wait. One whose
        # reads did not stop in time stays: it is never freed.
        self._draining: list[_NicStaging] = []

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
                at once if the NIC staging buffer cannot be created or
                :meth:`drain` was called for the context.

        Returns:
            A response completed with ``finish``'s result or exception.
        """
        with self._lock:
            if self._poller is None:
                self._poller = HostPoller("lmcache-nic-dma")
            poller = self._poller
            # Under the lock, so that drain() either counts this retrieve or
            # makes it fail.
            staging = None
            if cache_context not in self._released:
                try:
                    staging = self._staging.get(id(cache_context)) or self._create(
                        cache_context
                    )
                except Exception:
                    # Allocation and registration errors have many types.
                    logger.exception("Cannot create the NIC staging buffer")
                else:
                    with staging.idle:
                        staging.active += 1
        if staging is None:
            response = DeferredResponse()
            _complete(cache_context, response, finish, False)
            return response
        return _Retrieve(cache_context, staging, poller, runs, finish).response

    def drain(self, cache_context: BaseCacheContext, timeout_s: float) -> bool:
        """Wait until every retrieve of ``cache_context`` completed. Call
        before closing the context, and not on its affinity worker: closing
        deregisters the slots reads land in and unmaps the KV cache their
        kernels fill. From this call on, a read of the context fails and its
        waves that have not started are dropped, so the wait covers only the
        reads in flight. Returns at once after :meth:`close`.

        Args:
            cache_context: The context about to be closed.
            timeout_s: How long to wait.

        Returns:
            False if a retrieve is still running after ``timeout_s``: the
            context must then stay open. Its NIC staging buffer is kept
            for good, since a read may still write to it.
        """
        with self._lock:
            self._released.add(cache_context)
            staging = self._staging.pop(id(cache_context), None)
            if staging is None:
                return True
            staging.released = True
            self._draining.append(staging)
        with staging.idle:
            drained = staging.idle.wait_for(
                lambda: staging.active == 0 or self._closed, timeout_s
            )
        if drained:
            with self._lock:
                self._draining.remove(staging)
        return drained

    def close(self) -> None:
        """Stop the poller; reads in flight are abandoned."""
        with self._lock:
            poller = self._poller
        if poller is not None:
            poller.stop()
        self._closed = True
        with self._lock:
            draining = list(self._draining)
        for staging in draining:
            with staging.idle:
                staging.idle.notify_all()

    def _create(self, cache_context: BaseCacheContext) -> _NicStaging:
        if not isinstance(cache_context, StagingBufferFactory):
            raise NotImplementedError(
                f"{type(cache_context).__name__} cannot create a NIC staging buffer"
            )
        groups = range(cache_context.kv_layer_groups_manager.num_object_groups)
        row = sum(
            cache_context.get_temp_object_group_buffer(0, g).nbytes for g in groups
        )
        num_slots = max(2 * cache_context.max_batch_size, NIC_STAGING_BYTES // row)
        registration = TransferChannelStagingRegistration()
        buffer = cache_context.create_staging_buffer(
            num_slots + _SPARE_ROWS, registration
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
