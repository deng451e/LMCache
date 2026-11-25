# SPDX-License-Identifier: Apache-2.0
"""What a transfer hands to the NIC DMA engine: the runs of objects a peer
owns, and the staging slots the engine reads them into.

A cache context's transfer callback returns nothing
(:class:`~lmcache.v1.platform.base.object_group_transfer.ObjectGroupTransfer`),
so the runs reach the caller through :func:`collect_peer_runs`.
"""

# Standard
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# Third Party
import torch

# First Party
from lmcache.v1.common.buffer_registration import BufferRegistration
from lmcache.v1.memory_management import MemoryObj


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


@runtime_checkable
class StagingBufferFactory(Protocol):
    """A cache context that can allocate staging buffers apart from the one
    its transfers share. The NIC DMA engine needs one; a context without it
    cannot retrieve objects a peer owns."""

    def create_staging_buffer(
        self,
        num_slots: int,
        slot_align_bytes: int,
        registration: BufferRegistration,
    ) -> StagingBuffer:
        """Allocate a staging buffer of ``num_slots`` slots with the
        context's layout.

        The buffer is registered with ``registration`` on the context's
        stream and deregistered when the context is closed.

        Args:
            num_slots: Slots in the buffer.
            slot_align_bytes: Every slot starts on this power of two.
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

    A transfer returns it instead of reading inline; the NIC DMA engine reads
    ``memory_objs`` into staging slots it picks, then queues the kernels.

    Attributes:
        memory_objs: The objects to read, in slot order.
        targets: ``(staging, slots) -> regions``: where each object lands
            when the run gets ``slots`` of ``staging``.
        launch: ``(staging, slots) -> None``: queues the kernels that copy
            the landed bytes into the paged KV cache.
        spans_slots: The object may extend past its slot into the following
            ones (an unfull chunk with lead blocks), so the run needs the
            slot at the end of the buffer that has room after it.
    """

    memory_objs: Sequence[MemoryObj]
    targets: Callable[[StagingBuffer, Sequence[int]], list[torch.Tensor]]
    launch: Callable[[StagingBuffer, Sequence[int]], None]
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
