# SPDX-License-Identifier: Apache-2.0
"""Callback contract for a cache context's default object-group transfer."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

# Third Party
import torch

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.common.memory_management import MemoryObj
    from lmcache.v1.platform.base.cache_context import BaseCacheContext
    import lmcache.lmcache_native as lmcache_native


class StagingBuffer(Protocol):
    """Staging slots with a cache context's layout.

    Slot ``i`` holds one chunk: object group ``g`` at
    ``get_temp_object_group_buffer(i, g)``, kernel group ``k`` (as the block
    kernel reads it) at ``get_temp_kernel_group_buffer(i, k)``.
    """

    @property
    def max_batch_size(self) -> int:
        """Number of slots."""
        ...

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


class ObjectGroupTransfer(Protocol):
    """Enqueue one object group's copies on the supplied context's stream.

    The server supplies the default callback at context construction; device
    contexts may instead override ``transfer_object_group`` to match their
    staging layout.
    """

    def __call__(
        self,
        cache_context: BaseCacheContext,
        block_ids_gpu: list[torch.Tensor],
        memory_objs: Sequence[MemoryObj | None],
        object_group_id: int,
        batch_size: int,
        skip_first_n_tokens: int,
        direction: lmcache_native.TransferDirection,
        *,
        transfer_key: str,
        block_ids_host: Sequence[Sequence[int]] = (),
    ) -> list[DeferredNicRun] | None:
        """Transfer one object group without committing storage.

        Args:
            cache_context: KV layout, pointers, staging buffers and stream.
            block_ids_gpu: Staged block IDs per kernel group, in chunk order.
            memory_objs: One object per chunk. Retrieve rejects None entries;
                store skips their batches.
            object_group_id: Object group index.
            batch_size: Chunks per batch.
            skip_first_n_tokens: Leading tokens to leave untouched.
            direction: H2D for retrieve or D2H for store.
            transfer_key: Operation ID for phase timing.
            block_ids_host: Host block IDs; empty forces the kernel copy path.

        Returns:
            On retrieve, the runs of objects a peer owns, for the NIC DMA
            engine to read; None when there are none. Every other copy is
            enqueued on the context's stream.

        Raises:
            ValueError: If a retrieve encounters a missing object, or a store
                targets an object a peer owns.

        Copy failures propagate to the caller. The caller owns completion,
        storage commit and the lifetime of the context and memory objects.
        """
        ...
