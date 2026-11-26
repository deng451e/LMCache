# SPDX-License-Identifier: Apache-2.0
"""Retrieve steps of engines whose bytes land off the stream.

Most engines stage on the context's stream, so a transfer stages and
launches its kernels inline: the stream orders the kernels after the copy.
An engine whose bytes arrive by a path the stream does not wait for (a NIC
writing into GPU memory) cannot do that. A transfer hands each of its steps
as an :class:`OffStreamRun` to the caller's :func:`collect_off_stream_runs`
block, and the caller gives the collected runs to an
:class:`OffStreamEngine`, which stages them and queues their kernels once
the bytes landed.
"""

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Callable, Protocol, Sequence

# Third Party
import torch

# First Party
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.affinity_pool import DeferredResponse
from lmcache.v1.platform.base.cache_context import BaseCacheContext


class Staging(Protocol):
    """Where one transfer step is staged, indexed by its chunks: chunk ``i``
    of the step is in slot ``i``. A cache context is one (its shared
    buffer); an engine that stages elsewhere passes a view of its own."""

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


@dataclass
class OffStreamRun:
    """One retrieve step of an engine that stages outside the stream.

    A transfer hands it over instead of staging inline; the engine stages
    ``memory_objs`` into staging slots it picks, then queues the kernels.

    Attributes:
        memory_objs: The objects to stage, in chunk order.
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


_run_sink: ContextVar[list[OffStreamRun] | None] = ContextVar(
    "lmcache_run_sink", default=None
)


@contextmanager
def collect_off_stream_runs(runs: list[OffStreamRun]) -> Iterator[None]:
    """Append to ``runs`` the runs that transfers called inside the block,
    on this thread, hand off.

    Args:
        runs: Receives the runs, in transfer order.
    """
    token = _run_sink.set(runs)
    try:
        yield
    finally:
        _run_sink.reset(token)


def hand_off_run(run: OffStreamRun) -> None:
    """Hand one run to the caller's :func:`collect_off_stream_runs` block, with
    its own copy of the block ids: the next transfer restages the buffer
    they live in.

    Args:
        run: The run a transfer did not stage.

    Raises:
        NotImplementedError: If the caller collects none: it cannot run
            off-stream steps.
    """
    sink = _run_sink.get()
    if sink is None:
        raise NotImplementedError(
            "This transfer has off-stream steps: call it inside "
            "collect_off_stream_runs()"
        )
    sink.append(replace(run, block_ids=[ids.clone() for ids in run.block_ids]))


class OffStreamEngine(Protocol):
    """Runs retrieves' off-stream runs without blocking the worker."""

    def read(
        self,
        cache_context: BaseCacheContext,
        runs: Sequence[OffStreamRun],
        finish: Callable[[bool], object],
    ) -> DeferredResponse:
        """Stage ``runs`` and queue their kernels; return at once. Call on
        the context's affinity worker after the retrieve queued its other
        work.

        Args:
            cache_context: The retrieve's context.
            runs: The runs its transfers handed off.
            finish: Called once on the worker, under the context's device
                and stream: True after the last run's kernels are queued,
                False if the runs could not all be staged.

        Returns:
            A response completed with ``finish``'s result or exception.
        """
        ...

    def drain(self, cache_context: BaseCacheContext) -> bool:
        """Wait until every retrieve of ``cache_context`` completed. Call
        before closing the context, and not on its affinity worker. From
        this call on, a read of the context fails.

        Args:
            cache_context: The context about to be closed.

        Returns:
            False if a retrieve is still running after the engine's
            timeout: the context must then stay open.
        """
        ...

    def close(self) -> None:
        """Stop the engine; runs in flight are abandoned."""
        ...
