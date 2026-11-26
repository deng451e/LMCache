# SPDX-License-Identifier: Apache-2.0
"""Transfer engines: which one moves an object, decided from its L1 type.

An engine only moves bytes between objects and staging regions; what the
regions hold (a full chunk, an unfull chunk) is the caller's business.
"""

# Standard
from typing import Callable, Protocol, Sequence
import enum

# Third Party
import torch

# First Party
from lmcache.v1.memory_management import (
    BorrowedMemoryObj,
    GDSMemoryObject,
    MemoryObj,
)
from lmcache.v1.multiprocess.object_group_transfer.gpu_dma import GpuDmaEngine
from lmcache.v1.multiprocess.object_group_transfer.nic_dma import NicDmaEngine
from lmcache.v1.multiprocess.object_group_transfer.nvme_dma import NvmeDmaEngine
from lmcache.v1.multiprocess.object_group_transfer.off_stream import (
    OffStreamRun,
    OffStreamEngine,
    hand_off_run,
)
import lmcache.lmcache_native as lmcache_native


class TransferEngine(enum.Enum):
    """How an L1 object's bytes travel to or from GPU memory."""

    GPU_DMA = "gpu_dma"
    """Host memory (DRAM, DevDAX): a host/GPU copy."""
    NVME_DMA = "nvme_dma"
    """Storage-backed object: a file read or write against GPU memory."""
    NIC_DMA = "nic_dma"
    """Object a peer owns: its bytes are read into GPU memory through the
    object's reader, retrieve only. Not
    staged here: the transfer hands its runs to the NIC DMA engine
    (``nic_dma.py``), which reads them without blocking."""


def select_engine(memory_obj: MemoryObj) -> TransferEngine:
    """Return the engine for ``memory_obj``'s type.

    Args:
        memory_obj: An L1 object.

    Returns:
        The engine that moves objects of that type.
    """
    if isinstance(memory_obj, GDSMemoryObject):
        return TransferEngine.NVME_DMA
    if isinstance(memory_obj, BorrowedMemoryObj):
        return TransferEngine.NIC_DMA
    return TransferEngine.GPU_DMA


class DMAEngine(Protocol):
    """What every transfer engine implements.

    An engine only moves bytes between objects and staging regions; it
    knows nothing about chunks, block ids or kernels.
    """

    def stage(
        self,
        memory_objs: Sequence[MemoryObj],
        buffers: Sequence[torch.Tensor],
        direction: "lmcache_native.TransferDirection",
    ) -> None:
        """Move objects to or from their staging regions.

        Args:
            memory_objs: Objects of this engine's type.
            buffers: One staging region per object.
            direction: H2D fills the regions, D2H fills the objects.
        """
        ...


_ENGINES: dict[TransferEngine, DMAEngine] = {
    TransferEngine.GPU_DMA: GpuDmaEngine(),
    TransferEngine.NVME_DMA: NvmeDmaEngine(),
}


# Engines whose bytes land off the stream: their retrieve steps are handed
# to the off-stream engine.
_OFF_STREAM = frozenset({TransferEngine.NIC_DMA})


def create_off_stream_engine(drain_timeout_s: float) -> OffStreamEngine:
    """Return a new engine that runs the steps :func:`staged_off_stream` hands over.

    Args:
        drain_timeout_s: How long its ``drain`` waits for a context's
            retrieves.
    """
    return NicDmaEngine(drain_timeout_s)


def stage(
    engine: TransferEngine,
    memory_objs: Sequence[MemoryObj],
    buffers: Sequence[torch.Tensor],
    direction: "lmcache_native.TransferDirection",
) -> None:
    """Move objects to or from their staging regions with ``engine``.

    On return the copies are queued on the stream ahead of anything queued
    later.

    Args:
        engine: The engine of every object in ``memory_objs``.
        memory_objs: The objects.
        buffers: One staging region per object.
        direction: H2D fills the regions, D2H fills the objects.

    Raises:
        ValueError: If a host-memory object and its region differ in size,
            or ``engine`` does not stage on the stream (see
            :func:`staged_off_stream`), as for a store with such an engine.
    """
    if engine not in _ENGINES:
        raise ValueError(
            f"The {engine.value} engine does not stage on the stream: its "
            "objects are only retrieved, as off-stream runs"
        )
    _ENGINES[engine].stage(memory_objs, buffers, direction)


def staged_off_stream(
    engine: TransferEngine, make_run: Callable[[], OffStreamRun]
) -> bool:
    """Hand a retrieve step to ``engine`` if it stages outside the stream.

    Args:
        engine: The engine of every object of the step.
        make_run: Builds the step (its objects and the kernels that follow
            them); called only for an off-stream engine.

    Returns:
        True if the engine took the run and will stage it and launch its
        kernels later; False for an engine that stages on the stream, where
        the caller stages and launches now.

    Raises:
        NotImplementedError: If the engine is off-stream and the caller of
            the transfer collects no off-stream runs.
    """
    if engine not in _OFF_STREAM:
        return False
    hand_off_run(make_run())
    return True


def split_into_runs(
    memory_objs: Sequence[MemoryObj | None],
) -> list[tuple[TransferEngine, int, int]]:
    """Split a group's chunks into consecutive same-engine runs.

    Args:
        memory_objs: One object per chunk; a None entry joins the next
            object's run, or the previous run at the end.

    Returns:
        ``(engine, start, end)`` per run, in chunk order.
    """
    engines: list[TransferEngine | None] = [
        None if memory_obj is None else select_engine(memory_obj)
        for memory_obj in memory_objs
    ]
    following: TransferEngine | None = None
    for index in range(len(engines) - 1, -1, -1):
        if engines[index] is None:
            engines[index] = following
        following = engines[index]
    runs: list[tuple[TransferEngine, int, int]] = []
    for index, engine in enumerate(engines):
        # Trailing None entries have no following object.
        current = engine or (runs[-1][0] if runs else TransferEngine.GPU_DMA)
        if runs and runs[-1][0] is current:
            runs[-1] = (current, runs[-1][1], index + 1)
        else:
            runs.append((current, index, index + 1))
    return runs
