# SPDX-License-Identifier: Apache-2.0
"""Per-chunk engine selection: object type to engine, the split into runs,
and the copy path of each run. Copy paths are stubbed, so no CUDA build is
needed."""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.memory_management import (
    BorrowedMemoryObj,
    GDSMemoryObject,
    MemoryObj,
)
from lmcache.v1.multiprocess.object_group_transfer import (
    dispatch,
    transfer_kv_per_object_group,
)
from lmcache.v1.multiprocess.object_group_transfer.engines import (
    TransferEngine,
    select_engine,
    split_into_runs,
    stage,
    staged_off_stream,
)
from lmcache.v1.multiprocess.object_group_transfer.off_stream import (
    OffStreamRun,
    collect_off_stream_runs,
    hand_off_run,
)
import lmcache.lmcache_native as lmcache_native

GPU = TransferEngine.GPU_DMA
NVME = TransferEngine.NVME_DMA
NIC = TransferEngine.NIC_DMA


def _host() -> MemoryObj:
    return MagicMock(spec=MemoryObj)


def _storage() -> MemoryObj:
    return MagicMock(spec=GDSMemoryObject)


def _peer() -> MemoryObj:
    return MagicMock(spec=BorrowedMemoryObj)


def _objects(pattern: str) -> list[MemoryObj | None]:
    """``h`` host object, ``s`` storage-backed object, ``p`` object a peer
    owns, ``-`` missing."""
    make = {"h": _host, "s": _storage, "p": _peer, "-": lambda: None}
    return [make[c]() for c in pattern]


def test_engine_follows_the_object_type():
    assert select_engine(_host()) is GPU
    assert select_engine(_storage()) is NVME
    assert select_engine(_peer()) is NIC


def test_storing_into_an_object_a_peer_owns_is_rejected():
    with pytest.raises(ValueError, match="retrieved"):
        stage(
            NIC,
            [_peer()],
            [MagicMock()],
            lmcache_native.TransferDirection.D2H,
        )


def test_only_the_nic_engine_hands_a_retrieve_step_to_the_collector():
    """A caller that collects no off-stream runs cannot read objects a peer
    owns, as before the NIC DMA engine."""
    run = OffStreamRun(
        [_peer()], [torch.arange(2)], lambda staging: [], lambda staging, ids: None
    )
    built: list = []

    def make_run() -> OffStreamRun:
        built.append(run)
        return run

    # An engine that stages on the stream does not even build the run.
    assert not staged_off_stream(GPU, make_run)
    assert not staged_off_stream(NVME, make_run)
    assert built == []
    with pytest.raises(NotImplementedError):
        staged_off_stream(NIC, make_run)
    collected: list = []
    with collect_off_stream_runs(collected):
        assert staged_off_stream(NIC, make_run)
    assert [taken.memory_objs for taken in collected] == [run.memory_objs]


def _handed_off(block_ids: list[torch.Tensor]) -> OffStreamRun:
    return OffStreamRun(
        [_peer()], block_ids, lambda staging: [], lambda staging, ids: None
    )


def test_runs_of_one_staging_share_one_copy_of_the_block_ids():
    """The batches of a transfer pass the same staged block ids: they are
    copied once, and restaging the buffer afterwards does not reach the copy."""
    buffer = torch.arange(8)
    staged_ids = [buffer[:4], buffer[4:]]
    collected: list[OffStreamRun] = []
    with collect_off_stream_runs(collected):
        hand_off_run(_handed_off(staged_ids))
        hand_off_run(_handed_off(staged_ids))
        buffer.fill_(-1)
    first, second = collected
    assert first.block_ids is second.block_ids
    assert first.block_ids is not staged_ids
    assert [ids.tolist() for ids in first.block_ids] == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_a_restaged_buffer_gets_its_own_copy_of_the_block_ids():
    """A later transfer restages the buffer and passes new views of it: its
    runs get the new ids and the earlier runs keep theirs. A new collect
    block starts without a copy, even for the same staged list."""
    buffer = torch.arange(4)
    first_staging = [buffer[:]]
    collected: list[OffStreamRun] = []
    with collect_off_stream_runs(collected):
        hand_off_run(_handed_off(first_staging))
        buffer.add_(10)
        hand_off_run(_handed_off([buffer[:]]))
    with collect_off_stream_runs(collected):
        buffer.add_(10)
        hand_off_run(_handed_off(first_staging))
    assert [run.block_ids[0].tolist() for run in collected] == [
        [0, 1, 2, 3],
        [10, 11, 12, 13],
        [20, 21, 22, 23],
    ]


@pytest.mark.parametrize(
    "pattern, expected",
    [
        ("", []),
        ("hhhh", [(GPU, 0, 4)]),
        ("ssss", [(NVME, 0, 4)]),
        ("hhsshh", [(GPU, 0, 2), (NVME, 2, 4), (GPU, 4, 6)]),
        ("shhs", [(NVME, 0, 1), (GPU, 1, 3), (NVME, 3, 4)]),
        ("hppsp", [(GPU, 0, 1), (NIC, 1, 3), (NVME, 3, 4), (NIC, 4, 5)]),
        # A missing chunk joins the run of the next object ...
        ("--ss", [(NVME, 0, 4)]),
        ("hh-ss", [(GPU, 0, 2), (NVME, 2, 5)]),
        # ... or the previous one when nothing follows it.
        ("ss--", [(NVME, 0, 4)]),
        ("---", [(GPU, 0, 3)]),
    ],
)
def test_chunks_split_into_consecutive_same_engine_runs(pattern, expected):
    assert split_into_runs(_objects(pattern)) == expected


def _transfer(monkeypatch, pattern: str) -> list[tuple[str, tuple[int, int]]]:
    """Run the dispatcher with the copy paths stubbed and no direct copy;
    return each run's path and chunk range, in call order."""
    calls: list[tuple[str, tuple[int, int]]] = []
    monkeypatch.setattr(dispatch, "batch_memcpy_available", lambda: False)
    monkeypatch.setattr(dispatch, "native_plan_available", lambda: True)
    monkeypatch.setattr(
        dispatch,
        "run_object_group_transfer_plan",
        lambda *a, **k: calls.append(("plan", k["chunk_range"])),
    )
    monkeypatch.setattr(
        dispatch,
        "run_staged_loop",
        lambda *a, **k: calls.append((k["engine"].value, k["chunk_range"])),
    )
    transfer_kv_per_object_group(
        MagicMock(),
        block_ids_gpu=[MagicMock()],
        memory_objs=_objects(pattern),
        object_group_id=0,
        batch_size=4,
        skip_first_n_tokens=0,
        direction=lmcache_native.TransferDirection.H2D,
        transfer_key="test",
        block_ids_host=[[1, 2]],
    )
    return calls


def test_single_engine_group_is_one_transfer_over_every_chunk(monkeypatch):
    assert _transfer(monkeypatch, "hhhhh") == [("plan", (0, 5))]


def test_empty_group_runs_no_copy_path(monkeypatch):
    assert _transfer(monkeypatch, "") == []


def test_each_run_goes_to_its_engine_in_chunk_order(monkeypatch):
    """Only host memory may use the native plan; the other engines stage one
    batch at a time."""
    assert _transfer(monkeypatch, "hhpppsh") == [
        ("plan", (0, 2)),
        ("nic_dma", (2, 5)),
        ("nvme_dma", (5, 6)),
        ("plan", (6, 7)),
    ]


def test_host_memory_stages_per_batch_without_the_native_plan(monkeypatch):
    calls: list[tuple[str, tuple[int, int]]] = []
    monkeypatch.setattr(dispatch, "batch_memcpy_available", lambda: False)
    monkeypatch.setattr(dispatch, "native_plan_available", lambda: False)
    monkeypatch.setattr(
        dispatch,
        "run_staged_loop",
        lambda *a, **k: calls.append((k["engine"].value, k["chunk_range"])),
    )
    transfer_kv_per_object_group(
        MagicMock(),
        block_ids_gpu=[MagicMock()],
        memory_objs=_objects("hh"),
        object_group_id=0,
        batch_size=4,
        skip_first_n_tokens=0,
        direction=lmcache_native.TransferDirection.H2D,
        transfer_key="test",
    )
    assert calls == [("gpu_dma", (0, 2))]
