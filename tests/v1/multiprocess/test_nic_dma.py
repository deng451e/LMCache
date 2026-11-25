# SPDX-License-Identifier: Apache-2.0
"""NIC DMA engine: a retrieve's runs of objects a peer owns, returned by the
transfers, are read in waves into the NIC staging buffer, one read per peer
per wave, and their kernels are queued only after the reads land. Native and
device code is faked, so these run without CUDA."""

# Standard
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Callable, Sequence
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.transfer_channel import (
    TransferChannelStagingRegistration,
    disable_staging_buffers,
    enable_staging_buffers,
)
from lmcache.v1.memory_management import (
    BorrowedMemoryObj,
    MemoryFormat,
    MemoryObjMetadata,
    PendingRead,
)
from lmcache.v1.multiprocess.object_group_transfer import (
    nic_dma,
    staged,
    transfer_kv_per_object_group,
    transfer_unfull_chunk,
    unfull_chunk,
)
from lmcache.v1.multiprocess.object_group_transfer.nic_dma import (
    NicDmaEngine,
    StagingSlotPool,
)
from lmcache.v1.platform.base.object_group_transfer import DeferredNicRun
import lmcache.lmcache_native as lmcache_native

CHUNK = 16
BLOCKS_PER_CHUNK = 4
OBJ_BYTES = 256
BATCH = 2
# Four full-chunk slots (waves of 2 chunks), then two spare rows at the end.
SLOTS = 6
SPARE = 4
REGION = 9
ALIGN = 64
H2D = lmcache_native.TransferDirection.H2D


# ---------------------------------------------------------------- fakes


class _Read(PendingRead):
    """A read the test finishes."""

    def __init__(self) -> None:
        self.finished = False
        self.ok = True

    def is_finished(self) -> bool:
        return self.finished

    def succeeded(self) -> bool:
        return self.ok


_NAMES: dict[int, str] = {}


class _Reader:
    """A peer: records each read as (peer, names, slots)."""

    def __init__(self, name: str, log: list) -> None:
        self.name = name
        self._log = log
        self.reads: list[_Read] = []

    def read_many(
        self, region_id: int, targets: Sequence[tuple[BorrowedMemoryObj, int]]
    ) -> PendingRead:
        assert region_id == REGION
        self._log.append(
            (
                "read",
                self.name,
                [_NAMES[id(obj)] for obj, _ in targets],
                [offset // OBJ_BYTES for _, offset in targets],
            )
        )
        read = _Read()
        self.reads.append(read)
        return read


def _peer_object(reader: _Reader, name: str) -> BorrowedMemoryObj:
    meta = MemoryObjMetadata(
        shape=torch.Size([OBJ_BYTES]),
        dtype=torch.uint8,
        address=0x1000,
        phy_size=OBJ_BYTES,
        ref_count=1,
        fmt=MemoryFormat.KV_2LTD,
        shapes=[torch.Size([OBJ_BYTES])],
        dtypes=[torch.uint8],
    )
    obj = BorrowedMemoryObj(meta)
    obj.bind((lambda: True, lambda: None), reader=reader, address=0x1000, source="peer")
    _NAMES[id(obj)] = name
    return obj


class _Staging:
    """A StagingBuffer: one object group per slot over one CPU tensor;
    kernel group ``k`` sits 64 bytes into its object."""

    def __init__(self, num_slots: int) -> None:
        self.max_batch_size = num_slots
        self.buffer = torch.zeros(num_slots * OBJ_BYTES, dtype=torch.uint8)

    def get_temp_object_group_buffer(self, index: int, group: int) -> torch.Tensor:
        assert group == 0
        return self.buffer[index * OBJ_BYTES : (index + 1) * OBJ_BYTES]

    def get_temp_kernel_group_buffer(self, index: int, group: int) -> torch.Tensor:
        return self.buffer[index * OBJ_BYTES + group * 64 : (index + 1) * OBJ_BYTES]

    def slot_of(self, ptr: int) -> int:
        return (ptr - self.buffer.data_ptr()) // OBJ_BYTES


class _Poller:
    """Runs watched jobs only when the test polls."""

    def __init__(self) -> None:
        self.jobs: list[tuple[Callable[[], bool], Callable, Callable]] = []

    def watch(self, check, on_ready, on_abandon) -> None:
        self.jobs.append((check, on_ready, on_abandon))

    def poll(self) -> None:
        progressed = True
        while progressed:
            progressed = False
            for job in list(self.jobs):
                if job[0]():
                    self.jobs.remove(job)
                    job[1]()
                    progressed = True

    def abandon_all(self) -> None:
        jobs, self.jobs = self.jobs, []
        for job in jobs:
            job[2]()

    def stop(self) -> None:
        self.abandon_all()


class _Event:
    def __init__(self) -> None:
        self.complete = False

    def is_complete(self) -> bool:
        return self.complete


@pytest.fixture
def h(monkeypatch):
    """A context whose NIC staging buffer has 4 full slots and 2 spare rows;
    kernels and events are faked."""
    log: list = []
    events: list[_Event] = []
    created: list[tuple[int, int]] = []
    holder: dict[str, _Staging] = {}
    channel = MagicMock()
    channel.register_memory_region.return_value = REGION
    enable_staging_buffers(channel, ALIGN)

    def create_staging_buffer(num_slots, slot_align_bytes, registration):
        assert isinstance(registration, TransferChannelStagingRegistration)
        created.append((num_slots, slot_align_bytes))
        staging = _Staging(num_slots)
        registration.register(staging.buffer)
        holder["staging"] = staging
        return staging

    shared = _Staging(BATCH)
    cc = MagicMock()
    cc.get_temp_object_group_buffer.side_effect = shared.get_temp_object_group_buffer
    cc.get_temp_kernel_group_buffer.side_effect = shared.get_temp_kernel_group_buffer
    cc.create_staging_buffer.side_effect = create_staging_buffer
    cc.max_batch_size = BATCH
    cc.lmcache_tokens_per_chunk = CHUNK
    cc.kv_layer_groups_manager = SimpleNamespace(
        num_object_groups=1,
        object_groups=[SimpleNamespace(kernel_group_indices=[0, 1])],
        get_attn_desc=lambda: SimpleNamespace(is_full_attention=lambda og: True),
        get_subchunk_sw_size_tokens=lambda kg: CHUNK,
    )
    cc.calculate_num_blocks.side_effect = lambda tokens, kg: tokens // 4
    cc.device = torch.device("cpu")
    cc.stream = "ctx-stream"

    def kernel(paged, ptrs, block_ids, *args, **kwargs) -> None:
        staging = holder.get("staging")
        # Log kernel group 0 of the NIC staging buffer only.
        if (
            staging is not None
            and (ptrs[0] - staging.buffer.data_ptr()) % OBJ_BYTES == 0
        ):
            log.append(
                ("kernel", [staging.slot_of(p) for p in ptrs], block_ids.tolist())
            )

    ops = MagicMock()
    ops.multi_layer_block_kv_transfer.side_effect = kernel
    monkeypatch.setattr(staged, "device_ops", ops)
    monkeypatch.setattr(unfull_chunk, "device_ops", ops)
    monkeypatch.setattr(
        nic_dma,
        "torch_dev",
        SimpleNamespace(device=lambda d: nullcontext(), stream=lambda s: nullcontext()),
    )

    def record(device, stream):
        assert stream == "ctx-stream"
        log.append("event")
        event = _Event()
        events.append(event)
        return event

    monkeypatch.setattr(nic_dma, "record_completion_event", record)
    # Room for 4 full-chunk slots: twice the batch.
    monkeypatch.setattr(nic_dma, "NIC_STAGING_BYTES", 4 * OBJ_BYTES)
    poller = _Poller()
    yield SimpleNamespace(
        cc=cc,
        log=log,
        events=events,
        created=created,
        holder=holder,
        poller=poller,
        engine=NicDmaEngine(poller),
        reader=_Reader("p", log),
    )
    disable_staging_buffers()


def _block_ids(num_chunks: int) -> list[torch.Tensor]:
    return [torch.arange(num_chunks * BLOCKS_PER_CHUNK) for _ in range(2)]


def _runs(h, objs, skip_first_n_tokens: int = 0, block_ids=None):
    """The runs the dispatcher returns for retrieving ``objs``."""
    return transfer_kv_per_object_group(
        h.cc,
        block_ids if block_ids is not None else _block_ids(len(objs)),
        objs,
        0,
        BATCH,
        skip_first_n_tokens,
        H2D,
        transfer_key="t",
    )


def _read(h, runs):
    """Hand ``runs`` to the engine; return the response and the outcomes
    ``finish`` was called with."""
    outcome: list[bool] = []

    def finish(ok: bool) -> str:
        outcome.append(ok)
        return f"done:{ok}"

    return h.engine.read(h.cc, runs, finish), outcome


def _reads(log) -> list:
    return [entry[1:] for entry in log if entry[0] == "read"]


def _kernels(log) -> list:
    return [entry[1:] for entry in log if entry[0] == "kernel"]


# ---------------------------------------------------------------- pool


class _Wave:
    def __init__(self, name: str, num_slots: int, log: list) -> None:
        self.name = name
        self.num_slots = num_slots
        self._log = log
        self.slots: list[int] = []

    def start(self, slots: list[int]) -> None:
        self.slots = slots
        self._log.append(self.name)


def test_pool_waves_take_half_the_slots() -> None:
    assert StagingSlotPool(range(4)).wave_slots == 2
    assert StagingSlotPool(range(3)).wave_slots == 1
    with pytest.raises(ValueError):
        StagingSlotPool(range(0))
    with pytest.raises(ValueError):
        StagingSlotPool(range(4)).submit(_Wave("big", 3, []))


def test_pool_starts_waves_in_order_as_slots_free() -> None:
    log: list = []
    pool = StagingSlotPool(range(4))
    a, b, c, d = (_Wave(n, 2, log) for n in "abcd")
    for wave in (a, b, c):
        pool.submit(wave)
    small = _Wave("small", 1, log)
    pool.submit(small)  # waits behind c although one slot would do
    assert log == ["a", "b"]
    assert set(a.slots) | set(b.slots) == {0, 1, 2, 3}
    pool.release(a.slots[:1])
    assert log == ["a", "b"]
    pool.release(a.slots[1:])
    assert log == ["a", "b", "c"]
    pool.withdraw([small, d])
    pool.release(b.slots)
    assert log == ["a", "b", "c"]


# ---------------------------------------------------------------- dispatch


def test_retrieve_returns_one_run_per_kernel_batch(h) -> None:
    objs = [_peer_object(h.reader, n) for n in "abc"]
    runs = _runs(h, objs)
    assert [[_NAMES[id(o)] for o in run.memory_objs] for run in runs] == [
        ["a", "b"],
        ["c"],
    ]
    assert not any(run.spans_slots for run in runs)
    # Nothing was read or queued yet.
    assert h.log == []


def test_runs_leave_out_chunks_the_kernel_skips(h) -> None:
    objs = [_peer_object(h.reader, n) for n in "abc"]
    runs = _runs(h, objs, skip_first_n_tokens=CHUNK)
    assert [[_NAMES[id(o)] for o in run.memory_objs] for run in runs] == [
        ["b"],
        ["c"],
    ]


# ---------------------------------------------------------------- engine


def test_reads_land_in_nic_slots_and_kernels_follow_each_wave(h) -> None:
    objs = [_peer_object(h.reader, n) for n in "abcde"]
    response, outcome = _read(h, _runs(h, objs))
    assert h.created == [(SLOTS, ALIGN)]
    # Two waves fill the four slots, one read each; the third waits.
    first, second = _reads(h.log)
    assert (first[:2], second[:2]) == (("p", ["a", "b"]), ("p", ["c", "d"]))
    assert set(first[2]) | set(second[2]) == {0, 1, 2, 3}
    assert _kernels(h.log) == [] and not response.done()

    h.reader.reads[0].finished = True
    h.poller.poll()
    assert _kernels(h.log) == [(first[2], list(range(0, 8)))]
    assert len(_reads(h.log)) == 2 and outcome == []

    # Slots are free again only once the wave's kernels finished.
    h.reader.reads[1].finished = True
    h.poller.poll()
    assert len(_reads(h.log)) == 2
    h.events[0].complete = True
    h.poller.poll()
    third = _reads(h.log)[2]
    assert third[:2] == ("p", ["e"]) and set(third[2]) <= set(first[2])

    h.reader.reads[2].finished = True
    h.poller.poll()
    assert outcome == [True] and response.result(timeout=1) == "done:True"
    assert _kernels(h.log)[-1][1] == list(range(16, 20))


def test_the_nic_staging_buffer_is_created_once_per_context(h) -> None:
    _read(h, _runs(h, [_peer_object(h.reader, "a")]))
    _read(h, _runs(h, [_peer_object(h.reader, "b")]))
    assert h.created == [(SLOTS, ALIGN)]


def test_kernels_use_the_block_ids_the_transfer_was_given(h) -> None:
    """The staged block ids live in a buffer the next transfer restages."""
    block_ids = _block_ids(2)
    runs = _runs(h, [_peer_object(h.reader, n) for n in "ab"], block_ids=block_ids)
    for ids in block_ids:
        ids.fill_(-1)
    _read(h, runs)
    h.reader.reads[0].finished = True
    h.poller.poll()
    assert _kernels(h.log)[0][1] == list(range(0, 8))


def test_one_read_per_peer_per_wave(h) -> None:
    other = _Reader("q", h.log)
    _read(h, _runs(h, [_peer_object(h.reader, "a"), _peer_object(other, "b")]))
    assert sorted(r[:2] for r in _reads(h.log)) == [("p", ["a"]), ("q", ["b"])]


def test_a_failed_read_fails_the_retrieve_without_kernels(h) -> None:
    objs = [_peer_object(h.reader, n) for n in "abcde"]
    response, outcome = _read(h, _runs(h, objs))
    h.reader.reads[0].ok = False
    h.reader.reads[0].finished = True
    h.poller.poll()
    # The other read in flight could still land in its slots.
    assert outcome == []
    h.reader.reads[1].finished = True
    h.poller.poll()
    assert outcome == [False] and response.result(timeout=1) == "done:False"
    assert _kernels(h.log) == []
    assert len(_reads(h.log)) == 2  # the third wave is never read
    # Every slot is free again for the next retrieve.
    _read(h, _runs(h, [_peer_object(h.reader, n) for n in "wxyz"]))
    assert len(_reads(h.log)) == 4


def test_retrieves_share_the_slots_in_order(h) -> None:
    _read(h, _runs(h, [_peer_object(h.reader, n) for n in "abcdef"]))
    _read(h, _runs(h, [_peer_object(h.reader, n) for n in "gh"]))
    assert [r[1] for r in _reads(h.log)] == [["a", "b"], ["c", "d"]]
    for read in h.reader.reads:
        read.finished = True
    h.poller.poll()
    h.events[0].complete = True
    h.poller.poll()
    assert [r[1] for r in _reads(h.log)][2:] == [["e", "f"]]
    h.events[1].complete = True
    h.poller.poll()
    assert [r[1] for r in _reads(h.log)][3:] == [["g", "h"]]


def test_an_abandoned_read_fails_the_retrieve(h) -> None:
    response, outcome = _read(h, _runs(h, [_peer_object(h.reader, n) for n in "ab"]))
    h.poller.abandon_all()
    assert outcome == [False] and response.done()
    assert _kernels(h.log) == []


def test_an_exception_from_finish_fails_the_response(h) -> None:
    def finish(ok: bool) -> None:
        raise RuntimeError("tail failed")

    response = h.engine.read(h.cc, _runs(h, [_peer_object(h.reader, "a")]), finish)
    h.reader.reads[0].finished = True
    h.poller.poll()
    with pytest.raises(RuntimeError, match="tail failed"):
        response.result(timeout=1)


# ---------------------------------------------------------------- unfull


def _unfull_run(h, name: str) -> DeferredNicRun:
    run = transfer_unfull_chunk(
        h.cc,
        [torch.arange(4) for _ in range(2)],
        [0, 0],
        [0, 0],
        _peer_object(h.reader, name),
        0,
        h.cc.get_temp_object_group_buffer(0, 0),
        H2D,
    )
    assert isinstance(run, DeferredNicRun)
    return run


def test_an_unfull_chunk_a_peer_owns_is_deferred(h) -> None:
    run = _unfull_run(h, "tail")
    assert run.spans_slots and h.log == []


def test_unfull_chunks_use_the_spare_slot_one_at_a_time(h) -> None:
    """Their object can extend past one slot, so they land in the slot at the
    end of the buffer, which has room after it."""
    first, second = _unfull_run(h, "u1"), _unfull_run(h, "u2")
    _read(h, [first])
    _, outcome = _read(h, [second])
    assert _reads(h.log) == [("p", ["u1"], [SPARE])]
    h.reader.reads[0].finished = True
    h.poller.poll()
    assert _kernels(h.log) == [([SPARE], [0, 1, 2, 3])]
    h.events[0].complete = True
    h.poller.poll()
    assert _reads(h.log)[1] == ("p", ["u2"], [SPARE])
    h.reader.reads[1].finished = True
    h.poller.poll()
    assert outcome == [True]
