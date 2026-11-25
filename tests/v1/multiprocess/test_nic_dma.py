# SPDX-License-Identifier: Apache-2.0
"""NIC DMA engine: a retrieve's peer-owned batches, returned by the transfer,
are read in waves into the NIC staging buffer, one read per peer per wave,
and their kernels are queued only after the reads land (CPU, fakes)."""

# Standard
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed import transfer_channel
from lmcache.v1.memory_management import (
    BorrowedMemoryObj,
    MemoryFormat,
    MemoryObjMetadata,
)
from lmcache.v1.multiprocess.object_group_transfer import (
    nic_dma,
    staged,
    transfer_kv_per_object_group,
    transfer_unfull_chunk,
    unfull_chunk,
)
import lmcache.lmcache_native as lmcache_native

OBJ = 256  # bytes per object and per slot
H2D = lmcache_native.TransferDirection.H2D
# NIC buffer: 4 slots (two batches of 2, waves of 2), then spare rows 4-5.
SLOTS, SPARE = 6, 4


class _Read:
    def __init__(self) -> None:
        self.finished, self.ok = False, True

    def is_finished(self) -> bool:
        return self.finished

    def succeeded(self) -> bool:
        return self.ok


class _Reader:
    """A peer; logs ("read", peer, names, slots)."""

    def __init__(self, name: str, log: list) -> None:
        self.name, self.log, self.reads = name, log, []

    def read_many(self, region_id, targets):
        assert region_id == 9
        names = [o.name for o, _ in targets]
        self.log.append(("read", self.name, names, [t // OBJ for _, t in targets]))
        self.reads.append(_Read())
        return self.reads[-1]


def _peer(reader: _Reader, name: str) -> BorrowedMemoryObj:
    meta = MemoryObjMetadata(
        shape=torch.Size([OBJ]),
        dtype=torch.uint8,
        address=0,
        phy_size=OBJ,
        ref_count=1,
        fmt=MemoryFormat.KV_2LTD,
        shapes=[torch.Size([OBJ])],
        dtypes=[torch.uint8],
    )
    obj = BorrowedMemoryObj(meta)
    obj.bind((lambda: True, lambda: None), reader=reader, address=0, source="p")
    obj.name = name  # type: ignore[attr-defined]
    return obj


class _Staging:
    """One object group per slot; kernel group k sits 64 bytes in."""

    def __init__(self, num_slots: int) -> None:
        self.max_batch_size = num_slots
        self.buffer = torch.zeros(num_slots * OBJ, dtype=torch.uint8)

    def get_temp_object_group_buffer(self, slot: int, group: int) -> torch.Tensor:
        return self.buffer[slot * OBJ : (slot + 1) * OBJ]

    def get_temp_kernel_group_buffer(self, slot: int, group: int) -> torch.Tensor:
        return self.buffer[slot * OBJ + group * 64 : (slot + 1) * OBJ]


class _Poller:
    """Runs watched jobs only when the test polls."""

    def __init__(self) -> None:
        self.jobs: list = []

    def watch(self, check, on_ready, on_abandon) -> None:
        self.jobs.append((check, on_ready, on_abandon))

    def poll(self) -> None:
        while ready := [job for job in self.jobs if job[0]()]:
            for job in ready:
                self.jobs.remove(job)
                job[1]()

    def stop(self) -> None:
        jobs, self.jobs = self.jobs, []
        for job in jobs:
            job[2]()


@pytest.fixture
def h(monkeypatch):
    log: list = []
    events: list = []
    nic: dict = {}
    channel = MagicMock(align_bytes=64)
    channel.register_memory_region.return_value = 9
    for module in (transfer_channel, nic_dma):
        monkeypatch.setattr(module, "get_transfer_channel_context", lambda: channel)

    def create(num_slots, align, registration):
        nic["args"] = (num_slots, align)
        nic["buf"] = _Staging(num_slots)
        registration.register(nic["buf"].buffer)
        return nic["buf"]

    shared = _Staging(2)
    cc = MagicMock(
        max_batch_size=2,
        lmcache_tokens_per_chunk=16,
        device=torch.device("cpu"),
        stream="s",
        kv_layer_groups_manager=SimpleNamespace(
            num_object_groups=1,
            object_groups=[SimpleNamespace(kernel_group_indices=[0, 1])],
            get_attn_desc=lambda: SimpleNamespace(is_full_attention=lambda g: True),
            get_subchunk_sw_size_tokens=lambda kg: 16,
        ),
    )
    cc.get_temp_object_group_buffer.side_effect = shared.get_temp_object_group_buffer
    cc.get_temp_kernel_group_buffer.side_effect = shared.get_temp_kernel_group_buffer
    cc.create_staging_buffer.side_effect = create
    cc.calculate_num_blocks.side_effect = lambda tokens, kg: tokens // 4

    def kernel(paged, ptrs, block_ids, *args) -> None:
        base = nic["buf"].buffer.data_ptr()
        if (ptrs[0] - base) % OBJ == 0:  # kernel group 0 only
            slots = [(p - base) // OBJ for p in ptrs]
            log.append(("kernel", slots, block_ids.tolist()))

    def record(device, stream):
        event = SimpleNamespace(done=False)
        events.append(event)
        return SimpleNamespace(is_complete=lambda: event.done)

    ops = MagicMock()
    ops.multi_layer_block_kv_transfer.side_effect = kernel
    monkeypatch.setattr(staged, "device_ops", ops)
    monkeypatch.setattr(unfull_chunk, "device_ops", ops)
    monkeypatch.setattr(
        nic_dma, "torch_dev", MagicMock(device=nullcontext, stream=nullcontext)
    )
    monkeypatch.setattr(nic_dma, "record_completion_event", record)
    monkeypatch.setattr(nic_dma, "NIC_STAGING_BYTES", OBJ)  # floor: 2 batches
    poller = _Poller()
    engine = nic_dma.NicDmaEngine(poller)  # type: ignore[arg-type]
    p = _Reader("p", log)

    def runs(objs, skip=0, block_ids=None):
        ids = block_ids or [torch.arange(4 * len(objs)) for _ in range(2)]
        return transfer_kv_per_object_group(
            cc, ids, objs, 0, 2, skip, H2D, transfer_key="t"
        )

    def retrieve(names, skip=0, block_ids=None):
        outcome: list = []
        objs = [_peer(p, n) for n in names]
        response = engine.read(
            cc, runs(objs, skip, block_ids), lambda ok: outcome.append(ok) or ok
        )
        return response, outcome

    yield SimpleNamespace(
        cc=cc,
        log=log,
        events=events,
        nic=nic,
        poller=poller,
        engine=engine,
        p=p,
        runs=runs,
        retrieve=retrieve,
    )


def _reads(log):
    return [e[1:] for e in log if e[0] == "read"]


def _kernels(log):
    return [e[1:] for e in log if e[0] == "kernel"]


def test_reads_land_in_nic_slots_and_kernels_follow_each_wave(h) -> None:
    response, outcome = h.retrieve("abcde")
    assert h.nic["args"] == (SLOTS, 64)
    # Two waves fill the four slots, one read each; the third waits.
    first, second = _reads(h.log)
    assert (first[:2], second[:2]) == (("p", ["a", "b"]), ("p", ["c", "d"]))
    assert set(first[2]) | set(second[2]) == {0, 1, 2, 3}
    assert _kernels(h.log) == [] and not response.done()

    h.p.reads[0].finished = True
    h.poller.poll()
    assert _kernels(h.log) == [(first[2], list(range(8)))]
    # Slots are free only once the wave's kernels finished.
    h.p.reads[1].finished = True
    h.poller.poll()
    assert len(_reads(h.log)) == 2
    h.events[0].done = True
    h.poller.poll()
    third = _reads(h.log)[2]
    assert third[:2] == ("p", ["e"]) and set(third[2]) <= set(first[2])

    h.p.reads[2].finished = True
    h.poller.poll()
    assert outcome == [True] and response.result(timeout=1) is True
    assert _kernels(h.log)[-1][1] == list(range(16, 20))
    h.retrieve("f")
    assert h.cc.create_staging_buffer.call_count == 1  # once per context


def test_batches_skip_chunks_and_keep_their_own_block_ids(h) -> None:
    """Chunks the kernel skips are not read; the block ids live in a buffer
    the next transfer restages, so each run copies them."""
    ids = [torch.arange(12) for _ in range(2)]
    h.retrieve("abc", skip=16, block_ids=ids)
    for t in ids:
        t.fill_(-1)
    assert [r[1] for r in _reads(h.log)] == [["b", "c"]]
    h.p.reads[0].finished = True
    h.poller.poll()
    assert [k[1] for k in _kernels(h.log)] == [list(range(8)), list(range(8, 12))]


def test_one_read_per_peer_per_wave(h) -> None:
    q = _Reader("q", h.log)
    h.engine.read(h.cc, h.runs([_peer(h.p, "a"), _peer(q, "b")]), lambda ok: ok)
    assert sorted(r[:2] for r in _reads(h.log)) == [("p", ["a"]), ("q", ["b"])]


def test_a_failed_read_fails_the_retrieve_without_kernels(h) -> None:
    _, outcome = h.retrieve("abcde")
    h.p.reads[0].ok = False
    h.p.reads[0].finished = True
    h.poller.poll()
    assert outcome == []  # the other read could still land in its slots
    h.p.reads[1].finished = True
    h.poller.poll()
    assert outcome == [False] and _kernels(h.log) == []
    assert len(_reads(h.log)) == 2  # the third wave is never read
    h.retrieve("wxyz")  # every slot is free again
    assert len(_reads(h.log)) == 4


def test_retrieves_share_the_slots_in_order(h) -> None:
    h.retrieve("abcdef")
    h.retrieve("gh")
    for read in h.p.reads:
        read.finished = True
    h.poller.poll()
    for event in h.events[:2]:
        event.done = True
        h.poller.poll()
    names = [r[1] for r in _reads(h.log)]
    assert names == [["a", "b"], ["c", "d"], ["e", "f"], ["g", "h"]]


def test_shutdown_abandons_the_reads(h) -> None:
    response, outcome = h.retrieve("ab")
    h.poller.stop()
    assert outcome == [False] and response.done() and _kernels(h.log) == []


def test_a_staging_buffer_that_cannot_be_created_fails_the_retrieve(h) -> None:
    """E.g. the transfer channel refuses the region: the retrieve completes
    as failed, so its read locks are released and vLLM recomputes."""
    h.cc.create_staging_buffer.side_effect = RuntimeError("no GPUDirect RDMA")
    response, outcome = h.retrieve("ab")
    assert outcome == [False] and response.result(timeout=1) is False
    assert _reads(h.log) == []


def _drain(h) -> threading.Thread:
    thread = threading.Thread(target=h.engine.drain, args=(h.cc, 30), daemon=True)
    thread.start()
    thread.join(0.1)
    return thread


def test_drain_waits_until_the_contexts_retrieves_completed(h) -> None:
    """A context is closed only after its reads landed and their kernels
    were queued: closing deregisters the slots and unmaps the KV cache."""
    assert not _drain(h).is_alive()  # nothing was read for this context
    h.retrieve("ab")
    drain = _drain(h)
    assert drain.is_alive()
    h.p.reads[0].finished = True
    h.poller.poll()
    drain.join(5)
    assert not drain.is_alive() and len(_kernels(h.log)) == 1


def test_drain_gives_up_after_its_timeout(h) -> None:
    """The caller then leaves the context open; a read that lands later
    still finds its slots and queues its kernels."""
    assert h.engine.drain(h.cc, 0.05) is True  # nothing to wait for
    _, outcome = h.retrieve("ab")
    assert h.engine.drain(h.cc, 0.05) is False
    h.p.reads[0].finished = True
    h.poller.poll()
    assert outcome == [True] and len(_kernels(h.log)) == 1


def test_drain_does_not_wait_after_close(h) -> None:
    """At shutdown the workers may be gone, so abandoned retrieves never
    complete."""
    h.retrieve("ab")
    h.poller.stop = lambda: None
    h.engine.close()
    assert not _drain(h).is_alive()


def test_unfull_chunks_use_the_spare_rows_one_at_a_time(h) -> None:
    """An unfull chunk's object can extend past its slot."""
    buffer = h.cc.get_temp_object_group_buffer(0, 0)
    runs = [
        transfer_unfull_chunk(
            h.cc, [torch.arange(4)] * 2, [0, 0], [0, 0], _peer(h.p, n), 0, buffer, H2D
        )
        for n in ("u1", "u2")
    ]
    assert all(run.spans_slots for run in runs) and h.log == []
    h.engine.read(h.cc, runs[:1], lambda ok: ok)
    h.engine.read(h.cc, runs[1:], lambda ok: ok)
    assert _reads(h.log) == [("p", ["u1"], [SPARE])]
    h.p.reads[0].finished = True
    h.poller.poll()
    assert _kernels(h.log) == [([SPARE], [0, 1, 2, 3])]
    h.events[0].done = True
    h.poller.poll()
    assert _reads(h.log)[1] == ("p", ["u2"], [SPARE])
