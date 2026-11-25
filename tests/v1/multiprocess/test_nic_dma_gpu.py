# SPDX-License-Identifier: Apache-2.0
"""NIC DMA engine with the real block kernel: peer objects read into the NIC
staging buffer fill the paged KV cache exactly as the same bytes staged in
the shared buffer do. The peer read is a device copy on another stream."""

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type

pytest.importorskip("cupy", reason="GPU cache context tests require cupy")
pytest.importorskip("lmcache.cuda_ops", reason="Requires lmcache.cuda_ops")

# First Party
from lmcache.v1.memory_management import (  # noqa: E402
    BorrowedMemoryObj,
    MemoryFormat,
    MemoryObjMetadata,
)
from lmcache.v1.multiprocess.host_poller import HostPoller  # noqa: E402
from lmcache.v1.multiprocess.object_group_transfer import (  # noqa: E402
    downsample_and_stage_block_ids,
    nic_dma,
    staged,
    transfer_kv_per_object_group,
)
from lmcache.v1.multiprocess.object_group_transfer.engines import (  # noqa: E402
    TransferEngine,
)
from lmcache.v1.multiprocess.object_group_transfer.nic_staging import (  # noqa: E402
    collect_peer_runs,
)
from tests.v1.platform.devices.cuda.test_gdr_staging_region import (  # noqa: E402
    _channel,
    _context,
)
import lmcache.lmcache_native as lmcache_native  # noqa: E402

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(torch_device_type != "cuda", reason="Requires the CUDA backend"),
    pytest.mark.skipif(not torch_dev.is_available(), reason="Requires a GPU"),
]

_H2D = lmcache_native.TransferDirection.H2D
_BLOCK_IDS = [6, 7, 0, 1, 4, 5]  # 3 chunks of 2 blocks


class _CopyReader:
    """A peer whose read is a device copy into the NIC staging buffer."""

    def __init__(self) -> None:
        self.payloads: dict[int, torch.Tensor] = {}
        self.staging: torch.Tensor | None = None
        self.stream = torch.cuda.Stream()

    def read_many(self, region_id, targets):
        assert self.staging is not None
        with torch.cuda.stream(self.stream):
            for obj, offset in targets:
                data = self.payloads[id(obj)]
                self.staging[offset : offset + data.numel()].copy_(
                    data, non_blocking=True
                )
            event = torch.cuda.Event()
            event.record(self.stream)
        return type(
            "_Read", (), {"is_finished": event.query, "succeeded": lambda self: True}
        )()


def test_peer_reads_and_staged_copies_fill_the_kv_cache_alike(monkeypatch) -> None:
    _channel(monkeypatch)
    nic_ctx, ref_ctx = _context(), _context()
    row = nic_ctx.get_temp_object_group_buffer(0, 0).nbytes
    payloads = [
        torch.randint(0, 256, (row,), dtype=torch.uint8).pin_memory() for _ in range(3)
    ]
    reader = _CopyReader()
    objs = []
    for payload in payloads:
        meta = MemoryObjMetadata(
            shape=torch.Size([row]),
            dtype=torch.uint8,
            address=0,
            phy_size=row,
            ref_count=1,
            fmt=MemoryFormat.KV_2LTD,
        )
        obj = BorrowedMemoryObj(meta)
        obj.bind((lambda: True, lambda: None), reader=reader, address=0, source="p")
        reader.payloads[id(obj)] = payload
        objs.append(obj)
    create = nic_ctx.create_staging_buffer

    def create_and_watch(*args):
        staging = create(*args)
        reader.staging = staging.buffer
        return staging

    monkeypatch.setattr(nic_ctx, "create_staging_buffer", create_and_watch)
    # The reference: the same bytes staged in the shared buffer.
    monkeypatch.setattr(
        staged,
        "stage",
        lambda engine, srcs, buffers, direction: [
            buffer.copy_(src) for src, buffer in zip(srcs, buffers, strict=True)
        ],
    )
    engine = nic_dma.NicDmaEngine(HostPoller())
    try:
        with torch.cuda.stream(nic_ctx.stream):
            ids = downsample_and_stage_block_ids(nic_ctx, [list(_BLOCK_IDS)] * 2)
            runs: list = []
            with collect_peer_runs(runs):
                transfer_kv_per_object_group(
                    nic_ctx,
                    ids,
                    objs,
                    0,
                    nic_ctx.max_batch_size,
                    0,
                    _H2D,
                    transfer_key="n",
                )
        assert runs and engine.read(nic_ctx, runs, lambda ok: ok).result(timeout=30)
        nic_ctx.stream.synchronize()

        with torch.cuda.stream(ref_ctx.stream):
            ids = downsample_and_stage_block_ids(ref_ctx, [list(_BLOCK_IDS)] * 2)
            staged.run_staged_loop(
                ref_ctx, ids, payloads, 0, 1, 0, _H2D, engine=TransferEngine.GPU_DMA
            )
        ref_ctx.stream.synchronize()

        # Random bytes include NaN patterns: compare bits.
        for nic, ref in zip(nic_ctx.kv_tensors, ref_ctx.kv_tensors, strict=True):
            assert torch.equal(nic.view(torch.int16), ref.view(torch.int16))
            assert nic.view(torch.int16).any()
    finally:
        engine.close()
        nic_ctx.close()
        ref_ctx.close()
