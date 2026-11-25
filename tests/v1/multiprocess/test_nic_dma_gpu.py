# SPDX-License-Identifier: Apache-2.0
"""NIC DMA engine on a real GPU cache context: peer objects read into the NIC
staging buffer and copied by the real block kernel land in the paged KV cache
exactly as the same bytes staged in the shared buffer do. The peer read is a
device copy on another stream, so no transfer channel is needed (CUDA)."""

# Standard
from typing import Sequence
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.utils import EngineType

pytest.importorskip("cupy", reason="GPU cache context tests require cupy")
pytest.importorskip("lmcache.cuda_ops", reason="Requires lmcache.cuda_ops")

# First Party
from lmcache.v1.distributed.transfer_channel import (  # noqa: E402
    disable_staging_buffers,
    enable_staging_buffers,
)
from lmcache.v1.memory_management import (  # noqa: E402
    BorrowedMemoryObj,
    MemoryFormat,
    MemoryObjMetadata,
    PendingRead,
)
from lmcache.v1.multiprocess.host_poller import HostPoller  # noqa: E402
from lmcache.v1.multiprocess.object_group_transfer import (  # noqa: E402
    downsample_and_stage_block_ids,
    nic_dma,
    staged,
    transfer_kv_per_object_group,
)
from lmcache.v1.platform.devices.cuda.cache_context import (  # noqa: E402
    GPUCacheContext,
)
import lmcache.lmcache_native as lmcache_native  # noqa: E402

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(torch_device_type != "cuda", reason="Requires the CUDA backend"),
    pytest.mark.skipif(not torch_dev.is_available(), reason="Requires a GPU"),
]

_ALIGN = 65536
_H2D = lmcache_native.TransferDirection.H2D
# Chunks 0..2 go to these blocks (2 blocks of 16 tokens per 32-token chunk).
_BLOCK_IDS = [6, 7, 0, 1, 4, 5]


class _FakeIPCWrapper:
    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def to_tensor(self) -> torch.Tensor:
        return self._tensor

    def close(self) -> None:
        return None


def _context() -> tuple[GPUCacheContext, list[torch.Tensor]]:
    tensors = [
        torch.zeros([2, 8, 16, 4, 32], dtype=torch.bfloat16, device="cuda:0")
        for _ in range(2)
    ]
    ctx = GPUCacheContext(
        [_FakeIPCWrapper(t) for t in tensors],  # type: ignore[misc]
        lmcache_tokens_per_chunk=32,
        engine_type=EngineType.VLLM,
    )
    return ctx, tensors


class _DeviceCopy(PendingRead):
    def __init__(self, event: torch.cuda.Event) -> None:
        self._event = event

    def is_finished(self) -> bool:
        return self._event.query()

    def succeeded(self) -> bool:
        return True


class _CopyReader:
    """A peer whose read is a device copy into the registered NIC buffer."""

    def __init__(self, data: dict[int, torch.Tensor]) -> None:
        self.data = data
        self.staging: torch.Tensor | None = None
        self.reads = 0
        self._stream = torch.cuda.Stream()

    def read_many(
        self, region_id: int, targets: Sequence[tuple[BorrowedMemoryObj, int]]
    ) -> PendingRead:
        assert self.staging is not None
        self.reads += 1
        with torch.cuda.stream(self._stream):
            for obj, offset in targets:
                payload = self.data[id(obj)]
                self.staging[offset : offset + payload.numel()].copy_(
                    payload, non_blocking=True
                )
            event = torch.cuda.Event()
            event.record(self._stream)
        return _DeviceCopy(event)


def _peer_object(reader: _CopyReader, payload: torch.Tensor) -> BorrowedMemoryObj:
    meta = MemoryObjMetadata(
        shape=torch.Size([payload.numel()]),
        dtype=torch.uint8,
        address=0,
        phy_size=payload.numel(),
        ref_count=1,
        fmt=MemoryFormat.KV_2LTD,
        shapes=[torch.Size([payload.numel()])],
        dtypes=[torch.uint8],
    )
    obj = BorrowedMemoryObj(meta)
    obj.bind((lambda: True, lambda: None), reader=reader, address=0, source="peer")
    reader.data[id(obj)] = payload
    return obj


def test_peer_reads_and_staged_copies_fill_the_kv_cache_alike(monkeypatch) -> None:
    nic_ctx, nic_kv = _context()
    ref_ctx, ref_kv = _context()
    row = nic_ctx.get_temp_object_group_buffer(0, 0).nbytes
    # The floor of two kernel batches: 8 slots, waves of 4 chunks.
    monkeypatch.setattr(nic_dma, "NIC_STAGING_BYTES", row)
    payloads = [
        torch.randint(0, 256, (row,), dtype=torch.uint8).pin_memory() for _ in range(3)
    ]
    reader = _CopyReader({})
    objs = [_peer_object(reader, p) for p in payloads]
    original_create = nic_ctx.create_staging_buffer

    def create(num_slots, slot_align_bytes, registration):
        staging = original_create(num_slots, slot_align_bytes, registration)
        reader.staging = staging.buffer
        return staging

    monkeypatch.setattr(nic_ctx, "create_staging_buffer", create)
    channel = MagicMock()
    channel.register_memory_region.return_value = 1
    enable_staging_buffers(channel, _ALIGN)
    poller = HostPoller()
    engine = nic_dma.NicDmaEngine(poller)
    try:
        with torch.cuda.stream(nic_ctx.stream):
            block_ids = downsample_and_stage_block_ids(
                nic_ctx, [list(_BLOCK_IDS), list(_BLOCK_IDS)]
            )
            runs = transfer_kv_per_object_group(
                nic_ctx,
                block_ids,
                objs,
                0,
                nic_ctx.max_batch_size,
                0,
                _H2D,
                transfer_key="nic",
            )
        assert runs is not None
        response = engine.read(nic_ctx, runs, lambda ok: ok)
        assert response.result(timeout=30) is True
        nic_ctx.stream.synchronize()
        assert reader.reads == 1  # one wave, one peer

        # The same bytes through the shared staging buffer, one chunk a batch.
        with torch.cuda.stream(ref_ctx.stream):
            ref_ids = downsample_and_stage_block_ids(
                ref_ctx, [list(_BLOCK_IDS), list(_BLOCK_IDS)]
            )
            for chunk, payload in enumerate(payloads):
                ref_ctx.get_temp_object_group_buffer(0, 0).copy_(
                    payload, non_blocking=True
                )
                staged.launch_batch_kernels(ref_ctx, ref_ids, 0, chunk, 1, 0, _H2D)
        ref_ctx.stream.synchronize()

        # Random bytes include NaN patterns: compare bits, not values.
        for nic, ref in zip(nic_kv, ref_kv, strict=True):
            assert torch.equal(nic.view(torch.int16), ref.view(torch.int16))
        assert any(bool(t.view(torch.int16).any()) for t in nic_kv)
    finally:
        engine.close()
        nic_ctx.close()
        ref_ctx.close()
        disable_staging_buffers()
