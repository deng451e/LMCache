# SPDX-License-Identifier: Apache-2.0
"""GPU cache context: the NIC staging buffer, registered as a
transfer-channel device region for GPU-direct borrowing (CUDA)."""

# Standard
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
from lmcache.v1.distributed import transfer_channel  # noqa: E402
from lmcache.v1.distributed.transfer_channel import (  # noqa: E402
    TransferChannelStagingRegistration,
)
from lmcache.v1.multiprocess.object_group_transfer import nic_dma  # noqa: E402
from lmcache.v1.platform.devices.cuda.cache_context import (  # noqa: E402
    GPUCacheContext,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(torch_device_type != "cuda", reason="Requires the CUDA backend"),
    pytest.mark.skipif(not torch_dev.is_available(), reason="Requires a GPU"),
]

_ALIGN = 65536


class _FakeIPCWrapper:
    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def to_tensor(self) -> torch.Tensor:
        return self._tensor

    def close(self) -> None:
        return None


def _channel(monkeypatch) -> MagicMock:
    """Install a fake global transfer channel context."""
    context = MagicMock(align_bytes=_ALIGN)
    context.register_memory_region.return_value = 1
    for module in (transfer_channel, nic_dma):
        monkeypatch.setattr(module, "get_transfer_channel_context", lambda: context)
    return context


def _context() -> GPUCacheContext:
    tensors = [
        torch.zeros([2, 8, 16, 4, 32], dtype=torch.bfloat16, device="cuda:0")
        for _ in range(2)
    ]
    return GPUCacheContext(
        [_FakeIPCWrapper(t) for t in tensors],  # type: ignore[misc]
        lmcache_tokens_per_chunk=32,
        engine_type=EngineType.VLLM,
    )


def test_nic_staging_buffer_is_aligned_registered_and_released_on_close(
    monkeypatch,
) -> None:
    context = _channel(monkeypatch)
    registration = TransferChannelStagingRegistration()
    ctx = _context()
    staging = ctx.create_staging_buffer(6, registration)
    assert staging.max_batch_size == 6 and registration.region_id == 1
    (call_,) = context.register_memory_region.call_args_list
    ptr, size, device_index = call_.args
    assert (ptr, size) == (staging.buffer.data_ptr(), staging.buffer.nbytes)
    assert ptr % _ALIGN == 0 and size % _ALIGN == 0 and device_index == 0
    # Every slot starts on the alignment and has the context's layout.
    for slot in range(6):
        buffer = staging.get_temp_object_group_buffer(slot, 0)
        offset = buffer.data_ptr() - ptr
        assert offset % _ALIGN == 0 and offset + buffer.nbytes <= size
        assert buffer.nbytes == ctx.get_temp_object_group_buffer(0, 0).nbytes
        assert (
            staging.get_temp_kernel_group_buffer(slot, 0).shape
            == ctx.get_temp_kernel_group_buffer(0, 0).shape
        )
    # At shutdown the transfer channel may close first, which already
    # released the region.
    context.deregister_memory_region.side_effect = KeyError(1)
    ctx.close()
    context.deregister_memory_region.assert_called_once_with(1)


def test_a_nic_staging_buffer_needs_a_transfer_channel() -> None:
    ctx = _context()
    try:
        with pytest.raises(RuntimeError, match="not initialized"):
            ctx.create_staging_buffer(2, TransferChannelStagingRegistration())
    finally:
        ctx.close()
