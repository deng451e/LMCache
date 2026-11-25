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
from lmcache.v1.distributed.transfer_channel import (  # noqa: E402
    TransferChannelStagingRegistration,
    disable_staging_buffers,
    enable_staging_buffers,
    find_staging_region,
)
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


def _enabled_context() -> MagicMock:
    """Enable staging buffers on a fake transfer channel context."""
    context = MagicMock()
    context.register_memory_region.return_value = 1
    enable_staging_buffers(context, _ALIGN)
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


def test_nic_staging_buffer_is_aligned_registered_and_released_on_close() -> None:
    context = _enabled_context()
    try:
        ctx = _context()
        staging = ctx.create_staging_buffer(
            6, _ALIGN, TransferChannelStagingRegistration()
        )
        try:
            assert staging.max_batch_size == 6
            (call_,) = context.register_memory_region.call_args_list
            ptr, size, device_index = call_.args
            assert ptr == staging.buffer.data_ptr() and size == staging.buffer.nbytes
            assert ptr % _ALIGN == 0 and size % _ALIGN == 0 and device_index == 0
            # Every slot a peer's object is read into starts on the alignment,
            # has the context's layout, and is found by address.
            for slot in range(staging.max_batch_size):
                buffer = staging.get_temp_object_group_buffer(slot, 0)
                offset = buffer.data_ptr() - ptr
                assert offset % _ALIGN == 0 and offset + buffer.nbytes <= size
                assert buffer.nbytes == ctx.get_temp_object_group_buffer(0, 0).nbytes
                assert (
                    staging.get_temp_kernel_group_buffer(slot, 0).shape
                    == ctx.get_temp_kernel_group_buffer(0, 0).shape
                )
                region = find_staging_region(buffer.data_ptr())
                assert region is not None and region.region_id == 1
            # The buffer the context's own transfers share is not registered.
            shared = ctx.get_temp_object_group_buffer(0, 0).data_ptr()
            assert find_staging_region(shared) is None
        finally:
            ctx.close()
    finally:
        disable_staging_buffers()
    context.deregister_memory_region.assert_called_once_with(1)
    assert find_staging_region(ptr) is None


def test_a_nic_staging_buffer_needs_gdr() -> None:
    ctx = _context()
    try:
        with pytest.raises(RuntimeError, match="p2p-load-mode gdr"):
            ctx.create_staging_buffer(2, _ALIGN, TransferChannelStagingRegistration())
    finally:
        ctx.close()


def test_close_tolerates_a_transfer_channel_that_closed_first() -> None:
    """At shutdown the P2P controller may close the transfer channel, which
    releases every region, before the cache context deregisters its own."""
    context = _enabled_context()
    context.deregister_memory_region.side_effect = KeyError(1)
    try:
        ctx = _context()
        ctx.create_staging_buffer(2, _ALIGN, TransferChannelStagingRegistration())
    finally:
        disable_staging_buffers()
    ctx.close()
    context.deregister_memory_region.assert_called_once_with(1)
