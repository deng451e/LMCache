# SPDX-License-Identifier: Apache-2.0
"""GPU cache context: the staging buffer registered as a transfer-channel
device region for GPU-direct borrowing (CUDA)."""

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


def test_context_registers_its_staging_buffer_with_aligned_slots() -> None:
    context = _enabled_context()
    try:
        ctx = _context()
        ctx.register_staging_buffer(TransferChannelStagingRegistration())
    finally:
        disable_staging_buffers()
    try:
        assert ctx.max_batch_size == 4
        (call_,) = context.register_memory_region.call_args_list
        ptr, size, device_index = call_.args
        assert ptr % _ALIGN == 0
        assert size % _ALIGN == 0
        assert device_index == 0

        # Every slot a peer's object is read into starts on the alignment,
        # and the NIC DMA engine finds the region from any slot address.
        for batch in range(ctx.max_batch_size):
            slot = ctx.get_temp_object_group_buffer(batch, 0)
            offset = slot.data_ptr() - ptr
            assert offset % _ALIGN == 0 and offset + slot.nbytes <= size
            region = find_staging_region(slot.data_ptr())
            assert region is not None and region.region_id == 1
    finally:
        ctx.close()
    context.deregister_memory_region.assert_called_once_with(1)
    assert find_staging_region(ptr) is None


def test_registration_does_nothing_while_staging_buffers_are_disabled() -> None:
    context = MagicMock()
    ctx = _context()
    try:
        ctx.register_staging_buffer(TransferChannelStagingRegistration())
    finally:
        ctx.close()
    context.register_memory_region.assert_not_called()


def test_an_unaligned_buffer_is_refused_before_registration() -> None:
    """A context whose staging layout ignores the alignment (for example a
    subclass with its own ``_make_temp_buffer``) must fail loudly."""
    context = _enabled_context()
    try:
        buffer = torch.empty(_ALIGN + 512, dtype=torch.uint8, device="cuda:0")[512:]
        with pytest.raises(RuntimeError, match="align"):
            TransferChannelStagingRegistration().register(buffer)
    finally:
        disable_staging_buffers()
    context.register_memory_region.assert_not_called()


def test_close_tolerates_a_transfer_channel_that_closed_first() -> None:
    """At shutdown the P2P controller may close the transfer channel, which
    releases every region, before the cache context deregisters its own."""
    context = _enabled_context()
    context.deregister_memory_region.side_effect = KeyError(1)
    try:
        ctx = _context()
        ctx.register_staging_buffer(TransferChannelStagingRegistration())
    finally:
        disable_staging_buffers()
    ctx.close()
    context.deregister_memory_region.assert_called_once_with(1)
