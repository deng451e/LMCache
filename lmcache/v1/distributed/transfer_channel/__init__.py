# SPDX-License-Identifier: Apache-2.0
# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING
import threading

# First Party
from lmcache.v1.common.buffer_registration import BufferRegistration
from lmcache.v1.common.logging import init_logger
from lmcache.v1.common.transfer_channel.api import (
    TransferChannelAddress,
    TransferChannelTransferResult,
)
from lmcache.v1.distributed.transfer_channel.abstract import (
    TransferChannelClient,
    TransferChannelContext,
    TransferChannelServer,
)
from lmcache.v1.distributed.transfer_channel.factory import (
    create_transfer_channel_context,
)

# Import the implementations so they self-register their factories.
import lmcache.v1.distributed.transfer_channel.impl  # noqa: F401

if TYPE_CHECKING:
    # Third Party
    import torch

    # First Party
    from lmcache.v1.distributed.internal_api import L1MemoryDesc

__all__ = [
    "TransferChannelAddress",
    "TransferChannelTransferResult",
    "TransferChannelContext",
    "TransferChannelServer",
    "TransferChannelClient",
    "initialize_transfer_channel_context",
    "get_transfer_channel_context",
    "delete_transfer_channel_context",
    "TransferChannelStagingRegistration",
]

logger = init_logger(__name__)

_context: TransferChannelContext | None = None
_context_lock = threading.Lock()


def initialize_transfer_channel_context(
    transfer_channel_type: str,
    l1_memory_desc: "L1MemoryDesc",
    listen_url: str,
    advertise_url: str,
    **kwargs,
) -> TransferChannelContext:
    """Create the global transfer channel context.

    Args:
        transfer_channel_type: Registered implementation name, such as
            ``"nixl"`` or ``"mooncake"``.
        l1_memory_desc: Describes the L1 memory region to register.
        listen_url: ``host:port`` this peer's singleton server binds to.
        advertise_url: ``host:port`` this peer advertises as its identity (the
            key peers store its reverse client under).
        **kwargs: Implementation-specific keyword arguments forwarded to the
            factory.

    Returns:
        The created context (also retrievable via ``get_transfer_channel_context``).

    Raises:
        RuntimeError: If a context has already been initialized.
        ValueError: If no factory is registered for ``transfer_channel_type``.
    """
    global _context
    with _context_lock:
        if _context is not None:
            raise RuntimeError(
                "Transfer channel context already initialized; call "
                "delete_transfer_channel_context() first."
            )
        _context = create_transfer_channel_context(
            transfer_channel_type=transfer_channel_type,
            l1_memory_desc=l1_memory_desc,
            listen_url=listen_url,
            advertise_url=advertise_url,
            **kwargs,
        )
        return _context


def get_transfer_channel_context() -> TransferChannelContext:
    """Get the global transfer channel context.

    Raises:
        RuntimeError: If the context has not been initialized yet.
    """
    with _context_lock:
        if _context is None:
            raise RuntimeError("Transfer channel context not initialized.")
        return _context


def delete_transfer_channel_context() -> None:
    """Delete the global transfer channel context, if it exists."""
    global _context
    with _context_lock:
        if _context is not None:
            _context.close()
            _context = None


class TransferChannelStagingRegistration(BufferRegistration):
    """Registers one staging buffer as a region of the global transfer
    channel context, so peers' objects can be read straight into its slots."""

    def __init__(self) -> None:
        self._context: TransferChannelContext | None = None
        self.region_id = -1

    def staging_alignment(self) -> int:
        """The transfer channel context's ``align_bytes``.

        Raises:
            RuntimeError: If there is no transfer channel context.
        """
        return get_transfer_channel_context().align_bytes

    def register(self, buffer: torch.Tensor) -> None:
        """Register ``buffer``; :attr:`region_id` is then its region.

        Raises:
            RuntimeError: If there is no transfer channel context, or it
                refuses the region.
        """
        target = get_transfer_channel_context()
        try:
            self.region_id = target.register_memory_region(
                buffer.data_ptr(), buffer.nbytes, buffer.device.index
            )
        except ValueError as error:
            raise RuntimeError("The transfer channel refused the buffer") from error
        self._context = target

    def deregister(self, buffer: torch.Tensor) -> None:
        """Release the region; nothing to do if :meth:`register` failed."""
        if self._context is None:
            return
        try:
            self._context.deregister_memory_region(self.region_id)
        except KeyError:
            # The context was closed first, which released every region.
            logger.debug("Staging region %d was already released", self.region_id)
        self._context = None
