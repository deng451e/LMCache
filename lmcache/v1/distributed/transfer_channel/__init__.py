# SPDX-License-Identifier: Apache-2.0
# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
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
    "StagingRegion",
    "TransferChannelStagingRegistration",
    "enable_staging_buffers",
    "disable_staging_buffers",
    "find_staging_region",
    "staging_alignment",
    "acquire_transfer_channel_client",
    "release_transfer_channel_client",
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
            closed = _context
            closed.close()
            _context = None
        else:
            return
    with _clients_lock:
        for key in [key for key in _client_users if key[0] == id(closed)]:
            del _client_users[key]


# GPU-direct borrowing state, guarded by ``_context_lock``:
# - the staging target and its alignment: enabled by the P2P controller once
#   the context exists, disabled when it closes; read when the NIC DMA engine
#   registers a NIC staging buffer (TransferChannelStagingRegistration).
# - the staging regions: one per registered staging buffer, keyed by base
#   address; read by the NIC DMA engine on every borrowed retrieve.
_staging_target: TransferChannelContext | None = None
_staging_align_bytes = 1


@dataclass(frozen=True)
class StagingRegion:
    """A NIC staging buffer as a transfer-channel region.

    Attributes:
        context: The transfer channel context the buffer is registered with.
        region_id: The context's region id.
        base_ptr: Start address.
        size: Size in bytes.
    """

    context: TransferChannelContext
    region_id: int
    base_ptr: int
    size: int


_staging_regions: dict[int, StagingRegion] = {}


def enable_staging_buffers(context: TransferChannelContext, align_bytes: int) -> None:
    """Let NIC staging buffers register with ``context``, with slots aligned
    to ``align_bytes``.

    Only a GPU-direct borrowing reader calls this: its peers' objects are read
    straight into its NIC staging slots. Copy mode never registers GPU memory,
    so its hosts need no GPUDirect support.

    Args:
        context: The transfer channel context to register with.
        align_bytes: The alignment its regions need, the L1 alignment.

    Raises:
        ValueError: If ``align_bytes`` is not a positive power of two.
    """
    if align_bytes < 1 or align_bytes & (align_bytes - 1):
        raise ValueError(
            f"Staging slot alignment must be a power of two, got {align_bytes}"
        )
    global _staging_target, _staging_align_bytes
    with _context_lock:
        _staging_target = context
        _staging_align_bytes = align_bytes


def disable_staging_buffers() -> None:
    """Stop NIC staging buffers from registering."""
    global _staging_target, _staging_align_bytes
    with _context_lock:
        _staging_target = None
        _staging_align_bytes = 1


def staging_alignment() -> int:
    """Return the slot alignment NIC staging buffers need (1 when disabled)."""
    with _context_lock:
        return _staging_align_bytes


class TransferChannelStagingRegistration(BufferRegistration):
    """Register a NIC staging buffer with the transfer channel, so the NIC
    DMA engine can read a peer's objects straight into its slots.

    Pass it to :meth:`BaseCacheContext.create_staging_buffer`; it needs
    :func:`enable_staging_buffers`.
    """

    def register(self, buffer: torch.Tensor) -> None:
        """Register ``buffer`` as one transfer-channel region.

        Args:
            buffer: The NIC staging buffer.

        Raises:
            RuntimeError: If staging buffers are not enabled (no
                ``--p2p-load-mode gdr``), the buffer does not start and end on
                the staging alignment, or the transfer channel refuses the
                region.
        """
        with _context_lock:
            target = _staging_target
            align = _staging_align_bytes
        if target is None:
            raise RuntimeError(
                "Reading objects a peer owns into GPU memory needs --p2p-load-mode gdr"
            )
        ptr = buffer.data_ptr()
        if ptr % align or buffer.nbytes % align:
            raise RuntimeError(
                f"The staging buffer ({buffer.nbytes} bytes at {ptr:#x}) is not "
                f"aligned to {align} bytes; GPU-direct borrowing cannot read "
                "into it"
            )
        try:
            region_id = target.register_memory_region(
                ptr, buffer.nbytes, buffer.device.index
            )
        except ValueError as error:
            raise RuntimeError(
                "The transfer channel refused the staging buffer"
            ) from error
        with _context_lock:
            _staging_regions[ptr] = StagingRegion(target, region_id, ptr, buffer.nbytes)

    def deregister(self, buffer: torch.Tensor) -> None:
        """Forget ``buffer``'s region and release it; a buffer that was not
        registered is ignored.

        Args:
            buffer: The buffer previously passed to :meth:`register`.
        """
        with _context_lock:
            region = _staging_regions.pop(buffer.data_ptr(), None)
        if region is None:
            return
        try:
            region.context.deregister_memory_region(region.region_id)
        except KeyError:
            # The context was closed first, which released every region.
            logger.debug("Staging region %d was already released", region.region_id)


def find_staging_region(ptr: int) -> StagingRegion | None:
    """Return the registered staging region containing ``ptr``, if any.

    Args:
        ptr: An address inside a staging buffer.
    """
    with _context_lock:
        for region in _staging_regions.values():
            if region.base_ptr <= ptr < region.base_ptr + region.size:
                return region
    return None


# Peer clients shared by several adapters of the same peer (a removal can be
# deferred while a re-added adapter already uses the client): users per
# ``(context, peer url)``, guarded by ``_clients_lock``. A separate lock, as
# acquiring may connect to the peer.
_client_users: dict[tuple[int, str], int] = {}
_clients_lock = threading.Lock()


def acquire_transfer_channel_client(
    context: TransferChannelContext, peer_advertise_url: str
) -> TransferChannelClient:
    """Get the client for a peer and count the caller as one of its users.

    Every call must be paired with :func:`release_transfer_channel_client`.

    Args:
        context: The transfer channel context.
        peer_advertise_url: The peer's ``host:port``.

    Returns:
        The context's client for the peer.
    """
    key = (id(context), peer_advertise_url)
    with _clients_lock:
        client = context.get_transfer_channel_client(peer_advertise_url)
        _client_users[key] = _client_users.get(key, 0) + 1
    return client


def release_transfer_channel_client(
    context: TransferChannelContext, peer_advertise_url: str
) -> None:
    """Drop one user of a peer's client; the last one removes the client.

    Releasing a peer with no users does nothing.

    Args:
        context: The transfer channel context.
        peer_advertise_url: The peer's ``host:port``.
    """
    key = (id(context), peer_advertise_url)
    with _clients_lock:
        users = _client_users.get(key, 0)
        if users == 0:
            return
        if users > 1:
            _client_users[key] = users - 1
            return
        del _client_users[key]
        context.remove_transfer_channel_client(peer_advertise_url)
