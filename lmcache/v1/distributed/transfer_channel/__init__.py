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
