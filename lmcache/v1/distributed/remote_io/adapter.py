# SPDX-License-Identifier: Apache-2.0
"""Abstract interface for RemoteIOAdapter.

Combines the lookup protocol (ZMQ LookupRequest / UnpinRequest fan-out) with
RDMA data transfer (NIXL/UCX). Internal to RemoteL2Adapter — not called
directly by PrefetchController or other external components.
"""

# Future
from __future__ import annotations

# Standard
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # First Party
    from lmcache.native_storage_ops import Bitmap

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.memory_management import MemoryObj

IOTaskId = int
"""Opaque task identifier returned by submit_lookup_task and submit_fetch_task."""


class LocalMemHandle:
    """Opaque handle for a locally registered memory buffer.

    Returned by register_local_memory(). Reused across all fetches against
    the same buffer — register once at startup.
    """


class RemoteIOAdapter(ABC):
    """Client-side I/O adapter for remote KV-cache peers.

    Combines the lookup protocol (ZMQ LookupRequest / UnpinRequest fan-out)
    with RDMA data transfer (NIXL/UCX). All methods are thread-safe.

    Handle cache lifecycle:
        _handle_cache[task_id] is created by submit_lookup_task, read (not
        removed) by submit_fetch_task, and fully released by submit_unlock.
        submit_unlock is called twice per request — once for unneeded keys
        (before fetch) and once for plan keys (after fetch) — together
        covering all found keys.
    """

    ###########################
    # Local memory registration
    ###########################

    @abstractmethod
    def register_local_memory(
        self,
        ptr: int,
        size: int,
        device: str,
    ) -> LocalMemHandle:
        """Register the L1 buffer for RDMA access.

        Called once at startup. The returned handle is used implicitly by all
        subsequent fetch operations.

        Args:
            ptr:    Base address of the L1 buffer.
            size:   Buffer size in bytes.
            device: "cpu" or "cuda".

        Returns:
            Opaque handle for the registered buffer.

        Raises:
            RuntimeError: If memory registration fails.
        """

    #############################################
    # Metadata (called by RemoteController only)
    #############################################

    @abstractmethod
    def get_local_metadata(self) -> bytes:
        """Return serialised NIXL agent descriptor for the Init handshake.

        Called by RemoteController during register_peer() to send in
        InitRequest.local_agent_metadata.

        Returns:
            Opaque serialised agent descriptor bytes.
        """

    @abstractmethod
    def get_local_xfer_descs(self) -> bytes:
        """Return serialised local transfer descriptors for the MemReg handshake.

        Called by RemoteController during register_peer() after Init completes.

        Returns:
            Opaque serialised transfer descriptor bytes.
        """

    ###############################################
    # Peer lifecycle (called by RemoteController)
    ###############################################

    @abstractmethod
    def connect_peer(
        self,
        peer_id: str,
        endpoint: str,
        unpin_endpoint: str,
        peer_metadata: bytes,
        peer_xfer_descs: bytes,
    ) -> None:
        """Register a peer for lookup and RDMA.

        Called by RemoteController after the Init/MemReg handshake completes.
        Creates a ZMQ REQ socket to `endpoint` for lookup traffic (Lazy
        Pirate) and a ZMQ PUSH socket to `unpin_endpoint` for fire-and-forget
        UnpinRequests. Also registers the peer's NIXL descriptors for RDMA.

        Args:
            peer_id:         Logical peer identifier.
            endpoint:        ZMQ REQ endpoint, e.g. "tcp://host:5200".
            unpin_endpoint:  ZMQ PUSH endpoint, e.g. "tcp://host:5201".
            peer_metadata:   Serialised NIXL agent descriptor from InitResponse.
            peer_xfer_descs: Serialised NIXL transfer descriptors from MemRegResponse.

        Raises:
            RuntimeError: If NIXL descriptor registration fails.
        """

    @abstractmethod
    def disconnect_peer(self, peer_id: str) -> None:
        """Remove peer state, drain in-flight RDMA, and close ZMQ sockets.

        Waits for _peer_rdma_inflight[peer_id] == 0 before tearing down NIXL
        descriptors to avoid corrupting in-flight transfers.

        Called by RemoteController during unregister_peer().

        Args:
            peer_id: Must match a previously connected peer.
        """

    @abstractmethod
    def get_disconnected_peers(self) -> list[str]:
        """Return peer_ids that timed out during lookup fan-out.

        Called by RemoteController's reconnect thread. A peer is removed from
        this list once connect_peer() is called successfully for it.

        Returns:
            List of peer_ids currently in a disconnected state.
        """

    ##################################
    # Lookup (called by RemoteL2Adapter)
    ##################################

    @abstractmethod
    def submit_lookup_task(self, keys: list[ObjectKey]) -> IOTaskId:
        """Fan-out ZMQ LookupRequest to all connected peers.

        Non-blocking. The fan-out runs in a background thread pool. On
        completion, per-key remote handles are cached internally in
        _handle_cache[task_id] and get_lookup_event_fd() is signaled.

        Applies lookup_policy to resolve keys found on multiple peers.
        Peers that time out are added to the disconnected set.

        Args:
            keys: Keys to look up across all peers.

        Returns:
            Task ID for use with query_lookup_result().
        """

    @abstractmethod
    def query_lookup_result(self, task_id: IOTaskId) -> "Bitmap | None":
        """Non-blockingly query the result of a lookup task.

        Returns a Bitmap where bit i == 1 if keys[i] was found on any peer.
        One-shot: returns non-None exactly once per task_id.

        Args:
            task_id: From submit_lookup_task().

        Returns:
            Bitmap of found keys, or None if not yet complete.
        """

    ##################################
    # Fetch (called by RemoteL2Adapter)
    ##################################

    @abstractmethod
    def submit_fetch_task(
        self,
        keys: list[ObjectKey],
        local_objs: list[MemoryObj],
        lookup_task_id: IOTaskId | None = None,
    ) -> IOTaskId:
        """Issue RDMA READs for keys using handles cached by the prior lookup.

        Non-blocking. Reads (does not remove) handles from
        _handle_cache[lookup_task_id]. Increments _peer_rdma_inflight per
        peer for each key routed. Issues NIXL_READ per key.

        Args:
            keys:           Keys to fetch. Must be a subset of a prior
                            lookup's found keys.
            local_objs:     L1 write buffers, one per key (same order).
            lookup_task_id: Task ID of the prior submit_lookup_task() whose
                            cached handles to use. Should always be provided;
                            falls back to per-key routing dict when None.

        Returns:
            Task ID for use with query_fetch_result().

        Raises:
            KeyError: If a key has no cached handle from a prior lookup.
            ValueError: If len(keys) != len(local_objs).
        """

    @abstractmethod
    def query_fetch_result(self, task_id: IOTaskId) -> "Bitmap | None":
        """Non-blockingly query the result of a fetch task.

        Returns a Bitmap where bit i == 1 if keys[i] was successfully loaded.
        One-shot: returns non-None exactly once per task_id.

        Args:
            task_id: From submit_fetch_task().

        Returns:
            Bitmap of successfully loaded keys, or None if not yet complete.
        """

    ###################################
    # Unlock (called by RemoteL2Adapter)
    ###################################

    @abstractmethod
    def submit_unlock(
        self,
        keys: list[ObjectKey],
        lookup_task_id: IOTaskId | None = None,
    ) -> None:
        """Send ZMQ UnpinRequest to each owning peer for the given keys.

        The only removal point for _handle_cache: reads each key's peer from
        _handle_cache[lookup_task_id], sends UnpinRequest via the peer's PUSH
        socket, removes the entry. Deletes _handle_cache[lookup_task_id] once
        all keys for that task have been unlocked.

        Called twice per request — once for unneeded keys (before fetch) and
        once for plan keys (after fetch) — together covering all found keys.

        Args:
            keys:           Keys whose remote read locks should be released.
            lookup_task_id: Task ID of the originating submit_lookup_task().
                            Should always be provided by RemoteL2Adapter.
        """

    ###########
    # Eventfds
    ###########

    @abstractmethod
    def get_lookup_event_fd(self) -> int:
        """File descriptor signaled when a lookup task result is available.

        Must be distinct from get_fetch_event_fd() and from the eventfds of
        all other L2 adapters (per L2AdapterInterface contract).

        Returns:
            The eventfd file descriptor.
        """

    @abstractmethod
    def get_fetch_event_fd(self) -> int:
        """File descriptor signaled when a fetch task result is available.

        Must be distinct from get_lookup_event_fd().

        Returns:
            The eventfd file descriptor.
        """

    #########
    # Cleanup
    #########

    @abstractmethod
    def close(self) -> None:
        """Shutdown adapter and release all registered memory and sockets."""
