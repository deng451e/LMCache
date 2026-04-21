# Interface: RemoteIOAdapter

**Status**: Draft  
**Author**: weishu@tensormesh.ai  
**Date**: 2026-04-21

Client-side I/O adapter for remote KV-cache peers. Combines the lookup
protocol (ZMQ `LookupRequest / UnpinRequest` fan-out) with RDMA data transfer
(NIXL/UCX). Internal to `RemoteL2Adapter` — not called directly by
`PrefetchController` or other external components.

`RemoteController` calls `connect_peer() / disconnect_peer()` to notify this
adapter of peer lifecycle events. The adapter manages its own ZMQ REQ sockets
(for lookup traffic) and NIXL remote descriptors per peer.

---

## 1. Handle Types and Task ID

```python
class LocalMemHandle:
    """Opaque handle for a locally registered memory buffer.

    Returned by register_local_memory(). Reused across all fetches
    against the same buffer — register once at startup.
    """

IOTaskId = int
"""Opaque task identifier returned by submit_lookup_task and submit_fetch_task."""
```

---

## 2. Config

```python
@dataclass
class RemoteIOAdapterConfig:
    """Configuration for RemoteIOAdapter."""

    lookup_policy: str = "first_found"
    """Key resolution across peers: 'first_found' | 'round_robin'.

    first_found — first peer (in registration order) that has a key wins.
    round_robin — cycles through peers across successive lookups; spreads
                  read load evenly.
    """

    zmq_timeout_ms: int = 5000
    """Per-peer ZMQ request timeout for lookup traffic."""
```

---

## 3. Abstract Interface

```python
class RemoteIOAdapter(ABC):

    ###########################
    # Local memory registration
    ###########################

    @abstractmethod
    def register_local_memory(
        self, ptr: int, size: int, device: str
    ) -> LocalMemHandle:
        """Register the L1 buffer for RDMA access.

        Called once at startup. The returned handle is passed implicitly to
        all subsequent fetch operations.

        Args:
            ptr:    Base address of the L1 buffer.
            size:   Buffer size in bytes.
            device: "cpu" or "cuda".

        Returns:
            Opaque handle for the registered buffer.

        Raises:
            RuntimeError: If NIXL memory registration fails.
        """

    #############################################
    # Metadata (called by RemoteController only)
    #############################################

    @abstractmethod
    def get_local_metadata(self) -> bytes:
        """Return serialised NIXL agent descriptor for the Init handshake.

        Called by RemoteController during register_peer() to send in
        InitRequest.local_agent_metadata.
        """

    @abstractmethod
    def get_local_xfer_descs(self) -> bytes:
        """Return serialised local transfer descriptors for the MemReg handshake.

        Called by RemoteController during register_peer() after Init completes.
        """

    ###############################################
    # Peer lifecycle (called by RemoteController)
    ###############################################

    @abstractmethod
    def connect_peer(
        self,
        peer_id:         str,
        endpoint:        str,
        unpin_endpoint:  str,
        peer_metadata:   bytes,
        peer_xfer_descs: bytes,
    ) -> None:
        """Register a peer for lookup and RDMA.

        Called by RemoteController after the Init/MemReg handshake completes.
        Creates two ZMQ sockets to the peer: a REQ socket to `endpoint` for
        lookup traffic (Lazy Pirate, one in-flight request), and a PUSH socket
        to `unpin_endpoint` for fire-and-forget UnpinRequests. Also registers
        the peer's NIXL descriptors for RDMA.

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
        """Remove peer state and close its ZMQ lookup socket.

        Called by RemoteController during unregister_peer().

        Args:
            peer_id: Must match a previously connected peer.
        """

    @abstractmethod
    def get_disconnected_peers(self) -> list[str]:
        """Return peer_ids that timed out during lookup fan-out.

        Called by RemoteController's reconnect thread to identify peers
        that need re-registration. A peer is removed from this list once
        connect_peer() is called successfully for it.

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
        completion, per-key remote handles are cached internally and
        get_lookup_event_fd() is signaled.

        Applies lookup_policy to resolve keys found on multiple peers.
        Peers that time out are added to the disconnected set.

        Args:
            keys: Keys to look up across all peers.

        Returns:
            Task ID for use with query_lookup_result().
        """

    @abstractmethod
    def query_lookup_result(self, task_id: IOTaskId) -> Bitmap | None:
        """Non-blockingly query the result of a lookup task.

        Returns a Bitmap where bit i = 1 if keys[i] was found on any peer.
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
        keys:           list[ObjectKey],
        local_objs:     list[MemoryObj],
        lookup_task_id: IOTaskId | None = None,
    ) -> IOTaskId:
        """Issue RDMA READs for keys using handles cached by the prior lookup.

        Non-blocking. Uses remote handles cached by submit_lookup_task() for
        the same keys. local_objs are pre-allocated L1 write buffers
        (write destination); the caller manages their lifecycle.

        Args:
            keys:           Keys to fetch. Must be a subset of a prior lookup's
                            found keys.
            local_objs:     L1 write buffers, one per key (same order).
            lookup_task_id: Task ID of the prior submit_lookup_task() call whose
                            cached handles should be used. When provided, handles
                            are looked up from _handle_cache[lookup_task_id],
                            which is unambiguous even with concurrent overlapping
                            requests under round_robin policy. When None, the
                            implementation falls back to per-key routing
                            (last-write-wins), which is safe only for
                            first_found policy with non-overlapping requests.

        Returns:
            Task ID for use with query_fetch_result().

        Raises:
            KeyError: If a key has no cached handle from a prior lookup.
            ValueError: If len(keys) != len(local_objs).
        """

    @abstractmethod
    def query_fetch_result(self, task_id: IOTaskId) -> Bitmap | None:
        """Non-blockingly query the result of a fetch task.

        Returns a Bitmap where bit i = 1 if keys[i] was successfully loaded.
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
        keys:           list[ObjectKey],
        lookup_task_id: IOTaskId | None = None,
    ) -> None:
        """Send ZMQ UnpinRequest to each owning peer for the given keys.

        Fire-and-forget. The implementation must guarantee eventual delivery
        (internal retry).

        When `lookup_task_id` is provided the adapter routes each key to the
        peer recorded in `_handle_cache[lookup_task_id]` and removes those
        entries from the cache (releasing the cache entry entirely once all
        keys for that task have been unlocked or fetched). This is the only
        correct path when `round_robin` lookup policy is active or when
        multiple concurrent requests share overlapping keys.

        When `lookup_task_id` is None the adapter falls back to a per-key
        routing dict (last-write-wins). Safe only for `first_found` policy
        with non-overlapping concurrent requests.

        Args:
            keys:           Keys whose remote read locks should be released.
            lookup_task_id: Task ID of the originating submit_lookup_task()
                            call. Should always be provided by RemoteL2Adapter.
        """

    ###########
    # Eventfds
    ###########

    @abstractmethod
    def get_lookup_event_fd(self) -> int:
        """Fd signaled when a lookup task result is available.

        Must be distinct from get_fetch_event_fd() and from the eventfds
        of all other L2 adapters (per L2AdapterInterface contract).
        """

    @abstractmethod
    def get_fetch_event_fd(self) -> int:
        """Fd signaled when a fetch task result is available.

        Must be distinct from get_lookup_event_fd().
        """

    #########
    # Cleanup
    #########

    @abstractmethod
    def close(self) -> None:
        """Shutdown adapter and release all registered memory and sockets."""
```

---

## 4. Concrete Implementation: NixlIOAdapter

`NixlIOAdapter` — ZMQ lookup fan-out + NIXL/UCX RDMA.

- `register_local_memory` → `NixlAgent.register_buffer`
- `get_local_metadata` → `NixlAgent.get_agent_metadata`
- `get_local_xfer_descs` → `NixlAgent.get_xfer_descs`
- `connect_peer` → create ZMQ REQ socket to `endpoint` (lookup); create ZMQ PUSH
  socket to `unpin_endpoint` (unlock); `NixlAgent.add_remote_agent` + `add_remote_descs`
- `disconnect_peer` → drain `_peer_rdma_inflight`; close both ZMQ sockets;
  `NixlAgent.remove_remote_agent`
- `submit_lookup_task` → thread-pool fan-out of ZMQ `LookupRequest` per peer;
  results cached in `_handle_cache[task_id]`
- `submit_fetch_task(keys, objs, lookup_task_id)` → reads handles from
  `_handle_cache[lookup_task_id]` (**does not remove them** — still needed by
  the subsequent `submit_unlock` for routing); issues `NIXL_READ` per key
- `submit_unlock(keys, lookup_task_id)` → the **only** removal point: reads each
  key's peer from `_handle_cache[lookup_task_id]`, sends `UnpinRequest` via the
  peer's PUSH socket, removes the entry; deletes `_handle_cache[lookup_task_id]`
  once all keys for that task have been unlocked
- `query_fetch_result` → background thread: `NixlAgent.check_request`; or
  `select.poll()` on NIXL completion fd if exposed

**Handle cache lifecycle**: `_handle_cache[task_id]` is created by
`submit_lookup_task`, read (not removed) by `submit_fetch_task`, and fully
released by `submit_unlock` (called twice — unneeded keys before fetch, plan
keys after fetch — together covering all found keys). There is no other cleanup path — callers must
always call `submit_unlock` for every key returned by a lookup, even when no
fetch is issued.

**ZMQ channel resilience (Lazy Pirate)**: each per-peer REQ socket recreates
itself on timeout before retrying, preventing stuck socket state. After all
retries exhausted, the peer is added to `_disconnected_peers`.
