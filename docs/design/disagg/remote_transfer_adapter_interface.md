# Interface: RemoteTransferAdapter

**Status**: Draft  
**Author**: weishu@tensormesh.ai  
**Date**: 2026-04-19

Pure data-plane transport. No protocol knowledge, no peer state, no ZMQ.
Internal to `RemoteL2Adapter` — called after `RemoteController.lookup()`
populates the adapter's handle cache. External components do not call it
directly.

---

## 1. Handle Types

```python
class LocalMemHandle:
    """Opaque handle for a locally registered memory buffer.

    Returned by register_local_memory(). Reused across all transfers
    against the same buffer — register once at startup.
    """

class RemoteMemHandle:
    """Opaque handle for a remote peer's registered memory.

    Returned by connect_peer(). Stored in RemoteController peer registry
    and passed to read/write at transfer time.
    """

class TransferHandle:
    """Opaque handle for a submitted, possibly in-flight transfer.

    Passed to poll() and release().
    """

class TransferStatus(Enum):
    IN_PROGRESS = "in_progress"
    DONE        = "done"
    ERROR       = "error"
```

---

## 2. Abstract Interface

```python
class RemoteTransferAdapter(ABC):

    @abstractmethod
    def get_local_metadata(self) -> bytes:
        """Return serialised agent descriptor for the Init handshake.

        Called by RemoteController during register_peer() to exchange
        NIXL agent identity with the new peer.

        Returns:
            Opaque bytes sent as InitRequest.local_agent_metadata.
        """

    @abstractmethod
    def get_local_xfer_descs(self) -> bytes:
        """Return serialised local transfer descriptors for the MemReg handshake.

        Called by RemoteController during register_peer() after Init completes.

        Returns:
            Opaque bytes sent as MemRegRequest.local_xfer_descs.
        """

    @abstractmethod
    def register_local_memory(
        self, ptr: int, size: int, device: str
    ) -> LocalMemHandle:
        """Register a local memory buffer for RDMA access.

        Called once at startup against the L1Manager buffer. The returned
        handle is reused for all subsequent read/write calls.

        Args:
            ptr:    Base address of the buffer.
            size:   Buffer size in bytes.
            device: "cpu" or "cuda".

        Returns:
            Opaque handle for the registered buffer.

        Raises:
            RuntimeError: If memory registration fails.
        """

    @abstractmethod
    def connect_peer(
        self,
        peer_metadata:  bytes,
        peer_xfer_descs: bytes,
    ) -> RemoteMemHandle:
        """Register a remote peer's memory for RDMA access.

        Called by RemoteController during register_peer() after exchanging
        Init/MemReg messages with the peer over ZMQ.

        Args:
            peer_metadata:   Serialised agent descriptor from peer InitResponse.
            peer_xfer_descs: Serialised transfer descriptors from peer MemRegResponse.

        Returns:
            Opaque handle for the remote memory, stored in peer registry.

        Raises:
            RuntimeError: If peer connection or descriptor exchange fails.
        """

    @abstractmethod
    def read(
        self,
        local_handle:  LocalMemHandle,
        local_pages:   list[int],
        remote_handle: RemoteMemHandle,
        remote_pages:  list[int],
    ) -> TransferHandle:
        """Submit a non-blocking RDMA READ: remote pages -> local pages.

        Args:
            local_handle:  Registered local buffer (write destination).
            local_pages:   Page indices within local buffer to write into.
            remote_handle: Registered remote buffer (read source).
            remote_pages:  Page indices within remote buffer to read from.

        Returns:
            Handle to poll for completion.

        Raises:
            ValueError: If local_pages and remote_pages lengths differ.
        """

    @abstractmethod
    def write(
        self,
        local_handle:  LocalMemHandle,
        local_pages:   list[int],
        remote_handle: RemoteMemHandle,
        remote_pages:  list[int],
    ) -> TransferHandle:
        """Submit a non-blocking RDMA WRITE: local pages -> remote pages.

        Same semantics as read() with reversed direction.

        Args:
            local_handle:  Registered local buffer (read source).
            local_pages:   Page indices within local buffer to read from.
            remote_handle: Registered remote buffer (write destination).
            remote_pages:  Page indices within remote buffer to write into.

        Returns:
            Handle to poll for completion.

        Raises:
            ValueError: If local_pages and remote_pages lengths differ.
        """

    @abstractmethod
    def poll(self, handle: TransferHandle) -> TransferStatus:
        """Check transfer status. Non-blocking.

        Args:
            handle: Handle returned by read() or write().

        Returns:
            Current status of the transfer.
        """

    @abstractmethod
    def release(self, handle: TransferHandle) -> None:
        """Release resources for a completed or failed transfer.

        Must be called exactly once per TransferHandle after poll()
        returns DONE or ERROR.

        Args:
            handle: Handle to release.
        """

    @abstractmethod
    def close(self) -> None:
        """Shut down the adapter and release all registered memory."""
```

---

## 3. Concrete Implementation

**`NixlTransferBackend`** — wraps `NixlAgent` with UCX backend.

- `register_local_memory` → `NixlAgent.register_buffer`
- `connect_peer` → `NixlAgent.add_remote_agent` + `add_remote_descs`
- `read / write` → `NixlAgent.issue_request` with `NIXL_READ / NIXL_WRITE`
- `poll` → `NixlAgent.check_request`
- `release` → `NixlAgent.release_request`
