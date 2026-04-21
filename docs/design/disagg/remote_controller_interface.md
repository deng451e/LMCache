# Interface: RemoteController

**Status**: Draft  
**Author**: weishu@tensormesh.ai  
**Date**: 2026-04-19

Control-plane coordinator: peer registry, ZMQ channels, metadata exchange,
lookup policy, dedup cache, and L1Manager pin management (server side).

`RemoteController` is an **internal dependency of `RemoteL2Adapter`** — it is
not called directly by `PrefetchController`, the Engine, or `StorageManager`.
At runtime, `lookup()` returns remote handles that `RemoteL2Adapter` caches
internally and passes to `RemoteTransferAdapter.read()`. `RemoteController`
does not call `RemoteTransferAdapter` at runtime — only during `register_peer()`
to exchange NIXL descriptors via `connect_peer()`.

---

## 1. Config

```python
@dataclass
class PeerConfig:
    """Connection info for one remote peer."""

    peer_id: str
    """Logical name, e.g. 'decode-0' or 'peer-gpu-1'."""

    host: str
    port: int


@dataclass
class RemoteControllerConfig:
    """Configuration for RemoteController."""

    mode: str
    """Deployment role: 'p2p' | 'pd_prefill' | 'pd_decode'."""

    serve_host: str        = "0.0.0.0"
    serve_port: int        = 5200
    peers: list[PeerConfig] = field(default_factory=list)

    lookup_policy:    str = "first_found"
    """Key resolution across peers: 'first_found' | 'round_robin'."""

    zmq_timeout_ms:   int = 5000
    remote_pin_ttl_s: int = 60
    """Server-side read-lock TTL. Unpin expires after this if UnpinRequest is lost."""
```

---

## 2. Result Types

```python
@dataclass
class RemoteKeyInfo:
    """Per-key transfer info returned by lookup(). Passed to RemoteTransferAdapter."""

    peer_id:       str
    remote_handle: RemoteMemHandle
    """Handle for the owning peer's L1 buffer. Sourced from peer registry."""

    remote_pages:  list[int]
    """Page indices within the remote L1 buffer for this key."""


@dataclass
class LookupResult:
    """Aggregated result of a remote lookup across all peers."""

    found_keys: list[ObjectKey]
    """Subset of queried keys that exist on at least one peer."""

    key_info:   dict[ObjectKey, RemoteKeyInfo]
    """Transfer info for each found key. Used to drive RemoteTransferAdapter.read()."""
```

---

## 3. Abstract Interface

```python
class RemoteController(ABC):

    @abstractmethod
    def lookup(
        self,
        request_id: str,
        keys:       list[ObjectKey],
    ) -> LookupResult:
        """Look up keys across all registered peers and pin them for reading.

        Sends LookupRequest to all peers in parallel, applies lookup_policy
        to resolve keys found on multiple peers, and returns remote transfer
        info for each found key. Remote read locks are held until unlock()
        is called with the same request_id.

        Idempotent: retrying with the same request_id returns the cached
        response without re-pinning.

        Args:
            request_id: Stable ID for this lookup; must be unique per
                        concurrent lookup and reused on retry.
            keys:       Keys to look up.

        Returns:
            LookupResult with found keys and per-key RemoteKeyInfo.

        Raises:
            TimeoutError: If all peer LookupRequests time out.
        """

    @abstractmethod
    def unlock(
        self,
        request_id: str,
        found_keys: list[ObjectKey],
    ) -> None:
        """Release remote read locks acquired by lookup().

        Sends UnpinRequest to each peer that owns at least one found key.
        Must be called for ALL keys in LookupResult.found_keys, including
        keys whose local reserve_write or RDMA transfer failed.

        Args:
            request_id: Must match the request_id used in lookup().
            found_keys: All keys from LookupResult.found_keys.
        """

    @abstractmethod
    def register_peer(self, config: PeerConfig) -> None:
        """Add a new peer at runtime.

        Performs Init/MemReg ZMQ handshake synchronously, then calls
        RemoteTransferAdapter.connect_peer() to register the peer's memory.
        Safe to call concurrently with ongoing lookup/unlock operations.

        Args:
            config: Connection info for the new peer.

        Raises:
            ConnectionError: If ZMQ handshake with peer fails.
        """

    @abstractmethod
    def unregister_peer(self, peer_id: str) -> None:
        """Remove a peer and release its resources.

        Waits for all in-flight lookup/unlock operations for this peer
        to complete before removing it from the registry.

        Args:
            peer_id: ID of the peer to remove.

        Raises:
            KeyError: If peer_id is not registered.
        """

    @abstractmethod
    def start(self) -> None:
        """Bind ZMQ server socket and begin serving incoming requests.

        Must be called before any peer connects to this instance.
        """

    @abstractmethod
    def stop(self) -> None:
        """Stop serving incoming requests and release all resources."""
```

---

## 4. Internal Usage (RemoteL2Adapter)

`RemoteController` is an internal dependency of `RemoteL2Adapter`; external
components do not call it directly.

```python
# Inside RemoteL2Adapter — thread-pool task backing submit_lookup_and_lock_task()
result = remote_controller.lookup(request_id, keys)
# result.key_info cached internally; only a Bitmap is exposed to PrefetchController

# Inside RemoteL2Adapter — submit_load_task() uses cached key_info to drive RTA
for key, obj in zip(keys, local_objs):
    info = _handle_cache[task_id][key]
    handle = remote_transfer_adapter.read(
        local_handle, local_pages[key],
        info.remote_handle, info.remote_pages,
    )
    # ... poll loop in background thread ...

# Inside RemoteL2Adapter — submit_unlock() routes by originating request_id
remote_controller.unlock(request_id, per_request_keys)
```

`RemoteController.register_peer()` is called once per peer during
`RemoteL2Adapter.__init__` (or at runtime for dynamic peers) to exchange NIXL
descriptors and establish the ZMQ channel.

---

## 5. Server-Side Behaviour (Concrete Implementation)

Incoming ZMQ messages handled internally — not part of the public interface:

| Message | Handler |
|---|---|
| `InitRequest` | Return local agent metadata |
| `MemRegRequest` | Return local xfer_descs |
| `LookupRequest` | `l1_manager.reserve_read(keys)` → store in dedup cache → `LookupResponse` |
| `UnpinRequest` | `l1_manager.finish_read(found_keys)` → evict dedup entry → `UnpinResponse` |
