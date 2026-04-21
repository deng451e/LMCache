# Interface: RemoteController

**Status**: Draft  
**Author**: weishu@tensormesh.ai  
**Date**: 2026-04-19

Server-side handler and peer connection manager. `RemoteController` has two
responsibilities:

1. **Server**: runs the ZMQ REP socket; handles `LookupRequest / UnpinRequest`
   from remote peers by calling `L1Manager.reserve_read / finish_read`.
2. **Peer lifecycle**: `register_peer()` runs the Init/MemReg handshake and
   calls `RemoteIOAdapter.connect_peer()` so the adapter can set up its ZMQ
   lookup sockets and NIXL descriptors. `unregister_peer()` mirrors this.

`RemoteController` does **not** issue lookup requests to peers — that is owned
entirely by `RemoteIOAdapter`. It is an internal dependency of `RemoteL2Adapter`.

---

## 1. Config

```python
@dataclass
class PeerConfig:
    """Connection info for one remote peer."""

    peer_id: str
    """Logical name, e.g. 'decode-0' or 'peer-gpu-1'."""

    host: str
    port:       int
    unpin_port: int
    """PULL socket port for UnpinRequest (fire-and-forget unlock traffic)."""


@dataclass
class RemoteControllerConfig:
    """Configuration for RemoteController."""

    mode: str
    """Deployment role: 'p2p' | 'pd_prefill' | 'pd_decode'."""

    serve_host:       str         = "0.0.0.0"
    serve_port:       int         = 5200
    serve_unpin_port: int         = 5201
    """PULL socket port for incoming UnpinRequests. Must differ from serve_port."""
    peers: list[PeerConfig]       = field(default_factory=list)

    zmq_timeout_ms:   int = 5000
    remote_pin_ttl_s: int = 60
    """Server-side read-lock TTL. Unpin expires after this if UnpinRequest is lost."""
```

> `lookup_policy` (`first_found` / `round_robin`) has moved to
> `RemoteIOAdapterConfig` — see [remote_io_adapter_interface.md](remote_io_adapter_interface.md).

---

## 2. Abstract Interface

```python
class RemoteController(ABC):

    @abstractmethod
    def register_peer(self, config: PeerConfig) -> None:
        """Add a new peer at runtime.

        Performs Init/MemReg ZMQ handshake synchronously, retrieves the peer's
        NIXL metadata and xfer_descs, then calls
        RemoteIOAdapter.connect_peer(peer_id, endpoint, peer_metadata, peer_xfer_descs).
        Safe to call concurrently with ongoing server operations.

        Args:
            config: Connection info for the new peer.

        Raises:
            ConnectionError: If ZMQ handshake with peer fails.
        """

    @abstractmethod
    def unregister_peer(self, peer_id: str) -> None:
        """Remove a peer and release its resources.

        Calls RemoteIOAdapter.disconnect_peer(peer_id), then removes
        the peer from the registry.

        Args:
            peer_id: ID of the peer to remove.

        Raises:
            KeyError: If peer_id is not registered.
        """

    @abstractmethod
    def start(self) -> None:
        """Bind ZMQ REP server socket and begin serving incoming requests.

        Must be called before any peer connects to this instance.
        """

    @abstractmethod
    def stop(self) -> None:
        """Stop serving incoming requests and release all resources."""
```

---

## 3. Internal Usage (RemoteL2Adapter)

`RemoteController` is an internal dependency of `RemoteL2Adapter`. External
components do not call it directly. Its role at runtime is purely server-side;
`RemoteIOAdapter` owns all client-side I/O.

```python
# At startup — RemoteL2Adapter wires everything together
rc.start()

# At runtime — rc is never called on the hot path
# All lookup / fetch / unlock calls go directly to rio

# Peer lifecycle (dynamic peers only)
rc.register_peer(PeerConfig(peer_id="new-peer", host=..., port=...))
rc.unregister_peer("old-peer")
```

---

## 4. Server-Side Behaviour (Concrete Implementation)

Incoming ZMQ messages handled internally — not part of the public interface:

| Message | Handler |
|---|---|
| `InitRequest` | Return `_io.get_local_metadata()` |
| `MemRegRequest` | Return `_io.get_local_xfer_descs()` |
| `LookupRequest` | `l1_manager.reserve_read(keys)` → store in dedup cache → `LookupResponse` |
| `UnpinRequest` | `l1_manager.finish_read(found_keys)` → evict dedup entry (no reply — PULL socket) |
