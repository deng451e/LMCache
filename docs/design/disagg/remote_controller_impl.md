# Implementation Plan: RemoteController & RemoteTransferAdapter

**Status**: Draft  
**Author**: weishu@tensormesh.ai  
**Date**: 2026-04-19

Companion to [remote_controller.md](remote_controller.md),
[remote_controller_interface.md](remote_controller_interface.md), and
[remote_transfer_adapter_interface.md](remote_transfer_adapter_interface.md).

Covers `ZMQRemoteController` internals, `NixlTransferBackend`, peer registration
protocol, lookup idempotency, config schema, StorageManager integration, and
file structure.

---

## 1. ZMQRemoteController Internals

The concrete implementation is `ZMQRemoteController(RemoteController)`.

### 1.1 State

```python
class PeerStatus(Enum):
    CONNECTED    = "connected"
    DISCONNECTED = "disconnected"  # ZMQ retries exhausted; excluded from lookup fan-out

@dataclass
class PeerState:
    config:        PeerConfig
    zmq_channel:   ZMQControlChannel   # one REQ socket per peer
    remote_handle: RemoteMemHandle     # NIXL memory descriptors for this peer
    in_flight:     int        = 0      # active lookup/unlock ops (for drain on unregister)
    status:        PeerStatus = PeerStatus.CONNECTED

class ZMQRemoteController:
    _l1_manager:  L1Manager
    _transfer:    RemoteTransferAdapter
    _peers:       dict[str, PeerState]
    _peers_lock:  RWLock              # read for lookup; write for register/unregister
    _dedup:       dict[str, LookupResponse]
    _dedup_lock:  threading.Lock
```

`_transfer` is referenced only during `register_peer()` to call `connect_peer()`.
At lookup runtime the caller drives `RemoteTransferAdapter` directly using the
`RemoteMemHandle` stored in `LookupResult.key_info`.

### 1.3 Peer Disconnect Detection and Reconnect

When `ZMQControlChannel.send_request()` exhausts all retries (§3.1), it raises
`TimeoutError`. The `lookup()` fan-out catches this and marks the peer as
`DISCONNECTED` before skipping it:

```python
try:
    responses[peer_id] = fut.result(timeout=...)
except TimeoutError:
    with _peers_lock.write():
        _peers[peer_id].status = PeerStatus.DISCONNECTED
    # peer excluded from this and all future lookup fan-outs until reconnected
```

`lookup()` skips any peer whose `status == DISCONNECTED` before building the
fan-out futures — so a stale `RemoteMemHandle` is never placed into
`LookupResult.key_info` and never reaches `RTA.read()`.

A background reconnect thread polls disconnected peers on a fixed interval
(default 30 s):

```python
for peer_id, state in _peers.items():
    if state.status == PeerStatus.DISCONNECTED:
        try:
            register_peer(state.config)   # full Init/MemReg handshake + connect_peer
        except ConnectionError:
            pass   # retry next interval
```

`register_peer()` on success replaces the `PeerState` entry (fresh `zmq_channel`
and `remote_handle`) and sets `status = CONNECTED` under the write lock.

### 1.2 ZMQ Server Thread

A single background thread runs the ZMQ REP socket event loop. Incoming `bytes`
are decoded with `msgspec` and dispatched by message type tag. Handlers call
`_l1_manager` directly — no cross-thread queuing.

```python
while running:
    raw      = rep_socket.recv()
    msg      = msgspec.msgpack.decode(raw, type=ZMQMessage)
    response = _dispatch(msg)
    rep_socket.send(msgspec.msgpack.encode(response))
```

| Message | Server handler |
|---|---|
| `InitRequest` | Return `_transfer.get_local_metadata()` |
| `MemRegRequest` | Return `_transfer.get_local_xfer_descs()` |
| `LookupRequest` | Dedup check → `_l1_manager.reserve_read(keys)` → cache → `LookupResponse` |
| `UnpinRequest` | `_l1_manager.finish_read(found_keys)` → evict dedup entry → `UnpinResponse` |

---

## 2. Peer Registration Protocol

`register_peer(config)` is called at startup for pre-configured peers and at
runtime for dynamic peers. The ZMQ handshake runs outside the `_peers_lock` write
lock; the lock is acquired only for the final `_peers` insert.

```
register_peer() caller                        Remote ZMQ server (already running)
──────────────────────────────────────────    ────────────────────────────────────
channel = ZMQControlChannel(host, port)
send InitRequest(local_agent_metadata)   ──→  handler: return local_agent_metadata
peer_metadata = InitResponse.metadata    ←──
send MemRegRequest(local_xfer_descs)     ──→  handler: return local_xfer_descs
peer_xfer_descs = MemRegResponse.descs   ←──
remote_handle = _transfer.connect_peer(peer_metadata, peer_xfer_descs)
with _peers_lock.write():
    _peers[peer_id] = PeerState(config, channel, remote_handle)
```

`unregister_peer(peer_id)` acquires the write lock, marks the peer as draining,
waits for `in_flight == 0`, then removes the entry and closes the ZMQ channel.

---

## 3. Lookup Fan-out

`lookup(request_id, keys)` fans out `LookupRequest` to all registered peers
concurrently using a thread pool, then aggregates results by `lookup_policy`.

```python
def lookup(self, request_id: str, keys: list[ObjectKey]) -> LookupResult:
    with self._peers_lock.read():
        peers = list(self._peers.values())

    futures = {
        p.config.peer_id: executor.submit(_send_lookup, p, request_id, keys)
        for p in peers
    }

    responses: dict[str, LookupResponse] = {}
    for peer_id, fut in futures.items():
        try:
            responses[peer_id] = fut.result(timeout=zmq_timeout_ms / 1000)
        except TimeoutError:
            pass   # peer treated as not-found for this request

    return _apply_lookup_policy(responses, keys)
```

`_apply_lookup_policy`:

- `first_found` — first peer (in registration order) that has a key wins.
- `round_robin` — cycles through peers across successive lookups using a
  per-key modulo on a request counter. Spreads read load evenly.

### 3.1 ZMQ Channel Resilience (Lazy Pirate)

ZMQ REQ sockets are strictly alternating: `send` must be followed by `recv`
before the next `send`. If `RCVTIMEO` fires, the socket enters a broken state —
the next `send` blocks indefinitely.

`ZMQControlChannel.send_request()` recovers by closing and recreating the socket
on each timeout:

```python
def send_request(self, msg: bytes, retries: int = 3) -> bytes:
    for attempt in range(retries):
        self._socket.send(msg)
        if self._socket.poll(self._timeout_ms):
            return self._socket.recv()
        # Timeout: REQ state machine is broken — recreate the socket
        self._socket.close()
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._socket.connect(self._endpoint)
    raise TimeoutError(f"all {retries} attempts to {self._endpoint} timed out")
```

Key properties:
- No state leaks: old socket is closed before the new one connects.
- Transparent to `lookup()`: `_send_lookup` only sees `TimeoutError` on exhaustion.
- Safe to retry: the server dedup cache ensures a late-arriving `LookupRequest`
  retried with the same `request_id` does not double-pin.

---

## 4. Lookup Idempotency

### Problem

`LookupRequest` calls `_l1_manager.reserve_read(keys)` on the server, incrementing
a read-lock counter per key. A ZMQ timeout followed by retry would double-pin the
key; a single `UnpinRequest` would then leave a ghost read-lock until
`remote_pin_ttl_s` expires.

### Server-side dedup cache

```python
_dedup: dict[str, LookupResponse]
# keyed by request_id; entries expire after remote_pin_ttl_s + 10 s
```

**`LookupRequest` handler:**

```python
if request_id in _dedup:
    return _dedup[request_id]          # cached — no reserve_read
result   = _l1_manager.reserve_read(keys)
response = LookupResponse(...)
_dedup[request_id] = response          # store before replying
return response
```

**`UnpinRequest` handler:**

```python
_l1_manager.finish_read(found_keys)
_dedup.pop(request_id, None)           # eager eviction
```

TTL-based expiry ensures cleanup even if `UnpinRequest` is lost. After
`remote_pin_ttl_s + 10 s` the read lock itself has also expired, so a belated
retry returns an empty `found_bitmap` rather than a stale cache hit.

`request_id` is generated once per caller operation and never regenerated on
retry. The same `request_id` is sent to all peers in the fan-out, so per-peer
retries are also safe.

`UnpinRequest` is idempotent: `finish_read` on an already-released key logs a
warning but does not corrupt state. The `_dedup.pop` is a no-op on a missing key.

---

## 5. Configuration

```python
@dataclass
class PeerConfig:
    peer_id: str   # logical name e.g. "decode-0", "peer-gpu-1"
    host:    str
    port:    int


@dataclass
class RemoteControllerConfig:
    mode: str   # "p2p" | "pd_prefill" | "pd_decode"

    serve_host: str              = "0.0.0.0"
    serve_port: int              = 5200

    peers: list[PeerConfig]      = field(default_factory=list)
    # Pre-configured peers. register_peer() can add more at runtime.

    lookup_policy:    str = "first_found"  # "first_found" | "round_robin"
    zmq_timeout_ms:   int = 5000
    remote_pin_ttl_s: int = 60
```

Components active per mode:

```
mode          ZMQ server   peer registry    RemoteTransferAdapter
──────────────────────────────────────────────────────────────────
p2p           yes          yes (≥1 peer)    yes (RDMA READ)
pd_prefill    yes          no               no
pd_decode     yes          yes (prefiller)  yes (RDMA READ)
```

`pd_prefill` runs only the ZMQ REP server — it does not initiate lookups.  
`pd_decode` connects to the prefill host as a peer and issues lookup/pull
using the same protocol as P2P.

---

## 6. StorageManager Integration

`RemoteL2Adapter` is wired in via `StorageManagerConfig` and added to the
**prefetch-only** adapter list. It must not be in the `StoreController`'s list —
remote peers are read sources, not write destinations.

`StorageManager` therefore maintains two separate adapter lists:

```python
@dataclass
class StorageManagerConfig:
    # ... existing fields ...
    remote_controller_config: RemoteControllerConfig | None = None
```

`StorageManager.__init__` builds and starts the adapter:

```python
if cfg.remote_controller_config:
    transfer = NixlTransferBackend(...)
    local_handle = transfer.register_local_memory(
        l1_buf.ptr, l1_buf.size, device="cpu"
    )
    rc = build_remote_controller(cfg.remote_controller_config, l1_manager, transfer)
    rc.start()
    remote_l2 = RemoteL2Adapter(
        rc, transfer, local_handle,
        l1_manager.get_l1_memory_desc().align_bytes,
    )
    remote_l2.start()
    self._prefetch_adapters.append(remote_l2)
    # self._store_adapters does NOT include remote_l2
```

`PrefetchController` is constructed with `prefetch_adapters`; `StoreController`
with `store_adapters`. `PrefetchController` itself is unchanged — it sees
`RemoteL2Adapter` as an ordinary `L2AdapterInterface` entry.

### 6.1 RemoteL2Adapter Internals

`RemoteL2Adapter` implements `L2AdapterInterface` and orchestrates
`RemoteController` + `RemoteTransferAdapter` behind two eventfds, keeping
`PrefetchController` completely unaware of the remote path.

#### State

```python
class RemoteL2Adapter(L2AdapterInterface):
    _rc:           RemoteController
    _rta:          RemoteTransferAdapter
    _local_handle: LocalMemHandle
    _l1_align:     int

    _next_task_id: int
    _lookup_lock:  threading.Lock
    # Populated by lookup thread; consumed by query_lookup_and_lock_result (one-shot)
    _completed_lookups: dict[L2TaskId, Bitmap]
    # Per-task remote handle cache: consumed by submit_load_task
    _handle_cache: dict[L2TaskId, dict[ObjectKey, RemoteKeyInfo]]
    # Key → request_id: used by submit_unlock to route to rc.unlock()
    _key_to_reqid: dict[ObjectKey, str]

    _load_lock:    threading.Lock
    # task_id → [(key, TransferHandle)] for in-flight RDMA reads
    _pending_loads:   dict[L2TaskId, list[tuple[ObjectKey, TransferHandle]]]
    # Populated by RDMA poll thread; consumed by query_load_result (one-shot)
    _completed_loads: dict[L2TaskId, Bitmap]

    # Eventfds (distinct per L2AdapterInterface contract)
    _lookup_event_fd: int   # signaled when a lookup task result is ready
    _load_event_fd:   int   # signaled when all RDMA reads for a load task finish

    _lookup_executor:  ThreadPoolExecutor   # rc.lookup() calls (blocking network I/O)
    _rdma_poll_thread: threading.Thread     # polls rta.poll() for in-flight transfers
```

#### Lookup path

`submit_lookup_and_lock_task(keys)` submits a thread-pool task that calls
`rc.lookup(request_id, keys)` (blocking ZMQ round-trip). On completion:

1. Cache `result.key_info` in `_handle_cache[task_id]`.
2. Record `{key: request_id}` for each found key in `_key_to_reqid`.
3. Build `Bitmap` from `result.found_keys`.
4. Under `_lookup_lock`: store bitmap in `_completed_lookups[task_id]`.
5. Signal `_lookup_event_fd`.

`query_lookup_and_lock_result(task_id)` pops and returns the cached bitmap
(one-shot), or `None` if not yet ready.

#### Load path

`submit_load_task(keys, local_objs)` retrieves `_handle_cache[task_id]` for
per-key `RemoteKeyInfo`, then calls `rta.read()` for each key. The resulting
`TransferHandle`s are stored in `_pending_loads[task_id]`.

The RDMA poll thread wakes on a short timer (or on NIXL's completion fd if
exposed) and calls `rta.poll(handle)` for all pending handles:

- `DONE` → mark key successful; call `rta.release(handle)`.
- `ERROR` → mark key failed; call `rta.release(handle)`.

When all handles for a `task_id` are resolved:

1. Build result `Bitmap`.
2. Under `_load_lock`: store in `_completed_loads[task_id]`.
3. Signal `_load_event_fd`.

`query_load_result(task_id)` pops and returns the cached bitmap (one-shot).

#### Unlock path

`submit_unlock(keys)` groups keys by their `request_id` from `_key_to_reqid`,
calls `rc.unlock(request_id, per_request_keys)` for each group (fire-and-forget;
`RemoteController` retries internally), then cleans up `_key_to_reqid` entries.

#### NIXL completion fd (optional optimisation)

If `NixlTransferBackend` exposes a hardware completion fd, the RDMA poll thread
can `select.poll()` on it instead of spinning — eliminating busy-wait CPU use.
The lookup thread is unaffected (it blocks on ZMQ I/O).

---

## 7. File Structure

```
lmcache/v1/distributed/
│
├── storage_manager.py                      # + remote_controller_config field   [modified]
│                                           # + separate store/prefetch adapter lists
├── l1_manager.py                           # (unchanged)
├── memory_manager.py                       # (unchanged)
│
├── storage_controllers/
│   ├── prefetch_controller.py              # (unchanged)
│   └── store_controller.py                 # (unchanged)
│
├── l2_adapters/
│   ├── remote_l2_adapter.py               # RemoteL2Adapter                    [NEW]
│   └── nixl_store_l2_adapter.py           # (unchanged)
│
├── remote_controller/                      # Control-plane coordinator          [NEW]
│   ├── __init__.py
│   ├── config.py                           # PeerConfig, RemoteControllerConfig
│   ├── protocol.py                         # ZMQ message structs (msgspec)
│   │                                       # Init/MemReg/Lookup/Unpin Req+Resp
│   ├── types.py                            # RemoteKeyInfo, LookupResult
│   ├── controller.py                       # RemoteController ABC + ZMQRemoteController
│   └── factory.py                          # build_remote_controller(...)
│
└── remote_transfer/                        # Data-plane transport               [NEW]
    ├── __init__.py
    ├── adapter.py                          # RemoteTransferAdapter ABC
    │                                       # + LocalMemHandle, RemoteMemHandle,
    │                                       #   TransferHandle, TransferStatus
    └── backends/
        ├── __init__.py
        └── nixl_backend.py                 # NixlTransferBackend (UCX/RDMA)
```

---

## 8. Relation to Existing Code

| Existing component | Reused as |
|---|---|
| `NixlChannel` — ZMQ REQ/REP + handshake | Template for `ZMQControlChannel` and `InitRequest/MemRegRequest` sequence |
| `NixlAgentWrapper` — agent init + xfer_descs | Moved into `NixlTransferBackend.register_local_memory` and `get_local_xfer_descs` |
| `NixlStoreL2Adapter` — eventfd + async poll loop | Template for `RemoteL2Adapter` lookup/load eventfds and RDMA poll thread |
| `L1Manager.reserve_read / finish_read` | Called by `ZMQRemoteController` server-side handlers for `LookupRequest` and `UnpinRequest` |
| `L1Manager.reserve_write / finish_write` | Called by `PrefetchController` (unchanged) after lookup returns found keys |
| `PrefetchController` | Unchanged — treats `RemoteL2Adapter` as a standard L2 adapter |
| `pd_backend.py` — proxy key-ready notification | Unchanged; sends key list to decode side after KV store completes |
