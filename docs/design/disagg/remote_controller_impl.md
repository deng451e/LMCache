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

`RemoteController` is wired in via `StorageManagerConfig`:

```python
@dataclass
class StorageManagerConfig:
    # ... existing fields ...
    remote_controller_config: RemoteControllerConfig | None = None
```

`StorageManager.__init__` builds and starts the controller:

```python
if cfg.remote_controller_config:
    transfer = NixlTransferBackend(...)
    local_handle = transfer.register_local_memory(
        l1_buf.ptr, l1_buf.size, device="cpu"
    )
    rc = build_remote_controller(cfg.remote_controller_config, l1_manager, transfer)
    rc.start()
    self._remote_controller = rc
    self._local_handle      = local_handle
```

`PrefetchController` receives `remote_controller: RemoteController | None`,
`remote_transfer_adapter: RemoteTransferAdapter | None`, and
`local_handle: LocalMemHandle | None` as optional constructor dependencies.
`StoreController` is unchanged.

### 6.1 PrefetchController Integration (P2P and PD-decode modes)

**Design principle**: `rc.lookup()` is never called on the `PrefetchController`
background thread. The full remote pipeline runs in a thread-pool task and signals
an eventfd on completion, preserving the non-blocking event-driven invariant of the
existing loop.

#### New phase

```python
class PrefetchPhase(enum.Enum):
    LOOKUP        = enum.auto()   # L2 adapter lookup_and_lock (unchanged)
    REMOTE_LOOKUP = enum.auto()   # rc.lookup() + rta.read() in thread-pool task
    PLAN_AND_LOAD = enum.auto()   # L2 load (unchanged)
```

#### New result type

```python
@dataclass
class RemotePrefetchResult:
    """Produced by the thread-pool task; consumed by the background loop."""
    request_id:  str              # matches the RemoteController.lookup() call
    found_keys:  list[ObjectKey]  # all keys pinned by rc.lookup(); passed to rc.unlock()
    loaded_keys: list[ObjectKey]  # RDMA succeeded; already read-locked in L1
    failed_keys: list[ObjectKey]  # RDMA failed; already cleaned from L1
    prefix_hits: int              # contiguous prefix count among loaded_keys
```

#### New InFlightPrefetchRequest field

```python
remote_future: Future[RemotePrefetchResult] | None = None
```

#### Thread-pool task (remote pipeline)

Submitted by `_submit_remote_lookup()`. Runs entirely outside the background thread.
`request_id` is `str(InFlightPrefetchRequest.request_id)` — stable across retries.

```
rc.lookup(request_id, missed_keys)
  → for each found_key: l1.reserve_write([found_key], ...)
  → rta.read(local_handle, local_pages, info.remote_handle, info.remote_pages)
  → busy-poll rta.poll(handle) until DONE or ERROR; rta.release(handle)
  → l1.finish_write_and_reserve_read(loaded_keys, extra_count)
  → l1.finish_write(failed_keys); l1.delete(failed_keys)
  → rc.unlock(request_id, found_keys)
  → compute prefix_hits over loaded_keys
  → store RemotePrefetchResult in _completed_remote[request_id]; signal _remote_efd
```

#### Eventfd and thread pool

```python
self._remote_efd      = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
self._remote_executor = ThreadPoolExecutor(max_workers=max_in_flight)

# Background-thread-only dict: request_id -> RemotePrefetchResult
self._completed_remote: dict[PrefetchRequestId, RemotePrefetchResult] = {}
```

`_remote_efd` is registered in the `select.poll()` set alongside the existing
L2 adapter eventfds. `_process_remote_completions()` drains `_completed_remote`
and finalises any request in `REMOTE_LOOKUP` phase whose result has arrived.

#### Integration flow

After L2 lookup completes (`_transition_to_load_phase`):

1. Compute L2 load plan for keys found in L2 (unchanged).
2. Keys **not** in the L2 load plan: if `_remote_controller` is configured, call
   `_submit_remote_lookup(request, missed_keys)`.
3. Transition request to `REMOTE_LOOKUP` phase. The L2 `PLAN_AND_LOAD` phase and
   the remote pipeline task run **concurrently** — they operate on disjoint key sets
   so there is no conflict.
4. When `_remote_efd` fires, `_process_remote_completions()` merges the remote
   `prefix_hits` with any L2 prefix count already recorded and calls
   `_complete_request()`.

#### No L2 adapters (PD-decode or remote-only P2P)

When `self._l2_adapters` is empty and `_remote_controller` is configured,
`_start_lookup_phase` skips the L2 fan-out entirely and calls
`_submit_remote_lookup(request, all_keys)` immediately, transitioning straight
to `REMOTE_LOOKUP` phase.

---

## 7. File Structure

```
lmcache/v1/distributed/
│
├── storage_manager.py                      # + remote_controller_config field  [modified]
├── l1_manager.py                           # (unchanged)
├── memory_manager.py                       # (unchanged)
│
├── storage_controllers/
│   ├── prefetch_controller.py              # + REMOTE_LOOKUP phase, eventfd,    [modified]
│   │                                       #   thread-pool task, RemotePrefetchResult
│   └── store_controller.py                 # (unchanged)
│
├── l2_adapters/
│   └── nixl_store_l2_adapter.py           # template for NixlTransferBackend   (unchanged)
│
├── remote_controller/                      # Control-plane coordinator          [NEW]
│   ├── __init__.py                         #                                    [NEW]
│   ├── config.py                           # PeerConfig, RemoteControllerConfig [NEW]
│   ├── protocol.py                         # ZMQ message structs (msgspec)      [NEW]
│   │                                       # Init/MemReg/Lookup/Unpin Req+Resp
│   ├── types.py                            # RemoteKeyInfo, LookupResult        [NEW]
│   ├── controller.py                       # RemoteController ABC               [NEW]
│   │                                       # + ZMQRemoteController
│   └── factory.py                          # build_remote_controller(...)       [NEW]
│
└── remote_transfer/                        # Data-plane transport               [NEW]
    ├── __init__.py                         #                                    [NEW]
    ├── adapter.py                          # RemoteTransferAdapter ABC          [NEW]
    │                                       # + LocalMemHandle, RemoteMemHandle,
    │                                       #   TransferHandle, TransferStatus
    └── backends/
        ├── __init__.py                     #                                    [NEW]
        └── nixl_backend.py                 # NixlTransferBackend (UCX/RDMA)     [NEW]
```

---

## 8. Relation to Existing Code

| Existing component | Reused as |
|---|---|
| `NixlChannel` — ZMQ REQ/REP + handshake | Template for `ZMQControlChannel` and `InitRequest/MemRegRequest` sequence |
| `NixlAgentWrapper` — agent init + xfer_descs | Moved into `NixlTransferBackend.register_local_memory` and `get_local_xfer_descs` |
| `NixlStoreL2Adapter` — eventfd + async poll loop | Template for `_remote_efd` pattern and `_process_remote_completions()` in `PrefetchController` |
| `L1Manager.reserve_read / finish_read` | Called by `ZMQRemoteController` server-side handlers for `LookupRequest` and `UnpinRequest` |
| `L1Manager.reserve_write / finish_write` | Called by thread-pool task inside `_submit_remote_lookup()` |
| `PrefetchController` | Add `REMOTE_LOOKUP` phase, `_remote_efd`, `_remote_executor`; thread-pool task drives full remote pipeline |
| `pd_backend.py` — proxy key-ready notification | Unchanged; sends key list to decode side after KV store completes |
