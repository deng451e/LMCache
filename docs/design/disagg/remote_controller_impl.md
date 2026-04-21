# Implementation Plan: RemoteController & RemoteIOAdapter

**Status**: Draft  
**Author**: weishu@tensormesh.ai  
**Date**: 2026-04-19

Companion to [remote_controller.md](remote_controller.md),
[remote_controller_interface.md](remote_controller_interface.md), and
[remote_io_adapter_interface.md](remote_io_adapter_interface.md).

Covers `ZMQRemoteController` internals, `NixlIOAdapter`, peer registration
protocol, lookup idempotency, config schema, StorageManager integration, and
file structure.

---

## 1. ZMQRemoteController Internals

The concrete implementation is `ZMQRemoteController(RemoteController)`.

### 1.1 State

```python
class PeerStatus(Enum):
    CONNECTED    = "connected"
    DISCONNECTED = "disconnected"  # reported by RIO; excluded from reconnect until resolved

@dataclass
class PeerState:
    config:      PeerConfig
    zmq_channel: ZMQControlChannel   # one REQ socket per peer (for Init/MemReg + reconnect)
    in_flight:   int        = 0      # active register/unregister ops (for drain on unregister)
    status:      PeerStatus = PeerStatus.CONNECTED
    # remote_handle removed — now owned by RemoteIOAdapter

class ZMQRemoteController:
    _l1_manager: L1Manager
    _io:         RemoteIOAdapter
    _peers:      dict[str, PeerState]
    _peers_lock: RWLock              # write for register/unregister; read for reconnect scan
    _dedup:      dict[str, LookupResponse]
    _dedup_lock: threading.Lock
```

`_io` is called during `register_peer()` to exchange peer descriptors. At
server runtime `_io` is not involved — handlers call `_l1_manager` directly.

### 1.2 ZMQ Server Thread

A single background thread polls two sockets — a REP socket (`serve_port`) for
lookup/init/memreg traffic and a PULL socket (`serve_unpin_port`) for fire-and-forget
unpin traffic. Using `zmq.Poller` keeps both on one thread without blocking:

```python
poller = zmq.Poller()
poller.register(rep_socket,  zmq.POLLIN)
poller.register(pull_socket, zmq.POLLIN)

while running:
    for sock, _ in poller.poll():
        raw = sock.recv()
        msg = msgspec.msgpack.decode(raw, type=ZMQMessage)
        response = _dispatch(msg)        # None for UnpinRequest
        if response is not None:
            sock.send(msgspec.msgpack.encode(response))
```

| Socket | Message | Server handler |
|---|---|---|
| REP | `InitRequest` | Return `_io.get_local_metadata()` |
| REP | `MemRegRequest` | Return `_io.get_local_xfer_descs()` |
| REP | `LookupRequest` | Dedup check → `_l1_manager.reserve_read(keys)` → cache → `LookupResponse` |
| PULL | `UnpinRequest` | `_l1_manager.finish_read(found_keys)` → evict dedup entry (no reply) |

### 1.3 Peer Disconnect Detection and Reconnect

Disconnect detection is owned by `RemoteIOAdapter`, which detects ZMQ timeout
during its lookup fan-out. RIO marks the peer as disconnected internally and
exposes it via `get_disconnected_peers() → list[str]`.

`ZMQRemoteController`'s reconnect thread polls on a fixed interval (default 30 s):

```python
for peer_id in _io.get_disconnected_peers():
    state = _peers.get(peer_id)
    if state is None:
        continue
    try:
        register_peer(state.config)   # full Init/MemReg handshake + _io.connect_peer()
    except ConnectionError:
        pass   # retry next interval
```

`register_peer()` on success calls `_io.connect_peer()` which resets the peer
to `CONNECTED` in RIO's internal state.

---

## 2. Peer Registration Protocol

`register_peer(config)` is called at startup for pre-configured peers and at
runtime for dynamic peers. The ZMQ handshake runs outside the `_peers_lock` write
lock; the lock is acquired only for the final `_peers` insert.

```
register_peer() caller                        Remote ZMQ server (already running)
──────────────────────────────────────────    ────────────────────────────────────
channel = ZMQControlChannel(host, port)
send InitRequest(_io.get_local_metadata()) ──→  handler: return local metadata
peer_metadata = InitResponse.metadata      ←──
send MemRegRequest(_io.get_local_xfer_descs()) ──→  handler: return local xfer_descs
peer_xfer_descs = MemRegResponse.descs     ←──
endpoint       = f"tcp://{host}:{port}"
unpin_endpoint = f"tcp://{host}:{unpin_port}"
_io.connect_peer(peer_id, endpoint, unpin_endpoint, peer_metadata, peer_xfer_descs)
with _peers_lock.write():
    _peers[peer_id] = PeerState(config, channel)
    # no remote_handle — now owned by _io
```

`unregister_peer(peer_id)` acquires the write lock, waits for `in_flight == 0`,
calls `_io.disconnect_peer(peer_id)`, then removes the entry and closes the ZMQ
channel.

---

## 3. Lookup Fan-out (moved to NixlIOAdapter)

The lookup fan-out, ZMQ channel resilience (Lazy Pirate), and lookup policy
(`first_found` / `round_robin`) are implemented in `NixlIOAdapter` — the
concrete `RemoteIOAdapter`. See
[remote_io_adapter_interface.md](remote_io_adapter_interface.md) for the full
interface and §6.2 below for `NixlIOAdapter` internals.

`ZMQRemoteController` has no `lookup()` or `unlock()` methods.

---

## 4. Lookup Idempotency

### Problem

`LookupRequest` calls `_l1_manager.reserve_read(keys)` on the server,
incrementing a read-lock counter per key. A ZMQ timeout followed by retry would
double-pin the key; a single `UnpinRequest` would then leave a ghost read-lock
until `remote_pin_ttl_s` expires.

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

TTL-based expiry ensures cleanup even if `UnpinRequest` is lost.

`request_id` is generated once per `NixlIOAdapter` task and never regenerated on
retry, so per-peer retries are safe.

`UnpinRequest` is idempotent: `finish_read` on an already-released key logs a
warning but does not corrupt state.

---

## 5. Configuration

```python
@dataclass
class PeerConfig:
    peer_id:    str   # logical name e.g. "decode-0", "peer-gpu-1"
    host:       str
    port:       int   # REP socket port (lookup / init / memreg)
    unpin_port: int   # PULL socket port (UnpinRequest)


@dataclass
class RemoteControllerConfig:
    mode: str   # "p2p" | "pd_prefill" | "pd_decode"

    serve_host:       str              = "0.0.0.0"
    serve_port:       int              = 5200
    serve_unpin_port: int              = 5201

    peers: list[PeerConfig]            = field(default_factory=list)
    # Pre-configured peers. register_peer() can add more at runtime.

    zmq_timeout_ms:   int = 5000
    remote_pin_ttl_s: int = 60
    # lookup_policy moved to RemoteIOAdapterConfig
```

Components active per mode:

```
mode          ZMQ server   peer registry    RemoteIOAdapter
──────────────────────────────────────────────────────────────────────────────
p2p           yes          yes (≥1 peer)    yes (full: register + lookup + RDMA)
pd_prefill    yes          no               yes (registration only: register_local_memory
                                                 + get_local_metadata + get_local_xfer_descs)
pd_decode     yes          yes (prefiller)  yes (full: register + lookup + RDMA)
```

`pd_prefill` does not initiate lookups but still requires `NixlIOAdapter` to
register its L1 buffer and expose NIXL metadata: the ZMQ server responds to
`InitRequest` with `_io.get_local_metadata()` and to `MemRegRequest` with
`_io.get_local_xfer_descs()`, which decode needs to issue RDMA READs. Without
these, the Init/MemReg handshake cannot complete and no RDMA transfers are
possible. `RemoteL2Adapter` is NOT created for `pd_prefill`.

---

## 6. StorageManager Integration

`RemoteL2Adapter` is wired in via `StorageManagerConfig` and added to the
**prefetch-only** adapter list. It must not be in the `StoreController`'s list.

`StorageManager` maintains two separate adapter lists:

```python
@dataclass
class StorageManagerConfig:
    # ... existing fields ...
    remote_controller_config:   RemoteControllerConfig   | None = None
    remote_io_adapter_config:   RemoteIOAdapterConfig    | None = None
```

`StorageManager.__init__` builds and starts the components:

```python
if cfg.remote_controller_config:
    rio = NixlIOAdapter(cfg.remote_io_adapter_config)
    rio.register_local_memory(l1_buf.ptr, l1_buf.size, device="cpu")
    rc = build_remote_controller(cfg.remote_controller_config, l1_manager, rio)
    rc.start()
    if cfg.remote_controller_config.mode != "pd_prefill":
        remote_l2 = RemoteL2Adapter(rc, rio)
        self._prefetch_adapters.append(remote_l2)
        # self._store_adapters does NOT include remote_l2
```

`NixlIOAdapter` starts its background threads (`_lookup_executor`,
`_fetch_poll_thread`) in `__init__` and tears them down in `close()`.

`PrefetchController` is constructed with `prefetch_adapters`; `StoreController`
with `store_adapters`. `PrefetchController` requires one small interface change
to pass the lookup task_id through to the load phase:

```python
# L2AdapterInterface — two new optional parameters (backward-compatible):
def submit_load_task(
    self,
    keys:           list[ObjectKey],
    objects:        list[MemoryObj],
    lookup_task_id: L2TaskId | None = None,
) -> L2TaskId: ...

def submit_unlock(
    self,
    keys:           list[ObjectKey],
    lookup_task_id: L2TaskId | None = None,
) -> None: ...

# PrefetchController — InFlightPrefetchRequest gains one new field:
completed_lookup_task_ids: dict[int, L2TaskId]  # adapter_idx -> lookup task_id
# Populated in _process_lookup_completions; passed to submit_load_task,
# _unlock_unneeded_keys, and _unlock_all_plan_keys.
```

All other L2 adapters accept and ignore both new parameters. Only `RemoteL2Adapter`
forwards them to `RemoteIOAdapter` for correct handle cache routing and cleanup.

**Why `submit_unlock` also needs `lookup_task_id`**: without it, two bugs arise:
1. *Handle cache leak* — if L1 reservation fails for all found keys, no fetch is
   issued and `_handle_cache[lookup_task_id]` is never freed.
2. *Routing correctness* — under `round_robin` with concurrent overlapping keys,
   a per-key routing dict is overwritten by the later lookup, causing `UnpinRequest`
   to be sent to the wrong peer while the correct peer retains a dangling read lock
   until `remote_pin_ttl_s` expires.

### 6.1 RemoteL2Adapter Internals

`RemoteL2Adapter` is a **thin wrapper** — its only role is to translate
`L2AdapterInterface` calls into `RemoteIOAdapter` calls. All task tracking,
handle caching, ZMQ fan-out, and RDMA polling live in `RemoteIOAdapter`.

#### State

```python
class RemoteL2Adapter(L2AdapterInterface):
    _rc:  RemoteController   # for peer lifecycle: register_peer / unregister_peer
    _rio: RemoteIOAdapter    # for all I/O: lookup, fetch, unlock
    # No internal task state — fully delegated to _rio
```

#### Method delegation

| `L2AdapterInterface` method | Delegation |
|---|---|
| `get_lookup_and_lock_event_fd()` | `_rio.get_lookup_event_fd()` |
| `get_load_event_fd()` | `_rio.get_fetch_event_fd()` |
| `get_store_event_fd()` | dummy fd that never fires |
| `submit_lookup_and_lock_task(keys)` | `_rio.submit_lookup_task(keys)` |
| `query_lookup_and_lock_result(task_id)` | `_rio.query_lookup_result(task_id)` |
| `submit_load_task(keys, objs, lookup_task_id)` | `_rio.submit_fetch_task(keys, objs, lookup_task_id)` |
| `query_load_result(task_id)` | `_rio.query_fetch_result(task_id)` |
| `submit_unlock(keys, lookup_task_id)` | `_rio.submit_unlock(keys, lookup_task_id)` |
| `submit_store_task(...)` | no-op (read-only adapter) |
| `pop_completed_store_tasks()` | returns `{}` |
| `delete()` / `get_usage()` | no-op / default |

### 6.2 NixlIOAdapter Internals

`NixlIOAdapter` is the concrete `RemoteIOAdapter`. It manages one ZMQ REQ
socket per peer (for lookup traffic) and one NIXL agent (for RDMA).

**State:**

```python
class NixlIOAdapter(RemoteIOAdapter):
    _agent:        NixlAgent
    _local_handle: LocalMemHandle
    _l1_align:     int

    # Per-peer state (set by connect_peer, cleared by disconnect_peer)
    _peer_lookup_channels: dict[str, ZMQControlChannel]  # ZMQ REQ — lookup traffic only
    _peer_unlock_channels: dict[str, zmq.Socket]          # ZMQ PUSH — UnpinRequest only
    _peer_remote_handle:   dict[str, RemoteMemHandle]     # NIXL descriptors per peer
    _disconnected_peers:   set[str]                       # reported via get_disconnected_peers()
    _peer_rdma_inflight:   dict[str, int]                 # active NIXL transfers per peer
    _peer_rdma_cv:         dict[str, threading.Condition] # signaled when count reaches 0
    _peers_lock:           RWLock

    # Task state
    _next_task_id:       int
    _lookup_lock:        threading.Lock
    _completed_lookups:  dict[IOTaskId, Bitmap]
    _handle_cache:       dict[IOTaskId, dict[ObjectKey, tuple[str, list[int]]]]
    # (peer_id, remote_pages) per key, for submit_fetch_task

    _load_lock:          threading.Lock
    _pending_fetches:    dict[IOTaskId, list[tuple[ObjectKey, TransferHandle]]]
    _completed_fetches:  dict[IOTaskId, Bitmap]

    _lookup_executor:    ThreadPoolExecutor   # ZMQ lookup fan-out
    _fetch_poll_thread:  threading.Thread    # NIXL completion polling
    _lookup_event_fd:    int
    _fetch_event_fd:     int

    _lookup_policy:      str   # "first_found" | "round_robin"
```

**Lookup fan-out** (`submit_lookup_task`): submits a thread-pool task that fans out
`LookupRequest` to all connected peers via their per-peer `ZMQControlChannel`,
applies `lookup_policy`, caches handles, and signals `_lookup_event_fd`. ZMQ
timeout causes the peer to be added to `_disconnected_peers`.

**ZMQ channel resilience** (Lazy Pirate): identical to the former
`ZMQControlChannel.send_request()` — recreates the REQ socket on each timeout
before retrying. Transparent to the fan-out task.

**Fetch** (`submit_fetch_task`): reads handles from `_handle_cache[lookup_task_id]`
(**does not remove them** — they are still needed by the subsequent `submit_unlock`
for routing). Increments `_peer_rdma_inflight[peer_id]` per key. Calls
`_agent.issue_request(NIXL_READ)` and stores `TransferHandle`s in `_pending_fetches`.

**Unlock** (`submit_unlock`): the **only** removal point for `_handle_cache`.
For each key, reads peer from `_handle_cache[lookup_task_id]`, sends `UnpinRequest`
via the peer's PUSH socket, removes the entry. Deletes `_handle_cache[lookup_task_id]`
once all keys for that task have been unlocked. Called twice per request —
once for unneeded keys (before fetch) and once for plan keys (after fetch) —
together covering all found keys. No TTL or background sweep needed.

**Fetch poll thread**: calls `_agent.check_request(handle)` for all pending
handles. On completion, decrements `_peer_rdma_inflight[peer_id]` for each
completed key and signals `_peer_rdma_cv[peer_id]` when the count reaches 0.
Builds result `Bitmap` and signals `_fetch_event_fd`.
If NIXL exposes a completion fd, this thread can `select.poll()` on it instead
of spinning.

**`disconnect_peer` drain**: before calling `NixlAgent.remove_remote_agent`,
waits on `_peer_rdma_cv[peer_id]` until `_peer_rdma_inflight[peer_id] == 0`.
This ensures no NIXL transfer is in flight against the peer's descriptors when
they are torn down. Peers marked disconnected (`_disconnected_peers`) will have
no new transfers submitted to them, so the drain always terminates.

---

## 7. File Structure

```
lmcache/v1/distributed/
│
├── storage_manager.py                      # + separate store/prefetch adapter lists [modified]
├── l1_manager.py                           # (unchanged)
├── memory_manager.py                       # (unchanged)
│
├── storage_controllers/
│   ├── prefetch_controller.py              # completed_lookup_task_ids + lookup_task_id passthrough [modified]
│   └── store_controller.py                 # (unchanged)
│
├── l2_adapters/
│   ├── remote_l2_adapter.py               # RemoteL2Adapter (thin wrapper)      [NEW]
│   └── nixl_store_l2_adapter.py           # (unchanged)
│
├── remote_controller/                      # Server + peer connection management  [NEW]
│   ├── __init__.py
│   ├── config.py                           # PeerConfig, RemoteControllerConfig
│   ├── protocol.py                         # ZMQ message structs (msgspec)
│   │                                       # Init/MemReg/Lookup/Unpin Req+Resp
│   ├── controller.py                       # RemoteController ABC + ZMQRemoteController
│   └── factory.py                          # build_remote_controller(...)
│
└── remote_io/                              # Client-side I/O: lookup + RDMA       [NEW]
    ├── __init__.py
    ├── adapter.py                          # RemoteIOAdapter ABC
    │                                       # + LocalMemHandle, IOTaskId
    ├── config.py                           # RemoteIOAdapterConfig (lookup_policy)
    └── backends/
        ├── __init__.py
        └── nixl_backend.py                 # NixlIOAdapter (ZMQ fan-out + NIXL/UCX)
```

> `remote_controller/types.py` (RemoteKeyInfo, LookupResult) is removed — those
> types are now internal to `NixlIOAdapter`.

---

## 8. Relation to Existing Code

| Existing component | Reused as |
|---|---|
| `NixlChannel` — ZMQ REQ/REP + handshake | Template for `ZMQControlChannel` in RC (Init/MemReg) and per-peer ZMQ REQ sockets in `NixlIOAdapter` (lookup traffic) |
| `NixlAgentWrapper` — agent init + xfer_descs | Moved into `NixlIOAdapter.register_local_memory` and `get_local_xfer_descs` |
| `NixlStoreL2Adapter` — eventfd + async poll loop | Template for `NixlIOAdapter` lookup/fetch eventfds and fetch poll thread |
| `L1Manager.reserve_read / finish_read` | Called by `ZMQRemoteController` server-side handlers for `LookupRequest` and `UnpinRequest` |
| `L1Manager.reserve_write / finish_write` | Called by `PrefetchController` (unchanged) after lookup returns found keys |
| `PrefetchController` | Unchanged — treats `RemoteL2Adapter` as a standard L2 adapter |
| `pd_backend.py` — proxy key-ready notification | Unchanged; sends key list to decode side after KV store completes |
