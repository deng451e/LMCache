# Implementation: CXL Adaptor

**Status**: Draft
**Author**: weishu@tensormesh.ai
**Date**: 2026-04-23

Implementation companion to [cxl_adaptor.md](cxl_adaptor.md).

---

## 1. `CxlAdaptor` — Local CXL Region

Implements `L1ManagerListener` + `L2AdapterInterface`.  Owns the local CXL
allocator and `_index`.

### Config

`CxlAdaptorConfig` extends `L2AdapterConfigBase` (from
`distributed/l2_adapters/config.py`) so `StorageManager` can construct it
from config files via the standard `from_dict` / `help` interface.

```python
class CxlAdaptorConfig(L2AdapterConfigBase):
    dax_device_path:   str  # e.g. "/dev/dax0.0" — same path on all hosts
    region_size:       int  # full shared region size in bytes (same on all hosts)
    subregion_offset:  int  # byte offset of this host's owned sub-region
    subregion_size:    int  # size of this host's owned sub-region
    align_bytes:       int  # allocation alignment; passed to TensorMemoryAllocator
    cxl_numa_node:     int  # NUMA node ID of the CXL device on this host
```

### Internal index

```python
@dataclass
class CxlIndexEntry:
    obj:            TensorMemoryObj  # suballocation; carries data_ptr(), phy_size,
                                     # shapes, dtypes directly
    pending_free:   bool = False     # deferred free flag
    l2_lock_count:  int  = 0        # >0 while any caller (local or remote) holds
                                     # a lock between lookup and unlock
```

### Interface

```python
class CxlAdaptor(L1ManagerListener, L2AdapterInterface):

    # -----------------------------------------------------------------------
    # L2AdapterInterface — event fds
    # -----------------------------------------------------------------------

    def get_store_event_fd(self) -> int:
        """Eventfd signaled when a store task completes (immediate no-op)."""

    def get_lookup_and_lock_event_fd(self) -> int:
        """Eventfd signaled when a lookup_and_lock task result is ready."""

    def get_load_event_fd(self) -> int:
        """Eventfd signaled when a shadow-registration (load) task completes."""

    def requires_pre_allocation(self) -> bool:
        """Return False.

        PrefetchController must NOT call reserve_write before submit_load_task
        for CXL hits.  The adapter re-registers a CXL_SHADOW object in
        L1Manager; no DRAM buffer is needed."""
        return False

    # -----------------------------------------------------------------------
    # L2AdapterInterface — store (no-op: data already in CXL)
    # -----------------------------------------------------------------------

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """No-op — objects are CXL_SHADOW; data is already in CXL.
        Signals get_store_event_fd() immediately."""

    def pop_completed_store_tasks(self) -> dict[L2TaskId, bool]:
        """Return all immediately-completed no-op tasks."""

    # -----------------------------------------------------------------------
    # L2AdapterInterface — lookup and lock
    # -----------------------------------------------------------------------

    def submit_lookup_and_lock_task(self, keys: list[ObjectKey]) -> L2TaskId:
        """Check _index for each key; increment l2_lock_count for found keys.
        Skips entries with pending_free=True (being evicted; not returned as hits).
        Non-blocking; result available immediately (signals lookup_efd)."""

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        """Return found bitmap once ready. One-shot."""

    # -----------------------------------------------------------------------
    # L2AdapterInterface — unlock
    # -----------------------------------------------------------------------

    def submit_unlock(
        self,
        keys: list[ObjectKey],
        lookup_task_id: L2TaskId | None = None,
    ) -> None:
        """Decrement l2_lock_count for each key.
        If l2_lock_count reaches 0 and pending_free is True, free CXL pages."""

    # -----------------------------------------------------------------------
    # L2AdapterInterface — load (shadow re-registration, no copy)
    # -----------------------------------------------------------------------

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],   # always [] for CXL (requires_pre_allocation=False)
        lookup_task_id: L2TaskId | None = None,
    ) -> L2TaskId:
        """Re-register each found key as a write-locked shadow in L1Manager.
        objects must be [] (CxlAdaptor owns allocation). Non-blocking."""

    def query_load_result(self, task_id: L2TaskId) -> Bitmap | None:
        """Return shadow-registered bitmap once ready. One-shot."""

    # -----------------------------------------------------------------------
    # Shadow page lifecycle (called from L1Manager.reserve_write and read path)
    # -----------------------------------------------------------------------

    def allocate_and_register_shadow(
        self,
        key: ObjectKey,
        layout_desc: MemoryLayoutDesc,
        is_temporary: bool = False,
    ) -> TensorMemoryObj:
        """Allocate from _allocator, set meta.fmt = CXL_SHADOW, store in
        _index, and register as write-locked shadow in L1Manager."""

    def reregister_shadow(self, key: ObjectKey) -> TensorMemoryObj:
        """Re-register a shadow whose L1 entry was evicted.
        Reconstructs CXL_SHADOW obj from _index[key].obj and calls
        L1Manager.register_shadow."""

    def transfer_to_gpu(self, obj: TensorMemoryObj, dst: MemoryObj) -> Future:
        """Async cudaMemcpy: CXL NUMA VA → GPU buffer (ThreadPoolExecutor)."""

    # -----------------------------------------------------------------------
    # Server-side methods (called by CxlRemoteController in ZMQ thread)
    # -----------------------------------------------------------------------

    def server_lookup_and_lock(
        self,
        keys: list[ObjectKey],
    ) -> dict[ObjectKey, tuple[int, int]]:
        """Synchronously look up keys and increment l2_lock_count.

        Does NOT interact with L1Manager.  Reads _index only.
        Skips entries with pending_free=True (same guard as submit_lookup_and_lock_task).

        Args:
            keys: Keys to look up.

        Returns:
            {key: (byte_offset, byte_size)} for found keys only.
            byte_offset = obj.data_ptr() - _region_va_base.
        """

    def server_unpin(self, keys: list[ObjectKey]) -> None:
        """Decrement l2_lock_count; free if pending_free.
        Called by CxlRemoteController._handle_unpin()."""

    def get_subregion_meta(self) -> CxlSubregionMeta:
        """Return {subregion_offset, subregion_size} for this host's owned sub-region.
        Called by CxlRemoteController during peer handshake."""

    # -----------------------------------------------------------------------
    # L1ManagerListener (all 6 callbacks)
    # -----------------------------------------------------------------------

    def on_l1_keys_reserved_read(self, keys: list[ObjectKey]) -> None:
        """No-op."""

    def on_l1_keys_reserved_write(self, keys: list[ObjectKey]) -> None:
        """No-op."""

    def on_l1_keys_write_finished(self, keys: list[ObjectKey]) -> None:
        """No-op — StoreController handles L2 persistence; data is in CXL."""

    def on_l1_keys_finish_write_and_reserve_read(self, keys: list[ObjectKey]) -> None:
        """No-op."""

    def on_l1_keys_read_finished(self, keys: list[ObjectKey]) -> None:
        """Queue pending-free keys for the deferred-free background thread.

        Called by L1Manager from inside finish_read() — which holds
        L1Manager._lock — so this callback MUST NOT call any L1Manager method
        (deadlock).  Instead it enqueues candidate keys and signals the
        background _deferred_free_thread via _deferred_free_efd.
        """
        to_queue: list[ObjectKey] = []
        with self._lock:
            for key in keys:
                entry = self._index.get(key)
                if entry is not None and entry.pending_free and entry.l2_lock_count == 0:
                    to_queue.append(key)
        if to_queue:
            with self._deferred_free_lock:
                self._deferred_free_queue.extend(to_queue)
            os.eventfd_write(self._deferred_free_efd, 1)

    def on_l1_keys_deleted_by_manager(self, keys: list[ObjectKey]) -> None:
        """Shadow evicted by L1Manager (called while L1 lock is held).

        Acquire _lock.  CXL pages survive — _index entry stays.
        If pending_free=True and l2_lock_count==0, free CXL pages now.
        """
```

`CxlAdaptor` holds an injected `L1Manager` reference for `register_shadow`
and CXL-side eviction.

### `L1Manager.reserve_write` tiering hook

```python
# inside L1Manager.reserve_write (simplified new branch):
for i, key in enumerate(need_to_allocate):
    if self._cxl_adaptor and self._tiering_policy.should_use_cxl(key, layout_desc):
        try:
            obj = self._cxl_adaptor.allocate_and_register_shadow(
                key, layout_desc, is_temporary=is_temporary[i]
            )
        except OutOfMemoryError:
            obj = self._memory_manager.allocate_one(layout_desc)  # DRAM fallback
    else:
        obj = self._memory_manager.allocate_one(layout_desc)
    result[key] = (L1Error.OK, obj)
```

`TieringPolicy` protocol:

```python
class TieringPolicy(Protocol):
    def should_use_cxl(self, key: ObjectKey, layout: MemoryLayoutDesc) -> bool: ...
```

Both `_cxl_adaptor` and `_tiering_policy` are `None` when CXL is not
configured — existing path is completely unchanged.

Two concrete implementations ship with the initial CXL integration:

```python
class AlwaysCxlPolicy:
    """Route every allocation to CXL (useful for testing / all-CXL deployments)."""
    def should_use_cxl(self, key: ObjectKey, layout: MemoryLayoutDesc) -> bool:
        return True

class SizeThresholdCxlPolicy:
    """Route to CXL when the tensor byte-size exceeds a threshold.

    Large KV tensors benefit from CXL's capacity; small tensors (e.g. metadata
    chunks) stay in DRAM to avoid CXL-access latency on the hot path.

    Args:
        min_bytes: Minimum total byte size for CXL routing (default 1 MiB).
    """
    def __init__(self, min_bytes: int = 1 << 20) -> None:
        self._min_bytes = min_bytes

    def should_use_cxl(self, key: ObjectKey, layout: MemoryLayoutDesc) -> bool:
        return layout.total_bytes() >= self._min_bytes
```

---

### Eviction retry invariant

A CXL allocation is freed only when **both** conditions hold:
1. `l2_lock_count == 0` — no outstanding lookup results (local or remote).
2. L1 delete succeeds — the L1 shadow is not read-locked.

When either condition fails, `pending_free=True` is set and the free is retried from
the first callback that observes both conditions satisfied:

| Trigger | When it fires |
|---|---|
| `submit_unlock` | Local PrefetchController releases L2 lock after load |
| `server_unpin` | Remote client sends CxlUnpinRequest |
| `on_l1_keys_read_finished` | L1 read lock released (engine called `finish_read`) |
| `on_l1_keys_deleted_by_manager` | L1Manager evicts the shadow under memory pressure |

Lookups (`submit_lookup_and_lock_task`, `server_lookup_and_lock`) skip keys with
`pending_free=True` so no new lock-holders are created for a draining entry.

### Eviction policy

Eviction is triggered lazily: `allocate_and_register_shadow` calls
`_allocator.batched_allocate()`; if that raises `OutOfMemoryError`, the
adaptor evicts candidate entries before retrying.

**Candidate selection** — LRU order over `_index` entries where:
- `l2_lock_count == 0` (no outstanding local or remote lock)
- `pending_free == False` (not already being freed)

**Eviction procedure** (repeated until enough space is reclaimed or no
candidates remain):

```python
# Inside CxlAdaptor._evict_one() — called under _lock NOT held:
with self._lock:
    candidate = _pick_lru_candidate()        # None if all locked
    if candidate is None:
        raise OutOfMemoryError("CXL region full; all entries locked")
    candidate.pending_free = True

err = self._l1_manager.delete(candidate.key)
with self._lock:
    if err == L1Error.SUCCESS and candidate.l2_lock_count == 0:
        self._allocator.batched_free([candidate.obj])
        del self._index[candidate.key]
    # else: leave pending_free=True; deferred-free thread or callbacks retry
```

**LRU tracking** — `CxlAdaptor` maintains an `OrderedDict` or a
`deque`-based LRU keyed on `ObjectKey`, updated on every
`submit_lookup_and_lock_task` hit and `server_lookup_and_lock` hit.
Entries removed from `_index` are also removed from the LRU structure.

### Deferred-free background thread

`on_l1_keys_read_finished` is called while `L1Manager._lock` is held
(`finish_read` is `@l1_mgr_synchronized`).  Calling `l1_manager.delete()`
inline would deadlock.  `CxlAdaptor` therefore runs a lightweight
**`_deferred_free_thread`** that does the actual delete + free:

```python
# CxlAdaptor internal state (added):
_deferred_free_queue: list[ObjectKey]  # guarded by _deferred_free_lock
_deferred_free_lock:  threading.Lock
_deferred_free_efd:   int              # eventfd, non-blocking

def _deferred_free_loop(self) -> None:
    """Background thread: retry l1_manager.delete + allocator free for
    keys enqueued by on_l1_keys_read_finished."""
    while not self._stop_event.is_set():
        try:
            os.eventfd_read(self._deferred_free_efd)
        except BlockingIOError:
            time.sleep(0.01)
            continue

        with self._deferred_free_lock:
            keys, self._deferred_free_queue = self._deferred_free_queue, []

        for key in keys:
            with self._lock:
                entry = self._index.get(key)
                if entry is None or not entry.pending_free or entry.l2_lock_count > 0:
                    continue
            err = self._l1_manager.delete(key)
            with self._lock:
                entry = self._index.get(key)
                if entry is None:
                    continue
                if err == L1Error.SUCCESS and entry.l2_lock_count == 0:
                    self._allocator.batched_free([entry.obj])
                    del self._index[key]
                # else KEY_IS_LOCKED or other error: leave pending_free=True;
                # next on_l1_keys_read_finished or on_l1_keys_deleted_by_manager retries
```

`_deferred_free_thread` is a daemon thread started in `CxlAdaptor.__init__`
and stopped (via `_stop_event`) in `close()`.

---

## 2. `CXL_SHADOW` Memory Format

No `CXLMemoryObj` subclass needed.  `allocate_and_register_shadow` sets
`meta.fmt = MemoryFormat.CXL_SHADOW` on the standard `TensorMemoryObj`
returned by `_allocator.batched_allocate()`.

### `L1MemoryManager.free()` guard

```python
def free(self, objs: list[MemoryObj]) -> None:
    dram_objs = [o for o in objs if o.get_memory_format() != MemoryFormat.CXL_SHADOW]
    self._allocator.batched_free(dram_objs)
```

---

## 3. `L1Manager.register_shadow()`

```python
def register_shadow(
    self,
    key: ObjectKey,
    obj: TensorMemoryObj,   # meta.fmt must be CXL_SHADOW or REMOTE_CXL_SHADOW
    is_temporary: bool = False,
) -> L1Error:
    """Register an externally-allocated shadow TensorMemoryObj without
    calling _memory_manager.allocate().  Entry is created write-locked.

    Accepted formats:
      CXL_SHADOW        — local CXL page; called by CxlAdaptor.
      REMOTE_CXL_SHADOW — peer DAX window VA; called by CxlRemoteL2Adapter
                          in gpu_direct mode.

    Both formats are skipped by L1MemoryManager.free() — the caller owns
    the underlying memory and is responsible for releasing it.

    Returns L1Error.EXIST if key is already present.
    """
```

---

## 4. `L2AdapterInterface.requires_pre_allocation()`

```python
class L2AdapterInterface(ABC):
    def requires_pre_allocation(self) -> bool:
        """Whether PrefetchController should reserve L1 write buffers before
        calling submit_load_task.

        Default True (all existing adapters unchanged).
        CxlAdaptor returns False: load re-registers a CXL_SHADOW object;
        no DRAM buffer is needed or expected.
        """
        return True
```

### `PrefetchController._transition_to_load_phase` update

The load phase is split into per-adapter allocation:

```python
# (replaces the single reserve_write call, lines ~567-638 of prefetch_controller.py)
pre_alloc_plan: dict[int, Bitmap] = {}
no_pre_alloc_plan: dict[int, Bitmap] = {}
for adapter_idx, bitmap in trimmed_plan.items():
    if self._l2_adapters[adapter_idx].requires_pre_allocation():
        pre_alloc_plan[adapter_idx] = bitmap
    else:
        no_pre_alloc_plan[adapter_idx] = bitmap

# Reserve L1 write buffers only for adapters that need them.
pre_alloc_keys = set()
for bitmap in pre_alloc_plan.values():
    pre_alloc_keys.update(bitmap.gather(request.keys))
write_results = (
    l1_mgr.reserve_write(list(pre_alloc_keys), ...)
    if pre_alloc_keys
    else {}
)
# ... rest of reservation logic unchanged ...

# Submit load tasks: pass objects for pre-alloc adapters, [] for CXL.
for adapter_idx, bitmap in trimmed_plan.items():
    per_adapter_keys = bitmap.gather(request.keys)
    if adapter_idx in pre_alloc_plan:
        per_adapter_objs = [request.write_reserved_objs[k] for k in per_adapter_keys]
    else:
        per_adapter_objs = []
    task_id = self._l2_adapters[adapter_idx].submit_load_task(
        per_adapter_keys, per_adapter_objs,
        lookup_task_id=request.completed_lookup_task_ids.get(adapter_idx),
    )
    request.pending_load_tasks[adapter_idx] = task_id
```

The `_finalize_load` path is unchanged: it calls
`l1_mgr.finish_write_and_reserve_read(loaded_keys)` for all loaded keys
regardless of adapter type, because `CxlAdaptor.submit_load_task` already
wrote a write-locked shadow into `L1Manager` — `finish_write_and_reserve_read`
just transitions it to read-locked.

---

## 5. `CxlRemoteController`

Analogous to `ZMQRemoteController`.  Server queries `CxlAdaptor` instead of
`L1Manager`.  No `MemReg` step.

```python
@dataclass
class CxlControllerConfig:
    serve_host:       str   = "0.0.0.0"
    serve_port:       int   = 5300
    serve_unpin_port: int   = 5301
    peers:            list[PeerConfig] = field(default_factory=list)
    zmq_timeout_ms:   int   = 5000
    remote_pin_ttl_s: int   = 60   # must exceed worst-case lookup→GPU-DMA-done latency
    reconnect_interval_s: int = 30
```

Reuses `PeerConfig` from `remote_controller/config.py`.

### `PinCache` — shared utility

`ZMQRemoteController` already implements the same TTL dedup pattern
(`_DeduplicatedEntry` + expiry logic in `_handle_lookup_request`).  Rather
than duplicating it, extract a `PinCache` utility class into
`distributed/remote_controller/pin_cache.py` and use it from both
`ZMQRemoteController` and `CxlRemoteController`.

```python
# distributed/remote_controller/pin_cache.py

@dataclass
class PinEntry:
    request_id: str
    found_keys: list[WireObjectKey]
    response:   Any              # caller stores whatever response type it needs
    expires_at: float            # time.monotonic() + ttl_s

class PinCache:
    """Thread-safe TTL cache for pinned-key dedup.

    Used by ZMQRemoteController and CxlRemoteController.
    """

    def __init__(self, ttl_s: int) -> None: ...

    def get(self, request_id: str) -> PinEntry | None:
        """Return entry if present and not yet expired."""

    def put(self, entry: PinEntry) -> None:
        """Insert or overwrite entry for request_id."""

    def pop(self, request_id: str) -> PinEntry | None:
        """Remove and return entry; None if missing."""

    def sweep_expired(self) -> list[PinEntry]:
        """Remove and return all entries whose expires_at <= now."""
```

`CxlRemoteController._pin_cache: PinCache` replaces the ad-hoc
`dict[str, CxlPinEntry]`.  `_sweep_expired_pins` calls
`_pin_cache.sweep_expired()` and then `server_unpin` for each returned entry.

### Server loop

```
REP socket:
  CxlInitRequest  → CxlInitResponse { server_meta = _cxl_adaptor.get_subregion_meta() }
  CxlLookupRequest → _handle_lookup() → CxlLookupResponse
  (unknown msgs)  → empty CxlLookupResponse

PULL socket:
  CxlUnpinRequest → _handle_unpin()
```

### `_handle_lookup`

```python
def _handle_lookup(self, msg: CxlLookupRequest) -> CxlLookupResponse:
    # 1. Check _pin_cache; return cached response if present.
    # 2. Call _cxl_adaptor.server_lookup_and_lock(keys).
    # 3. Build found_positions, byte_offsets, byte_sizes from returned dict.
    # 4. Store CxlPinEntry(response, found_keys, expires_at) in _pin_cache; return response.
```

### `_handle_unpin`

```python
def _handle_unpin(self, msg: CxlUnpinRequest) -> None:
    # 1. Pop entry from _pin_cache by request_id; no-op if already expired+swept.
    # 2. Convert entry.found_keys (WireObjectKey → ObjectKey).
    # 3. Call _cxl_adaptor.server_unpin(keys).
```

### `_sweep_expired_pins`

Background thread started in `__init__`, stopped via `_stop_event` in `close()`.

```python
def _sweep_expired_pins(self) -> None:
    while not self._stop_event.wait(timeout=self._config.remote_pin_ttl_s / 2):
        expired = self._pin_cache.sweep_expired()  # thread-safe; acquires internal lock
        for entry in expired:
            keys = [wire_key_to_object_key(k) for k in entry.found_keys]
            self._cxl_adaptor.server_unpin(keys)
```

`server_unpin` is called outside `PinCache`'s internal lock to avoid holding
it during CXL index operations.

### Peer lifecycle

`register_peer(config)`:
1. Open ZMQ REQ channel to peer.
2. Send `CxlInitRequest { local_meta=_cxl_adaptor.get_subregion_meta() }`, receive `CxlInitResponse`.
3. Call `_remote_l2_adapter.connect_peer(peer_id, endpoint, unpin_endpoint, response.server_meta)`.
4. Store `PeerState`.

`unregister_peer(peer_id)`: calls `_remote_l2_adapter.disconnect_peer(peer_id)`.

Reconnect thread: same pattern as `ZMQRemoteController`.

`CxlRemoteController` holds `_remote_l2_adapter: CxlRemoteL2Adapter` as a
concrete type (not `L2AdapterInterface`) so it can call peer lifecycle methods
not part of the interface.

---

## 6. `CxlRemoteL2Adapter`

Implements `L2AdapterInterface` directly — no separate IO adapter layer.
Owns ZMQ channels and the handle cache.  Reads peer data directly from the
shared region using the same VA base injected from `CxlAdaptor` — no
per-peer mmap is needed.
`CxlRemoteController` additionally calls peer lifecycle methods on a concrete
reference; `PrefetchController` uses it only through `L2AdapterInterface`.

Supports two access modes via `CxlRemoteL2AdapterConfig.access_mode`:

- **`dram_bounce`** (default): peer CXL → CPU `memcpy` → local DRAM → GPU.
  `requires_pre_allocation() = True`.
- **`gpu_direct`**: peer CXL → GPU DMA directly from shared region VA.
  `requires_pre_allocation() = False`.  Also implements `L1ManagerListener`
  to defer `CxlUnpinRequest` until the L1 read lock is released (GPU done).

### Config

```python
@dataclass
class CxlRemoteL2AdapterConfig:
    access_mode: Literal["dram_bounce", "gpu_direct"] = "dram_bounce"
```

`CxlRemoteL2Adapter.__init__` receives `region_va_base: int` and
`region_size: int` injected by `StorageManager` from `CxlAdaptor`'s mapped
region — no DAX device path is needed here.

### Internal types

```python
@dataclass
class CxlRemoteHandle:
    peer_id:     str
    byte_offset: int
    byte_size:   int
```

### Interface

```python
class CxlRemoteL2Adapter(L2AdapterInterface):
    """Client-side remote CXL adapter: ZMQ lookup fan-out + shared region access.

    Implements L2AdapterInterface for PrefetchController.
    In gpu_direct mode also implements L1ManagerListener to defer remote
    unlock until the engine releases its L1 read lock (GPU DMA complete).

    Store operations raise NotImplementedError (remote peers are read-only).

    Shared region access: _region_va_base and _region_size are injected from
    CxlAdaptor at construction.  No per-peer mmap is needed — all peers'
    data is accessible via the same VA base using the absolute byte_offset
    returned by CxlLookupResponse.

    Handle cache lifecycle:
      _handle_cache[task_id] created by submit_lookup_and_lock_task,
      read (not removed) by submit_load_task,
      released by submit_unlock (dram_bounce) or on_l1_keys_read_finished
      (gpu_direct).
    """

    # ------------------------------------------------------------------
    # Peer lifecycle — called by CxlRemoteController (concrete reference)
    # ------------------------------------------------------------------

    def connect_peer(
        self,
        peer_id: str,
        endpoint: str,
        unpin_endpoint: str,
        peer_meta: CxlSubregionMeta,
    ) -> None:
        """Open ZMQ channels to peer; validate sub-region does not overlap ours.

        No mmap is performed — the shared region is already mapped and
        cudaHostRegistered by CxlAdaptor at startup.

        Args:
            peer_id:        Logical peer ID.
            endpoint:       ZMQ REQ endpoint for lookup traffic.
            unpin_endpoint: ZMQ PUSH endpoint for fire-and-forget unpins.
            peer_meta:      CxlSubregionMeta from CxlInitResponse (for validation).
        """

    def disconnect_peer(self, peer_id: str) -> None:
        """Close ZMQ channels for peer.

        In gpu_direct mode, also drains _pending_remote_unpin of any entries
        belonging to this peer (identified via CxlRemoteHandle.peer_id).
        For each drained key the corresponding _pending_task_key_count is
        decremented; when it hits zero _handle_cache[task_id] is deleted.
        No CxlUnpinRequest is sent — the peer is gone and the remote
        l2_lock_count is irrelevant.
        """

    def get_disconnected_peers(self) -> list[str]:
        """Return peers that timed out during lookup fan-out."""

    # ------------------------------------------------------------------
    # L2AdapterInterface — event fds
    # ------------------------------------------------------------------

    def get_store_event_fd(self) -> int:
        """Not supported; raises NotImplementedError."""
        raise NotImplementedError

    def get_lookup_and_lock_event_fd(self) -> int:
        """Eventfd signaled when a lookup task result is ready."""

    def get_load_event_fd(self) -> int:
        """Eventfd signaled when a load task completes."""

    # ------------------------------------------------------------------
    # L2AdapterInterface — requires_pre_allocation
    # ------------------------------------------------------------------

    def requires_pre_allocation(self) -> bool:
        """Return True in dram_bounce mode, False in gpu_direct mode.

        gpu_direct: submit_load_task registers REMOTE_CXL_SHADOW objects
        in L1Manager; no DRAM buffer is needed or expected (objects=[]).
        """
        return self._access_mode == "dram_bounce"

    # ------------------------------------------------------------------
    # L2AdapterInterface — store (unsupported)
    # ------------------------------------------------------------------

    def submit_store_task(self, keys, objects) -> L2TaskId:
        raise NotImplementedError

    def pop_completed_store_tasks(self) -> dict[L2TaskId, bool]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # L2AdapterInterface — lookup and lock
    # ------------------------------------------------------------------

    def submit_lookup_and_lock_task(self, keys: list[ObjectKey]) -> L2TaskId:
        """Fan-out CxlLookupRequest to all connected peers (background thread).

        Builds _handle_cache[task_id] from aggregated responses.
        Signals lookup_and_lock_efd on completion.

        Args:
            keys: Keys to look up across all peers.

        Returns:
            Task ID for use with query_lookup_and_lock_result.
        """

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        """One-shot query. Returns Bitmap where bit i = key i found on any peer."""

    # ------------------------------------------------------------------
    # L2AdapterInterface — unlock
    # ------------------------------------------------------------------

    def submit_unlock(
        self,
        keys: list[ObjectKey],
        lookup_task_id: L2TaskId | None = None,
    ) -> None:
        """Release remote lock for each key.

        dram_bounce: sends CxlUnpinRequest immediately (data already in DRAM).
        gpu_direct:  no-op — actual CxlUnpinRequest is deferred to
                     on_l1_keys_read_finished (fired when GPU DMA completes and
                     engine releases L1 read lock).

        Args:
            keys:           Keys whose remote locks should be released.
            lookup_task_id: Task ID from submit_lookup_and_lock_task.
        """

    # ------------------------------------------------------------------
    # L2AdapterInterface — load
    # ------------------------------------------------------------------

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        lookup_task_id: L2TaskId | None = None,
    ) -> L2TaskId:
        """Load found keys from remote CXL peer.

        dram_bounce: memcpy(objects[i].data_ptr() ← peer_va, byte_size) per
            key using ThreadPoolExecutor. Signals load_efd on completion.

        gpu_direct: for each key, create TensorMemoryObj(peer_va,
            REMOTE_CXL_SHADOW) and call L1Manager.register_shadow write-locked.
            Store key → CxlRemoteHandle in _pending_remote_unpin.
            Signals load_efd immediately (no copy). objects must be [].

        Args:
            keys:           Keys to load (subset of prior lookup found keys).
            objects:        Pre-allocated DRAM buffers (dram_bounce) or []
                            (gpu_direct).
            lookup_task_id: Originating lookup task ID; resolves handle cache.

        Returns:
            Task ID for use with query_load_result.
        """

    def query_load_result(self, task_id: L2TaskId) -> Bitmap | None:
        """One-shot query. Returns Bitmap of successfully loaded keys."""

    # ------------------------------------------------------------------
    # L1ManagerListener — gpu_direct mode only
    # ------------------------------------------------------------------

    def on_l1_keys_read_finished(self, keys: list[ObjectKey]) -> None:
        """Fire deferred CxlUnpinRequests for REMOTE_CXL_SHADOW keys.

        Called by L1Manager (inside finish_read) when the engine releases
        its read lock — i.e., after GPU DMA completes and the engine calls
        finish_read_prefetched().  Only acts on keys present in
        _pending_remote_unpin; ignores all others.

        Must be fast and must NOT call any L1Manager method (deadlock).
        Sends CxlUnpinRequest via ZMQ PUSH socket (fire-and-forget).

        After sending, decrements _pending_task_key_count for the originating
        task_id.  When the count reaches zero all keys for that task have been
        unlocked and _handle_cache[task_id] is deleted.

        Args:
            keys: Keys whose L1 read locks were just released.
        """

    def on_l1_keys_reserved_read(self, keys: list[ObjectKey]) -> None:
        """No-op."""

    def on_l1_keys_reserved_write(self, keys: list[ObjectKey]) -> None:
        """No-op."""

    def on_l1_keys_write_finished(self, keys: list[ObjectKey]) -> None:
        """No-op."""

    def on_l1_keys_finish_write_and_reserve_read(self, keys: list[ObjectKey]) -> None:
        """No-op."""

    def on_l1_keys_deleted_by_manager(self, keys: list[ObjectKey]) -> None:
        """No-op."""

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close all peer ZMQ sockets. Shared region cleanup is owned by CxlAdaptor."""
```

### Peer address formula

```python
# On server (CxlRemoteController._handle_lookup):
byte_offset = obj.data_ptr() - _region_va_base   # absolute offset in global region
byte_size   = obj.phy_size

# On client — dram_bounce (submit_load_task):
peer_va = _region_va_base + byte_offset           # client's own VA base + same offset
ctypes.memmove(objects[i].data_ptr(), peer_va, byte_size)

# On client — gpu_direct (submit_load_task):
peer_va = _region_va_base + byte_offset
# TensorMemoryObj wrapping peer_va with meta.fmt = REMOTE_CXL_SHADOW
# GPU cudaMemcpy(gpu_dst ← peer_va) issued by PrefetchController / engine
```

Both sides map the same physical CXL region so the absolute offset resolves
to the correct physical address from either VA base.  No page-size dependency.

### `gpu_direct` internal state

```python
# Added fields for gpu_direct mode:
_l1_manager:             L1Manager                                   # required; passed to __init__ when access_mode == "gpu_direct"
_pending_remote_unpin:   dict[ObjectKey, tuple[L2TaskId, CxlRemoteHandle]]
_pending_task_key_count: dict[L2TaskId, int]                         # ref-count; reaches 0 → delete _handle_cache[task_id]
_pending_unpin_lock:     threading.Lock                              # guards both dicts above
```

`_pending_remote_unpin` maps each loaded key to `(lookup_task_id, handle)`.
`submit_load_task` sets `_pending_task_key_count[task_id] = len(keys)` when
it populates the map.  `on_l1_keys_read_finished` decrements the count for
each drained key; when it reaches zero it also deletes
`_handle_cache[task_id]`, closing the handle-cache leak that would otherwise
accumulate in gpu_direct mode.  The lock guards both dicts against concurrent
access between the PrefetchController thread (writer) and the L1Manager
callback thread (reader/drainer).

---

## 7. Integration Points Summary

| Component | Change |
|---|---|
| `memory_management.py` | Add `MemoryFormat.CXL_SHADOW`, `MemoryFormat.REMOTE_CXL_SHADOW` |
| `distributed/l1_manager.py` | Add `register_shadow()` (accepts both shadow formats) + TieringPolicy hook in `reserve_write` |
| `distributed/memory_manager.py` | Skip `CXL_SHADOW` and `REMOTE_CXL_SHADOW` objects in `free()` |
| `distributed/l2_adapters/base.py` | Add `requires_pre_allocation() → bool` (default `True`) |
| `distributed/storage_controllers/prefetch_controller.py` | Split `_transition_to_load_phase` to handle `requires_pre_allocation=False` adapters |
| `distributed/storage_manager.py` | Construct and wire `CxlAdaptor` + `CxlRemoteL2Adapter`; register both as `L1ManagerListener` (see construction snippet below) |
| `distributed/remote_controller/pin_cache.py` | **New**: `PinCache` utility extracted from `ZMQRemoteController`; shared with `CxlRemoteController` |
| `distributed/remote_controller/controller.py` | Refactor to use `PinCache` (replaces inline `_DeduplicatedEntry` dict) |

### `StorageManager` construction and wiring

```python
# Inside StorageManager.__init__ (simplified):

if config.cxl is not None:
    cxl_adaptor = CxlAdaptor(config.cxl.adaptor, l1_manager=l1_manager)
    l1_manager.register_listener(cxl_adaptor)       # eviction + deferred-free callbacks
    l2_adapters.append(cxl_adaptor)

    if config.cxl.remote is not None:
        l1_manager_ref = l1_manager if config.cxl.remote.access_mode == "gpu_direct" else None
        remote_l2 = CxlRemoteL2Adapter(
            config.cxl.remote,
            region_va_base=cxl_adaptor.region_va_base,  # shared mmap; no second mmap
            region_size=config.cxl.adaptor.region_size,
            l1_manager=l1_manager_ref,
        )
        cxl_controller = CxlRemoteController(
            config.cxl.controller,
            cxl_adaptor=cxl_adaptor,
            remote_l2_adapter=remote_l2,
        )
        if config.cxl.remote.access_mode == "gpu_direct":
            l1_manager.register_listener(remote_l2)  # deferred CxlUnpinRequest callbacks
        l2_adapters.append(remote_l2)
```

`CxlRemoteL2Adapter.__init__` signature:

```python
def __init__(
    self,
    config: CxlRemoteL2AdapterConfig,
    region_va_base: int,               # injected from CxlAdaptor; VA base of full shared region
    region_size: int,                  # injected from CxlAdaptor; full region size in bytes
    l1_manager: L1Manager | None = None,  # required when access_mode == "gpu_direct"
) -> None:
    if config.access_mode == "gpu_direct" and l1_manager is None:
        raise ValueError("l1_manager is required for gpu_direct mode")
    self._region_va_base = region_va_base
    self._region_size    = region_size
    self._l1_manager     = l1_manager
    ...
```

`CxlAdaptorConfig` + top-level CXL config schema:

```python
@dataclass
class CxlConfig:
    adaptor:    CxlAdaptorConfig
    controller: CxlControllerConfig           = field(default_factory=CxlControllerConfig)
    remote:     CxlRemoteL2AdapterConfig | None = None
    tiering_policy: Literal["always", "size_threshold"] = "always"
    tiering_min_bytes: int = 1 << 20          # used when tiering_policy == "size_threshold"
```

---

## 8. File Structure

```
lmcache/v1/
├── memory_management.py                        # modified: + MemoryFormat.CXL_SHADOW
└── distributed/
    ├── l1_manager.py                           # modified: + register_shadow(), TieringPolicy hook
    ├── memory_manager.py                       # modified: + CXL_SHADOW guard in free()
    ├── l2_adapters/
    │   └── base.py                             # modified: + requires_pre_allocation()
    ├── remote_controller/
    │   └── pin_cache.py                        # new: PinCache utility (shared by ZMQRemoteController + CxlRemoteController)
    │
    ├── storage_controllers/
    │   └── prefetch_controller.py              # modified: per-adapter pre_allocation check
    │
    └── cxl/
        ├── __init__.py
        ├── adaptor.py                          # CxlAdaptor, CxlAdaptorConfig,
        │                                       # CxlIndexEntry, TieringPolicy
        │                                       # register_l2_adapter_factory("cxl", ...)
        ├── controller.py                       # CxlRemoteController, CxlControllerConfig
        │                                       # reuses ZMQControlChannel (lazy-pirate) from remote_controller/
        ├── protocol.py                         # CxlSubregionMeta, CxlInitRequest/Response,
        │                                       # CxlLookupRequest/Response, CxlUnpinRequest
        └── remote_l2_adapter.py                # CxlRemoteL2Adapter, CxlRemoteL2AdapterConfig
                                                # CxlRemoteHandle

tests/v1/distributed/cxl/
├── test_cxl_adaptor.py                         # alloc, store, lookup, shadow cycle
├── test_cxl_shadow_registration.py             # register_shadow / eviction
├── test_cxl_eviction_consistency.py            # Path A + Path B under concurrent access
├── test_cxl_server_methods.py                  # server_lookup_and_lock / server_unpin
├── test_cxl_remote_controller.py               # handshake, lookup, dedup, unpin
├── test_cxl_remote_l2_adapter.py               # dram_bounce + gpu_direct modes;
│                                               # connect_peer, load, unlock, deferred unpin
└── test_cxl_e2e.py                             # end-to-end: local + remote CXL prefetch
                                                # (both dram_bounce and gpu_direct)
```
