# Design: RemoteController

**Status**: Draft  
**Author**: weishu@tensormesh.ai  
**Date**: 2026-04-16

For implementation details (interfaces, config, file structure, adapter
internals, idempotency) see [remote_controller_impl.md](remote_controller_impl.md).

---

## 1. Goal

Enable direct memory-to-memory transfer of KV-cache tensors between two LMCache
instances over RDMA (or any pluggable transport), without an intermediate storage
layer.  Supports two use cases:

- **P2P** — symmetric cache sharing: either host can prefetch from the other's L1.
- **PD** — Prefill-Decode disaggregation: Prefill stores KV locally, notifies
  the proxy, and Decode pulls it over RDMA on demand.

---

## 2. Component Placement

Each host runs the same set of components.  The `mode` config field determines
which are active.

```mermaid
graph LR
    subgraph LOCAL["This Host"]
        direction TB
        ENG["Engine"]
        PC["PrefetchController (P2P, optional)"]
        SC["StoreController"]
        RC["RemoteController"]
        L1["L1Manager"]
        RTA["RemoteTransferAdapter"]
    end

    subgraph PEER["Peer Host"]
        direction TB
        RCP["RemoteController"]
        L1P["L1Manager"]
    end

    PRX["Proxy (PD only)"]

    ENG -- "lookup / unlock" --> RC
    PC -. "lookup / unlock" .-> RC
    SC -- "KV ready (PD only)" --> PRX
    RC -- "reserve_read / finish_read" --> L1
    RC -. "connect_peer (init only)" .-> RTA
    ENG -- "read" --> RTA
    PC -. "read" .-> RTA
    RC -- "ZMQ: Lookup / Unpin" --> RCP
    RCP -- "reserve_read / finish_read" --> L1P
    RTA -- "RDMA READ" --> L1P
```

**`RemoteController`** — all control-plane and peer coordination:
- Peer registry: ZMQ channels per peer, remote memory handles, connection lifecycle
- Metadata exchange: `InitRequest / MemRegRequest` handshake at startup
- Lookup: sends `LookupRequest` to peers, applies lookup policy, manages dedup cache
- Server side: handles incoming `LookupRequest / UnpinRequest`, calls `L1Manager.reserve_read / finish_read`

**`RemoteTransferAdapter`** — data-plane transport, no protocol knowledge:
- `NixlTransferBackend` — RDMA READ/WRITE over UCX (InfiniBand / RoCE)
- Interface: `read/write(local_handle, local_pages, remote_handle, remote_pages)`

| Mode | Required | Optional |
|------|----------|---------|
| P2P (both hosts) | `RemoteController` + `RemoteTransferAdapter` | `PrefetchController` (speculative prefetch) |
| PD prefill | `RemoteController` + `StoreController` | — |
| PD decode | `RemoteController` + `RemoteTransferAdapter` | — |

---

## 3. Control Protocol Messages

All messages use `msgspec.Struct` with `tag=True`.  Encoded as msgpack.

```
Message                 Direction          Purpose
──────────────────────────────────────────────────────────────────
InitRequest             client → server    Exchange NIXL agent metadata
InitResponse            server → client    ack + server metadata
MemRegRequest           client → server    Exchange NIXL xfer_descs
MemRegResponse          server → client    ack + server xfer_descs
──────────────────────────────────────────────────────────────────
LookupRequest           client → server    Which keys exist in remote L1?
                                           fields: request_id, keys
                                           request_id is stable across retries (see impl §3)
LookupResponse          server → client    bitmap + page_indices per found key
──────────────────────────────────────────────────────────────────
UnpinRequest            client → server    Release read locks for keys
                                           fields: request_id, found_keys
                                           request_id matches the originating LookupRequest
UnpinResponse           server → client    ack
──────────────────────────────────────────────────────────────────
```

`InitRequest / MemRegRequest / Response` are reused verbatim from the existing
`NixlChannel` handshake protocol.  All messages are handled by `RemoteController`
on both the sending and receiving side.  `RemoteTransferAdapter` has no protocol
knowledge and handles no control messages.

> **PD pull reuses P2P messages.** Both flows use `LookupRequest → RDMA READ →
> UnpinRequest`.  There are no PD-specific control messages.

---

## 4. Data Flows

### 4.1 P2P Pull (host_b fetches from host_a)

Single-peer is the degenerate case (N=1); the flow is identical. Diagram shows
`PrefetchController` as the initiator; without it the Engine drives the same
`RemoteController.lookup()` path on L1 miss.

#### Sequence diagram

```mermaid
sequenceDiagram
    participant PC  as PrefetchCtl (b)
    participant L1B as L1Mgr (b)
    participant RCB as RemoteController (b)
    participant RCA as RemoteController (peer a)
    participant L1A as L1Mgr (a)
    participant RTA as RemoteTransferAdapter (b)

    PC ->> RCB: lookup(keys)
    activate RCB
    par fan-out LookupRequest to all N peers
        RCB ->> RCA: LookupRequest(request_id, keys)
        Note over RCA: reserve_read(keys) on L1A
        RCA ->> RCB: LookupResponse(found_bitmap_i, pages_i)
    end
    RCB ->> RCB: aggregate, first_found policy, unpin losers
    RCB ->> PC: LookupResult(found_keys, key_info)
    deactivate RCB

    PC ->> L1B: reserve_write(found_keys, is_temporary, layout, mode=new)
    L1B ->> PC: write_bitmap + local_objs

    par RDMA READ per peer
        PC ->> RTA: read(local_handle, local_pages, info.remote_handle, info.remote_pages)
        RTA ->> L1A: RDMA READ
        L1A -->> RTA: RDMA DONE
        RTA -->> PC: TransferHandle (DONE)
    end

    alt loaded_keys
        PC ->> L1B: finish_write_and_reserve_read(loaded_keys)
    else load_failed_keys
        PC ->> L1B: finish_write(load_failed_keys)
        PC ->> L1B: delete(load_failed_keys)
    end

    PC ->> RCB: unlock(found_keys)
    activate RCB
    par UnpinRequest per peer
        RCB ->> RCA: UnpinRequest(request_id, peer_found_keys)
        RCA ->> RCB: UnpinResponse
    end
    deactivate RCB
```

> **Implementation note (PrefetchController async path).** The sequence above
> shows the logical flow. In practice, `PrefetchController` never calls
> `rc.lookup()` directly on its background thread. Instead, the full remote
> pipeline (`lookup → reserve_write → rta.read → poll → unlock`) runs in a
> thread-pool task and signals `_remote_efd` on completion. The background loop
> picks up the result via `_process_remote_completions()`, keeping the loop
> non-blocking even if a peer is slow or disconnected. See
> [remote_controller_impl.md §6.1](remote_controller_impl.md).

### 4.2 PD Pull (decode pulls from prefill via RemoteTransferAdapter)

Prefill stores KV in its own L1, notifies the proxy, and serves RDMA READs.
Decode pulls KV on demand — the same `LookupRequest → RDMA READ → UnpinRequest`
path as P2P pull.

For multi-peer routing (round_robin / broadcast) see
[remote_controller_impl.md §2](remote_controller_impl.md).

#### Sequence diagram

```mermaid
sequenceDiagram
    participant PRX as Proxy
    participant SCP as StoreCtl (prefill)
    participant L1P as L1Mgr (prefill)
    participant RCP as RemoteController (prefill)
    participant ENG as Engine (decode)
    participant RCD as RemoteController (decode)
    participant RTA as RemoteTransferAdapter (decode)
    participant L1D as L1Mgr (decode)

    PRX ->> SCP: dispatch prefill request
    activate SCP
    Note over SCP,L1P: prefill computes KV, stores in local L1
    L1P ->> SCP: on_write_finished(keys)
    SCP ->> PRX: KV ready (request_id, keys)
    deactivate SCP

    PRX ->> ENG: forward request + key list (request_id, keys)
    ENG ->> RCD: lookup(request_id, keys)
    activate RCD
    RCD ->> RCP: LookupRequest(request_id, keys)
    activate RCP
    RCP ->> L1P: reserve_read(keys)
    L1P ->> RCP: found_keys + pages
    RCP ->> RCP: dedup cache: store request_id
    RCP ->> RCD: LookupResponse(found_bitmap, pages_per_found)
    deactivate RCP
    RCD ->> ENG: LookupResult(found_keys, key_info)
    deactivate RCD

    ENG ->> L1D: reserve_write(found_keys, is_temporary=False, layout, mode=new)
    L1D ->> ENG: write_bitmap + local_objs

    ENG ->> RTA: read(local_handle, local_pages, info.remote_handle, info.remote_pages)
    activate RTA
    RTA ->> L1P: RDMA READ
    L1P -->> RTA: RDMA DONE
    RTA -->> ENG: TransferHandle (DONE)
    deactivate RTA

    alt loaded_keys
        ENG ->> L1D: finish_write_and_reserve_read(loaded_keys)
    else load_failed_keys
        ENG ->> L1D: finish_write(load_failed_keys)
        ENG ->> L1D: delete(load_failed_keys)
    end

    ENG ->> RCD: unlock(request_id, found_keys)
    activate RCD
    RCD ->> RCP: UnpinRequest(request_id, found_keys)
    activate RCP
    RCP ->> L1P: finish_read(found_keys)
    RCP ->> RCP: dedup cache: evict request_id
    RCP ->> RCD: UnpinResponse
    deactivate RCP
    deactivate RCD

    ENG ->> L1D: finish_read(loaded_keys)
```

