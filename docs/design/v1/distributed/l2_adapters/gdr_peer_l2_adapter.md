# GPU-Direct Borrowing Peer Adapter (GDR)

`lmcache/v1/distributed/l2_adapters/gdr_peer_l2_adapter.py`,
`BorrowingPeerL2AdapterBase` in `lmcache/v1/distributed/l2_adapters/p2p_l2_adapter.py`,
`NicDmaEngine` in `lmcache/v1/multiprocess/object_group_transfer/nic_dma.py`,
`HostPoller` in `lmcache/v1/multiprocess/host_poller.py`

The copying P2P adapter ([`p2p_l2_adapter.md`](p2p_l2_adapter.md)) pulls a
peer's object into the local DRAM L1 at lookup, then copies it to the GPU at
retrieve. The GDR adapter **borrows** it instead: the owner keeps its read
lock, and the bytes move once, at retrieve, from the owner's DRAM straight
into GPU memory. Nothing is staged in the local L1.

```
--p2p-load-mode l1 (copy)                  --p2p-load-mode gdr (borrow)
lookup:   owner DRAM ─RDMA─► local L1      lookup:   owner read-locks; shadow L1 indexes a handle
retrieve: local L1 ─PCIe─► GPU             retrieve: owner DRAM ─RDMA─► GPU slot ─kernel─► paged KV
unlock:   after the copy                   unlock:   after the last local reader
```

Enable with `--p2p-load-mode gdr --p2p-transfer-engine nixl`. Borrowed objects
are indexed by a DRAM shadow L1
([`../l1_manager/l1-manager-types.md`](../l1_manager/l1-manager-types.md));
the server adds one (`shadow.dram`) when none is configured. `gdr` excludes
`--enable-p2p-write`.

## Classes

`GdrPeerL2Adapter` extends `BorrowingPeerL2AdapterBase`; the class tree is in
[`p2p_l2_adapter.md`](p2p_l2_adapter.md).

- Each lookup hit adds one reservation for its key.
- `submit_load_task` binds the shadow L1's `BorrowedMemoryObj` placeholder to
  one reservation, which becomes the object's lease. Nothing is copied.
- `submit_unlock` returns the reservations that were not borrowed.
- Ending a lease fires the unlock RPC without waiting for the answer, so
  `finish_read` never blocks on the owner.
- `close` refuses while a lease is live; the P2P controller retries the
  removal every discovery cycle.

## Lease

A lease lasts until `lookup start + read TTL - margin`, with
`margin = min(1 s, TTL / 10)`, and must have `load_timeout_s` left for a read
to start. After that the object is invalid, a read of it is refused, and the key
is a miss. An expired reservation is not unlocked; the owner's TTL reclaims it.

The TTL is not sent over the wire: **every node must use the same
`--l1-read-ttl-seconds`**, and the borrower uses its own. A peer with a shorter
TTL would let the borrower read bytes the owner already reused. Nothing
checks this today (known limitation); sending the owner's TTL with the lookup
reply is a possible follow-up.

## Retrieve

`transfer_kv_per_object_group` splits a group into runs by transfer engine
(see [`lmcache_driven_transfer_copy_paths.md`](../../multiprocess/lmcache_driven_transfer_copy_paths.md)).
Runs of peer-owned objects are **deferred, not staged**, and read without
blocking the retrieve. `platform/base` is not involved: the context's
transfer callback still returns nothing, so the transfer hands the runs to
the `collect_peer_runs(...)` block the handler wraps around the call
(`object_group_transfer/nic_staging.py`). A caller without such a block gets
`NotImplementedError` for a peer-owned object, as before:

```
RETRIEVE handler (affinity worker)
  host-memory runs → queued on the stream as usual
  peer runs        → collected as DeferredNicRun (one per kernel batch, own block ids)
  NicDmaEngine.read(runs) → DeferredResponse              worker is free again

per wave (≤ half the NIC staging slots), FIFO across the context's retrieves:
  take free slots of the NIC staging buffer (`_SlotPool`)
  one BorrowedReader.read_many per peer: owner DRAM ─RDMA─► NIC slots   one NIXL submit
  HostPoller sees the reads finished → on the worker:
      block kernels: NIC slots → paged KV, then record an event
  HostPoller sees the event complete  → on the worker: slots free again

after the last wave's kernels are queued (on the worker):
  record the completion event, queue finish_read_by_owner, MP_RETRIEVE_END
  → DeferredResponse = (event handle, True)
```

Why it is built this way:

- **A NIC staging buffer.** The shared staging buffer (GPU DMA, NVMe) is
  reused by every copy queued on the stream, so a read into it would first
  wait for the stream. The NIC buffer is written only by peer reads and read
  only by the kernels queued after them. `create_staging_buffer` (a
  context that has it satisfies the `StagingBufferFactory` protocol; today
  the CUDA context) allocates it on a context's first peer read with the shared buffer's slot layout,
  aligned to the L1 alignment and registered as one transfer-channel region
  (`TransferChannelStagingRegistration`); the context deregisters it at
  `close`. Size: `NIC_STAGING_BYTES` = 512 MiB of slots (at least two kernel
  batches) plus two rows for an unfull chunk, whose object can extend past
  one slot. Over a 400 Gb/s link, waves of half the slots ran at 47.9–48.5
  GB/s for 6 MiB and 80 MiB chunks; a one-slot pool ran at 40 GB/s.
- **One read per peer per wave** (`BorrowedReader.read_many`). Per-object
  submits gave the same bandwidth but 10x the poller CPU and up to 13 ms
  tail delay before the kernels were queued.
- **The host orders the work; no stream waits on a host-written value.** A
  stream parked on `cuStreamWaitValue32` blocks every first-time kernel
  launch in the process and deadlocks if the releasing thread launches one
  (the hang behind PR #6687). The poller only checks reads and events,
  backing off from 50 µs to 1 ms, and hands every step to the instance's
  affinity worker, which owns the slot pools.
- **Own block ids.** The staged block ids live in a buffer the next transfer
  restages, so each run clones them on the stream when it is made.

Under `--rocm-host-event-wait` the waiter completes the request from the
handler's `DeferredResponse`. Only a retrieve of a borrowed object creates
a NIC staging buffer, so copy mode never registers GPU memory and needs no
GPUDirect support. Separate staging buffers
and CUDA streams for GPU DMA and NVMe are a planned follow-up, needed once
mixed DRAM + GDS retrieves are supported.

## Peer clients

The transfer channel keeps one client per peer URL. A GDR adapter whose
removal waits for its leases (see Failures) can still hold that client when
the controller adds a new adapter for the same peer. Adapters therefore take
the client with `acquire_transfer_channel_client` and give it back with
`release_transfer_channel_client`; the client is removed only when its last
user releases it.

## Failures

| Failure | Behavior |
|---|---|
| Read fails | The retrieve completes with `False` once every read it started has stopped; its later waves are never read and no kernel is queued after the failure. vLLM recomputes the blocks (`kv_load_failure_policy=recompute`). |
| Read exceeds `load_timeout_s` | It is cancelled (`TransferChannelClient.cancel`) and fails once the transport confirms it stopped, also if it finished before the cancel took effect: the owner's lock may lapse before late bytes are used. Until then its slots stay taken: a read still running could land in them. A cancel the transport never confirms keeps the retrieve waiting (known gap). |
| NIC staging buffer cannot be allocated or registered | The retrieve completes with `False` at once, so its read locks are released and vLLM recomputes. The next borrowed retrieve tries again. |
| Instance unregistered or reaped with reads in flight | `NicDmaEngine.drain` waits until the context's retrieves completed before the context is closed: closing deregisters the NIC slots and unmaps the KV cache. From the drain on, a read of the context fails at once and its waves still waiting for slots are dropped, so only reads in flight are waited for. The wait runs on the request loop and is bounded by `--p2p-load-timeout` + 1 s; if reads still run after that, the context and its NIC buffer are left open rather than freed under them. Such a context is never closed, also not once its reads stop: its stream is not synchronized and its imported events are not released until the process exits. |
| Server shuts down with reads in flight | The poller abandons them: the retrieve fails and their slots are never reused. |
| Read TTL not above `--p2p-load-timeout` + 1 s | The server refuses to start in `gdr` mode. |
| Lease expired | The key is a miss; a read of a held object is refused. |
| Owner unreachable for unlocks | The unlock is lost; the owner's TTL reclaims the locks. |
| Peer removed with live leases or a read in flight | `close()` raises; the controller retries each cycle. |

## Known limitations

- A cancelled read relies on nixl's contract that a successful release of an
  active transfer means no more bytes land. The tests only cancel a transfer
  that already finished.
- At shutdown, if an adapter could not be removed (leases or reads still
  live), the P2P controller leaves the transfer channel open instead of
  closing it under a read; the process exit reclaims it.
- A read that completes between a poll and its cancel is reported failed.
