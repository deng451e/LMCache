# SPDX-License-Identifier: Apache-2.0
"""CxlRemoteL2Adapter: access remote peers' CXL sub-regions as an L2 source.

Since all hosts mmap the same full physical CXL region via the DAX device,
"remote" data is already in the local process VA space at:
    peer_va = local_region_va_base + byte_offset

where byte_offset is the absolute offset within the global shared region
returned by CxlLookupResponse.  No NIXL/RDMA is required; the load step
is a direct memory copy from CXL NUMA VA to the destination MemoryObj.

Socket layout (per peer):
  One ZMQ REQ socket for CxlInitRequest + CxlLookupRequest (REP side).
  One ZMQ PUSH socket for CxlUnpinRequest (PULL side, no reply).
"""

# Future
from __future__ import annotations

# Standard
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING
import ctypes
import os
import threading
import uuid

# Third Party
import msgspec
import torch
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.cxl.protocol import (
    CxlInitRequest,
    CxlInitResponse,
    CxlLookupRequest,
    CxlLookupResponse,
    CxlRepMessage,
    CxlSubregionMeta,
    CxlUnpinRequest,
)
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface, L2TaskId
from lmcache.v1.distributed.remote_controller.protocol import WireObjectKey
from lmcache.v1.memory_management import MemoryObj

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.config import CxlConfig

logger = init_logger(__name__)

# Per-lookup request timeout in milliseconds.
_ZMQ_TIMEOUT_MS = 5_000
# Number of retries for Lazy Pirate pattern.
_ZMQ_RETRIES = 3


def _key_to_wire(k: ObjectKey) -> WireObjectKey:
    return WireObjectKey(
        chunk_hash=k.chunk_hash, model_name=k.model_name, kv_rank=k.kv_rank
    )


# -----------------------------------------------------------------------
# Per-peer ZMQ channel
# -----------------------------------------------------------------------


@dataclass
class _PeerMeta:
    """Server sub-region info received during handshake."""

    subregion_offset: int
    subregion_size: int


class _CxlPeerChannel:
    """ZMQ REQ + PUSH sockets for a single CXL peer.

    Args:
        ctx: Shared ZMQ context.
        host: Peer hostname or IP.
        lookup_port: REP socket port on the peer.
        unpin_port: PULL socket port on the peer.
        timeout_ms: Per-request recv timeout in milliseconds.
    """

    def __init__(
        self,
        ctx: zmq.Context,
        host: str,
        lookup_port: int,
        unpin_port: int,
        timeout_ms: int = _ZMQ_TIMEOUT_MS,
    ) -> None:
        self._ctx = ctx
        self._endpoint = f"tcp://{host}:{lookup_port}"
        self._unpin_endpoint = f"tcp://{host}:{unpin_port}"
        self._timeout_ms = timeout_ms

        self._enc = msgspec.msgpack.Encoder()
        self._rep_dec = msgspec.msgpack.Decoder(CxlRepMessage)

        self._req_socket = self._new_req_socket()

        self._push_socket: zmq.Socket = ctx.socket(zmq.PUSH)
        self._push_socket.setsockopt(zmq.LINGER, 0)
        self._push_socket.connect(self._unpin_endpoint)

        self._meta: _PeerMeta | None = None

    def _new_req_socket(self) -> zmq.Socket:
        sock = self._ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(self._endpoint)
        return sock

    def _send_req(self, msg: bytes, retries: int = _ZMQ_RETRIES) -> bytes:
        """Send a REQ message with Lazy Pirate retry on timeout.

        Args:
            msg: Encoded request bytes.
            retries: Number of retry attempts.

        Returns:
            Raw response bytes.

        Raises:
            TimeoutError: If all retries time out.
        """
        for _ in range(retries):
            self._req_socket.send(msg)
            if self._req_socket.poll(self._timeout_ms):
                return self._req_socket.recv()
            self._req_socket.close()
            self._req_socket = self._new_req_socket()
        raise TimeoutError(f"all {retries} attempts to {self._endpoint} timed out")

    def handshake(self, local_meta: CxlSubregionMeta) -> _PeerMeta:
        """Send CxlInitRequest and store the peer's sub-region descriptor.

        Args:
            local_meta: This host's sub-region descriptor.

        Returns:
            The peer's sub-region descriptor.

        Raises:
            TimeoutError: If the handshake times out.
        """
        req = CxlInitRequest(local_meta=local_meta)
        raw = msgspec.msgpack.encode(req)
        resp_raw = self._send_req(raw)
        resp = msgspec.msgpack.decode(resp_raw, type=CxlInitResponse)
        self._meta = _PeerMeta(
            subregion_offset=resp.server_meta.subregion_offset,
            subregion_size=resp.server_meta.subregion_size,
        )
        return self._meta

    def lookup(self, request_id: str, keys: list[ObjectKey]) -> CxlLookupResponse:
        """Send a CxlLookupRequest to the peer.

        Args:
            request_id: Stable request ID for dedup on the server side.
            keys: Keys to look up.

        Returns:
            CxlLookupResponse from the peer.

        Raises:
            TimeoutError: If the request times out after all retries.
        """
        wire_keys = [_key_to_wire(k) for k in keys]
        req = CxlLookupRequest(request_id=request_id, keys=wire_keys)
        raw = msgspec.msgpack.encode(req)
        resp_raw = self._send_req(raw)
        return msgspec.msgpack.decode(resp_raw, type=CxlLookupResponse)

    def unpin(self, request_id: str, found_wire_keys: list[WireObjectKey]) -> None:
        """Fire-and-forget CxlUnpinRequest via PUSH socket.

        Args:
            request_id: Must match the originating CxlLookupRequest.
            found_wire_keys: Keys returned in the prior CxlLookupResponse.
        """
        msg = CxlUnpinRequest(request_id=request_id, found_keys=found_wire_keys)
        self._push_socket.send(msgspec.msgpack.encode(msg), zmq.NOBLOCK)

    def close(self) -> None:
        """Close ZMQ sockets."""
        self._req_socket.close(linger=0)
        self._push_socket.close(linger=0)


# -----------------------------------------------------------------------
# Per-task state
# -----------------------------------------------------------------------


@dataclass
class _LookupState:
    """State retained from a completed lookup for use in load and unpin."""

    request_id: str
    """Stable ID sent to all peers; used for dedup + unpin routing."""

    key_routing: dict[ObjectKey, tuple[int, int]]
    """Maps each found key to (peer_va, byte_size) in local VA space."""

    found_by_peer: dict[int, list[WireObjectKey]]
    """Maps peer index to the list of wire keys found at that peer.
    Used to route unpin messages."""


# -----------------------------------------------------------------------
# CxlRemoteL2Adapter
# -----------------------------------------------------------------------


class CxlRemoteL2Adapter(L2AdapterInterface):
    """L2 adapter that reads from remote peers' CXL sub-regions.

    All peers share the same DAX-mmap'd CXL region.  Remote data is
    accessible at ``local_region_va_base + byte_offset`` without RDMA.
    The load step is a direct CXL-NUMA-to-destination memcpy in a
    background thread pool.

    ``requires_pre_allocation()`` returns True: the PrefetchController must
    supply destination MemoryObj buffers before calling ``submit_load_task``.

    Args:
        config: CxlConfig with peer addresses and region metadata.
        local_region_va_base: VA of the full CXL region on this host.
        local_meta: This host's sub-region descriptor (for handshakes).
    """

    def __init__(
        self,
        config: "CxlConfig",
        local_region_va_base: int,
        local_meta: CxlSubregionMeta,
    ) -> None:
        super().__init__()
        self._region_va_base = local_region_va_base
        self._local_meta = local_meta

        self._zmq_ctx = zmq.Context()
        self._channels: list[_CxlPeerChannel] = []
        for peer in config.peers:
            ch = _CxlPeerChannel(
                ctx=self._zmq_ctx,
                host=peer.host,
                lookup_port=peer.port,
                unpin_port=peer.unpin_port,
            )
            try:
                ch.handshake(local_meta)
                logger.info(
                    "CxlRemoteL2Adapter: connected to peer %s:%d",
                    peer.host,
                    peer.port,
                )
            except TimeoutError:
                logger.warning(
                    "CxlRemoteL2Adapter: handshake to %s:%d timed out; peer skipped",
                    peer.host,
                    peer.port,
                )
                ch.close()
                continue
            self._channels.append(ch)

        self._lock = threading.Lock()
        self._next_task_id: L2TaskId = 0

        self._completed_lookup_tasks: dict[L2TaskId, Bitmap] = {}
        self._completed_load_tasks: dict[L2TaskId, Bitmap] = {}
        self._lookup_states: dict[L2TaskId, _LookupState] = {}

        self._store_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self._lookup_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self._load_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)

        self._executor = ThreadPoolExecutor(
            max_workers=max(4, len(self._channels) * 2),
            thread_name_prefix="cxl-remote",
        )

    # -------------------------------------------------------------------
    # L2AdapterInterface — event fds
    # -------------------------------------------------------------------

    def get_store_event_fd(self) -> int:
        """Not supported; remote CXL peers are read-only.

        Returns:
            Unused store eventfd.
        """
        return self._store_efd

    def get_lookup_and_lock_event_fd(self) -> int:
        """Return the eventfd signaled when a lookup task completes.

        Returns:
            Lookup completion eventfd.
        """
        return self._lookup_efd

    def get_load_event_fd(self) -> int:
        """Return the eventfd signaled when a load (copy) task completes.

        Returns:
            Load completion eventfd.
        """
        return self._load_efd

    def requires_pre_allocation(self) -> bool:
        """Return True — destination MemoryObj buffers must be pre-allocated.

        The load step copies CXL NUMA VA data into the caller-supplied buffers.

        Returns:
            True.
        """
        return True

    # -------------------------------------------------------------------
    # Store (unsupported)
    # -------------------------------------------------------------------

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """Not supported; remote CXL peers are read-only sources.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "CxlRemoteL2Adapter does not support store operations"
        )

    def pop_completed_store_tasks(self) -> dict[L2TaskId, bool]:
        """Not supported; remote CXL peers are read-only sources.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "CxlRemoteL2Adapter does not support store operations"
        )

    # -------------------------------------------------------------------
    # Lookup and lock
    # -------------------------------------------------------------------

    def submit_lookup_and_lock_task(self, keys: list[ObjectKey]) -> L2TaskId:
        """Fan out CxlLookupRequest to all connected peers in a background thread.

        Non-blocking.  The task completes when all per-peer requests have
        returned (or timed out).

        Args:
            keys: Keys to look up across all connected peers.

        Returns:
            Task ID for use with query_lookup_and_lock_result.
        """
        with self._lock:
            task_id = self._next_task_id
            self._next_task_id += 1

        request_id = str(uuid.uuid4())
        self._executor.submit(self._run_lookup, task_id, request_id, keys)
        return task_id

    def _run_lookup(
        self,
        task_id: L2TaskId,
        request_id: str,
        keys: list[ObjectKey],
    ) -> None:
        """Worker: fan out lookup to all peers and merge results.

        Args:
            task_id: Identifies this lookup task.
            request_id: Stable UUID for server-side dedup.
            keys: Keys to look up.
        """
        num_keys = len(keys)
        bitmap = Bitmap(num_keys)
        key_routing: dict[ObjectKey, tuple[int, int]] = {}
        found_by_peer: dict[int, list[WireObjectKey]] = {}

        for peer_idx, ch in enumerate(self._channels):
            try:
                resp = ch.lookup(request_id, keys)
            except TimeoutError:
                logger.warning(
                    "CxlRemoteL2Adapter: lookup to peer %d timed out", peer_idx
                )
                continue

            peer_wire_found: list[WireObjectKey] = []
            for list_pos, global_pos in enumerate(resp.found_positions):
                if global_pos >= num_keys:
                    continue
                key = keys[global_pos]
                if key in key_routing:
                    # Already found at a closer peer; release server lock on this peer
                    continue
                byte_offset = resp.byte_offsets[list_pos]
                byte_size = resp.byte_sizes[list_pos]
                peer_va = self._region_va_base + byte_offset
                key_routing[key] = (peer_va, byte_size)
                bitmap.set(global_pos)
                peer_wire_found.append(
                    WireObjectKey(
                        chunk_hash=key.chunk_hash,
                        model_name=key.model_name,
                        kv_rank=key.kv_rank,
                    )
                )
            if peer_wire_found:
                found_by_peer[peer_idx] = peer_wire_found

        state = _LookupState(
            request_id=request_id,
            key_routing=key_routing,
            found_by_peer=found_by_peer,
        )
        with self._lock:
            self._completed_lookup_tasks[task_id] = bitmap
            self._lookup_states[task_id] = state
        os.eventfd_write(self._lookup_efd, 1)

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        """Return the found bitmap for this task (one-shot).

        Args:
            task_id: From submit_lookup_and_lock_task.

        Returns:
            Bitmap with bit i set for found keys, or None if not yet complete.
        """
        with self._lock:
            return self._completed_lookup_tasks.pop(task_id, None)

    # -------------------------------------------------------------------
    # Unlock
    # -------------------------------------------------------------------

    def submit_unlock(
        self,
        keys: list[ObjectKey],
        lookup_task_id: L2TaskId | None = None,
    ) -> None:
        """Send CxlUnpinRequest to each owning peer for the given keys.

        Retrieves per-peer routing from the lookup state.  If lookup_task_id
        is None or the state is missing, unlock is silently skipped.

        Args:
            keys: Keys whose server-side CXL locks should be released.
            lookup_task_id: Task ID of the originating
                submit_lookup_and_lock_task call.
        """
        if lookup_task_id is None:
            return
        with self._lock:
            state = self._lookup_states.get(lookup_task_id)
        if state is None:
            return

        key_set = set(keys)
        for peer_idx, wire_keys in state.found_by_peer.items():
            to_unpin = [
                w
                for w in wire_keys
                if ObjectKey(w.chunk_hash, w.model_name, w.kv_rank) in key_set
            ]
            if to_unpin:
                self._channels[peer_idx].unpin(state.request_id, to_unpin)

    # -------------------------------------------------------------------
    # Load (CXL VA -> destination buffer)
    # -------------------------------------------------------------------

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        lookup_task_id: L2TaskId | None = None,
    ) -> L2TaskId:
        """Copy CXL NUMA data to destination buffers in a background thread.

        For each key, the source VA is computed as
        ``local_region_va_base + byte_offset`` (the global CXL offset returned
        during lookup).  Data is copied to the provided MemoryObj via
        torch.Tensor.copy_().

        Args:
            keys: Keys to copy from remote CXL sub-region.
            objects: Destination MemoryObj buffers (one per key, same order).
            lookup_task_id: Task ID of the originating lookup call; used to
                retrieve per-key VA routing.

        Returns:
            Task ID for use with query_load_result.
        """
        with self._lock:
            task_id = self._next_task_id
            self._next_task_id += 1
            state = (
                self._lookup_states.pop(lookup_task_id, None)
                if lookup_task_id is not None
                else None
            )

        self._executor.submit(self._run_load, task_id, keys, objects, state)
        return task_id

    def _run_load(
        self,
        task_id: L2TaskId,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        state: _LookupState | None,
    ) -> None:
        """Worker: copy each key's CXL data to its destination buffer.

        Args:
            task_id: Identifies this load task.
            keys: Keys to load.
            objects: Destination buffers (one per key).
            state: Lookup state with per-key VA routing.
        """
        bitmap = Bitmap(len(keys))

        if state is None:
            logger.warning(
                "CxlRemoteL2Adapter: load task %d has no lookup state", task_id
            )
        else:
            for i, (key, dst_obj) in enumerate(zip(keys, objects, strict=False)):
                routing = state.key_routing.get(key)
                if routing is None:
                    continue
                peer_va, byte_size = routing
                try:
                    self._copy_cxl_to_obj(peer_va, byte_size, dst_obj)
                    bitmap.set(i)
                except Exception:
                    logger.exception("CxlRemoteL2Adapter: copy failed for key %s", key)

        with self._lock:
            self._completed_load_tasks[task_id] = bitmap
        os.eventfd_write(self._load_efd, 1)

    def _copy_cxl_to_obj(self, src_va: int, byte_size: int, dst_obj: MemoryObj) -> None:
        """Copy ``byte_size`` bytes from ``src_va`` (CXL NUMA) into ``dst_obj``.

        Uses torch.Tensor.copy_() so both CPU-pinned and GPU destinations
        are handled correctly.

        Args:
            src_va: Source virtual address in the CXL DAX mmap.
            byte_size: Number of bytes to copy.
            dst_obj: Destination MemoryObj (DRAM or GPU tensor).
        """
        src_buf = (ctypes.c_uint8 * byte_size).from_address(src_va)
        src_tensor = torch.frombuffer(src_buf, dtype=torch.uint8)
        dst_tensor = dst_obj.raw_tensor
        if dst_tensor is None:
            raise ValueError("Destination MemoryObj has no raw tensor")
        flat_dst = dst_tensor.view(-1).view(torch.uint8)
        flat_dst.copy_(src_tensor[: flat_dst.numel()])

    def query_load_result(self, task_id: L2TaskId) -> Bitmap | None:
        """Return the copy result bitmap for this task (one-shot).

        Args:
            task_id: From submit_load_task.

        Returns:
            Bitmap with bit i set for successfully copied keys, or None if not ready.
        """
        with self._lock:
            return self._completed_load_tasks.pop(task_id, None)

    # -------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------

    def close(self) -> None:
        """Shut down thread pool, close ZMQ sockets and context."""
        self._executor.shutdown(wait=False)
        for ch in self._channels:
            ch.close()
        self._zmq_ctx.term()
        for fd in (self._store_efd, self._lookup_efd, self._load_efd):
            try:
                os.close(fd)
            except OSError:
                pass

    def report_status(self) -> dict:
        """Return a status dict for the CXL remote L2 adapter.

        Returns:
            Dict with is_healthy and peer count.
        """
        return {
            "is_healthy": True,
            "type": "CxlRemoteL2Adapter",
            "peer_count": len(self._channels),
            "region_va_base": hex(self._region_va_base),
        }
