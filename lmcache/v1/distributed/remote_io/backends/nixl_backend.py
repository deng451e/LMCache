# SPDX-License-Identifier: Apache-2.0
"""NixlIOAdapter: RemoteIOAdapter backed by NIXL RDMA + ZMQ control plane.

ZMQ layer (lookup, unpin) is fully implemented.
NIXL RDMA layer (register_local_memory, fetch) is stubbed and raises
NotImplementedError until the NIXL bindings are wired in.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
import os
import threading
import uuid

# Third Party
import msgspec
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.remote_controller.protocol import (
    LookupRequest,
    LookupResponse,
    UnpinRequest,
    WireObjectKey,
)
from lmcache.v1.distributed.remote_io.adapter import (
    IOTaskId,
    LocalMemHandle,
    RemoteIOAdapter,
)
from lmcache.v1.distributed.remote_io.config import RemoteIOAdapterConfig
from lmcache.v1.memory_management import MemoryObj

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Per-peer handle: result of a single ZMQ LookupResponse for one key
# ---------------------------------------------------------------------------


@dataclass
class _RemoteKeyHandle:
    """Cached result for one found key from one lookup task."""

    peer_id: str
    found_position: int
    pages: list[int]


# ---------------------------------------------------------------------------
# Per-lookup task entry in the handle cache
# ---------------------------------------------------------------------------


@dataclass
class _LookupTaskEntry:
    """All per-key handles from one submit_lookup_task call."""

    handles: dict[int, _RemoteKeyHandle]
    """Maps key index (position in the original keys list) -> handle."""


# ---------------------------------------------------------------------------
# Per-fetch task state
# ---------------------------------------------------------------------------


@dataclass
class _FetchTaskState:
    """Tracks one in-flight submit_fetch_task call."""

    result: Bitmap | None = None
    done: bool = False


# ---------------------------------------------------------------------------
# Per-peer ZMQ state
# ---------------------------------------------------------------------------


@dataclass
class _PeerState:
    """ZMQ sockets and RDMA state for one connected peer."""

    peer_id: str
    lookup_socket: zmq.Socket
    unpin_socket: zmq.Socket
    rdma_inflight: int = 0
    rdma_cv: threading.Condition = field(
        default_factory=threading.Condition, repr=False
    )


# ---------------------------------------------------------------------------
# NixlIOAdapter
# ---------------------------------------------------------------------------


class NixlIOAdapter(RemoteIOAdapter):
    """RemoteIOAdapter backed by ZMQ lookup + NIXL RDMA.

    ZMQ lookup fan-out and UnpinRequest dispatch are fully implemented.
    NIXL RDMA (register_local_memory, submit_fetch_task) are stubbed.

    Background threads:
        - One thread pool worker per submit_lookup_task call (fan-out).
        - One thread per submit_fetch_task call (NIXL READ, stubbed).

    Args:
        config: Adapter configuration (lookup_policy, zmq_timeout_ms).
    """

    def __init__(self, config: RemoteIOAdapterConfig) -> None:
        self._config = config

        self._zmq_ctx = zmq.Context(1)

        # Peer registry
        self._peers: dict[str, _PeerState] = {}
        self._peers_lock = threading.Lock()
        self._disconnected_peers: set[str] = set()
        self._disconnected_lock = threading.Lock()

        # Round-robin counter (lookup_policy == "round_robin")
        self._rr_counter: int = 0
        self._rr_lock = threading.Lock()

        # Handle cache: lookup_task_id -> per-key handles
        self._handle_cache: dict[IOTaskId, _LookupTaskEntry] = {}
        self._handle_cache_lock = threading.Lock()

        # Lookup task results
        self._lookup_results: dict[IOTaskId, Bitmap] = {}
        self._lookup_lock = threading.Lock()

        # Fetch task results
        self._fetch_results: dict[IOTaskId, Bitmap] = {}
        self._fetch_lock = threading.Lock()

        # Task ID counter (shared for lookup and fetch)
        self._next_task_id: int = 0
        self._task_id_lock = threading.Lock()

        # Eventfds
        self._lookup_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self._fetch_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_id(self) -> IOTaskId:
        with self._task_id_lock:
            task_id = self._next_task_id
            self._next_task_id += 1
            return task_id

    def _new_req_socket(self, endpoint: str) -> zmq.Socket:
        sock = self._zmq_ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, self._config.zmq_timeout_ms)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(endpoint)
        return sock

    def _new_push_socket(self, endpoint: str) -> zmq.Socket:
        sock = self._zmq_ctx.socket(zmq.PUSH)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(endpoint)
        return sock

    def _send_lookup(
        self,
        peer_id: str,
        sock: zmq.Socket,
        request_id: str,
        wire_keys: list[WireObjectKey],
    ) -> LookupResponse | None:
        """Send a LookupRequest and return the response, or None on timeout."""
        req = LookupRequest(request_id=request_id, keys=wire_keys)
        raw = msgspec.msgpack.encode(req)
        for _ in range(3):
            sock.send(raw)
            if sock.poll(self._config.zmq_timeout_ms):
                resp_raw = sock.recv()
                return msgspec.msgpack.decode(resp_raw, type=LookupResponse)
            sock.close()
            # Re-connect to recover broken REQ state
            sock = self._new_req_socket(sock._connect_target)  # type: ignore[attr-defined]
        return None

    # ------------------------------------------------------------------
    # Local memory registration (NIXL stub)
    # ------------------------------------------------------------------

    def register_local_memory(
        self,
        ptr: int,
        size: int,
        device: str,
    ) -> LocalMemHandle:
        """Register the L1 buffer with NIXL for RDMA access.

        Args:
            ptr:    Base address of the L1 buffer.
            size:   Buffer size in bytes.
            device: "cpu" or "cuda".

        Returns:
            Opaque handle for the registered buffer.

        Raises:
            NotImplementedError: NIXL registration not yet implemented.
        """
        raise NotImplementedError("NIXL memory registration not yet implemented")

    # ------------------------------------------------------------------
    # Metadata (NIXL stub)
    # ------------------------------------------------------------------

    def get_local_metadata(self) -> bytes:
        """Return serialised NIXL agent descriptor for the Init handshake.

        Raises:
            NotImplementedError: NIXL agent not yet implemented.
        """
        raise NotImplementedError("NIXL agent metadata not yet implemented")

    def get_local_xfer_descs(self) -> bytes:
        """Return serialised local transfer descriptors for the MemReg handshake.

        Raises:
            NotImplementedError: NIXL agent not yet implemented.
        """
        raise NotImplementedError("NIXL xfer descs not yet implemented")

    # ------------------------------------------------------------------
    # Peer lifecycle
    # ------------------------------------------------------------------

    def connect_peer(
        self,
        peer_id: str,
        endpoint: str,
        unpin_endpoint: str,
        peer_metadata: bytes,
        peer_xfer_descs: bytes,
    ) -> None:
        """Register a peer's ZMQ sockets and NIXL descriptors.

        Creates a ZMQ REQ socket to ``endpoint`` for lookup traffic and a
        ZMQ PUSH socket to ``unpin_endpoint`` for fire-and-forget UnpinRequests.
        NIXL descriptor registration is stubbed.

        Args:
            peer_id:         Logical peer identifier.
            endpoint:        ZMQ REQ endpoint, e.g. "tcp://host:5200".
            unpin_endpoint:  ZMQ PUSH endpoint, e.g. "tcp://host:5201".
            peer_metadata:   Serialised NIXL agent descriptor (ignored until
                             NIXL is implemented).
            peer_xfer_descs: Serialised NIXL transfer descriptors (ignored
                             until NIXL is implemented).

        Raises:
            RuntimeError: If ZMQ socket creation fails.
        """
        lookup_sock = self._new_req_socket(endpoint)
        unpin_sock = self._new_push_socket(unpin_endpoint)
        state = _PeerState(
            peer_id=peer_id,
            lookup_socket=lookup_sock,
            unpin_socket=unpin_sock,
        )
        with self._peers_lock:
            old = self._peers.pop(peer_id, None)
            if old is not None:
                old.lookup_socket.close()
                old.unpin_socket.close()
            self._peers[peer_id] = state
        with self._disconnected_lock:
            self._disconnected_peers.discard(peer_id)
        logger.info("NixlIOAdapter: connected peer %s at %s", peer_id, endpoint)

    def disconnect_peer(self, peer_id: str) -> None:
        """Drain in-flight RDMA and close ZMQ sockets for the peer.

        Waits for in-flight RDMA count to reach zero before tearing down
        sockets to avoid corrupting in-flight transfers.

        Args:
            peer_id: Must match a previously connected peer.
        """
        with self._peers_lock:
            state = self._peers.pop(peer_id, None)
        if state is None:
            return
        with state.rdma_cv:
            while state.rdma_inflight > 0:
                state.rdma_cv.wait(timeout=1.0)
        state.lookup_socket.close()
        state.unpin_socket.close()
        logger.info("NixlIOAdapter: disconnected peer %s", peer_id)

    def get_disconnected_peers(self) -> list[str]:
        """Return peer_ids that timed out during a recent lookup fan-out.

        Returns:
            List of peer_ids currently in a disconnected state.
        """
        with self._disconnected_lock:
            return list(self._disconnected_peers)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def submit_lookup_task(self, keys: list[ObjectKey]) -> IOTaskId:
        """Fan-out ZMQ LookupRequest to all connected peers.

        Non-blocking. The fan-out runs in a background thread. On completion,
        per-key handles are cached in _handle_cache[task_id] and
        get_lookup_event_fd() is signaled.

        Args:
            keys: Keys to look up across all peers.

        Returns:
            Task ID for use with query_lookup_result.
        """
        task_id = self._next_id()
        thread = threading.Thread(
            target=self._lookup_worker,
            args=(task_id, keys),
            daemon=True,
        )
        thread.start()
        return task_id

    def _lookup_worker(self, task_id: IOTaskId, keys: list[ObjectKey]) -> None:
        """Background: fan-out lookup and populate handle cache + result."""
        wire_keys = [
            WireObjectKey(
                chunk_hash=k.chunk_hash,
                model_name=k.model_name,
                kv_rank=k.kv_rank,
            )
            for k in keys
        ]
        request_id = str(uuid.uuid4())

        num_keys = len(keys)
        merged_bitmap = Bitmap(num_keys)
        entry = _LookupTaskEntry(handles={})

        with self._peers_lock:
            peers = list(self._peers.values())

        for state in peers:
            try:
                resp = self._send_lookup_via_state(state, request_id, wire_keys)
                if resp is None:
                    with self._disconnected_lock:
                        self._disconnected_peers.add(state.peer_id)
                    continue
                for pos, pages in zip(
                    resp.found_positions, resp.pages_per_found, strict=False
                ):
                    if pos in entry.handles:
                        if self._config.lookup_policy == "first_found":
                            continue
                        # round_robin: replace if this peer is next in rotation
                        with self._rr_lock:
                            ordered = sorted(p.peer_id for p in peers)
                            chosen = ordered[self._rr_counter % len(ordered)]
                            self._rr_counter += 1
                        if state.peer_id != chosen:
                            continue
                    entry.handles[pos] = _RemoteKeyHandle(
                        peer_id=state.peer_id,
                        found_position=pos,
                        pages=pages,
                    )
                    merged_bitmap.set(pos)
            except Exception:
                logger.exception(
                    "NixlIOAdapter: lookup error for peer %s", state.peer_id
                )
                with self._disconnected_lock:
                    self._disconnected_peers.add(state.peer_id)

        with self._handle_cache_lock:
            self._handle_cache[task_id] = entry
        with self._lookup_lock:
            self._lookup_results[task_id] = merged_bitmap
        os.eventfd_write(self._lookup_efd, 1)

    def _send_lookup_via_state(
        self,
        state: _PeerState,
        request_id: str,
        wire_keys: list[WireObjectKey],
    ) -> LookupResponse | None:
        """Send LookupRequest to one peer using Lazy Pirate pattern."""
        req = LookupRequest(request_id=request_id, keys=wire_keys)
        raw = msgspec.msgpack.encode(req)
        sock = state.lookup_socket
        for _ in range(3):
            sock.send(raw)
            if sock.poll(self._config.zmq_timeout_ms):
                resp_raw = sock.recv()
                return msgspec.msgpack.decode(resp_raw, type=LookupResponse)
            sock.close()
            sock = self._new_req_socket(sock.underlying_addr)  # type: ignore[attr-defined]
            state.lookup_socket = sock
        return None

    def query_lookup_result(self, task_id: IOTaskId) -> Bitmap | None:
        """Non-blockingly query the result of a lookup task.

        One-shot: returns non-None exactly once per task_id.

        Args:
            task_id: From submit_lookup_task.

        Returns:
            Bitmap of found keys, or None if not yet complete.
        """
        with self._lookup_lock:
            return self._lookup_results.pop(task_id, None)

    # ------------------------------------------------------------------
    # Fetch (NIXL stub)
    # ------------------------------------------------------------------

    def submit_fetch_task(
        self,
        keys: list[ObjectKey],
        local_objs: list[MemoryObj],
        lookup_task_id: IOTaskId | None = None,
    ) -> IOTaskId:
        """Issue NIXL RDMA READs for keys using handles from the prior lookup.

        Args:
            keys:           Keys to fetch.
            local_objs:     L1 write buffers (one per key, same order).
            lookup_task_id: Task ID of the prior submit_lookup_task whose
                            cached handles to use.

        Returns:
            Task ID for use with query_fetch_result.

        Raises:
            NotImplementedError: NIXL RDMA not yet implemented.
        """
        raise NotImplementedError("NIXL RDMA fetch not yet implemented")

    def query_fetch_result(self, task_id: IOTaskId) -> Bitmap | None:
        """Non-blockingly query the result of a fetch task.

        Args:
            task_id: From submit_fetch_task.

        Returns:
            Bitmap of successfully loaded keys, or None if not yet complete.
        """
        with self._fetch_lock:
            return self._fetch_results.pop(task_id, None)

    # ------------------------------------------------------------------
    # Unlock
    # ------------------------------------------------------------------

    def submit_unlock(
        self,
        keys: list[ObjectKey],
        lookup_task_id: IOTaskId | None = None,
    ) -> None:
        """Send ZMQ UnpinRequest to each owning peer for the given keys.

        Reads routing from _handle_cache[lookup_task_id]. Removes the
        handle cache entry once all keys for the task have been unlocked.

        Args:
            keys:           Keys whose remote read locks should be released.
            lookup_task_id: Task ID of the originating submit_lookup_task.
                            Should always be provided; falls back to an
                            empty handle set when None.
        """
        if not keys:
            return

        with self._handle_cache_lock:
            entry = (
                self._handle_cache.get(lookup_task_id)
                if lookup_task_id is not None
                else None
            )

        if entry is None:
            logger.warning(
                "submit_unlock: no handle cache entry for task_id=%s", lookup_task_id
            )
            return

        wire_keys_per_peer: dict[str, list[WireObjectKey]] = {}
        key_index_map = {k: i for i, k in enumerate(keys)}

        with self._handle_cache_lock:
            for key in keys:
                idx = key_index_map.get(key)
                if idx is None:
                    continue
                handle = entry.handles.pop(idx, None)
                if handle is None:
                    continue
                wire_key = WireObjectKey(
                    chunk_hash=key.chunk_hash,
                    model_name=key.model_name,
                    kv_rank=key.kv_rank,
                )
                wire_keys_per_peer.setdefault(handle.peer_id, []).append(wire_key)

            if lookup_task_id is not None and not entry.handles:
                self._handle_cache.pop(lookup_task_id, None)

        request_id = str(uuid.uuid4())
        with self._peers_lock:
            peers_snapshot = dict(self._peers)

        for peer_id, wire_keys in wire_keys_per_peer.items():
            state = peers_snapshot.get(peer_id)
            if state is None:
                logger.warning(
                    "submit_unlock: peer %s not found, dropping unpin for %d keys",
                    peer_id,
                    len(wire_keys),
                )
                continue
            try:
                unpin = UnpinRequest(request_id=request_id, found_keys=wire_keys)
                state.unpin_socket.send(msgspec.msgpack.encode(unpin), zmq.NOBLOCK)
            except zmq.ZMQError:
                logger.warning(
                    "submit_unlock: failed to send UnpinRequest to %s", peer_id
                )

    # ------------------------------------------------------------------
    # Eventfds
    # ------------------------------------------------------------------

    def get_lookup_event_fd(self) -> int:
        """Return the eventfd signaled when a lookup result is available.

        Returns:
            The lookup eventfd file descriptor.
        """
        return self._lookup_efd

    def get_fetch_event_fd(self) -> int:
        """Return the eventfd signaled when a fetch result is available.

        Returns:
            The fetch eventfd file descriptor.
        """
        return self._fetch_efd

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Shut down all ZMQ sockets, terminate the context, and close eventfds."""
        with self._peers_lock:
            for state in self._peers.values():
                state.lookup_socket.close()
                state.unpin_socket.close()
            self._peers.clear()
        self._zmq_ctx.term()
        os.close(self._lookup_efd)
        os.close(self._fetch_efd)
