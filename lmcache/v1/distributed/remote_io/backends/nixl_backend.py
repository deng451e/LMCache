# SPDX-License-Identifier: Apache-2.0
"""NixlIOAdapter: RemoteIOAdapter backed by NIXL RDMA + ZMQ control plane.

ZMQ layer (lookup, unpin) is fully implemented.
NIXL RDMA layer (register_local_memory, fetch) is implemented using the UCX
backend via nixl_agent. When peer_metadata is empty (stub mode), the ZMQ
control plane works alone and RDMA falls back to a zero-hit bitmap.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
from typing import Any
import os
import threading
import time
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
# Per-peer handle: routing info from one lookup hit for one key
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
    """All per-key handles from one submit_lookup_task call.

    Handles are keyed by ObjectKey (not original position) to allow
    submit_fetch_task and submit_unlock to look up by key value rather
    than by index, which would not be stable across key-list subsets.
    """

    handles: dict[ObjectKey, _RemoteKeyHandle]


# ---------------------------------------------------------------------------
# Per-peer ZMQ + NIXL state
# ---------------------------------------------------------------------------


@dataclass
class _PeerState:
    """ZMQ sockets and NIXL transfer state for one connected peer."""

    peer_id: str
    lookup_socket: zmq.Socket
    unpin_socket: zmq.Socket
    rdma_inflight: int = 0
    rdma_cv: threading.Condition = field(
        default_factory=threading.Condition, repr=False
    )
    remote_agent_name: str = ""
    remote_xfer_handler: Any = None  # nixl_prepped_dlist_handle, or None
    lookup_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    """Serializes concurrent lookups from multiple _lookup_worker threads.

    ZMQ REQ sockets are not thread-safe; multiple concurrent lookup tasks
    for the same peer must hold this lock around send/recv.
    """


# ---------------------------------------------------------------------------
# NixlIOAdapter
# ---------------------------------------------------------------------------


class NixlIOAdapter(RemoteIOAdapter):
    """RemoteIOAdapter backed by ZMQ lookup + NIXL RDMA.

    ZMQ lookup fan-out and UnpinRequest dispatch are fully implemented.
    NIXL RDMA (register_local_memory, submit_fetch_task) are implemented
    using the UCX backend when NIXL is available.

    If peer_metadata is empty (peer stub mode), RDMA is unavailable for that
    peer and submit_fetch_task returns a zero-hit bitmap for those keys.

    Background threads:
        - One daemon thread per submit_lookup_task call (fan-out).
        - One daemon thread per submit_fetch_task call (NIXL READ poll loop).

    Args:
        config: Adapter configuration (lookup_policy, zmq_timeout_ms,
                align_bytes, nixl_backends).
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

        # Handle cache: lookup_task_id -> per-key handles (keyed by ObjectKey)
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

        # NIXL agent and registered memory (populated by register_local_memory)
        self._nixl_agent: Any = None
        self._reg_descs: Any = None
        self._xfer_descs: Any = None
        self._xfer_handler: Any = None
        self._align_bytes: int = config.align_bytes
        self._mem_type: str = "DRAM"
        self._nixl_lock = threading.Lock()

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

    # ------------------------------------------------------------------
    # Local memory registration (NIXL)
    # ------------------------------------------------------------------

    def register_local_memory(
        self,
        ptr: int,
        size: int,
        device: str,
    ) -> LocalMemHandle:
        """Register the L1 buffer with NIXL for RDMA access.

        Creates a NIXL agent with the UCX backend, registers the whole L1
        buffer as one region, and builds one xfer_desc entry per
        align_bytes-sized page so that page indices map 1-to-1 with
        MemoryObj.meta.address // align_bytes.

        Args:
            ptr:    Base address (data_ptr()) of the L1 buffer.
            size:   Buffer size in bytes.
            device: PyTorch device type — "cpu" (DRAM) or "cuda" (VRAM).

        Returns:
            Opaque LocalMemHandle (unused beyond signaling success).

        Raises:
            RuntimeError: If NIXL is unavailable or registration fails.
        """
        try:
            # Third Party
            from nixl._api import nixl_agent as NixlAgent  # noqa: PLC0415
            from nixl._api import nixl_agent_config as NixlAgentConfig
        except ImportError as err:
            raise RuntimeError("NIXL is not available") from err

        mem_type = "DRAM" if device == "cpu" else "VRAM"
        # device_id: 0 for CPU; for CUDA use the current device index.
        if device == "cuda":
            try:
                # Third Party
                import torch  # noqa: PLC0415

                device_id = torch.cuda.current_device()
            except Exception:
                device_id = 0
        else:
            device_id = 0

        page_size = self._align_bytes
        if page_size == 0:
            raise RuntimeError(
                "align_bytes must be set in RemoteIOAdapterConfig before "
                "calling register_local_memory"
            )

        agent = NixlAgent(
            str(uuid.uuid4()),
            NixlAgentConfig(backends=self._config.nixl_backends),
        )

        reg_descs = agent.register_memory(
            [(ptr, size, device_id, "")], mem_type=mem_type
        )

        xfer_desc = [
            (ptr + i * page_size, page_size, device_id)
            for i in range(size // page_size)
        ]
        xfer_descs = agent.get_xfer_descs(xfer_desc, mem_type=mem_type)
        xfer_handler = agent.prep_xfer_dlist("", xfer_descs, mem_type=mem_type)

        with self._nixl_lock:
            self._nixl_agent = agent
            self._reg_descs = reg_descs
            self._xfer_descs = xfer_descs
            self._xfer_handler = xfer_handler
            self._mem_type = mem_type

        logger.info(
            "[Remote] NIXL registered local memory: "
            "ptr=0x%x size=%d device=%s pages=%d",
            ptr,
            size,
            device,
            len(xfer_desc),
        )
        return LocalMemHandle()

    # ------------------------------------------------------------------
    # Metadata (NIXL agent descriptor exchange)
    # ------------------------------------------------------------------

    def get_local_metadata(self) -> bytes:
        """Return serialised NIXL agent descriptor for the Init handshake.

        Returns empty bytes when register_local_memory has not yet been called
        (ZMQ-only mode). Both sides must have non-empty metadata for RDMA to
        be available; a peer with empty metadata will skip NIXL registration
        and fall back to ZMQ-only lookup.

        Returns:
            Serialised NIXL agent descriptor, or empty bytes.
        """
        with self._nixl_lock:
            if self._nixl_agent is None:
                return b""
            return self._nixl_agent.get_agent_metadata()

    def get_local_xfer_descs(self) -> bytes:
        """Return serialised transfer descriptors for the MemReg handshake.

        Returns empty bytes when register_local_memory has not been called.
        A peer receiving empty xfer_descs will skip RDMA registration.

        Returns:
            Serialised xfer_descs bytes, or empty bytes.
        """
        with self._nixl_lock:
            if self._nixl_agent is None or self._xfer_descs is None:
                return b""
            return self._nixl_agent.get_serialized_descs(self._xfer_descs)

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

        Creates ZMQ REQ and PUSH sockets. If both the local NIXL agent and
        the peer's metadata/xfer_descs are non-empty, also registers the peer
        as a remote NIXL agent and prepares an xfer_dlist handle for RDMA.

        Args:
            peer_id:         Logical peer identifier.
            endpoint:        ZMQ REQ endpoint for lookup traffic.
            unpin_endpoint:  ZMQ PUSH endpoint for UnpinRequests.
            peer_metadata:   Serialised NIXL agent descriptor (empty → RDMA
                             unavailable for this peer).
            peer_xfer_descs: Serialised NIXL transfer descriptors (empty →
                             RDMA unavailable for this peer).

        Raises:
            RuntimeError: If ZMQ socket creation fails.
        """
        lookup_sock = self._new_req_socket(endpoint)
        unpin_sock = self._new_push_socket(unpin_endpoint)

        remote_agent_name = ""
        remote_xfer_handler = None

        if peer_metadata and peer_xfer_descs:
            with self._nixl_lock:
                agent = self._nixl_agent
            if agent is not None:
                try:
                    remote_agent_name = agent.add_remote_agent(peer_metadata)
                    remote_xfer_dlist = agent.deserialize_descs(peer_xfer_descs)
                    remote_xfer_handler = agent.prep_xfer_dlist(
                        remote_agent_name, remote_xfer_dlist
                    )
                    logger.info(
                        "[Remote] NIXL registered remote peer %s (agent=%s)",
                        peer_id,
                        remote_agent_name,
                    )
                except Exception:
                    logger.exception(
                        "[Remote] NIXL registration failed for peer %s — "
                        "RDMA unavailable, falling back to ZMQ-only",
                        peer_id,
                    )
                    remote_agent_name = ""
                    remote_xfer_handler = None
            else:
                logger.warning(
                    "[Remote] local NIXL agent not initialised; peer %s "
                    "registered for ZMQ lookup only",
                    peer_id,
                )
        else:
            logger.info(
                "[Remote] peer %s has empty NIXL metadata/xfer_descs — "
                "ZMQ lookup only (no RDMA)",
                peer_id,
            )

        state = _PeerState(
            peer_id=peer_id,
            lookup_socket=lookup_sock,
            unpin_socket=unpin_sock,
            remote_agent_name=remote_agent_name,
            remote_xfer_handler=remote_xfer_handler,
        )
        with self._peers_lock:
            old = self._peers.pop(peer_id, None)
            if old is not None:
                old.lookup_socket.close()
                old.unpin_socket.close()
            self._peers[peer_id] = state
        with self._disconnected_lock:
            self._disconnected_peers.discard(peer_id)
        logger.info(
            "[Remote] peer connection established: %s at %s (unpin=%s, rdma=%s)",
            peer_id,
            endpoint,
            unpin_endpoint,
            "yes" if remote_xfer_handler is not None else "no",
        )

    def disconnect_peer(self, peer_id: str) -> None:
        """Drain in-flight RDMA and release ZMQ sockets and NIXL resources.

        Waits for rdma_inflight to reach zero before releasing the remote
        NIXL xfer_handler to avoid corrupting in-flight RDMA transfers.

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

        if state.remote_agent_name:
            with self._nixl_lock:
                agent = self._nixl_agent
            if agent is not None:
                try:
                    if state.remote_xfer_handler is not None:
                        agent.release_dlist_handle(state.remote_xfer_handler)
                    agent.remove_remote_agent(state.remote_agent_name)
                except Exception:
                    logger.exception("[Remote] NIXL cleanup error for peer %s", peer_id)

        state.lookup_socket.close()
        state.unpin_socket.close()
        logger.info("[Remote] peer disconnected: %s", peer_id)

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

        Non-blocking. The fan-out runs in a background daemon thread. On
        completion, per-key handles are cached in _handle_cache[task_id]
        (keyed by ObjectKey) and get_lookup_event_fd() is signaled.

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
        """Background: fan-out lookup to all peers and populate handle cache."""
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
                    logger.warning(
                        "[Remote] lookup task %d: peer %s timed out",
                        task_id,
                        state.peer_id,
                    )
                    with self._disconnected_lock:
                        self._disconnected_peers.add(state.peer_id)
                    continue
                peer_hits = len(resp.found_positions)
                logger.info(
                    "[Remote] lookup task %d: peer %s has %d/%d keys",
                    task_id,
                    state.peer_id,
                    peer_hits,
                    num_keys,
                )
                for pos, pages in zip(
                    resp.found_positions, resp.pages_per_found, strict=False
                ):
                    key = keys[pos]
                    if key in entry.handles:
                        if self._config.lookup_policy == "first_found":
                            continue
                        # round_robin: replace when this peer is chosen
                        with self._rr_lock:
                            ordered = sorted(p.peer_id for p in peers)
                            chosen = ordered[self._rr_counter % len(ordered)]
                            self._rr_counter += 1
                        if state.peer_id != chosen:
                            continue
                    entry.handles[key] = _RemoteKeyHandle(
                        peer_id=state.peer_id,
                        found_position=pos,
                        pages=pages,
                    )
                    merged_bitmap.set(pos)
            except Exception:
                logger.exception("[Remote] lookup error for peer %s", state.peer_id)
                with self._disconnected_lock:
                    self._disconnected_peers.add(state.peer_id)

        total_hits = merged_bitmap.popcount()
        logger.info(
            "[Remote] lookup task %d complete: %d/%d keys found across %d peer(s)",
            task_id,
            total_hits,
            num_keys,
            len(peers),
        )

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
        """Send LookupRequest to one peer using Lazy Pirate pattern.

        Acquires state.lookup_lock because ZMQ REQ sockets are not
        thread-safe and concurrent _lookup_worker threads may call this
        for the same peer simultaneously.
        """
        req = LookupRequest(request_id=request_id, keys=wire_keys)
        raw = msgspec.msgpack.encode(req)
        with state.lookup_lock:
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
    # Fetch (NIXL RDMA READ)
    # ------------------------------------------------------------------

    def submit_fetch_task(
        self,
        keys: list[ObjectKey],
        local_objs: list[MemoryObj],
        lookup_task_id: IOTaskId | None = None,
    ) -> IOTaskId:
        """Issue NIXL RDMA READs for keys using handles from the prior lookup.

        Non-blocking. A daemon thread issues a batched NIXL READ per peer,
        polls check_xfer_state until DONE or ERR, then signals the fetch
        eventfd. Falls back to a zero-hit bitmap for any key whose peer has
        no remote_xfer_handler (ZMQ-only peer or NIXL registration failure).

        Args:
            keys:           Keys to fetch (subset of prior lookup's found set).
            local_objs:     L1 write buffers (one per key, same order).
            lookup_task_id: Task ID of the prior submit_lookup_task whose
                            cached handles to use. Should always be provided.

        Returns:
            Task ID for use with query_fetch_result.
        """
        task_id = self._next_id()
        thread = threading.Thread(
            target=self._fetch_worker,
            args=(task_id, keys, local_objs, lookup_task_id),
            daemon=True,
        )
        thread.start()
        return task_id

    def _fetch_worker(
        self,
        task_id: IOTaskId,
        keys: list[ObjectKey],
        local_objs: list[MemoryObj],
        lookup_task_id: IOTaskId | None,
    ) -> None:
        """Background: perform NIXL RDMA READs and write results to bitmap."""
        result_bitmap = Bitmap(len(keys))

        with self._nixl_lock:
            nixl_agent = self._nixl_agent
            xfer_handler = self._xfer_handler
            align_bytes = self._align_bytes

        if nixl_agent is None or xfer_handler is None:
            logger.info(
                "[Remote] fetch task %d: NIXL agent not initialised, "
                "returning zero-hit bitmap (ZMQ-only mode)",
                task_id,
            )
            self._finalize_fetch(task_id, result_bitmap)
            return

        # Snapshot handle cache (copy under lock so we don't hold lock during I/O)
        with self._handle_cache_lock:
            entry = (
                self._handle_cache.get(lookup_task_id)
                if lookup_task_id is not None
                else None
            )
            key_to_handle: dict[ObjectKey, _RemoteKeyHandle] = (
                dict(entry.handles) if entry is not None else {}
            )

        if not key_to_handle:
            logger.warning(
                "[Remote] fetch task %d: no handle cache for lookup_task_id=%s",
                task_id,
                lookup_task_id,
            )
            self._finalize_fetch(task_id, result_bitmap)
            return

        # Group keys by peer: build local/remote page index lists per peer.
        per_peer_local: dict[str, list[int]] = {}
        per_peer_remote: dict[str, list[int]] = {}
        per_peer_key_idxs: dict[str, list[int]] = {}

        for i, (key, local_obj) in enumerate(zip(keys, local_objs, strict=False)):
            handle = key_to_handle.get(key)
            if handle is None:
                continue

            local_start = local_obj.meta.address // align_bytes
            local_num = local_obj.meta.phy_size // align_bytes
            local_pages = list(range(local_start, local_start + local_num))
            remote_pages = handle.pages

            if len(local_pages) != len(remote_pages):
                logger.warning(
                    "[Remote] fetch task %d key %d: "
                    "local pages %d != remote pages %d, skipping",
                    task_id,
                    i,
                    len(local_pages),
                    len(remote_pages),
                )
                continue

            pid = handle.peer_id
            per_peer_local.setdefault(pid, []).extend(local_pages)
            per_peer_remote.setdefault(pid, []).extend(remote_pages)
            per_peer_key_idxs.setdefault(pid, []).append(i)

        with self._peers_lock:
            peers_snap = dict(self._peers)

        total_success = 0
        for peer_id, local_indices in per_peer_local.items():
            remote_indices = per_peer_remote[peer_id]
            state = peers_snap.get(peer_id)
            if state is None:
                continue
            if state.remote_xfer_handler is None:
                logger.info(
                    "[Remote] fetch task %d: peer %s has no NIXL xfer handle "
                    "(ZMQ-only), skipping RDMA for %d keys",
                    task_id,
                    peer_id,
                    len(per_peer_key_idxs[peer_id]),
                )
                continue

            with state.rdma_cv:
                state.rdma_inflight += 1
            try:
                xfer_handle = nixl_agent.make_prepped_xfer(
                    "READ",
                    xfer_handler,
                    local_indices,
                    state.remote_xfer_handler,
                    remote_indices,
                )
                xfer_state = nixl_agent.transfer(xfer_handle)
                while xfer_state not in ("DONE", "ERR"):
                    time.sleep(0.001)
                    xfer_state = nixl_agent.check_xfer_state(xfer_handle)
                nixl_agent.release_xfer_handle(xfer_handle)

                if xfer_state == "DONE":
                    for ki in per_peer_key_idxs[peer_id]:
                        result_bitmap.set(ki)
                    total_success += len(per_peer_key_idxs[peer_id])
                    logger.info(
                        "[Remote] fetch task %d: peer %s RDMA READ done — "
                        "%d keys transferred",
                        task_id,
                        peer_id,
                        len(per_peer_key_idxs[peer_id]),
                    )
                else:
                    logger.error(
                        "[Remote] fetch task %d: NIXL READ ERR from peer %s",
                        task_id,
                        peer_id,
                    )
            except Exception:
                logger.exception(
                    "[Remote] fetch task %d: NIXL READ exception from peer %s",
                    task_id,
                    peer_id,
                )
            finally:
                with state.rdma_cv:
                    state.rdma_inflight -= 1
                    state.rdma_cv.notify_all()

        logger.info(
            "[Remote] fetch task %d complete: %d/%d keys fetched via NIXL RDMA",
            task_id,
            total_success,
            len(keys),
        )
        self._finalize_fetch(task_id, result_bitmap)

    def _finalize_fetch(self, task_id: IOTaskId, bitmap: Bitmap) -> None:
        """Store fetch result and signal the fetch eventfd."""
        with self._fetch_lock:
            self._fetch_results[task_id] = bitmap
        os.eventfd_write(self._fetch_efd, 1)

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

        Routes via _handle_cache[lookup_task_id].handles (keyed by ObjectKey).
        Removes the handle cache entry once all keys for the task have been
        unlocked.

        Args:
            keys:           Keys whose remote read locks should be released.
            lookup_task_id: Task ID of the originating submit_lookup_task.
                            Should always be provided.
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
                "[Remote] submit_unlock: no handle cache entry for task_id=%s",
                lookup_task_id,
            )
            return

        wire_keys_per_peer: dict[str, list[WireObjectKey]] = {}

        with self._handle_cache_lock:
            for key in keys:
                handle = entry.handles.pop(key, None)
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
                    "[Remote] submit_unlock: peer %s not found, dropping unpin "
                    "for %d keys",
                    peer_id,
                    len(wire_keys),
                )
                continue
            try:
                unpin = UnpinRequest(request_id=request_id, found_keys=wire_keys)
                state.unpin_socket.send(msgspec.msgpack.encode(unpin), zmq.NOBLOCK)
            except zmq.ZMQError:
                logger.warning(
                    "[Remote] submit_unlock: failed to send UnpinRequest to %s",
                    peer_id,
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
        """Shut down ZMQ sockets, release NIXL resources, and close eventfds."""
        with self._peers_lock:
            for state in self._peers.values():
                state.lookup_socket.close()
                state.unpin_socket.close()
            self._peers.clear()

        with self._nixl_lock:
            agent = self._nixl_agent
            self._nixl_agent = None

        if agent is not None:
            try:
                if self._xfer_handler is not None:
                    agent.release_dlist_handle(self._xfer_handler)
                if self._reg_descs is not None:
                    agent.deregister_memory(self._reg_descs)
            except Exception:
                logger.exception("[Remote] NIXL cleanup error during close")

        self._zmq_ctx.term()
        os.close(self._lookup_efd)
        os.close(self._fetch_efd)
