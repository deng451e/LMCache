# SPDX-License-Identifier: Apache-2.0
"""RemoteController ABC and ZMQRemoteController concrete implementation."""

# Future
from __future__ import annotations

# Standard
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import enum
import threading
import time

# Third Party
import msgspec
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.remote_controller.config import (
    PeerConfig,
    RemoteControllerConfig,
)
from lmcache.v1.distributed.remote_controller.protocol import (
    InitRequest,
    InitResponse,
    LookupRequest,
    LookupResponse,
    MemRegRequest,
    MemRegResponse,
    UnpinRequest,
    UnpinResponse,
    WireObjectKey,
    ZMQMessage,
)
from lmcache.v1.distributed.remote_controller.types import (
    LookupResult,
    RemoteKeyInfo,
)
from lmcache.v1.distributed.remote_transfer.adapter import (
    RemoteMemHandle,
    RemoteTransferAdapter,
)

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# ZMQControlChannel — per-peer REQ socket with Lazy Pirate recovery
# ---------------------------------------------------------------------------


class ZMQControlChannel:
    """Client-side ZMQ REQ socket for a single peer with timeout recovery.

    Implements the Lazy Pirate pattern: on RCVTIMEO the socket is closed and
    recreated to avoid the broken REQ state machine.
    """

    def __init__(
        self,
        ctx: zmq.Context,
        endpoint: str,
        timeout_ms: int,
    ) -> None:
        self._ctx = ctx
        self._endpoint = endpoint
        self._timeout_ms = timeout_ms
        self._socket = self._new_socket()

    def _new_socket(self) -> zmq.Socket:
        sock = self._ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(self._endpoint)
        return sock

    def send_request(self, msg: bytes, retries: int = 3) -> bytes:
        """Send a request and return the response bytes.

        Recreates the socket on timeout to recover from broken REQ state.

        Args:
            msg:     Encoded request bytes.
            retries: Number of attempts before raising TimeoutError.

        Returns:
            Raw response bytes from the server.

        Raises:
            TimeoutError: If all retry attempts time out.
        """
        for _ in range(retries):
            self._socket.send(msg)
            if self._socket.poll(self._timeout_ms):
                return self._socket.recv()
            # RCVTIMEO fired — REQ state machine is broken; recreate socket
            self._socket.close()
            self._socket = self._new_socket()
        raise TimeoutError(f"all {retries} attempts to {self._endpoint} timed out")

    def close(self) -> None:
        """Close the underlying ZMQ socket."""
        self._socket.close()


# ---------------------------------------------------------------------------
# PeerState
# ---------------------------------------------------------------------------


class PeerStatus(enum.Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"


@dataclass
class PeerState:
    config: PeerConfig
    zmq_channel: ZMQControlChannel
    remote_handle: RemoteMemHandle
    status: PeerStatus = PeerStatus.CONNECTED


# ---------------------------------------------------------------------------
# Server-side dedup cache entry
# ---------------------------------------------------------------------------


@dataclass
class _DeduplicatedEntry:
    found_positions: list[int]
    pages_per_found: list[list[int]]
    expires_at: float  # monotonic


# ---------------------------------------------------------------------------
# RemoteController ABC
# ---------------------------------------------------------------------------


class RemoteController(ABC):
    """Control-plane coordinator for remote L1 cache access.

    Manages peer registry, ZMQ channels, metadata exchange, lookup policy,
    dedup cache, and server-side L1Manager pin management.
    """

    @abstractmethod
    def lookup(
        self,
        request_id: str,
        keys: list[ObjectKey],
    ) -> LookupResult:
        """Look up keys across all registered peers and pin them for reading.

        Sends LookupRequest to all peers in parallel, applies lookup_policy
        to resolve keys found on multiple peers, and returns remote transfer
        info for each found key. Remote read locks are held until unlock()
        is called with the same request_id.

        Idempotent: retrying with the same request_id returns the cached
        response without re-pinning.

        Args:
            request_id: Stable ID for this lookup; must be unique per
                        concurrent lookup and reused on retry.
            keys:       Keys to look up.

        Returns:
            LookupResult with found keys and per-key RemoteKeyInfo.

        Raises:
            TimeoutError: If all peer LookupRequests time out.
        """

    @abstractmethod
    def unlock(
        self,
        request_id: str,
        found_keys: list[ObjectKey],
    ) -> None:
        """Release remote read locks acquired by lookup().

        Sends UnpinRequest to each peer that owns at least one found key.
        Must be called for ALL keys in LookupResult.found_keys, including
        keys whose local reserve_write or RDMA transfer failed.

        Args:
            request_id: Must match the request_id used in lookup().
            found_keys: All keys from LookupResult.found_keys.
        """

    @abstractmethod
    def register_peer(self, config: PeerConfig) -> None:
        """Add a new peer at runtime.

        Performs Init/MemReg ZMQ handshake synchronously, then calls
        RemoteTransferAdapter.connect_peer() to register the peer's memory.
        Safe to call concurrently with ongoing lookup/unlock operations.

        Args:
            config: Connection info for the new peer.

        Raises:
            ConnectionError: If ZMQ handshake with peer fails.
        """

    @abstractmethod
    def unregister_peer(self, peer_id: str) -> None:
        """Remove a peer and release its resources.

        Args:
            peer_id: ID of the peer to remove.

        Raises:
            KeyError: If peer_id is not registered.
        """

    @abstractmethod
    def start(self) -> None:
        """Bind ZMQ server socket and begin serving incoming requests.

        Must be called before any peer connects to this instance.
        """

    @abstractmethod
    def stop(self) -> None:
        """Stop serving incoming requests and release all resources."""


# ---------------------------------------------------------------------------
# ZMQRemoteController
# ---------------------------------------------------------------------------


class ZMQRemoteController(RemoteController):
    """Concrete RemoteController using ZMQ REQ/REP for control plane.

    Server side: single background thread on a ZMQ REP socket handles
    InitRequest, MemRegRequest, LookupRequest, and UnpinRequest.

    Client side: lookup() fans out LookupRequests to all CONNECTED peers
    concurrently via a thread pool.

    Args:
        config:     Controller configuration.
        l1_manager: Local L1Manager (for server-side reserve_read/finish_read).
        l1_mem_desc: L1 memory descriptor (provides align_bytes for page index
                     computation).
        transfer:   RemoteTransferAdapter (for connect_peer during handshake).
    """

    def __init__(
        self,
        config: RemoteControllerConfig,
        l1_manager: L1Manager,
        l1_mem_desc: L1MemoryDesc,
        transfer: RemoteTransferAdapter,
    ) -> None:
        self._config = config
        self._l1_manager = l1_manager
        self._align_bytes = l1_mem_desc.align_bytes
        self._transfer = transfer

        self._peers: dict[str, PeerState] = {}
        self._peers_lock = threading.Lock()

        self._dedup: dict[str, _DeduplicatedEntry] = {}
        self._dedup_lock = threading.Lock()

        self._zmq_ctx = zmq.Context(1)
        self._stop_event = threading.Event()

        # key_info stored per (request_id, peer_id) for unlock routing
        # maps request_id -> {peer_id -> list[ObjectKey]}
        self._pending_unlocks: dict[str, dict[str, list[ObjectKey]]] = {}
        self._pending_unlocks_lock = threading.Lock()

        max_workers = max(8, len(config.peers) * 2)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        self._server_thread = threading.Thread(
            target=self._server_loop, daemon=True, name="rc-server"
        )
        self._reconnect_thread = threading.Thread(
            target=self._reconnect_loop, daemon=True, name="rc-reconnect"
        )
        self._round_robin_counter = 0
        self._rr_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def lookup(
        self,
        request_id: str,
        keys: list[ObjectKey],
    ) -> LookupResult:
        """Fan-out LookupRequest to all CONNECTED peers and return aggregated result.

        Args:
            request_id: Stable ID; reused on retry for idempotency.
            keys:       Keys to look up.

        Returns:
            LookupResult with found keys and per-key RemoteKeyInfo.
        """
        with self._peers_lock:
            active_peers = [
                (pid, state)
                for pid, state in self._peers.items()
                if state.status == PeerStatus.CONNECTED
            ]

        if not active_peers:
            return LookupResult(found_keys=[], key_info={})

        wire_keys = [WireObjectKey(k.chunk_hash, k.model_name, k.kv_rank) for k in keys]
        req_bytes = msgspec.msgpack.encode(
            LookupRequest(request_id=request_id, keys=wire_keys)
        )

        futures: dict[str, tuple[Future, PeerState]] = {
            pid: (
                self._executor.submit(self._send_lookup, state, req_bytes),
                state,
            )
            for pid, state in active_peers
        }

        timeout_s = self._config.zmq_timeout_ms / 1000
        peer_responses: dict[str, tuple[list[int], list[list[int]]]] = {}

        for peer_id, (fut, _state) in futures.items():
            try:
                resp = fut.result(timeout=timeout_s + 1)
                peer_responses[peer_id] = resp
            except TimeoutError:
                logger.warning(
                    "Lookup timed out for peer %s; marking disconnected", peer_id
                )
                with self._peers_lock:
                    if peer_id in self._peers:
                        self._peers[peer_id].status = PeerStatus.DISCONNECTED
            except Exception:
                logger.exception("Lookup failed for peer %s", peer_id)

        return self._apply_policy(peer_responses, keys, active_peers, request_id)

    def unlock(
        self,
        request_id: str,
        found_keys: list[ObjectKey],
    ) -> None:
        """Send UnpinRequest to all peers that own at least one found key.

        Args:
            request_id: Must match the request_id used in lookup().
            found_keys: All keys from LookupResult.found_keys.
        """
        if not found_keys:
            return

        with self._pending_unlocks_lock:
            peer_map = self._pending_unlocks.pop(request_id, {})

        if not peer_map:
            return

        for peer_id, peer_keys in peer_map.items():
            keys_to_unpin = [k for k in peer_keys if k in set(found_keys)]
            if not keys_to_unpin:
                continue
            with self._peers_lock:
                state = self._peers.get(peer_id)
            if state is None or state.status != PeerStatus.CONNECTED:
                continue
            self._executor.submit(self._send_unpin, state, request_id, keys_to_unpin)

    def register_peer(self, config: PeerConfig) -> None:
        """Perform Init/MemReg handshake and add peer to registry.

        Args:
            config: Connection info for the new peer.

        Raises:
            ConnectionError: If ZMQ handshake fails.
        """
        endpoint = f"tcp://{config.host}:{config.port}"
        channel = ZMQControlChannel(
            self._zmq_ctx, endpoint, self._config.zmq_timeout_ms
        )
        try:
            local_metadata = self._transfer.get_local_metadata()
            init_req = msgspec.msgpack.encode(
                InitRequest(local_agent_metadata=local_metadata)
            )
            init_resp_bytes = channel.send_request(init_req)
            init_resp = msgspec.msgpack.decode(init_resp_bytes, type=ZMQMessage)
            if not isinstance(init_resp, InitResponse):
                raise ConnectionError(
                    f"Unexpected response to InitRequest: {type(init_resp)}"
                )

            local_xfer_descs = self._transfer.get_local_xfer_descs()
            mem_req = msgspec.msgpack.encode(
                MemRegRequest(local_xfer_descs=local_xfer_descs)
            )
            mem_resp_bytes = channel.send_request(mem_req)
            mem_resp = msgspec.msgpack.decode(mem_resp_bytes, type=ZMQMessage)
            if not isinstance(mem_resp, MemRegResponse):
                raise ConnectionError(
                    f"Unexpected response to MemRegRequest: {type(mem_resp)}"
                )

            remote_handle = self._transfer.connect_peer(
                init_resp.server_agent_metadata,
                mem_resp.server_xfer_descs,
            )

        except TimeoutError as err:
            channel.close()
            raise ConnectionError(
                f"Handshake with {config.peer_id} at {endpoint} timed out"
            ) from err

        with self._peers_lock:
            old = self._peers.get(config.peer_id)
            if old is not None:
                old.zmq_channel.close()
            self._peers[config.peer_id] = PeerState(
                config=config,
                zmq_channel=channel,
                remote_handle=remote_handle,
                status=PeerStatus.CONNECTED,
            )
        logger.info("Registered peer %s at %s", config.peer_id, endpoint)

    def unregister_peer(self, peer_id: str) -> None:
        """Remove a peer from the registry and close its ZMQ channel.

        Args:
            peer_id: ID of the peer to remove.

        Raises:
            KeyError: If peer_id is not registered.
        """
        with self._peers_lock:
            state = self._peers.pop(peer_id, None)
        if state is None:
            raise KeyError(f"Peer {peer_id!r} is not registered")
        state.zmq_channel.close()
        logger.info("Unregistered peer %s", peer_id)

    def start(self) -> None:
        """Bind ZMQ server socket and start background threads.

        Registers pre-configured peers after the server is running.
        """
        self._server_thread.start()
        self._reconnect_thread.start()
        for peer_config in self._config.peers:
            try:
                self.register_peer(peer_config)
            except ConnectionError:
                logger.warning(
                    "Could not connect to pre-configured peer %s; "
                    "reconnect thread will retry",
                    peer_config.peer_id,
                )
                with self._peers_lock:
                    self._peers[peer_config.peer_id] = PeerState(
                        config=peer_config,
                        zmq_channel=ZMQControlChannel(
                            self._zmq_ctx,
                            f"tcp://{peer_config.host}:{peer_config.port}",
                            self._config.zmq_timeout_ms,
                        ),
                        remote_handle=None,  # type: ignore[arg-type]
                        status=PeerStatus.DISCONNECTED,
                    )

    def stop(self) -> None:
        """Stop all background threads and release ZMQ resources."""
        self._stop_event.set()
        self._server_thread.join(timeout=5)
        self._reconnect_thread.join(timeout=5)
        self._executor.shutdown(wait=False)
        with self._peers_lock:
            for state in self._peers.values():
                state.zmq_channel.close()
        self._zmq_ctx.term()

    # ------------------------------------------------------------------
    # Server loop
    # ------------------------------------------------------------------

    def _server_loop(self) -> None:
        """Background thread: serve incoming ZMQ REP requests."""
        socket = self._zmq_ctx.socket(zmq.REP)
        socket.setsockopt(zmq.RCVTIMEO, 200)  # poll interval for stop check
        socket.setsockopt(zmq.LINGER, 0)
        endpoint = f"tcp://{self._config.serve_host}:{self._config.serve_port}"
        socket.bind(endpoint)
        logger.info("RemoteController server listening on %s", endpoint)

        while not self._stop_event.is_set():
            try:
                raw = socket.recv()
            except zmq.Again:
                continue  # RCVTIMEO — check stop_event
            except Exception:
                logger.exception("Error receiving ZMQ message")
                continue

            try:
                msg = msgspec.msgpack.decode(raw, type=ZMQMessage)
                response = self._dispatch(msg)
                socket.send(msgspec.msgpack.encode(response))
            except Exception:
                logger.exception("Error dispatching ZMQ message")
                # Must send something or REP socket gets stuck
                socket.send(msgspec.msgpack.encode(UnpinResponse()))

        socket.close()

    def _dispatch(self, msg: ZMQMessage) -> ZMQMessage:
        """Dispatch an incoming message to the appropriate handler.

        Args:
            msg: Decoded incoming message.

        Returns:
            Response message to send back to the client.
        """
        if isinstance(msg, InitRequest):
            return InitResponse(
                server_agent_metadata=self._transfer.get_local_metadata()
            )

        if isinstance(msg, MemRegRequest):
            return MemRegResponse(
                server_xfer_descs=self._transfer.get_local_xfer_descs()
            )

        if isinstance(msg, LookupRequest):
            return self._handle_lookup(msg)

        if isinstance(msg, UnpinRequest):
            return self._handle_unpin(msg)

        logger.warning("Unexpected message type: %s", type(msg))
        return UnpinResponse()

    def _handle_lookup(self, msg: LookupRequest) -> LookupResponse:
        """Handle an incoming LookupRequest.

        Checks the dedup cache before calling reserve_read to ensure
        idempotency across retries with the same request_id.

        Args:
            msg: Decoded LookupRequest.

        Returns:
            LookupResponse with found positions and page indices.
        """
        now = time.monotonic()
        with self._dedup_lock:
            # Lazy cleanup of expired entries
            expired = [
                rid for rid, entry in self._dedup.items() if entry.expires_at < now
            ]
            for rid in expired:
                self._dedup.pop(rid, None)

            cached = self._dedup.get(msg.request_id)
            if cached is not None:
                return LookupResponse(
                    found_positions=cached.found_positions,
                    pages_per_found=cached.pages_per_found,
                )

        keys = [ObjectKey(w.chunk_hash, w.model_name, w.kv_rank) for w in msg.keys]
        results = self._l1_manager.reserve_read(keys)

        found_positions: list[int] = []
        pages_per_found: list[list[int]] = []
        for i, key in enumerate(keys):
            entry = results.get(key)
            if entry is None:
                continue
            err, obj = entry
            if err != L1Error.SUCCESS or obj is None:
                continue
            found_positions.append(i)
            start_page = obj.meta.address // self._align_bytes
            num_pages = obj.meta.phy_size // self._align_bytes
            pages_per_found.append(list(range(start_page, start_page + num_pages)))

        expires_at = now + self._config.remote_pin_ttl_s + 10
        with self._dedup_lock:
            self._dedup[msg.request_id] = _DeduplicatedEntry(
                found_positions=found_positions,
                pages_per_found=pages_per_found,
                expires_at=expires_at,
            )

        return LookupResponse(
            found_positions=found_positions,
            pages_per_found=pages_per_found,
        )

    def _handle_unpin(self, msg: UnpinRequest) -> UnpinResponse:
        """Handle an incoming UnpinRequest.

        Args:
            msg: Decoded UnpinRequest.

        Returns:
            UnpinResponse ack.
        """
        keys = [
            ObjectKey(w.chunk_hash, w.model_name, w.kv_rank) for w in msg.found_keys
        ]
        if keys:
            self._l1_manager.finish_read(keys)
        with self._dedup_lock:
            self._dedup.pop(msg.request_id, None)
        return UnpinResponse()

    # ------------------------------------------------------------------
    # Reconnect loop
    # ------------------------------------------------------------------

    def _reconnect_loop(self) -> None:
        """Background thread: retry register_peer for disconnected peers."""
        while not self._stop_event.wait(timeout=self._config.reconnect_interval_s):
            with self._peers_lock:
                disconnected = [
                    state.config
                    for state in self._peers.values()
                    if state.status == PeerStatus.DISCONNECTED
                ]
            for config in disconnected:
                try:
                    self.register_peer(config)
                    logger.info("Reconnected to peer %s", config.peer_id)
                except ConnectionError:
                    pass  # will retry next interval

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def _send_lookup(
        self,
        state: PeerState,
        req_bytes: bytes,
    ) -> tuple[list[int], list[list[int]]]:
        """Send a LookupRequest to one peer and return parsed response.

        Args:
            state:     PeerState for the target peer.
            req_bytes: Pre-encoded LookupRequest bytes.

        Returns:
            Tuple of (found_positions, pages_per_found).

        Raises:
            TimeoutError: If the channel times out on all retries.
        """
        resp_bytes = state.zmq_channel.send_request(req_bytes)
        resp = msgspec.msgpack.decode(resp_bytes, type=ZMQMessage)
        if not isinstance(resp, LookupResponse):
            raise RuntimeError(f"Expected LookupResponse, got {type(resp)}")
        return resp.found_positions, resp.pages_per_found

    def _send_unpin(
        self,
        state: PeerState,
        request_id: str,
        keys: list[ObjectKey],
    ) -> None:
        """Send an UnpinRequest to one peer (best-effort, fire-and-forget).

        Args:
            state:      PeerState for the target peer.
            request_id: request_id matching the original lookup.
            keys:       Keys to unpin.
        """
        wire_keys = [WireObjectKey(k.chunk_hash, k.model_name, k.kv_rank) for k in keys]
        req_bytes = msgspec.msgpack.encode(
            UnpinRequest(request_id=request_id, found_keys=wire_keys)
        )
        try:
            state.zmq_channel.send_request(req_bytes)
        except TimeoutError:
            logger.warning(
                "UnpinRequest to peer %s timed out (TTL will expire)",
                state.config.peer_id,
            )

    def _apply_policy(
        self,
        peer_responses: dict[str, tuple[list[int], list[list[int]]]],
        keys: list[ObjectKey],
        active_peers: list[tuple[str, PeerState]],
        request_id: str,
    ) -> LookupResult:
        """Aggregate peer responses into a LookupResult using the configured policy.

        Args:
            peer_responses: Mapping peer_id -> (found_positions, pages_per_found).
            keys:           The original ordered keys list.
            active_peers:   Peers consulted in this lookup, in registration order.
            request_id:     Used to build the pending_unlocks map for unlock().

        Returns:
            LookupResult with winner assignment and per-key RemoteKeyInfo.
        """
        if self._config.lookup_policy == "round_robin":
            peer_order = self._rotated_peers(active_peers)
        else:
            peer_order = active_peers  # first_found: registration order

        # For each key position, find the first (in policy order) peer that has it
        key_assignment: dict[int, tuple[str, list[int]]] = {}
        for peer_id, _state in peer_order:
            resp = peer_responses.get(peer_id)
            if resp is None:
                continue
            found_positions, pages_per_found = resp
            for i, key_pos in enumerate(found_positions):
                if key_pos not in key_assignment:
                    key_assignment[key_pos] = (peer_id, pages_per_found[i])

        # Unpin losers asynchronously (keys found by a peer but not chosen)
        loser_map: dict[str, list[ObjectKey]] = {}
        for peer_id, (found_positions, _) in peer_responses.items():
            for key_pos in found_positions:
                winner_pid = key_assignment.get(key_pos, (None,))[0]
                if winner_pid != peer_id:
                    loser_map.setdefault(peer_id, []).append(keys[key_pos])

        for peer_id, loser_keys in loser_map.items():
            with self._peers_lock:
                state = self._peers.get(peer_id)
            if state is not None and state.status == PeerStatus.CONNECTED:
                self._executor.submit(self._send_unpin, state, request_id, loser_keys)

        # Build LookupResult and register pending_unlocks for unlock()
        found_keys: list[ObjectKey] = []
        key_info: dict[ObjectKey, RemoteKeyInfo] = {}
        winner_map: dict[str, list[ObjectKey]] = {}  # peer_id -> keys it owns

        for key_pos in sorted(key_assignment):
            peer_id, remote_pages = key_assignment[key_pos]
            key = keys[key_pos]
            with self._peers_lock:
                state = self._peers.get(peer_id)
            if state is None:
                continue
            found_keys.append(key)
            key_info[key] = RemoteKeyInfo(
                peer_id=peer_id,
                remote_handle=state.remote_handle,
                remote_pages=remote_pages,
            )
            winner_map.setdefault(peer_id, []).append(key)

        with self._pending_unlocks_lock:
            self._pending_unlocks[request_id] = winner_map

        return LookupResult(found_keys=found_keys, key_info=key_info)

    def _rotated_peers(
        self,
        active_peers: list[tuple[str, PeerState]],
    ) -> list[tuple[str, PeerState]]:
        """Return active_peers rotated by the round-robin counter.

        Args:
            active_peers: Peers in registration order.

        Returns:
            Same list rotated so that a different peer is first each call.
        """
        with self._rr_lock:
            offset = self._round_robin_counter % len(active_peers)
            self._round_robin_counter += 1
        return active_peers[offset:] + active_peers[:offset]
