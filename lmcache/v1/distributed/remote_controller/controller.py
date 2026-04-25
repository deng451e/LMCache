# SPDX-License-Identifier: Apache-2.0
"""RemoteController ABC and ZMQRemoteController concrete implementation.

ZMQRemoteController is responsible for:
  1. Server: ZMQ REP socket (Init/MemReg/Lookup) + PULL socket (Unpin).
  2. Peer lifecycle: register_peer() runs Init/MemReg handshake and calls
     RemoteIOAdapter.connect_peer(); reconnect thread polls for disconnected
     peers and retries.

All client-side I/O (lookup fan-out, RDMA fetch, handle caching) lives in
RemoteIOAdapter, not here.
"""

# Future
from __future__ import annotations

# Standard
from abc import ABC, abstractmethod
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
    ZMQRepMessage,
)
from lmcache.v1.distributed.remote_io.adapter import RemoteIOAdapter

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# ZMQControlChannel — per-peer REQ socket with Lazy Pirate recovery
# ---------------------------------------------------------------------------


class ZMQControlChannel:
    """Client-side ZMQ REQ socket for a single peer with timeout recovery.

    Implements the Lazy Pirate pattern: on timeout the socket is closed and
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
    status: PeerStatus = PeerStatus.CONNECTED


# ---------------------------------------------------------------------------
# Server-side dedup cache entry
# ---------------------------------------------------------------------------


@dataclass
class _DeduplicatedEntry:
    found_positions: list[int]
    byte_offsets: list[int]
    byte_sizes: list[int]
    expires_at: float  # monotonic


# ---------------------------------------------------------------------------
# RemoteController ABC
# ---------------------------------------------------------------------------


class RemoteController(ABC):
    """Control-plane coordinator: ZMQ server + peer lifecycle management.

    Does NOT own client-side lookup fan-out or RDMA — those live in
    RemoteIOAdapter.
    """

    @abstractmethod
    def register_peer(self, config: PeerConfig) -> None:
        """Add a new peer at runtime.

        Performs Init/MemReg ZMQ handshake synchronously, then calls
        RemoteIOAdapter.connect_peer() to set up lookup and RDMA channels.

        Args:
            config: Connection info for the new peer.

        Raises:
            ConnectionError: If ZMQ handshake with peer fails.
        """

    @abstractmethod
    def unregister_peer(self, peer_id: str) -> None:
        """Remove a peer and release its resources.

        Calls RemoteIOAdapter.disconnect_peer(), then removes the peer from
        the registry.

        Args:
            peer_id: ID of the peer to remove.

        Raises:
            KeyError: If peer_id is not registered.
        """

    @abstractmethod
    def start(self) -> None:
        """Bind ZMQ server sockets and begin serving incoming requests."""

    @abstractmethod
    def stop(self) -> None:
        """Stop serving and release all resources."""


# ---------------------------------------------------------------------------
# ZMQRemoteController
# ---------------------------------------------------------------------------


class ZMQRemoteController(RemoteController):
    """Concrete RemoteController using ZMQ REQ/REP + PUSH/PULL.

    Server side: a single background thread polls both a REP socket
    (Init/MemReg/Lookup) and a PULL socket (Unpin, fire-and-forget).

    Peer lifecycle: register_peer() runs the Init/MemReg handshake and calls
    _io.connect_peer(); a reconnect thread polls _io.get_disconnected_peers()
    and retries register_peer() on a fixed interval.

    Args:
        config:      Controller configuration.
        l1_manager:  Local L1Manager for server-side reserve_read/finish_read.
        l1_mem_desc: L1 memory descriptor providing align_bytes.
        io:          RemoteIOAdapter for peer connect/disconnect and metadata.
    """

    def __init__(
        self,
        config: RemoteControllerConfig,
        l1_manager: L1Manager,
        l1_mem_desc: L1MemoryDesc,
        io: RemoteIOAdapter,
    ) -> None:
        self._config = config
        self._l1_manager = l1_manager
        self._align_bytes = l1_mem_desc.align_bytes
        self._io = io

        self._peers: dict[str, PeerState] = {}
        self._peers_lock = threading.Lock()

        self._dedup: dict[str, _DeduplicatedEntry] = {}
        self._dedup_lock = threading.Lock()

        self._zmq_ctx = zmq.Context(1)
        self._stop_event = threading.Event()

        self._server_thread = threading.Thread(
            target=self._server_loop, daemon=True, name="rc-server"
        )
        self._reconnect_thread = threading.Thread(
            target=self._reconnect_loop, daemon=True, name="rc-reconnect"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def register_peer(self, config: PeerConfig) -> None:
        """Perform Init/MemReg handshake and add peer to registry.

        Args:
            config: Connection info for the new peer.

        Raises:
            ConnectionError: If ZMQ handshake fails.
        """
        endpoint = f"tcp://{config.host}:{config.port}"
        unpin_endpoint = f"tcp://{config.host}:{config.unpin_port}"
        channel = ZMQControlChannel(
            self._zmq_ctx, endpoint, self._config.zmq_timeout_ms
        )
        try:
            local_metadata = self._io.get_local_metadata()
            init_resp_bytes = channel.send_request(
                msgspec.msgpack.encode(InitRequest(local_agent_metadata=local_metadata))
            )
            init_resp = msgspec.msgpack.decode(init_resp_bytes, type=ZMQRepMessage)
            if not isinstance(init_resp, InitResponse):
                raise ConnectionError(
                    f"Unexpected response to InitRequest: {type(init_resp)}"
                )

            local_xfer_descs = self._io.get_local_xfer_descs()
            mem_resp_bytes = channel.send_request(
                msgspec.msgpack.encode(MemRegRequest(local_xfer_descs=local_xfer_descs))
            )
            mem_resp = msgspec.msgpack.decode(mem_resp_bytes, type=ZMQRepMessage)
            if not isinstance(mem_resp, MemRegResponse):
                raise ConnectionError(
                    f"Unexpected response to MemRegRequest: {type(mem_resp)}"
                )

            self._io.connect_peer(
                config.peer_id,
                endpoint,
                unpin_endpoint,
                init_resp.server_agent_metadata,
                mem_resp.server_xfer_descs,
            )

        except TimeoutError as err:
            channel.close()
            raise ConnectionError(
                f"Handshake with {config.peer_id} at {endpoint} timed out: {err}"
            ) from err
        except Exception as err:
            import traceback
            logger.error(
                "[Remote] register_peer handshake failed: peer=%s endpoint=%s unpin=%s error=%s: %s\n%s",
                config.peer_id, endpoint, unpin_endpoint,
                type(err).__name__, err, traceback.format_exc(),
            )
            try:
                channel.close()
            except Exception:
                pass
            raise ConnectionError(
                f"Handshake with {config.peer_id} at {endpoint} failed: {type(err).__name__}: {err}"
            ) from err

        with self._peers_lock:
            old = self._peers.get(config.peer_id)
            if old is not None:
                old.zmq_channel.close()
            self._peers[config.peer_id] = PeerState(
                config=config,
                zmq_channel=channel,
                status=PeerStatus.CONNECTED,
            )
        logger.info(
            "[Remote] peer connection established: %s at %s (unpin=%s)",
            config.peer_id,
            endpoint,
            unpin_endpoint,
        )

    def unregister_peer(self, peer_id: str) -> None:
        """Remove a peer from the registry and release its resources.

        Args:
            peer_id: ID of the peer to remove.

        Raises:
            KeyError: If peer_id is not registered.
        """
        with self._peers_lock:
            state = self._peers.pop(peer_id, None)
        if state is None:
            raise KeyError(f"Peer {peer_id!r} is not registered")
        self._io.disconnect_peer(peer_id)
        state.zmq_channel.close()
        logger.info("Unregistered peer %s", peer_id)

    def start(self) -> None:
        """Bind ZMQ server sockets and start background threads.

        Registers pre-configured peers after the server is running.
        """
        self._server_thread.start()
        self._reconnect_thread.start()
        for peer_config in self._config.peers:
            endpoint = f"tcp://{peer_config.host}:{peer_config.port}"
            unpin_endpoint = f"tcp://{peer_config.host}:{peer_config.unpin_port}"
            logger.info(
                "[Remote] attempting to register pre-configured peer %s at %s (unpin=%s)",
                peer_config.peer_id, endpoint, unpin_endpoint,
            )
            try:
                self.register_peer(peer_config)
            except ConnectionError as err:
                logger.warning(
                    "[Remote] FAILED to register peer %s at %s: %s; "
                    "reconnect thread will retry",
                    peer_config.peer_id, endpoint, err,
                )
                with self._peers_lock:
                    self._peers[peer_config.peer_id] = PeerState(
                        config=peer_config,
                        zmq_channel=ZMQControlChannel(
                            self._zmq_ctx, endpoint, self._config.zmq_timeout_ms,
                        ),
                        status=PeerStatus.DISCONNECTED,
                    )
            except Exception as err:
                import traceback
                logger.error(
                    "[Remote] UNEXPECTED error registering peer %s at %s: %s: %s\n%s",
                    peer_config.peer_id, endpoint,
                    type(err).__name__, err, traceback.format_exc(),
                )
                with self._peers_lock:
                    self._peers[peer_config.peer_id] = PeerState(
                        config=peer_config,
                        zmq_channel=ZMQControlChannel(
                            self._zmq_ctx, endpoint, self._config.zmq_timeout_ms,
                        ),
                        status=PeerStatus.DISCONNECTED,
                    )

    def stop(self) -> None:
        """Stop all background threads and release ZMQ resources."""
        self._stop_event.set()
        self._server_thread.join(timeout=5)
        self._reconnect_thread.join(timeout=5)
        with self._peers_lock:
            for state in self._peers.values():
                state.zmq_channel.close()
        self._zmq_ctx.term()

    # ------------------------------------------------------------------
    # Server loop
    # ------------------------------------------------------------------

    def _server_loop(self) -> None:
        """Background thread: poll REP + PULL sockets and dispatch messages."""
        rep_socket = self._zmq_ctx.socket(zmq.REP)
        rep_socket.setsockopt(zmq.LINGER, 0)
        rep_endpoint = f"tcp://{self._config.serve_host}:{self._config.serve_port}"
        rep_socket.bind(rep_endpoint)

        pull_socket = self._zmq_ctx.socket(zmq.PULL)
        pull_socket.setsockopt(zmq.LINGER, 0)
        unpin_endpoint = (
            f"tcp://{self._config.serve_host}:{self._config.serve_unpin_port}"
        )
        pull_socket.bind(unpin_endpoint)

        logger.info(
            "RemoteController server listening on %s (lookup) / %s (unpin)",
            rep_endpoint,
            unpin_endpoint,
        )

        poller = zmq.Poller()
        poller.register(rep_socket, zmq.POLLIN)
        poller.register(pull_socket, zmq.POLLIN)

        while not self._stop_event.is_set():
            try:
                ready = dict(poller.poll(timeout=200))
            except Exception:
                logger.exception("Error in ZMQ poll")
                continue

            if rep_socket in ready:
                try:
                    raw = rep_socket.recv()
                    msg = msgspec.msgpack.decode(raw, type=ZMQRepMessage)
                    response = self._dispatch_rep(msg)
                    rep_socket.send(msgspec.msgpack.encode(response))
                except Exception:
                    logger.exception("Error handling REP message")
                    rep_socket.send(msgspec.msgpack.encode(LookupResponse([], [])))

            if pull_socket in ready:
                try:
                    raw = pull_socket.recv()
                    msg = msgspec.msgpack.decode(raw, type=UnpinRequest)
                    self._handle_unpin(msg)
                except Exception:
                    logger.exception("Error handling PULL (unpin) message")

        rep_socket.close()
        pull_socket.close()

    def _dispatch_rep(self, msg: ZMQRepMessage) -> ZMQRepMessage:
        """Dispatch an incoming REP-socket message.

        Args:
            msg: Decoded incoming message.

        Returns:
            Response message to send back to the client.
        """
        if isinstance(msg, InitRequest):
            logger.info("[Remote] incoming peer handshake: InitRequest received")
            return InitResponse(server_agent_metadata=self._io.get_local_metadata())
        if isinstance(msg, MemRegRequest):
            logger.info(
                "[Remote] incoming peer handshake: "
                "MemRegRequest received — peer fully connected"
            )
            return MemRegResponse(server_xfer_descs=self._io.get_local_xfer_descs())
        if isinstance(msg, LookupRequest):
            return self._handle_lookup(msg)
        logger.warning("Unexpected REP message type: %s", type(msg))
        return LookupResponse(found_positions=[], byte_offsets=[], byte_sizes=[])

    def _handle_lookup(self, msg: LookupRequest) -> LookupResponse:
        """Handle an incoming LookupRequest with server-side dedup.

        Args:
            msg: Decoded LookupRequest.

        Returns:
            LookupResponse with found positions, byte offsets and sizes.
            Wire format ships only (offset, size) per key; page indices are
            reconstructed client-side using align_bytes (~3000x smaller than
            the prior pages_per_found list-of-lists encoding).
        """
        now = time.monotonic()
        with self._dedup_lock:
            expired = [rid for rid, e in self._dedup.items() if e.expires_at < now]
            for rid in expired:
                self._dedup.pop(rid, None)

            cached = self._dedup.get(msg.request_id)
            if cached is not None:
                return LookupResponse(
                    found_positions=cached.found_positions,
                    byte_offsets=cached.byte_offsets,
                    byte_sizes=cached.byte_sizes,
                )

        keys = [ObjectKey(w.chunk_hash, w.model_name, w.kv_rank) for w in msg.keys]
        results = self._l1_manager.reserve_read(keys)

        found_positions: list[int] = []
        byte_offsets: list[int] = []
        byte_sizes: list[int] = []
        for i, key in enumerate(keys):
            entry = results.get(key)
            if entry is None:
                continue
            err, obj = entry
            if err != L1Error.SUCCESS or obj is None:
                continue
            found_positions.append(i)
            byte_offsets.append(obj.meta.address)
            byte_sizes.append(obj.meta.phy_size)

        logger.info(
            "[Remote] lookup: found %d/%d keys in local L1 (request_id=%s)",
            len(found_positions),
            len(keys),
            msg.request_id,
        )

        expires_at = now + self._config.remote_pin_ttl_s + 10
        with self._dedup_lock:
            self._dedup[msg.request_id] = _DeduplicatedEntry(
                found_positions=found_positions,
                byte_offsets=byte_offsets,
                byte_sizes=byte_sizes,
                expires_at=expires_at,
            )

        return LookupResponse(
            found_positions=found_positions,
            byte_offsets=byte_offsets,
            byte_sizes=byte_sizes,
        )

    def _handle_unpin(self, msg: UnpinRequest) -> None:
        """Handle an incoming UnpinRequest from the PULL socket (no reply).

        Args:
            msg: Decoded UnpinRequest.
        """
        keys = [
            ObjectKey(w.chunk_hash, w.model_name, w.kv_rank) for w in msg.found_keys
        ]
        if keys:
            self._l1_manager.finish_read(keys)
            logger.info(
                "[Remote] unpin: released %d read locks (request_id=%s)",
                len(keys),
                msg.request_id,
            )
        with self._dedup_lock:
            self._dedup.pop(msg.request_id, None)

    # ------------------------------------------------------------------
    # Reconnect loop
    # ------------------------------------------------------------------

    def _reconnect_loop(self) -> None:
        """Background thread: retry register_peer for disconnected peers."""
        while not self._stop_event.wait(timeout=self._config.reconnect_interval_s):
            disconnected_ids = self._io.get_disconnected_peers()
            for peer_id in disconnected_ids:
                with self._peers_lock:
                    state = self._peers.get(peer_id)
                if state is None:
                    continue
                try:
                    self.register_peer(state.config)
                    logger.info("Reconnected to peer %s", peer_id)
                except ConnectionError:
                    pass  # will retry next interval
