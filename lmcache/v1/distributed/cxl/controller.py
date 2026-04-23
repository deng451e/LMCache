# SPDX-License-Identifier: Apache-2.0
"""CxlRemoteController: ZMQ control-plane server for the CXL shared memory tier.

Responsibilities:
  - Run a ZMQ REP socket for CxlInitRequest (handshake) and CxlLookupRequest
    (key lookup + pin on behalf of a remote peer).
  - Run a ZMQ PULL socket for CxlUnpinRequest (fire-and-forget unlock).
  - Deduplicate retried lookup requests via PinCache.
  - Sweep expired pin entries and release server-side CXL read locks.

The server does NOT own client-side state (fan-out, peer VA mapping).
That logic lives in CxlRemoteL2Adapter.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING
import threading
import time

# Third Party
import msgspec
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.cxl.protocol import (
    CxlInitRequest,
    CxlInitResponse,
    CxlLookupRequest,
    CxlLookupResponse,
    CxlRepMessage,
    CxlUnpinRequest,
)
from lmcache.v1.distributed.remote_controller.pin_cache import PinCache, PinEntry
from lmcache.v1.distributed.remote_controller.protocol import WireObjectKey

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.config import CxlConfig
    from lmcache.v1.distributed.cxl.adaptor import CxlAdaptor

logger = init_logger(__name__)

# Seconds until an unexpired pin entry is swept and the server-side lock released.
_PIN_TTL_S = 60
# Poll timeout in milliseconds for the server loop.
_SERVER_POLL_MS = 100
# Interval in seconds between sweep-expired runs.
_SWEEP_INTERVAL_S = 10


def _wire_to_key(w: WireObjectKey) -> ObjectKey:
    return ObjectKey(
        chunk_hash=w.chunk_hash, model_name=w.model_name, kv_rank=w.kv_rank
    )


def _key_to_wire(k: ObjectKey) -> WireObjectKey:
    return WireObjectKey(
        chunk_hash=k.chunk_hash, model_name=k.model_name, kv_rank=k.kv_rank
    )


class CxlRemoteController:
    """ZMQ server that serves CXL key lookups to remote peers.

    Registered as a listener on CxlAdaptor for its server-side pin/unpin
    callbacks.  Does not interact with L1Manager.

    Args:
        config: CxlConfig with DAX path, sub-region sizes, and serve ports.
        cxl_adaptor: The local CxlAdaptor instance owning the CXL sub-region.
    """

    def __init__(
        self,
        config: "CxlConfig",
        cxl_adaptor: "CxlAdaptor",
    ) -> None:
        self._config = config
        self._cxl_adaptor = cxl_adaptor
        self._pin_cache = PinCache(ttl_s=_PIN_TTL_S)

        self._zmq_ctx = zmq.Context()

        serve_port = config.serve_port
        serve_unpin_port = config.serve_unpin_port or (serve_port + 1)

        self._rep_socket: zmq.Socket = self._zmq_ctx.socket(zmq.REP)
        self._rep_socket.bind(f"tcp://*:{serve_port}")

        self._pull_socket: zmq.Socket = self._zmq_ctx.socket(zmq.PULL)
        self._pull_socket.bind(f"tcp://*:{serve_unpin_port}")

        self._enc = msgspec.msgpack.Encoder()
        self._rep_dec = msgspec.msgpack.Decoder(CxlRepMessage)
        self._unpin_dec = msgspec.msgpack.Decoder(CxlUnpinRequest)

        self._stop = threading.Event()
        self._server_thread = threading.Thread(
            target=self._server_loop,
            daemon=True,
            name="cxl-remote-server",
        )
        self._sweep_thread = threading.Thread(
            target=self._sweep_loop,
            daemon=True,
            name="cxl-remote-sweep",
        )

        logger.info(
            "CxlRemoteController: REP=%d PULL=%d",
            serve_port,
            serve_unpin_port,
        )

    def start(self) -> None:
        """Start server and sweep background threads."""
        self._server_thread.start()
        self._sweep_thread.start()

    def stop(self) -> None:
        """Signal threads to stop and wait for them to exit."""
        self._stop.set()
        self._server_thread.join(timeout=5)
        self._sweep_thread.join(timeout=5)
        self._rep_socket.close(linger=0)
        self._pull_socket.close(linger=0)
        self._zmq_ctx.term()

    # -----------------------------------------------------------------------
    # Server loop
    # -----------------------------------------------------------------------

    def _server_loop(self) -> None:
        poller = zmq.Poller()
        poller.register(self._rep_socket, zmq.POLLIN)
        poller.register(self._pull_socket, zmq.POLLIN)

        while not self._stop.is_set():
            ready = dict(poller.poll(timeout=_SERVER_POLL_MS))

            if self._rep_socket in ready:
                try:
                    raw = self._rep_socket.recv()
                    msg = self._rep_dec.decode(raw)
                    reply = self._handle_rep(msg)
                    self._rep_socket.send(self._enc.encode(reply))
                except Exception:
                    logger.exception("CxlRemoteController: error handling REP message")

            if self._pull_socket in ready:
                try:
                    raw = self._pull_socket.recv()
                    msg = self._unpin_dec.decode(raw)
                    self._handle_unpin(msg)
                except Exception:
                    logger.exception("CxlRemoteController: error handling PULL message")

    # -----------------------------------------------------------------------
    # Message handlers
    # -----------------------------------------------------------------------

    def _handle_rep(self, msg: CxlRepMessage) -> CxlInitResponse | CxlLookupResponse:
        """Dispatch a REP-socket message to the appropriate handler.

        Args:
            msg: Decoded CxlRepMessage (CxlInitRequest or CxlLookupRequest).

        Returns:
            The response struct to send back to the client.
        """
        if isinstance(msg, CxlInitRequest):
            return self._handle_init(msg)
        if isinstance(msg, CxlLookupRequest):
            return self._handle_lookup(msg)
        logger.warning(
            "CxlRemoteController: unhandled message type %s", type(msg).__name__
        )
        return CxlLookupResponse(found_positions=[], byte_offsets=[], byte_sizes=[])

    def _handle_init(self, msg: CxlInitRequest) -> CxlInitResponse:
        """Exchange sub-region metadata with a connecting peer.

        Args:
            msg: CxlInitRequest from the connecting peer.

        Returns:
            CxlInitResponse carrying this host's sub-region descriptor.
        """
        logger.info(
            "CxlRemoteController: peer handshake — peer sub-region offset=%d size=%d",
            msg.local_meta.subregion_offset,
            msg.local_meta.subregion_size,
        )
        return CxlInitResponse(server_meta=self._cxl_adaptor.get_subregion_meta())

    def _handle_lookup(self, msg: CxlLookupRequest) -> CxlLookupResponse:
        """Look up and pin keys in the local CXL index.

        Deduplicates retried requests via PinCache (same request_id returns
        the cached response without double-pinning).

        Args:
            msg: CxlLookupRequest from a remote peer.

        Returns:
            CxlLookupResponse with found positions, byte offsets, and sizes.
        """
        cached = self._pin_cache.get(msg.request_id)
        if cached is not None:
            return cached.response  # type: ignore[return-value]

        keys = [_wire_to_key(w) for w in msg.keys]
        result = self._cxl_adaptor.server_lookup_and_lock(keys)

        found_positions: list[int] = []
        byte_offsets: list[int] = []
        byte_sizes: list[int] = []
        found_wire_keys: list[WireObjectKey] = []

        for i, key in enumerate(keys):
            if key in result:
                offset, size = result[key]
                found_positions.append(i)
                byte_offsets.append(offset)
                byte_sizes.append(size)
                found_wire_keys.append(msg.keys[i])

        response = CxlLookupResponse(
            found_positions=found_positions,
            byte_offsets=byte_offsets,
            byte_sizes=byte_sizes,
        )
        self._pin_cache.put(
            PinEntry(
                request_id=msg.request_id,
                found_keys=found_wire_keys,
                response=response,
                expires_at=time.monotonic() + _PIN_TTL_S,
            )
        )
        return response

    def _handle_unpin(self, msg: CxlUnpinRequest) -> None:
        """Release server-side CXL read locks for a previously-looked-up request.

        Removes the dedup cache entry and decrements l2_lock_count for each
        found key.

        Args:
            msg: CxlUnpinRequest identifying the originating lookup request.
        """
        cached = self._pin_cache.pop(msg.request_id)
        if cached is None:
            return
        keys = [_wire_to_key(w) for w in msg.found_keys]
        self._cxl_adaptor.server_unpin(keys)

    # -----------------------------------------------------------------------
    # Sweep loop
    # -----------------------------------------------------------------------

    def _sweep_loop(self) -> None:
        """Periodically sweep expired pin entries and release server-side locks."""
        while not self._stop.is_set():
            self._stop.wait(timeout=_SWEEP_INTERVAL_S)
            try:
                expired = self._pin_cache.sweep_expired()
                for entry in expired:
                    keys = [_wire_to_key(w) for w in entry.found_keys]
                    self._cxl_adaptor.server_unpin(keys)
                    logger.debug(
                        "CxlRemoteController: swept expired pin %s (%d keys)",
                        entry.request_id,
                        len(keys),
                    )
            except Exception:
                logger.exception("CxlRemoteController: error in sweep loop")

    def report_status(self) -> dict:
        """Return a status dict for the CXL remote controller.

        Returns:
            Dict with is_healthy and thread liveness flags.
        """
        return {
            "is_healthy": self._server_thread.is_alive(),
            "type": "CxlRemoteController",
            "server_thread_alive": self._server_thread.is_alive(),
            "sweep_thread_alive": self._sweep_thread.is_alive(),
            "serve_port": self._config.serve_port,
            "serve_unpin_port": self._config.serve_unpin_port
            or (self._config.serve_port + 1),
        }
