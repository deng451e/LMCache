# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Optional, Union, cast
import asyncio
import threading
import time

# Third Party
import msgspec
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.rpc_utils import get_zmq_context, get_zmq_socket
from lmcache.v1.transfer_channel.abstract import BaseTransferChannel
from lmcache.v1.transfer_channel.transfer_utils import (
    InitSideMsgBase,
    InitSideRetMsgBase,
    SideMsg,
)

if TYPE_CHECKING:
    # Third Party
    from uccl import p2p

logger = init_logger(__name__)

############################################################
# Message types for UCCL init protocol
############################################################


class UcclMsgBase(msgspec.Struct, tag=True):
    pass


class UcclInitRequest(UcclMsgBase):
    local_id: str
    reply_init_url: Optional[str] = None


class UcclInitResponse(UcclMsgBase):
    server_metadata: bytes


class UcclPageExchangeRequest(UcclMsgBase):
    """Initiator sends its advertised page fifo_blobs and local_id."""

    local_id: str
    fifo_blobs: list[bytes]


class UcclPageExchangeResponse(UcclMsgBase):
    """Acceptor responds with its advertised page fifo_blobs."""

    fifo_blobs: list[bytes]


class UcclErrorResponse(UcclMsgBase):
    error: str


UcclMsg = Union[
    UcclInitRequest,
    UcclInitResponse,
    UcclPageExchangeRequest,
    UcclPageExchangeResponse,
    UcclErrorResponse,
]

DecodedInitMsg = Union[UcclMsg, SideMsg]


class UcclChannel(BaseTransferChannel):
    """
    One-sided RDMA transfer channel backed by uccl.p2p.

    Important UCCL rule:
      - the side that calls connect() owns the send channel group
      - the side that calls accept() does NOT initiate write/read on that conn

    Therefore we keep TWO logical directions per peer:
      - outbound connection: created by this side via connect()
        used for local batched_write()/batched_read()
      - inbound connection: accepted by this side via accept()
        used only so the remote peer can access our memory
    """

    def __init__(
        self,
        async_mode: bool = False,
        device: Optional[str] = None,
        **kwargs,
    ):
        assert "role" in kwargs
        assert "buffer_ptr" in kwargs
        assert "buffer_size" in kwargs
        assert "align_bytes" in kwargs
        assert "tp_rank" in kwargs
        assert "peer_init_url" in kwargs

        self.role = kwargs["role"]
        self.buffer_ptr: int = kwargs["buffer_ptr"]
        self.buffer_size: int = kwargs["buffer_size"]
        self.page_size: int = kwargs["align_bytes"]
        self.tp_rank: int = kwargs["tp_rank"]
        self.num_cpus: int = kwargs.get("num_cpus", 4)

        assert self.page_size > 0
        assert self.buffer_size > 0
        assert self.buffer_size % self.page_size == 0, (
            f"buffer_size={self.buffer_size} must be divisible by "
            f"page_size={self.page_size}"
        )

        self.uccl_wrapper = UcclAgentWrapper(
            buffer_ptr=self.buffer_ptr,
            buffer_size=self.buffer_size,
            page_size=self.page_size,
            tp_rank=self.tp_rank,
            device=device,
            num_cpus=self.num_cpus,
        )
        self.mr_id: int = self.uccl_wrapper.mr_id

        # Identity / reverse-connect metadata
        self.local_id: Optional[str] = kwargs.get("local_id")
        self.peer_init_urls: dict[str, str] = {}
        self.auto_bidirectional_connect: bool = kwargs.get(
            "auto_bidirectional_connect", False
        )

        # Outbound = this side called connect(); use these for local read/write.
        self.outbound_conn_ids: dict[str, int] = {}
        self.outbound_remote_page_offsets: dict[str, list[bytes]] = {}

        # Inbound = this side called accept(); remote peer uses these to reach us.
        self.inbound_conn_ids: dict[str, int] = {}
        self.inbound_remote_page_offsets: dict[str, list[bytes]] = {}

        # Cache local advertised page blobs per conn_id.
        # If UCCL advertisev() blobs remain valid for the connection lifetime,
        # this avoids redundant advertisement work.
        self._advertised_local_pages: dict[int, list[bytes]] = {}

        # Prevent duplicate outbound init races for the same peer.
        self._peer_init_locks: dict[str, threading.Lock] = {}
        self._peer_init_locks_guard = threading.Lock()
        self._async_peer_init_locks: dict[str, asyncio.Lock] = {}
        self._async_peer_init_locks_guard: Optional[asyncio.Lock] = None

        self.peer_lookup_url = kwargs.get("peer_lookup_url", None)

        self.running = True
        self.side_channels: list[zmq.Socket] = []
        self.running_threads: list[threading.Thread] = []

        self.async_mode = async_mode
        self.zmq_context = get_zmq_context(use_asyncio=async_mode)

        self.peer_init_url = kwargs["peer_init_url"]
        self.event_loop = kwargs.get("event_loop", None)

        self._init_side_channels()

    ############################################################
    # Initialization functions
    ############################################################

    def _get_or_create_peer_lock(self, peer_id: str) -> threading.Lock:
        with self._peer_init_locks_guard:
            lock = self._peer_init_locks.get(peer_id)
            if lock is None:
                lock = threading.Lock()
                self._peer_init_locks[peer_id] = lock
            return lock

    async def _get_or_create_async_peer_lock(self, peer_id: str) -> asyncio.Lock:
        if self._async_peer_init_locks_guard is None:
            self._async_peer_init_locks_guard = asyncio.Lock()

        async with self._async_peer_init_locks_guard:
            lock = self._async_peer_init_locks.get(peer_id)
            if lock is None:
                lock = asyncio.Lock()
                self._async_peer_init_locks[peer_id] = lock
            return lock

    def _get_local_metadata(self) -> bytes:
        """
        Fetch fresh metadata from the endpoint instead of caching it forever.
        This avoids sending stale or uninitialized metadata such as port=-1.
        """
        md = self.uccl_wrapper.ep.get_metadata()
        try:
            parsed = self.uccl_wrapper.ep.parse_metadata(md)
            logger.debug("Local UCCL metadata parsed as: %r", parsed)
        except Exception:
            logger.debug("Unable to parse local UCCL metadata for debug logging")
        return md

    def _parse_and_validate_peer_metadata(
        self,
        peer_id: str,
        server_metadata: bytes,
    ) -> tuple[str, int, int]:
        """
        Assumes parse_metadata returns (ip, port, gpu).
        If UCCL's API differs in your build, update only this helper.
        """
        parsed = self.uccl_wrapper.ep.parse_metadata(server_metadata)
        logger.info("Peer %s metadata parsed as: %r", peer_id, parsed)

        if not isinstance(parsed, tuple) or len(parsed) != 3:
            raise RuntimeError(
                f"Unexpected UCCL metadata format from peer {peer_id}: {parsed!r}"
            )

        ip, port, r_gpu = parsed

        if port is None or int(port) < 0:
            raise RuntimeError(
                f"Invalid UCCL metadata from peer {peer_id}: "
                f"ip={ip}, port={port}, gpu={r_gpu}"
            )

        return ip, int(port), int(r_gpu)

    def _advertise_local_pages(self, conn_id: int) -> list[bytes]:
        cached = self._advertised_local_pages.get(conn_id)
        if cached is not None:
            return cached

        uccl = self.uccl_wrapper
        num_pages = uccl.num_pages

        ok, fifo_blob_v = uccl.ep.advertisev(
            conn_id,
            uccl.full_mr_ids,
            uccl.local_page_offset,
            uccl.full_page_sizes,
            num_pages,
        )
        assert ok, "ep.advertisev() failed"

        self._advertised_local_pages[conn_id] = fifo_blob_v
        return fifo_blob_v

    def _store_outbound_connection(
        self,
        peer_id: str,
        conn_id: int,
        remote_fifo_blobs: list[bytes],
    ) -> None:
        self.outbound_conn_ids[peer_id] = conn_id
        self.outbound_remote_page_offsets[peer_id] = remote_fifo_blobs

    def _store_inbound_connection(
        self,
        peer_id: str,
        conn_id: int,
    ) -> None:
        self.inbound_conn_ids[peer_id] = conn_id

    def _get_outbound_conn_id(self, peer_id: str) -> int:
        conn_id = self.outbound_conn_ids.get(peer_id)
        if conn_id is None:
            raise RuntimeError(
                f"No outbound UCCL connection for peer_id={peer_id}. "
                "You must initialize an outbound connect()-side connection first."
            )
        return conn_id

    def _get_outbound_remote_blobs(self, peer_id: str) -> list[bytes]:
        blobs = self.outbound_remote_page_offsets.get(peer_id)
        if blobs is None:
            raise RuntimeError(
                f"No outbound remote page map for peer_id={peer_id}. "
                "Peer page exchange for outbound connection is incomplete."
            )
        return blobs

    def lazy_init_peer_connection(
        self,
        local_id: str,
        peer_id: str,
        peer_init_url: str,
        init_side_msg: Optional[InitSideMsgBase] = None,
    ) -> Optional[InitSideRetMsgBase]:
        """
        Create / refresh the outbound connection to peer_id.

        This side calls connect(), so this outbound connection is the one used
        for local writev/readv.
        """
        init_tmp_socket = get_zmq_socket(
            self.zmq_context,
            peer_init_url,
            "tcp",
            zmq.REQ,
            "connect",
        )

        try:
            # Round-trip 1: exchange endpoint metadata and give peer our init URL
            # so it can reverse-connect later if needed.
            init_tmp_socket.send(
                msgspec.msgpack.encode(
                    UcclInitRequest(
                        local_id=local_id,
                        reply_init_url=self.peer_init_url,
                    )
                )
            )
            resp_bytes = init_tmp_socket.recv()
            uccl_init_resp = msgspec.msgpack.decode(resp_bytes, type=UcclMsg)

            if isinstance(uccl_init_resp, UcclErrorResponse):
                raise RuntimeError(
                    f"Server returned error during init: {uccl_init_resp.error}"
                )
            assert isinstance(uccl_init_resp, UcclInitResponse)

            ip, port, r_gpu = self._parse_and_validate_peer_metadata(
                peer_id,
                uccl_init_resp.server_metadata,
            )

            logger.info(
                "Connecting outbound UCCL connection to peer %s at %s:%s gpu=%s",
                peer_id,
                ip,
                port,
                r_gpu,
            )
            ok, conn_id = self.uccl_wrapper.ep.connect(ip, r_gpu, remote_port=port)
            assert ok, "ep.connect() failed"

            local_pages = self._advertise_local_pages(conn_id)

            # Round-trip 2: exchange advertised page blobs for THIS outbound conn.
            init_tmp_socket.send(
                msgspec.msgpack.encode(
                    UcclPageExchangeRequest(
                        local_id=local_id,
                        fifo_blobs=local_pages,
                    )
                )
            )
            resp_bytes = init_tmp_socket.recv()
            page_resp = msgspec.msgpack.decode(resp_bytes, type=UcclMsg)

            if isinstance(page_resp, UcclErrorResponse):
                raise RuntimeError(f"Page exchange failed: {page_resp.error}")
            assert isinstance(page_resp, UcclPageExchangeResponse)

            self._store_outbound_connection(
                peer_id=peer_id,
                conn_id=conn_id,
                remote_fifo_blobs=page_resp.fifo_blobs,
            )

            init_ret_msg: Optional[InitSideRetMsgBase] = None
            if init_side_msg is not None:
                init_ret_msg = self.send_init_side_msg(init_tmp_socket, init_side_msg)

            return init_ret_msg
        finally:
            init_tmp_socket.close()

    async def async_lazy_init_peer_connection(
        self,
        local_id: str,
        peer_id: str,
        peer_init_url: str,
        init_side_msg: Optional[InitSideMsgBase] = None,
    ) -> Optional[InitSideRetMsgBase]:
        """
        Async version of outbound connect()-side initialization.
        """
        init_tmp_socket = get_zmq_socket(
            self.zmq_context,
            peer_init_url,
            "tcp",
            zmq.REQ,
            "connect",
        )

        try:
            logger.info("Sending init request to %s", peer_init_url)

            await init_tmp_socket.send(
                msgspec.msgpack.encode(
                    UcclInitRequest(
                        local_id=local_id,
                        reply_init_url=self.peer_init_url,
                    )
                )
            )
            resp_bytes = await init_tmp_socket.recv()
            uccl_init_resp = msgspec.msgpack.decode(resp_bytes, type=UcclMsg)

            logger.info("Received init response from %s", peer_init_url)

            if isinstance(uccl_init_resp, UcclErrorResponse):
                raise RuntimeError(
                    f"Server returned error during init: {uccl_init_resp.error}"
                )
            assert isinstance(uccl_init_resp, UcclInitResponse)

            ip, port, r_gpu = self._parse_and_validate_peer_metadata(
                peer_id,
                uccl_init_resp.server_metadata,
            )
            logger.info(
                "Connecting outbound UCCL connection to peer %s at %s:%s gpu=%s",
                peer_id,
                ip,
                port,
                r_gpu,
            )

            ok, conn_id = self.uccl_wrapper.ep.connect(ip, r_gpu, remote_port=port)
            assert ok, "ep.connect() failed"

            local_pages = self._advertise_local_pages(conn_id)

            await init_tmp_socket.send(
                msgspec.msgpack.encode(
                    UcclPageExchangeRequest(
                        local_id=local_id,
                        fifo_blobs=local_pages,
                    )
                )
            )
            resp_bytes = await init_tmp_socket.recv()
            page_resp = msgspec.msgpack.decode(resp_bytes, type=UcclMsg)
            if isinstance(page_resp, UcclErrorResponse):
                raise RuntimeError(f"Page exchange failed: {page_resp.error}")
            assert isinstance(page_resp, UcclPageExchangeResponse)

            self._store_outbound_connection(
                peer_id=peer_id,
                conn_id=conn_id,
                remote_fifo_blobs=page_resp.fifo_blobs,
            )

            init_ret_msg: Optional[InitSideRetMsgBase] = None
            if init_side_msg is not None:
                init_ret_msg = await self.async_send_init_side_msg(
                    init_tmp_socket, init_side_msg
                )

            return init_ret_msg
        finally:
            init_tmp_socket.close()

    def _ensure_outbound_peer_connection(
        self,
        peer_id: str,
        transfer_spec: dict,
    ) -> None:
        """
        Ensure the local side has an outbound connect()-side connection for peer_id.
        """
        if peer_id in self.outbound_conn_ids:
            return

        lock = self._get_or_create_peer_lock(peer_id)
        with lock:
            if peer_id in self.outbound_conn_ids:
                return

            local_id = transfer_spec.get("local_id", self.local_id)
            peer_init_url = transfer_spec.get(
                "peer_init_url", self.peer_init_urls.get(peer_id)
            )

            if local_id is None or peer_init_url is None:
                raise RuntimeError(
                    f"Missing outbound UCCL connection for peer_id={peer_id}. "
                    "Need local_id and peer_init_url to create a "
                    "connect()-side connection."
                )

            self.lazy_init_peer_connection(
                local_id=local_id,
                peer_id=peer_id,
                peer_init_url=peer_init_url,
            )

    async def _async_ensure_outbound_peer_connection(
        self,
        peer_id: str,
        transfer_spec: dict,
    ) -> None:
        if peer_id in self.outbound_conn_ids:
            return

        lock = await self._get_or_create_async_peer_lock(peer_id)
        async with lock:
            if peer_id in self.outbound_conn_ids:
                return

            local_id = transfer_spec.get("local_id", self.local_id)
            peer_init_url = transfer_spec.get(
                "peer_init_url", self.peer_init_urls.get(peer_id)
            )

            if local_id is None or peer_init_url is None:
                raise RuntimeError(
                    f"Missing outbound UCCL connection for peer_id={peer_id}. "
                    "Need local_id and peer_init_url to create a "
                    "connect()-side connection."
                )

            await self.async_lazy_init_peer_connection(
                local_id=local_id,
                peer_id=peer_id,
                peer_init_url=peer_init_url,
            )

    def remote_xfer_handler_exists(self, receiver_or_sender_id: str) -> bool:
        # "exists for local initiated ops" means outbound exists.
        return receiver_or_sender_id in self.outbound_conn_ids

    def _init_side_channels(self):
        if self.peer_init_url is None:
            return

        if self.async_mode:
            if self.event_loop is None:
                raise RuntimeError("event_loop must be provided when async_mode=True")
            fut = asyncio.run_coroutine_threadsafe(
                self._async_init_loop(), self.event_loop
            )
            # Keep a reference so exceptions are not silently dropped.
            self._async_init_future = fut
        else:
            init_thread = threading.Thread(target=self._init_loop, daemon=True)
            init_thread.start()
            self.running_threads.append(init_thread)

    def _handle_init_msg(
        self, req: Union[UcclMsg, InitSideMsgBase]
    ) -> Union[UcclMsg, InitSideRetMsgBase]:
        resp: Union[UcclMsg, InitSideRetMsgBase]
        if isinstance(req, UcclInitRequest):
            # Remember how to connect back to this peer later.
            if req.reply_init_url is not None:
                self.peer_init_urls[req.local_id] = req.reply_init_url

            # Reply with fresh endpoint metadata.
            resp = UcclInitResponse(server_metadata=self._get_local_metadata())

        elif isinstance(req, UcclPageExchangeRequest):
            # This request belongs to the peer-initiated inbound connection.
            conn_id = self.inbound_conn_ids.get(req.local_id)
            if conn_id is None:
                raise RuntimeError(
                    f"No inbound accepted connection for peer {req.local_id}. "
                    "accept() has not completed yet."
                )

            # Store peer-advertised blobs for the inbound connection separately.
            self.inbound_remote_page_offsets[req.local_id] = req.fifo_blobs

            local_pages = self._advertise_local_pages(conn_id)
            resp = UcclPageExchangeResponse(fifo_blobs=local_pages)

        elif isinstance(req, InitSideMsgBase):
            resp = self.handle_init_side_msg(req)
            logger.info("Replying P2P init side response")
        else:
            raise ValueError(f"Unsupported InitMsg type: {type(req)}")

        return resp

    def _post_init_accept(self, req: UcclInitRequest) -> None:
        logger.info("Accepting inbound UCCL connection for %s", req.local_id)
        ok, _r_ip, _r_gpu, conn_id = self.uccl_wrapper.ep.accept()
        assert ok, "ep.accept() failed"
        self._store_inbound_connection(req.local_id, conn_id)

        if (
            self.auto_bidirectional_connect
            and req.local_id not in self.outbound_conn_ids
            and req.local_id in self.peer_init_urls
            and self.local_id is not None
        ):
            try:
                self.lazy_init_peer_connection(
                    local_id=self.local_id,
                    peer_id=req.local_id,
                    peer_init_url=self.peer_init_urls[req.local_id],
                )
                logger.info(
                    "Created reverse outbound UCCL connection to peer %s",
                    req.local_id,
                )
            except Exception as e:
                logger.warning(
                    "Failed to auto-create reverse outbound connection to %s: %s",
                    req.local_id,
                    e,
                )

    async def _async_post_init_accept(self, req: UcclInitRequest) -> None:
        logger.info("Accepting inbound UCCL connection for %s", req.local_id)
        ok, _r_ip, _r_gpu, conn_id = self.uccl_wrapper.ep.accept()
        assert ok, "ep.accept() failed"
        self._store_inbound_connection(req.local_id, conn_id)

        if (
            self.auto_bidirectional_connect
            and req.local_id not in self.outbound_conn_ids
            and req.local_id in self.peer_init_urls
            and self.local_id is not None
        ):
            try:
                await self.async_lazy_init_peer_connection(
                    local_id=self.local_id,
                    peer_id=req.local_id,
                    peer_init_url=self.peer_init_urls[req.local_id],
                )
                logger.info(
                    "Created reverse outbound UCCL connection to peer %s",
                    req.local_id,
                )
            except Exception as e:
                logger.warning(
                    "Failed to auto-create reverse outbound connection to %s: %s",
                    req.local_id,
                    e,
                )

    def _init_loop(self):
        self.init_side_channel = get_zmq_socket(
            self.zmq_context,
            self.peer_init_url,
            "tcp",
            zmq.REP,
            "bind",
        )
        self.init_side_channel.setsockopt(zmq.LINGER, 0)
        self.side_channels.append(self.init_side_channel)

        while self.running:
            received = False
            try:
                req_bytes = self.init_side_channel.recv()
                received = True
                req = msgspec.msgpack.decode(req_bytes, type=DecodedInitMsg)

                # Important ordering:
                # 1. UcclInitRequest -> reply metadata, then accept inbound conn.
                # 2. UcclPageExchangeRequest -> use stored inbound conn_id.
                resp = self._handle_init_msg(req)
                self.init_side_channel.send(msgspec.msgpack.encode(resp))

                if isinstance(req, UcclInitRequest):
                    self._post_init_accept(req)

            except Exception as e:
                logger.error("Failed to process initialization loop: %s", str(e))
                if received:
                    try:
                        err_reply = msgspec.msgpack.encode(
                            UcclErrorResponse(error=str(e))
                        )
                        self.init_side_channel.send(err_reply, zmq.NOBLOCK)
                    except Exception:
                        pass
                if self.running:
                    time.sleep(0.01)

    async def _async_init_loop(self):
        self.init_side_channel = get_zmq_socket(
            self.zmq_context,
            self.peer_init_url,
            "tcp",
            zmq.REP,
            "bind",
        )
        self.init_side_channel.setsockopt(zmq.LINGER, 0)
        self.side_channels.append(self.init_side_channel)

        while self.running:
            received = False
            try:
                req_bytes = await self.init_side_channel.recv()
                received = True
                req = msgspec.msgpack.decode(req_bytes, type=DecodedInitMsg)

                resp = self._handle_init_msg(req)
                await self.init_side_channel.send(msgspec.msgpack.encode(resp))

                if isinstance(req, UcclInitRequest):
                    await self._async_post_init_accept(req)

            except Exception as e:
                logger.error("Failed to process initialization loop: %s", str(e))
                if received:
                    try:
                        err_reply = msgspec.msgpack.encode(
                            UcclErrorResponse(error=str(e))
                        )
                        await self.init_side_channel.send(err_reply, zmq.NOBLOCK)
                    except Exception:
                        pass
                if self.running:
                    await asyncio.sleep(0.01)

    ############################################################
    # Utility functions
    ############################################################

    def get_local_mem_indices(
        self, objects: Union[list[bytes], list[MemoryObj]]
    ) -> list[int]:
        if not objects:
            return []

        first = objects[0]
        if isinstance(first, MemoryObj):
            mem_objects = cast(list[MemoryObj], objects)
            return [mem_obj.meta.address for mem_obj in mem_objects]

        if isinstance(first, bytes):
            raise NotImplementedError(
                "Sending raw bytes is not supported in UCCL channel"
            )

        raise TypeError(f"Unsupported object type: {type(first)}")

    def _prepare_xfer(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        peer_id: str,
        remote_indexes: list[int],
    ) -> tuple[int, list[int], list[int], list[int], list[bytes], int]:
        local_indexes = self.get_local_mem_indices(objects)
        n = len(local_indexes)

        if n != len(remote_indexes):
            raise ValueError(
                f"Local object count {n} != remote index count {len(remote_indexes)}"
            )

        uccl = self.uccl_wrapper
        local_page_offset = uccl.local_page_offset

        local_blobs = [local_page_offset[i] for i in local_indexes]

        remote_page_blobs = self._get_outbound_remote_blobs(peer_id)
        remote_blobs = [remote_page_blobs[i] for i in remote_indexes]

        conn_id = self._get_outbound_conn_id(peer_id)
        local_mr_ids = uccl.slice_mr_ids(n)
        local_sizes = uccl.slice_page_sizes(n)

        return conn_id, local_mr_ids, local_sizes, local_blobs, remote_blobs, n

    async def _poll_transfer(
        self,
        transfer_id: int,
        initial_sleep_s: float = 0.0,
        max_sleep_s: float = 0.001,
    ) -> None:
        """
        Lighter async polling:
          - first try immediate polls
          - then back off up to max_sleep_s
        """
        ep = self.uccl_wrapper.ep

        sleep_s = initial_sleep_s
        spins = 0

        while True:
            ok, is_done = ep.poll_async(transfer_id)
            assert ok, "poll_async failed"
            if is_done:
                return

            spins += 1
            if spins < 8:
                await asyncio.sleep(0)
                continue

            sleep_s = max_sleep_s if sleep_s == 0.0 else min(max_sleep_s, sleep_s * 2)
            await asyncio.sleep(sleep_s)

    ############################################################
    # Send/Recv functions
    ############################################################

    def batched_send(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        raise NotImplementedError

    def batched_recv(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        raise NotImplementedError

    async def async_batched_send(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        raise NotImplementedError

    async def async_batched_recv(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        raise NotImplementedError

    ############################################################
    # Read/Write functions
    ############################################################

    def batched_write(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        assert transfer_spec is not None
        receiver_id = transfer_spec["receiver_id"]
        remote_indexes = transfer_spec["remote_indexes"]

        # Ensure THIS side owns a connect()-side outbound connection to receiver.
        self._ensure_outbound_peer_connection(receiver_id, transfer_spec)

        (
            conn_id,
            local_mr_ids,
            local_sizes,
            local_blobs,
            remote_blobs,
            n,
        ) = self._prepare_xfer(objects, receiver_id, remote_indexes)

        ok = self.uccl_wrapper.ep.writev(
            conn_id,
            local_mr_ids,
            local_blobs,
            local_sizes,
            remote_blobs,
            n,
        )
        assert ok, "ep.writev() failed"
        return n

    def batched_read(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        assert transfer_spec is not None
        sender_id = transfer_spec["sender_id"]
        remote_indexes = transfer_spec["remote_indexes"]

        # Ensure THIS side owns a connect()-side outbound connection to sender.
        self._ensure_outbound_peer_connection(sender_id, transfer_spec)

        (
            conn_id,
            local_mr_ids,
            local_sizes,
            local_blobs,
            remote_blobs,
            n,
        ) = self._prepare_xfer(buffers, sender_id, remote_indexes)

        ok = self.uccl_wrapper.ep.readv(
            conn_id,
            local_mr_ids,
            local_blobs,
            local_sizes,
            remote_blobs,
            n,
        )
        assert ok, "ep.readv() failed"
        return n

    async def async_batched_write(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        assert transfer_spec is not None
        receiver_id = transfer_spec["receiver_id"]
        remote_indexes = transfer_spec["remote_indexes"]

        await self._async_ensure_outbound_peer_connection(receiver_id, transfer_spec)

        (
            conn_id,
            local_mr_ids,
            local_sizes,
            local_blobs,
            remote_blobs,
            n,
        ) = self._prepare_xfer(objects, receiver_id, remote_indexes)

        ok, transfer_id = self.uccl_wrapper.ep.writev_async(
            conn_id,
            local_mr_ids,
            local_blobs,
            local_sizes,
            remote_blobs,
            n,
        )
        assert ok, "writev_async failed"

        await self._poll_transfer(transfer_id)
        return n

    async def async_batched_read(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        assert transfer_spec is not None
        sender_id = transfer_spec["sender_id"]
        remote_indexes = transfer_spec["remote_indexes"]

        await self._async_ensure_outbound_peer_connection(sender_id, transfer_spec)

        (
            conn_id,
            local_mr_ids,
            local_sizes,
            local_blobs,
            remote_blobs,
            n,
        ) = self._prepare_xfer(buffers, sender_id, remote_indexes)

        ok, transfer_id = self.uccl_wrapper.ep.readv_async(
            conn_id,
            local_mr_ids,
            local_blobs,
            local_sizes,
            remote_blobs,
            n,
        )
        assert ok, "readv_async failed"

        await self._poll_transfer(transfer_id)
        return n

    ############################################################
    # Cleanup
    ############################################################

    def close(self):
        self.running = False

        for channel in self.side_channels:
            try:
                channel.close(0)
            except Exception:
                pass

        for thread in self.running_threads:
            thread.join(timeout=1.0)

        if hasattr(self, "_async_init_future"):
            try:
                self._async_init_future.cancel()
            except Exception:
                pass

        try:
            self.zmq_context.term()
        except Exception:
            pass


class UcclAgentWrapper:
    ep: "p2p.Endpoint"
    mr_id: int
    local_page_offset: list[int]
    num_pages: int

    def __init__(
        self,
        buffer_ptr: int,
        buffer_size: int,
        page_size: int,
        tp_rank: int,
        device: Optional[str] = None,
        num_cpus: int = 4,
    ):
        try:
            # Third Party
            from uccl import p2p as uccl_p2p
        except ImportError as err:
            raise RuntimeError("UCCL p2p is not available") from err

        if device == "cpu":
            self.ep = uccl_p2p.Endpoint(num_cpus=num_cpus)
        else:
            self.ep = uccl_p2p.Endpoint(local_gpu_idx=tp_rank, num_cpus=num_cpus)

        num_pages = buffer_size // page_size
        self.num_pages = num_pages
        self.local_page_offset = [buffer_ptr + i * page_size for i in range(num_pages)]

        ok, self.mr_id = self.ep.reg(buffer_ptr, buffer_size)
        assert ok, "ep.reg() failed for pre-allocated transfer buffer"

        # Reused vectors to reduce hot-path list allocations.
        self.full_mr_ids = [self.mr_id] * num_pages
        self.full_page_sizes = [page_size] * num_pages

    @property
    def local_metadata(self) -> bytes:
        # Kept as a property for compatibility, but always fetch fresh metadata.
        return self.ep.get_metadata()

    def slice_mr_ids(self, n: int) -> list[int]:
        return self.full_mr_ids[:n]

    def slice_page_sizes(self, n: int) -> list[int]:
        return self.full_page_sizes[:n]
