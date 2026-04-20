# SPDX-License-Identifier: Apache-2.0
"""NixlTransferBackend: UCX/RDMA-based RemoteTransferAdapter."""

# Standard
from typing import TYPE_CHECKING
import uuid

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.remote_transfer.adapter import (
    LocalMemHandle,
    RemoteMemHandle,
    RemoteTransferAdapter,
    TransferHandle,
    TransferStatus,
)

if TYPE_CHECKING:
    # Third Party
    from nixl._api import nixl_agent as NixlAgent

logger = init_logger(__name__)

_DONE_STATE = "DONE"
_ERROR_STATE = "ERR"


class NixlLocalMemHandle(LocalMemHandle):
    """LocalMemHandle backed by a NIXL prepped dlist for the L1 buffer."""

    def __init__(self, dlist: object, reg_descs: object) -> None:
        self._dlist = dlist
        self._reg_descs = reg_descs


class NixlRemoteMemHandle(RemoteMemHandle):
    """RemoteMemHandle backed by a NIXL prepped dlist for a peer's L1 buffer."""

    def __init__(self, dlist: object, remote_agent_name: bytes) -> None:
        self._dlist = dlist
        self._remote_agent_name = remote_agent_name


class NixlTransferHandle(TransferHandle):
    """TransferHandle wrapping a NIXL xfer handle."""

    def __init__(self, xfer_handle: object) -> None:
        self._xfer_handle = xfer_handle


class NixlTransferBackend(RemoteTransferAdapter):
    """RemoteTransferAdapter backed by NIXL UCX for RDMA READ/WRITE.

    Wraps NixlAgent with UCX backend. register_local_memory() must be called
    once before any transfer or peer connection operation.

    Args:
        align_bytes: Page size of the L1 memory buffer.
    """

    def __init__(self, align_bytes: int) -> None:
        self._align_bytes = align_bytes
        self._agent: "NixlAgent | None" = None
        self._local_xfer_descs: object = None
        self._local_reg_descs: object = None

    def _require_agent(self) -> "NixlAgent":
        if self._agent is None:
            raise RuntimeError(
                "NixlTransferBackend: register_local_memory() must be called first"
            )
        return self._agent

    def get_local_metadata(self) -> bytes:
        """Return serialised NIXL agent descriptor for the Init handshake.

        Returns:
            Opaque bytes representing this agent's identity.
        """
        return self._require_agent().get_agent_metadata()

    def get_local_xfer_descs(self) -> bytes:
        """Return serialised local transfer descriptors for the MemReg handshake.

        Returns:
            Opaque bytes representing this agent's L1 memory descriptors.
        """
        return self._require_agent().get_serialized_descs(self._local_xfer_descs)

    def register_local_memory(
        self,
        ptr: int,
        size: int,
        device: str,
    ) -> LocalMemHandle:
        """Register the L1 memory buffer with the NIXL UCX backend.

        Creates a NixlAgent with UCX backend, registers the buffer, and
        prepares per-page transfer descriptors. Must be called once at startup.

        Args:
            ptr:    Base address of the L1 buffer.
            size:   Buffer size in bytes.
            device: "cpu" or "cuda".

        Returns:
            NixlLocalMemHandle wrapping the prepared dlist.

        Raises:
            RuntimeError: If NIXL is not installed or registration fails.
        """
        try:
            # Third Party
            from nixl._api import nixl_agent as NixlAgent
            from nixl._api import nixl_agent_config as NixlAgentConfig
        except ImportError as err:
            raise RuntimeError("NIXL is not available") from err

        agent_name = f"NixlRTA_{uuid.uuid4().hex[:8]}"
        agent = NixlAgent(agent_name, NixlAgentConfig(backends=[]))
        agent.create_backend("UCX", {})

        mem_type = "DRAM" if device == "cpu" else "VRAM"
        reg_list = [(ptr, size, 0, "")]
        reg_descs = agent.register_memory(reg_list, mem_type=mem_type)

        xfer_desc = [
            (base_addr, self._align_bytes, 0)
            for base_addr in range(ptr, ptr + size, self._align_bytes)
        ]
        xfer_descs = agent.get_xfer_descs(xfer_desc, mem_type=mem_type)
        local_dlist = agent.prep_xfer_dlist("", xfer_descs, mem_type=mem_type)

        self._agent = agent
        self._local_xfer_descs = xfer_descs
        self._local_reg_descs = reg_descs

        return NixlLocalMemHandle(dlist=local_dlist, reg_descs=reg_descs)

    def connect_peer(
        self,
        peer_metadata: bytes,
        peer_xfer_descs: bytes,
    ) -> RemoteMemHandle:
        """Register a remote peer's L1 memory for RDMA access.

        Args:
            peer_metadata:   Serialised agent descriptor from the peer's InitResponse.
            peer_xfer_descs: Serialised transfer descriptors from the peer's
                MemRegResponse.

        Returns:
            NixlRemoteMemHandle wrapping the peer's prepared dlist.

        Raises:
            RuntimeError: If agent registration or descriptor exchange fails.
        """
        agent = self._require_agent()
        remote_agent_name = agent.add_remote_agent(peer_metadata)
        remote_xfer_dlist = agent.deserialize_descs(peer_xfer_descs)
        remote_dlist = agent.prep_xfer_dlist(remote_agent_name, remote_xfer_dlist)
        return NixlRemoteMemHandle(
            dlist=remote_dlist, remote_agent_name=remote_agent_name
        )

    def read(
        self,
        local_handle: LocalMemHandle,
        local_pages: list[int],
        remote_handle: RemoteMemHandle,
        remote_pages: list[int],
    ) -> TransferHandle:
        """Submit a non-blocking RDMA READ: remote pages -> local pages.

        Args:
            local_handle:  Registered local buffer (write destination).
            local_pages:   Page indices within local dlist to write into.
            remote_handle: Registered remote buffer (read source).
            remote_pages:  Page indices within remote dlist to read from.

        Returns:
            NixlTransferHandle to poll for completion.

        Raises:
            ValueError: If local_pages and remote_pages lengths differ.
        """
        if len(local_pages) != len(remote_pages):
            raise ValueError(
                f"local_pages length {len(local_pages)} != "
                f"remote_pages length {len(remote_pages)}"
            )
        if not isinstance(local_handle, NixlLocalMemHandle):
            raise ValueError("local_handle must be a NixlLocalMemHandle")
        if not isinstance(remote_handle, NixlRemoteMemHandle):
            raise ValueError("remote_handle must be a NixlRemoteMemHandle")

        agent = self._require_agent()
        xfer_handle = agent.make_prepped_xfer(
            "READ",
            local_handle._dlist,
            local_pages,
            remote_handle._dlist,
            remote_pages,
        )
        agent.transfer(xfer_handle)
        return NixlTransferHandle(xfer_handle)

    def write(
        self,
        local_handle: LocalMemHandle,
        local_pages: list[int],
        remote_handle: RemoteMemHandle,
        remote_pages: list[int],
    ) -> TransferHandle:
        """Submit a non-blocking RDMA WRITE: local pages -> remote pages.

        Args:
            local_handle:  Registered local buffer (read source).
            local_pages:   Page indices within local dlist to read from.
            remote_handle: Registered remote buffer (write destination).
            remote_pages:  Page indices within remote dlist to write into.

        Returns:
            NixlTransferHandle to poll for completion.

        Raises:
            ValueError: If local_pages and remote_pages lengths differ.
        """
        if len(local_pages) != len(remote_pages):
            raise ValueError(
                f"local_pages length {len(local_pages)} != "
                f"remote_pages length {len(remote_pages)}"
            )
        if not isinstance(local_handle, NixlLocalMemHandle):
            raise ValueError("local_handle must be a NixlLocalMemHandle")
        if not isinstance(remote_handle, NixlRemoteMemHandle):
            raise ValueError("remote_handle must be a NixlRemoteMemHandle")

        agent = self._require_agent()
        xfer_handle = agent.make_prepped_xfer(
            "WRITE",
            local_handle._dlist,
            local_pages,
            remote_handle._dlist,
            remote_pages,
        )
        agent.transfer(xfer_handle)
        return NixlTransferHandle(xfer_handle)

    def poll(self, handle: TransferHandle) -> TransferStatus:
        """Check transfer status without blocking.

        Args:
            handle: Handle returned by read() or write().

        Returns:
            Current TransferStatus of the operation.
        """
        if not isinstance(handle, NixlTransferHandle):
            raise ValueError("handle must be a NixlTransferHandle")
        state = self._require_agent().check_xfer_state(handle._xfer_handle)
        if state == _DONE_STATE:
            return TransferStatus.DONE
        if state == _ERROR_STATE:
            return TransferStatus.ERROR
        return TransferStatus.IN_PROGRESS

    def release(self, handle: TransferHandle) -> None:
        """Release resources for a completed or failed transfer.

        Must be called exactly once after poll() returns DONE or ERROR.

        Args:
            handle: Handle to release.
        """
        if not isinstance(handle, NixlTransferHandle):
            raise ValueError("handle must be a NixlTransferHandle")
        self._require_agent().release_xfer_handle(handle._xfer_handle)

    def close(self) -> None:
        """Shut down the NIXL agent and release all registered memory."""
        if self._agent is None:
            return
        if self._local_reg_descs is not None:
            try:
                self._agent.deregister_memory(self._local_reg_descs)
            except Exception:
                logger.warning("Failed to deregister local memory during close")
        self._agent = None
