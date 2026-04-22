# SPDX-License-Identifier: Apache-2.0
"""RemoteL2Adapter: L2AdapterInterface wrapper around RemoteIOAdapter.

Bridges the prefetch controller's L2AdapterInterface contract to
RemoteIOAdapter's lookup/fetch/unlock primitives. Store operations are
not supported (remote peers are read-only from this side).
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # First Party
    from lmcache.native_storage_ops import Bitmap

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface, L2TaskId
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.remote_io.adapter import RemoteIOAdapter
from lmcache.v1.memory_management import MemoryObj

# ---------------------------------------------------------------------------
# Sentinel config — used only for AdapterDescriptor; not parsed from JSON
# ---------------------------------------------------------------------------


@dataclass
class RemoteL2AdapterConfig(L2AdapterConfigBase):
    """Sentinel config for RemoteL2Adapter.

    Not intended for JSON-based instantiation. Used exclusively to populate
    AdapterDescriptor so the prefetch policy can identify this adapter.
    """

    @classmethod
    def from_dict(cls, d: dict) -> "RemoteL2AdapterConfig":
        """Return a default instance (no JSON fields are consumed).

        Args:
            d: Ignored.

        Returns:
            A new RemoteL2AdapterConfig instance.
        """
        return cls()

    @classmethod
    def help(cls) -> str:
        """Return help text for this adapter type.

        Returns:
            Description string.
        """
        return "Remote P2P/PD adapter; configured via --remote-mode, not --l2-adapter."


register_l2_adapter_type("remote", RemoteL2AdapterConfig)


class RemoteL2Adapter(L2AdapterInterface):
    """L2AdapterInterface backed by a RemoteIOAdapter.

    Delegates lookup, fetch, and unlock to the provided RemoteIOAdapter.
    Store operations raise NotImplementedError because remote peers serve
    as read sources, not write targets.

    The lookup_task_id returned by submit_lookup_and_lock_task is passed
    back by the prefetch controller to submit_load_task and submit_unlock,
    enabling the adapter to route handles and unlock requests to the
    correct peer even under round-robin lookup policy.

    Args:
        io: Configured RemoteIOAdapter (already registered local memory
            and connected to peers).
    """

    def __init__(self, io: RemoteIOAdapter) -> None:
        super().__init__()
        self._io = io

    # ------------------------------------------------------------------
    # Event fd interface
    # ------------------------------------------------------------------

    def get_store_event_fd(self) -> int:
        """Not supported; remote peers are read-only sources.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("RemoteL2Adapter does not support store operations")

    def get_lookup_and_lock_event_fd(self) -> int:
        """Return the eventfd signaled on lookup completion.

        Returns:
            File descriptor from the underlying RemoteIOAdapter.
        """
        return self._io.get_lookup_event_fd()

    def get_load_event_fd(self) -> int:
        """Return the eventfd signaled on fetch completion.

        Returns:
            File descriptor from the underlying RemoteIOAdapter.
        """
        return self._io.get_fetch_event_fd()

    # ------------------------------------------------------------------
    # Store interface (unsupported)
    # ------------------------------------------------------------------

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """Not supported; remote peers are read-only sources.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("RemoteL2Adapter does not support store operations")

    def pop_completed_store_tasks(self) -> dict[L2TaskId, bool]:
        """Not supported; remote peers are read-only sources.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("RemoteL2Adapter does not support store operations")

    # ------------------------------------------------------------------
    # Lookup and lock interface
    # ------------------------------------------------------------------

    def submit_lookup_and_lock_task(self, keys: list[ObjectKey]) -> L2TaskId:
        """Fan-out ZMQ LookupRequest to all connected peers.

        Non-blocking. The returned task ID must be passed back to
        submit_load_task and submit_unlock so they can resolve the
        correct per-peer handles.

        Args:
            keys: Keys to look up across all connected peers.

        Returns:
            Task ID for use with query_lookup_and_lock_result.
        """
        return self._io.submit_lookup_task(keys)

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> "Bitmap | None":
        """Non-blockingly query the result of a lookup task.

        One-shot: returns non-None exactly once per task_id.

        Args:
            task_id: From submit_lookup_and_lock_task.

        Returns:
            Bitmap of found keys, or None if not yet complete.
        """
        return self._io.query_lookup_result(task_id)

    # ------------------------------------------------------------------
    # Unlock interface
    # ------------------------------------------------------------------

    def submit_unlock(
        self,
        keys: list[ObjectKey],
        lookup_task_id: "L2TaskId | None" = None,
    ) -> None:
        """Send ZMQ UnpinRequest to the owning peer for each key.

        Must be called twice per request — once for keys not in the load
        plan (before fetch) and once for all plan keys (after fetch).

        Args:
            keys:           Keys whose remote read locks should be released.
            lookup_task_id: Task ID of the originating
                submit_lookup_and_lock_task call. Used to route UnpinRequest
                to the correct peer via the handle cache. Should always be
                provided; falls back to per-key routing dict when None.
        """
        self._io.submit_unlock(keys, lookup_task_id=lookup_task_id)

    # ------------------------------------------------------------------
    # Load interface
    # ------------------------------------------------------------------

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        lookup_task_id: "L2TaskId | None" = None,
    ) -> L2TaskId:
        """Issue RDMA READs for keys using handles cached by the prior lookup.

        Non-blocking.

        Args:
            keys:           Keys to fetch from remote peers.
            objects:        L1 write buffers (one per key, same order).
            lookup_task_id: Task ID of the originating
                submit_lookup_and_lock_task call. Used to route fetch to the
                correct peer via the handle cache. Should always be provided.

        Returns:
            Task ID for use with query_load_result.
        """
        return self._io.submit_fetch_task(keys, objects, lookup_task_id=lookup_task_id)

    def query_load_result(self, task_id: L2TaskId) -> "Bitmap | None":
        """Non-blockingly query the result of a fetch task.

        One-shot: returns non-None exactly once per task_id.

        Args:
            task_id: From submit_load_task.

        Returns:
            Bitmap of successfully loaded keys, or None if not yet complete.
        """
        return self._io.query_fetch_result(task_id)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Shut down the underlying RemoteIOAdapter."""
        self._io.close()
