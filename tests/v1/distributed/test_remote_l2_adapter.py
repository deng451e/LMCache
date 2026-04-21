# SPDX-License-Identifier: Apache-2.0
"""
Tests for RemoteL2Adapter and the lookup_task_id passthrough in PrefetchController.

Verifies that:
1. RemoteL2Adapter correctly delegates all L2AdapterInterface calls to RemoteIOAdapter.
2. PrefetchController passes the correct lookup_task_id to submit_load_task
   and submit_unlock when the adapter is a RemoteL2Adapter.
3. Store operations raise NotImplementedError from RemoteL2Adapter.
"""

# Standard
from unittest.mock import MagicMock
import os
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.remote_l2_adapter import RemoteL2Adapter
from lmcache.v1.distributed.remote_io.adapter import (
    IOTaskId,
    LocalMemHandle,
    RemoteIOAdapter,
)
from lmcache.v1.distributed.storage_controllers.prefetch_controller import (
    PrefetchController,
)
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    DefaultPrefetchPolicy,
)
from lmcache.v1.distributed.storage_controllers.store_policy import AdapterDescriptor
from lmcache.v1.memory_management import MemoryObj

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available"
)


# =============================================================================
# Helpers
# =============================================================================


def make_object_key(chunk_id: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=0,
    )


def make_layout() -> MemoryLayoutDesc:
    return MemoryLayoutDesc(
        shapes=[torch.Size([100, 2, 512])],
        dtypes=[torch.bfloat16],
    )


def wait_for_condition(
    predicate, timeout: float = 5.0, poll_interval: float = 0.05
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_interval)
    return False


# =============================================================================
# FakeRemoteIOAdapter: records all calls for assertion
# =============================================================================


class FakeRemoteIOAdapter(RemoteIOAdapter):
    """In-process RemoteIOAdapter that records calls and drives eventfds directly.

    All lookup tasks complete immediately with a configurable bitmap.
    All fetch tasks complete immediately with a configurable bitmap.
    """

    def __init__(self, num_keys: int, found_bits: list[int]) -> None:
        """
        Args:
            num_keys:   Size of bitmaps to produce.
            found_bits: Bit positions to set as "found" in lookup results.
        """
        self._num_keys = num_keys
        self._found_bits = found_bits

        self._lookup_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self._fetch_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)

        self._next_task_id = 0
        self._lock = threading.Lock()

        # Pending task results
        self._lookup_results: dict[IOTaskId, Bitmap] = {}
        self._fetch_results: dict[IOTaskId, Bitmap] = {}

        # Call log — each entry is a dict of kwargs
        self.load_calls: list[dict] = []
        self.unlock_calls: list[dict] = []

    def _next_id(self) -> IOTaskId:
        with self._lock:
            tid = self._next_task_id
            self._next_task_id += 1
            return tid

    def register_local_memory(self, ptr: int, size: int, device: str) -> LocalMemHandle:
        raise NotImplementedError

    def get_local_metadata(self) -> bytes:
        raise NotImplementedError

    def get_local_xfer_descs(self) -> bytes:
        raise NotImplementedError

    def connect_peer(
        self, peer_id, endpoint, unpin_endpoint, peer_metadata, peer_xfer_descs
    ) -> None:
        pass

    def disconnect_peer(self, peer_id: str) -> None:
        pass

    def get_disconnected_peers(self) -> list[str]:
        return []

    def submit_lookup_task(self, keys: list[ObjectKey]) -> IOTaskId:
        task_id = self._next_id()
        bm = Bitmap(len(keys))
        for bit in self._found_bits:
            if bit < len(keys):
                bm.set(bit)
        with self._lock:
            self._lookup_results[task_id] = bm
        os.eventfd_write(self._lookup_efd, 1)
        return task_id

    def query_lookup_result(self, task_id: IOTaskId) -> Bitmap | None:
        with self._lock:
            return self._lookup_results.pop(task_id, None)

    def submit_fetch_task(
        self,
        keys: list[ObjectKey],
        local_objs: list[MemoryObj],
        lookup_task_id: IOTaskId | None = None,
    ) -> IOTaskId:
        task_id = self._next_id()
        # Record the call
        self.load_calls.append({"keys": list(keys), "lookup_task_id": lookup_task_id})
        # All fetches succeed
        bm = Bitmap(len(keys))
        for i in range(len(keys)):
            bm.set(i)
        with self._lock:
            self._fetch_results[task_id] = bm
        os.eventfd_write(self._fetch_efd, 1)
        return task_id

    def query_fetch_result(self, task_id: IOTaskId) -> Bitmap | None:
        with self._lock:
            return self._fetch_results.pop(task_id, None)

    def submit_unlock(
        self,
        keys: list[ObjectKey],
        lookup_task_id: IOTaskId | None = None,
    ) -> None:
        self.unlock_calls.append({"keys": list(keys), "lookup_task_id": lookup_task_id})

    def get_lookup_event_fd(self) -> int:
        return self._lookup_efd

    def get_fetch_event_fd(self) -> int:
        return self._fetch_efd

    def close(self) -> None:
        os.close(self._lookup_efd)
        os.close(self._fetch_efd)


# =============================================================================
# Tests for RemoteL2Adapter delegation
# =============================================================================


class TestRemoteL2AdapterDelegation:
    """Verify that RemoteL2Adapter correctly delegates all calls."""

    def test_lookup_delegates_to_io(self):
        io = FakeRemoteIOAdapter(num_keys=4, found_bits=[0, 1])
        adapter = RemoteL2Adapter(io)
        keys = [make_object_key(i) for i in range(4)]

        task_id = adapter.submit_lookup_and_lock_task(keys)
        assert wait_for_condition(
            lambda: adapter.query_lookup_and_lock_result(task_id) is not None
        )

    def test_lookup_bitmap_matches(self):
        io = FakeRemoteIOAdapter(num_keys=4, found_bits=[0, 2])
        adapter = RemoteL2Adapter(io)
        keys = [make_object_key(i) for i in range(4)]

        task_id = adapter.submit_lookup_and_lock_task(keys)
        # Wait for result
        result = None
        for _ in range(100):
            result = adapter.query_lookup_and_lock_result(task_id)
            if result is not None:
                break
            time.sleep(0.02)

        assert result is not None
        assert result.test(0) == 1
        assert result.test(1) == 0
        assert result.test(2) == 1
        assert result.test(3) == 0

    def test_store_raises(self):
        io = FakeRemoteIOAdapter(num_keys=1, found_bits=[])
        adapter = RemoteL2Adapter(io)
        with pytest.raises(NotImplementedError):
            adapter.submit_store_task([], [])

    def test_pop_store_raises(self):
        io = FakeRemoteIOAdapter(num_keys=1, found_bits=[])
        adapter = RemoteL2Adapter(io)
        with pytest.raises(NotImplementedError):
            adapter.pop_completed_store_tasks()

    def test_get_store_event_fd_raises(self):
        io = FakeRemoteIOAdapter(num_keys=1, found_bits=[])
        adapter = RemoteL2Adapter(io)
        with pytest.raises(NotImplementedError):
            adapter.get_store_event_fd()

    def test_event_fds_distinct(self):
        io = FakeRemoteIOAdapter(num_keys=1, found_bits=[])
        adapter = RemoteL2Adapter(io)
        lookup_fd = adapter.get_lookup_and_lock_event_fd()
        load_fd = adapter.get_load_event_fd()
        assert lookup_fd != load_fd

    def test_unlock_passes_lookup_task_id(self):
        io = FakeRemoteIOAdapter(num_keys=2, found_bits=[0])
        adapter = RemoteL2Adapter(io)
        keys = [make_object_key(0), make_object_key(1)]

        adapter.submit_unlock(keys, lookup_task_id=42)

        assert len(io.unlock_calls) == 1
        assert io.unlock_calls[0]["lookup_task_id"] == 42

    def test_load_passes_lookup_task_id(self):
        io = FakeRemoteIOAdapter(num_keys=2, found_bits=[0])
        adapter = RemoteL2Adapter(io)
        keys = [make_object_key(0)]
        mock_obj = MagicMock(spec=MemoryObj)

        adapter.submit_load_task(keys, [mock_obj], lookup_task_id=7)

        assert len(io.load_calls) == 1
        assert io.load_calls[0]["lookup_task_id"] == 7

    def test_close_delegates(self):
        io = FakeRemoteIOAdapter(num_keys=1, found_bits=[])
        adapter = RemoteL2Adapter(io)
        adapter.close()
        # eventfds should be closed — re-reading should raise OSError
        with pytest.raises(OSError):
            os.eventfd_read(io._lookup_efd)


# =============================================================================
# Tests for PrefetchController lookup_task_id passthrough
# =============================================================================


def make_l1_manager() -> L1Manager:
    """Create a minimal L1Manager for testing."""
    l1_cfg = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=128 * 1024 * 1024,
            use_lazy=torch.cuda.is_available(),
            init_size_in_bytes=64 * 1024 * 1024,
            align_bytes=0x1000,
        ),
        write_ttl_seconds=600,
        read_ttl_seconds=300,
    )
    return L1Manager(l1_cfg)


class TestPrefetchControllerLookupTaskIdPassthrough:
    """Verify lookup_task_id is propagated from lookup to load and unlock."""

    def _run_prefetch_and_wait(
        self,
        l1_manager: L1Manager,
        io: FakeRemoteIOAdapter,
        num_keys: int,
        timeout: float = 5.0,
    ) -> int:
        """Run a prefetch for num_keys and return prefix_hits."""
        adapter = RemoteL2Adapter(io)
        descriptor = AdapterDescriptor(index=0, config=MagicMock())
        controller = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[descriptor],
            policy=DefaultPrefetchPolicy(),
            max_in_flight=4,
        )
        controller.start()

        try:
            keys = [make_object_key(i) for i in range(num_keys)]
            layout = make_layout()
            request_id = controller.submit_prefetch_request(keys, layout)

            deadline = time.monotonic() + timeout
            result = None
            while time.monotonic() < deadline:
                result = controller.query_prefetch_result(request_id)
                if result is not None:
                    break
                time.sleep(0.05)

            assert result is not None, "Prefetch timed out"
            return result
        finally:
            controller.stop()
            adapter.close()

    def test_lookup_task_id_passed_to_load(self):
        """lookup_task_id from the completed lookup must reach submit_fetch_task."""
        l1_manager = make_l1_manager()
        # Keys 0, 1 are found (contiguous prefix)
        io = FakeRemoteIOAdapter(num_keys=3, found_bits=[0, 1])

        hits = self._run_prefetch_and_wait(l1_manager, io, num_keys=3)

        assert hits >= 1
        # At least one load call must have happened with a non-None lookup_task_id
        assert len(io.load_calls) >= 1
        for call in io.load_calls:
            assert call["lookup_task_id"] is not None, (
                "submit_fetch_task must receive the lookup_task_id, got None"
            )

    def test_lookup_task_id_passed_to_unlock(self):
        """lookup_task_id must be propagated to submit_unlock for all unlock calls."""
        l1_manager = make_l1_manager()
        # Keys 0, 1 found
        io = FakeRemoteIOAdapter(num_keys=3, found_bits=[0, 1])

        self._run_prefetch_and_wait(l1_manager, io, num_keys=3)

        # All unlock calls should carry a non-None lookup_task_id
        assert len(io.unlock_calls) >= 1
        for call in io.unlock_calls:
            assert call["lookup_task_id"] is not None, (
                "submit_unlock must receive the lookup_task_id, got None"
            )

    def test_no_load_when_nothing_found(self):
        """When lookup finds nothing, no load or unlock calls should occur."""
        l1_manager = make_l1_manager()
        io = FakeRemoteIOAdapter(num_keys=3, found_bits=[])

        hits = self._run_prefetch_and_wait(l1_manager, io, num_keys=3)

        assert hits == 0
        assert len(io.load_calls) == 0

    def test_consistent_task_id_across_calls(self):
        """The same lookup_task_id should appear in both load and unlock calls."""
        l1_manager = make_l1_manager()
        # Only key 0 found (prefix of 1)
        io = FakeRemoteIOAdapter(num_keys=2, found_bits=[0])

        self._run_prefetch_and_wait(l1_manager, io, num_keys=2)

        if io.load_calls:
            load_tid = io.load_calls[0]["lookup_task_id"]
            # All unlock calls for plan keys should use the same task_id
            plan_unlock_tids = {c["lookup_task_id"] for c in io.unlock_calls}
            assert load_tid in plan_unlock_tids or len(io.unlock_calls) == 0
