# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for PrefetchController's remote lookup path (REMOTE_LOOKUP phase).

Tests verify the end-to-end remote prefetch flow without requiring real ZMQ
or NIXL: lookup in remote peers → reserve L1 → mock RDMA READ → read-locked →
report prefix hits.

Uses a real L1Manager and mock RemoteController/RemoteTransferAdapter to
exercise the full integration path without external dependencies.
"""

# Standard
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.remote_controller.config import PeerConfig
from lmcache.v1.distributed.remote_controller.controller import RemoteController
from lmcache.v1.distributed.remote_controller.types import LookupResult, RemoteKeyInfo
from lmcache.v1.distributed.remote_transfer.adapter import (
    LocalMemHandle,
    RemoteMemHandle,
    RemoteTransferAdapter,
    TransferHandle,
    TransferStatus,
)
from lmcache.v1.distributed.storage_controllers.prefetch_controller import (
    PrefetchController,
)
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    DefaultPrefetchPolicy,
)
from lmcache.v1.distributed.storage_controllers.store_policy import AdapterDescriptor
from lmcache.v1.memory_management import MemoryObjMetadata, TensorMemoryObj

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available"
)


# =============================================================================
# Mock infrastructure
# =============================================================================


class MockLocalMemHandle(LocalMemHandle):
    """Trivial local handle for tests — no actual RDMA memory."""


class MockRemoteMemHandle(RemoteMemHandle):
    """Trivial remote handle for tests."""


class MockTransferHandle(TransferHandle):
    """Transfer handle whose poll() result is fixed at construction."""

    def __init__(self, succeed: bool = True) -> None:
        self.succeed = succeed


class MockRemoteTransferAdapter(RemoteTransferAdapter):
    """Mock transport: read() returns immediately DONE.

    Tracks read_count and released_count for assertions.
    """

    def __init__(self) -> None:
        self.read_count: int = 0
        self.released_count: int = 0

    def get_local_metadata(self) -> bytes:
        return b"mock_metadata"

    def get_local_xfer_descs(self) -> bytes:
        return b"mock_xfer_descs"

    def register_local_memory(self, ptr: int, size: int, device: str) -> LocalMemHandle:
        return MockLocalMemHandle()

    def connect_peer(
        self, peer_metadata: bytes, peer_xfer_descs: bytes
    ) -> RemoteMemHandle:
        return MockRemoteMemHandle()

    def read(
        self,
        local_handle: LocalMemHandle,
        local_pages: list[int],
        remote_handle: RemoteMemHandle,
        remote_pages: list[int],
    ) -> TransferHandle:
        self.read_count += 1
        return MockTransferHandle(succeed=True)

    def write(
        self,
        local_handle: LocalMemHandle,
        local_pages: list[int],
        remote_handle: RemoteMemHandle,
        remote_pages: list[int],
    ) -> TransferHandle:
        return MockTransferHandle(succeed=True)

    def poll(self, handle: TransferHandle) -> TransferStatus:
        assert isinstance(handle, MockTransferHandle)
        return TransferStatus.DONE if handle.succeed else TransferStatus.ERROR

    def release(self, handle: TransferHandle) -> None:
        self.released_count += 1

    def close(self) -> None:
        pass


class MockRemoteController(RemoteController):
    """Mock remote controller backed by an in-memory key set.

    lookup() returns keys from ``available_keys`` in the order they appear
    in the query. unlock() is tracked for assertions.

    Args:
        available_keys: The set of keys this peer "has" in its L1.
    """

    def __init__(self, available_keys: set[ObjectKey]) -> None:
        self._available_keys = available_keys
        self._remote_handle = MockRemoteMemHandle()
        self._unlock_lock = threading.Lock()
        self.unlock_calls: list[tuple[str, list[ObjectKey]]] = []

    def lookup(self, request_id: str, keys: list[ObjectKey]) -> LookupResult:
        """Return found keys with stub RemoteKeyInfo."""
        found_keys: list[ObjectKey] = []
        key_info: dict[ObjectKey, RemoteKeyInfo] = {}
        for key in keys:
            if key in self._available_keys:
                found_keys.append(key)
                key_info[key] = RemoteKeyInfo(
                    peer_id="mock-peer",
                    remote_handle=self._remote_handle,
                    remote_pages=[0],  # ignored by MockRemoteTransferAdapter
                )
        return LookupResult(found_keys=found_keys, key_info=key_info)

    def unlock(self, request_id: str, found_keys: list[ObjectKey]) -> None:
        with self._unlock_lock:
            self.unlock_calls.append((request_id, list(found_keys)))

    def register_peer(self, config: PeerConfig) -> None:
        pass

    def unregister_peer(self, peer_id: str) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


# =============================================================================
# Test helpers
# =============================================================================


def make_object_key(chunk_id: int) -> ObjectKey:
    """Create a test ObjectKey with the given chunk ID."""
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=0,
    )


def make_layout() -> MemoryLayoutDesc:
    """Create a small MemoryLayoutDesc for testing."""
    return MemoryLayoutDesc(
        shapes=[torch.Size([100, 2, 512])],
        dtypes=[torch.bfloat16],
    )


def should_use_lazy_alloc() -> bool:
    return torch.cuda.is_available()


def wait_for_condition(
    predicate,
    timeout: float = 5.0,
    poll_interval: float = 0.05,
) -> bool:
    """Poll until predicate() returns True or timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_interval)
    return False


def wait_for_prefetch_result(
    ctrl: PrefetchController,
    req_id: int,
    timeout: float = 5.0,
    poll_interval: float = 0.05,
) -> int | None:
    """Poll query_prefetch_result until it returns a non-None value."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = ctrl.query_prefetch_result(req_id)
        if result is not None:
            return result
        time.sleep(poll_interval)
    return None


def store_keys_in_l2(
    adapter: MockL2Adapter,
    keys: list[ObjectKey],
    layout: MemoryLayoutDesc,
) -> None:
    """Store test data directly in an L2 adapter and wait for completion."""
    if not keys:
        return
    objs = []
    for _ in keys:
        tensor = torch.randn(layout.shapes[0], dtype=layout.dtypes[0])
        metadata = MemoryObjMetadata(
            shape=layout.shapes[0],
            dtype=layout.dtypes[0],
            address=0,
            phy_size=tensor.nelement() * tensor.element_size(),
            ref_count=0,
        )
        obj = TensorMemoryObj(raw_data=tensor, metadata=metadata, parent_allocator=None)
        objs.append(obj)
    adapter.submit_store_task(keys, objs)  # type: ignore[arg-type]
    ok = wait_for_condition(
        lambda: all(adapter.debug_has_key(k) for k in keys),
        timeout=5.0,
    )
    assert ok, "Failed to store test data in L2 adapter"


def make_l2_adapter() -> MockL2Adapter:
    """Create a MockL2Adapter with fast mock bandwidth."""
    return MockL2Adapter(MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0))


def make_descriptor(index: int) -> AdapterDescriptor:
    """Create an AdapterDescriptor for testing."""
    return AdapterDescriptor(
        index=index,
        config=MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0),
    )


def make_ctrl(
    l1_manager: L1Manager,
    rc: RemoteController,
    rta: MockRemoteTransferAdapter,
    local_handle: MockLocalMemHandle,
    l2_adapters: list[L2AdapterInterface] | None = None,
    descriptors: list[AdapterDescriptor] | None = None,
) -> PrefetchController:
    """Build a PrefetchController with the given remote components."""
    return PrefetchController(
        l1_manager=l1_manager,
        l2_adapters=l2_adapters or [],
        adapter_descriptors=descriptors or [],
        policy=DefaultPrefetchPolicy(),
        remote_controller=rc,
        remote_transfer=rta,
        local_handle=local_handle,
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def l1_manager():
    """Real L1Manager with 128 MB."""
    config = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=128 * 1024 * 1024,
            use_lazy=should_use_lazy_alloc(),
            init_size_in_bytes=64 * 1024 * 1024,
            align_bytes=0x1000,
        ),
        write_ttl_seconds=600,
        read_ttl_seconds=300,
    )
    mgr = L1Manager(config)
    yield mgr
    mgr.close()


@pytest.fixture
def rta() -> MockRemoteTransferAdapter:
    """Shared MockRemoteTransferAdapter."""
    return MockRemoteTransferAdapter()


@pytest.fixture
def local_handle() -> MockLocalMemHandle:
    """Shared MockLocalMemHandle."""
    return MockLocalMemHandle()


# =============================================================================
# Lifecycle tests
# =============================================================================


class TestRemoteControllerLifecycle:
    """PrefetchController start/stop behavior when remote is configured."""

    def test_start_stop_with_remote(self, l1_manager, rta, local_handle):
        """Controller starts and stops cleanly when remote is configured."""
        rc = MockRemoteController(available_keys=set())
        ctrl = make_ctrl(l1_manager, rc, rta, local_handle)
        ctrl.start()
        assert ctrl._thread.is_alive()
        ctrl.stop()
        assert not ctrl._thread.is_alive()

    def test_remote_efd_closed_after_stop(self, l1_manager, rta, local_handle):
        """The remote eventfd is allocated then closed on stop."""
        rc = MockRemoteController(available_keys=set())
        ctrl = make_ctrl(l1_manager, rc, rta, local_handle)
        assert ctrl._remote_efd is not None
        ctrl.start()
        ctrl.stop()
        # After stop(), _remote_efd is closed (OS fd is released).
        # A second close attempt should raise OSError.
        # Standard
        import os

        with pytest.raises(OSError):
            os.close(ctrl._remote_efd)

    def test_no_requests_submitted(self, l1_manager, rta, local_handle):
        """Clean start/stop with no requests ever submitted."""
        rc = MockRemoteController(available_keys={make_object_key(0)})
        ctrl = make_ctrl(l1_manager, rc, rta, local_handle)
        ctrl.start()
        ctrl.stop()


# =============================================================================
# No-L2 / direct-remote path (PD-decode scenario)
# =============================================================================


class TestNoL2DirectRemote:
    """PrefetchController with no L2 adapters; all keys go directly to remote."""

    def test_full_prefix_hit(self, l1_manager, rta, local_handle):
        """Remote has all requested keys → prefix_hits = total keys."""
        keys = [make_object_key(i) for i in range(4)]
        layout = make_layout()
        rc = MockRemoteController(available_keys=set(keys))
        ctrl = make_ctrl(l1_manager, rc, rta, local_handle)
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(keys, layout)
        result = wait_for_prefetch_result(ctrl, req_id)

        assert result == 4, f"Expected 4 prefix hits, got {result}"
        assert rta.read_count == 4, "Expected one RDMA read per found key"

        # All keys must be read-locked in L1
        read_results = l1_manager.unsafe_read(keys)
        for key in keys:
            assert read_results[key][0] == L1Error.SUCCESS, (
                f"Expected {key} to be read-locked in L1"
            )

        # Cleanup read locks
        l1_manager.finish_read(keys)
        ctrl.stop()

    def test_prefix_with_gap(self, l1_manager, rta, local_handle):
        """Remote has keys {0,1,3} (gap at 2) → prefix_hits = 2."""
        all_keys = [make_object_key(i) for i in range(4)]
        # key at index 2 is absent
        available = {all_keys[0], all_keys[1], all_keys[3]}
        layout = make_layout()
        rc = MockRemoteController(available_keys=available)
        ctrl = make_ctrl(l1_manager, rc, rta, local_handle)
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(all_keys, layout)
        result = wait_for_prefetch_result(ctrl, req_id)

        assert result == 2, f"Expected 2 prefix hits (gap at index 2), got {result}"
        # 3 keys were found and RDMA-read (0, 1, 3) even though prefix = 2
        assert rta.read_count == 3

        # Prefix keys are read-locked in L1
        prefix_keys = all_keys[:2]
        read_results = l1_manager.unsafe_read(prefix_keys)
        for key in prefix_keys:
            assert read_results[key][0] == L1Error.SUCCESS

        # key at index 2 is not in L1
        absent_results = l1_manager.reserve_read([all_keys[2]])
        assert absent_results[all_keys[2]][0] == L1Error.KEY_NOT_EXIST

        # Cleanup: release all loaded keys (0, 1, 3) before L1 teardown
        l1_manager.finish_read(list(available))
        ctrl.stop()

    def test_remote_miss(self, l1_manager, rta, local_handle):
        """Remote has no keys → prefix_hits = 0."""
        keys = [make_object_key(i) for i in range(3)]
        layout = make_layout()
        rc = MockRemoteController(available_keys=set())
        ctrl = make_ctrl(l1_manager, rc, rta, local_handle)
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(keys, layout)
        result = wait_for_prefetch_result(ctrl, req_id)

        assert result == 0
        assert rta.read_count == 0, "No RDMA reads should be issued on a full miss"

        ctrl.stop()

    def test_unlock_called_with_found_keys(self, l1_manager, rta, local_handle):
        """unlock() is called exactly once with all keys returned by lookup()."""
        keys = [make_object_key(i) for i in range(3)]
        layout = make_layout()
        available = {keys[0], keys[1]}
        rc = MockRemoteController(available_keys=available)
        ctrl = make_ctrl(l1_manager, rc, rta, local_handle)
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(keys, layout)
        result = wait_for_prefetch_result(ctrl, req_id)

        assert result == 2

        with rc._unlock_lock:
            calls = list(rc.unlock_calls)

        assert len(calls) == 1, "unlock() must be called exactly once"
        call_rid, call_found_keys = calls[0]
        assert call_rid == str(req_id), (
            "unlock() request_id must match lookup request_id"
        )
        assert set(call_found_keys) == available, (
            "unlock() must be called with all found keys"
        )

        # Cleanup
        l1_manager.finish_read(list(available))
        ctrl.stop()

    def test_unlock_called_on_full_miss(self, l1_manager, rta, local_handle):
        """unlock() is called even when remote finds nothing (with empty list)."""
        keys = [make_object_key(i) for i in range(2)]
        layout = make_layout()
        rc = MockRemoteController(available_keys=set())
        ctrl = make_ctrl(l1_manager, rc, rta, local_handle)
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(keys, layout)
        result = wait_for_prefetch_result(ctrl, req_id)

        assert result == 0

        with rc._unlock_lock:
            calls = list(rc.unlock_calls)

        assert len(calls) == 1
        _, call_found_keys = calls[0]
        assert call_found_keys == [], "unlock() must be called with empty list on miss"

        ctrl.stop()

    def test_multiple_requests(self, l1_manager, rta, local_handle):
        """Multiple sequential remote requests each complete correctly."""
        layout = make_layout()
        # Use disjoint key sets to avoid L1 conflicts between requests
        batch1 = [make_object_key(i) for i in range(3)]
        batch2 = [make_object_key(i) for i in range(10, 13)]
        rc = MockRemoteController(available_keys=set(batch1 + batch2))
        ctrl = make_ctrl(l1_manager, rc, rta, local_handle)
        ctrl.start()

        req1 = ctrl.submit_prefetch_request(batch1, layout)
        result1 = wait_for_prefetch_result(ctrl, req1)
        assert result1 == 3

        req2 = ctrl.submit_prefetch_request(batch2, layout)
        result2 = wait_for_prefetch_result(ctrl, req2)
        assert result2 == 3

        # Cleanup
        l1_manager.finish_read(batch1)
        l1_manager.finish_read(batch2)
        ctrl.stop()


# =============================================================================
# L2 + Remote integration path
# =============================================================================


class TestL2PlusRemote:
    """L2 loads some keys; missed keys go to remote."""

    def test_l2_and_remote_combine(self, l1_manager, rta, local_handle):
        """L2 prefix + remote prefix → total prefix hits correct."""
        all_keys = [make_object_key(i) for i in range(5)]
        l2_keys = all_keys[:2]  # keys 0, 1
        remote_available = {all_keys[2], all_keys[3]}  # keys 2, 3 (4 is absent)
        layout = make_layout()

        adapter = make_l2_adapter()
        store_keys_in_l2(adapter, l2_keys, layout)

        rc = MockRemoteController(available_keys=remote_available)
        ctrl = make_ctrl(
            l1_manager,
            rc,
            rta,
            local_handle,
            l2_adapters=[adapter],
            descriptors=[make_descriptor(0)],
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(all_keys, layout)
        result = wait_for_prefetch_result(ctrl, req_id)

        # L2 loads keys 0, 1 → l2_prefix_hits = 2
        # Remote missed = [2, 3, 4]; found = [2, 3] → remote_prefix_hits = 2
        # Total = 4
        assert result == 4, f"Expected 4 prefix hits, got {result}"
        assert rta.read_count == 2, "Expected 2 RDMA reads for remote keys 2 and 3"

        # All 4 loaded keys are read-locked in L1
        loaded_keys = all_keys[:4]
        read_results = l1_manager.unsafe_read(loaded_keys)
        for key in loaded_keys:
            assert read_results[key][0] == L1Error.SUCCESS

        # Cleanup
        l1_manager.finish_read(loaded_keys)
        ctrl.stop()
        adapter.close()

    def test_remote_gap_in_missed_keys(self, l1_manager, rta, local_handle):
        """Remote has gap within the missed keys → total hits trimmed to gap."""
        all_keys = [make_object_key(i) for i in range(5)]
        l2_keys = all_keys[:2]  # keys 0, 1
        # Remote has keys 2 and 4 but not 3 → breaks prefix at missed_keys[1]
        remote_available = {all_keys[2], all_keys[4]}
        layout = make_layout()

        adapter = make_l2_adapter()
        store_keys_in_l2(adapter, l2_keys, layout)

        rc = MockRemoteController(available_keys=remote_available)
        ctrl = make_ctrl(
            l1_manager,
            rc,
            rta,
            local_handle,
            l2_adapters=[adapter],
            descriptors=[make_descriptor(0)],
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(all_keys, layout)
        result = wait_for_prefetch_result(ctrl, req_id)

        # L2: prefix 2; remote missed=[2,3,4], found=[2,4]
        # remote prefix: key 2 ✓ → 1, key 3 ✗ → break → remote_prefix_hits = 1
        # Total = 2 + 1 = 3
        assert result == 3, f"Expected 3 prefix hits, got {result}"

        ctrl.stop()
        adapter.close()

    def test_remote_miss_after_l2(self, l1_manager, rta, local_handle):
        """L2 loads prefix; remote finds nothing → no extra hits."""
        all_keys = [make_object_key(i) for i in range(4)]
        l2_keys = all_keys[:2]  # keys 0, 1
        layout = make_layout()

        adapter = make_l2_adapter()
        store_keys_in_l2(adapter, l2_keys, layout)

        rc = MockRemoteController(available_keys=set())
        ctrl = make_ctrl(
            l1_manager,
            rc,
            rta,
            local_handle,
            l2_adapters=[adapter],
            descriptors=[make_descriptor(0)],
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(all_keys, layout)
        result = wait_for_prefetch_result(ctrl, req_id)

        assert result == 2, f"Expected 2 prefix hits (L2 only), got {result}"
        assert rta.read_count == 0, "No RDMA reads should occur when remote misses"

        l1_manager.finish_read(l2_keys)
        ctrl.stop()
        adapter.close()

    def test_l2_full_hit_no_remote_lookup(self, l1_manager, rta, local_handle):
        """If L2 loads all keys, remote pipeline is never invoked."""
        all_keys = [make_object_key(i) for i in range(3)]
        layout = make_layout()

        adapter = make_l2_adapter()
        store_keys_in_l2(adapter, all_keys, layout)

        rc = MockRemoteController(available_keys=set(all_keys))
        ctrl = make_ctrl(
            l1_manager,
            rc,
            rta,
            local_handle,
            l2_adapters=[adapter],
            descriptors=[make_descriptor(0)],
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(all_keys, layout)
        result = wait_for_prefetch_result(ctrl, req_id)

        assert result == 3
        # Remote should not have been called at all
        assert rta.read_count == 0, (
            "Remote should not be called when L2 loads everything"
        )
        with rc._unlock_lock:
            calls = list(rc.unlock_calls)
        assert len(calls) == 0, "unlock() must not be called when remote is not invoked"

        l1_manager.finish_read(all_keys)
        ctrl.stop()
        adapter.close()
