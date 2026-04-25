# SPDX-License-Identifier: Apache-2.0
"""CxlAdaptor: local CXL NUMA region as an L2 cache tier.

Implements L2AdapterInterface + L1ManagerListener.  Owns the CXL allocator
and _index.  Registered as an L2 adapter AND as an L1ManagerListener by
StorageManager.

Shadow page model:
  L1Manager._objects holds TensorMemoryObj backed by CXL VA (CXL_SHADOW).
  CxlAdaptor._index is the authoritative allocation record.
  L1MemoryManager.free() skips CXL_SHADOW objects; CxlAdaptor frees them.
"""

# Future
from __future__ import annotations

# Standard
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol
import ctypes
import mmap
import os
import stat
import threading

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.cxl.protocol import CxlSubregionMeta
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1ManagerListener
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface, L2TaskId
from lmcache.v1.memory_management import (
    MemoryFormat,
    TensorMemoryAllocator,
    TensorMemoryObj,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.l1_manager import L1Manager

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Device-agnostic mmap helper
# ---------------------------------------------------------------------------


def _open_and_mmap(path: str, region_size: int) -> mmap.mmap:
    """Open *path* and return a MAP_SHARED mmap of *region_size* bytes.

    Supports two path categories:

    * **DAX / block / char devices** (``/dev/dax0.0``, ``/dev/pmem0``, …):
      opened with ``O_RDWR``; the device must already expose at least
      ``region_size`` bytes.

    * **Regular files** (``/dev/shm/cxl.bin``, any writable path):
      opened with ``O_RDWR | O_CREAT``; ``ftruncate`` is called when the
      file is smaller than ``region_size`` so the mmap always succeeds
      without a separate ``truncate`` step.

    Args:
        path: Filesystem path to the DAX device or backing file.
        region_size: Number of bytes to map.

    Returns:
        A writeable ``mmap.mmap`` object covering the full region.

    Raises:
        OSError: If the path cannot be opened or mapped.
    """
    try:
        st = os.stat(path)
        is_device = stat.S_ISBLK(st.st_mode) or stat.S_ISCHR(st.st_mode)
    except FileNotFoundError:
        is_device = False

    if is_device:
        fd = os.open(path, os.O_RDWR)
    else:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        if os.fstat(fd).st_size < region_size:
            os.ftruncate(fd, region_size)

    try:
        return mmap.mmap(
            fd,
            region_size,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# CUDA host registration (optional; skipped on non-CUDA systems)
# ---------------------------------------------------------------------------

_libcuda: ctypes.CDLL | None = None
_cudaHostRegister_fn = None
_cudaHostUnregister_fn = None


def _init_cuda_bindings() -> None:
    global _libcuda, _cudaHostRegister_fn, _cudaHostUnregister_fn
    if _libcuda is not None:
        return
    try:
        _libcuda = ctypes.CDLL("libcuda.so")
        _cudaHostRegister_fn = _libcuda.cuMemHostRegister_v2
        _cudaHostRegister_fn.restype = ctypes.c_int
        _cudaHostRegister_fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        _cudaHostUnregister_fn = _libcuda.cuMemHostUnregister
        _cudaHostUnregister_fn.restype = ctypes.c_int
        _cudaHostUnregister_fn.argtypes = [ctypes.c_void_p]
    except OSError:
        logger.warning("libcuda.so not found; cudaHostRegister skipped for CXL region")
        _libcuda = None


def _cuda_host_register(va: int, size: int) -> None:
    _init_cuda_bindings()
    if _cudaHostRegister_fn is None:
        return
    # CU_MEMHOSTREGISTER_PORTABLE (0x1) | CU_MEMHOSTREGISTER_DEVICEMAP (0x2)
    # — DEVICEMAP makes the mapping usable as a GPU device pointer,
    #   enabling GPU-Direct DMA between GPU KV and CXL memory.
    flags = 0x1 | 0x2
    ret = _cudaHostRegister_fn(
        ctypes.c_void_p(va), ctypes.c_size_t(size), ctypes.c_uint(flags)
    )
    if ret != 0:
        logger.warning("cudaHostRegister returned %d for CXL region", ret)


def _cuda_host_unregister(va: int) -> None:
    if _cudaHostUnregister_fn is None:
        return
    _cudaHostUnregister_fn(ctypes.c_void_p(va))


# ---------------------------------------------------------------------------
# Tiering policy
# ---------------------------------------------------------------------------


class TieringPolicy(Protocol):
    """Decides per-key/layout whether to allocate in CXL or DRAM."""

    def should_use_cxl(self, key: ObjectKey, layout: MemoryLayoutDesc) -> bool:
        """Return True to route this allocation to CXL.

        Args:
            key: Object key being allocated.
            layout: Memory layout for the allocation.

        Returns:
            True if CXL should be used; False for DRAM.
        """
        ...


class AlwaysCxlPolicy:
    """Route every allocation to CXL (useful for testing / all-CXL deployments).

    Args:
        None.
    """

    def should_use_cxl(self, key: ObjectKey, layout: MemoryLayoutDesc) -> bool:
        """Always returns True.

        Args:
            key: Object key (unused).
            layout: Memory layout (unused).

        Returns:
            True.
        """
        return True


class SizeThresholdCxlPolicy:
    """Route to CXL when the total byte size of the tensor exceeds a threshold.

    Large KV tensors benefit from CXL capacity; small tensors (e.g. metadata)
    stay in DRAM to avoid CXL-access latency on the hot path.

    Args:
        min_bytes: Minimum total byte size for CXL routing (default 1 MiB).
    """

    def __init__(self, min_bytes: int = 1 << 20) -> None:
        self._min_bytes = min_bytes

    def should_use_cxl(self, key: ObjectKey, layout: MemoryLayoutDesc) -> bool:
        """Return True if total byte size >= min_bytes.

        Args:
            key: Object key (unused).
            layout: Memory layout; total_bytes computed from shapes/dtypes.

        Returns:
            True if CXL should be used.
        """
        # First Party
        from lmcache.integration.vllm.utils import get_size_bytes  # noqa: PLC0415

        total = get_size_bytes(layout.shapes, layout.dtypes)
        return total >= self._min_bytes


# ---------------------------------------------------------------------------
# CXL index entry
# ---------------------------------------------------------------------------


@dataclass
class CxlIndexEntry:
    """Per-key state in CxlAdaptor._index."""

    obj: TensorMemoryObj
    """The CXL-backed allocation; data_ptr() is in CXL NUMA VA space."""

    pending_free: bool = False
    """Set when eviction is requested.  New locks not granted; free deferred."""

    l2_lock_count: int = 0
    """Count of outstanding lookup results (local or remote) not yet unlocked."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class CxlAdaptorConfig:
    """Configuration for CxlAdaptor.

    Args:
        dax_device_path: Linux DAX device path (same on all hosts), e.g. /dev/dax0.0.
        region_size: Full shared region size in bytes (same on all hosts).
        subregion_offset: Byte offset of this host's owned sub-region.
        subregion_size: Size of this host's owned sub-region.
        align_bytes: Allocation alignment; passed to TensorMemoryAllocator.
        cxl_numa_node: NUMA node ID of the CXL device on this host (-1 to disable).
    """

    dax_device_path: str
    region_size: int
    subregion_offset: int
    subregion_size: int
    align_bytes: int = field(default=0x1000)
    cxl_numa_node: int = field(default=-1)


# ---------------------------------------------------------------------------
# CxlAdaptor
# ---------------------------------------------------------------------------


class CxlAdaptor(L1ManagerListener, L2AdapterInterface):
    """Local CXL NUMA region as an L2 cache adapter.

    Implements both L1ManagerListener (eviction callbacks) and
    L2AdapterInterface (lookup/load/store for PrefetchController).

    The full CXL shared region is mmap'd at startup via DAX device.
    Each host allocates only within its owned sub-region
    (subregion_offset, subregion_size); all hosts can read any offset.

    Args:
        config: CxlAdaptorConfig with DAX path, region sizes, and alignment.
        l1_manager: The local L1Manager instance (for register_shadow and delete).
    """

    def __init__(
        self,
        config: CxlAdaptorConfig,
        l1_manager: L1Manager,
    ) -> None:
        super().__init__()
        self._config = config
        self._l1_manager = l1_manager

        # --- Map the full shared CXL region ---
        # Works for DAX/char/block devices and regular files (/dev/shm/*).
        self._mmap = _open_and_mmap(config.dax_device_path, config.region_size)

        self._region_va_base: int = ctypes.addressof(
            ctypes.c_char.from_buffer(self._mmap)
        )

        # --- Wrap the owned sub-region as a tensor for the allocator ---
        # Keep the ctypes array alive to prevent GC from releasing the buffer.
        self._sub_buf = (ctypes.c_uint8 * config.subregion_size).from_address(
            self._region_va_base + config.subregion_offset
        )
        cxl_tensor = torch.frombuffer(self._sub_buf, dtype=torch.uint8)
        self._allocator = TensorMemoryAllocator(
            cxl_tensor, align_bytes=config.align_bytes
        )

        # --- Register the full region with CUDA once ---
        _cuda_host_register(self._region_va_base, config.region_size)

        # --- Index and LRU tracking ---
        self._index: dict[ObjectKey, CxlIndexEntry] = {}
        self._lru: OrderedDict[ObjectKey, None] = OrderedDict()
        self._lock = threading.Lock()

        # --- Task ID management and completion queues ---
        self._next_task_id: L2TaskId = 0
        self._completed_store_tasks: dict[L2TaskId, bool] = {}
        self._completed_lookup_tasks: dict[L2TaskId, Bitmap] = {}
        self._completed_load_tasks: dict[L2TaskId, Bitmap] = {}

        # --- Event file descriptors ---
        self._store_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self._lookup_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self._load_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self._deferred_free_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)

        # --- Deferred-free background thread ---
        # on_l1_keys_read_finished is called while L1Manager._lock is held,
        # so we cannot call l1_manager.delete() inline (deadlock).
        self._deferred_free_queue: list[ObjectKey] = []
        self._deferred_free_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._deferred_free_thread = threading.Thread(
            target=self._deferred_free_loop,
            daemon=True,
            name="cxl-deferred-free",
        )
        self._deferred_free_thread.start()

        # --- GPU transfer pool ---
        self._transfer_executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="cxl-gpu"
        )

        logger.info(
            "CxlAdaptor: mapped %s region_size=%d subregion=[%d, %d)",
            config.dax_device_path,
            config.region_size,
            config.subregion_offset,
            config.subregion_offset + config.subregion_size,
        )

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def region_va_base(self) -> int:
        """VA base of the full mapped CXL shared region."""
        return self._region_va_base

    def get_subregion_meta(self) -> CxlSubregionMeta:
        """Return this host's sub-region descriptor for the handshake.

        Returns:
            CxlSubregionMeta with subregion_offset and subregion_size.
        """
        return CxlSubregionMeta(
            subregion_offset=self._config.subregion_offset,
            subregion_size=self._config.subregion_size,
        )

    # -----------------------------------------------------------------------
    # L2AdapterInterface — event fds
    # -----------------------------------------------------------------------

    def get_store_event_fd(self) -> int:
        """Return the store completion event fd.

        Returns:
            Event fd signaled when store tasks complete (immediately, no-op).
        """
        return self._store_efd

    def get_lookup_and_lock_event_fd(self) -> int:
        """Return the lookup-and-lock completion event fd.

        Returns:
            Event fd signaled when lookup tasks complete.
        """
        return self._lookup_efd

    def get_load_event_fd(self) -> int:
        """Return the load completion event fd.

        Returns:
            Event fd signaled when shadow re-registration (load) completes.
        """
        return self._load_efd

    def requires_pre_allocation(self) -> bool:
        """Return False.

        PrefetchController must NOT call reserve_write before submit_load_task.
        CxlAdaptor registers a CXL_SHADOW object in L1Manager during load;
        no DRAM buffer is needed or expected.

        Returns:
            False.
        """
        return False

    # -----------------------------------------------------------------------
    # L2AdapterInterface — store (no-op: data already in CXL)
    # -----------------------------------------------------------------------

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list,
    ) -> L2TaskId:
        """No-op store: objects are CXL_SHADOW and data is already in CXL.

        Args:
            keys: Keys to store (unused).
            objects: MemoryObj list (unused).

        Returns:
            Task ID for the immediately-completed no-op task.
        """
        with self._lock:
            task_id = self._next_task_id
            self._next_task_id += 1
            self._completed_store_tasks[task_id] = True
        os.eventfd_write(self._store_efd, 1)
        return task_id

    def pop_completed_store_tasks(self) -> dict[L2TaskId, bool]:
        """Return all immediately-completed no-op store tasks.

        Returns:
            Dict mapping task ID to True (always success for CXL no-op store).
        """
        with self._lock:
            completed = self._completed_store_tasks
            self._completed_store_tasks = {}
        return completed

    # -----------------------------------------------------------------------
    # L2AdapterInterface — lookup and lock
    # -----------------------------------------------------------------------

    def submit_lookup_and_lock_task(self, keys: list[ObjectKey]) -> L2TaskId:
        """Check _index for each key; increment l2_lock_count for found keys.

        Skips entries with pending_free=True (being evicted).
        Non-blocking; result available immediately.

        Args:
            keys: Keys to look up and lock.

        Returns:
            Task ID for the completed lookup task.
        """
        bitmap = Bitmap(len(keys))
        with self._lock:
            for i, key in enumerate(keys):
                entry = self._index.get(key)
                if entry is not None and not entry.pending_free:
                    entry.l2_lock_count += 1
                    bitmap.set(i)
                    self._lru.move_to_end(key)
            task_id = self._next_task_id
            self._next_task_id += 1
            self._completed_lookup_tasks[task_id] = bitmap
        os.eventfd_write(self._lookup_efd, 1)
        return task_id

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        """Return the found bitmap for this task (one-shot).

        Args:
            task_id: Task ID from submit_lookup_and_lock_task.

        Returns:
            Bitmap with bit i=1 for found keys, or None if not ready.
        """
        with self._lock:
            return self._completed_lookup_tasks.pop(task_id, None)

    def submit_unlock(
        self,
        keys: list[ObjectKey],
        lookup_task_id: L2TaskId | None = None,
    ) -> None:
        """Decrement l2_lock_count for each key; free if pending_free and count==0.

        Args:
            keys: Keys whose CXL locks should be released.
            lookup_task_id: Unused by CxlAdaptor.
        """
        to_free: list[ObjectKey] = []
        with self._lock:
            for key in keys:
                entry = self._index.get(key)
                if entry is None:
                    continue
                if entry.l2_lock_count > 0:
                    entry.l2_lock_count -= 1
                if entry.l2_lock_count == 0 and entry.pending_free:
                    to_free.append(key)
        for key in to_free:
            self._try_free(key)

    # -----------------------------------------------------------------------
    # L2AdapterInterface — load (shadow re-registration, no copy)
    # -----------------------------------------------------------------------

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list,
        lookup_task_id: L2TaskId | None = None,
    ) -> L2TaskId:
        """Re-register each found key as a write-locked shadow in L1Manager.

        No data copy occurs.  objects must be [] (CxlAdaptor owns allocation).
        Called from PrefetchController thread (L1 lock NOT held).

        Args:
            keys: Keys to load (subset of prior lookup found keys).
            objects: Must be empty list (CxlAdaptor manages memory).
            lookup_task_id: Unused by CxlAdaptor.

        Returns:
            Task ID for the completed load task.
        """
        bitmap = Bitmap(len(keys))
        for i, key in enumerate(keys):
            with self._lock:
                entry = self._index.get(key)
                if entry is None or entry.pending_free:
                    continue
                obj = entry.obj

            err = self._l1_manager.register_shadow(key, obj, is_temporary=True)
            if err in (L1Error.SUCCESS, L1Error.KEY_NOT_WRITABLE):
                # KEY_NOT_WRITABLE means already in L1 (previous registration);
                # treat as a load miss so PrefetchController re-evaluates.
                if err == L1Error.SUCCESS:
                    bitmap.set(i)

        with self._lock:
            task_id = self._next_task_id
            self._next_task_id += 1
            self._completed_load_tasks[task_id] = bitmap
        os.eventfd_write(self._load_efd, 1)
        return task_id

    def query_load_result(self, task_id: L2TaskId) -> Bitmap | None:
        """Return the load result bitmap for this task (one-shot).

        Args:
            task_id: Task ID from submit_load_task.

        Returns:
            Bitmap with bit i=1 for successfully loaded keys, or None if not ready.
        """
        with self._lock:
            return self._completed_load_tasks.pop(task_id, None)

    # -----------------------------------------------------------------------
    # CXL allocation — called from L1Manager.reserve_write (lock held by L1)
    # -----------------------------------------------------------------------

    def cxl_allocate(
        self,
        key: ObjectKey,
        layout_desc: MemoryLayoutDesc,
        is_temporary: bool,
    ) -> TensorMemoryObj | None:
        """Allocate from CXL sub-region and record in _index.

        Called from inside L1Manager.reserve_write while the L1 lock is held,
        so this method MUST NOT call any L1Manager method (including via eviction,
        since _evict_one -> _try_free -> l1_manager.delete would deadlock).

        Returns None when the CXL region is full.  The caller (L1Manager) handles
        the OOM case; background eviction via submit_unlock / deferred-free thread
        reclaims space between allocations.

        Args:
            key: Object key for this allocation.
            layout_desc: Shapes/dtypes for the allocation.
            is_temporary: Whether this object is temporary (passed through to
                L1ObjectState by the L1Manager caller).

        Returns:
            TensorMemoryObj backed by CXL NUMA VA with fmt=CXL_SHADOW,
            or None if the CXL region is full.
        """
        objs = self._allocator.batched_allocate(
            layout_desc.shapes,
            layout_desc.dtypes,
            1,
            fmt=MemoryFormat.CXL_SHADOW,
        )
        if objs is None:
            return None
        obj = objs[0]
        with self._lock:
            self._index[key] = CxlIndexEntry(obj=obj)
            self._lru[key] = None
        return obj

    def _evict_one(self) -> bool:
        """Mark the LRU unlocked entry as pending_free and attempt to free it.

        Returns:
            True if a candidate was found (eviction may be deferred).
            False if no evictable candidate exists.
        """
        with self._lock:
            candidate_key: ObjectKey | None = None
            for k in self._lru:  # LRU → MRU order
                e = self._index.get(k)
                if e and e.l2_lock_count == 0 and not e.pending_free:
                    candidate_key = k
                    break
            if candidate_key is None:
                return False
            self._index[candidate_key].pending_free = True

        # Call l1_manager.delete outside _lock to respect lock ordering
        # (L1 lock always acquired before CXL _lock).
        self._try_free(candidate_key)
        return True

    def _try_free(self, key: ObjectKey) -> None:
        """Attempt to delete from L1 and free CXL pages for a pending_free entry.

        Safe to call without _lock held; acquires _lock internally around
        the actual free.

        Args:
            key: CXL index key to free.
        """
        with self._lock:
            entry = self._index.get(key)
            if entry is None or not entry.pending_free or entry.l2_lock_count > 0:
                return

        results = self._l1_manager.delete([key])
        err = results.get(key, L1Error.KEY_NOT_EXIST)

        with self._lock:
            entry = self._index.get(key)
            if entry is None:
                return
            if (
                err in (L1Error.SUCCESS, L1Error.KEY_NOT_EXIST)
                and entry.l2_lock_count == 0
            ):
                self._allocator.batched_free([entry.obj])
                del self._index[key]
                self._lru.pop(key, None)
            # else: L1 key is locked; leave pending_free=True for later retry

    # -----------------------------------------------------------------------
    # Server-side methods (called by CxlRemoteController in ZMQ thread)
    # -----------------------------------------------------------------------

    def server_lookup_and_lock(
        self,
        keys: list[ObjectKey],
    ) -> dict[ObjectKey, tuple[int, int]]:
        """Synchronously look up keys and increment l2_lock_count.

        Does NOT interact with L1Manager.  Reads _index only.
        Skips entries with pending_free=True.

        Args:
            keys: Keys to look up.

        Returns:
            {key: (byte_offset, byte_size)} for found keys only.
            byte_offset is absolute within the global shared CXL region.
        """
        result: dict[ObjectKey, tuple[int, int]] = {}
        with self._lock:
            for key in keys:
                entry = self._index.get(key)
                if entry is not None and not entry.pending_free:
                    entry.l2_lock_count += 1
                    byte_offset = entry.obj.data_ptr - self._region_va_base
                    byte_size = entry.obj.meta.phy_size
                    result[key] = (byte_offset, byte_size)
                    self._lru.move_to_end(key)
        return result

    def server_unpin(self, keys: list[ObjectKey]) -> None:
        """Decrement l2_lock_count; free if pending_free.

        Called by CxlRemoteController._handle_unpin().

        Args:
            keys: Keys whose remote read locks should be released.
        """
        self.submit_unlock(keys)

    # -----------------------------------------------------------------------
    # GPU transfer
    # -----------------------------------------------------------------------

    def transfer_to_gpu(self, obj: TensorMemoryObj, dst) -> Future:
        """Async cudaMemcpy: CXL NUMA VA -> GPU buffer.

        Args:
            obj: CXL_SHADOW TensorMemoryObj (source).
            dst: Destination MemoryObj with a GPU tensor.

        Returns:
            Future that resolves when the copy completes.
        """
        src_tensor = obj.raw_tensor
        dst_tensor = dst.raw_tensor

        def _copy():
            if src_tensor is None or dst_tensor is None:
                raise ValueError(
                    "Invalid source or destination tensor for CXL GPU transfer"
                )
            dst_tensor.copy_(src_tensor)

        return self._transfer_executor.submit(_copy)

    # -----------------------------------------------------------------------
    # L1ManagerListener callbacks
    # -----------------------------------------------------------------------

    def on_l1_keys_reserved_read(self, keys: list[ObjectKey]) -> None:
        """No-op."""

    def on_l1_keys_reserved_write(self, keys: list[ObjectKey]) -> None:
        """No-op."""

    def on_l1_keys_write_finished(self, keys: list[ObjectKey]) -> None:
        """No-op — StoreController handles L2 persistence; data is in CXL."""

    def on_l1_keys_finish_write_and_reserve_read(self, keys: list[ObjectKey]) -> None:
        """No-op."""

    def on_l1_keys_read_finished(self, keys: list[ObjectKey]) -> None:
        """Queue pending-free keys for the deferred-free background thread.

        Called while L1Manager._lock is held (finish_read is @l1_mgr_synchronized),
        so MUST NOT call any L1Manager method.  Enqueues candidates and signals
        the deferred-free thread via _deferred_free_efd.

        Args:
            keys: Keys whose L1 read locks were just released.
        """
        to_queue: list[ObjectKey] = []
        with self._lock:
            for key in keys:
                entry = self._index.get(key)
                if (
                    entry is not None
                    and entry.pending_free
                    and entry.l2_lock_count == 0
                ):
                    to_queue.append(key)
        if to_queue:
            with self._deferred_free_lock:
                self._deferred_free_queue.extend(to_queue)
            os.eventfd_write(self._deferred_free_efd, 1)

    def on_l1_keys_deleted_by_manager(self, keys: list[ObjectKey]) -> None:
        """Shadow evicted by L1Manager (called while L1 lock is held).

        CXL pages survive when L1 evicts the shadow — _index entry stays.
        If pending_free=True and l2_lock_count==0, free CXL pages now.

        Args:
            keys: Keys whose L1 entries were just deleted.
        """
        with self._lock:
            for key in keys:
                entry = self._index.get(key)
                if (
                    entry is not None
                    and entry.pending_free
                    and entry.l2_lock_count == 0
                ):
                    self._allocator.batched_free([entry.obj])
                    del self._index[key]
                    self._lru.pop(key, None)

    # -----------------------------------------------------------------------
    # Deferred-free background thread
    # -----------------------------------------------------------------------

    def _deferred_free_loop(self) -> None:
        """Background thread: retry l1_manager.delete + CXL free for queued keys."""
        while not self._stop_event.is_set():
            try:
                os.eventfd_read(self._deferred_free_efd)
            except BlockingIOError:
                self._stop_event.wait(timeout=0.01)
                continue

            with self._deferred_free_lock:
                keys, self._deferred_free_queue = self._deferred_free_queue, []

            for key in keys:
                self._try_free(key)

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------

    def close(self) -> None:
        """Stop background threads, unregister CUDA mapping, and unmap CXL region."""
        self._stop_event.set()
        try:
            os.eventfd_write(self._deferred_free_efd, 1)
        except OSError:
            pass
        self._deferred_free_thread.join(timeout=5)
        self._transfer_executor.shutdown(wait=False)

        _cuda_host_unregister(self._region_va_base)
        self._mmap.close()

        for fd in (
            self._store_efd,
            self._lookup_efd,
            self._load_efd,
            self._deferred_free_efd,
        ):
            try:
                os.close(fd)
            except OSError:
                pass

    def report_status(self) -> dict:
        """Return a status dict for the CXL adaptor.

        Returns:
            Dict with is_healthy, entry counts, and lock counts.
        """
        with self._lock:
            total = len(self._index)
            locked = sum(1 for e in self._index.values() if e.l2_lock_count > 0)
            pending = sum(1 for e in self._index.values() if e.pending_free)
        return {
            "is_healthy": True,
            "type": "CxlAdaptor",
            "total_entries": total,
            "locked_entries": locked,
            "pending_free_entries": pending,
            "region_va_base": hex(self._region_va_base),
            "subregion_offset": self._config.subregion_offset,
            "subregion_size": self._config.subregion_size,
        }
