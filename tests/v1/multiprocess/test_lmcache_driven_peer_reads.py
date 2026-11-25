# SPDX-License-Identifier: Apache-2.0
"""RETRIEVE with objects a peer owns: the handler hands the runs its
transfers return to the NIC DMA engine and answers with its deferred
response; the retrieve completes (completion event, read-lock release, end
event) only once the runs' kernels are queued."""

# Standard
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, Callable, cast

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.multiprocess.affinity_pool import DeferredResponse
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer
from tests.v1.multiprocess.test_event_ipc_handle_path import (
    _FakeEventBackend,
    _FakeStorageManager,
    _NoopDispatcher,
)


class _FakeNicDma:
    """Stands in for NicDmaEngine; the test decides when the runs' kernels
    are queued."""

    def __init__(self) -> None:
        self.reads: list[tuple[list, Callable[[bool], Any], DeferredResponse]] = []

    def read(self, cache_context, runs, finish) -> DeferredResponse:
        response = DeferredResponse()
        self.reads.append((list(runs), finish, response))
        return response

    def queued(self, ok: bool) -> None:
        _, finish, response = self.reads[-1]
        response.set_result(finish(ok))

    def close(self) -> None:
        return None


class _Dispatcher(_NoopDispatcher):
    def stop(self) -> None:
        return None


class _Storage(_FakeStorageManager):
    def unsafe_read(self, keys, l1_owners=None):
        return list(keys), [SimpleNamespace(get_size=lambda: 64) for _ in keys]


@pytest.fixture
def t(monkeypatch: pytest.MonkeyPatch):
    order: list[Any] = []
    monkeypatch.setattr(
        lmcache_driven_transfer, "DeviceHostFuncDispatcher", _Dispatcher
    )
    monkeypatch.setattr(
        lmcache_driven_transfer, "downsample_and_stage_block_ids", lambda c, b: b
    )
    monkeypatch.setattr(
        lmcache_driven_transfer.torch_dev, "device", lambda d: nullcontext()
    )
    monkeypatch.setattr(
        lmcache_driven_transfer.torch_dev, "stream", lambda s: nullcontext()
    )
    monkeypatch.setattr(
        lmcache_driven_transfer,
        "submit_callback_to_stream",
        lambda stream, kind, payload: order.append(kind),
    )
    monkeypatch.setattr(lmcache_driven_transfer, "fd_channel_path", lambda port: "")
    server_context = SimpleNamespace(
        chunk_size=1,
        hash_algorithm_name="blake3",
        null_block_id=0,
        storage_manager=_Storage(),
        get_read_owners=lambda request_id: None,
        event_bus=SimpleNamespace(
            publish=lambda event: None,
            publish_on_stream=lambda stream, event: order.append(event.event_type),
            has_subscribers=lambda event_type: False,
        ),
        resolve_obj_keys=lambda key, group_ids: [["k0", "k1"]],
        layout_desc_registry=SimpleNamespace(
            register=lambda *args, **kwargs: None,
            unregister=lambda *args, **kwargs: None,
        ),
    )
    build_ctx = SimpleNamespace(
        mp_config=SimpleNamespace(
            supported_transfer_mode="lmcache_driven",
            rocm_host_event_wait=False,
            port=0,
        ),
        engine_context=server_context,
    )
    module: Any = lmcache_driven_transfer.LMCacheDrivenTransferModule.create(
        cast(Any, build_ctx)
    )
    module._nic_dma.close()
    nic = _FakeNicDma()
    module._nic_dma = nic
    returned: list[Any] = []

    def transfer(block_ids, memory_objs, **kwargs) -> Any:
        order.append("queued")
        return list(returned) or None

    backend = _FakeEventBackend()
    original_record = backend.record_event

    def record(event: object, stream: object) -> None:
        order.append("record")
        original_record(event, stream)

    backend.record_event = record  # type: ignore[method-assign]
    cache_context = SimpleNamespace(
        device=torch.device("cpu"),
        stream="transfer-stream",
        cupy_stream="cupy-stream",
        hold_imported_event=lambda event: 0,
        release_imported_event=lambda token: None,
        max_batch_size=4,
        kv_layer_groups_manager=SimpleNamespace(
            num_object_groups=1,
            num_kernel_groups=1,
            object_groups=[SimpleNamespace(kernel_group_indices=[0])],
            get_attn_desc=lambda: SimpleNamespace(
                num_chunks_in_sw=[-1], group_kinds=()
            ),
        ),
        calculate_num_blocks=lambda chunk_size, group_idx: 1,
        transfer_object_group=transfer,
        num_layers=1,
        close=lambda: None,
    )
    entry = lmcache_driven_transfer.ContextEntry(
        cache_context=cast(Any, cache_context),
        model_name="model",
        world_size=1,
        event_backend=cast(Any, backend),
    )
    monkeypatch.setattr(module, "get_and_touch_context_entry", lambda iid: entry)
    yield SimpleNamespace(module=module, order=order, nic=nic, returned=returned)
    module.close()


def _key() -> Any:
    return SimpleNamespace(
        request_id="request",
        cache_salt="",
        worker_id=0,
        start=0,
        end=2,
        has_unfull_chunk=lambda chunk_size: False,
    )


def _retrieve(t: Any) -> Any:
    return t.module.retrieve(_key(), 1, [[0, 1]], b"producer")


_TAIL = ["record", "finish_read_by_owner", EventType.MP_RETRIEVE_END]


def test_peer_runs_defer_the_response_until_their_kernels_are_queued(t) -> None:
    t.returned.append("run")
    response = _retrieve(t)
    assert isinstance(response, DeferredResponse) and not response.done()
    # Nothing completes the retrieve while the reads run.
    assert not set(_TAIL) & set(t.order)
    ((runs, _, _),) = t.nic.reads
    assert runs == ["run"]
    t.nic.queued(True)
    assert response.result(timeout=5) == (b"completion-handle", True)
    assert t.order[-3:] == _TAIL


def test_a_failed_read_completes_the_retrieve_as_failed(t) -> None:
    """vLLM then recomputes the blocks; the read locks are still released."""
    t.returned.append("run")
    response = _retrieve(t)
    t.nic.queued(False)
    assert response.result(timeout=5) == (b"completion-handle", False)
    assert t.order[-3:] == _TAIL


def test_a_retrieve_without_peer_runs_completes_inline(t) -> None:
    assert _retrieve(t) == (b"completion-handle", True)
    assert t.order[-3:] == _TAIL
    assert t.nic.reads == []
