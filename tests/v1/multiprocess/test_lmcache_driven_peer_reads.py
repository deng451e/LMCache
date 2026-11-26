# SPDX-License-Identifier: Apache-2.0
"""RETRIEVE of objects a peer owns answers with the NIC DMA engine's deferred
response; the retrieve completes (event, read-lock release, end event) only
once their kernels are queued."""

# Standard
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, cast

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.multiprocess.affinity_pool import DeferredResponse
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as transfer
from lmcache.v1.multiprocess.object_group_transfer.deferred import defer_runs
from tests.v1.multiprocess.test_event_ipc_handle_path import (
    _FakeEventBackend,
    _FakeStorageManager,
    _NoopDispatcher,
)

_TAIL = ["record", "finish_read_by_owner", EventType.MP_RETRIEVE_END]


class _Storage(_FakeStorageManager):
    def unsafe_read(self, keys, l1_owners=None):
        return list(keys), [SimpleNamespace(get_size=lambda: 64) for _ in keys]


@pytest.fixture
def t(monkeypatch: pytest.MonkeyPatch):
    order: list[Any] = []
    reads: list = []
    returned: list = []
    monkeypatch.setattr(transfer, "DeviceHostFuncDispatcher", _NoopDispatcher)
    monkeypatch.setattr(transfer, "downsample_and_stage_block_ids", lambda c, b: b)
    monkeypatch.setattr(transfer.torch_dev, "device", lambda d: nullcontext())
    monkeypatch.setattr(transfer.torch_dev, "stream", lambda s: nullcontext())
    monkeypatch.setattr(
        transfer, "submit_callback_to_stream", lambda s, kind, p: order.append(kind)
    )
    ctx = SimpleNamespace(
        chunk_size=1,
        storage_manager=_Storage(),
        get_read_owners=lambda request_id: None,
        event_bus=SimpleNamespace(
            publish=lambda event: None,
            publish_on_stream=lambda stream, event: order.append(event.event_type),
        ),
        resolve_obj_keys=lambda key, group_ids: [["k0", "k1"]],
    )
    module: Any = transfer.LMCacheDrivenTransferModule(cast(Any, ctx))

    def read(cache_context, runs, finish) -> DeferredResponse:
        reads.append((list(runs), finish, DeferredResponse()))
        return reads[-1][2]

    module._deferred_engine = SimpleNamespace(read=read)
    backend = _FakeEventBackend()
    record = backend.record_event
    backend.record_event = lambda e, s: order.append("record") or record(e, s)  # type: ignore[method-assign]
    cache_context = SimpleNamespace(
        device=torch.device("cpu"),
        stream="stream",
        cupy_stream="cupy",
        hold_imported_event=lambda event: 0,
        max_batch_size=4,
        kv_layer_groups_manager=SimpleNamespace(
            num_object_groups=1,
            num_kernel_groups=1,
            get_attn_desc=lambda: SimpleNamespace(
                num_chunks_in_sw=[-1], group_kinds=()
            ),
        ),
        calculate_num_blocks=lambda chunk_size, group_idx: 1,
        transfer_object_group=lambda *a, **k: defer_runs(returned),
    )
    entry = transfer.ContextEntry(
        cast(Any, cache_context), "model", 1, event_backend=cast(Any, backend)
    )
    monkeypatch.setattr(module, "get_and_touch_context_entry", lambda iid: entry)
    key = SimpleNamespace(
        request_id="r", cache_salt="", end=2, has_unfull_chunk=lambda size: False
    )
    yield SimpleNamespace(
        retrieve=lambda: module.retrieve(key, 1, [[0, 1]], b"producer"),
        order=order,
        reads=reads,
        returned=returned,
    )


@pytest.mark.parametrize("ok", [True, False])
def test_peer_runs_defer_the_response_until_their_kernels_are_queued(t, ok) -> None:
    """A failed read completes the retrieve as failed (vLLM recomputes); the
    read locks are released either way."""
    t.returned.append("run")
    response = t.retrieve()
    assert isinstance(response, DeferredResponse) and not response.done()
    assert not set(_TAIL) & set(t.order)
    ((runs, finish, deferred),) = t.reads
    assert runs == ["run"] and deferred is response
    response.set_result(finish(ok))
    assert response.result(timeout=5) == (b"completion-handle", ok)
    assert t.order[-3:] == _TAIL

