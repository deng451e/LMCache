# SPDX-License-Identifier: Apache-2.0
"""ZMQ control-plane message structs for RemoteController.

All messages are encoded as msgpack via msgspec with tag=True so the Union
decoder can distinguish them without an explicit type annotation at the call
site.
"""

# Standard
from typing import Union

# Third Party
import msgspec


class InitRequest(msgspec.Struct, tag=True):
    """Client -> server: exchange NIXL agent metadata."""

    local_agent_metadata: bytes
    """Serialised agent descriptor from the initiating side."""


class InitResponse(msgspec.Struct, tag=True):
    """Server -> client: ack + server agent metadata."""

    server_agent_metadata: bytes
    """Serialised agent descriptor from the server side."""


class MemRegRequest(msgspec.Struct, tag=True):
    """Client -> server: exchange NIXL transfer descriptors."""

    local_xfer_descs: bytes
    """Serialised transfer descriptors from the initiating side."""


class MemRegResponse(msgspec.Struct, tag=True):
    """Server -> client: ack + server transfer descriptors."""

    server_xfer_descs: bytes
    """Serialised transfer descriptors from the server side."""


class WireObjectKey(msgspec.Struct):
    """Wire-format representation of ObjectKey for msgspec serialisation."""

    chunk_hash: bytes
    model_name: str
    kv_rank: int


class LookupRequest(msgspec.Struct, tag=True):
    """Client -> server: which keys exist in remote L1?

    request_id is stable across retries; the server dedup cache uses it to
    avoid double-pinning on retry.
    """

    request_id: str
    keys: list[WireObjectKey]


class LookupResponse(msgspec.Struct, tag=True):
    """Server -> client: which keys were found and their page indices.

    found_positions[i] is the index into the original keys list.
    pages_per_found[i] are the remote page indices for that key.
    """

    found_positions: list[int]
    pages_per_found: list[list[int]]


class UnpinRequest(msgspec.Struct, tag=True):
    """Client -> server: release read locks for the given keys.

    request_id must match the originating LookupRequest so the server can
    evict the dedup cache entry.
    """

    request_id: str
    found_keys: list[WireObjectKey]


class UnpinResponse(msgspec.Struct, tag=True):
    """Server -> client: ack."""


ZMQMessage = Union[
    InitRequest,
    InitResponse,
    MemRegRequest,
    MemRegResponse,
    LookupRequest,
    LookupResponse,
    UnpinRequest,
    UnpinResponse,
]
