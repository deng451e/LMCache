# SPDX-License-Identifier: Apache-2.0
"""ZMQ control-plane message structs for RemoteController.

All messages are encoded as msgpack via msgspec with tag=True.

Socket layout:
  REP socket (serve_port):   InitRequest/Response, MemRegRequest/Response,
                              LookupRequest/Response
  PULL socket (serve_unpin_port): UnpinRequest (no reply)
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
    """Server -> client: which keys were found, with compact addr/size.

    found_positions[i] is the index into the original keys list.
    byte_offsets[i] is the remote address (offset within the registered L1 MR)
    of that key's payload; byte_sizes[i] is its size. The client converts
    these to NIXL prep_xfer_dlist page indices on demand using its known
    align_bytes (no per-page list shipped on the wire).
    """

    found_positions: list[int]
    byte_offsets: list[int]
    byte_sizes: list[int]


class UnpinRequest(msgspec.Struct, tag=True):
    """Client -> server (PUSH/PULL, no reply): release read locks.

    request_id must match the originating LookupRequest so the server can
    evict the dedup cache entry.
    """

    request_id: str
    found_keys: list[WireObjectKey]


# Union used only for REP socket messages (InitRequest through LookupResponse).
# UnpinRequest arrives on the PULL socket and is decoded separately.
ZMQRepMessage = Union[
    InitRequest,
    InitResponse,
    MemRegRequest,
    MemRegResponse,
    LookupRequest,
    LookupResponse,
]
