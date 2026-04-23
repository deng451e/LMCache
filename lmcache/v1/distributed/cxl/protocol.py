# SPDX-License-Identifier: Apache-2.0
"""ZMQ control-plane message structs for CxlRemoteController.

All messages are encoded as msgpack via msgspec with tag=True.

Socket layout:
  REP socket (serve_port):        CxlInitRequest/Response,
                                   CxlLookupRequest/Response
  PULL socket (serve_unpin_port): CxlUnpinRequest (no reply)
"""

# Standard
from typing import Union

# Third Party
import msgspec

# First Party
# Re-export WireObjectKey from remote_controller.protocol for convenience.
from lmcache.v1.distributed.remote_controller.protocol import WireObjectKey


class CxlSubregionMeta(msgspec.Struct):
    """Describes one host's owned sub-region within the global CXL region.

    Exchanged during handshake for validation (not for routing).
    Both sides already have the full region mapped.
    """

    subregion_offset: int
    """Byte offset of this host's owned sub-region within the global region."""

    subregion_size: int
    """Size in bytes of this host's owned sub-region."""


class CxlInitRequest(msgspec.Struct, tag=True):
    """Client -> server: exchange sub-region metadata."""

    local_meta: CxlSubregionMeta
    """Initiating side's sub-region descriptor."""


class CxlInitResponse(msgspec.Struct, tag=True):
    """Server -> client: ack + server sub-region metadata."""

    server_meta: CxlSubregionMeta
    """Server side's sub-region descriptor."""


class CxlLookupRequest(msgspec.Struct, tag=True):
    """Client -> server: which keys exist in remote CXL index?

    request_id is stable across retries; the server dedup cache uses it to
    avoid double-pinning on retry.
    """

    request_id: str
    keys: list[WireObjectKey]


class CxlLookupResponse(msgspec.Struct, tag=True):
    """Server -> client: found keys and their absolute byte offsets.

    found_positions[i] is the index into the original keys list.
    byte_offsets[i] is the absolute byte offset within the global shared region.
    byte_sizes[i] is the object size in bytes.
    """

    found_positions: list[int]
    byte_offsets: list[int]
    byte_sizes: list[int]


class CxlUnpinRequest(msgspec.Struct, tag=True):
    """Client -> server (PUSH/PULL, no reply): release CXL read locks.

    request_id must match the originating CxlLookupRequest so the server can
    evict the dedup cache entry.
    """

    request_id: str
    found_keys: list[WireObjectKey]


# Union used only for REP socket messages.
# CxlUnpinRequest arrives on the PULL socket and is decoded separately.
CxlRepMessage = Union[
    CxlInitRequest,
    CxlInitResponse,
    CxlLookupRequest,
    CxlLookupResponse,
]
