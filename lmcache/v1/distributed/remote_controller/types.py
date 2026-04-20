# SPDX-License-Identifier: Apache-2.0
"""Result types returned by RemoteController."""

# Standard
from dataclasses import dataclass

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.remote_transfer.adapter import RemoteMemHandle


@dataclass
class RemoteKeyInfo:
    """Per-key transfer info returned by lookup().

    Passed directly to RemoteTransferAdapter.read() by the caller.
    """

    peer_id: str
    """ID of the peer that holds this key."""

    remote_handle: RemoteMemHandle
    """Handle for the owning peer's L1 buffer. Sourced from peer registry."""

    remote_pages: list[int]
    """Page indices within the remote L1 buffer for this key."""


@dataclass
class LookupResult:
    """Aggregated result of a remote lookup across all peers."""

    found_keys: list[ObjectKey]
    """Subset of queried keys that exist on at least one peer."""

    key_info: dict[ObjectKey, RemoteKeyInfo]
    """Transfer info for each found key. Used to drive RemoteTransferAdapter.read()."""
