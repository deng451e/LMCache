# SPDX-License-Identifier: Apache-2.0
"""Configuration for RemoteIOAdapter."""

# Standard
from dataclasses import dataclass, field


@dataclass
class RemoteIOAdapterConfig:
    """Configuration for RemoteIOAdapter."""

    lookup_policy: str = "first_found"
    """Key resolution across peers: 'first_found' | 'round_robin'.

    first_found — first peer (in registration order) that has a key wins.
    round_robin — cycles through peers across successive lookups; spreads
                  read load evenly.
    """

    zmq_timeout_ms: int = 5000
    """Per-peer ZMQ request timeout for lookup traffic."""

    align_bytes: int = 4096
    """L1 allocation alignment in bytes (== page size for NIXL xfer_desc).

    Set by StorageManager from L1MemoryDesc.align_bytes before constructing
    the adapter so that register_local_memory can build page-sized descriptors.
    """

    nixl_backends: list[str] = field(default_factory=lambda: ["UCX"])
    """NIXL transport backends to enable (e.g. ["UCX"], ["UCT"])."""
