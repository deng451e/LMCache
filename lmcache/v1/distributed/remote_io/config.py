# SPDX-License-Identifier: Apache-2.0
"""Configuration for RemoteIOAdapter."""

# Standard
from dataclasses import dataclass


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
