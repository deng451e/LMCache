# SPDX-License-Identifier: Apache-2.0
"""Configuration types for RemoteController."""

# Standard
from dataclasses import dataclass, field


@dataclass
class PeerConfig:
    """Connection info for one remote peer."""

    peer_id: str
    """Logical name, e.g. 'decode-0' or 'peer-gpu-1'."""

    host: str
    """Hostname or IP address of the peer."""

    port: int
    """ZMQ server port on the peer."""


@dataclass
class RemoteControllerConfig:
    """Configuration for RemoteController."""

    mode: str
    """Deployment role: 'p2p' | 'pd_prefill' | 'pd_decode'."""

    serve_host: str = "0.0.0.0"
    """Address the local ZMQ server binds to."""

    serve_port: int = 5200
    """Port the local ZMQ server listens on."""

    peers: list[PeerConfig] = field(default_factory=list)
    """Pre-configured peers. register_peer() can add more at runtime."""

    lookup_policy: str = "first_found"
    """Key resolution across peers: 'first_found' | 'round_robin'."""

    zmq_timeout_ms: int = 5000
    """Per-request ZMQ timeout in milliseconds."""

    remote_pin_ttl_s: int = 60
    """Server-side read-lock TTL. Unpin expires after this if UnpinRequest is lost."""

    reconnect_interval_s: int = 30
    """Interval between reconnect attempts for disconnected peers."""
