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
    """ZMQ REP socket port (lookup / Init / MemReg traffic)."""

    unpin_port: int
    """ZMQ PULL socket port (UnpinRequest fire-and-forget traffic)."""


@dataclass
class RemoteControllerConfig:
    """Configuration for RemoteController."""

    mode: str
    """Deployment role: 'p2p' | 'pd_prefill' | 'pd_decode'."""

    serve_host: str = "0.0.0.0"
    """Address the local ZMQ server binds to."""

    serve_port: int = 5200
    """Port the local ZMQ REP server listens on (lookup / Init / MemReg)."""

    serve_unpin_port: int = 5201
    """Port the local ZMQ PULL server listens on (UnpinRequest). Must differ
    from serve_port."""

    peers: list[PeerConfig] = field(default_factory=list)
    """Pre-configured peers. register_peer() can add more at runtime."""

    zmq_timeout_ms: int = 5000
    """Per-request ZMQ timeout in milliseconds for lookup traffic."""

    remote_pin_ttl_s: int = 60
    """Server-side read-lock TTL. Unpin expires after this if UnpinRequest
    is lost."""

    reconnect_interval_s: int = 30
    """Interval between reconnect attempts for disconnected peers."""
