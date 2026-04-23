# SPDX-License-Identifier: Apache-2.0

"""
Configuration for distributed storage manager
"""

# Standard
from dataclasses import dataclass, field
from typing import Literal
import argparse

# First Party
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdaptersConfig,
    add_l2_adapters_args,
    parse_args_to_l2_adapters_config,
)
from lmcache.v1.distributed.remote_controller.config import (
    PeerConfig,
    RemoteControllerConfig,
)


@dataclass
class L1MemoryManagerConfig:
    """
    The configuration for L1 memory manager.
    """

    size_in_bytes: int
    """ The size of L1 memory in bytes. """

    use_lazy: bool
    """ Whether to use lazy initialization for L1 memory. """

    init_size_in_bytes: int = field(default=20 << 30)
    """ The initial size when using lazy allocation. Default is 20GB. """

    align_bytes: int = field(default=0x1000)
    """ The alignment size in bytes. Default is 4KB. """

    def __post_init__(self):
        self.init_size_in_bytes = min(self.init_size_in_bytes, self.size_in_bytes)


@dataclass
class L1ManagerConfig:
    """
    Special config for the L1 Object/Key manager
    """

    memory_config: L1MemoryManagerConfig
    """ The memory manager configuration for L1 cache. """

    write_ttl_seconds: int = field(default=600)
    """ Time to live for each object's write lock. Default is 600s (10 minutes). """

    read_ttl_seconds: int = field(default=300)
    """ Time to live for each object's read lock. Default is 300s (5 minutes). """


@dataclass
class EvictionConfig:
    """
    The configuration for eviction policies (L1 and optionally L2).
    """

    eviction_policy: Literal["LRU", "noop"]
    """ The eviction policy to use. """

    trigger_watermark: float = field(default=0.8)
    """ The memory usage watermark to trigger eviction (0.0 to 1.0). """

    eviction_ratio: float = field(default=0.2)
    """ The fraction of *allocated* memory to evict when triggered (0.0 to 1.0). """


@dataclass
class CxlConfig:
    """Configuration for the CXL shared memory cache tier.

    All hosts sharing the CXL region must use the same ``dax_device_path``
    and ``region_size``.  Each host owns a disjoint sub-region
    (``subregion_offset``, ``subregion_size``) for writes; any host can
    read any byte offset.
    """

    dax_device_path: str
    """Linux DAX device path, e.g. /dev/dax0.0 (same on all hosts)."""

    region_size: int
    """Full shared CXL region size in bytes (same on all hosts)."""

    subregion_offset: int
    """Byte offset of this host's owned sub-region within the global region."""

    subregion_size: int
    """Size of this host's owned sub-region in bytes."""

    serve_port: int = field(default=5300)
    """ZMQ REP port for CxlRemoteController init/lookup messages."""

    serve_unpin_port: int | None = field(default=None)
    """ZMQ PULL port for unpin fire-and-forget messages.
    Defaults to serve_port + 1 when None."""

    peers: list[PeerConfig] = field(default_factory=list)
    """Pre-configured CXL peers (peer_id, host, port) to connect to."""

    tiering_policy: str = field(default="always_cxl")
    """Tiering policy name: 'always_cxl' routes everything to CXL;
    'size_threshold' routes only large tensors."""

    align_bytes: int = field(default=0x1000)
    """CXL allocator alignment in bytes."""

    cxl_numa_node: int = field(default=-1)
    """NUMA node of the CXL device (-1 to disable NUMA binding)."""


@dataclass
class StorageManagerConfig:
    """
    The configuration for the distributed storage manager.
    """

    l1_manager_config: L1ManagerConfig
    """ The configuration for the L1 manager. """

    eviction_config: EvictionConfig
    """ The configuration for eviction policies. """

    l2_adapter_config: L2AdaptersConfig = field(
        default_factory=lambda: L2AdaptersConfig([])
    )
    """ The configuration for L2 adapters. """

    store_policy: str = "default"
    """ The L2 store policy name. """

    prefetch_policy: str = "default"
    """ The L2 prefetch policy name. """

    prefetch_max_in_flight: int = 8
    """ Maximum number of concurrent prefetch requests. """

    remote_controller_config: RemoteControllerConfig | None = None
    """ Optional remote controller config. None means no remote P2P/PD. """

    cxl_config: CxlConfig | None = None
    """Optional CXL shared memory tier config. None means CXL is disabled."""


def add_storage_manager_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """
    Add storage manager configuration arguments to an existing parser.

    This function allows other modules to integrate storage manager arguments
    into their own argument parsers. Arguments are organized into groups to
    avoid naming conflicts with other modules.

    Args:
        parser: The argument parser to add arguments to.

    Returns:
        argparse.ArgumentParser: The same parser with storage manager
            arguments added.

    Example:
        >>> # In another module that needs its own arguments
        >>> parser = argparse.ArgumentParser(description="My Application")
        >>> parser.add_argument("--my-arg", type=str)
        >>> add_storage_manager_args(parser)
        >>> args = parser.parse_args()
        >>> config = parse_args_to_config(args)
    """
    # L1 Memory Manager Config
    memory_group = parser.add_argument_group(
        "L1 Memory Manager", "Configuration for L1 memory manager"
    )
    memory_group.add_argument(
        "--l1-size-gb",
        type=float,
        required=True,
        help="The size of L1 memory in GB.",
    )
    memory_group.add_argument(
        "--l1-use-lazy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use lazy loading for L1 memory. (Default is True)",
    )
    memory_group.add_argument(
        "--l1-init-size-gb",
        type=int,
        default=20,
        help="The initial size (GB) when using lazy allocation. Default is 20.",
    )
    memory_group.add_argument(
        "--l1-align-bytes",
        type=int,
        default=4096,
        help="The alignment size in bytes. Default is 4KB (4096 bytes).",
    )

    # L1 Manager Config (TTL settings)
    ttl_group = parser.add_argument_group(
        "L1 Manager TTL", "TTL configuration for L1 manager locks"
    )
    ttl_group.add_argument(
        "--l1-write-ttl-seconds",
        type=int,
        default=600,
        help="Time to live for each object's write lock. Default is 600s.",
    )
    ttl_group.add_argument(
        "--l1-read-ttl-seconds",
        type=int,
        default=300,
        help="Time to live for each object's read lock. Default is 300s.",
    )

    # Eviction Config
    eviction_group = parser.add_argument_group(
        "Eviction Policy", "Configuration for eviction policies"
    )
    eviction_group.add_argument(
        "--eviction-policy",
        type=str,
        choices=["LRU", "noop"],
        required=True,
        help="The eviction policy to use ('LRU' or 'noop').",
    )
    eviction_group.add_argument(
        "--eviction-trigger-watermark",
        type=float,
        default=0.8,
        help="The memory usage watermark to trigger eviction (0.0 to 1.0). "
        "Default is 0.8.",
    )
    eviction_group.add_argument(
        "--eviction-ratio",
        type=float,
        default=0.2,
        help="The fraction of memory to evict when triggered (0.0 to 1.0). "
        "Default is 0.2.",
    )

    # L2 Policies
    # Import here to break circular dependency:
    # config.py <-> storage_controllers (via eviction_controller)
    # Safe because config.py is fully initialized by the time this
    # function is called.
    # First Party
    from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
        get_registered_prefetch_policies,
    )
    from lmcache.v1.distributed.storage_controllers.store_policy import (
        get_registered_store_policies,
    )
    import lmcache.v1.distributed.storage_controllers  # noqa: F401

    policy_group = parser.add_argument_group(
        "L2 Policies", "Store and prefetch policy selection for L2 adapters"
    )
    policy_group.add_argument(
        "--l2-store-policy",
        type=str,
        choices=get_registered_store_policies(),
        default="default",
        help="L2 store policy. Determines which adapters receive each key "
        "and whether keys are deleted from L1 after L2 store. "
        "Default is 'default' (store all keys to all adapters, keep L1).",
    )
    policy_group.add_argument(
        "--l2-prefetch-policy",
        type=str,
        choices=get_registered_prefetch_policies(),
        default="default",
        help="L2 prefetch policy. Determines which adapter loads each key "
        "when multiple adapters have it. "
        "Default is 'default' (pick the first adapter by index).",
    )
    policy_group.add_argument(
        "--l2-prefetch-max-in-flight",
        type=int,
        default=8,
        help="Maximum number of concurrent prefetch requests. Default is 8.",
    )

    # Remote Controller
    remote_group = parser.add_argument_group(
        "Remote Controller",
        "P2P / PD disaggregation via ZMQ + NIXL RDMA. Omit --remote-mode to disable.",
    )
    remote_group.add_argument(
        "--remote-mode",
        type=str,
        choices=["p2p", "pd_prefill", "pd_decode"],
        default=None,
        help="Enable remote KV-cache transfer. 'p2p' for symmetric sharing; "
        "'pd_prefill'/'pd_decode' for prefill-decode disaggregation.",
    )
    remote_group.add_argument(
        "--remote-serve-port",
        type=int,
        default=5200,
        help="ZMQ REP port this server listens on for lookup / Init / MemReg. "
        "Default: 5200.",
    )
    remote_group.add_argument(
        "--remote-serve-unpin-port",
        type=int,
        default=None,
        help="ZMQ PULL port for UnpinRequest (fire-and-forget). "
        "Default: --remote-serve-port + 1.",
    )
    remote_group.add_argument(
        "--remote-peer",
        type=str,
        action="append",
        default=[],
        metavar="ID:HOST:PORT",
        help="Pre-configure a remote peer. Format: peer_id:host:lookup_port. "
        "The unpin port is derived as lookup_port + 1. Repeatable.",
    )
    remote_group.add_argument(
        "--remote-zmq-timeout-ms",
        type=int,
        default=5000,
        help="Per-request ZMQ timeout in milliseconds. Default: 5000.",
    )

    # CXL shared memory tier
    cxl_group = parser.add_argument_group(
        "CXL Tier",
        "Shared CXL NUMA memory as an L2 cache tier.  "
        "Omit --cxl-dax-device to disable.",
    )
    cxl_group.add_argument(
        "--cxl-dax-device",
        type=str,
        default=None,
        metavar="PATH",
        help="Linux DAX device path (same on all hosts), e.g. /dev/dax0.0. "
        "A regular file works for testing (lmcache will mmap it). "
        "Omit to disable the CXL tier.",
    )
    cxl_group.add_argument(
        "--cxl-region-size-gb",
        type=float,
        default=4.0,
        help="Full shared CXL region size in GiB (same on all hosts). Default: 4.",
    )
    cxl_group.add_argument(
        "--cxl-subregion-offset-gb",
        type=float,
        default=0.0,
        help="Byte offset (in GiB) of this host's owned sub-region. Default: 0.",
    )
    cxl_group.add_argument(
        "--cxl-subregion-size-gb",
        type=float,
        default=2.0,
        help="Size (in GiB) of this host's owned sub-region. Default: 2.",
    )
    cxl_group.add_argument(
        "--cxl-serve-port",
        type=int,
        default=5300,
        help="ZMQ REP port for CXL init/lookup requests. Default: 5300.",
    )
    cxl_group.add_argument(
        "--cxl-serve-unpin-port",
        type=int,
        default=None,
        help="ZMQ PULL port for CXL unpin messages. Default: --cxl-serve-port + 1.",
    )
    cxl_group.add_argument(
        "--cxl-peer",
        type=str,
        action="append",
        default=[],
        metavar="ID:HOST:PORT",
        help="Pre-configure a CXL peer. Format: peer_id:host:lookup_port. "
        "The unpin port is derived as lookup_port + 1. Repeatable.",
    )
    cxl_group.add_argument(
        "--cxl-tiering-policy",
        type=str,
        choices=["always_cxl", "size_threshold"],
        default="always_cxl",
        help="Tiering policy: 'always_cxl' routes every allocation to CXL; "
        "'size_threshold' only routes large tensors. Default: always_cxl.",
    )
    cxl_group.add_argument(
        "--cxl-numa-node",
        type=int,
        default=-1,
        help="NUMA node of the CXL device for allocation affinity. -1 to disable. "
        "Default: -1.",
    )

    # Adapter config
    add_l2_adapters_args(parser)
    return parser


def get_arg_parser() -> argparse.ArgumentParser:
    """
    Get a standalone argument parser for storage manager configuration.

    This creates a new parser with only storage manager arguments.
    For integrating with other modules' parsers, use add_storage_manager_args()
    instead.

    Returns:
        argparse.ArgumentParser: The argument parser with all storage manager
            configuration options.
    """
    parser = argparse.ArgumentParser(
        description="Distributed Storage Manager Configuration"
    )
    return add_storage_manager_args(parser)


def parse_args_to_config(
    args: argparse.Namespace,
) -> StorageManagerConfig:
    """
    Convert parsed command line arguments to a StorageManagerConfig.

    Args:
        args: Parsed arguments from the argument parser.

    Returns:
        StorageManagerConfig: The configuration object.
    """
    memory_config = L1MemoryManagerConfig(
        size_in_bytes=int(args.l1_size_gb * (1 << 30)),
        use_lazy=args.l1_use_lazy,
        init_size_in_bytes=int(args.l1_init_size_gb * (1 << 30)),
        align_bytes=args.l1_align_bytes,
    )

    l1_manager_config = L1ManagerConfig(
        memory_config=memory_config,
        write_ttl_seconds=args.l1_write_ttl_seconds,
        read_ttl_seconds=args.l1_read_ttl_seconds,
    )

    eviction_config = EvictionConfig(
        eviction_policy=args.eviction_policy,
        trigger_watermark=args.eviction_trigger_watermark,
        eviction_ratio=args.eviction_ratio,
    )

    l2_adapter_config = parse_args_to_l2_adapters_config(args)

    remote_controller_config: RemoteControllerConfig | None = None
    if getattr(args, "remote_mode", None) is not None:
        serve_port: int = getattr(args, "remote_serve_port", 5200)
        serve_unpin_port: int = getattr(args, "remote_serve_unpin_port", None) or (
            serve_port + 1
        )
        peers: list[PeerConfig] = []
        for peer_str in getattr(args, "remote_peer", None) or []:
            parts = peer_str.split(":", 2)
            if len(parts) != 3:
                raise ValueError(
                    f"--remote-peer must be 'peer_id:host:port', got {peer_str!r}"
                )
            peer_id, host, port_str = parts
            port = int(port_str)
            peers.append(
                PeerConfig(peer_id=peer_id, host=host, port=port, unpin_port=port + 1)
            )
        remote_controller_config = RemoteControllerConfig(
            mode=args.remote_mode,
            serve_port=serve_port,
            serve_unpin_port=serve_unpin_port,
            peers=peers,
            zmq_timeout_ms=getattr(args, "remote_zmq_timeout_ms", 5000),
        )

    cxl_config: CxlConfig | None = None
    if getattr(args, "cxl_dax_device", None) is not None:
        cxl_serve_port: int = getattr(args, "cxl_serve_port", 5300)
        cxl_serve_unpin_port: int = getattr(args, "cxl_serve_unpin_port", None) or (
            cxl_serve_port + 1
        )
        cxl_peers: list[PeerConfig] = []
        for peer_str in getattr(args, "cxl_peer", None) or []:
            parts = peer_str.split(":", 2)
            if len(parts) != 3:
                raise ValueError(
                    f"--cxl-peer must be 'peer_id:host:port', got {peer_str!r}"
                )
            peer_id, host, port_str = parts
            port = int(port_str)
            cxl_peers.append(
                PeerConfig(peer_id=peer_id, host=host, port=port, unpin_port=port + 1)
            )
        cxl_config = CxlConfig(
            dax_device_path=args.cxl_dax_device,
            region_size=int(getattr(args, "cxl_region_size_gb", 4.0) * (1 << 30)),
            subregion_offset=int(
                getattr(args, "cxl_subregion_offset_gb", 0.0) * (1 << 30)
            ),
            subregion_size=int(getattr(args, "cxl_subregion_size_gb", 2.0) * (1 << 30)),
            serve_port=cxl_serve_port,
            serve_unpin_port=cxl_serve_unpin_port,
            peers=cxl_peers,
            tiering_policy=getattr(args, "cxl_tiering_policy", "always_cxl"),
            cxl_numa_node=getattr(args, "cxl_numa_node", -1),
        )

    return StorageManagerConfig(
        l1_manager_config=l1_manager_config,
        eviction_config=eviction_config,
        l2_adapter_config=l2_adapter_config,
        store_policy=args.l2_store_policy,
        prefetch_policy=args.l2_prefetch_policy,
        prefetch_max_in_flight=args.l2_prefetch_max_in_flight,
        remote_controller_config=remote_controller_config,
        cxl_config=cxl_config,
    )


def parse_args(args: list[str] | None = None) -> StorageManagerConfig:
    """
    Parse command line arguments and return a StorageManagerConfig.

    This is a convenience function that combines get_arg_parser() and
    parse_args_to_config().

    Args:
        args: Optional list of arguments to parse. If None, uses sys.argv.

    Returns:
        StorageManagerConfig: The configuration object.
    """
    parser = get_arg_parser()
    parsed_args = parser.parse_args(args)
    return parse_args_to_config(parsed_args)
