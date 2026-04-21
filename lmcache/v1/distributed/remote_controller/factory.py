# SPDX-License-Identifier: Apache-2.0
"""Factory for building RemoteController instances."""

# First Party
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.remote_controller.config import RemoteControllerConfig
from lmcache.v1.distributed.remote_controller.controller import (
    RemoteController,
    ZMQRemoteController,
)
from lmcache.v1.distributed.remote_io.adapter import RemoteIOAdapter


def build_remote_controller(
    config: RemoteControllerConfig,
    l1_manager: L1Manager,
    l1_mem_desc: L1MemoryDesc,
    io: RemoteIOAdapter,
) -> RemoteController:
    """Construct and return a RemoteController for the given configuration.

    Currently always returns a ZMQRemoteController. Future implementations
    may select different backends based on config.mode.

    Args:
        config:      Controller configuration (mode, peers, ports, etc.).
        l1_manager:  Local L1Manager for server-side pin management.
        l1_mem_desc: L1 memory descriptor providing align_bytes.
        io:          RemoteIOAdapter for peer connect/disconnect and lookups.

    Returns:
        Configured RemoteController (not yet started).
    """
    return ZMQRemoteController(
        config=config,
        l1_manager=l1_manager,
        l1_mem_desc=l1_mem_desc,
        io=io,
    )
