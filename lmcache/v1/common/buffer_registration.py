# SPDX-License-Identifier: Apache-2.0
"""Storage-neutral registration contract for device staging buffers."""

# Standard
from typing import Protocol

# Third Party
import torch


class BufferRegistration(Protocol):
    """Register a staging allocation for an external transfer resource.

    The context owns the buffer and invokes both methods on its transfer
    stream. Implementations must undo partial registration before raising.
    """

    def staging_alignment(self) -> int:
        """Slot alignment a context applies when it creates a staging buffer
        for this registration (``create_staging_buffer``): the buffer and
        each of its slots start and end on it.

        Returns:
            A power of two. The default of 1 asks for no such layout; it says
            nothing about alignment the resource handles itself.
        """
        return 1

    def register(self, buffer: torch.Tensor) -> None:
        """Register the full staging allocation before its first transfer.

        Args:
            buffer: Contiguous device staging allocation, owned by the context.

        Raises:
            RuntimeError: If registration fails; no registration may remain.
        """
        ...

    def deregister(self, buffer: torch.Tensor) -> None:
        """Release a successful registration after transfers have drained.

        Args:
            buffer: The allocation previously passed to ``register``.

        Raises:
            RuntimeError: If the registration cannot be released safely.
        """
        ...
