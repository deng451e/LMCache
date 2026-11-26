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
        """Bytes the allocation and each of its slots must start and end on.

        Returns:
            A power of two; 1 when the resource needs no alignment.
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
