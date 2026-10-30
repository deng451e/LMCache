# SPDX-License-Identifier: Apache-2.0
"""Storage-neutral registration contract for device staging buffers."""

# Standard
from typing import Protocol, Sequence
import threading

# Third Party
import torch

# First Party
from lmcache.v1.common.logging import init_logger

logger = init_logger(__name__)


class BufferRegistration(Protocol):
    """Register a staging allocation for an external transfer resource.

    The context owns the buffer and invokes both methods on its transfer
    stream. Implementations must undo partial registration before raising.
    """

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


class CompositeBufferRegistration(BufferRegistration):
    """Apply several registrations to one staging buffer.

    Members register in order and deregister in reverse. A failed register
    deregisters the members already registered, so no registration remains.

    Args:
        registrations: The registrations to apply, in order.
    """

    def __init__(self, registrations: Sequence[BufferRegistration]) -> None:
        self._registrations = tuple(registrations)

    @property
    def registrations(self) -> tuple[BufferRegistration, ...]:
        """The member registrations, in registration order."""
        return self._registrations

    def register(self, buffer: torch.Tensor) -> None:
        """Register ``buffer`` with every member, in order.

        Args:
            buffer: Contiguous device staging allocation, owned by the context.

        Raises:
            RuntimeError: If a member fails; the members registered before it
                are deregistered first.
        """
        registered: list[BufferRegistration] = []
        try:
            for registration in self._registrations:
                registration.register(buffer)
                registered.append(registration)
        except BaseException:
            for registration in reversed(registered):
                try:
                    registration.deregister(buffer)
                except Exception:
                    logger.exception("Rolling back a staging registration failed")
            raise

    def deregister(self, buffer: torch.Tensor) -> None:
        """Deregister ``buffer`` from every member, in reverse order.

        Every member is deregistered even if an earlier one fails.

        Args:
            buffer: The allocation previously passed to ``register``.

        Raises:
            RuntimeError: If any member failed to deregister.
        """
        failure: Exception | None = None
        for registration in reversed(self._registrations):
            try:
                registration.deregister(buffer)
            except Exception as error:
                logger.exception("Releasing a staging registration failed")
                failure = failure or error
        if failure is not None:
            raise RuntimeError("A staging registration was not released") from failure


# Alignment of every staging slot, in bytes; 1 means no alignment. A
# transfer target that reads straight into staging slots (the transfer
# channel under GPU-direct borrowing) sets it before cache contexts are
# created; contexts read it when they build their staging buffer.
_slot_alignment = 1
_slot_alignment_lock = threading.Lock()


def set_staging_slot_alignment(align_bytes: int) -> None:
    """Set the alignment new staging buffers give every slot.

    Args:
        align_bytes: A power of two; 1 removes the requirement.

    Raises:
        ValueError: If ``align_bytes`` is not a positive power of two.
    """
    if align_bytes < 1 or align_bytes & (align_bytes - 1):
        raise ValueError(
            f"Staging slot alignment must be a power of two, got {align_bytes}"
        )
    global _slot_alignment
    with _slot_alignment_lock:
        _slot_alignment = align_bytes


def staging_slot_alignment() -> int:
    """Return the alignment new staging buffers give every slot (1 = none)."""
    with _slot_alignment_lock:
        return _slot_alignment
