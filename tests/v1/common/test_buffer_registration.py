# SPDX-License-Identifier: Apache-2.0
"""Composite staging-buffer registration and the staging slot alignment."""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.common.buffer_registration import (
    CompositeBufferRegistration,
    set_staging_slot_alignment,
    staging_slot_alignment,
)


class _Recorder:
    """A registration that logs its calls and can fail on demand."""

    def __init__(
        self,
        name: str,
        log: list[tuple[str, str]],
        fail_register: bool = False,
        fail_deregister: bool = False,
    ) -> None:
        self._name = name
        self._log = log
        self._fail_register = fail_register
        self._fail_deregister = fail_deregister

    def register(self, buffer: torch.Tensor) -> None:
        self._log.append(("register", self._name))
        if self._fail_register:
            raise RuntimeError(f"{self._name} register failed")

    def deregister(self, buffer: torch.Tensor) -> None:
        self._log.append(("deregister", self._name))
        if self._fail_deregister:
            raise RuntimeError(f"{self._name} deregister failed")


def test_composite_registers_in_order_and_deregisters_in_reverse() -> None:
    log: list[tuple[str, str]] = []
    composite = CompositeBufferRegistration([_Recorder("a", log), _Recorder("b", log)])
    buffer = torch.empty(8, dtype=torch.uint8)
    composite.register(buffer)
    composite.deregister(buffer)
    assert log == [
        ("register", "a"),
        ("register", "b"),
        ("deregister", "b"),
        ("deregister", "a"),
    ]


def test_composite_rolls_back_earlier_registrations_when_one_fails() -> None:
    log: list[tuple[str, str]] = []
    composite = CompositeBufferRegistration(
        [_Recorder("a", log), _Recorder("b", log, fail_register=True)]
    )
    with pytest.raises(RuntimeError, match="b register failed"):
        composite.register(torch.empty(8, dtype=torch.uint8))
    assert log == [("register", "a"), ("register", "b"), ("deregister", "a")]


def test_composite_deregisters_every_member_even_after_a_failure() -> None:
    log: list[tuple[str, str]] = []
    composite = CompositeBufferRegistration(
        [_Recorder("a", log), _Recorder("b", log, fail_deregister=True)]
    )
    buffer = torch.empty(8, dtype=torch.uint8)
    composite.register(buffer)
    with pytest.raises(RuntimeError):
        composite.deregister(buffer)
    assert log[-2:] == [("deregister", "b"), ("deregister", "a")]


def test_staging_slot_alignment_defaults_to_one_and_can_be_set_and_reset() -> None:
    assert staging_slot_alignment() == 1
    set_staging_slot_alignment(65536)
    try:
        assert staging_slot_alignment() == 65536
    finally:
        set_staging_slot_alignment(1)
    assert staging_slot_alignment() == 1


@pytest.mark.parametrize("bad", [0, -4, 3, 65535])
def test_staging_slot_alignment_rejects_non_powers_of_two(bad: int) -> None:
    with pytest.raises(ValueError):
        set_staging_slot_alignment(bad)
