# SPDX-License-Identifier: Apache-2.0
"""Peer clients shared by several adapters: removed only by the last user."""

# Standard
from unittest.mock import MagicMock

# First Party
from lmcache.v1.distributed.transfer_channel import (
    acquire_transfer_channel_client,
    release_transfer_channel_client,
)


def _context() -> MagicMock:
    context = MagicMock()
    context.get_transfer_channel_client.side_effect = lambda url: f"client:{url}"
    return context


def test_client_is_removed_only_after_its_last_user_releases_it() -> None:
    context = _context()
    first = acquire_transfer_channel_client(context, "peer:1")
    second = acquire_transfer_channel_client(context, "peer:1")
    assert first == second == "client:peer:1"

    release_transfer_channel_client(context, "peer:1")
    context.remove_transfer_channel_client.assert_not_called()

    release_transfer_channel_client(context, "peer:1")
    context.remove_transfer_channel_client.assert_called_once_with("peer:1")


def test_users_of_different_peers_do_not_share_a_count() -> None:
    context = _context()
    acquire_transfer_channel_client(context, "peer:1")
    acquire_transfer_channel_client(context, "peer:2")
    release_transfer_channel_client(context, "peer:1")
    context.remove_transfer_channel_client.assert_called_once_with("peer:1")
    release_transfer_channel_client(context, "peer:2")
    assert context.remove_transfer_channel_client.call_count == 2


def test_a_peer_acquired_again_after_removal_gets_a_fresh_count() -> None:
    context = _context()
    acquire_transfer_channel_client(context, "peer:1")
    release_transfer_channel_client(context, "peer:1")
    acquire_transfer_channel_client(context, "peer:1")
    release_transfer_channel_client(context, "peer:1")
    assert context.remove_transfer_channel_client.call_count == 2


def test_releasing_an_unacquired_peer_removes_nothing() -> None:
    context = _context()
    release_transfer_channel_client(context, "peer:9")
    context.remove_transfer_channel_client.assert_not_called()
