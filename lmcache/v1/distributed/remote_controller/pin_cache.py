# SPDX-License-Identifier: Apache-2.0
"""Thread-safe TTL cache for pinned-key dedup.

Used by ZMQRemoteController and CxlRemoteController to avoid double-pinning
on retry and to reclaim l2_lock_count increments from crashed clients.
"""

# Standard
from dataclasses import dataclass
from typing import Any
import threading
import time


@dataclass
class PinEntry:
    """One server-side cached pin entry."""

    request_id: str
    """Stable across retries; used as the cache key."""

    found_keys: list
    """list[WireObjectKey] — keys that were locked on the server side."""

    response: Any
    """The cached response object (type depends on caller)."""

    expires_at: float
    """time.monotonic() deadline; entry is swept when now >= expires_at."""


class PinCache:
    """Thread-safe TTL cache for pinned-key deduplication.

    Args:
        ttl_s: Seconds until an entry expires after insertion.
    """

    def __init__(self, ttl_s: int) -> None:
        self._ttl_s = ttl_s
        self._cache: dict[str, PinEntry] = {}
        self._lock = threading.Lock()

    def get(self, request_id: str) -> PinEntry | None:
        """Return the entry for request_id if present and not expired.

        Args:
            request_id: The lookup key.

        Returns:
            PinEntry if found and unexpired, else None.
        """
        with self._lock:
            entry = self._cache.get(request_id)
            if entry is None:
                return None
            if entry.expires_at <= time.monotonic():
                return None
            return entry

    def put(self, entry: PinEntry) -> None:
        """Insert or overwrite the entry for entry.request_id.

        Args:
            entry: Entry to store.
        """
        with self._lock:
            self._cache[entry.request_id] = entry

    def pop(self, request_id: str) -> PinEntry | None:
        """Remove and return the entry for request_id.

        Args:
            request_id: The lookup key.

        Returns:
            The removed PinEntry, or None if not present.
        """
        with self._lock:
            return self._cache.pop(request_id, None)

    def sweep_expired(self) -> list[PinEntry]:
        """Remove and return all entries whose expires_at <= now.

        Returns:
            List of expired entries (may be empty).
        """
        now = time.monotonic()
        expired: list[PinEntry] = []
        with self._lock:
            expired_ids = [rid for rid, e in self._cache.items() if e.expires_at <= now]
            for rid in expired_ids:
                expired.append(self._cache.pop(rid))
        return expired
