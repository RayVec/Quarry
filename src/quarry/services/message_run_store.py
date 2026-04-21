from __future__ import annotations

from datetime import UTC, datetime, timedelta

from quarry.domain.models import MessageRunState


class MessageRunNotFoundError(KeyError):
    """Raised when a message run cannot be found."""


class MessageRunStoreEntry:
    __slots__ = ("message_run", "expires_at")

    def __init__(self, message_run: MessageRunState, expires_at: datetime) -> None:
        self.message_run = message_run
        self.expires_at = expires_at


class MessageRunStore:
    def __init__(self, *, ttl_minutes: int = 180, max_runs: int = 2000) -> None:
        self._runs: dict[str, MessageRunStoreEntry] = {}
        self._ttl = timedelta(minutes=ttl_minutes)
        self._max_runs = max_runs

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC)

    def _purge_expired(self) -> None:
        now = self._now()
        expired = [run_id for run_id, entry in self._runs.items() if entry.expires_at <= now]
        for run_id in expired:
            del self._runs[run_id]

    def _evict_if_needed(self) -> None:
        if len(self._runs) < self._max_runs:
            return
        oldest_run_id = min(self._runs.keys(), key=lambda run_id: self._runs[run_id].expires_at)
        del self._runs[oldest_run_id]

    def save(self, message_run: MessageRunState) -> MessageRunState:
        self._purge_expired()
        self._evict_if_needed()
        self._runs[message_run.message_run_id] = MessageRunStoreEntry(
            message_run=message_run,
            expires_at=self._now() + self._ttl,
        )
        return message_run

    def get(self, message_run_id: str) -> MessageRunState:
        self._purge_expired()
        try:
            entry = self._runs[message_run_id]
        except KeyError as exc:
            raise MessageRunNotFoundError(message_run_id) from exc
        entry.expires_at = self._now() + self._ttl
        return entry.message_run

    def delete(self, message_run_id: str) -> None:
        self._purge_expired()
        if message_run_id not in self._runs:
            raise MessageRunNotFoundError(message_run_id)
        del self._runs[message_run_id]
