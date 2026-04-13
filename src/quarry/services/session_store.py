from __future__ import annotations

from datetime import UTC, datetime, timedelta

from quarry.domain.models import SessionState


class SessionNotFoundError(KeyError):
    """Raised when a session cannot be found."""


class SessionStoreEntry:
    __slots__ = ("session", "expires_at")

    def __init__(self, session: SessionState, expires_at: datetime) -> None:
        self.session = session
        self.expires_at = expires_at


class SessionStore:
    def __init__(self, *, ttl_minutes: int = 180, max_sessions: int = 2000) -> None:
        self._sessions: dict[str, SessionStoreEntry] = {}
        self._ttl = timedelta(minutes=ttl_minutes)
        self._max_sessions = max_sessions

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC)

    def _purge_expired(self) -> None:
        now = self._now()
        expired = [session_id for session_id, entry in self._sessions.items() if entry.expires_at <= now]
        for session_id in expired:
            del self._sessions[session_id]

    def _evict_if_needed(self) -> None:
        if len(self._sessions) < self._max_sessions:
            return
        oldest_session_id = min(self._sessions.keys(), key=lambda session_id: self._sessions[session_id].expires_at)
        del self._sessions[oldest_session_id]

    def save(self, session: SessionState) -> SessionState:
        self._purge_expired()
        self._evict_if_needed()
        self._sessions[session.session_id] = SessionStoreEntry(session=session, expires_at=self._now() + self._ttl)
        return session

    def get(self, session_id: str) -> SessionState:
        self._purge_expired()
        try:
            entry = self._sessions[session_id]
        except KeyError as exc:
            raise SessionNotFoundError(session_id) from exc
        entry.expires_at = self._now() + self._ttl
        return entry.session

    def delete(self, session_id: str) -> None:
        self._purge_expired()
        if session_id not in self._sessions:
            raise SessionNotFoundError(session_id)
        del self._sessions[session_id]

    def clear(self) -> None:
        self._sessions.clear()
