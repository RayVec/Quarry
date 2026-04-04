from __future__ import annotations

from quarry.domain.models import SessionState


class SessionNotFoundError(KeyError):
    """Raised when a session cannot be found."""


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def save(self, session: SessionState) -> SessionState:
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> SessionState:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise SessionNotFoundError(session_id) from exc

    def delete(self, session_id: str) -> None:
        if session_id not in self._sessions:
            raise SessionNotFoundError(session_id)
        del self._sessions[session_id]

    def clear(self) -> None:
        self._sessions.clear()
