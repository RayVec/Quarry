import pytest

from quarry.domain.models import SessionState
from quarry.services.session_store import SessionNotFoundError, SessionStore


def test_session_store_expires_sessions_after_ttl() -> None:
    store = SessionStore(ttl_minutes=1, max_sessions=10)
    session = SessionState(session_id="s1", original_query="q")
    store.save(session)

    # simulate ttl by forcing entry expiration through implementation detail:
    store._sessions["s1"].expires_at = store._now()  # type: ignore[attr-defined]

    with pytest.raises(SessionNotFoundError):
        store.get("s1")


def test_session_store_evicts_oldest_when_capacity_reached() -> None:
    store = SessionStore(ttl_minutes=10, max_sessions=2)
    store.save(SessionState(session_id="s1", original_query="q1"))
    store.save(SessionState(session_id="s2", original_query="q2"))
    store._sessions["s1"].expires_at = store._now()  # type: ignore[attr-defined]
    store._sessions["s2"].expires_at = store._now() + (store._ttl * 2)  # type: ignore[attr-defined]

    store.save(SessionState(session_id="s3", original_query="q3"))

    with pytest.raises(SessionNotFoundError):
        store.get("s1")
    assert store.get("s3").original_query == "q3"
