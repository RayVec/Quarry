from pathlib import Path
from time import sleep

from fastapi.testclient import TestClient

from quarry.api.app import create_app
from quarry.config import Settings


def _build_settings(tmp_path: Path) -> Settings:
    settings = Settings(
        corpus_dir=tmp_path / "corpus",
        artifacts_dir=tmp_path / "artifacts",
        runtime_mode="hybrid",
        use_local_models=False,
    )
    settings.corpus_dir.mkdir(parents=True, exist_ok=True)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return settings


def test_session_not_found_uses_structured_error(tmp_path: Path) -> None:
    app = create_app(_build_settings(tmp_path))
    with TestClient(app) as client:
        response = client.get("/api/v1/sessions/missing")
    assert response.status_code == 404
    payload = response.json()
    assert payload["detail"]["code"] == "SESSION_NOT_FOUND"
    assert payload["detail"]["message"] == "Session not found."


def test_scoped_retrieval_validates_sentence_index(tmp_path: Path) -> None:
    app = create_app(_build_settings(tmp_path))
    with TestClient(app) as client:
        response = client.post("/api/v1/sessions/s1/citations/1/scoped", json={"sentence_index": -1})
    assert response.status_code == 422


def test_message_run_not_found_uses_structured_error(tmp_path: Path) -> None:
    app = create_app(_build_settings(tmp_path))
    with TestClient(app) as client:
        response = client.get("/api/v1/message-runs/missing")
    assert response.status_code == 404
    payload = response.json()
    assert payload["detail"]["code"] == "MESSAGE_RUN_NOT_FOUND"
    assert payload["detail"]["message"] == "Message run not found."


def test_start_message_returns_orchestration_run_immediately(tmp_path: Path) -> None:
    app = create_app(_build_settings(tmp_path))
    with TestClient(app) as client:
        response = client.post("/api/v1/messages/start", json={"message": "What is FEED maturity?"})
        assert response.status_code == 200
        payload = response.json()
        message_run = payload["message_run"]
        assert message_run["status"] == "running"
        assert message_run["stage"] == "orchestrating"

        message_run_id = message_run["message_run_id"]
        followup = None
        for _ in range(10):
            sleep(0.05)
            next_response = client.get(f"/api/v1/message-runs/{message_run_id}")
            assert next_response.status_code == 200
            followup = next_response.json()["message_run"]
            if followup.get("session") is not None or followup.get("assistant_turn") is not None:
                break

        assert followup is not None
        assert followup.get("session") is not None or followup.get("assistant_turn") is not None
