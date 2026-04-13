from pathlib import Path

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
