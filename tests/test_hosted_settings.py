from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from quarry.api.app import create_app
from quarry.config import Settings
from quarry.domain.models import HostedProviderPreset, HostedSettingsUpdateRequest, SessionState
from quarry.hosted_settings import (
    build_hosted_settings_envelope,
    load_provider_catalog,
    persist_hosted_settings,
)


def test_persist_hosted_settings_switches_to_openrouter_and_preserves_existing_key(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[runtime]
mode = "hybrid"

[hosted]
provider = "openai_compatible"
provider_preset = "openai"
llm_base_url = "https://api.openai.com/v1"
llm_api_key = "existing-key"
llm_model = "gpt-4.1-mini"
use_live_generation = true
"""
    )

    envelope = persist_hosted_settings(
        HostedSettingsUpdateRequest(
            provider_preset="openrouter",
            selected_model_id="stepfun/step-3.5-flash:free",
        ),
        config_path=config_path,
    )
    settings = Settings.from_env(config_path=config_path)
    saved_text = config_path.read_text()

    assert settings.llm_provider == "openai_compatible"
    assert settings.llm_base_url == "https://openrouter.ai/api/v1"
    assert settings.llm_model == "stepfun/step-3.5-flash:free"
    assert settings.llm_api_key == "existing-key"
    assert settings.use_live_generation is True
    assert envelope.settings.provider_preset == "openrouter"
    assert [provider.label for provider in envelope.providers] == [
        "OpenRouter",
        "Azure OpenAI",
        "Gemini",
        "Custom OpenAI-Compatible",
    ]
    assert 'provider_preset = "openrouter"' in saved_text


def test_persist_hosted_settings_writes_azure_specific_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"

    envelope = persist_hosted_settings(
        HostedSettingsUpdateRequest(
            provider_preset="azure_openai",
            selected_model_id="gpt-5.2-chat",
            api_key="azure-key",
            azure_base_url="https://team-resource.cognitiveservices.azure.com/openai/v1",
        ),
        config_path=config_path,
    )
    settings = Settings.from_env(config_path=config_path)
    saved_text = config_path.read_text()

    assert settings.llm_provider == "openai_compatible"
    assert settings.llm_base_url == "https://team-resource.cognitiveservices.azure.com/openai/v1"
    assert settings.llm_model == "gpt-5.2-chat"
    assert settings.llm_api_key == "azure-key"
    assert envelope.settings.azure_base_url == "https://team-resource.cognitiveservices.azure.com/openai/v1"
    assert envelope.settings.azure_deployment_name == "gpt-5.2-chat"
    assert 'azure_model_family = "gpt-5.2-chat"' in saved_text
    assert '[hosted.saved_providers.azure_openai]' in saved_text
    assert 'azure_base_url = "https://team-resource.cognitiveservices.azure.com/openai/v1"' in saved_text


def test_persist_hosted_settings_allows_azure_deployment_override(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"

    envelope = persist_hosted_settings(
        HostedSettingsUpdateRequest(
            provider_preset="azure_openai",
            selected_model_id="gpt-5.2-chat",
            api_key="azure-key",
            azure_base_url="https://team-resource.cognitiveservices.azure.com/openai/v1/",
            azure_deployment_name="gpt-5.2-chat",
        ),
        config_path=config_path,
    )
    settings = Settings.from_env(config_path=config_path)
    saved_text = config_path.read_text()

    assert settings.llm_base_url == "https://team-resource.cognitiveservices.azure.com/openai/v1"
    assert settings.llm_model == "gpt-5.2-chat"
    assert envelope.settings.azure_deployment_name == "gpt-5.2-chat"
    assert 'azure_deployment_name = "gpt-5.2-chat"' not in saved_text


def test_build_hosted_settings_envelope_reads_legacy_azure_resource_name(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[runtime]
mode = "hybrid"

[hosted]
provider = "openai_compatible"
provider_preset = "azure_openai"
llm_base_url = "https://team-resource.cognitiveservices.azure.com/openai/v1"
llm_api_key = "azure-key"
llm_model = "gpt-5.2-chat"
azure_resource_name = "team-resource"
azure_model_family = "gpt-5.2-chat"
use_live_generation = true
"""
    )

    envelope = build_hosted_settings_envelope(config_path)

    assert envelope.settings.azure_base_url == "https://team-resource.cognitiveservices.azure.com/openai/v1"


def test_persist_hosted_settings_normalizes_root_azure_uri(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"

    envelope = persist_hosted_settings(
        HostedSettingsUpdateRequest(
            provider_preset="azure_openai",
            selected_model_id="gpt-5.2-chat",
            api_key="azure-key",
            azure_base_url="https://team-resource.cognitiveservices.azure.com",
        ),
        config_path=config_path,
    )

    assert envelope.settings.azure_base_url == "https://team-resource.cognitiveservices.azure.com/openai/v1"
    assert Settings.from_env(config_path=config_path).llm_base_url == "https://team-resource.cognitiveservices.azure.com/openai/v1"


def test_persist_hosted_settings_remembers_saved_values_per_provider(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"

    persist_hosted_settings(
        HostedSettingsUpdateRequest(
            provider_preset="gemini",
            selected_model_id="gemini-3-flash-preview",
            api_key="gemini-key",
        ),
        config_path=config_path,
    )

    persist_hosted_settings(
        HostedSettingsUpdateRequest(
            provider_preset="openrouter",
            selected_model_id="qwen/qwen3.6-plus",
            api_key="openrouter-key",
        ),
        config_path=config_path,
    )

    envelope = persist_hosted_settings(
        HostedSettingsUpdateRequest(
            provider_preset="gemini",
            selected_model_id="gemini-3-flash-preview",
            api_key=None,
            clear_api_key=False,
        ),
        config_path=config_path,
    )
    settings = Settings.from_env(config_path=config_path)
    saved_text = config_path.read_text()

    assert settings.llm_provider == "gemini"
    assert settings.llm_model == "gemini-3-flash-preview"
    assert settings.llm_api_key == "gemini-key"
    assert envelope.settings.saved_provider_settings[HostedProviderPreset.GEMINI].api_key_configured is True
    assert envelope.settings.saved_provider_settings[HostedProviderPreset.OPENROUTER].api_key_configured is True
    assert '[hosted.saved_providers.gemini]' in saved_text
    assert '[hosted.saved_providers.openrouter]' in saved_text


def test_persist_hosted_settings_refuses_when_env_overrides_are_active(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.toml"
    monkeypatch.setenv("QUARRY_LLM_API_KEY", "env-key")

    with pytest.raises(RuntimeError, match="environment variables"):
        persist_hosted_settings(
            HostedSettingsUpdateRequest(
                provider_preset="openrouter",
                selected_model_id="qwen/qwen3.6-plus",
            ),
            config_path=config_path,
        )


def test_load_provider_catalog_uses_external_hosted_models_file(tmp_path: Path, monkeypatch) -> None:
    hosted_models_path = tmp_path / "hosted_models.toml"
    hosted_models_path.write_text(
        """
[providers.openrouter]
models = [
  { id = "custom/router-model", label = "Router Model", description = "Custom router model." },
]

[providers.azure_openai]
models = [
  { id = "custom/azure-model", label = "Azure Model", description = "Custom azure model." },
]

[providers.gemini]
models = [
  { id = "custom/gemini-model", label = "Gemini Model", description = "Custom gemini model." },
]

[providers.custom_openai_compatible]
models = []
"""
    )
    monkeypatch.setenv("QUARRY_HOSTED_MODELS_PATH", str(hosted_models_path))

    catalog = load_provider_catalog()
    openrouter = next(provider for provider in catalog if provider.preset == HostedProviderPreset.OPENROUTER)
    gemini = next(provider for provider in catalog if provider.preset == HostedProviderPreset.GEMINI)

    assert [model.id for model in openrouter.models] == ["custom/router-model"]
    assert openrouter.models[0].label == "Router Model"
    assert [model.id for model in gemini.models] == ["custom/gemini-model"]


def test_load_provider_catalog_requires_all_visible_provider_entries(tmp_path: Path, monkeypatch) -> None:
    hosted_models_path = tmp_path / "hosted_models.toml"
    hosted_models_path.write_text(
        """
[providers.openrouter]
models = [
  { id = "custom/router-model", label = "Router Model", description = "Custom router model." },
]
"""
    )
    monkeypatch.setenv("QUARRY_HOSTED_MODELS_PATH", str(hosted_models_path))

    with pytest.raises(ValueError, match=r"missing \[providers\.azure_openai\]"):
        load_provider_catalog()


def test_hosted_settings_api_updates_runtime_without_dropping_existing_sessions(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[runtime]
mode = "hybrid"

[hosted]
provider = "openai_compatible"
provider_preset = "openai"
llm_base_url = "https://api.openai.com/v1"
llm_api_key = "existing-key"
llm_model = "gpt-4.1-mini"
use_live_generation = true
"""
    )
    settings = Settings(
        corpus_dir=tmp_path / "corpus",
        artifacts_dir=tmp_path / "artifacts",
        use_local_models=False,
        runtime_mode="hybrid",
        llm_provider="openai_compatible",
        llm_base_url="https://api.openai.com/v1",
        llm_api_key="existing-key",
        llm_model="gpt-4.1-mini",
        use_live_generation=True,
    )
    settings.corpus_dir.mkdir(parents=True, exist_ok=True)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)

    app = create_app(settings, config_path=str(config_path))
    existing_store = app.state.pipeline_service.session_store
    existing_store.save(SessionState(session_id="existing-session", original_query="What changed?"))

    with TestClient(app) as client:
        response = client.get("/api/v1/settings/hosted")
        assert response.status_code == 200
        payload = response.json()
        assert payload["settings"]["provider_preset"] == "custom_openai_compatible"

        response = client.put(
            "/api/v1/settings/hosted",
            json={
                "provider_preset": "gemini",
                "selected_model_id": "gemini-3-flash-preview",
                "api_key": "gemini-key",
                "clear_api_key": False,
            },
        )

    assert response.status_code == 200
    assert app.state.pipeline_service.session_store is existing_store
    assert existing_store.get("existing-session").original_query == "What changed?"
    assert app.state.settings.llm_provider == "gemini"
    assert app.state.settings.llm_model == "gemini-3-flash-preview"
    assert build_hosted_settings_envelope(config_path).settings.provider_preset == "gemini"


def test_hosted_settings_api_returns_structured_validation_error(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    settings = Settings(
        corpus_dir=tmp_path / "corpus",
        artifacts_dir=tmp_path / "artifacts",
        use_local_models=False,
        runtime_mode="hybrid",
    )
    settings.corpus_dir.mkdir(parents=True, exist_ok=True)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(settings, config_path=str(config_path))

    with TestClient(app) as client:
        response = client.put(
            "/api/v1/settings/hosted",
            json={
                "provider_preset": "azure_openai",
                "selected_model_id": "gpt-5.2-chat",
                "azure_base_url": "https://invalid-host.example.com/v1",
            },
        )

    assert response.status_code == 422
    payload = response.json()
    assert payload["detail"]["code"] == "SETTINGS_VALIDATION_ERROR"
