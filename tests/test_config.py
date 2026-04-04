import pytest

from quarry.config import Settings


def test_settings_from_env_defaults_to_apple_profile_on_apple_silicon(monkeypatch) -> None:
    monkeypatch.delenv("QUARRY_RUNTIME_PROFILE", raising=False)
    monkeypatch.setattr("quarry.config.is_apple_silicon_host", lambda: True)

    settings = Settings.from_env()

    assert settings.runtime_profile == "apple_lite_mlx"


def test_settings_from_env_respects_explicit_runtime_profile(monkeypatch) -> None:
    monkeypatch.setenv("QUARRY_RUNTIME_PROFILE", "full_local_transformers")
    monkeypatch.setattr("quarry.config.is_apple_silicon_host", lambda: True)

    settings = Settings.from_env()

    assert settings.runtime_profile == "full_local_transformers"
    monkeypatch.delenv("QUARRY_RUNTIME_PROFILE", raising=False)


def test_legacy_live_llm_flag_only_enables_generation(monkeypatch) -> None:
    monkeypatch.setenv("QUARRY_USE_LIVE_LLM", "1")
    monkeypatch.delenv("QUARRY_USE_LIVE_GENERATION", raising=False)
    monkeypatch.delenv("QUARRY_USE_LIVE_DECOMPOSITION", raising=False)
    monkeypatch.delenv("QUARRY_USE_LIVE_METADATA_ENRICHMENT", raising=False)

    settings = Settings.from_env()

    assert settings.use_live_generation is True
    assert settings.use_live_decomposition is False
    assert settings.use_live_metadata_enrichment is False


def test_explicit_generation_flag_overrides_legacy_live_llm_alias(monkeypatch) -> None:
    monkeypatch.setenv("QUARRY_USE_LIVE_LLM", "1")
    monkeypatch.setenv("QUARRY_USE_LIVE_GENERATION", "0")

    settings = Settings.from_env()

    assert settings.use_live_generation is False


def test_legacy_runtime_mode_names_are_rejected(monkeypatch) -> None:
    monkeypatch.setenv("QUARRY_RUNTIME_MODE", "degraded_local")

    with pytest.raises(ValueError, match="Unsupported QUARRY runtime mode"):
        Settings.from_env()


def test_settings_from_toml_config_file(tmp_path) -> None:
    config_path = tmp_path / "quarry.local.toml"
    config_path.write_text(
        """
[runtime]
mode = "hybrid"
profile = "apple_lite_mlx"

[hosted]
llm_base_url = "https://openrouter.ai/api/v1"
llm_api_key = "secret"
llm_model = "stepfun/step-3.5-flash:free"
use_live_generation = true
use_live_decomposition = false
"""
    )

    settings = Settings.from_env(config_path=config_path)

    assert settings.runtime_mode == "hybrid"
    assert settings.runtime_profile == "apple_lite_mlx"
    assert settings.llm_base_url == "https://openrouter.ai/api/v1"
    assert settings.llm_model == "stepfun/step-3.5-flash:free"
    assert settings.use_live_generation is True
    assert settings.use_live_decomposition is False


def test_env_vars_override_toml_config(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "quarry.local.toml"
    config_path.write_text(
        """
[runtime]
mode = "hybrid"

[hosted]
llm_model = "stepfun/step-3.5-flash:free"
"""
    )
    monkeypatch.setenv("QUARRY_RUNTIME_MODE", "local")
    monkeypatch.setenv("QUARRY_LLM_MODEL", "override-model")

    settings = Settings.from_env(config_path=config_path)

    assert settings.runtime_mode == "local"
    assert settings.llm_model == "override-model"
