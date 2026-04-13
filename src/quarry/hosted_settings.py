from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import tomllib
from typing import cast

from quarry.config import Settings, load_raw_file_config, resolve_config_path, write_raw_file_config
from quarry.domain.models import (
    HostedEnvOverride,
    HostedModelOption,
    HostedProviderDescriptor,
    HostedProviderPreset,
    HostedSavedProviderState,
    HostedSettingsEnvelope,
    HostedSettingsState,
    HostedSettingsUpdateRequest,
)
from quarry.hosted_auth import is_azure_openai_base_url, normalize_azure_openai_base_url

BASE_VISIBLE_PROVIDER_CATALOG = [
    HostedProviderDescriptor(
        preset=HostedProviderPreset.OPENROUTER,
        label="OpenRouter",
        provider_family="openai_compatible",
        description="OpenRouter's OpenAI-compatible gateway.",
        model_label="Model",
        models=[],
        supports_custom_model=False,
    ),
    HostedProviderDescriptor(
        preset=HostedProviderPreset.AZURE_OPENAI,
        label="Azure OpenAI",
        provider_family="openai_compatible",
        description="Azure OpenAI through the OpenAI v1-compatible endpoint.",
        model_label="Model",
        requires_base_url=True,
        models=[],
        supports_custom_model=False,
    ),
    HostedProviderDescriptor(
        preset=HostedProviderPreset.GEMINI,
        label="Gemini",
        provider_family="gemini",
        description="Google AI Studio Gemini API.",
        model_label="Model",
        models=[],
        supports_custom_model=False,
    ),
    HostedProviderDescriptor(
        preset=HostedProviderPreset.CUSTOM_OPENAI_COMPATIBLE,
        label="Custom OpenAI-Compatible",
        provider_family="openai_compatible",
        description="Any OpenAI-compatible provider with a custom base URL.",
        model_label="Model",
        requires_base_url=True,
        models=[],
    ),
]

LEGACY_PROVIDER_DESCRIPTORS = {
    HostedProviderPreset.OPENAI: HostedProviderDescriptor(
        preset=HostedProviderPreset.OPENAI,
        label="OpenAI",
        provider_family="openai_compatible",
        description="Legacy direct OpenAI preset kept only for backward compatibility.",
        model_label="Model",
        models=[],
    ),
}


def resolve_hosted_models_path(config_path: str | Path | None = None) -> Path:
    env_path = os.getenv("QUARRY_HOSTED_MODELS_PATH")
    if env_path:
        return Path(env_path)

    if config_path is not None:
        sibling = resolve_config_path(config_path).with_name("hosted_models.toml")
        if sibling.exists():
            return sibling

    return Path("hosted_models.toml")


def _load_raw_hosted_models_config(config_path: str | Path | None = None) -> dict[str, object]:
    resolved = resolve_hosted_models_path(config_path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Hosted model catalog file not found: {resolved}. "
            "Create hosted_models.toml to define supported provider models."
        )
    payload = tomllib.loads(resolved.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Hosted model catalog must be a TOML table: {resolved}.")
    return payload


def _model_options_from_raw(value: object, *, provider_name: str) -> list[HostedModelOption]:
    if not isinstance(value, list):
        raise ValueError(f"[providers.{provider_name}] models must be a TOML array.")
    options: list[HostedModelOption] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"[providers.{provider_name}] models must contain inline tables.")
        model_id = item.get("id")
        label = item.get("label")
        description = item.get("description", "")
        if not isinstance(model_id, str) or not isinstance(label, str):
            raise ValueError(
                f"[providers.{provider_name}] each model must define string id and label fields."
            )
        if not isinstance(description, str):
            raise ValueError(
                f"[providers.{provider_name}] description must be a string."
            )
        options.append(
            HostedModelOption(
                id=model_id,
                label=label,
                description=description,
            )
        )
    return options


def load_provider_catalog(config_path: str | Path | None = None) -> list[HostedProviderDescriptor]:
    resolved = resolve_hosted_models_path(config_path)
    raw_catalog = _load_raw_hosted_models_config(config_path)
    raw_providers = raw_catalog.get("providers", {})
    if not isinstance(raw_providers, dict):
        raise ValueError(f"Hosted model catalog must define a [providers] table: {resolved}.")
    provider_overrides = raw_providers

    catalog: list[HostedProviderDescriptor] = []
    for base_descriptor in BASE_VISIBLE_PROVIDER_CATALOG:
        descriptor = base_descriptor.model_copy(deep=True)
        override = provider_overrides.get(base_descriptor.preset.value)
        if not isinstance(override, dict):
            raise ValueError(
                f"Hosted model catalog is missing [providers.{base_descriptor.preset.value}] in {resolved}."
            )
        descriptor.models = _model_options_from_raw(
            override.get("models"),
            provider_name=base_descriptor.preset.value,
        )
        if not descriptor.supports_custom_model and not descriptor.models:
            raise ValueError(
                f"[providers.{base_descriptor.preset.value}] must define at least one model in {resolved}."
            )
        catalog.append(descriptor)
    return catalog


def provider_catalog_by_preset(
    config_path: str | Path | None = None,
) -> dict[HostedProviderPreset, HostedProviderDescriptor]:
    return {
        **{
            preset: descriptor.model_copy(deep=True)
            for preset, descriptor in LEGACY_PROVIDER_DESCRIPTORS.items()
        },
        **{descriptor.preset: descriptor for descriptor in load_provider_catalog(config_path)},
    }


def _normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None


def _raw_hosted_section(config_path: str | Path | None = None) -> dict[str, object]:
    raw = load_raw_file_config(config_path)
    hosted = raw.get("hosted", {})
    return hosted if isinstance(hosted, dict) else {}


def _raw_saved_provider_sections(raw_hosted: dict[str, object]) -> dict[HostedProviderPreset, dict[str, object]]:
    raw_saved = raw_hosted.get("saved_providers", {})
    if not isinstance(raw_saved, dict):
        return {}
    sections: dict[HostedProviderPreset, dict[str, object]] = {}
    for preset_value, payload in raw_saved.items():
        if not isinstance(preset_value, str) or not isinstance(payload, dict):
            continue
        try:
            preset = HostedProviderPreset(preset_value)
        except ValueError:
            continue
        sections[preset] = dict(payload)
    return sections


def _legacy_azure_base_url(resource_name: str | None) -> str | None:
    normalized = _normalize_text(resource_name)
    if not normalized:
        return None
    return f"https://{normalized}.cognitiveservices.azure.com/openai/v1"


def _infer_provider_preset(settings: Settings, raw_hosted: dict[str, object]) -> HostedProviderPreset:
    preset_value = raw_hosted.get("provider_preset")
    if isinstance(preset_value, str):
        if preset_value == HostedProviderPreset.OPENAI.value:
            return HostedProviderPreset.CUSTOM_OPENAI_COMPATIBLE
        try:
            preset = HostedProviderPreset(preset_value)
            if preset == HostedProviderPreset.GEMINI and settings.llm_provider == "gemini":
                return preset
            if (
                preset in {
                    HostedProviderPreset.OPENROUTER,
                    HostedProviderPreset.AZURE_OPENAI,
                    HostedProviderPreset.CUSTOM_OPENAI_COMPATIBLE,
                }
                and settings.llm_provider == "openai_compatible"
            ):
                return preset
        except ValueError:
            pass

    if settings.llm_provider == "gemini":
        return HostedProviderPreset.GEMINI

    base_url = (settings.llm_base_url or "").rstrip("/")
    if is_azure_openai_base_url(base_url):
        return HostedProviderPreset.AZURE_OPENAI
    if base_url == "https://openrouter.ai/api/v1":
        return HostedProviderPreset.OPENROUTER
    if base_url == "https://api.openai.com/v1":
        return HostedProviderPreset.CUSTOM_OPENAI_COMPATIBLE
    if base_url:
        return HostedProviderPreset.CUSTOM_OPENAI_COMPATIBLE
    return HostedProviderPreset.OPENROUTER


def _model_selection(descriptor: HostedProviderDescriptor, value: str | None) -> tuple[str | None, str | None]:
    normalized = _normalize_text(value)
    if not normalized:
        if descriptor.models:
            return descriptor.models[0].id, None
        if descriptor.supports_custom_model:
            return "custom", None
        return None, None
    if any(option.id == normalized for option in descriptor.models):
        return normalized, None
    if descriptor.supports_custom_model:
        return "custom", normalized
    if descriptor.models:
        return descriptor.models[0].id, None
    return None, None


def _active_provider_profile_payload(
    settings: Settings,
    raw_hosted: dict[str, object],
) -> dict[str, object]:
    profile: dict[str, object] = {}
    if settings.llm_base_url:
        profile["llm_base_url"] = settings.llm_base_url
    if settings.llm_api_key:
        profile["llm_api_key"] = settings.llm_api_key
    if settings.llm_model:
        profile["llm_model"] = settings.llm_model
    for key in ("azure_base_url", "azure_deployment_name", "azure_model_family", "custom_base_url"):
        value = _normalize_text(cast(str | None, raw_hosted.get(key)))
        if value:
            profile[key] = value
    return profile


def _build_saved_provider_state(
    descriptor: HostedProviderDescriptor,
    profile: dict[str, object],
) -> HostedSavedProviderState:
    llm_model = _normalize_text(cast(str | None, profile.get("llm_model")))
    llm_base_url = _normalize_text(cast(str | None, profile.get("llm_base_url")))

    if descriptor.preset == HostedProviderPreset.AZURE_OPENAI:
        selected_model_id, custom_model_id = _model_selection(
            descriptor,
            _normalize_text(cast(str | None, profile.get("azure_model_family"))),
        )
        normalized_azure_base_url = normalize_azure_openai_base_url(
            _normalize_text(cast(str | None, profile.get("azure_base_url")))
            or llm_base_url
            or _legacy_azure_base_url(cast(str | None, profile.get("azure_resource_name")))
            or ""
        )
        return HostedSavedProviderState(
            api_key_configured=bool(_normalize_text(cast(str | None, profile.get("llm_api_key")))),
            selected_model_id=selected_model_id,
            custom_model_id=custom_model_id,
            azure_base_url=normalized_azure_base_url,
            azure_deployment_name=_normalize_text(cast(str | None, profile.get("azure_deployment_name")))
            or llm_model,
            azure_model_family=custom_model_id or selected_model_id,
        )

    if descriptor.preset == HostedProviderPreset.CUSTOM_OPENAI_COMPATIBLE:
        selected_model_id, custom_model_id = _model_selection(descriptor, llm_model)
        return HostedSavedProviderState(
            api_key_configured=bool(_normalize_text(cast(str | None, profile.get("llm_api_key")))),
            selected_model_id=selected_model_id,
            custom_model_id=custom_model_id,
            custom_base_url=_normalize_text(cast(str | None, profile.get("custom_base_url")))
            or llm_base_url,
        )

    selected_model_id, custom_model_id = _model_selection(descriptor, llm_model)
    return HostedSavedProviderState(
        api_key_configured=bool(_normalize_text(cast(str | None, profile.get("llm_api_key")))),
        selected_model_id=selected_model_id,
        custom_model_id=custom_model_id,
    )


def detect_hosted_env_overrides(settings: Settings | None = None) -> list[HostedEnvOverride]:
    current_provider = settings.llm_provider if settings is not None else None
    overrides: list[HostedEnvOverride] = []

    if os.getenv("QUARRY_LLM_PROVIDER"):
        overrides.append(HostedEnvOverride(field="provider", env_var="QUARRY_LLM_PROVIDER"))
    if os.getenv("QUARRY_HOSTED_PROVIDER"):
        overrides.append(HostedEnvOverride(field="provider", env_var="QUARRY_HOSTED_PROVIDER"))
    if os.getenv("QUARRY_LLM_BASE_URL"):
        overrides.append(HostedEnvOverride(field="base_url", env_var="QUARRY_LLM_BASE_URL"))
    if os.getenv("QUARRY_LLM_MODEL"):
        overrides.append(HostedEnvOverride(field="model", env_var="QUARRY_LLM_MODEL"))
    if os.getenv("QUARRY_USE_LIVE_GENERATION"):
        overrides.append(HostedEnvOverride(field="use_live_generation", env_var="QUARRY_USE_LIVE_GENERATION"))
    if os.getenv("QUARRY_USE_LIVE_LLM"):
        overrides.append(HostedEnvOverride(field="use_live_generation", env_var="QUARRY_USE_LIVE_LLM"))
    if os.getenv("QUARRY_LLM_API_KEY"):
        overrides.append(HostedEnvOverride(field="api_key", env_var="QUARRY_LLM_API_KEY"))
    if current_provider == "gemini":
        if os.getenv("QUARRY_GEMINI_API_KEY"):
            overrides.append(HostedEnvOverride(field="api_key", env_var="QUARRY_GEMINI_API_KEY"))
        if os.getenv("GEMINI_API_KEY"):
            overrides.append(HostedEnvOverride(field="api_key", env_var="GEMINI_API_KEY"))

    deduped: list[HostedEnvOverride] = []
    seen: set[tuple[str, str]] = set()
    for override in overrides:
        key = (override.field, override.env_var)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(override)
    return deduped


def build_hosted_settings_envelope(config_path: str | Path | None = None) -> HostedSettingsEnvelope:
    resolved_config_path = resolve_config_path(config_path)
    settings = Settings.from_env(config_path=resolved_config_path)
    raw_hosted = _raw_hosted_section(resolved_config_path)
    raw_saved_provider_sections = _raw_saved_provider_sections(raw_hosted)
    visible_provider_catalog = load_provider_catalog(resolved_config_path)
    provider_by_preset = provider_catalog_by_preset(resolved_config_path)
    preset = _infer_provider_preset(settings, raw_hosted)
    descriptor = provider_by_preset[preset]
    env_overrides = detect_hosted_env_overrides(settings)
    notices: list[str] = []

    if env_overrides:
        env_var_names = ", ".join(override.env_var for override in env_overrides)
        notices.append(
            f"Hosted settings are currently overridden by environment variables: {env_var_names}."
        )
    if not settings.use_live_generation:
        notices.append("Hosted answer generation is currently turned off.")

    selected_model_id: str | None = None
    custom_model_id: str | None = None
    azure_base_url: str | None = None
    azure_deployment_name: str | None = None
    azure_model_family: str | None = None
    custom_base_url: str | None = None

    if preset == HostedProviderPreset.AZURE_OPENAI:
        azure_base_url = normalize_azure_openai_base_url(
            _normalize_text(cast(str | None, raw_hosted.get("azure_base_url")))
            or _normalize_text(settings.llm_base_url)
            or _legacy_azure_base_url(cast(str | None, raw_hosted.get("azure_resource_name")))
            or ""
        )
        azure_deployment_name = _normalize_text(cast(str | None, raw_hosted.get("azure_deployment_name"))) or _normalize_text(settings.llm_model)
        selected_model_id, custom_model_id = _model_selection(
            descriptor,
            _normalize_text(cast(str | None, raw_hosted.get("azure_model_family"))),
        )
        azure_model_family = custom_model_id or selected_model_id
        if selected_model_id == "custom":
            selected_model_id = "custom"
    elif preset == HostedProviderPreset.CUSTOM_OPENAI_COMPATIBLE:
        custom_base_url = _normalize_text(cast(str | None, raw_hosted.get("custom_base_url"))) or _normalize_text(settings.llm_base_url)
        selected_model_id, custom_model_id = _model_selection(descriptor, settings.llm_model)
    else:
        selected_model_id, custom_model_id = _model_selection(descriptor, settings.llm_model)

    saved_provider_settings: dict[HostedProviderPreset, HostedSavedProviderState] = {}
    active_profile = _active_provider_profile_payload(settings, raw_hosted)
    for provider_descriptor in visible_provider_catalog:
        profile = dict(raw_saved_provider_sections.get(provider_descriptor.preset, {}))
        if provider_descriptor.preset == preset:
            profile.update(active_profile)
        saved_provider_settings[provider_descriptor.preset] = _build_saved_provider_state(
            provider_descriptor,
            profile,
        )

    state = HostedSettingsState(
        config_path=str(resolved_config_path.resolve()),
        config_exists=resolved_config_path.exists(),
        provider_preset=preset,
        llm_provider=settings.llm_provider,
        runtime_mode=settings.runtime_mode,
        api_key_configured=bool(settings.llm_api_key),
        base_url=settings.llm_base_url,
        selected_model_id=selected_model_id,
        custom_model_id=custom_model_id,
        azure_base_url=azure_base_url,
        azure_deployment_name=azure_deployment_name,
        azure_model_family=azure_model_family,
        custom_base_url=custom_base_url,
        env_overrides=env_overrides,
        notices=notices,
        saved_provider_settings=saved_provider_settings,
    )
    return HostedSettingsEnvelope(settings=state, providers=visible_provider_catalog)


def _resolve_general_model(update: HostedSettingsUpdateRequest, descriptor: HostedProviderDescriptor) -> str:
    selected_model = _normalize_text(update.selected_model_id)
    custom_model = _normalize_text(update.custom_model_id)
    allowed_model_ids = {option.id for option in descriptor.models}

    if selected_model == "custom" or custom_model:
        if not descriptor.supports_custom_model:
            raise ValueError("Choose one of the supported models for this provider.")
        if not custom_model:
            raise ValueError("Choose a model or provide a custom model ID.")
        return custom_model
    if selected_model and selected_model in allowed_model_ids:
        return selected_model
    if selected_model:
        raise ValueError("Choose one of the supported models for this provider.")
    if descriptor.models:
        return descriptor.models[0].id
    raise ValueError("A model is required for this provider.")


def _resolve_azure_model_family(update: HostedSettingsUpdateRequest, descriptor: HostedProviderDescriptor) -> str:
    selected_model = _normalize_text(update.selected_model_id)
    custom_model = _normalize_text(update.custom_model_id)
    allowed_model_ids = {option.id for option in descriptor.models}

    if selected_model == "custom" or custom_model:
        if not descriptor.supports_custom_model:
            raise ValueError("Choose one of the supported Azure models.")
        return custom_model or descriptor.models[0].id
    if selected_model and selected_model in allowed_model_ids:
        return selected_model
    if selected_model:
        raise ValueError("Choose one of the supported Azure models.")
    if descriptor.models:
        return descriptor.models[0].id
    raise ValueError("Azure OpenAI has no configured model families.")


def persist_hosted_settings(
    update: HostedSettingsUpdateRequest,
    *,
    config_path: str | Path | None = None,
) -> HostedSettingsEnvelope:
    resolved_config_path = resolve_config_path(config_path)
    current_settings = Settings.from_env(config_path=resolved_config_path)
    overrides = detect_hosted_env_overrides(current_settings)
    if overrides:
        env_var_names = ", ".join(override.env_var for override in overrides)
        raise RuntimeError(
            f"Hosted settings are controlled by environment variables: {env_var_names}. "
            "Remove those environment variables before editing settings in the UI."
        )

    raw_config = deepcopy(load_raw_file_config(resolved_config_path))
    hosted = raw_config.get("hosted", {})
    if not isinstance(hosted, dict):
        hosted = {}
    hosted = dict(hosted)
    raw_saved_providers = hosted.get("saved_providers", {})
    if not isinstance(raw_saved_providers, dict):
        raw_saved_providers = {}
    saved_providers = dict(raw_saved_providers)
    runtime = raw_config.get("runtime", {})
    if not isinstance(runtime, dict):
        runtime = {}
    runtime = dict(runtime)

    provider_by_preset = provider_catalog_by_preset(resolved_config_path)
    descriptor = provider_by_preset[update.provider_preset]
    llm_provider = descriptor.provider_family
    llm_base_url: str | None
    llm_model: str
    legacy_provider_preset = _normalize_text(cast(str | None, hosted.get("provider_preset")))
    legacy_seed_api_key = (
        _normalize_text(current_settings.llm_api_key)
        if legacy_provider_preset == HostedProviderPreset.OPENAI.value
        or (current_settings.llm_base_url or "").rstrip("/") == "https://api.openai.com/v1"
        else None
    )

    if update.provider_preset == HostedProviderPreset.OPENAI:
        llm_base_url = "https://api.openai.com/v1"
        llm_model = _resolve_general_model(update, descriptor)
    elif update.provider_preset == HostedProviderPreset.OPENROUTER:
        llm_base_url = "https://openrouter.ai/api/v1"
        llm_model = _resolve_general_model(update, descriptor)
    elif update.provider_preset == HostedProviderPreset.AZURE_OPENAI:
        azure_base_url = _normalize_text(update.azure_base_url)
        model_family = _resolve_azure_model_family(update, descriptor)
        deployment_name = _normalize_text(update.azure_deployment_name) or model_family
        if not azure_base_url:
            raise ValueError("Azure OpenAI requires a URI.")
        normalized_azure_base_url = normalize_azure_openai_base_url(azure_base_url)
        if not normalized_azure_base_url:
            raise ValueError("Enter a valid Azure OpenAI v1 URI.")
        llm_base_url = normalized_azure_base_url
        llm_model = deployment_name
        hosted["azure_base_url"] = llm_base_url
        hosted["azure_model_family"] = model_family
        if deployment_name != model_family:
            hosted["azure_deployment_name"] = deployment_name
        else:
            hosted.pop("azure_deployment_name", None)
    elif update.provider_preset == HostedProviderPreset.GEMINI:
        llm_base_url = None
        llm_model = _resolve_general_model(update, descriptor)
    else:
        custom_base_url = _normalize_text(update.custom_base_url)
        if not custom_base_url:
            raise ValueError("Custom OpenAI-compatible providers require a base URL.")
        llm_base_url = custom_base_url.rstrip("/")
        llm_model = _resolve_general_model(update, descriptor)
        hosted["custom_base_url"] = llm_base_url

    hosted["provider"] = llm_provider
    hosted["provider_preset"] = update.provider_preset.value
    hosted["llm_model"] = llm_model
    hosted["use_live_decomposition"] = False
    hosted["use_live_metadata_enrichment"] = False

    saved_profile_raw = saved_providers.get(update.provider_preset.value, {})
    if not isinstance(saved_profile_raw, dict):
        saved_profile_raw = {}
    saved_profile = dict(saved_profile_raw)
    saved_profile["provider"] = llm_provider
    saved_profile["llm_model"] = llm_model

    if llm_base_url is None:
        hosted.pop("llm_base_url", None)
        saved_profile.pop("llm_base_url", None)
    else:
        hosted["llm_base_url"] = llm_base_url
        saved_profile["llm_base_url"] = llm_base_url

    api_key = _normalize_text(update.api_key)
    if update.clear_api_key:
        hosted.pop("llm_api_key", None)
        saved_profile.pop("llm_api_key", None)
    elif api_key:
        hosted["llm_api_key"] = api_key
        saved_profile["llm_api_key"] = api_key
    else:
        saved_api_key = _normalize_text(cast(str | None, saved_profile.get("llm_api_key")))
        if saved_api_key:
            hosted["llm_api_key"] = saved_api_key
        elif legacy_seed_api_key:
            hosted["llm_api_key"] = legacy_seed_api_key
            saved_profile["llm_api_key"] = legacy_seed_api_key
        else:
            hosted.pop("llm_api_key", None)

    if update.provider_preset != HostedProviderPreset.AZURE_OPENAI:
        hosted.pop("azure_base_url", None)
        hosted.pop("azure_resource_name", None)
        hosted.pop("azure_deployment_name", None)
        hosted.pop("azure_model_family", None)
        saved_profile.pop("azure_base_url", None)
        saved_profile.pop("azure_resource_name", None)
        saved_profile.pop("azure_deployment_name", None)
        saved_profile.pop("azure_model_family", None)
    else:
        saved_profile["azure_base_url"] = hosted["azure_base_url"]
        saved_profile.pop("azure_resource_name", None)
        saved_profile["azure_model_family"] = hosted["azure_model_family"]
        if "azure_deployment_name" in hosted:
            saved_profile["azure_deployment_name"] = hosted["azure_deployment_name"]
        else:
            saved_profile.pop("azure_deployment_name", None)
    if update.provider_preset != HostedProviderPreset.CUSTOM_OPENAI_COMPATIBLE:
        hosted.pop("custom_base_url", None)
        saved_profile.pop("custom_base_url", None)
    elif llm_base_url is not None:
        saved_profile["custom_base_url"] = llm_base_url

    saved_providers[update.provider_preset.value] = saved_profile
    hosted["saved_providers"] = saved_providers

    raw_config["runtime"] = runtime
    raw_config["hosted"] = hosted

    write_raw_file_config(raw_config, resolved_config_path)
    return build_hosted_settings_envelope(resolved_config_path)
