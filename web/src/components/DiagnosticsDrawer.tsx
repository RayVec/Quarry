import { useEffect, useMemo, useState } from "react";
import { AlertCircleIcon, CheckCircle2, Loader2, X } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type {
  HostedProviderDescriptor,
  HostedProviderPreset,
  HostedSettingsState,
  HostedSettingsUpdatePayload,
  SessionState,
} from "../types";

type DrawerTab = "settings" | "diagnostics";

interface DiagnosticsDrawerProps {
  session: SessionState | null;
  open: boolean;
  activeTab: DrawerTab;
  settings: HostedSettingsState | null;
  providers: HostedProviderDescriptor[];
  settingsLoading: boolean;
  settingsSaving: boolean;
  settingsError: string | null;
  settingsSaveNotice: string | null;
  onTabChange: (tab: DrawerTab) => void;
  onClose: () => void;
  onSaveSettings: (payload: HostedSettingsUpdatePayload) => void;
}

interface HostedSettingsDraft {
  providerPreset: HostedProviderPreset;
  selectedModelId: string;
  customModelId: string;
  apiKey: string;
  apiKeyMasked: boolean;
  clearApiKey: boolean;
  customBaseUrl: string;
  azureBaseUrl: string;
}

const SAVED_API_KEY_MASK = "saved-api-key";

function baseUrlValueForDraft(draft: HostedSettingsDraft): string {
  return draft.providerPreset === "azure_openai"
    ? draft.azureBaseUrl
    : draft.customBaseUrl;
}

function providerStateForPreset(
  settings: HostedSettingsState | null,
  preset: HostedProviderPreset,
) {
  if (!settings) {
    return null;
  }
  if (settings.provider_preset === preset) {
    return {
      api_key_configured: settings.api_key_configured,
      selected_model_id: settings.selected_model_id ?? null,
      custom_model_id: settings.custom_model_id ?? null,
      custom_base_url: settings.custom_base_url ?? settings.base_url ?? null,
      azure_base_url: settings.azure_base_url ?? settings.base_url ?? null,
      azure_deployment_name: settings.azure_deployment_name ?? null,
      azure_model_family: settings.azure_model_family ?? null,
    };
  }
  return settings.saved_provider_settings?.[preset] ?? null;
}

function defaultModelIdForProvider(
  descriptor: HostedProviderDescriptor | null,
  selectedModelId: string | null,
): string {
  if (
    selectedModelId &&
    ((selectedModelId !== "custom" &&
      descriptor?.models.some((item) => item.id === selectedModelId)) ||
      (selectedModelId === "custom" && descriptor?.supports_custom_model))
  ) {
    return selectedModelId;
  }
  return descriptor?.models[0]?.id ?? "custom";
}

function buildDraft(
  settings: HostedSettingsState | null,
  providers: HostedProviderDescriptor[],
): HostedSettingsDraft {
  const visiblePreset = settings?.provider_preset
    ? providers.find((item) => item.preset === settings.provider_preset)?.preset
    : null;
  const fallbackPreset: HostedProviderPreset =
    visiblePreset ?? providers[0]?.preset ?? "openrouter";
  const descriptor =
    providers.find((item) => item.preset === fallbackPreset) ??
    providers[0] ??
    null;
  const providerState = providerStateForPreset(settings, fallbackPreset);
  const defaultModelId = defaultModelIdForProvider(
    descriptor,
    providerState?.selected_model_id ?? null,
  );

  return {
    providerPreset: fallbackPreset,
    selectedModelId: defaultModelId,
    customModelId:
      defaultModelId === "custom" ? providerState?.custom_model_id ?? "" : "",
    apiKey: "",
    apiKeyMasked: Boolean(providerState?.api_key_configured),
    clearApiKey: false,
    customBaseUrl: providerState?.custom_base_url ?? "",
    azureBaseUrl: providerState?.azure_base_url ?? "",
  };
}

function applyPresetDefaults(
  preset: HostedProviderPreset,
  providers: HostedProviderDescriptor[],
  settings: HostedSettingsState | null,
  current: HostedSettingsDraft,
): HostedSettingsDraft {
  const descriptor = providers.find((item) => item.preset === preset);
  const providerState = providerStateForPreset(settings, preset);
  const nextModelId = defaultModelIdForProvider(
    descriptor ?? null,
    providerState?.selected_model_id ?? null,
  );

  return {
    ...current,
    providerPreset: preset,
    selectedModelId: nextModelId,
    customModelId: nextModelId === "custom" ? providerState?.custom_model_id ?? "" : "",
    apiKey: "",
    customBaseUrl:
      preset === "custom_openai_compatible"
        ? providerState?.custom_base_url ?? ""
        : "",
    azureBaseUrl:
      preset === "azure_openai"
        ? providerState?.azure_base_url ?? ""
        : "",
    apiKeyMasked: Boolean(providerState?.api_key_configured),
    clearApiKey: false,
  };
}

function validationMessage(
  descriptor: HostedProviderDescriptor | null,
  draft: HostedSettingsDraft,
): string | null {
  if (!descriptor) {
    return "Choose a provider.";
  }
  if (descriptor.requires_base_url && !baseUrlValueForDraft(draft).trim()) {
    return draft.providerPreset === "azure_openai"
      ? "Enter your Azure URI."
      : "Enter a base URL for the custom provider.";
  }
  if (draft.selectedModelId === "custom" && !draft.customModelId.trim()) {
    return "Enter a custom model ID.";
  }
  return null;
}

function SettingsField({
  label,
  description,
  children,
}: {
  label: string;
  description?: string;
  children: React.ReactNode;
}) {
  return (
    <label className="settings-field-label">
      <span className="settings-field-label-text">{label}</span>
      {children}
      {description ? (
        <span className="settings-field-label-description">{description}</span>
      ) : null}
    </label>
  );
}

function DiagnosticsDatum({
  label,
  value,
  testId,
}: {
  label: string;
  value: string | number;
  testId?: string;
}) {
  return (
    <div className="diagnostics-datum">
      <span className="diagnostics-datum-label">
        {label}
      </span>
      <span className="diagnostics-datum-value" data-testid={testId}>
        {value}
      </span>
    </div>
  );
}

export function DiagnosticsDrawer({
  session,
  open,
  activeTab,
  settings,
  providers,
  settingsLoading,
  settingsSaving,
  settingsError,
  settingsSaveNotice,
  onTabChange,
  onClose,
  onSaveSettings,
}: DiagnosticsDrawerProps) {
  const [draft, setDraft] = useState<HostedSettingsDraft>(() =>
    buildDraft(settings, providers),
  );

  useEffect(() => {
    setDraft(buildDraft(settings, providers));
  }, [settings, providers]);

  const selectedProvider = useMemo(
    () =>
      providers.find((item) => item.preset === draft.providerPreset) ?? null,
    [draft.providerPreset, providers],
  );
  const formValidationMessage = useMemo(
    () => validationMessage(selectedProvider, draft),
    [draft, selectedProvider],
  );
  const connectionLabel =
    draft.providerPreset === "azure_openai" ? "Azure URI" : "Base URL";
  const connectionPlaceholder =
    draft.providerPreset === "azure_openai"
      ? "https://your-resource.cognitiveservices.azure.com/openai/v1"
      : "https://example.com/v1";
  const selectedProviderState = useMemo(
    () => providerStateForPreset(settings, draft.providerPreset),
    [draft.providerPreset, settings],
  );
  const hasSavedApiKeyForSelectedProvider = Boolean(
    selectedProviderState?.api_key_configured,
  );
  const canClearSavedApiKey = Boolean(
    hasSavedApiKeyForSelectedProvider &&
      !draft.clearApiKey &&
      !draft.apiKey.trim(),
  );
  const apiKeyValue =
    draft.apiKeyMasked && canClearSavedApiKey ? SAVED_API_KEY_MASK : draft.apiKey;
  const apiKeyPlaceholder =
    draft.clearApiKey || !hasSavedApiKeyForSelectedProvider
      ? "blank api key"
      : undefined;
  const settingsLocked = (settings?.env_overrides.length ?? 0) > 0;
  const canSave =
    !settingsLoading &&
    !settingsSaving &&
    !settingsLocked &&
    formValidationMessage === null;

  return (
    <Sheet open={open} onOpenChange={(nextOpen) => !nextOpen && onClose()}>
      <SheetContent
        className="diagnostics-drawer workspace-drawer diagnostics-drawer--sized"
        data-testid="diagnostics-drawer"
        side="right"
      >
        <SheetHeader className="workspace-drawer-header">
          <SheetTitle asChild>
            <h3 className="workspace-drawer-title">
              Settings
            </h3>
          </SheetTitle>
          <SheetDescription className="sr-only">
            Manage hosted provider credentials and inspect runtime diagnostics.
          </SheetDescription>
        </SheetHeader>

        <div className="workspace-drawer-content">
          <Tabs
            className="workspace-drawer-tabs"
            value={activeTab}
            onValueChange={(value) => onTabChange(value as DrawerTab)}
          >
            <TabsList className="workspace-drawer-tabs-list">
              <TabsTrigger value="settings">Provider</TabsTrigger>
              <TabsTrigger value="diagnostics">Diagnostics</TabsTrigger>
            </TabsList>

            <TabsContent value="settings" className="workspace-drawer-panel">
              {settingsLoading ? (
                <Alert>
                  <Loader2 className="spin-icon" />
                  <AlertTitle>Loading</AlertTitle>
                  <AlertDescription>
                    Loading saved provider settings…
                  </AlertDescription>
                </Alert>
              ) : null}

              {settings?.notices.map((notice) => (
                <Alert key={notice}>
                  <AlertCircleIcon />
                  <AlertTitle>Notice</AlertTitle>
                  <AlertDescription>{notice}</AlertDescription>
                </Alert>
              ))}

              {settingsError ? (
                <Alert variant="destructive">
                  <AlertCircleIcon />
                  <AlertTitle>Save error</AlertTitle>
                  <AlertDescription>{settingsError}</AlertDescription>
                </Alert>
              ) : null}

              {settingsLocked ? (
                <Alert>
                  <AlertCircleIcon />
                  <AlertTitle>Environment-managed settings</AlertTitle>
                  <AlertDescription>
                    These hosted settings are currently controlled by
                    environment variables, so the web form is read-only.
                  </AlertDescription>
                </Alert>
              ) : null}

              <Card className="surface-card">
                <CardHeader>
                  <CardTitle className="tiny-label">Provider configuration</CardTitle>
                </CardHeader>
                <CardContent className="stack-gap-4">
                  <SettingsField label="Provider">
                    <Select
                      value={draft.providerPreset}
                      onValueChange={(value) =>
                        setDraft((current) =>
                          applyPresetDefaults(
                            value as HostedProviderPreset,
                            providers,
                            settings,
                            current,
                          ),
                        )
                      }
                    >
                      <SelectTrigger className="provider-select-trigger" aria-label="Provider">
                        <SelectValue placeholder="Choose provider" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectGroup>
                          {providers.map((provider) => (
                            <SelectItem
                              key={provider.preset}
                              value={provider.preset}
                            >
                              {provider.label}
                            </SelectItem>
                          ))}
                        </SelectGroup>
                      </SelectContent>
                    </Select>
                  </SettingsField>

                  {selectedProvider?.requires_base_url ? (
                    <SettingsField label={connectionLabel}>
                      <Input
                        placeholder={connectionPlaceholder}
                        value={baseUrlValueForDraft(draft)}
                        onChange={(event) =>
                          setDraft((current) => ({
                            ...current,
                            ...(current.providerPreset === "azure_openai"
                              ? { azureBaseUrl: event.target.value }
                              : { customBaseUrl: event.target.value }),
                          }))
                        }
                      />
                    </SettingsField>
                  ) : null}

                  <SettingsField label={selectedProvider?.model_label ?? "Model"}>
                    <Select
                      value={draft.selectedModelId}
                      onValueChange={(value) =>
                        setDraft((current) => ({
                          ...current,
                          selectedModelId: value,
                          customModelId:
                            value === "custom" ? current.customModelId : "",
                        }))
                      }
                    >
                      <SelectTrigger
                        aria-label={selectedProvider?.model_label ?? "Model"}
                        className="full-width"
                      >
                        <SelectValue placeholder="Choose model" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectGroup>
                          {selectedProvider?.models.map((model) => (
                            <SelectItem key={model.id} value={model.id}>
                              {model.label}
                            </SelectItem>
                          ))}
                          {selectedProvider?.supports_custom_model ? (
                            <SelectItem value="custom">Custom</SelectItem>
                          ) : null}
                        </SelectGroup>
                      </SelectContent>
                    </Select>

                    {draft.selectedModelId === "custom" ? (
                      <Input
                        autoCapitalize="off"
                        autoComplete="off"
                        autoCorrect="off"
                        aria-label="Custom model ID"
                        name="quarry-model-id"
                        placeholder={
                          draft.providerPreset === "openrouter"
                            ? "provider/model-name"
                            : "model-id"
                        }
                        spellCheck={false}
                        value={draft.customModelId}
                        onChange={(event) =>
                          setDraft((current) => ({
                            ...current,
                            customModelId: event.target.value,
                          }))
                        }
                      />
                    ) : null}
                  </SettingsField>

                  <SettingsField label="API key">
                    <div className="api-key-input-shell">
                      <Input
                        aria-label="API key"
                        autoCapitalize="off"
                        autoComplete="off"
                        autoCorrect="off"
                        className={`api-key-input ${canClearSavedApiKey ? "api-key-input--with-clear" : ""}`.trim()}
                        inputMode="text"
                        name="quarry-api-key"
                        placeholder={apiKeyPlaceholder}
                        spellCheck={false}
                        type="text"
                        value={apiKeyValue}
                        onBlur={() => {
                          if (
                            hasSavedApiKeyForSelectedProvider &&
                            !draft.apiKey.trim() &&
                            !draft.clearApiKey
                          ) {
                            setDraft((current) => ({
                              ...current,
                              apiKeyMasked: true,
                            }));
                          }
                        }}
                        onChange={(event) =>
                          setDraft((current) => ({
                            ...current,
                            apiKey: event.target.value,
                            apiKeyMasked: false,
                            clearApiKey: event.target.value ? false : current.clearApiKey,
                          }))
                        }
                        onFocus={() => {
                          if (draft.apiKeyMasked && !draft.clearApiKey) {
                            setDraft((current) => ({
                              ...current,
                              apiKeyMasked: false,
                            }));
                          }
                        }}
                      />
                      {canClearSavedApiKey ? (
                        <Button
                          aria-label="Clear saved API key"
                          className="api-key-clear"
                          onClick={() =>
                            setDraft((current) => ({
                              ...current,
                              apiKey: "",
                              apiKeyMasked: false,
                              clearApiKey: true,
                            }))
                          }
                          size="icon-xs"
                          type="button"
                          variant="ghost"
                        >
                          <X />
                        </Button>
                      ) : null}
                    </div>
                  </SettingsField>
                </CardContent>
              </Card>

              <div className="workspace-drawer-actions">
                {formValidationMessage ? (
                  <p className="workspace-validation-message">
                    {formValidationMessage}
                  </p>
                ) : null}

                <Button
                  className="workspace-save-button"
                  disabled={!canSave}
                  onClick={() =>
                    onSaveSettings({
                      provider_preset: draft.providerPreset,
                      selected_model_id: draft.selectedModelId,
                      custom_model_id:
                        draft.selectedModelId === "custom"
                          ? draft.customModelId.trim()
                          : null,
                      api_key: draft.apiKey.trim() || null,
                      clear_api_key: draft.clearApiKey,
                      custom_base_url:
                        draft.providerPreset === "custom_openai_compatible"
                          ? draft.customBaseUrl.trim() || null
                          : null,
                      azure_base_url:
                        draft.providerPreset === "azure_openai"
                          ? draft.azureBaseUrl.trim() || null
                          : null,
                      azure_deployment_name: null,
                      azure_model_family:
                        draft.selectedModelId === "custom"
                          ? draft.customModelId.trim() || null
                          : draft.selectedModelId,
                    })
                  }
                  size="lg"
                  type="button"
                >
                  {settingsSaving ? (
                    <>
                      <Loader2
                        className="spin-icon"
                        data-icon="inline-start"
                      />
                      Saving…
                    </>
                  ) : (
                    "Save"
                  )}
                </Button>

                {settingsSaveNotice ? (
                  <div className="workspace-save-feedback">
                    <p className="workspace-save-notice-row">
                      <CheckCircle2 />
                      <span>{settingsSaveNotice}</span>
                    </p>
                    <p className="workspace-save-path">
                      Saved to{" "}
                      <code>{settings?.config_path ?? "config.toml"}</code>.
                      Future queries will use the updated provider settings.
                    </p>
                  </div>
                ) : null}
              </div>
            </TabsContent>

            <TabsContent value="diagnostics" className="workspace-drawer-panel">
              {session ? (
                <>
                  <Card className="surface-card">
                    <CardHeader>
                      <CardTitle className="tiny-label">Session</CardTitle>
                    </CardHeader>
                    <CardContent className="diagnostics-grid diagnostics-grid--two-column">
                      <DiagnosticsDatum
                        label="Session"
                        testId="status-mode"
                        value={session.session_id}
                      />
                      <DiagnosticsDatum
                        label="Mode"
                        value={session.response_mode}
                      />
                      <DiagnosticsDatum
                        label="Query status"
                        testId="status-query-status"
                        value={session.query_status}
                      />
                      <DiagnosticsDatum
                        label="Current stage"
                        testId="status-query-stage"
                        value={session.query_stage_label}
                      />
                      <DiagnosticsDatum
                        label="Runtime"
                        testId="status-runtime-mode"
                        value={session.runtime_mode}
                      />
                      <DiagnosticsDatum
                        label="Profile"
                        testId="status-runtime-profile"
                        value={session.runtime_profile}
                      />
                      <DiagnosticsDatum
                        label="Generator"
                        testId="status-generation-provider"
                        value={session.generation_provider}
                      />
                      <DiagnosticsDatum
                        label="Parser"
                        testId="status-parser-provider"
                        value={session.parser_provider}
                      />
                      <DiagnosticsDatum
                        label="Refinements"
                        testId="status-refinements"
                        value={session.refinement_count}
                      />
                      <DiagnosticsDatum
                        label="Citations"
                        testId="status-citations"
                        value={session.citation_index?.length ?? 0}
                      />
                      <DiagnosticsDatum
                        label="Sentences"
                        testId="status-sentences"
                        value={session.parsed_sentences?.length ?? 0}
                      />
                    </CardContent>
                  </Card>

                  {(session.ui_messages?.length ?? 0) ? (
                    <Card className="surface-card">
                      <CardHeader>
                        <CardTitle className="tiny-label">
                          Pipeline messages
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="stack-gap-3">
                        {session.ui_messages.map((message) => (
                          <Alert key={`${message.code}-${message.message}`}>
                            <AlertTitle>{message.code}</AlertTitle>
                            <AlertDescription>
                              {message.message}
                            </AlertDescription>
                          </Alert>
                        ))}
                      </CardContent>
                    </Card>
                  ) : null}

                  <Card className="surface-card">
                    <CardHeader>
                      <CardTitle className="tiny-label">Feedback</CardTitle>
                    </CardHeader>
                    <CardContent className="diagnostics-grid diagnostics-grid--three-column">
                      <DiagnosticsDatum
                        label="Comments"
                        value={session.feedback.comments?.length ?? 0}
                      />
                      <DiagnosticsDatum
                        label="Resolved"
                        value={session.feedback.resolved_comments?.length ?? 0}
                      />
                      <DiagnosticsDatum
                        label="Citation replacements"
                        value={
                          session.feedback.citation_replacements?.length ?? 0
                        }
                      />
                    </CardContent>
                  </Card>
                </>
              ) : (
                <Alert>
                  <AlertCircleIcon />
                  <AlertTitle>No session selected</AlertTitle>
                  <AlertDescription>
                    Start or reopen a research thread to inspect runtime
                    diagnostics here.
                  </AlertDescription>
                </Alert>
              )}
            </TabsContent>
          </Tabs>
        </div>
      </SheetContent>
    </Sheet>
  );
}
