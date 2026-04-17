import { startTransition, useEffect, useState } from "react";
import { api } from "@/api";
import type {
  HostedProviderDescriptor,
  HostedSettingsState,
  HostedSettingsUpdatePayload,
} from "@/types";

export function useHostedSettings() {
  const [settings, setSettings] = useState<HostedSettingsState | null>(null);
  const [providers, setProviders] = useState<HostedProviderDescriptor[]>([]);
  const [settingsLoading, setSettingsLoading] = useState(true);
  const [settingsSaving, setSettingsSaving] = useState(false);
  const [settingsError, setSettingsError] = useState<string | null>(null);
  const [settingsSaveNotice, setSettingsSaveNotice] = useState<string | null>(
    null,
  );

  useEffect(() => {
    let cancelled = false;

    async function loadHostedSettings() {
      setSettingsLoading(true);
      try {
        const response = await api.getHostedSettings();
        if (cancelled) {
          return;
        }
        setSettings(response.settings);
        setProviders(response.providers);
        setSettingsError(null);
      } catch (error) {
        if (cancelled) {
          return;
        }
        setSettingsError(
          error instanceof Error
            ? error.message
            : "Could not load hosted settings.",
        );
      } finally {
        if (!cancelled) {
          setSettingsLoading(false);
        }
      }
    }

    void loadHostedSettings();
    return () => {
      cancelled = true;
    };
  }, []);

  async function saveSettings(payload: HostedSettingsUpdatePayload) {
    setSettingsSaving(true);
    setSettingsError(null);
    setSettingsSaveNotice(null);
    try {
      const response = await api.updateHostedSettings(payload);
      startTransition(() => {
        setSettings(response.settings);
        setProviders(response.providers);
      });
      setSettingsSaveNotice("Provider settings saved successfully.");
    } catch (error) {
      setSettingsError(
        error instanceof Error
          ? error.message
          : "Could not save hosted settings.",
      );
    } finally {
      setSettingsSaving(false);
    }
  }

  function clearSaveNotice() {
    setSettingsSaveNotice(null);
  }

  return {
    settings,
    providers,
    settingsLoading,
    settingsSaving,
    settingsError,
    settingsSaveNotice,
    saveSettings,
    clearSaveNotice,
  };
}
