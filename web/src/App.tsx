import { useState } from "react";
import { CitationDialog } from "@/components/CitationDialog";
import { DiagnosticsDrawer } from "@/components/DiagnosticsDrawer";
import { ThreadActionsProvider } from "@/context/threadActions";
import { AppSidebar } from "@/features/thread/components/AppSidebar";
import { ConversationStage } from "@/features/thread/components/ConversationStage";
import { LandingStage } from "@/features/thread/components/LandingStage";
import { useHostedSettings } from "@/features/settings/useHostedSettings";
import { useThreadController } from "@/features/thread/useThreadController";
import { useDockedComposerOffset } from "@/hooks/useDockedComposerOffset";
import "./styles/app.css";

type DrawerTab = "settings" | "diagnostics";

export default function App() {
  const [diagnosticsOpen, setDiagnosticsOpen] = useState(false);
  const [drawerTab, setDrawerTab] = useState<DrawerTab>("settings");
  const hostedSettings = useHostedSettings();
  const threadController = useThreadController();
  const { workspaceColumnRef, dockedComposerRef, dockedComposerOffset } =
    useDockedComposerOffset(threadController.thread.length > 0);

  function openWorkspaceDrawer(tab: DrawerTab) {
    setDrawerTab(tab);
    hostedSettings.clearSaveNotice();
    setDiagnosticsOpen(true);
  }

  function handleNewSearch() {
    threadController.handleNewSearch();
    setDiagnosticsOpen(false);
  }

  return (
    <ThreadActionsProvider value={threadController.threadActions}>
      <div className="app-shell">
        <AppSidebar
          latestUserQuery={threadController.latestUserQuery}
          recentResearch={threadController.recentResearch}
          onDeleteRecentResearch={threadController.handleDeleteRecentResearch}
          onNewSearch={handleNewSearch}
          onResumeResearch={(query) =>
            threadController.submitQuery(query, { fresh: true })
          }
        />

        <div className="workspace-shell">
          {threadController.thread.length === 0 ? (
            <LandingStage
              loading={threadController.loading}
              query={threadController.query}
              onChangeQuery={threadController.setQuery}
              onOpenSettings={() => openWorkspaceDrawer("settings")}
              onSubmitQuery={() => void threadController.submitQuery()}
            />
          ) : (
            <ConversationStage
              dockedComposerOffset={dockedComposerOffset}
              dockedComposerRef={dockedComposerRef}
              latestUserQuery={threadController.latestUserQuery}
              loading={threadController.loading}
              onChangeQuery={threadController.setQuery}
              onOpenDiagnostics={() => openWorkspaceDrawer("diagnostics")}
              onSubmitQuery={() => void threadController.submitQuery()}
              query={threadController.query}
              thread={threadController.thread}
              workspaceColumnRef={workspaceColumnRef}
            />
          )}
        </div>

        <DiagnosticsDrawer
          session={threadController.latestSession}
          open={diagnosticsOpen}
          activeTab={drawerTab}
          settings={hostedSettings.settings}
          providers={hostedSettings.providers}
          settingsLoading={hostedSettings.settingsLoading}
          settingsSaving={hostedSettings.settingsSaving}
          settingsError={hostedSettings.settingsError}
          settingsSaveNotice={hostedSettings.settingsSaveNotice}
          onTabChange={setDrawerTab}
          onClose={() => setDiagnosticsOpen(false)}
          onSaveSettings={(payload) => void hostedSettings.saveSettings(payload)}
        />

        {threadController.activeCitation ? (
          <CitationDialog
            citation={threadController.activeCitation.citation}
            sentenceIndex={threadController.activeCitation.sentenceIndex}
            referenceQuote={threadController.activeCitation.referenceQuote}
            readOnly={threadController.activeCitation.readOnly}
            session={threadController.activeCitation.session}
            initialAlternatives={
              threadController.activeCitationAlternativesCacheEntry
                ?.alternatives ?? []
            }
            initialAlternativesLoaded={
              threadController.activeCitationAlternativesCacheEntry
                ?.hasLoaded ?? false
            }
            onAlternativesLoaded={
              threadController.storeActiveCitationAlternatives
            }
            onClose={threadController.closeCitation}
            onSessionUpdate={threadController.handleCitationSessionUpdate}
          />
        ) : null}
      </div>
    </ThreadActionsProvider>
  );
}
