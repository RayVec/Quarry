import type { CSSProperties, Ref } from "react";
import { Cog } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ConversationMessage } from "@/components/ConversationMessage";
import { PendingConversationMessage } from "@/components/PendingConversationMessage";
import { QueryComposer } from "@/components/QueryComposer";
import type { ThreadEntry } from "@/features/thread/model";

interface ConversationStageProps {
  dockedComposerOffset: number;
  dockedComposerRef: Ref<HTMLFormElement>;
  threadTitleQuery: string | null;
  loading: boolean;
  onChangeQuery: (nextQuery: string) => void;
  onOpenDiagnostics: () => void;
  onSubmitQuery: () => void;
  query: string;
  thread: ThreadEntry[];
  workspaceColumnRef: Ref<HTMLElement>;
}

export function ConversationStage({
  dockedComposerOffset,
  dockedComposerRef,
  threadTitleQuery,
  loading,
  onChangeQuery,
  onOpenDiagnostics,
  onSubmitQuery,
  query,
  thread,
  workspaceColumnRef,
}: ConversationStageProps) {
  return (
    <main
      className="conversation-stage with-docked-offset"
      style={
        {
          "--docked-composer-offset": `${dockedComposerOffset}px`,
        } as CSSProperties
      }
    >
      <section className="workspace-column" ref={workspaceColumnRef}>
        <div className="thread-intro">
          <div className="thread-intro-copy">
            <span className="tiny-label">Conversation</span>
            <h2>{threadTitleQuery ?? "Current discussion"}</h2>
          </div>
          <Button
            className="diagnostics-trigger"
            data-testid="open-diagnostics"
            onClick={onOpenDiagnostics}
          >
            <span className="sr-only">Provider settings and diagnostics</span>
            <Cog aria-hidden="true" focusable="false" />
          </Button>
        </div>

        <div className="thread-column" data-testid="conversation-thread">
          {thread.map((entry) =>
            entry.kind === "user" ? (
              <Card className="thread-message user-message" key={entry.id}>
                <CardContent className="user-query-card-content">
                  <p>{entry.query}</p>
                </CardContent>
              </Card>
            ) : entry.kind === "pending-assistant" ? (
              <PendingConversationMessage
                key={entry.id}
                messageRun={entry.messageRun}
                session={entry.session}
              />
            ) : (
              <ConversationMessage
                key={entry.id}
                entryId={entry.id}
                entry={entry}
              />
            ),
          )}
        </div>
      </section>

      <QueryComposer
        className="docked"
        id="thread-query-input"
        label=""
        loading={loading}
        placeholder="Ask a question or reply"
        query={query}
        formRef={dockedComposerRef}
        onChange={onChangeQuery}
        onSubmit={onSubmitQuery}
      />
    </main>
  );
}
