import { Alert, AlertDescription } from "@/components/ui/alert";
import { CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { ResponseReview } from "./ResponseReview";
import { ReviewPanel } from "./ReviewPanel";
import { useThreadActions } from "../context/threadActions";
import type { ParsedSentence, SessionState } from "../types";

export type AssistantMessageSource = "query" | "refinement";

interface ConversationMessageProps {
  entryId: string;
  source: AssistantMessageSource;
  session: SessionState;
  interactive: boolean;
}

function messageTitle(source: AssistantMessageSource, session: SessionState) {
  if (source === "refinement") {
    return "Refined response";
  }
  if (session.response_mode === "generation_failed") {
    return "Answer unavailable";
  }
  return "QUARRY response";
}

function messageLead(session: SessionState) {
  if (session.response_mode === "generation_failed") {
    return "I couldn't turn the available evidence into a verified answer this time. You can revise the question, leave review feedback, or try again with a narrower scope.";
  }
  return null;
}

function removedSentenceBanner(session: SessionState) {
  if (!session.removed_ungrounded_claim_count) {
    return null;
  }
  if (session.removed_ungrounded_claim_count === 1) {
    return "One sentence was removed because it could not be verified against the source documents.";
  }
  return `${session.removed_ungrounded_claim_count} sentences were removed because they could not be verified against the source documents.`;
}

export function ConversationMessage({
  entryId,
  source,
  session,
  interactive,
}: ConversationMessageProps) {
  const readOnly = !interactive;
  const actions = useThreadActions();

  return (
    <section className={`thread-message assistant-message ${readOnly ? "archived" : "interactive"}`}>
      <CardHeader className="thread-message-header">
        <div>
          <span className="eyebrow">{messageTitle(source, session)}</span>
          <CardTitle className="sr-only">{messageTitle(source, session)}</CardTitle>
          {source !== "query" ? (
            <CardDescription className="message-subtitle">
              Appended below the prior answer for comparison.
            </CardDescription>
          ) : null}
        </div>
      </CardHeader>

      <CardContent className="flex flex-col gap-4">
        {messageLead(session) ? <p className="assistant-lead">{messageLead(session)}</p> : null}
        {removedSentenceBanner(session) ? (
          <Alert className="response-warning-banner border-warning-medium/70 bg-[var(--warning-surface)] text-[var(--warning-ink)]">
            <AlertDescription>{removedSentenceBanner(session)}</AlertDescription>
          </Alert>
        ) : null}

        <ResponseReview
          session={session}
          readOnly={readOnly}
          onOpenCitation={(sentence, citationId, referenceQuote) =>
            actions.openCitation(entryId, session, sentence, citationId, referenceQuote, readOnly)
          }
          onSaveComment={(payload) => actions.saveComment(entryId, session.session_id, payload)}
          onUpdateComment={(commentId, commentText) =>
            actions.updateComment(entryId, session.session_id, commentId, commentText)
          }
          onDeleteComment={(commentId) => actions.deleteComment(entryId, session.session_id, commentId)}
        />
      </CardContent>

      {interactive ? (
        <ReviewPanel
          session={session}
          interactive={interactive}
          onRefine={() => actions.refine(session.session_id)}
        />
      ) : null}
    </section>
  );
}
