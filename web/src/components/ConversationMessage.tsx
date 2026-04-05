import { useEffect, useState } from "react";
import { ResponseReview } from "./ResponseReview";
import { ReviewPanel } from "./ReviewPanel";
import type { ParsedSentence, SessionState } from "../types";

export type AssistantMessageSource = "query" | "refinement";

interface ConversationMessageProps {
  source: AssistantMessageSource;
  session: SessionState;
  interactive: boolean;
  onOpenCitation: (session: SessionState, sentence: ParsedSentence, citationId: number, referenceQuote: string, readOnly: boolean) => void;
  onSaveComment: (
    session: SessionState,
    payload: {
      text_selection: string;
      char_start: number;
      char_end: number;
      comment_text: string;
    },
  ) => Promise<void>;
  onUpdateComment: (session: SessionState, commentId: string, commentText: string) => Promise<void>;
  onDeleteComment: (session: SessionState, commentId: string) => Promise<void>;
  onRefine: () => Promise<void>;
  onRunClarificationSuggestion: (query: string) => void;
}

function messageTitle(source: AssistantMessageSource, session: SessionState) {
  if (source === "refinement") {
    return "Refined response";
  }
  if (session.response_mode === "clarification_required") {
    return "Clarification needed";
  }
  if (session.response_mode === "generation_failed") {
    return "Answer unavailable";
  }
  return "QUARRY response";
}

function messageLead(session: SessionState) {
  if (session.response_mode === "clarification_required") {
    return "I need a bit more detail to search effectively. Try one of these reformulations or write your own.";
  }
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
  source,
  session,
  interactive,
  onOpenCitation,
  onSaveComment,
  onUpdateComment,
  onDeleteComment,
  onRefine,
  onRunClarificationSuggestion,
}: ConversationMessageProps) {
  const readOnly = !interactive;

  return (
    <article className={`thread-message assistant-message ${readOnly ? "archived" : "interactive"}`}>
      <div className="thread-message-header">
        <div>
          <span className="eyebrow">{messageTitle(source, session)}</span>
          {source !== "query" ? <p className="message-subtitle">Appended below the prior answer for comparison.</p> : null}
        </div>
      </div>

      {messageLead(session) ? <p className="assistant-lead">{messageLead(session)}</p> : null}
      {removedSentenceBanner(session) ? <div className="response-warning-banner">{removedSentenceBanner(session)}</div> : null}

      {session.response_mode === "clarification_required" ? (
        <div className="clarification-block" data-testid="clarification-required">
          <div className="clarification-chip-row">
            {session.clarification_suggestions.map((suggestion) => (
              <button
                className="clarification-chip"
                data-testid={`clarification-suggestion-${suggestion.replace(/[^a-z0-9]+/gi, "-").toLowerCase()}`}
                disabled={readOnly}
                key={suggestion}
                onClick={() => onRunClarificationSuggestion(suggestion)}
              >
                {suggestion}
              </button>
            ))}
          </div>
        </div>
      ) : (
        <ResponseReview
          session={session}
          readOnly={readOnly}
          onOpenCitation={(sentence, citationId, referenceQuote) =>
            onOpenCitation(session, sentence, citationId, referenceQuote, readOnly)
          }
          onSaveComment={(payload) => onSaveComment(session, payload)}
          onUpdateComment={(commentId, commentText) => onUpdateComment(session, commentId, commentText)}
          onDeleteComment={(commentId) => onDeleteComment(session, commentId)}
        />
      )}

      {interactive && session.response_mode !== "clarification_required" ? (
        <ReviewPanel
          session={session}
          interactive={interactive}
          onRefine={onRefine}
        />
      ) : null}
    </article>
  );
}
