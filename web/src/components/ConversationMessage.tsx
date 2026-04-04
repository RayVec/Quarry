import { useEffect, useState } from "react";
import { ResponseReview } from "./ResponseReview";
import { ReviewPanel } from "./ReviewPanel";
import type { ParsedSentence, SessionState } from "../types";

export type AssistantMessageSource = "query" | "supplement" | "refinement";

interface ConversationMessageProps {
  source: AssistantMessageSource;
  session: SessionState;
  interactive: boolean;
  onOpenCitation: (session: SessionState, sentence: ParsedSentence, citationId: number, referenceQuote: string, readOnly: boolean) => void;
  onSaveDisagreement: (session: SessionState, sentenceIndex: number, note: string) => Promise<void>;
  onSupplement: (selectedFacets: string[]) => Promise<void>;
  onRefine: (selectedFacets: string[]) => Promise<void>;
  onRunClarificationSuggestion: (query: string) => void;
}

function messageTitle(source: AssistantMessageSource, session: SessionState) {
  if (source === "supplement") {
    return "Supplement";
  }
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
  onSaveDisagreement,
  onSupplement,
  onRefine,
  onRunClarificationSuggestion,
}: ConversationMessageProps) {
  const [reviewOpen, setReviewOpen] = useState(false);
  const [selectedFacetGaps, setSelectedFacetGaps] = useState<string[]>([]);
  const readOnly = !interactive;

  useEffect(() => {
    setReviewOpen(false);
    setSelectedFacetGaps([]);
  }, [session.session_id, session.response_mode, source]);

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
          onSaveDisagreement={(sentenceIndex, note) => onSaveDisagreement(session, sentenceIndex, note)}
        />
      )}

      {interactive && session.response_mode !== "clarification_required" ? (
        <ReviewPanel
          session={session}
          interactive={interactive}
          open={reviewOpen}
          selectedFacetGaps={selectedFacetGaps}
          onToggle={() => setReviewOpen((current) => !current)}
          onFacetToggle={(facet) =>
            setSelectedFacetGaps((current) =>
              current.includes(facet) ? current.filter((item) => item !== facet) : [...current, facet],
            )
          }
          onSupplement={onSupplement}
          onRefine={onRefine}
        />
      ) : null}
    </article>
  );
}
