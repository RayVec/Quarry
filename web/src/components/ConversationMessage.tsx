import { CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useThreadActions } from "@/context/threadActions";
import {
  isResearchAssistantEntry,
  researchAssistantText,
  type AssistantThreadEntry,
} from "@/features/thread/model";
import { ResponseReview } from "./ResponseReview";
import { ReviewPanel } from "./ReviewPanel";

interface ConversationMessageProps {
  entryId: string;
  entry: AssistantThreadEntry;
}

export function ConversationMessage({
  entryId,
  entry,
}: ConversationMessageProps) {
  const actions = useThreadActions();

  if (!isResearchAssistantEntry(entry)) {
    return (
      <section className="thread-message assistant-message chat-only">
        <CardHeader className="thread-message-header">
          <div>
            <span className="eyebrow">QUARRY response</span>
            <CardTitle className="sr-only">QUARRY response</CardTitle>
          </div>
        </CardHeader>

        <CardContent className="assistant-message-content">
          <p className="assistant-chat-copy">{entry.turn.content}</p>
        </CardContent>
      </section>
    );
  }

  const { interactive, session } = entry;
  const readOnly = !interactive;
  const generationFailed = session.response_mode === "generation_failed";
  const title = generationFailed ? "Answer unavailable" : "QUARRY response";
  const lead = generationFailed
    ? "I couldn't turn the available evidence into a verified answer this time. You can revise the question, leave review feedback, or try again with a narrower scope."
    : null;
  const hasVisibleResponse = session.generated_response.trim().length > 0;

  return (
    <section
      className={`thread-message assistant-message ${readOnly ? "archived" : "interactive"}`}
    >
      <CardHeader className="thread-message-header">
        <div>
          <span className="eyebrow">{title}</span>
          <CardTitle className="sr-only">{title}</CardTitle>
        </div>
      </CardHeader>

      <CardContent className="assistant-message-content">
        {lead ? <p className="assistant-lead">{lead}</p> : null}

        {generationFailed && !hasVisibleResponse ? (
          <p className="assistant-chat-copy">
            {researchAssistantText(session)}
          </p>
        ) : (
          <ResponseReview
            session={session}
            readOnly={readOnly}
            onOpenCitation={(sentence, citationId, referenceQuote) =>
              actions.openCitation(
                entryId,
                session,
                sentence,
                citationId,
                referenceQuote,
                readOnly,
              )
            }
            onSaveComment={(payload) =>
              actions.saveComment(entryId, session.session_id, payload)
            }
            onUpdateComment={(commentId, commentText) =>
              actions.updateComment(
                entryId,
                session.session_id,
                commentId,
                commentText,
              )
            }
            onDeleteComment={(commentId) =>
              actions.deleteComment(entryId, session.session_id, commentId)
            }
          />
        )}
      </CardContent>

      {interactive ? (
        <ReviewPanel
          session={session}
          onRefine={() => actions.refine(session.session_id)}
        />
      ) : null}
    </section>
  );
}
