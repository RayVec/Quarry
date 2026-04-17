import { startTransition, useState } from "react";
import { Button } from "@/components/ui/button";
import type { SessionState } from "../types";

interface ReviewPanelProps {
  session: SessionState;
  interactive: boolean;
  onRefine: () => Promise<void>;
}

export function ReviewPanel({
  session,
  interactive,
  onRefine,
}: ReviewPanelProps) {
  const [busy, setBusy] = useState<"refine" | null>(null);
  const commentCount = session.feedback?.comments?.length ?? 0;
  const resolvedCount = session.feedback?.resolved_comments?.length ?? 0;
  const replacementCount = session.feedback?.citation_replacements?.length ?? 0;
  const dislikedCitationCount =
    session.feedback?.citation_feedback?.filter(
      (feedback) => feedback.feedback_type === "dislike",
    ).length ?? 0;
  const removedSentenceCount = session.removed_ungrounded_claim_count;
  const anyFeedback =
    commentCount > 0 || dislikedCitationCount > 0 || replacementCount > 0;

  return (
    <section className="review-panel-shell" data-testid="review-panel">
      <div className="feedback-action-row">
        <div className="feedback-stats-group">
          <div className="feedback-stats-text" data-testid="feedback-summary">
            {commentCount} comments captured, {replacementCount} citation
            replacements pending.
            {dislikedCitationCount
              ? ` · ${dislikedCitationCount} disliked citations`
              : ""}
            {resolvedCount ? ` · ${resolvedCount} resolved comments` : ""}
            {removedSentenceCount
              ? ` · ${removedSentenceCount} unverified removed`
              : ""}
          </div>
        </div>

        <Button
          className="review-panel-refine-button"
          data-testid="run-refinement"
          disabled={!interactive || !anyFeedback || busy !== null}
          variant="secondary"
          onClick={async () => {
            setBusy("refine");
            try {
              await onRefine();
            } finally {
              startTransition(() => setBusy(null));
            }
          }}
        >
          {busy === "refine" ? "Refining..." : "Refine"}
        </Button>
      </div>
    </section>
  );
}
