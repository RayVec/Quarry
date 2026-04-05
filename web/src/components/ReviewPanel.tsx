import { startTransition, useState } from "react";
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
  const [open, setOpen] = useState(false);
  const commentCount = session.feedback?.comments?.length ?? 0;
  const resolvedCount = session.feedback?.resolved_comments?.length ?? 0;
  const replacementCount = session.feedback?.citation_replacements?.length ?? 0;
  const removedSentenceCount = session.removed_ungrounded_claim_count;
  const anyFeedback = commentCount > 0 || replacementCount > 0;

  return (
    <section className={`review-panel-shell ${open ? "open" : ""}`} data-testid="review-panel">
      <button
        className="review-panel-toggle"
        data-testid="toggle-review-panel"
        onClick={() => setOpen((value) => !value)}
      >
        <span>Review and refine</span>
        <span>{open ? "Hide" : "Open"}</span>
      </button>
      {open ? (
        <div className="feedback-action-row">
          <div className="feedback-stats-group">
            <div className="feedback-stats-text" data-testid="feedback-summary">
              {commentCount} comments captured, {replacementCount} citation replacements pending.
              {resolvedCount ? ` · ${resolvedCount} resolved comments` : ""}
              {removedSentenceCount ? ` · ${removedSentenceCount} unverified removed` : ""}
            </div>
          </div>

          <button
            className="primary-button subtle"
            data-testid="run-refinement"
            disabled={!interactive || !anyFeedback || busy !== null}
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
          </button>
        </div>
      ) : null}
    </section>
  );
}
