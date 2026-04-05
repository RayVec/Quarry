import { startTransition, useState } from "react";
import { MessageSquare } from "lucide-react";
import type { SessionState } from "../types";

interface ReviewPanelProps {
  session: SessionState;
  interactive: boolean;
  onSaveResponseComment: (note: string) => Promise<void>;
  onRefine: () => Promise<void>;
}

export function ReviewPanel({
  session,
  interactive,
  onSaveResponseComment,
  onRefine,
}: ReviewPanelProps) {
  const [busy, setBusy] = useState<"comment" | "refine" | null>(null);
  const [responseComment, setResponseComment] = useState("");
  const [popupOpen, setPopupOpen] = useState(false);
  const [open, setOpen] = useState(false);
  const commentCount = session.feedback?.comments?.length ?? 0;
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
        <div className="feedback-comment-trigger-container">
          <button
            className="icon-button"
            data-testid="toggle-response-comment"
            onClick={() => {
              setPopupOpen((prev) => !prev);
              if (!popupOpen) setResponseComment("");
            }}
            disabled={!interactive}
          >
            <span className="sr-only">Leave comments</span>
            <MessageSquare size={18} aria-hidden="true" focusable="false" />
          </button>

          {popupOpen ? (
            <div className="feedback-comment-popup">
              <textarea
                data-testid="response-comment"
                disabled={!interactive || busy !== null}
                value={responseComment}
                onChange={(event) => setResponseComment(event.target.value)}
                placeholder="Leave comments for the overall response"
              />
              <div className="feedback-comment-popup-actions">
                <button
                  className="ghost-button"
                  disabled={busy !== null}
                  onClick={() => setPopupOpen(false)}
                >
                  Cancel
                </button>
                <button
                  className="primary-button subtle"
                  data-testid="save-response-comment"
                  disabled={!interactive || !responseComment.trim() || busy !== null}
                  onClick={async () => {
                    setBusy("comment");
                    try {
                      await onSaveResponseComment(responseComment.trim());
                      setResponseComment("");
                      setPopupOpen(false);
                    } finally {
                      startTransition(() => setBusy(null));
                    }
                  }}
                >
                  {busy === "comment" ? "Saving..." : "Save"}
                </button>
              </div>
            </div>
          ) : null}
        </div>

        <div className="feedback-stats-text" data-testid="feedback-summary">
          {commentCount} comments captured, {replacementCount} citation replacements pending.
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
