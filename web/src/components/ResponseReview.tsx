import { startTransition, useMemo, useRef, useState } from "react";
import { MessageSquare } from "lucide-react";
import type { ParsedSentence, Reference, SessionState } from "../types";
import { buildDisplayCitationMap } from "../utils/citationDisplay";

interface ResponseReviewProps {
  session: SessionState;
  readOnly: boolean;
  onOpenCitation: (sentence: ParsedSentence, citationId: number, referenceQuote: string) => void;
  onSaveComment: (selection: {
    text_selection: string;
    char_start: number;
    char_end: number;
    comment_text: string;
  }) => Promise<void>;
  onUpdateComment: (commentId: string, commentText: string) => Promise<void>;
  onDeleteComment: (commentId: string) => Promise<void>;
}

function sentenceStatusNote(sentence: ParsedSentence) {
  if (sentence.sentence_type === "synthesis" && sentence.status === "ungrounded") {
    return "This synthesis could not be fully verified. The reasoning may be valid, but QUARRY could not link it to exact source text.";
  }
  return null;
}

function sentenceRailTooltip(sentence: ParsedSentence) {
  if (sentence.warnings.includes("structural_fact")) {
    return "This sentence may contain an unsourced factual claim.";
  }
  if (sentence.warnings.includes("confidence_unknown")) {
    return "Verifier confidence is unavailable for this sentence.";
  }
  switch (sentence.status) {
    case "verified":
      return "This sentence is verified against source text.";
    case "partially_verified":
      return "This sentence is only partially verified by the source text.";
    case "ungrounded":
      return "This sentence could not be grounded to source text.";
    case "no_ref":
      return "This sentence has no supporting citation.";
    default:
      return "";
  }
}

function referenceTooltip(reference: Reference) {
  const lines = [reference.reference_quote, reference.document_title, reference.section_heading]
    .map((line) => line?.trim())
    .filter((line): line is string => Boolean(line));
  return lines.join("\n");
}

export function ResponseReview({
  session,
  readOnly,
  onOpenCitation,
  onSaveComment,
  onUpdateComment,
  onDeleteComment,
}: ResponseReviewProps) {
  const [selectionDraft, setSelectionDraft] = useState<{
    text: string;
    start: number;
    end: number;
  } | null>(null);
  const [note, setNote] = useState("");
  const [activeCommentId, setActiveCommentId] = useState<string | null>(null);
  const contentRef = useRef<HTMLDivElement | null>(null);
  const displayCitationMap = useMemo(() => buildDisplayCitationMap(session), [session]);
  const visibleSentences = useMemo(
    () => (session.parsed_sentences ?? []).filter((sentence) => sentence.sentence_text.trim().length > 0),
    [session.parsed_sentences],
  );
  const paragraphs = useMemo(() => {
    const grouped = new Map<number, ParsedSentence[]>();
    for (const sentence of visibleSentences) {
      const key = sentence.paragraph_index ?? 0;
      if (!grouped.has(key)) grouped.set(key, []);
      grouped.get(key)!.push(sentence);
    }
    return [...grouped.entries()].sort((a, b) => a[0] - b[0]);
  }, [visibleSentences]);
  const flatText = useMemo(
    () => visibleSentences.map((sentence) => sentence.sentence_text).join(" "),
    [visibleSentences],
  );
  const activeComment = useMemo(
    () => session.feedback.comments.find((comment) => comment.comment_id === activeCommentId) ?? null,
    [session.feedback.comments, activeCommentId],
  );

  function handleMouseUp() {
    if (readOnly) return;
    const selected = window.getSelection();
    if (!selected || selected.isCollapsed) return;
    const selectedText = selected.toString().trim();
    if (!selectedText) return;
    const start = flatText.indexOf(selectedText);
    if (start < 0) return;
    setSelectionDraft({ text: selectedText, start, end: start + selectedText.length });
    setNote("");
    setActiveCommentId(null);
  }

  return (
    <section className="response-review">
      <div className="response-reading-flow" onMouseUp={handleMouseUp} ref={contentRef}>
        {!paragraphs.length ? (
          <div className="response-empty-state">
            <span className="tiny-label">No verified answer to show</span>
            <p>I couldn't prepare a grounded response from the current evidence. You can try a narrower question or ask QUARRY to try again with different feedback.</p>
          </div>
        ) : null}
        {paragraphs.map(([paragraphIndex, paragraphSentences]) => (
          <div className="response-paragraph" key={`paragraph-${paragraphIndex}`}>
            {paragraphSentences.map((sentence) => {
              const commentable = true;
              const matchQuality = sentence.match_quality ?? "none";
              return (
                <div
                  className="response-inline-sentence"
                  data-testid={`sentence-${sentence.sentence_index}`}
                  key={sentence.sentence_index}
                >
                  <span className="response-inline-sentence-text">
                    {sentence.sentence_text}{" "}
                    {sentence.references.map((reference, index) =>
                      reference.citation_id && matchQuality !== "none" ? (
                        <button
                          className={`citation-pill citation-pill--${matchQuality} ${reference.replacement_pending ? "replaced" : ""} hover-tooltip`}
                          data-testid={`citation-${sentence.sentence_index}-${reference.citation_id}`}
                          key={`${sentence.sentence_index}-${index}`}
                          data-tooltip={referenceTooltip(reference)}
                          onClick={() => onOpenCitation(sentence, reference.citation_id!, reference.reference_quote)}
                        >
                          [{displayCitationMap.get(reference.citation_id) ?? reference.citation_id}]
                        </button>
                      ) : null,
                    )}
                  </span>
                  <span className="response-inline-sentence-actions">
                    {commentable ? (
                      <button
                        className={`inline-flag-button hover-tooltip ${sentenceRailTooltip(sentence) ? "" : "no-tooltip"}`}
                        data-testid={`disagree-${sentence.sentence_index}`}
                        data-tooltip={sentenceRailTooltip(sentence) || "Make comments"}
                        onClick={() => {
                          const fallbackText = sentence.sentence_text.trim();
                          const start = flatText.indexOf(fallbackText);
                          if (start >= 0) {
                            setSelectionDraft({
                              text: fallbackText,
                              start,
                              end: start + fallbackText.length,
                            });
                            setNote("");
                            setActiveCommentId(null);
                          }
                        }}
                      >
                        <span className="sr-only">Make comments</span>
                        <MessageSquare aria-hidden="true" focusable="false" />
                      </button>
                    ) : null}
                  </span>

                  {sentenceStatusNote(sentence) ? (
                    <p className="sentence-status-note" data-testid={`sentence-note-${sentence.sentence_index}`}>
                      {sentenceStatusNote(sentence)}
                    </p>
                  ) : null}

                </div>
              );
            })}
          </div>
        ))}

        {selectionDraft ? (
          <div className="inline-note-editor selection-comment-editor" data-testid="selection-comment-editor">
            <span className="tiny-label">Selected text</span>
            <p className="selection-preview">"{selectionDraft.text}"</p>
            <textarea
              data-testid="selection-comment-input"
              value={note}
              onChange={(event) => setNote(event.target.value)}
              placeholder="Leave your comment on this selection"
            />
            <div className="inline-note-actions">
              <button
                className="ghost-button"
                onClick={() => {
                  setSelectionDraft(null);
                  setNote("");
                }}
              >
                Cancel
              </button>
              <button
                className="primary-button subtle"
                data-testid="save-selection-comment"
                disabled={!note.trim()}
                onClick={async () => {
                  await onSaveComment({
                    text_selection: selectionDraft.text,
                    char_start: selectionDraft.start,
                    char_end: selectionDraft.end,
                    comment_text: note.trim(),
                  });
                  startTransition(() => {
                    setSelectionDraft(null);
                    setNote("");
                  });
                }}
              >
                Save comment
              </button>
            </div>
          </div>
        ) : null}

        {activeComment ? (
          <div className="inline-note-editor selection-comment-editor" data-testid="selection-comment-active-editor">
            <span className="tiny-label">Selected text</span>
            <p className="selection-preview">"{activeComment.text_selection}"</p>
            <textarea
              data-testid="selection-comment-edit-input"
              value={note || activeComment.comment_text}
              onChange={(event) => setNote(event.target.value)}
              placeholder="Edit comment"
            />
            <div className="inline-note-actions">
              <button
                className="ghost-button"
                onClick={() => {
                  setActiveCommentId(null);
                  setNote("");
                }}
              >
                Close
              </button>
              <button
                className="ghost-button"
                data-testid="delete-selection-comment"
                onClick={async () => {
                  await onDeleteComment(activeComment.comment_id);
                  startTransition(() => {
                    setActiveCommentId(null);
                    setNote("");
                  });
                }}
              >
                Delete
              </button>
              <button
                className="primary-button subtle"
                data-testid="update-selection-comment"
                disabled={!(note || activeComment.comment_text).trim()}
                onClick={async () => {
                  await onUpdateComment(activeComment.comment_id, (note || activeComment.comment_text).trim());
                  startTransition(() => {
                    setActiveCommentId(null);
                    setNote("");
                  });
                }}
              >
                Save
              </button>
            </div>
          </div>
        ) : null}

        {(session.feedback.comments ?? []).length ? (
          <div className="selection-highlight-list" data-testid="selection-highlight-list">
            {(session.feedback.comments ?? []).map((comment) => (
              <button
                key={comment.comment_id}
                className="selection-highlight-chip"
                data-testid={`selection-highlight-${comment.comment_id}`}
                onClick={() => {
                  setActiveCommentId(comment.comment_id);
                  setNote(comment.comment_text);
                  setSelectionDraft(null);
                }}
              >
                <span>"{comment.text_selection}"</span>
              </button>
            ))}
          </div>
        ) : null}
      </div>
    </section>
  );
}
