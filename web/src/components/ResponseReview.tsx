import {
  startTransition,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { MessageSquare, ThumbsUp, ThumbsDown } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import type {
  MatchQuality,
  ParsedSentence,
  Reference,
  SessionState,
} from "../types";
import { buildDisplayCitationMap } from "../utils/citationDisplay";
import {
  describeUnifiedMatchQuality,
  hasExactQuoteMatch,
  referenceQuoteCoverage,
} from "../utils/retrievalDisplay";

interface ResponseReviewProps {
  session: SessionState;
  readOnly: boolean;
  onOpenCitation: (
    sentence: ParsedSentence,
    citationId: number,
    referenceQuote: string,
  ) => void;
  onSaveComment: (selection: {
    text_selection: string;
    char_start: number;
    char_end: number;
    comment_text: string;
  }) => Promise<void>;
  onUpdateComment: (commentId: string, commentText: string) => Promise<void>;
  onDeleteComment: (commentId: string) => Promise<void>;
}

type SelectionDraft = {
  text: string;
  start: number;
  end: number;
  rect: DOMRect; // bounding rect of the first line (used for popup anchor)
  spanTop: number; // viewport top of the entire selection
  spanBottom: number; // viewport bottom of the entire selection
};

type CommentIndicator = {
  commentId: string;
  commentIds: string[];
  top: number;
};

type PopupState = {
  mode: "new" | "existing";
  rect: DOMRect;
  commentIds: string[];
};

function sentenceStatusNote(sentence: ParsedSentence) {
  if (
    sentence.sentence_type === "synthesis" &&
    sentence.status === "ungrounded"
  ) {
    return "This synthesis could not be fully verified. The reasoning may be valid, but QUARRY could not link it to exact source text.";
  }
  return null;
}

function matchQualityTooltipLabel(quality: MatchQuality): string {
  switch (quality) {
    case "strong":
      return "Good match";
    case "partial":
      return "Partial match";
    case "none":
      return "No match";
    default:
      return "No match";
  }
}

function clampPosition(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

const COMMENT_GUTTER_INSET = 4;
const COMMENT_TRIGGER_SIZE = 32;
const COMMENT_CARD_FALLBACK_WIDTH = 420;
const SENTENCE_COPY_SELECTOR = "[data-sentence-copy-index]";

function _closestSentenceCopy(node: Node | null): HTMLElement | null {
  if (!node) {
    return null;
  }
  if (node instanceof HTMLElement) {
    return node.closest<HTMLElement>(SENTENCE_COPY_SELECTOR);
  }
  return (
    node.parentElement?.closest<HTMLElement>(SENTENCE_COPY_SELECTOR) ?? null
  );
}

function _rangePointOffsetInSentence(
  sentenceCopyElement: HTMLElement,
  boundaryContainer: Node,
  boundaryOffset: number,
): number | null {
  const localRange = document.createRange();
  localRange.selectNodeContents(sentenceCopyElement);
  try {
    localRange.setEnd(boundaryContainer, boundaryOffset);
  } catch {
    return null;
  }
  return localRange.toString().length;
}

export function ResponseReview({
  session,
  readOnly,
  onOpenCitation,
  onSaveComment,
  onUpdateComment,
  onDeleteComment,
}: ResponseReviewProps) {
  const contentRef = useRef<HTMLDivElement | null>(null);
  const popupRef = useRef<HTMLDivElement | null>(null);
  const [selectionDraft, setSelectionDraft] = useState<SelectionDraft | null>(
    null,
  );
  const [popupState, setPopupState] = useState<PopupState | null>(null);

  const citationFeedbackMap = useMemo(() => {
    const map = new Map<string, "like" | "dislike" | "neutral">();
    session.feedback.citation_feedback?.forEach((fb) => {
      map.set(`${fb.sentence_index}:${fb.citation_id}`, fb.feedback_type);
    });
    return map;
  }, [session.feedback.citation_feedback]);
  const [draftNote, setDraftNote] = useState("");
  const [editNotes, setEditNotes] = useState<Record<string, string>>({});
  const [commentIndicators, setCommentIndicators] = useState<
    CommentIndicator[]
  >([]);

  const displayCitationMap = useMemo(
    () => buildDisplayCitationMap(session),
    [session],
  );
  const citationById = useMemo(
    () =>
      new Map(
        session.citation_index.map((citation) => [
          citation.citation_id,
          citation,
        ]),
      ),
    [session.citation_index],
  );
  const visibleSentences = useMemo(
    () =>
      (session.parsed_sentences ?? []).filter(
        (sentence) => sentence.sentence_text.trim().length > 0,
      ),
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

  const sentenceRanges = useMemo(() => {
    const ranges = new Map<number, { start: number; end: number }>();
    let cursor = 0;
    for (const sentence of visibleSentences) {
      const start = cursor;
      const end = start + sentence.sentence_text.length;
      ranges.set(sentence.sentence_index, { start, end });
      cursor = end + 1;
    }
    return ranges;
  }, [visibleSentences]);

  const comments = session.feedback.comments ?? [];
  const commentsById = useMemo(
    () => new Map(comments.map((comment) => [comment.comment_id, comment])),
    [comments],
  );
  const hasCommentOverlay = Boolean(selectionDraft || popupState);

  function dismissPopup() {
    setPopupState(null);
    setSelectionDraft(null);
    setDraftNote("");
    setEditNotes({});
    window.getSelection()?.removeAllRanges();
  }

  // Measure annotation highlight positions to render margin indicators (Google Docs style)
  useLayoutEffect(() => {
    const container = contentRef.current;
    if (!container || !comments.length) {
      setCommentIndicators([]);
      return;
    }

    const containerRect = container.getBoundingClientRect();
    const seen = new Map<string, CommentIndicator>();

    const highlights =
      container.querySelectorAll<HTMLElement>("[data-comment-id]");
    for (const el of highlights) {
      const ids = (el.dataset.commentId ?? "").split(",").filter(Boolean);
      for (const id of ids) {
        if (seen.has(id)) continue;
        const rect = el.getBoundingClientRect();
        const top =
          rect.top - containerRect.top + container.scrollTop + rect.height / 2;
        seen.set(id, { commentId: id, commentIds: ids, top });
      }
    }

    setCommentIndicators([...seen.values()]);
  }, [comments, paragraphs, popupState]);

  useEffect(() => {
    function handleMouseDown(event: MouseEvent) {
      const target = event.target as Node;

      // Never dismiss if clicking inside the popup card
      if (popupRef.current?.contains(target)) return;

      if (
        (event.target as HTMLElement).closest?.(
          '[data-testid="selection-comment-trigger"]',
        )
      ) {
        return;
      }

      if (popupState) {
        dismissPopup();
        return;
      }

      if (contentRef.current?.contains(target)) return;

      dismissPopup();
    }
    document.addEventListener("mousedown", handleMouseDown);
    return () => document.removeEventListener("mousedown", handleMouseDown);
  }, [popupState]);

  function handleMouseUp(event: React.MouseEvent) {
    if (readOnly) return;
    // Ignore mouseup events originating inside the comment card
    // (e.g. clicking the textarea) — they bubble up and would clear the draft
    if (popupRef.current?.contains(event.target as Node)) return;
    // Ignore clicks on the comment trigger button — let its onClick handle the transition
    if (
      (event.target as HTMLElement).closest?.(
        '[data-testid="selection-comment-trigger"]',
      )
    )
      return;
    // Ignore clicks on comment margin indicators — they open existing comments
    if ((event.target as HTMLElement).closest?.(".comment-margin-indicator"))
      return;

    const selected = window.getSelection();
    if (!selected || selected.rangeCount === 0 || selected.isCollapsed) {
      // Don't clear draft if a popup is open — user may just be clicking inside the card
      if (!popupState) setSelectionDraft(null);
      return;
    }

    const range = selected.getRangeAt(0);
    const container = contentRef.current;
    if (!container) return;
    if (!container.contains(range.commonAncestorContainer)) return;

    const selectedText = selected.toString().trim();
    if (!selectedText) {
      setSelectionDraft(null);
      return;
    }

    const startSentenceCopy = _closestSentenceCopy(range.startContainer);
    const endSentenceCopy = _closestSentenceCopy(range.endContainer);
    if (!startSentenceCopy || !endSentenceCopy) {
      setSelectionDraft(null);
      return;
    }

    const startSentenceIndex = Number(
      startSentenceCopy.dataset.sentenceCopyIndex,
    );
    const endSentenceIndex = Number(endSentenceCopy.dataset.sentenceCopyIndex);
    if (
      !Number.isInteger(startSentenceIndex) ||
      !Number.isInteger(endSentenceIndex)
    ) {
      setSelectionDraft(null);
      return;
    }

    const startSentenceRange = sentenceRanges.get(startSentenceIndex);
    const endSentenceRange = sentenceRanges.get(endSentenceIndex);
    if (!startSentenceRange || !endSentenceRange) {
      setSelectionDraft(null);
      return;
    }

    const localStart = _rangePointOffsetInSentence(
      startSentenceCopy,
      range.startContainer,
      range.startOffset,
    );
    const localEnd = _rangePointOffsetInSentence(
      endSentenceCopy,
      range.endContainer,
      range.endOffset,
    );
    if (localStart === null || localEnd === null) {
      setSelectionDraft(null);
      return;
    }

    const start = Math.max(0, startSentenceRange.start + localStart);
    const end = Math.max(start, endSentenceRange.start + localEnd);

    const sentenceConcatenated = visibleSentences
      .map((s) => s.sentence_text)
      .join(" ");
    const cleanSelected = sentenceConcatenated
      .slice(start, end)
      .replace(/\s+/g, " ")
      .trim();
    if (!cleanSelected) return; // Should not trigger for just selecting brackets or spaces

    const rect = range.getBoundingClientRect();
    if (!rect.width && !rect.height) return;

    // Use getClientRects() to get the true vertical span across all lines
    const clientRects = Array.from(range.getClientRects()).filter(
      (r) => r.width > 0 && r.height > 0,
    );
    const spanTop = clientRects.length
      ? Math.min(...clientRects.map((r) => r.top))
      : rect.top;
    const spanBottom = clientRects.length
      ? Math.max(...clientRects.map((r) => r.bottom))
      : rect.bottom;

    setSelectionDraft({
      text: cleanSelected || selectedText,
      start,
      end,
      rect,
      spanTop,
      spanBottom,
    });
    setPopupState(null);
    setDraftNote("");
    setEditNotes({});
  }

  function openExistingPopup(commentIds: string[], rect: DOMRect) {
    setSelectionDraft(null);
    setDraftNote("");
    setEditNotes((current) => {
      const next = { ...current };
      for (const id of commentIds) {
        const comment = commentsById.get(id);
        if (comment && !next[id]) {
          next[id] = comment.comment_text;
        }
      }
      return next;
    });
    setPopupState({ mode: "existing", rect, commentIds });
    const selection = window.getSelection();
    selection?.removeAllRanges();
  }

  const popupPosition = useMemo(() => {
    if (!popupState || !contentRef.current) return null;
    const containerRect = contentRef.current.getBoundingClientRect();
    const popupWidth =
      popupRef.current?.offsetWidth ?? COMMENT_CARD_FALLBACK_WIDTH;
    const left = clampPosition(
      containerRect.width - popupWidth - COMMENT_GUTTER_INSET,
      0,
      Math.max(0, containerRect.width - popupWidth),
    );

    if (popupState.mode === "new" && selectionDraft) {
      const selectionMidpoint =
        (selectionDraft.spanTop + selectionDraft.spanBottom) / 2;
      const top =
        selectionMidpoint -
        containerRect.top +
        contentRef.current.scrollTop -
        COMMENT_TRIGGER_SIZE / 2;
      return { left, top };
    }

    // "existing" mode: position near the clicked annotation highlight (convert to container coords)
    const top =
      popupState.rect.top - containerRect.top + contentRef.current.scrollTop;
    return { left, top };
  }, [popupState, selectionDraft]);

  const triggerPosition = useMemo(() => {
    if (!selectionDraft || !contentRef.current) return null;
    const containerRect = contentRef.current.getBoundingClientRect();

    // Keep the trigger inside the reserved right gutter so it never widens the page.
    const left = Math.max(
      0,
      containerRect.width - COMMENT_TRIGGER_SIZE - COMMENT_GUTTER_INSET,
    );

    // Vertical: true center across ALL selected lines, relative to the container
    // spanTop/spanBottom are viewport coords; containerRect.top is also viewport
    // contentRef.current.scrollTop handles any scrolling inside the container
    const selectionMidpoint =
      (selectionDraft.spanTop + selectionDraft.spanBottom) / 2;
    const top =
      selectionMidpoint -
      containerRect.top +
      contentRef.current.scrollTop -
      COMMENT_TRIGGER_SIZE / 2;

    return { left, top };
  }, [selectionDraft]);

  return (
    <section className="response-review">
      <div
        className="response-reading-flow"
        onMouseUp={handleMouseUp}
        ref={contentRef}
      >
        {!paragraphs.length ? (
          <Alert className="response-empty-state border-border/70 bg-card/90">
            <AlertTitle className="tiny-label">
              No verified answer to show
            </AlertTitle>
            <AlertDescription>
              I couldn't prepare a grounded response from the current evidence.
              You can try a narrower question or ask QUARRY to try again with
              different feedback.
            </AlertDescription>
          </Alert>
        ) : null}

        {paragraphs.map(([paragraphIndex, paragraphSentences]) => (
          <div
            className="response-paragraph"
            key={`paragraph-${paragraphIndex}`}
          >
            {paragraphSentences.map((sentence) => {
              const matchQuality = sentence.match_quality ?? "none";
              const range = sentenceRanges.get(sentence.sentence_index);
              const sentenceStart = range?.start ?? 0;
              const sentenceEnd = range?.end ?? sentenceStart;

              const overlapping = comments
                .filter(
                  (comment) =>
                    comment.char_start < sentenceEnd &&
                    comment.char_end > sentenceStart,
                )
                .map((comment) => ({
                  id: comment.comment_id,
                  start: Math.max(0, comment.char_start - sentenceStart),
                  end: Math.min(
                    sentence.sentence_text.length,
                    comment.char_end - sentenceStart,
                  ),
                }))
                .filter((segment) => segment.end > segment.start);

              // Draft selection highlight: when the comment popup is open, render the
              // selected text range as a DOM-based highlight so it persists even after
              // focus moves to the textarea (native ::selection disappears on blur).
              const hasDraftHighlight =
                popupState?.mode === "new" &&
                selectionDraft &&
                selectionDraft.start < sentenceEnd &&
                selectionDraft.end > sentenceStart;
              const draftLocalStart = hasDraftHighlight
                ? Math.max(0, selectionDraft!.start - sentenceStart)
                : 0;
              const draftLocalEnd = hasDraftHighlight
                ? Math.min(
                    sentence.sentence_text.length,
                    selectionDraft!.end - sentenceStart,
                  )
                : 0;

              const boundaries = new Set<number>([
                0,
                sentence.sentence_text.length,
              ]);
              for (const segment of overlapping) {
                boundaries.add(segment.start);
                boundaries.add(segment.end);
              }
              if (hasDraftHighlight && draftLocalEnd > draftLocalStart) {
                boundaries.add(draftLocalStart);
                boundaries.add(draftLocalEnd);
              }
              const boundaryList = [...boundaries].sort((a, b) => a - b);

              return (
                <div
                  className="response-inline-sentence"
                  data-testid={`sentence-${sentence.sentence_index}`}
                  key={sentence.sentence_index}
                >
                  <span className="response-inline-sentence-text">
                    <span
                      className="response-inline-sentence-copy"
                      data-sentence-copy-index={sentence.sentence_index}
                    >
                      {boundaryList.slice(0, -1).map((segmentStart, i) => {
                        const segmentEnd = boundaryList[i + 1];
                        const segmentText = sentence.sentence_text.slice(
                          segmentStart,
                          segmentEnd,
                        );
                        const globalStart = sentenceStart + segmentStart;
                        const segmentCommentIds = comments
                          .filter(
                            (comment) =>
                              comment.char_start <
                                globalStart + segmentText.length &&
                              comment.char_end > globalStart,
                          )
                          .map((comment) => comment.comment_id);
                        const inDraft =
                          hasDraftHighlight &&
                          draftLocalEnd > draftLocalStart &&
                          segmentStart >= draftLocalStart &&
                          segmentStart < draftLocalEnd;
                        if (!segmentCommentIds.length) {
                          return (
                            <span
                              key={`${sentence.sentence_index}-seg-${i}`}
                              className={
                                inDraft
                                  ? "draft-selection-highlight"
                                  : undefined
                              }
                            >
                              {segmentText}
                            </span>
                          );
                        }
                        return (
                          <span
                            key={`${sentence.sentence_index}-seg-${i}`}
                            className={`annotation-highlight${inDraft ? " draft-selection-highlight" : ""}`}
                            data-testid={`annotation-highlight-${sentence.sentence_index}-${i}`}
                            data-comment-id={segmentCommentIds.join(",")}
                            role="button"
                            tabIndex={0}
                            onClick={(event) =>
                              openExistingPopup(
                                segmentCommentIds,
                                (
                                  event.currentTarget as HTMLElement
                                ).getBoundingClientRect(),
                              )
                            }
                            onKeyDown={(event) => {
                              if (event.key === "Enter" || event.key === " ") {
                                event.preventDefault();
                                openExistingPopup(
                                  segmentCommentIds,
                                  (
                                    event.currentTarget as HTMLElement
                                  ).getBoundingClientRect(),
                                );
                              }
                            }}
                          >
                            {segmentText}
                          </span>
                        );
                      })}
                    </span>{" "}
                    <span className="response-inline-citations">
                      {sentence.references.map((reference, index) => {
                        if (!reference.citation_id) return null;

                        const citation = citationById.get(
                          reference.citation_id,
                        );
                        const unifiedMatch = citation
                          ? describeUnifiedMatchQuality(
                              citation.retrieval_score,
                              citation.ambiguity_review_required,
                              citation.ambiguity_gap,
                              {
                                sentenceStatus: sentence.status,
                                referenceVerified: reference.verified,
                                referenceQuoteCoverage: referenceQuoteCoverage(
                                  sentence.sentence_text,
                                  reference.reference_quote,
                                ),
                                referenceQuoteExactMatch: hasExactQuoteMatch(
                                  citation.text,
                                  reference.reference_quote,
                                ),
                                referenceConfidenceLabel:
                                  reference.confidence_label,
                                referenceConfidenceUnknown:
                                  reference.confidence_unknown,
                              },
                            )
                          : null;
                        const pillQuality: MatchQuality = unifiedMatch
                          ? unifiedMatch.level === "strong" ||
                            unifiedMatch.level === "good"
                            ? "strong"
                            : unifiedMatch.level === "fair"
                              ? "partial"
                              : "none"
                          : matchQuality;
                        if (pillQuality === "none") return null;

                        const feedback = citationFeedbackMap.get(
                          `${sentence.sentence_index}:${reference.citation_id}`,
                        );
                        const tooltipQuote =
                          reference.reference_quote?.trim() ?? "";
                        const tooltipDoc =
                          reference.document_title?.trim() || "—";
                        const tooltipMatchLabel =
                          unifiedMatch?.headline ??
                          matchQualityTooltipLabel(pillQuality);

                        return (
                          <Button
                            className={`citation-pill citation-pill--${pillQuality} ${reference.replacement_pending ? "replaced" : ""} ${
                              hasCommentOverlay
                                ? "tooltip-suppressed"
                                : "citation-pill-tooltip-anchor"
                            }`}
                            data-testid={`citation-${sentence.sentence_index}-${reference.citation_id}`}
                            key={`${sentence.sentence_index}-${index}`}
                            onClick={() =>
                              onOpenCitation(
                                sentence,
                                reference.citation_id!,
                                reference.reference_quote,
                              )
                            }
                            type="button"
                          >
                            [
                            {displayCitationMap.get(reference.citation_id) ??
                              reference.citation_id}
                            ]
                            {feedback === "like" && (
                              <ThumbsUp
                                aria-label="Liked"
                                className="ml-1 inline"
                              />
                            )}
                            {feedback === "dislike" && (
                              <ThumbsDown
                                aria-label="Disliked"
                                className="ml-1 inline"
                              />
                            )}
                            {!hasCommentOverlay ? (
                              <span
                                className="citation-pill-tooltip"
                                role="tooltip"
                              >
                                <span className="citation-pill-tooltip-quote">
                                  {tooltipQuote}
                                </span>
                                <hr className="citation-pill-tooltip-rule" />
                                <span className="citation-pill-tooltip-doc">
                                  {tooltipDoc}
                                </span>
                                <hr className="citation-pill-tooltip-rule" />
                                <span className="citation-pill-tooltip-match">
                                  {tooltipMatchLabel}
                                </span>
                              </span>
                            ) : null}
                          </Button>
                        );
                      })}
                    </span>
                  </span>

                  {sentenceStatusNote(sentence) ? (
                    <Alert
                      className="sentence-status-note mt-3 border-warning-medium/70 bg-[var(--warning-surface)] text-[var(--warning-ink)]"
                      data-testid={`sentence-note-${sentence.sentence_index}`}
                    >
                      <AlertDescription>
                        {sentenceStatusNote(sentence)}
                      </AlertDescription>
                    </Alert>
                  ) : null}
                </div>
              );
            })}
          </div>
        ))}

        {selectionDraft && triggerPosition && !popupState ? (
          <Button
            className="selection-comment-trigger"
            data-testid="selection-comment-trigger"
            style={{
              left: `${triggerPosition.left}px`,
              top: `${triggerPosition.top}px`,
              position: "absolute",
            }}
            onMouseDown={(e) => e.preventDefault()}
            type="button"
            onClick={() => {
              setPopupState({
                mode: "new",
                rect: selectionDraft.rect,
                commentIds: [],
              });
              // Clear native selection — the DOM-based draft highlight takes over
              window.getSelection()?.removeAllRanges();
            }}
          >
            <MessageSquare aria-hidden="true" focusable="false" />
            <span className="sr-only">Add comment</span>
          </Button>
        ) : null}

        {commentIndicators.map((indicator) => {
          const container = contentRef.current;
          if (!container) return null;
          const containerRect = container.getBoundingClientRect();
          const left = Math.max(
            0,
            containerRect.width - COMMENT_TRIGGER_SIZE - COMMENT_GUTTER_INSET,
          );
          return (
            <Button
              key={`comment-indicator-${indicator.commentId}`}
              className="comment-margin-indicator"
              data-testid={`comment-indicator-${indicator.commentId}`}
              style={{
                left: `${left}px`,
                top: `${indicator.top - 10}px`,
                position: "absolute",
              }}
              onMouseDown={(e) => e.preventDefault()}
              type="button"
              onClick={(e) => {
                openExistingPopup(
                  indicator.commentIds,
                  (e.currentTarget as HTMLElement).getBoundingClientRect(),
                );
              }}
            >
              <MessageSquare aria-hidden="true" focusable="false" />
            </Button>
          );
        })}

        {popupState && popupPosition ? (
          <Card
            className="selection-comment-card gap-0 border-border/70 bg-card/98 py-0"
            data-testid="selection-comment-card"
            ref={popupRef}
            style={{
              left: `${popupPosition.left}px`,
              top: `${popupPosition.top}px`,
              position: "absolute",
            }}
            onMouseDown={(e) => e.preventDefault()}
          >
            {popupState.mode === "new" && selectionDraft ? (
              <CardContent
                className="selection-comment-compose p-0"
                data-testid="selection-comment-editor"
              >
                <Textarea
                  autoFocus
                  data-testid="selection-comment-input"
                  value={draftNote}
                  onChange={(event) => setDraftNote(event.target.value)}
                  placeholder="Add a comment"
                />
                <div className="selection-comment-actions">
                  <button
                    className="text-button selection-comment-cancel"
                    data-testid="cancel-selection-comment"
                    onClick={dismissPopup}
                    type="button"
                  >
                    Cancel
                  </button>
                  <Button
                    data-testid="save-selection-comment"
                    disabled={!draftNote.trim()}
                    type="button"
                    variant="secondary"
                    onClick={async () => {
                      await onSaveComment({
                        text_selection: selectionDraft.text,
                        char_start: selectionDraft.start,
                        char_end: selectionDraft.end,
                        comment_text: draftNote.trim(),
                      });
                      startTransition(() => {
                        dismissPopup();
                      });
                    }}
                  >
                    Comment
                  </Button>
                </div>
              </CardContent>
            ) : null}

            {popupState.mode === "existing" ? (
              <CardContent
                className="selection-comment-thread p-0"
                data-testid="selection-comment-active-editor"
              >
                {popupState.commentIds.map((commentId) => {
                  const comment = commentsById.get(commentId);
                  if (!comment) return null;
                  const value = editNotes[commentId] ?? comment.comment_text;
                  return (
                    <div className="selection-comment-item" key={commentId}>
                      <Textarea
                        data-testid={`selection-comment-edit-input-${commentId}`}
                        value={value}
                        onChange={(event) =>
                          setEditNotes((current) => ({
                            ...current,
                            [commentId]: event.target.value,
                          }))
                        }
                      />
                      <div className="selection-comment-item-actions">
                        <Button
                          data-testid={`delete-selection-comment-${commentId}`}
                          type="button"
                          variant="ghost"
                          onClick={async () => {
                            await onDeleteComment(commentId);
                            startTransition(() => {
                              setPopupState((current) => {
                                if (!current || current.mode !== "existing")
                                  return current;
                                const nextIds = current.commentIds.filter(
                                  (id) => id !== commentId,
                                );
                                if (!nextIds.length) return null;
                                return { ...current, commentIds: nextIds };
                              });
                            });
                          }}
                        >
                          Delete
                        </Button>
                        <Button
                          data-testid={`update-selection-comment-${commentId}`}
                          disabled={!value.trim()}
                          type="button"
                          variant="secondary"
                          onClick={async () => {
                            await onUpdateComment(commentId, value.trim());
                          }}
                        >
                          Save
                        </Button>
                      </div>
                    </div>
                  );
                })}
              </CardContent>
            ) : null}
          </Card>
        ) : null}
      </div>
    </section>
  );
}
