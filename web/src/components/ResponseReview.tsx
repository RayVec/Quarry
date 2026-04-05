import { startTransition, useMemo, useState } from "react";
import { MessageSquare } from "lucide-react";
import type { ParsedSentence, Reference, SessionState } from "../types";
import { buildDisplayCitationMap } from "../utils/citationDisplay";

interface ResponseReviewProps {
  session: SessionState;
  readOnly: boolean;
  onOpenCitation: (sentence: ParsedSentence, citationId: number, referenceQuote: string) => void;
  onSaveComment: (sentenceIndex: number, note: string) => Promise<void>;
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
}: ResponseReviewProps) {
  const [editingSentenceIndex, setEditingSentenceIndex] = useState<number | null>(null);
  const [note, setNote] = useState("");
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

  return (
    <section className="response-review">
      <div className="response-reading-flow">
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
                          setEditingSentenceIndex((current) =>
                            current === sentence.sentence_index ? null : sentence.sentence_index,
                          );
                          setNote("");
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

                  {commentable && editingSentenceIndex === sentence.sentence_index ? (
                    <div className="inline-note-editor">
                      <textarea
                        data-testid="disagreement-note"
                        value={note}
                        onChange={(event) => setNote(event.target.value)}
                        placeholder="Leave your comments on this sentence"
                      />
                      <div className="inline-note-actions">
                        <button
                          className="ghost-button"
                          onClick={() => {
                            setEditingSentenceIndex(null);
                            setNote("");
                          }}
                        >
                          Cancel
                        </button>
                        <button
                          className="primary-button subtle"
                          data-testid="save-disagreement"
                          disabled={!note.trim()}
                          onClick={async () => {
                            await onSaveComment(sentence.sentence_index, note.trim());
                            startTransition(() => {
                              setEditingSentenceIndex(null);
                              setNote("");
                            });
                          }}
                        >
                          Save note
                        </button>
                      </div>
                    </div>
                  ) : null}
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </section>
  );
}
