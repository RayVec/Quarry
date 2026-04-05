import {
  startTransition,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Check, Minus, X } from "lucide-react";
import { api } from "../api";
import type { CitationIndexEntry, Reference, SessionState } from "../types";
import { buildDisplayCitationMap } from "../utils/citationDisplay";
import { lockBodyScroll, unlockBodyScroll } from "../utils/bodyScrollLock";
import type { MatchQuality } from "../types";

const MATCH_QUALITY_ICON_SIZE = 18;
const MATCH_QUALITY_ICON_STROKE = 2.3;

function MatchQualityIcon({ level }: { level: MatchQuality }) {
  const common = { size: MATCH_QUALITY_ICON_SIZE, strokeWidth: MATCH_QUALITY_ICON_STROKE, "aria-hidden": true as const };
  switch (level) {
    case "strong":
      return <Check {...common} />;
    case "partial":
      return <Minus {...common} />;
    case "none":
      return null;
    default:
      return <X {...common} />;
  }
}

interface CitationDialogProps {
  citation: CitationIndexEntry;
  sentenceIndex: number;
  referenceQuote: string;
  readOnly: boolean;
  session: SessionState;
  onClose: () => void;
  onSessionUpdate: (session: SessionState) => void;
}

function highlightReference(text: string, referenceQuote: string) {
  if (!referenceQuote || !text.includes(referenceQuote)) {
    return text;
  }
  const [before, after] = text.split(referenceQuote, 2);
  return (
    <>
      {before}
      <mark>{referenceQuote}</mark>
      {after}
    </>
  );
}

function CitationQuotePanel({
  text,
  referenceQuote,
}: {
  text: string;
  referenceQuote: string;
}) {
  const [expanded, setExpanded] = useState(false);
  const [needsToggle, setNeedsToggle] = useState(false);
  const measureRef = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    const el = measureRef.current;
    if (!el) return;
    const measure = () => {
      const maxPx = window.innerHeight * 0.5;
      setNeedsToggle(el.scrollHeight > maxPx + 1);
    };
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [text, referenceQuote]);

  return (
    <>
      <div
        className={`quoted-passage-viewport ${expanded ? "quoted-passage-viewport--expanded" : ""}`}
      >
        <div ref={measureRef}>
          <p className="quoted-passage">{highlightReference(text, referenceQuote)}</p>
        </div>
      </div>
      {needsToggle ? (
        <button
          type="button"
          className="text-button quoted-passage-toggle"
          data-testid="citation-quote-read-more"
          aria-expanded={expanded}
          onClick={() => setExpanded((value) => !value)}
        >
          {expanded ? "Show less" : "Read more"}
        </button>
      ) : null}
    </>
  );
}

export function CitationDialog({
  citation,
  sentenceIndex,
  referenceQuote,
  readOnly,
  session,
  onClose,
  onSessionUpdate,
}: CitationDialogProps) {
  const [scopedResults, setScopedResults] = useState<CitationIndexEntry[]>([]);
  const [loadingMore, setLoadingMore] = useState(false);
  const [mismatchNote, setMismatchNote] = useState("");
  const displayCitationMap = useMemo(
    () => buildDisplayCitationMap(session),
    [session],
  );
  const displayCitationId =
    displayCitationMap.get(citation.citation_id) ?? citation.citation_id;
  const sourceSentence = useMemo(
    () =>
      session.parsed_sentences.find(
        (sentence) => sentence.sentence_index === sentenceIndex,
      ) ?? null,
    [session.parsed_sentences, sentenceIndex],
  );
  const sourceReference = useMemo<Reference | null>(() => {
    if (!sourceSentence) {
      return null;
    }
    return (
      sourceSentence.references.find(
        (reference) =>
          reference.citation_id === citation.citation_id &&
          reference.reference_quote === referenceQuote,
      ) ??
      sourceSentence.references.find(
        (reference) => reference.citation_id === citation.citation_id,
      ) ??
      sourceSentence.references.find(
        (reference) => reference.reference_quote === referenceQuote,
      ) ??
      null
    );
  }, [sourceSentence, citation.citation_id, referenceQuote]);
  const matchQuality = useMemo(() => {
    const level: MatchQuality = sourceSentence?.match_quality ?? "none";
    if (level === "strong") {
      return {
        level,
        headline: "Strong match",
        detail: "This passage clearly supports the sentence.",
      };
    }
    if (level === "partial") {
      return {
        level,
        headline: "Partial match",
        detail: "This passage is grounded, but you may want to inspect the source context.",
      };
    }
    return {
      level,
      headline: "No citation expected",
      detail: "This sentence is structural or inferential, so no citation badge is shown in the response.",
    };
  }, [sourceSentence?.match_quality]);

  useEffect(() => {
    lockBodyScroll();
    return () => unlockBodyScroll();
  }, []);

  async function handleLoadMore() {
    setLoadingMore(true);
    try {
      const response = await api.scopedRetrieval(
        session.session_id,
        sentenceIndex,
        citation.citation_id,
      );
      startTransition(() => {
        setScopedResults(response.citations);
      });
    } finally {
      setLoadingMore(false);
    }
  }

  async function handleReplace(replacementChunkId: string) {
    const response = await api.replaceCitation(
      session.session_id,
      sentenceIndex,
      citation.citation_id,
      replacementChunkId,
    );
    startTransition(() => {
      onSessionUpdate(response.session);
      onClose();
    });
  }

  async function handleMismatch() {
    const response = await api.addComment(
      session.session_id,
      mismatchNote || "Citation does not support this sentence.",
      sentenceIndex,
    );
    startTransition(() => {
      onSessionUpdate(response.session);
      setMismatchNote("");
    });
  }

  async function handleUndoReplacement() {
    const response = await api.undoReplacement(
      session.session_id,
      citation.citation_id,
    );
    startTransition(() => {
      onSessionUpdate(response.session);
      onClose();
    });
  }

  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <aside
        className="citation-drawer"
        data-testid="citation-dialog"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="drawer-header">
          <div>
            <span className="eyebrow">Citation [{displayCitationId}]</span>
          </div>
          <button
            type="button"
            className="icon-button drawer-close-trigger"
            data-testid="close-citation-dialog"
            aria-label="Close citation"
            onClick={onClose}
          >
            <X aria-hidden size={20} strokeWidth={2} />
          </button>
        </div>

        <div className="drawer-stack">
          <section className="drawer-section">
            <span className="tiny-label">Reference quote in context</span>
            <CitationQuotePanel
              key={citation.citation_id}
              text={citation.text}
              referenceQuote={referenceQuote}
            />
          </section>

          <div className="citation-drawer-doc-page-row">
            <div className="citation-drawer-doc-page-column">
              <section className="drawer-section citation-drawer-info-card">
                <span className="tiny-label">Document</span>
                <p className="citation-drawer-info-value">{citation.document_title}</p>
              </section>

              <section className="drawer-section citation-drawer-info-card">
                <span className="tiny-label">Page</span>
                <p className="citation-drawer-info-value">
                  {citation.page_end != null && citation.page_end !== citation.page_number
                    ? `${citation.page_number}-${citation.page_end}`
                    : `${citation.page_number}`}
                </p>
              </section>
            </div>

            <section className="drawer-section citation-drawer-info-card citation-drawer-info-card--match-quality">
              <span className="tiny-label">Match quality</span>
              <div
                className={`retrieval-strength retrieval-strength--${matchQuality.level === "partial" ? "fair" : matchQuality.level}`}
              >
                <div className="retrieval-strength-headline-row">
                  <span className="retrieval-strength-icon">
                    <MatchQualityIcon level={matchQuality.level} />
                  </span>
                  <p className="retrieval-strength-headline">{matchQuality.headline}</p>
                </div>
                <p className="retrieval-strength-detail">{matchQuality.detail}</p>
              </div>
            </section>
          </div>

          <section className="drawer-section">
            <div className="drawer-action-row">
              <button
                data-testid="show-more-citations"
                className="ghost-button"
                disabled={readOnly || loadingMore}
                onClick={handleLoadMore}
              >
                {loadingMore ? "Searching nearby passages..." : "Show me more"}
              </button>
              {citation.replacement_pending ? (
                <button
                  data-testid="undo-citation-replacement"
                  className="ghost-button"
                  disabled={readOnly}
                  onClick={handleUndoReplacement}
                >
                  Undo replacement
                </button>
              ) : null}
            </div>
          </section>

          <section className="drawer-section">
            <span className="tiny-label">Mark as mismatch</span>
            <textarea
              data-testid="mismatch-note"
              placeholder="Optional note for why this citation does not support the sentence."
              value={mismatchNote}
              onChange={(event) => setMismatchNote(event.target.value)}
              disabled={readOnly}
            />
            <div className="drawer-action-row">
              <button
                data-testid="save-mismatch"
                className="primary-button subtle"
                disabled={readOnly}
                onClick={handleMismatch}
              >
                Save mismatch
              </button>
            </div>
          </section>

          {scopedResults.length ? (
            <section className="drawer-section">
              <span className="tiny-label">Alternative passages</span>
              <div className="candidate-list">
                {scopedResults.map((candidate) => (
                  <article className="candidate-item" key={candidate.chunk_id}>
                    <p>{candidate.text}</p>
                    <button
                      data-testid={`replace-citation-${candidate.chunk_id}`}
                      className="ghost-button"
                      disabled={readOnly}
                      onClick={() => handleReplace(candidate.chunk_id)}
                    >
                      Replace with this passage
                    </button>
                  </article>
                ))}
              </div>
            </section>
          ) : null}

        </div>
      </aside>
    </div>
  );
}
