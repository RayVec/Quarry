import {
  startTransition,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Check, Minus, ThumbsUp, ThumbsDown, X } from "lucide-react";
import { api } from "../api";
import type { CitationIndexEntry, Reference, SessionState } from "../types";
import { buildDisplayCitationMap } from "../utils/citationDisplay";
import { lockBodyScroll, unlockBodyScroll } from "../utils/bodyScrollLock";
import type { MatchQuality } from "../types";

const MATCH_QUALITY_ICON_SIZE = 18;
const MATCH_QUALITY_ICON_STROKE = 2.3;

function MatchQualityIcon({ level }: { level: MatchQuality }) {
  const common = {
    size: MATCH_QUALITY_ICON_SIZE,
    strokeWidth: MATCH_QUALITY_ICON_STROKE,
    "aria-hidden": true as const,
  };
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
          <p className="quoted-passage">
            {highlightReference(text, referenceQuote)}
          </p>
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
  const drawerStackRef = useRef<HTMLDivElement>(null);
  const [showAlternativesSection, setShowAlternativesSection] = useState(false);
  const [alternatives, setAlternatives] = useState<CitationIndexEntry[]>([]);
  const [loadingAlternatives, setLoadingAlternatives] = useState(false);
  const [expandedAlternatives, setExpandedAlternatives] = useState<Set<number>>(new Set());
  
  const currentFeedback = useMemo(() => {
    const feedback = session.feedback.citation_feedback?.find(
      (fb) =>
        fb.sentence_index === sentenceIndex && fb.citation_id === citation.citation_id
    );
    return feedback?.feedback_type ?? "neutral";
  }, [session.feedback.citation_feedback, sentenceIndex, citation.citation_id]);
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
        detail:
          "This passage is grounded, but you may want to inspect the source context.",
      };
    }
    return {
      level,
      headline: "No citation expected",
      detail:
        "This sentence is structural or inferential, so no citation badge is shown in the response.",
    };
  }, [sourceSentence?.match_quality]);

  useEffect(() => {
    lockBodyScroll();
    return () => unlockBodyScroll();
  }, []);

  useEffect(() => {
    if (!(currentFeedback === "dislike" && showAlternativesSection)) {
      return;
    }
    const stack = drawerStackRef.current;
    if (!stack) {
      return;
    }
    stack.scrollTop = stack.scrollHeight;
  }, [currentFeedback, showAlternativesSection]);

  async function handleFeedback(feedbackType: "like" | "dislike") {
    const newFeedbackType = currentFeedback === feedbackType ? "neutral" : feedbackType;
    const response = await api.setCitationFeedback(
      session.session_id,
      sentenceIndex,
      citation.citation_id,
      newFeedbackType
    );
    startTransition(() => {
      onSessionUpdate(response.session);
    });
    
    if (newFeedbackType === "dislike") {
      setShowAlternativesSection(true);
      setAlternatives([]);
    } else {
      setShowAlternativesSection(false);
      setAlternatives([]);
    }
  }

  async function handleLoadAlternatives() {
    setLoadingAlternatives(true);
    try {
      const altResponse = await api.getCitationAlternatives(
        session.session_id,
        citation.citation_id
      );
      setAlternatives(altResponse.citations.slice(0, 3));
    } catch (error) {
      console.error("Failed to load alternatives:", error);
    } finally {
      setLoadingAlternatives(false);
    }
  }

  async function handleReplaceWithAlternative(replacementCitationId: number) {
    const response = await api.replaceWithAlternative(
      session.session_id,
      sentenceIndex,
      citation.citation_id,
      replacementCitationId
    );
    startTransition(() => {
      onSessionUpdate(response.session);
      setShowAlternativesSection(false);
      setAlternatives([]);
      onClose();
    });
  }

  function toggleExpandAlternative(citationId: number) {
    setExpandedAlternatives(prev => {
      const next = new Set(prev);
      if (next.has(citationId)) {
        next.delete(citationId);
      } else {
        next.add(citationId);
      }
      return next;
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

        <div ref={drawerStackRef} className="drawer-stack">
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
                <p className="citation-drawer-info-value">
                  {citation.document_title}
                </p>
              </section>

              <section className="drawer-section citation-drawer-info-card">
                <span className="tiny-label">Page</span>
                <p className="citation-drawer-info-value">
                  {citation.page_end != null &&
                  citation.page_end !== citation.page_number
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
                  <p className="retrieval-strength-headline">
                    {matchQuality.headline}
                  </p>
                </div>
                <p className="retrieval-strength-detail">
                  {matchQuality.detail}
                </p>
              </div>
            </section>
          </div>

          <section className="drawer-section">
            <span className="tiny-label">Citation feedback</span>
            <div className="drawer-action-row" style={{ gap: "8px" }}>
              <button
                data-testid="like-citation"
                className={currentFeedback === "like" ? "primary-button" : "ghost-button"}
                disabled={readOnly || loadingAlternatives}
                onClick={() => handleFeedback("like")}
                style={{ display: "flex", alignItems: "center", gap: "6px" }}
              >
                <ThumbsUp size={16} aria-hidden />
                {currentFeedback === "like" ? "Liked" : "Like"}
              </button>
              <button
                data-testid="dislike-citation"
                className={currentFeedback === "dislike" ? "primary-button" : "ghost-button"}
                disabled={readOnly || loadingAlternatives}
                onClick={() => handleFeedback("dislike")}
                style={{ display: "flex", alignItems: "center", gap: "6px" }}
              >
                <ThumbsDown size={16} aria-hidden />
                {currentFeedback === "dislike" ? "Disliked" : "Dislike"}
              </button>
            </div>
          </section>

          {currentFeedback === "dislike" && showAlternativesSection && (
            <section className="drawer-section" style={{ backgroundColor: "#f9fafb", padding: "16px", borderRadius: "8px" }}>
              <span className="tiny-label" style={{ marginBottom: "8px", display: "block" }}>
                You can select similar passages to replace
              </span>
              
              {alternatives.length === 0 ? (
                <button
                  data-testid="load-alternatives"
                  className="primary-button"
                  disabled={readOnly || loadingAlternatives}
                  onClick={handleLoadAlternatives}
                  style={{ width: "100%" }}
                >
                  {loadingAlternatives ? "Searching..." : "Show me similar passages"}
                </button>
              ) : (
                <div className="candidate-list">
                  {alternatives.map((alt) => {
                    const isExpanded = expandedAlternatives.has(alt.citation_id);
                    return (
                      <article 
                        className="candidate-item" 
                        key={alt.citation_id}
                        style={{ marginBottom: "16px" }}
                      >
                        <div style={{ marginBottom: "8px", fontSize: "0.875rem", color: "#666" }}>
                          <strong>{alt.document_title}</strong>
                          {alt.section_heading && <span> • {alt.section_heading}</span>}
                          <span> (Page {alt.page_number})</span>
                        </div>
                        <p 
                          style={{ 
                            display: "-webkit-box",
                            WebkitLineClamp: isExpanded ? "unset" : 6,
                            WebkitBoxOrient: "vertical",
                            overflow: "hidden",
                            marginBottom: "12px",
                            lineHeight: "1.5"
                          }}
                        >
                          {alt.text}
                        </p>
                        <div style={{ display: "flex", gap: "8px" }}>
                          <button
                            data-testid={`expand-alternative-${alt.citation_id}`}
                            className="ghost-button citation-alt-action-button"
                            onClick={() => toggleExpandAlternative(alt.citation_id)}
                          >
                            {isExpanded ? "Show less" : "Show more"}
                          </button>
                          <button
                            data-testid={`replace-with-alternative-${alt.citation_id}`}
                            className="primary-button subtle citation-alt-action-button"
                            disabled={readOnly}
                            onClick={() => handleReplaceWithAlternative(alt.citation_id)}
                          >
                            Replace with this passage
                          </button>
                        </div>
                      </article>
                    );
                  })}
                </div>
              )}
            </section>
          )}
        </div>
      </aside>
    </div>
  );
}
