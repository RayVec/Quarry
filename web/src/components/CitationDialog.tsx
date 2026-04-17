import {
  startTransition,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  Check,
  LoaderCircle,
  Minus,
  ThumbsDown,
  ThumbsUp,
  X,
} from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { api } from "../api";
import type { CitationIndexEntry, Reference, SessionState } from "../types";
import { buildDisplayCitationMap } from "../utils/citationDisplay";
import {
  describeUnifiedMatchQuality,
  hasExactQuoteMatch,
  referenceQuoteCoverage,
  type UnifiedMatchLevel,
} from "../utils/retrievalDisplay";

const MATCH_QUALITY_ICON_SIZE = 18;
const MATCH_QUALITY_ICON_STROKE = 2.3;


function MatchQualityIcon({ level }: { level: UnifiedMatchLevel }) {
  const common = {
    size: MATCH_QUALITY_ICON_SIZE,
    strokeWidth: MATCH_QUALITY_ICON_STROKE,
    "aria-hidden": true as const,
  };
  switch (level) {
    case "strong":
    case "good":
      return <Check {...common} />;
    case "fair":
      return <Minus {...common} />;
    case "weak":
      return <X {...common} />;
    default:
      return null;
  }
}

interface CitationDialogProps {
  citation: CitationIndexEntry;
  sentenceIndex: number;
  referenceQuote: string;
  readOnly: boolean;
  session: SessionState;
  initialAlternatives: CitationIndexEntry[];
  initialAlternativesLoaded: boolean;
  onAlternativesLoaded: (alternatives: CitationIndexEntry[]) => void;
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
      const parent = el.parentElement;
      if (!parent) return;

      const wasExpanded = parent.classList.contains("quoted-passage-viewport--expanded");
      if (wasExpanded) parent.classList.remove("quoted-passage-viewport--expanded");

      setNeedsToggle(el.scrollHeight > parent.clientHeight + 1);

      if (wasExpanded) parent.classList.add("quoted-passage-viewport--expanded");
    };

    measure();
    const resizeObserver = new ResizeObserver(measure);
    resizeObserver.observe(el);

    const fonts = document.fonts;
    let cancelled = false;
    if (fonts?.ready) {
      void fonts.ready.then(() => {
        if (!cancelled) {
          measure();
        }
      });
    }

    return () => {
      cancelled = true;
      resizeObserver.disconnect();
    };
  }, [text, referenceQuote]);

  useEffect(() => {
    setExpanded(false);
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
          aria-expanded={expanded}
          className="quoted-passage-toggle"
          data-testid="citation-quote-read-more"
          onClick={() => setExpanded((value) => !value)}
          type="button"
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
  initialAlternatives,
  initialAlternativesLoaded,
  onAlternativesLoaded,
  onClose,
  onSessionUpdate,
}: CitationDialogProps) {
  const drawerStackRef = useRef<HTMLDivElement>(null);
  const [alternatives, setAlternatives] = useState<CitationIndexEntry[]>(
    initialAlternatives,
  );
  const [hasLoadedAlternatives, setHasLoadedAlternatives] = useState(
    initialAlternativesLoaded,
  );
  const [loadingAlternatives, setLoadingAlternatives] = useState(false);
  const [replacingAlternativeId, setReplacingAlternativeId] = useState<number | null>(null);
  const [expandedAlternatives, setExpandedAlternatives] = useState<Set<number>>(new Set());

  const currentFeedback = useMemo(() => {
    const feedback = session.feedback.citation_feedback?.find(
      (fb) =>
        fb.sentence_index === sentenceIndex && fb.citation_id === citation.citation_id,
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
  const isReplacementPending =
    citation.replacement_pending || sourceReference?.replacement_pending === true;
  const matchQuality = useMemo(() => {
    return describeUnifiedMatchQuality(
      citation.retrieval_score,
      citation.ambiguity_review_required,
      citation.ambiguity_gap,
      {
        sentenceStatus: sourceSentence?.status,
        referenceVerified: sourceReference?.verified,
        referenceQuoteCoverage: referenceQuoteCoverage(sourceSentence?.sentence_text, referenceQuote),
        referenceQuoteExactMatch: hasExactQuoteMatch(citation.text, referenceQuote),
        referenceConfidenceLabel: sourceReference?.confidence_label,
        referenceConfidenceUnknown: sourceReference?.confidence_unknown,
      },
    );
  }, [
    citation.ambiguity_gap,
    citation.ambiguity_review_required,
    citation.retrieval_score,
    citation.text,
    referenceQuote,
    sourceReference?.confidence_label,
    sourceReference?.confidence_unknown,
    sourceReference?.verified,
    sourceSentence?.status,
    sourceSentence?.sentence_text,
  ]);
  const matchQualityDisplay = isReplacementPending
    ? {
        headline: "Pending review",
        detail:
          "This citation was replaced manually. Match quality will update after the next verification or refinement pass.",
        level: "pending" as const,
      }
    : matchQuality;

  const showAlternativesSection = currentFeedback === "dislike";

  useEffect(() => {
    setAlternatives(initialAlternatives);
    setHasLoadedAlternatives(initialAlternativesLoaded);
    setExpandedAlternatives(new Set());
  }, [
    citation.citation_id,
    citation.chunk_id,
    initialAlternatives,
    initialAlternativesLoaded,
    session.session_id,
  ]);

  useEffect(() => {
    if (!showAlternativesSection) {
      return;
    }
    const stack = drawerStackRef.current;
    if (!stack) {
      return;
    }
    stack.scrollTop = stack.scrollHeight;
  }, [currentFeedback, showAlternativesSection]);

  useEffect(() => {
    if (
      !(
        showAlternativesSection &&
        (alternatives.length > 0 || hasLoadedAlternatives)
      )
    ) {
      return;
    }
    const stack = drawerStackRef.current;
    if (!stack) {
      return;
    }
    stack.scrollTop = stack.scrollHeight;
  }, [
    alternatives.length,
    hasLoadedAlternatives,
    showAlternativesSection,
  ]);

  async function handleFeedback(feedbackType: "like" | "dislike") {
    const newFeedbackType = currentFeedback === feedbackType ? "neutral" : feedbackType;
    const response = await api.setCitationFeedback(
      session.session_id,
      sentenceIndex,
      citation.citation_id,
      newFeedbackType,
    );
    startTransition(() => {
      onSessionUpdate(response.session);
    });
  }

  async function handleLoadAlternatives() {
    setLoadingAlternatives(true);
    try {
      const altResponse = await api.getCitationAlternatives(
        session.session_id,
        citation.citation_id,
      );
      const nextAlternatives = altResponse.citations.slice(0, 3);
      setAlternatives(nextAlternatives);
      setHasLoadedAlternatives(true);
      onAlternativesLoaded(nextAlternatives);
    } catch (error) {
      console.error("Failed to load alternatives:", error);
    } finally {
      setLoadingAlternatives(false);
    }
  }

  async function handleReplaceWithAlternative(replacementCitationId: number) {
    setReplacingAlternativeId(replacementCitationId);
    try {
      const response = await api.replaceWithAlternative(
        session.session_id,
        sentenceIndex,
        citation.citation_id,
        replacementCitationId,
      );
      startTransition(() => {
        setAlternatives([]);
        setHasLoadedAlternatives(false);
        setExpandedAlternatives(new Set());
        onSessionUpdate(response.session);
      });
    } catch (error) {
      console.error("Failed to replace citation with alternative:", error);
    } finally {
      setReplacingAlternativeId(null);
    }
  }

  function toggleExpandAlternative(citationId: number) {
    setExpandedAlternatives((prev) => {
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
    <Sheet open onOpenChange={(nextOpen) => !nextOpen && onClose()}>
      <SheetContent
        className="citation-drawer citation-drawer--sized"
        data-testid="citation-dialog"
        side="right"
      >
        <SheetHeader className="citation-drawer-header">
          <span className="eyebrow">Citation {displayCitationId}</span>
          <SheetTitle className="citation-drawer-title">
            Review citation support
          </SheetTitle>
        </SheetHeader>

        <div
          ref={drawerStackRef}
          className="drawer-stack citation-drawer-stack"
        >
          <Card className="citation-drawer-card citation-drawer-card--surface citation-drawer-card--overflow-visible" size="sm">
            <CardHeader>
              <CardTitle>Reference quote in context</CardTitle>
            </CardHeader>
            <CardContent>
              <CitationQuotePanel
                key={citation.citation_id}
                referenceQuote={referenceQuote}
                text={citation.text}
              />
            </CardContent>
          </Card>

          <div className="citation-drawer-meta-layout">
            <Card className="citation-drawer-card citation-drawer-card--surface citation-drawer-info-card citation-drawer-info-card--document" size="sm">
              <CardHeader>
                <CardTitle>Document</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="citation-drawer-info-value">{citation.document_title}</p>
              </CardContent>
            </Card>

            <Card className="citation-drawer-card citation-drawer-card--surface citation-drawer-info-card citation-drawer-info-card--page" size="sm">
              <CardHeader>
                <CardTitle>Page</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="citation-drawer-info-value">
                  {citation.page_end != null && citation.page_end !== citation.page_number
                    ? `${citation.page_number}-${citation.page_end}`
                    : `${citation.page_number}`}
                </p>
              </CardContent>
            </Card>

            <Card className="citation-drawer-card citation-drawer-card--surface citation-drawer-info-card citation-drawer-info-card--match-quality citation-drawer-match-card" size="sm">
              <CardHeader>
                <CardTitle>Match quality</CardTitle>
              </CardHeader>
              <CardContent className="citation-drawer-match-card-content">
                <div
                  className={`retrieval-strength retrieval-strength--${matchQualityDisplay.level}`}
                >
                  <div className="retrieval-strength-headline-row">
                    <span className="retrieval-strength-icon">
                      {isReplacementPending ? (
                        <LoaderCircle
                          aria-hidden
                          size={MATCH_QUALITY_ICON_SIZE}
                          strokeWidth={MATCH_QUALITY_ICON_STROKE}
                        />
                      ) : (
                        <MatchQualityIcon level={matchQuality.level} />
                      )}
                    </span>
                    <p className="retrieval-strength-headline">{matchQualityDisplay.headline}</p>
                  </div>
                  <p className="retrieval-strength-detail">{matchQualityDisplay.detail}</p>
                </div>
              </CardContent>
            </Card>
            <Card className="citation-drawer-card citation-drawer-card--surface citation-drawer-feedback-card" size="sm">
              <CardHeader>
                <CardTitle>Citation feedback</CardTitle>
              </CardHeader>
              <CardContent className="citation-feedback-actions">
                <Button
                  data-testid="like-citation"
                  disabled={readOnly || loadingAlternatives || replacingAlternativeId !== null}
                  onClick={() => handleFeedback("like")}
                  type="button"
                  variant={currentFeedback === "like" ? "default" : "outline"}
                >
                  <ThumbsUp data-icon="inline-start" />
                  {currentFeedback === "like" ? "Liked" : "Like"}
                </Button>
                <Button
                  data-testid="dislike-citation"
                  disabled={readOnly || loadingAlternatives || replacingAlternativeId !== null}
                  onClick={() => handleFeedback("dislike")}
                  type="button"
                  variant={currentFeedback === "dislike" ? "default" : "outline"}
                >
                  <ThumbsDown data-icon="inline-start" />
                  {currentFeedback === "dislike" ? "Disliked" : "Dislike"}
                </Button>
              </CardContent>
            </Card>
          </div>

          {showAlternativesSection ? (
            <Card
              className="citation-drawer-card citation-drawer-card--surface citation-drawer-alternatives-card citation-alternatives-card"
              size="sm"
            >
              <CardHeader>
                <CardTitle>
                  You can select similar passages to replace
                </CardTitle>
              </CardHeader>
              <CardContent>
                {!hasLoadedAlternatives ? (
                  <Button
                    className="citation-alternatives-load-button"
                    data-testid="load-alternatives"
                    disabled={readOnly || loadingAlternatives || replacingAlternativeId !== null}
                    onClick={handleLoadAlternatives}
                    type="button"
                  >
                    {loadingAlternatives ? "Searching..." : "Show me similar passages"}
                  </Button>
                ) : (
                  <div className="citation-candidate-list">
                    {alternatives.map((alt) => {
                      const isExpanded = expandedAlternatives.has(alt.citation_id);
                      const isReplacing = replacingAlternativeId === alt.citation_id;
                      return (
                        <article className="citation-candidate-item" key={alt.citation_id}>
                          <div className="citation-candidate-content">
                            <p className="citation-candidate-meta">
                              <strong>{alt.document_title}</strong>
                              {alt.section_heading ? <span> • {alt.section_heading}</span> : null}
                              <span> (Page {alt.page_number})</span>
                            </p>
                            <p
                              className={`citation-candidate-passage ${isExpanded ? "citation-candidate-passage--expanded" : ""}`}
                            >
                              {alt.text}
                            </p>
                            <div className="citation-candidate-actions">
                              <button
                                className="citation-alt-text-button"
                                data-testid={`expand-alternative-${alt.citation_id}`}
                                onClick={() => toggleExpandAlternative(alt.citation_id)}
                                type="button"
                              >
                                {isExpanded ? "Read less" : "Read more"}
                              </button>
                              <Button
                                className="citation-alt-replace-button"
                                data-testid={`replace-with-alternative-${alt.citation_id}`}
                                disabled={readOnly || replacingAlternativeId !== null}
                                onClick={() => handleReplaceWithAlternative(alt.citation_id)}
                                type="button"
                                variant="default"
                              >
                                {isReplacing ? (
                                  <>
                                    <LoaderCircle className="spin-icon" data-icon="inline-start" />
                                    Replacing...
                                  </>
                                ) : (
                                  "Replace"
                                )}
                              </Button>
                            </div>
                          </div>
                        </article>
                      );
                    })}
                  </div>
                )}
              </CardContent>
            </Card>
          ) : null}

          {!sourceReference ? (
            <Alert>
              <AlertTitle>Reference metadata unavailable</AlertTitle>
              <AlertDescription>
                The supporting sentence is available, but the exact reference metadata could not be resolved from this session snapshot.
              </AlertDescription>
            </Alert>
          ) : null}
        </div>
      </SheetContent>
    </Sheet>
  );
}
