import { startTransition, useMemo, useState } from "react";
import { X } from "lucide-react";
import { api } from "../api";
import type { CitationIndexEntry, SessionState } from "../types";
import { buildDisplayCitationMap } from "../utils/citationDisplay";
import { describeUnifiedMatchQuality } from "../utils/retrievalDisplay";

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
  const displayCitationMap = useMemo(() => buildDisplayCitationMap(session), [session]);
  const displayCitationId = displayCitationMap.get(citation.citation_id) ?? citation.citation_id;
  const matchQuality = useMemo(
    () =>
      describeUnifiedMatchQuality(
        citation.retrieval_score,
        citation.ambiguity_review_required,
        citation.ambiguity_gap,
      ),
    [citation.retrieval_score, citation.ambiguity_review_required, citation.ambiguity_gap],
  );

  async function handleLoadMore() {
    setLoadingMore(true);
    try {
      const response = await api.scopedRetrieval(session.session_id, sentenceIndex, citation.citation_id);
      startTransition(() => {
        setScopedResults(response.citations);
      });
    } finally {
      setLoadingMore(false);
    }
  }

  async function handleReplace(replacementChunkId: string) {
    const response = await api.replaceCitation(session.session_id, sentenceIndex, citation.citation_id, replacementChunkId);
    startTransition(() => {
      onSessionUpdate(response.session);
      onClose();
    });
  }

  async function handleMismatch() {
    const response = await api.addMismatch(session.session_id, citation.citation_id, mismatchNote || undefined);
    startTransition(() => {
      onSessionUpdate(response.session);
      setMismatchNote("");
    });
  }

  async function handleUndoReplacement() {
    const response = await api.undoReplacement(session.session_id, citation.citation_id);
    startTransition(() => {
      onSessionUpdate(response.session);
      onClose();
    });
  }

  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <aside className="citation-drawer" data-testid="citation-dialog" onClick={(event) => event.stopPropagation()}>
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
            <p className="quoted-passage">{highlightReference(citation.text, referenceQuote)}</p>
          </section>

          <section className="drawer-section">
            <span className="tiny-label">Source metadata</span>
            <p>{citation.document_title}</p>
            <p>{citation.section_path}</p>
            <p>
              page {citation.page_number}
              {citation.page_end ? `-${citation.page_end}` : ""}
            </p>
          </section>

          <section className="drawer-section">
            <span className="tiny-label">Match quality</span>
            <div className="retrieval-human">
              <div className={`retrieval-strength retrieval-strength--${matchQuality.level}`}>
                <p className="retrieval-strength-headline">{matchQuality.headline}</p>
                <p className="retrieval-strength-detail">{matchQuality.detail}</p>
              </div>
            </div>
          </section>

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
              disabled={readOnly}
              value={mismatchNote}
              onChange={(event) => setMismatchNote(event.target.value)}
              placeholder="Optional note for why this passage does not support the sentence."
            />
            <button data-testid="save-mismatch" className="ghost-button" disabled={readOnly} onClick={handleMismatch}>
              Save mismatch
            </button>
          </section>

          {scopedResults.length ? (
            <section className="drawer-section">
              <span className="tiny-label">Nearby candidate passages</span>
              <div className="candidate-list">
                {scopedResults.map((result) => (
                  <button
                    className="candidate-item"
                    data-testid={`replace-citation-${result.chunk_id}`}
                    disabled={readOnly}
                    key={result.chunk_id}
                    onClick={() => handleReplace(result.chunk_id)}
                  >
                    <span className="candidate-meta">
                      {result.document_title} · {result.section_heading} · p. {result.page_number}
                    </span>
                    <span>{result.text}</span>
                  </button>
                ))}
              </div>
            </section>
          ) : null}
        </div>
      </aside>
    </div>
  );
}
