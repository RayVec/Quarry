import { startTransition, useMemo, useState } from "react";
import { api } from "../api";
import type { CitationIndexEntry, SessionState } from "../types";
import { buildDisplayCitationMap } from "../utils/citationDisplay";

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
  const retrievalModes = useMemo(
    () =>
      Object.entries(citation.retrieval_scores)
        .map(([name, score]) => `${name}:${score.toFixed(3)}`)
        .join(" · "),
    [citation.retrieval_scores],
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
            <h2>{citation.section_heading}</h2>
            <p>
              {citation.document_title} · p. {citation.page_number}
            </p>
          </div>
          <button className="icon-button" data-testid="close-citation-dialog" onClick={onClose}>
            Close
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
            <span className="tiny-label">Retrieval info</span>
            <p>
              score {citation.retrieval_score.toFixed(3)} · facet {citation.source_facet}
            </p>
            {retrievalModes ? <p>{retrievalModes}</p> : null}
            {citation.ambiguity_gap != null ? <p>ambiguity gap {citation.ambiguity_gap.toFixed(3)}</p> : null}
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
