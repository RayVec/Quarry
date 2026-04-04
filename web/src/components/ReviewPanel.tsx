import { startTransition, useState } from "react";
import type { SessionState } from "../types";

interface ReviewPanelProps {
  session: SessionState;
  interactive: boolean;
  open: boolean;
  selectedFacetGaps: string[];
  onToggle: () => void;
  onFacetToggle: (facet: string) => void;
  onSupplement: (selectedFacets: string[]) => Promise<void>;
  onRefine: (selectedFacets: string[]) => Promise<void>;
}

function facetChecklistVisible(session: SessionState) {
  return session.query_type === "multi_hop" && session.facets.length > 1;
}

export function ReviewPanel({
  session,
  interactive,
  open,
  selectedFacetGaps,
  onToggle,
  onFacetToggle,
  onSupplement,
  onRefine,
}: ReviewPanelProps) {
  const [busy, setBusy] = useState<"supplement" | "refine" | null>(null);
  const mismatchCount = session.feedback.citation_mismatches.length;
  const disagreementCount = session.feedback.claim_disagreements.length;
  const replacementCount = session.feedback.citation_replacements.length;
  const persistedFacetCount = session.feedback.facet_gaps.length;
  const effectiveFacetGapCount = new Set([...session.feedback.facet_gaps, ...selectedFacetGaps]).size;
  const removedSentenceCount = session.removed_ungrounded_claim_count;
  const anyFeedback =
    mismatchCount > 0 ||
    disagreementCount > 0 ||
    replacementCount > 0 ||
    effectiveFacetGapCount > 0;

  return (
    <section className={`review-panel-shell ${open ? "open" : ""}`} data-testid="review-panel">
      <button className="review-panel-toggle" data-testid="toggle-review-panel" onClick={onToggle}>
        <span>Review and refine</span>
        <span>{open ? "Hide" : "Open"}</span>
      </button>

      {open ? (
        <div className="review-panel-body">
          {facetChecklistVisible(session) ? (
            <div className="review-panel-section">
              <span className="tiny-label">Facet completeness</span>
              <div className="facet-checklist">
                {session.facets.map((facet) => {
                  const checked = selectedFacetGaps.includes(facet);
                  return (
                    <label className="facet-option" key={facet}>
                      <input
                        data-testid={`facet-toggle-${facet.replace(/[^a-z0-9]+/gi, "-").toLowerCase()}`}
                        disabled={!interactive || busy !== null}
                        type="checkbox"
                        checked={checked}
                        onChange={() => onFacetToggle(facet)}
                      />
                      <span>{facet}</span>
                    </label>
                  );
                })}
              </div>
            </div>
          ) : null}

          <div className="review-panel-section">
            <span className="tiny-label">Feedback summary</span>
            <p data-testid="feedback-summary">
              {mismatchCount} citations flagged as mismatch, {disagreementCount} claim disagreements, {effectiveFacetGapCount} facet gaps.
            </p>
            {replacementCount ? <p>{replacementCount} citation replacements are still pending regeneration.</p> : null}
            {removedSentenceCount ? (
              <p className="review-summary-note">
                {removedSentenceCount} unverifiable {removedSentenceCount === 1 ? "sentence was" : "sentences were"} removed from the visible response. Use the saved citations and your review notes to investigate what QUARRY could not ground.
              </p>
            ) : null}
            {!persistedFacetCount && !selectedFacetGaps.length && !mismatchCount && !disagreementCount && !replacementCount ? (
              <p>No feedback has been captured yet.</p>
            ) : null}
          </div>

          <div className="review-panel-actions">
            <button
              className="ghost-button"
              data-testid="supplement-selected-facets"
              disabled={!interactive || !selectedFacetGaps.length || busy !== null}
              onClick={async () => {
                setBusy("supplement");
                try {
                  await onSupplement(selectedFacetGaps);
                } finally {
                  startTransition(() => setBusy(null));
                }
              }}
            >
              {busy === "supplement" ? "Adding coverage..." : "Supplement"}
            </button>
            <button
              className="primary-button subtle"
              data-testid="run-refinement"
              disabled={!interactive || !anyFeedback || busy !== null}
              onClick={async () => {
                setBusy("refine");
                try {
                  await onRefine(selectedFacetGaps);
                } finally {
                  startTransition(() => setBusy(null));
                }
              }}
            >
              {busy === "refine" ? "Refining..." : "Refine"}
            </button>
          </div>
        </div>
      ) : null}
    </section>
  );
}
