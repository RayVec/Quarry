import type { SessionState } from "../types";

/** Internal `citation_id` → 1-based display index by ascending referenced `citation_id`. */
export function buildDisplayCitationMap(session: SessionState): Map<number, number> {
  const referencedCitationIds = new Set<number>();
  for (const sentence of session.parsed_sentences) {
    for (const reference of sentence.references) {
      const citationId = reference.citation_id;
      if (!citationId) continue;
      referencedCitationIds.add(citationId);
    }
  }

  const displayByCitationId = new Map<number, number>();
  [...referencedCitationIds]
    .sort((left, right) => left - right)
    .forEach((citationId, index) => {
      displayByCitationId.set(citationId, index + 1);
    });

  return displayByCitationId;
}
