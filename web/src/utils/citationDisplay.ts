import type { SessionState } from "../types";

/** Internal `citation_id` → 1-based display index by order of first reference in `parsed_sentences`. */
export function buildDisplayCitationMap(session: SessionState): Map<number, number> {
  const displayByCitationId = new Map<number, number>();
  let nextDisplayId = 1;

  for (const sentence of session.parsed_sentences) {
    for (const reference of sentence.references) {
      const citationId = reference.citation_id;
      if (!citationId) continue;
      if (displayByCitationId.has(citationId)) continue;
      displayByCitationId.set(citationId, nextDisplayId);
      nextDisplayId += 1;
    }
  }

  return displayByCitationId;
}
