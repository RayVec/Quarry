import { useRef, useState } from "react";
import {
  citationAlternativesCacheKey,
  clearCitationAlternativesCache,
  resolveActiveCitationContext,
  type ActiveCitationContext,
  type CitationAlternativesCacheEntry,
} from "@/features/thread/model";
import type { CitationIndexEntry, ParsedSentence, SessionState } from "@/types";

export function useCitationDialogState() {
  const [activeCitation, setActiveCitation] =
    useState<ActiveCitationContext | null>(null);
  const citationAlternativesCache = useRef(
    new Map<string, CitationAlternativesCacheEntry>(),
  );

  function openCitation(
    entryId: string,
    session: SessionState,
    sentence: ParsedSentence,
    citationId: number,
    referenceQuote: string,
    readOnly: boolean,
  ) {
    const citation = session.citation_index.find(
      (item) => item.citation_id === citationId,
    );
    if (!citation) {
      return;
    }
    setActiveCitation({
      entryId,
      session,
      sentenceIndex: sentence.sentence_index,
      referenceQuote,
      citation,
      readOnly,
    });
  }

  function closeCitation() {
    setActiveCitation(null);
  }

  function clearCitationCache() {
    citationAlternativesCache.current.clear();
  }

  function reset() {
    clearCitationCache();
    closeCitation();
  }

  function syncInteractiveSession(nextSession: SessionState) {
    setActiveCitation((current) => {
      if (!current || current.session.session_id !== nextSession.session_id) {
        return current;
      }
      return resolveActiveCitationContext(current, nextSession);
    });
  }

  function syncEntrySession(entryId: string, nextSession: SessionState) {
    setActiveCitation((current) => {
      if (!current || current.entryId !== entryId) {
        return current;
      }
      return resolveActiveCitationContext(current, nextSession);
    });
  }

  function getActiveCitationCacheEntry() {
    if (!activeCitation) {
      return null;
    }
    return (
      citationAlternativesCache.current.get(
        citationAlternativesCacheKey(
          activeCitation.session.session_id,
          activeCitation.citation.citation_id,
          activeCitation.citation.chunk_id,
        ),
      ) ?? null
    );
  }

  function storeAlternativesForActive(alternatives: CitationIndexEntry[]) {
    if (!activeCitation) {
      return;
    }
    clearCitationAlternativesCache(
      citationAlternativesCache.current,
      activeCitation.session.session_id,
      activeCitation.citation.citation_id,
    );
    citationAlternativesCache.current.set(
      citationAlternativesCacheKey(
        activeCitation.session.session_id,
        activeCitation.citation.citation_id,
        activeCitation.citation.chunk_id,
      ),
      {
        hasLoaded: true,
        alternatives,
      },
    );
  }

  return {
    activeCitation,
    openCitation,
    closeCitation,
    clearCitationCache,
    reset,
    syncInteractiveSession,
    syncEntrySession,
    getActiveCitationCacheEntry,
    storeAlternativesForActive,
  };
}
