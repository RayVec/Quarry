import type { CitationIndexEntry, SessionEnvelope } from "./types";

const API_ROOT = "http://127.0.0.1:8000/api/v1";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return response.json() as Promise<T>;
}

export const api = {
  startQuery(query: string) {
    return request<SessionEnvelope>("/query/start", {
      method: "POST",
      body: JSON.stringify({ query }),
    });
  },
  runQuery(query: string) {
    return request<SessionEnvelope>("/query", {
      method: "POST",
      body: JSON.stringify({ query }),
    });
  },
  getSession(sessionId: string) {
    return request<SessionEnvelope>(`/sessions/${sessionId}`);
  },
  addMismatch(sessionId: string, citationId: number, reviewerNote?: string) {
    return request<SessionEnvelope>(`/sessions/${sessionId}/feedback/mismatch`, {
      method: "POST",
      body: JSON.stringify({ citation_id: citationId, reviewer_note: reviewerNote }),
    });
  },
  addDisagreement(sessionId: string, sentenceIndex: number, reviewerNote?: string) {
    return request<SessionEnvelope>(`/sessions/${sessionId}/feedback/disagreement`, {
      method: "POST",
      body: JSON.stringify({ sentence_index: sentenceIndex, reviewer_note: reviewerNote }),
    });
  },
  addFacetGaps(sessionId: string, facets: string[]) {
    return request<SessionEnvelope>(`/sessions/${sessionId}/feedback/facet-gaps`, {
      method: "POST",
      body: JSON.stringify({ facets }),
    });
  },
  supplement(sessionId: string, facets: string[]) {
    return request<SessionEnvelope>(`/sessions/${sessionId}/supplement`, {
      method: "POST",
      body: JSON.stringify({ facets }),
    });
  },
  refine(sessionId: string) {
    return request<SessionEnvelope>(`/sessions/${sessionId}/refine`, { method: "POST" });
  },
  scopedRetrieval(sessionId: string, sentenceIndex: number, citationId: number) {
    return request<{ citations: CitationIndexEntry[] }>(`/sessions/${sessionId}/scoped-retrieval`, {
      method: "POST",
      body: JSON.stringify({ sentence_index: sentenceIndex, citation_id: citationId, top_k: 3 }),
    });
  },
  replaceCitation(sessionId: string, sentenceIndex: number, citationId: number, replacementChunkId: string) {
    return request<SessionEnvelope>(`/sessions/${sessionId}/citations/replace`, {
      method: "POST",
      body: JSON.stringify({
        sentence_index: sentenceIndex,
        citation_id: citationId,
        replacement_chunk_id: replacementChunkId,
      }),
    });
  },
  undoReplacement(sessionId: string, citationId: number) {
    return request<SessionEnvelope>(`/sessions/${sessionId}/citations/${citationId}/replacement`, {
      method: "DELETE",
    });
  },
};
