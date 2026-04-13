import type {
  CitationIndexEntry,
  HostedSettingsEnvelope,
  HostedSettingsUpdatePayload,
  SessionEnvelope,
} from "./types";

const maybeEnv = (
  import.meta as ImportMeta & { env?: Record<string, string | undefined> }
).env;
const API_ROOT =
  maybeEnv?.VITE_API_ROOT?.replace(/\/+$/, "") ||
  "http://127.0.0.1:8000/api/v1";

type ApiErrorPayload = {
  code?: string;
  message?: string;
  details?: Record<string, unknown> | null;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });
  if (!response.ok) {
    const raw = await response.text();
    let detail: string | null = null;
    let code: string | null = null;
    try {
      const payload = JSON.parse(raw) as { detail?: string | ApiErrorPayload };
      if (typeof payload.detail === "string" && payload.detail.trim()) {
        detail = payload.detail;
      } else if (payload.detail && typeof payload.detail === "object") {
        detail =
          typeof payload.detail.message === "string" &&
          payload.detail.message.trim()
            ? payload.detail.message
            : null;
        code =
          typeof payload.detail.code === "string" && payload.detail.code.trim()
            ? payload.detail.code
            : null;
      }
    } catch {
      detail = null;
    }
    throw new Error(code ? `${code}: ${detail ?? raw}` : (detail ?? raw));
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return response.json() as Promise<T>;
}

export const api = {
  getHostedSettings() {
    return request<HostedSettingsEnvelope>("/settings/hosted");
  },
  updateHostedSettings(payload: HostedSettingsUpdatePayload) {
    return request<HostedSettingsEnvelope>("/settings/hosted", {
      method: "PUT",
      body: JSON.stringify(payload),
    });
  },
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
  addComment(
    sessionId: string,
    payload: {
      text_selection: string;
      char_start: number;
      char_end: number;
      comment_text: string;
    },
  ) {
    return request<SessionEnvelope>(`/sessions/${sessionId}/comments`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },
  updateComment(sessionId: string, commentId: string, commentText: string) {
    return request<SessionEnvelope>(
      `/sessions/${sessionId}/comments/${commentId}`,
      {
        method: "PATCH",
        body: JSON.stringify({ comment_text: commentText }),
      },
    );
  },
  deleteComment(sessionId: string, commentId: string) {
    return request<SessionEnvelope>(
      `/sessions/${sessionId}/comments/${commentId}`,
      {
        method: "DELETE",
      },
    );
  },
  scopedRetrieval(
    sessionId: string,
    sentenceIndex: number,
    citationId: number,
  ) {
    return request<{ citations: CitationIndexEntry[] }>(
      `/sessions/${sessionId}/citations/${citationId}/scoped`,
      {
        method: "POST",
        body: JSON.stringify({ sentence_index: sentenceIndex }),
      },
    );
  },
  replaceCitation(
    sessionId: string,
    sentenceIndex: number,
    citationId: number,
    replacementChunkId: string,
  ) {
    return request<SessionEnvelope>(
      `/sessions/${sessionId}/citations/${citationId}/replace`,
      {
        method: "POST",
        body: JSON.stringify({
          sentence_index: sentenceIndex,
          replacement_chunk_id: replacementChunkId,
        }),
      },
    );
  },
  undoReplacement(sessionId: string, citationId: number) {
    return request<SessionEnvelope>(
      `/sessions/${sessionId}/citations/${citationId}/undo`,
      {
        method: "POST",
      },
    );
  },
  setCitationFeedback(
    sessionId: string,
    sentenceIndex: number,
    citationId: number,
    feedbackType: "like" | "dislike" | "neutral",
  ) {
    return request<SessionEnvelope>(
      `/sessions/${sessionId}/citations/${citationId}/feedback`,
      {
        method: "POST",
        body: JSON.stringify({
          sentence_index: sentenceIndex,
          feedback_type: feedbackType,
        }),
      },
    );
  },
  getCitationAlternatives(sessionId: string, citationId: number) {
    return request<{ citations: CitationIndexEntry[] }>(
      `/sessions/${sessionId}/citations/${citationId}/alternatives`,
    );
  },
  replaceWithAlternative(
    sessionId: string,
    sentenceIndex: number,
    citationId: number,
    replacementCitationId: number,
  ) {
    return request<SessionEnvelope>(
      `/sessions/${sessionId}/citations/${citationId}/replace`,
      {
        method: "PUT",
        body: JSON.stringify({
          sentence_index: sentenceIndex,
          replacement_citation_id: replacementCitationId,
        }),
      },
    );
  },
  refine(sessionId: string) {
    return request<SessionEnvelope>(`/sessions/${sessionId}/refine`, {
      method: "POST",
    });
  },
};
