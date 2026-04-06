import { startTransition, useEffect, useLayoutEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { Cog, Trash2 } from "lucide-react";
import { api } from "./api";
import { CitationDialog } from "./components/CitationDialog";
import { ConversationMessage, type AssistantMessageSource } from "./components/ConversationMessage";
import { DiagnosticsDrawer } from "./components/DiagnosticsDrawer";
import { PendingConversationMessage } from "./components/PendingConversationMessage";
import { QueryComposer } from "./components/QueryComposer";
import type { CitationIndexEntry, ParsedSentence, SessionState } from "./types";
import "./styles/app.css";

type UserThreadEntry = {
  id: string;
  kind: "user";
  query: string;
  createdAt: string;
};

type AssistantThreadEntry = {
  id: string;
  kind: "assistant";
  source: AssistantMessageSource;
  session: SessionState;
  interactive: boolean;
};

type PendingAssistantThreadEntry = {
  id: string;
  kind: "pending-assistant";
  session: SessionState | null;
};

type ThreadEntry = UserThreadEntry | AssistantThreadEntry | PendingAssistantThreadEntry;

interface PersistedThreadPayload {
  version: 1;
  thread: ThreadEntry[];
}

const THREAD_STORAGE_KEY = "quarry-thread-v1";

interface RecentResearchItem {
  id: string;
  query: string;
  createdAt: string;
}

interface PersistedRecentResearchPayload {
  version: 1;
  recentResearch: RecentResearchItem[];
}

const RECENT_RESEARCH_STORAGE_KEY = "quarry-recent-research-v1";
const MAX_RECENT_RESEARCH_ITEMS = 8;

interface ActiveCitationContext {
  entryId: string;
  session: SessionState;
  sentenceIndex: number;
  referenceQuote: string;
  citation: CitationIndexEntry;
  readOnly: boolean;
}

function makeId() {
  if ("randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function assistantEntry(source: AssistantMessageSource, session: SessionState, interactive: boolean): AssistantThreadEntry {
  return {
    id: makeId(),
    kind: "assistant",
    source,
    session,
    interactive,
  };
}

function isAssistantEntry(entry: ThreadEntry): entry is AssistantThreadEntry {
  return entry.kind === "assistant";
}

function isPendingAssistantEntry(entry: ThreadEntry): entry is PendingAssistantThreadEntry {
  return entry.kind === "pending-assistant";
}

function sleep(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isThreadEntry(value: unknown): value is ThreadEntry {
  if (!isRecord(value) || typeof value.id !== "string" || typeof value.kind !== "string") {
    return false;
  }
  if (value.kind === "user") {
    return typeof value.query === "string";
  }
  if (value.kind === "assistant") {
    return typeof value.source === "string" && typeof value.interactive === "boolean" && isRecord(value.session);
  }
  if (value.kind === "pending-assistant") {
    return value.session === null || isRecord(value.session);
  }
  return false;
}

function normalizeThread(entries: ThreadEntry[]): ThreadEntry[] {
  const cleaned = entries.flatMap<ThreadEntry>((entry): ThreadEntry[] => {
    if (entry.kind === "user") {
      return [
        {
          ...entry,
          createdAt: entry.createdAt ?? new Date().toISOString(),
        },
      ];
    }
    if (entry.kind === "pending-assistant") {
      if (!entry.session) {
        return [];
      }
      if (entry.session.query_status === "completed") {
        return [
          {
            id: entry.id,
            kind: "assistant" as const,
            source: "query" as const,
            session: entry.session,
            interactive: true,
          },
        ];
      }
    }
    return [entry];
  });

  const hasPending = cleaned.some((entry) => entry.kind === "pending-assistant");
  if (hasPending) {
    return cleaned.map((entry) => (entry.kind === "assistant" ? { ...entry, interactive: false } : entry));
  }

  let interactiveAssigned = false;
  return [...cleaned].reverse().map((entry) => {
    if (entry.kind !== "assistant") {
      return entry;
    }
    if (!interactiveAssigned) {
      interactiveAssigned = true;
      return { ...entry, interactive: true };
    }
    return entry.interactive ? { ...entry, interactive: false } : entry;
  }).reverse();
}

function loadPersistedThread(): ThreadEntry[] {
  if (typeof window === "undefined") {
    return [];
  }

  try {
    const raw = window.localStorage.getItem(THREAD_STORAGE_KEY);
    if (!raw) {
      return [];
    }

    const payload = JSON.parse(raw) as PersistedThreadPayload | ThreadEntry[];
    const maybeThread = Array.isArray(payload)
      ? payload
      : isRecord(payload) && payload.version === 1 && Array.isArray(payload.thread)
        ? payload.thread
        : [];

    return normalizeThread(maybeThread.filter(isThreadEntry));
  } catch {
    return [];
  }
}

function persistThread(thread: ThreadEntry[]) {
  if (typeof window === "undefined") {
    return;
  }

  if (!thread.length) {
    window.localStorage.removeItem(THREAD_STORAGE_KEY);
    return;
  }

  const payload: PersistedThreadPayload = {
    version: 1,
    thread,
  };
  window.localStorage.setItem(THREAD_STORAGE_KEY, JSON.stringify(payload));
}

function isRecentResearchItem(value: unknown): value is RecentResearchItem {
  return (
    isRecord(value) &&
    typeof value.id === "string" &&
    typeof value.query === "string" &&
    typeof value.createdAt === "string"
  );
}

function deriveRecentResearchFromThread(thread: ThreadEntry[]): RecentResearchItem[] {
  const seen = new Set<string>();
  const recent: RecentResearchItem[] = [];

  [...thread]
    .reverse()
    .forEach((entry) => {
      if (entry.kind !== "user") {
        return;
      }
      const key = entry.query.trim().toLowerCase();
      if (!key || seen.has(key)) {
        return;
      }
      seen.add(key);
      recent.push({
        id: entry.id,
        query: entry.query,
        createdAt: entry.createdAt,
      });
    });

  return recent.slice(0, MAX_RECENT_RESEARCH_ITEMS);
}

function loadPersistedRecentResearch(initialThread: ThreadEntry[]): RecentResearchItem[] {
  if (typeof window === "undefined") {
    return deriveRecentResearchFromThread(initialThread);
  }

  try {
    const raw = window.localStorage.getItem(RECENT_RESEARCH_STORAGE_KEY);
    if (!raw) {
      return deriveRecentResearchFromThread(initialThread);
    }

    const payload = JSON.parse(raw) as PersistedRecentResearchPayload | RecentResearchItem[];
    const items = Array.isArray(payload)
      ? payload
      : isRecord(payload) && payload.version === 1 && Array.isArray(payload.recentResearch)
        ? payload.recentResearch
        : [];

    return items.filter(isRecentResearchItem).slice(0, MAX_RECENT_RESEARCH_ITEMS);
  } catch {
    return deriveRecentResearchFromThread(initialThread);
  }
}

function persistRecentResearch(recentResearch: RecentResearchItem[]) {
  if (typeof window === "undefined") {
    return;
  }

  if (!recentResearch.length) {
    window.localStorage.removeItem(RECENT_RESEARCH_STORAGE_KEY);
    return;
  }

  const payload: PersistedRecentResearchPayload = {
    version: 1,
    recentResearch,
  };
  window.localStorage.setItem(RECENT_RESEARCH_STORAGE_KEY, JSON.stringify(payload));
}

function updateRecentResearch(current: RecentResearchItem[], query: string): RecentResearchItem[] {
  const normalized = query.trim().toLowerCase();
  if (!normalized) {
    return current;
  }

  const nextItem: RecentResearchItem = {
    id: makeId(),
    query: query.trim(),
    createdAt: new Date().toISOString(),
  };

  const deduped = current.filter((item) => item.query.trim().toLowerCase() !== normalized);
  return [nextItem, ...deduped].slice(0, MAX_RECENT_RESEARCH_ITEMS);
}

function formatRecentResearchDate(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "RECENT";
  }

  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
  })
    .format(date)
    .toUpperCase();
}

export default function App() {
  const [query, setQuery] = useState("");
  const [thread, setThread] = useState<ThreadEntry[]>(() => loadPersistedThread());
  const [recentResearch, setRecentResearch] = useState<RecentResearchItem[]>(() =>
    loadPersistedRecentResearch(loadPersistedThread()),
  );
  const [loading, setLoading] = useState(false);
  const [diagnosticsOpen, setDiagnosticsOpen] = useState(false);
  const [activeCitation, setActiveCitation] = useState<ActiveCitationContext | null>(null);
  const resumedPersistedPendingRef = useRef(false);
  const workspaceColumnRef = useRef<HTMLElement | null>(null);
  const dockedComposerRef = useRef<HTMLFormElement | null>(null);
  const localCommentIds = useRef(new Set<string>());
  const [dockedComposerOffset, setDockedComposerOffset] = useState(0);

  const latestAssistant = useMemo(
    () => [...thread].reverse().find(isAssistantEntry) ?? null,
    [thread],
  );
  const interactiveAssistant = useMemo(
    () => [...thread].reverse().find((entry): entry is AssistantThreadEntry => isAssistantEntry(entry) && entry.interactive) ?? null,
    [thread],
  );
  const latestPendingSession = useMemo(
    () => [...thread].reverse().find((entry): entry is PendingAssistantThreadEntry => isPendingAssistantEntry(entry) && entry.session !== null)?.session ?? null,
    [thread],
  );
  const latestSession = interactiveAssistant?.session ?? latestAssistant?.session ?? latestPendingSession ?? null;
  const latestUserQuery = useMemo(
    () => [...thread].reverse().find((entry): entry is UserThreadEntry => entry.kind === "user")?.query ?? null,
    [thread],
  );

  useEffect(() => {
    persistThread(thread);
  }, [thread]);

  useEffect(() => {
    persistRecentResearch(recentResearch);
  }, [recentResearch]);

  useEffect(() => {
    if (resumedPersistedPendingRef.current) {
      return;
    }
    resumedPersistedPendingRef.current = true;

    const pending = [...thread].reverse().find(
      (entry): entry is PendingAssistantThreadEntry =>
        isPendingAssistantEntry(entry) && entry.session !== null && entry.session.query_status === "running",
    );
    if (!pending?.session) {
      return;
    }

    setLoading(true);
    void pollQueryProgress(pending.id, pending.session.session_id).catch(() => {
      setLoading(false);
    });
  }, [thread]);

  useLayoutEffect(() => {
    if (!thread.length) {
      setDockedComposerOffset(0);
      return;
    }

    const measure = () => {
      const workspaceColumn = workspaceColumnRef.current;
      const dockedComposer = dockedComposerRef.current;
      if (!workspaceColumn || !dockedComposer) {
        setDockedComposerOffset(0);
        return;
      }

      const dockedComposerRect = dockedComposer.getBoundingClientRect();
      const clearance = 16;
      const nextOffset = Math.ceil(dockedComposerRect.height + clearance);
      setDockedComposerOffset(nextOffset);
    };

    measure();

    const resizeObserver = new ResizeObserver(() => measure());
    if (workspaceColumnRef.current) {
      resizeObserver.observe(workspaceColumnRef.current);
    }
    if (dockedComposerRef.current) {
      resizeObserver.observe(dockedComposerRef.current);
    }

    window.addEventListener("resize", measure);
    return () => {
      resizeObserver.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, [thread]);

  function replaceInteractiveSession(nextSession: SessionState) {
    setThread((current) => {
      const next = [...current];
      for (let index = next.length - 1; index >= 0; index -= 1) {
        const entry = next[index];
        if (isAssistantEntry(entry) && entry.interactive) {
          next[index] = { ...entry, session: nextSession };
          break;
        }
      }
      return next;
    });

    setActiveCitation((current) => {
      if (!current || current.session.session_id !== nextSession.session_id) {
        return current;
      }
      const citation = nextSession.citation_index.find((item) => item.citation_id === current.citation.citation_id);
      if (!citation) {
        return null;
      }
      return { ...current, session: nextSession, citation };
    });
  }

  function appendAssistantSession(nextSession: SessionState, source: AssistantMessageSource) {
    setThread((current) => [
      ...current.map((entry) => (isAssistantEntry(entry) ? { ...entry, interactive: false } : entry)),
      assistantEntry(source, nextSession, true),
    ]);
  }

  function updatePendingSession(pendingId: string, nextSession: SessionState) {
    setThread((current) =>
      current.map((entry) =>
        isPendingAssistantEntry(entry) && entry.id === pendingId ? { ...entry, session: nextSession } : entry,
      ),
    );
  }

  function completePendingSession(pendingId: string, nextSession: SessionState) {
    setThread((current) =>
      current.map((entry) => {
        if (isPendingAssistantEntry(entry) && entry.id === pendingId) {
          return assistantEntry("query", nextSession, true);
        }
        if (isAssistantEntry(entry)) {
          return { ...entry, interactive: false };
        }
        return entry;
      }),
    );
  }

  function replaceAssistantEntrySession(entryId: string, nextSession: SessionState) {
    setThread((current) =>
      current.map((entry) =>
        isAssistantEntry(entry) && entry.id === entryId ? { ...entry, session: nextSession } : entry,
      ),
    );
    setActiveCitation((current) => {
      if (!current || current.entryId !== entryId) {
        return current;
      }
      const citation = nextSession.citation_index.find(
        (item) => item.citation_id === current.citation.citation_id,
      );
      if (!citation) {
        return null;
      }
      return { ...current, session: nextSession, citation };
    });
  }

  function failPendingSession(pendingId: string, nextSession: SessionState) {
    setThread((current) =>
      current.map((entry) =>
        isPendingAssistantEntry(entry) && entry.id === pendingId ? { ...entry, session: nextSession } : entry,
      ),
    );
  }

  async function pollQueryProgress(pendingId: string, sessionId: string) {
    while (true) {
      const response = await api.getSession(sessionId);
      const nextSession = response.session;
      if (nextSession.query_status === "completed") {
        startTransition(() => completePendingSession(pendingId, nextSession));
        setLoading(false);
        return;
      }
      if (nextSession.query_status === "failed") {
        startTransition(() => failPendingSession(pendingId, nextSession));
        setLoading(false);
        return;
      }
      startTransition(() => updatePendingSession(pendingId, nextSession));
      await sleep(900);
    }
  }

  function handleNewSearch() {
    startTransition(() => {
      setThread([]);
      setQuery("");
      setActiveCitation(null);
      setDiagnosticsOpen(false);
    });
    setLoading(false);
  }

  function handleDeleteRecentResearch(itemId: string) {
    startTransition(() => {
      setRecentResearch((current) => current.filter((item) => item.id !== itemId));
    });
  }

  async function submitQuery(queryOverride?: string, options?: { fresh?: boolean }) {
    const submittedQuery = (queryOverride ?? query).trim();
    if (!submittedQuery) return;

    const pendingId = makeId();
    startTransition(() => {
      setRecentResearch((current) => updateRecentResearch(current, submittedQuery));
      setThread((current) => [
        ...(options?.fresh ? [] : current.map((entry) => (isAssistantEntry(entry) ? { ...entry, interactive: false } : entry))),
        { id: makeId(), kind: "user", query: submittedQuery, createdAt: new Date().toISOString() },
        { id: pendingId, kind: "pending-assistant", session: null },
      ]);
      setQuery("");
      setActiveCitation(null);
    });
    setLoading(true);
    try {
      const response = await api.startQuery(submittedQuery);
      startTransition(() => updatePendingSession(pendingId, response.session));
      await pollQueryProgress(pendingId, response.session.session_id);
    } catch (error) {
      startTransition(() => {
        setThread((current) => current.filter((entry) => !(isPendingAssistantEntry(entry) && entry.id === pendingId)));
      });
      setLoading(false);
    }
  }

  async function handleSaveSelectionComment(
    entryId: string,
    sessionId: string,
    payload: {
      text_selection: string;
      char_start: number;
      char_end: number;
      comment_text: string;
    },
  ) {
    try {
      const response = await api.addComment(sessionId, payload);
      startTransition(() => replaceAssistantEntrySession(entryId, response.session));
    } catch {
      // Backend unavailable — save comment locally
      const localComment = {
        comment_id: makeId(),
        text_selection: payload.text_selection,
        char_start: payload.char_start,
        char_end: payload.char_end,
        comment_text: payload.comment_text,
        resolved: false,
      };
      localCommentIds.current.add(localComment.comment_id);
      startTransition(() => {
        setThread((current) =>
          current.map((entry) => {
            if (!isAssistantEntry(entry) || entry.id !== entryId) return entry;
            return {
              ...entry,
              session: {
                ...entry.session,
                feedback: {
                  ...entry.session.feedback,
                  comments: [...entry.session.feedback.comments, localComment],
                },
              },
            };
          }),
        );
      });
    }
  }

  function updateCommentLocally(entryId: string, commentId: string, commentText: string) {
    startTransition(() => {
      setThread((current) =>
        current.map((entry) => {
          if (!isAssistantEntry(entry) || entry.id !== entryId) return entry;
          return {
            ...entry,
            session: {
              ...entry.session,
              feedback: {
                ...entry.session.feedback,
                comments: entry.session.feedback.comments.map((c) =>
                  c.comment_id === commentId ? { ...c, comment_text: commentText } : c,
                ),
              },
            },
          };
        }),
      );
    });
  }

  function deleteCommentLocally(entryId: string, commentId: string) {
    localCommentIds.current.delete(commentId);
    startTransition(() => {
      setThread((current) =>
        current.map((entry) => {
          if (!isAssistantEntry(entry) || entry.id !== entryId) return entry;
          return {
            ...entry,
            session: {
              ...entry.session,
              feedback: {
                ...entry.session.feedback,
                comments: entry.session.feedback.comments.filter((c) => c.comment_id !== commentId),
              },
            },
          };
        }),
      );
    });
  }

  async function handleUpdateSelectionComment(entryId: string, sessionId: string, commentId: string, commentText: string) {
    if (localCommentIds.current.has(commentId)) {
      updateCommentLocally(entryId, commentId, commentText);
      return;
    }
    try {
      const response = await api.updateComment(sessionId, commentId, commentText);
      startTransition(() => replaceAssistantEntrySession(entryId, response.session));
    } catch {
      updateCommentLocally(entryId, commentId, commentText);
    }
  }

  async function handleDeleteSelectionComment(entryId: string, sessionId: string, commentId: string) {
    if (localCommentIds.current.has(commentId)) {
      deleteCommentLocally(entryId, commentId);
      return;
    }
    try {
      const response = await api.deleteComment(sessionId, commentId);
      startTransition(() => replaceInteractiveSession(response.session));
    } catch {
      deleteCommentLocally(entryId, commentId);
    }
  }

  async function handleRefine() {
    if (!interactiveAssistant) return;
    const response = await api.refine(interactiveAssistant.session.session_id);
    startTransition(() => appendAssistantSession(response.session, "refinement"));
  }

  return (
    <div className="app-shell">
      <aside className="app-sidebar">
        <div className="sidebar-brand">
          <span className="sidebar-mark">Q</span>
          <div>
            <p className="sidebar-title">QUARRY</p>
            <p className="sidebar-subtitle">Architectural Intelligence</p>
          </div>
        </div>

        <button className="sidebar-primary-action" onClick={handleNewSearch}>
          + New Search
        </button>

        <div className="sidebar-section">
          <span className="tiny-label">Recent Research</span>
          {recentResearch.length ? (
            <div className="recent-research-list">
              {recentResearch.map((item, index) => (
                <div
                  className={`recent-research-item ${index === 0 && latestUserQuery === item.query ? "active" : ""}`}
                  key={item.id}
                >
                  <button className="recent-research-open" onClick={() => void submitQuery(item.query, { fresh: true })}>
                    <strong>{item.query}</strong>
                    <span>{formatRecentResearchDate(item.createdAt)}</span>
                  </button>
                  <button
                    className="recent-research-delete"
                    aria-label={`Delete search ${item.query}`}
                    onClick={() => handleDeleteRecentResearch(item.id)}
                  >
                    <Trash2 aria-hidden="true" focusable="false" />
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <p className="sidebar-empty">
              Your recent research questions will appear here so you can return to them quickly.
            </p>
          )}
        </div>
      </aside>

      <div className="workspace-shell">
        {thread.length === 0 ? (
          <main className="landing-stage">
            <section className="hero-block">
              <h1>Intelligence for the built environment.</h1>
              <p>
                Ask complex questions, analyze technical reports, and verify construction guidance with editorial clarity and structural precision.
              </p>
            </section>

            <QueryComposer
              className="landing"
              id="query-input"
              loading={loading}
              placeholder="What is PDRI maturity?"
              query={query}
              onChange={setQuery}
              onSubmit={() => void submitQuery()}
            />
          </main>
        ) : (
          <main
            className="conversation-stage with-docked-offset"
            style={{ "--docked-composer-offset": `${dockedComposerOffset}px` } as CSSProperties}
          >
            <section className="workspace-column" ref={workspaceColumnRef}>
              <div className="thread-intro">
                <div className="thread-intro-copy">
                  <span className="tiny-label">Research Thread</span>
                  <h2>{latestUserQuery ?? "Current analysis"}</h2>
                </div>
                <button
                  className="ghost-button diagnostics-trigger"
                  data-testid="open-diagnostics"
                  onClick={() => setDiagnosticsOpen(true)}
                >
                  <span className="sr-only">Diagnostics</span>
                  <Cog aria-hidden="true" focusable="false" />
                </button>
              </div>

              <div className="thread-column" data-testid="conversation-thread">
                {thread.map((entry) =>
                  entry.kind === "user" ? (
                    <article className="thread-message user-message" key={entry.id}>
                      <span className="tiny-label">Query</span>
                      <p>{entry.query}</p>
                    </article>
                  ) : entry.kind === "pending-assistant" ? (
                    <PendingConversationMessage key={entry.id} session={entry.session} />
                  ) : (
                    <ConversationMessage
                      key={entry.id}
                      source={entry.source}
                      session={entry.session}
                      interactive={entry.interactive}
                      onOpenCitation={(session, sentence, citationId, referenceQuote, readOnly) => {
                        const citation = session.citation_index.find((item) => item.citation_id === citationId);
                        if (!citation) return;
                        setActiveCitation({
                          entryId: entry.id,
                          session,
                          sentenceIndex: sentence.sentence_index,
                          referenceQuote,
                          citation,
                          readOnly,
                        });
                      }}
                      onSaveComment={(session, payload) =>
                        handleSaveSelectionComment(entry.id, session.session_id, payload)
                      }
                      onUpdateComment={(session, commentId, commentText) =>
                        handleUpdateSelectionComment(entry.id, session.session_id, commentId, commentText)
                      }
                      onDeleteComment={(session, commentId) =>
                        handleDeleteSelectionComment(entry.id, session.session_id, commentId)
                      }
                      onRefine={handleRefine}
                      onRunClarificationSuggestion={(suggestion) => void submitQuery(suggestion, { fresh: true })}
                    />
                  ),
                )}
              </div>
            </section>

            <QueryComposer
              className="docked"
              id="thread-query-input"
              label=""
              loading={loading}
              placeholder="Ask follow-up questions"
              query={query}
              formRef={dockedComposerRef}
              onChange={setQuery}
              onSubmit={() => void submitQuery()}
            />
          </main>
        )}
      </div>

      <DiagnosticsDrawer session={latestSession} open={diagnosticsOpen} onClose={() => setDiagnosticsOpen(false)} />

      {activeCitation ? (
        <CitationDialog
          citation={activeCitation.citation}
          sentenceIndex={activeCitation.sentenceIndex}
          referenceQuote={activeCitation.referenceQuote}
          readOnly={activeCitation.readOnly}
          session={activeCitation.session}
          onClose={() => setActiveCitation(null)}
          onSessionUpdate={(nextSession) => startTransition(() => replaceInteractiveSession(nextSession))}
        />
      ) : null}
    </div>
  );
}
