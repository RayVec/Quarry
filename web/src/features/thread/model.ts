import type {
  AssistantMessageSource,
  CitationIndexEntry,
  SessionState,
} from "@/types";

export type UserThreadEntry = {
  id: string;
  kind: "user";
  query: string;
  createdAt: string;
};

export type AssistantThreadEntry = {
  id: string;
  kind: "assistant";
  source: AssistantMessageSource;
  session: SessionState;
  interactive: boolean;
};

export type PendingAssistantThreadEntry = {
  id: string;
  kind: "pending-assistant";
  session: SessionState | null;
};

export type ThreadEntry =
  | UserThreadEntry
  | AssistantThreadEntry
  | PendingAssistantThreadEntry;

export interface PersistedThreadPayload {
  version: 1;
  thread: ThreadEntry[];
}

export interface RecentResearchItem {
  id: string;
  query: string;
  createdAt: string;
}

export interface PersistedRecentResearchPayload {
  version: 1;
  recentResearch: RecentResearchItem[];
}

export interface ActiveCitationContext {
  entryId: string;
  session: SessionState;
  sentenceIndex: number;
  referenceQuote: string;
  citation: CitationIndexEntry;
  readOnly: boolean;
}

export interface CitationAlternativesCacheEntry {
  hasLoaded: boolean;
  alternatives: CitationIndexEntry[];
}

export const THREAD_STORAGE_KEY = "quarry-thread-v1";
export const RECENT_RESEARCH_STORAGE_KEY = "quarry-recent-research-v1";
export const MAX_RECENT_RESEARCH_ITEMS = 8;
export const STORAGE_SOFT_LIMIT_BYTES = 4_500_000;

export function makeId() {
  if ("randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function assistantEntry(
  source: AssistantMessageSource,
  session: SessionState,
  interactive: boolean,
): AssistantThreadEntry {
  return {
    id: makeId(),
    kind: "assistant",
    source,
    session,
    interactive,
  };
}

export function isAssistantEntry(entry: ThreadEntry): entry is AssistantThreadEntry {
  return entry.kind === "assistant";
}

export function isPendingAssistantEntry(
  entry: ThreadEntry,
): entry is PendingAssistantThreadEntry {
  return entry.kind === "pending-assistant";
}

export function citationAlternativesCacheKey(
  sessionId: string,
  citationId: number,
  chunkId: string,
) {
  return `${sessionId}:${citationId}:${chunkId}`;
}

export function resolveActiveCitationContext(
  current: ActiveCitationContext,
  nextSession: SessionState,
) {
  const sentence =
    nextSession.parsed_sentences.find(
      (item) => item.sentence_index === current.sentenceIndex,
    ) ?? null;

  const resolvedCitationId =
    sentence?.references.find(
      (reference) =>
        reference.reference_quote === current.referenceQuote &&
        reference.citation_id != null,
    )?.citation_id ??
    sentence?.references.find(
      (reference) => reference.citation_id === current.citation.citation_id,
    )?.citation_id ??
    current.citation.citation_id;

  const citation = nextSession.citation_index.find(
    (item) => item.citation_id === resolvedCitationId,
  );
  if (!citation) {
    return null;
  }

  return { ...current, session: nextSession, citation };
}

export function clearCitationAlternativesCache(
  cache: Map<string, CitationAlternativesCacheEntry>,
  sessionId: string,
  citationId: number,
) {
  const prefix = `${sessionId}:${citationId}:`;
  for (const key of cache.keys()) {
    if (key.startsWith(prefix)) {
      cache.delete(key);
    }
  }
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

export function isThreadEntry(value: unknown): value is ThreadEntry {
  if (
    !isRecord(value) ||
    typeof value.id !== "string" ||
    typeof value.kind !== "string"
  ) {
    return false;
  }
  if (value.kind === "user") {
    return typeof value.query === "string";
  }
  if (value.kind === "assistant") {
    return (
      typeof value.source === "string" &&
      typeof value.interactive === "boolean" &&
      isRecord(value.session)
    );
  }
  if (value.kind === "pending-assistant") {
    return value.session === null || isRecord(value.session);
  }
  return false;
}

export function normalizeThread(entries: ThreadEntry[]): ThreadEntry[] {
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

  const hasPending = cleaned.some(
    (entry) => entry.kind === "pending-assistant",
  );
  if (hasPending) {
    return cleaned.map((entry) =>
      entry.kind === "assistant" ? { ...entry, interactive: false } : entry,
    );
  }

  let interactiveAssigned = false;
  return [...cleaned]
    .reverse()
    .map((entry) => {
      if (entry.kind !== "assistant") {
        return entry;
      }
      if (!interactiveAssigned) {
        interactiveAssigned = true;
        return { ...entry, interactive: true };
      }
      return entry.interactive ? { ...entry, interactive: false } : entry;
    })
    .reverse();
}

export function isRecentResearchItem(value: unknown): value is RecentResearchItem {
  return (
    isRecord(value) &&
    typeof value.id === "string" &&
    typeof value.query === "string" &&
    typeof value.createdAt === "string"
  );
}

export function deriveRecentResearchFromThread(
  thread: ThreadEntry[],
): RecentResearchItem[] {
  const seen = new Set<string>();
  const recent: RecentResearchItem[] = [];

  [...thread].reverse().forEach((entry) => {
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

export function updateRecentResearch(
  current: RecentResearchItem[],
  query: string,
): RecentResearchItem[] {
  const normalized = query.trim().toLowerCase();
  if (!normalized) {
    return current;
  }

  const nextItem: RecentResearchItem = {
    id: makeId(),
    query: query.trim(),
    createdAt: new Date().toISOString(),
  };

  const deduped = current.filter(
    (item) => item.query.trim().toLowerCase() !== normalized,
  );
  return [nextItem, ...deduped].slice(0, MAX_RECENT_RESEARCH_ITEMS);
}

export function formatRecentResearchDate(value: string) {
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

export function findResumablePendingEntry(thread: ThreadEntry[]) {
  return (
    [...thread]
      .reverse()
      .find(
        (entry): entry is PendingAssistantThreadEntry =>
          isPendingAssistantEntry(entry) &&
          entry.session !== null &&
          entry.session.query_status === "running",
      ) ?? null
  );
}
