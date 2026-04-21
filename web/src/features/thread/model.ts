import type {
  AssistantMessageSource,
  AssistantTurnState,
  CitationIndexEntry,
  ConversationContextTurn,
  MessageRunState,
  SessionState,
} from "@/types";

export type UserThreadEntry = {
  id: string;
  kind: "user";
  query: string;
  createdAt: string;
  synthetic?: boolean;
  researchBacked?: boolean;
};

export type ResearchAssistantThreadEntry = {
  id: string;
  kind: "assistant";
  assistantKind: "research";
  source: AssistantMessageSource;
  session: SessionState;
  interactive: boolean;
};

export type ChatAssistantThreadEntry = {
  id: string;
  kind: "assistant";
  assistantKind: "chat";
  source: AssistantMessageSource;
  turn: AssistantTurnState;
  interactive: false;
};

export type AssistantThreadEntry =
  | ResearchAssistantThreadEntry
  | ChatAssistantThreadEntry;

export type PendingAssistantThreadEntry = {
  id: string;
  kind: "pending-assistant";
  userEntryId: string;
  messageRun: MessageRunState | null;
  session: SessionState | null;
};

export type ThreadEntry =
  | UserThreadEntry
  | AssistantThreadEntry
  | PendingAssistantThreadEntry;

export interface PersistedThreadPayload {
  version: 2;
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
): ResearchAssistantThreadEntry {
  return {
    id: makeId(),
    kind: "assistant",
    assistantKind: "research",
    source,
    session,
    interactive,
  };
}

export function chatAssistantEntry(
  source: AssistantMessageSource,
  turn: AssistantTurnState,
): ChatAssistantThreadEntry {
  return {
    id: makeId(),
    kind: "assistant",
    assistantKind: "chat",
    source,
    turn,
    interactive: false,
  };
}

export function userEntry(
  query: string,
  options: {
    createdAt: string;
    synthetic?: boolean;
    researchBacked?: boolean;
  },
): UserThreadEntry {
  return {
    id: makeId(),
    kind: "user",
    query,
    createdAt: options.createdAt,
    synthetic: options.synthetic ?? false,
    researchBacked: options.researchBacked,
  };
}

export function isAssistantEntry(entry: ThreadEntry): entry is AssistantThreadEntry {
  return entry.kind === "assistant";
}

export function isResearchAssistantEntry(
  entry: ThreadEntry,
): entry is ResearchAssistantThreadEntry {
  return isAssistantEntry(entry) && entry.assistantKind === "research";
}

export function isChatAssistantEntry(
  entry: ThreadEntry,
): entry is ChatAssistantThreadEntry {
  return isAssistantEntry(entry) && entry.assistantKind === "chat";
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

function isAssistantTurnState(value: unknown): value is AssistantTurnState {
  return (
    isRecord(value) &&
    typeof value.turn_id === "string" &&
    typeof value.content === "string" &&
    typeof value.used_search === "boolean" &&
    typeof value.response_basis === "string"
  );
}

function isMessageRunState(value: unknown): value is MessageRunState {
  return (
    isRecord(value) &&
    typeof value.message_run_id === "string" &&
    typeof value.status === "string" &&
    typeof value.stage === "string" &&
    typeof value.stage_label === "string" &&
    typeof value.stage_detail === "string"
  );
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
    return (
      typeof value.query === "string" &&
      (value.synthetic === undefined || typeof value.synthetic === "boolean") &&
      (value.researchBacked === undefined ||
        typeof value.researchBacked === "boolean")
    );
  }
  if (value.kind === "assistant") {
    if (value.assistantKind === "chat") {
      return (
        typeof value.source === "string" &&
        typeof value.interactive === "boolean" &&
        isAssistantTurnState(value.turn)
      );
    }
    return (
      typeof value.source === "string" &&
      typeof value.interactive === "boolean" &&
      isRecord(value.session)
    );
  }
  if (value.kind === "pending-assistant") {
    return (
      (value.userEntryId === undefined ||
        typeof value.userEntryId === "string") &&
      (value.messageRun === undefined ||
        value.messageRun === null ||
        isMessageRunState(value.messageRun)) &&
      (value.session === undefined ||
        value.session === null ||
        isRecord(value.session))
    );
  }
  return false;
}

function normalizeAssistantEntry(entry: AssistantThreadEntry): AssistantThreadEntry {
  if (entry.assistantKind === "chat") {
    return {
      ...entry,
      interactive: false,
    };
  }
  return entry;
}

export function normalizeThread(entries: ThreadEntry[]): ThreadEntry[] {
  const cleaned = entries.flatMap<ThreadEntry>((entry, index, source): ThreadEntry[] => {
    if (entry.kind === "user") {
      return [
        {
          ...entry,
          createdAt: entry.createdAt ?? new Date().toISOString(),
          synthetic: entry.synthetic ?? false,
        },
      ];
    }
    if (entry.kind === "pending-assistant") {
      const fallbackUserEntryId =
        entry.userEntryId ??
        [...source.slice(0, index)]
          .reverse()
          .find((candidate): candidate is UserThreadEntry => candidate.kind === "user")
          ?.id ??
        entry.id;
      const normalizedEntry: PendingAssistantThreadEntry = {
        ...entry,
        userEntryId: fallbackUserEntryId,
      };
      if (normalizedEntry.session?.query_status === "completed") {
        return [assistantEntry("query", normalizedEntry.session, true)];
      }
      if (normalizedEntry.messageRun?.assistant_turn) {
        return [chatAssistantEntry("query", normalizedEntry.messageRun.assistant_turn)];
      }
      if (!normalizedEntry.session && !normalizedEntry.messageRun) {
        return [];
      }
      return [normalizedEntry];
    }
    if (entry.assistantKind === undefined) {
      const legacyAssistant = entry as ResearchAssistantThreadEntry & {
        assistantKind?: "research";
      };
      return [
        {
          ...legacyAssistant,
          assistantKind: "research",
        } satisfies ResearchAssistantThreadEntry,
      ];
    }
    return [normalizeAssistantEntry(entry)];
  });

  const hasPending = cleaned.some(
    (entry) => entry.kind === "pending-assistant",
  );
  if (hasPending) {
    return cleaned.map((entry) =>
      isResearchAssistantEntry(entry) ? { ...entry, interactive: false } : entry,
    );
  }

  let interactiveAssigned = false;
  return [...cleaned]
    .reverse()
    .map((entry) => {
      if (!isResearchAssistantEntry(entry)) {
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

function getThreadTitleEntry(thread: ThreadEntry[]): UserThreadEntry | null {
  return (
    thread.find(
      (entry): entry is UserThreadEntry =>
        entry.kind === "user" && !entry.synthetic,
    ) ?? null
  );
}

export function getThreadTitleQuery(thread: ThreadEntry[]): string | null {
  return getThreadTitleEntry(thread)?.query ?? null;
}

export function deriveRecentResearchFromThread(
  thread: ThreadEntry[],
): RecentResearchItem[] {
  const titleEntry = getThreadTitleEntry(thread);
  const hasResearch = thread.some(
    (entry) =>
      (entry.kind === "user" && entry.researchBacked === true) ||
      isResearchAssistantEntry(entry),
  );

  if (!titleEntry || !hasResearch) {
    return [];
  }

  return [
    {
      id: titleEntry.id,
      query: titleEntry.query,
      createdAt: titleEntry.createdAt,
    },
  ];
}

export function updateRecentResearch(
  current: RecentResearchItem[],
  thread: ThreadEntry[],
): RecentResearchItem[] {
  const titleEntry = getThreadTitleEntry(thread);
  const title = titleEntry?.query.trim() ?? "";
  const normalized = title.toLowerCase();

  if (!titleEntry || !normalized) {
    return current;
  }

  const existing = current.find(
    (item) => item.query.trim().toLowerCase() === normalized,
  );
  const nextItem: RecentResearchItem = {
    id: existing?.id ?? titleEntry.id,
    query: existing?.query ?? title,
    createdAt: existing?.createdAt ?? titleEntry.createdAt,
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
          ((entry.session !== null && entry.session.query_status === "running") ||
            (entry.messageRun !== null && entry.messageRun.status === "running")),
      ) ?? null
  );
}

export function buildConversationContextTurns(
  thread: ThreadEntry[],
  limit = 8,
): ConversationContextTurn[] {
  const turns = thread.flatMap<ConversationContextTurn>((entry) => {
    if (entry.kind === "pending-assistant") {
      return [];
    }
    if (entry.kind === "user") {
      return [
        {
          role: "user",
          text: entry.query,
          search_backed: entry.researchBacked !== false,
          derived_from_session_id: undefined,
        },
      ];
    }
    if (entry.assistantKind === "chat") {
      return [
        {
          role: "assistant",
          text: entry.turn.content,
          search_backed: false,
          session_id: entry.turn.linked_session_id ?? null,
          derived_from_session_id: entry.turn.derived_from_session_id ?? null,
        },
      ];
    }
    return [
      {
        role: "assistant",
        text: researchAssistantText(entry.session),
        search_backed: true,
        session_id: entry.session.session_id,
        derived_from_session_id: entry.session.derived_from_session_id ?? null,
      },
    ];
  });

  return turns.slice(-limit);
}

export function researchAssistantText(session: SessionState) {
  const response = session.generated_response.trim();
  if (response) {
    return response;
  }
  const latestMessage = session.ui_messages.at(-1)?.message?.trim();
  if (latestMessage) {
    return latestMessage;
  }
  return session.query_stage_detail.trim() || "Answer unavailable.";
}
