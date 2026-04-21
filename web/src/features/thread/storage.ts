import {
  deriveRecentResearchFromThread,
  isRecentResearchItem,
  isRecord,
  isThreadEntry,
  MAX_RECENT_RESEARCH_ITEMS,
  normalizeThread,
  RECENT_RESEARCH_STORAGE_KEY,
  STORAGE_SOFT_LIMIT_BYTES,
  THREAD_STORAGE_KEY,
  type PersistedRecentResearchPayload,
  type PersistedThreadPayload,
  type RecentResearchItem,
  type ThreadEntry,
} from "./model";

interface StorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

export interface InitialClientState {
  thread: ThreadEntry[];
  recentResearch: RecentResearchItem[];
}

function getBrowserStorage(): StorageLike | null {
  if (typeof window === "undefined") {
    return null;
  }
  return window.localStorage;
}

export function loadPersistedThread(
  storage: StorageLike | null = getBrowserStorage(),
): ThreadEntry[] {
  if (!storage) {
    return [];
  }

  try {
    const raw = storage.getItem(THREAD_STORAGE_KEY);
    if (!raw) {
      return [];
    }

    const payload = JSON.parse(raw) as
      | PersistedThreadPayload
      | { version?: number; thread?: unknown }
      | ThreadEntry[];
    const maybeThread = Array.isArray(payload)
      ? payload
      : isRecord(payload) &&
          (payload.version === 1 || payload.version === 2) &&
          Array.isArray(payload.thread)
        ? payload.thread
        : [];

    return normalizeThread(maybeThread.filter(isThreadEntry));
  } catch {
    return [];
  }
}

export function persistThread(
  thread: ThreadEntry[],
  storage: StorageLike | null = getBrowserStorage(),
) {
  if (!storage) {
    return;
  }

  if (!thread.length) {
    storage.removeItem(THREAD_STORAGE_KEY);
    return;
  }

  const payload: PersistedThreadPayload = {
    version: 2,
    thread,
  };
  const serialized = JSON.stringify(payload);
  if (serialized.length > STORAGE_SOFT_LIMIT_BYTES) {
    const compactThread = normalizeThread(thread).slice(-80);
    storage.setItem(
      THREAD_STORAGE_KEY,
      JSON.stringify({
        version: 2,
        thread: compactThread,
      } satisfies PersistedThreadPayload),
    );
    return;
  }

  try {
    storage.setItem(THREAD_STORAGE_KEY, serialized);
  } catch (error) {
    if (error instanceof DOMException && error.name === "QuotaExceededError") {
      const compactThread = normalizeThread(thread).slice(-40);
      storage.setItem(
        THREAD_STORAGE_KEY,
        JSON.stringify({
          version: 2,
          thread: compactThread,
        } satisfies PersistedThreadPayload),
      );
      return;
    }
    throw error;
  }
}

export function loadPersistedRecentResearch(
  initialThread: ThreadEntry[],
  storage: StorageLike | null = getBrowserStorage(),
): RecentResearchItem[] {
  if (!storage) {
    return deriveRecentResearchFromThread(initialThread);
  }

  try {
    const raw = storage.getItem(RECENT_RESEARCH_STORAGE_KEY);
    if (!raw) {
      return deriveRecentResearchFromThread(initialThread);
    }

    const payload = JSON.parse(raw) as
      | PersistedRecentResearchPayload
      | RecentResearchItem[];
    const items = Array.isArray(payload)
      ? payload
      : isRecord(payload) &&
          payload.version === 1 &&
          Array.isArray(payload.recentResearch)
        ? payload.recentResearch
        : [];

    return items
      .filter(isRecentResearchItem)
      .slice(0, MAX_RECENT_RESEARCH_ITEMS);
  } catch {
    return deriveRecentResearchFromThread(initialThread);
  }
}

export function persistRecentResearch(
  recentResearch: RecentResearchItem[],
  storage: StorageLike | null = getBrowserStorage(),
) {
  if (!storage) {
    return;
  }

  if (!recentResearch.length) {
    storage.removeItem(RECENT_RESEARCH_STORAGE_KEY);
    return;
  }

  const payload: PersistedRecentResearchPayload = {
    version: 1,
    recentResearch,
  };
  try {
    storage.setItem(RECENT_RESEARCH_STORAGE_KEY, JSON.stringify(payload));
  } catch (error) {
    if (error instanceof DOMException && error.name === "QuotaExceededError") {
      storage.setItem(
        RECENT_RESEARCH_STORAGE_KEY,
        JSON.stringify({
          version: 1,
          recentResearch: recentResearch.slice(
            0,
            Math.max(1, Math.floor(MAX_RECENT_RESEARCH_ITEMS / 2)),
          ),
        } satisfies PersistedRecentResearchPayload),
      );
      return;
    }
    throw error;
  }
}

export function loadInitialClientState(
  storage: StorageLike | null = getBrowserStorage(),
): InitialClientState {
  const thread = loadPersistedThread(storage);
  return {
    thread,
    recentResearch: loadPersistedRecentResearch(thread, storage),
  };
}
