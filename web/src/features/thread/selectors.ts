import type { SessionState } from "@/types";
import {
  getThreadTitleQuery as getThreadTitleQueryFromModel,
  isChatAssistantEntry,
  isPendingAssistantEntry,
  isResearchAssistantEntry,
  type PendingAssistantThreadEntry,
  type ResearchAssistantThreadEntry,
  type ThreadEntry,
  type UserThreadEntry,
} from "./model";

export function getLatestAssistant(
  thread: ThreadEntry[],
): ResearchAssistantThreadEntry | null {
  return [...thread].reverse().find(isResearchAssistantEntry) ?? null;
}

export function getInteractiveAssistant(
  thread: ThreadEntry[],
): ResearchAssistantThreadEntry | null {
  return (
    [...thread]
      .reverse()
      .find(
        (entry): entry is ResearchAssistantThreadEntry =>
          isResearchAssistantEntry(entry) && entry.interactive,
      ) ?? null
  );
}

export function getLatestPendingSession(
  thread: ThreadEntry[],
): SessionState | null {
  return (
    [...thread]
      .reverse()
      .find(
        (entry): entry is PendingAssistantThreadEntry =>
          isPendingAssistantEntry(entry) && entry.session !== null,
      )?.session ?? null
  );
}

export function getLatestSession(thread: ThreadEntry[]): SessionState | null {
  const interactiveAssistant = getInteractiveAssistant(thread);
  const latestAssistant = getLatestAssistant(thread);
  const latestPendingSession = getLatestPendingSession(thread);

  return (
    interactiveAssistant?.session ??
    latestAssistant?.session ??
    latestPendingSession ??
    null
  );
}

export function getLatestGroundedSessionId(
  thread: ThreadEntry[],
): string | null {
  for (const entry of [...thread].reverse()) {
    if (
      isResearchAssistantEntry(entry) &&
      entry.session.query_status === "completed" &&
      entry.session.response_mode === "response_review" &&
      entry.session.generated_response.trim()
    ) {
      return entry.session.session_id;
    }
    if (isChatAssistantEntry(entry) && entry.turn.derived_from_session_id) {
      return entry.turn.derived_from_session_id;
    }
  }
  return null;
}

export function getLatestResearchQuery(thread: ThreadEntry[]): string | null {
  return (
    [...thread]
      .reverse()
      .find(
        (entry): entry is UserThreadEntry =>
          entry.kind === "user" &&
          !entry.synthetic &&
          entry.researchBacked === true,
      )
      ?.query ?? null
  );
}

export function getThreadTitleQuery(thread: ThreadEntry[]): string | null {
  return getThreadTitleQueryFromModel(thread);
}
