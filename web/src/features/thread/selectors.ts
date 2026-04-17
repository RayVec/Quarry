import type { SessionState } from "@/types";
import {
  isAssistantEntry,
  isPendingAssistantEntry,
  type AssistantThreadEntry,
  type PendingAssistantThreadEntry,
  type ThreadEntry,
  type UserThreadEntry,
} from "./model";

export function getLatestAssistant(
  thread: ThreadEntry[],
): AssistantThreadEntry | null {
  return [...thread].reverse().find(isAssistantEntry) ?? null;
}

export function getInteractiveAssistant(
  thread: ThreadEntry[],
): AssistantThreadEntry | null {
  return (
    [...thread]
      .reverse()
      .find(
        (entry): entry is AssistantThreadEntry =>
          isAssistantEntry(entry) && entry.interactive,
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

export function getLatestUserQuery(thread: ThreadEntry[]): string | null {
  return (
    [...thread]
      .reverse()
      .find((entry): entry is UserThreadEntry => entry.kind === "user")
      ?.query ?? null
  );
}
