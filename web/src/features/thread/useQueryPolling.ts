import { useEffect, useRef } from "react";
import { api } from "@/api";
import type { AssistantTurnState, MessageRunState, SessionState } from "@/types";
import { findResumablePendingEntry, type ThreadEntry } from "./model";

interface UseQueryPollingOptions {
  thread: ThreadEntry[];
  onResumePendingWork: () => void;
  onPendingMessageRunUpdated: (
    pendingId: string,
    nextMessageRun: MessageRunState,
  ) => void;
  onPendingMessageRunFailed: (
    pendingId: string,
    nextMessageRun: MessageRunState,
  ) => void;
  onPendingChatCompleted: (
    pendingId: string,
    nextTurn: AssistantTurnState,
  ) => void;
  onPendingResearchStarted: (
    pendingId: string,
    userEntryId: string,
    nextMessageRun: MessageRunState,
    nextSession: SessionState,
  ) => void;
  onPendingSessionUpdated: (
    pendingId: string,
    nextSession: SessionState,
  ) => void;
  onPendingSessionCompleted: (
    pendingId: string,
    nextSession: SessionState,
  ) => void;
  onPendingSessionFailed: (
    pendingId: string,
    nextSession: SessionState,
  ) => void;
  onPollingSettled: () => void;
}

function sleep(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

export function useQueryPolling({
  thread,
  onResumePendingWork,
  onPendingMessageRunUpdated,
  onPendingMessageRunFailed,
  onPendingChatCompleted,
  onPendingResearchStarted,
  onPendingSessionUpdated,
  onPendingSessionCompleted,
  onPendingSessionFailed,
  onPollingSettled,
}: UseQueryPollingOptions) {
  const pollTokenRef = useRef(0);
  const resumedPersistedPendingRef = useRef(false);
  const callbacksRef = useRef({
    onResumePendingWork,
    onPendingMessageRunUpdated,
    onPendingMessageRunFailed,
    onPendingChatCompleted,
    onPendingResearchStarted,
    onPendingSessionUpdated,
    onPendingSessionCompleted,
    onPendingSessionFailed,
    onPollingSettled,
  });

  callbacksRef.current = {
    onResumePendingWork,
    onPendingMessageRunUpdated,
    onPendingMessageRunFailed,
    onPendingChatCompleted,
    onPendingResearchStarted,
    onPendingSessionUpdated,
    onPendingSessionCompleted,
    onPendingSessionFailed,
    onPollingSettled,
  };

  async function pollQueryProgress(
    pendingId: string,
    sessionId: string,
    pollToken: number,
  ) {
    while (pollTokenRef.current === pollToken) {
      const response = await api.getSession(sessionId);
      const nextSession = response.session;
      if (nextSession.query_status === "completed") {
        callbacksRef.current.onPendingSessionCompleted(pendingId, nextSession);
        callbacksRef.current.onPollingSettled();
        return;
      }
      if (nextSession.query_status === "failed") {
        callbacksRef.current.onPendingSessionFailed(pendingId, nextSession);
        callbacksRef.current.onPollingSettled();
        return;
      }
      callbacksRef.current.onPendingSessionUpdated(pendingId, nextSession);
      await sleep(900);
    }
  }

  async function pollMessageRun(
    pendingId: string,
    userEntryId: string,
    messageRunId: string,
    pollToken: number,
  ) {
    while (pollTokenRef.current === pollToken) {
      const response = await api.getMessageRun(messageRunId);
      const nextMessageRun = response.message_run;
      if (nextMessageRun.assistant_turn) {
        callbacksRef.current.onPendingChatCompleted(
          pendingId,
          nextMessageRun.assistant_turn,
        );
        callbacksRef.current.onPollingSettled();
        return;
      }
      if (nextMessageRun.session) {
        callbacksRef.current.onPendingResearchStarted(
          pendingId,
          userEntryId,
          nextMessageRun,
          nextMessageRun.session,
        );
        await pollQueryProgress(
          pendingId,
          nextMessageRun.session.session_id,
          pollToken,
        );
        return;
      }
      if (nextMessageRun.status === "failed") {
        callbacksRef.current.onPendingMessageRunFailed(
          pendingId,
          nextMessageRun,
        );
        callbacksRef.current.onPollingSettled();
        return;
      }
      callbacksRef.current.onPendingMessageRunUpdated(
        pendingId,
        nextMessageRun,
      );
      await sleep(450);
    }
  }

  useEffect(() => {
    if (resumedPersistedPendingRef.current) {
      return;
    }
    resumedPersistedPendingRef.current = true;

    const pending = findResumablePendingEntry(thread);
    if (!pending) {
      return;
    }

    callbacksRef.current.onResumePendingWork();
    const pollToken = ++pollTokenRef.current;
    if (pending.session?.query_status === "running") {
      void pollQueryProgress(
        pending.id,
        pending.session.session_id,
        pollToken,
      ).catch(() => {
        callbacksRef.current.onPollingSettled();
      });
      return;
    }
    if (pending.messageRun?.status === "running") {
      void pollMessageRun(
        pending.id,
        pending.userEntryId,
        pending.messageRun.message_run_id,
        pollToken,
      ).catch(() => {
        callbacksRef.current.onPollingSettled();
      });
    }
  }, [thread]);

  useEffect(() => {
    return () => {
      pollTokenRef.current += 1;
    };
  }, []);

  function cancelPolling() {
    pollTokenRef.current += 1;
  }

  function startMessageRunPolling(
    pendingId: string,
    userEntryId: string,
    messageRunId: string,
  ) {
    const pollToken = ++pollTokenRef.current;
    return pollMessageRun(pendingId, userEntryId, messageRunId, pollToken);
  }

  function startSessionPolling(pendingId: string, sessionId: string) {
    const pollToken = ++pollTokenRef.current;
    return pollQueryProgress(pendingId, sessionId, pollToken);
  }

  return {
    cancelPolling,
    startMessageRunPolling,
    startSessionPolling,
  };
}
