import { useEffect, useRef } from "react";
import { api } from "@/api";
import type { SessionState } from "@/types";
import { findResumablePendingEntry, type ThreadEntry } from "./model";

interface UseQueryPollingOptions {
  thread: ThreadEntry[];
  onResumePendingSession: () => void;
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
  onResumePendingSession,
  onPendingSessionUpdated,
  onPendingSessionCompleted,
  onPendingSessionFailed,
  onPollingSettled,
}: UseQueryPollingOptions) {
  const pollTokenRef = useRef(0);
  const resumedPersistedPendingRef = useRef(false);
  const callbacksRef = useRef({
    onResumePendingSession,
    onPendingSessionUpdated,
    onPendingSessionCompleted,
    onPendingSessionFailed,
    onPollingSettled,
  });

  callbacksRef.current = {
    onResumePendingSession,
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

  useEffect(() => {
    if (resumedPersistedPendingRef.current) {
      return;
    }
    resumedPersistedPendingRef.current = true;

    const pending = findResumablePendingEntry(thread);
    if (!pending?.session) {
      return;
    }

    callbacksRef.current.onResumePendingSession();
    const pollToken = ++pollTokenRef.current;
    void pollQueryProgress(
      pending.id,
      pending.session.session_id,
      pollToken,
    ).catch(() => {
      callbacksRef.current.onPollingSettled();
    });
  }, [thread]);

  useEffect(() => {
    return () => {
      pollTokenRef.current += 1;
    };
  }, []);

  function cancelPolling() {
    pollTokenRef.current += 1;
  }

  function startPolling(pendingId: string, sessionId: string) {
    const pollToken = ++pollTokenRef.current;
    return pollQueryProgress(pendingId, sessionId, pollToken);
  }

  return {
    cancelPolling,
    startPolling,
  };
}
