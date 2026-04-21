import { startTransition, useReducer, useRef } from "react";
import { api } from "@/api";
import type { ThreadActions } from "@/context/threadActions";
import { useCitationDialogState } from "@/features/citations/useCitationDialogState";
import type { MessageRunState, SessionState } from "@/types";
import { buildConversationContextTurns, makeId } from "./model";
import {
  createThreadControllerState,
  threadControllerReducer,
} from "./reducer";
import {
  getLatestGroundedSessionId,
  getLatestSession,
  getThreadTitleQuery,
} from "./selectors";
import { loadInitialClientState } from "./storage";
import { usePersistentThread } from "./usePersistentThread";
import { useQueryPolling } from "./useQueryPolling";

type CommentPayload = {
  text_selection: string;
  char_start: number;
  char_end: number;
  comment_text: string;
};

type SessionComment = SessionState["feedback"]["comments"][number];

export function useThreadController() {
  const [state, dispatch] = useReducer(
    threadControllerReducer,
    undefined,
    () => createThreadControllerState(loadInitialClientState()),
  );
  const localCommentIds = useRef(new Set<string>());
  const citationDialog = useCitationDialogState();

  usePersistentThread(state.thread, state.recentResearch);

  const { cancelPolling, startMessageRunPolling, startSessionPolling } = useQueryPolling({
    thread: state.thread,
    onResumePendingWork() {
      dispatch({ type: "loading/set", loading: true });
    },
    onPendingMessageRunUpdated(pendingId, nextMessageRun) {
      startTransition(() => {
        dispatch({
          type: "thread/pendingMessageRunUpdated",
          pendingId,
          messageRun: nextMessageRun,
        });
      });
    },
    onPendingMessageRunFailed(pendingId, nextMessageRun) {
      startTransition(() => {
        dispatch({
          type: "thread/pendingMessageRunFailed",
          pendingId,
          messageRun: nextMessageRun,
        });
      });
    },
    onPendingChatCompleted(pendingId, nextTurn) {
      dispatch({
        type: "message/chatCompleted",
        pendingId,
        turn: nextTurn,
      });
    },
    onPendingResearchStarted(
      pendingId,
      userEntryId,
      nextMessageRun,
      nextSession,
    ) {
      dispatch({
        type: "message/searchStarted",
        pendingId,
        userEntryId,
        messageRun: nextMessageRun,
        session: nextSession,
      });
    },
    onPendingSessionUpdated(pendingId, nextSession) {
      startTransition(() => {
        dispatch({
          type: "thread/pendingSessionUpdated",
          pendingId,
          session: nextSession,
        });
      });
    },
    onPendingSessionCompleted(pendingId, nextSession) {
      startTransition(() => {
        dispatch({
          type: "thread/pendingSessionCompleted",
          pendingId,
          session: nextSession,
        });
      });
    },
    onPendingSessionFailed(pendingId, nextSession) {
      startTransition(() => {
        dispatch({
          type: "thread/pendingSessionFailed",
          pendingId,
          session: nextSession,
        });
      });
    },
    onPollingSettled() {
      dispatch({ type: "loading/set", loading: false });
    },
  });

  function setQuery(nextQuery: string) {
    dispatch({ type: "query/set", query: nextQuery });
  }

  async function handleMessageRunResponse(
    pendingId: string,
    userEntryId: string,
    messageRun: MessageRunState,
  ) {
    dispatch({
      type: "message/runStarted",
      pendingId,
      messageRun,
    });

    if (messageRun.assistant_turn) {
      dispatch({
        type: "message/chatCompleted",
        pendingId,
        turn: messageRun.assistant_turn,
      });
      dispatch({ type: "loading/set", loading: false });
      return;
    }

    if (messageRun.session) {
      dispatch({
        type: "message/searchStarted",
        pendingId,
        userEntryId,
        messageRun,
        session: messageRun.session,
      });
      await startSessionPolling(pendingId, messageRun.session.session_id);
      return;
    }

    await startMessageRunPolling(pendingId, userEntryId, messageRun.message_run_id);
  }

  function replaceInteractiveSession(nextSession: SessionState) {
    citationDialog.syncInteractiveSession(nextSession);
    startTransition(() => {
      dispatch({
        type: "thread/interactiveSessionReplaced",
        session: nextSession,
      });
    });
  }

  function replaceAssistantEntrySession(
    entryId: string,
    nextSession: SessionState,
  ) {
    citationDialog.syncEntrySession(entryId, nextSession);
    startTransition(() => {
      dispatch({
        type: "thread/assistantSessionReplaced",
        entryId,
        session: nextSession,
      });
    });
  }

  function appendAssistantSession(nextSession: SessionState) {
    startTransition(() => {
      dispatch({
        type: "thread/refinementCompleted",
        session: nextSession,
        userQuery: "Please refine the previous answer.",
        createdAt: new Date().toISOString(),
      });
    });
  }

  function handleNewSearch() {
    cancelPolling();
    citationDialog.reset();
    startTransition(() => {
      dispatch({ type: "search/reset" });
    });
  }

  function handleDeleteRecentResearch(itemId: string) {
    startTransition(() => {
      dispatch({ type: "recentResearch/delete", itemId });
    });
  }

  async function submitQuery(
    queryOverride?: string,
    options?: { fresh?: boolean },
  ) {
    const submittedQuery = (queryOverride ?? state.query).trim();
    if (!submittedQuery) {
      return;
    }

    cancelPolling();
    if (options?.fresh) {
      citationDialog.clearCitationCache();
    }
    citationDialog.closeCitation();

    const contextTurns = options?.fresh
      ? []
      : buildConversationContextTurns(state.thread);
    const latestGroundedSessionId = options?.fresh
      ? null
      : getLatestGroundedSessionId(state.thread);
    const pendingId = makeId();
    const userEntryId = makeId();
    dispatch({
      type: "message/submitStarted",
      pendingId,
      userEntryId,
      query: submittedQuery,
      fresh: Boolean(options?.fresh),
      createdAt: new Date().toISOString(),
    });

    try {
      const response = await api.startMessage(submittedQuery, {
        contextTurns,
        latestGroundedSessionId,
      });
      await handleMessageRunResponse(
        pendingId,
        userEntryId,
        response.message_run,
      );
    } catch {
      dispatch({ type: "query/startFailed", pendingId });
    }
  }

  async function handleSaveSelectionComment(
    entryId: string,
    sessionId: string,
    payload: CommentPayload,
  ) {
    try {
      const response = await api.addComment(sessionId, payload);
      replaceAssistantEntrySession(entryId, response.session);
    } catch {
      const localComment: SessionComment = {
        comment_id: makeId(),
        text_selection: payload.text_selection,
        char_start: payload.char_start,
        char_end: payload.char_end,
        comment_text: payload.comment_text,
        resolved: false,
      };
      localCommentIds.current.add(localComment.comment_id);
      startTransition(() => {
        dispatch({
          type: "thread/localCommentAdded",
          entryId,
          comment: localComment,
        });
      });
    }
  }

  function updateCommentLocally(
    entryId: string,
    commentId: string,
    commentText: string,
  ) {
    startTransition(() => {
      dispatch({
        type: "thread/localCommentUpdated",
        entryId,
        commentId,
        commentText,
      });
    });
  }

  function deleteCommentLocally(entryId: string, commentId: string) {
    localCommentIds.current.delete(commentId);
    startTransition(() => {
      dispatch({
        type: "thread/localCommentDeleted",
        entryId,
        commentId,
      });
    });
  }

  async function handleUpdateSelectionComment(
    entryId: string,
    sessionId: string,
    commentId: string,
    commentText: string,
  ) {
    if (localCommentIds.current.has(commentId)) {
      updateCommentLocally(entryId, commentId, commentText);
      return;
    }

    try {
      const response = await api.updateComment(
        sessionId,
        commentId,
        commentText,
      );
      replaceAssistantEntrySession(entryId, response.session);
    } catch {
      updateCommentLocally(entryId, commentId, commentText);
    }
  }

  async function handleDeleteSelectionComment(
    entryId: string,
    sessionId: string,
    commentId: string,
  ) {
    if (localCommentIds.current.has(commentId)) {
      deleteCommentLocally(entryId, commentId);
      return;
    }

    try {
      const response = await api.deleteComment(sessionId, commentId);
      replaceInteractiveSession(response.session);
    } catch {
      deleteCommentLocally(entryId, commentId);
    }
  }

  const threadActions: ThreadActions = {
    openCitation: citationDialog.openCitation,
    saveComment: handleSaveSelectionComment,
    updateComment: handleUpdateSelectionComment,
    deleteComment: handleDeleteSelectionComment,
    refine: async (sessionId: string) => {
      try {
        const response = await api.refine(sessionId);
        appendAssistantSession(response.session);
      } catch {
        // Keep the current interactive response unchanged if refine fails.
      }
    },
  };

  return {
    query: state.query,
    thread: state.thread,
    recentResearch: state.recentResearch,
    loading: state.loading,
    latestSession: getLatestSession(state.thread),
    threadTitleQuery: getThreadTitleQuery(state.thread),
    setQuery,
    submitQuery,
    handleNewSearch,
    handleDeleteRecentResearch,
    threadActions,
    activeCitation: citationDialog.activeCitation,
    activeCitationAlternativesCacheEntry:
      citationDialog.getActiveCitationCacheEntry(),
    closeCitation: citationDialog.closeCitation,
    storeActiveCitationAlternatives:
      citationDialog.storeAlternativesForActive,
    handleCitationSessionUpdate: replaceInteractiveSession,
  };
}
