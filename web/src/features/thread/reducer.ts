import type { AssistantTurnState, MessageRunState, SessionState } from "@/types";
import {
  assistantEntry,
  chatAssistantEntry,
  isResearchAssistantEntry,
  isPendingAssistantEntry,
  normalizeThread,
  userEntry,
  updateRecentResearch,
  type ThreadEntry,
  type RecentResearchItem,
} from "./model";

type SessionComment = SessionState["feedback"]["comments"][number];

export interface ThreadControllerState {
  query: string;
  thread: ThreadEntry[];
  recentResearch: RecentResearchItem[];
  loading: boolean;
}

interface ThreadControllerInitialState {
  query?: string;
  thread: ThreadEntry[];
  recentResearch: RecentResearchItem[];
  loading?: boolean;
}

type ThreadControllerAction =
  | { type: "query/set"; query: string }
  | { type: "loading/set"; loading: boolean }
  | { type: "search/reset" }
  | { type: "recentResearch/delete"; itemId: string }
  | {
      type: "message/submitStarted";
      pendingId: string;
      userEntryId: string;
      query: string;
      fresh: boolean;
      createdAt: string;
    }
  | { type: "query/startFailed"; pendingId?: string }
  | {
      type: "message/runStarted";
      pendingId: string;
      messageRun: MessageRunState;
    }
  | {
      type: "message/chatCompleted";
      pendingId: string;
      turn: AssistantTurnState;
    }
  | {
      type: "message/searchStarted";
      pendingId: string;
      userEntryId: string;
      messageRun: MessageRunState;
      session: SessionState;
    }
  | {
      type: "thread/pendingMessageRunUpdated";
      pendingId: string;
      messageRun: MessageRunState;
    }
  | {
      type: "thread/pendingMessageRunFailed";
      pendingId: string;
      messageRun: MessageRunState;
    }
  | {
      type: "thread/pendingSessionUpdated";
      pendingId: string;
      session: SessionState;
    }
  | {
      type: "thread/pendingSessionCompleted";
      pendingId: string;
      session: SessionState;
    }
  | {
      type: "thread/pendingSessionFailed";
      pendingId: string;
      session: SessionState;
    }
  | { type: "thread/interactiveSessionReplaced"; session: SessionState }
  | {
      type: "thread/assistantSessionReplaced";
      entryId: string;
      session: SessionState;
    }
  | {
      type: "thread/refinementCompleted";
      session: SessionState;
      userQuery: string;
      createdAt: string;
    }
  | {
      type: "thread/localCommentAdded";
      entryId: string;
      comment: SessionComment;
    }
  | {
      type: "thread/localCommentUpdated";
      entryId: string;
      commentId: string;
      commentText: string;
    }
  | {
      type: "thread/localCommentDeleted";
      entryId: string;
      commentId: string;
    };

export function createThreadControllerState(
  initialState: ThreadControllerInitialState,
): ThreadControllerState {
  return {
    query: initialState.query ?? "",
    thread: initialState.thread,
    recentResearch: initialState.recentResearch,
    loading: initialState.loading ?? false,
  };
}

function archiveResearchAssistants(thread: ThreadEntry[]) {
  return thread.map((entry) =>
    isResearchAssistantEntry(entry) ? { ...entry, interactive: false } : entry,
  );
}

export function replaceInteractiveSessionInThread(
  thread: ThreadEntry[],
  nextSession: SessionState,
) {
  const next = [...thread];
  for (let index = next.length - 1; index >= 0; index -= 1) {
    const entry = next[index];
    if (isResearchAssistantEntry(entry) && entry.interactive) {
      next[index] = { ...entry, session: nextSession };
      break;
    }
  }
  return next;
}

export function appendChatTurnToThread(thread: ThreadEntry[], nextTurn: AssistantTurnState) {
  return [...thread, chatAssistantEntry("query", nextTurn)];
}

export function appendRefinementToThread(
  thread: ThreadEntry[],
  nextSession: SessionState,
  options: {
    userQuery: string;
    createdAt: string;
  },
) {
  return [
    ...archiveResearchAssistants(thread),
    userEntry(options.userQuery, {
      createdAt: options.createdAt,
      synthetic: true,
      researchBacked: true,
    }),
    assistantEntry("refinement", nextSession, true),
  ];
}

export function attachMessageRunToPendingThread(
  thread: ThreadEntry[],
  pendingId: string,
  nextMessageRun: MessageRunState,
) {
  return thread.map((entry) =>
    isPendingAssistantEntry(entry) && entry.id === pendingId
      ? { ...entry, messageRun: nextMessageRun }
      : entry,
  );
}

export function attachPendingSessionToThread(
  thread: ThreadEntry[],
  pendingId: string,
  userEntryId: string,
  nextMessageRun: MessageRunState,
  nextSession: SessionState,
) {
  return archiveResearchAssistants(
    thread.map((entry) => {
      if (entry.kind === "user" && entry.id === userEntryId) {
        return { ...entry, researchBacked: true };
      }
      if (isPendingAssistantEntry(entry) && entry.id === pendingId) {
        return { ...entry, messageRun: nextMessageRun, session: nextSession };
      }
      return entry;
    }),
  );
}

export function updatePendingSessionInThread(
  thread: ThreadEntry[],
  pendingId: string,
  nextSession: SessionState,
) {
  return thread.map((entry) =>
    isPendingAssistantEntry(entry) && entry.id === pendingId
      ? { ...entry, session: nextSession }
      : entry,
  );
}

export function completePendingSessionInThread(
  thread: ThreadEntry[],
  pendingId: string,
  nextSession: SessionState,
) {
  return normalizeThread(
    thread.map((entry) =>
      isPendingAssistantEntry(entry) && entry.id === pendingId
        ? assistantEntry("query", nextSession, true)
        : entry,
    ),
  );
}

export function completePendingChatTurnInThread(
  thread: ThreadEntry[],
  pendingId: string,
  nextTurn: AssistantTurnState,
) {
  return normalizeThread(
    thread.map((entry) =>
      isPendingAssistantEntry(entry) && entry.id === pendingId
        ? chatAssistantEntry("query", nextTurn)
        : entry,
    ),
  );
}

export function failPendingSessionInThread(
  thread: ThreadEntry[],
  pendingId: string,
  nextSession: SessionState,
) {
  return thread.map((entry) =>
    isPendingAssistantEntry(entry) && entry.id === pendingId
      ? { ...entry, session: nextSession }
      : entry,
  );
}

export function failPendingMessageRunInThread(
  thread: ThreadEntry[],
  pendingId: string,
  nextMessageRun: MessageRunState,
) {
  return thread.map((entry) =>
    isPendingAssistantEntry(entry) && entry.id === pendingId
      ? { ...entry, messageRun: nextMessageRun }
      : entry,
  );
}

export function replaceAssistantEntrySessionInThread(
  thread: ThreadEntry[],
  entryId: string,
  nextSession: SessionState,
) {
  return thread.map((entry) =>
    isResearchAssistantEntry(entry) && entry.id === entryId
      ? { ...entry, session: nextSession }
      : entry,
  );
}

export function addLocalCommentToThread(
  thread: ThreadEntry[],
  entryId: string,
  comment: SessionComment,
) {
  return thread.map((entry) => {
    if (!isResearchAssistantEntry(entry) || entry.id !== entryId) {
      return entry;
    }
    return {
      ...entry,
      session: {
        ...entry.session,
        feedback: {
          ...entry.session.feedback,
          comments: [...entry.session.feedback.comments, comment],
        },
      },
    };
  });
}

export function updateThreadComment(
  thread: ThreadEntry[],
  entryId: string,
  commentId: string,
  commentText: string,
) {
  return thread.map((entry) => {
    if (!isResearchAssistantEntry(entry) || entry.id !== entryId) {
      return entry;
    }
    return {
      ...entry,
      session: {
        ...entry.session,
        feedback: {
          ...entry.session.feedback,
          comments: entry.session.feedback.comments.map((comment) =>
            comment.comment_id === commentId
              ? { ...comment, comment_text: commentText }
              : comment,
          ),
        },
      },
    };
  });
}

export function deleteThreadComment(
  thread: ThreadEntry[],
  entryId: string,
  commentId: string,
) {
  return thread.map((entry) => {
    if (!isResearchAssistantEntry(entry) || entry.id !== entryId) {
      return entry;
    }
    return {
      ...entry,
      session: {
        ...entry.session,
        feedback: {
          ...entry.session.feedback,
          comments: entry.session.feedback.comments.filter(
            (comment) => comment.comment_id !== commentId,
          ),
        },
      },
    };
  });
}

export function threadControllerReducer(
  state: ThreadControllerState,
  action: ThreadControllerAction,
): ThreadControllerState {
  switch (action.type) {
    case "query/set":
      return { ...state, query: action.query };
    case "loading/set":
      return { ...state, loading: action.loading };
    case "search/reset":
      return {
        ...state,
        thread: [],
        query: "",
        loading: false,
      };
    case "recentResearch/delete":
      return {
        ...state,
        recentResearch: state.recentResearch.filter(
          (item) => item.id !== action.itemId,
        ),
      };
    case "message/submitStarted":
      return {
        ...state,
        loading: true,
        query: "",
        thread: [
          ...archiveResearchAssistants(action.fresh ? [] : state.thread),
          {
            id: action.userEntryId,
            kind: "user",
            query: action.query,
            createdAt: action.createdAt,
            researchBacked: false,
          },
          {
            id: action.pendingId,
            kind: "pending-assistant",
            userEntryId: action.userEntryId,
            messageRun: null,
            session: null,
          },
        ],
      };
    case "query/startFailed":
      return {
        ...state,
        loading: false,
        thread:
          action.pendingId == null
            ? state.thread
            : state.thread.filter(
                (entry) =>
                  !(
                    isPendingAssistantEntry(entry) &&
                    entry.id === action.pendingId
                  ),
              ),
      };
    case "message/runStarted":
      return {
        ...state,
        thread: attachMessageRunToPendingThread(
          state.thread,
          action.pendingId,
          action.messageRun,
        ),
      };
    case "message/chatCompleted":
      return {
        ...state,
        loading: false,
        thread: completePendingChatTurnInThread(
          state.thread,
          action.pendingId,
          action.turn,
        ),
      };
    case "message/searchStarted":
      const nextThread = attachPendingSessionToThread(
        state.thread,
        action.pendingId,
        action.userEntryId,
        action.messageRun,
        action.session,
      );
      return {
        ...state,
        recentResearch: updateRecentResearch(state.recentResearch, nextThread),
        thread: nextThread,
      };
    case "thread/pendingMessageRunUpdated":
      return {
        ...state,
        thread: attachMessageRunToPendingThread(
          state.thread,
          action.pendingId,
          action.messageRun,
        ),
      };
    case "thread/pendingMessageRunFailed":
      return {
        ...state,
        loading: false,
        thread: failPendingMessageRunInThread(
          state.thread,
          action.pendingId,
          action.messageRun,
        ),
      };
    case "thread/pendingSessionUpdated":
      return {
        ...state,
        thread: updatePendingSessionInThread(
          state.thread,
          action.pendingId,
          action.session,
        ),
      };
    case "thread/pendingSessionCompleted":
      return {
        ...state,
        thread: completePendingSessionInThread(
          state.thread,
          action.pendingId,
          action.session,
        ),
      };
    case "thread/pendingSessionFailed":
      return {
        ...state,
        thread: failPendingSessionInThread(
          state.thread,
          action.pendingId,
          action.session,
        ),
      };
    case "thread/interactiveSessionReplaced":
      return {
        ...state,
        thread: replaceInteractiveSessionInThread(state.thread, action.session),
      };
    case "thread/assistantSessionReplaced":
      return {
        ...state,
        thread: replaceAssistantEntrySessionInThread(
          state.thread,
          action.entryId,
          action.session,
        ),
      };
    case "thread/refinementCompleted":
      return {
        ...state,
        thread: appendRefinementToThread(state.thread, action.session, {
          userQuery: action.userQuery,
          createdAt: action.createdAt,
        }),
      };
    case "thread/localCommentAdded":
      return {
        ...state,
        thread: addLocalCommentToThread(
          state.thread,
          action.entryId,
          action.comment,
        ),
      };
    case "thread/localCommentUpdated":
      return {
        ...state,
        thread: updateThreadComment(
          state.thread,
          action.entryId,
          action.commentId,
          action.commentText,
        ),
      };
    case "thread/localCommentDeleted":
      return {
        ...state,
        thread: deleteThreadComment(
          state.thread,
          action.entryId,
          action.commentId,
        ),
      };
    default:
      return state;
  }
}
