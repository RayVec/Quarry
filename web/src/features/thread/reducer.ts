import type { AssistantMessageSource, SessionState } from "@/types";
import {
  assistantEntry,
  isAssistantEntry,
  isPendingAssistantEntry,
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
      type: "query/submitStarted";
      pendingId: string;
      userEntryId: string;
      query: string;
      fresh: boolean;
      createdAt: string;
    }
  | { type: "query/startFailed"; pendingId: string }
  | {
      type: "thread/pendingUpdated";
      pendingId: string;
      session: SessionState;
    }
  | {
      type: "thread/pendingCompleted";
      pendingId: string;
      session: SessionState;
    }
  | {
      type: "thread/pendingFailed";
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
      type: "thread/assistantAppended";
      source: AssistantMessageSource;
      session: SessionState;
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

export function replaceInteractiveSessionInThread(
  thread: ThreadEntry[],
  nextSession: SessionState,
) {
  const next = [...thread];
  for (let index = next.length - 1; index >= 0; index -= 1) {
    const entry = next[index];
    if (isAssistantEntry(entry) && entry.interactive) {
      next[index] = { ...entry, session: nextSession };
      break;
    }
  }
  return next;
}

export function appendAssistantSessionToThread(
  thread: ThreadEntry[],
  nextSession: SessionState,
  source: AssistantMessageSource,
) {
  return [
    ...thread.map((entry) =>
      isAssistantEntry(entry) ? { ...entry, interactive: false } : entry,
    ),
    assistantEntry(source, nextSession, true),
  ];
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
  return thread.map((entry) => {
    if (isPendingAssistantEntry(entry) && entry.id === pendingId) {
      return assistantEntry("query", nextSession, true);
    }
    if (isAssistantEntry(entry)) {
      return { ...entry, interactive: false };
    }
    return entry;
  });
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

export function replaceAssistantEntrySessionInThread(
  thread: ThreadEntry[],
  entryId: string,
  nextSession: SessionState,
) {
  return thread.map((entry) =>
    isAssistantEntry(entry) && entry.id === entryId
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
    if (!isAssistantEntry(entry) || entry.id !== entryId) {
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
    if (!isAssistantEntry(entry) || entry.id !== entryId) {
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
    if (!isAssistantEntry(entry) || entry.id !== entryId) {
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
    case "query/submitStarted":
      return {
        ...state,
        loading: true,
        query: "",
        recentResearch: updateRecentResearch(
          state.recentResearch,
          action.query,
        ),
        thread: [
          ...(action.fresh
            ? []
            : state.thread.map((entry) =>
                isAssistantEntry(entry)
                  ? { ...entry, interactive: false }
                  : entry,
              )),
          {
            id: action.userEntryId,
            kind: "user",
            query: action.query,
            createdAt: action.createdAt,
          },
          { id: action.pendingId, kind: "pending-assistant", session: null },
        ],
      };
    case "query/startFailed":
      return {
        ...state,
        loading: false,
        thread: state.thread.filter(
          (entry) =>
            !(isPendingAssistantEntry(entry) && entry.id === action.pendingId),
        ),
      };
    case "thread/pendingUpdated":
      return {
        ...state,
        thread: updatePendingSessionInThread(
          state.thread,
          action.pendingId,
          action.session,
        ),
      };
    case "thread/pendingCompleted":
      return {
        ...state,
        thread: completePendingSessionInThread(
          state.thread,
          action.pendingId,
          action.session,
        ),
      };
    case "thread/pendingFailed":
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
    case "thread/assistantAppended":
      return {
        ...state,
        thread: appendAssistantSessionToThread(
          state.thread,
          action.session,
          action.source,
        ),
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
