import { createContext, useContext } from "react";
import type { ReactNode } from "react";
import type { ParsedSentence, SessionState } from "../types";

type CommentPayload = {
  text_selection: string;
  char_start: number;
  char_end: number;
  comment_text: string;
};

export interface ThreadActions {
  openCitation: (
    entryId: string,
    session: SessionState,
    sentence: ParsedSentence,
    citationId: number,
    referenceQuote: string,
    readOnly: boolean,
  ) => void;
  saveComment: (entryId: string, sessionId: string, payload: CommentPayload) => Promise<void>;
  updateComment: (entryId: string, sessionId: string, commentId: string, commentText: string) => Promise<void>;
  deleteComment: (entryId: string, sessionId: string, commentId: string) => Promise<void>;
  refine: (sessionId: string) => Promise<void>;
}

const ThreadActionsContext = createContext<ThreadActions | null>(null);

export function ThreadActionsProvider({
  value,
  children,
}: {
  value: ThreadActions;
  children: ReactNode;
}) {
  return <ThreadActionsContext.Provider value={value}>{children}</ThreadActionsContext.Provider>;
}

export function useThreadActions() {
  const context = useContext(ThreadActionsContext);
  if (!context) {
    throw new Error("useThreadActions must be used within ThreadActionsProvider");
  }
  return context;
}
