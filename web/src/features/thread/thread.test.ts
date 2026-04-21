import { describe, expect, it } from "vitest";
import {
  buildConversationContextTurns,
  chatAssistantEntry,
  deriveRecentResearchFromThread,
  findResumablePendingEntry,
  normalizeThread,
  resolveActiveCitationContext,
  type ActiveCitationContext,
  type PendingAssistantThreadEntry,
  type ResearchAssistantThreadEntry,
  type ThreadEntry,
} from "./model";
import {
  createThreadControllerState,
  addLocalCommentToThread,
  appendChatTurnToThread,
  attachPendingSessionToThread,
  appendRefinementToThread,
  completePendingSessionInThread,
  deleteThreadComment,
  threadControllerReducer,
  updateThreadComment,
} from "./reducer";
import {
  getLatestGroundedSessionId,
  getLatestResearchQuery,
  getThreadTitleQuery,
} from "./selectors";
import type {
  AssistantTurnState,
  CitationIndexEntry,
  MessageRunState,
  ParsedSentence,
  SessionState,
} from "@/types";

function makeCitation(
  citationId: number,
  overrides: Partial<CitationIndexEntry> = {},
): CitationIndexEntry {
  return {
    citation_id: citationId,
    chunk_id: `chunk-${citationId}`,
    text: `Citation ${citationId} text`,
    document_id: "doc-1",
    document_title: "Specification",
    section_heading: "Section",
    section_path: "Section",
    page_number: 1,
    retrieval_score: 0.9,
    source_facet: "schedule",
    source_facets: ["schedule"],
    replacement_pending: false,
    ambiguity_review_required: false,
    retrieval_scores: {},
    ...overrides,
  };
}

function makeSentence(
  sentenceIndex: number,
  citationId: number,
  referenceQuote = "quote",
  overrides: Partial<ParsedSentence> = {},
): ParsedSentence {
  return {
    sentence_index: sentenceIndex,
    sentence_text: "This is a sentence.",
    sentence_type: "claim",
    references: [
      {
        reference_quote: referenceQuote,
        verified: true,
        citation_id: citationId,
        replacement_pending: false,
        confidence_unknown: false,
      },
    ],
    status: "verified",
    match_quality: "strong",
    paragraph_index: 0,
    warnings: [],
    raw_text: "This is a sentence.",
    ...overrides,
  };
}

function makeSession(overrides: Partial<SessionState> = {}): SessionState {
  return {
    session_id: "session-1",
    original_query: "What is schedule risk?",
    source_message: "What is schedule risk?",
    resolved_query: "What is schedule risk?",
    query_type: "single_hop",
    facets: [],
    citation_index: [makeCitation(1)],
    generated_response: "Response",
    parsed_sentences: [makeSentence(0, 1)],
    feedback: {
      comments: [],
      resolved_comments: [],
      citation_replacements: [],
      citation_feedback: [],
    },
    refinement_count: 0,
    retrieval_diagnostics: [],
    ui_messages: [],
    removed_ungrounded_claim_count: 0,
    response_mode: "response_review",
    generation_provider: "stub",
    parser_provider: "stub",
    runtime_mode: "hosted",
    runtime_profile: "apple_silicon",
    local_model_status: {},
    active_model_ids: [],
    query_status: "completed",
    query_stage: "completed",
    query_stage_label: "Completed",
    query_stage_detail: "",
    ...overrides,
  };
}

function makeAssistantEntry(
  id: string,
  session: SessionState,
  interactive = true,
): ResearchAssistantThreadEntry {
  return {
    id,
    kind: "assistant",
    assistantKind: "research",
    source: "query",
    session,
    interactive,
  };
}

function makePendingEntry(
  id: string,
  session: SessionState | null,
  overrides: Partial<PendingAssistantThreadEntry> = {},
): PendingAssistantThreadEntry {
  return {
    id,
    kind: "pending-assistant",
    userEntryId: "user-1",
    messageRun: null,
    session,
    ...overrides,
  };
}

function makeMessageRun(
  overrides: Partial<MessageRunState> = {},
): MessageRunState {
  return {
    message_run_id: "run-1",
    status: "running",
    stage: "orchestrating",
    stage_label: "Deciding whether to search",
    stage_detail: "Deciding whether to search.",
    stage_catalog: [],
    assistant_turn: null,
    session: null,
    ...overrides,
  };
}

function makeChatTurn(
  overrides: Partial<AssistantTurnState> = {},
): AssistantTurnState {
  return {
    turn_id: "turn-1",
    content: "Understood.",
    used_search: false,
    response_basis: "social",
    derived_from_session_id: "session-a",
    ...overrides,
  };
}

describe("thread model and reducer", () => {
  it("keeps persisted running pending sessions resumable and archives prior assistants", () => {
    const persistedThread = normalizeThread([
      {
        id: "user-1",
        kind: "user",
        query: "How does procurement affect risk?",
        createdAt: "2026-04-17T12:00:00.000Z",
        researchBacked: true,
      },
      makeAssistantEntry("assistant-1", makeSession({ session_id: "session-a" })),
      makePendingEntry(
        "pending-1",
        makeSession({
          session_id: "session-pending",
          query_status: "running",
          query_stage: "writing",
          query_stage_label: "Writing",
        }),
      ),
    ]);

    expect(findResumablePendingEntry(persistedThread)?.id).toBe("pending-1");
    expect(
      persistedThread.filter(
        (entry) => entry.kind === "assistant" && entry.interactive,
      ),
    ).toHaveLength(0);
  });

  it("keeps persisted running orchestration runs resumable before a session exists", () => {
    const persistedThread = normalizeThread([
      {
        id: "user-1",
        kind: "user",
        query: "good",
        createdAt: "2026-04-17T12:00:00.000Z",
        researchBacked: false,
      },
      makePendingEntry("pending-1", null, {
        messageRun: makeMessageRun(),
      }),
    ]);

    expect(findResumablePendingEntry(persistedThread)).toMatchObject({
      id: "pending-1",
      messageRun: expect.objectContaining({ message_run_id: "run-1" }),
    });
  });

  it("keeps exactly one interactive research assistant when a pending query completes", () => {
    const previousAssistant = makeAssistantEntry(
      "assistant-1",
      makeSession({ session_id: "session-a" }),
    );
    const thread: ThreadEntry[] = [
      {
        id: "user-1",
        kind: "user",
        query: "How does procurement affect risk?",
        createdAt: "2026-04-17T12:00:00.000Z",
        researchBacked: true,
      },
      previousAssistant,
      makePendingEntry(
        "pending-1",
        makeSession({
          session_id: "session-pending",
          query_status: "running",
          query_stage: "writing",
          query_stage_label: "Writing",
        }),
      ),
    ];

    const completedThread = completePendingSessionInThread(
      thread,
      "pending-1",
      makeSession({ session_id: "session-completed" }),
    );

    const interactiveAssistants = completedThread.filter(
      (entry) => entry.kind === "assistant" && entry.interactive,
    );

    expect(interactiveAssistants).toHaveLength(1);
    expect(interactiveAssistants[0]).toMatchObject({
      kind: "assistant",
      interactive: true,
      session: expect.objectContaining({ session_id: "session-completed" }),
    });
    expect(
      completedThread.find(
        (entry) => entry.kind === "assistant" && entry.id === "assistant-1",
      ),
    ).toMatchObject({ interactive: false });
  });

  it("keeps the latest grounded research answer interactive when a chat turn is appended", () => {
    const thread: ThreadEntry[] = [
      {
        id: "user-1",
        kind: "user",
        query: "How does procurement affect risk?",
        createdAt: "2026-04-17T12:00:00.000Z",
        researchBacked: true,
      },
      makeAssistantEntry("assistant-1", makeSession({ session_id: "session-a" })),
    ];

    const withChat = appendChatTurnToThread(thread, makeChatTurn());

    expect(withChat.at(-1)).toMatchObject({
      kind: "assistant",
      assistantKind: "chat",
      turn: expect.objectContaining({ content: "Understood." }),
    });
    expect(
      withChat.find(
        (entry) =>
          entry.kind === "assistant" &&
          entry.assistantKind === "research" &&
          entry.id === "assistant-1",
      ),
    ).toMatchObject({ interactive: true });
  });

  it("supports local comment add, update, and delete fallbacks without backend data", () => {
    const thread: ThreadEntry[] = [
      makeAssistantEntry("assistant-1", makeSession({ session_id: "session-a" })),
    ];
    const comment = {
      comment_id: "comment-1",
      text_selection: "selection",
      char_start: 0,
      char_end: 9,
      comment_text: "Initial comment",
      resolved: false,
    };

    const withComment = addLocalCommentToThread(thread, "assistant-1", comment);
    expect(
      withComment[0].kind === "assistant" &&
        withComment[0].assistantKind === "research"
        ? withComment[0].session.feedback.comments
        : [],
    ).toHaveLength(1);

    const updatedComment = updateThreadComment(
      withComment,
      "assistant-1",
      "comment-1",
      "Updated comment",
    );
    expect(
      updatedComment[0].kind === "assistant" &&
        updatedComment[0].assistantKind === "research"
        ? updatedComment[0].session.feedback.comments[0]?.comment_text
        : null,
    ).toBe("Updated comment");

    const withoutComment = deleteThreadComment(
      updatedComment,
      "assistant-1",
      "comment-1",
    );
    expect(
      withoutComment[0].kind === "assistant" &&
        withoutComment[0].assistantKind === "research"
        ? withoutComment[0].session.feedback.comments
        : [],
    ).toHaveLength(0);
  });

  it("re-resolves active citation context when a citation id changes after review", () => {
    const currentSession = makeSession({
      session_id: "session-a",
      citation_index: [makeCitation(1)],
      parsed_sentences: [makeSentence(0, 1, "same quote")],
    });
    const activeCitation: ActiveCitationContext = {
      entryId: "assistant-1",
      session: currentSession,
      sentenceIndex: 0,
      referenceQuote: "same quote",
      citation: currentSession.citation_index[0],
      readOnly: false,
    };

    const nextSession = makeSession({
      session_id: "session-a",
      citation_index: [makeCitation(2)],
      parsed_sentences: [makeSentence(0, 2, "same quote")],
    });

    expect(resolveActiveCitationContext(activeCitation, nextSession)).toMatchObject({
      session: nextSession,
      citation: expect.objectContaining({ citation_id: 2 }),
    });
  });

  it("appends refinement as a synthetic user turn without replacing the latest real research query", () => {
    const thread: ThreadEntry[] = [
      {
        id: "user-1",
        kind: "user",
        query: "How does procurement affect risk?",
        createdAt: "2026-04-17T12:00:00.000Z",
        researchBacked: true,
      },
      makeAssistantEntry("assistant-1", makeSession({ session_id: "session-a" })),
    ];

    const refinedThread = appendRefinementToThread(
      thread,
      makeSession({
        session_id: "session-b",
        derived_from_session_id: "session-a",
      }),
      {
        userQuery: "Please refine the previous answer.",
        createdAt: "2026-04-17T12:05:00.000Z",
      },
    );

    expect(refinedThread.at(-2)).toMatchObject({
      kind: "user",
      query: "Please refine the previous answer.",
      synthetic: true,
    });
    expect(getLatestResearchQuery(refinedThread)).toBe(
      "How does procurement affect risk?",
    );
  });

  it("derives recent research only from search-backed user turns", () => {
    const thread: ThreadEntry[] = [
      {
        id: "user-1",
        kind: "user",
        query: "What is FEED maturity?",
        createdAt: "2026-04-17T12:00:00.000Z",
        researchBacked: true,
      },
      makeAssistantEntry("assistant-1", makeSession({ session_id: "session-a" })),
      {
        id: "user-2",
        kind: "user",
        query: "good",
        createdAt: "2026-04-17T12:02:00.000Z",
        researchBacked: false,
      },
      chatAssistantEntry("query", makeChatTurn()),
    ];

    expect(deriveRecentResearchFromThread(thread)).toEqual([
      {
        id: "user-1",
        query: "What is FEED maturity?",
        createdAt: "2026-04-17T12:00:00.000Z",
      },
    ]);
  });

  it("uses the first real user query as the thread title and recent research title", () => {
    const thread: ThreadEntry[] = [
      {
        id: "user-1",
        kind: "user",
        query: "What is FEED maturity?",
        createdAt: "2026-04-17T12:00:00.000Z",
        researchBacked: false,
      },
      chatAssistantEntry("query", makeChatTurn()),
      {
        id: "user-2",
        kind: "user",
        query: "Explain more about FEED maturity",
        createdAt: "2026-04-17T12:05:00.000Z",
        researchBacked: true,
      },
      makeAssistantEntry("assistant-1", makeSession({ session_id: "session-a" })),
    ];

    expect(getThreadTitleQuery(thread)).toBe("What is FEED maturity?");
    expect(deriveRecentResearchFromThread(thread)).toEqual([
      {
        id: "user-1",
        query: "What is FEED maturity?",
        createdAt: "2026-04-17T12:00:00.000Z",
      },
    ]);
  });

  it("keeps a single recent research item when follow-up research happens in the same thread", () => {
    const state = createThreadControllerState({
      thread: [
        {
          id: "user-1",
          kind: "user",
          query: "What is FEED maturity?",
          createdAt: "2026-04-17T12:00:00.000Z",
          researchBacked: true,
        },
        makeAssistantEntry("assistant-1", makeSession({ session_id: "session-a" })),
        {
          id: "user-2",
          kind: "user",
          query: "Explain more about FEED maturity",
          createdAt: "2026-04-17T12:05:00.000Z",
          researchBacked: false,
        },
        makePendingEntry("pending-1", null, {
          userEntryId: "user-2",
          messageRun: makeMessageRun(),
        }),
      ],
      recentResearch: [
        {
          id: "user-1",
          query: "What is FEED maturity?",
          createdAt: "2026-04-17T12:00:00.000Z",
        },
      ],
    });

    const nextState = threadControllerReducer(state, {
      type: "message/searchStarted",
      pendingId: "pending-1",
      userEntryId: "user-2",
      messageRun: makeMessageRun({
        session: makeSession({
          session_id: "session-b",
          query_status: "running",
          query_stage: "writing",
          query_stage_label: "Writing",
        }),
      }),
      session: makeSession({
        session_id: "session-b",
        query_status: "running",
        query_stage: "writing",
        query_stage_label: "Writing",
      }),
    });

    expect(nextState.recentResearch).toEqual([
      {
        id: "user-1",
        query: "What is FEED maturity?",
        createdAt: "2026-04-17T12:00:00.000Z",
      },
    ]);
  });

  it("builds conversation context turns with both research and chat messages", () => {
    const thread: ThreadEntry[] = [
      {
        id: "user-1",
        kind: "user",
        query: "What is FEED maturity?",
        createdAt: "2026-04-17T12:00:00.000Z",
        researchBacked: true,
      },
      makeAssistantEntry(
        "assistant-1",
        makeSession({
          session_id: "session-a",
          generated_response: "Grounded answer.",
        }),
      ),
      {
        id: "user-2",
        kind: "user",
        query: "good",
        createdAt: "2026-04-17T12:02:00.000Z",
        researchBacked: false,
      },
      chatAssistantEntry("query", makeChatTurn()),
    ];

    expect(buildConversationContextTurns(thread)).toEqual([
      {
        role: "user",
        text: "What is FEED maturity?",
        search_backed: true,
        derived_from_session_id: undefined,
      },
      {
        role: "assistant",
        text: "Grounded answer.",
        search_backed: true,
        session_id: "session-a",
        derived_from_session_id: null,
      },
      {
        role: "user",
        text: "good",
        search_backed: false,
        derived_from_session_id: undefined,
      },
      {
        role: "assistant",
        text: "Understood.",
        search_backed: false,
        session_id: null,
        derived_from_session_id: "session-a",
      },
    ]);
  });

  it("uses the latest derived grounded session id when the latest turn is chat", () => {
    const thread: ThreadEntry[] = [
      {
        id: "user-1",
        kind: "user",
        query: "What is FEED maturity?",
        createdAt: "2026-04-17T12:00:00.000Z",
        researchBacked: true,
      },
      makeAssistantEntry("assistant-1", makeSession({ session_id: "session-a" })),
      {
        id: "user-2",
        kind: "user",
        query: "good",
        createdAt: "2026-04-17T12:02:00.000Z",
        researchBacked: false,
      },
      chatAssistantEntry("query", makeChatTurn()),
    ];

    expect(getLatestGroundedSessionId(thread)).toBe("session-a");
  });

  it("marks the triggering user turn as research-backed when search starts", () => {
    const thread: ThreadEntry[] = [
      {
        id: "user-1",
        kind: "user",
        query: "Explain FEED maturity",
        createdAt: "2026-04-17T12:00:00.000Z",
        researchBacked: false,
      },
    ];

    const nextThread = attachPendingSessionToThread(
      [
        ...thread,
        makePendingEntry("pending-1", null),
      ],
      "pending-1",
      "user-1",
      makeMessageRun(),
      makeSession({
        session_id: "session-search",
        query_status: "running",
        query_stage: "understanding",
      }),
    );

    expect(nextThread[0]).toMatchObject({
      kind: "user",
      researchBacked: true,
    });
    expect(nextThread.at(-1)).toMatchObject({
      kind: "pending-assistant",
      messageRun: expect.objectContaining({ message_run_id: "run-1" }),
      session: expect.objectContaining({ session_id: "session-search" }),
    });
  });

  it("replaces a pending orchestration entry with a chat turn when direct reply completes", () => {
    const state = createThreadControllerState({
      thread: [
        {
          id: "user-1",
          kind: "user",
          query: "good",
          createdAt: "2026-04-17T12:00:00.000Z",
          researchBacked: false,
        },
        makePendingEntry("pending-1", null, {
          messageRun: makeMessageRun(),
        }),
      ],
      recentResearch: [],
    });

    const nextState = threadControllerReducer(state, {
      type: "message/chatCompleted",
      pendingId: "pending-1",
      turn: makeChatTurn({ derived_from_session_id: null }),
    });

    expect(nextState.thread.at(-1)).toMatchObject({
      kind: "assistant",
      assistantKind: "chat",
      turn: expect.objectContaining({ content: "Understood." }),
    });
    expect(
      nextState.thread.some((entry) => entry.kind === "pending-assistant"),
    ).toBe(false);
  });
});
