import { describe, expect, it } from "vitest";
import { resolveActiveCitationContext, normalizeThread, findResumablePendingEntry } from "./model";
import {
  addLocalCommentToThread,
  completePendingSessionInThread,
  deleteThreadComment,
  updateThreadComment,
} from "./reducer";
import type {
  ActiveCitationContext,
  AssistantThreadEntry,
  PendingAssistantThreadEntry,
  ThreadEntry,
} from "./model";
import type {
  CitationIndexEntry,
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
): AssistantThreadEntry {
  return {
    id,
    kind: "assistant",
    source: "query",
    session,
    interactive,
  };
}

function makePendingEntry(
  id: string,
  session: SessionState,
): PendingAssistantThreadEntry {
  return {
    id,
    kind: "pending-assistant",
    session,
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

  it("keeps exactly one interactive assistant when a pending query completes", () => {
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
      withComment[0].kind === "assistant"
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
      updatedComment[0].kind === "assistant"
        ? updatedComment[0].session.feedback.comments[0]?.comment_text
        : null,
    ).toBe("Updated comment");

    const withoutComment = deleteThreadComment(
      updatedComment,
      "assistant-1",
      "comment-1",
    );
    expect(
      withoutComment[0].kind === "assistant"
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
});
