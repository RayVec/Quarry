from __future__ import annotations

import asyncio

from quarry.domain.models import (
    ConversationAction,
    MessageRequest,
    QueryProgressStage,
    QueryRequest,
    ResponseBasis,
    ResponseMode,
    SessionState,
)
from quarry.services.message_service import MessageService
from quarry.services.message_run_store import MessageRunStore
from quarry.services.session_store import SessionNotFoundError


class StubPipelineService:
    def __init__(self, sessions: dict[str, SessionState] | None = None) -> None:
        self.sessions = sessions or {}
        self.begin_requests: list[QueryRequest] = []

    def get_session(self, session_id: str) -> SessionState:
        try:
            return self.sessions[session_id]
        except KeyError as exc:
            raise SessionNotFoundError(session_id) from exc

    def begin_query(self, request: QueryRequest) -> SessionState:
        self.begin_requests.append(request)
        session = SessionState(
            session_id="session-search",
            original_query=request.source_message or request.query,
            source_message=request.source_message or request.query,
            resolved_query=request.query,
            query_status="running",
            query_stage=QueryProgressStage.UNDERSTANDING,
            query_stage_label="Reading your question",
            query_stage_detail="I'm getting clear on what you want to know.",
        )
        self.sessions[session.session_id] = session
        return session


class StubLLM:
    def __init__(self, response: str | None = None, *, error: Exception | None = None) -> None:
        self.response = response or ""
        self.error = error
        self.prompts: list[str] = []

    async def complete(self, prompt: str, *, temperature: float = 0.1, operation: str = "completion") -> str:
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        return self.response


def make_grounded_session(session_id: str = "session-1") -> SessionState:
    return SessionState(
        session_id=session_id,
        original_query="What is FEED maturity?",
        source_message="What is FEED maturity?",
        resolved_query="What is FEED maturity?",
        generated_response="FEED maturity describes how complete the FEED deliverables are before detailed design.",
        response_mode=ResponseMode.RESPONSE_REVIEW,
        query_status="completed",
        query_stage="completed",
        query_stage_label="Completed",
        query_stage_detail="",
    )


def test_message_service_defaults_to_search_when_hosted_orchestration_is_unavailable() -> None:
    service = MessageService(
        pipeline_service=StubPipelineService(),
        orchestration_llm=None,
        message_run_store=MessageRunStore(),
    )

    decision = asyncio.run(service.decide(MessageRequest(message="good")))

    assert decision.action == ConversationAction.SEARCH
    assert decision.search_query == "good"
    assert decision.response_basis == ResponseBasis.CORPUS_SEARCH


def test_message_service_returns_direct_social_reply_when_hosted_planner_says_respond() -> None:
    llm = StubLLM(
        '{"action":"respond","response_basis":"social","assistant_text":"Understood.","search_query":"","derived_from_session_id":""}'
    )
    service = MessageService(
        pipeline_service=StubPipelineService(),
        orchestration_llm=llm,
        message_run_store=MessageRunStore(),
    )

    decision = asyncio.run(service.decide(MessageRequest(message="good")))

    assert decision.action == ConversationAction.RESPOND
    assert decision.assistant_text == "Understood."
    assert decision.response_basis == ResponseBasis.SOCIAL


def test_message_service_rejects_thread_context_only_reply_without_grounded_anchor() -> None:
    llm = StubLLM(
        '{"action":"respond","response_basis":"thread_context_only","assistant_text":"Here is a quick restatement.","search_query":"","derived_from_session_id":""}'
    )
    service = MessageService(
        pipeline_service=StubPipelineService(),
        orchestration_llm=llm,
        message_run_store=MessageRunStore(),
    )

    decision = asyncio.run(service.decide(MessageRequest(message="say more")))

    assert decision.action == ConversationAction.SEARCH
    assert decision.search_query == "say more"


def test_message_service_keeps_thread_context_reply_when_grounded_anchor_exists() -> None:
    grounded = make_grounded_session()
    llm = StubLLM(
        '{"action":"respond","response_basis":"thread_context_only","assistant_text":"In other words, it measures how complete the FEED package is before detailed design.","search_query":"","derived_from_session_id":""}'
    )
    service = MessageService(
        pipeline_service=StubPipelineService({grounded.session_id: grounded}),
        orchestration_llm=llm,
        message_run_store=MessageRunStore(),
    )

    decision = asyncio.run(
        service.decide(
            MessageRequest(
                message="say more",
                latest_grounded_session_id=grounded.session_id,
            )
        )
    )

    assert decision.action == ConversationAction.RESPOND
    assert decision.response_basis == ResponseBasis.THREAD_CONTEXT_ONLY
    assert decision.derived_from_session_id == grounded.session_id


def test_message_service_falls_back_to_search_on_invalid_planner_output() -> None:
    llm = StubLLM("not json")
    grounded = make_grounded_session()
    service = MessageService(
        pipeline_service=StubPipelineService({grounded.session_id: grounded}),
        orchestration_llm=llm,
        message_run_store=MessageRunStore(),
    )

    decision = asyncio.run(
        service.decide(
            MessageRequest(
                message="why?",
                latest_grounded_session_id=grounded.session_id,
            )
        )
    )

    assert decision.action == ConversationAction.SEARCH
    assert decision.search_query == "why?"
    assert decision.derived_from_session_id == grounded.session_id


def test_message_service_begins_message_run_at_orchestration_stage() -> None:
    service = MessageService(
        pipeline_service=StubPipelineService(),
        orchestration_llm=None,
        message_run_store=MessageRunStore(),
    )

    message_run = service.begin_message_run()

    assert message_run.status == "running"
    assert message_run.stage == "orchestrating"
    assert message_run.stage_label == "Deciding whether to search"


def test_message_service_attaches_search_session_when_orchestration_routes_to_search() -> None:
    pipeline_service = StubPipelineService()
    service = MessageService(
        pipeline_service=pipeline_service,
        orchestration_llm=None,
        message_run_store=MessageRunStore(),
    )
    message_run = service.begin_message_run()

    next_message_run, query_payload = asyncio.run(
        service.run_message_for_run(
            message_run.message_run_id,
            MessageRequest(message="say more"),
        )
    )

    assert query_payload is not None
    assert query_payload.query == "say more"
    assert next_message_run.session is not None
    assert next_message_run.session.session_id == "session-search"
    assert pipeline_service.begin_requests[0].query == "say more"
