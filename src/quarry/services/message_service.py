from __future__ import annotations

from uuid import uuid4

from pydantic import ValidationError

from quarry.adapters.production import CompletionLLM
from quarry.domain.models import (
    AssistantTurnState,
    ConversationDecision,
    ConversationAction,
    MessageRequest,
    MessageRunState,
    QueryProgressStage,
    QueryRequest,
    QueryRunStatus,
    ResponseBasis,
    ResponseMode,
    SessionState,
    resolve_query_stage_descriptor,
)
from quarry.logging_utils import logger_with_trace
from quarry.prompts import message_orchestration_prompt, parse_json_response
from quarry.services.message_run_store import MessageRunStore
from quarry.services.pipeline_service import PipelineService
from quarry.services.session_store import SessionNotFoundError


logger = logger_with_trace(__name__)


class MessageService:
    def __init__(
        self,
        *,
        pipeline_service: PipelineService,
        orchestration_llm: CompletionLLM | None,
        message_run_store: MessageRunStore,
    ) -> None:
        self.pipeline_service = pipeline_service
        self.orchestration_llm = orchestration_llm
        self.message_run_store = message_run_store

    def begin_message_run(self) -> MessageRunState:
        descriptor = resolve_query_stage_descriptor(QueryProgressStage.ORCHESTRATING)
        return self.message_run_store.save(
            MessageRunState(
                message_run_id=str(uuid4()),
                stage=QueryProgressStage.ORCHESTRATING,
                stage_label=descriptor.label if descriptor is not None else "Deciding whether to search",
                stage_detail=descriptor.detail
                if descriptor is not None
                else "I'm deciding whether this needs report search or a direct response.",
            )
        )

    def get_message_run(self, message_run_id: str) -> MessageRunState:
        return self.message_run_store.get(message_run_id)

    def fail_message_run(self, message_run_id: str) -> MessageRunState:
        message_run = self.message_run_store.get(message_run_id)
        message_run.status = QueryRunStatus.FAILED
        message_run.stage = QueryProgressStage.ORCHESTRATING
        message_run.stage_label = "I hit a problem"
        message_run.stage_detail = "I couldn't get started on this request."
        return self.message_run_store.save(message_run)

    async def run_message_for_run(
        self,
        message_run_id: str,
        request: MessageRequest,
    ) -> tuple[MessageRunState, QueryRequest | None]:
        decision = await self.decide(request)
        message_run = self.message_run_store.get(message_run_id)

        if decision.action == ConversationAction.RESPOND:
            message_run.status = QueryRunStatus.COMPLETED
            message_run.assistant_turn = self.build_assistant_turn(decision)
            message_run.stage = QueryProgressStage.ORCHESTRATING
            message_run.stage_label = "Answer ready"
            message_run.stage_detail = "I can answer this directly without searching the reports."
            return self.message_run_store.save(message_run), None

        query_payload = QueryRequest(
            query=decision.search_query or request.message,
            source_message=request.message,
            derived_from_session_id=decision.derived_from_session_id,
        )
        session = self.pipeline_service.begin_query(query_payload)
        message_run.status = QueryRunStatus.COMPLETED
        message_run.stage = session.query_stage
        message_run.stage_label = session.query_stage_label
        message_run.stage_detail = session.query_stage_detail
        message_run.session = session
        return self.message_run_store.save(message_run), query_payload

    async def decide(self, request: MessageRequest) -> ConversationDecision:
        message = request.message.strip()
        latest_grounded_session = self._resolve_grounded_session(request.latest_grounded_session_id)

        if self.orchestration_llm is None:
            logger.info(
                "hosted orchestration unavailable; defaulting to search",
                extra={"message_preview": message[:200], "console_visible": True},
            )
            return self._fallback_search_decision(message, latest_grounded_session)

        try:
            raw = await self.orchestration_llm.complete(
                message_orchestration_prompt(
                    message=message,
                    context_turns=request.context_turns,
                    latest_grounded_session=latest_grounded_session,
                ),
                temperature=0.0,
                operation="message_orchestration",
            )
            payload = parse_json_response(raw)
            decision = ConversationDecision.model_validate(payload)
            return self._normalize_decision(decision, message, latest_grounded_session)
        except (ValidationError, ValueError, TypeError, KeyError) as exc:
            logger.warning(
                "message orchestration produced invalid output; defaulting to search",
                extra={
                    "message_preview": message[:200],
                    "error": str(exc),
                    "console_visible": True,
                },
            )
            return self._fallback_search_decision(message, latest_grounded_session)
        except Exception as exc:
            logger.warning(
                "message orchestration failed; defaulting to search",
                extra={
                    "message_preview": message[:200],
                    "error": str(exc),
                    "console_visible": True,
                },
            )
            return self._fallback_search_decision(message, latest_grounded_session)

    def build_assistant_turn(self, decision: ConversationDecision) -> AssistantTurnState:
        content = (decision.assistant_text or "").strip()
        return AssistantTurnState(
            content=content,
            used_search=False,
            response_basis=decision.response_basis,
            linked_session_id=None,
            derived_from_session_id=decision.derived_from_session_id,
        )

    def _resolve_grounded_session(self, session_id: str | None) -> SessionState | None:
        if not session_id:
            return None
        try:
            session = self.pipeline_service.get_session(session_id)
        except SessionNotFoundError:
            return None
        if session.query_status != QueryRunStatus.COMPLETED:
            return None
        if session.response_mode != ResponseMode.RESPONSE_REVIEW:
            return None
        if not session.generated_response.strip():
            return None
        return session

    def _normalize_decision(
        self,
        decision: ConversationDecision,
        message: str,
        latest_grounded_session: SessionState | None,
    ) -> ConversationDecision:
        derived_from_session_id = decision.derived_from_session_id or None
        if derived_from_session_id is None and latest_grounded_session is not None:
            derived_from_session_id = latest_grounded_session.session_id

        if decision.action == ConversationAction.RESPOND:
            assistant_text = (decision.assistant_text or "").strip()
            if not assistant_text:
                return self._fallback_search_decision(message, latest_grounded_session)
            if decision.response_basis == ResponseBasis.CORPUS_SEARCH:
                return self._fallback_search_decision(message, latest_grounded_session)
            if decision.response_basis == ResponseBasis.THREAD_CONTEXT_ONLY and latest_grounded_session is None:
                return self._fallback_search_decision(message, latest_grounded_session)
            return ConversationDecision(
                action=ConversationAction.RESPOND,
                assistant_text=assistant_text,
                search_query=None,
                response_basis=decision.response_basis,
                derived_from_session_id=derived_from_session_id,
            )

        search_query = (decision.search_query or "").strip() or message
        return ConversationDecision(
            action=ConversationAction.SEARCH,
            assistant_text=None,
            search_query=search_query,
            response_basis=ResponseBasis.CORPUS_SEARCH,
            derived_from_session_id=derived_from_session_id,
        )

    def _fallback_search_decision(
        self,
        message: str,
        latest_grounded_session: SessionState | None,
    ) -> ConversationDecision:
        return ConversationDecision(
            action=ConversationAction.SEARCH,
            assistant_text=None,
            search_query=message,
            response_basis=ResponseBasis.CORPUS_SEARCH,
            derived_from_session_id=latest_grounded_session.session_id if latest_grounded_session is not None else None,
        )
