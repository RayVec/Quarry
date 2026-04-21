from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response, status

from quarry.domain.models import (
    ApiError,
    CitationFeedbackRequest,
    CitationReplaceRequest,
    CitationReplacementRequest,
    HostedSettingsEnvelope,
    HostedSettingsUpdateRequest,
    MessageRequest,
    MessageRunEnvelope,
    QueryRequest,
    ReviewCommentRequest,
    ReviewCommentUpdateRequest,
    ScopedRetrievalRequest,
    ScopedRetrievalEnvelope,
    SessionEnvelope,
)
from quarry.hosted_settings import build_hosted_settings_envelope, persist_hosted_settings
from quarry.services.message_service import MessageService
from quarry.services.message_run_store import MessageRunNotFoundError
from quarry.services.pipeline_service import PipelineService
from quarry.services.session_store import SessionNotFoundError
from quarry.config import Settings


router = APIRouter(prefix="/api/v1", tags=["quarry"])


def get_service(request: Request) -> PipelineService:
    return request.app.state.pipeline_service


def get_message_service(request: Request) -> MessageService:
    return request.app.state.message_service


def _config_path_from_request(request: Request) -> str | Path | None:
    return getattr(request.app.state, "config_path", None)


def api_error(
    *,
    status_code: int,
    code: str,
    message: str,
    details: dict[str, str | int | float | bool | None] | None = None,
) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail=ApiError(code=code, message=message, details=details).model_dump(),
    )


def _schedule_query_task(
    *,
    app: FastAPI,
    service: PipelineService,
    session_id: str,
    payload: QueryRequest,
) -> None:
    task = asyncio.create_task(service.run_query_for_session(session_id, payload))
    query_tasks: set[asyncio.Task] = app.state.query_tasks
    query_tasks.add(task)

    def _cleanup_query_task(done_task: asyncio.Task) -> None:
        query_tasks.discard(done_task)
        try:
            done_task.result()
        except Exception as exc:
            app.state.background_error_logger.exception(
                "query task failed",
                extra={"error": str(exc), "task": "run_query_for_session", "console_visible": True},
            )

    task.add_done_callback(_cleanup_query_task)


async def _process_message_run(
    *,
    app: FastAPI,
    message_service: MessageService,
    pipeline_service: PipelineService,
    message_run_id: str,
    payload: MessageRequest,
) -> None:
    try:
        message_run, query_payload = await message_service.run_message_for_run(message_run_id, payload)
        if query_payload is not None and message_run.session is not None:
            _schedule_query_task(
                app=app,
                service=pipeline_service,
                session_id=message_run.session.session_id,
                payload=query_payload,
            )
    except Exception as exc:
        try:
            message_service.fail_message_run(message_run_id)
        except MessageRunNotFoundError:
            pass
        app.state.background_error_logger.exception(
            "message run task failed",
            extra={"error": str(exc), "task": "run_message_for_run", "console_visible": True},
        )


def _schedule_message_run_task(
    *,
    app: FastAPI,
    message_service: MessageService,
    pipeline_service: PipelineService,
    message_run_id: str,
    payload: MessageRequest,
) -> None:
    task = asyncio.create_task(
        _process_message_run(
            app=app,
            message_service=message_service,
            pipeline_service=pipeline_service,
            message_run_id=message_run_id,
            payload=payload,
        )
    )
    message_run_tasks: set[asyncio.Task] = app.state.message_run_tasks
    message_run_tasks.add(task)

    def _cleanup_message_run_task(done_task: asyncio.Task) -> None:
        message_run_tasks.discard(done_task)
        try:
            done_task.result()
        except Exception:
            pass

    task.add_done_callback(_cleanup_message_run_task)


@router.get("/settings/hosted", response_model=HostedSettingsEnvelope)
async def get_hosted_settings(request: Request) -> HostedSettingsEnvelope:
    return build_hosted_settings_envelope(_config_path_from_request(request))


@router.put("/settings/hosted", response_model=HostedSettingsEnvelope)
async def update_hosted_settings(
    payload: HostedSettingsUpdateRequest,
    request: Request,
) -> HostedSettingsEnvelope:
    config_path = _config_path_from_request(request)
    try:
        envelope = persist_hosted_settings(payload, config_path=config_path)
        next_settings = Settings.from_env(config_path=config_path)
        request.app.state.reconfigure_runtime(next_settings)
        return envelope
    except RuntimeError as exc:
        raise api_error(
            status_code=409,
            code="SETTINGS_CONFLICT",
            message=str(exc),
        ) from exc
    except ValueError as exc:
        raise api_error(
            status_code=422,
            code="SETTINGS_VALIDATION_ERROR",
            message=str(exc),
        ) from exc


@router.post("/query", response_model=SessionEnvelope)
async def run_query(payload: QueryRequest, service: PipelineService = Depends(get_service)) -> SessionEnvelope:
    session = await service.run_query(payload)
    return SessionEnvelope(session=session)


@router.post("/query/start", response_model=SessionEnvelope)
async def start_query(
    payload: QueryRequest,
    request: Request,
    service: PipelineService = Depends(get_service),
) -> SessionEnvelope:
    session = service.begin_query(payload)
    _schedule_query_task(app=request.app, service=service, session_id=session.session_id, payload=payload)
    return SessionEnvelope(session=session)


@router.post("/messages/start", response_model=MessageRunEnvelope)
async def start_message(
    payload: MessageRequest,
    request: Request,
    service: PipelineService = Depends(get_service),
    message_service: MessageService = Depends(get_message_service),
) -> MessageRunEnvelope:
    message_run = message_service.begin_message_run()
    _schedule_message_run_task(
        app=request.app,
        message_service=message_service,
        pipeline_service=service,
        message_run_id=message_run.message_run_id,
        payload=payload,
    )
    return MessageRunEnvelope(message_run=message_run)


@router.get("/message-runs/{message_run_id}", response_model=MessageRunEnvelope)
async def get_message_run(
    message_run_id: str,
    message_service: MessageService = Depends(get_message_service),
) -> MessageRunEnvelope:
    try:
        message_run = message_service.get_message_run(message_run_id)
    except MessageRunNotFoundError as exc:
        raise api_error(status_code=404, code="MESSAGE_RUN_NOT_FOUND", message="Message run not found.") from exc
    return MessageRunEnvelope(message_run=message_run)


@router.get("/sessions/{session_id}", response_model=SessionEnvelope)
async def get_session(session_id: str, service: PipelineService = Depends(get_service)) -> SessionEnvelope:
    try:
        session = service.get_session(session_id)
    except SessionNotFoundError as exc:
        raise api_error(status_code=404, code="SESSION_NOT_FOUND", message="Session not found.") from exc
    return SessionEnvelope(session=session)


@router.get("/sessions/{session_id}/review-state", response_model=SessionEnvelope)
async def get_review_state(session_id: str, service: PipelineService = Depends(get_service)) -> SessionEnvelope:
    try:
        session = service.review_snapshot(session_id)
    except SessionNotFoundError as exc:
        raise api_error(status_code=404, code="SESSION_NOT_FOUND", message="Session not found.") from exc
    return SessionEnvelope(session=session)


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def close_session(session_id: str, service: PipelineService = Depends(get_service)) -> Response:
    try:
        service.close_session(session_id)
    except SessionNotFoundError as exc:
        raise api_error(status_code=404, code="SESSION_NOT_FOUND", message="Session not found.") from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/sessions/{session_id}/comments", response_model=SessionEnvelope)
async def add_review_comment(
    session_id: str,
    payload: ReviewCommentRequest,
    service: PipelineService = Depends(get_service),
) -> SessionEnvelope:
    try:
        session = service.add_review_comment(session_id, payload)
    except SessionNotFoundError as exc:
        raise api_error(status_code=404, code="SESSION_NOT_FOUND", message="Session not found.") from exc
    return SessionEnvelope(session=session)


@router.patch("/sessions/{session_id}/comments/{comment_id}", response_model=SessionEnvelope)
async def update_review_comment(
    session_id: str,
    comment_id: str,
    payload: ReviewCommentUpdateRequest,
    service: PipelineService = Depends(get_service),
) -> SessionEnvelope:
    try:
        session = service.update_review_comment(session_id, comment_id, payload.comment_text)
    except SessionNotFoundError as exc:
        raise api_error(status_code=404, code="SESSION_NOT_FOUND", message="Session not found.") from exc
    return SessionEnvelope(session=session)


@router.delete("/sessions/{session_id}/comments/{comment_id}", response_model=SessionEnvelope)
async def delete_review_comment(
    session_id: str,
    comment_id: str,
    service: PipelineService = Depends(get_service),
) -> SessionEnvelope:
    try:
        session = service.delete_review_comment(session_id, comment_id)
    except SessionNotFoundError as exc:
        raise api_error(status_code=404, code="SESSION_NOT_FOUND", message="Session not found.") from exc
    return SessionEnvelope(session=session)


@router.post("/sessions/{session_id}/citations/{citation_id}/scoped", response_model=ScopedRetrievalEnvelope)
async def scoped_retrieval(
    session_id: str,
    citation_id: int,
    payload: ScopedRetrievalRequest,
    service: PipelineService = Depends(get_service),
) -> ScopedRetrievalEnvelope:
    try:
        return await service.scoped_retrieval(session_id, payload.sentence_index, citation_id)
    except SessionNotFoundError as exc:
        raise api_error(status_code=404, code="SESSION_NOT_FOUND", message="Session not found.") from exc


@router.post("/sessions/{session_id}/citations/{citation_id}/replace", response_model=SessionEnvelope)
async def replace_citation(
    session_id: str,
    citation_id: int,
    payload: CitationReplacementRequest,
    service: PipelineService = Depends(get_service),
) -> SessionEnvelope:
    try:
        session = service.replace_citation(session_id, payload.sentence_index, citation_id, payload)
    except SessionNotFoundError as exc:
        raise api_error(status_code=404, code="SESSION_NOT_FOUND", message="Session not found.") from exc
    return SessionEnvelope(session=session)


@router.post("/sessions/{session_id}/citations/{citation_id}/feedback", response_model=SessionEnvelope)
async def set_citation_feedback(
    session_id: str,
    citation_id: int,
    payload: CitationFeedbackRequest,
    service: PipelineService = Depends(get_service),
) -> SessionEnvelope:
    try:
        session = service.set_citation_feedback(
            session_id,
            payload.sentence_index,
            citation_id,
            payload.feedback_type,
        )
    except SessionNotFoundError as exc:
        raise api_error(status_code=404, code="SESSION_NOT_FOUND", message="Session not found.") from exc
    return SessionEnvelope(session=session)


@router.get("/sessions/{session_id}/citations/{citation_id}/alternatives", response_model=ScopedRetrievalEnvelope)
async def get_citation_alternatives(
    session_id: str,
    citation_id: int,
    service: PipelineService = Depends(get_service),
) -> ScopedRetrievalEnvelope:
    try:
        return await service.get_citation_alternatives(session_id, citation_id)
    except SessionNotFoundError as exc:
        raise api_error(status_code=404, code="SESSION_NOT_FOUND", message="Session not found.") from exc


@router.put("/sessions/{session_id}/citations/{citation_id}/replace", response_model=SessionEnvelope)
async def replace_with_alternative(
    session_id: str,
    citation_id: int,
    payload: CitationReplaceRequest,
    service: PipelineService = Depends(get_service),
) -> SessionEnvelope:
    try:
        session = service.replace_with_alternative(
            session_id,
            payload.sentence_index,
            citation_id,
            payload.replacement_citation_id,
        )
    except SessionNotFoundError as exc:
        raise api_error(status_code=404, code="SESSION_NOT_FOUND", message="Session not found.") from exc
    return SessionEnvelope(session=session)


@router.post("/sessions/{session_id}/citations/{citation_id}/undo", response_model=SessionEnvelope)
async def undo_citation_replacement(
    session_id: str,
    citation_id: int,
    service: PipelineService = Depends(get_service),
) -> SessionEnvelope:
    try:
        session = service.undo_replacement(session_id, citation_id)
    except SessionNotFoundError as exc:
        raise api_error(status_code=404, code="SESSION_NOT_FOUND", message="Session not found.") from exc
    return SessionEnvelope(session=session)


@router.post("/sessions/{session_id}/refine", response_model=SessionEnvelope)
async def refine_response(session_id: str, service: PipelineService = Depends(get_service)) -> SessionEnvelope:
    try:
        session = await service.refine(session_id)
    except SessionNotFoundError as exc:
        raise api_error(status_code=404, code="SESSION_NOT_FOUND", message="Session not found.") from exc
    return SessionEnvelope(session=session)
