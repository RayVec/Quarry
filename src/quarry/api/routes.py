from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from quarry.domain.models import (
    CitationReplacementRequest,
    QueryRequest,
    ReviewCommentRequest,
    ReviewCommentUpdateRequest,
    ScopedRetrievalEnvelope,
    SessionEnvelope,
)
from quarry.services.pipeline_service import PipelineService
from quarry.services.session_store import SessionNotFoundError


router = APIRouter(prefix="/api/v1", tags=["quarry"])


def get_service(request: Request) -> PipelineService:
    return request.app.state.pipeline_service


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
    task = asyncio.create_task(service.run_query_for_session(session.session_id, payload))
    query_tasks: set[asyncio.Task] = request.app.state.query_tasks
    query_tasks.add(task)

    def _cleanup_query_task(done_task: asyncio.Task) -> None:
        query_tasks.discard(done_task)
        try:
            done_task.result()
        except Exception:
            pass

    task.add_done_callback(_cleanup_query_task)
    return SessionEnvelope(session=session)


@router.get("/sessions/{session_id}", response_model=SessionEnvelope)
async def get_session(session_id: str, service: PipelineService = Depends(get_service)) -> SessionEnvelope:
    try:
        session = service.get_session(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Session not found.") from exc
    return SessionEnvelope(session=session)


@router.get("/sessions/{session_id}/review-state", response_model=SessionEnvelope)
async def get_review_state(session_id: str, service: PipelineService = Depends(get_service)) -> SessionEnvelope:
    try:
        session = service.review_snapshot(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Session not found.") from exc
    return SessionEnvelope(session=session)


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def close_session(session_id: str, service: PipelineService = Depends(get_service)) -> Response:
    try:
        service.close_session(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Session not found.") from exc
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
        raise HTTPException(status_code=404, detail="Session not found.") from exc
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
        raise HTTPException(status_code=404, detail="Session not found.") from exc
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
        raise HTTPException(status_code=404, detail="Session not found.") from exc
    return SessionEnvelope(session=session)


@router.post("/sessions/{session_id}/citations/{citation_id}/scoped", response_model=ScopedRetrievalEnvelope)
async def scoped_retrieval(
    session_id: str,
    citation_id: int,
    payload: dict,
    service: PipelineService = Depends(get_service),
) -> ScopedRetrievalEnvelope:
    try:
        sentence_index = int(payload.get("sentence_index", -1))
        return await service.scoped_retrieval(session_id, sentence_index, citation_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Session not found.") from exc


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
        raise HTTPException(status_code=404, detail="Session not found.") from exc
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
        raise HTTPException(status_code=404, detail="Session not found.") from exc
    return SessionEnvelope(session=session)


@router.post("/sessions/{session_id}/refine", response_model=SessionEnvelope)
async def refine_response(session_id: str, service: PipelineService = Depends(get_service)) -> SessionEnvelope:
    try:
        session = await service.refine(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Session not found.") from exc
    return SessionEnvelope(session=session)
