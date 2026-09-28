"""HTTP API layer.

Endpoints:
    POST /api/research              create a research job (async)
    GET  /api/research              list recent sessions
    GET  /api/research/{id}         full state
    GET  /api/research/{id}/status  progress summary (polling)
    GET  /api/research/{id}/events  live trace events (polling)
    GET  /api/research/{id}/report  final structured report
    POST /api/research/{id}/cancel  cancel a running job
    POST /api/documents             upload a document (multipart)
    GET  /api/documents             list documents
    GET  /api/documents/{id}        document info + chunks
    DELETE /api/documents/{id}      delete document
    GET  /api/tools                  tool catalog
    GET  /api/health                 health check
"""
from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse

from app.core.config import get_settings
from app.core.errors import DocumentIngestError, NotFoundError
from app.core.logging import get_logger
from app.schemas.api import (
    DocumentIngestResponse,
    HealthResponse,
    ResearchCreateRequest,
    ResearchCreateResponse,
    ResearchStatusResponse,
    SourceListResponse,
    SubQuestionsResponse,
    TraceResponse,
)
from app.schemas.research import (
    EvidenceConfidence,
    ReportFormat,
    ResearchDepth,
    ResearchStage,
    SourceOrigin,
)
from app.services.document_service import DocumentService
from app.services.research_service import ResearchService
from app.services.session_repository import SessionRepository

logger = get_logger(__name__)
router = APIRouter(prefix="/api")

research_service = ResearchService()
repository = SessionRepository()
document_service = DocumentService()


def _not_found(exc: NotFoundError) -> HTTPException:
    return HTTPException(status_code=404, detail=str(exc))


# ---------------------------------------------------------------- research
@router.post("/research", response_model=ResearchCreateResponse, status_code=201)
async def create_research(req: ResearchCreateRequest) -> ResearchCreateResponse:
    """Create a research job. Returns immediately; poll /status for progress."""
    session_id = uuid.uuid4().hex[:12]
    try:
        await research_service.start(
            session_id=session_id,
            question=req.question,
            depth=ResearchDepth(req.depth),
            report_format=ReportFormat(req.report_format),
            document_ids=req.document_ids,
        )
    except Exception as exc:  # noqa: BLE001 — surface clean 500s, never crash
        logger.exception("research_start_failed")
        raise HTTPException(status_code=500, detail=f"failed to start research: {exc}") from exc
    return ResearchCreateResponse(
        session_id=session_id,
        status=ResearchStage.PENDING,
        question=req.question,
        depth=req.depth,
    )


@router.get("/research")
async def list_research(limit: int = Query(20, ge=1, le=100)) -> dict:
    return {"sessions": await repository.list_sessions(limit=limit)}


@router.get("/research/{session_id}")
async def get_research(session_id: str) -> dict:
    try:
        state = await repository.get_state(session_id)
    except NotFoundError as exc:
        raise _not_found(exc) from exc
    return state.model_dump(mode="json")


@router.get("/research/{session_id}/status", response_model=ResearchStatusResponse)
async def research_status(session_id: str) -> ResearchStatusResponse:
    try:
        summary = await repository.status_summary(session_id)
    except NotFoundError as exc:
        raise _not_found(exc) from exc
    return ResearchStatusResponse(**summary)


@router.get("/research/{session_id}/events", response_model=TraceResponse)
async def research_events(
    session_id: str, after: int = Query(0, ge=0)
) -> TraceResponse:
    try:
        if not await repository.exists(session_id):
            raise NotFoundError(f"research session {session_id} not found")
        events = await repository.get_events(session_id, after_index=after)
    except NotFoundError as exc:
        raise _not_found(exc) from exc
    return TraceResponse(session_id=session_id, events=events)


@router.get("/research/{session_id}/events/stream")
async def research_events_stream(session_id: str) -> StreamingResponse:
    """Server-Sent Events stream: true real-time trace (no polling).

    Yields each new event as `data: {json}\\n\\n`, then a final `status`
    payload when the job reaches a terminal state, then closes. Clients
    should fall back to polling /events if the stream errors.
    """
    import asyncio
    import json as _json

    try:
        if not await repository.exists(session_id):
            raise NotFoundError(f"research session {session_id} not found")
    except NotFoundError as exc:
        raise _not_found(exc) from exc

    async def generator():
        sent = 0
        idle_polls = 0
        while True:
            events = await repository.get_events(session_id, after_index=sent)
            if events:
                idle_polls = 0
                for e in events:
                    sent += 1
                    payload = _json.dumps(e.model_dump(mode="json"), default=str)
                    yield f"data: {payload}\n\n"
            else:
                idle_polls += 1

            # status check: close the stream on terminal states
            try:
                summary = await repository.status_summary(session_id)
            except NotFoundError:
                break
            terminal = summary["status"] in (
                "report_ready", "failed", "cancelled", "completed_partial"
            )
            if terminal and not events:
                # drain any events that raced in, then send final status + close
                final = await repository.get_events(session_id, after_index=sent)
                for e in final:
                    sent += 1
                    payload = _json.dumps(e.model_dump(mode="json"), default=str)
                    yield f"data: {payload}\n\n"
                yield f"data: {_json.dumps({'type': 'status', 'status': summary['status'], 'progress_pct': summary['progress_pct'], 'stage_label': summary['stage_label']})}\n\n"
                yield "data: {\"type\": \"done\"}\n\n"
                break
            if terminal and events:
                continue  # loop once more to send the closing status
            # safety: don't stream forever if the job vanished
            if idle_polls > 900:  # ~15 min at 1s
                yield "data: {\"type\": \"done\"}\n\n"
                break
            await asyncio.sleep(1.0)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/research/{session_id}/report")
async def research_report(session_id: str) -> dict:
    try:
        state = await repository.get_state(session_id)
    except NotFoundError as exc:
        raise _not_found(exc) from exc
    if state.report is None:
        raise HTTPException(
            status_code=409,
            detail=f"report not ready (current status: {state.status.value})",
        )
    return state.report.model_dump(mode="json")


@router.get("/research/{session_id}/subquestions", response_model=SubQuestionsResponse)
async def research_subquestions(session_id: str) -> SubQuestionsResponse:
    try:
        state = await repository.get_state(session_id)
    except NotFoundError as exc:
        raise _not_found(exc) from exc
    return SubQuestionsResponse(session_id=session_id, subquestions=state.subquestions)


@router.post("/research/{session_id}/cancel")
async def cancel_research(session_id: str) -> dict:
    try:
        if not await repository.exists(session_id):
            raise NotFoundError(f"research session {session_id} not found")
    except NotFoundError as exc:
        raise _not_found(exc) from exc
    cancelled = await research_service.cancel(session_id)
    return {"session_id": session_id, "cancel_requested": cancelled}


@router.get("/research/{session_id}/sources", response_model=SourceListResponse)
async def research_sources(session_id: str) -> SourceListResponse:
    try:
        state = await repository.get_state(session_id)
    except NotFoundError as exc:
        raise _not_found(exc) from exc
    sources = [s.model_dump(mode="json") for s in state.sources.values()]
    sources.sort(key=lambda s: s.get("trust_score", 0), reverse=True)
    return SourceListResponse(sources=sources, total=len(sources))


# ---------------------------------------------------------------- documents
@router.post("/documents", response_model=DocumentIngestResponse, status_code=201)
async def upload_document(file: UploadFile = File(...)) -> DocumentIngestResponse:
    raw = await file.read()
    filename = file.filename or "upload.txt"
    try:
        info = await document_service.ingest(filename, raw)
    except DocumentIngestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return DocumentIngestResponse(
        document_id=info.id,
        filename=info.filename,
        status=info.status,
        chunks=info.chunks,
        message="document ingested successfully",
    )


@router.get("/documents")
async def list_documents() -> dict:
    return {"documents": [d.__dict__ for d in await document_service.list_documents()]}


@router.get("/documents/{doc_id}")
async def get_document(doc_id: str, chunks: bool = Query(False)) -> dict:
    try:
        info = await document_service.get_document(doc_id)
        out = info.__dict__
        if chunks:
            out["chunks"] = [
                {"chunk_index": c.chunk_index, "text": c.text[:400]}
                for c in await document_service.get_document_chunks(doc_id)
            ]
        return out
    except NotFoundError as exc:
        raise _not_found(exc) from exc


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str) -> dict:
    try:
        removed = await document_service.delete_document(doc_id)
    except NotFoundError as exc:
        raise _not_found(exc) from exc
    return {"document_id": doc_id, "chunks_removed": removed}


# ---------------------------------------------------------------- tools
@router.get("/tools")
async def tool_catalog() -> dict:
    from app.tools import build_tool_registry

    return {"tools": build_tool_registry().catalog()}


# ---------------------------------------------------------------- health
@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    from datetime import datetime, timezone

    from sqlalchemy import text

    from app.core.database import get_engine

    settings = get_settings()
    db_ok = True
    try:
        engine = await get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception:  # noqa: BLE001
        db_ok = False
    return HealthResponse(
        status="ok" if db_ok else "degraded",
        app_env=settings.APP_ENV,
        llm_provider=settings.LLM_PROVIDER,
        web_search_provider=settings.WEB_SEARCH_PROVIDER,
        database="ok" if db_ok else "unavailable",
        time=datetime.now(timezone.utc).isoformat(),
    )
