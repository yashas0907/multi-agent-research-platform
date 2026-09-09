"""Research job service — async job architecture.

POST /api/research creates a job row and returns immediately. A background
asyncio task runs the orchestrator, persisting state snapshots + events as
they happen. The frontend polls /status and /events (or reads the SSE stream).

This module owns the in-memory registry of running jobs (with cancellation
support) and delegates persistence to the session repository.
"""
from __future__ import annotations

import asyncio
from typing import Any

from app.core.database import get_sessionmaker
from app.core.logging import get_logger
from app.models.database import AgentEventModel, ResearchSessionModel
from app.schemas.research import (
    AgentEvent,
    ResearchDepth,
    ResearchState,
    ReportFormat,
    ResearchStage,
    SourceOrigin,
    SourceRecord,
    SourceType,
)
from app.workflows.orchestrator import Orchestrator

logger = get_logger(__name__)

# session_id -> asyncio.Task
_RUNNING: dict[str, asyncio.Task] = {}


class ResearchService:
    def __init__(self, orchestrator_factory=None) -> None:
        # injected for tests
        self._factory = orchestrator_factory

    async def start(
        self,
        session_id: str,
        question: str,
        depth: ResearchDepth,
        report_format: ReportFormat,
        document_ids: list[str] | None = None,
    ) -> None:
        state = ResearchState(
            session_id=session_id,
            question=question,
            depth=depth,
            requested_format=report_format,
        )
        # persist initial row immediately
        await self._persist_state(state)
        task = asyncio.create_task(self._execute(state, document_ids or []))
        _RUNNING[session_id] = task

    async def _execute(self, state: ResearchState, document_ids: list[str]) -> None:
        try:
            doc_sources = await self._load_document_sources(document_ids)
            if self._factory is not None:
                orch = self._factory(state)
            else:
                from app.tools import build_tool_registry

                orch = Orchestrator(
                    state,
                    build_tool_registry(),
                    document_sources=doc_sources,
                    event_sink=self._make_event_sink(state.session_id),
                    token_sync=self._make_token_sync(state),
                )
                _ORCHESTRATORS[state.session_id] = orch

            async def progress_persist() -> None:
                """Persist state snapshots so /status reflects live progress."""
                while not _DONE.get(state.session_id, True):
                    await asyncio.sleep(1.0)
                    try:
                        await self._persist_state(orch.state)
                    except Exception:  # noqa: BLE001
                        pass

            _DONE[state.session_id] = False
            monitor = asyncio.create_task(progress_persist())
            await self._persist_event(
                AgentEvent(
                    id=state.session_id + "0",
                    session_id=state.session_id,
                    agent="orchestrator",
                    event_type="stage_started",
                    message="Research started",
                    stage="pending",
                )
            )
            final_state = await orch.run()

            # failure path: also persist the terminal event
            if final_state.status in (ResearchStage.FAILED, ResearchStage.CANCELLED):
                await self._persist_event(
                    AgentEvent(
                        id=state.session_id + "z",
                        session_id=state.session_id,
                        agent="orchestrator",
                        event_type="stage_failed",
                        message=final_state.error or "Research cancelled",
                        stage=final_state.status.value,
                    )
                )
            await self._persist_state(final_state)
        except Exception as exc:  # noqa: BLE001 — worker must never die silently
            logger.exception("research_worker_crashed", session=state.session_id)
            state.error = f"worker crash: {exc}"
            state.status = ResearchStage.FAILED
            await self._persist_state(state)
        finally:
            _DONE[state.session_id] = True
            _RUNNING.pop(state.session_id, None)
            _ORCHESTRATORS.pop(state.session_id, None)

    @staticmethod
    def _make_token_sync(state: ResearchState):
        def sync(total_tokens: int) -> None:
            state.tokens_used = total_tokens

        return sync

    # event sink: persist every safe operational event to the DB as it happens
    def _make_event_sink(self, session_id: str):
        async def sink(event) -> None:
            try:
                await self._persist_event(event)
            except Exception:  # noqa: BLE001 — observability must not break research
                pass

        return sink

    async def cancel(self, session_id: str) -> bool:
        task = _RUNNING.get(session_id)
        if task is None or task.done():
            return False
        # orchestrators register themselves via factory closure; find via event
        # simpler: keep a side-map of orchestrators
        orch = _ORCHESTRATORS.get(session_id)
        if orch is not None:
            await orch.cancel()
            return True
        return False

    # ------------------------------------------------------------ persistence
    async def _persist_state(self, state: ResearchState) -> None:
        maker = await get_sessionmaker()
        async with maker() as session:
            obj = await session.get(ResearchSessionModel, state.session_id)
            if obj is None:
                obj = ResearchSessionModel(
                    id=state.session_id,
                    question=state.question,
                    depth=state.depth.value,
                    report_format=state.requested_format.value,
                )
                session.add(obj)
            obj.status = state.status.value
            obj.error = state.error
            obj.state_json = state.model_dump(mode="json")
            await session.commit()

    async def _persist_event(self, event: AgentEvent) -> None:
        maker = await get_sessionmaker()
        async with maker() as session:
            session.add(
                AgentEventModel(
                    id=event.id,
                    session_id=event.session_id,
                    agent=event.agent,
                    event_type=event.event_type.value,
                    stage=event.stage,
                    message=event.message,
                    data=event.data,
                )
            )
            await session.commit()

    async def _load_document_sources(self, document_ids: list[str]) -> list[SourceRecord]:
        """Convert ingested documents into SourceRecords (origin=user_document)."""
        from app.services.document_service import DocumentService

        doc_service = DocumentService()
        sources: list[SourceRecord] = []
        for doc_id in document_ids:
            try:
                doc = await doc_service.get_document(doc_id)
            except Exception:  # noqa: BLE001 — bad doc ids degrade gracefully
                continue
            chunks = await doc_service.get_document_chunks(doc_id, limit=200)
            content = "\n\n".join(c.text for c in chunks)
            sources.append(
                SourceRecord(
                    id=f"doc_{doc.id}",
                    title=doc.title or doc.filename,
                    url=None,
                    source_type=SourceType.USER_DOCUMENT,
                    origin=SourceOrigin.USER_DOC,
                    domain=None,
                    snippet=content[:400],
                    content_text=content,
                    trust_score=0.9,  # user-provided: high trust, not externally verified
                    relevance_score=1.0,
                    authority_score=0.9,
                    is_primary=True,
                    evaluation_notes="User-provided document (trusted as primary input)",
                    fetched_ok=True,
                    suspected_injection=_detect_injection(content),
                )
            )
        return sources


def _detect_injection(content: str) -> bool:
    """Flag injection-style content in user documents (defense-in-depth).

    The pipeline NEVER executes document content regardless of this flag —
    this exists so the security trace warns users their upload contains
    suspicious instruction-like text.
    """
    from app.tools.web_fetch import detect_injection

    try:
        return detect_injection(content)
    except Exception:  # noqa: BLE001
        return False


# session_id -> live orchestrator (for cancellation)
_ORCHESTRATORS: dict[str, Orchestrator] = {}
_DONE: dict[str, bool] = {}


async def wait_for_completion(session_id: str, timeout: float = 600) -> dict[str, Any]:
    """Utility for tests: wait until the job leaves the running set."""
    task = _RUNNING.get(session_id)
    if task is not None:
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
    return {}
