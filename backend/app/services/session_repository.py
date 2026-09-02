"""Read-side repository for research sessions and events."""
from __future__ import annotations

from sqlalchemy import func, select as sa_select

from app.core.database import get_sessionmaker
from app.core.errors import NotFoundError
from app.models.database import AgentEventModel, ResearchSessionModel
from app.schemas.research import (
    AgentEvent,
    AgentEventType,
    ResearchState,
    ResearchStage,
)

STAGE_PROGRESS: dict[str, int] = {
    "pending": 0,
    "planning": 10,
    "searching": 30,
    "gathering_evidence": 45,
    "fact_checking": 55,
    "contradiction_check": 62,
    "critiquing": 70,
    "synthesizing": 80,
    "citing": 90,
    "report_ready": 100,
    "completed_partial": 100,
    "cancelled": 0,
    "failed": 0,
}


class SessionRepository:
    async def get_state(self, session_id: str) -> ResearchState:
        maker = await get_sessionmaker()
        async with maker() as session:
            row = await session.get(ResearchSessionModel, session_id)
        if row is None:
            raise NotFoundError(f"research session {session_id} not found")
        return ResearchState.model_validate(row.state_json)

    async def exists(self, session_id: str) -> bool:
        maker = await get_sessionmaker()
        async with maker() as session:
            row = await session.get(ResearchSessionModel, session_id)
            return row is not None

    async def get_events(
        self, session_id: str, after_index: int = 0, limit: int = 500
    ) -> list[AgentEvent]:
        maker = await get_sessionmaker()
        async with maker() as session:
            rows = (
                await session.execute(
                    sa_select(AgentEventModel)
                    .where(AgentEventModel.session_id == session_id)
                    .order_by(AgentEventModel.timestamp, AgentEventModel.id)
                    .limit(limit)
                    .offset(after_index)
                )
            ).scalars().all()
        return [
            AgentEvent(
                id=r.id,
                session_id=r.session_id,
                agent=r.agent,
                event_type=AgentEventType(r.event_type),
                message=r.message,
                stage=r.stage,
                data=dict(r.data or {}),
                timestamp=r.timestamp.isoformat() if r.timestamp else "",
            )
            for r in rows
        ]

    async def status_summary(self, session_id: str) -> dict:
        state = await self.get_state(session_id)
        return {
            "session_id": session_id,
            "status": state.status.value,
            "stage_label": state.status.value.replace("_", " ").title(),
            "progress_pct": STAGE_PROGRESS.get(state.status.value, 0),
            "current_iteration": state.current_iteration,
            "question": state.question,
            "depth": state.depth.value,
            "error": state.error,
            "budget": state.budget_snapshot(),
        }

    async def list_sessions(self, limit: int = 20) -> list[dict]:
        maker = await get_sessionmaker()
        async with maker() as session:
            rows = (
                await session.execute(
                    sa_select(ResearchSessionModel)
                    .order_by(ResearchSessionModel.created_at.desc())
                    .limit(limit)
                )
            ).scalars().all()
        out = []
        for r in rows:
            state = ResearchState.model_validate(r.state_json) if r.state_json else None
            out.append(
                {
                    "session_id": r.id,
                    "question": r.question,
                    "status": (state.status.value if state else r.status),
                    "depth": r.depth,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
            )
        return out

    async def count_sessions(self) -> int:
        maker = await get_sessionmaker()
        async with maker() as session:
            return (
                await session.execute(sa_select(func.count(ResearchSessionModel.id)))
            ).scalar() or 0
