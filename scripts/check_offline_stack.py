"""Development helper: verify the whole offline stack quickly.

Runs from repo root:  python scripts/check_offline_stack.py
- imports all modules
- builds the tool registry
- runs one tiny research session (offline providers)
- asserts the structured report contract
"""
from __future__ import annotations

import asyncio
import os
import pathlib
import sys

os.environ.update(
    APP_ENV="test", LLM_PROVIDER="mock", EMBEDDING_PROVIDER="mock",
    WEB_SEARCH_PROVIDER="offline", LOG_LEVEL="ERROR",
    DATABASE_URL="sqlite+aiosqlite://",
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))


async def main() -> None:
    import app.core.database as db_mod
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import StaticPool

    engine = create_async_engine(
        "sqlite+aiosqlite://", poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    db_mod._engine = engine
    db_mod._sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    from app.models.database import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    from app.main import app  # full import surface incl. API layer
    from app.schemas.research import ResearchDepth, ReportFormat, ResearchState
    from app.tools import build_tool_registry
    from app.workflows.orchestrator import Orchestrator

    assert app.title, "FastAPI app failed to construct"

    registry = build_tool_registry()
    names = {t.name for t in registry.all()}
    assert names >= {"web_search", "web_fetch", "calculator", "vector_search"}

    state = ResearchState(
        session_id="check",
        question="What are the leading approaches to RAG evaluation?",
        depth=ResearchDepth.QUICK,
        requested_format=ReportFormat.DETAILED,
    )
    from app.workflows.budget import ResearchBudget

    budget = ResearchBudget(max_iterations=1, max_searches=4, max_sources=4,
                            max_tokens=10**9, max_runtime_seconds=60)
    final = await Orchestrator(state, registry, budget=budget).run()

    assert final.status.value == "report_ready", f"status={final.status.value}"
    assert final.report and final.report.executive_summary
    assert isinstance(final.report.key_findings, list)
    assert isinstance(final.report.sources, list)

    print("OFFLINE STACK OK")
    print(f"  app      : {app.title}")
    print(f"  tools    : {sorted(names)}")
    print(f"  pipeline : {final.status.value} "
          f"(sources={len(final.sources)}, evidence={len(final.evidence)})")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
