"""Development helper: run a research question end-to-end from the CLI.

Usage (from repo root):
    python scripts/run_research.py "your research question" --depth standard
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import sys

# Offline defaults so the script works without any API keys; override via env.
os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("EMBEDDING_PROVIDER", "mock")
os.environ.setdefault("WEB_SEARCH_PROVIDER", "offline")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./data/scripts.db")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run one research session end-to-end")
    parser.add_argument("question", help="the research question")
    parser.add_argument("--depth", default="standard", choices=["quick", "standard", "deep"])
    parser.add_argument("--format", default="detailed",
                        choices=["detailed", "executive_summary", "comparison"])
    parser.add_argument("--save-report", default=None,
                        help="path to write the JSON report")
    args = parser.parse_args()

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

    from app.schemas.research import ResearchDepth, ReportFormat, ResearchState
    from app.tools import build_tool_registry
    from app.workflows.budget import ResearchBudget
    from app.workflows.orchestrator import Orchestrator

    state = ResearchState(
        session_id="cli_run",
        question=args.question,
        depth=ResearchDepth(args.depth),
        requested_format=ReportFormat(args.format),
    )
    budget = ResearchBudget.from_settings()
    orch = Orchestrator(state, build_tool_registry(), budget=budget)
    final = await orch.run()

    print("\n" + "=" * 64)
    print(f"STATUS      : {final.status.value}")
    print(f"SUBQUESTIONS: {len(final.subquestions)}")
    print(f"SOURCES     : {len(final.sources)}")
    print(f"EVIDENCE    : {len(final.evidence)}")
    print(f"CLAIMS      : {len(final.claims)}")
    print(f"TOKENS      : {final.tokens_used} (LLM calls: {final.llm_calls})")

    if final.report:
        print("\nEXECUTIVE SUMMARY")
        print("-" * 64)
        print(final.report.executive_summary)
        if final.report.key_findings:
            print("\nKEY FINDINGS")
            print("-" * 64)
            for f in final.report.key_findings:
                conf = f.confidence.value
                marker = " [interpretation]" if f.is_interpretation else ""
                print(f"- [{conf}]{marker} {f.statement}")
        if final.report.sources:
            print("\nSOURCES CITED")
            print("-" * 64)
            for s in final.report.sources:
                print(f"- {s.title[:70]}  ({s.domain or 'user document'}, trust={s.trust_score})")
        if args.save_report:
            path = pathlib.Path(args.save_report)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(final.report.model_dump_json(indent=2), encoding="utf-8")
            print(f"\nreport written to {path}")
    elif final.error:
        print(f"\nERROR: {final.error}")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
