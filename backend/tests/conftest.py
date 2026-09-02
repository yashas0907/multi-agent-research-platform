"""Pytest configuration: offline/mock environment, fresh DB per test."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make `app` importable when pytest is run from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("EMBEDDING_PROVIDER", "mock")
os.environ.setdefault("WEB_SEARCH_PROVIDER", "offline")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite://")

import pytest_asyncio

import app.core.database as db_mod
from app.core.llm import reset_llm_client
from app.retrieval.embeddings import reset_embedding_service
from app.retrieval.vector_store import reset_vector_store


@pytest_asyncio.fixture(autouse=True)
async def fresh_db():
    """Every test gets a brand-new in-memory database (single shared
    connection via StaticPool so schema is visible to all sessions)."""
    from sqlalchemy.pool import StaticPool
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    engine = create_async_engine(
        "sqlite+aiosqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    db_mod._engine = engine
    db_mod._sessionmaker = async_sessionmaker(engine, expire_on_commit=False)

    from app.models.database import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    reset_llm_client()
    reset_embedding_service()
    reset_vector_store()

    yield engine

    await engine.dispose()
    db_mod._engine = None
    db_mod._sessionmaker = None
    reset_llm_client()
    reset_embedding_service()
    reset_vector_store()
