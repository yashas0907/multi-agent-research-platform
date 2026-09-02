"""Async SQLAlchemy engine/session management."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None
_init_lock = asyncio.Lock()


async def get_engine() -> AsyncEngine:
    global _engine, _sessionmaker
    if _engine is None:
        async with _init_lock:
            if _engine is None:
                settings = get_settings()
                _ensure_sqlite_dir(settings.DATABASE_URL)
                _engine = create_async_engine(settings.DATABASE_URL, echo=False)
                _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
                logger.info("database_engine_created", url_safe=_safe_url(settings.DATABASE_URL))
    return _engine


def _ensure_sqlite_dir(url: str) -> None:
    """Create the parent directory for file-based SQLite URLs (aiosqlite
    does not create missing directories itself)."""
    if "sqlite" in url and ":///" in url:
        path = url.split("///", 1)[1].split("?", 1)[0]
        if path and path != ":memory:":
            from pathlib import Path

            parent = Path(path).parent
            if str(parent) not in ("", "."):
                parent.mkdir(parents=True, exist_ok=True)


def _safe_url(url: str) -> str:
    # Never log credentials embedded in a DB URL.
    if "://" in url and "@" in url:
        scheme, rest = url.split("://", 1)
        if "@" in rest:
            creds, host = rest.rsplit("@", 1)
            return f"{scheme}://***@{host}"
    return url


async def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    await get_engine()
    assert _sessionmaker is not None
    return _sessionmaker


async def session_iterator() -> AsyncIterator[AsyncSession]:
    maker = await get_sessionmaker()
    async with maker() as session:
        yield session


async def init_db() -> None:
    """Create tables. Called once at startup (dev/test); Alembic in prod."""
    from app.models.database import Base

    engine = await get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("database_initialized")


async def dispose_engine() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _sessionmaker = None
