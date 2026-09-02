"""SQLAlchemy database models.

The database is the durable record of research: sessions, tasks, sources,
evidence, claims, contradictions, reports, documents, and agent events.
Nothing is persisted to random JSON files.
"""
from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return uuid.uuid4().hex[:12]


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Base(DeclarativeBase):
    pass


class JSONMixin:
    """Helpers for Pydantic <-> ORM conversion."""

    @staticmethod
    def dumps(v: Any) -> str:
        import json

        return json.dumps(v, default=str)

    @staticmethod
    def loads(s: str) -> Any:
        import json

        return json.loads(s)


class ResearchSessionModel(Base):
    __tablename__ = "research_sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    question: Mapped[str] = mapped_column(Text)
    depth: Mapped[str] = mapped_column(String(16), default="standard")
    report_format: Mapped[str] = mapped_column(String(32), default="detailed")
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)
    completed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Snapshot of the full ResearchState (Pydantic → JSON) — the audit record.
    state_json: Mapped[dict] = mapped_column(JSON, default=dict)

    events: Mapped[list["AgentEventModel"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    tasks: Mapped[list["ResearchTaskModel"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class ResearchTaskModel(Base):
    __tablename__ = "research_tasks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(ForeignKey("research_sessions.id"), index=True)
    stage: Mapped[str] = mapped_column(String(32))
    agent: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default="pending")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    session: Mapped[ResearchSessionModel] = relationship(back_populates="tasks")


class AgentEventModel(Base):
    __tablename__ = "agent_events"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: uuid.uuid4().hex)
    session_id: Mapped[str] = mapped_column(ForeignKey("research_sessions.id"), index=True)
    agent: Mapped[str] = mapped_column(String(64))
    event_type: Mapped[str] = mapped_column(String(32))
    stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    message: Mapped[str] = mapped_column(Text)
    data: Mapped[dict] = mapped_column(JSON, default=dict)
    timestamp: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    session: Mapped[ResearchSessionModel] = relationship(back_populates="events")


class DocumentModel(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(String(256))
    content_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16), default="processing")
    chunks: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    # Document metadata the parser could genuinely extract (no invention).
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    author: Mapped[str | None] = mapped_column(String(256), nullable=True)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)


class VectorChunkModel(Base):
    __tablename__ = "vector_chunks"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: uuid.uuid4().hex)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[str] = mapped_column(JSON)  # float list
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
