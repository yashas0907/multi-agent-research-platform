"""Vector store abstraction.

`VectorStore` interface supports add / query / delete with optional metadata
filtering. Two backends:
  * `SQLiteVectorStore` — embeddings stored in the app database (portable,
    zero external services; cosine similarity computed in-process). Good for
    development, demos, and small/medium corpora.
  * Swappable: production can plug in pgvector/Qdrant/Milvus by implementing
    the same interface (see docs/vector_store.md).
"""
from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import delete as sa_delete, select as sa_select

from app.core.database import get_sessionmaker
from app.core.errors import VectorStoreError
from app.core.logging import get_logger
from app.models.database import VectorChunkModel

logger = get_logger(__name__)


@dataclass
class StoredChunk:
    id: str
    document_id: str
    text: str
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


class VectorStore(abc.ABC):
    """Backend-agnostic vector storage interface."""

    @abc.abstractmethod
    async def add_chunks(
        self, chunks: list[dict[str, Any]]
    ) -> int:
        """chunks: [{id?, document_id, chunk_index, text, embedding, metadata}]"""

    @abc.abstractmethod
    async def query(
        self,
        embedding: list[float],
        *,
        top_k: int = 5,
        document_ids: list[str] | None = None,
        min_score: float = -1.0,
    ) -> list[StoredChunk]:
        """min_score: gate AFTER sorting (-1.0 = no gate; the retriever's
        combined-score gating is the quality control, not this filter)."""

    @abc.abstractmethod
    async def delete_document(self, document_id: str) -> int:
        ...

    @abc.abstractmethod
    async def count(self, document_ids: list[str] | None = None) -> int:
        ...


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise VectorStoreError("embedding dimension mismatch")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class SQLiteVectorStore(VectorStore):
    """App-database-backed vector store with in-process cosine scoring."""

    async def add_chunks(self, chunks: list[dict[str, Any]]) -> int:
        if not chunks:
            return 0
        maker = await get_sessionmaker()
        added = 0
        async with maker() as session:
            for c in chunks:
                model = VectorChunkModel(
                    document_id=c["document_id"],
                    chunk_index=c.get("chunk_index", 0),
                    text=c["text"],
                    embedding=[float(v) for v in c["embedding"]],
                    metadata_json=c.get("metadata", {}),
                )
                session.add(model)
                added += 1
            await session.commit()
        return added

    async def query(
        self,
        embedding: list[float],
        *,
        top_k: int = 5,
        document_ids: list[str] | None = None,
        min_score: float = -1.0,
    ) -> list[StoredChunk]:
        maker = await get_sessionmaker()
        async with maker() as session:
            stmt = sa_select(VectorChunkModel)
            if document_ids is not None:
                if not document_ids:
                    return []
                stmt = stmt.where(VectorChunkModel.document_id.in_(document_ids))
            rows = (await session.execute(stmt)).scalars().all()

        scored: list[tuple[float, VectorChunkModel]] = []
        for row in rows:
            score = cosine_similarity(embedding, row.embedding)
            if score >= min_score:
                scored.append((score, row))
        scored.sort(key=lambda p: p[0], reverse=True)

        return [
            StoredChunk(
                id=row.id,
                document_id=row.document_id,
                text=row.text,
                score=score,
                metadata=dict(row.metadata_json or {}),
            )
            for score, row in scored[:top_k]
        ]

    async def delete_document(self, document_id: str) -> int:
        maker = await get_sessionmaker()
        async with maker() as session:
            stmt = sa_delete(VectorChunkModel).where(
                VectorChunkModel.document_id == document_id
            )
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount or 0

    async def count(self, document_ids: list[str] | None = None) -> int:
        maker = await get_sessionmaker()
        async with maker() as session:
            stmt = sa_select(VectorChunkModel.id)
            if document_ids is not None:
                if not document_ids:
                    return 0
                stmt = stmt.where(VectorChunkModel.document_id.in_(document_ids))
            rows = (await session.execute(stmt)).scalars().all()
            return len(rows)


_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = SQLiteVectorStore()
    return _store


def reset_vector_store() -> None:
    global _store
    _store = None
