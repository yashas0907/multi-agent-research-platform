"""Document ingestion service.

Pipeline: upload bytes → parse → clean → chunk → metadata → embeddings →
vector store. Uploads are size-limited, extension-allowlisted, and parsed
defensively (see retrieval/parser.py). Ingestion failures are recorded on the
document row — never silently swallowed.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

from sqlalchemy import select as sa_select

from app.core.config import get_settings
from app.core.database import get_sessionmaker
from app.core.errors import DocumentIngestError, NotFoundError
from app.core.logging import get_logger
from app.models.database import DocumentModel, VectorChunkModel
from app.retrieval.chunker import Chunk, chunk_text, clean_text
from app.retrieval.embeddings import get_embedding_service
from app.retrieval.parser import ParsedDocument, parse_document
from app.retrieval.vector_store import get_vector_store

logger = get_logger(__name__)


@dataclass
class DocumentInfo:
    id: str
    filename: str
    title: str | None
    author: str | None
    page_count: int | None
    status: str
    chunks: int
    size_bytes: int
    error: str | None


@dataclass
class ChunkInfo:
    id: str
    chunk_index: int
    text: str


class DocumentService:
    async def ingest(self, filename: str, raw: bytes) -> DocumentInfo:
        settings = get_settings()
        if len(raw) > settings.max_upload_bytes:
            raise DocumentIngestError(
                f"file exceeds max upload size ({settings.MAX_UPLOAD_SIZE_MB} MB)"
            )
        if not raw.strip():
            raise DocumentIngestError("file is empty")

        maker = await get_sessionmaker()
        async with maker() as session:
            doc = DocumentModel(filename=filename[:250], size_bytes=len(raw), status="processing")
            session.add(doc)
            await session.commit()
            await session.refresh(doc)
            doc_id = doc.id

        try:
            parsed = parse_document(filename, raw)
            chunks = self._make_chunks(parsed, doc_id)
            n_stored = await self._embed_and_store(doc_id, chunks, parsed)
            async with maker() as session:
                doc = await session.get(DocumentModel, doc_id)
                doc.status = "ready"
                doc.chunks = n_stored
                doc.title = parsed.title
                doc.author = parsed.author
                doc.page_count = parsed.page_count
                doc.content_type = parsed.content_type
                await session.commit()
            logger.info("document_ingested", doc_id=doc_id, chunks=n_stored, filename=filename)
            return DocumentInfo(
                id=doc_id, filename=filename, title=parsed.title, author=parsed.author,
                page_count=parsed.page_count, status="ready", chunks=n_stored,
                size_bytes=len(raw), error=None,
            )
        except DocumentIngestError as exc:
            await self._fail(doc_id, str(exc))
            raise
        except Exception as exc:  # noqa: BLE001
            await self._fail(doc_id, f"ingestion crashed: {exc}")
            raise DocumentIngestError(f"ingestion failed: {exc}") from exc

    def _make_chunks(self, parsed: ParsedDocument, doc_id: str) -> list[Chunk]:
        text = clean_text(parsed.text)
        if len(text.strip()) < 20:
            raise DocumentIngestError("document contains no usable text")
        return chunk_text(text, doc_id)

    async def _embed_and_store(
        self, doc_id: str, chunks: list[Chunk], parsed: ParsedDocument
    ) -> int:
        if not chunks:
            raise DocumentIngestError("chunking produced no chunks")
        embeddings = get_embedding_service()
        vectors = await embeddings.embed([c.text for c in chunks])
        store = get_vector_store()
        added = await store.add_chunks(
            [
                {
                    "document_id": doc_id,
                    "chunk_index": c.index,
                    "text": c.text,
                    "embedding": vec,
                    "metadata": {
                        "title": parsed.title or f"document {doc_id}",
                        "filename": None,
                        "content_type": parsed.content_type,
                    },
                }
                for c, vec in zip(chunks, vectors)
            ]
        )
        return added

    async def _fail(self, doc_id: str, message: str) -> None:
        maker = await get_sessionmaker()
        async with maker() as session:
            doc = await session.get(DocumentModel, doc_id)
            if doc is not None:
                doc.status = "failed"
                doc.error = message[:500]
                await session.commit()

    async def list_documents(self) -> list[DocumentInfo]:
        maker = await get_sessionmaker()
        async with maker() as session:
            rows = (
                await session.execute(
                    sa_select(DocumentModel).order_by(DocumentModel.created_at.desc())
                )
            ).scalars().all()
            return [
                DocumentInfo(
                    id=r.id, filename=r.filename, title=r.title, author=r.author,
                    page_count=r.page_count, status=r.status, chunks=r.chunks,
                    size_bytes=r.size_bytes, error=r.error,
                )
                for r in rows
            ]

    async def get_document(self, doc_id: str) -> DocumentInfo:
        maker = await get_sessionmaker()
        async with maker() as session:
            doc = await session.get(DocumentModel, doc_id)
        if doc is None:
            raise NotFoundError(f"document {doc_id} not found")
        return DocumentInfo(
            id=doc.id, filename=doc.filename, title=doc.title, author=doc.author,
            page_count=doc.page_count, status=doc.status, chunks=doc.chunks,
            size_bytes=doc.size_bytes, error=doc.error,
        )

    async def get_document_chunks(self, doc_id: str, limit: int = 50) -> list[ChunkInfo]:
        maker = await get_sessionmaker()
        async with maker() as session:
            rows = (
                await session.execute(
                    sa_select(VectorChunkModel)
                    .where(VectorChunkModel.document_id == doc_id)
                    .order_by(VectorChunkModel.chunk_index)
                    .limit(limit)
                )
            ).scalars().all()
            return [ChunkInfo(id=r.id, chunk_index=r.chunk_index, text=r.text) for r in rows]

    async def delete_document(self, doc_id: str) -> int:
        maker = await get_sessionmaker()
        async with maker() as session:
            doc = await session.get(DocumentModel, doc_id)
            if doc is None:
                raise NotFoundError(f"document {doc_id} not found")
            await session.delete(doc)
            await session.commit()
        removed = await get_vector_store().delete_document(doc_id)
        logger.info("document_deleted", doc_id=doc_id, vectors_removed=removed)
        return removed
