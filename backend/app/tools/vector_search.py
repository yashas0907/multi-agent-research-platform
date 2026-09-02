"""Vector search tool — exposes the RAG subsystem to agents via the tool API."""
from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from app.core.errors import ToolError
from app.core.logging import get_logger
from app.retrieval.retriever import get_retriever
from app.tools.base import BaseTool

logger = get_logger(__name__)


class VectorSearchInput(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)
    document_ids: list[str] = Field(default_factory=list)


class VectorSearchHit(BaseModel):
    document_id: str
    chunk_id: str
    text: str
    score: float
    source_title: str | None = None


class VectorSearchOutput(BaseModel):
    query: str
    hits: list[VectorSearchHit] = Field(default_factory=list)
    total: int = 0


class VectorSearchTool(BaseTool[VectorSearchInput, VectorSearchOutput]):
    name: ClassVar[str] = "vector_search"
    description: ClassVar[str] = (
        "Search the user's uploaded document corpus (semantic + keyword). "
        "Returns chunks with document ids and scores."
    )

    def input_schema(self) -> type[VectorSearchInput]:
        return VectorSearchInput

    def output_schema(self) -> type[VectorSearchOutput]:
        return VectorSearchOutput

    async def run(self, params: VectorSearchInput) -> VectorSearchOutput:
        retriever = get_retriever()
        doc_ids = params.document_ids or None
        chunks = await retriever.retrieve(
            params.query, top_k=params.top_k, document_ids=doc_ids
        )
        hits = [
            VectorSearchHit(
                document_id=c.document_id,
                chunk_id=c.id,
                text=c.text[:1200],
                score=round(c.combined_score, 4),
                source_title=c.metadata.get("title"),
            )
            for c in chunks
        ]
        if not hits:
            logger.info("vector_search_empty", query_len=len(params.query))
        return VectorSearchOutput(query=params.query, hits=hits, total=len(hits))
