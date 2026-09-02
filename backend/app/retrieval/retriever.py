"""The retriever: hybrid search (vector + keyword overlap) with score gating.

Design decisions (documented in docs/retrieval.md):
  * Never blindly return top-k: results below `min_score` are dropped.
  * Keyword overlap re-ranking boosts chunks containing exact query terms
    (compensates for weak offline embeddings).
  * Metadata filtering by document ids enables per-session doc corpora.
  * Every retrieved chunk keeps its document_id + chunk_index so evidence can
    cite a precise location.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.core.logging import get_logger
from app.retrieval.embeddings import EmbeddingService, get_embedding_service
from app.retrieval.vector_store import StoredChunk, VectorStore, get_vector_store

logger = get_logger(__name__)


@dataclass
class RetrievedChunk(StoredChunk):
    keyword_score: float = 0.0
    combined_score: float = 0.0


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        embeddings: EmbeddingService,
        *,
        min_score: float = 0.05,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> None:
        self.store = store
        self.embeddings = embeddings
        self.min_score = min_score
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        document_ids: list[str] | None = None,
        fetch_multiplier: int = 3,
    ) -> list[RetrievedChunk]:
        """Retrieve and re-rank chunks for a query.

        fetch_multiplier: we fetch top_k*multiplier candidates from the vector
        store, then re-rank and trim — a cheap approximation of hybrid search.
        """
        query_vec = await self.embeddings.embed_query(query)

        candidates = await self.store.query(
            query_vec,
            top_k=top_k * fetch_multiplier,
            document_ids=document_ids,
            min_score=0.0,  # gate after re-ranking, not before
        )
        if not candidates:
            return []

        q_terms = _terms(query)
        reranked: list[RetrievedChunk] = []
        for c in candidates:
            kw = _keyword_score(c.text, q_terms)
            combined = self.vector_weight * c.score + self.keyword_weight * kw
            reranked.append(
                RetrievedChunk(
                    id=c.id,
                    document_id=c.document_id,
                    text=c.text,
                    score=c.score,
                    metadata=c.metadata,
                    keyword_score=kw,
                    combined_score=combined,
                )
            )
        reranked.sort(key=lambda r: r.combined_score, reverse=True)

        gated = [r for r in reranked if r.combined_score >= self.min_score]
        dropped = len(reranked) - len(gated)
        if dropped:
            logger.info("retrieval_gated", dropped=dropped, kept=len(gated))
        return gated[:top_k]


def _terms(text: str) -> set[str]:
    return {t for t in text.lower().replace("[^a-z0-9 ]", "").split() if len(t) > 2}


def _keyword_score(text: str, query_terms: set[str]) -> float:
    if not query_terms:
        return 0.0
    text_terms = _terms(text)
    if not text_terms:
        return 0.0
    overlap = len(query_terms & text_terms)
    return overlap / len(query_terms)


def get_retriever() -> Retriever:
    return Retriever(get_vector_store(), get_embedding_service())
