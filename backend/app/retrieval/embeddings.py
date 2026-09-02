"""Embedding abstraction.

Two providers:
  * MockEmbeddingProvider — deterministic hashing-based embeddings (offline,
    zero network). Not semantically meaningful, but stable, fast, and
    sufficient for plumbing, tests, and offline demos.
  * OpenAIEmbeddingProvider — real embeddings via the OpenAI-compatible API.

The interface is async and batched; the vector store only requires that the
same provider be used for indexing and querying.
"""
from __future__ import annotations

import abc
import hashlib
import math

import httpx
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.errors import PlatformError
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingProvider(abc.ABC):
    """Provider-agnostic embedding interface."""

    name: str = "abstract"
    dim: int = 0

    @abc.abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

    async def embed_query(self, text: str) -> list[float]:
        vecs = await self.embed_texts([text])
        return vecs[0]


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic feature-hashing embeddings.

    Each text is tokenized; token buckets get weights (with log-scaled
    sublinear term frequency). Documents sharing tokens land near each other,
    which gives the retriever meaningful behavior offline.
    """

    name = "mock"
    dim = 256

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = [t for t in text.lower().split() if len(t) > 2]
        if not tokens:
            tokens = [text.lower()[:8] or "empty"]
        for token in tokens:
            h = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big")
            idx = h % self.dim
            sign = 1.0 if (h >> 63) & 1 == 0 else -1.0
            vec[idx] += sign * 1.0
        # log tf scaling
        vec = [math.log1p(abs(v)) * (1 if v >= 0 else -1) for v in vec]
        return _normalize(vec)


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embeddings via OpenAI-compatible /embeddings endpoint."""

    name = "openai"

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        *,
        dim: int = 256,
        timeout: int = 30,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.dim = dim
        self.timeout = timeout

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        # batch to keep payloads sane
        for i in range(0, len(texts), 64):
            batch = texts[i : i + 64]
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model, "input": batch},
                )
            if resp.status_code != 200:
                raise PlatformError(f"embedding API error {resp.status_code}", retryable=True)
            data = resp.json()
            out.extend(item["embedding"] for item in data["data"])
        return out


class EmbeddingService(BaseModel):
    """Facade used by ingestion + retrieval."""

    model_config = {"arbitrary_types_allowed": True}

    provider: object = None

    @classmethod
    def create(cls) -> "EmbeddingService":
        settings = get_settings()
        if settings.EMBEDDING_PROVIDER == "openai":
            provider: EmbeddingProvider = OpenAIEmbeddingProvider(
                model=settings.EMBEDDING_MODEL,
                api_key=settings.LLM_API_KEY,
                dim=settings.EMBEDDING_DIM,
            )
        else:
            provider = MockEmbeddingProvider(dim=settings.EMBEDDING_DIM)
        return cls(provider=provider)

    def embed_texts_sync(self, texts: list[str]) -> list[list[float]]:
        import asyncio

        p: EmbeddingProvider = self.provider  # type: ignore[assignment]
        return asyncio.run(p.embed_texts(texts))

    async def embed(self, texts: list[str]) -> list[list[float]]:
        p: EmbeddingProvider = self.provider  # type: ignore[assignment]
        return await p.embed_texts(texts)

    async def embed_query(self, text: str) -> list[float]:
        p: EmbeddingProvider = self.provider  # type: ignore[assignment]
        return await p.embed_query(text)


_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService.create()
    return _embedding_service


def reset_embedding_service() -> None:
    global _embedding_service
    _embedding_service = None
