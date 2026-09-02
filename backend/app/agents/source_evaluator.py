"""Source Evaluation Agent.

Evaluates candidate sources on relevance, authority, recency, and primary/
secondary status. The LLM provides relevance/authority judgments; recency and
trust composites are computed by deterministic code from verifiable metadata.
Methodology documented in docs/source_evaluation.md.
"""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from app.agents.base import BaseAgent, new_id
from app.schemas.research import (
    SourceRecord,
    SourceType,
    SubQuestion,
)

# Domain reputation tiers used by the deterministic component of scoring.
HIGH_TRUST_DOMAINS = {
    "arxiv.org", "dl.acm.org", "ieee.org", "nature.com", "science.org",
    "docs.python.org", "learn.microsoft.com", "developer.mozilla.org",
    "docs.smith.langchain.com", "python.langchain.com", "openai.com",
    "ai.meta.com", "ai.googleblog.com", "research.google", "github.com",
}
MEDIUM_TRUST_DOMAINS = {
    "medium.com", "dev.to", "stackoverflow.com", "huggingface.co",
    "trulens.org", "llamaindex.ai", "zilliz.com", "pinecone.io",
    "wandb.ai", "research.aimultiple.com",
}


class SourceEval(BaseModel):
    relevance: float = Field(ge=0.0, le=1.0, default=0.5)
    authority: float = Field(ge=0.0, le=1.0, default=0.5)
    is_primary: bool = False
    source_type: str = "other"
    reasoning: str = ""
    discard: bool = False


class SourceEvaluatorAgent(BaseAgent):
    name = "source_evaluator"
    prompt_name = "source_evaluator"

    async def evaluate(
        self,
        candidate: SourceRecord,
        subquestion: SubQuestion,
    ) -> SourceRecord:
        user_payload = self._prompt.render_user(
            subquestion=subquestion.text,
            title=candidate.title,
            url=candidate.url or "n/a",
            snippet=(candidate.snippet or "")[:500],
        )
        ev = await self._structured(user_payload, SourceEval)

        recency = _recency_score(candidate.published_at)
        domain_trust = _domain_trust(candidate.domain)

        candidate.relevance_score = ev.relevance
        candidate.authority_score = max(ev.authority, domain_trust)
        candidate.recency_score = recency
        # Composite: relevance weighted highest; sources failing all get dropped.
        candidate.trust_score = round(
            0.45 * ev.relevance + 0.3 * max(ev.authority, domain_trust) + 0.25 * recency,
            3,
        )
        candidate.is_primary = ev.is_primary
        try:
            candidate.source_type = SourceType(ev.source_type)
        except ValueError:
            candidate.source_type = SourceType.OTHER
        candidate.evaluation_notes = ev.reasoning[:400]

        if ev.discard:
            candidate.trust_score = 0.0
        return candidate


def _recency_score(published_at: str | None) -> float:
    """Deterministic recency scoring from *stated* publication date.

    None (unknown date) gets a neutral 0.5 — we never guess dates.
    """
    if not published_at:
        return 0.5
    try:
        published = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    except ValueError:
        return 0.5
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    age_days = (datetime.now(timezone.utc) - published).days
    if age_days <= 180:
        return 1.0
    if age_days <= 365:
        return 0.8
    if age_days <= 730:
        return 0.6
    if age_days <= 1460:
        return 0.4
    return 0.2


def _domain_trust(domain: str | None) -> float:
    if not domain:
        return 0.2
    d = domain.lower().lstrip("www.")
    if any(d == h or d.endswith("." + h) for h in HIGH_TRUST_DOMAINS):
        return 0.9
    if any(d == h or d.endswith("." + h) for h in MEDIUM_TRUST_DOMAINS):
        return 0.7
    return 0.3
