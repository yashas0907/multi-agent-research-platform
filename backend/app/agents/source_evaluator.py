"""Source Evaluation Agent.

Evaluates candidate sources on relevance, authority, recency, and primary/
secondary status. The LLM provides relevance/authority judgments; recency and
trust composites are computed by deterministic code from verifiable metadata.
Methodology documented in docs/source_evaluation.md.
"""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field, model_validator

from app.agents.base import BaseAgent, new_id
from app.prompts.library import get_prompt_library
from app.schemas.research import (
    SourceRecord,
    SourceType,
    SubQuestion,
)


def _accept_bare_list(key: str):
    def _before(data: object) -> object:
        if isinstance(data, list):
            return {key: data}
        return data

    return model_validator(mode="before")(_before)
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


class BatchSourceEval(BaseModel):
    """One LLM call evaluates a whole batch of candidate sources."""

    evals: list[BatchEvalItem] = Field(default_factory=list, max_length=20)
    _wrap = _accept_bare_list("evals")


class BatchEvalItem(BaseModel):
    index: int = Field(ge=0, le=100)
    relevance: float = Field(ge=0.0, le=1.0, default=0.5)
    authority: float = Field(ge=0.0, le=1.0, default=0.5)
    is_primary: bool = False
    source_type: str = "other"
    discard: bool = False
    reasoning: str = ""


class SourceEvaluatorAgent(BaseAgent):
    name = "source_evaluator"
    prompt_name = "source_evaluator"

    async def evaluate_deterministic(
        self,
        candidate: SourceRecord,
        subquestion: SubQuestion,
    ) -> SourceRecord:
        """Zero-LLM evaluation: keyword relevance + domain tiers + recency.

        Used by quick/standard depths to preserve the LLM token budget for
        fact-checking and synthesis (where judgment matters most). Same
        composite scoring as the LLM path; relevance is keyword-overlap.
        """
        terms = {t for t in subquestion.text.lower().split() if len(t) > 2}
        text = (candidate.title + " " + (candidate.snippet or "")).lower()
        if terms:
            overlap = sum(1 for t in terms if t in text) / len(terms)
        else:
            overlap = 0.5
        relevance = min(1.0, 0.4 + 0.6 * overlap)
        recency = _recency_score(candidate.published_at)
        domain_trust = _domain_trust(candidate.domain)
        candidate.relevance_score = round(relevance, 3)
        candidate.authority_score = domain_trust
        candidate.recency_score = recency
        candidate.trust_score = round(
            0.45 * relevance + 0.3 * domain_trust + 0.25 * recency, 3
        )
        candidate.evaluation_notes = (
            "Deterministic evaluation (keyword relevance + domain trust + recency; no LLM)"
        )
        return candidate

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
        return self._apply(candidate, ev)

    async def evaluate_batch(
        self,
        candidates: list[SourceRecord],
        subquestion: SubQuestion,
    ) -> list[SourceRecord]:
        """Evaluate up to 8 candidates in ONE LLM call (free-tier economics)."""
        if not candidates:
            return []
        if len(candidates) == 1:
            return [await self.evaluate(candidates[0], subquestion)]

        batch_prompt = get_prompt_library().get("source_evaluator_batch")
        block = "\n".join(
            f"[{i}] title: {c.title}\n    url: {c.url or 'n/a'}\n"
            f"    snippet: {(c.snippet or '')[:180]}"
            for i, c in enumerate(candidates[:8])
        )
        user_payload = batch_prompt.render_user(
            subquestion=subquestion.text, candidates_block=block
        )
        messages = [
            {"role": "system", "content": batch_prompt.system},
            {"role": "user", "content": user_payload},
        ]
        out = await self.ctx.llm.complete_structured(messages, BatchSourceEval)

        by_index = {e.index: e for e in out.evals}
        results: list[SourceRecord] = []
        for i, candidate in enumerate(candidates[:8]):
            ev = by_index.get(i)
            if ev is None:
                # missing eval → neutral deterministic defaults, never guessed high
                ev = BatchEvalItem(index=i)
            results.append(self._apply(candidate, ev, batch_reason=True))
        return results

    def _apply(self, candidate: SourceRecord, ev: SourceEval | BatchEvalItem, batch_reason: bool = False) -> SourceRecord:
        recency = _recency_score(candidate.published_at)
        domain_trust = _domain_trust(candidate.domain)

        candidate.relevance_score = ev.relevance
        candidate.authority_score = max(ev.authority, domain_trust)
        candidate.recency_score = recency
        candidate.trust_score = round(
            0.45 * ev.relevance + 0.3 * max(ev.authority, domain_trust) + 0.25 * recency,
            3,
        )
        candidate.is_primary = ev.is_primary
        try:
            candidate.source_type = SourceType(ev.source_type)
        except ValueError:
            candidate.source_type = SourceType.OTHER
        candidate.evaluation_notes = (
            ("Batch evaluation: " if batch_reason else "") + (ev.reasoning or "")[:380]
        ) or None

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
