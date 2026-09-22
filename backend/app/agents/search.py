"""Search Agent — generates effective, non-redundant queries per subquestion."""
from __future__ import annotations

from pydantic import AliasChoices, BaseModel, Field, model_validator

from app.agents.base import AgentContext, BaseAgent
from app.schemas.research import ResearchState, SubQuestion


def _accept_bare_list(key: str):
    def _before(data: object) -> object:
        if isinstance(data, list):
            # models often return a bare list of plain strings
            return {key: [{"query": q} if isinstance(q, str) else q for q in data]}
        return data

    return model_validator(mode="before")(_before)


class SearchQueries(BaseModel):
    """Structured output of the Search Agent. Accepts objects or bare strings."""

    queries: list[QuerySpec] = Field(default_factory=list, max_length=10)
    _wrap = _accept_bare_list("queries")


class QuerySpec(BaseModel):
    query: str = Field(min_length=3, max_length=250)
    reasoning: str = Field(default="", validation_alias=AliasChoices("reasoning", "reason", "rationale"))
    site: str | None = None  # restrict to a domain (optional)
    recency_months: int | None = Field(default=None, ge=1, le=60)


class SearchAgent(BaseAgent):
    name = "search"
    prompt_name = "search"

    async def run(self, state: ResearchState, subquestion: SubQuestion) -> SearchQueries:
        executed = [r.query for r in state.executed_queries]
        user_payload = self._prompt.render_user(
            subquestion=subquestion.text,
            context=state.question[:300],
            executed="; ".join(executed[-15:]) or "none",
            n_queries=self.ctx.profile["queries_per_subquestion"],
        )
        out = await self._structured(user_payload, SearchQueries)

        # Dedupe against executed queries + within batch (research memory)
        seen = {r.query.strip().lower() for r in state.executed_queries}
        unique: list[QuerySpec] = []
        for q in out.queries:
            key = q.query.strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(q)
        n_allowed = self.ctx.profile["queries_per_subquestion"]
        out.queries = unique[:n_allowed]

        await self.emit(
            "info",
            f"Generated {len(out.queries)} queries for subquestion: {subquestion.text[:80]}",
            queries=len(out.queries),
        )
        return out
