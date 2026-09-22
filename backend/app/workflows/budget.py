"""Research budget tracker — cost control for agent loops.

Enforces: max iterations, max searches, max sources, max tokens, max runtime.
Every check is deterministic code, never left to model discretion.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.core.config import get_settings
from app.core.errors import BudgetExceededError
from app.schemas.research import ResearchState


@dataclass
class ResearchBudget:
    max_iterations: int
    max_searches: int
    max_sources: int
    max_tokens: int
    max_runtime_seconds: int
    max_llm_calls: int = 60
    started_at: float = field(default_factory=time.monotonic)

    @classmethod
    def from_settings(cls) -> ResearchBudget:
        s = get_settings()
        return cls(
            max_iterations=s.RESEARCH_MAX_ITERATIONS,
            max_searches=s.RESEARCH_MAX_SEARCHES,
            max_sources=s.RESEARCH_MAX_SOURCES,
            max_tokens=s.RESEARCH_MAX_TOKENS,
            max_runtime_seconds=s.RESEARCH_MAX_RUNTIME_SECONDS,
            max_llm_calls=s.RESEARCH_MAX_LLM_CALLS,
        )

    # -- soft checks (agent may still gracefully finish) --
    def can_search(self, state: ResearchState) -> bool:
        return len(state.executed_queries) < self.max_searches

    def can_add_source(self, state: ResearchState) -> bool:
        return len(state.sources) < self.max_sources

    def can_call_llm(self, state: ResearchState) -> bool:
        """Free-tier protection: stop LLM-driven steps when call budget is out."""
        return state.llm_calls < self.max_llm_calls

    # -- hard checks (abort workflow) --
    def check_hard_limits(self, state: ResearchState) -> None:
        self.check_runtime()
        if state.tokens_used > self.max_tokens:
            raise BudgetExceededError(
                f"token budget exhausted ({state.tokens_used}/{self.max_tokens})",
                budget_type="tokens",
                limit=self.max_tokens,
            )
        if state.llm_calls > self.max_llm_calls:
            raise BudgetExceededError(
                f"LLM call budget exhausted ({state.llm_calls}/{self.max_llm_calls})",
                budget_type="llm_calls",
                limit=self.max_llm_calls,
            )

    def check_runtime(self) -> None:
        elapsed = time.monotonic() - self.started_at
        if elapsed > self.max_runtime_seconds:
            raise BudgetExceededError(
                f"runtime budget exhausted ({int(elapsed)}s/{self.max_runtime_seconds}s)",
                budget_type="runtime",
                limit=self.max_runtime_seconds,
            )

    def snapshot(self, state: ResearchState) -> dict:
        return {
            "iteration": state.current_iteration,
            "max_iterations": self.max_iterations,
            "searches": len(state.executed_queries),
            "max_searches": self.max_searches,
            "sources": len(state.sources),
            "max_sources": self.max_sources,
            "tokens_used": state.tokens_used,
            "max_tokens": self.max_tokens,
            "llm_calls": state.llm_calls,
            "max_llm_calls": self.max_llm_calls,
            "runtime_seconds": int(time.monotonic() - self.started_at),
            "max_runtime_seconds": self.max_runtime_seconds,
        }
