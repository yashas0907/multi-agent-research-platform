"""Unit tests: source evaluation (deterministic components), budget logic."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.agents.source_evaluator import _domain_trust, _recency_score
from app.schemas.research import SourceOrigin, SourceRecord, SourceType
from app.workflows.budget import ResearchBudget
from app.schemas.research import ResearchState, ResearchDepth, SearchQueryRecord


class TestRecencyScore:
    def test_recent_scores_high(self):
        recent = datetime.now(timezone.utc) - timedelta(days=30)
        assert _recency_score(recent.isoformat()) == 1.0

    def test_old_scores_low(self):
        old = datetime.now(timezone.utc) - timedelta(days=2000)
        assert _recency_score(old.isoformat()) == 0.2

    def test_unknown_date_is_neutral_never_guessed(self):
        assert _recency_score(None) == 0.5
        assert _recency_score("garbage") == 0.5


class TestDomainTrust:
    def test_arxiv_high(self):
        assert _domain_trust("arxiv.org") == 0.9
        assert _domain_trust("www.arxiv.org") == 0.9

    def test_medium_tier(self):
        assert _domain_trust("medium.com") == 0.7

    def test_unknown_low(self):
        assert _domain_trust("totally-random-site.example") == 0.3
        assert _domain_trust(None) == 0.2


class TestBudget:
    def _state(self, searches: int = 0, tokens: int = 0) -> ResearchState:
        s = ResearchState(session_id="t", question="q", depth=ResearchDepth.STANDARD)
        for _ in range(searches):
            s.executed_queries.append(SearchQueryRecord(query=f"q{_}"))
        s.tokens_used = tokens
        return s

    def test_search_budget(self):
        b = ResearchBudget(max_iterations=3, max_searches=2, max_sources=5,
                           max_tokens=1000, max_runtime_seconds=60)
        assert b.can_search(self._state(searches=1)) is True
        assert b.can_search(self._state(searches=2)) is False

    def test_source_budget(self):
        b = ResearchBudget(max_iterations=3, max_searches=2, max_sources=2,
                           max_tokens=1000, max_runtime_seconds=60)
        s = self._state()
        s.sources["a"] = SourceRecord(id="a", title="t")
        s.sources["b"] = SourceRecord(id="b", title="t")
        assert b.can_add_source(s) is False

    def test_token_budget_raises(self):
        import pytest
        from app.core.errors import BudgetExceededError

        b = ResearchBudget(max_iterations=3, max_searches=2, max_sources=5,
                           max_tokens=100, max_runtime_seconds=60)
        with pytest.raises(BudgetExceededError):
            b.check_hard_limits(self._state(tokens=1000))

    def test_runtime_budget_raises(self):
        import time as _t
        import pytest
        from app.core.errors import BudgetExceededError

        b = ResearchBudget(max_iterations=3, max_searches=2, max_sources=5,
                           max_tokens=1000, max_runtime_seconds=0)
        _t.sleep(0.01)
        with pytest.raises(BudgetExceededError):
            b.check_runtime()

    def test_snapshot_shape(self):
        b = ResearchBudget(max_iterations=3, max_searches=2, max_sources=5,
                           max_tokens=1000, max_runtime_seconds=60)
        snap = b.snapshot(self._state())
        assert snap["max_searches"] == 2
        assert "searches" in snap and "tokens_used" in snap
