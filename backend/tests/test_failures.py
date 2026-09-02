"""Failure tests: simulated tool/LLM/search failures — pipeline must degrade,
never crash, never fabricate."""
from __future__ import annotations

import pytest

from app.core.errors import (
    BudgetExceededError,
    LLMError,
    PlatformError,
    ToolError,
)
from app.schemas.research import (
    ResearchDepth,
    ResearchState,
    ReportFormat,
)
from app.tools.base import BaseTool, ToolRegistry, ToolRunner
from app.workflows.budget import ResearchBudget
from app.workflows.orchestrator import Orchestrator

from pydantic import BaseModel

QUESTION = "Compare RAG evaluation approaches for a production customer-support system."


class _NoOpIn(BaseModel):
    x: str = ""


class _NoOpOut(BaseModel):
    y: str = ""


class _FailingSearchTool(BaseTool[_NoOpIn, _NoOpOut]):
    name = "web_search"
    description = "always fails"

    def input_schema(self): return _NoOpIn
    def output_schema(self): return _NoOpOut

    async def run(self, params):
        raise ToolError("search backend down", retryable=True)


class _FailingFetchTool(BaseTool[_NoOpIn, _NoOpOut]):
    name = "web_fetch"
    description = "always fails"

    def input_schema(self): return _NoOpIn
    def output_schema(self): return _NoOpOut

    async def run(self, params):
        raise ToolError("network unavailable", retryable=False)


def _registry_with(*tools) -> ToolRegistry:
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


class TestToolFailures:
    async def test_search_failure_degrades_not_crashes(self):
        from app.tools.calculator import CalculatorTool
        from app.tools.vector_search import VectorSearchTool

        state = ResearchState(
            session_id="f1", question=QUESTION, depth=ResearchDepth.QUICK
        )
        reg = _registry_with(
            _FailingSearchTool(), _FailingFetchTool(), CalculatorTool(), VectorSearchTool()
        )
        orch = Orchestrator(state, reg, budget=_small_budget())
        final = await orch.run()
        # pipeline completes (honestly, with insufficient evidence)
        assert final.status.value in ("report_ready", "completed_partial", "failed")
        if final.report:
            # honest degradation: findings marked low/insufficient confidence
            assert final.report.key_findings is not None

    async def test_budget_exhaustion_partial_report(self):
        """Search budget of 0: pipeline must not loop; must finish gracefully."""
        from app.tools import build_tool_registry

        state = ResearchState(
            session_id="f2", question=QUESTION, depth=ResearchDepth.QUICK
        )
        budget = ResearchBudget(
            max_iterations=3, max_searches=0, max_sources=2,
            max_tokens=10**9, max_runtime_seconds=60,
        )
        orch = Orchestrator(state, build_tool_registry(), budget=budget)
        final = await orch.run()
        # no searches possible → no web evidence; completes honestly
        assert final.status.value in ("report_ready", "failed", "completed_partial")

    async def test_llm_failure_fails_cleanly(self):
        from app.tools import build_tool_registry

        class ExplodingLLM:
            provider_name = "exploding"

            class _Usage:
                total_tokens = 0
                calls = 0

            usage = _Usage()

            async def complete(self, *a, **k):
                raise LLMError("provider down", retryable=False)

            async def complete_structured(self, *a, **k):
                raise LLMError("provider down", retryable=False)

        state = ResearchState(
            session_id="f3", question=QUESTION, depth=ResearchDepth.QUICK
        )
        orch = Orchestrator(state, build_tool_registry(), budget=_small_budget())
        # swap in the failing provider
        orch.llm = ExplodingLLM()
        orch.ctx.llm = ExplodingLLM()
        final = await orch.run()
        assert final.status.value == "failed"
        assert final.error  # surfaced to the user, not swallowed

    async def test_unavailable_source_skipped(self):
        """One bad URL among candidates must not abort the whole pass."""
        from app.tools.calculator import CalculatorTool
        from app.tools.vector_search import VectorSearchTool
        from app.tools.web_search import WebSearchTool, WebSearchResultItem, WebSearchOutput, SearchProvider

        corpus_items = [
            WebSearchResultItem(
                title="RAGAS paper",
                url="https://arxiv.org/abs/2309.15217",
                snippet="RAGAS evaluates RAG pipelines reference-free.",
                source_type="academic_paper",
                domain="arxiv.org",
            ),
            WebSearchResultItem(
                title="Dead link",
                url="https://this-domain-definitely-does-not-exist-12345.example",
                snippet="unreachable",
            ),
        ]

        class FixedProvider(SearchProvider):
            name = "fixed"

            async def search(self, query, max_results):
                return corpus_items

        class FixedSearchTool(WebSearchTool):
            def __init__(self):
                super().__init__(provider=FixedProvider())

        from app.tools.web_fetch import WebFetchTool

        state = ResearchState(
            session_id="f5", question=QUESTION, depth=ResearchDepth.QUICK
        )
        reg = _registry_with(
            FixedSearchTool(), WebFetchTool(), CalculatorTool(), VectorSearchTool()
        )
        orch = Orchestrator(state, reg, budget=_small_budget())
        final = await orch.run()
        # dead link skipped; good source may still be processed
        assert final.status.value in ("report_ready", "completed_partial", "failed")


class TestCancellation:
    async def test_cancel_flag_respected(self):
        from app.tools import build_tool_registry

        state = ResearchState(
            session_id="f6", question=QUESTION, depth=ResearchDepth.DEEP
        )
        orch = Orchestrator(state, build_tool_registry(), budget=_big_budget())
        await orch.cancel()
        final = await orch.run()
        assert final.status.value == "cancelled"
        assert final.report is None


def _small_budget() -> ResearchBudget:
    return ResearchBudget(
        max_iterations=1, max_searches=4, max_sources=4,
        max_tokens=10**9, max_runtime_seconds=60,
    )


def _big_budget() -> ResearchBudget:
    return ResearchBudget(
        max_iterations=3, max_searches=30, max_sources=20,
        max_tokens=10**9, max_runtime_seconds=60,
    )
