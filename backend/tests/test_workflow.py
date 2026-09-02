"""Integration tests: full orchestrator pipeline, LLM structured outputs, security."""
from __future__ import annotations

import pytest

from app.agents.base import AgentContext
from app.agents.evidence import EvidenceAgent
from app.agents.planner import PlannerAgent
from app.core.config import DEPTH_PROFILES
from app.core.llm import MockLLMClient, extract_json
from app.core.errors import StructuredOutputError
from app.schemas.research import (
    ResearchDepth,
    ResearchState,
    ReportFormat,
    SourceOrigin,
    SourceRecord,
    SourceType,
    SubQuestion,
)
from app.workflows.orchestrator import Orchestrator
from app.workflows.budget import ResearchBudget


def _ctx() -> AgentContext:
    return AgentContext(
        llm=MockLLMClient(), event_emitter=_noop, profile=DEPTH_PROFILES["standard"]
    )


async def _noop(payload: dict) -> None:
    pass


QUESTION = (
    "Compare RAGAS, TruLens and DeepEval for RAG evaluation and recommend "
    "one for a production customer-support system."
)


class TestLLMStructuredOutput:
    async def test_valid_json_extracted_from_fences(self):
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
        assert extract_json('Sure! Here: {"a": [1,2]} hope that helps') == {"a": [1, 2]}

    async def test_invalid_json_raises(self):
        with pytest.raises(ValueError):
            extract_json("no json here at all")

    async def test_repair_pass_on_bad_schema(self):
        class FailingMock(MockLLMClient):
            def __init__(self):
                super().__init__()
                self.calls = 0

            async def complete(self, messages, *, temperature=None, max_tokens=None):
                self.calls += 1
                if self.calls == 1:
                    return '{"wrong_field": true}'  # valid JSON, wrong schema
                return '{"research_goal": "g", "subquestions": ["q1"]}'

        from app.schemas.research import ResearchPlan

        llm = FailingMock()
        ctx = AgentContext(llm=llm, event_emitter=_noop, profile=DEPTH_PROFILES["quick"])
        agent = PlannerAgent(ctx)
        plan = await agent.run(
            ResearchState(session_id="x", question="q", depth=ResearchDepth.QUICK)
        )
        assert plan.subquestions == ["q1"]
        assert llm.calls == 2  # repair pass happened

    async def test_structured_output_error_after_repair(self):
        class AlwaysBad(MockLLMClient):
            async def complete(self, messages, *, temperature=None, max_tokens=None):
                return "not json at all"

        from app.schemas.research import ResearchPlan

        llm = AlwaysBad()
        with pytest.raises(StructuredOutputError):
            await llm.complete_structured(
                [{"role": "system", "content": "TASK:planner"}, {"role": "user", "content": "q"}],
                ResearchPlan,
            )


class TestPlannerAgent:
    async def test_depth_profile_enforced(self):
        ctx = AgentContext(
            llm=MockLLMClient(),
            event_emitter=_noop,
            profile=DEPTH_PROFILES["quick"],  # max 3 subquestions
        )
        state = ResearchState(session_id="s", question=QUESTION, depth=ResearchDepth.QUICK)
        plan = await PlannerAgent(ctx).run(state)
        assert len(plan.subquestions) <= 3
        assert len(state.subquestions) == len(plan.subquestions)


class TestEvidenceAgent:
    async def test_ungrounded_evidence_dropped(self):
        src = SourceRecord(
            id="src1",
            title="RAGAS paper",
            url="https://arxiv.org/abs/2309.15217",
            origin=SourceOrigin.WEB,
            source_type=SourceType.ACADEMIC_PAPER,
            content_text=(
                "We introduce RAGAS (Retrieval Augmented Generation Assessment), "
                "a framework for reference-free evaluation of RAG pipelines. "
                "RAGAS measures faithfulness, answer relevance, and context relevance "
                "using an LLM-based judge without ground truth answers."
            ),
        )
        sq = SubQuestion(id="sq1", text="What are the leading approaches to RAG evaluation?")
        state = ResearchState(session_id="s", question=QUESTION, depth=ResearchDepth.STANDARD)
        agent = EvidenceAgent(_ctx())
        items = await agent.run(state, src, sq)
        assert len(items) >= 1
        # every returned snippet must be grounded in the source text
        for it in items:
            assert _snip_in(it.snippet, src.content_text)


def _snip_in(snippet: str, text: str) -> bool:
    from app.agents.evidence import _grounded

    return _grounded(snippet, text)


class TestFullPipeline:
    async def test_end_to_end_produces_report(self):
        from app.tools import build_tool_registry

        state = ResearchState(
            session_id="e2e1",
            question=QUESTION,
            depth=ResearchDepth.STANDARD,
            requested_format=ReportFormat.COMPARISON,
        )
        budget = ResearchBudget(
            max_iterations=2, max_searches=8, max_sources=6,
            max_tokens=10**9, max_runtime_seconds=120,
        )
        orch = Orchestrator(state, build_tool_registry(), budget=budget)
        final = await orch.run()
        assert final.status.value == "report_ready"
        report = final.report
        assert report is not None
        # structured output contract
        assert report.executive_summary
        assert report.methodology
        assert report.key_findings is not None
        assert report.sources is not None
        assert isinstance(report.confidence_summary, dict)
        # every cited source has provenance metadata
        for s in report.sources:
            assert s.title
            assert s.url is None or s.url.startswith("http")
        # subquestions tracked
        assert len(final.subquestions) >= 3

    async def test_events_emitted_safe(self):
        from app.tools import build_tool_registry

        events: list = []

        async def sink(event) -> None:
            events.append(event)

        state = ResearchState(
            session_id="e2e2",
            question=QUESTION,
            depth=ResearchDepth.QUICK,
        )
        budget = ResearchBudget(
            max_iterations=1, max_searches=4, max_sources=4,
            max_tokens=10**9, max_runtime_seconds=120,
        )
        orch = Orchestrator(state, build_tool_registry(), budget=budget, event_sink=sink)
        await orch.run()
        assert len(events) > 10
        # no secrets/prompts leak into event data
        for e in events:
            blob = str(e.data).lower() + e.message.lower()
            assert "api_key" not in blob
            assert "system prompt" not in blob


class TestDepthProfiles:
    def test_quick_standard_deep_differ(self):
        assert DEPTH_PROFILES["quick"]["max_subquestions"] < DEPTH_PROFILES["standard"]["max_subquestions"]
        assert DEPTH_PROFILES["standard"]["max_subquestions"] < DEPTH_PROFILES["deep"]["max_subquestions"]
        assert DEPTH_PROFILES["deep"]["critic_passes"] > DEPTH_PROFILES["quick"]["critic_passes"]
