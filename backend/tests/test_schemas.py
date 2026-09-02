"""Unit tests: research schemas, planning invariants, confidence assignment."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.research import (
    Claim,
    ClaimStatus,
    Contradiction,
    Evidence,
    EvidenceConfidence,
    ResearchPlan,
    ResearchState,
    SourceRecord,
    SubQuestion,
    SubQuestionStatus,
)


class TestResearchPlan:
    def test_dedupes_subquestions(self):
        plan = ResearchPlan(
            research_goal="goal",
            subquestions=["What is A?", " what is a? ", "What is B?"],
        )
        assert len(plan.subquestions) == 2

    def test_empty_subquestions_rejected(self):
        with pytest.raises(ValidationError):
            ResearchPlan(research_goal="g", subquestions=["  "])

    def test_question_type_enum(self):
        plan = ResearchPlan(
            research_goal="g", subquestions=["q1"], question_type="comparative"
        )
        assert plan.question_type == "comparative"


class TestSourceRecord:
    def test_rejects_non_http_url(self):
        with pytest.raises(ValidationError):
            SourceRecord(id="s", title="t", url="ftp://bad.example")

    def test_none_url_allowed_for_user_docs(self):
        s = SourceRecord(id="s", title="user doc", url=None)
        assert s.url is None

    def test_retrieval_timestamp_defaults(self):
        s = SourceRecord(id="s", title="t")
        assert s.retrieved_at  # auto-populated


class TestEvidence:
    def test_empty_snippet_rejected(self):
        with pytest.raises(ValidationError):
            Evidence(id="e", claim_summary="c", source_id="s", snippet="  ")

    def test_snippet_clipped_to_1500(self):
        e = Evidence(id="e", claim_summary="c", source_id="s", snippet="x" * 5000)
        assert len(e.snippet) == 1500


class TestResearchState:
    def test_query_dedupe_memory(self):
        state = ResearchState(session_id="x", question="q")
        from app.schemas.research import SearchQueryRecord

        state.executed_queries.append(SearchQueryRecord(query="RAG evaluation"))
        assert state.query_already_executed("rag evaluation") is True
        assert state.query_already_executed("RAG EVALUATION") is True
        assert state.query_already_executed("vector db") is False

    def test_next_subquestion_returns_pending(self):
        state = ResearchState(session_id="x", question="q")
        state.subquestions = [
            SubQuestion(id="sq1", text="a", status=SubQuestionStatus.ANSWERED),
            SubQuestion(id="sq2", text="b"),
        ]
        nxt = state.next_subquestion_to_research()
        assert nxt is not None and nxt.id == "sq2"

    def test_budget_snapshot_counts(self):
        state = ResearchState(session_id="x", question="q")
        snap = state.budget_snapshot()
        assert snap["searches"] == 0 and snap["sources"] == 0


class TestConfidenceRules:
    """Documented confidence assignment (docs/confidence.md)."""

    def test_claim_defaults_insufficient(self):
        c = Claim(id="c", text="t")
        assert c.status == ClaimStatus.INSUFFICIENT_EVIDENCE

    def test_evidence_confidence_categories(self):
        assert EvidenceConfidence.HIGH.value == "HIGH_EVIDENCE"
        assert EvidenceConfidence.INSUFFICIENT.value == "INSUFFICIENT_EVIDENCE"


class TestContradiction:
    def test_holds_both_sides(self):
        c = Contradiction(
            id="x", topic="t",
            claim_a="A", source_id_a="s1",
            claim_b="B", source_id_b="s2",
        )
        assert c.source_id_a != c.source_id_b
