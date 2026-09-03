"""Integration tests: hybrid research (user documents + web), RAG wiring,
session-scoped document search, research memory across critic loops."""
from __future__ import annotations

from app.agents.base import AgentContext
from app.core.config import DEPTH_PROFILES
from app.core.llm import MockLLMClient
from app.retrieval.embeddings import get_embedding_service
from app.retrieval.vector_store import get_vector_store
from app.schemas.research import (
    Evidence,
    ResearchDepth,
    ResearchState,
    SourceOrigin,
    SourceRecord,
    SourceType,
    SubQuestion,
)
from app.workflows.orchestrator import Orchestrator
from app.workflows.budget import ResearchBudget


DOC_A_TEXT = (
    "Team Alpha benchmark report. RAGAS was faster to integrate into our "
    "customer support pipeline than TruLens, requiring two days instead of "
    "one week of engineering effort. TruLens provided richer tracing "
    "instrumentation which helped debugging production issues."
)

DOC_B_TEXT = (
    "Unrelated quarterly financial review. Revenue grew quarter over quarter "
    "driven by enterprise contract expansion in the EMEA region. Operating "
    "margins improved due to reduced infrastructure spending on legacy "
    "systems and renegotiated vendor contracts across the organization."
)


async def _ingest_two_documents() -> tuple[str, str]:
    """Ingest two docs into the vector store; return their raw ids."""
    from app.retrieval.chunker import chunk_text, clean_text

    store = get_vector_store()
    embeddings = get_embedding_service()
    ids = []
    for i, text in enumerate((DOC_A_TEXT, DOC_B_TEXT)):
        doc_id = f"hybdocrun{i}"
        chunks = chunk_text(clean_text(text), doc_id)
        vectors = await embeddings.embed([c.text for c in chunks])
        await store.add_chunks(
            [
                {
                    "document_id": doc_id,
                    "chunk_index": c.index,
                    "text": c.text,
                    "embedding": v,
                    "metadata": {"title": f"test doc {i}"},
                }
                for c, v in zip(chunks, vectors)
            ]
        )
        ids.append(doc_id)
    return ids[0], ids[1]


def _doc_source(source_id: str, text: str) -> SourceRecord:
    return SourceRecord(
        id=source_id,
        title=f"document {source_id}",
        url=None,
        source_type=SourceType.USER_DOCUMENT,
        origin=SourceOrigin.USER_DOC,
        content_text=text,
        snippet=text[:300],
        trust_score=0.9,
        relevance_score=1.0,
        authority_score=0.9,
        is_primary=True,
        fetched_ok=True,
        evaluation_notes="User-provided document",
    )


class TestHybridDocumentResearch:
    async def test_document_search_is_session_scoped(self):
        """A session's vector search must only see its own documents."""
        doc_a, doc_b = await _ingest_two_documents()

        state = ResearchState(
            session_id="hyb1",
            question="Which RAG evaluation framework was faster to integrate for customer support?",
            depth=ResearchDepth.QUICK,
        )
        orch = Orchestrator(
            state,
            _registry_without_web(),
            document_sources=[_doc_source(f"doc_{doc_a}", DOC_A_TEXT)],
            budget=_budget(),
        )
        # Call the private helper directly to assert scoping
        sq = SubQuestion(
            id="sqx", text="Which framework was faster to integrate for customer support workloads?"
        )
        hits = await orch._search_user_documents(sq)
        assert all(h["document_id"] == doc_a for h in hits), (
            f"leaked documents into session: {[h['document_id'] for h in hits]}"
        )

    async def test_doc_evidence_has_exact_provenance(self):
        """Doc hits map back to SourceRecords by id; evidence is grounded."""
        doc_a, _doc_b = await _ingest_two_documents()

        state = ResearchState(
            session_id="hyb2",
            question="Which RAG evaluation framework was faster to integrate for customer support?",
            depth=ResearchDepth.QUICK,
        )
        orch = Orchestrator(
            state,
            _registry_without_web(),
            document_sources=[_doc_source(f"doc_{doc_a}", DOC_A_TEXT)],
            budget=_budget(),
        )
        final = await orch.run()
        doc_evidence = [
            e for e in final.evidence.values() if e.origin == SourceOrigin.USER_DOC
        ]
        assert doc_evidence, "user-document evidence was dropped"
        for ev in doc_evidence:
            assert ev.source_id == f"doc_{doc_a}"
            assert ev.origin == SourceOrigin.USER_DOC
            # every doc-evidence snippet is grounded in the document text
            from app.agents.evidence import _grounded

            assert _grounded(ev.snippet, DOC_A_TEXT)

    async def test_untitled_documents_still_produce_evidence(self):
        """The id-mapping must not depend on document titles."""
        doc_a, _ = await _ingest_two_documents()
        # NOTE: title in vector metadata is 'test doc 0' but the SourceRecord
        # title is 'document doc_...' — old title-matching would fail here.
        state = ResearchState(
            session_id="hyb3",
            question="Which framework was faster to integrate?",
            depth=ResearchDepth.QUICK,
        )
        orch = Orchestrator(
            state,
            _registry_without_web(),
            document_sources=[_doc_source(f"doc_{doc_a}", DOC_A_TEXT)],
            budget=_budget(),
        )
        final = await orch.run()
        assert any(
            e.origin == SourceOrigin.USER_DOC for e in final.evidence.values()
        )


class TestResearchMemory:
    async def test_no_duplicate_searches_within_session(self):
        """Executed queries are remembered; the same query never runs twice."""
        from app.schemas.research import SearchQueryRecord

        state = ResearchState(
            session_id="mem1", question="RAG evaluation approaches", depth=ResearchDepth.QUICK
        )
        state.executed_queries.append(
            SearchQueryRecord(query="rag evaluation approaches overview")
        )
        assert state.query_already_executed("RAG evaluation approaches OVERVIEW")

    async def test_followup_queries_flow_from_critic(self):
        """Critic followup queries become next-pass searches (budget-limited)."""
        state = ResearchState(
            session_id="mem2",
            question="Compare approaches to LLM evaluation for support systems.",
            depth=ResearchDepth.STANDARD,
        )
        orch = Orchestrator(state, _registry_without_web(), budget=_budget())
        plan = await _plan(orch)
        assert plan is not None
        # simulate critic output routing
        state.followup_queries = ["llm evaluation production case studies"]
        await orch._research_pass()
        assert not state.followup_queries, "followups were not consumed"


def _registry_without_web():
    """Registry with tools that work fully offline (no network tools)."""
    from app.tools import build_tool_registry

    reg = build_tool_registry()
    # Drop network tools so this test is hermetic.
    from app.tools.base import BaseTool, ToolRegistry

    fresh = ToolRegistry()
    for tool in reg.all():
        if tool.name in ("web_fetch",):
            continue
        fresh.register(tool)
    return fresh


def _budget() -> ResearchBudget:
    return ResearchBudget(
        max_iterations=1, max_searches=5, max_sources=5,
        max_tokens=10**9, max_runtime_seconds=60,
    )


async def _plan(orch: Orchestrator):
    from app.agents.planner import PlannerAgent

    agent = PlannerAgent(orch.ctx)
    return await agent.run(orch.state)
