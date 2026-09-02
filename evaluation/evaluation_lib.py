"""Evaluation framework — measures real metrics on a curated dataset.

Metrics (all genuinely computed, never fabricated):

  Retrieval:
    * Recall@K    — fraction of known-relevant chunks retrieved in top-K
    * Precision@K — fraction of top-K retrieved that are relevant

  RAG (heuristic, deterministic — documented in docs/evaluation.md):
    * Context relevance — keyword overlap between query and retrieved context
    * Faithfulness      — evidence snippets grounded in source text (the same
                          grounding guard used in production)
    * Answer relevance  — question-term coverage in the generated answer

  Research pipeline (end-to-end, offline providers):
    * Source quality    — mean trust score of cited sources
    * Citation integrity — all report evidence ids exist & are grounded
    * Claim support      — no claim reported as SUPPORTED without evidence
    * Completeness      — subquestions answered ratio
    * Honesty           — low-evidence questions produce INSUFFICIENT/LOW
                          confidence rather than fabricated conclusions

Run:  python evaluation/run_evaluation.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.core.config import get_settings  # noqa: E402


@dataclass
class RetrievalCase:
    query: str
    relevant_doc_ids: set[str]  # document ids known relevant


@dataclass
class RagCase:
    question: str
    expected_topics: list[str]  # terms that should appear in a good answer


@dataclass
class ResearchCase:
    question: str
    depth: str
    expect_answerable: bool


@dataclass
class EvalResult:
    name: str
    value: float
    detail: dict = field(default_factory=dict)


def recall_at_k(retrieved: list[str], relevant: set[str]) -> float | None:
    if not relevant:
        return None
    hits = sum(1 for r in retrieved if r in relevant)
    return hits / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str]) -> float | None:
    if not retrieved:
        return None
    hits = sum(1 for r in retrieved if r in relevant)
    return hits / len(retrieved)


# ---------------------------------------------------------------------------
# Curated evaluation dataset (small, hand-checked)
# ---------------------------------------------------------------------------
def retrieval_cases() -> list[RetrievalCase]:
    """Queries with known-relevant documents in the offline corpus."""
    corpus = json.loads(
        (Path(__file__).parent.parent / "backend/app/tools/offline_corpus.json")
        .read_text(encoding="utf-8")
    )
    by_domain = {c["domain"]: c["url"] for c in corpus}
    return [
        RetrievalCase(
            query="ragas reference-free evaluation faithfulness metrics",
            relevant_doc_ids={
                by_domain["arxiv.org"],  # RAGAS + ARES papers both match terms
            },
        ),
        RetrievalCase(
            query="trulens rag triad groundedness feedback functions",
            relevant_doc_ids={by_domain["trulens.org"]},
        ),
        RetrievalCase(
            query="deepeval pytest unit testing llm metrics",
            relevant_doc_ids={by_domain["github.com"]},
        ),
        RetrievalCase(
            query="giskard vulnerability scanning hallucination testing",
            relevant_doc_ids={by_domain["github.com"]},
        ),
        RetrievalCase(
            query="vector database retrieval hit rate mrr evaluation",
            relevant_doc_ids={by_domain["zilliz.com"]},
        ),
    ]


def rag_cases() -> list[RagCase]:
    return [
        RagCase(
            question="What is RAGAS and what does it measure?",
            expected_topics=["ragas", "faithfulness", "relevance"],
        ),
        RagCase(
            question="How does TruLens evaluate RAG applications?",
            expected_topics=["trulens", "groundedness", "relevance"],
        ),
        RagCase(
            question="What is DeepEval used for?",
            expected_topics=["deepeval", "testing", "metrics"],
        ),
    ]


def research_cases() -> list[ResearchCase]:
    return [
        ResearchCase(
            question=(
                "Compare RAGAS, TruLens and DeepEval for evaluating a RAG-based "
                "customer support assistant and recommend the most suitable."
            ),
            depth="standard",
            expect_answerable=True,
        ),
        ResearchCase(
            question=(
                "What is the exact revenue of Acme Corporation in fiscal year 2031?"
            ),
            depth="quick",
            expect_answerable=False,  # unknowable — must be reported as insufficient
        ),
    ]
