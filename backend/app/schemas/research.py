"""Core domain schemas — the typed contracts shared by every agent.

These models ARE the research state. Agents never pass giant prose blobs to
each other; they exchange these structured objects through `ResearchState`.
"""
from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------
def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Source / provenance
# ---------------------------------------------------------------------------
class SourceType(str, enum.Enum):
    WEB_PAGE = "web_page"
    DOCUMENTATION = "documentation"
    ACADEMIC_PAPER = "academic_paper"
    TECHNICAL_BLOG = "technical_blog"
    USER_DOCUMENT = "user_document"
    GOVERNMENT = "government"
    COMPANY_DOC = "company_doc"
    NEWS = "news"
    OTHER = "other"


class SourceOrigin(str, enum.Enum):
    """Where evidence came from — required for the hybrid research layer."""

    WEB = "web"
    USER_DOC = "user_document"
    INTERNAL = "internal"


class SourceRecord(BaseModel):
    """A discovered source with full provenance metadata.

    Policy: never invent metadata. Missing fields stay `None` and are rendered
    as "not available" in the UI.
    """

    id: str
    title: str
    url: str | None = None
    source_type: SourceType = SourceType.OTHER
    origin: SourceOrigin = SourceOrigin.WEB
    # Publication date *as stated by the source* (ISO string) — None if unknown.
    published_at: str | None = None
    retrieved_at: str = Field(default_factory=lambda: utcnow().isoformat())
    domain: str | None = None
    snippet: str | None = None
    # Source evaluation scores (documented methodology — see docs/source_evaluation.md)
    relevance_score: float = Field(ge=0.0, le=1.0, default=0.0)
    authority_score: float = Field(ge=0.0, le=1.0, default=0.0)
    recency_score: float = Field(ge=0.0, le=1.0, default=0.0)
    trust_score: float = Field(ge=0.0, le=1.0, default=0.0)  # composite
    is_primary: bool = False
    evaluation_notes: str | None = None
    fetched_ok: bool = False
    # Defense-in-depth signal: source content contains instruction-like text.
    # The pipeline never executes source content regardless of this flag.
    suspected_injection: bool = False
    content_text: str | None = None  # retrieved body text (untrusted content)

    @field_validator("url")
    @classmethod
    def _validate_url(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("url must be http(s)")
        return v or None


# ---------------------------------------------------------------------------
# Research planning
# ---------------------------------------------------------------------------
class ResearchPlan(BaseModel):
    """Structured output of the Planner Agent."""

    research_goal: str
    question_type: Literal["comparative", "factual", "exploratory", "how_to"] = "exploratory"
    subquestions: list[str] = Field(min_length=1)
    required_evidence: list[str] = Field(default_factory=list)
    comparison_subjects: list[str] = Field(default_factory=list)
    completion_criteria: list[str] = Field(default_factory=list)
    planned_queries: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_loose_shapes(cls, data: object) -> object:
        """Tolerant parsing: models sometimes return subquestions as objects
        ({'question': ...} / {'text': ...}) or a numbered dict
        ({'1': '...', '2': '...'}) — extract the strings."""
        if isinstance(data, dict):
            subs = data.get("subquestions")
            if isinstance(subs, list):
                fixed: list[str] = []
                for item in subs:
                    if isinstance(item, str):
                        fixed.append(item)
                    elif isinstance(item, dict):
                        text = (
                            item.get("question") or item.get("text")
                            or item.get("subquestion") or ""
                        )
                        if isinstance(text, str) and text.strip():
                            fixed.append(text.strip())
                data = {**data, "subquestions": fixed}
            elif isinstance(subs, dict):
                values = [
                    (v.get("question") if isinstance(v, dict) else v)
                    for v in subs.values()
                ]
                fixed = [v for v in values if isinstance(v, str) and v.strip()]
                data = {**data, "subquestions": fixed}
        return data

    @field_validator("subquestions")
    @classmethod
    def _dedupe(cls, v: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for s in v:
            key = s.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(s.strip())
        if not out:
            raise ValueError("at least one unique subquestion required")
        return out


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------
class Evidence(BaseModel):
    """A single extracted piece of evidence with full provenance.

    The `source_id` link is NEVER dropped: reports cite through it.
    """

    id: str
    claim_summary: str
    source_id: str
    # Location within the source (page, section, chunk index…)
    source_location: str | None = None
    snippet: str  # verbatim (or lightly trimmed) excerpt — untrusted content
    subquestion_id: str | None = None
    subquestion: str | None = None
    origin: SourceOrigin = SourceOrigin.WEB
    confidence: Literal["high", "moderate", "low"] = "moderate"
    extracted_at: str = Field(default_factory=lambda: utcnow().isoformat())

    @field_validator("snippet")
    @classmethod
    def _clip_snippet(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("snippet must not be empty")
        return v[:1500]


# ---------------------------------------------------------------------------
# Claims & fact-checking
# ---------------------------------------------------------------------------
class ClaimStatus(str, enum.Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class Claim(BaseModel):
    """A candidate factual claim proposed from evidence."""

    id: str
    text: str
    subquestion_id: str | None = None
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    status: ClaimStatus = ClaimStatus.INSUFFICIENT_EVIDENCE
    verification_rationale: str | None = None


class Contradiction(BaseModel):
    """Two credible sources disagree — surfaced, never silently resolved."""

    id: str
    topic: str
    claim_a: str
    source_id_a: str
    claim_b: str
    source_id_b: str
    possible_reason: str | None = None
    resolving_evidence_needed: str | None = None


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------
class EvidenceConfidence(str, enum.Enum):
    """Evidence-based categories (NOT pseudo-probabilities).

    Assignment rules (documented in docs/confidence.md):
      HIGH          — >=2 independent credible sources, no contradiction
      MODERATE      — 1 credible source, or 2+ weaker sources, no contradiction
      LOW           — single weak source OR contradicted but plausibly resolved
      INSUFFICIENT  — no usable evidence located
    """

    HIGH = "HIGH_EVIDENCE"
    MODERATE = "MODERATE_EVIDENCE"
    LOW = "LOW_EVIDENCE"
    INSUFFICIENT = "INSUFFICIENT_EVIDENCE"


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------
class CriticFindingType(str, enum.Enum):
    MISSING_EVIDENCE = "missing_evidence"
    WEAK_SOURCE = "weak_source"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    DUPLICATE = "duplicate_finding"
    LOGICAL_INCONSISTENCY = "logical_inconsistency"
    OVERCONFIDENT = "overconfident_conclusion"
    UNANSWERED_SUBQUESTION = "unanswered_subquestion"


class CriticFinding(BaseModel):
    id: str
    finding_type: CriticFindingType
    description: str
    affected_subquestion_id: str | None = None
    severity: Literal["low", "medium", "high"] = "medium"
    suggested_followup_query: str | None = None


class CriticVerdict(BaseModel):
    """Output of one critic pass."""

    findings: list[CriticFinding] = Field(default_factory=list)
    research_sufficient: bool = False
    overall_assessment: str = "not evaluated"
    followup_queries: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
class ReportFormat(str, enum.Enum):
    EXECUTIVE_SUMMARY = "executive_summary"
    DETAILED = "detailed"
    COMPARISON = "comparison"


class Finding(BaseModel):
    statement: str
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: EvidenceConfidence = EvidenceConfidence.INSUFFICIENT
    is_interpretation: bool = False
    caveat: str | None = None


class ComparisonRow(BaseModel):
    subject: str
    criteria: dict[str, str] = Field(default_factory=dict)
    advantages: list[str] = Field(default_factory=list)
    disadvantages: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class ResearchReport(BaseModel):
    """Final structured report — the ONLY shape the frontend renders."""

    session_id: str
    question: str
    format: ReportFormat = ReportFormat.DETAILED
    generated_at: str = Field(default_factory=lambda: utcnow().isoformat())
    executive_summary: str
    methodology: str
    key_findings: list[Finding] = Field(default_factory=list)
    comparison: list[ComparisonRow] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    conclusion: str
    recommendation: str | None = None
    sources: list[SourceRecord] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    confidence_summary: dict[str, int] = Field(default_factory=dict)
    subquestion_answers: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Live research trace (safe operational events only)
# ---------------------------------------------------------------------------
class AgentEventType(str, enum.Enum):
    STAGE_STARTED = "stage_started"
    STAGE_COMPLETED = "stage_completed"
    STAGE_FAILED = "stage_failed"
    INFO = "info"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    LOOP = "research_loop"
    WARNING = "warning"
    BUDGET = "budget_update"


class AgentEvent(BaseModel):
    """A safe, concise operational event for the live trace.

    Forbidden content: prompts, completions, chain-of-thought, secrets.
    """

    id: str
    session_id: str
    timestamp: str = Field(default_factory=lambda: utcnow().isoformat())
    agent: str
    event_type: AgentEventType
    message: str
    stage: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Search queries (research memory)
# ---------------------------------------------------------------------------
class SearchQueryRecord(BaseModel):
    query: str
    subquestion_id: str | None = None
    executed_at: str = Field(default_factory=lambda: utcnow().isoformat())
    result_count: int = 0
    status: Literal["pending", "ok", "failed", "skipped_duplicate"] = "pending"


# ---------------------------------------------------------------------------
# Research state — the single source of truth for a running research job
# ---------------------------------------------------------------------------
class ResearchStage(str, enum.Enum):
    PENDING = "pending"
    PLANNING = "planning"
    SEARCHING = "searching"
    GATHERING_EVIDENCE = "gathering_evidence"
    FACT_CHECKING = "fact_checking"
    CONTRADICTION_CHECK = "contradiction_check"
    CRITIQUING = "critiquing"
    SYNTHESIZING = "synthesizing"
    CITING = "citing"
    REPORT_READY = "report_ready"
    CANCELLED = "cancelled"
    FAILED = "failed"
    COMPLETED_PARTIAL = "completed_partial"


STAGE_ORDER: list[ResearchStage] = [
    ResearchStage.PENDING,
    ResearchStage.PLANNING,
    ResearchStage.SEARCHING,
    ResearchStage.GATHERING_EVIDENCE,
    ResearchStage.FACT_CHECKING,
    ResearchStage.CONTRADICTION_CHECK,
    ResearchStage.CRITIQUING,
    ResearchStage.SYNTHESIZING,
    ResearchStage.CITING,
    ResearchStage.REPORT_READY,
]


class ResearchDepth(str, enum.Enum):
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


class SubQuestionStatus(str, enum.Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    ANSWERED = "answered"
    INSUFFICIENT = "insufficient"


class SubQuestion(BaseModel):
    id: str
    text: str
    status: SubQuestionStatus = SubQuestionStatus.PENDING
    answer: str | None = None
    confidence: EvidenceConfidence | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class ResearchState(BaseModel):
    """Explicit structured state maintained across the workflow.

    This is what makes the pipeline a state machine rather than a prompt chain.
    """

    model_config = ConfigDict(validate_assignment=True)

    session_id: str
    question: str
    depth: ResearchDepth = ResearchDepth.STANDARD
    requested_format: ReportFormat = ReportFormat.DETAILED
    status: ResearchStage = ResearchStage.PENDING
    current_iteration: int = 0
    error: str | None = None

    # Plan
    research_plan: ResearchPlan | None = None

    # Subquestions with status (research memory)
    subquestions: list[SubQuestion] = Field(default_factory=list)

    # Memory of executed searches (dedupe + no-repeat)
    executed_queries: list[SearchQueryRecord] = Field(default_factory=list)

    # Sources & evidence
    sources: dict[str, SourceRecord] = Field(default_factory=dict)
    evidence: dict[str, Evidence] = Field(default_factory=dict)
    claims: dict[str, Claim] = Field(default_factory=dict)
    contradictions: list[Contradiction] = Field(default_factory=list)

    # Critic loop
    critic_verdicts: list[CriticVerdict] = Field(default_factory=list)
    followup_queries: list[str] = Field(default_factory=list)

    # Report
    report: ResearchReport | None = None

    # Budget / accounting
    tokens_used: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    started_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    # ---- helpers ----
    def add_event_safe(self, event: AgentEvent) -> None:
        """Events are stored externally; this hook exists for orchestrator use."""
        self.updated_at = utcnow()

    def next_subquestion_to_research(self) -> SubQuestion | None:
        for sq in self.subquestions:
            if sq.status in (SubQuestionStatus.PENDING, SubQuestionStatus.IN_PROGRESS):
                return sq
        return None

    def subquestion_answered(self, sq_id: str) -> bool:
        sq = next((s for s in self.subquestions if s.id == sq_id), None)
        return sq is not None and sq.status == SubQuestionStatus.ANSWERED

    def query_already_executed(self, query: str) -> bool:
        q = query.strip().lower()
        return any(r.query.strip().lower() == q for r in self.executed_queries)

    def sources_by_origin(self, origin: SourceOrigin) -> list[SourceRecord]:
        return [s for s in self.sources.values() if s.origin == origin]

    def budget_snapshot(self) -> dict[str, Any]:
        return {
            "iteration": self.current_iteration,
            "searches": len(self.executed_queries),
            "sources": len(self.sources),
            "tokens": self.tokens_used,
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
        }
