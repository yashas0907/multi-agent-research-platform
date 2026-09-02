"""API request/response schemas (separate from internal domain schemas)."""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.research import (
    AgentEvent,
    ResearchDepth,
    ResearchReport,
    ResearchStage,
    SubQuestion,
)


class ResearchCreateRequest(BaseModel):
    question: str = Field(min_length=10, max_length=2000)
    depth: ResearchDepth = ResearchDepth.STANDARD
    report_format: str = Field(default="detailed", pattern="^(detailed|executive_summary|comparison)$")
    document_ids: list[str] = Field(default_factory=list, description="User documents to include as evidence")


class ResearchCreateResponse(BaseModel):
    session_id: str
    status: ResearchStage
    question: str
    depth: ResearchDepth


class ResearchStatusResponse(BaseModel):
    session_id: str
    status: ResearchStage
    stage_label: str
    progress_pct: int
    current_iteration: int
    question: str
    depth: ResearchDepth
    error: str | None = None
    budget: dict = Field(default_factory=dict)


class TraceResponse(BaseModel):
    session_id: str
    events: list[AgentEvent]


class SubQuestionsResponse(BaseModel):
    session_id: str
    subquestions: list[SubQuestion]


class DocumentIngestResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    chunks: int
    message: str


class HealthResponse(BaseModel):
    status: str
    app_env: str
    llm_provider: str
    web_search_provider: str
    database: str
    time: str


class SourceListResponse(BaseModel):
    sources: list[dict]
    total: int
