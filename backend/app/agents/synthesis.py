"""Synthesis Agent — combines verified evidence into the draft report."""
from __future__ import annotations

from pydantic import AliasChoices, BaseModel, Field, model_validator

from app.agents.base import BaseAgent
from app.schemas.research import (
    Claim,
    ClaimStatus,
    ComparisonRow,
    Contradiction,
    Evidence,
    Finding,
    ResearchState,
)


class FindingSpec(BaseModel):
    statement: str = Field(
        validation_alias=AliasChoices("statement", "finding", "text", "claim"),
        min_length=5,
        max_length=1000,
    )
    evidence_ids: list[str] = Field(default_factory=list, validation_alias=AliasChoices("evidence_ids", "evidence", "citations"))
    confidence: str = "LOW_EVIDENCE"
    is_interpretation: bool = False
    caveat: str | None = None


class ComparisonSpec(BaseModel):
    subject: str
    criteria: dict[str, str] = Field(default_factory=dict)
    advantages: list[str] = Field(default_factory=list)
    disadvantages: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class SynthesisOutput(BaseModel):
    executive_summary: str = Field(min_length=20, max_length=3000)
    methodology: str = Field(min_length=10, max_length=2000)
    key_findings: list[FindingSpec] = Field(default_factory=list, max_length=15)
    comparison: list[ComparisonSpec] = Field(default_factory=list, max_length=8)
    limitations: list[str] = Field(default_factory=list, max_length=12)
    conclusion: str = Field(min_length=10, max_length=3000)
    recommendation: str | None = None
    subquestion_answers: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _coerce_loose_shapes(cls, data: object) -> object:
        """Tolerant parsing for real-model output variance.

        Handles: key_findings under 'findings'/'findings_list'; the model's
        per-subquestion dict shape ('subquestions': {id: {answer, ...}})
        flattened into subquestion_answers.
        """
        if not isinstance(data, dict):
            return data
        d = dict(data)
        if not d.get("key_findings") and isinstance(d.get("findings"), list):
            d["key_findings"] = d["findings"]
        # per-subquestion dict shape → flat answers
        subs = d.get("subquestions")
        if isinstance(subs, dict) and not d.get("subquestion_answers"):
            answers: dict[str, str] = {}
            for sq_id, val in subs.items():
                if isinstance(val, str):
                    answers[sq_id] = val
                elif isinstance(val, dict):
                    text = val.get("answer") or val.get("summary") or ""
                    if isinstance(text, str) and text.strip():
                        answers[sq_id] = text.strip()
            d["subquestion_answers"] = answers
        return d


class SynthesisAgent(BaseAgent):
    name = "synthesis"
    prompt_name = "synthesis"

    async def run(
        self,
        state: ResearchState,
        evidence: list[Evidence],
        claims: list[Claim],
        contradictions: list[Contradiction],
    ) -> SynthesisOutput:
        user_payload = self._prompt.render_user(
            question=state.question,
            format=state.requested_format.value,
            verified_block=_format_verified(evidence, claims, state),
            contradictions_block=_format_contradictions(contradictions) or "none",
            subjects=", ".join(state.research_plan.comparison_subjects) if state.research_plan else "none",
        )
        out = await self._structured(user_payload, SynthesisOutput)

        # Sanitize: only allow evidence ids that exist and are verified.
        valid_ids = {e.id for e in evidence}
        for f in out.key_findings:
            f.evidence_ids = [i for i in f.evidence_ids if i in valid_ids]
        for c in out.comparison:
            c.evidence_ids = [i for i in c.evidence_ids if i in valid_ids]
        # map subquestion answers by index position -> id (LLM sees sq ids)
        if out.subquestion_answers:
            remapped: dict[str, str] = {}
            for sq in state.subquestions:
                if sq.text[:80] in out.subquestion_answers:
                    remapped[sq.id] = out.subquestion_answers[sq.text[:80]]
                elif sq.id in out.subquestion_answers:
                    remapped[sq.id] = out.subquestion_answers[sq.id]
            out.subquestion_answers = remapped

        await self.emit(
            "info",
            f"Draft synthesized: {len(out.key_findings)} findings, "
            f"{len(out.comparison)} comparison rows",
            findings=len(out.key_findings),
        )
        return out


def _format_verified(
    evidence: list[Evidence], claims: list[Claim], state: ResearchState
) -> str:
    lines: list[str] = []
    ev_by_id = {e.id: e for e in evidence}
    for sq in state.subquestions:
        lines.append(f"SUBQUESTION [{sq.id}]: {sq.text}")
        sq_evidence = [e for e in evidence if e.subquestion_id == sq.id]
        if not sq_evidence:
            lines.append("  (no evidence — answer honestly as insufficient)")
        for e in sq_evidence:
            lines.append(f"  EVIDENCE {e.id} (source {e.source_id}, {e.confidence}): {e.claim_summary}")
            lines.append(f"    snippet: {e.snippet[:220]}")
        sq_claims = [c for c in claims if c.subquestion_id == sq.id]
        for c in sq_claims:
            sup = ", ".join(c.supporting_evidence_ids) or "none"
            lines.append(f"  CLAIM {c.id} [{c.status.value}]: {c.text} (evidence: {sup})")
        lines.append("")
    return "\n".join(lines)


def _format_contradictions(contradictions: list[Contradiction]) -> str:
    lines = []
    for c in contradictions:
        lines.append(
            f"- Topic: {c.topic}\n  A: {c.claim_a} (source {c.source_id_a})\n"
            f"  B: {c.claim_b} (source {c.source_id_b})\n"
            f"  Possible reason: {c.possible_reason}\n"
            f"  Resolving evidence: {c.resolving_evidence_needed}"
        )
    return "\n".join(lines)
