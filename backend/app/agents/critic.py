"""Critic Agent — quality gate with the power to request more research."""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.agents.base import BaseAgent
from app.schemas.research import (
    Claim,
    CriticFinding,
    CriticFindingType,
    CriticVerdict,
    Evidence,
    ResearchState,
    SubQuestion,
)


class CriticFindingSpec(BaseModel):
    finding_type: str
    description: str = Field(min_length=5, max_length=800)
    severity: str = "medium"
    subquestion_id: str | None = None
    suggested_followup_query: str | None = None


class CriticOutput(BaseModel):
    findings: list[CriticFindingSpec] = Field(default_factory=list, max_length=15)
    research_sufficient: bool = False
    overall_assessment: str = "not evaluated"
    followup_queries: list[str] = Field(default_factory=list, max_length=5)


class CriticAgent(BaseAgent):
    name = "critic"
    prompt_name = "critic"

    async def run(
        self,
        state: ResearchState,
        evidence: list[Evidence],
        claims: list[Claim],
        pass_num: int,
    ) -> CriticVerdict:
        user_payload = self._prompt.render_user(
            question=state.question,
            subquestions_block=_format_subquestions(state.subquestions),
            evidence_summary=_evidence_summary(evidence),
            claims_summary=_claims_summary(claims),
            contradictions_count=len(state.contradictions),
            iteration=state.current_iteration,
            max_iterations=self.ctx.profile.get("_max_iterations", 3),
            critic_passes=self.ctx.profile["critic_passes"],
            pass_num=pass_num,
        )
        out = await self._structured(user_payload, CriticOutput)

        valid_sq_ids = {sq.id for sq in state.subquestions}
        findings: list[CriticFinding] = []
        for f in out.findings:
            try:
                ftype = CriticFindingType(f.finding_type)
            except ValueError:
                ftype = CriticFindingType.LOGICAL_INCONSISTENCY
            findings.append(
                CriticFinding(
                    id=f"cf_{len(findings)}_{id(f) % 10000}",
                    finding_type=ftype,
                    description=f.description,
                    affected_subquestion_id=(
                        f.subquestion_id if f.subquestion_id in valid_sq_ids else None
                    ),
                    severity=f.severity if f.severity in ("low", "medium", "high") else "medium",
                    suggested_followup_query=f.suggested_followup_query,
                )
            )

        verdict = CriticVerdict(
            findings=findings,
            research_sufficient=out.research_sufficient,
            overall_assessment=out.overall_assessment[:500],
            followup_queries=[
                q for q in out.followup_queries if q.strip()
            ][:5],
        )
        await self.emit(
            "info",
            f"Critic pass {pass_num}: "
            + (
                "research sufficient"
                if verdict.research_sufficient
                else f"{len(findings)} issue(s), followup needed"
            ),
            findings=len(findings),
            sufficient=verdict.research_sufficient,
        )
        return verdict


def _format_subquestions(subquestions: list[SubQuestion]) -> str:
    lines = []
    for sq in subquestions:
        ev_count = len(sq.evidence_ids)
        lines.append(
            f"- [{sq.status.value}] {sq.text} (evidence items: {ev_count})"
        )
    return "\n".join(lines) or "none"


def _evidence_summary(evidence: list[Evidence]) -> str:
    if not evidence:
        return "NO EVIDENCE GATHERED YET"
    by_conf: dict[str, int] = {}
    for e in evidence:
        by_conf[e.confidence] = by_conf.get(e.confidence, 0) + 1
    distinct_sources = {e.source_id for e in evidence}
    return (
        f"{len(evidence)} evidence items from {len(distinct_sources)} sources; "
        f"confidence distribution: {by_conf}"
    )


def _claims_summary(claims: list[Claim]) -> str:
    if not claims:
        return "no claims proposed yet"
    by_status: dict[str, int] = {}
    for c in claims:
        by_status[c.status.value] = by_status.get(c.status.value, 0) + 1
    return f"{len(claims)} claims; statuses: {by_status}"
