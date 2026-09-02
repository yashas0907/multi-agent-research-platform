"""Fact-Checking Agent — classifies claims against evidence."""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.agents.base import BaseAgent
from app.schemas.research import Claim, ClaimStatus, Evidence


class FactVerdict(BaseModel):
    claim_id: str
    status: str  # SUPPORTED | PARTIALLY_SUPPORTED | CONTRADICTED | INSUFFICIENT_EVIDENCE
    rationale: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class FactCheckOutput(BaseModel):
    verdicts: list[FactVerdict] = Field(default_factory=list)


class FactCheckerAgent(BaseAgent):
    name = "factchecker"
    prompt_name = "factchecker"

    async def run(self, claims: list[Claim], evidence: list[Evidence]) -> list[Claim]:
        if not claims:
            return claims

        claims_block = _format_claims(claims, evidence)
        user_payload = self._prompt.render_user(claims_block=claims_block)
        out = await self._structured(user_payload, FactCheckOutput)

        by_id = {c.id: c for c in claims}
        valid_evidence_ids = {e.id for e in evidence}
        for verdict in out.verdicts:
            claim = by_id.get(verdict.claim_id)
            if claim is None:
                continue
            try:
                claim.status = ClaimStatus(verdict.status.upper())
            except ValueError:
                claim.status = ClaimStatus.INSUFFICIENT_EVIDENCE
            claim.verification_rationale = verdict.rationale[:400]
            # validate cited evidence ids exist
            claim.supporting_evidence_ids = [
                eid for eid in verdict.evidence_ids if eid in valid_evidence_ids
            ]

        # Any claim without a verdict stays INSUFFICIENT_EVIDENCE (conservative).
        checked = sum(1 for c in claims if c.verification_rationale)
        await self.emit(
            "info",
            f"Verified {checked}/{len(claims)} claims",
            checked=checked,
            total=len(claims),
        )
        return claims


def _format_claims(claims: list[Claim], evidence: list[Evidence]) -> str:
    lines: list[str] = []
    for c in claims[:25]:
        lines.append(f"CLAIM {c.id}: {c.text}")
        linked = [e for e in evidence if e.id in c.supporting_evidence_ids]
        if linked:
            for e in linked[:3]:
                lines.append(f"  EVIDENCE {e.id} (source {e.source_id}): {e.snippet[:250]}")
        else:
            lines.append("  (proposed evidence: none yet)")
        lines.append("")
    return "\n".join(lines)
