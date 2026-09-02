"""Contradiction Detection Agent — surfaces disagreements, never resolves them silently."""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.agents.base import BaseAgent, new_id
from app.schemas.research import Contradiction, Evidence


class ContradictionSpec(BaseModel):
    topic: str = Field(min_length=3, max_length=300)
    evidence_id_a: str
    evidence_id_b: str
    claim_a: str = ""
    claim_b: str = ""
    possible_reason: str | None = None
    resolving_evidence_needed: str | None = None


class ContradictionOutput(BaseModel):
    contradictions: list[ContradictionSpec] = Field(default_factory=list, max_length=20)


class ContradictionAgent(BaseAgent):
    name = "contradiction"
    prompt_name = "contradiction"

    async def run(
        self, evidence: list[Evidence], source_trust: dict[str, float]
    ) -> list[Contradiction]:
        # Only compare evidence from credible-enough sources (trust >= 0.35).
        credible = [e for e in evidence if source_trust.get(e.source_id, 0.0) >= 0.35]
        if len(credible) < 2:
            return []

        user_payload = self._prompt.render_user(
            evidence_block=_format_evidence(credible)
        )
        out = await self._structured(user_payload, ContradictionOutput)

        ev_by_id = {e.id: e for e in credible}
        results: list[Contradiction] = []
        seen_pairs: set[frozenset[str]] = set()
        for spec in out.contradictions:
            ea, eb = ev_by_id.get(spec.evidence_id_a), ev_by_id.get(spec.evidence_id_b)
            if ea is None or eb is None or ea.id == eb.id:
                continue
            pair = frozenset((ea.id, eb.id))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            results.append(
                Contradiction(
                    id=new_id("contra"),
                    topic=spec.topic,
                    claim_a=spec.claim_a or ea.claim_summary,
                    source_id_a=ea.source_id,
                    claim_b=spec.claim_b or eb.claim_summary,
                    source_id_b=eb.source_id,
                    possible_reason=spec.possible_reason,
                    resolving_evidence_needed=spec.resolving_evidence_needed,
                )
            )
        if results:
            await self.emit(
                "info",
                f"Detected {len(results)} contradiction(s) — will be surfaced in the report",
                count=len(results),
            )
        return results


def _format_evidence(evidence: list[Evidence]) -> str:
    lines = []
    for e in evidence[:40]:
        lines.append(f"EVIDENCE {e.id} (source={e.source_id}, subq={e.subquestion_id})")
        lines.append(f"  CLAIM: {e.claim_summary}")
        lines.append(f"  SNIPPET: {e.snippet[:300]}")
        lines.append("")
    return "\n".join(lines)
