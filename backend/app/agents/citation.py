"""Citation Agent — audits claim→source links; refuses fabricated citations."""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.agents.base import BaseAgent
from app.schemas.research import Finding


class CitationIssue(BaseModel):
    finding_index: int
    issue_type: str  # missing_evidence_link | dangling_reference | weak_link
    detail: str = ""


class CitationOutput(BaseModel):
    citation_map: list[dict] = Field(default_factory=list)
    issues: list[CitationIssue] = Field(default_factory=list, max_length=20)
    summary: str = ""


class CitationAgent(BaseAgent):
    name = "citation"
    prompt_name = "citation"

    async def run(self, findings: list[Finding], valid_evidence_ids: list[str]) -> tuple[list[Finding], list[dict]]:
        """Audit findings; drop dangling evidence refs; return cleaned findings."""
        if not findings:
            return findings, []

        user_payload = self._prompt.render_user(
            findings_block=_format_findings(findings),
            evidence_ids=", ".join(valid_evidence_ids) or "none",
        )
        out = await self._structured(user_payload, CitationOutput)

        valid = set(valid_evidence_ids)
        cleaned = []
        for i, f in enumerate(findings):
            f.evidence_ids = [eid for eid in f.evidence_ids if eid in valid]
            issue = next((iss for iss in out.issues if iss.finding_index == i), None)
            if issue is not None and issue.issue_type == "dangling_reference":
                continue  # drop findings the auditor flagged as wholly unsupported
            if not f.evidence_ids and not f.is_interpretation:
                # unlinked factual finding: downgrade to interpretation with caveat
                f.is_interpretation = True
                f.caveat = (f.caveat or "") + " [citation audit: no direct evidence link]"
            cleaned.append(f)

        await self.emit(
            "info",
            f"Citation audit: {len(out.issues)} issue(s), {len(cleaned)} findings kept",
            issues=len(out.issues),
        )
        return cleaned, [i.model_dump() for i in out.issues]


def _format_findings(findings: list[Finding]) -> str:
    lines = []
    for i, f in enumerate(findings):
        lines.append(
            f"FINDING {i}: {f.statement}\n"
            f"  evidence_ids: {f.evidence_ids or 'NONE'}\n"
            f"  is_interpretation: {f.is_interpretation}, confidence: {f.confidence.value}"
        )
    return "\n".join(lines)
