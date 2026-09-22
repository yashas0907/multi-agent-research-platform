"""Contradiction Detection Agent — surfaces disagreements, never resolves them silently."""
from __future__ import annotations

from pydantic import AliasChoices, BaseModel, Field, model_validator

from app.agents.base import BaseAgent, new_id
from app.schemas.research import Contradiction, Evidence


def _accept_bare_list(key: str):
    def _before(data: object) -> object:
        if isinstance(data, list):
            return {key: data}
        return data

    return model_validator(mode="before")(_before)


class ContradictionSpec(BaseModel):
    topic: str = Field(min_length=3, max_length=300)
    evidence_id_a: str = Field(validation_alias=AliasChoices("evidence_id_a", "evidence_a"))
    evidence_id_b: str = Field(validation_alias=AliasChoices("evidence_id_b", "evidence_b"))
    claim_a: str = ""
    claim_b: str = ""
    possible_reason: str | None = Field(default=None, validation_alias=AliasChoices("possible_reason", "reason"))
    resolving_evidence_needed: str | None = Field(default=None, validation_alias=AliasChoices("resolving_evidence_needed", "resolving_evidence", "resolution"))


class ContradictionOutput(BaseModel):
    contradictions: list[ContradictionSpec] = Field(default_factory=list, max_length=20)
    reason: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_loose_shapes(cls, data: object) -> object:
        """Tolerant parsing for real-model output variance.

        Handles: bare lists; list under 'conflict'/'conflicts'; items shaped
        as single-side {'claim', 'source'} pairs (consecutive items are the
        two sides); items referencing source ids instead of evidence ids.
        """
        if isinstance(data, list):
            data = {"contradictions": data}
        if not isinstance(data, dict):
            return data
        d = dict(data)
        if not isinstance(d.get("contradictions"), list):
            for key in ("conflict", "conflicts", "contradiction"):
                if isinstance(d.get(key), list):
                    d["contradictions"] = d[key]
                    break
        items = d.get("contradictions")
        if not isinstance(items, list):
            return d

        coerced: list[dict] = []
        for item in items:
            if isinstance(item, dict):
                coerced.append(_coerce_contra_item(item))
        # single-side pairs: {claim, source} lists → consecutive items are A/B
        if coerced and all("claim_a" not in c for c in coerced):
            paired: list[dict] = []
            for i in range(0, len(coerced) - 1, 2):
                a, b = coerced[i], coerced[i + 1]
                if a.get("_claim") and b.get("_claim"):
                    paired.append(
                        {
                            "topic": a.get("topic") or (a.get("_claim") or "")[:80],
                            "claim_a": a["_claim"],
                            "evidence_id_a": a.get("evidence_id_a") or a.get("_source") or "",
                            "claim_b": b["_claim"],
                            "evidence_id_b": b.get("evidence_id_b") or b.get("_source") or "",
                            "possible_reason": d.get("reason"),
                        }
                    )
            d["contradictions"] = paired
        else:
            # top-level reason flows down to items missing one
            top_reason = d.get("reason")
            if isinstance(top_reason, str) and top_reason:
                for c in d["contradictions"]:
                    if isinstance(c, dict) and not c.get("possible_reason"):
                        c["possible_reason"] = top_reason
        return d


def _coerce_contra_item(item: dict) -> dict:
    """Map one contradiction item's fields from common model variants."""
    out = dict(item)
    # two-side ids (evidence or source naming)
    out["evidence_id_a"] = (
        item.get("evidence_id_a") or item.get("evidence_a")
        or item.get("source_id_a") or item.get("source_a") or ""
    )
    out["evidence_id_b"] = (
        item.get("evidence_id_b") or item.get("evidence_b")
        or item.get("source_id_b") or item.get("source_b") or ""
    )
    # single-side shape: keep raw claim/source for pairing
    if "claim" in item and "evidence_id_a" not in item:
        out["_claim"] = item.get("claim")
        out["_source"] = item.get("source")
    return out


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
        seen_topics: set[str] = set()
        for spec in out.contradictions:
            resolved = self._resolve_pair(spec, ev_by_id)
            if resolved is None:
                continue
            ea, eb = resolved
            pair = frozenset((ea.id, eb.id))
            topic_key = spec.topic.strip().lower()[:100]
            # dedupe by evidence pair AND by topic (models often report the
            # same disagreement from multiple evidence pairs)
            if pair in seen_pairs or topic_key in seen_topics:
                continue
            seen_pairs.add(pair)
            seen_topics.add(topic_key)
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

    @staticmethod
    def _resolve_pair(spec, ev_by_id: dict[str, Evidence]) -> tuple[Evidence, Evidence] | None:
        """Resolve a spec's two sides to evidence items.

        Models may reference evidence ids or source ids — try both. Unresolvable
        pairs are skipped (never guessed).
        """
        ea = ev_by_id.get(spec.evidence_id_a)
        eb = ev_by_id.get(spec.evidence_id_b)
        if ea is not None and eb is not None and ea.id != eb.id:
            return ea, eb
        # fall back: ids may be source ids — find evidence from those sources
        if ea is None:
            ea = next(
                (e for e in ev_by_id.values() if e.source_id == spec.evidence_id_a), None
            )
        if eb is None:
            eb = next(
                (e for e in ev_by_id.values() if e.source_id == spec.evidence_id_b), None
            )
        if ea is not None and eb is not None and ea.id != eb.id:
            return ea, eb
        return None


def _format_evidence(evidence: list[Evidence]) -> str:
    lines = []
    for e in evidence[:24]:
        lines.append(f"EVIDENCE {e.id} (source={e.source_id}, subq={e.subquestion_id})")
        lines.append(f"  CLAIM: {e.claim_summary[:180]}")
        lines.append(f"  SNIPPET: {e.snippet[:180]}")
        lines.append("")
    return "\n".join(lines)
