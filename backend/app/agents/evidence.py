"""Evidence Extraction Agent — pulls cited evidence from (untrusted) source text."""
from __future__ import annotations

from pydantic import AliasChoices, BaseModel, Field, model_validator

from app.agents.base import BaseAgent, new_id
from app.schemas.research import Evidence, SourceRecord, SubQuestion


def _accept_bare_list(key: str):
    """Tolerant parsing: models often return a bare JSON array instead of
    {key: [...]} — wrap it before validation instead of failing."""

    def _before(data: object) -> object:
        if isinstance(data, list):
            return {key: data}
        return data

    return model_validator(mode="before")(_before)


class ExtractedEvidence(BaseModel):
    claim_summary: str = Field(
        validation_alias=AliasChoices("claim_summary", "claim", "summary", "statement"),
        min_length=4,
        max_length=500,
    )
    source_location: str | None = Field(default=None, validation_alias=AliasChoices("source_location", "location", "section"))
    snippet: str = Field(min_length=4, max_length=1500)
    confidence: str = "moderate"


class EvidenceList(BaseModel):
    evidence: list[ExtractedEvidence] = Field(default_factory=list, max_length=15)
    _wrap = _accept_bare_list("evidence")


class EvidenceAgent(BaseAgent):
    name = "evidence"
    prompt_name = "evidence"

    async def run(
        self,
        state,  # ResearchState (import cycle avoided)
        source: SourceRecord,
        subquestion: SubQuestion,
    ) -> list[Evidence]:
        # 4000 chars ≈ 1000 tokens: wiki pages carry heavy navigation
        # boilerplate before content. Rather than always taking the head
        # (definitions can sit anywhere), pick the window with the highest
        # keyword overlap with the subquestion.
        raw_text = source.content_text or source.snippet or ""
        text = _best_window(raw_text, subquestion.text, window=4000)
        if not text.strip():
            return []

        user_payload = self._prompt.render_user(
            subquestion=subquestion.text,
            title=source.title,
            url=source.url or "n/a",
            text=text,
        )
        out = await self._structured(user_payload, EvidenceList)

        items: list[Evidence] = []
        for ev in out.evidence[: self.ctx.profile["max_evidence_per_subquestion"]]:
            # Anti-fabrication guard: snippet must actually appear in (or be a
            # close substring of) the source text. Otherwise we drop the item —
            # never keep evidence we cannot ground.
            if not _grounded(ev.snippet, text):
                await self.emit(
                    "warning",
                    "Dropped ungrounded evidence snippet (not found in source text)",
                    source_id=source.id,
                )
                continue
            items.append(
                Evidence(
                    id=new_id("ev"),
                    claim_summary=ev.claim_summary,
                    source_id=source.id,
                    source_location=ev.source_location,
                    snippet=ev.snippet,
                    subquestion_id=subquestion.id,
                    subquestion=subquestion.text,
                    origin=source.origin,
                    confidence=ev.confidence if ev.confidence in ("high", "moderate", "low") else "moderate",
                )
            )
        await self.emit(
            "info",
            f"Extracted {len(items)} evidence items from '{source.title[:60]}'",
            source_id=source.id,
            extracted=len(items),
        )
        return items


def _grounded(snippet: str, source_text: str) -> bool:
    """Verify the snippet is grounded in the source text.

    Accepts exact match or a high-overlap fuzzy match (agents may trim
    whitespace/punctuation). Strict enough to block fabrication.
    """
    s = _norm(snippet)
    t = _norm(source_text)
    if not s:
        return False
    if s in t:
        return True
    # fuzzy: check 8-gram overlap coverage
    grams = [s[i : i + 8] for i in range(0, max(1, len(s) - 7), 4)]
    if not grams:
        return False
    hits = sum(1 for g in grams if g in t)
    return hits / len(grams) >= 0.8


def _best_window(text: str, subquestion: str, window: int = 4000) -> str:
    """Pick the `window`-char slice of `text` most relevant to the subquestion.

    Wiki/dictionary pages front-load hundreds of chars of navigation; the
    definition may sit mid-page. A relevance-picked window beats a fixed
    head-of-text window.
    """
    text = text.strip()
    if len(text) <= window:
        return text
    import re as _re

    terms = {t for t in _re.sub(r"[^a-z0-9]+", " ", subquestion.lower()).split() if len(t) > 3}
    step = window // 2  # overlapping windows
    best_text = text[:window]
    best_score = -1
    for start in range(0, len(text) - window + 1, step):
        chunk = text[start : start + window]
        if not terms:
            break
        chunk_terms = set(_re.sub(r"[^a-z0-9]+", " ", chunk.lower()).split())
        score = len(terms & chunk_terms)
        if score > best_score:
            best_score = score
            best_text = chunk
    return best_text


def _norm(text: str) -> str:
    import re

    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()
