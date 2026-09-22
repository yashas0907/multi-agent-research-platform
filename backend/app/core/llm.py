"""LLM abstraction layer.

Design:
  * `LLMClient` is a thin, provider-agnostic interface: `complete()` and
    `complete_structured()`.
  * `complete_structured` takes a Pydantic model class, requests JSON, and
    *validates* it. Malformed output triggers at most one deterministic repair
    pass, then raises `StructuredOutputError`.
  * Concrete providers: OpenAI-compatible HTTP (works for OpenAI and Groq —
    both expose the same chat-completions shape) and a deterministic
    `MockLLMClient` used for offline development and tests.
  * Token usage is tracked and surfaced for budget accounting (mock included).
"""
from __future__ import annotations

import abc
import asyncio
import json
import time
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import get_settings
from app.core.errors import LLMError, StructuredOutputError
from app.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class RateLimiter:
    """Adaptive rate limiter for LLM calls.

    Two mechanisms:
      1. Global minimum interval between requests (all callers serialized).
      2. Token-aware pacing: the client records `x-ratelimit-remaining-tokens`
         and `x-ratelimit-reset-tokens` from each response; when headroom is
         too low for the next call, acquire() waits for the reset window.
         This maximizes throughput within the provider's REAL token budget
         instead of guessing a fixed interval.
    """

    def __init__(self, min_interval: float, min_headroom_tokens: int = 3000) -> None:
        self.min_interval = max(0.0, min_interval)
        self.min_headroom_tokens = min_headroom_tokens
        self._last_call = 0.0
        self._lock = asyncio.Lock()
        self._remaining_tokens = 10**9
        self._reset_seconds = 0.0

    def observe(self, remaining_tokens: int, reset_seconds: float) -> None:
        """Called by the client after each response with provider headers."""
        self._remaining_tokens = remaining_tokens
        self._reset_seconds = reset_seconds

    async def acquire(self) -> None:
        if self.min_interval <= 0 and self._remaining_tokens > self.min_headroom_tokens:
            return
        async with self._lock:
            # token-aware pacing: not enough headroom → wait for reset window
            if self._remaining_tokens <= self.min_headroom_tokens and self._reset_seconds > 0:
                wait = self._reset_seconds + 0.5
                logger.info("llm_token_pacing", waiting_for_reset=round(wait, 1))
                await asyncio.sleep(wait)
                self._remaining_tokens = 10**9  # assume reset restored headroom
                self._reset_seconds = 0.0
            now = time.monotonic()
            wait_for = self._last_call + self.min_interval - now
            if wait_for > 0:
                await asyncio.sleep(wait_for)
            self._last_call = time.monotonic()


class UsageReport(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0
    retries: int = 0
    model: str = ""


class LLMClient(abc.ABC):
    """Provider-agnostic LLM interface."""

    provider_name: str = "abstract"

    def __init__(self, model: str) -> None:
        self.model = model
        self.usage = UsageReport(model=model)

    @abc.abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Return the assistant message text."""

    async def complete_structured(
        self,
        messages: list[dict[str, str]],
        schema: type[T],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> T:
        """Request, parse, and validate a structured output.

        Failure policy: one repair attempt with the validation error appended;
        then raise `StructuredOutputError` (callers decide retry/skip/degrade).
        """
        attempt_messages = list(messages)
        last_error: Exception | None = None
        for attempt in (1, 2):
            raw = await self.complete(
                attempt_messages, temperature=temperature, max_tokens=max_tokens
            )
            try:
                data = extract_json(raw)
                return schema.model_validate(data)
            except (ValueError, ValidationError) as exc:
                last_error = exc
                logger.warning(
                    "structured_output_invalid",
                    provider=self.provider_name,
                    attempt=attempt,
                    error=str(exc)[:400],
                )
                if attempt == 1:
                    attempt_messages = attempt_messages + [
                        {"role": "assistant", "content": raw},
                        {
                            "role": "user",
                            "content": (
                                "Your previous reply was not valid JSON matching the "
                                f"schema of `{schema.__name__}` ({str(exc)[:500]}). "
                                "Respond again with ONLY a valid JSON object. No "
                                "markdown fences, no commentary."
                            ),
                        },
                    ]
        raise StructuredOutputError(
            f"model failed to produce valid {schema.__name__} after repair pass",
            retryable=True,
        )


class OpenAICompatibleClient(LLMClient):
    """Chat-completions client for OpenAI and any compatible API (Groq, etc.)."""

    provider_name = "openai-compatible"

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        *,
        timeout: int = 60,
        max_retries: int = 2,
        min_interval: float = 0.0,
        fallback_models: list[str] | None = None,
    ) -> None:
        super().__init__(model)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        # Fallback chain: free-tier model buckets are PER-MODEL — when the
        # primary is rate-limited, the next model (separate bucket) takes over.
        self.fallback_models = fallback_models or []
        self._model_index = 0
        self._limiter = RateLimiter(min_interval)

    @property
    def current_model(self) -> str:
        models = [self.model] + self.fallback_models
        return models[self._model_index % len(models)]

    def _advance_model(self) -> bool:
        """Move to the next fallback model. False when the chain is exhausted."""
        if self._model_index >= len(self.fallback_models):
            return False
        self._model_index += 1
        logger.info(
            "llm_fallback_switch", from_model=self.model, to_model=self.current_model
        )
        return True

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        settings = get_settings()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": settings.LLM_TEMPERATURE if temperature is None else temperature,
            "max_tokens": settings.LLM_MAX_OUTPUT_TOKENS if max_tokens is None else max_tokens,
        }
        if settings.LLM_REASONING_EFFORT and _supports_reasoning_effort(self.current_model):
            # gpt-oss reasoning models: cut chain-of-thought token burn.
            # NOT sent for models that reject it (qwen, compound, llama…).
            payload["reasoning_effort"] = settings.LLM_REASONING_EFFORT
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        # Free-tier 429s: first retry within the model's bucket (capped waits),
        # then fall through the fallback chain (separate per-model buckets).
        rate_limit_retries = 2
        while True:
            try:
                await self._limiter.acquire()
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(
                        f"{self.base_url}/chat/completions",
                        json={**payload, "model": self.current_model},
                        headers=headers,
                    )
                if resp.status_code == 429:
                    retry_after = _retry_after_seconds(resp.headers)
                    last_error = LLMError(
                        f"rate limited ({retry_after:.0f}s)", retryable=True
                    )
                    if rate_limit_retries > 0:
                        rate_limit_retries -= 1
                        self.usage.retries += 1
                        logger.warning(
                            "llm_rate_limited",
                            model=self.current_model,
                            retry_after=retry_after,
                            retries_left=rate_limit_retries,
                        )
                        await asyncio.sleep(retry_after)
                        continue
                    # model bucket drained → try the fallback chain
                    if self._advance_model():
                        rate_limit_retries = 2
                        continue
                    raise last_error
                if resp.status_code >= 500:
                    last_error = LLMError(f"provider server error {resp.status_code}", retryable=True)
                    if rate_limit_retries > 0:
                        rate_limit_retries -= 1
                        self.usage.retries += 1
                        await asyncio.sleep(0.5 * (2**self.max_retries))
                        continue
                    raise last_error
                if resp.status_code >= 400:
                    # Self-healing: if the model rejected a parameter we sent
                    # (e.g. reasoning_effort on a non-reasoning fallback model),
                    # strip it and retry once — handles provider API drift
                    # across the fallback chain automatically.
                    if _strip_rejected_param(payload, resp.text):
                        continue
                    raise LLMError(
                        f"provider client error {resp.status_code}: {resp.text[:300]}",
                        retryable=False,
                    )
                data = resp.json()
                usage = data.get("usage", {})
                self.usage.prompt_tokens += usage.get("prompt_tokens", 0)
                self.usage.completion_tokens += usage.get("completion_tokens", 0)
                self.usage.total_tokens += usage.get("total_tokens", 0)
                self.usage.calls += 1
                # token-aware pacing: record provider rate-limit state
                self._limiter.observe(
                    remaining_tokens=_int_header(resp.headers, "x-ratelimit-remaining-tokens", 10**9),
                    reset_seconds=_parse_reset_seconds(resp.headers.get("x-ratelimit-reset-tokens")),
                )
                return data["choices"][0]["message"]["content"] or ""
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if rate_limit_retries > 0:
                    rate_limit_retries -= 1
                    self.usage.retries += 1
                    await asyncio.sleep(1.0)
                else:
                    raise LLMError(f"LLM call failed after retries: {exc}", retryable=False)
            except LLMError as exc:
                if exc is last_error or not exc.retryable:
                    raise
                # retryable non-429 errors fall through to the next attempt


def _supports_reasoning_effort(model: str) -> bool:
    """reasoning_effort is only accepted by reasoning models (gpt-oss family)."""
    m = model.lower()
    return "gpt-oss" in m


def _strip_rejected_param(payload: dict[str, Any], error_text: str) -> bool:
    """Remove a payload parameter the provider rejected; True if stripped.

    Handles provider API drift: a 400 mentioning an unknown/unsupported
    parameter (by name) gets that parameter removed and the call retried.
    """
    for param in list(payload.keys()):
        if param in ("model", "messages"):
            continue
        if param in error_text and (
            "not supported" in error_text
            or "unknown" in error_text
            or "does not support" in error_text
            or "unexpected" in error_text
            or "invalid" in error_text
        ):
            payload.pop(param)
            logger.info("llm_param_stripped", param=param)
            return True
    return False


def _retry_after_seconds(headers: httpx.Headers) -> float:
    """Parse Retry-After; default to a conservative free-tier wait.

    Capped at 30s: when the token window is drained, a longer wait still
    cannot succeed before the refill — the client retries instead of
    stalling for minutes.
    """
    raw = headers.get("retry-after")
    if raw:
        try:
            return min(30.0, max(1.0, float(raw)))
        except ValueError:
            pass
    return 15.0


def _int_header(headers: httpx.Headers, name: str, default: int) -> int:
    raw = headers.get(name)
    if raw:
        try:
            return int(float(raw))
        except ValueError:
            pass
    return default


def _parse_reset_seconds(raw: str | None) -> float:
    """Parse Groq's reset format like '929ms', '2.5s', '30m14.4s' → seconds."""
    if not raw:
        return 0.0
    raw = raw.strip().lower()
    total = 0.0
    import re

    for value, unit in re.findall(r"([\d.]+)(ms|s|m|h)", raw):
        v = float(value)
        total += v / 1000 if unit == "ms" else v * 60 if unit == "m" else v * 3600 if unit == "h" else v
    return total


# ---------------------------------------------------------------------------
# Deterministic offline provider
# ---------------------------------------------------------------------------
def extract_json(raw: str) -> Any:
    """Extract a JSON object/array from a model response.

    Handles: bare JSON, ```json fences, leading/trailing prose, and — as a
    last resort — *truncated* JSON (output token cutoff mid-array), repaired
    by closing open structures. Raises ValueError when nothing is parsable.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("` \n")
        if text[:4].lower() == "json":
            text = text[4:]
        text = text.strip()
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        raise ValueError("no JSON found in response")
    ends = [i for i in (text.rfind("}"), text.rfind("]")) if i != -1]
    if not ends:
        raise ValueError("no JSON terminator found in response")
    candidate = text[min(starts) : max(ends) + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return _repair_truncated_json(candidate)


def _repair_truncated_json(candidate: str) -> Any:
    """Repair JSON truncated by an output-token cutoff.

    Scans open brackets/braces (string-aware) and closes them, then parses.
    Trailing incomplete items are dropped — partial-but-valid output beats
    total failure, and downstream agents handle partial lists gracefully.
    """
    opens: list[str] = []
    in_string = False
    escaped = False
    for ch in candidate:
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            opens.append(ch)
        elif ch in "}]" and opens:
            opens.pop()
    repaired = candidate
    if in_string:
        repaired += '"'
    for ch in reversed(opens):
        repaired += "}" if ch == "{" else "]"
    return json.loads(repaired)


def _find_task_marker(system: str) -> str | None:
    markers = (
        "planner",
        "search",
        "source_evaluator",
        "evidence",
        "factchecker",
        "contradiction",
        "critic",
        "synthesis",
        "citation",
    )
    for m in markers:
        if f"TASK:{m}" in system:
            return m
    return None


def _payload_field(user: str, key: str, default: Any = None) -> Any:
    """Mock handlers receive agent payloads in the user message.

    Payloads may be JSON or the plain-text 'Key: value' template format —
    support both.
    """
    text = user.strip()
    # try JSON first
    if text.startswith(("{", "[", "```")):
        try:
            data = extract_json(text)
            if isinstance(data, dict) and key in data:
                return data[key]
        except (ValueError, json.JSONDecodeError):
            pass
    # plain-text template: 'Key: value' or multi-line 'Key:\nvalue'
    import re as _re

    pattern = _re.compile(
        rf"^{_re.escape(key)}:\s*(.+?)(?=^\w+:|\Z)", _re.S | _re.M
    )
    match = pattern.search(user)
    if match is not None:
        return match.group(1).strip()
    return default


def _first_text_block(user: str) -> str:
    """Fall back to treating the user message as plain text."""
    return user.strip()


def _question_from(user: str) -> str:
    q = _payload_field(user, "question")
    if isinstance(q, str) and q:
        return q
    return _first_text_block(user)[:300]


def _is_comparative(question: str) -> bool:
    q = question.lower()
    return any(w in q for w in ("compare", " vs ", "versus", "better", "which approach"))


def _subjects_from(question: str) -> list[str]:
    """Extract plausible comparison subjects from the question text."""
    q = question.lower()
    found: list[str] = []
    for kw in ("ragas", "trulens", "deepeval", "giskard", "langsmith", "arette"):
        if kw in q and kw not in found:
            found.append(kw)
    if found:
        return found
    # generic split on " vs "
    if " vs " in q:
        parts = question.split(" vs ")
        if len(parts) >= 2:
            a = parts[-2].strip().strip(",.;:")
            b = parts[-1].strip()
            for cut in (",", ".", ";", "?", " would", " is", " for"):
                b = b.split(cut)[0].strip()
            if a and b:
                return [a[-60:], b[:60]]
    return []


def _subquestions_for(question: str) -> list[str]:
    base = [
        f"What are the leading approaches relevant to: {question[:150]}?",
        "What evidence exists comparing these approaches on quality and reliability?",
        "What are the documented limitations and failure modes of each approach?",
    ]
    if _is_comparative(question):
        base.append(
            "Under which conditions does each approach perform best, and what are the tradeoffs?"
        )
    base.append("What do practitioners report about production usage of these approaches?")
    return base


class MockLLMClient(LLMClient):
    """Deterministic offline provider for development, CI, and tests.

    It inspects the *task marker* embedded in the system prompt (see
    `app/prompts/`) and emits schema-valid structured outputs for each agent,
    so the entire pipeline runs offline without pretending to be a real model.
    All mock responses are clearly marked as offline-provider output.
    """

    provider_name = "mock"

    def __init__(self, model: str = "mock-research-model") -> None:
        super().__init__(model)

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        task = _find_task_marker(system)

        self.usage.calls += 1
        self.usage.prompt_tokens += max(1, len(system + user) // 4)
        self.usage.completion_tokens += 60
        self.usage.total_tokens = self.usage.prompt_tokens + self.usage.completion_tokens

        handler = _MOCK_HANDLERS.get(task or "")
        if handler is None:
            return json.dumps({"note": "mock provider: no task marker found"})
        return handler(user, system)


# ---------------------------------------------------------------------------
# Mock task handlers
# ---------------------------------------------------------------------------
def _mock_search(user: str, system: str) -> str:
    subquestion = _extract_prompt_section(user, "Subquestion") or ""
    if not subquestion:
        subquestion = _first_text_block(user)[:150]
    # generate keyword-focused queries from the subquestion
    base = subquestion.strip().rstrip("?").strip()
    queries = [base]
    if len(base.split()) > 4:
        queries.append(" ".join(base.split()[:6]))
    queries.append(f"{base} benchmark comparison")
    out = []
    for q in queries[:3]:
        if q.strip():
            out.append(
                {
                    "query": q.strip()[:200],
                    "reasoning": "high-signal query for the subquestion",
                    "site": None,
                    "recency_months": None,
                }
            )
    return json.dumps({"queries": out})


def _mock_planner(user: str, system: str) -> str:
    question = _extract_prompt_section(user, "Research question") or _first_text_block(user)
    question = question.strip()
    return json.dumps(
        {
            "research_goal": f"Determine: {question[:300]}",
            "question_type": "comparative" if _is_comparative(question) else "exploratory",
            "subquestions": _subquestions_for(question),
            "required_evidence": [
                "Primary documentation for each approach under consideration",
                "Independent benchmarks or evaluations where available",
                "Documented limitations and failure modes",
                "Practitioner reports on production usage",
            ],
            "comparison_subjects": _subjects_from(question),
            "completion_criteria": [
                "Each subquestion answered with at least one credible source",
                "No high-severity critic findings outstanding",
            ],
            "planned_queries": _subquestions_for(question)[:3],
        }
    )


def _mock_source_eval(user: str, system: str) -> str:
    return json.dumps(
        {
            "relevance": 0.75,
            "authority": 0.7,
            "is_primary": False,
            "source_type": "documentation",
            "reasoning": "Mock evaluation (offline provider): source appears relevant to the subquestion and reasonably authoritative.",
            "discard": False,
        }
    )


def _mock_evidence(user: str, system: str) -> str:
    """Return a REAL snippet from the provided source text (anti-fabrication).

    Picks the sentence with the highest keyword overlap with the subquestion,
    skipping navigational/boilerplate lines. The grounding check in
    EvidenceAgent then verifies the snippet verbatim — so the offline
    pipeline exercises real extraction logic without fabrication.
    """
    text = _extract_prompt_section(user, "Source text") or ""
    subquestion = _extract_prompt_section(user, "Subquestion") or ""
    if not text.strip():
        return json.dumps({"evidence": []})

    import re as _re

    # Candidate passages: sentences from substantive lines only.
    junk_markers = (
        "learn more", "sign in", "log in", "search", "skip to", "menu",
        "download", "subscribe", "cookie", "newsletter", "click here",
        "read more", "share", "follow us", "©", "all rights reserved",
        "press enter", "submit", "donate", "navigate", "home",
        "streamline", "get started", "try now", "contact us", "book a",
        "watch", "webinar", "case studi", "pricing", "talk to", "request a demo",
        "administrative operations", "→",
    )
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    candidates: list[str] = []
    for ln in lines:
        if len(ln) < 50:
            continue
        if any(m in ln.lower() for m in junk_markers):
            continue
        # split long lines into sentences
        for s in _re.split(r"(?<=[.!?])\s+", ln):
            s = s.strip()
            # must look like a real sentence: starts with a letter, ends with punctuation
            if not (40 <= len(s) <= 400):
                continue
            if not (s[0].isalpha() and s[-1] in ".!?"):
                continue
            # reject sentence-fragment stacks (multiple '→' or short word runs)
            words = s.split()
            if len(words) < 8:
                continue
            if sum(1 for w in words if len(w) <= 2) > len(words) // 2:
                continue
            candidates.append(s)
    if not candidates:
        return json.dumps({"evidence": []})

    q_terms = {t for t in _re.split(r"[^a-z0-9]+", subquestion.lower()) if len(t) > 3}

    def score(s: str) -> tuple[int, int]:
        terms = {t for t in _re.split(r"[^a-z0-9]+", s.lower()) if len(t) > 3}
        overlap = len(q_terms & terms)
        # prefer medium-length informative sentences
        length_bonus = 1 if 60 <= len(s) <= 260 else 0
        return (overlap, length_bonus)

    best = max(candidates, key=score)
    return json.dumps(
        {
            "evidence": [
                {
                    "claim_summary": best[:180],
                    "source_location": "extracted passage",
                    "snippet": best,
                    "confidence": "moderate",
                }
            ]
        }
    )


def _extract_prompt_section(user: str, key: str) -> str | None:
    """Extract a section from the plain-text prompt payloads.

    Sections start with 'Key ...:' at line start and end at the next known
    footer line ('Extract ... as JSON.' / 'Evaluate as JSON.' / 'Detect ...'
    / 'Review as JSON.' / 'Synthesize as JSON.' / 'Produce ...' / 'Return
    verdicts as JSON.') or a line that begins a known template section.
    """
    import re as _re

    # Known prompt-template section headers (from app/prompts/library.py) —
    # these delimit sections. Content lines never start with these.
    headers = (
        "Subquestion:", "Research context:", "Already-executed queries:",
        "Source title:", "Source URL:", "Source text", "Source snippet",
        "Research question:", "Research depth:", "Report format requested:",
        "Claims with their proposed supporting evidence:", "Evidence set",
        "Subquestions with status:", "Evidence summary:", "Claim verification summary:",
        "Contradictions found:", "Report format:", "Verified findings per subquestion:",
        "Contradictions to surface:", "Comparison subjects", "Draft findings:",
        "Valid evidence ids:", "Iteration ", "Critic pass ",
        "Generate up to", "Extract evidence as JSON.", "Evaluate as JSON.",
        "Detect contradictions as JSON.", "Review as JSON.", "Synthesize as JSON.",
        "Audit citations as JSON.", "Return verdicts as JSON.",
        "Produce the research plan as JSON.",
    )
    header_pattern = "|".join(_re.escape(h) for h in headers)
    pattern = _re.compile(
        rf"^{_re.escape(key)}[^:\n]*:\s*\n?(.+?)^(?:{header_pattern})", _re.S | _re.M
    )
    m = pattern.search(user)
    if m is not None:
        return m.group(1).strip()
    # fallback: section to end of message
    pattern2 = _re.compile(rf"^{_re.escape(key)}[^:\n]*:\s*\n?(.+)$", _re.S | _re.M)
    m2 = pattern2.search(user)
    if m2 is not None:
        return m2.group(1).strip()
    return None


def _mock_factchecker(user: str, system: str) -> str:
    """Parse 'CLAIM <id>: ...' lines from the payload; mark all PARTIALLY_SUPPORTED.

    Rationale: evidence-backed claims derived from real sources with one
    supporting item deserve partial support — conservative, never upgrades
    unsupported claims to SUPPORTED.
    """
    import re as _re

    claim_ids = _re.findall(r"^CLAIM (cl_\w+):", user, _re.M)
    verdicts = []
    for cid in claim_ids[:20]:
        # find its linked evidence lines
        block = _re.search(
            rf"^CLAIM {cid}:.*?(?=^CLAIM |\Z)", user, _re.S | _re.M
        )
        ev_ids = _re.findall(r"EVIDENCE (ev_\w+)", block.group(0)) if block else []
        verdicts.append(
            {
                "claim_id": cid,
                "status": "PARTIALLY_SUPPORTED" if ev_ids else "INSUFFICIENT_EVIDENCE",
                "rationale": "Offline mock verification: grounded in retrieved source text.",
                "evidence_ids": ev_ids,
            }
        )
    return json.dumps({"verdicts": verdicts})


def _mock_contradiction(user: str, system: str) -> str:
    return json.dumps({"contradictions": []})


def _mock_critic(user: str, system: str) -> str:
    return json.dumps(
        {
            "findings": [
                {
                    "finding_type": "missing_evidence",
                    "description": "Offline mock critic: additional primary-source evidence would strengthen conclusions.",
                    "severity": "low",
                    "subquestion_id": None,
                    "suggested_followup_query": None,
                }
            ],
            "research_sufficient": True,
            "overall_assessment": "Mock critic pass: evidence base is sufficient for a low-confidence offline demonstration.",
            "followup_queries": [],
        }
    )


def _mock_synthesis(user: str, system: str) -> str:
    question = _extract_prompt_section(user, "Research question") or _first_text_block(user)
    # collect the real evidence ids listed in the verified findings block
    import re as _re

    ev_ids = _re.findall(r"EVIDENCE (ev_\w+)", user)
    # Answerable questions with evidence → MODERATE; anything else must stay
    # LOW/INSUFFICIENT (anti-fabrication: never claim moderate support without
    # retrieved evidence).
    confidence = "MODERATE_EVIDENCE" if ev_ids else "LOW_EVIDENCE"
    findings = [
        {
            "statement": (
                "Evidence was retrieved from "
                f"{len(set(ev_ids))} evidence item(s) relevant to the research question "
                "(offline mock synthesis)."
            ),
            "evidence_ids": ev_ids[:5],
            "confidence": confidence,
            "is_interpretation": False,
        }
    ]
    # subquestion answers: map every SUBQUESTION id present in the payload
    answers = {}
    for sq_id in _re.findall(r"SUBQUESTION \[(sq_\w+)\]", user):
        answers[sq_id] = (
            "Answered with the retrieved evidence (offline mock, moderate confidence)."
            if ev_ids
            else "Insufficient evidence retrieved to answer this subquestion."
        )
    return json.dumps(
        {
            "executive_summary": (
                "Offline mock synthesis: based on the gathered evidence, the research "
                f"question ({question[:120]}) "
                + (
                    "has moderate support from retrieved sources; findings below are "
                    "moderate confidence."
                    if ev_ids
                    else "cannot be answered with high confidence from the retrieved "
                    "material alone; findings below are low-to-moderate confidence."
                )
            ),
            "methodology": (
                "Structured multi-agent pipeline: plan, search, source evaluation, "
                "evidence extraction, claim verification, contradiction detection, "
                "critique, synthesis, citation mapping. (Offline mock provider.)"
            ),
            "key_findings": findings,
            "comparison": [],
            "limitations": [
                "Offline deterministic provider used; conclusions are illustrative only.",
                "Source coverage depends on the offline corpus.",
            ],
            "conclusion": (
                "Evidence retrieved provides moderate support for preliminary "
                "conclusions; verify with a production LLM provider."
                if ev_ids
                else "Insufficient independent evidence was available for a "
                "high-confidence conclusion; treat all findings as preliminary."
            ),
            "recommendation": None,
            "subquestion_answers": answers,
        }
    )


def _mock_citation(user: str, system: str) -> str:
    return json.dumps(
        {
            "citation_map": [],
            "issues": [],
            "summary": "Offline mock citation pass: no fabricated citations added.",
        }
    )


_MOCK_HANDLERS = {
    "planner": _mock_planner,
    "search": _mock_search,
    "source_evaluator": _mock_source_eval,
    "evidence": _mock_evidence,
    "factchecker": _mock_factchecker,
    "contradiction": _mock_contradiction,
    "critic": _mock_critic,
    "synthesis": _mock_synthesis,
    "citation": _mock_citation,
}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_llm_instance: LLMClient | None = None


def _fallback_list(raw: str) -> list[str]:
    return [m.strip() for m in raw.split(",") if m.strip()]


def get_llm_client() -> LLMClient:
    """Build (once) the configured LLM client. Provider switchable via env."""
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance
    settings = get_settings()
    fallbacks = _fallback_list(settings.LLM_FALLBACK_MODELS)
    if settings.LLM_PROVIDER == "mock":
        _llm_instance = MockLLMClient()
    elif settings.LLM_PROVIDER == "groq":
        if not settings.GROQ_API_KEY:
            raise LLMError("GROQ_API_KEY not set", retryable=False)
        _llm_instance = OpenAICompatibleClient(
            model=settings.GROQ_MODEL,
            api_key=settings.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            timeout=settings.LLM_TIMEOUT_SECONDS,
            max_retries=settings.LLM_MAX_RETRIES,
            min_interval=settings.LLM_MIN_INTERVAL_SECONDS,
            fallback_models=fallbacks,
        )
    else:  # openai
        if not settings.LLM_API_KEY:
            raise LLMError("LLM_API_KEY not set", retryable=False)
        _llm_instance = OpenAICompatibleClient(
            model=settings.LLM_MODEL,
            api_key=settings.LLM_API_KEY,
            base_url="https://api.openai.com/v1",
            timeout=settings.LLM_TIMEOUT_SECONDS,
            max_retries=settings.LLM_MAX_RETRIES,
            min_interval=settings.LLM_MIN_INTERVAL_SECONDS,
            fallback_models=fallbacks,
        )
    logger.info("llm_client_ready", provider=_llm_instance.provider_name, model=_llm_instance.model)
    return _llm_instance


def reset_llm_client() -> None:
    """For tests: force re-creation of the client after settings changes."""
    global _llm_instance
    _llm_instance = None
