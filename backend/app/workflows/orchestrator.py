"""The Research Orchestrator — an explicit state machine, not a prompt chain.

Flow (with the critic loop):
    plan → [per-subquestion: search → fetch → evaluate → extract] →
    fact-check → contradiction-check → critic ─┐
        ↑ ____________________________________│ (followup searches, if budget allows)
        └────────────────────────────────────-┘
    → synthesis → citation → report

Every stage transition:
  * updates `ResearchState.status` (the UI reflects REAL backend state)
  * emits safe operational events (persisted to the DB + streamed)
  * is budget-checked (iterations/searches/tokens/runtime)
  * is failure-tolerant: tool/LLM failures degrade the stage, never crash the
    pipeline; the report notes what failed.

Cancellation: `cancel()` flips a flag checked between stages and inside the
subquestion loop — the job then ends as CANCELLED, not FAILED.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable

from app.agents.base import AgentContext
from app.agents.citation import CitationAgent
from app.agents.contradiction import ContradictionAgent
from app.agents.critic import CriticAgent
from app.agents.evidence import EvidenceAgent
from app.agents.factchecker import FactCheckerAgent
from app.agents.planner import PlannerAgent
from app.agents.search import SearchAgent
from app.agents.source_evaluator import SourceEvaluatorAgent
from app.agents.synthesis import SynthesisAgent
from app.core.config import DEPTH_PROFILES
from app.core.errors import (
    BudgetExceededError,
    LLMError,
    PlatformError,
    ResearchCancelledError,
)
from app.core.llm import LLMClient, get_llm_client
from app.core.logging import get_logger
from app.schemas.research import (
    AgentEvent,
    AgentEventType,
    Claim,
    Evidence,
    EvidenceConfidence,
    Finding,
    ResearchReport,
    ResearchStage,
    SourceOrigin,
    SourceRecord,
    SubQuestionStatus,
)
from app.schemas.research import ResearchState
from app.tools.base import ToolRegistry, ToolRunner
from app.tools.web_search import WebSearchTool
from app.tools.web_fetch import WebFetchTool
from app.workflows.budget import ResearchBudget

logger = get_logger(__name__)

EventSink = Callable[[AgentEvent], Awaitable[None]]


class Orchestrator:
    def __init__(
        self,
        state: ResearchState,
        registry: ToolRegistry,
        *,
        document_sources: list[SourceRecord] | None = None,
        llm: LLMClient | None = None,
        budget: ResearchBudget | None = None,
        event_sink: EventSink | None = None,
        token_sync: Callable[[int], None] | None = None,
    ) -> None:
        self.state = state
        self.registry = registry
        self.llm = llm or get_llm_client()
        self.budget = budget or ResearchBudget.from_settings()
        self.event_sink = event_sink
        self.token_sync = token_sync  # pull usage counters into state

        self._cancelled = asyncio.Event()
        self._events: list[AgentEvent] = []

        profile = dict(DEPTH_PROFILES[state.depth.value])
        profile["_max_iterations"] = self.budget.max_iterations

        async def emitter(payload: dict) -> None:
            await self._emit(**payload)

        self.ctx = AgentContext(llm=self.llm, event_emitter=emitter, profile=profile)
        self.runner = ToolRunner(registry, on_event=emitter)

        # Hybrid research: pre-ingested user documents become sources.
        self.document_sources = document_sources or []

    # ------------------------------------------------------------------ api
    async def cancel(self) -> None:
        self._cancelled.set()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    # ------------------------------------------------------------- events
    async def _emit(
        self,
        agent: str,
        event_type: str,
        message: str,
        stage: str | None = None,
        data: dict | None = None,
        **extra: object,
    ) -> None:
        if data is None:
            data = {}
        if extra:
            data = {**data, **{k: v for k, v in extra.items() if v is not None}}
        event = AgentEvent(
            id=uuid.uuid4().hex[:16],
            session_id=self.state.session_id,
            agent=agent,
            event_type=AgentEventType(event_type),
            message=message,
            stage=stage or self.state.status.value,
            data=data,
        )
        self._events.append(event)
        if self.event_sink is not None:
            try:
                await self.event_sink(event)
            except Exception:  # noqa: BLE001 — observability must not break research
                pass

    def _sync_usage(self) -> None:
        if self.token_sync is not None:
            self.token_sync(self.llm.usage.total_tokens)
        self.state.llm_calls = self.llm.usage.calls
        self.state.tool_calls = self.runner.calls_made

    # ------------------------------------------------------------- helpers
    def _check_cancel(self) -> None:
        if self.cancelled:
            raise ResearchCancelledError("research cancelled by user")

    def _set_stage(self, stage: ResearchStage) -> None:
        self.state.status = stage
        logger.info("stage_transition", session=self.state.session_id, stage=stage.value)

    async def _stage(self, name: str, stage: ResearchStage, agent: str) -> None:
        self._check_cancel()
        self.budget.check_runtime()
        self._set_stage(stage)
        await self._emit(agent, "stage_started", f"{name} started", stage=stage.value)

    # ---------------------------------------------------------------- run
    async def run(self) -> ResearchState:
        state = self.state
        try:
            await self._run_pipeline()
        except ResearchCancelledError:
            state.status = ResearchStage.CANCELLED
            await self._emit("orchestrator", "stage_failed", "Research cancelled", stage="cancelled")
        except BudgetExceededError as exc:
            # Produce best-effort partial report if we have anything usable.
            await self._emit(
                "orchestrator",
                "warning",
                f"Budget exhausted: {exc.message}. Producing partial report.",
                budget_type=exc.budget_type,
            )
            await self._finish_partial(exc.message)
        except LLMError as exc:
            state.error = f"LLM provider failure: {exc.message}"
            state.status = ResearchStage.FAILED
            await self._emit("orchestrator", "stage_failed", state.error[:200])
        except PlatformError as exc:
            state.error = exc.message
            state.status = ResearchStage.FAILED
            await self._emit("orchestrator", "stage_failed", f"Pipeline error: {exc.message[:200]}")
        except Exception as exc:  # noqa: BLE001 — never crash the worker
            state.error = f"unexpected pipeline failure: {exc}"
            state.status = ResearchStage.FAILED
            logger.exception("pipeline_crashed", session=state.session_id)
            await self._emit("orchestrator", "stage_failed", "Unexpected internal failure")
        self._sync_usage()
        return state

    # ----------------------------------------------------------- pipeline
    async def _run_pipeline(self) -> None:
        state = self.state
        profile = self.ctx.profile

        # ---- 1. Planning -------------------------------------------------
        await self._stage("Planning", ResearchStage.PLANNING, "planner")
        planner = PlannerAgent(self.ctx)
        await planner.run(state)
        self._sync_usage()

        # ---- 2..7 research iterations ------------------------------------
        critic_agent = CriticAgent(self.ctx)
        max_iterations = self.budget.max_iterations

        for iteration in range(1, max_iterations + 1):
            state.current_iteration = iteration
            await self._emit(
                "orchestrator",
                "research_loop",
                f"Research iteration {iteration}/{max_iterations}",
                iteration=iteration,
            )

            await self._research_pass()

            # -- fact checking --
            await self._stage("Fact checking", ResearchStage.FACT_CHECKING, "factchecker")
            claims = self._derive_claims()
            if claims:
                factchecker = FactCheckerAgent(self.ctx)
                await factchecker.run(claims, list(state.evidence.values()))
                self._sync_usage()
                for c in claims:
                    state.claims[c.id] = c

            # -- contradiction check --
            await self._stage("Contradiction check", ResearchStage.CONTRADICTION_CHECK, "contradiction")
            contradiction_agent = ContradictionAgent(self.ctx)
            trust = {sid: s.trust_score for sid, s in state.sources.items()}
            found = await contradiction_agent.run(list(state.evidence.values()), trust)
            state.contradictions.extend(found)
            self._sync_usage()

            # -- update subquestion statuses from evidence --
            self._update_subquestion_status()

            # -- critic --
            await self._stage("Critiquing", ResearchStage.CRITIQUING, "critic")
            verdict = await critic_agent.run(
                state, list(state.evidence.values()), list(state.claims.values()), pass_num=iteration
            )
            state.critic_verdicts.append(verdict)
            self._sync_usage()

            if verdict.research_sufficient or iteration >= max_iterations:
                break

            if not profile["allow_followup_search"]:
                break
            if not self.budget.can_search(state):
                await self._emit(
                    "orchestrator", "warning", "Search budget exhausted — finishing with available evidence"
                )
                break
            # queue followup queries for next pass
            state.followup_queries = verdict.followup_queries[: profile["queries_per_subquestion"]]

        # ---- synthesis -----------------------------------------------------
        await self._stage("Synthesizing", ResearchStage.SYNTHESIZING, "synthesis")
        synthesis = SynthesisAgent(self.ctx)
        synth_out = await synthesis.run(
            state, list(state.evidence.values()), list(state.claims.values()), state.contradictions
        )
        self._sync_usage()

        # ---- citation audit --------------------------------------------------
        await self._stage("Citation mapping", ResearchStage.CITING, "citation")
        findings = [
            Finding(
                statement=f.statement,
                evidence_ids=[i for i in f.evidence_ids],
                confidence=EvidenceConfidence(f.confidence),
                is_interpretation=f.is_interpretation,
                caveat=f.caveat,
            )
            for f in synth_out.key_findings
        ]
        citation_agent = CitationAgent(self.ctx)
        findings, _issues = await citation_agent.run(
            findings, [e.id for e in state.evidence.values()]
        )
        self._sync_usage()

        report = self._build_report(synth_out, findings)
        state.report = report
        self._set_stage(ResearchStage.REPORT_READY)
        await self._emit(
            "orchestrator",
            "stage_completed",
            "Research complete — report ready",
            stage="report_ready",
        )

    # ----------------------------------------------------- one research pass
    async def _research_pass(self) -> None:
        state = self.state
        profile = self.ctx.profile
        search_agent = SearchAgent(self.ctx)
        evaluator = SourceEvaluatorAgent(self.ctx)
        evidence_agent = EvidenceAgent(self.ctx)

        # Promote user documents as candidate sources for this session (once).
        for doc_source in self.document_sources:
            if doc_source.id not in state.sources:
                if self.budget.can_add_source(state):
                    state.sources[doc_source.id] = doc_source
                    if doc_source.suspected_injection:
                        await self._emit(
                            "security",
                            "warning",
                            "Uploaded document contains instruction-like content "
                            "(treated as data, never executed)",
                        )

        targets = [
            sq for sq in state.subquestions if sq.status != SubQuestionStatus.ANSWERED
        ]
        if not targets:
            targets = state.subquestions

        # Followup queries from the previous critic pass take priority.
        followup_map: dict[str, list[str]] = {}
        if state.followup_queries:
            pending_sq = targets[0] if targets else None
            if pending_sq is not None:
                followup_map[pending_sq.id] = list(state.followup_queries)
            state.followup_queries = []

        for sq in targets:
            self._check_cancel()
            sq.status = SubQuestionStatus.IN_PROGRESS

            # --- vector search on user docs (hybrid research) ---
            doc_hits = await self._search_user_documents(sq)

            # --- web queries ---
            queries: list[str] = list(followup_map.get(sq.id, []))
            if len(queries) < profile["queries_per_subquestion"] and self.budget.can_search(state):
                generated = await search_agent.run(state, sq)
                queries.extend(q.query for q in generated.queries)

            if not queries and not doc_hits:
                sq.status = SubQuestionStatus.INSUFFICIENT
                continue

            # Search → fetch → evaluate → extract, per query
            for query in queries:
                self._check_cancel()
                if not self.budget.can_search(state):
                    break
                await self._emit(
                    "search", "info", f"Searching: {query[:120]}", query_len=len(query)
                )
                result = await self.runner.execute(
                    "web_search", {"query": query, "max_results": profile["max_sources_per_query"]}
                )
                state.executed_queries.append(
                    _qrecord(query, sq.id, result)
                )
                if not result.ok:
                    continue

                for item in (result.output or {}).get("results", []):
                    if not self.budget.can_add_source(state):
                        break
                    await self._ingest_candidate_source(
                        item, sq, evaluator, evidence_agent
                    )

            # extract from doc hits for this subquestion
            if doc_hits:
                await self._extract_from_doc_hits(doc_hits, sq)

        # doc hits already consumed; avoid double extraction next iteration
        self.document_sources = []

    async def _search_user_documents(self, sq) -> list[dict]:
        if not self.document_sources:
            return []
        # Scope strictly to THIS session's documents — never search other
        # sessions' corpora (privacy + relevance).
        doc_ids = [self._raw_doc_id(s.id) for s in self.document_sources]
        result = await self.runner.execute(
            "vector_search",
            {"query": sq.text, "top_k": 6, "document_ids": doc_ids},
        )
        if not result.ok:
            return []
        hits = (result.output or {}).get("hits", [])
        await self._emit(
            "vector", "info", f"Document search: {len(hits)} chunk(s) for subquestion", hits=len(hits)
        )
        return hits

    @staticmethod
    def _raw_doc_id(source_id: str) -> str:
        """SourceRecords for documents use id='doc_<raw_id>' (see
        ResearchService._load_document_sources); the vector store indexes the
        raw document id. Strip the prefix to map between them."""
        return source_id.removeprefix("doc_")

    async def _ingest_candidate_source(
        self, item: dict, sq, evaluator, evidence_agent
    ) -> None:
        state = self.state
        url = item.get("url", "")
        if not url or url in {s.url for s in state.sources.values()}:
            return

        candidate = SourceRecord(
            id=f"src_{uuid.uuid4().hex[:10]}",
            title=(item.get("title") or "Untitled")[:300],
            url=url,
            domain=item.get("domain"),
            snippet=(item.get("snippet") or "")[:600],
            published_at=item.get("published_at"),
        )
        candidate = await evaluator.evaluate(candidate, sq)
        if candidate.trust_score <= 0.0:
            await self._emit(
                "source_evaluator", "info", f"Discarded low-quality source: {candidate.title[:60]}"
            )
            return
        state.sources[candidate.id] = candidate

        # Fetch content for evidence extraction (untrusted data)
        fetch = await self.runner.execute("web_fetch", {"url": url})
        if fetch.ok and fetch.output:
            candidate.content_text = fetch.output.get("content_text", "")
            candidate.fetched_ok = True
            if fetch.output.get("suspected_injection"):
                await self._emit(
                    "security",
                    "warning",
                    f"Source may contain injection-style content (treated as data): {candidate.domain}",
                )
        else:
            await self._emit(
                "source_evaluator",
                "warning",
                f"Could not fetch source content: {candidate.title[:60]}",
            )

        if candidate.content_text:
            items = await evidence_agent.run(state, candidate, sq)
            for e in items:
                state.evidence[e.id] = e
                if sq.id not in [s.id for s in state.subquestions]:
                    continue
                sq.evidence_ids.append(e.id)

    async def _extract_from_doc_hits(self, hits: list[dict], sq) -> None:
        """Turn user-document chunks into evidence items with provenance.

        Chunks are matched back to their SourceRecord by document id —
        exact mapping, never title heuristics (untitled docs would be lost).
        """
        from app.agents.evidence import _grounded

        for hit in hits:
            text = hit.get("text", "")
            raw_doc_id = hit.get("document_id", "")
            if not text or not raw_doc_id:
                continue
            doc_source = next(
                (s for s in self.document_sources if s.id == f"doc_{raw_doc_id}"),
                None,
            )
            if doc_source is None:
                continue
            # Grounding: chunk text must come from this document's content.
            snippet = text[:600]
            if not _grounded(snippet, doc_source.content_text or ""):
                continue  # never keep an ungrounded chunk
            ev = Evidence(
                id=f"ev_{uuid.uuid4().hex[:10]}",
                claim_summary=f"From user document: {snippet[:180]}",
                source_id=doc_source.id,
                source_location=f"chunk {hit.get('chunk_id', '?')}",
                snippet=snippet,
                subquestion_id=sq.id,
                subquestion=sq.text,
                origin=SourceOrigin.USER_DOC,
                confidence="high" if hit.get("score", 0) > 0.5 else "moderate",
            )
            self.state.evidence[ev.id] = ev
            sq.evidence_ids.append(ev.id)

    # ------------------------------------------------------------ claims
    def _derive_claims(self) -> list[Claim]:
        """Evidence summaries become candidate claims (one claim per evidence
        item with a summary; grouping happens at fact-check)."""
        state = self.state
        claims: list[Claim] = []
        for e in state.evidence.values():
            if e.id in state.claims:
                continue
            claims.append(
                Claim(
                    id=f"cl_{uuid.uuid4().hex[:10]}",
                    text=e.claim_summary,
                    subquestion_id=e.subquestion_id,
                    supporting_evidence_ids=[e.id],
                )
            )
        return claims

    def _update_subquestion_status(self) -> None:
        state = self.state
        for sq in state.subquestions:
            ev_ids = [e for e in state.evidence.values() if e.subquestion_id == sq.id]
            if not ev_ids:
                if sq.status == SubQuestionStatus.IN_PROGRESS:
                    sq.status = SubQuestionStatus.INSUFFICIENT
                continue
            sq.evidence_ids = [e.id for e in ev_ids]
            supported = [
                c
                for c in state.claims.values()
                if c.subquestion_id == sq.id
                and c.status.value in ("SUPPORTED", "PARTIALLY_SUPPORTED")
            ]
            sq.status = SubQuestionStatus.ANSWERED if supported else SubQuestionStatus.INSUFFICIENT
            sq.confidence = _confidence_for(ev_ids, state)

    # ------------------------------------------------------------ report
    def _build_report(self, synth_out, findings: list[Finding]) -> ResearchReport:
        state = self.state
        used_source_ids = {
            e.source_id for e in state.evidence.values()
        }
        sources = [s for sid, s in state.sources.items() if sid in used_source_ids or s.fetched_ok]
        # Deduplicate sources by URL keeping the highest trust
        seen: dict[str, SourceRecord] = {}
        for s in sources:
            key = s.url or s.id
            if key not in seen or s.trust_score > seen[key].trust_score:
                seen[key] = s
        sources = list(seen.values())

        report = ResearchReport(
            session_id=state.session_id,
            question=state.question,
            format=state.requested_format,
            executive_summary=synth_out.executive_summary,
            methodology=synth_out.methodology,
            key_findings=findings,
            comparison=[
                _row_from_spec(c) for c in synth_out.comparison
            ],
            contradictions=state.contradictions,
            limitations=synth_out.limitations,
            conclusion=synth_out.conclusion,
            recommendation=synth_out.recommendation,
            sources=sources,
            evidence=sorted(state.evidence.values(), key=lambda e: e.subquestion_id or ""),
            claims=sorted(state.claims.values(), key=lambda c: c.subquestion_id or ""),
            subquestion_answers=synth_out.subquestion_answers,
        )
        report.confidence_summary = _confidence_counts(findings)
        return report

    async def _finish_partial(self, reason: str) -> None:
        """Budget exhausted: build an honest partial report from what exists."""
        state = self.state
        evidence = list(state.evidence.values())
        state.status = ResearchStage.COMPLETED_PARTIAL
        if not evidence:
            state.status = ResearchStage.FAILED
            state.error = f"No evidence gathered before budget exhaustion: {reason}"
            return
        # Reuse synthesis if it can run; otherwise degrade to raw findings.
        try:
            synthesis = SynthesisAgent(self.ctx)
            synth_out = await synthesis.run(
                state, evidence, list(state.claims.values()), state.contradictions
            )
            findings = [
                Finding(
                    statement=f.statement,
                    evidence_ids=f.evidence_ids,
                    confidence=EvidenceConfidence(f.confidence),
                    is_interpretation=f.is_interpretation,
                    caveat=(f.caveat or "") + f" [partial research: {reason}]",
                )
                for f in synth_out.key_findings
            ]
            report = self._build_report(synth_out, findings)
            report.limitations = report.limitations + [
                f"Research stopped early due to budget: {reason}"
            ]
            state.report = report
            await self._emit("orchestrator", "stage_completed", "Partial report generated", stage="report_ready")
        except PlatformError as exc:
            state.status = ResearchStage.FAILED
            state.error = f"could not build partial report: {exc.message}"


def _qrecord(query: str, sq_id, result) -> "SearchQueryRecord":
    from app.schemas.research import SearchQueryRecord

    return SearchQueryRecord(
        query=query,
        subquestion_id=sq_id,
        status="ok" if result.ok else "failed",
        result_count=len((result.output or {}).get("results", [])) if result.ok else 0,
    )


def _row_from_spec(spec) -> "ComparisonRow":
    from app.schemas.research import ComparisonRow

    return ComparisonRow(
        subject=spec.subject,
        criteria=spec.criteria,
        advantages=spec.advantages,
        disadvantages=spec.disadvantages,
        evidence_ids=spec.evidence_ids,
    )


def _confidence_for(ev_list: list[Evidence], state: ResearchState) -> EvidenceConfidence:
    high = sum(1 for e in ev_list if e.confidence == "high")
    if high >= 2:
        return EvidenceConfidence.HIGH
    if high == 1 or len(ev_list) >= 2:
        return EvidenceConfidence.MODERATE
    return EvidenceConfidence.LOW


def _confidence_counts(findings: list[Finding]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for f in findings:
        counts[f.confidence.value] = counts.get(f.confidence.value, 0) + 1
    return counts
