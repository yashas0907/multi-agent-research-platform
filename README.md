# Multi-Agent Research & Intelligence Platform

An autonomous research system that takes a complex question and produces a
structured, cited, verified research report — using a pipeline of specialized
agents, real tools, retrieval, verification, and explicit cost budgets.

This is not a chatbot. There is no single `question → LLM → answer` path.
Every claim in the final report is traceable to a real retrieved source, and
when evidence is insufficient, the report says so.

---

## Example

> *"Compare the current leading approaches to RAG evaluation and determine
> which approach would be most suitable for a production customer-support
> system."*

The platform: decomposes this into subquestions → searches web + your
documents → evaluates source quality → extracts grounded evidence →
fact-checks claims → detects contradictions → critiques its own research
(loops for more if needed) → synthesizes with citations → audits every
claim→source link → emits a structured report with confidence categories,
contradictions, and limitations.

---

## Why multi-agent?

A single-agent design collapses under this task's requirements:

- **Planning and synthesis need different context shapes.** Planning wants the
  question; synthesis wants dozens of verified evidence items. One prompt
  holding both exceeds working memory and degrades both.
- **Verification must be independent.** The agent that *writes* a claim
  cannot be the one that *verifies* it — separation of duties is the only
  reliable anti-hallucination structure.
- **The critic loop needs a role with authority to reject.** A
  self-critiquing monologue rationalizes; a separate critic with budget
  authority loops the pipeline.
- **Failure isolation.** A search outage must degrade one stage, not corrupt
  the entire answer. Agents give failures blast radius.

When *would* single-agent be right? Direct Q&A over one known document, or
any task where planning/verification loops add latency without evidence
gains. This system's complexity is justified by verification requirements,
not fashion — each agent exists because a requirement demanded it.

---

## Architecture

```
                 USER QUESTION
                       │
                PLANNER AGENT
                       │
              RESEARCH PLAN (typed)
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   SEARCH AGENT    (VECTOR SEARCH   (SOURCE EVALUATOR
   query gen         over user        scores relevance/
        │            documents)       authority/recency)
        ▼                                │
   web_search/web_fetch tools ───────────┤
        │                                │
        └──────────────┬─────────────────┘
                       ▼
                EVIDENCE AGENT
          (grounded extraction, provenance)
                       │
                       ▼
                FACT-CHECKER AGENT
     SUPPORTED / PARTIAL / CONTRADICTED / INSUFFICIENT
                       │
                       ▼
             CONTRADICTION DETECTION
                       │
                       ▼
                  CRITIC AGENT ──── insufficient? ──► followup loop
                       │                  (budget-limited)
                       ▼
                SYNTHESIS AGENT
                       │
                       ▼
                CITATION AGENT (audit claim→source links)
                       │
                       ▼
                STRUCTURED REPORT (JSON) → UI
```

The orchestrator (`app/workflows/orchestrator.py`) is an explicit async state
machine over `ResearchState` — a typed Pydantic model holding the plan,
subquestions with status, executed queries, sources, evidence, claims,
contradictions, critic verdicts, and budget counters. Agents never exchange
prose blobs; they read and write structured state.

## Agent responsibilities

| Agent | Responsibility | Key output |
|---|---|---|
| **Planner** | Decompose objective; define evidence requirements & completion criteria | `ResearchPlan` |
| **Search** | Generate non-redundant, high-signal queries (dedupes against executed-query memory) | `SearchQueries` |
| **Source Evaluator** | Score relevance/authority/recency; classify primary vs secondary; discard weak sources | updated `SourceRecord` |
| **Evidence** | Extract claims with **verbatim grounded snippets** + provenance (source, location, subquestion) | `Evidence[]` |
| **Fact-Checker** | Verify each claim against evidence; conservative default INSUFFICIENT | `Claim(status)` |
| **Contradiction** | Surface disagreements between credible sources with reasons + resolution paths; never picks a winner | `Contradiction[]` |
| **Critic** | Find gaps/weak sources/overconfidence; trigger followup loop within budget | `CriticVerdict` |
| **Synthesis** | Combine verified evidence only; separate facts from interpretation; state uncertainty | draft findings |
| **Citation** | Audit claim→source links; drop dangling refs; downgrade unlinked factual claims to interpretations | audited findings |

## Workflow state

`ResearchState` (see `app/schemas/research.py`) is the single source of truth:

```
question, depth, status(stage), iteration
research_plan, subquestions[{status, answer, evidence_ids}]
executed_queries[]        ← research memory (no duplicate searches)
sources{}, evidence{}, claims{}, contradictions[]
critic_verdicts[], followup_queries[]
tokens_used, llm_calls, tool_calls
report
```

Stage transitions are real: the API's `/status` reflects the actual state
machine, and the UI's progress bar is driven by it — no fake progress.

## Tool architecture

Tools are classes with **Pydantic input/output schemas**, executed through
`ToolRunner`, which validates arguments, enforces per-session call budgets,
emits safe observability events, and converts every failure into a typed
error envelope (validation / timeout / upstream / budget / crash) — a
crashing tool can never take the pipeline down.

| Tool | Purpose |
|---|---|
| `web_search` | Offline curated corpus (default, reproducible) or live DuckDuckGo |
| `web_fetch` | Sanitized page text; treats content as **untrusted data**; heuristic injection flag |
| `vector_search` | Hybrid search over user documents |
| `calculator` | AST-whitelisted arithmetic for numeric claims (no `eval`) |

## RAG architecture

```
Document → Parser (txt/md/pdf*/html/json) → Cleaner → Chunker
(paragraph-aware, 180-word window, 40-word overlap) → Metadata
→ Embeddings (mock hashing | OpenAI) → Vector Store (SQLite, swappable)
→ Retriever (0.7·cosine + 0.3·keyword, score gating, metadata filters)
→ Evidence (verbatim-grounded, provenance-tracked)
```

`*` PDFs require a text layer; encrypted/scanned PDFs are rejected explicitly.

**Hybrid research:** user documents are searched in parallel with web sources
and become first-class evidence with `origin=user_document` — visible in the
UI and report. See `docs/retrieval.md`.

## Source verification & citations

- Source quality: documented methodology (`docs/source_evaluation.md`) —
  LLM relevance/authority + deterministic domain tiers and recency; composite
  trust decides what gets extracted.
- Anti-fabrication: **evidence snippets must appear in the source text**
  (verbatim or ≥80% 8-gram overlap) — enforced at runtime, and reused as the
  faithfulness metric in evaluation.
- Citation audit: every finding keeps its evidence ids; dangling references
  are stripped; unlinked factual claims are downgraded to interpretations
  with an explicit caveat. No invented titles, URLs, authors, or dates —
  missing metadata renders as "not available".

## Confidence & uncertainty

Evidence-based categories, never pseudo-probabilities
(`docs/confidence.md`):

```
HIGH_EVIDENCE          ≥2 independent credible sources agree
MODERATE_EVIDENCE      1 credible source, no contradiction
LOW_EVIDENCE           weak/contradicted evidence
INSUFFICIENT_EVIDENCE  no usable evidence → report says so
```

## Evaluation

`evaluation/run_evaluation.py` measures (on a curated dataset, results in
`evaluation/results/`):

- **Retrieval:** Recall@3, Precision@3
- **RAG:** context relevance, faithfulness (via the production grounding guard)
- **Research:** source quality, citation integrity, claim support, subquestion
  completeness, honesty (unanswerable questions must NOT get confident answers)

Latest offline run (mock providers): Recall@3 0.80 · Precision@3 0.27 ·
context relevance 0.69 · source trust 0.70 · completeness 0.70 · honesty 0.50.
Precision is honestly low by construction (closely-related corpus entries are
near-misses), and the offline mock LLM cannot semantically detect
unanswerable questions — both noted in the results JSON. These are the real
measured numbers for the offline configuration; rerun with real providers for
production figures.

## Tech stack

- **Backend:** Python 3.12+, FastAPI, SQLAlchemy 2 (async, SQLite/aiosqlite),
  Pydantic v2, httpx, structlog
- **Frontend:** React 18, TypeScript (strict), Vite
- **LLM:** provider abstraction — OpenAI / Groq (OpenAI-compatible) /
  deterministic offline mock for CI
- **Vector store:** SQLite-backed with a swappable interface
- **Testing:** pytest + pytest-asyncio (77 tests)
- **Deploy:** Docker + docker-compose

## Model abstraction

`LLM_PROVIDER=openai|groq|mock` in `.env`. The OpenAI-compatible client
serves both OpenAI and Groq (same chat-completions shape). All agents call
`complete_structured(messages, Schema)` — outputs are parsed, validated, and
get one deterministic repair pass before failing loudly. Switching providers
is a config change; nothing else in the codebase references a vendor.

## Database

Entities: `research_sessions` (full state snapshots for audit),
`research_tasks`, `agent_events` (safe traces), `documents`, `vector_chunks`.
Async SQLAlchemy; schema created at startup (Alembic-ready for prod).

## API

```
POST /api/research              create job (returns immediately)
GET  /api/research             list sessions
GET  /api/research/{id}        full state
GET  /api/research/{id}/status progress + budget (polling)
GET  /api/research/{id}/events live trace events (safe summaries only)
GET  /api/research/{id}/report structured report (409 until ready)
GET  /api/research/{id}/subquestions
GET  /api/research/{id}/sources
POST /api/research/{id}/cancel
POST /api/documents            upload (multipart, validated & size-capped)
GET  /api/documents            list
GET  /api/documents/{id}       info (+chunks)
DELETE /api/documents/{id}
GET  /api/tools                tool catalog (names + JSON schemas)
GET  /api/health               health (DB, providers)
```

### Async execution
`POST /api/research` → job row created → background asyncio task runs the
orchestrator → state + events persisted as they happen → frontend polls
`/status` + `/events`. Cancellation is cooperative between stages.

## Frontend

- **Workspace:** question, depth (Quick/Standard/Deep — actually changes
  subquestion counts, query counts, critic passes), report type, document
  upload/selection.
- **Progress:** stage list + percentage driven by real backend state.
- **Live trace:** operational events only (agent, message, stage) — prompts
  and chain-of-thought never appear.
- **Report:** executive summary, confidence distribution, findings with
  click-through evidence, contradictions, limitations, methodology, source
  list with trust bars, comparison table.

## Security

Retrieved/uploaded content is **data, never instructions** — the contract is
in every system prompt and enforced by typed fields, schema-validated tool
args, the grounding guard, and trace sanitization. See `docs/security.md`.

## Cost controls

```
RESEARCH_MAX_ITERATIONS   research loop cap
RESEARCH_MAX_SEARCHES     total searches per session
RESEARCH_MAX_SOURCES     source ingestion cap
RESEARCH_MAX_TOKENS      token budget (hard abort → partial report)
RESEARCH_MAX_RUNTIME_SECONDS  wall-clock cap
+ per-depth profiles (subquestions, queries/evidence caps, critic passes)
+ duplicate-query prevention via executed-query memory
```

Budgets are enforced by code (`ResearchBudget`), never entrusted to the
model. Exhaustion produces an honest partial report, not a silent truncation.

## Failure recovery

Typed error taxonomy with retry policy: transport errors and rate limits get
exponential backoff (bounded); validation failures never retry; a tool crash
becomes an error envelope; a failing LLM fails the session with the error
surfaced. Cancellation and budget exhaustion stop cleanly between stages.
Covered by `tests/test_failures.py`.

## Setup

```bash
# Backend
cd backend
python -m venv .venv && .venv\Scripts\activate   # or source .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example ../.env                       # defaults work offline (mock providers)
python -m uvicorn app.main:app --reload          # http://localhost:8000

# Frontend
cd ../frontend
npm install
npm run dev                                      # http://localhost:5173

# Tests
cd ../backend && python -m pytest

# Evaluation
python ../evaluation/run_evaluation.py

# One research session from the CLI (offline providers by default)
python ../scripts/run_research.py "Compare RAGAS and TruLens for RAG evaluation" --depth quick

# Quick stack sanity check (imports, tools, one pipeline run)
python ../scripts/check_offline_stack.py
```

The default configuration (mock LLM + offline corpus + mock embeddings) runs
the **entire** pipeline with zero API keys — deliberately, so the system is
demonstrable and CI-reproducible. For real research: set `LLM_PROVIDER`,
`LLM_API_KEY`, `EMBEDDING_PROVIDER`, `WEB_SEARCH_PROVIDER` in `.env`.

### Real research on free tiers (verified)

```env
LLM_PROVIDER=groq
GROQ_API_KEY=<your free key from console.groq.com>
GROQ_MODEL=openai/gpt-oss-120b
LLM_FALLBACK_MODELS=qwen/qwen3.8-27b,groq/compound-mini
LLM_MIN_INTERVAL_SECONDS=8
LLM_REASONING_EFFORT=low
EMBEDDING_PROVIDER=mock
WEB_SEARCH_PROVIDER=duckduckgo
```

Measured on the free tier (all free, no credit card):

- **Full research session: ~3–5 minutes end-to-end** (standard depth)
- Live web search via DuckDuckGo (no key, real results)
- Free-tier resilience is engineered, not hoped for: per-model rate-limit
  buckets with an automatic **fallback model chain** (120b → qwen →
  compound-mini), **token-aware adaptive pacing** from provider rate-limit
  headers, capped retry waits, and **graceful degradation** — a rate-limited
  step degrades honestly (noted in the report) instead of failing the session
- Real analysis quality: verbatim-grounded evidence, honest confidence
  distribution (e.g. 5 HIGH / 2 MODERATE / 4 INSUFFICIENT), contradictions
  surfaced, real cited sources
- Model output shape drift is handled by tolerant coercion layers on every
  agent schema + explicit JSON shapes in every prompt

Note: Groq free tier has no embeddings API — document search uses the
deterministic hashing embedder (works, weaker semantics). Swap
`EMBEDDING_PROVIDER` when a free embedding source is configured.

### Docker

```bash
docker compose up --build
# backend :8000, frontend :5173, persistent volume for data
```

### Live deployment (free tiers)

Deploy the backend to [Render](https://render.com) and the frontend to
[Vercel](https://vercel.com) — both free:

1. **Backend (Render):** render.com → New → Blueprint → import this repo.
   Render reads `render.yaml` automatically. Set the secret `GROQ_API_KEY`
   in the dashboard; set `CORS_ORIGINS` to your Vercel URL after step 2.
2. **Frontend (Vercel):** vercel.com → Add New → Project → import this repo.
   **Set Root Directory to `frontend`** (Configure Project → Root Directory),
   then Settings → Environment Variables → add `VITE_API_URL` = your Render
   URL (e.g. `https://research-platform-api.onrender.com`). No other config
   needed — Vercel auto-detects the Vite build.
3. Redeploy the backend so CORS picks up the frontend URL.

Free-tier notes: Render sleeps after ~15 min idle (~50s cold start on the
next request); SSE streaming is supported by both platforms.

## Testing

82 tests across four layers:

- **Unit:** schemas/invariants, source scoring (recency/domain trust), budget
  logic, tool validation & budgets, calculator injection resistance, chunker,
  parser, embeddings, vector store, grounding guard
- **Integration:** full pipeline end-to-end, planner depth enforcement,
  structured-output repair pass, evidence grounding, event safety
- **API:** lifecycle (create → status → report), validation errors, 404s,
  document upload/list/delete, tool catalog, health
- **Failure:** search outage, unreachable sources, LLM provider death, zero
  search budget, cancellation mid-run — all must degrade, never crash, never
  fabricate.
- **Hybrid research:** session-scoped document search (no cross-session
  leakage), document evidence provenance by exact id mapping, research memory
  (no duplicate queries, followup consumption from critic verdicts).

CI (`.github/workflows/ci.yml`) runs all backend tests, the evaluation
framework (asserting ≥ 6 measured metrics), and the strict-TypeScript
frontend build on every push — fully offline, no secrets required.

## Limitations

1. Offline mock providers cannot semantically judge answerability or produce
   genuinely analytical synthesis — with mock LLM the report is structurally
   complete but analytically shallow.
2. The offline corpus is small and AI-topic-biased; live DuckDuckGo search is
   best-effort (no API key, HTML scraping) and can be rate-limited.
3. SQLite vector store scans linearly — fine for thousands of chunks, not
   millions (swap the backend per docs).
4. No auth layer; single-user assumption.
5. Evidence grounding is textual: it blocks fabrication, not misquoted-out-of-
   context snippets.
6. Prompt-injection detection is heuristic; the architectural guarantees are
   the real defense.

## Future improvements

- Alembic migrations; Postgres + pgvector backend
- SSE/WebSocket streaming (polling today)
- Per-question cached search/evidence reuse across sessions
- Human feedback loop on verdicts (annotation store)
- Egress allowlist + authn/authz + per-user quotas
- Re-ranking model (cross-encoder) for retrieval
- Langfuse/OpenTelemetry exporter for traces
