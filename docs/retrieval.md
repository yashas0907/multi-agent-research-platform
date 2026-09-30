# Retrieval & RAG Design

## Pipeline

```
Document (upload)          Web source (fetched)
      │                          │
      ▼                          ▼
   Parser ── (txt/md/pdf/html/json, defensive)
      │
      ▼
   Cleaner ── (normalize whitespace, drop junk lines)
      │
      ▼
   Chunker ── (paragraph-aware sliding window, ~180 words, 40-word overlap)
      │
      ▼
   Metadata ── (title, content_type, document_id, chunk_index)
      │
      ▼
   Embeddings ── (provider abstraction: mock hashing | OpenAI)
      │
      ▼
   Vector Store ── (SQLiteVectorStore | swappable backend)
      │
      ▼
   Retriever ── (hybrid: cosine + keyword overlap, score gating)
      │
      ▼
   Evidence ── (grounded snippets with provenance)
```

## Design decisions

### Hybrid retrieval, not blind top-k
The retriever fetches `top_k × 3` candidates from the vector store, re-ranks
by `0.7 × cosine + 0.3 × keyword-overlap`, then **drops everything below a
minimum combined score** before trimming to top-k. Retrieval results that are
all weak are reported as empty rather than passed along.

### Metadata filtering
`Retriever.retrieve(document_ids=...)` restricts search to a session's
documents — the same interface supports future per-user / per-corpus
filtering.

### Swappable vector backend
`VectorStore` is a 5-method interface (add/query/delete/count). The default
`SQLiteVectorStore` keeps the system dependency-free; production backends
(pgvector, Qdrant, Milvus) implement the same interface. See
`app/retrieval/vector_store.py`.

### Mock embeddings (offline)
Deterministic feature-hashing (SHA-256 buckets, log-scaled TF, L2
normalized). Documents sharing terms land near each other — enough signal for
offline development, CI, and reproducible evaluation. Switch to
`EMBEDDING_PROVIDER=openai` for real semantics; the interface is identical.

### Grounding
Every evidence snippet must appear verbatim (or ≥ 80% 8-gram overlap) in the
source text — enforced in `EvidenceAgent` at runtime. This is the
anti-fabrication backstop and doubles as the faithfulness metric in
evaluation.

## Search resilience chain (live-deployment learnings)

```
web_search: DuckDuckGo ──(blocked/empty)──► Wikipedia ──(empty)──► offline corpus
```

Found during live deployment verification (Render, Oregon datacenter):

1. **DuckDuckGo tarpits datacenter IPs** — HTML scraping times out (15s) from
   cloud providers while working fine from residential IPs. Local testing
   alone would never catch this.
2. **Wikipedia API works from datacenters** — free, no key, reliable. Covers
   encyclopedic topics; the REST summary endpoint supplies actual definitions
   for `what is X` / `definition of X` queries.
3. **Dictionary words have no Wikipedia article** — single-word definitional
   queries also get the **Wiktionary** entry (the free dictionary), where
   `blatant`, `obfuscate`, etc. actually live.
4. **Evidence extraction window is relevance-picked** — wiki pages front-load
   hundreds of chars of navigation; Wiktionary's definition of "blatant" sat
   at char ~2200, right at a naive head-of-text window's edge. The 4000-char
   window is now selected by keyword overlap with the subquestion.

The chain guarantees every deployed research session has real, citable
sources even when search engines block the datacenter.
