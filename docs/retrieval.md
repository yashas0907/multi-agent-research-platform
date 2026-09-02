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
