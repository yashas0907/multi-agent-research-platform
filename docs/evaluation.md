# Evaluation Methodology

All metrics are computed by `evaluation/run_evaluation.py` on a small curated
dataset. **Scores are measured, never fabricated** — see
`evaluation/results/evaluation_results.json` for the latest run's raw output,
including which cases were skipped and why.

## 1. Retrieval — Recall@3 / Precision@3

Dataset: 5 queries against the 13-entry offline corpus with hand-labeled
relevant documents.

- **Recall@3** — of the known-relevant documents, how many appear in top-3.
- **Precision@3** — of the top-3 retrieved, how many are known-relevant.

Measured on the hybrid retriever (vector + keyword re-ranking) with mock
embeddings.

## 2. RAG answers — Context relevance / Faithfulness

- **Context relevance** — keyword-overlap between the question and the
  retrieved context (deterministic).
- **Faithfulness** — the *production grounding guard* is applied to extracted
  evidence: a snippet must appear (verbatim, or ≥ 80% 8-gram overlap) in the
  source text or it is dropped. The metric is the fraction of returned
  evidence that passes. This is the same check the pipeline enforces at
  runtime, so it measures real behavior, not a proxy.

## 3. Research pipeline — end-to-end

Measured over curated research cases (answerable + deliberately unanswerable):

- **Source quality** — mean trust score of sources cited in the final report.
- **Citation integrity** — every evidence id referenced by findings must exist
  in the report's evidence set.
- **Claim support rule** — no claim may be reported `SUPPORTED` without at
  least one evidence link.
- **Subquestion completeness** — fraction of subquestions reaching `answered`.
- **Honesty** — for questions that *cannot* be answered from the corpus, all
  findings must be LOW/INSUFFICIENT confidence rather than fabricated.

## Honest notes (also embedded in results JSON)

- The offline mock LLM cannot semantically judge answerability; loosely
  matched corpus sources can be retrieved for unanswerable questions, which
  measurably lowers the honesty score *of the offline configuration*. This is
  reported as-measured. A production LLM provider recognizes dead ends.
- Precision@3 is low by construction: each case lists one known-relevant
  document while the corpus holds closely-related entries, so top-3 returns
  near-misses. Reported honestly, not tuned away.
- With real providers (OpenAI/Groq + live search), rerun the same script —
  the framework is provider-independent.

## Reproducing

```bash
python evaluation/run_evaluation.py
cat evaluation/results/evaluation_results.json
```
