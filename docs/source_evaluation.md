# Source Evaluation Methodology

## Purpose

The Source Evaluation Agent assesses each candidate source before evidence
extraction. Its goal is to *prioritize* sources — never to pretend quality is
perfectly quantifiable.

## Scoring components

Scores are composites of an LLM judgment (relevance, authority) and
**deterministic code** (recency, domain trust). The deterministic parts are
unit-tested and auditable; the LLM parts are prompt-constrained and their
reasoning is stored in `evaluation_notes` for review.

### 1. Relevance (LLM, 0–1)
How directly the source addresses the subquestion. Prompt requires reasoning
to be given. A `discard` verdict zeroes the score.

### 2. Authority (LLM + deterministic domain tiers, 0–1)
The LLM judges authority from the source's self-presentation; the final value
is `max(llm_authority, domain_trust)`, where domain trust comes from a
maintained tier list:

- **0.9 — High trust:** arxiv.org, ACM, IEEE, official vendor docs
  (docs.python.org, learn.microsoft.com, MDN, OpenAI, Meta AI, Google Research)
- **0.7 — Medium trust:** established engineering blogs and communities
  (medium.com, dev.to, stackoverflow.com, huggingface.co, vendor tech blogs)
- **0.3 — Default:** any unknown domain
- **0.2 — Missing domain**

A source claiming high authority from an unknown domain cannot inflate its
score above its domain tier.

### 3. Recency (deterministic, 0–1)
Computed from the *stated* publication date only. Unknown dates get a neutral
0.5 — **the system never guesses dates.**

| Age          | Score |
|--------------|-------|
| ≤ 180 days   | 1.0   |
| ≤ 1 year     | 0.8   |
| ≤ 2 years    | 0.6   |
| ≤ 4 years    | 0.4   |
| older        | 0.2   |
| unknown      | 0.5   |

### 4. Composite trust
```
trust = 0.45 × relevance + 0.30 × authority + 0.25 × recency
```
Relevance dominates: an irrelevant-but-authoritative source still ranks low.
Sources with `trust ≤ 0` (explicit discard) are excluded from evidence
extraction.

## Primary vs secondary

The LLM classifies primary sources (official docs, papers, original data) vs
secondary (news coverage, summaries). This is surfaced in the UI and in the
report's source list.

## Known limitations

- Domain tiers are maintained by hand and biased toward technical/AI topics.
- LLM relevance judgments are imperfect; that is why extraction-then-grounding
  (snippet must appear verbatim in source text) provides the second guard.
- Recency weighting is domain-blind: some topics age slower than others.
