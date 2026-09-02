# Confidence & Uncertainty Model

The platform never presents arbitrary LLM confidence numbers as probabilities.
Instead it assigns **evidence-based categories**:

| Category | Meaning |
|----------|---------|
| `HIGH_EVIDENCE` | ≥ 2 independent credible sources agree; no unresolved contradiction |
| `MODERATE_EVIDENCE` | 1 credible source, or ≥ 2 weaker sources; no contradiction |
| `LOW_EVIDENCE` | Single weak source, or contradicted-but-plausible |
| `INSUFFICIENT_EVIDENCE` | No usable evidence located |

## Where each category is assigned

1. **Evidence items** (`Evidence.confidence`): the extraction agent rates how
   directly the snippet supports the claim (high requires direct, unambiguous
   support in the snippet).
2. **Subquestions** (`SubQuestion.confidence`): deterministic code from the
   evidence set — ≥ 2 high-confidence items → HIGH; 1 high or ≥ 2 any →
   MODERATE; else LOW. No evidence → INSUFFICIENT.
3. **Findings** (`Finding.confidence`): the synthesis agent must justify the
   category from cited evidence; the citation auditor flags findings whose
   stated confidence is not supported by their evidence links, and unlinked
   factual findings are downgraded to interpretations.

## Enforcement rules (code, not prompts)

- A claim with no verdict from the fact-checker stays `INSUFFICIENT_EVIDENCE`.
- Evidence ids that do not exist are stripped from findings at synthesis.
- Findings with zero evidence links lose factual status (marked as
  interpretation with an explicit caveat).
- Reports always include a `confidence_summary` distribution so readers see
  how strong the overall evidence base is.

## Why categories, not numbers

A "0.87 confidence" from an LLM is not a calibrated probability and invites
false precision. Categories map directly onto countable properties of the
evidence (number of sources, their independence, contradictions), making them
auditable and reproducible.
