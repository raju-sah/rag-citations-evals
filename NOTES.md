# NOTES — RAG with Citations and Evals

Running log of decisions, findings, and gotchas. Newest first.

## Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-10-28 | Repo scaffolded; stack defaults in PLAN.md pending /grill-me | — |

## Reference notes

- **Hybrid fusion:** Reciprocal Rank Fusion `score = Σ 1/(k + rank)`, `k≈60`. No score
  normalisation needed between BM25 and cosine.
- **Reranker:** `bge-reranker-v2-m3` is a cross-encoder — score (query, chunk) pairs for the
  top-20 only; it's the slowest step on CPU (~0.5–2 s for 20 pairs).
- **Citation precision/recall (ALCE, Gao et al. 2023):**
  - recall = fraction of answer sentences whose cited chunks (jointly) entail the sentence
  - precision = fraction of individual citations that are actually needed/supportive
- **Golden set hygiene:** generate Qs per chunk with an LLM, then hand-verify; dedupe near-
  identical questions; keep a held-out slice you never look at while tuning.
- **Unanswerable questions:** essential for measuring refusal rate; otherwise the system is
  rewarded for hallucinating.

## Experiment log

See `experiments/ablations.csv` — one row per run. Add free-text observations here.

## Gotchas

- _(fill in as you hit them)_
