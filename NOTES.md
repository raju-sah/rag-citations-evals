# NOTES — RAG with Citations and Evals

Running log of decisions, findings, and gotchas. Newest first.

## Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-10-28 | **UI Scope**: 2-tab Streamlit app (Q&A demo with inline citations + sources drawer + reranker toggle; Ablation dashboard). | Keeps UI focused and clean (<200 lines) while highlighting both the interactive legal assistant and the eval results. |
| 2026-10-28 | **Ablation Plan**: 8 core rows (Baseline, Dense-only, BM25-only, Naive hybrid without routing, Reranker OFF, Fixed 500 tok, Fixed 300 tok, Fixed 800 tok) + 1 prompt ablation. | Isolates one variable at a time against baseline to demonstrate exact value of hybrid search, cross-encoder, structure-aware chunking, and language routing. |
| 2026-10-28 | **Eval Framework**: Own custom metrics first (ALCE cite-P/R, prompt-cached LLM judge with JSON schema, retrieval metrics) + 20-sample RAGAS cross-validation. | Transparent, fully auditable, avoids heavy black-box dependencies while ensuring strict sentence-level citation verification. |
| 2026-10-28 | **Compute & Reranking**: Kaggle T4 (1–2 GPU h) for corpus embeddings & eval rerank-score caching; laptop CPU for query inference and local evaluation runs. | Prevents CPU bottleneck on bge-reranker-v2-m3 (which would take ~2.5 h per 200-question eval on CPU) while remaining laptop-executable. |
| 2026-10-28 | **Hybrid Fusion**: RRF (k=60) with language-aware routing (Devanagari query → dense-only), `bm25s`, FAISS `IndexFlatIP`. | Avoids BM25 vocabulary mismatch degrading Nepali query retrieval when searching English corpus; `bm25s` provides fast serialization. |
| 2026-10-28 | **Chunking Strategy**: Structure-aware baseline (one chunk per Article/Section, max 500 tokens, sub-section splitting, header path prepended); fixed 300/500/800 windows as ablations. | Legal text is naturally hierarchical; section header paths (`Act › Chapter › Section`) boost both dense and sparse retrieval accuracy and provide clean citations. |
| 2026-10-28 | **Golden Eval Set**: 200+ questions (~150 dev / ~50 test), stratified mix (~25–30% Nepali, ~15% unanswerable, ~15% multi-section), section-level gold IDs. | Statistically rigorous sample size for credible confidence intervals; section-level IDs prevent invalidating ground truth across chunk-size ablations. |
| 2026-10-28 | **Judge LLM**: Cross-family Groq (`llama-3.3-70b-versatile`) + 40-item human calibration set reporting Cohen's κ; Gemini fallback for Nepali. | Avoids LLM judge self-preference bias; reporting human agreement makes faithfulness metrics credible for portfolio evaluation. |
| 2026-10-28 | **LLM Generator**: Gemini Flash (free AI Studio key, `temp=0.0`) with disk-based cache keyed on `(model, prompt)`. | Free tier, high instruction following for sentence citations `[n]`, bilingual quality; disk cache avoids hitting rate limits on repeat evals. |
| 2026-10-28 | **Language**: Cross-lingual day one (English documents, English + Nepali queries via `bge-m3`); Nepali documents gated on 1h extraction spike. | Native cross-lingual retrieval works out-of-the-box with `bge-m3`; prevents legacy Devanagari font extraction issues from blocking initial progress. |
| 2026-10-28 | **Corpus**: Nepal Law (Constitution 2015 + ~12 key Acts + Muluki Civil & Criminal Codes). | High legal structure, low prior LLM pre-training memorization, strong real-world motivation for sentence-level citation verification. |
| 2026-10-28 | Repo scaffolded; stack defaults in PLAN.md resolved via /grill-me. | Initial repository setup. |

## Reference notes

- **Language-aware routing:** For Devanagari queries against English text, word-level token overlap is zero. BM25 produces near-random scores. Routing Devanagari queries directly to dense search avoids polluting RRF rank positions.
- **Section-level evaluation:** Evaluating retrieval against section IDs (e.g. `labour-act-2017:s12`) instead of chunk IDs (e.g. `c042`) ensures that changing chunk size (300 vs 500 vs 800) does not invalidate ground truth labels. A retrieved chunk counts as a hit if its section ID intersects the gold section IDs.
- **Hybrid fusion:** Reciprocal Rank Fusion `score = Σ 1/(k + rank)`, `k≈60`. No score normalisation needed between BM25 and cosine.
- **Reranker latency:** `bge-reranker-v2-m3` has 568M parameters. On CPU (i5-8250U), 20 pairs takes ~30–60s per query. Caching (query, candidate) reranker scores on Kaggle T4 enables instant, free local eval sweeps.
- **Citation precision/recall (ALCE, Gao et al. 2023):**
  - recall = fraction of answer sentences whose cited chunks (jointly) entail the sentence
  - precision = fraction of individual citations that are actually needed/supportive
- **Golden set hygiene:** LLM-draft questions per section, run an anti-leak paraphrase pass to remove lexical overlap that unfairly benefits BM25, then hand-verify. Keep 50 items strictly held out.
- **Unanswerable questions:** Essential for measuring refusal rate; otherwise the system is rewarded for hallucinating.


## Experiment log

See `experiments/ablations.csv` — one row per run. Add free-text observations here.

## Gotchas

- _(fill in as you hit them)_
