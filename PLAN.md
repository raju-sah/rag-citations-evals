# PLAN — RAG with Citations and Evals

> Project 1 of 3 · Difficulty ⭐⭐ · GPU: optional (1–2 h Kaggle T4 for batch embedding/rerank caching) · Weeks 1–2

## What it is

Retrieval-Augmented Generation: the user asks a question (in English or Nepali), we retrieve relevant chunks from
our legal corpus, and the LLM answers **using only those chunks**. Every claim cites the chunk it
came from, e.g. `[2]`. Evals measure (a) whether retrieval found the right legal sections and (b)
whether the answer is faithful to them.

**Portfolio thesis:** the eval harness + ablation table matter more than the chat UI.
Target headline: *"hybrid + reranker raised Recall@5 from 0.62 → 0.86 and faithfulness 0.71 → 0.92, while language-aware routing preserved Nepali retrieval accuracy"*.

## Stack (Finalized via /grill-me)

| Layer | Selection | Rationale |
|---|---|---|
| Corpus | Nepal Law (Constitution 2015 + ~12 key Acts + Muluki Codes) | Realistic legal structure (Part/Article/Section), low memorization in LLMs, high domain value. |
| Embeddings | `BAAI/bge-m3` | Multilingual (EN + Nepali), 8192 context; embedded via Kaggle T4, fast query inference on CPU. |
| Vector store | FAISS (`IndexFlatIP`) | Simple exact search on CPU, no server overhead, 100% deterministic. |
| Keyword search | `bm25s` + language-aware routing | Fast sparse search with native disk serialization; Devanagari queries bypass BM25 to avoid noise. |
| Reranker | `BAAI/bge-reranker-v2-m3` (top-20 → top-5) | State-of-the-art cross-encoder; batch scores cached for evals via Kaggle T4; CPU latency benchmarked for demo. |
| LLM Generator | Google Gemini Flash (`temperature=0.0`) | Fast, capable in English and Nepali, strict instruction following; calls cached to disk. |
| Judge LLM | Groq (`llama-3.3-70b-versatile`) + Gemini fallback | Cross-family evaluation to avoid self-preference bias; validated against 40 human calibration items (Cohen's κ). |
| Evals | Own metrics + lightweight RAGAS cross-validation | ALCE citation precision/recall, JSON-schema LLM judge for faithfulness, retrieval metrics, 20-sample RAGAS check. |
| UI | Streamlit 2-tab demo | Tab 1: Interactive Q&A with citation badges, sources drawer, and reranker toggle; Tab 2: Ablation dashboard. |

## Resolved Decisions

- [x] **Corpus** — Nepal Law: English translations of Constitution 2015 + ~12 key Acts + Muluki Civil and Criminal Codes (~6–8k chunks).
- [x] **Language** — Cross-lingual day one: English documents, English + Nepali queries (answering in query language). Nepali documents gated on a 1-hour extraction spike (clean Unicode).
- [x] **LLM Generator** — Gemini Flash via free Google AI Studio key, with strict disk-based caching keyed on `(model, prompt)`.
- [x] **Judge LLM** — Cross-family judge on Groq free tier (`llama-3.3-70b-versatile`) + 40-item human calibration set (reporting Cohen's κ); Gemini fallback for Nepali items if κ is low.
- [x] **Golden-set size** — 200+ questions (~150 dev, ~50 held-out test), stratified mix (~25–30% Nepali, ~15% unanswerable, ~15% multi-section, paraphrased), section-level gold IDs.
- [x] **Chunking strategy** — Structure-aware baseline (one chunk per Article/Section, max 500 tokens, sub-section splitting, header path prepended); fixed 300/500/800 windows as ablations.
- [x] **Hybrid fusion** — RRF (k=60) + language-aware routing (Devanagari query → dense only); `bm25s` + FAISS FlatIP.
- [x] **Compute & reranker** — Kaggle T4 (~1–2 GPU h) for corpus embeddings and golden-set rerank score caching; laptop CPU for query inference, UI demo, and eval execution.
- [x] **Eval framework** — Own transparent metrics in `src/ragcite/evals/` (ALCE citation P/R, custom prompt-cached LLM judge, exact retrieval metrics) + 20-sample RAGAS correlation check.
- [x] **Ablations** — 8 core rows (Baseline, Dense-only, BM25-only, Naive hybrid without routing, Reranker OFF, Fixed 500, Fixed 300, Fixed 800) + 1 prompt ablation.
- [x] **UI scope** — 2-tab Streamlit app: (1) Q&A demo with inline citation badges, expandable sources drawer, and reranker toggle; (2) Ablations & metrics dashboard reading `experiments/ablations.csv`.

## Steps

### Phase 1 — Pipeline (Week 1)
- [x] 1. **Ingest corpus**: Download Constitution, ~12 Acts, and Muluki Codes from Law Commission into `data/raw/`. Run 1-hour Nepali PDF extraction spike to determine if Nepali docs are clean Unicode.
- [x] 2. **Structure-aware chunker** (`src/ragcite/chunk.py`, `ingest.py`): Extract hierarchy (Act › Chapter › Section/Article), split large sections at subsection boundaries (max 500 tokens), merge tiny definitions (<80 tokens), prepend header path. Persist `data/processed/chunks.jsonl`.
- [x] 3. **Index builder** (`src/ragcite/index.py`): Kaggle script/local fallback to produce `bge-m3` FAISS index + `bm25s` sparse index. Persist to `data/index/`.
- [x] 4. **Retrieve & rerank** (`src/ragcite/retrieve.py`, `rerank.py`): Hybrid top-20 with language routing (Devanagari → dense-only; English → RRF k=60) → `bge-reranker-v2-m3` cross-encoder → top-5. Benchmark CPU rerank latency.
- [x] 5. **Generate with citations** (`src/ragcite/generate.py`): Prompt with numbered sources `[1]..[5]`, strict sentence-level citation requirement, refusal on unsupported questions. Disk-cached LLM client.
- [x] 6. **Citation parser & validator** (`src/ragcite/citations.py`): Parse `[n]` citations per sentence, validate source bounds, compute structural citation recall.
- [x] 7. **Streamlit UI** (`app/streamlit_app.py`): Tab 1 interactive demo with citation badges, sources drawer, and reranker toggle; Tab 2 ablation viewer.

### Phase 2 — Evals & ablations (Week 2)
- [x] 8. **Golden eval set** (`evals/golden/`): 200+ questions (~150 dev, ~50 test) with reference answers + gold section IDs. 25–30% Nepali, 15% unanswerable, 15% multi-section. LLM-drafted + hand-verified. Create 40-item calibration set.
- [x] 9. **Eval harness** (`src/ragcite/evals/`): Retrieval metrics (Recall@k, MRR, nDCG@k), ALCE citation precision/recall, JSON-schema LLM judge for faithfulness & answer relevancy, Cohen's κ calculation. Optional 20-sample RAGAS check.
- [x] 10. **Run 8 ablations**: Log rows to `experiments/ablations.csv` with bootstrap 95% confidence intervals.
- [x] 11. **Write-up** in README: Ablation table, failure analysis, cost/latency breakdown.

## Definition of done

- [x] `python scripts/run_ablations.py` reproduces all 8+ ablation rows end-to-end
- [x] Judge calibration harness in place on 40-item hand-verified human calibration set
- [x] Streamlit 2-tab demo working with Q&A citation drawer and ablation dashboard
- [x] Detailed write-up: what helped, what failed (e.g. naive hybrid on Nepali queries, fixed vs structure-aware chunking), and latency/cost trade-offs
