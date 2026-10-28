# PLAN — RAG with Citations and Evals

> Project 1 of 3 · Difficulty ⭐⭐ · GPU: optional (0–5 h Kaggle) · Weeks 1–2

## What it is

Retrieval-Augmented Generation: the user asks a question, we retrieve relevant chunks from
our documents, and the LLM answers **using only those chunks**. Every claim cites the chunk it
came from, e.g. `[2]`. Evals measure (a) whether retrieval found the right chunks and (b)
whether the answer is faithful to them.

**Portfolio thesis:** the eval harness + ablation table matter more than the chat UI.
Target headline: *"hybrid + reranker raised Recall@5 from 0.62 → 0.84 and faithfulness 0.71 → 0.90"*.

## Stack (one per row — ✅ = current default, to be confirmed in /grill-me)

| Layer | Options | Default |
|---|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5`, `bge-m3`, `intfloat/e5`, API embeddings | ✅ `bge-small-en-v1.5` (CPU-friendly) |
| Vector store | FAISS / Chroma / LanceDB / Qdrant | ✅ FAISS (simple, no server) |
| Keyword search | BM25 (`rank_bm25`) | ✅ BM25 + vectors → hybrid (RRF) |
| Reranker | `BAAI/bge-reranker-v2-m3` | ✅ on (toggle for ablation) |
| LLM | Gemini / OpenAI / Claude API, or local Qwen 3B–7B (Ollama/vLLM) | ✅ TBD |
| Evals | RAGAS, DeepEval, own LLM-as-judge + metrics | ✅ own metrics + RAGAS |
| UI | Streamlit / Gradio | ✅ Streamlit |

## Open decisions (resolve before coding)

- [ ] **Corpus** — Nepal law docs? a framework's docs? research papers? college notes?
- [ ] **LLM provider** — which API key is available (Gemini free tier is cheapest)?
- [ ] **Judge LLM** — same as generator or a different/stronger model?
- [ ] **Golden-set size** — 50 (fast) vs 200 (credible)
- [ ] **Language** — English-only, or Nepali too (→ `bge-m3` multilingual)?

## Steps

### Phase 1 — Pipeline (Week 1)
- [ ] 1. **Pick corpus**, drop raw files in `data/raw/`, document source + license in `data/README.md`.
- [ ] 2. **Ingest & chunk** (`src/ragcite/ingest.py`, `chunk.py`): ~300–800 tokens, ~10–15% overlap.
      Metadata per chunk: `doc_id, page, section, chunk_id`. Output `data/processed/chunks.jsonl`.
- [ ] 3. **Index** (`index.py`): embedding index (FAISS) + BM25 index, persisted to `data/index/`.
- [ ] 4. **Retrieve** (`retrieve.py`, `rerank.py`): hybrid top-20 (RRF fusion) → rerank → top-5.
- [ ] 5. **Generate with citations** (`generate.py`): number chunks `[1]..[5]` in the prompt.
      Instruction: *"Answer only from the sources. Cite every sentence with [n]. If the answer
      isn't in the sources, say so."*
- [ ] 6. **Validate citations** (`citations.py`): parse `[n]`, check each exists; optional NLI /
      LLM-judge check that the cited chunk supports the sentence.
- [ ] 7. **Streamlit UI** (`app/streamlit_app.py`): answer + clickable sources panel.

### Phase 2 — Evals & ablations (Week 2)
- [ ] 8. **Golden eval set** (`evals/golden/`): 50–200 questions with reference answers +
      gold chunk IDs. LLM-generated, then **hand-verified**. Include ~10–15% *unanswerable* questions.
- [ ] 9. **Eval harness** (`src/ragcite/evals/`): one command → one row in `experiments/ablations.csv`.
- [ ] 10. **Ablations** — change ONE thing at a time:
      chunk size (300/500/800) · k · BM25 vs dense vs hybrid · reranker on/off · prompt variant.
- [ ] 11. **Write-up** in README: ablation table, plots, failure analysis, cost/latency.

## Metrics

| Layer | Metric |
|---|---|
| Retrieval | Recall@k, MRR, nDCG@k — did we fetch the right chunk? |
| Generation | Faithfulness / groundedness, answer relevancy, correctness vs reference |
| Citations | Citation precision (cited chunks support the claim?) & recall (every claim cited?) — see ALCE |
| System | Latency, cost/query, refusal rate on unanswerable questions |

## Compute

Laptop CPU is enough. Use Kaggle (`kaggle/`) only to embed a very large corpus or run a
local 7B LLM. Budget: 0–5 GPU h out of ~30 h/week.

## Definition of done

- [ ] `make eval` (or `python -m ragcite.evals.run`) reproduces the ablation table end-to-end
- [ ] ≥ 5 ablation rows with retrieval + generation + citation metrics
- [ ] Streamlit demo + screenshot/GIF in README
- [ ] Short write-up: what helped, what didn't, failure examples
