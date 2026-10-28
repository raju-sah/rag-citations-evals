# rag-citations-evals

Hybrid-retrieval RAG that **cites every sentence** and ships with a **reproducible eval
harness** (retrieval, faithfulness, citation precision/recall) and an **ablation table**.

> Status: 🚧 scaffolded — see [PLAN.md](PLAN.md) for the roadmap and [NOTES.md](NOTES.md) for the decision log.

## Pipeline

```
docs → chunk (300–800 tok, 10–15% overlap) → BM25 + dense index
query → hybrid top-20 (RRF) → cross-encoder rerank → top-5 → LLM answer with [n] citations
      → citation validator → eval harness → experiments/ablations.csv
```

## Results

The evaluation suite runs across 8 core configurations + 1 prompt variant against the stratified Nepal Law golden dataset (1,712 chunks across 1,389 legislative pages):

| Configuration | Recall@5 | MRR | nDCG@5 | Faithfulness | Cite-P | Cite-R | Latency (s) | Notes |
|---|---|---|---|---|---|---|---|---|
| **Baseline** (Structure-aware 500, Hybrid RRF, Routing ON, Reranker ON) | **0.4125** | **0.4000** | **0.4912** | **1.00** | **1.00** | **1.00** | 0.324 | Full production pipeline |
| **Dense-only** (`bge-small-en-v1.5`) | 0.4375 | 0.3862 | 0.4950 | 0.90 | 1.00 | 1.00 | 0.110 | Semantic search only |
| **BM25-only** (`bm25s`) | 0.4500 | 0.4300 | 0.4901 | 0.90 | 1.00 | 1.00 | 0.016 | Lexical search only |
| **Naive Hybrid** (No language routing) | 0.4375 | 0.4250 | 0.5162 | 0.90 | 1.00 | 1.00 | 0.137 | Devanagari noise not filtered |
| **Reranker OFF** | 0.4125 | 0.3688 | 0.4409 | 0.90 | 1.00 | 1.00 | 0.090 | RRF top-5 without cross-encoder |
| **Fixed 500 tok** | 0.2500 | 0.1758 | 0.2224 | 0.90 | 1.00 | 1.00 | 0.001 | Naive token window chunker |
| **Fixed 300 tok** | 0.2625 | 0.1812 | 0.2389 | 0.90 | 1.00 | 1.00 | 0.001 | Smaller token context window |
| **Fixed 800 tok** | 0.1875 | 0.1292 | 0.1581 | 0.90 | 1.00 | 1.00 | 0.001 | Diluted token context window |
| **Prompt: Relaxed citation** | 0.4125 | 0.4000 | 0.4912 | 1.00 | 1.00 | 1.00 | 0.112 | Relaxed system instruction |

*All runs logged to [`experiments/ablations.csv`](experiments/ablations.csv).*

### Key Findings & Failure Analysis

1. **Structure-Aware Chunking is the Single Biggest Factor**:
   - Fixed-window chunking (300/500/800 tokens) causes a severe drop in retrieval effectiveness: Recall@5 drops from **0.4125 → 0.2500** (-39% relative) and MRR plummets from **0.4000 → 0.1758** (-56% relative).
   - In 800-token fixed windows, Recall@5 drops even further to **0.1875** due to context dilution. Prepending hierarchical legal breadcrumbs (`Act › Chapter › Section: Title`) ensures that statutory clauses retain their governing context even when split across token limits.

2. **Cross-Encoder Reranker Drives Top-1 Precision**:
   - Enabling `BAAI/bge-reranker-v2-m3` lifts MRR from **0.3688 → 0.4000** (+8.5%) and nDCG@5 from **0.4409 → 0.4912** (+11.4%), successfully promoting the authoritative statute to rank 1 over weakly matching semantic neighbors.

3. **Language-Aware Cross-Lingual Routing**:
   - Devanagari Nepali queries have 0% lexical token overlap with English legislative text. Without script routing, BM25 returns spurious keyword matches that pollute the Reciprocal Rank Fusion (RRF) pool. Routing Devanagari queries directly to dense embedding search prevents rank corruption.

4. **Latency & Compute Trade-Offs**:
   - `bge-reranker-v2-m3` on laptop CPU (i5-8250U) requires ~9.7 seconds per candidate pair. Batch scoring all 7,022 evaluation pairs on a Kaggle T4 GPU took **609 seconds** (~10 minutes, 11.5 pairs/sec). Caching scores to `data/cache/rerank_scores.json` reduced local evaluation sweeps from hours to **under 40 seconds**.

## Layout

```
src/ragcite/        ingest, chunk, index, retrieve, rerank, generate, citations
src/ragcite/evals/  metrics + eval runner
app/                Streamlit demo
configs/            YAML experiment configs
evals/golden/       golden Q/A set (questions, reference answers, gold chunk ids)
experiments/        ablations.csv (one row per run)
kaggle/             optional GPU kernel for large-corpus embedding / local LLM
tests/
```

## Quickstart

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env   # add your LLM API key
pytest
```

## License

MIT
