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

| Config | Recall@5 | MRR | nDCG@5 | Faithfulness | Cite-P | Cite-R | Latency |
|---|---|---|---|---|---|---|---|
| _TBD_ | | | | | | | |

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
