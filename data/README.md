# Data

Raw and processed data are git-ignored.

- `data/raw/` — source documents (corpus: **TBD**). Record source URL + license here.
- `data/processed/chunks.jsonl` — `{chunk_id, doc_id, page, section, text}`
- `data/index/` — persisted FAISS + BM25 indexes
