import tempfile
from pathlib import Path

import bm25s
import faiss
import numpy as np

from ragcite.chunk import Chunk
from ragcite.index import (
    build_bm25_index,
    load_bm25_index,
    load_dense_index,
)


def _make_sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="doc1:s1:c1",
            doc_id="doc1",
            section_id="doc1:s1",
            header_path="Doc 1 > Section 1: Employment",
            text="Doc 1 > Section 1: Employment\n\nContracts of employment and remuneration.",
            content="Contracts of employment and remuneration.",
            page_start=1,
            page_end=1,
            token_count=15,
        ),
        Chunk(
            chunk_id="doc1:s2:c1",
            doc_id="doc1",
            section_id="doc1:s2",
            header_path="Doc 1 > Section 2: Environment",
            text="Doc 1 > Section 2: Environment\n\nPollution control and environmental impact assessment.",
            content="Pollution control and environmental impact assessment.",
            page_start=2,
            page_end=2,
            token_count=16,
        ),
    ]


def test_bm25_index_build_and_load():
    chunks = _make_sample_chunks()
    with tempfile.TemporaryDirectory() as tmp_dir:
        retriever = build_bm25_index(chunks, output_dir=tmp_dir)
        assert retriever is not None

        loaded = load_bm25_index(tmp_dir)
        query_tokens = bm25s.tokenize(["remuneration"])
        results, _scores = loaded.retrieve(query_tokens, k=1)
        assert results[0][0] == 0  # Chunk 0 has 'remuneration'


def test_load_dense_index():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        dim = 16
        index = faiss.IndexFlatIP(dim)
        vecs = np.random.randn(2, dim).astype("float32")
        faiss.normalize_L2(vecs)
        index.add(vecs)
        faiss.write_index(index, str(tmp_path / "faiss.index"))

        import json

        with open(tmp_path / "metadata.json", "w") as f:
            json.dump([{"chunk_id": "c1"}, {"chunk_id": "c2"}], f)

        loaded_idx, meta = load_dense_index(tmp_path)
        assert loaded_idx.ntotal == 2
        assert len(meta) == 2
