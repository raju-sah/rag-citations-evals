"""Build and persist FAISS (dense) and BM25 (sparse) indexes."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import bm25s
import faiss

from ragcite.chunk import Chunk, load_chunks_jsonl

logger = logging.getLogger(__name__)


def build_bm25_index(
    chunks: Sequence[Chunk], output_dir: str | Path = "data/index/bm25"
) -> bm25s.BM25:
    """Build and persist a BM25 sparse index using bm25s."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    corpus_texts = [c.text for c in chunks]
    logger.info("Tokenizing %d documents for BM25...", len(corpus_texts))
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en")

    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    retriever.save(str(out))
    logger.info("Saved BM25 index to %s", out)
    return retriever


def load_bm25_index(index_dir: str | Path = "data/index/bm25") -> bm25s.BM25:
    """Load a persisted bm25s index."""
    inp = Path(index_dir)
    if not inp.exists():
        raise FileNotFoundError(f"BM25 index directory not found: {inp}")
    return bm25s.BM25.load(str(inp))


def build_dense_index(
    chunks: Sequence[Chunk],
    model_name: str = "BAAI/bge-m3",
    output_dir: str | Path = "data/index",
    batch_size: int = 16,
    device: str = "cpu",
) -> tuple[faiss.Index, list[dict]]:
    """Embed chunks with SentenceTransformer and index with FAISS IndexFlatIP."""
    from sentence_transformers import SentenceTransformer

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Loading embedding model %s on %s ...", model_name, device)
    model = SentenceTransformer(model_name, device=device)

    corpus_texts = [c.text for c in chunks]
    logger.info("Encoding %d chunks (batch_size=%d)...", len(corpus_texts), batch_size)
    embeddings = model.encode(
        corpus_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    dimension = embeddings.shape[1]
    logger.info("Building FAISS IndexFlatIP with dimension %d...", dimension)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    index_file = out / "faiss.index"
    faiss.write_index(index, str(index_file))

    metadata = [
        {
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "section_id": c.section_id,
            "header_path": c.header_path,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "token_count": c.token_count,
            "text": c.text,
        }
        for c in chunks
    ]
    metadata_file = out / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)

    logger.info("Saved FAISS index to %s and metadata to %s", index_file, metadata_file)
    return index, metadata


def load_dense_index(
    index_dir: str | Path = "data/index",
) -> tuple[faiss.Index, list[dict]]:
    """Load FAISS index and chunk metadata."""
    inp = Path(index_dir)
    index_file = inp / "faiss.index"
    metadata_file = inp / "metadata.json"

    if not index_file.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_file}")
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    index = faiss.read_index(str(index_file))
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata


def build_indexes(
    chunks_path: str | Path = "data/processed/chunks.jsonl",
    output_dir: str | Path = "data/index",
    embedding_model: str = "BAAI/bge-m3",
    skip_dense: bool = False,
    device: str = "cpu",
) -> None:
    """Build both BM25 and FAISS indexes."""
    chunks = load_chunks_jsonl(chunks_path)
    logger.info("Loaded %d chunks from %s", len(chunks), chunks_path)

    out = Path(output_dir)
    # Build BM25 index
    build_bm25_index(chunks, output_dir=out / "bm25")

    # Build dense index
    if not skip_dense:
        build_dense_index(
            chunks,
            model_name=embedding_model,
            output_dir=out,
            device=device,
        )


def load_indexes(
    index_dir: str | Path = "data/index",
) -> tuple[faiss.Index, list[dict], bm25s.BM25]:
    """Load both dense and sparse indexes for retrieval."""
    inp = Path(index_dir)
    faiss_idx, metadata = load_dense_index(inp)
    bm25_idx = load_bm25_index(inp / "bm25")
    return faiss_idx, metadata, bm25_idx


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Build retrieval indexes")
    parser.add_argument("--chunks", default="data/processed/chunks.jsonl", help="Chunks JSONL path")
    parser.add_argument("--output-dir", default="data/index", help="Index output directory")
    parser.add_argument("--model", default="BAAI/bge-m3", help="SentenceTransformer model name")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--skip-dense", action="store_true", help="Only build BM25 index")
    args = parser.parse_args()

    build_indexes(
        chunks_path=args.chunks,
        output_dir=args.output_dir,
        embedding_model=args.model,
        skip_dense=args.skip_dense,
        device=args.device,
    )

