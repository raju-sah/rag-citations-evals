"""Hybrid retrieval: Dense (FAISS) + Sparse (BM25) fused with Reciprocal Rank Fusion.

Includes language-aware routing: Devanagari queries bypass BM25 when searching
English corpora to avoid score pollution from lexical vocabulary mismatch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import bm25s

from ragcite.index import load_bm25_index, load_dense_index

logger = logging.getLogger(__name__)


@dataclass
class RetrievedCandidate:
    """A single retrieved candidate chunk with retrieval provenance."""

    chunk_id: str
    doc_id: str
    section_id: str
    header_path: str
    text: str
    page_start: int
    page_end: int
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None
    final_rank: int = 0

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "section_id": self.section_id,
            "header_path": self.header_path,
            "text": self.text,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "score": self.score,
            "dense_rank": self.dense_rank,
            "bm25_rank": self.bm25_rank,
            "final_rank": self.final_rank,
        }


def is_devanagari(text: str) -> bool:
    """Detect if a query is predominantly in Devanagari script."""
    stripped = text.replace(" ", "")
    if not stripped:
        return False
    devanagari_chars = sum(1 for ch in stripped if "\u0900" <= ch <= "\u097f")
    return (devanagari_chars / len(stripped)) > 0.25


def reciprocal_rank_fusion(
    rankings: list[list[str]], rrf_k: int = 60
) -> list[tuple[str, float]]:
    """Combine ranked candidate IDs using Reciprocal Rank Fusion: score = Σ 1 / (k + rank)."""
    fused_scores: dict[str, float] = {}
    for rank_list in rankings:
        for rank, cid in enumerate(rank_list, start=1):
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    """Manages dense FAISS index, sparse BM25 index, and cross-lingual routing."""

    def __init__(
        self,
        index_dir: str | Path = "data/index",
        embedding_model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
    ) -> None:
        self.index_dir = Path(index_dir)
        self.embedding_model_name = embedding_model_name
        self.device = device
        self._embedder = None

        # Load dense index & metadata
        self.faiss_index, self.metadata = load_dense_index(self.index_dir)
        self.chunk_by_id = {m["chunk_id"]: m for m in self.metadata}
        self.chunk_by_idx = {i: m for i, m in enumerate(self.metadata)}

        # Load BM25 index
        self.bm25_index = load_bm25_index(self.index_dir / "bm25")

    @property
    def embedder(self):
        """Lazy load SentenceTransformer embedder."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedder %s for queries on %s...", self.embedding_model_name, self.device)
            self._embedder = SentenceTransformer(self.embedding_model_name, device=self.device)
        return self._embedder

    def search_dense(self, query: str, top_k: int = 20) -> list[tuple[str, float, int]]:
        """Dense embedding search via FAISS. Returns [(chunk_id, cosine_score, rank)]."""
        query_vec = self.embedder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")

        scores, indices = self.faiss_index.search(query_vec, top_k)
        results: list[tuple[str, float, int]] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx in self.chunk_by_idx:
                cid = self.chunk_by_idx[idx]["chunk_id"]
                results.append((cid, float(score), rank))
        return results

    def search_bm25(self, query: str, top_k: int = 20) -> list[tuple[str, float, int]]:
        """Sparse search via bm25s. Returns [(chunk_id, bm25_score, rank)]."""
        query_tokens = bm25s.tokenize([query], stopwords="en")
        results, scores = self.bm25_index.retrieve(query_tokens, k=top_k)

        out: list[tuple[str, float, int]] = []
        for rank, (idx, score) in enumerate(zip(results[0], scores[0]), start=1):
            if idx in self.chunk_by_idx:
                cid = self.chunk_by_idx[idx]["chunk_id"]
                out.append((cid, float(score), rank))
        return out

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        mode: str = "hybrid",
        route_devanagari: bool = True,
        rrf_k: int = 60,
    ) -> list[RetrievedCandidate]:
        """Execute retrieval according to specified mode and script routing."""
        query_is_devanagari = is_devanagari(query)

        # Handle routing: Devanagari query skips BM25 if routing enabled
        effective_mode = mode
        if mode == "hybrid" and query_is_devanagari and route_devanagari:
            logger.info("Query detected as Devanagari; routing to dense-only retrieval.")
            effective_mode = "dense"

        if effective_mode == "dense":
            dense_res = self.search_dense(query, top_k=top_k)
            candidates: list[RetrievedCandidate] = []
            for rank, (cid, score, d_rank) in enumerate(dense_res, start=1):
                meta = self.chunk_by_id[cid]
                candidates.append(
                    RetrievedCandidate(
                        chunk_id=cid,
                        doc_id=meta["doc_id"],
                        section_id=meta["section_id"],
                        header_path=meta["header_path"],
                        text=meta["text"],
                        page_start=meta["page_start"],
                        page_end=meta["page_end"],
                        score=score,
                        dense_rank=d_rank,
                        bm25_rank=None,
                        final_rank=rank,
                    )
                )
            return candidates

        elif effective_mode == "bm25":
            bm25_res = self.search_bm25(query, top_k=top_k)
            candidates = []
            for rank, (cid, score, b_rank) in enumerate(bm25_res, start=1):
                meta = self.chunk_by_id[cid]
                candidates.append(
                    RetrievedCandidate(
                        chunk_id=cid,
                        doc_id=meta["doc_id"],
                        section_id=meta["section_id"],
                        header_path=meta["header_path"],
                        text=meta["text"],
                        page_start=meta["page_start"],
                        page_end=meta["page_end"],
                        score=score,
                        dense_rank=None,
                        bm25_rank=b_rank,
                        final_rank=rank,
                    )
                )
            return candidates

        elif effective_mode == "hybrid":
            dense_res = self.search_dense(query, top_k=top_k)
            bm25_res = self.search_bm25(query, top_k=top_k)

            dense_ranks = {cid: r for cid, _, r in dense_res}
            bm25_ranks = {cid: r for cid, _, r in bm25_res}

            dense_order = [cid for cid, _, _ in dense_res]
            bm25_order = [cid for cid, _, _ in bm25_res]

            fused = reciprocal_rank_fusion([dense_order, bm25_order], rrf_k=rrf_k)
            top_fused = fused[:top_k]

            candidates = []
            for rank, (cid, score) in enumerate(top_fused, start=1):
                meta = self.chunk_by_id[cid]
                candidates.append(
                    RetrievedCandidate(
                        chunk_id=cid,
                        doc_id=meta["doc_id"],
                        section_id=meta["section_id"],
                        header_path=meta["header_path"],
                        text=meta["text"],
                        page_start=meta["page_start"],
                        page_end=meta["page_end"],
                        score=score,
                        dense_rank=dense_ranks.get(cid),
                        bm25_rank=bm25_ranks.get(cid),
                        final_rank=rank,
                    )
                )
            return candidates

        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")


def hybrid_search(
    query: str,
    top_k: int = 20,
    index_dir: str | Path = "data/index",
    mode: str = "hybrid",
    route_devanagari: bool = True,
) -> list[RetrievedCandidate]:
    """Convenience functional wrapper for hybrid retrieval."""
    retriever = HybridRetriever(index_dir=index_dir)
    return retriever.retrieve(
        query=query, top_k=top_k, mode=mode, route_devanagari=route_devanagari
    )

