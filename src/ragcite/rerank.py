"""Cross-encoder reranking (BAAI/bge-reranker-v2-m3): top-20 -> top-5.

Includes disk-based score caching to enable instant, zero-cost repeatable evaluation
runs without repeatedly bottlenecking on CPU cross-encoders.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path

from ragcite.retrieve import RetrievedCandidate

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_PATH = Path("data/cache/rerank_scores.json")


def _load_cache(cache_path: Path) -> dict[str, float]:
    """Load cached reranker scores."""
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as err:
            logger.warning("Failed to read reranker cache %s: %s", cache_path, err)
    return {}


def _save_cache(cache_path: Path, cache: dict[str, float]) -> None:
    """Save reranker scores cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
    except OSError as err:
        logger.warning("Failed to save reranker cache %s: %s", cache_path, err)


class CrossEncoderReranker:
    """Wraps cross-encoder model scoring with disk cache support."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        cache_path: str | Path | None = _DEFAULT_CACHE_PATH,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.cache_path = Path(cache_path) if cache_path else None
        self.device = device
        self._model = None
        self.cache: dict[str, float] = (
            _load_cache(self.cache_path) if self.cache_path else {}
        )

    @property
    def model(self):
        """Lazy load CrossEncoder."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder %s on %s...", self.model_name, self.device)
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def score_pairs(self, query: str, candidates: Sequence[RetrievedCandidate]) -> list[float]:
        """Score (query, candidate.text) pairs using cache where possible."""
        scores: list[float] = []
        uncached_indices: list[int] = []
        uncached_pairs: list[tuple[str, str]] = []

        for idx, cand in enumerate(candidates):
            key = f"{query}::{cand.chunk_id}"
            if key in self.cache:
                scores.append(self.cache[key])
            else:
                scores.append(0.0)  # placeholder
                uncached_indices.append(idx)
                uncached_pairs.append((query, cand.text))

        if uncached_pairs:
            logger.info("Computing cross-encoder scores for %d uncached pairs...", len(uncached_pairs))
            new_scores = self.model.predict(uncached_pairs, show_progress_bar=False)
            if hasattr(new_scores, "tolist"):
                new_scores = new_scores.tolist()

            for idx, score_val in zip(uncached_indices, new_scores):
                score = float(score_val)
                scores[idx] = score
                key = f"{query}::{candidates[idx].chunk_id}"
                self.cache[key] = score

            if self.cache_path:
                _save_cache(self.cache_path, self.cache)

        return scores

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedCandidate],
        top_k: int = 5,
    ) -> list[RetrievedCandidate]:
        """Rerank candidates down to top_k by cross-encoder score."""
        if not candidates:
            return []

        scores = self.score_pairs(query, candidates)
        scored_pairs = list(zip(candidates, scores))
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        reranked: list[RetrievedCandidate] = []
        for rank, (cand, score) in enumerate(scored_pairs[:top_k], start=1):
            reranked.append(
                RetrievedCandidate(
                    chunk_id=cand.chunk_id,
                    doc_id=cand.doc_id,
                    section_id=cand.section_id,
                    header_path=cand.header_path,
                    text=cand.text,
                    page_start=cand.page_start,
                    page_end=cand.page_end,
                    score=score,
                    dense_rank=cand.dense_rank,
                    bm25_rank=cand.bm25_rank,
                    final_rank=rank,
                )
            )
        return reranked


def rerank(
    query: str,
    candidates: Sequence[RetrievedCandidate],
    top_k: int = 5,
    model_name: str = "BAAI/bge-reranker-v2-m3",
    cache_path: str | Path | None = _DEFAULT_CACHE_PATH,
) -> list[RetrievedCandidate]:
    """Functional wrapper for cross-encoder reranking."""
    reranker = CrossEncoderReranker(model_name=model_name, cache_path=cache_path)
    return reranker.rerank(query=query, candidates=candidates, top_k=top_k)


def benchmark_reranker_latency(
    query: str,
    candidates: Sequence[RetrievedCandidate],
    model_name: str = "BAAI/bge-reranker-v2-m3",
) -> dict:
    """Benchmark CPU inference latency for cross-encoder reranking without caching."""
    from sentence_transformers import CrossEncoder

    pairs = [(query, c.text) for c in candidates]
    logger.info("Benchmarking %s on %d pairs...", model_name, len(pairs))
    encoder = CrossEncoder(model_name, device="cpu")

    start_time = time.perf_counter()
    _ = encoder.predict(pairs)
    duration_s = time.perf_counter() - start_time

    per_pair_ms = (duration_s / len(pairs)) * 1000.0 if pairs else 0.0
    return {
        "num_pairs": len(pairs),
        "total_latency_s": duration_s,
        "per_pair_ms": per_pair_ms,
        "model": model_name,
    }

