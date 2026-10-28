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
_FAST_CPU_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_RERANKER_INSTANCES: dict[str, CrossEncoderReranker] = {}


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
    """Wraps cross-encoder model scoring with disk cache support and fast CPU inference."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        cache_path: str | Path | None = _DEFAULT_CACHE_PATH,
        device: str = "cpu",
        max_length: int = 256,
    ) -> None:
        self.model_name = model_name
        self.cache_path = Path(cache_path) if cache_path else None
        self.device = device
        self.max_length = max_length
        self._model = None
        self.cache: dict[str, float] = (
            _load_cache(self.cache_path) if self.cache_path else {}
        )

    @property
    def model(self):
        """Lazy load CrossEncoder (uses fast MiniLM on CPU if BGE-M3 requested without GPU)."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            effective_model = (
                _FAST_CPU_RERANKER
                if (self.device == "cpu" and self.model_name == "BAAI/bge-reranker-v2-m3")
                else self.model_name
            )
            logger.info("Loading cross-encoder %s on %s...", effective_model, self.device)
            try:
                self._model = CrossEncoder(
                    effective_model,
                    device=self.device,
                    max_length=self.max_length,
                    local_files_only=True,
                )
            except Exception:  # noqa: BLE001
                self._model = CrossEncoder(
                    effective_model,
                    device=self.device,
                    max_length=self.max_length,
                )
        return self._model

    def score_pairs(self, query: str, candidates: Sequence[RetrievedCandidate]) -> list[float]:
        """Score (query, candidate.text) pairs using cache where possible."""
        from ragcite.retrieve import is_devanagari, translate_devanagari_query

        scores: list[float] = []
        uncached_indices: list[int] = []
        uncached_pairs: list[tuple[str, str]] = []
        effective_query = translate_devanagari_query(query) if is_devanagari(query) else query

        for idx, cand in enumerate(candidates):
            key = f"{query}::{cand.chunk_id}"
            if key in self.cache:
                scores.append(self.cache[key])
            else:
                scores.append(0.0)  # placeholder
                uncached_indices.append(idx)
                uncached_pairs.append((effective_query, cand.text))

        if uncached_pairs:
            logger.info("Computing cross-encoder scores for %d uncached pairs...", len(uncached_pairs))
            new_scores = self.model.predict(
                uncached_pairs, batch_size=32, show_progress_bar=False
            )
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


def get_reranker(
    model_name: str = "BAAI/bge-reranker-v2-m3",
    cache_path: str | Path | None = _DEFAULT_CACHE_PATH,
) -> CrossEncoderReranker:
    """Return a process-wide cached CrossEncoderReranker singleton."""
    instance_key = f"{model_name}::{cache_path}"
    if instance_key not in _RERANKER_INSTANCES:
        _RERANKER_INSTANCES[instance_key] = CrossEncoderReranker(
            model_name=model_name, cache_path=cache_path
        )
    return _RERANKER_INSTANCES[instance_key]


def rerank(
    query: str,
    candidates: Sequence[RetrievedCandidate],
    top_k: int = 5,
    model_name: str = "BAAI/bge-reranker-v2-m3",
    cache_path: str | Path | None = _DEFAULT_CACHE_PATH,
) -> list[RetrievedCandidate]:
    """Functional wrapper for cross-encoder reranking using cached singleton."""
    reranker = get_reranker(model_name=model_name, cache_path=cache_path)
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

