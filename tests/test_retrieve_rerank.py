import json
import tempfile
from pathlib import Path

from ragcite.rerank import CrossEncoderReranker
from ragcite.retrieve import (
    RetrievedCandidate,
    is_devanagari,
    reciprocal_rank_fusion,
)


def test_is_devanagari():
    assert is_devanagari("नेपालको संविधानको धारा १७ मा के छ?") is True
    assert is_devanagari("What does Section 10 of Labour Act say?") is False
    assert is_devanagari("Labour Act 2017") is False


def test_reciprocal_rank_fusion():
    ranking1 = ["c1", "c2", "c3"]
    ranking2 = ["c2", "c1", "c4"]
    fused = reciprocal_rank_fusion([ranking1, ranking2], rrf_k=60)
    # c1 and c2 should have higher scores than c3 and c4
    fused_cids = [cid for cid, _ in fused]
    assert fused_cids[0] in ("c1", "c2")
    assert fused_cids[1] in ("c1", "c2")
    assert "c3" in fused_cids
    assert "c4" in fused_cids


def test_reranker_caching():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        cache_path = Path(f.name)

    try:
        # Pre-seed cache
        seed = {"query1::c1": 0.95, "query1::c2": 0.40}
        with open(cache_path, "w") as f:
            json.dump(seed, f)

        reranker = CrossEncoderReranker(cache_path=cache_path)
        cands = [
            RetrievedCandidate(
                chunk_id="c2",
                doc_id="d1",
                section_id="d1:s2",
                header_path="D1 > S2",
                text="Second chunk",
                page_start=1,
                page_end=1,
                score=0.5,
            ),
            RetrievedCandidate(
                chunk_id="c1",
                doc_id="d1",
                section_id="d1:s1",
                header_path="D1 > S1",
                text="First chunk",
                page_start=1,
                page_end=1,
                score=0.3,
            ),
        ]

        reranked = reranker.rerank("query1", cands, top_k=2)
        assert len(reranked) == 2
        # c1 has higher cached score (0.95 vs 0.40), so should be rank 1
        assert reranked[0].chunk_id == "c1"
        assert reranked[0].score == 0.95
        assert reranked[1].chunk_id == "c2"
    finally:
        cache_path.unlink(missing_ok=True)
